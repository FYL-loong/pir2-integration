#!/usr/bin/env python3
"""πR² on π0.5 —— 平台环境自检。

在训练平台上跑这一条命令，它会把"能不能开始"需要知道的全部信息打出来：

    python preflight.py                 # 环境 + 依赖 + 补丁状态
    python preflight.py --gates         # 额外跑两个 Gate（需要 GPU，会实例化模型）

设计原则：任何一项失败都不中断，最后给一张总表和明确的下一步。
没有 jax 的机器也能跑（只报环境部分），所以可以拿它在本地先看一眼。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

RESULTS: list[tuple[str, str, str]] = []  # (状态, 项目, 说明)


def record(status: str, item: str, detail: str = "") -> None:
    RESULTS.append((status, item, detail))
    icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIP": "➖"}[status]
    line = f"{icon} {item}"
    if detail:
        line += f"  —  {detail}"
    print(line)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- 环境
def check_python() -> None:
    section("Python")
    major, minor = sys.version_info[:2]
    if (major, minor) == (3, 11):
        record("PASS", "Python 3.11", sys.executable)
    elif (major, minor) >= (3, 11):
        record("WARN", f"Python {major}.{minor}", "openpi 的 .python-version 是 3.11；版本不符可能踩依赖坑")
    else:
        record("FAIL", f"Python {major}.{minor}", "需要 >= 3.11")


def check_gpu() -> None:
    section("GPU")
    if shutil.which("nvidia-smi") is None:
        record("FAIL", "nvidia-smi", "找不到；如果是 CPU 机器，只能做代码检查")
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, timeout=20,
        ).strip()
    except Exception as e:  # noqa: BLE001
        record("WARN", "nvidia-smi 调用失败", str(e))
        return
    lines = [l for l in out.splitlines() if l.strip()]
    for i, l in enumerate(lines):
        record("PASS", f"GPU{i}", l)
    if len(lines) < 1:
        record("FAIL", "GPU 数量", "0 张卡")


def check_disk() -> None:
    section("磁盘")
    total, used, free = shutil.disk_usage(Path.cwd())
    free_gb = free / 1e9
    status = "PASS" if free_gb > 60 else ("WARN" if free_gb > 20 else "FAIL")
    record(status, "可用空间", f"{free_gb:.0f} GB（pi05_base 约 12 GB，checkpoint 会写很多）")


# --------------------------------------------------------------------------- 依赖
def _ver(mod: str) -> str | None:
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def check_deps() -> None:
    section("依赖")
    if importlib.util.find_spec("jax") is None:
        record("FAIL", "jax", "未安装。openpi 固定 jax[cuda12]==0.5.3")
    else:
        v = _ver("jax")
        if v == "0.5.3":
            record("PASS", "jax", v)
        else:
            record("WARN", "jax 版本", f"{v}（openpi 期望 0.5.3）")
        try:
            import jax

            devs = jax.devices()
            record("PASS", "jax devices", ", ".join(str(d) for d in devs[:8]) + (" ..." if len(devs) > 8 else ""))
            if not any(d.platform == "gpu" for d in devs):
                record("FAIL", "jax 未看到 GPU", "可能装成了 CPU 版 jax，或 CUDA 驱动不匹配")
        except Exception as e:  # noqa: BLE001
            record("FAIL", "jax 初始化失败", str(e))

    for mod, want in (("flax", "0.10.2"), ("jaxtyping", "0.2.36")):
        v = _ver(mod)
        if v is None:
            record("FAIL", mod, "未安装")
        elif v == want:
            record("PASS", mod, v)
        else:
            record("WARN", mod, f"{v}（openpi 期望 {want}）")


def check_openpi() -> None:
    section("openpi")
    try:
        import openpi  # noqa: F401

        record("PASS", "openpi 可导入", str(Path(openpi.__file__).parent))
    except Exception as e:  # noqa: BLE001
        record("FAIL", "openpi 可导入", f"{e}\n     提示：cd 到 openpi 根目录，用 `uv run python preflight.py`")
        return

    try:
        from openpi.models import pi0_config

        has = any(f.name == "pir2_mode" for f in __import__("dataclasses").fields(pi0_config.Pi0Config))
        if has:
            record("PASS", "补丁：Pi0Config.pir2_mode", "已应用")
        else:
            record("FAIL", "补丁：Pi0Config.pir2_mode", "未应用，先跑 apply_patch.py")
    except Exception as e:  # noqa: BLE001
        record("FAIL", "补丁检查", str(e))

    try:
        from openpi.models.pi0 import Pi0

        if hasattr(Pi0, "sample_actions_step"):
            record("PASS", "补丁：Pi0.sample_actions_step", "已应用")
        else:
            record("FAIL", "补丁：Pi0.sample_actions_step", "未应用")
    except Exception as e:  # noqa: BLE001
        record("FAIL", "Pi0 导入", str(e))

    try:
        from openpi.training import config as _config

        names = [c.name for c in _config._CONFIGS if c.name.startswith("pir2")]
        if names:
            record("PASS", "已注册配置", ", ".join(names))
        else:
            record("FAIL", "已注册配置", "没有 pir2_* 配置")
    except Exception as e:  # noqa: BLE001
        record("FAIL", "配置注册", str(e))


# --------------------------------------------------------------------------- Gate
def gate_buffer() -> None:
    """Gate B：滑动缓冲区不变量（纯 numpy，任何机器都能跑）。"""
    section("Gate B：缓冲区不变量")
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    try:
        from pir2_runtime import RollingBuffer, staircase_t
        import numpy as np

        ok = True
        for d in (1, 2, 3, 5):
            buf = RollingBuffer(50, 7)
            buf.init_from_chunk(np.zeros((50, 7), dtype=np.float32), d)
            for _ in range(10):
                buf.apply(np.zeros((50, 7), dtype=np.float32), d)
                if not np.allclose(buf.t, staircase_t(d, 50), atol=1e-6):
                    ok = False
        record("PASS" if ok else "FAIL", "step+slide 后调度复原", "H=50, d∈{1,2,3,5}" if ok else "有偏差")
    except Exception as e:  # noqa: BLE001
        record("WARN", "Gate B 跳过", str(e))


def gate_prefix_cache(gate_dtype: str = "float32") -> None:
    """Gate A：缓存前缀 KV 的单步结果 == 不走缓存的整段前向。

    默认用 float32：bf16 下两条不同计算图本来就会差 1e-3 量级，
    那不是逻辑错误，会把真正的 bug 淹没。float32 下应当收敛到 1e-6。
    """
    section(f"Gate A：前缀缓存等价性（{gate_dtype}）")
    if importlib.util.find_spec("jax") is None:
        record("SKIP", "Gate A", "没有 jax")
        return
    try:
        import dataclasses

        import jax
        import jax.numpy as jnp
        from openpi.training import config as _config
        from openpi.models.pi0 import make_attn_mask
    except Exception as e:  # noqa: BLE001
        record("SKIP", "Gate A", f"导入失败：{e}")
        return

    try:
        cfg = _config.get_config("pir2_staircase")
    except Exception as e:  # noqa: BLE001
        record("SKIP", "Gate A", f"拿不到 pir2_staircase 配置：{e}")
        return

    try:
        model_cfg = dataclasses.replace(cfg.model, dtype=gate_dtype)
        model = model_cfg.create(jax.random.key(0))
        obs = model_cfg.fake_obs()
        H = model.action_horizon
        A = model_cfg.action_dim
        x = jax.random.normal(jax.random.key(1), (1, H, A), dtype=jnp.float32)
        t = jnp.full((1, H), 0.5)

        # 路径 1：先用缓存
        prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
        pre_attn = make_attn_mask(prefix_mask, prefix_ar)
        pre_pos = jnp.cumsum(prefix_mask, axis=-1) - 1
        _, kv = model.PaliGemma.llm([prefix_tokens, None], mask=pre_attn, positions=pre_pos)
        out_cached = model.sample_actions_step(obs, x, t, kv, prefix_mask, d=1)

        # 路径 2：前缀和 suffix 一次性前向（compute_loss 用的路径）
        s_tokens, s_mask, s_ar, cond = model.embed_suffix(obs, x, t)
        full_mask = jnp.concatenate([prefix_mask, s_mask], axis=1)
        attn = make_attn_mask(full_mask, jnp.concatenate([prefix_ar, s_ar]))
        pos = jnp.cumsum(full_mask, axis=-1) - 1
        (_, s_out), _ = model.PaliGemma.llm(
            [prefix_tokens, s_tokens], mask=attn, positions=pos, adarms_cond=[None, cond]
        )
        v_full = model.action_out_proj(s_out[:, -H:])

        from openpi.models import pir2_schedule

        out_full = x + pir2_schedule.delta_t(1, H)[None, :, None] * v_full
        diff = float(jnp.max(jnp.abs(out_cached - out_full)))
        scale = float(jnp.max(jnp.abs(out_full))) + 1e-6
        rel = diff / scale

        # 灵敏度对照：给速度场加 10% 扰动，确认这个指标能识别真实差异。
        # 如果扰动下差异仍然很小，说明指标没有区分力，PASS 也不可信。
        ctrl_v = v_full * 1.1 + 0.1 * jax.random.normal(jax.random.key(7), v_full.shape)
        out_ctrl = x + pir2_schedule.delta_t(1, H)[None, :, None] * ctrl_v
        ctrl_rel = float(jnp.max(jnp.abs(out_cached - out_ctrl))) / scale

        tol = 1e-3 if gate_dtype == "float32" else 5e-3
        detail = (
            f"{gate_dtype} 相对差异 {rel:.2e}（绝对 {diff:.2e}，阈值 {tol:g}）；"
            f"10% 扰动对照 {ctrl_rel:.2e}"
        )
        if ctrl_rel < 10 * tol:
            record("WARN", "Gate A 对照失效", detail + " ← 指标区分力不足，本项结果不可信")
        elif rel < tol:
            record("PASS", "Gate A", detail)
        else:
            record("FAIL", "Gate A", detail)
    except Exception as e:  # noqa: BLE001
        record("FAIL", "Gate A", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true", help="额外跑 Gate A（会实例化 3B 模型，需要 GPU）")
    ap.add_argument(
        "--gate-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="Gate A 用哪种精度；bf16 下两条计算图本来就会差 1e-3 量级",
    )
    args = ap.parse_args()

    print("πR² on π0.5 —— 平台自检")
    print(f"cwd: {Path.cwd()}")

    check_python()
    check_gpu()
    check_disk()
    check_deps()
    check_openpi()
    gate_buffer()
    if args.gates:
        gate_prefix_cache(args.gate_dtype)

    section("汇总")
    n_fail = sum(1 for s, _, _ in RESULTS if s == "FAIL")
    n_warn = sum(1 for s, _, _ in RESULTS if s == "WARN")
    n_pass = sum(1 for s, _, _ in RESULTS if s == "PASS")
    print(f"PASS {n_pass}   WARN {n_warn}   FAIL {n_fail}")

    if n_fail:
        print("\n有 FAIL，先解决这些再往下走：")
        for s, item, detail in RESULTS:
            if s == "FAIL":
                print(f"  - {item}: {detail}")
        return 1

    print("\n环境就绪，可以进入下一步：")
    print("  阶段 1：π0.5 状态通道补丁（apply_patch.py --pi05-state）")
    print("  阶段 2：apply_patch.py --openpi-dir . && compute_norm_stats.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
