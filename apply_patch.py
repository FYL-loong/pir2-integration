#!/usr/bin/env python3
"""把 πR² 打到 openpi 上。

设计成"锚点替换"而不是 git diff：每处改动都先断言锚点唯一存在，
对不上立刻报错退出，绝不会改坏你机器上的版本。可重复执行（幂等）。

用法：
    python apply_patch.py --openpi-dir ~/openpi --dry-run     # 先看会改什么
    python apply_patch.py --openpi-dir ~/openpi               # 落地
    python apply_patch.py --openpi-dir ~/openpi --check       # 检查是否已打上
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARK = "[pir2]"

# --------------------------------------------------------------------------- 新增文件
NEW_FILES = [
    ("files/src/openpi/models/pir2_schedule.py", "src/openpi/models/pir2_schedule.py"),
    ("files/src/openpi/training/pir2_configs.py", "src/openpi/training/pir2_configs.py"),
]

# --------------------------------------------------------------------------- 文本改动
# (相对路径, 旧文本, 新文本, 人类可读说明)
EDITS: list[tuple[str, str, str, str]] = [
    # 0. 导入调度模块
    (
        "src/openpi/models/pi0.py",
        "import openpi.models.gemma as _gemma\n",
        "import openpi.models.gemma as _gemma\nimport openpi.models.pir2_schedule as pir2_schedule  # [pir2]\n",
        "pi0.py: 导入 pir2_schedule",
    ),
    # 1. 配置字段
    (
        "src/openpi/models/pi0_config.py",
        "    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.\n"
        "    discrete_state_input: bool = None  # type: ignore\n",
        "    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.\n"
        "    discrete_state_input: bool = None  # type: ignore\n"
        "\n"
        "    # [pir2] 噪声调度模式：flow（基线）/ rtc（基线）/ staircase（πR²）\n"
        '    pir2_mode: str = "flow"\n'
        "    # [pir2] 训练时采样的最大推理延迟（控制 tick），以及视觉陈旧度上限\n"
        "    d_max: int = 5\n"
        "    d_max_vis: int = 5\n"
        "    # [pir2] 阶梯抖动量、标准 flow 预热分支比例\n"
        "    pir2_jitter: float = 0.05\n"
        "    pir2_warmup_prob: float = 0.2\n",
        "pi0_config.py: 增加 pir2_mode / d_max / d_max_vis / jitter / warmup",
    ),
    # 2. __init__ 记下配置
    (
        "src/openpi/models/pi0.py",
        "        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)\n"
        "\n"
        "        # This attribute gets automatically set by model.train() and model.eval().\n",
        "        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)\n"
        "\n"
        "        # [pir2] 调度相关配置（不引入新参数，保证能直接加载 pi0_base 权重）\n"
        '        self.pir2_mode = getattr(config, "pir2_mode", "flow")\n'
        '        self.pir2_d_max = getattr(config, "d_max", 5)\n'
        '        self.pir2_d_max_vis = getattr(config, "d_max_vis", 5)\n'
        '        self.pir2_jitter = getattr(config, "pir2_jitter", 0.05)\n'
        '        self.pir2_warmup_prob = getattr(config, "pir2_warmup_prob", 0.2)\n'
        "\n"
        "        # This attribute gets automatically set by model.train() and model.eval().\n",
        "pi0.py: __init__ 记录 πR² 配置",
    ),
    # 3. compute_loss：换成逐位置时间步 + 前段 inpaint
    (
        "src/openpi/models/pi0.py",
        "        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001\n"
        "        time_expanded = time[..., None, None]\n"
        "        x_t = time_expanded * noise + (1 - time_expanded) * actions\n"
        "        u_t = noise - actions\n",
        "        # [pir2] time 约定是 t=1 噪声、t=0 干净；τ 约定相反（τ=1 干净）\n"
        '        if self.pir2_mode == "flow":\n'
        "            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001\n"
        "            time_expanded = time[..., None, None]\n"
        "            x_t = time_expanded * noise + (1 - time_expanded) * actions\n"
        "            loss_mask = None\n"
        "        else:\n"
        '            assert actions.ndim == 3, "πR² 只支持 [B, H, A] 的批次"\n'
        "            b, h = actions.shape[0], actions.shape[1]\n"
        "            d_rng, jit_rng, warm_rng = jax.random.split(rng, 3)\n"
        "            d = jax.random.randint(d_rng, (b,), 1, self.pir2_d_max + 1)\n"
        '            if self.pir2_mode == "rtc":\n'
        "                # Train-Time RTC：前 d 个 clamp 成 GT，其余 H-d 共享一个 τ\n"
        "                shared = jax.random.uniform(jit_rng, (b,), minval=0.0, maxval=1.0)\n"
        "                tau = jnp.broadcast_to(shared[:, None], (b, h))\n"
        "                warm = jnp.zeros((b,), dtype=jnp.bool_)\n"
        "            else:\n"
        "                tau = pir2_schedule.staircase_tau(d, h)\n"
        "                jit = jax.random.uniform(jit_rng, tau.shape, minval=-self.pir2_jitter, maxval=self.pir2_jitter)\n"
        "                tau = jnp.clip(tau + jit, 0.0, 1.0)\n"
        "                warm = jax.random.uniform(warm_rng, (b,)) < self.pir2_warmup_prob\n"
        "                flow_tau = jax.random.uniform(jit_rng, (b, 1), minval=0.0, maxval=1.0)\n"
        "                tau = jnp.where(warm[:, None], jnp.broadcast_to(flow_tau, tau.shape), tau)\n"
        "            time = 1.0 - tau\n"
        "            x_t = time[..., None] * noise + (1 - time[..., None]) * actions\n"
        "            front = (jnp.arange(h)[None, :] < d[:, None]) & ~warm[:, None]\n"
        "            x_t = jnp.where(front[..., None], actions, x_t)\n"
        "            loss_mask = jnp.where(front, 0.0, 1.0)\n"
        "        u_t = noise - actions\n",
        "pi0.py: compute_loss 支持 staircase / RTC 与前段 inpaint",
    ),
    # 4. compute_loss 返回值：加权
    (
        "src/openpi/models/pi0.py",
        "        return jnp.mean(jnp.square(v_t - u_t), axis=-1)\n",
        "        se = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [pir2] [B, H]\n"
        "        if loss_mask is None:\n"
        "            return se\n"
        "        # 保持 [B, H] 形状（满足 typecheck），用缩放让 jnp.mean 等于 mask 内均值\n"
        "        return se * loss_mask * (loss_mask.size / jnp.maximum(loss_mask.sum(), 1.0))\n",
        "pi0.py: loss 按 inpaint mask 加权",
    ),
    # 5. embed_suffix 去掉 typecheck（timestep 要支持 [B, H]）
    (
        "src/openpi/models/pi0.py",
        "    @at.typecheck\n    def embed_suffix(\n",
        "    # [pir2] 去掉 at.typecheck：timestep 需要支持 [B, H] 的逐位置时间步\n    def embed_suffix(\n",
        "pi0.py: embed_suffix 移除 at.typecheck",
    ),
    # 6. 逐位置时间嵌入
    (
        "src/openpi/models/pi0.py",
        "        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)\n",
        "        # [pir2] 兼容 timestep 为 [B] 或 [B, H]\n"
        "        _tf = timestep.reshape(-1)\n"
        "        time_emb = posemb_sincos(_tf, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)\n"
        "        time_emb = time_emb.reshape(*timestep.shape, self.action_in_proj.out_features)\n",
        "pi0.py: 时间嵌入支持逐位置",
    ),
    # 7. π0（非 05）分支复用逐位置时间
    (
        "src/openpi/models/pi0.py",
        '            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)\n',
        "            # [pir2] 逐位置时不要 repeat\n"
        "            time_tokens = (\n"
        "                time_emb\n"
        '                if time_emb.ndim == 3\n'
        '                else einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)\n'
        "            )\n",
        "pi0.py: π0 分支兼容逐位置时间",
    ),
    # 8. 视觉陈旧度编码（不加新参数，保持 checkpoint 兼容）
    (
        "src/openpi/models/pi0.py",
        "        action_tokens = self.action_in_proj(noisy_actions)\n",
        "        action_tokens = self.action_in_proj(noisy_actions)\n"
        "        # [pir2] 视觉陈旧度：用固定的正余弦编码，不引入新参数\n"
        '        if self.pir2_mode != "flow":\n'
        '            _dvis = getattr(obs, "vision_delay", None)\n'
        "            if _dvis is None:\n"
        "                _dvis = jnp.zeros((action_tokens.shape[0],), dtype=jnp.float32)\n"
        "            _code = posemb_sincos(\n"
        "                _dvis.astype(jnp.float32), action_tokens.shape[-1], min_period=4e-3, max_period=4.0\n"
        "            )\n"
        "            action_tokens = action_tokens + 0.1 * _code[:, None, :]\n",
        "pi0.py: 视觉陈旧度编码",
    ),
    # 9. 三步之外的：单步推理方法
    (
        "src/openpi/models/pi0.py",
        "        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))\n        return x_0\n",
        "        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))\n"
        "        return x_0\n"
        "\n"
        "    def sample_actions_step(\n"
        "        self,\n"
        "        observation: _model.Observation,\n"
        "        buffer_x: at.Float[at.Array, \"b h ad\"],\n"
        "        buffer_t: at.Float[at.Array, \"b h\"],\n"
        "        kv_cache,\n"
        "        prefix_mask: at.Bool[at.Array, \"b p\"],\n"
        "        d: int,\n"
        "    ) -> at.Float[at.Array, \"b h ad\"]:\n"
        '        """[pir2] 一次调用：单步 Euler，复用 prefix KV 缓存（论文 Alg. 2）。\n'
        "\n"
        "        buffer_t 用 openpi 约定（t=1 噪声，t=0 干净）。返回更新后的 buffer_x。\n"
        '        """\n'
        "        observation = _model.preprocess_observation(None, observation, train=False)\n"
        "        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(\n"
        "            observation, buffer_x, buffer_t\n"
        "        )\n"
        "        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)\n"
        "        prefix_attn_mask = einops.repeat(prefix_mask, \"b p -> b s p\", s=suffix_tokens.shape[1])\n"
        "        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)\n"
        "        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1\n"
        "        (_, suffix_out), _ = self.PaliGemma.llm(\n"
        "            [None, suffix_tokens],\n"
        "            mask=full_attn_mask,\n"
        "            positions=positions,\n"
        "            kv_cache=kv_cache,\n"
        "            adarms_cond=[None, adarms_cond],\n"
        "        )\n"
        "        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])\n"
        "        dt = pir2_schedule.delta_t(d, self.action_horizon)\n"
        "        return buffer_x + dt[None, :, None] * v_t\n",
        "pi0.py: 新增 sample_actions_step",
    ),
    # 10. gemma adaRMS 支持 3D 条件
    (
        "src/openpi/models/gemma.py",
        "        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)\n",
        "        # [pir2] 允许 cond 已经是 [B, H, D]（逐位置调制）\n"
        "        if modulation.ndim == 2:\n"
        "            modulation = modulation[:, None, :]\n"
        "        scale, shift, gate = jnp.split(modulation, 3, axis=-1)\n",
        "gemma.py: adaRMSNorm 支持逐位置条件",
    ),
    # 11. 注册配置
    (
        "src/openpi/training/config.py",
        "if len({config.name for config in _CONFIGS}) != len(_CONFIGS):\n",
        "from openpi.training import pir2_configs as _pir2_configs  # [pir2]\n"
        "\n"
        "_CONFIGS += _pir2_configs.all_configs()\n"
        "\n"
        "if len({config.name for config in _CONFIGS}) != len(_CONFIGS):\n",
        "config.py: 注册 pir2_flow / pir2_rtc / pir2_staircase",
    ),
]


def apply_edit(path: Path, old: str, new: str, desc: str, *, dry: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return f"skip   {desc}（已应用）"
    n = text.count(old)
    if n != 1:
        raise SystemExit(
            f"!! 锚点数量异常（{n} 次）: {desc}\n   文件: {path}\n"
            "   说明你的 openpi 版本和本补丁针对的版本不一致，请手动对照 PATCH.md 修改。"
        )
    if not dry:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"{'would ' if dry else ''}patch  {desc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi-dir", default="openpi", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    root = args.openpi_dir.expanduser().resolve()
    if not (root / "src" / "openpi" / "models" / "pi0.py").exists():
        print(f"找不到 {root}/src/openpi/models/pi0.py，请用 --openpi-dir 指定 openpi 根目录")
        return 2

    here = Path(__file__).resolve().parent
    ok, fail = 0, 0

    # 新增文件
    for src_rel, dst_rel in NEW_FILES:
        src, dst = here / src_rel, root / dst_rel
        if args.check:
            print(f"{'ok ' if dst.exists() else 'MISS'}   {dst_rel}")
            ok += int(dst.exists())
            continue
        if args.dry_run:
            print(f"would copy {dst_rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"copy   {dst_rel}")

    # 文本改动
    for rel, old, new, desc in EDITS:
        path = root / rel
        if not path.exists():
            print(f"MISS   {rel}（文件不存在）")
            fail += 1
            continue
        try:
            msg = apply_edit(path, old, new, desc, dry=args.dry_run or args.check)
        except SystemExit as e:
            print(e)
            fail += 1
            continue
        print(msg)
        ok += 1

    print(f"\n完成：{ok} 项成功，{fail} 项失败。")
    if fail:
        print("有失败项，请先按 PATCH.md 人工核对。")
        return 1
    if not args.dry_run and not args.check:
        print(
            "\n下一步：\n"
            "  1) python -c \"from openpi.training import config; print([c.name for c in config._CONFIGS if c.name.startswith('pir2')])\"\n"
            "  2) python scripts/compute_norm_stats.py --config-name pir2_staircase\n"
            "  3) 见 RUN.md 的训练命令"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
