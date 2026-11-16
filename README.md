# πR² on π0.5 — integration kit

把 **πR²**（Reactive Real-time Flow Policies, Park & Tulsiani, CMU）的机制接到
openpi 的 **π0.5 / π0** 上所需的全部东西。

## 里面有什么

| 文件 | 作用 |
|---|---|
| `preflight.py` | 平台自检：GPU / jax / flax / openpi / 补丁状态 + Gate B，一条命令看全部 |
| `apply_patch.py` | 给 openpi 打补丁（锚点校验、幂等、失败即停） |
| `files/` | 会被复制进 openpi 的两个新模块（`pir2_schedule.py`、`pir2_configs.py`） |
| `pir2_runtime.py` | 机器人端运行时：滚动缓冲区 + 双线程 + 延迟自动估计（含自检） |
| `pir2_openpi.py` | openpi 约定（t=1 噪声）下的调度与缓冲区，用于对照 |
| `RUN.md` | 训练平台上的完整命令清单 |
| `PATCH.md` | 逐行改动说明 |

## 快速开始

```bash
# 1. 自检（还没打补丁时，三个"补丁未应用"的 FAIL 是预期的）
uv run python preflight.py

# 2. 先看会改什么
python apply_patch.py --openpi-dir /path/to/openpi --dry-run

# 3. 落地
python apply_patch.py --openpi-dir /path/to/openpi

# 4. 再自检，应该全绿
uv run python preflight.py --gates
```

## 已验证 / 未验证

- ✅ 补丁脚本在 openpi 官方源码上 12 项改动 + 2 个文件全部命中，幂等，改完 45 个 .py 语法通过
- ✅ `pir2_runtime.py` 自检：理想速度场下缓冲区误差 1.5e-08
- ✅ 调度数学 20 个单元测试（`pir2_repro/tests`）
- ❌ **JAX 前向没有在真实模型上跑过**（开发机无 GPU）。第一次上机务必先跑 `preflight.py --gates`
- ❌ π0.5 的连续状态通道改造需要手工完成，见 `PATCH.md` §0

## 出处

对应的完整复现代码、玩具任务实验与结果分析在 `pir2_repro/` 目录（不在此仓库内分发）。
