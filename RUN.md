# 训练平台上的执行清单

目标机器假设：**Linux + 8×H100 80GB**（或 8×A100 80GB）。显存不够就走 LoRA，见 §4.3。
下面所有命令都在 openpi 仓库根目录执行。

先说明两条路线的差别，别选错：

| | 改动量 | 何时用 |
|---|---|---|
| **π0** | 打补丁即可，状态本来就是连续向量进 suffix | 第一优先，先验证方法 |
| **π0.5** | 还要手工恢复连续状态通道（`PATCH.md` §0） | π0 跑通、且确实需要 0.5 的泛化 |

---

## 0. 先跑自检（每次上机第一步）

```bash
cd ~/openpi
python /path/to/pir2_integration/preflight.py          # 环境 + 依赖 + 补丁状态
python /path/to/pir2_integration/preflight.py --gates  # 再加跑 Gate A（会实例化 3B 模型）
```

它会逐项打印 PASS / WARN / FAIL，并在最后给出"能不能往下走"的结论。
任何 FAIL 都不要跳过——尤其是 `jax 未看到 GPU` 和 `补丁未应用` 这两类。

## 0.1 环境

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv run pip install -e .
```

把本仓库的 `integrations/openpi/` 整个拷到 openpi 旁边（不要放进 `src/`）：

```bash
cp -r /path/to/pir2_repro/integrations/openpi  ./pir2_integration
```

## 1. 预训练权重

```bash
# π0 base（约 8 GB）
uv run scripts/download_assets.py pi0_base
# π0.5 base（走 π0.5 路线时再下，约 12 GB）
# uv run scripts/download_assets.py pi05_base
```

默认从 `gs://openpi-assets` 拉。国内机器可以先 `gcloud auth login`，
或从 HF 镜像 `physical-intelligence/pi0_base` 下载后设 `PIR2_BASE_CKPT=<本地路径>/params`。

## 2. 打补丁

```bash
python pir2_integration/apply_patch.py --openpi-dir . --dry-run   # 先看会改什么
python pir2_integration/apply_patch.py --openpi-dir .             # 落地
```

预期输出 12 项 `patch` + 2 项 `copy`。任何一项报「锚点数量异常」就说明 openpi 版本变了，
停下来按 `PATCH.md` 人工核对，**不要**跳过。

校验：

```bash
uv run python -c "
from openpi.training import config
print([c.name for c in config._CONFIGS if c.name.startswith('pir2')])
"
# 期望：['pir2_flow', 'pir2_rtc', 'pir2_staircase']
```

补丁是幂等的，重复执行只会打印 `skip`。

## 3. 数据

数据必须是 LeRobot 格式。改 `src/openpi/training/pir2_configs.py` 里这两行：

```python
DATASET = os.environ.get("PIR2_DATASET", "your-org/your-dataset")
```

ALOHA 形态可以直接用；其它机器人要照 `config.py` 里 `LeRobotAlohaDataConfig` 的注释
改 `repack_transforms`（相机 key、state/action 的 key）。

然后算归一化统计量（三条腿都要算，或者算完复用）：

```bash
export PIR2_DATASET=your-org/your-dataset
for c in pir2_flow pir2_rtc pir2_staircase; do
  uv run scripts/compute_norm_stats.py --config-name $c
done
```

`PIR2_HORIZON` 默认 50（25 Hz 下 2 秒，与论文真机设置一致）。
**如果你的控制频率不是 25 Hz，必须改**，否则 `d` 的 tick 换算会错。

## 4. 训练

### 4.1 三条腿（论文可比性的核心）

三条腿只有 `model.pir2_mode` 不同，数据、checkpoint、步数、batch 全一样。
**不要**为了让 πR² 好看而单独加训练量，那样数字没有意义。

```bash
export PIR2_STEPS=30000 PIR2_BATCH=32
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

uv run scripts/train.py pir2_flow      --exp-name=seed0 --overwrite
uv run scripts/train.py pir2_rtc       --exp-name=seed0 --overwrite
uv run scripts/train.py pir2_staircase --exp-name=seed0 --overwrite
```

8×H100 上 30k 步大约 12~14 小时/条，三条腿约 40 GPU·小时。
显存吃紧时加 `--fsdp-devices=2` 或 4。

### 4.2 多卡/多机

```bash
# 8 卡数据并行（openpi 默认就是这样，fsdp_devices=1）
uv run scripts/train.py pir2_staircase --exp-name=seed0 --batch-size=32

# 显存不够，切 FSDP
uv run scripts/train.py pir2_staircase --exp-name=seed0 --fsdp-devices=4
```

### 4.3 显存不够：LoRA

```bash
# 改 pir2_configs.py 里 _cfg(..., lora=True)，或直接复制一份配置
PIR2_BATCH=64 uv run scripts/train.py pir2_staircase_lora --exp-name=seed0
```

LoRA 只训低秩适配器，24 GB 卡也能跑，但 action head 的表达能力会打折，
**先确认全参版本的趋势，再用 LoRA 降本**。

### 4.4 训练开始后先看这三个数

```bash
# 1) loss 是否在降（前 500 步内应明显下降）
# 2) staircase 的 masked_frac 应接近 d_mean/H ≈ 3/50 = 0.06
# 3) 如果 loss 长期不动，先检查 DATA 的 action 维度与 PIR2_HORIZON
```

## 5. 评测

### 5.1 先测"前缀缓存等价性"（最容易出错，必须测）

如果这一步不成立，异步就是假的：

```bash
uv run python - <<'PY'
import jax, jax.numpy as jnp
from openpi.training import config as _config
from openpi.models import model as _model

cfg = _config.get_config("pir2_staircase")
model = cfg.model.create(jax.random.key(0))
obs = cfg.model.fake_obs()
params = _model.state_dict(model)

# 一次完整 prefix 前向 -> kv_cache, prefix_mask
prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
attn = model.make_attn_mask(prefix_mask, prefix_ar)
pos = jnp.cumsum(prefix_mask, -1) - 1
_, kv = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=pos)

# 用缓存做单步；再和不带缓存重算的结果比
x = jax.random.normal(jax.random.key(1), (1, model.action_horizon, cfg.model.action_dim))
t = jnp.full((1, model.action_horizon), 0.5)
out_cached = model.sample_actions_step(obs, x, t, kv, prefix_mask, d=1)

# 对照：把 prefix 重新算一遍（这里省略，思路是把 kv 换成新算的）
print("cached step ok, |v| =", float(jnp.linalg.norm(out_cached)))
PY
```

预期：不报错，且 `sample_actions_step` 的输出与"重算 prefix"版本在 1e-4 内一致。

### 5.2 离线评动作误差

用留出的 episode 比 `pred vs gt` 的分位置误差，看：
- `d=1` 时靠前位置（马上要执行的）误差是否够小；
- 误差是否随位置迅速上升（上升太快说明位置加权或数据不够）。

### 5.3 真机

四条任务照论文：接触密集 + 需要即时反应。
每格至少 20 次，记录成功率与分阶段完成度。
**基线必须用同一个 checkpoint、同样步数**，否则数字不可比。

## 6. 测延迟，定 `d_max`

部署前先量这三个数（单位：控制 tick）：

```bash
uv run python - <<'PY'
import time, jax, numpy as np
# 伪代码：换成你的模型和真实观测
# t_prefill = 图像预处理 + VLM 前向
# t_step    = 一次 suffix 前向（1 NFE）
tick = 0.04                      # 25 Hz
for name, fn in [("prefill", prefill), ("step", step)]:
    fn(); jax.block_until_ready(...)      # 预热
    t0 = time.perf_counter()
    for _ in range(20):
        out = fn()
        jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / 20
    print(f"{name}: {dt*1000:.1f} ms = {dt/tick:.2f} tick")
PY
```

把 `step` 的 tick 数向上取整设为 `d_max`（论文是 5，对应 200 ms）。
若 `step` 已经超过 1 tick，`d` 会自动大于 1，缓冲区按实际值工作。

## 7. 部署

机器人端用 `pir2_runtime.py`（已通过离线自检）：

```bash
python pir2_integration/pir2_runtime.py     # 只跑自检
```

接线三步：

```python
from pir2_runtime import Pir2Runtime

rt = Pir2Runtime(horizon=50, action_dim=32, tick_seconds=0.04, d_max=5)

# 1) 视觉线程：只跑图像+语言，产出 kv_cache 与 prefix_mask
rt.start_vision_worker(prefill=my_prefill, get_obs=read_camera_and_prompt)

# 2) 动作循环：每 tick 一次单步去噪
while running:
    actions = rt.tick(get_obs=read_proprio, step_fn=my_step)   # 返回 d 个动作
    for a in actions:                     # 按控制周期逐个下发
        robot.send(a); sleep_until_next_tick()
```

`my_step` 就是 `model.sample_actions_step`，包一层 jit 即可。
`d` 由 `LatencyTracker` 用滚动窗口自动估计，不需要你手填。

## 8. 常见坑

1. **`H > 2d`**。`pir2_configs` 里 `H=50, d_max=5`，留有余量。若把 `d_max` 调到 25 以上，
   调度会退化（发射窗口和噪声尾重叠），直接报错。
2. **控制频率变了但 `PIR2_HORIZON` 没变**。`d` 是按 tick 算的，频率一改全部错位。
3. **π0.5 忘了改状态通道**。补丁不含这部分，必须按 `PATCH.md` §0 手工做，
   否则"快通道"根本没接上，但代码不会报错——这是最危险的失败模式。
4. **三条腿训练量不一致**。基线少训一半，等于自己造结论。
5. **拿单卡跑 πR²、双卡跑基线**。论文附录里 πR² 用 2×A5000 是为了让 VLM 和 DiT 不抢带宽，
   你自己测的时候要么都单卡、要么都双卡。

## 9. 想对齐论文数字，还需要什么

| 论文 | 你需要 |
|---|---|
| 仿真 Fig.3 左（h 扫描） | MuJoCo Playground Leap Cube，纯状态，`H=16` |
| 仿真 Fig.3 右（延迟） | `d0 ∈ {1,2,3}`，VLM ≈ 0.75 d0，1 NFE ≈ 0.25 d0 |
| 真机 Tab.1 | xArm6 + XHand（或等效），4 个接触密集任务，每任务 100~300 条演示 |
| 25 Hz / 40 ms | 控制频率 25 Hz，`H=50` |
| +23% / +30% | 三条腿同预算训练，四任务各 20 次 |

本地的玩具任务（`pir2_repro/scripts/train_toy.py`）可以在同一台机器上先跑一遍，
用来确认你对 `d`、`h`、warm start 的理解没错，再去烧 GPU。
