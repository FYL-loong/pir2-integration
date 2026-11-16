# 把 πR² 接到 openpi 的 π0 / π0.5 上：改动清单

这份文档对应 `third_party/openpi` 里下载的源码（commit 以 jsDelivr 的 `@main` 为准）。
所有行号引用的是 2025 版 `src/openpi/models/pi0.py`，改动本身不依赖行号。

> 结论先说：**π0 几乎可以直接改；π0.5 必须先恢复"连续状态快通道"，否则 πR² 的核心机制不成立。**

---

## 0. 为什么 π0.5 要先动状态通道

`pi0.py:151-157`：

```python
if not self.pi05:
    # add a single state token
    state_token = self.state_proj(obs.state)[:, None, :]
    tokens.append(state_token)
```

也就是说 **π0 把状态作为连续向量塞进 suffix（动作专家侧），π0.5 不这么做**。
π0.5 的状态在 `tokenizer.py` 里被离散化成 256 个桶的整数，拼进了语言 prompt：

```python
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

一旦状态进了 prefix，**每刷新一次关节角就要重跑一次 VLM prefill**，πR² 的"快通道"就没有了。

### 0.1 恢复连续状态通道（π0.5 专用）

**a) `src/openpi/models/tokenizer.py`**：π0.5 分支去掉 State 段，把 `max_token_len` 降回 48 左右。

```python
# 原
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
# 改
full_prompt = f"Task: {cleaned_text};\nAction: "
```

**b) `src/openpi/models/pi0_config.py`**：`discrete_state_input = False`（保留 `pi05=True` 以继续用 adaRMSNorm）。

**c) `src/openpi/models/pi0.py` 的 `embed_suffix`**：把 π0 的 state token 加回 π0.5 分支。

```python
# 放在 action_tokens 之前，无条件执行
state_token = self.state_proj(obs.state)[:, None, :]
tokens.append(state_token)
input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
ar_mask += [True]   # prefix 不看 state/action
```

并在 `__init__` 里让 π0.5 也创建 `state_proj`：

```python
if config.pi05:
    self.time_mlp_in = nnx.Linear(...)
    self.time_mlp_out = nnx.Linear(...)
self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)  # 两个分支都要
```

> 代价：这个 `state_proj` 在 π0.5 预训练权重里不存在，微调时是从零学的。
> 演示数据少的时候，这条路线会比直接用 π0 更容易训崩。

---

## 1. 逐位置 adaLN / adaRMS（两处）

### 1.1 π0.5：adaRMSNorm 支持 3D 条件

`src/openpi/models/gemma.py:127-131`：

```python
# 原
modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
# 改：允许 cond 已经是 [B, H, D]
modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
if modulation.ndim == 2:
    modulation = modulation[:, None, :]
scale, shift, gate = jnp.split(modulation, 3, axis=-1)
```

`normed_inputs` 是 `[B, H, D]`，`scale/shift/gate` 也是 `[B, H, D]`，广播天然成立；
`_gated_residual(x, y, gate)` 同样成立。

### 1.2 π0.5：`embed_suffix` 产出逐位置条件

```python
# 原：timestep 是标量 [B]
time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
# 改：timestep 是 [B, H]
w = self.action_in_proj.out_features
time_emb = posemb_sincos(timestep.reshape(-1), w, min_period=4e-3, max_period=4.0)
time_emb = time_emb.reshape(*timestep.shape, w)
```

`posemb_sincos` 的签名是 `"b"`，所以先 flatten 再 reshape 即可，函数本身不用改。

### 1.3 π0：时间已经混进 token，天然支持

`pi0.py:172-173`：

```python
# 原
time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
# 改（time 已是 [B, H] 时）
time_tokens = time_emb
```

---

## 2. 训练目标：延迟采样 + 前段 inpaint

`compute_loss` 里，把"一个标量 time"换成"逐位置 time"，并加 mask。

```python
batch_shape = actions.shape[:-2]
noise = jax.random.normal(noise_rng, actions.shape)
u_t = noise - actions

if config.pir2:
    d_rng, jit_rng, warm_rng = jax.random.split(rng, 3)
    d = jax.random.randint(d_rng, batch_shape, 1, config.d_max + 1)      # [B]

    tau = staircase_tau(d, self.action_horizon)                          # [B,H], 1=干净
    tau = jnp.clip(tau + jax.random.uniform(jit_rng, tau.shape,
                                            minval=-config.pir2_jitter,
                                            maxval=config.pir2_jitter), 0, 1)

    # 20% 的 batch 走标准 flow，用来在 episode 开始整块去噪
    warm = jax.random.uniform(warm_rng, batch_shape) < config.pir2_warmup_prob
    flow_tau = jax.random.uniform(jit_rng, batch_shape)[:, None]
    tau = jnp.where(warm[:, None], flow_tau, tau)

    t = 1.0 - tau                          # openpi 约定：t=1 噪声
    x_t = t[..., None] * noise + (1 - t[..., None]) * actions

    front = jnp.arange(self.action_horizon)[None, :] < d[:, None]
    front = jnp.where(warm[:, None], False, front)       # warmup 分支不 clamp
    x_t = jnp.where(front[..., None], actions, x_t)
    loss_mask = (~front).astype(jnp.float32)
else:
    time = ...
    ...
    loss_mask = None
```

loss 部分：

```python
se = jnp.square(v_t - u_t).mean(axis=-1)                 # [B,H]
if loss_mask is None:
    return se.mean(axis=-1)
return (se * loss_mask).sum() / loss_mask.sum().clip(min=1.0)
```

> `staircase_tau` 用仓库里的 `pir2/schedule.py`（论文约定 τ=1 干净），或者用
> `integrations/openpi/pir2_openpi.py` 的 `staircase_t`（openpi 约定）再取 `1 - t`。

---

## 3. 视觉陈旧度嵌入

模型里加一个查表（零初始化，保证未微调时等价于无延迟）：

```python
# __init__，π0 和 π0.5 都要
self.pir2_delay_emb = nnx.Embed(
    num_embeddings=config.d_max_vis + 1,
    features=action_expert_config.width,
    embedding_init=nn.initializers.zeros,
    rngs=rngs,
)
```

`embed_suffix` 里加在动作 token 上（论文原文就是这么做的，广播到 H 个位置）：

```python
action_tokens = self.action_in_proj(noisy_actions)
if config.pir2:
    action_tokens = action_tokens + self.pir2_delay_emb(obs.vision_delay)[:, None, :]
```

需要在 `_model.Observation` 里加一个字段：

```python
vision_delay: at.Int[at.Array, " b"] | None = None   # 缓存视觉特征落后了多少 tick
```

---

## 4. 数据侧：喂"旧帧"

论文的做法是：每个 batch 采样 `dvis ~ U{0..dmax_vis}`，把当前图像替换成 `dvis` 步之前的帧，
本体感觉保持当前。openpi 里最省事的实现是在 `transforms.py` 加一个 transform，按 episode 索引取历史帧：

```python
@dataclasses.dataclass(frozen=True)
class Pir2StaleVision(DataTransformFn):
    d_max_vis: int
    def __call__(self, data: DataDict) -> DataDict:
        # data 需要带 episode 内的时间索引；按 dvis 回退取帧
        ...
```

如果你的数据是 LeRobot 格式，`delta_timestamps` 已经支持按时间偏移取帧，
可以直接在 `LeRobotDataset` 构造时给 image key 配 `[-dvis*dt, 0.0]` 两个时间点，
训练时随机二选一，省掉自定义 transform。

---

## 5. 推理：滚动缓冲区

模型端新增一个"单步"方法（对照 `sample_actions`，它每次都要重跑 prefix + K 次去噪）：

```python
def sample_actions_step(self, observation, buffer_x, buffer_t, kv_cache, d):
    """buffer_x [H, A]，buffer_t [H]（openpi 约定），返回更新后的 buffer_x。"""
    obs = _model.preprocess_observation(None, observation, train=False)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        obs, buffer_x[None], buffer_t[None]
    )
    # suffix 看 prefix 的掩码 + 自身
    prefix_attn = einops.repeat(obs_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_mask = jnp.concatenate([prefix_attn, make_attn_mask(suffix_mask, suffix_ar_mask)], axis=-1)
    positions = jnp.sum(obs_prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, -1) - 1
    (_, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens], mask=full_mask, positions=positions,
        kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
    )
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])[0]
    return buffer_x + delta_t(d, self.action_horizon)[:, None] * v_t   # Δt 为负
```

机器人端（对齐论文 Alg. 2）：

```python
buf = RollingBuffer(H, action_dim)
chunk = model.sample_actions(rng, obs, num_steps=10)     # warm start
buf.init_from_chunk(chunk[0], d0)

while True:                                              # 每个控制 tick
    fresh_proprio = read_robot()
    obs = obs_with(fresh_proprio, cached_vlm_feature, dvis=age_of(cache))
    if new_chunk_ready:
        ...
    v = model.sample_actions_step(obs, buf.x, buf.t, kv_cache, d)
    actions = buf.step(v, d)                             # d 个新动作
    send(actions)
```

关键：**`kv_cache` 只在 VLM 后台线程刷新时重建**，动作循环里只跑一次 suffix 前向。
`sample_actions` 里 `pi0.py:233-237` 已经把 prefix 单独前向并返回 `kv_cache`，这条路是通的。

---

## 6. 配置

```python
@dataclasses.dataclass(frozen=True)
class Pi0Pir2Config(Pi0Config):
    pi05: bool = True
    pir2: bool = True
    d_max: int = 5
    d_max_vis: int = 5
    pir2_jitter: float = 0.05
    pir2_warmup_prob: float = 0.2
    action_horizon: int = 50
```

---

## 7. 改完先跑这几个检查

1. **退化为原模型**：`pir2=False` 时，改动前后 `compute_loss` 数值一致。
2. **零延迟一致性**：`d=1`、`jitter=0`、`warmup_prob=0` 时，`tau[:, 0]` 恒为 1、`tau[:, -1]` 恒为 0。
3. **调度复原**：连续 `d` 不变地跑 20 次 `step`，`buf.t` 应精确等于 `staircase_t(d, H)`（见 `tests/test_openpi_convention.py`）。
4. **前缀缓存等价**：用缓存 KV 做单步去噪，与"重新跑一遍 prefix"的结果在 1e-4 内一致。

第 4 条最容易出错，也最值得先测——它决定了异步到底成不成立。
