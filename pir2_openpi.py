"""πR² 调度在 openpi 约定下的实现（numpy，无 JAX 依赖）。

openpi 的约定与论文相反：
    t = 1 -> 纯噪声
    t = 0 -> 干净动作
    x_t = t * ε + (1 - t) * a
    v   = ε - a
    Euler: x += dt * v,  dt = -1/K

因此论文的 τ 与 openpi 的 t 满足  t = 1 - τ，
论文的 Δτ 对应这里的 Δt = -Δτ。

这个文件只做"调度 + 缓冲区"的数值逻辑，方便在机器人端（或 openpi 的
sample_actions 改写里）直接复用；模型前向仍在 openpi 的 JAX 代码里。
"""

from __future__ import annotations

import numpy as np


def staircase_t(d: int, horizon: int) -> np.ndarray:
    """openpi 约定下的 t*_{·,d}：前段 t=0（干净），尾段 t=1（噪声）。"""
    if horizon <= 2 * d:
        raise ValueError(f"需要 H > 2d，收到 H={horizon}, d={d}")
    idx = np.arange(horizon)
    t = np.empty(horizon, dtype=np.float32)
    t[idx < d] = 0.0
    interior = (idx >= d) & (idx < horizon - d)
    t[interior] = (idx[interior] - d) / float(horizon - 2 * d)
    t[idx >= horizon - d] = 1.0
    return t


def delta_t(d: int, horizon: int) -> np.ndarray:
    """一次 Euler 子步每位置的 Δt（负值，t 递减方向）。"""
    t = staircase_t(d, horizon)
    target = np.zeros_like(t)
    target[d:] = t[: horizon - d]
    delta = target - t
    delta[:d] = 0.0
    return delta


def shift(t: np.ndarray, d: int, fill: float = 1.0) -> np.ndarray:
    """右移 d 格；尾部新补的槽在 openpi 约定下是纯噪声，故 t=1。"""
    out = np.full_like(t, fill)
    out[: t.shape[0] - d] = t[d:]
    return out


class RollingBuffer:
    """设备端滚动缓冲区，语义与论文 Fig. 2 一致。"""

    def __init__(self, horizon: int, action_dim: int):
        self.h = horizon
        self.a = action_dim
        self.x = np.zeros((horizon, action_dim), dtype=np.float32)
        self.t = np.zeros(horizon, dtype=np.float32)

    def init_from_chunk(self, clean_chunk: np.ndarray, d: int, rng: np.random.Generator | None = None) -> None:
        """用一次标准 flow 的整块去噪结果初始化，再重新加噪到 t*_d。"""
        rng = rng or np.random.default_rng()
        t = staircase_t(d, self.h)
        eps = rng.standard_normal(clean_chunk.shape).astype(np.float32)
        self.x = t[:, None] * eps + (1 - t)[:, None] * clean_chunk.astype(np.float32)
        self.t = t

    def clamp_front(self, inflight: np.ndarray) -> None:
        d = inflight.shape[0]
        self.x[:d] = inflight
        self.t[:d] = 0.0

    def euler(self, velocity: np.ndarray, d: int) -> None:
        dt = delta_t(d, self.h)
        self.x = self.x + dt[:, None] * velocity
        self.t = np.clip(self.t + dt, 0.0, 1.0)

    def emit(self, d: int) -> np.ndarray:
        return self.x[d : 2 * d].copy()

    def slide(self, d: int, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        fresh = rng.standard_normal((d, self.a)).astype(np.float32)
        self.x = np.concatenate([self.x[d:], fresh], axis=0)
        self.t = shift(self.t, d)

    def step(self, velocity: np.ndarray, d: int, rng: np.random.Generator | None = None) -> np.ndarray:
        self.euler(velocity, d)
        out = self.emit(d)
        self.slide(d, rng)
        return out
