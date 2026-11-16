"""πR² 调度（openpi 约定：t=1 噪声，t=0 干净）。

由 apply_patch.py 复制到 openpi 的 src/openpi/models/ 下。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def staircase_tau(d: jax.Array, horizon: int) -> jax.Array:
    """论文约定下的 τ*_{·,d}：前段 1（干净），尾段 0（噪声）。d: [B] -> [B, H]。"""
    idx = jnp.arange(horizon)[None, :]
    dd = d[:, None]
    interior_denom = jnp.maximum(horizon - 2 * dd, 1)
    ramp = 1.0 - (idx - dd) / interior_denom
    tau = jnp.where(idx < dd, 1.0, ramp)
    tau = jnp.where(idx >= horizon - dd, 0.0, tau)
    return tau


def delta_tau(d: int, horizon: int) -> jnp.ndarray:
    """一次 Euler 子步每个位置的 Δτ（论文约定，正值）。"""
    tau = staircase_tau(jnp.asarray([d]), horizon)[0]
    target = jnp.zeros_like(tau)
    target = target.at[d:].set(tau[: horizon - d])
    delta = target - tau
    return delta.at[:d].set(0.0)


def delta_t(d: int, horizon: int) -> jnp.ndarray:
    """openpi 约定下的 Δt（t 递减，因此为负）。"""
    return -delta_tau(d, horizon)
