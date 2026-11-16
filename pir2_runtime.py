"""机器人端的 πR² 运行时：滚动缓冲区 + 延迟自适应 + 双线程。

这个文件里，**纯逻辑部分（LatencyTracker / RollingBuffer 的用法）可以离线测试**，
与 openpi 对接的部分（prefill / step 两个 jitted 函数）需要你在训练机上按自己的
serving 方式接一下，形状和调用顺序都写清楚了。

为什么是"进程内推理"而不是 policy server：
    openpi 的 websocket server 只暴露 sample_actions（返回动作），不暴露 KV cache。
    要做异步视觉，必须能拿到 prefix 的 kv_cache —— 所以这里直接把模型跑在进程内。
    如果要走远端，需要给 server 加两个接口：/prefill 返回 (kv_cache, prefix_mask)，
    /step 接收 buffer 返回下一步。这条路的网络往返会抬高 d，论文也把它列为限制。
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- 延迟估计
@dataclass
class LatencyTracker:
    """用最近 W 次调用的墙钟耗时估计每次调用占几个控制 tick。"""

    tick_seconds: float
    window: int = 20
    d_max: int = 5
    _samples: deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self):
        self._samples = deque(maxlen=self.window)

    def add(self, seconds: float) -> None:
        self._samples.append(seconds)

    def d(self) -> int:
        if not self._samples:
            return 1
        mean = sum(self._samples) / len(self._samples)
        return int(min(max(1, round(mean / self.tick_seconds)), self.d_max))


# --------------------------------------------------------------------------- 缓冲区
def staircase_t(d: int, horizon: int) -> np.ndarray:
    """openpi 约定：t=1 噪声，t=0 干净。"""
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
    """一次 Euler 子步每位置的 Δt（负值）。滑动后应精确复原 t*_d。"""
    t = staircase_t(d, horizon)
    target = np.empty_like(t)
    target[:d] = t[:d]          # 前 d 个不推进
    target[d:] = t[: horizon - d]
    return target - t


class RollingBuffer:
    def __init__(self, horizon: int, action_dim: int):
        self.h, self.a = horizon, action_dim
        self.x = np.zeros((horizon, action_dim), dtype=np.float32)
        self.t = np.zeros(horizon, dtype=np.float32)

    def init_from_chunk(self, clean: np.ndarray, d: int, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        t = staircase_t(d, self.h)
        eps = rng.standard_normal(clean.shape).astype(np.float32)
        self.x = t[:, None] * eps + (1 - t)[:, None] * clean.astype(np.float32)
        self.t = t

    def clamp_front(self, inflight: np.ndarray) -> None:
        self.x[: inflight.shape[0]] = inflight
        self.t[: inflight.shape[0]] = 0.0

    def apply(self, velocity: np.ndarray, d: int) -> np.ndarray:
        """一步 Euler -> 取出 [d, 2d) -> 滑动。输入是**速度**。"""
        dt = delta_t(d, self.h)
        self.x = self.x + dt[:, None] * velocity
        self.t = np.clip(self.t + dt, 0.0, 1.0)
        return self._emit_and_slide(d)

    def commit(self, new_x: np.ndarray, d: int) -> np.ndarray:
        """输入是**已更新好的 buffer_x**（例如 model.sample_actions_step 的返回值）。

        注意：不要把已经更新过的 buffer_x 再送给 apply()，那会积分两次。
        """
        self.x = np.asarray(new_x, dtype=np.float32)
        self.t = np.clip(self.t + delta_t(d, self.h), 0.0, 1.0)
        return self._emit_and_slide(d)

    def _emit_and_slide(self, d: int) -> np.ndarray:
        out = self.x[d : 2 * d].copy()
        fresh = np.random.standard_normal((d, self.a)).astype(np.float32)
        self.x = np.concatenate([self.x[d:], fresh], axis=0)
        self.t = np.concatenate([self.t[d:], np.ones(d, dtype=np.float32)])
        return out


# --------------------------------------------------------------------------- 双线程
class Pir2Runtime:
    """视觉 worker 在后台刷新 prefix 特征；动作 worker 每 tick 做一次单步去噪。

    与 openpi 的接口（你需要提供的两个函数）：
        prefill(observation) -> (kv_cache, prefix_mask)
            跑一次 embed_prefix + LLM prefill，只依赖图像和语言，代价最大。
        step(observation, buffer_x, buffer_t, kv_cache, prefix_mask, d) -> velocity
            即 Pi0.sample_actions_step 的返回值减去 buffer_x 后的增量，
            或者直接返回新的 buffer_x（二选一，见注释）。
    """

    def __init__(self, horizon: int, action_dim: int, tick_seconds: float, d_max: int = 5):
        self.buf = RollingBuffer(horizon, action_dim)
        self.lat = LatencyTracker(tick_seconds=tick_seconds, d_max=d_max)
        self._lock = threading.Lock()
        self._cache = None          # (kv_cache, prefix_mask, stamp)
        self._cache_age = 0
        self._stop = threading.Event()

    # ---------------- 后台视觉线程
    def start_vision_worker(self, prefill, get_obs, period_seconds: float = 0.0):
        def _loop():
            while not self._stop.is_set():
                t0 = time.perf_counter()
                obs = get_obs()
                cache = prefill(obs)
                with self._lock:
                    self._cache = (cache[0], cache[1], time.perf_counter())
                dt = time.perf_counter() - t0
                if period_seconds > dt:
                    time.sleep(period_seconds - dt)

        th = threading.Thread(target=_loop, daemon=True)
        th.start()
        return th

    def cached(self):
        with self._lock:
            if self._cache is None:
                return None, None, 0
            kv, mask, stamp = self._cache
            return kv, mask, stamp

    def vision_age_ticks(self, tick_seconds: float) -> int:
        _, _, stamp = self.cached()
        if stamp == 0:
            return 0
        return int(min(round((time.perf_counter() - stamp) / tick_seconds), 8))

    # ---------------- 主循环
    def tick(self, get_obs, step_fn) -> np.ndarray:
        """返回下一个要下发的动作。调用方保证按控制周期调用。"""
        kv, mask, _ = self.cached()
        if kv is None:
            raise RuntimeError("视觉缓存还没就绪，先跑一次 prefill 或等 worker 完成首帧")

        obs = get_obs()
        d = self.lat.d()
        t0 = time.perf_counter()
        new_x = step_fn(obs, self.buf.x, self.buf.t, kv, mask, d)
        self.lat.add(time.perf_counter() - t0)
        # step_fn 返回的已经是更新后的 buffer_x，这里只做"取出 + 滑动"，
        # 不要再乘 delta_t（那样会积分两次）
        return self.buf.commit(np.asarray(new_x), d)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- 自检
def _self_test():
    """纯逻辑自检：不依赖 openpi，可在任何机器上跑。"""
    rng = np.random.default_rng(0)
    H, A = 16, 2
    target = np.array([0.3, -0.2], dtype=np.float32)

    buf = RollingBuffer(H, A)
    # warm start 的输入应当是"标准 flow 整块去噪"的结果，这里用理想解 target 代替
    buf.init_from_chunk(np.broadcast_to(target, (H, A)).copy(), 1)

    # 理想速度场：openpi 约定 v = ε - c，x += Δt·v（Δt<0）
    def ideal_velocity(x, t):
        c = np.broadcast_to(target, x.shape)
        tau = 1.0 - t
        denom = np.clip(t, 1e-3, None)[:, None]
        eps_hat = (x - tau[:, None] * c) / denom
        v = eps_hat - c
        return np.where(t[:, None] < 1e-3, 0.0, v)

    errs = []
    for _ in range(12):
        out = buf.apply(ideal_velocity(buf.x, buf.t), 1)
        errs.append(float(np.abs(out - target).max()))
    assert max(errs) < 1e-4, errs
    print(f"RollingBuffer OK, max|err|={max(errs):.2e}")

    tr = LatencyTracker(tick_seconds=0.04, d_max=5)
    for _ in range(10):
        tr.add(0.15)
    assert tr.d() == 4, tr.d()
    for _ in range(20):          # 填满滚动窗口，旧样本被挤出去
        tr.add(0.03)
    assert tr.d() == 1, tr.d()
    print("LatencyTracker OK")


if __name__ == "__main__":
    _self_test()
