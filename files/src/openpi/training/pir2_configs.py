"""πR² 的训练配置：三条腿 × 两种底座。

由 apply_patch.py 复制到 openpi 的 src/openpi/training/ 下，并注册进 _CONFIGS。

注册 6 条：
    pir2_flow / pir2_rtc / pir2_staircase                # π0 底座
    pir2_flow_pi05 / pir2_rtc_pi05 / pir2_staircase_pi05 # π0.5 底座 + 连续状态快通道

同一条腿的两种底座共享除 model 以外的一切，三条腿之间只差 pir2_mode。

================================ EDIT ME ================================
1) DATASET：改成你的 LeRobot 数据集（或用环境变量 PIR2_DATASET）。
2) _data()：相机的 key、state/action 维度按你的机器人改。
3) 训练前先算归一化统计量：
       python scripts/compute_norm_stats.py --config-name pir2_staircase_pi05
=========================================================================
"""

from __future__ import annotations

import os

import openpi.training.config as _cfgmod  # 只绑定模块对象，避免与 config.py 循环导入
from openpi.models import pi0_config
from openpi.training import weight_loaders

DATASET = os.environ.get("PIR2_DATASET", "your-org/your-dataset")

CKPT_PI0 = os.environ.get("PIR2_CKPT_PI0", "gs://openpi-assets/checkpoints/pi0_base/params")
CKPT_PI05 = os.environ.get("PIR2_CKPT_PI05", "gs://openpi-assets/checkpoints/pi05_base/params")

# 论文真机设置：25 Hz，H=50（2 秒），d_max=5 tick（200 ms）
ACTION_HORIZON = int(os.environ.get("PIR2_HORIZON", "50"))
ACTION_DIM = int(os.environ.get("PIR2_ACTION_DIM", "32"))
D_MAX = int(os.environ.get("PIR2_D_MAX", "5"))
D_MAX_VIS = int(os.environ.get("PIR2_D_MAX_VIS", "5"))
NUM_STEPS = int(os.environ.get("PIR2_STEPS", "30000"))
BATCH = int(os.environ.get("PIR2_BATCH", "32"))


def _data():
    return _cfgmod.LeRobotAlohaDataConfig(
        repo_id=DATASET,
        base_config=_cfgmod.DataConfig(prompt_from_task=True),
        # 默认去 ./assets/<config_name>/<repo_id> 找 norm stats
        assets=_cfgmod.AssetsConfig(),
    )


def _cfg(name: str, mode: str, *, pi05: bool, lora: bool = False):
    kwargs = {}
    model_kwargs = dict(
        pi05=pi05,
        pir2_mode=mode,
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        d_max=D_MAX,
        d_max_vis=D_MAX_VIS,
    )
    if pi05:
        # 需要 --pi05-state 补丁；没打时给一个能看懂的报错，而不是 TypeError
        import dataclasses as _dc

        _fields = {f.name for f in _dc.fields(pi0_config.Pi0Config)}
        if "pir2_pi05_state" not in _fields:
            raise RuntimeError(
                "π0.5 配置需要连续状态快通道补丁：先跑 "
                "`python apply_patch.py --openpi-dir . --pi05-state`，再重新 import。"
            )
        # 关键：保留 prompt 里的离散状态，同时给动作专家加一个连续状态 token
        model_kwargs["pir2_pi05_state"] = True

    if lora:
        model = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            **model_kwargs,
        )
        kwargs["ema_decay"] = None
        kwargs["freeze_filter"] = model.get_freeze_filter()
    else:
        model = pi0_config.Pi0Config(**model_kwargs)

    return _cfgmod.TrainConfig(
        name=name,
        model=model,
        data=_data(),
        weight_loader=weight_loaders.CheckpointWeightLoader(CKPT_PI05 if pi05 else CKPT_PI0),
        num_train_steps=NUM_STEPS,
        batch_size=BATCH,
        save_interval=2000,
        **kwargs,
    )


def all_configs():
    """三条腿 × 两种底座。

    同一底座的三个配置共用数据、checkpoint、步数与 batch，只有 pir2_mode 不同。
    """
    out = []
    for pi05 in (False, True):
        suffix = "_pi05" if pi05 else ""
        for mode in ("flow", "rtc", "staircase"):
            out.append(_cfg(f"pir2_{mode}{suffix}", mode, pi05=pi05))
    return out
