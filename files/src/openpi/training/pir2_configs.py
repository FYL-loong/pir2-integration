"""πR² 的训练配置：三条腿只差噪声调度。

由 apply_patch.py 复制到 openpi 的 src/openpi/training/ 下，并注册进 _CONFIGS。

================================ EDIT ME ================================
1) DATASET：改成你的 LeRobot 数据集（或用环境变量 PIR2_DATASET）。
2) _data()：相机的 key、state/action 维度按你的机器人改。ALOHA 形态可直接用
   LeRobotAlohaDataConfig；其它形态请照 config.py 里的注释写一个 DataConfigFactory 子类。
3) 训练前必须先算归一化统计量：
       python scripts/compute_norm_stats.py --config-name pir2_staircase
=========================================================================
"""

from __future__ import annotations

import os

import openpi.training.config as _cfgmod  # 只绑定模块对象，避免与 config.py 循环导入
from openpi.models import pi0_config
from openpi.training import weight_loaders

DATASET = os.environ.get("PIR2_DATASET", "your-org/your-dataset")

# π0.5 换成 gs://openpi-assets/checkpoints/pi05_base/params
BASE_CKPT = os.environ.get("PIR2_BASE_CKPT", "gs://openpi-assets/checkpoints/pi0_base/params")

# 论文真机设置：25 Hz，H=50（2 秒），d_max=5 tick（200 ms）
ACTION_HORIZON = int(os.environ.get("PIR2_HORIZON", "50"))
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


def _cfg(name: str, mode: str, *, lora: bool = False):
    kwargs = {}
    if lora:
        model = pi0_config.Pi0Config(
            pir2_mode=mode,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=ACTION_HORIZON,
            d_max=D_MAX,
            d_max_vis=D_MAX_VIS,
        )
        kwargs["ema_decay"] = None
        kwargs["freeze_filter"] = model.get_freeze_filter()
    else:
        model = pi0_config.Pi0Config(
            pir2_mode=mode,
            action_horizon=ACTION_HORIZON,
            d_max=D_MAX,
            d_max_vis=D_MAX_VIS,
        )

    return _cfgmod.TrainConfig(
        name=name,
        model=model,
        data=_data(),
        weight_loader=weight_loaders.CheckpointWeightLoader(BASE_CKPT),
        num_train_steps=NUM_STEPS,
        batch_size=BATCH,
        save_interval=2000,
        **kwargs,
    )


def all_configs():
    """三条腿：flow（基线）/ rtc（基线）/ staircase（πR²）。

    共用同一份数据、同一个 checkpoint、同样的步数与 batch，
    只有 model.pir2_mode 不同 —— 这是可比性的唯一来源。
    """
    return [
        _cfg("pir2_flow", "flow"),
        _cfg("pir2_rtc", "rtc"),
        _cfg("pir2_staircase", "staircase"),
    ]
