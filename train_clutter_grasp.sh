#!/bin/bash

# ------------------------------------------------
# train_clutter_grasp.sh (最终修正版)
#
# 修正了 `--cfg_train` 参数，使其正确指向我们自己的 PPO 配置文件。
# 移除了与 Hydra 冲突的 `train=` 参数。
# ------------------------------------------------

# 1. 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gym

# 2. 设置 PYTHONPATH
export PYTHONPATH="/home/evan/isaacgym/python:/home/evan/IsaacGymEnvs:$PYTHONPATH"

# 2. 调用 train.py 并传入正确的参数
#    --cfg_train: [核心修正] 明确指定使用哪个训练配置文件
#    task: [保持不变] 仍然需要它来加载任务配置
python src/train.py \
    --cfg_train ClutterGraspPPO \
    task=ClutterGrasp \
    headless=False \
    num_envs=64 \
    --exp_name='ClutterGrasp' \
    --logdir='logs/' \
    --run_device_id=0

