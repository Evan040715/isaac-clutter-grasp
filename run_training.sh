#!/bin/bash

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gym

# 设置 Python 路径
export PYTHONPATH="/home/evan/isaacgym/python:/home/evan/IsaacGymEnvs:$PYTHONPATH"

# 切换到项目目录
cd /home/evan/isaac-clutter-grasp

# 运行训练脚本
python src/train.py task=ClutterGrasp "$@"

