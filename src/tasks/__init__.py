# Copyright (c) 2021-2023, NVIDIA Corporation
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from typing import Any, Dict, Optional

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig, OmegaConf

# 只保留我们自定义的 ClutterGrasp 任务的导入和注册
from .clutter_grasp_task import ClutterGraspTask
isaacgym_task_map["ClutterGrasp"] = ClutterGraspTask


def print_cfg(d: Dict, indent: int = 0) -> None:
    """打印环境配置"""
    for key, value in d.items():
        if isinstance(value, dict):
            print_cfg(value, indent + 1)
        else:
            print("  |   " * indent + "  |-- {}: {}".format(key, value))


def parse_hydra_config(
    task_name: str = "",
    isaacgymenvs_path: str = "",
    show_cfg: bool = True,
    args: Optional[Any] = None,
    config_file: str = "config",
) -> DictConfig:
    """解析Hydra配置"""
    import isaacgym
    import isaacgymenvs
    from hydra._internal.hydra import Hydra
    from hydra._internal.utils import create_automatic_config_search_path, get_args_parser
    from hydra.types import RunMode

    # 检查命令行参数中是否定义了任务
    defined = False
    for arg in sys.argv:
        if arg.startswith("task="):
            defined = True
            break
    # 如果未在命令行中定义，则使用函数参数
    if not defined:
        if task_name:
            if "task=" not in " ".join(sys.argv):
                sys.argv.append("task={}".format(task_name))
        else:
            raise ValueError("未定义任务名称。请设置 task_name 参数或使用 task=<task_name> 作为命令行参数")

    # 获取isaacgymenvs的路径
    if isaacgymenvs_path == "":
        isaacgymenvs_path = list(isaacgymenvs.__path__)[0]

    # 项目根目录和配置路径
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_dir, "cfg")

    # 注册omegaconf解析器
    try: OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    except Exception: pass
    try: OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
    except Exception: pass
    try: OmegaConf.register_new_resolver("if", lambda condition, a, b: a if condition else b)
    except Exception: pass
    try: OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)
    except Exception: pass

    # 获取hydra配置
    args = args if args else get_args_parser().parse_args()
    search_path = create_automatic_config_search_path(config_file, None, config_path)
    hydra_object = Hydra.create_main_hydra2(task_name="load_isaacgymenv", config_search_path=search_path)
    config = hydra_object.compose_config(config_file, args.overrides, run_mode=RunMode.RUN)

    if show_cfg:
        print("\nIsaac Gym environment ({})".format(config.task.name))
        print_cfg(omegaconf_to_dict(config.task))

    return config


def create_env_from_config(config: OmegaConf) -> VecTask:
    """根据配置创建环境"""
    cfg = omegaconf_to_dict(config.task)

    # 统一 try 和 except 的调用签名，使其更健壮
    try:
        env = isaacgym_task_map[config.task.name](
            cfg=cfg,
            rl_device=config.rl_device,
            sim_device=config.sim_device,
            graphics_device_id=config.graphics_device_id,
            headless=config.headless,
            virtual_screen_capture=config.capture_video if "capture_video" in config else False,
            force_render=config.force_render if "force_render" in config else False,
        )
    except TypeError as e:
        # 为旧版任务定义提供备用方案
        print(f"初始任务创建失败，尝试备用方案。错误: {e}")
        env = isaacgym_task_map[config.task.name](
            cfg=cfg,
            sim_device=config.sim_device,
            graphics_device_id=config.graphics_device_id,
            headless=config.headless,
        )

    return env


def load_isaacgym_env(
    task_name: str = "",
    isaacgymenvs_path: str = "",
    show_cfg: bool = True,
    args: Optional[Any] = None,
) -> VecTask:
    """加载Isaac Gym环境的主函数"""
    config = parse_hydra_config(task_name=task_name, isaacgymenvs_path=isaacgymenvs_path, show_cfg=show_cfg, args=args)
    env = create_env_from_config(config)
    return env

