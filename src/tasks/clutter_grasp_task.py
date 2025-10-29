
import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

import math
import random
import numpy as np
import os
import torch
from omegaconf import OmegaConf, ListConfig

# 用于查找 URDF 文件的辅助函数
def find_graspnet_urdfs(root_path):
    all_urdf_files = []
    models_dir = os.path.join(root_path, "models")
    if not os.path.exists(models_dir):
        return []
    
    for obj_folder in sorted(os.listdir(models_dir)):
        urdf_path = os.path.join(models_dir, obj_folder, f"{obj_folder}.urdf")
        if os.path.exists(urdf_path):
            relative_path = os.path.relpath(urdf_path, root_path)
            all_urdf_files.append(relative_path)
    return all_urdf_files

# 用于观测项规格的辅助类
class ObservationSpec:
    def __init__(self, name, dim, **kwargs):
        self.name = name
        self.dim = dim
        self.shape = kwargs.get("shape", [dim])
        self.attr = kwargs.get("attr", name)
        self.tags = kwargs.get("tags", [])

class ClutterGraspTask(VecTask):
    _xarm_right_init_dof_positions = {
        "joint1": 0.0,
        "joint2":-1,
        "joint3":-0.5,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
    }
    _allegro_hand_init_dof_positions = {
        "joint_0.0": 0,
        "joint_1.0": 0.0,
        "joint_2.0": 0,
        "joint_3.0": 0,
        "joint_4.0": 0.0,
        "joint_5.0": 0.0,
        "joint_6.0": 0,
        "joint_7.0": 0,
        "joint_8.0": 0,
        "joint_9.0": 0,
        "joint_10.0": 0,
        "joint_11.0": 0,
        "joint_12.0": 0,
        "joint_13.0": 0.0,
        "joint_14.0": 0,
        "joint_15.0": 0.0,
    }
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = 0.1

        # --- 为 PPO 添加必需的属性 ---
        self.observation_info = self.cfg["env"]["observationSpace"]
        self.stack_frame_number = self.cfg["env"].get("stackFrameNumber", 1)
        
        # <<< 核心修正: 添加 PPO 日志记录逻辑必需的 object_codes 属性 >>>
        # 我们提供一个简单的占位符列表来满足接口要求。
        self.object_codes = ["clutter_object"]
        self.label_paths = "place_holder"
        self.num_objects = "place_holder"
        self.object_cat = "place_holder"
        self.max_per_cat = "place_holder"
        self.object_geo_level = "place_holder"
        self.object_scale = "place_holder"
        self.reward_type = "place_holder"
        self.mode = "train"
        
        # 初始化每个环境的物体数量
        self.num_objects_per_env = 5

        # --- 首先获取机器人的关节数量 ---
        self._get_robot_dof_count()

        # --- 现在使用已知的关节数量来配置 MDP 空间 ---
        self._configure_mdp_spaces()
        
        # --- 调用父类的构造函数 (这会触发 create_sim) ---
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # --- 获取张量并创建缓冲区 ---
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)
        self.rb_states = gymtorch.wrap_tensor(rb_state_tensor).view(self.num_envs, -1, 13)
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.dof_pos = self.dof_states[..., 0]
        self.dof_vel = self.dof_states[..., 1]

        self.allegro_hand_dof_positions = self.dof_states[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end, 0]
        self.allegro_hand_dof_velocities = self.dof_states[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end, 1]
        
        self.curr_targets_buffer = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.curr_targets = self.curr_targets_buffer[:, self.allegro_hand_dof_start : self.allegro_hand_dof_end]

        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self.target_pos_buf = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        # --- 初始化观测属性以避免错误 ---
        for spec in self._observation_space:
            setattr(self, spec.attr, torch.zeros((self.num_envs, *spec.shape), device=self.device))


    def _get_robot_dof_count(self):
        temp_gym = gymapi.acquire_gym()
        temp_sim_params = gymapi.SimParams()
        temp_sim = temp_gym.create_sim(0, 0, gymapi.SIM_PHYSX, temp_sim_params)
        asset_root = "assets"
        robot_asset_file = "my_robot/xarm6_allegro_right.urdf" 
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.fix_base_link = True
        robot_asset = temp_gym.load_asset(temp_sim, asset_root, robot_asset_file, robot_asset_options)
        self.num_dofs = temp_gym.get_asset_dof_count(robot_asset)
        temp_gym.destroy_sim(temp_sim)

    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        self.asset_root = "assets"
        print(f"Loading assets from root: '{self.asset_root}'")

        table_dims = gymapi.Vec3(0.8, 1.2, 0.4)
        table_options = gymapi.AssetOptions(); table_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_options)

        bin_asset_file = "urdf/tray/traybox.urdf"
        bin_options = gymapi.AssetOptions(); bin_options.fix_base_link = True
        bin_asset = self.gym.load_asset(self.sim, self.asset_root, bin_asset_file, bin_options)

        robot_asset_file = "my_robot/xarm6_allegro_right.urdf"
        robot_asset_options = gymapi.AssetOptions(); robot_asset_options.fix_base_link = True; robot_asset_options.disable_gravity = True
        robot_asset = self.gym.load_asset(self.sim, self.asset_root, robot_asset_file, robot_asset_options)
        
        # num_dofs = self.gym.get_asset_dof_count(robot_asset)
        
        dof_init_positions = [0.0 for _ in range(self.num_dofs)]
        dof_init_velocities = [0.0 for _ in range(self.num_dofs)]
        
        for name, value in self._xarm_right_init_dof_positions.items():
            dof_init_positions[self.gym.find_asset_dof_index(robot_asset, name)] = value
        for name, value in self._allegro_hand_init_dof_positions.items():
            dof_init_positions[self.gym.find_asset_dof_index(robot_asset, name)] = value
            
        self.dof_init_positions = torch.tensor(dof_init_positions).float().to(self.device)
        self.dof_init_velocities = torch.tensor(dof_init_velocities).float().to(self.device)
        
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS); dof_props["stiffness"].fill(400.0); dof_props["damping"].fill(40.0)

        self.num_object_models_to_load = 1
        self.object_assets = []
        
        all_graspnet_files = find_graspnet_urdfs(os.path.join(self.asset_root, "graspnet"))
        
        if len(all_graspnet_files) >= self.num_object_models_to_load:
            print(f"共找到 {len(all_graspnet_files)} 个模型，将随机加载其中的 {self.num_object_models_to_load} 个。")
            files_to_load = random.sample(all_graspnet_files, self.num_object_models_to_load)
        else:
            print(f"找到的模型数量不足 {self.num_object_models_to_load} 个，将加载所有 {len(all_graspnet_files)} 个模型。")
            files_to_load = all_graspnet_files

        for urdf_file in files_to_load:
            opts = gymapi.AssetOptions(); opts.use_mesh_materials = True; opts.vhacd_enabled = True
            opts.vhacd_params.resolution = 100000
            asset = self.gym.load_asset(self.sim, os.path.join(self.asset_root, "graspnet"), urdf_file, opts)
            self.object_assets.append(asset)
        
        if not self.object_assets:
            print("\n*** 错误: 未能成功加载任何物体模型，请检查路径。程序退出。")
            quit()
        
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.robot_handles = []
        self.object_actor_idxs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_ptr)
            
            table_pose = gymapi.Transform(); table_pose.p = gymapi.Vec3(0.0, 0.0, table_dims.z / 2.0)
            self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            
            bin_pose = gymapi.Transform(); bin_pose.p = gymapi.Vec3(0, 0, table_dims.z)
            self.gym.create_actor(env_ptr, bin_asset, bin_pose, "bin", i, 0)

            robot_pose = gymapi.Transform(); robot_pose.p = gymapi.Vec3(-0.5, 0.0, table_dims.z)
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, robot_pose, "robot", i, 1)
            robot_handle_idx = self.gym.get_actor_index(env_ptr, robot_handle, gymapi.DOMAIN_SIM)
            # print(i, robot_handle, robot_handle_idx)
            self.robot_handles.append(robot_handle_idx)
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)
            
            obj_start_idx = self.gym.get_actor_count(env_ptr)
            self.num_objects_per_env = 5
            for j in range(self.num_objects_per_env):
                asset = random.choice(self.object_assets)
                pose = gymapi.Transform()
                pose.p.z = table_dims.z + 0.2
                self.gym.create_actor(env_ptr, asset, pose, f"object_{j}", i, 1)
            obj_end_idx = self.gym.get_actor_count(env_ptr)
            self.object_actor_idxs.append(torch.arange(obj_start_idx, obj_end_idx, device=self.device, dtype=torch.long))
            
        env = self.envs[-1]
        allegro_hand = self.gym.find_actor_handle(env, "robot")
        self.allegro_hand_dof_start = self.gym.get_actor_dof_index(env, allegro_hand, 0, gymapi.DOMAIN_ENV)
        self.allegro_hand_dof_end = self.allegro_hand_dof_start + self.num_dofs

        # 尝试找到末端执行器链接
        EE_LINK_NAMES = ["link_15.0_tip", "link_11.0_tip", "link_7.0_tip", "link_3.0_tip"]
        self.ee_handle_idx = -1
        for link_name in EE_LINK_NAMES:
            self.ee_handle_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_handles[0], link_name)
            if self.ee_handle_idx != -1:
                print(f"找到末端执行器链接: {link_name}")
                break
        
        if self.ee_handle_idx == -1:
            print("警告: 未找到末端执行器链接，使用默认链接")
            # 使用最后一个链接作为末端执行器
            self.ee_handle_idx = self.gym.get_actor_rigid_body_count(self.envs[0], self.robot_handles[0]) - 1
        
        self.set_initial_state()

    def set_initial_state(self):
        self.initial_dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)

    def compute_observations(self, env_ids=None):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.robot_dof_pos = self.dof_pos
        self.robot_dof_vel = self.dof_vel
        self.eef_pos = self.rb_states[:, self.ee_handle_idx, :3]
        self.target_pos = self.target_pos_buf

        observation_dict = self.retrieve_observation_dict()
        self.obs_buf = torch.cat([observation_dict[spec.name].view(self.num_envs, -1) for spec in self._observation_space], dim=-1)
        
        self.obs_dict["obs"] = self.obs_buf
        return self.obs_dict

    def compute_reward(self, actions):
        dist_to_target = torch.norm(self.eef_pos - self.target_pos_buf, dim=-1)
        
        reward = -dist_to_target
        success_bonus = 500.0
        
        is_success = dist_to_target < 0.05
        reward = torch.where(is_success, reward + success_bonus, reward)
        self.reset_buf = torch.where(is_success, torch.ones_like(self.reset_buf), self.reset_buf)

        self.rew_buf[:] = reward
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def reset_idx(self, env_ids):
        pos = self.initial_dof_pos[env_ids]
        self.dof_pos[env_ids] = pos
        self.dof_vel[env_ids] = 0.0
        
        self.allegro_hand_dof_positions[env_ids, :] = self.dof_init_positions
        self.allegro_hand_dof_velocities[env_ids, :] = self.dof_init_velocities

        self.curr_targets[env_ids] = self.dof_init_positions

        robot_indices = torch.tensor(self.robot_handles, dtype=torch.int32, device=self.device)[env_ids].flatten()
        # breakpoint()
        if len(robot_indices) > 0:
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_states), gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
        
        for env_id in env_ids:
            actor_idxs = self.object_actor_idxs[env_id]
            for actor_idx in actor_idxs:
                self.root_states[env_id, actor_idx, 0] = torch_rand_float(-0.15, 0.15, (1,1), self.device)
                self.root_states[env_id, actor_idx, 1] = torch_rand_float(-0.15, 0.15, (1,1), self.device)
                self.root_states[env_id, actor_idx, 2] = 0.45
                # self.root_states[env_id, actor_idx, 3:7] = random_quaternion(1, self.device).squeeze()
            
            target_actor_local_idx = torch.randint(0, self.num_objects_per_env, (1,)).item()
            target_actor_global_idx = actor_idxs[target_actor_local_idx]
            self.target_pos_buf[env_id] = self.root_states[env_id, target_actor_global_idx, :3]
        
        object_global_indices = torch.cat([self.object_actor_idxs[i] for i in env_ids]).flatten()
        if len(object_global_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(object_global_indices.to(torch.int32)), len(object_global_indices))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids.clone()] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.dof_pos + self.actions * self.action_scale
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        self.compute_reward(self.actions)

    def _configure_mdp_spaces(self):
        self.cfg["env"]["observationSpecs"]["__dim__"]["robot_dof"] = self.num_dofs
        
        observation_specs_def = self.cfg["env"]["observationSpecs"]
        dims = observation_specs_def.pop("__dim__")
        
        self._observation_specs = []
        for name, info in observation_specs_def.items():
            shape = info["shape"]
            if not isinstance(shape, (list, tuple, ListConfig)):
                shape = [shape]
            
            resolved_shape = [dims.get(d, d) for d in shape]
            dim = int(np.prod(resolved_shape))
            
            spec = ObservationSpec(name, dim, **info)
            spec.shape = resolved_shape
            self._observation_specs.append(spec)

        self._observation_space = [self._get_observation_spec(name) for name in self.cfg["env"]["observationSpace"]]
        
        self.cfg["env"]["numActions"] = self.num_dofs
        self.cfg["env"]["numObservations"] = sum([spec.dim for spec in self._observation_space])
        
    def export_observation_metainfo(self):
        """ 导出神经网络所需的观测元数据 """
        metainfo = []
        current = 0
        for spec in self._observation_space:
            metainfo.append({ 
                "name": spec.name, 
                "start": current, 
                "end": current + spec.dim,
                "tags": spec.tags,
                "dim": spec.dim
            })
            current += spec.dim
        return metainfo
    
    def _get_observation_spec(self, name: str):
        for spec in self._observation_specs:
            if spec.name == name:
                return spec
        raise ValueError(f"Observation spec '{name}' not found.")

    def retrieve_observation_dict(self):
        obs_dict = {spec.name: getattr(self, spec.attr) for spec in self._observation_space}
        return obs_dict
    
    def reset(self, env_ids=None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        return self.compute_observations()

    def train(self):
        self.training = True

    def eval(self, vis=False):
        self.training = False   
