# ------------------------------------------------
# ClutterDexGrasp 论文环境复现脚本 (机械臂集成版)
# ------------------------------------------------
# 功能:
# - 新增: 加载并显示一个用户自定义的机械臂URDF。
# - 稳定地在桌面上的围栏内生成杂乱物体场景。
# ------------------------------------------------

import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil

import math
import random
import numpy as np
import os
import argparse
import time

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Isaac Gym Cluttered Scene Generator with GraspNet Models")
    parser.add_argument('--density', type=str, default='sparse',
                        choices=['sparse', 'dense', 'ultra_dense'],
                        help='要生成的场景的杂乱等级')
    parser.add_argument('--headless', action='store_true', help='在无头模式下运行 (不创建图形窗口)')
    args = parser.parse_args()
    return args

def find_graspnet_urdfs(root_path):
    """
    在指定路径下查找 GraspNet 模型的 URDF 文件。
    """
    all_urdf_files = []
    models_dir = os.path.join(root_path, "models")
    if not os.path.exists(models_dir):
        print(f"错误: 在 '{root_path}' 中找不到 'models' 文件夹。")
        return []

    for obj_folder in sorted(os.listdir(models_dir)):
        urdf_path = os.path.join(models_dir, obj_folder, f"{obj_folder}.urdf")
        if os.path.exists(urdf_path):
            relative_path = os.path.relpath(urdf_path, root_path)
            all_urdf_files.append(relative_path)
    
    return all_urdf_files


# --- 主程序开始 ---

# 1. 初始化
args = parse_args()
gym = gymapi.acquire_gym()

# 2. 定义杂乱等级
clutter_density_ranges = {
    'sparse': (4, 8),
    'dense': (9, 15),
    'ultra_dense': (16, 25)
}
min_objects, max_objects = clutter_density_ranges[args.density]
num_objects_to_spawn = random.randint(min_objects, max_objects)
print(f"正在生成一个 '{args.density}' 场景, 将掉落 {num_objects_to_spawn} 个物体。")


# 3. 配置仿真参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.use_gpu = True
compute_device_id = 0
graphics_device_id = 0 if not args.headless else -1
sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# 4. 创建地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 5. 加载资源
# 5.1 GraspNet 物体模型
asset_root_graspnet = "../assets/graspnet"
NUM_MODELS_TO_LOAD = 5

all_asset_files = find_graspnet_urdfs(asset_root_graspnet)

if not all_asset_files:
    print("未能找到任何 GraspNet 模型，程序退出。")
    quit()

if len(all_asset_files) > NUM_MODELS_TO_LOAD:
    asset_files_to_load = random.sample(all_asset_files, NUM_MODELS_TO_LOAD)
else:
    asset_files_to_load = all_asset_files

print(f"共找到 {len(all_asset_files)} 个模型, 为节省显存，本次将随机加载其中的 {len(asset_files_to_load)} 个。")

object_assets = []
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.disable_gravity = False
asset_options.use_mesh_materials = True
asset_options.vhacd_enabled = True
asset_options.vhacd_params = gymapi.VhacdParams()
asset_options.vhacd_params.resolution = 50000
asset_options.linear_damping = 0.2
asset_options.angular_damping = 0.2

for asset_file in asset_files_to_load:
    asset = gym.load_asset(sim, asset_root_graspnet, asset_file, asset_options)
    if asset is None:
        print(f"*** 错误: 加载模型 '{asset_file}' 失败。")
        continue
    
    props = gym.get_asset_rigid_shape_properties(asset)
    for p in props:
        p.friction = 1.0
        p.restitution = 0.1
    gym.set_asset_rigid_shape_properties(asset, props)
    object_assets.append(asset)

if not object_assets:
    print("所有模型都加载失败，程序退出。")
    quit()

# 5.2 加载箱子和桌子模型
asset_root_general = "../assets"
bin_asset_file = "urdf/tray/traybox.urdf"
bin_options = gymapi.AssetOptions()
bin_options.fix_base_link = True
bin_asset = gym.load_asset(sim, asset_root_general, bin_asset_file, bin_options)

props = gym.get_asset_rigid_shape_properties(bin_asset)
for p in props:
    p.friction = 1.0
    p.restitution = 0.1
gym.set_asset_rigid_shape_properties(bin_asset, props)

table_dims = gymapi.Vec3(1.6, 1.6, 0.4)
table_options = gymapi.AssetOptions()
table_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, table_options)

# 5.3 <<< 新增: 加载您的机械臂模型
# =================================================================
# === 请在这里修改为您自己的URDF文件路径 ===
MY_ROBOT_FOLDER = "my_robot" # 机械臂所在的文件夹名
MY_ROBOT_URDF_FILE = "xarm6_allegro_right.urdf" # 您的URDF文件名
# =================================================================
robot_asset_file = os.path.join(MY_ROBOT_FOLDER, MY_ROBOT_URDF_FILE)

robot_asset_options = gymapi.AssetOptions()
robot_asset_options.fix_base_link = True
robot_asset_options.flip_visual_attachments = False # 根据您的URDF调整
robot_asset_options.collapse_fixed_joints = True
robot_asset_options.disable_gravity = True

print(f"正在从 '{robot_asset_file}' 加载机械臂...")
robot_asset = gym.load_asset(sim, asset_root_general, robot_asset_file, robot_asset_options)
if robot_asset is None:
    print(f"*** 错误: 加载机械臂 '{robot_asset_file}' 失败。请检查路径和URDF文件。")
    quit()


# 6. 设置环境
num_envs = 1
envs_per_row = 1
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, envs_per_row)

# 7. 在环境中生成 Actor
# 7.1 创建桌子 Actor
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.0, 0.0, table_dims.z / 2.0)
table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

# 7.2 创建箱子 Actor 并放在桌面上
bin_pose = gymapi.Transform()
bin_pose.p = gymapi.Vec3(0, 0, table_dims.z)
bin_handle = gym.create_actor(env, bin_asset, bin_pose, "bin", 0, 0)
black_color = gymapi.Vec3(0.1, 0.1, 0.1)
gym.set_rigid_body_color(env, bin_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, black_color)

# 7.3 在箱子内生成杂乱物体
drop_area_dims = (0.4, 0.4)
drop_height_above_table = 0.2
for i in range(num_objects_to_spawn):
    asset = random.choice(object_assets)
    pose = gymapi.Transform()
    pose.p.x = np.random.uniform(-drop_area_dims[0] / 2, drop_area_dims[0] / 2)
    pose.p.y = np.random.uniform(-drop_area_dims[1] / 2, drop_area_dims[1] / 2)
    pose.p.z = table_dims.z + drop_height_above_table
    quat = np.random.uniform(-1, 1, 4)
    pose.r = gymapi.Quat(*quat).normalize()
    # <<< 修改: 设置碰撞组，让物体只与机械臂碰撞 (组1)
    actor_handle = gym.create_actor(env, asset, pose, f"object_{i}", 0, 1)

# 7.4 <<< 新增: 创建机械臂 Actor
robot_pose = gymapi.Transform()
# 将机械臂放在桌子旁边
robot_pose.p = gymapi.Vec3(-0.5, 0.0, table_dims.z) 
# 3. 调整姿态: 移除所有旋转，让机械臂默认竖直站立
robot_pose.r = gymapi.Quat(0, 0, 0, 1) # 这是默认的“无旋转”姿态
# --- 修改结束 ---
robot_handle = gym.create_actor(env, robot_asset, robot_pose, "robot", 0, 1)


# 8. 创建观察器 (Viewer)
viewer = None
if not args.headless:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    cam_pos = gymapi.Vec3(1.5, 0, table_dims.z + 1.0) 
    cam_target = gymapi.Vec3(1.0, 1.0, table_dims.z)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 9. 主仿真循环
print("\n仿真已开始。")
if args.headless:
    print("正在无头模式下运行5秒钟...")
frame_count = 0
start_time = time.time()
while True:
    if not args.headless and gym.query_viewer_has_closed(viewer):
        break
    if args.headless and time.time() - start_time > 5.0:
        break
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    if not args.headless:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
    frame_count += 1
# 10. 清理
print(f"仿真结束。成功运行了 {frame_count} 帧。")
if not args.headless:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

