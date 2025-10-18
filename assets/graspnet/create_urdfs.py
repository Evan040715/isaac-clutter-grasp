import os

# URDF 文件模板
# 这个模板定义了一个简单的、只有一个链接的物体
# 它会使用 textured.obj 作为视觉模型和碰撞模型
URDF_TEMPLATE = """
<robot name="{object_name}">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="textured.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="textured.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <!-- 这里的惯性参数是估算的，但对于掉落仿真已经足够 -->
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
</robot>
"""

def create_urdf_for_models(models_dir):
    """
    遍历所有模型文件夹，并为每个模型创建一个URDF文件。
    """
    if not os.path.isdir(models_dir):
        print(f"错误: 找不到目录 '{models_dir}'")
        print("请确保此脚本与 'models' 文件夹在同一目录下。")
        return

    print(f"正在为 '{models_dir}' 中的模型生成URDF文件...")
    
    model_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
    
    for folder_name in model_folders:
        object_path = os.path.join(models_dir, folder_name)
        urdf_file_path = os.path.join(object_path, f"{folder_name}.urdf")
        
        # 检查 textured.obj 是否存在
        if not os.path.exists(os.path.join(object_path, "textured.obj")):
            print(f"警告: 在 '{object_path}' 中找不到 'textured.obj'，跳过。")
            continue

        # 填充模板并写入文件
        urdf_content = URDF_TEMPLATE.format(object_name=folder_name)
        with open(urdf_file_path, 'w') as f:
            f.write(urdf_content)
            
    print(f"完成！为 {len(model_folders)} 个模型生成了URDF文件。")

if __name__ == "__main__":
    # 假设 'models' 文件夹就在当前目录下
    current_directory = os.path.dirname(os.path.abspath(__file__))
    models_directory = os.path.join(current_directory, "models")
    
    create_urdf_for_models(models_directory)
