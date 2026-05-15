import os
import json
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
import tempfile
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_launch_description():
    NUM_ROBOTS = 16
    
    subprocess.run(['pkill', '-9', '-f', 'gz sim'], stderr=subprocess.DEVNULL)
    subprocess.run(['pkill', '-9', '-f', 'ruby.*gz'], stderr=subprocess.DEVNULL)
    subprocess.run(['pkill', '-9', '-f', 'parameter_bridge'], stderr=subprocess.DEVNULL)
    import time; time.sleep(1.0)

    bfs_map_path = os.path.join(SCRIPT_DIR, 'bfs_map_multi.json')
    sdf_path = os.path.join(SCRIPT_DIR, 'multi_maze_world.sdf')
    if not os.path.exists(bfs_map_path) or not os.path.exists(sdf_path):
        print('🎲 Đang sinh 16 mê cung ngẫu nhiên MỚI...')
        subprocess.run(['python3', os.path.join(SCRIPT_DIR, 'src', 'environment', 'multi_maze_generator.py')])
    else:
        print('♻️ Khôi phục sau Crash: Đang tái sử dụng 16 mê cung cũ để giữ nguyên "Đề thi"...')
    
    try:
        with open(os.path.join(SCRIPT_DIR, 'bfs_map_multi.json'), 'r') as f:
            multi_bfs = json.load(f)
    except Exception as e:
        print(f"❌ Lỗi đọc bfs_map_multi.json: {e}")
        return LaunchDescription([])

    nvidia_gpu_env = {
        '__NV_PRIME_RENDER_OFFLOAD': '1',
        '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
        '__EGL_VENDOR_LIBRARY_FILENAMES': '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
    }
    
    gz_proc = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', os.path.join(SCRIPT_DIR, 'multi_maze_world.sdf')],
        output='log',
        additional_env=nvidia_gpu_env
    )
    cmds = [gz_proc]

    with open(os.path.join(SCRIPT_DIR, 'robot_bao_template.sdf'), 'r') as f:
        robot_template = f.read()

    bridge_args = ""

    for i in range(NUM_ROBOTS):
        robot_id = i + 1
        robot_name = f'robot_{robot_id}'
        bitmask = str(2 ** (i % 16))

        sdf_content = robot_template.replace('{{ROBOT_NAME}}', robot_name)
        sdf_content = sdf_content.replace('{{BITMASK}}', bitmask)

        tmp_sdf = os.path.join(tempfile.gettempdir(), f'{robot_name}.sdf')
        with open(tmp_sdf, 'w') as f:
            f.write(sdf_content)

        start_x = multi_bfs['mazes'][i]['start']['x']
        start_y = multi_bfs['mazes'][i]['start']['y']

        spawn_cmd = (
            f"sleep {5 + i * 0.25} && "
            f"gz service -s /world/baohet/create "
            f"--reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 5000 "
            f"--req 'sdf_filename: \"{tmp_sdf}\", name: \"{robot_name}\", "
            f"pose: {{position: {{x: {start_x}, y: {start_y}, z: 0.030}}, "
            f"orientation: {{x: 0, y: 0, z: 0, w: 1}}}}'"
        )

        cmds.append(ExecuteProcess(
            cmd=['bash', '-c', spawn_cmd],
            output='screen'
        ))

        bridge_args += f" /{robot_name}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
        bridge_args += f" /{robot_name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"
        bridge_args += f" /{robot_name}/ground_truth@nav_msgs/msg/Odometry[gz.msgs.Odometry"

    bridge_args += " /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"

    cmds.append(ExecuteProcess(
        cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge'] + bridge_args.strip().split(),
        output='screen'
    ))

    train_cmd = f"sleep 15 && export PYTHONUNBUFFERED=1 && python3 -u {os.path.join(SCRIPT_DIR, 'train_multi_ga.py')}"
    train_proc = ExecuteProcess(
        cmd=['bash', '-c', train_cmd],
        output='screen'
    )
    cmds.append(train_proc)

    cmds.append(RegisterEventHandler(
        OnProcessExit(target_action=train_proc, on_exit=[EmitEvent(event=Shutdown())])
    ))
    cmds.append(RegisterEventHandler(
        OnProcessExit(target_action=gz_proc, on_exit=[EmitEvent(event=Shutdown())])
    ))

    return LaunchDescription(cmds)
