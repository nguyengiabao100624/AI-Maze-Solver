import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
import tempfile
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_launch_description():
    NUM_ROBOTS = 1 # CHỈ 1 XE ĐỂ TEST
    START_X = 0.0
    START_Y = -1.5

    subprocess.run(['pkill', '-9', '-f', 'gz sim'], stderr=subprocess.DEVNULL)
    subprocess.run(['pkill', '-9', '-f', 'ruby.*gz'], stderr=subprocess.DEVNULL)
    subprocess.run(['pkill', '-9', '-f', 'parameter_bridge'], stderr=subprocess.DEVNULL)
    import time; time.sleep(1.0)

    print('🎲 Đang sinh mê cung ngẫu nhiên mới cho bài Test...')
    # subprocess.run(['python3', os.path.join(SCRIPT_DIR, 'generate_maze.py')])
    # subprocess.run(['python3', os.path.join(SCRIPT_DIR, 'plot_current_maze.py')])

    try:
        import json
        with open(os.path.join(SCRIPT_DIR, 'bfs_map.json'), 'r') as f:
            wp = json.load(f)
            START_X = wp['start']['x']
            START_Y = wp['start']['y']
            print(f'📍 Tọa độ xuất phát (CỐ ĐỊNH): ({START_X}, {START_Y})')
    except Exception:
        print(f"📚 Khởi tạo với tọa độ mặc định.")

    nvidia_gpu_env = {
        '__NV_PRIME_RENDER_OFFLOAD': '1',
        '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
        '__EGL_VENDOR_LIBRARY_FILENAMES': '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
    }
    
    # BỎ -s (headless) ĐỂ CÓ THỂ XEM GUI CỦA GAZEBO KHI TEST
    gz_proc = ExecuteProcess(
        cmd=['gz', 'sim', '-r', os.path.join(SCRIPT_DIR, 'maze_world_fixed.sdf')],
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

        spawn_cmd = (
            f"sleep 2 && "
            f"gz service -s /world/baohet/create "
            f"--reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 5000 "
            f"--req 'sdf_filename: \"{tmp_sdf}\", name: \"{robot_name}\", "
            f"pose: {{position: {{x: {START_X}, y: {START_Y}, z: 0.030}}, "
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

    # TẮT train_ga.py KHI CHẠY TEST
    #cmds.append(ExecuteProcess(...))

    cmds.append(RegisterEventHandler(
        OnProcessExit(target_action=gz_proc, on_exit=[EmitEvent(event=Shutdown())])
    ))

    return LaunchDescription(cmds)
