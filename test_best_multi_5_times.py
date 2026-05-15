"""
Test Brain Multi (12 Maps) IN SINGLE GAZEBO MAZE — Chạy 5 lần liên tục
===================================================================================
Script này dùng ROS2 để điều khiển 1 robot (robot_1) trong Gazebo (maze_world.sdf) 
và quan sát hành vi của bộ não tốt nhất train từ Multi Maze.

CÁCH DÙNG:
  1. DỪNG process auto_multi.py đang chạy (Ctrl+C)
  2. DỪNG Gazebo nếu đang mở.
  3. Chạy lệnh: ros2 launch start_ai.launch.py (Để mở Gazebo với map đơn)
  4. Mở terminal mới, chạy: python3 test_best_multi_5_times.py
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
import time
import os
import sys
import json
import math
import threading
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.agent.robot import RobotAgent, OBS_DIM
import src.agent.robot
src.agent.robot.WALL_DEATH_DIST = 0.10 # Ghi đè chỉ dành riêng cho bài Test này
from src.ros_layer.ros_bridge import ROSBridgeNode
from src.core.ga_model import GARobotModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_STEPS = 2000

def get_best_multi_champion():
    best_file = 'champions/multi_champion_12_maps_1778058917.npy'
    best_path = os.path.join(SCRIPT_DIR, best_file)
    
    if not os.path.exists(best_path):
        return None, None
        
    return best_path, best_file

def main():
    print("=" * 60)
    print("  🧠 TEST 5 LẦN BỘ NÃO MULTI TỐT NHẤT TRONG GAZEBO")
    print("=" * 60)

    best_path, best_name = get_best_multi_champion()
    if not best_path:
        print("❌ Không tìm thấy file multi_ga_best.npy!")
        return

    weights = np.load(best_path)
    print(f"✅ Đã tải bộ não Quán Quân: {best_name}")

    # Load BFS map
    try:
        with open(os.path.join(SCRIPT_DIR, 'bfs_map.json'), 'r') as f:
            bfs_data = json.load(f)
    except:
        print("❌ Không tìm thấy bfs_map.json. Bạn đã chạy start_ai.launch.py chưa?")
        return

    sx = bfs_data['start']['x']
    sy = bfs_data['start']['y']
    gx = bfs_data['goal']['x']
    gy = bfs_data['goal']['y']

    # Init ROS2
    rclpy.init()

    # Chỉ dùng 1 robot để test
    robots = [RobotAgent(1, sx, sy, gx, gy, bfs_data)]

    ros_node = ROSBridgeNode(robots)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(ros_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    time.sleep(2.0)  # Chờ ROS2 kết nối

    print(f"\n📍 Start: ({sx:.2f}, {sy:.2f})")
    print(f"🎯 Goal: ({gx:.2f}, {gy:.2f})")
    print(f"📊 Bắt đầu chạy 5 lần thử nghiệm...\n")

    results = []

    for test_run in range(1, 6):
        print(f"\n{'─' * 60}")
        print(f"  🧪 LẦN CHẠY THỨ {test_run}/5")
        print(f"{'─' * 60}")

        model = GARobotModel()
        model.set_weights(weights)
        model.reset_memory()

        r = robots[0]

        # Reset robot vật lý
        ros_node.physical_reset(robots)
        r.reset()

        # Chờ Lidar
        timeout = time.time() + 5.0
        while time.time() < timeout:
            if r.lidar_updated:
                break
            time.sleep(0.01)

        if not r.lidar_updated:
            print("  ⚠️ Lidar chưa cập nhật! Bỏ qua test này.")
            results.append("Lỗi Lidar")
            continue

        obs = r._get_obs()
        step_count = 0
        trajectory = [(r.x, r.y)]

        while rclpy.ok():
            time.sleep(0.0)

            if r.is_dead or r.is_done:
                ros_node.publish_cmd(r.id, 0.0, 0.0, 0.0)
                break

            if r.lidar_updated:
                r.lidar_updated = False
                step_count += 1

                v_l, v_r = model.act(obs)
                obs, _, done, info = r.step((v_l, v_r), MAX_STEPS)

                if r.is_dead or r.is_done:
                    ros_node.publish_cmd(r.id, 0.0, 0.0, 0.0)
                    break
                else:
                    ros_node.publish_cmd(r.id, r.cmd_x, 0.0, r.cmd_az)

                trajectory.append((r.x, r.y))

                # In tiến trình mỗi 200 steps
                if step_count % 200 == 0:
                    cur_bfs = r.get_bfs_distance(r.x, r.y)
                    print(f"  Step {step_count}: pos=({r.x:.2f},{r.y:.2f}) bfs={cur_bfs:.2f}")
            else:
                pass

        # Kết quả
        fitness = r.get_fitness(MAX_STEPS)
        dist = sum(math.hypot(trajectory[i+1][0]-trajectory[i][0],
                              trajectory[i+1][1]-trajectory[i][1])
                   for i in range(len(trajectory)-1))

        emoji = '✅ VỀ ĐÍCH!' if r.death_why == 'Goal' else '❌ CHẾT DO ' + r.death_why
        print(f"\n  🏁 Kết quả: {emoji}")
        print(f"  📊 Fitness: {fitness:.2f}")
        print(f"  🏃 Steps: {step_count}")
        print(f"  📏 Quãng đường: {dist:.2f}m")
        print(f"  📍 Vị trí cuối: ({r.x:.2f}, {r.y:.2f})")
        
        results.append(emoji)

        # Dừng robot
        ros_node.publish_cmd(r.id, 0.0, 0.0, 0.0)
        time.sleep(2.0)

    print(f"\n{'═' * 60}")
    print(f"  ✅ TỔNG KẾT 5 LẦN CHẠY: {best_name}")
    print(f"{'═' * 60}")
    for i, res in enumerate(results):
        print(f"  Lần {i+1}: {res}")

    try:
        rclpy.shutdown()
    except:
        pass


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Đã dừng test.")
        try:
            rclpy.shutdown()
        except:
            pass
