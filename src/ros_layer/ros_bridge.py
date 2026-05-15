"""
ROS 2 Bridge - v4.0 PRODUCTION
==============================
NGUYÊN TẮC: Bridge CHỈ làm 2 việc:
1. CẬP NHẬT dữ liệu (lidar, odometry) vào robot objects
2. GỬI lệnh (cmd_vel) xuống Gazebo

KHÔNG check wall collision. KHÔNG có game logic.
Tất cả game logic nằm trong robot.py step().
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
import math
import numpy as np
import subprocess
import time

LIDAR_MIN_RANGE = 0.022
LIDAR_MAX_RANGE = 12.0


class ROSBridgeNode(Node):
    def __init__(self, robots):
        super().__init__('ppo_bridge')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.robots = robots
        self.pubs = {}
        
        for r in self.robots:
            rid = r.id
            self.pubs[rid] = self.create_publisher(Twist, f'/robot_{rid}/cmd_vel', 10)
            self.create_subscription(
                Odometry, f'/robot_{rid}/ground_truth',
                lambda msg, robot_id=rid: self._odom_cb(msg, robot_id),
                qos_profile_sensor_data)
            self.create_subscription(
                LaserScan, f'/robot_{rid}/scan',
                lambda msg, robot_id=rid: self._scan_cb(msg, robot_id),
                qos_profile_sensor_data)
    
    def _odom_cb(self, msg, robot_id):
        """Cập nhật vị trí + hướng robot."""
        r = self.robots[robot_id - 1]
        r.x = msg.pose.pose.position.x
        r.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        r.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    def _scan_cb(self, msg, robot_id):
        """Cập nhật dữ liệu lidar. KHÔNG check collision."""
        if len(msg.ranges) == 0:
            return
        r = self.robots[robot_id - 1]
        raw = np.array(msg.ranges, dtype=np.float32)
        # Clean: NaN/Inf → max range, below min_range → min_range
        raw = np.nan_to_num(raw, nan=LIDAR_MAX_RANGE,
                            posinf=LIDAR_MAX_RANGE, neginf=LIDAR_MAX_RANGE)
        raw = np.clip(raw, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE)
        # Sensor đã cấu hình 48 tia khớp AI input, downsample nếu cần
        if len(raw) == 48:
            r.lidar = raw
        else:
            idx = np.linspace(0, len(raw) - 1, 48, dtype=int)
            r.lidar = raw[idx]
        
        r.lidar_updated = True
    
    def publish_cmd(self, robot_id, lx, ly, az):
        cmd = Twist()
        cmd.linear.x = float(lx)
        cmd.linear.y = float(ly)
        cmd.angular.z = float(az)
        self.pubs[robot_id].publish(cmd)
    
    def stop_all(self, robots):
        """Dừng tất cả xe."""
        for _ in range(5):
            for r in robots:
                self.publish_cmd(r.id, 0.0, 0.0, 0.0)
            time.sleep(0.1)
    
    def physical_reset(self, robots):
        """Teleport robots về vị trí xuất phát (SONG SONG để tối đa tốc độ)."""
        # Dừng xe VÀ CHỜ bánh xe dừng hẳn TRƯỚC KHI teleport
        for _ in range(15):
            for r in robots:
                self.publish_cmd(r.id, 0.0, 0.0, 0.0)
            time.sleep(0.05)
        
        # Teleport SONG SONG tất cả xe cùng lúc
        z_val = 0.030
        procs = []
        for r in robots:
            cmd = (
                f"gz service -s /world/baohet/set_pose "
                f"--reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 3000 "
                f"--req 'name: \"robot_{r.id}\", "
                f"position: {{x: {r.start_x}, y: {r.start_y}, z: {z_val}}}, "
                f"orientation: {{x: 0, y: 0, z: 0, w: 1}}'"
            )
            procs.append(subprocess.Popen(cmd, shell=True,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        
        # Chờ tất cả hoàn thành
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        
        # Dừng lại sau teleport (triệt tiêu momentum)
        for _ in range(10):
            for r in robots:
                self.publish_cmd(r.id, 0.0, 0.0, 0.0)
            time.sleep(0.05)
