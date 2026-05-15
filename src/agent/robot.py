"""
Robot Agent - v4.0 PRODUCTION
=============================
NGUYÊN TẮC: Mọi logic game (chết, phần thưởng, kẹt) đều diễn ra SYNCHRONOUSLY
bên trong hàm step(). Không dùng ROS callback để quyết định sống/chết.
"""

import math
import numpy as np

# ===================== OBSERVATION =====================
NUM_LIDAR_RAYS = 48          # Tia vật lý (sensor Gazebo) — giữ cao để check va chạm chính xác
AI_LIDAR_RAYS = 24           # Tia AI nhận được (downsample từ 48) — giữ gọn để não dễ học
LIDAR_CLIP_RANGE = 1.0       # Giới hạn tầm nhìn Lidar là 1.0m
OBS_DIM = AI_LIDAR_RAYS      # Không gian quan sát = 24 tia gom từ 48 tia vật lý

# ===================== KINEMATICS LIMITS =====================
MAX_LINEAR_SPEED = 0.35      # Cân bằng: đủ nhanh để không timeout, đủ chậm để ôm cua
MAX_ANGULAR_SPEED = 1.75     # Cân bằng: đủ gắt để rẽ trong hành lang hẹp, không quá gắt gây xoay tròn

# ===================== PHYSICS & ENVIRONMENT =====================
WALL_DEATH_DIST = 0.14       # Ngưỡng đâm tường (mét) — giữ chuẩn 0.14 để training ổn định
GOAL_RADIUS = 0.30           # Bán kính đến đích
MAP_BOUND = 1.55             # OOB: Tránh trường hợp văng ra xa hoặc lách ra ngoài rìa mê cung



class RobotAgent:
    def __init__(self, agent_id, start_x, start_y, goal_x, goal_y, bfs_data):
        self.id = agent_id
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.bfs_data = bfs_data
        
        # Max BFS distance
        grid = bfs_data['bfs_grid']
        cs = bfs_data['cell_size']
        max_val = max(v for row in grid for v in row if v != -1)
        self.max_bfs_dist = max(max_val * cs + 1.0, 1.0)
        
        # State
        self.x = start_x
        self.y = start_y
        self.yaw = 0.0
        self.lidar = np.ones(48) * 12.0
        
        self.cmd_x = 0.0
        self.cmd_az = 0.0
        self.last_x = start_x
        self.last_y = start_y
        
        self.reset()
    
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.yaw = 0.0 # Nhìn thẳng theo trục X (Forward)
        
        # Lidar set to max to avoid premature death before first update
        self.lidar = np.ones(48) * 12.0
        self.lidar_updated = False
        
        self.steps = 0
        self.cmd_x = 0.0
        self.cmd_az = 0.0
        
        # Grace period steps (với 20Hz, đặt về 0 để ép AI cẩn thận ngay từ đầu)
        self.grace_steps = 0
        
        self.last_x = self.start_x
        self.last_y = self.start_y
        self.next_wx = self.goal_x
        self.next_wy = self.goal_y
        
        self.start_bfs_dist = self.get_bfs_distance(self.start_x, self.start_y)
        self.min_bfs_reached = self.start_bfs_dist
        
        # Stagnation detection
        self.last_progress_step = 0    # Step cuối cùng có tiến bộ BFS
        
        self.is_dead = False
        self.is_done = False
        self.death_why = ''
        
        return self._get_obs()
    
    def get_bfs_distance(self, x, y):
        cx = self.bfs_data['half_x']
        cy = self.bfs_data['half_y']
        cs = self.bfs_data['cell_size']
        grid = self.bfs_data['bfs_grid']
        rows, cols = self.bfs_data['maze_size']
        
        # TRỤC X LÀ ROWS (Tiến-Lùi), TRỤC Y LÀ COLS (Trái-Phải)
        row = int(math.floor((x + cx) / cs))
        col = int(math.floor((y + cy) / cs))
        
        # Nếu ra khỏi mảng -> Phạt nặng để cấm tuyệt đối trò "lùi ra ngoài hack điểm"
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return self.max_bfs_dist * 2.0

        
        bfs_val = grid[row][col]
        if bfs_val == -1:
            return self.max_bfs_dist
        if bfs_val == 0:
            self.next_wx, self.next_wy = self.goal_x, self.goal_y
            return max(math.hypot(x - self.goal_x, y - self.goal_y), 0.001)
        
        best_val, best_nr, best_nc = bfs_val, row, col
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                v = grid[nr][nc]
                # CHỈ CHẤP NHẬN Ô KẾ TIẾP TRÊN ĐƯỜNG GIẢI (v == bfs_val - 1)
                if v != -1 and v == bfs_val - 1:
                    best_val, best_nr, best_nc = v, nr, nc
                    break
        
        # Tọa độ cell center thực tế (X-Forward Standard)
        self.next_wx = best_nr * cs - cx + cs / 2.0
        self.next_wy = best_nc * cs - cy + cs / 2.0
        return best_val * cs + math.hypot(x - self.next_wx, y - self.next_wy)
    
    def _get_obs(self):
        clipped = np.clip(self.lidar, 0.0, LIDAR_CLIP_RANGE)
        rays = 1.0 - (clipped / LIDAR_CLIP_RANGE)
        # Thuật toán Max-Pooling: Gộp nhóm tia vật lý để không lọt lưới vật cản nhỏ.
        # Vì rays càng lớn (gần 1.0) nghĩa là vật cản càng gần, ta lấy hàm Max.
        if len(rays) == 48 and AI_LIDAR_RAYS == 24:
            # Gom 48 tia thành 24 cặp (mỗi cặp 2 tia), lấy giá trị lớn nhất của từng cặp
            lidar_obs = rays.reshape(24, 2).max(axis=1).astype(np.float32)
        else:
            # Fallback nếu số lượng tia bị lẻ
            idx = np.linspace(0, len(rays) - 1, AI_LIDAR_RAYS, dtype=int)
            lidar_obs = rays[idx].astype(np.float32)
        
        # --- COMPASS INPUTS ---
        # 1. Normalized distance to waypoint
        dist_to_wp = math.hypot(self.next_wx - self.x, self.next_wy - self.y)
        norm_dist = np.clip(dist_to_wp / 2.0, 0.0, 1.0)  # Chuẩn hóa (tối đa 2 mét)
        
        # 2. Relative heading to waypoint
        target_angle = math.atan2(self.next_wy - self.y, self.next_wx - self.x)
        relative_angle = target_angle - self.yaw
        
        # Đưa góc lệch về khoảng [-pi, pi]
        while relative_angle > math.pi:
            relative_angle -= 2 * math.pi
        while relative_angle < -math.pi:
            relative_angle += 2 * math.pi
            
        norm_angle = relative_angle / math.pi  # Chuẩn hóa về [-1.0, 1.0]
        
        return np.concatenate([lidar_obs, [norm_dist, norm_angle]]).astype(np.float32)

    def step(self, action_continuous, max_steps):
        if self.is_dead or self.is_done:
            return self._get_obs(), 0.0, True, {'reason': self.death_why}
        
        self.steps += 1
        
        v_left, v_right = action_continuous
        self.cmd_x = ((v_left + v_right) / 2.0) * MAX_LINEAR_SPEED
        self.cmd_az = (v_right - v_left) * MAX_ANGULAR_SPEED
        
        # 1. Về đích
        dist_to_goal = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
        if dist_to_goal < GOAL_RADIUS:
            self.is_done = True
            self.death_why = 'Goal'
            self.cmd_x, self.cmd_az = 0.0, 0.0
            return self._get_obs(), 0.0, True, {'reason': 'Goal'}
        
        # 2. Đâm tường
        min_lidar = float(np.min(self.lidar))
        if min_lidar < WALL_DEATH_DIST:
            self.is_dead = True
            self.death_why = 'Wall'
            self.cmd_x, self.cmd_az = 0.0, 0.0
            return self._get_obs(), 0.0, True, {'reason': 'Wall'}
            
        # 3. Timeout / OOB
        if self.steps >= max_steps or abs(self.x) > MAP_BOUND or abs(self.y) > MAP_BOUND:
            self.is_dead = True
            self.death_why = 'Timeout'
            self.cmd_x, self.cmd_az = 0.0, 0.0
            return self._get_obs(), 0.0, True, {'reason': 'Timeout'}
            
        # --- CẬP NHẬT KHOẢNG CÁCH BFS ---
        cur_bfs = self.get_bfs_distance(self.x, self.y)
        if cur_bfs < self.min_bfs_reached - 0.05:
            self.min_bfs_reached = cur_bfs
            self.last_progress_step = self.steps
            
        # --- PHÁT HIỆN BẾ TẮC ---
        if self.steps - self.last_progress_step > 250:
            self.is_dead = True
            self.death_why = 'Timeout'
            self.cmd_x, self.cmd_az = 0.0, 0.0
            return self._get_obs(), 0.0, True, {'reason': 'Timeout'}
        
        return self._get_obs(), 0.0, False, {}

    def get_fitness(self, max_steps):
        # ═══ TÍNH ĐIỂM FITNESS (ĐƠN GIẢN & HOÀN HẢO) ═══
        progress = self.start_bfs_dist - self.min_bfs_reached
        progress = min(progress, self.max_bfs_dist)
        fitness = max(0.0, progress)
        
        # THƯỞNG VỀ ĐÍCH
        if self.death_why == 'Goal':
            fitness += 50.0
            
        return fitness

