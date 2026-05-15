import math
import numpy as np

NUM_LIDAR_RAYS = 48
AI_LIDAR_RAYS = 24
LIDAR_CLIP_RANGE = 1.0
OBS_DIM = AI_LIDAR_RAYS

MAX_LINEAR_SPEED = 0.35
MAX_ANGULAR_SPEED = 1.75
WALL_DEATH_DIST = 0.140
GOAL_RADIUS = 0.30
MAP_BOUND_LOCAL = 1.55 # from center of its specific maze

class MultiRobotAgent:
    def __init__(self, agent_id, maze_data, global_bfs_data):
        self.id = agent_id
        self.maze_data = maze_data
        self.global_bfs_data = global_bfs_data
        
        self.offset_x = maze_data['offset_x']
        self.offset_y = maze_data['offset_y']
        
        self.start_x = maze_data['start']['x']
        self.start_y = maze_data['start']['y']
        self.goal_x = maze_data['goal']['x']
        self.goal_y = maze_data['goal']['y']
        
        grid = maze_data['bfs_grid']
        cs = global_bfs_data['cell_size']
        max_val = max(v for row in grid for v in row if v != -1)
        self.max_bfs_dist = max(max_val * cs + 1.0, 1.0)
        
        self.x = self.start_x
        self.y = self.start_y
        self.yaw = 0.0
        self.lidar = np.ones(48) * 12.0
        
        self.cmd_x = 0.0
        self.cmd_az = 0.0
        self.reset()
    
    def reset(self):
        self.x = self.start_x
        self.y = self.start_y
        self.yaw = 0.0
        
        self.lidar = np.ones(48) * 12.0
        self.lidar_updated = False
        
        self.steps = 0
        self.cmd_x = 0.0
        self.cmd_az = 0.0
        
        self.next_wx = self.goal_x
        self.next_wy = self.goal_y
        
        self.start_bfs_dist = self.get_bfs_distance(self.start_x, self.start_y)
        self.min_bfs_reached = self.start_bfs_dist
        
        self.last_progress_step = 0
        
        self.is_dead = False
        self.is_done = False
        self.death_why = ''
        
        return self._get_obs()
    
    def get_bfs_distance(self, x, y):
        cx = self.global_bfs_data['half_x']
        cy = self.global_bfs_data['half_y']
        cs = self.global_bfs_data['cell_size']
        grid = self.maze_data['bfs_grid']
        rows, cols = self.global_bfs_data['maze_size']
        
        local_x = x - self.offset_x
        local_y = y - self.offset_y
        
        row = int(math.floor((local_x + cx) / cs))
        col = int(math.floor((local_y + cy) / cs))
        
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
                if v != -1 and v == bfs_val - 1:
                    best_val, best_nr, best_nc = v, nr, nc
                    break
        
        local_next_wx = best_nr * cs - cx + cs / 2.0
        local_next_wy = best_nc * cs - cy + cs / 2.0
        self.next_wx = local_next_wx + self.offset_x
        self.next_wy = local_next_wy + self.offset_y
        
        return best_val * cs + math.hypot(x - self.next_wx, y - self.next_wy)
    
    def _get_obs(self):
        clipped = np.clip(self.lidar, 0.0, LIDAR_CLIP_RANGE)
        rays = 1.0 - (clipped / LIDAR_CLIP_RANGE)
        if len(rays) == 48 and AI_LIDAR_RAYS == 24:
            lidar_obs = rays.reshape(24, 2).max(axis=1).astype(np.float32)
        else:
            idx = np.linspace(0, len(rays) - 1, AI_LIDAR_RAYS, dtype=int)
            lidar_obs = rays[idx].astype(np.float32)
        
        dist_to_wp = math.hypot(self.next_wx - self.x, self.next_wy - self.y)
        norm_dist = np.clip(dist_to_wp / 2.0, 0.0, 1.0)
        
        target_angle = math.atan2(self.next_wy - self.y, self.next_wx - self.x)
        relative_angle = target_angle - self.yaw
        
        while relative_angle > math.pi: relative_angle -= 2 * math.pi
        while relative_angle < -math.pi: relative_angle += 2 * math.pi
            
        norm_angle = relative_angle / math.pi
        
        return np.concatenate([lidar_obs, [norm_dist, norm_angle]]).astype(np.float32)

    def step(self, action_continuous, max_steps):
        if self.is_dead or self.is_done:
            return self._get_obs(), 0.0, True, {'reason': self.death_why}
        
        self.steps += 1
        
        v_left, v_right = action_continuous
        self.cmd_x = ((v_left + v_right) / 2.0) * MAX_LINEAR_SPEED
        self.cmd_az = (v_right - v_left) * MAX_ANGULAR_SPEED
        
        dist_to_goal = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
        if dist_to_goal < GOAL_RADIUS:
            self.is_done = True
            self.death_why = 'Goal'
            self.cmd_x, self.cmd_az = 0.0, 0.0
            return self._get_obs(), 0.0, True, {'reason': 'Goal'}
        
        min_lidar = float(np.min(self.lidar))
        if min_lidar < WALL_DEATH_DIST:
            self.is_dead = True
            self.death_why = 'Wall'
            self.cmd_x, self.cmd_az = 0.0, 0.0
            return self._get_obs(), 0.0, True, {'reason': 'Wall'}
            
        local_x = self.x - self.offset_x
        local_y = self.y - self.offset_y
        if self.steps >= max_steps or abs(local_x) > MAP_BOUND_LOCAL or abs(local_y) > MAP_BOUND_LOCAL:
            self.is_dead = True
            self.death_why = 'Timeout'
            self.cmd_x, self.cmd_az = 0.0, 0.0
            return self._get_obs(), 0.0, True, {'reason': 'Timeout'}
            
        cur_bfs = self.get_bfs_distance(self.x, self.y)
        if cur_bfs < self.min_bfs_reached - 0.05:
            self.min_bfs_reached = cur_bfs
            self.last_progress_step = self.steps
            
        if self.steps - self.last_progress_step > 200:
            self.is_dead = True
            self.death_why = 'Timeout'
            self.cmd_x, self.cmd_az = 0.0, 0.0
            return self._get_obs(), 0.0, True, {'reason': 'Timeout'}
        
        return self._get_obs(), 0.0, False, {}

    def get_fitness(self, max_steps):
        progress = self.start_bfs_dist - self.min_bfs_reached
        progress = min(progress, self.max_bfs_dist)
        fitness = max(0.0, progress)
        
        if self.death_why == 'Goal':
            fitness += 50.0
            
        return fitness
