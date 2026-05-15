import random
import json
import os
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def _p(filename): 
    return os.path.join(PROJECT_DIR, filename)

def generate_single_maze(rows, cols, cell_size):
    goal_row = rows - 1
    while True:
        start_c = random.randint(0, cols - 1)
        goal_col = random.randint(0, cols - 1)

        h_walls = [[True] * cols for _ in range(rows - 1)]
        v_walls = [[True] * (cols - 1) for _ in range(rows)]
        visited = [[False] * cols for _ in range(rows)]
        
        def get_neighbors(r, c):
            n = []
            if r > 0: n.append((r - 1, c, 'N'))
            if r < rows - 1: n.append((r + 1, c, 'S'))
            if c > 0: n.append((r, c - 1, 'W'))
            if c < cols - 1: n.append((r, c + 1, 'E'))
            random.shuffle(n)
            return n

        start_r = 0
        stack = [(start_r, start_c)]
        visited[start_r][start_c] = True
        
        while stack:
            r, c = stack[-1]
            unvisited = [n for n in get_neighbors(r, c) if not visited[n[0]][n[1]]]
            if unvisited:
                nr, nc, d = unvisited[0]
                if d == 'N': h_walls[r - 1][c] = False
                elif d == 'S': h_walls[r][c] = False
                elif d == 'W': v_walls[r][c - 1] = False
                elif d == 'E': v_walls[r][c] = False
                visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()

        bfs_map = [[-1] * cols for _ in range(rows)]
        queue = deque([(goal_row, goal_col, 0)])
        bfs_map[goal_row][goal_col] = 0
        
        while queue:
            r, c, dist = queue.popleft()
            if r > 0 and not h_walls[r-1][c] and bfs_map[r-1][c] == -1:
                bfs_map[r-1][c] = dist + 1
                queue.append((r-1, c, dist + 1))
            if r < rows - 1 and not h_walls[r][c] and bfs_map[r+1][c] == -1:
                bfs_map[r+1][c] = dist + 1
                queue.append((r+1, c, dist + 1))
            if c > 0 and not v_walls[r][c-1] and bfs_map[r][c-1] == -1:
                bfs_map[r][c-1] = dist + 1
                queue.append((r, c-1, dist + 1))
            if c < cols - 1 and not v_walls[r][c] and bfs_map[r][c+1] == -1:
                bfs_map[r][c+1] = dist + 1
                queue.append((r, c+1, dist + 1))
                
        start_bfs_dist = bfs_map[start_r][start_c]
        if start_bfs_dist >= 20 or (rows * cols < 25): 
            return start_r, start_c, goal_row, goal_col, h_walls, v_walls, bfs_map

def generate_multi_maze_sdf_and_bfs(num_mazes=16, rows=5, cols=5, cell_size=0.6, spacing=10.0):
    thickness = 0.05
    height = 0.5
    half_x = (rows * cell_size) / 2.0
    half_y = (cols * cell_size) / 2.0

    mazes_data = []
    walls_sdf = ""

    def add_wall(name, x, y, sx, sy):
        return (
            f'\n        <collision name="c_{name}"><pose>{x:.4f} {y:.4f} 0.25 0 0 0</pose>'
            f'<geometry><box><size>{sx:.4f} {sy:.4f} {height}</size></box></geometry>'
            f'<surface><contact><collide_bitmask>65535</collide_bitmask></contact></surface></collision>'
            f'\n        <visual name="v_{name}"><visibility_flags>1</visibility_flags>'
            f'<pose>{x:.4f} {y:.4f} 0.25 0 0 0</pose>'
            f'<geometry><box><size>{sx:.4f} {sy:.4f} {height}</size></box></geometry>'
            f'<material><ambient>0.2 0.8 0.3 1</ambient><diffuse>0.2 0.8 0.3 1</diffuse></material></visual>'
        )

    grid_side = int(num_mazes ** 0.5)
    if grid_side * grid_side < num_mazes: grid_side += 1

    for i in range(num_mazes):
        start_r, start_c, goal_row, goal_col, h_walls, v_walls, bfs_map = generate_single_maze(rows, cols, cell_size)
        
        offset_x = (i // grid_side) * spacing
        offset_y = (i % grid_side) * spacing

        def get_x(r): return r * cell_size - half_x + cell_size / 2 + offset_x
        def get_y(c): return c * cell_size - half_y + cell_size / 2 + offset_y

        start_x_final = get_x(start_r) - 0.25 
        start_y_final = get_y(start_c)
        goal_x_final = get_x(goal_row)
        goal_y_final = get_y(goal_col)

        mazes_data.append({
            'id': i,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'start': {'x': start_x_final, 'y': start_y_final, 'c': start_c},
            'goal': {'x': goal_x_final, 'y': goal_y_final, 'c': goal_col},
            'bfs_grid': bfs_map
        })

        walls_sdf += add_wall(f'm{i}_out_L', offset_x, offset_y - half_y, 2 * half_x, thickness)
        walls_sdf += add_wall(f'm{i}_out_R', offset_x, offset_y + half_y, 2 * half_x, thickness)

        bot_y_gate = start_y_final
        bot_l_len = half_y + (bot_y_gate - offset_y) - cell_size/2
        bot_r_len = half_y - (bot_y_gate - offset_y) - cell_size/2
        bot_l_mid = offset_y - half_y + bot_l_len / 2
        bot_r_mid = offset_y + half_y - bot_r_len / 2
        if bot_l_len > 0.001: walls_sdf += add_wall(f'm{i}_out_Bot_L', offset_x - half_x, bot_l_mid, thickness, bot_l_len)
        if bot_r_len > 0.001: walls_sdf += add_wall(f'm{i}_out_Bot_R', offset_x - half_x, bot_r_mid, thickness, bot_r_len)

        top_y_gate = goal_y_final
        top_l_len = half_y + (top_y_gate - offset_y) - cell_size/2
        top_r_len = half_y - (top_y_gate - offset_y) - cell_size/2
        top_l_mid = offset_y - half_y + top_l_len / 2
        top_r_mid = offset_y + half_y - top_r_len / 2
        if top_l_len > 0.001: walls_sdf += add_wall(f'm{i}_out_Top_L', offset_x + half_x, top_l_mid, thickness, top_l_len)
        if top_r_len > 0.001: walls_sdf += add_wall(f'm{i}_out_Top_R', offset_x + half_x, top_r_mid, thickness, top_r_len)

        idx = 0
        for r in range(rows - 1):
            for c in range(cols):
                if h_walls[r][c]:
                    walls_sdf += add_wall(f'm{i}_h_{idx}', get_x(r) + cell_size / 2, get_y(c), thickness, cell_size)
                    idx += 1
        for r in range(rows):
            for c in range(cols - 1):
                if v_walls[r][c]:
                    walls_sdf += add_wall(f'm{i}_v_{idx}', get_x(r), get_y(c) + cell_size / 2, cell_size, thickness)
                    idx += 1

    bfs_json = {
        'maze_size': [rows, cols],
        'cell_size': cell_size,
        'half_x': half_x,
        'half_y': half_y,
        'num_mazes': num_mazes,
        'mazes': mazes_data
    }
    with open(_p('bfs_map_multi.json'), 'w') as f:
        json.dump(bfs_json, f, indent=2)

    world_sdf = f'''
    <model name="ground_plane">
      <static>true</static>
      <pose>{spacing*grid_side/2} {spacing*grid_side/2} 0 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{spacing*grid_side*1.5} {spacing*grid_side*1.5} 0.01</size></box></geometry>
          <surface><contact><collide_bitmask>65535</collide_bitmask></contact></surface></collision>
        <visual name="visual">
          <transparency>0.85</transparency>
          <geometry><box><size>{spacing*grid_side*1.5} {spacing*grid_side*1.5} 0.01</size></box></geometry>
          <material><ambient>0.8 0.8 0.8 0.1</ambient><diffuse>0.8 0.8 0.8 0.1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="multi_maze_{rows}x{cols}">
      <static>true</static><pose>0 0 0 0 0 0</pose><link name="maze_walls">{walls_sdf}</link>
    </model>'''

    sdf = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="baohet">
    <physics name="1ms" type="ignored">
      <max_step_size>0.008</max_step_size>
      <real_time_factor>0</real_time_factor>
      <real_time_update_rate>0</real_time_update_rate>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <scene><background>0.85 0.85 0.9 1</background><ambient>0.5 0.5 0.5 1</ambient></scene>
{world_sdf}
  </world>
</sdf>
'''
    with open(_p('multi_maze_world.sdf'), 'w') as f:
        f.write(sdf)
    print(f"🤖 Đã tạo multi_maze_world.sdf ({num_mazes} mazes) và bfs_map_multi.json")

if __name__ == '__main__':
    generate_multi_maze_sdf_and_bfs(16)
