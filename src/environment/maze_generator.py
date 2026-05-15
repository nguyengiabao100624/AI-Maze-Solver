import random
import json
import os
from collections import deque

# File này nằm ở AI/src/environment/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Nhảy lên 2 cấp để lấy thư mục AI/
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def _p(filename): 
    return os.path.join(PROJECT_DIR, filename)

def generate_maze_sdf_and_bfs(rows=5, cols=5, cell_size=0.6):
    """
    Tạo Mê cung tĩnh chuẩn X-Forward:
    - Trục X: Chạy từ hàng 0 đến hàng N-1 (Tiến/Lùi)
    - Trục Y: Chạy từ cột 0 đến cột N-1 (Trái/Phải)
    """
    cell = cell_size
    thickness = 0.05
    height = 0.5
    
    # ── 1. Khởi tạo Mê cung bằng DFS & ÉP ĐỘ KHÓ >= 20 ──────────
    # ĐÍCH LÀ HÀNG CUỐI (Row N-1)
    goal_row = rows - 1
    
    while True:
        # Chọn ngẫu nhiên cửa xuất phát (Top) và cửa đích (Bottom)
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

        # DFS từ ô (0, start_c)
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

        # ── 2. Tạo BFS Distance Map từ Đích ──────────
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
                
        # KIỂM TRA ĐỘ DÀI ĐƯỜNG ĐI
        start_bfs_dist = bfs_map[start_r][start_c]
        
        # Yêu cầu BFS từ vạch xuất phát phải >= 20
        # (Chỉ ép giới hạn nếu map có khả năng dài >= 20, ví dụ 5x5 có 25 ô thì dài tối đa là 24)
        if (12 <= start_bfs_dist <= 16) or (rows * cols < 16):
            print(f"🎯 Đã tìm thấy mê cung có độ khó đạt chuẩn! (Quãng đường BFS = {start_bfs_dist} ô)")
            break


    # ── 3. Sinh đồ họa vật lý Mê cung (SDF) ──────────
    half_x = (rows * cell) / 2.0
    half_y = (cols * cell) / 2.0

    def get_x(r): return r * cell - half_x + cell / 2
    def get_y(c): return c * cell - half_y + cell / 2

    # TỌA ĐỘ XUẤT PHÁT: Lùi lại 0.25m so với tâm ô đầu tiên
    start_x_final = get_x(start_r) - 0.25 
    start_y_final = get_y(start_c)

    # ĐÍCH
    goal_x_final = get_x(goal_row)
    goal_y_final = get_y(goal_col)

    bfs_json = {
        'maze_size': [rows, cols],
        'cell_size': cell,
        'half_x': half_x,
        'half_y': half_y,
        'start': {'x': start_x_final, 'y': start_y_final, 'c': start_c},
        'goal': {'x': goal_x_final, 'y': goal_y_final, 'c': goal_col},
        'bfs_grid': bfs_map,
        'h_walls': h_walls,
        'v_walls': v_walls
    }
    with open(_p('bfs_map.json'), 'w') as f:
        json.dump(bfs_json, f, indent=2)

    walls_sdf = ""
    def add_wall(name, x, y, sx, sy):
        # Bitmask 65535 (16-bit)
        return (
            f'\n        <collision name="c_{name}"><pose>{x:.4f} {y:.4f} 0.25 0 0 0</pose>'
            f'<geometry><box><size>{sx:.4f} {sy:.4f} {height}</size></box></geometry>'
            f'<surface><contact><collide_bitmask>65535</collide_bitmask></contact></surface></collision>'
            f'\n        <visual name="v_{name}"><visibility_flags>1</visibility_flags>'
            f'<pose>{x:.4f} {y:.4f} 0.25 0 0 0</pose>'
            f'<geometry><box><size>{sx:.4f} {sy:.4f} {height}</size></box></geometry>'
            f'<material><ambient>0.2 0.3 0.8 1</ambient><diffuse>0.2 0.3 0.8 1</diffuse></material></visual>'
        )

    # Tường bao biên
    walls_sdf += add_wall('out_L', 0, -half_y, 2 * half_x, thickness)
    walls_sdf += add_wall('out_R', 0,  half_y, 2 * half_x, thickness)

    # Đầu vào (-X)
    bot_y_gate = start_y_final
    bot_l_len = half_y + bot_y_gate - cell/2
    bot_r_len = half_y - bot_y_gate - cell/2
    bot_l_mid = -half_y + bot_l_len / 2
    bot_r_mid =  half_y - bot_r_len / 2
    if bot_l_len > 0.001: walls_sdf += add_wall('out_Bot_L', -half_x, bot_l_mid, thickness, bot_l_len)
    if bot_r_len > 0.001: walls_sdf += add_wall('out_Bot_R', -half_x, bot_r_mid, thickness, bot_r_len)

    # Đầu ra (+X)
    top_y_gate = goal_y_final
    top_l_len = half_y + top_y_gate - cell/2
    top_r_len = half_y - top_y_gate - cell/2
    top_l_mid = -half_y + top_l_len / 2
    top_r_mid =  half_y - top_r_len / 2
    if top_l_len > 0.001: walls_sdf += add_wall('out_Top_L', half_x, top_l_mid, thickness, top_l_len)
    if top_r_len > 0.001: walls_sdf += add_wall('out_Top_R', half_x, top_r_mid, thickness, top_r_len)

    # Sinh tường trong Maze
    idx = 0
    for r in range(rows - 1):
        for c in range(cols):
            if h_walls[r][c]:
                walls_sdf += add_wall(f'h_{idx}', get_x(r) + cell / 2, get_y(c), thickness, cell)
                idx += 1
    for r in range(rows):
        for c in range(cols - 1):
            if v_walls[r][c]:
                walls_sdf += add_wall(f'v_{idx}', get_x(r), get_y(c) + cell / 2, cell, thickness)
                idx += 1

    # MẶT ĐẤT BẮT BUỘC PHẢI CÓ BITMASK 65535 ĐỂ ĐỠ ĐƯỢC 16 XE
    world_sdf = f'''
    <model name="ground_plane">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>20 20 0.01</size></box></geometry>
          <surface><contact><collide_bitmask>65535</collide_bitmask></contact></surface></collision>
        <visual name="visual">
          <transparency>0.85</transparency>
          <geometry><box><size>20 20 0.01</size></box></geometry>
          <material><ambient>0.8 0.8 0.8 0.1</ambient><diffuse>0.8 0.8 0.8 0.1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="maze_{rows}x{cols}">
      <static>true</static><pose>0 0 0 0 0 0</pose><link name="maze_walls">{walls_sdf}</link>
    </model>'''

    sdf = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="baohet">
    <physics name="1ms" type="ignored">
      <max_step_size>0.008</max_step_size>
      <real_time_factor>1.0</real_time_factor>
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
    with open(_p('maze_world.sdf'), 'w') as f:
        f.write(sdf)
    print(f"🤖 [Môi Trường] Đã tạo maze_world.sdf và bfs_map.json tại: {PROJECT_DIR}")

if __name__ == '__main__':
    import sys
    r = int(sys.argv[1]) if len(sys.argv) >= 3 else 5
    c = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    generate_maze_sdf_and_bfs(r, c)
