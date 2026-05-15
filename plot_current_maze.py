import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_current_maze():
    json_path = os.path.join(SCRIPT_DIR, 'bfs_map.json')
    if not os.path.exists(json_path):
        print("Không tìm thấy bfs_map.json!")
        return

    with open(json_path, 'r') as f:
        bfs_data = json.load(f)

    rows, cols = bfs_data['maze_size']
    cs = bfs_data['cell_size']
    cx = bfs_data['half_x']
    cy = bfs_data['half_y']
    grid = bfs_data['bfs_grid']
    start_x, start_y = bfs_data['start']['x'], bfs_data['start']['y']
    goal_x, goal_y = bfs_data['goal']['x'], bfs_data['goal']['y']

    # Lấy thông tin tường
    h_walls = bfs_data.get('h_walls', [])
    v_walls = bfs_data.get('v_walls', [])

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Vẽ nền (Heatmap của BFS)
    grid_arr = np.array(grid, dtype=float)
    max_val = np.max(grid_arr)
    
    for r in range(rows):
        for c in range(cols):
            x_center = r * cs - cx + cs / 2.0
            y_center = c * cs - cy + cs / 2.0
            rect_plot_x = y_center - cs / 2.0
            rect_plot_y = x_center - cs / 2.0
            
            val = grid_arr[r][c]
            if val != -1:
                intensity = 1.0 - (val / max_val) * 0.7 
                color = (intensity, intensity, 1.0)
                ax.add_patch(patches.Rectangle((rect_plot_x, rect_plot_y), cs, cs, facecolor=color, edgecolor='none', alpha=0.6))
                ax.text(y_center, x_center, str(int(val)), color='black', ha='center', va='center', fontsize=10, alpha=0.5)

    # 2. Vẽ tường (Wall) màu Xanh đậm
    wall_color = '#2980b9'
    lw = 3

    # Hàm vẽ line (lưu ý: plot_x = Y Gazebo, plot_y = X Gazebo)
    def draw_line(y1, x1, y2, x2):
        ax.plot([y1, y2], [x1, x2], color=wall_color, linewidth=lw, solid_capstyle='round')

    # Vẽ khung bao ngoài (Outer Bounds)
    draw_line(-cy, -cx, cy, -cx)  # Bot
    draw_line(-cy, cx, cy, cx)    # Top
    draw_line(-cy, -cx, -cy, cx)  # Left
    draw_line(cy, -cx, cy, cx)    # Right

    # Cửa vào và ra (Tẩy khung bao)
    # Tẩy cửa Start
    start_c = bfs_data['start'].get('c', cols // 2)
    start_y1 = start_c * cs - cy
    start_y2 = (start_c + 1) * cs - cy
    ax.plot([start_y1, start_y2], [-cx, -cx], color='white', linewidth=lw+2)

    # Tẩy cửa Goal
    goal_c = bfs_data['goal'].get('c', cols // 2)
    goal_y1 = goal_c * cs - cy
    goal_y2 = (goal_c + 1) * cs - cy
    ax.plot([goal_y1, goal_y2], [cx, cx], color='white', linewidth=lw+2)

    # Vẽ vách ngăn ngang (h_walls)
    if h_walls:
        for r in range(rows - 1):
            for c in range(cols):
                if h_walls[r][c]:
                    # Tường giữa r và r+1
                    wall_x = (r + 1) * cs - cx
                    wall_y1 = c * cs - cy
                    wall_y2 = (c + 1) * cs - cy
                    draw_line(wall_y1, wall_x, wall_y2, wall_x)

    # Vẽ vách ngăn dọc (v_walls)
    if v_walls:
        for r in range(rows):
            for c in range(cols - 1):
                if v_walls[r][c]:
                    # Tường giữa c và c+1
                    wall_y = (c + 1) * cs - cy
                    wall_x1 = r * cs - cx
                    wall_x2 = (r + 1) * cs - cx
                    draw_line(wall_y, wall_x1, wall_y, wall_x2)

    # 3. Đánh dấu Start & Goal
    ax.scatter(start_y, start_x, c='green', s=200, marker='o', label='Start (Robot)', zorder=5)
    ax.scatter(goal_y, goal_x, c='red', s=300, marker='*', label='Goal', zorder=5)

    # Vòng tròn bán kính Goal
    goal_circle = patches.Circle((goal_y, goal_x), 0.30, color='red', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(goal_circle)

    # Format
    ax.set_aspect('equal')
    ax.set_xlim(cy + 0.5, -cy - 0.5) # Đảo ngược để Trái (+Y) sang bên trái, Phải (-Y) sang bên phải
    ax.set_ylim(-cx - 0.5, cx + 0.5)
    ax.set_xlabel('Y (Trái-Phải) - meters')
    ax.set_ylabel('X (Tiến-Lùi) - meters')
    ax.set_title('📍 SƠ ĐỒ MÊ CUNG & BFS hiện tại', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')

    save_path = os.path.join(SCRIPT_DIR, 'current_maze_preview.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"🗺️ Đã xuất ảnh bản đồ mê cung hiện tại ra file: {save_path}")

if __name__ == '__main__':
    plot_current_maze()
