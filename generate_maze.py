"""
Wrapper cho generate_maze.py cũ để đảm bảo tính tương thích ngược cho auto_train.sh nếu có gọi.
"""
import sys
from src.environment.maze_generator import generate_maze_sdf_and_bfs

if __name__ == '__main__':
    r = int(sys.argv[1]) if len(sys.argv) >= 3 else 5
    c = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    generate_maze_sdf_and_bfs(r, c)
