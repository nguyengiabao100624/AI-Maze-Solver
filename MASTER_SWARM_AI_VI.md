# TÀI LIỆU TỔNG HỢP MASTER — HỆ THỐNG SWARM AI GIẢI MÊ CUNG

> **Tác giả:** Nguyễn Gia Bảo | **Phiên bản:** v5.0 TURBO | **Stack:** Genetic Algorithm + RNN + Gazebo + ROS2

---

## I. KIẾN TRÚC TỔNG THỂ

### 1.1 Sơ đồ 4 Tầng

```
┌───────────────────────────────────────────────────────────┐
│  TẦNG 4: TIẾN HÓA — train_ga.py / train_multi_ga.py      │
│  → Quản lý 30-96 bộ não, đánh giá, lai tạo, đột biến    │
├───────────────────────────────────────────────────────────┤
│  TẦNG 3: TRÍ TUỆ — ga_model.py (RNN ~1450 tham số)      │
│  → Nhận 26 input → Forward pass → Ra lệnh bánh xe        │
├───────────────────────────────────────────────────────────┤
│  TẦNG 2: GAME LOGIC — robot.py / multi_robot.py          │
│  → Gom Lidar 48→24, tính La bàn, check chết, tính BFS   │
├───────────────────────────────────────────────────────────┤
│  TẦNG 1: VẬT LÝ — Gazebo + ROS2 Bridge                  │
│  → gpu_lidar 48 tia, MecanumDrive, Teleport reset        │
└───────────────────────────────────────────────────────────┘
```

### 1.2 Luồng dữ liệu End-to-End

```
Gazebo gpu_lidar (48 tia 360°)
  → ROS2 /robot_X/scan
    → ros_bridge._scan_cb() → lọc NaN → r.lidar[48]
      → robot._get_obs() → clip 1m → đảo → maxpool 48→24 → +2 la bàn = 26
        → ga_model.act() → nối +8 memory = 34 → W1(34×32) ReLU → W2(32×10)
          → Sigmoid(2 action) + Tanh(8 memory mới)
            → robot.step() → (v_l,v_r) → (cmd_x, cmd_az)
              → ros_bridge.publish_cmd() → Twist → Gazebo
                → Robot di chuyển → Lidar quét lại → Lặp
                  
Khi chết → tính fitness BFS → GA xếp hạng → lai tạo/đột biến → thế hệ mới
```

### 1.3 Cây thư mục

```
AI/
├── robot_bao_template.sdf       # Template robot (Lidar 48 tia, 4 bánh Mecanum)
├── maze_world.sdf / multi_maze_world.sdf  # Thế giới mê cung (sinh tự động)
├── bfs_map.json / bfs_map_multi.json      # BFS distance map
├── start_ai.launch.py           # Launcher Single (1 map, 16 robot)
├── start_multi_ai.launch.py     # Launcher Multi (16 map, 16 robot)
├── train_ga.py                  # Huấn luyện Thiên Tài (96 pop)
├── train_multi_ga.py            # Huấn luyện Phổ Quát (30 pop)
├── auto_curriculum.py           # Tự đổi map khi tốt nghiệp
├── auto_multi.py                # Tự khôi phục khi crash
├── champions/                   # 70+ bộ não tốt nghiệp (.npy)
└── src/
    ├── agent/robot.py           # Game logic Single
    ├── agent/multi_robot.py     # Game logic Multi
    ├── core/ga_model.py         # Mạng Nơ-ron RNN
    ├── environment/maze_generator.py       # Sinh 1 mê cung DFS
    ├── environment/multi_maze_generator.py # Sinh 16 mê cung
    └── ros_layer/ros_bridge.py  # Cầu nối ROS2 ↔ Python
```

---

## II. ROBOT VẬT LÝ & MÊ CUNG

### 2.1 Robot (`robot_bao_template.sdf`)

Robot là xe 4 bánh Mecanum (25cm × 13cm × 5cm, nặng 1kg), gắn cảm biến Lidar ở mũi xe.

**Điểm trọng tâm cần hiểu:**

- **Lidar gpu_lidar:** Bắn 48 tia laser quét 360°, mỗi tia cách nhau 7.5°, tầm đo 2.2cm → 12m, tần số 10Hz.
- **Bitmask:** Mỗi robot có 1 bitmask riêng (`2^0` đến `2^15`). Tường có bitmask `65535`. Robot KHÔNG va chạm nhau nhưng ĐỀU va chạm tường.
- **Visibility flags:** Robot có flag=2, tường có flag=1. Lidar chỉ thấy flag=1 → robot KHÔNG "nhìn thấy" nhau qua Lidar.
- **MecanumDrive plugin:** Nhận lệnh `/robot_X/cmd_vel` (Twist message) → tự phân phối tốc độ 4 bánh.
- **OdometryPublisher:** Phát tọa độ chính xác tuyệt đối (ground truth) qua `/robot_X/ground_truth` ở 50Hz.

### 2.2 Sinh Mê cung (`maze_generator.py`)

Thuật toán sinh mê cung 5×5 bằng **DFS (Depth-First Search)**:

```python
# 1. KHỞI TẠO: Tất cả tường đều đóng
h_walls = [[True] * cols for _ in range(rows - 1)]  # Tường ngang
v_walls = [[True] * (cols - 1) for _ in range(rows)] # Tường dọc

# 2. DFS: Đi ngẫu nhiên, phá tường giữa 2 ô liền kề
stack = [(start_r, start_c)]
visited[start_r][start_c] = True
while stack:
    r, c = stack[-1]
    unvisited = [n for n in get_neighbors(r, c) if not visited[n[0]][n[1]]]
    if unvisited:
        nr, nc, d = unvisited[0]  # Chọn hàng xóm ngẫu nhiên (đã shuffle)
        # PHÁ TƯỜNG theo hướng di chuyển
        if d == 'N': h_walls[r - 1][c] = False   # Phá tường ngang phía Bắc
        elif d == 'S': h_walls[r][c] = False      # Phá tường ngang phía Nam
        elif d == 'W': v_walls[r][c - 1] = False  # Phá tường dọc phía Tây
        elif d == 'E': v_walls[r][c] = False      # Phá tường dọc phía Đông
        visited[nr][nc] = True
        stack.append((nr, nc))
    else:
        stack.pop()  # Quay lui (backtrack)
```

> **DFS đảm bảo mê cung "hoàn hảo"** — mọi ô đều có thể đến được, và chỉ có DUY NHẤT 1 đường đi giữa 2 ô bất kỳ (không có vòng lặp).

**BFS Distance Map — Hệ thống định vị GPS cho AI:**

```python
# Tính khoảng cách BFS từ ĐÍCH ngược về mọi ô
bfs_map = [[-1] * cols for _ in range(rows)]
queue = deque([(goal_row, goal_col, 0)])
bfs_map[goal_row][goal_col] = 0  # Đích = 0

while queue:
    r, c, dist = queue.popleft()
    # Mở rộng sang 4 hướng (chỉ khi KHÔNG CÓ TƯỜNG chắn)
    if r > 0 and not h_walls[r-1][c] and bfs_map[r-1][c] == -1:
        bfs_map[r-1][c] = dist + 1
        queue.append((r-1, c, dist + 1))
    # ... tương tự cho 3 hướng còn lại
```

> BFS map tạo ra một "bản đồ nhiệt" — ô nào gần đích hơn thì giá trị nhỏ hơn. AI dùng cái này để tính fitness: đi càng gần đích (giá trị BFS càng nhỏ) → điểm càng cao.

**Ép độ khó:** Mê cung được sinh lại cho đến khi BFS distance từ Start đến Goal >= 20 ô (đủ phức tạp).

### 2.3 Multi-Maze (`multi_maze_generator.py`)

Sinh 16 mê cung **độc lập** xếp trên lưới 4×4, cách nhau 10m:

```python
grid_side = 4  # √16 = 4
for i in range(16):
    offset_x = (i // grid_side) * spacing  # 0, 0, 0, 0, 10, 10, ...
    offset_y = (i % grid_side) * spacing   # 0, 10, 20, 30, 0, 10, ...
    # Sinh mê cung riêng biệt cho mỗi vị trí
    start_r, start_c, goal_row, goal_col, h_walls, v_walls, bfs_map = generate_single_maze(5, 5, 0.6)
```

---

## III. CẦU NỐI ROS2 (`ros_bridge.py`)

File này làm 2 việc duy nhất: **THU dữ liệu từ Gazebo** và **GỬI lệnh xuống Gazebo**.

### 3.1 Thu Lidar (`_scan_cb`)

```python
def _scan_cb(self, msg, robot_id):
    r = self.robots[robot_id - 1]
    raw = np.array(msg.ranges, dtype=np.float32)     # Mảng 48 số thực
    raw = np.nan_to_num(raw, nan=12.0, posinf=12.0, neginf=12.0)  # NaN/Inf → 12m
    raw = np.clip(raw, 0.022, 12.0)                  # Giới hạn trong [2.2cm, 12m]
    r.lidar = raw           # Ghi vào bộ nhớ robot
    r.lidar_updated = True  # BẬT cờ "có data mới" → AI mới được suy nghĩ
```

> **Cờ `lidar_updated`** là cơ chế đồng bộ: AI CHỈ được ra quyết định khi Gazebo đã cung cấp ảnh Lidar mới. Tránh AI chạy nhanh hơn vật lý → phán đoán trên dữ liệu cũ.

### 3.2 Thu vị trí (`_odom_cb`)

```python
def _odom_cb(self, msg, robot_id):
    r = self.robots[robot_id - 1]
    r.x = msg.pose.pose.position.x   # Tọa độ X (tiến/lùi)
    r.y = msg.pose.pose.position.y   # Tọa độ Y (trái/phải)
    # Giải mã Quaternion → Yaw (góc quay đầu xe, đơn vị radian)
    q = msg.pose.pose.orientation
    r.yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
```

### 3.3 Teleport Reset (`physical_reset`)

```python
def physical_reset(self, robots):
    # 1. Phanh gấp: Gửi lệnh dừng 15 lần liên tiếp (triệt tiêu quán tính)
    for _ in range(15):
        for r in robots:
            self.publish_cmd(r.id, 0.0, 0.0, 0.0)
        time.sleep(0.05)
    
    # 2. Teleport SONG SONG 16 xe cùng lúc (dùng subprocess.Popen)
    procs = []
    for r in robots:
        cmd = f"gz service -s /world/baohet/set_pose --req 'name: \"robot_{r.id}\", position: {{x: {r.start_x}, y: {r.start_y}, z: 0.030}}'"
        procs.append(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    for p in procs:
        p.wait(timeout=5)  # Chờ tất cả hoàn thành
    
    # 3. Phanh lại 10 lần nữa (triệt tiêu momentum sau teleport)
```

> Teleport song song thay vì tuần tự → nhanh gấp 16 lần.

---

## IV. GAME LOGIC — BỘ XỬ LÝ TRUNG TÂM

### 4.1 Xử lý cảm biến (`_get_obs` — robot.py dòng 127-157)

Đây là hàm **QUAN TRỌNG NHẤT** — nơi chế biến 48 tia Lidar thô thành 26 con số mà AI hiểu được.

```python
def _get_obs(self):
    # ═══ BƯỚC 1: CLIP & ĐẢO NGƯỢC ═══
    clipped = np.clip(self.lidar, 0.0, LIDAR_CLIP_RANGE)  # Cắt tầm nhìn xuống 1.0m
    rays = 1.0 - (clipped / LIDAR_CLIP_RANGE)
    # SAU PHÉP ĐẢO:
    #   Tường cách 0.0m → rays = 1.0 (NGUY HIỂM!)
    #   Tường cách 0.5m → rays = 0.5 (Vùng cảnh báo)
    #   Tường cách 1.0m+ → rays = 0.0 (An toàn)
    
    # ═══ BƯỚC 2: MAX-POOLING 48 → 24 ═══
    lidar_obs = rays.reshape(24, 2).max(axis=1).astype(np.float32)
    # Gom 48 tia thành 24 cặp, lấy MAX mỗi cặp
    # Tại sao MAX? Vì rays đã đảo → MAX = vật cản GẦN NHẤT
    # → Không bỏ sót bất kỳ mép tường nhỏ nào lọt giữa 2 tia!
    
    # ═══ BƯỚC 3: LA BÀN (COMPASS) — 2 input ═══
    # 3a. Khoảng cách đến waypoint tiếp theo (chuẩn hóa 0→1)
    dist_to_wp = math.hypot(self.next_wx - self.x, self.next_wy - self.y)
    norm_dist = np.clip(dist_to_wp / 2.0, 0.0, 1.0)
    
    # 3b. Góc lệch tương đối đến waypoint (-1 → +1)
    target_angle = math.atan2(self.next_wy - self.y, self.next_wx - self.x)
    relative_angle = target_angle - self.yaw  # Trừ góc đầu xe hiện tại
    # Chuẩn hóa về [-π, π] rồi chia π → [-1, 1]
    while relative_angle > math.pi: relative_angle -= 2*math.pi
    while relative_angle < -math.pi: relative_angle += 2*math.pi
    norm_angle = relative_angle / math.pi
    # norm_angle = +1.0 → mục tiêu ở phía sau bên trái
    # norm_angle =  0.0 → mục tiêu ngay trước mặt
    # norm_angle = -0.5 → mục tiêu ở bên phải 90°
    
    # ═══ GỘP LẠI: 24 + 2 = 26 INPUT ═══
    return np.concatenate([lidar_obs, [norm_dist, norm_angle]]).astype(np.float32)
```

### 4.2 BFS Waypoint — Hệ thống GPS dẫn đường

Hàm `get_bfs_distance()` không chỉ tính khoảng cách, mà còn **CẬP NHẬT waypoint tiếp theo** cho La bàn:

```python
def get_bfs_distance(self, x, y):
    # Chuyển tọa độ thực (mét) → chỉ số ô lưới (row, col)
    row = int(math.floor((x + cx) / cs))
    col = int(math.floor((y + cy) / cs))
    
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return self.max_bfs_dist * 2.0  # Ra ngoài → phạt nặng
    
    bfs_val = grid[row][col]
    if bfs_val == -1: return self.max_bfs_dist     # Ô tường → phạt
    if bfs_val == 0:                                # Đã ở ô đích!
        self.next_wx, self.next_wy = self.goal_x, self.goal_y
        return max(math.hypot(x - self.goal_x, y - self.goal_y), 0.001)
    
    # TÌM Ô KẾ TIẾP trên đường giải (BFS value nhỏ hơn 1)
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            v = grid[nr][nc]
            if v != -1 and v == bfs_val - 1:  # CHỈ chấp nhận ô có BFS = hiện tại - 1
                best_val, best_nr, best_nc = v, nr, nc
                break
    
    # Cập nhật waypoint → La bàn sẽ chỉ về đây
    self.next_wx = best_nr * cs - cx + cs / 2.0
    self.next_wy = best_nc * cs - cy + cs / 2.0
    return best_val * cs + math.hypot(x - self.next_wx, y - self.next_wy)
```

> **Tại sao cần waypoint thay vì chỉ thẳng đến đích?** Vì mê cung có tường! Nếu La bàn chỉ thẳng vào đích, robot sẽ lao vào tường. Waypoint dẫn robot đi theo đường giải BFS — rẽ từng khúc cua đúng lúc.

### 4.3 Game Logic (`step` — dòng 159-205)

```python
def step(self, action_continuous, max_steps):
    self.steps += 1
    v_left, v_right = action_continuous
    
    # QUY ĐỔI DIFFERENTIAL DRIVE:
    self.cmd_x = ((v_left + v_right) / 2.0) * 0.35   # Tốc độ tiến = trung bình 2 bánh
    self.cmd_az = (v_right - v_left) * 1.75            # Tốc độ xoay = chênh lệch 2 bánh
    # v_left=1, v_right=1 → đi thẳng max
    # v_left=0, v_right=1 → xoay phải gắt
    # v_left=1, v_right=0 → xoay trái gắt
    
    # CHECK 1: Về đích? (Khoảng cách < 0.30m)
    if math.hypot(self.goal_x - self.x, self.goal_y - self.y) < 0.30:
        self.is_done = True; self.death_why = 'Goal'; return ...
    
    # CHECK 2: Đâm tường? (Lidar min < 0.14m)
    if float(np.min(self.lidar)) < 0.140:
        self.is_dead = True; self.death_why = 'Wall'; return ...
    
    # CHECK 3: Timeout? (Hết 2000 steps hoặc ra ngoài map)
    if self.steps >= max_steps or abs(self.x) > 1.55 or abs(self.y) > 1.55:
        self.is_dead = True; self.death_why = 'Timeout'; return ...
    
    # CHECK 4: Bế tắc? (250 steps không tiến bộ BFS)
    cur_bfs = self.get_bfs_distance(self.x, self.y)
    if cur_bfs < self.min_bfs_reached - 0.05:
        self.min_bfs_reached = cur_bfs
        self.last_progress_step = self.steps
    if self.steps - self.last_progress_step > 250:
        self.is_dead = True; self.death_why = 'Timeout'; return ...
```

### 4.4 Tính điểm Fitness

```python
def get_fitness(self, max_steps):
    # Điểm = Quãng đường BFS đã vượt qua
    progress = self.start_bfs_dist - self.min_bfs_reached
    fitness = max(0.0, min(progress, self.max_bfs_dist))
    
    if self.death_why == 'Goal':
        fitness += 50.0  # THƯỞNG LỚN khi về đích
    return fitness
```

> **Ưu điểm BFS fitness:** Không thể gian lận! Robot chỉ được điểm khi THỰC SỰ tiến gần hơn đến đích theo đường giải mê cung. Đi vòng vòng = 0 điểm. Đi lùi = mất điểm.

### 4.5 Khác biệt Single vs Multi Robot

| Điểm | `robot.py` (Single) | `multi_robot.py` (Multi) |
|:-----|:---------------------|:--------------------------|
| Map | 1 mê cung duy nhất | 16 mê cung riêng biệt |
| Tọa độ | Tuyệt đối (0,0 = tâm map) | Có offset (`offset_x`, `offset_y`) |
| BFS | Dùng `bfs_data` trực tiếp | Dùng `maze_data['bfs_grid']` riêng cho mỗi map |
| OOB check | `abs(x) > 1.55` | `abs(local_x) > 1.55` (trừ offset trước) |
| Stagnation | 250 steps | 200 steps (khắt khe hơn) |


---

## V. MẠNG NƠ-RON RNN (`ga_model.py`)

### 5.1 Kiến trúc tổng quan

```
INPUT (26)          MEMORY (8)
   \                  /
    \                /
     ╔══════════════╗
     ║  NỐI LẠI     ║ → x = [26 + 8] = 34 phần tử
     ╠══════════════╣
     ║  W1: 34 × 32 ║ → h1 = ReLU(x·W1 + b1)  → 32 nơ-ron ẩn
     ╠══════════════╣
     ║  W2: 32 × 10 ║ → raw = h1·W2 + b2       → 10 output thô
     ╠══════════════╣
     ║  TÁCH RA:     ║
     ║  [0:2] → Sigmoid → v_left, v_right (0→1)
     ║  [2:10] → Tanh → memory mới (8 số, -1→+1)
     ╚══════════════╝

TỔNG THAM SỐ: 34×32 + 32 + 32×10 + 10 = 1088 + 32 + 320 + 10 = 1450
```

### 5.2 Code Forward Pass chi tiết

```python
def act(self, obs_lidar):
    # ═══ GIẢI NÉN TRỌNG SỐ ═══
    # Mảng 1D self.weights (1450 số) được cắt thành 4 miếng:
    idx = 0
    w1 = self.weights[0:1088].reshape(34, 32)      # Ma trận nhận thức
    b1 = self.weights[1088:1120].reshape(32,)       # Bias lớp ẩn
    w2 = self.weights[1120:1440].reshape(32, 10)    # Ma trận quyết định
    b2 = self.weights[1440:1450].reshape(10,)       # Bias đầu ra
    
    # ═══ NỐI CẢM BIẾN + BỘ NHỚ ═══
    x = np.concatenate([obs_lidar, self.memory])     # [26] + [8] = [34]
    
    # ═══ LỚP ẨN (Hidden Layer) ═══
    h1 = np.maximum(0, np.dot(x, w1) + b1)          # ReLU: chỉ giữ số dương
    # ReLU tự động "tắt" ~50% nơ-ron → giảm không gian tìm kiếm cho GA
    
    # ═══ LỚP ĐẦU RA ═══
    raw_out = np.dot(h1, w2) + b2                    # 10 số thô
    
    # ═══ KÍCH HOẠT ═══
    # Bánh xe: Sigmoid (ép về 0→1) → robot LUÔN đi tới, rẽ bằng chênh lệch
    actions = 1.0 / (1.0 + np.exp(-np.clip(raw_out[:2], -10, 10)))
    
    # Memory: Tanh (ép về -1→+1) → có thể nhớ thông tin âm/dương
    self.memory = np.tanh(raw_out[2:10])
    
    return actions[0], actions[1]  # v_left, v_right
```

### 5.3 Tại sao dùng NumPy thay vì PyTorch?

GA không cần gradient (backpropagation). GA cập nhật trọng số bằng cách **copy + đột biến trực tiếp** trên mảng 1D. NumPy nhanh hơn PyTorch 10-50x cho thao tác này vì không có overhead autograd.

### 5.4 RNN Memory — Bộ nhớ ngắn hạn

8 ô nhớ `self.memory` cho phép robot nhớ **trạng thái vài bước trước**. Ví dụ:
- "Tôi vừa rẽ trái ở ngã ba" → tránh quay lại
- "Tôi đã thấy tường dài bên phải" → giữ hướng đi

Memory được **reset về 0** mỗi đầu episode (`reset_memory()`), nên robot không nhớ xuyên episode.

---

## VI. HUẤN LUYỆN THIÊN TÀI CÁ NHÂN (`train_ga.py`)

### 6.1 Cấu hình

| Tham số | Giá trị | Ý nghĩa |
|:--------|:--------|:---------|
| `POPULATION_SIZE` | 96 | 96 bộ não mỗi thế hệ |
| `NUM_ROBOTS` | 16 | 16 robot vật lý trong Gazebo |
| Số batch | 96/16 = 6 | 6 đợt thi đấu mỗi thế hệ |
| `MAX_STEPS` | 2000 | Tối đa 2000 bước mỗi lượt |
| `ELITISM_COUNT` | 10 | Giữ lại 10 bộ não tốt nhất |
| `TOURNAMENT_K` | 6 | 6 người thi đấu mỗi vòng chọn |
| `BASE_MUTATION_RATE` | 0.15 | 15% gen bị đột biến |
| `MUTATION_POWER` | 0.15 | Biên độ nhiễu Gaussian |

### 6.2 Vòng lặp huấn luyện chính

```python
def run(self):
    for gen in range(1, GENERATIONS + 1):
        # ═══ ĐÁNH GIÁ: Chia 96 não thành 6 đợt × 16 xe ═══
        for i in range(0, POPULATION_SIZE, NUM_ROBOTS):  # i = 0, 16, 32, 48, 64, 80
            batch_models = self.population[i:i+16]
            fitness_res = self.evaluate_batch(batch_models)  # Chạy 16 xe trong Gazebo
            for j in range(len(batch_models)):
                self.fitnesses[i + j] = fitness_res[j]
```

### 6.3 Đánh giá 1 batch (`evaluate_batch`)

```python
def evaluate_batch(self, batch_models):
    current_obs = self._do_physical_reset()  # Teleport 16 xe về Start, chờ Lidar
    dones = [False] * 16
    for m in batch_models: m.reset_memory()  # Xóa bộ nhớ RNN
    
    while rclpy.ok():
        time.sleep(0.001)  # Nhường CPU cho ROS thread
        for i in range(16):
            r = self.robots[i]
            if not dones[i]:
                if r.lidar_updated:          # CHỈ khi Gazebo gửi Lidar mới
                    r.lidar_updated = False
                    v_l, v_r = batch_models[i].act(current_obs[i])  # AI suy nghĩ
                    o, _, d, _ = r.step((v_l, v_r), MAX_STEPS)     # Cập nhật game
                    if r.is_dead or r.is_done:
                        self.ros_node.publish_cmd(r.id, 0.0, 0.0, 0.0)  # Dừng xe
                        dones[i] = True
                    else:
                        self.ros_node.publish_cmd(r.id, r.cmd_x, 0.0, r.cmd_az)
                    current_obs[i] = o
        if all(dones): break  # Tất cả xe đã chết/về đích → kết thúc batch
    
    return [self.robots[i].get_fitness(MAX_STEPS) for i in range(16)]
```

### 6.4 Thuật toán Di truyền (Genetic Algorithm)

#### A. Sảnh Danh Vọng (Hall of Fame — All-time Top 10)

```python
# Thêm tất cả 96 cá thể vào danh sách ứng viên
for j in range(POPULATION_SIZE):
    fit = self.fitnesses[j]
    w = self.population[j].get_weights()
    # Chống trùng gen: nếu cùng bộ não nhưng điểm cao hơn → cập nhật
    for k, (hof_f, hof_w, hof_age) in enumerate(self.hall_of_fame):
        if np.allclose(w, hof_w):
            if fit > hof_f:
                self.hall_of_fame[k] = (fit, hof_w, 0)  # Reset tuổi
            break
    else:
        self.hall_of_fame.append((fit, w.copy(), 0))

# Sắp xếp và chỉ giữ TOP 10
self.hall_of_fame.sort(key=lambda x: x[0], reverse=True)
self.hall_of_fame = self.hall_of_fame[:ELITISM_COUNT]
```

#### B. Cơ chế Tuổi thọ & Giáng cấp (Anti-stagnation)

```python
for hof_f, hof_w, hof_age in self.hall_of_fame:
    hof_age += 1  # Tăng 1 tuổi mỗi thế hệ
    if hof_age >= 10:  # Ngồi vị trí 10 thế hệ mà không bị thay thế
        hof_f = max(hof_f * 0.85, 0.1)  # Giảm điểm 15%
        hof_age = 5  # Reset để tiếp tục giảm nếu vẫn ảo
```

> **Tại sao cần giáng cấp?** Đôi khi robot gặp may mắn đạt điểm cao 1 lần rồi không bao giờ lặp lại. Cơ chế decay loại bỏ dần những "huyền thoại ảo" này.

#### C. Lai tạo thế hệ mới

```python
new_population = []

# 1. ELITISM: Chèn 10 huyền thoại HoF nguyên vẹn (không đột biến)
for hof_f, hof_w, hof_age in self.hall_of_fame:
    elite = GARobotModel(); elite.set_weights(hof_w)
    new_population.append(elite)

# 2. TIÊM QUÁN QUÂN: 1-2 bộ não từ thư mục champions/
if self.champions_archive:
    for _ in range(min(2, len(self.champions_archive))):
        champ = GARobotModel()
        champ.set_weights(random.choice(self.champions_archive))
        new_population.append(champ)

# 3. TIÊM MÁU MỚI (khi bế tắc): 5 cá thể random hoàn toàn
if self.stagnation_counter >= 5:
    for _ in range(5):
        fresh = GARobotModel(); fresh.init_random_weights()
        new_population.append(fresh)

# 4. LAI TẠO phần còn lại:
while len(new_population) < 96:
    p1 = self.population[tournament_select()].get_weights()
    p2 = self.population[tournament_select()].get_weights()
    
    if random.random() < 0.10:  # 10% Crossover
        # Uniform Crossover: mỗi gen random chọn từ Bố hoặc Mẹ
        mask = np.random.rand(len(p1)) < 0.5
        child_w = np.where(mask, p1, p2)
    else:  # 90% Asexual (copy Bố)
        child_w = p1.copy()
    
    # MUTATION: 15% gen bị cộng nhiễu Gaussian
    mutation_mask = np.random.rand(len(child_w)) < 0.15
    child_w[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * 0.15
    
    child = GARobotModel(); child.set_weights(child_w)
    new_population.append(child)

self.population = new_population  # Thay thế toàn bộ thế hệ cũ
```

#### D. Tournament Selection

```python
def _tournament_selection(self):
    # Chọn ngẫu nhiên 6 cá thể, lấy kẻ có fitness cao nhất
    competitors = np.random.choice(96, 6, replace=False)
    return competitors[np.argmax(self.fitnesses[competitors])]
```

> **Tại sao Tournament thay vì Roulette?** Tournament selection có áp lực chọn lọc ổn định hơn. Không bị 1 cá thể siêu mạnh "độc chiếm" như Roulette.

### 6.5 Điều kiện Tốt nghiệp

```python
if goal_reaches >= 1:
    self.consecutive_goal_reaches += 1
else:
    self.consecutive_goal_reaches = 0

if self.consecutive_goal_reaches >= 3:  # 3 thế hệ liên tiếp có xe về đích
    champ_path = f"champions/champion_{int(time.time())}.npy"
    np.save(champ_path, self.historical_best_weights)  # Lưu Quán quân
    sys.exit(0)  # Thoát → auto_curriculum.py bắt tín hiệu → đổi map mới
```

> **Tại sao cần 3 lần liên tiếp?** Loại trừ yếu tố may mắn. Robot phải chứng minh nó THỰC SỰ biết giải map này, không phải random trúng.

### 6.6 Auto Curriculum (`auto_curriculum.py`)

```python
while True:
    result = subprocess.run(['ros2', 'launch', 'start_ai.launch.py'])
    if result.returncode == 0:       # Tốt nghiệp!
        print("Đang sinh Map mới...")
        map_count += 1
        time.sleep(3)                # Nghỉ CPU
    elif result.returncode == 130:   # Ctrl+C
        break
    else:                            # Crash
        time.sleep(5)                # Tự khởi động lại
```

> Đây là vòng lặp vô hạn: Train map → Tốt nghiệp → Sinh map mới → Train tiếp. Nhờ đó hệ thống tự động tạo ra 70+ Quán quân trên 70+ map khác nhau.


---

## VII. HUẤN LUYỆN TRÍ TUỆ PHỔ QUÁT (`train_multi_ga.py`)

### 7.1 Khác biệt cốt lõi so với Single

| Yếu tố | Single (`train_ga.py`) | Multi (`train_multi_ga.py`) |
|:--------|:-----------------------|:----------------------------|
| Quần thể | 96 não | 30 não |
| Cách đánh giá | 1 não → 1 xe/1 map | 1 não → 16 bản sao → 16 xe/16 map |
| Tính điểm | Fitness trực tiếp | **Trung bình** fitness 16 map |
| Tốc độ | 6 batch/thế hệ | 30 lần evaluate/thế hệ (chậm hơn nhiều) |
| Mục tiêu | Giải 1 map cụ thể | Giải ĐƯỢC BẤT KỲ map nào |

### 7.2 Đánh giá 1 bộ não trên 16 map (`evaluate_single_model`)

```python
def evaluate_single_model(self, model):
    current_obs = self._do_physical_reset()  # Reset 16 xe về 16 vị trí Start khác nhau
    
    # CRITICAL: Tạo 16 BẢN SAO não để giữ Memory (RNN) ĐỘC LẬP cho mỗi xe
    base_weights = model.get_weights()
    clones = []
    for _ in range(16):
        c = GARobotModel()
        c.set_weights(base_weights)  # CÙNG trọng số (cùng "kiến thức")
        c.reset_memory()             # KHÁC memory (khác "trải nghiệm")
        clones.append(c)
    
    # Vòng lặp chạy 16 xe song song trong Gazebo
    while rclpy.ok():
        for i in range(16):
            r = self.robots[i]
            if not dones[i] and r.lidar_updated:
                r.lidar_updated = False
                v_l, v_r = clones[i].act(current_obs[i])  # Clone riêng suy nghĩ
                o, _, d, _ = r.step((v_l, v_r), MAX_STEPS)
                # ... publish cmd ...
        if all(dones): break
    
    # TÍNH ĐIỂM TRUNG BÌNH 16 MAP
    fitnesses = [r.get_fitness(MAX_STEPS) for r in self.robots]
    goals = sum(1 for r in self.robots if r.death_why == 'Goal')
    return sum(fitnesses) / 16, goals  # Điểm TB và số map đã giải
```

> **Tại sao cần clone?** Vì RNN có memory. Nếu 16 xe dùng chung 1 object, xe 2 sẽ đọc memory mà xe 1 vừa ghi → chaos. Clone đảm bảo mỗi xe có "dòng suy nghĩ" riêng.

### 7.3 Seeding — Kế thừa từ Quán quân

```python
def _seed_population(self):
    loaded_count = 0
    # Ưu tiên 1: Hall of Fame (nếu đã từng train Multi trước đó)
    for i, (hf, hw, ha, hg) in enumerate(self.hall_of_fame):
        self.population[loaded_count].set_weights(hw.copy())
        loaded_count += 1
    
    # Ưu tiên 2: Quán quân từ thư mục champions/ (70+ bộ não Single)
    for champ_w in self.champions_archive:
        if loaded_count < POPULATION_SIZE:
            self.population[loaded_count].set_weights(champ_w.copy())
            loaded_count += 1
    
    # Phần còn lại: Đột biến từ các nguồn trên
    sources = [item[1] for item in self.hall_of_fame] + self.champions_archive
    while loaded_count < 30:
        source_w = sources[loaded_count % len(sources)]
        mutated = source_w.copy()
        mask = np.random.rand(len(mutated)) < 0.15
        mutated[mask] += np.random.randn(np.sum(mask)) * 0.15
        self.population[loaded_count].set_weights(mutated)
        loaded_count += 1
```

> **Curriculum Learning hoàn chỉnh:** 70 Quán quân Single (đã biết đi, né tường, nhìn la bàn) được dùng làm "vốn liếng" cho Multi. GA sẽ giữ lại gen "logic" và loại bỏ gen "học vẹt".

### 7.4 Lưu Quán quân Multi

```python
# Khi phá kỷ lục số map giải được (và >= 5 map)
if best_goals > self.max_historical_goals and best_goals >= 5:
    self.max_historical_goals = best_goals
    champ_file = f'champions/multi_champion_{best_goals}_maps_{int(time.time())}.npy'
    np.save(champ_file, self.population[best_idx].get_weights())
```

### 7.5 Auto Recovery (`auto_multi.py`)

```python
while True:
    result = subprocess.run(['ros2', 'launch', 'start_multi_ai.launch.py'])
    if result.returncode == 130: break        # Ctrl+C → dừng
    elif result.returncode == 0: break        # Hoàn tất 1000 gen → dừng
    else:                                     # Crash/OOM
        restart_count += 1
        time.sleep(5)  # Tự khởi động lại (dữ liệu đã backup qua emergency .npy)
```

---

## VIII. SO SÁNH: THIÊN TÀI CÁ NHÂN vs TRÍ TUỆ PHỔ QUÁT

### 8.1 Bảng so sánh đầy đủ

| Đặc điểm | 🧠 Thiên Tài (Single) | 🌐 Phổ Quát (Multi) |
|:----------|:-----------------------|:----------------------|
| **File train** | `train_ga.py` | `train_multi_ga.py` |
| **Số map/lần test** | 1 map cố định | 16 map ngẫu nhiên |
| **Population** | 96 não | 30 não |
| **Cách tính điểm** | Fitness trên 1 map | Trung bình 16 map |
| **Kiểu tư duy** | Học thuộc lòng (Memorization) | Logic phản xạ (Generalization) |
| **Ưu điểm** | Giải map cũ cực nhanh | Giải map MỚI CHƯA THẤY |
| **Nhược điểm** | Fail ngay trên map mới (Overfitting) | Chậm hội tụ, cần nhiều thời gian |
| **Vai trò** | Làm gen mồi (Seeding) | Sản phẩm AI cuối cùng |
| **Tốt nghiệp** | 3 lần về đích liên tiếp → .npy | Phá kỷ lục số map giải |
| **Output** | `champions/champion_*.npy` | `multi_ga_best.npy` |

### 8.2 Tại sao phải cần CẢ HAI? (Curriculum Learning)

```
GIAI ĐOẠN 1: LÒ ĐÀO TẠO (auto_curriculum.py)
═══════════════════════════════════════════════
Map 1 → train_ga.py → Tốt nghiệp → champion_001.npy
Map 2 → train_ga.py → Tốt nghiệp → champion_002.npy  
Map 3 → train_ga.py → Tốt nghiệp → champion_003.npy
...
Map 70 → train_ga.py → Tốt nghiệp → champion_070.npy

KẾT QUẢ: 70 bộ não, mỗi bộ GIỎI 1 map cụ thể.
Chúng biết đi, biết né tường, biết nhìn la bàn — nhưng "học vẹt".

GIAI ĐOẠN 2: LÒ MÀI GIŨ (auto_multi.py)
═══════════════════════════════════════════════
70 Quán quân → Nhồi vào train_multi_ga.py
→ Ép chạy 16 map CÙNG LÚC
→ Gen "học vẹt" bị loại (fail trên map lạ)
→ Gen "logic" được giữ (sống trên mọi map)
→ Sau N thế hệ: Bộ não giải được 12-16/16 map

KẾT QUẢ: 1 bộ não PHỔ QUÁT, giải được map CHƯA BAO GIỜ THẤY.
```

### 8.3 Ví dụ cụ thể về Overfitting vs Generalization

**Thiên Tài (Overfitting):** Robot học được: "Ở tọa độ (-1.2, 0.3) thì quẹo trái 90°". Điều này ĐÚNG trên map nó đã luyện, nhưng trên map mới tọa độ đó có thể là tường → CHẾT.

**Phổ Quát (Generalization):** Robot học được: "Nếu Lidar bên trái thấy tường gần (rays[6] > 0.7) VÀ La bàn chỉ bên phải (norm_angle < -0.3) → tăng v_right, giảm v_left để rẽ phải". Đây là **logic phản xạ** — đúng trên MỌI map.

---

## IX. CÔNG CỤ HỖ TRỢ

### 9.1 Test bộ não Multi (`test_best_multi_5_times.py`)

Chạy bộ não tốt nhất trên 1 map mới 5 lần, theo dõi: fitness, số steps, quãng đường, vị trí cuối, lý do chết. Giúp đánh giá khả năng generalization thực tế.

### 9.2 Vẽ biểu đồ Fitness (`plot_fitness.py`)

Đọc `fitness_log.csv`, vẽ 3 panel: (1) Best vs Avg fitness theo thế hệ, (2) Mutation rate, (3) Histogram điểm 30 gen cuối. Background tối (#1a1a2e), màu neon.

### 9.3 Vẽ bản đồ mê cung (`plot_current_maze.py`)

Đọc `bfs_map.json`, vẽ heatmap BFS + tường + Start/Goal. Giúp kiểm tra mê cung vừa sinh có đúng cấu trúc không.

### 9.4 Hệ thống Backup & Recovery

| File | Mục đích |
|:-----|:---------|
| `ga_best.npy` | Bộ não tốt nhất (cập nhật khi có xe về đích) |
| `ga_hof.npz` | Sảnh Danh Vọng Top 10 (fitness + weights + tuổi thọ) |
| `ga_emergency_backup.npy` | Backup khẩn cấp khi Ctrl+C hoặc crash |
| `fitness_log.csv` | Log toàn bộ lịch sử fitness |
| `champions/*.npy` | Kho tàng Quán quân tích lũy qua nhiều map |

Hệ thống **tự khôi phục** hoàn toàn: Khi khởi động lại, nó đọc HoF → Emergency → CSV theo thứ tự ưu tiên, nạp lại quần thể và tiếp tục train từ chỗ dang dở.

---

## X. KẾT LUẬN: TẠI SAO HỆ THỐNG NÀY ĐÁNG CHÚ Ý

### 10.1 Điểm mạnh kỹ thuật

1. **Curriculum Learning hoàn chỉnh:** Single → Multi, từ dễ đến khó, có hệ thống.
2. **Neuroevolution thuần túy:** Không cần gradient, không cần reward shaping phức tạp. GA + BFS fitness = đơn giản mà hiệu quả.
3. **Đồng bộ AI-Physics hoàn hảo:** Cờ `lidar_updated` đảm bảo AI không bao giờ phán đoán trên dữ liệu cũ.
4. **Chống gian lận BFS:** Robot chỉ được điểm khi THỰC SỰ tiến gần đích theo đường giải.
5. **Hệ thống tự vận hành 24/7:** Auto curriculum + auto recovery + emergency backup.

### 10.2 Luồng sự kiện tóm gọn

```
[Gazebo gpu_lidar 48 tia]
    → [ROS2 LaserScan]
        → [ros_bridge.py thu sóng, lọc NaN]
            → [robot.py: 48→24 Lidar + 2 La bàn = 26 input]
                → [ga_model.py: ×1450 gen → v_left, v_right]
                    → [ros_bridge.py: gửi Twist xuống Gazebo]
                        → [Robot di chuyển]

Nếu chết → BFS fitness → train_ga.py xếp hạng 96 con
    → Tournament chọn Bố Mẹ → Crossover 10% / Asexual 90% → Mutation 15%
        → Thế hệ mới → Lặp lại

3 thế hệ về đích liên tiếp → Tốt nghiệp → Lưu champion
    → auto_curriculum.py đổi map → Lặp lại

70 champion → Nhồi vào train_multi_ga.py → Ép 16 map
    → Loại gen học vẹt, giữ gen logic → Bộ não Phổ Quát hoàn chỉnh!
```

---

*Đây là tài liệu tổng hợp đầy đủ nhất, kết hợp toàn bộ 3 tài liệu gốc (system_architecture_vi.md, single_architecture_vi.md, single_vs_multi_brains_vi.md) cùng với giải thích chi tiết code từ toàn bộ 15 file source code của hệ thống.*
