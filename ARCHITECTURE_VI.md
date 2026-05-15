# Kiến trúc Hệ thống AI Maze Solver (PPO Thuần Túy)

Tài liệu này giải thích chi tiết **luồng dữ liệu (Data Flow)** và **nhiệm vụ của từng file** trong dự án AI học tăng cường. Kiến trúc hiện tại đảm bảo sự "thuần túy" 100%: không gian lận GPS, không dùng thuật toán bùa chú, chỉ có Cảm biến $\rightarrow$ Mạng Nơ-ron $\rightarrow$ Bánh xe.

---

## 🔄 Luồng Dữ Liệu Cốt Lõi (Mù đường $\rightarrow$ Nơ-ron $\rightarrow$ Bánh xe)

Toàn bộ vòng lặp tư duy của con AI diễn ra liên tục theo đúng 5 bước sau:

1. **Thu thập Cảm biến (Input):** Môi trường mô phỏng 3D (Gazebo) phát ra 24 tia laser (Lidar). File `ros_bridge.py` nhận 24 khoảng cách này qua ROS2 và cập nhật vào biến lưu trữ của robot.
2. **Trích xuất Đặc trưng:** File `robot.py` (tại hàm `_get_obs`) lấy đúng 24 con số đó, chuẩn hóa thành dải `[0, 1]` (để mạng nơ-ron dễ tính toán) và biến nó thành **Observation**. *Tuyệt đối không có dữ liệu GPS hay la bàn chỉ hướng ở đây.*
3. **Mạng Nơ-ron suy nghĩ (Brain):** File `ppo_model.py` nhận mảng 24 con số này. Đưa qua Mạng chập 1 chiều (CNN 1D) để nhận diện góc cạnh của bức tường $\rightarrow$ đưa qua Mạng dày đặc (Dense) $\rightarrow$ Quyết định bấm nút số 0, 1, hoặc 2.
4. **Biến thành Lệnh bánh xe (Output):** Các nút tương ứng là 0 (Tiến nhanh), 1 (Tiến chậm rẽ trái), 2 (Tiến chậm rẽ phải). File `robot.py` chuyển nút này thành tốc độ thật dạng Vector (`cmd_x`, `cmd_az`).
5. **Thực thi:** File `ros_bridge.py` đóng gói Vector này thành ROS2 Twist Message và gửi thẳng xuống động cơ bánh xe trong Gazebo để robot di chuyển.

---

## 📂 Phân tích chi tiết 5 File Cốt lõi

Toàn bộ logic trí tuệ nhân tạo nằm gọn trong 5 file sau:

### 1. `src/agent/robot.py` (Thể xác & Luật chơi)
File này định nghĩa con robot và các quy tắc vật lý/phần thưởng.
- **Đầu vào ở đâu?** Ở hàm `def _get_obs(self):`. Nó chỉ return đúng mảng 24 phần tử Lidar.
- **Quy đổi Bánh xe ở đâu?** Ở hàm `def step(...):`. Tùy vào Mạng nơ-ron chọn action 0, 1, 2 mà nó ánh xạ ra vận tốc tương ứng để chạy.
- **Hàm mục tiêu (Reward) ở đâu?** Nằm trong hàm `step()`. Nó kiểm tra:
  - Nếu đâm tường $\rightarrow$ Phạt nặng (-10 điểm).
  - Nếu bước đi có xu hướng tiến gần đích (dựa trên việc lén tra cứu bản đồ BFS ẩn) $\rightarrow$ Thưởng điểm.
  - *Lưu ý Quan Trọng: Hàm mục tiêu dùng bản đồ BFS để CHẤM ĐIỂM hành động, AI tuyệt đối không được xem bản đồ này. AI phải tự học cách kết hợp hình ảnh Lidar với việc nhận được điểm.*

### 2. `src/core/ppo_model.py` (Bộ Não / Mạng Nơ-ron)
File này định nghĩa hình dáng của não bộ.
- **Lớp `LidarCNN`:** Xử lý 24 tia Lidar giống như cách AI xử lý hình ảnh. Giúp nó nhận diện các mẫu (patterns) như "đây là ngõ cụt", "đây là góc cua bên phải".
- **Lớp `PPOActorCritic`:** Bộ não chính chia làm 2 phần:
  - **Actor (Người hành động):** Đưa ra quyết định bánh xe (Tiến, Trái, Phải).
  - **Critic (Người đánh giá):** Dự đoán xem "với tình trạng cảm biến hiện tại, tương lai mình sẽ gom được bao nhiêu điểm". (Chỉ dùng lúc huấn luyện để giúp Actor sửa sai nhanh hơn).

### 3. `src/ros_layer/ros_bridge.py` (Dây Thần Kinh)
File này làm nhiệm vụ "shipper" vận chuyển dữ liệu, hoàn toàn không có logic trí tuệ:
- `_scan_cb()`: Lấy dữ liệu LaserScan từ Gazebo về bỏ vào bộ nhớ của robot.
- `publish_cmd()`: Lấy vận tốc từ bộ nhớ robot ném xuống chủ đề `/cmd_vel` của Gazebo cho bánh xe chạy.

### 4. `train_ppo.py` (Giáo viên dạy học)
File này tổ chức lớp học PPO cho 16 học sinh robot chạy đồng thời.
- Cho robot chạy thử nghiệm (rollout) trong một khoảng thời gian (256 steps).
- Sau khi chạy xong, nó nhìn lại xem hành động nào sinh ra điểm cao, hành động nào dẫn đến đâm tường.
- **Thuật toán học ở đâu?** Tại dòng định nghĩa `self.optimizer = optim.SGD(...)`. Đây chính là thuật toán **Vanilla Gradient Descent** cơ bản nhất trong Machine Learning, không có momentum hay các kỹ thuật tối ưu hóa phức tạp. Dựa trên lỗi dự đoán, thuật toán này sẽ tính đạo hàm (Backpropagation) và cập nhật các trọng số (weights) bên trong file `ppo_model.py` để lần sau robot khôn hơn.

### 5. `start_ai.launch.py` (Công tắc điện)
Script khởi động toàn bộ hệ thống:
- Dọn dẹp tiến trình Gazebo cũ bị kẹt.
- Đọc vị trí xuất phát từ file bản đồ.
- Khởi động thế giới mô phỏng 3D ở chế độ Headless (không giao diện, tối ưu CPU/GPU).
- Gọi (spawn) ra 16 con robot vào mô phỏng.
- Kích hoạt file `train_ppo.py` để bắt đầu huấn luyện.

---

**Kết Luận:** Toàn bộ hệ thống là một đường thẳng minh bạch. Khả năng tự hành hoàn toàn sinh ra từ việc Mạng Nơ-ron bị Gradient Descent uốn nắn qua hàng triệu lần thử và sai.
