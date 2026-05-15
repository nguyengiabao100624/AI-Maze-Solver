"""
GA Trainer - v5.0 TURBO (Pure NumPy + Skid Steer)
================================================
Huấn luyện bầy đàn xe 4 bánh giải mê cung bằng Genetic Algorithm.
Sử dụng hàm tính điểm BFS cực kỳ chặt chẽ (không lỗ hổng).
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
import time
import os
import sys
import json
import csv
import math
import threading
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.agent.robot import RobotAgent, OBS_DIM
from src.ros_layer.ros_bridge import ROSBridgeNode
from src.core.ga_model import GARobotModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _p(filename): return os.path.join(SCRIPT_DIR, filename)

# ===================== HYPERPARAMETERS =====================
NUM_ROBOTS = 16              # 16 XE MỖI ĐỢT
POPULATION_SIZE = 96         # 6 ĐỢT CHẠY x 16 XE
MAX_STEPS = 2000             # 2000 steps — Đủ cho BFS=24 + quay đầu ngõ cụt
GENERATIONS = 1000           # Số thế hệ tối đa

# GA Parameters
TOURNAMENT_K = 6             
ELITISM_COUNT = 10           # Giữ lại 10 cá thể tinh anh
BASE_MUTATION_RATE = 0.15    
MUTATION_POWER = 0.15        # Tăng từ 0.10 → 0.15: đủ mạnh để thoát vùng kẹt 5-7đ

# Crossover: 90% Asexual + 10% Uniform Crossover

# Diversity Injection (Tiêm máu mới khi kẹt)
STAGNATION_THRESHOLD = 5     # Bắt đầu tiêm máu mới sau 5 thế hệ không cải thiện
FRESH_BLOOD_COUNT = 5        # Giảm từ 20→5: có crossover rồi, fresh blood ít quan trọng hơn

class GATrainer:
    def __init__(self):
        rclpy.init()
        
        # --- BFS Map ---
        with open(_p('bfs_map.json'), 'r') as f:
            self.bfs_data = json.load(f)
        
        sx = self.bfs_data['start']['x']
        sy = self.bfs_data['start']['y']
        gx = self.bfs_data['goal']['x']
        gy = self.bfs_data['goal']['y']
        
        # --- INIT ROBOTS ---
        self.robots = [RobotAgent(i, sx, sy, gx, gy, self.bfs_data) 
                       for i in range(1, NUM_ROBOTS + 1)]
        
        # --- ROS 2 NODE ---
        self.ros_node = ROSBridgeNode(self.robots)
        self.executor = MultiThreadedExecutor(num_threads=12)
        self.executor.add_node(self.ros_node)
        
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()
        
        # --- GA POPULATION ---
        self.population = [GARobotModel() for _ in range(POPULATION_SIZE)]
        for ind in self.population:
            ind.init_random_weights()
            
        self.fitnesses = np.zeros(POPULATION_SIZE)
        
        # --- STATS & LOGGING ---
        self.csv_path = _p('fitness_log.csv')
        self.best_global_fitness = -1.0
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'Generation', 'Best_Fitness', 'Avg_Fitness', 'Mutation_Rate', 'Goal_Reaches', 'Crossover_Alpha'
                ])
                
        self.stagnation_counter = 0  # Đếm số thế hệ không cải thiện kỷ lục
        
        # Load prev best or emergency backup
        # ⚠️ PHẢI load .npy TRƯỚC rồi mới đọc CSV — tránh bẫy "fitness ảo" từ CSV khi .npy bị xóa
        self.emergency_path = _p('ga_emergency_backup.npy')
        self.best_weights_path = _p('ga_best.npy')
        self.hof_path = _p('ga_hof.npz')
        
        self.historical_best_weights = None
        self.hall_of_fame = [] # Chứa tuple: (fitness, weights)
        loaded_emergency_w = None
        
        # 1. TẢI SẢNH DANH VỌNG (Hall of Fame)
        if os.path.exists(self.hof_path):
            try:
                data = np.load(self.hof_path)
                fits = data['fitnesses']
                wts = data['weights']
                
                # Tương thích ngược: Nếu file cũ chưa có 'ages', cho age = 0
                if 'ages' in data:
                    ages = data['ages']
                else:
                    ages = np.zeros(len(fits), dtype=int)
                    
                for f, w, a in zip(fits, wts, ages):
                    self.hall_of_fame.append((float(f), w, int(a)))
                print(f"✅ Đã tải Sảnh Danh Vọng (HoF) với {len(self.hall_of_fame)} huyền thoại.")
                if self.hall_of_fame:
                    self.historical_best_weights = self.hall_of_fame[0][1]
            except Exception as e:
                print(f"⚠️ Lỗi đọc ga_hof.npz: {e}")
                
        # 2. TẢI FILE DỰ PHÒNG CŨ (Nếu HoF chưa tồn tại)
        elif os.path.exists(self.best_weights_path):
            try:
                self.historical_best_weights = np.load(self.best_weights_path)
                print("✅ Đã tìm thấy ga_best.npy. Sẵn sàng chèn Kỷ lục lịch sử vào bầy đàn.")
            except Exception as e:
                print(f"⚠️ Lỗi đọc ga_best.npy: {e}")
        
        if self.historical_best_weights is not None:
            # BẢO VỆ TUYỆT ĐỐI: Phải chèn ngay vào Thế hệ 1 (Gen 1) ở vị trí đầu tiên
            self.population[0].set_weights(self.historical_best_weights.copy())
        
        # CHỈ đọc kỷ lục từ CSV nếu CÓ bộ não đi kèm (.npy tồn tại)
        # Nếu không có .npy, CSV fitness vô nghĩa → giữ best_global_fitness = -1.0
        if self.historical_best_weights is not None and os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            val = float(row['Best_Fitness'])
                            if val > self.best_global_fitness:
                                self.best_global_fitness = val
                        except:
                            pass
                print(f"📈 Kỷ lục hiện tại được khôi phục: {self.best_global_fitness:.2f}")
                
                # Nếu HoF rỗng nhưng có ga_best.npy, khởi tạo HoF với kỷ lục này
                if not self.hall_of_fame and self.historical_best_weights is not None:
                    self.hall_of_fame.append((self.best_global_fitness, self.historical_best_weights.copy(), 0))
            except Exception:
                pass

        # 3. TẢI KHO TÀNG TINH HOA (Champions Archive)
        self.champions_archive = []
        champions_dir = _p('champions')
        if os.path.exists(champions_dir):
            for file in os.listdir(champions_dir):
                if file.endswith('.npy'):
                    try:
                        champ_w = np.load(os.path.join(champions_dir, file))
                        self.champions_archive.append(champ_w)
                    except:
                        pass
        if self.champions_archive:
            print(f"🏛️ Đã mở Kho tàng: Tải thành công {len(self.champions_archive)} Quán quân lịch sử!")
            
        self.consecutive_goal_reaches = 0

        if os.path.exists(self.emergency_path):
            try:
                loaded_emergency_w = np.load(self.emergency_path)
                print("🚨 Đã tìm thấy file Backup Khẩn Cấp! Khôi phục thế hệ đang dở dang...")
                os.remove(self.emergency_path) # Xóa ngay để tránh nạp lại lần sau
            except Exception as e:
                print(f"⚠️ Lỗi đọc backup khẩn cấp: {e}")

        # Nạp bộ não — ƯU TIÊN TỪ SẢNH DANH VỌNG (HoF)
        if len(self.hall_of_fame) > 0:
            # CÓ HoF → Inject TẤT CẢ huyền thoại vào các slot đầu tiên
            for i, (hof_f, hof_w, hof_age) in enumerate(self.hall_of_fame):
                if i < POPULATION_SIZE:
                    self.population[i].set_weights(hof_w.copy())
            
            # Nếu có emergency backup, chèn thêm vào slot tiếp theo
            next_slot = len(self.hall_of_fame)
            if loaded_emergency_w is not None and next_slot < ELITISM_COUNT:
                self.population[next_slot].set_weights(loaded_emergency_w.copy())
                next_slot += 1
            
            # Tiêm Quán quân từ Kho Tàng vào ngay Gen 1 (thay vì chờ Gen 2)
            champ_injected = 0
            if self.champions_archive:
                for ci in range(min(2, len(self.champions_archive))):
                    slot = len(self.hall_of_fame) + ci
                    if slot < POPULATION_SIZE:
                        champ_w = self.champions_archive[np.random.randint(len(self.champions_archive))]
                        self.population[slot].set_weights(champ_w.copy())
                        champ_injected += 1
            
            # Đột biến phần còn lại từ các nguồn HoF
            start_mutation = max(next_slot + champ_injected, ELITISM_COUNT)
            sources = [item[1] for item in self.hall_of_fame]
            for i in range(start_mutation, POPULATION_SIZE):
                source_w = sources[i % len(sources)]
                mutated_w = source_w.copy()
                mutation_mask = np.random.rand(len(mutated_w)) < BASE_MUTATION_RATE
                mutated_w[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * MUTATION_POWER
                self.population[i].set_weights(mutated_w)
                
            print(f"✅ Đã SEED bộ não: {len(self.hall_of_fame)} Tinh anh, {champ_injected} Quán quân, {POPULATION_SIZE - ELITISM_COUNT - champ_injected} đột biến.")
        else:
            # KHÔNG CÓ HoF → Fallback: dùng emergency + best như cũ
            sources = []
            if loaded_emergency_w is not None: sources.append(loaded_emergency_w)
            if self.historical_best_weights is not None: sources.append(self.historical_best_weights)
            
            if len(sources) > 0:
                for i in range(ELITISM_COUNT):
                    source_w = sources[i % len(sources)]
                    self.population[i].set_weights(source_w)
                for i in range(ELITISM_COUNT, POPULATION_SIZE):
                    source_w = sources[i % len(sources)]
                    mutated_w = source_w.copy()
                    mutation_mask = np.random.rand(len(mutated_w)) < BASE_MUTATION_RATE
                    mutated_w[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * MUTATION_POWER
                    self.population[i].set_weights(mutated_w)
                print(f"✅ Đã SEED bộ não: {ELITISM_COUNT} tinh khiết (từ {len(sources)} nguồn), {POPULATION_SIZE - ELITISM_COUNT} đột biến.")

        print(f"\n{'═'*60}")
        print(f"  🧬 GA v5.0 TURBO - 16 XE / 96 POP")
        print(f"  PopSize={POPULATION_SIZE} | Robots={NUM_ROBOTS} | MaxSteps={MAX_STEPS}")
        print(f"  Elitism={ELITISM_COUNT} | BaseMut={BASE_MUTATION_RATE} | MutPow={MUTATION_POWER}")
        print(f"{'═'*60}\n")

    def _do_physical_reset(self):
        self.ros_node.physical_reset(self.robots)
        
        # 1. Reset trạng thái (bao gồm lidar_updated = False)
        for r in self.robots:
            r.reset()
            
        # 2. CHỜ LIDAR MỚI (Tránh lỗi xe bị "mù" ở những step đầu do ROS latency)
        timeout = time.time() + 3.0
        while time.time() < timeout:
            if all(r.lidar_updated for r in self.robots):
                break
            time.sleep(0.01)
            
        # 3. Lấy observation thực tế sau khi Lidar đã cập nhật
        obs_list = []
        for r in self.robots:
            obs_list.append(r._get_obs())
            
        return obs_list
        
    def evaluate_batch(self, batch_models):
        current_obs = self._do_physical_reset()
        num_in_batch = len(batch_models)
        dones = [False] * num_in_batch
        
        for m in batch_models:
            m.reset_memory()
            
        while rclpy.ok():
            time.sleep(0.001) # Nhường CPU cho thread ROS
            
            for i in range(num_in_batch):
                r = self.robots[i]
                if not dones[i]:
                    # CHỈ cho phép AI suy nghĩ và tiến lên 1 step KHI VÀ CHỈ KHI Gazebo đã gửi ảnh Lidar mới.
                    # Điều này đồng bộ hoàn hảo tốc độ của Python với mô phỏng vật lý của Gazebo!
                    if r.lidar_updated:
                        r.lidar_updated = False
                        
                        # AI Suy nghĩ
                        v_l, v_r = batch_models[i].act(current_obs[i])
                        
                        # Cập nhật game logic
                        o, _, d, _ = r.step((v_l, v_r), MAX_STEPS)
                        
                        if r.is_dead or r.is_done:
                            self.ros_node.publish_cmd(r.id, 0.0, 0.0, 0.0)
                            dones[i] = True
                            
                            # DEBUG: Báo lý do chết để user biết (chỉ báo khi xe bị chết ngang do Timeout)
                            if r.death_why == 'Timeout':
                                print(f"  [Robot {r.id}] ⏳ Timeout: Chạy hết {MAX_STEPS} steps vật lý hoặc Kẹt quá 250 steps.")
                        else:
                            self.ros_node.publish_cmd(r.id, r.cmd_x, 0.0, r.cmd_az)
                            
                        current_obs[i] = o
                else:
                    # Xe đã chết thì liên tục ép dừng (an toàn vật lý)
                    self.ros_node.publish_cmd(r.id, 0.0, 0.0, 0.0)
                    
            if all(dones): break
                
        batch_fitness = []
        for i in range(num_in_batch):
            batch_fitness.append(self.robots[i].get_fitness(MAX_STEPS))
        return batch_fitness

    def run(self):
        for gen in range(1, GENERATIONS + 1):
            goal_reaches = 0
            for i in range(0, POPULATION_SIZE, NUM_ROBOTS):
                end_i = min(i + NUM_ROBOTS, POPULATION_SIZE)
                batch_models = self.population[i:end_i]
                
                fitness_res = self.evaluate_batch(batch_models)
                for j in range(len(batch_models)):
                    self.fitnesses[i + j] = fitness_res[j]
                    if self.robots[j].death_why == 'Goal': goal_reaches += 1
                        
            best_idx = np.argmax(self.fitnesses)
            best_fitness = self.fitnesses[best_idx]
            avg_fitness = np.mean(self.fitnesses)
            
            print(f"\n{'─'*60}")
            print(f"  📊 Thế hệ {gen} hoàn tất! | Mục tiêu: {goal_reaches} xe về đích")
            print(f"  🏆 Tốt nhất: {best_fitness:.1f} | 📈 Trung bình: {avg_fitness:.1f}")
            print(f"{'─'*60}\n")
            
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([gen, f"{best_fitness:.2f}", f"{avg_fitness:.2f}", f"{MUTATION_POWER}", goal_reaches, "N/A"])
            
            # --- CẬP NHẬT SẢNH DANH VỌNG (ALL-TIME HALL OF FAME) ---
            
            # XỬ LÝ TUỔI THỌ VÀ HẠ CHUẨN CÁ NHÂN (Individual Decay)
            new_hof = []
            for hof_f, hof_w, hof_age in self.hall_of_fame:
                hof_age += 1 # Tăng 1 tuổi sau mỗi thế hệ
                if hof_age >= STAGNATION_THRESHOLD * 2: # 10 thế hệ không bị đẩy lùi
                    old_f = hof_f
                    hof_f = max(hof_f * 0.85, 0.1) # Hạ điểm 15% CHỈ CỦA RIÊNG NÓ
                    hof_age = STAGNATION_THRESHOLD # Trả về 5 để tiếp tục giảm nếu vẫn là điểm ảo
                    print(f"📉 [CÁ NHÂN] Giáng cấp Tinh anh: Điểm {old_f:.2f} 📉 {hof_f:.2f} (Đã ngồi vị trí 10 thế hệ)")
                new_hof.append((hof_f, hof_w, hof_age))
            self.hall_of_fame = new_hof
            
            # Thêm tất cả xe của thế hệ này vào danh sách ứng viên
            for j in range(POPULATION_SIZE):
                fit = self.fitnesses[j]
                w = self.population[j].get_weights()
                
                # Chống trùng lặp gen + CẬP NHẬT ĐIỂM NẾU CAO HƠN
                is_duplicate = False
                for k, (hof_f, hof_w, hof_age) in enumerate(self.hall_of_fame):
                    if np.allclose(w, hof_w):
                        is_duplicate = True
                        # BUG FIX: Nếu cùng bộ não nhưng lần chạy mới ĐIỂM CAO HƠN → Cập nhật!
                        if fit > hof_f:
                            self.hall_of_fame[k] = (fit, hof_w, 0)  # Reset tuổi vì nó vừa chứng minh thực lực
                            print(f"🔄 Cập nhật Tinh anh #{k+1}: {hof_f:.2f} → {fit:.2f} (Cùng não, điểm cao hơn!)")
                        break
                
                if not is_duplicate:
                    self.hall_of_fame.append((fit, w.copy(), 0)) # Tuổi thọ = 0
            
            # Sắp xếp và chỉ giữ lại TOP 10 (ELITISM_COUNT) Mọi Thời Đại
            self.hall_of_fame.sort(key=lambda x: x[0], reverse=True)
            self.hall_of_fame = self.hall_of_fame[:ELITISM_COUNT]
            
            # ============================================================
            # 🏆 IN BẢNG XẾP HẠNG SẢNH DANH VỌNG 🏆
            print(f"\n{'═'*45}")
            print(f" 🌟 SẢNH DANH VỌNG (TOP 10 MỌI THỜI ĐẠI) 🌟")
            print(f"{'═'*45}")
            print(f" Hạng | Điểm số | Tuổi thọ (Gens)")
            print(f"{'─'*45}")
            for rank, (hf, hw, ha) in enumerate(self.hall_of_fame):
                print(f" #{rank+1:<3} | {hf:>7.2f} | {ha:>3} gens")
            print(f"{'═'*45}\n")
            # ============================================================
            
            # --- LƯU TRỮ VÀ XỬ LÝ KẸT ---
            prev_global_fitness = self.best_global_fitness
            if self.hall_of_fame:
                self.best_global_fitness = self.hall_of_fame[0][0]
                # CHỈ cập nhật historical_best_weights khi có xe THỰC SỰ VỀ ĐÍCH
                # Tránh bị ghi đè bởi não tầm thường do Decay ăn mòn điểm Quán quân cũ
                if goal_reaches >= 1:
                    self.historical_best_weights = self.hall_of_fame[0][1]
                    print(f"🛡️ ga_best.npy được CẬP NHẬT! (Có {goal_reaches} xe về đích)")
            
            # Lưu file HoF (.npz) — luôn lưu bình thường
            try:
                fits = np.array([item[0] for item in self.hall_of_fame])
                wts = np.array([item[1] for item in self.hall_of_fame])
                ages = np.array([item[2] for item in self.hall_of_fame])
                np.savez(self.hof_path, fitnesses=fits, weights=wts, ages=ages)
                # ga_best.npy — CHỈ lưu khi historical_best_weights tồn tại
                if self.historical_best_weights is not None:
                    np.save(self.best_weights_path, self.historical_best_weights)
            except Exception as e:
                print(f"⚠️ Lỗi lưu não: {e}")

            # --- KIỂM TRA ĐIỀU KIỆN TỐT NGHIỆP ---
            if goal_reaches >= 1:
                self.consecutive_goal_reaches += 1
                print(f"🔥 ĐANG CHUỖI THẮNG: {self.consecutive_goal_reaches}/3 thế hệ liên tiếp có xe về đích!")
            else:
                if self.consecutive_goal_reaches > 0:
                    print(f"💔 Gãy chuỗi thắng! Trở về 0/3.")
                self.consecutive_goal_reaches = 0
                
            if self.consecutive_goal_reaches >= 3:
                import time
                champ_path = os.path.join(_p('champions'), f"champion_{int(time.time())}.npy")
                np.save(champ_path, self.historical_best_weights)
                print(f"\n{'⭐'*30}")
                print(f"🏆 TỐT NGHIỆP MAP NÀY! Đã lưu Quán quân vào Kho Tàng: {champ_path}")
                print(f"🚀 Kích hoạt Tự Động Đổi Map...")
                print(f"{'⭐'*30}\n")
                import sys
                sys.exit(0) # Thoát code 0 để script Auto Curriculum bắt tín hiệu
            if best_fitness > prev_global_fitness:
                self.stagnation_counter = 0
                print(f"🎉 Đã lưu bộ não ĐỈNH NHẤT MỚI: {best_fitness:.2f} điểm!")
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= STAGNATION_THRESHOLD:
                    print(f"🩸 TIÊM MÁU MỚI! Kẹt {self.stagnation_counter} thế hệ → Thêm {FRESH_BLOOD_COUNT} cá thể random")
                    # (Việc giảm điểm bây giờ đã tự động xử lý bởi chức năng Tuổi thọ Cá nhân ở trên)
            
            new_population = []
            
            # 1. THÊM TỪ SẢNH DANH VỌNG VÀO THẾ HỆ MỚI
            # Đây chính là mấu chốt: Các huyền thoại của Mọi Thời Đại được chuyển thẳng vào đời sau.
            for hof_f, hof_w, hof_age in self.hall_of_fame:
                elite_ind = GARobotModel()
                elite_ind.set_weights(hof_w)
                new_population.append(elite_ind)
                
            # Nếu HoF chưa đủ 10 (chỉ xảy ra lúc mới chạy), lấy thêm từ thế hệ hiện tại cho đủ
            sorted_indices = np.argsort(self.fitnesses)[::-1]
            curr_elite_idx = 0
            while len(new_population) < ELITISM_COUNT and curr_elite_idx < POPULATION_SIZE:
                elite_w = self.population[sorted_indices[curr_elite_idx]].get_weights()
                # Chống trùng lặp khi điền thêm
                is_duplicate = False
                for f, w, a in self.hall_of_fame:
                    if np.allclose(elite_w, w):
                        is_duplicate = True; break
                if not is_duplicate:
                    elite_ind = GARobotModel()
                    elite_ind.set_weights(elite_w)
                    new_population.append(elite_ind)
                curr_elite_idx += 1
                
            # 3. Tiêm máu mới: Quán quân lịch sử + Cá thể random
            inject_stagnation = self.stagnation_counter >= STAGNATION_THRESHOLD
            injected = 0
            
            # Tiêm 1-2 Quán quân từ Kho Tàng vào mỗi thế hệ (nếu có) để giữ mã gen vượt chướng ngại vật
            if self.champions_archive and len(new_population) < POPULATION_SIZE:
                num_champs = min(2, len(self.champions_archive))
                for _ in range(num_champs):
                    champ_ind = GARobotModel()
                    # Chọn ngẫu nhiên 1 Quán quân
                    champ_w = self.champions_archive[np.random.randint(len(self.champions_archive))]
                    champ_ind.set_weights(champ_w)
                    new_population.append(champ_ind)
            
            while len(new_population) < POPULATION_SIZE:
                if inject_stagnation and injected < FRESH_BLOOD_COUNT:
                    # Tiêm máu mới: cá thể hoàn toàn random
                    fresh_ind = GARobotModel()
                    fresh_ind.init_random_weights()
                    new_population.append(fresh_ind)
                    injected += 1
                else:
                    # UNIFORM CROSSOVER (10%) / ASEXUAL (90%) + MUTATION
                    p1_idx = self._tournament_selection()
                    p2_idx = self._tournament_selection()
                    # Đảm bảo 2 cha mẹ KHÁC NHAU (nếu trùng → d=0 → crossover vô nghĩa)
                    attempts = 0
                    while p2_idx == p1_idx and attempts < 3:
                        p2_idx = self._tournament_selection()
                        attempts += 1
                    p1_w = self.population[p1_idx].get_weights()
                    
                    # CÂN BẰNG TỐI ƯU NHẤT CHO NEURAL NETWORK (Chuẩn OpenAI / Neuroevolution)
                    # - 90% Sinh sản vô tính (Asexual): Giữ nguyên khối logic của cha/mẹ, chỉ đột biến nhẹ (Tối ưu cục bộ - Local Search).
                    # - 10% Lai ghép (Crossover): Thi thoảng "làm liều" xé gen để tạo ra bước nhảy vọt (Tối ưu toàn cục - Global Search).
                    if np.random.rand() < 0.10:
                        p2_w = self.population[p2_idx].get_weights()
                        child_w = self._uniform_crossover(p1_w, p2_w)
                    else:
                        child_w = p1_w.copy() # 90% bầy đàn sẽ giữ nguyên vẹn não xịn và chỉ đột biến
                    
                    # Mutation: nhiễu nhẹ sau crossover
                    mutation_mask = np.random.rand(len(child_w)) < BASE_MUTATION_RATE
                    child_w[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * MUTATION_POWER
                    
                    child_ind = GARobotModel()
                    child_ind.set_weights(child_w)
                    new_population.append(child_ind)
                
            self.population = new_population

    def _uniform_crossover(self, p1_w, p2_w):
        """
        Uniform Crossover (Giải quyết triệt để lỗi Trọng số phình to/Thoái hóa):
        - Bốc ngẫu nhiên từng gen từ Cha hoặc Mẹ với tỉ lệ 50-50.
        - Ưu điểm: KHÔNG BAO GIỜ làm trọng số to lên (vượt giới hạn).
        - Neural Network sẽ không bị bão hòa (bị đơ ở v_left=1.0, v_right=1.0).
        """
        mask = np.random.rand(len(p1_w)) < 0.5
        return np.where(mask, p1_w, p2_w)

    def _tournament_selection(self):
        competitors = np.random.choice(POPULATION_SIZE, TOURNAMENT_K, replace=False)
        best_idx = competitors[0]
        for idx in competitors[1:]:
            if self.fitnesses[idx] > self.fitnesses[best_idx]:
                best_idx = idx
        return best_idx

if __name__ == '__main__':
    trainer = None
    try:
        trainer = GATrainer()
        trainer.run()
    except KeyboardInterrupt:
        print("\n⚠️ Tắt thủ công (Ctrl+C). Đang lưu Backup Khẩn Cấp...")
        if trainer is not None:
            try:
                best_w = trainer.historical_best_weights
                if best_w is None:
                    best_idx = np.argmax(trainer.fitnesses)
                    best_w = trainer.population[best_idx].get_weights()
                np.save(trainer.emergency_path, best_w)
                print("💾 Đã lưu thành công `ga_emergency_backup.npy`!")
            except Exception as e:
                print(f"❌ Lỗi khi lưu backup: {e}")
        import sys
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Lỗi hệ thống: {e}. Đang lưu Backup Khẩn Cấp...")
        if trainer is not None:
            try:
                best_w = trainer.historical_best_weights
                if best_w is None:
                    best_idx = np.argmax(trainer.fitnesses)
                    best_w = trainer.population[best_idx].get_weights()
                np.save(trainer.emergency_path, best_w)
                print("💾 Đã lưu thành công `ga_emergency_backup.npy`!")
            except Exception as e2:
                print(f"❌ Lỗi khi lưu backup: {e2}")
        import sys
        sys.exit(1)
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
