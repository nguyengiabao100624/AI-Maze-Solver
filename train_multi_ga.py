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
from src.agent.multi_robot import MultiRobotAgent
from src.ros_layer.ros_bridge import ROSBridgeNode
from src.core.ga_model import GARobotModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _p(filename): return os.path.join(SCRIPT_DIR, filename)

# ===================== HYPERPARAMETERS =====================
NUM_ROBOTS = 16              
POPULATION_SIZE = 30         # Giảm từ 48 xuống 30 để train nhanh gấp rưỡi
MAX_STEPS = 2000             
GENERATIONS = 1000           

# GA Parameters
TOURNAMENT_K = 6             
ELITISM_COUNT = 10           
BASE_MUTATION_RATE = 0.15    
MUTATION_POWER = 0.15        

# Diversity Injection
STAGNATION_THRESHOLD = 5     
FRESH_BLOOD_COUNT = 5        

class MultiMazeGATrainer:
    def __init__(self):
        rclpy.init()
        
        # --- BFS Multi-Maze Data ---
        with open(_p('bfs_map_multi.json'), 'r') as f:
            self.bfs_data = json.load(f)
            
        self.robots = []
        for i in range(NUM_ROBOTS):
            maze_data = self.bfs_data['mazes'][i]
            r = MultiRobotAgent(i + 1, maze_data, self.bfs_data)
            self.robots.append(r)
            
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
        self.csv_path = _p('multi_fitness_log.csv')
        self.best_global_fitness = -1.0
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'Generation', 'Best_Avg_Fit', 'Pop_Avg_Fit', 'Best_Goals', 'Avg_Goals'
                ])
                
        self.stagnation_counter = 0
        self.max_historical_goals = 0
        self.emergency_path = _p('multi_ga_emergency.npy')
        self.best_weights_path = _p('multi_ga_best.npy')
        self.hof_path = _p('multi_ga_hof.npz')
        
        self.historical_best_weights = None
        self.hall_of_fame = [] # (fitness, weights, age, goals)
        
        # 1. Tải Sảnh Danh Vọng (HoF) nếu có
        if os.path.exists(self.hof_path):
            try:
                data = np.load(self.hof_path)
                fits, wts = data['fitnesses'], data['weights']
                ages = data['ages'] if 'ages' in data else np.zeros(len(fits), dtype=int)
                goals = data['goals'] if 'goals' in data else np.zeros(len(fits), dtype=int)
                
                for f, w, a, g in zip(fits, wts, ages, goals):
                    self.hall_of_fame.append((float(f), w, int(a), int(g)))
                if self.hall_of_fame: self.historical_best_weights = self.hall_of_fame[0][1]
            except: pass

        # 2. SEED TỪ KHO TÀNG, CÚP VÀ BACKUP KHẨN CẤP
        self.champions_archive = []
        
        # Load Cúp và Backup khẩn cấp (nếu có) để đảm bảo không rớt một nhịp nào khi tắt mở lại
        for fallback_path in [self.emergency_path, self.best_weights_path]:
            if os.path.exists(fallback_path):
                try: 
                    self.champions_archive.append(np.load(fallback_path))
                    if fallback_path == self.emergency_path: os.remove(fallback_path) # Đọc xong thì xóa
                except: pass

        champions_dir = _p('champions')
        if os.path.exists(champions_dir):
            for file in os.listdir(champions_dir):
                if file.endswith('.npy'):
                    try: self.champions_archive.append(np.load(os.path.join(champions_dir, file)))
                    except: pass
        if self.champions_archive:
            print(f"🏛️ Đã tải {len(self.champions_archive)} Quán quân lịch sử để lai tạo!")

        self._seed_population()

        print(f"\n{'═'*60}")
        print(f"  🧬 MULTI-MAZE GA - 16 MAPS / {POPULATION_SIZE} POP")
        print(f"{'═'*60}\n")

    def _seed_population(self):
        loaded_count = 0
        # Ưu tiên HoF trước
        for i, (hf, hw, ha, hg) in enumerate(self.hall_of_fame):
            if loaded_count < POPULATION_SIZE:
                self.population[loaded_count].set_weights(hw.copy())
                loaded_count += 1
                
        # Nhồi các Quán quân cũ vào
        if self.champions_archive:
            np.random.shuffle(self.champions_archive)
            for champ_w in self.champions_archive:
                if loaded_count < POPULATION_SIZE:
                    self.population[loaded_count].set_weights(champ_w.copy())
                    loaded_count += 1
                else: break
                
        # Đột biến phần còn lại từ các nguồn tinh hoa
        sources = [item[1] for item in self.hall_of_fame] + self.champions_archive
        if not sources: sources = [self.population[0].get_weights()]
        
        while loaded_count < POPULATION_SIZE:
            source_w = sources[loaded_count % len(sources)]
            mutated = source_w.copy()
            mask = np.random.rand(len(mutated)) < BASE_MUTATION_RATE
            mutated[mask] += np.random.randn(np.sum(mask)) * MUTATION_POWER
            self.population[loaded_count].set_weights(mutated)
            loaded_count += 1

    def _do_physical_reset(self):
        self.ros_node.physical_reset(self.robots)
        for r in self.robots: r.reset()
        
        timeout = time.time() + 5.0
        while time.time() < timeout:
            if all(r.lidar_updated for r in self.robots): break
            time.sleep(0)
            
        return [r._get_obs() for r in self.robots]

    def evaluate_single_model(self, model):
        current_obs = self._do_physical_reset()
        dones = [False] * NUM_ROBOTS
        
        # ⚠️ CRITICAL FIX: Tạo 16 bản sao não để giữ Memory (RNN state) hoàn toàn độc lập cho 16 xe
        base_weights = model.get_weights()
        clones = []
        for _ in range(NUM_ROBOTS):
            c = GARobotModel()
            c.set_weights(base_weights)
            c.reset_memory()
            clones.append(c)
        
        while rclpy.ok():
            time.sleep(0)
            for i in range(NUM_ROBOTS):
                r = self.robots[i]
                if not dones[i]:
                    if r.lidar_updated:
                        r.lidar_updated = False
                        # Dùng clone riêng của xe i để suy nghĩ, tránh việc xe 2 lấy nhầm trí nhớ của xe 1
                        v_l, v_r = clones[i].act(current_obs[i])
                        o, _, d, _ = r.step((v_l, v_r), MAX_STEPS)
                        
                        if r.is_dead or r.is_done:
                            self.ros_node.publish_cmd(r.id, 0.0, 0.0, 0.0)
                            dones[i] = True
                        else:
                            self.ros_node.publish_cmd(r.id, r.cmd_x, 0.0, r.cmd_az)
                        current_obs[i] = o
                else:
                    self.ros_node.publish_cmd(r.id, 0.0, 0.0, 0.0)
            if all(dones): break
            
        fitnesses = [r.get_fitness(MAX_STEPS) for r in self.robots]
        goals = sum(1 for r in self.robots if r.death_why == 'Goal')
        return sum(fitnesses) / NUM_ROBOTS, goals

    def run(self):
        for gen in range(1, GENERATIONS + 1):
            print(f"\n{'─'*60}")
            print(f"🚀 THẾ HỆ {gen}/{GENERATIONS} (Đang chấm điểm trên 16 Map...)")
            
            gen_goals = []
            for i in range(POPULATION_SIZE):
                avg_fit, goals = self.evaluate_single_model(self.population[i])
                self.fitnesses[i] = avg_fit
                gen_goals.append(goals)
                # Print chi tiết giống bản cũ
                print(f"  [Não {i:02d}] Điểm TB: {avg_fit:>5.2f} | Đã giải: {goals:>2}/16 xe")
                
            best_idx = np.argmax(self.fitnesses)
            best_fit, best_goals = self.fitnesses[best_idx], gen_goals[best_idx]
            avg_fit, avg_goals = np.mean(self.fitnesses), np.mean(gen_goals)
            
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([gen, f"{best_fit:.2f}", f"{avg_fit:.2f}", best_goals, f"{avg_goals:.2f}"])
            
            # --- XỬ LÝ TUỔI THỌ VÀ HẠ CHUẨN (DECAY) ---
            new_hof = []
            for hf, hw, ha, hg in self.hall_of_fame:
                ha += 1
                if ha >= STAGNATION_THRESHOLD * 2:
                    hf = max(hf * 0.85, 0.1)
                    ha = STAGNATION_THRESHOLD
                new_hof.append((hf, hw, ha, hg))
            self.hall_of_fame = new_hof
            
            # CẬP NHẬT HOF
            for j in range(POPULATION_SIZE):
                fit, w, goals = self.fitnesses[j], self.population[j].get_weights(), gen_goals[j]
                is_dup = False
                for k, (hf, hw, ha, hg) in enumerate(self.hall_of_fame):
                    if np.allclose(w, hw):
                        is_dup = True
                        if fit > hf:
                            self.hall_of_fame[k] = (fit, w, 0, goals)
                            print(f"🔄 Cập nhật Tinh anh #{k+1}: {hf:.2f} → {fit:.2f} (Cùng não, điểm cao hơn!)")
                        break
                if not is_dup: self.hall_of_fame.append((fit, w.copy(), 0, goals))
                
            self.hall_of_fame.sort(key=lambda x: x[0], reverse=True)
            self.hall_of_fame = self.hall_of_fame[:ELITISM_COUNT]
            
            print(f"\n{'═'*45}")
            print(f" 🌟 TOP ĐA NĂNG MỌI THỜI ĐẠI (GIẢI {NUM_ROBOTS} MAP)")
            print(f"{'═'*45}")
            print(f" Hạng | Điểm TB | Giải được | Tuổi thọ")
            print(f"{'─'*45}")
            for rank, (hf, hw, ha, hg) in enumerate(self.hall_of_fame):
                print(f" #{rank+1:<3} | {hf:>7.2f} | {hg:>2}/16 map | {ha:>3} gens")
            print(f"{'═'*45}\n")
            
            # LƯU KỶ LỤC VÀ HOF
            prev_global = self.best_global_fitness
            if self.hall_of_fame:
                self.best_global_fitness = self.hall_of_fame[0][0]
                self.historical_best_weights = self.hall_of_fame[0][1]
                
            try:
                fits = np.array([item[0] for item in self.hall_of_fame])
                wts = np.array([item[1] for item in self.hall_of_fame])
                ages = np.array([item[2] for item in self.hall_of_fame])
                goals = np.array([item[3] for item in self.hall_of_fame])
                np.savez(self.hof_path, fitnesses=fits, weights=wts, ages=ages, goals=goals)
                if self.historical_best_weights is not None:
                    np.save(self.best_weights_path, self.historical_best_weights)
            except Exception as e:
                print(f"⚠️ Lỗi lưu não: {e}")
                
            if best_fit > prev_global:
                self.stagnation_counter = 0
                print(f"🎉 Đỉnh cao mới: {best_fit:.2f} điểm trung bình!")
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= STAGNATION_THRESHOLD:
                    print(f"🩸 TIÊM MÁU MỚI! Kẹt {self.stagnation_counter} thế hệ → Bơm gen ngẫu nhiên")

            # Lưu vào Kho tàng Champions nếu phá kỷ lục số Map giải được
            if best_goals > self.max_historical_goals and best_goals >= 5:
                self.max_historical_goals = best_goals
                champ_dir = _p('champions')
                os.makedirs(champ_dir, exist_ok=True)
                champ_file = os.path.join(champ_dir, f'multi_champion_{best_goals}_maps_{int(time.time())}.npy')
                np.save(champ_file, self.population[best_idx].get_weights())
                print(f"🎖️ ĐÃ LƯU VÀO KHO TÀNG! Cột mốc mới: Giải được {best_goals} mê cung ({champ_file})")

            if self.hall_of_fame[0][3] == NUM_ROBOTS:
                print(f"\n{'⭐'*30}")
                print(f"🏆 TUYỆT VỜI! ĐÃ CÓ BỘ NÃO GIẢI ĐƯỢC 100% (16/16) MÊ CUNG!")
                print(f"Bộ não hoàn hảo nhất đã được lưu tại {self.best_weights_path}")
                print(f"{'⭐'*30}\n")
            
            # LAI TẠO THẾ HỆ MỚI
            new_pop = []
            for hf, hw, ha, hg in self.hall_of_fame:
                ind = GARobotModel(); ind.set_weights(hw); new_pop.append(ind)
                
            # Tiêm 1-2 Quán quân từ Kho Tàng vào mỗi thế hệ (nếu có) để giữ mã gen vượt chướng ngại vật
            if self.champions_archive and len(new_pop) < POPULATION_SIZE:
                num_champs = min(2, len(self.champions_archive))
                for _ in range(num_champs):
                    champ_ind = GARobotModel()
                    champ_w = self.champions_archive[np.random.randint(len(self.champions_archive))]
                    champ_ind.set_weights(champ_w)
                    new_pop.append(champ_ind)
                
            inject_stagnation = self.stagnation_counter >= STAGNATION_THRESHOLD
            injected = 0
            
            while len(new_pop) < POPULATION_SIZE:
                if inject_stagnation and injected < FRESH_BLOOD_COUNT:
                    ind = GARobotModel(); ind.init_random_weights()
                    new_pop.append(ind); injected += 1
                else:
                    p1 = self.population[self._tournament()].get_weights()
                    p2 = self.population[self._tournament()].get_weights()
                    child_w = self._uniform_crossover(p1, p2) if np.random.rand() < 0.10 else p1.copy()
                    
                    mask = np.random.rand(len(child_w)) < BASE_MUTATION_RATE
                    child_w[mask] += np.random.randn(np.sum(mask)) * MUTATION_POWER
                    
                    ind = GARobotModel(); ind.set_weights(child_w); new_pop.append(ind)
                    
            self.population = new_pop

    def _uniform_crossover(self, p1, p2):
        mask = np.random.rand(len(p1)) < 0.5
        return np.where(mask, p1, p2)

    def _tournament(self):
        comps = np.random.choice(POPULATION_SIZE, TOURNAMENT_K, replace=False)
        best = comps[0]
        for idx in comps[1:]:
            if self.fitnesses[idx] > self.fitnesses[best]: best = idx
        return best

if __name__ == '__main__':
    trainer = None
    try:
        trainer = MultiMazeGATrainer()
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
                print("💾 Đã lưu thành công `multi_ga_emergency.npy`!")
            except Exception as e:
                print(f"❌ Lỗi khi lưu backup: {e}")
        import sys; sys.exit(130)
    except Exception as e:
        print(f"\n❌ Lỗi hệ thống: {e}. Đang lưu Backup Khẩn Cấp...")
        if trainer is not None:
            try:
                best_w = trainer.historical_best_weights
                if best_w is None:
                    best_idx = np.argmax(trainer.fitnesses)
                    best_w = trainer.population[best_idx].get_weights()
                np.save(trainer.emergency_path, best_w)
                print("💾 Đã lưu thành công `multi_ga_emergency.npy`!")
            except Exception as e2:
                print(f"❌ Lỗi khi lưu backup: {e2}")
        import sys; sys.exit(1)
    finally:
        try: rclpy.shutdown()
        except: pass

