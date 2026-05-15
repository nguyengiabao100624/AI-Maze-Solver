import numpy as np

class GARobotModel:
    """
    Mạng Nơ-ron (RNN) cho thuật toán Genetic Algorithm (GA) sử dụng NumPy để tối ưu tốc độ.
    
    Cấu trúc:
    - Input: 26 (24 Lidar + 2 La Bàn) + 8 (Memory) = 34
    - Hidden: 32 (ReLU — dung lượng lớn hơn để kết hợp La Bàn và Lidar)
    - Output: 2 (v_left, v_right) + 8 (Memory) = 10
    - Tổng tham số: ~1450
    """
    def __init__(self, input_dim=26, memory_dim=8, hidden_dim=32, output_action_dim=2):
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.output_action_dim = output_action_dim
        
        self.total_input_dim = input_dim + memory_dim
        self.total_output_dim = output_action_dim + memory_dim
        
        # Shapes
        self.shapes = [
            (self.total_input_dim, self.hidden_dim),  # W1
            (self.hidden_dim,),                       # B1
            (self.hidden_dim, self.total_output_dim), # W2
            (self.total_output_dim,)                  # B2
        ]
        
        # Tính tổng số tham số
        self.num_params = sum(np.prod(shape) for shape in self.shapes)
        
        # Memory state
        self.memory = np.zeros(self.memory_dim)
        
        # Weights (mảng 1D)
        self.weights = np.zeros(self.num_params)

    def reset_memory(self):
        """Khởi tạo lại bộ nhớ khi bắt đầu vòng chạy mới"""
        self.memory = np.zeros(self.memory_dim)

    def set_weights(self, flat_weights):
        """Load trọng số (Nhiễm sắc thể) từ mảng 1D"""
        assert len(flat_weights) == self.num_params, "Sai kích thước trọng số!"
        self.weights = np.copy(flat_weights)

    def get_weights(self):
        """Trả về mảng trọng số 1D"""
        return np.copy(self.weights)

    def init_random_weights(self):
        """Khởi tạo trọng số ngẫu nhiên Gaussian (mu=0, std=0.5) — đủ lớn để tạo đa dạng hành vi cho GA"""
        self.weights = np.random.randn(self.num_params) * 0.5

    def act(self, obs_lidar):
        """
        Thực hiện một bước dự đoán (Forward Pass).
        Trả về (v_left, v_right) trong khoảng [0.0, 1.0].
        """
        # Giải nén mảng 1D thành các ma trận
        idx = 0
        w1_size = np.prod(self.shapes[0])
        w1 = self.weights[idx : idx + w1_size].reshape(self.shapes[0])
        idx += w1_size
        
        b1_size = np.prod(self.shapes[1])
        b1 = self.weights[idx : idx + b1_size].reshape(self.shapes[1])
        idx += b1_size
        
        w2_size = np.prod(self.shapes[2])
        w2 = self.weights[idx : idx + w2_size].reshape(self.shapes[2])
        idx += w2_size
        
        b2_size = np.prod(self.shapes[3])
        b2 = self.weights[idx : idx + b2_size].reshape(self.shapes[3])
        
        # Nối Lidar và Memory
        x = np.concatenate([obs_lidar, self.memory])
        
        # Forward pass
        # Hidden layer (ReLU — tự "tắt" ~50% nơ-ron, giảm không gian tìm kiếm hiệu quả cho GA)
        h1 = np.maximum(0, np.dot(x, w1) + b1)
        
        # Output layer
        raw_out = np.dot(h1, w2) + b2
        
        # Bánh xe dùng Sigmoid (0 đến 1) → luôn đi tới, rẽ bằng chênh lệch tốc độ (Theo yêu cầu của User)
        actions = 1.0 / (1.0 + np.exp(-np.clip(raw_out[:self.output_action_dim], -10, 10)))
        # Memory dùng Tanh (cần giá trị âm/dương)
        self.memory = np.tanh(raw_out[self.output_action_dim:])
        
        # Trả về v_left, v_right
        return actions[0], actions[1]
