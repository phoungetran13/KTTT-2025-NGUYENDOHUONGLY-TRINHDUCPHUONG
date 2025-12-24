import numpy as np

# --- CẤU HÌNH ---
dim = 3
num_salps = 30
max_iter = 100
lb = -5
ub = 5

# Hàm Styblinski-Tang (Vector hóa)
def objective_function(matrix_pos):
    # Tính toán song song cho cả hàng hoặc từng vector
    # axis=-1 nghĩa là cộng dồn theo chiều biến (x+y+z)
    term = (matrix_pos**4) - (16 * matrix_pos**2) + (5 * matrix_pos)
    return 0.5 * np.sum(term, axis=-1 if matrix_pos.ndim > 1 else 0)

def run_basic_ssa():
    # 1. Khởi tạo (Ma trận [30, 3])
    salp_positions = np.random.uniform(lb, ub, (num_salps, dim))
    
    # Tính fitness cho cả đàn 1 lúc
    salp_fitness = objective_function(salp_positions)
    
    # Tìm Best ban đầu
    min_idx = np.argmin(salp_fitness)
    best_pos = salp_positions[min_idx].copy()
    best_fit = salp_fitness[min_idx]

    # 2. Vòng lặp chính
    for t in range(max_iter):
        c1 = 2 * np.exp(-((4 * t / max_iter) ** 2)) # [cite: 26]
        
        # Backup vị trí cũ để tính Follower
        old_positions = salp_positions.copy()
        
        for i in range(num_salps):
            if i == 0: # Leader
                c2 = np.random.rand(dim) # Tạo vector ngẫu nhiên
                c3 = np.random.rand(dim)
                
                # Numpy cho phép cộng trừ nguyên vector không cần vòng lặp for j
                # [cite: 25]
                direction = np.where(c3 >= 0.5, 1, -1) # Mảng chứa 1 hoặc -1
                step = c1 * ((ub - lb) * c2 + lb)
                salp_positions[i] = best_pos + (direction * step)
                
            else: # Follower
                # [cite: 27]
                salp_positions[i] = 0.5 * (salp_positions[i] + old_positions[i-1])

        # Kẹp biên (Clip) toàn bộ ma trận
        salp_positions = np.clip(salp_positions, lb, ub) # [cite: 30]

        # Cập nhật Fitness & Best
        salp_fitness = objective_function(salp_positions)
        min_idx = np.argmin(salp_fitness)
        
        if salp_fitness[min_idx] < best_fit:
            best_fit = salp_fitness[min_idx]
            best_pos = salp_positions[min_idx].copy()

    return best_pos, best_fit

if __name__ == "__main__":
    print("--- SSA CƠ BẢN (NUMPY) ---")
    pos, val = run_basic_ssa()
    print(f"Vị trí: {np.round(pos, 4)}")
    print(f"Giá trị: {val:.4f}")