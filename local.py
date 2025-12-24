import numpy as np

dim = 3; num_salps = 30; max_iter = 100; lb = -5; ub = 5

def objective_function(p):
    return 0.5 * np.sum(p**4 - 16*p**2 + 5*p, axis=-1 if p.ndim > 1 else 0)

# === LOCAL SEARCH ===
def perform_local_search(start_pos, start_fit):
    curr_pos = start_pos.copy()
    curr_fit = start_fit
    
    # Thử 100 lần rung lắc
    for _ in range(100):
        # Tạo nhiễu ngẫu nhiên khoảng [-0.1, 0.1]
        noise = np.random.uniform(-0.1, 0.1, dim)
        new_pos = np.clip(curr_pos + noise, lb, ub)
        
        new_fit = objective_function(new_pos)
        
        if new_fit < curr_fit:
            curr_fit = new_fit
            curr_pos = new_pos # Numpy copy by assignment here is fine for vector
            
    return curr_pos, curr_fit

def run_localsearch_ssa():
    # SSA Gốc (Viết gọn)
    salp_positions = np.random.uniform(lb, ub, (num_salps, dim))
    salp_fitness = objective_function(salp_positions)
    
    idx = np.argmin(salp_fitness)
    best_pos = salp_positions[idx].copy(); best_fit = salp_fitness[idx]

    for t in range(max_iter):
        c1 = 2 * np.exp(-((4 * t / max_iter) ** 2))
        old_positions = salp_positions.copy()
        
        for i in range(num_salps):
            if i == 0:
                c2 = np.random.rand(dim); c3 = np.random.rand(dim)
                direction = np.where(c3 >= 0.5, 1, -1)
                salp_positions[i] = best_pos + direction * (c1 * ((ub - lb) * c2 + lb))
            else:
                salp_positions[i] = 0.5 * (salp_positions[i] + old_positions[i-1])
        
        salp_positions = np.clip(salp_positions, lb, ub)
        salp_fitness = objective_function(salp_positions)
        
        curr_min_idx = np.argmin(salp_fitness)
        if salp_fitness[curr_min_idx] < best_fit:
            best_fit = salp_fitness[curr_min_idx]
            best_pos = salp_positions[curr_min_idx].copy()

    print(f"SSA Gốc dừng tại: {best_fit:.4f}")
    
    # Chạy Local Search
    final_pos, final_fit = perform_local_search(best_pos, best_fit)
    return final_pos, final_fit

if __name__ == "__main__":
    print("--- LOCAL SEARCH SSA (NUMPY) ---")
    pos, val = run_localsearch_ssa()
    print(f"Vị trí cuối: {np.round(pos, 4)}")
    print(f"Giá trị cuối: {val:.4f}")