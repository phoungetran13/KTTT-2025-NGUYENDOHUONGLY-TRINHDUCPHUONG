import numpy as np

dim = 3; num_salps = 30; max_iter = 100; lb = -5; ub = 5
num_elites = 3 # Giữ lại 3 con tốt nhất

def objective_function(p):
    return 0.5 * np.sum(p**4 - 16*p**2 + 5*p, axis=-1 if p.ndim > 1 else 0)

def run_elitism_ssa():
    salp_positions = np.random.uniform(lb, ub, (num_salps, dim))
    salp_fitness = objective_function(salp_positions)
    
    # Best ban đầu
    idx = np.argmin(salp_fitness)
    best_pos = salp_positions[idx].copy()
    best_fit = salp_fitness[idx]

    for t in range(max_iter):
        # === BƯỚC SẮP XẾP ===
        # Lấy danh sách index đã sort theo fitness tăng dần
        sorted_indices = np.argsort(salp_fitness) 
        salp_positions = salp_positions[sorted_indices]
        salp_fitness = salp_fitness[sorted_indices]
        
        # Cập nhật Best Global
        if salp_fitness[0] < best_fit:
            best_fit = salp_fitness[0]
            best_pos = salp_positions[0].copy()

        c1 = 2 * np.exp(-((4 * t / max_iter) ** 2))
        old_positions = salp_positions.copy()

        for i in range(num_salps):
            # [ELITISM] Bỏ qua các con nằm trong top Elite
            if i < num_elites:
                continue 
            
            # Leader (Con ngay sau Elite)
            if i == num_elites:
                c2 = np.random.rand(dim); c3 = np.random.rand(dim)
                direction = np.where(c3 >= 0.5, 1, -1)
                salp_positions[i] = best_pos + direction * (c1 * ((ub - lb) * c2 + lb))
            else:
                salp_positions[i] = 0.5 * (salp_positions[i] + old_positions[i-1])

        salp_positions = np.clip(salp_positions, lb, ub)
        salp_fitness = objective_function(salp_positions)

    return best_pos, best_fit

if __name__ == "__main__":
    print("--- ELITISM SSA (NUMPY) ---")
    pos, val = run_elitism_ssa()
    print(f"Vị trí: {np.round(pos, 4)}")
    print(f"Giá trị: {val:.4f}")