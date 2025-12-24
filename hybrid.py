import numpy as np

dim = 3; num_salps = 30; max_iter = 100; lb = -5; ub = 5

def objective_function(p):
    return 0.5 * np.sum(p**4 - 16*p**2 + 5*p, axis=-1 if p.ndim > 1 else 0)

# === HYBRID INIT ===
def hybrid_init(n, d):
    pop = np.zeros((n, d))
    half = n // 2
    # 50% đầu: Random toàn dải [-5, 5]
    pop[:half] = np.random.uniform(lb, ub, (half, d))
    # 50% sau: Random vùng ÂM [-5, 0] để ưu tiên hố sâu
    pop[half:] = np.random.uniform(lb, 0, (n - half, d))
    return pop

def run_hybrid_ssa():
    salp_positions = hybrid_init(num_salps, dim)
    salp_fitness = objective_function(salp_positions)
    
    best_idx = np.argmin(salp_fitness)
    best_pos = salp_positions[best_idx].copy()
    best_fit = salp_fitness[best_idx]

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
        
        # Cập nhật Best
        salp_fitness = objective_function(salp_positions)
        current_best_idx = np.argmin(salp_fitness)
        if salp_fitness[current_best_idx] < best_fit:
            best_fit = salp_fitness[current_best_idx]
            best_pos = salp_positions[current_best_idx].copy()

    return best_pos, best_fit

if __name__ == "__main__":
    print("--- HYBRID SSA (NUMPY) ---")
    pos, val = run_hybrid_ssa()
    print(f"Vị trí: {np.round(pos, 4)}")
    print(f"Giá trị: {val:.4f}")