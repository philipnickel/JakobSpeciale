"""
Scalability Analysis: Time per Step vs Grid Size
=================================================

Measures wall-clock time vs grid size N to verify O(N log N) complexity.
"""

# %% Imports and setup -------------------------------------------------------
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK4, RK3

# %% Configuration -----------------------------------------------------------
DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

L = 30.0
c = 0.5
x0 = 0.0
N_VALUES = 2 ** np.arange(6, 10)  # Powers of 2: [64, 128, ..., 8192]
METHODS = {"RK4": RK4, "RK3": RK3}
MIN_STEPS = 200
MAX_STEPS = 2000
N_TRIALS = 5  # Number of timing trials for confidence intervals

print("=" * 70)
print("Scalability Analysis: KdV Solver")
print("=" * 70)
print(f"Testing N = {N_VALUES[0]} to {N_VALUES[-1]}")
print(f"Number of timing trials: {N_TRIALS}")
print(f"Methods: {', '.join(METHODS.keys())}\n")

# %% Run timing experiments --------------------------------------------------
results = []

for method_name, method_class in METHODS.items():
    print(f"\n{method_name}:")
    print("-" * 70)

    for N in N_VALUES:
        # Setup solver
        solver = KdVSolver(N, L, dealias=False)
        integrator = method_class()
        u0 = soliton(solver.x, 0.0, c, x0)

        # Estimate stable timestep
        u_max = float(np.max(np.abs(u0)))
        dt_stable = KdVSolver.stable_dt(
            N, L, u_max, integrator_name=method_name.lower()
        )
        if not np.isfinite(dt_stable) or dt_stable <= 0.0:
            dt_stable = 1e-3

        # Use moderate dt for timing (0.3× stable)
        dt = 0.3 * dt_stable
        dt = min(dt, 1.0 / MIN_STEPS)

        # Determine number of steps
        T_effective = min(1.0, MAX_STEPS * dt)
        T_effective = max(T_effective, MIN_STEPS * dt)
        n_steps = int(T_effective / dt)

        # Warm up (trigger JIT compilation)
        for _ in range(10):
            _ = integrator.step(solver.rhs, u0, 0.0, dt)

        # Run multiple timing trials
        timing_results = []
        for trial in range(N_TRIALS):
            u = u0.copy()
            t = 0.0
            wall_start = time.perf_counter()

            for _ in range(n_steps):
                u = integrator.step(solver.rhs, u, t, dt)
                t += dt

            wall_elapsed = time.perf_counter() - wall_start
            time_per_step = wall_elapsed / n_steps
            timing_results.append(time_per_step)

            # Store result for this trial
            results.append(
                {
                    "method": method_name,
                    "N": N,
                    "trial": trial,
                    "time_per_step": time_per_step,
                    "wall_time": wall_elapsed,
                    "n_steps": n_steps,
                }
            )

        # Print with mean timing across trials
        mean_time = np.mean(timing_results)
        std_time = np.std(timing_results)
        print(
            f"  N={N:5d}  time/step={mean_time:.6f}±{std_time:.6f}s  "
            f"(across {N_TRIALS} trials)"
        )

# %% Save results ------------------------------------------------------------
df = pd.DataFrame(results)
output_file = DATA_DIR / "scalability_timing.parquet"
df.to_parquet(output_file, index=False)

print("\n" + "=" * 70)
print(f"Results saved to: {output_file}")
print(f"Shape: {df.shape}")
print("=" * 70)
