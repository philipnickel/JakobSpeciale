"""
Work-Precision Analysis: RK3 vs RK4 with Varying Grid Resolution
=================================================================

Measures error vs computational cost for different time integrators
by varying grid size N. For each N, dt is automatically selected based
on a stability criterion (CFL-like condition).
"""

# %% Imports and setup -------------------------------------------------------
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK3, RK4

# %% Configuration -----------------------------------------------------------
DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Use same parameters as exercise_c
L = 40.0
X0 = 0.0
WAVE_SPEED = 0.5
T_FINAL = 1.0  # Fixed final time

# Vary N instead of dt
N_VALUES = np.arange(50, 400, 50)

# Automatic dt selection based on stability
# For spectral KdV, stability requires dt ~ 1/N or dt ~ 1/N^2
# We use dt = DT_FACTOR / N to ensure stability
# This gives more timesteps for larger N (better temporal resolution)
# Reduced factor for better stability at higher N
DT_FACTOR = 0.01

METHODS = {"RK3": RK3, "RK4": RK4}
N_TRIALS = 5  # Multiple trials for timing statistics

print("=" * 70)
print("Work-Precision Analysis: KdV Solver with Varying N")
print("=" * 70)
print(f"Domain: x ∈ [{-L}, {L}]")
print(f"Grid sizes: N ∈ [{N_VALUES[0]}, {N_VALUES[-1]}]")
print(f"Final time: T = {T_FINAL}")
print(f"Soliton: c = {WAVE_SPEED}, x0 = {X0}")
print(f"dt selection: dt = {DT_FACTOR} / N (stability-based)")
print(f"Number of timing trials per N: {N_TRIALS}")
print("=" * 70)

# %% Run work-precision experiments ------------------------------------------
results = []

for method_name, method_class in METHODS.items():
    print(f"\n{method_name} (Order {3 if method_name == 'RK3' else 4}):")
    print("-" * 70)

    # RHS evaluations per step
    rhs_evals_per_step = 3 if method_name == "RK3" else 4

    for N in N_VALUES:
        # Setup solver for this N
        solver = KdVSolver(N, L, dealias=True)
        x = solver.x
        dx = solver.dx

        # Automatically select dt based on stability: dt ~ 1/N
        dt = DT_FACTOR / N
        n_steps = int(np.round(T_FINAL / dt))
        t_final = n_steps * dt

        # Initial condition (soliton at t=0)
        u0 = soliton(x, 0.0, WAVE_SPEED, X0)

        # Warmup run (excluded from timing data)
        # Ensures JIT compilation and cache warming before measurements
        u_warmup = u0.copy()
        t_warmup = 0.0
        integrator_warmup = method_class()
        for step in range(n_steps):
            u_warmup = integrator_warmup.step(solver.rhs, u_warmup, t_warmup, dt)
            t_warmup += dt

        # Run multiple timing trials
        timing_trials = []
        for trial in range(N_TRIALS):
            t = 0.0
            u = u0.copy()
            integrator = method_class()  # Fresh integrator for each trial

            start_time = time.perf_counter()
            for step in range(n_steps):
                u = integrator.step(solver.rhs, u, t, dt)
                t += dt
            wall_time = time.perf_counter() - start_time
            timing_trials.append(wall_time)

        # Compute error once (separate from timing)
        u = u0.copy()
        t = 0.0
        integrator = method_class()
        for step in range(n_steps):
            u = integrator.step(solver.rhs, u, t, dt)
            t += dt

        u_exact = soliton(x, t_final, WAVE_SPEED, X0)
        diff = u - u_exact
        error_l2 = float(np.sqrt(np.sum(diff**2) * dx))
        error_linf = float(np.max(np.abs(diff)))

        # Total RHS evaluations
        total_rhs_evals = n_steps * rhs_evals_per_step

        # Store one result per trial (for error bar visualization)
        for trial, wall_time in enumerate(timing_trials):
            results.append(
                {
                    "method": method_name,
                    "N": N,
                    "L": L,
                    "dt": dt,
                    "T_FINAL": t_final,
                    "n_steps": n_steps,
                    "trial": trial,
                    "wall_time": wall_time,
                    "rhs_evaluations": total_rhs_evals,
                    "error_l2": error_l2,
                    "error_linf": error_linf,
                }
            )

        # Print with mean timing
        mean_time = np.mean(timing_trials)
        std_time = np.std(timing_trials)
        print(
            f"  N = {N:4d}, dt = {dt:.3e} ({n_steps:4d} steps)  "
            f"L2 err = {error_l2:.3e}  L∞ err = {error_linf:.3e}  "
            f"time = {mean_time:.4f}±{std_time:.4f}s"
        )

# %% Save results ------------------------------------------------------------
df = pd.DataFrame(results)
output_file = DATA_DIR / "work_precision.parquet"
df.to_parquet(output_file, index=False)

print("\n" + "=" * 70)
print(f"✓ Results saved to: {output_file}")
print(f"  Shape: {df.shape}")
print("=" * 70)
