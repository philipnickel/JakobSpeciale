"""
Scalability Analysis for KdV Solver
====================================

Measures sequential performance (computational complexity):

- Wall time vs N for single cases
- Compare all methods: RK4, RK3
- Expected scaling: O(N log N) due to FFT operations
"""

# %% Imports and setup
from pathlib import Path
import time

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK4, RK3

# %% Configuration
DATA_DIR = Path("data/A2/ex_g")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Timing parameters
L = 30.0
c = 0.5
x0 = 0.0
T_timing = 1.0  # Very short simulation for timing
METHODS = ("RK4", "RK3")

# Sequential timing: vary N (extended range to see asymptotic behavior)
N_values = [32, 64, 128, 256, 512]


# %% Helper: stable dt estimation
def estimate_stable_dt(
    N: int, L: float, method_name: str, c: float, safety_factor=0.1
) -> float:
    """Estimate stable dt with safety factor."""
    s = KdVSolver(N, L)
    u_max = float(np.max(np.abs(soliton(s.x, 0.0, c, x0))))
    dt_est = KdVSolver.stable_dt(N, L, u_max, integrator_name=method_name.lower())
    dt_safe = safety_factor * dt_est if np.isfinite(dt_est) else 1e-3
    return float(dt_safe)


# %% Sequential timing function
def time_single_case(method: str, N: int, L: float, c: float, T: float):
    """Time a single case and return timing metrics."""
    # Setup integrator
    integrator_map = {"RK4": RK4, "RK3": RK3}
    integ = integrator_map[method]()

    # Setup solver
    solver = KdVSolver(N, L)
    x = solver.x
    u0 = soliton(x, 0.0, c, x0)

    # Estimate stable dt
    dt = estimate_stable_dt(N, L, method, c, safety_factor=0.1)

    # Clear history for multi-step methods
    if hasattr(integ, "u_history"):
        integ.u_history, integ.f_history = [], []

    # Time the solve (use performance measurement)
    start_time = time.perf_counter()
    t_saved, u_hist, perf_metrics = solver.solve(
        u0.copy(),
        T,
        dt,
        integrator=integ,
        save_every=1000000,  # Don't save intermediate
        measure_performance=True,
    )
    end_time = time.perf_counter()

    wall_time = end_time - start_time
    n_steps = perf_metrics["nsteps"]
    time_per_step = perf_metrics["mean_step_time_ms"] / 1000.0  # Convert ms to s

    return {
        "method": method,
        "N": N,
        "L": L,
        "c": c,
        "T": T,
        "dt": dt,
        "n_steps": n_steps,
        "wall_time": wall_time,
        "time_per_step": time_per_step,
    }


if __name__ == "__main__":
    # %% Sequential performance (wall time vs N)
    print("=" * 60)
    print("Sequential Performance Analysis (Wall Time vs N)")
    print("=" * 60)

    timing_results = []

    for method in METHODS:
        print(f"\n{method}:")
        for N in N_values:
            print(f"  N={N:4d}...", end=" ", flush=True)
            result = time_single_case(method, N, L, c, T_timing)
            timing_results.append(result)
            print(
                f"time/step = {result['time_per_step']:.6f}s, total = {result['wall_time']:.3f}s"
            )

    # Save sequential timing results
    df_timing = pd.DataFrame(timing_results)
    df_timing["method"] = df_timing["method"].astype("category")
    out_timing = DATA_DIR / "scalability_timing.parquet"
    df_timing.to_parquet(out_timing, index=False)
    print(f"\nSaved timing data to {out_timing}")
    print(f"  Shape: {df_timing.shape}")

    print("\n" + "=" * 60)
    print("Scalability analysis complete!")
    print("=" * 60)
