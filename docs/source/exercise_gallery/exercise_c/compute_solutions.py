"""
Generate KdV Soliton Samples (RK4, RK3)
=======================================

Generates one tidy DataFrame with soliton solutions using parallel computation.
"""

# %% Imports and setup
import os
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from spectral.tdp import KdVSolver, soliton, RK4, RK3

# %% Configuration
DATA_DIR = Path("data/A2/ex_c")
DATA_DIR.mkdir(parents=True, exist_ok=True)

L, x0, T = 30.0, 0.0, 20.0
c_vals = [0.25, 0.5, 1.0]  # different wave speeds
N_vals = [200]  # different spatial resolutions

dt_scales = [0.4, 0.2]  # build dt ladder: dt = dt0 * scale
save_every_steps = 100
max_workers = os.cpu_count() or 4

METHODS = ("RK4", "RK3")


# %% Helper: compute stable timestep for each (N, method, c) combination
def estimate_stable_dt(N: int, method_name: str, c: float, fallback=1e-2) -> float:
    """Estimate dt0 per (N, method, c) from initial u_max."""
    s = KdVSolver(N, L)
    u_max = float(np.max(np.abs(soliton(s.x, 0.0, c, x0))))
    dt_est = KdVSolver.stable_dt(N, L, u_max, integrator_name=method_name.lower())
    return float(dt_est) if np.isfinite(dt_est) else float(fallback)


# %% Define single-case solver (used by parallel workers)
def solve_single_case(method: str, N: int, dt: float, c: float):
    """Solve one (method, N, dt, c) case and return tidy DataFrame."""
    # Setup integrator
    integrator_map = {"RK4": RK4, "RK3": RK3}
    integ = integrator_map[method]()

    # Setup solver
    solver = KdVSolver(N, L)
    x = solver.x
    dx = solver.dx
    u0 = soliton(x, 0.0, c, x0)

    # Clear history for multi-step methods
    if hasattr(integ, "u_history"):
        integ.u_history, integ.f_history = [], []

    # Solve
    t_saved, u_hist = solver.solve(
        u0.copy(), T, dt, integrator=integ, save_every=save_every_steps
    )

    # Build tidy DataFrame
    dfs = []
    for t, u in zip(t_saved, u_hist):
        df_t = pd.DataFrame(
            {
                "x": x,
                "u": u,
                "u_exact": soliton(x, float(t), c, x0),
                "t": t,
            }
        )
        dfs.append(df_t)

    df = pd.concat(dfs, ignore_index=True)

    # Add metadata (pandas broadcasts scalars)
    df["method"] = method
    df["N"] = N
    df["dx"] = dx
    df["dt"] = dt
    df["T"] = T
    df["c"] = c
    df["x0"] = x0
    df["L"] = L

    return df


if __name__ == "__main__":
    # %% Build task list
    tasks = []
    for method in METHODS:
        for c in c_vals:
            for N in N_vals:
                dt0 = estimate_stable_dt(N, method, c, fallback=1e-2)
                for scale in dt_scales:
                    tasks.append((method, N, float(dt0 * scale), c))

    # Deduplicate tasks just in case
    # tasks = sorted(set(tasks))

    print(f"Generated {len(tasks)} tasks for parallel execution")

    # %% Run all cases in parallel
    dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(solve_single_case, m, N, dt, c): (m, N, dt, c)
            for (m, N, dt, c) in tasks
        }

        for future in as_completed(futures):
            m, N, dt, c = futures[future]
            try:
                dfs.append(future.result())
            except Exception as e:
                # On failure, record a placeholder row with NaNs
                dfs.append(
                    pd.DataFrame(
                        [
                            {
                                "method": m,
                                "N": np.int32(N),
                                "dx": np.nan,
                                "dt": np.float32(dt),
                                "T": np.float32(T),
                                "t": np.nan,
                                "x": np.nan,
                                "u": np.nan,
                                "u_exact": np.nan,
                                "c": np.float32(c),
                                "x0": np.float32(x0),
                                "L": np.float32(L),
                            }
                        ]
                    )
                )
                print(
                    f"[warn] case failed: method={m}, N={N}, dt={dt:.3e}, c={c} -> {e}"
                )

    # %% Concatenate and save
    df = pd.concat(dfs, ignore_index=True)
    df["method"] = df["method"].astype("category")

    out = DATA_DIR / "kdv_solutions.parquet"
    df.to_parquet(out, index=False)

    print(f"Saved {len(df):,} rows → {out}")
