"""
Single Soliton Error and Conservation Analysis
===============================================

Investigates:

1. Domain investigation: Fixed N=128, varying L=[20, 30, 40]
2. Node investigation: Fixed L=30, varying N=[32, 64, 128]

For each configuration, computes:

- L2 and Linf errors over time
- Conservation quantities (M, V, E) over time
"""

# %% Imports and setup
import os
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from spectral.tdp import KdVSolver, soliton, RK4, RK3
from spectral.utils.norms import discrete_l2_error, discrete_linf_error

# %% Configuration
DATA_DIR = Path("data/A2/ex_d")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Common parameters
c_vals = [0.25, 0.5, 1.0]
x0 = 0.0
T = 20.0
METHODS = ("RK4", "RK3")
save_every_steps = 50  # Save every 50 steps

# Domain investigation: varying L, fixed N
DOMAIN_N = 128
DOMAIN_L_vals = [20.0, 30.0, 40.0]

# Node investigation: varying N, fixed L
NODES_L = 30.0
NODES_N_vals = [32, 64, 128]

max_workers = os.cpu_count() or 4


# %% Helper: compute stable timestep with safety factor
def estimate_stable_dt(
    N: int, L: float, method_name: str, c: float, safety_factor=0.1
) -> float:
    """Estimate stable dt with safety factor."""
    s = KdVSolver(N, L)
    u_max = float(np.max(np.abs(soliton(s.x, 0.0, c, x0))))
    dt_est = KdVSolver.stable_dt(N, L, u_max, integrator_name=method_name.lower())
    dt_safe = safety_factor * dt_est if np.isfinite(dt_est) else 1e-3
    return float(dt_safe)


# %% Single case solver
def solve_single_case(method: str, N: int, L: float, c: float, investigation_type: str):
    """Solve one case and compute errors + conservation quantities in long format."""
    # Setup integrator
    integrator_map = {"RK4": RK4, "RK3": RK3}
    integ = integrator_map[method]()

    # Setup solver
    solver = KdVSolver(N, L)
    x = solver.x
    dx = solver.dx
    u0 = soliton(x, 0.0, c, x0)

    # Estimate stable dt
    dt = estimate_stable_dt(N, L, method, c, safety_factor=0.1)

    # Clear history for multi-step methods
    if hasattr(integ, "u_history"):
        integ.u_history, integ.f_history = [], []

    # Solve
    t_saved, u_hist = solver.solve(
        u0.copy(), T, dt, integrator=integ, save_every=save_every_steps
    )

    # Compute initial conservation quantities for normalization
    M0, V0, E0 = KdVSolver.compute_conserved_quantities(u0, dx)

    # Build results in LONG format
    error_rows = []
    quantity_rows = []

    for t, u in zip(t_saved, u_hist):
        u_exact = soliton(x, float(t), c, x0)

        # Compute errors
        l2_err = discrete_l2_error(u_exact, u, 2 * L)
        linf_err = discrete_linf_error(u_exact, u)

        # Compute conservation quantities
        M, V, E = KdVSolver.compute_conserved_quantities(u, dx)

        # Common metadata
        metadata = {
            "t": t,
            "method": method,
            "N": N,
            "L": L,
            "c": c,
            "dt": dt,
            "investigation": investigation_type,
        }

        # Store errors in LONG format (one row per error type)
        for error_type, error_value in [("l2", l2_err), ("linf", linf_err)]:
            error_rows.append(
                {
                    **metadata,
                    "error_type": error_type,
                    "error": error_value,
                }
            )

        # Store quantities in LONG format (one row per quantity)
        for qty_name, qty_value, qty_initial in [
            ("M", M, M0),
            ("V", V, V0),
            ("E", E, E0),
        ]:
            rel_error = (
                (qty_value - qty_initial) / qty_initial if qty_initial != 0 else 0.0
            )
            quantity_rows.append(
                {
                    **metadata,
                    "quantity": qty_name,
                    "value": qty_value,
                    "rel_error": rel_error,
                }
            )

    return error_rows, quantity_rows


if __name__ == "__main__":
    # %% Build task list
    print("Building task list for exercise d)...")
    tasks = []

    # Domain investigation tasks
    for method in METHODS:
        for c in c_vals:
            for L in DOMAIN_L_vals:
                tasks.append((method, DOMAIN_N, L, c, "domain"))

    # Node investigation tasks
    for method in METHODS:
        for c in c_vals:
            for N in NODES_N_vals:
                tasks.append((method, N, NODES_L, c, "nodes"))

    print(f"Total tasks: {len(tasks)}")
    print(
        f"  Domain investigation: {len(METHODS) * len(c_vals) * len(DOMAIN_L_vals)} tasks"
    )
    print(
        f"  Node investigation: {len(METHODS) * len(c_vals) * len(NODES_N_vals)} tasks"
    )

    # %% Run all cases in parallel
    all_error_rows = []
    all_quantity_rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(solve_single_case, m, N, L, c, inv_type): (
                m,
                N,
                L,
                c,
                inv_type,
            )
            for (m, N, L, c, inv_type) in tasks
        }

        completed = 0
        for future in as_completed(futures):
            m, N, L, c, inv_type = futures[future]
            try:
                error_rows, quantity_rows = future.result()
                all_error_rows.extend(error_rows)
                all_quantity_rows.extend(quantity_rows)
                completed += 1
                if completed % 5 == 0:
                    print(f"Completed {completed}/{len(tasks)} tasks...")
            except Exception as e:
                print(
                    f"[warn] Task failed: method={m}, N={N}, L={L}, c={c}, type={inv_type} -> {e}"
                )

    # %% Create DataFrames and save
    print("\nCreating DataFrames...")

    # Error data (already in long format)
    df_errors = pd.DataFrame(all_error_rows)
    df_errors["method"] = df_errors["method"].astype("category")
    df_errors["investigation"] = df_errors["investigation"].astype("category")
    df_errors["error_type"] = df_errors["error_type"].astype("category")

    # Quantity data (already in long format)
    df_quantities = pd.DataFrame(all_quantity_rows)
    df_quantities["method"] = df_quantities["method"].astype("category")
    df_quantities["investigation"] = df_quantities["investigation"].astype("category")
    df_quantities["quantity"] = df_quantities["quantity"].astype("category")

    # Save domain investigation data
    df_domain_errors = df_errors[df_errors["investigation"] == "domain"].copy()
    df_domain_quantities = df_quantities[
        df_quantities["investigation"] == "domain"
    ].copy()

    out_domain_err = DATA_DIR / "domain_errors.parquet"
    out_domain_qty = DATA_DIR / "domain_quantities.parquet"
    df_domain_errors.to_parquet(out_domain_err, index=False)
    df_domain_quantities.to_parquet(out_domain_qty, index=False)

    print("\nDomain investigation:")
    print(f"  Saved {len(df_domain_errors):,} error rows → {out_domain_err}")
    print(f"  Saved {len(df_domain_quantities):,} quantity rows → {out_domain_qty}")

    # Save node investigation data
    df_nodes_errors = df_errors[df_errors["investigation"] == "nodes"].copy()
    df_nodes_quantities = df_quantities[
        df_quantities["investigation"] == "nodes"
    ].copy()

    out_nodes_err = DATA_DIR / "nodes_errors.parquet"
    out_nodes_qty = DATA_DIR / "nodes_quantities.parquet"
    df_nodes_errors.to_parquet(out_nodes_err, index=False)
    df_nodes_quantities.to_parquet(out_nodes_qty, index=False)

    print("\nNode investigation:")
    print(f"  Saved {len(df_nodes_errors):,} error rows → {out_nodes_err}")
    print(f"  Saved {len(df_nodes_quantities):,} quantity rows → {out_nodes_qty}")

    print("\nData generation complete!")
