"""
KdV Eigenvalue Stability Analysis - Data Generation
====================================================

Generates eigenvalue data for stability analysis:

1. eigenvalue_stability.parquet: eigenvalues for different integrators
2. eigenvalue_scaling.parquet: eigenvalue scaling with grid resolution
"""

# %% Imports and setup
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton

# %% Configuration
data_dir = Path("data/A2/ex_c")
data_dir.mkdir(parents=True, exist_ok=True)

# %% Parameters
N = 40
L = 30.0
c_values = [0.25, 0.5, 1.0]
x0 = 0.0
methods = ["rk4", "rk3"]

print("Generating eigenvalue stability data for KdV equation")
print(f"  Grid: N={N}, L={L}")
print(f"  Soliton speeds: c={c_values}, x0={x0}")
print(f"  Methods: {methods}")

# %% Generate eigenvalue stability data for each method and c value
stability_data = []

for c in c_values:
    print(f"\nProcessing c = {c}...")
    solver = KdVSolver(N, L)
    x = solver.x
    u0 = soliton(x, 0.0, c, x0)
    u_max = float(np.max(np.abs(u0)))

    # Compute eigenvalues
    eigvals = solver.compute_eigenvalues(u_max)

    for method in methods:
        # Estimate stable timestep
        stable_dt = KdVSolver.stable_dt(N, L, u_max, integrator_name=method)

        # Scale eigenvalues by timestep
        eigvals_scaled = eigvals * stable_dt

        # Store each eigenvalue
        for eig, eig_scaled in zip(eigvals, eigvals_scaled):
            stability_data.append(
                {
                    "c": c,
                    "method": method,
                    "N": N,
                    "L": L,
                    "u_max": u_max,
                    "stable_dt": stable_dt,
                    "eigval_real": eig.real,
                    "eigval_imag": eig.imag,
                    "eigval_scaled_real": eig_scaled.real,
                    "eigval_scaled_imag": eig_scaled.imag,
                }
            )

        print(f"  c={c}, {method.upper()}: stable_dt = {stable_dt:.4e}")

# Save stability data
stability_df = pd.DataFrame(stability_data)
stability_df["method"] = stability_df["method"].astype("category")
output_path = data_dir / "eigenvalue_stability.parquet"
stability_df.to_parquet(output_path, index=False)
print(f"\nSaved eigenvalue stability data to {output_path}")
print(f"  Shape: {stability_df.shape}")


N_values = [32, 64, 128, 256]
scaling_data = []

for c in c_values:
    print(f"\nScaling analysis for c = {c}...")
    for N_test in N_values:
        solver_test = KdVSolver(N_test, L)
        x_test = solver_test.x
        u0_test = soliton(x_test, 0.0, c, x0)
        u_max_test = float(np.max(np.abs(u0_test)))

        # Compute eigenvalues
        eigvals_test = solver_test.compute_eigenvalues(u_max_test)
        max_eig = float(np.max(np.abs(eigvals_test)))

        for method in methods:
            dt = KdVSolver.stable_dt(N_test, L, u_max_test, integrator_name=method)

            scaling_data.append(
                {
                    "c": c,
                    "method": method,
                    "N": N_test,
                    "L": L,
                    "dx": solver_test.dx,
                    "u_max": u_max_test,
                    "max_eigval": max_eig,
                    "stable_dt": dt,
                }
            )


# Save scaling data
scaling_df = pd.DataFrame(scaling_data)
output_path = data_dir / "eigenvalue_scaling.parquet"
scaling_df.to_parquet(output_path, index=False)
