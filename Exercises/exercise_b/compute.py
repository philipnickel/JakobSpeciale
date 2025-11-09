"""
Polar BVP - Data Generation
============================

Data stored in tidy dataframes:

1. convergence.parquet: Nr, Linf_err
2. solution.parquet: r, phi, u, u_exact, pointwise_err, Nr, r1, r2, L2_err, Linf_err

Each row in ex_b_solution.parquet represents one grid point (r, phi).
"""

# %% Imports
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.bvp import solve_polar_bvp

# %% Configuration
data_dir = Path("data/A2/ex_b")
data_dir.mkdir(parents=True, exist_ok=True)

# %% Convergence study
print("Running convergence study...")
Nrs = np.arange(10, 70, step=2)
Nthetas = np.arange(6, 30, step=2)
errors_Nr = np.zeros(Nrs.shape[0])
errors_Ntheta = np.zeros(Nthetas.shape[0])

r1 = 1
r2 = 10

for i, Nr in enumerate(Nrs):
    Phi, Phi_hat, Rs, Theta = solve_polar_bvp(r1, r2, Nr=Nr, Ntheta=50)
    errors_Nr[i] = np.max(np.abs(Phi - Phi_hat))
    print(f"  Nr={Nr}: max error = {errors_Nr[i]:.6e}")


for i, Ntheta in enumerate(Nthetas):
    Phi, Phi_hat, Rs, Theta = solve_polar_bvp(r1, r2, Nr=20, Ntheta=Ntheta)
    errors_Ntheta[i] = np.max(np.abs(Phi - Phi_hat))
    print(f"  Nr={Ntheta}: max error = {errors_Ntheta[i]:.6e}")

# %% Save convergence study results
convergence_df = pd.DataFrame({"Nr": Nrs, "Linf_err": errors_Nr})
convergence_df.to_parquet(data_dir / "convergence_r.parquet", index=False)
print(f"Saved convergence data: {data_dir}/convergence_r.parquet")

convergence_df = pd.DataFrame({"Ntheta": Nthetas, "Linf_err": errors_Ntheta})
convergence_df.to_parquet(data_dir / "convergence_theta.parquet", index=False)
print(f"Saved convergence data: {data_dir}/convergence_theta.parquet")

# %% Solve BVP for visualization
print("\nSolving BVP for visualization...")
r1 = 1
r2 = 10
Nr = 20
Ntheta = 20
Phi, Phi_hat, Rs, Theta = solve_polar_bvp(r1, r2, Nr, Ntheta)

print(f"  Solution shape: {Phi.shape}")
print(f"  Max error: {np.max(np.abs(Phi - Phi_hat)):.6e}")

# %% Build tidy DataFrame
# Flatten 2D grids into long-form data (one row per grid point)
pointwise_err = np.abs(Phi_hat - Phi)
dx = np.diff(Rs[0, :])
dx = np.append(dx, dx[-1])
dy = np.diff(Theta[:, 0])
dy = np.append(dy, dy[-1])

# Compute global errors (approximate L2 using grid integration)
dA = np.outer(dy, dx * Rs[0, :])  # Area element in polar coords: r*dr*dθ
L2_err = np.sqrt(np.sum(pointwise_err**2 * dA))
Linf_err = np.max(pointwise_err)

solution_df = pd.DataFrame(
    {
        "r": Rs.flatten(),
        "phi": Theta.flatten(),
        "u": Phi_hat.flatten(),
        "u_exact": Phi.flatten(),
        "pointwise_err": pointwise_err.flatten(),
        "Nr": Nr,
        "r1": r1,
        "r2": r2,
        "L2_err": L2_err,
        "Linf_err": Linf_err,
    }
)

solution_df.to_parquet(data_dir / "solution.parquet", index=False)
print(f"Saved solution data: {data_dir}/solution.parquet")

# %%
