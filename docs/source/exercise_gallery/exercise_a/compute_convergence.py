"""
Legendre Tau vs Collocation - Convergence Study
================================================

Generates convergence data: N, method, eps, Linf_err, L2_err
"""

# %% Imports and setup
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legval

from spectral.bvp import solve_legendre_collocation, solve_legendre_tau

# %% Configuration
data_dir = Path("data/A2/ex_a")
data_dir.mkdir(parents=True, exist_ok=True)


# %% Define exact solution
def exact_solution(x: np.ndarray, eps: float) -> np.ndarray:
    """Analytical solution for boundary value problem."""
    num = np.exp(-x / eps) + (x - 1.0) - np.exp(-1.0 / eps) * x
    den = np.exp(-1.0 / eps) - 1.0
    return num / den


# %% Convergence study parameters
eps_values = np.array([1e-1, 1e-2, 1e-3])  # Diffusion parameters
N_values = np.arange(10, 71, 2)  # Number of modes/nodes for convergence study

print(f"Running convergence study for {len(eps_values)} epsilon values")
print(f"  N values: {N_values[0]} to {N_values[-1]} (step={N_values[1] - N_values[0]})")

# %% Run convergence study
convergence_data = []

for eps in eps_values:
    print(f"\n  Processing ε = {eps}:")

    for N in N_values:
        # Evaluation points on reference domain
        xi = np.linspace(-1.0, 1.0, 2001)
        x = 0.5 * (xi + 1.0)  # Physical domain [0,1]

        # Build Vandermonde matrix
        V = np.column_stack([legval(xi, [0] * n + [1]) for n in range(N)])

        # Compute coefficients
        coeff_tau = solve_legendre_tau(eps, N)
        _, coeff_col = solve_legendre_collocation(eps, N)

        # Evaluate solutions
        u_tau = V @ coeff_tau
        u_col = V @ coeff_col
        u_exact = exact_solution(x, eps)

        # Compute errors
        err_tau = np.abs(u_tau - u_exact)
        err_col = np.abs(u_col - u_exact)

        dx = np.diff(x)
        dx = np.append(dx, dx[-1])

        L2_tau = np.sqrt(np.sum(err_tau**2 * dx))
        Linf_tau = err_tau.max()
        L2_col = np.sqrt(np.sum(err_col**2 * dx))
        Linf_col = err_col.max()

        # Store results
        convergence_data.append(
            {
                "N": N,
                "method": "Tau",
                "eps": eps,
                "L2_err": L2_tau,
                "Linf_err": Linf_tau,
            }
        )

        convergence_data.append(
            {
                "N": N,
                "method": "Collocation",
                "eps": eps,
                "L2_err": L2_col,
                "Linf_err": Linf_col,
            }
        )

    print(f"    N={N_values[-1]}: Tau L∞={L2_tau:.2e}, Collocation L∞={L2_col:.2e}")

# %% Save convergence data
convergence_df = pd.DataFrame(convergence_data)
convergence_df["method"] = convergence_df["method"].astype("category")

output_path = data_dir / "convergence.parquet"
convergence_df.to_parquet(output_path, index=False)

print(f"\nConvergence data saved to {output_path}")
print(f"  Shape: {convergence_df.shape}")
print(f"  Columns: {convergence_df.columns.tolist()}")
