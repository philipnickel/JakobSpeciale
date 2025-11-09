"""
Legendre Tau vs Collocation - Data Generation
==============================================

Data stored in a single tidy dataframe with columns:

method, eps, N, data_type, x, u, u_exact, pointwise_err, mode, abs_coeff, L2_err, Linf_err

where data_type ∈ {'solution', 'coefficient'}

- For solution rows: x, u, u_exact, pointwise_err are filled; mode, abs_coeff are NaN
- For coefficient rows: mode, abs_coeff are filled; x, u, u_exact, pointwise_err are NaN
- L2_err and Linf_err are the same for all rows in a (method, eps) group

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


# %% Problem parameters
eps_values = np.array([1e-1, 1e-2, 1e-3])  # Diffusion parameters
N = 50  # Number of modes/nodes
xi = np.linspace(-1.0, 1.0, 2001)  # Evaluation points on reference domain
x = 0.5 * (xi + 1.0)  # Physical domain [0,1]

print(f"Computing solutions for {len(eps_values)} epsilon values using N={N} modes")

# %% Build Vandermonde matrix for evaluation
V = np.column_stack([legval(xi, [0] * n + [1]) for n in range(N)])  # (M, N)

# %% Compute solutions and build unified tidy DataFrame
dfs = []

for eps in eps_values:
    print(f"  Processing ε = {eps}")

    # Compute coefficients
    coeff_tau = solve_legendre_tau(eps, N)
    _, coeff_col = solve_legendre_collocation(eps, N)

    # Evaluate solutions
    u_tau = V @ coeff_tau
    u_col = V @ coeff_col
    u_exact = exact_solution(x, eps)

    # Compute global errors
    err_tau = np.abs(u_tau - u_exact)
    err_col = np.abs(u_col - u_exact)
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])

    L2_tau = np.sqrt(np.sum(err_tau**2 * dx))
    Linf_tau = err_tau.max()
    L2_col = np.sqrt(np.sum(err_col**2 * dx))
    Linf_col = err_col.max()

    # Build DataFrames for each method
    for method_name, u_numerical, coeffs, L2, Linf in [
        ("Tau", u_tau, coeff_tau, L2_tau, Linf_tau),
        ("Collocation", u_col, coeff_col, L2_col, Linf_col),
        ("Exact", u_exact, np.zeros(N), 0.0, 0.0),
    ]:
        # Solution data rows
        df_sol = pd.DataFrame(
            {
                "method": method_name,
                "eps": eps,
                "N": N,
                "data_type": "solution",
                "x": x,
                "u": u_numerical,
                "u_exact": u_exact,
                "pointwise_err": np.abs(u_numerical - u_exact),
                "mode": np.nan,
                "abs_coeff": np.nan,
                "L2_err": L2,
                "Linf_err": Linf,
            }
        )

        # Coefficient data rows
        df_coef = pd.DataFrame(
            {
                "method": method_name,
                "eps": eps,
                "N": N,
                "data_type": "coefficient",
                "x": np.nan,
                "u": np.nan,
                "u_exact": np.nan,
                "pointwise_err": np.nan,
                "mode": np.arange(N),
                "abs_coeff": np.abs(coeffs),
                "L2_err": L2,
                "Linf_err": Linf,
            }
        )

        dfs.extend([df_sol, df_coef])

# %% Concatenate and save
df = pd.concat(dfs, ignore_index=True)
df["method"] = df["method"].astype("category")
df["data_type"] = df["data_type"].astype("category")

output_path = data_dir / "data.parquet"
df.to_parquet(output_path, index=False)

print(f"\nData saved to {output_path}")
