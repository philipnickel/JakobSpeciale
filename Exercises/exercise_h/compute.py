"""
Transport Equation Solution using Legendre Collocation
=======================================================

Computes transport equation solution using Legendre collocation method.
"""

# %% Imports and setup
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.bvp import solve_transport_spacetime
from spectral.polynomial import legendre_gauss_lobatto_nodes

# %% Configuration
data_dir = Path("data/A2/ex_h")
data_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Transport Equation - Legendre Collocation")
print("=" * 60)

# %% Problem parameters
Nt = 100
Nx = Nt
t1, t2 = 0.0, 1.0
x1, x2 = 0.0, 2 * np.pi
a = 1.0
sigma = 1.0
mu = x2 / 2

print(f"Grid: Nx = {Nx}, Nt = {Nt}")
print(f"Domain: x ∈ [{x1}, {x2:.2f}], t ∈ [{t1}, {t2}]")
print(f"Wave speed: a = {a}")


# %% Helper function for true solution
def gaussian_traveling_wave(Xs, Ts):
    """Gaussian traveling wave: u(x,t) = F(x - at)."""
    return (
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((Xs - a * Ts - mu) / sigma) ** 2)
    )


# %% Solve using space-time collocation
print("Solving linear system...")
ys = legendre_gauss_lobatto_nodes(Nx)
xs = 0.5 * (x2 - x1) * (ys + 1) + x1
ts = 0.5 * (t2 - t1) * (ys + 1) + t1
Ts, Xs = np.meshgrid(ts, xs)
Phi = gaussian_traveling_wave(Xs, Ts)

Phi, Phi_hat, Ts, Xs = solve_transport_spacetime(Nx, Nt, x1, x2, t1, t2, a, Phi)

# %% Compute and display errors
error = Phi - Phi_hat
error_l2 = np.linalg.norm(error)
error_max = np.abs(error).max()
xs = Xs[:, 0]
ts = Ts[0, :]

print(f"L2 error: {error_l2:.6e}")
print(f"Max error: {error_max:.6e}")

# %% Store in long/tidy format (seaborn-ready)
print("\nSaving solution data...")

dfs = []
for type_name, data in [("True", Phi), ("Predicted", Phi_hat), ("Error", error)]:
    df_type = (
        pd.DataFrame(data, index=xs, columns=ts)
        .rename_axis(index="x", columns="t")
        .stack()
        .reset_index(name="value")
        .assign(type=type_name)
    )
    dfs.append(df_type)

df = pd.concat(dfs, ignore_index=True)

# %% Save to parquet
output_path = data_dir / "solution.parquet"
df.to_parquet(output_path)
print("=" * 60)
print("Computation complete!")
print("=" * 60)

# %% Spatial Convergence Study (varying Nx, fixed Nt)
print("\n" + "=" * 60)
print("Spatial Convergence Study")
print("=" * 60)

# Fixed temporal resolution
Nt_fixed = 100

Nx_values = np.arange(5, 50, 5)  # 5, 10, 15, 20, ..., 100

spatial_convergence_data = []

for Nx in Nx_values:
    # Build meshgrid and exact solution
    ys_x = legendre_gauss_lobatto_nodes(Nx)
    ys_t = legendre_gauss_lobatto_nodes(Nt_fixed)
    xs_conv = 0.5 * (x2 - x1) * (ys_x + 1) + x1
    ts_conv = 0.5 * (t2 - t1) * (ys_t + 1) + t1
    Ts_conv, Xs_conv = np.meshgrid(ts_conv, xs_conv)
    Phi_true = gaussian_traveling_wave(Xs_conv, Ts_conv)

    # Solve
    _, Phi_hat_conv, _, _ = solve_transport_spacetime(
        Nx, Nt_fixed, x1, x2, t1, t2, a, Phi_true
    )

    # Compute errors
    error_conv = Phi_true - Phi_hat_conv
    error_l2_conv = np.linalg.norm(error_conv)
    error_max_conv = np.abs(error_conv).max()

    # Append in long format directly
    spatial_convergence_data.append(
        {
            "Nx": Nx,
            "Nt": Nt_fixed,
            "Error_Type": "L2_error",
            "Error": error_l2_conv,
        }
    )
    spatial_convergence_data.append(
        {
            "Nx": Nx,
            "Nt": Nt_fixed,
            "Error_Type": "Linf_error",
            "Error": error_max_conv,
        }
    )

    if Nx % 20 == 0:
        print(f"  Nx={Nx:3d}: L2={error_l2_conv:.6e}, L∞={error_max_conv:.6e}")

# Create DataFrame (already in long format)
df_spatial = pd.DataFrame(spatial_convergence_data)
output_spatial = data_dir / "spatial_convergence.parquet"
df_spatial.to_parquet(output_spatial, index=False)

print(f"\nSpatial convergence data saved to {output_spatial}")
print(f"  Shape: {df_spatial.shape}")

# %% Temporal Convergence Study (varying Nt, fixed Nx)
print("\n" + "=" * 60)
print("Temporal Convergence Study")
print("=" * 60)

# Fixed spatial resolution
Nx_fixed = 100
Nt_values = np.arange(2, 40, 1)  # 3, 5, 7, 9, ..., 101

temporal_convergence_data = []

for Nt in Nt_values:
    # Build meshgrid and exact solution
    ys_x = legendre_gauss_lobatto_nodes(Nx_fixed)
    ys_t = legendre_gauss_lobatto_nodes(Nt)
    xs_conv = 0.5 * (x2 - x1) * (ys_x + 1) + x1
    ts_conv = 0.5 * (t2 - t1) * (ys_t + 1) + t1
    Ts_conv, Xs_conv = np.meshgrid(ts_conv, xs_conv)
    Phi_true = gaussian_traveling_wave(Xs_conv, Ts_conv)

    # Solve
    _, Phi_hat_conv, _, _ = solve_transport_spacetime(
        Nx_fixed, Nt, x1, x2, t1, t2, a, Phi_true
    )

    # Compute errors
    error_conv = Phi_true - Phi_hat_conv
    error_l2_conv = np.linalg.norm(error_conv)
    error_max_conv = np.abs(error_conv).max()

    # Append in long format directly
    temporal_convergence_data.append(
        {
            "Nt": Nt,
            "Nx": Nx_fixed,
            "Error_Type": "L2_error",
            "Error": error_l2_conv,
        }
    )
    temporal_convergence_data.append(
        {
            "Nt": Nt,
            "Nx": Nx_fixed,
            "Error_Type": "Linf_error",
            "Error": error_max_conv,
        }
    )

    if Nt % 20 == 0:
        print(f"  Nt={Nt:3d}: L2={error_l2_conv:.6e}, L∞={error_max_conv:.6e}")

# Create DataFrame (already in long format)
df_temporal = pd.DataFrame(temporal_convergence_data)
output_temporal = data_dir / "temporal_convergence.parquet"
df_temporal.to_parquet(output_temporal, index=False)

print(f"\nTemporal convergence data saved to {output_temporal}")
print(f"  Shape: {df_temporal.shape}")

print("\n" + "=" * 60)
print("All convergence studies complete!")
print("=" * 60)
