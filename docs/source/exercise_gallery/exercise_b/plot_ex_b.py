"""
Polar BVP - Plotting
====================

Visualizes convergence study and solutions for the polar boundary value problem
using spectral collocation.
"""

# %%
# Convergence study
# -----------------
# Analyze spectral convergence for the polar BVP.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from spectral.utils.plotting import add_parameter_footer, get_repo_root

repo_root = get_repo_root()
data_dir = repo_root / "data/A2/ex_b"
save_dir = repo_root / "figures/A2/ex_b"
save_dir.mkdir(parents=True, exist_ok=True)

convergence_df = pd.read_parquet(data_dir / "convergence.parquet")
print(f"Loaded convergence data: {convergence_df.shape}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(
    data=convergence_df,
    x="Nr",
    y="Linf_err",
    marker="o",
    ax=ax,
    label="Spectral Collocation",
)

# Add reference line showing convergence rate
Nr_vals = convergence_df["Nr"].values
err_vals = convergence_df["Linf_err"].values

# Add reference line
slope = -2
Nr_ref = np.array([Nr_vals[0], Nr_vals[-1]])
err_ref = err_vals[0] * (Nr_ref / Nr_vals[0]) ** slope

ax.plot(
    Nr_ref, err_ref, "k--", alpha=0.5, linewidth=1.5, label=r"$\mathcal{O}(N_r^{-2})$"
)

ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"Number of radial nodes $N_r$",
    ylabel=r"$L^\infty$ Error",
)
ax.set_title("Convergence Study: Polar BVP")
ax.legend()
ax.grid(True, alpha=0.3)

# Add parameter footer
Nr_min = convergence_df["Nr"].min()
Nr_max = convergence_df["Nr"].max()
add_parameter_footer(fig, rf"$N_r \in [{Nr_min}, {Nr_max}]$")

fig.savefig(save_dir / "convergence.pdf", bbox_inches="tight")
print(f"Saved: {save_dir}/convergence.pdf")

# %%
# Load and prepare solution
# -------------------------
# Load the 2D polar solution data and prepare for visualization.

solution_df = pd.read_parquet(data_dir / "solution.parquet")
print(f"Loaded solution data: {solution_df.shape}")

# Get metadata
r1 = solution_df["r1"].iloc[0]
r2 = solution_df["r2"].iloc[0]
Nr = solution_df["Nr"].iloc[0]

# Get unique r and phi values to determine grid shape
r_unique = np.sort(solution_df["r"].unique())
phi_unique = np.sort(solution_df["phi"].unique())
n_phi = len(phi_unique)
n_r = len(r_unique)

print("\nGrid info:")
print(f"  Radial points: {n_r}")
print(f"  Angular points: {n_phi}")
print(f"  Total grid points: {len(solution_df)}")

# Wrap theta for continuous contours
phi_min = phi_unique.min()
wrap_df = solution_df[np.isclose(solution_df["phi"], phi_min)].copy()
wrap_df["phi"] = 2 * np.pi
grid_df = pd.concat([solution_df, wrap_df], ignore_index=True)

Theta_ext = grid_df.pivot(index="phi", columns="r", values="phi").values
Rs_ext = grid_df.pivot(index="phi", columns="r", values="r").values
Phi_hat_ext = grid_df.pivot(index="phi", columns="r", values="u").values
Phi_ext = grid_df.pivot(index="phi", columns="r", values="u_exact").values

# %%
# Visualize numerical solution
# -----------------------------
# Plot the solution on the polar domain.
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"}, layout="constrained")
con = ax.contourf(Theta_ext, Rs_ext, Phi_hat_ext, 100)
cbar = fig.colorbar(con, ax=ax)
cbar.set_label(r"$\Phi$")
ax.set_ylim(0, r2)
ax.set_title("Numerical Solution (Spectral Collocation)")

# Add parameter footer
add_parameter_footer(fig, rf"$N_r = {Nr}$, $r \in [{r1}, {r2}]$ ({n_r}×{n_phi} grid)")

fig.savefig(save_dir / "solution.pdf", bbox_inches="tight")
print(f"Saved: {save_dir}/solution.pdf")

# %%
# Visualize error
# ---------------
# Show the difference between numerical and exact solutions.
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"}, layout="constrained")
error = Phi_hat_ext - Phi_ext
con = ax.contourf(Theta_ext, Rs_ext, error, 100, cmap="RdBu_r")
cbar = fig.colorbar(con, ax=ax)
cbar.set_label(r"$\Phi_{\rm num} - \Phi_{\rm exact}$")
ax.set_ylim(0, r2)
ax.set_title("Error")

# Add parameter footer
add_parameter_footer(fig, rf"$N_r = {Nr}$, $r \in [{r1}, {r2}]$ ({n_r}×{n_phi} grid)")

fig.savefig(save_dir / "error.pdf", bbox_inches="tight")
print(f"Saved: {save_dir}/error.pdf")

print(f"\nAll plots saved to {save_dir}")
