"""
Scalability Analysis Results
=============================

Creates plot showing computational complexity:

- Wall time vs N (comparing all methods: RK4, RK3)
- Reference line showing expected O(N log N) scaling
"""

# %%
# Scalability analysis
# Study how runtime scales with problem size.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from spectral.utils.plotting import add_parameter_footer, get_repo_root

repo_root = get_repo_root()
data_dir = repo_root / "data/A2/ex_g"
save_dir = repo_root / "figures/A2/ex_g"
save_dir.mkdir(parents=True, exist_ok=True)

# %% Load data
print("Loading scalability data...")
df_timing = pd.read_parquet(data_dir / "scalability_timing.parquet")

print(f"  Timing data: {df_timing.shape}")

# %% Create two-panel plot
print("\nCreating scalability analysis plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ===== Panel 1: Absolute performance (log-log) =====
# Plot data for each method
sns.lineplot(
    data=df_timing,
    x="N",
    y="time_per_step",
    hue="method",
    style="method",
    markers=True,
    markersize=8,
    ax=ax1,
)

# Add reference line: N log N scaling
N_ref = df_timing["N"].unique()
N_ref = np.sort(N_ref)
# Normalize to match first data point of RK3 (fastest method)
first_point = df_timing[(df_timing["N"] == N_ref[0]) & (df_timing["method"] == "RK3")][
    "time_per_step"
].values[0]
n_log_n = N_ref * np.log(N_ref)
n_log_n_scaled = first_point * (n_log_n / n_log_n[0])

ax1.plot(
    N_ref,
    n_log_n_scaled,
    "--",
    linewidth=2,
    alpha=0.7,
    color="gray",
    label=r"$\mathcal{O}(N \log N)$",
)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"Number of grid points $N$")
ax1.set_ylabel("Time per timestep [s]")
ax1.set_title("Computational Complexity")
ax1.legend(title="Method", loc="best")
ax1.grid(True, alpha=0.3)

# ===== Panel 2: Normalized efficiency =====
# Compute normalized time for each method
for method in df_timing["method"].unique():
    subset = df_timing[df_timing["method"] == method].sort_values("N")
    N_vals = subset["N"].values
    time_vals = subset["time_per_step"].values

    # Normalize by N log N
    normalized = time_vals / (N_vals * np.log(N_vals))

    # Plot
    ax2.plot(
        N_vals,
        normalized,
        marker="o",
        markersize=8,
        linewidth=2,
        label=method,
        alpha=0.8,
    )

ax2.set_xlabel(r"Number of grid points $N$")
ax2.set_ylabel(r"Time / $(N \log N)$ [s]")
ax2.set_title("Scaling Efficiency (should be flat)")
ax2.set_xscale("log")
ax2.legend(title="Method")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

# Overall title
fig.suptitle("KdV Solver Scalability Analysis", fontsize=14, y=1.02)

# Add parameter footer
L_val = df_timing["L"].iloc[0] if "L" in df_timing.columns else None
T_val = df_timing["T"].iloc[0] if "T" in df_timing.columns else None
if L_val and T_val:
    add_parameter_footer(fig, rf"$L = {L_val:.1f}$, $T = {T_val:.1f}$")

output = save_dir / "scalability_analysis.pdf"
fig.savefig(output, bbox_inches="tight")
print(f"  Saved: {output}")

# %% Summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

print("\nTime per step at N=128:")
for method in df_timing["method"].unique():
    t = df_timing[(df_timing["method"] == method) & (df_timing["N"] == 128)][
        "time_per_step"
    ].values
    if len(t) > 0:
        print(f"  {method}: {t[0]:.6f} s")

print("\nScaling exponent (fit to N^α in log-log space):")
for method in df_timing["method"].unique():
    subset = df_timing[df_timing["method"] == method]
    log_N = np.log(subset["N"].values)
    log_t = np.log(subset["time_per_step"].values)
    # Linear fit in log-log space
    coef = np.polyfit(log_N, log_t, 1)
    print(f"  {method}: α = {coef[0]:.3f} (ideal: ~1.0-1.1 for N log N)")

print("\nNote: Low scaling exponents are expected for small N ranges.")
print("The O(N log N) behavior from FFT becomes dominant at larger N.")

print("\nPlot created!")
