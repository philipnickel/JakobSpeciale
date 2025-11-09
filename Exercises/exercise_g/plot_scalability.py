"""
Scalability Analysis
====================

Analyzes computational scaling of RK3 and RK4 methods with grid size,
demonstrating O(N log N) complexity from FFT operations.
"""

# %%
# Scalability plot
# Time per step vs grid size N

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns

from spectral.utils.plotting import get_repo_root
from spectral.utils.io import ensure_output_dir

repo_root = get_repo_root()
DATA_DIR = repo_root / "data/A2/ex_g"
OUTPUT_DIR = ensure_output_dir(repo_root / "figures/A2/ex_g")

print("Creating scalability plot...")

# Load data
df = pd.read_parquet(DATA_DIR / "scalability_timing.parquet")


# Create relplot with method in hue and style
g = sns.relplot(
    data=df,
    x="N",
    y="time_per_step",
    hue="method",
    style="method",
    kind="line",
    markers=True,
    errorbar="ci",
    err_style="bars",
    facet_kws={"legend_out": False},
)

# Set log scales
for ax in g.axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Add O(N log N) reference line
    N_ref = np.array([df["N"].min(), df["N"].max()])
    time_ref_base = df["time_per_step"].max() * 0.5
    N_mid = df["N"].median()
    time_theory = time_ref_base * (N_ref * np.log(N_ref)) / (N_mid * np.log(N_mid))

    ax.plot(N_ref, time_theory, "k--", alpha=0.5, label=r"$\mathcal{O}(N \log N)$")

g.set_axis_labels(r"Grid Size $N$", r"Time per Step (s)")

# Add parameter information to title
N_vals = sorted(df["N"].unique())
n_trials = df["trial"].nunique()

param_text = (
    rf"\tiny $N \in [{N_vals[0]}, {N_vals[-1]}]$, {n_trials} trials per configuration"
)

g.fig.suptitle(
    "Scalability Analysis: RK3 vs RK4" + "\n" + param_text,
    y=1.05,
)

# Update legend to include reference line
# handles, labels = g.axes.flat[0].get_legend_handles_labels()
# g.axes.flat[0].legend(handles, labels, loc="upper left")

# Save
output_file = OUTPUT_DIR / "scalability_analysis.pdf"
g.fig.savefig(output_file)
print(f"Saved: {output_file}")

print("\n✓ Scalability plot generated!")
