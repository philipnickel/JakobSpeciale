"""
Work-Precision Diagram: Varying Grid Resolution
================================================

Shows L² error vs wall time for RK3 and RK4, varying grid size N.
For each N, dt is automatically selected based on CFL condition.
"""

# %%
# Work-precision plot
# Error vs computational cost for varying N

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns

from spectral.utils.plotting import get_repo_root
from spectral.utils.io import ensure_output_dir

repo_root = get_repo_root()
DATA_DIR = repo_root / "data/A2/ex_g"
OUTPUT_DIR = ensure_output_dir(repo_root / "figures/A2/ex_g")

print("Creating work-precision plot...")

# Load data
df = pd.read_parquet(DATA_DIR / "work_precision.parquet")

# Filter out NaN errors (unstable runs)
df = df[np.isfinite(df["error_l2"])].copy()

# Add N as a categorical variable for styling
df["N_cat"] = df["N"].astype(str)

# Create relplot with method in hue, N in style
g = sns.relplot(
    data=df,
    x="wall_time",
    y="error_l2",
    hue="method",
    style="method",
    kind="line",
    markers=True,
    errorbar="ci",
    facet_kws={"legend_out": False},
)

# Set log scales
for ax in g.axes.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both", linestyle="--")

    # Annotate points with N values
    # Get unique (method, N) combinations for annotation
    for method in df["method"].unique():
        method_data = (
            df[df["method"] == method]
            .groupby("N")
            .agg({"wall_time": "mean", "error_l2": "first"})
            .reset_index()
        )


g.set_axis_labels(r"Wall Time (s)", r"$L^2$ Error")

# Add parameter information to title
N_vals = sorted(df["N"].unique())
T_final = df["T_FINAL"].iloc[0]
L_val = df["L"].iloc[0] if "L" in df.columns else 40.0

param_text = rf"\tiny $N \in [{N_vals[0]}, {N_vals[-1]}]$, $T = {T_final:.2f}$, $L = {L_val:.1f}$"

g.fig.suptitle(
    "Work-Precision Diagram: RK3 vs RK4" + "\n" + param_text,
    y=1.02,
)

# Save
output_file = OUTPUT_DIR / "work_precision.pdf"
g.fig.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Saved: {output_file}")

# Print summary
print("\n" + "=" * 70)
print("Work-Precision Analysis Summary")
print("=" * 70)

print("=" * 70)
print("\n✓ Work-precision plot generated!")
