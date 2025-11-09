"""
Conserved Quantities for Two-Soliton Collision
===============================================

Plots conserved quantities (mass, momentum, energy) for the two-soliton
collision in the KdV equation.
"""


# %%
# Conservation laws
# Visualize conservation of mass, momentum, and energy.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from spectral.tdp import KdVSolver
from spectral.utils.plotting import add_parameter_footer, get_repo_root
from spectral.utils.io import load_simulation_data, ensure_output_dir
from spectral.utils.formatting import extract_metadata, format_dt_latex

repo_root = get_repo_root()
data_dir = repo_root / "data/A2/ex_f"
save_dir = ensure_output_dir(repo_root / "figures/A2/ex_f")

print("=" * 60)
print("Exercise f – two-soliton invariants")
print("=" * 60)

# %% Load data ---------------------------------------------------------------
df = load_simulation_data(data_dir, "kdv_two_soliton")
print(f"Data shape: {df.shape}")

# Metadata
metadata = extract_metadata(
    df, ["dx", "dt", "N", "L", "save_every", "c1", "x01", "c2", "x02"]
)
dx = float(metadata.get("dx", df["x"].diff().abs().dropna().iloc[0]))

print("Metadata:")
for key, val in metadata.items():
    print(f"  {key} = {val}")

# %% Reshape grid ------------------------------------------------------------
x_vals = np.sort(df["x"].unique())
t_vals = np.sort(df["t"].unique())

U = (
    df.pivot(index="x", columns="t", values="u")
    .reindex(index=x_vals, columns=t_vals)
    .to_numpy()
)

print(f"Unique x count: {len(x_vals)}, unique t count: {len(t_vals)}")

# %% Compute conserved quantities -------------------------------------------
# Sort by time and space to ensure correct spatial ordering within each time slice
df_sorted = df.sort_values(["t", "x"])
grouped = df_sorted.groupby("t", sort=False)["u"]
quantities = grouped.apply(
    lambda u: KdVSolver.compute_conserved_quantities(u.to_numpy(), dx)
)
mass, momentum, energy = np.vstack(quantities.to_list()).T

df_abs = pd.DataFrame(
    {"t": t_vals, "Mass": mass, "Momentum": momentum, "Energy": energy}
)

M0, V0, E0 = df_abs.loc[0, ["Mass", "Momentum", "Energy"]]
denom = pd.Series(
    {"Mass": abs(M0) or 1.0, "Momentum": abs(V0) or 1.0, "Energy": abs(E0) or 1.0}
)
df_rel = df_abs.copy()
for col in ["Mass", "Momentum", "Energy"]:
    df_rel[col] = (df_rel[col] - df_rel[col].iloc[0]) / denom[col]

df_abs_long = df_abs.melt("t", var_name="Quantity", value_name="Value")
df_rel_long = df_rel.melt("t", var_name="Quantity", value_name="Relative drift")

# %% Plot --------------------------------------------------------------------
fig, axs = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"hspace": 0.15}
)

sns.lineplot(data=df_abs_long, x="t", y="Value", hue="Quantity", ax=axs[0])
axs[0].set_ylabel("Quantity value")
axs[0].set_title("Conserved quantities over time")
axs[0].legend(loc="best")

sns.lineplot(data=df_rel_long, x="t", y="Relative drift", hue="Quantity", ax=axs[1])
axs[1].set_xlabel(r"Time $t$")
axs[1].set_ylabel("Relative drift")
axs[1].set_title("Relative drift from initial value")
axs[1].legend(loc="best")

# Add parameter footer
N = metadata.get("N", "?")
L = metadata.get("L", "?")
dt = metadata.get("dt", "?")
c1 = metadata.get("c1", "?")
c2 = metadata.get("c2", "?")
dt_latex = format_dt_latex(dt)
add_parameter_footer(
    fig, rf"$N = {N}$, $L = {L}$, $\Delta t = {dt_latex}$, $c_1 = {c1}$, $c_2 = {c2}$"
)

output_path = save_dir / "invariants.pdf"
fig.savefig(output_path, bbox_inches="tight")
print(f"Saved invariants plot → {output_path}")

print("=" * 60)
print("Invariant plotting complete.")
print("=" * 60)
