"""
Aliasing Diagnostics for KdV Solver
====================================

Visualizes aliasing diagnostics comparing aliased vs dealiased FFT spectra for
the Fourier KdV solver.
"""


# %%
# Aliasing diagnostics
# Compare aliased vs dealiased FFT spectra.

from __future__ import annotations


import numpy as np
import pandas as pd
import seaborn as sns

from spectral.utils.plotting import get_repo_root

# Paths

repo_root = get_repo_root()
DATA_DIR = repo_root / "data/A2/ex_e"
RESULTS_DIR = repo_root / "figures/A2/ex_e"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPECTRA_PATH = DATA_DIR / "spectra.parquet"
CONSERVATION_PATH = DATA_DIR / "conservation.parquet"

# Load data

print("Loading spectra for exercise e)...")

spectra = pd.read_parquet(SPECTRA_PATH)
conservation = pd.read_parquet(CONSERVATION_PATH)


# Derived columns
spectra["k_norm_raw"] = spectra["k_abs"] / (np.pi * spectra["N"] / (2.0 * spectra["L"]))
spectra["k_norm"] = spectra["k_norm_raw"].clip(upper=1.0)
spectra["abs_u_hat_clipped"] = spectra["abs_u_hat"].clip(lower=1e-12)

scenario_names: dict[str, str] = {}
for _, row in spectra[["scenario", "N", "dealias"]].drop_duplicates().iterrows():
    scenario_names[row["scenario"]] = (
        f"N={int(row['N'])} ({'dealiased' if row['dealias'] else 'aliased'})"
    )

spectra["scenario_pretty"] = spectra["scenario"].map(scenario_names)


band_start = 2.0 / 3.0  # focus on the dealiasing cutoff (2/3 rule)
with_power = spectra.assign(band_power=spectra["abs_u_hat"] ** 2)
high_band = with_power[with_power["k_norm_raw"] >= band_start]


summary = (
    high_band.groupby(["N", "Treatment", "t"], sort=False, observed=True)["band_power"]
    .sum()
    .reset_index()
)

g = sns.relplot(
    data=summary,
    x="t",
    y="band_power",
    hue="Treatment",
    col="N",
    kind="line",
    facet_kws={"sharex": True, "sharey": True},
    markers=True,
)

for ax in g.axes.flat:
    ax.set_yscale("log")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Energy above $2/3 \, k_{\max}$")

# Add parameter information to title
N_unique = sorted(spectra["N"].unique())
L_val = spectra["L"].iloc[0]
T_val = spectra["t"].max()

if len(N_unique) > 1:
    param_text = rf"\tiny $N \in [{N_unique[0]}, {N_unique[-1]}]$, $L = {L_val:.1f}$, $T = {T_val:.1f}$"
else:
    param_text = rf"\tiny $N = {N_unique[0]}$, $L = {L_val:.1f}$, $T = {T_val:.1f}$"

g.fig.suptitle(
    "KdV Aliasing Diagnostic: High-Band Energy" + "\n" + param_text,
    y=1.02,
)

g.fig.savefig(RESULTS_DIR / "ex_e_high_band_power.pdf", bbox_inches="tight")


# Get final time for each N separately (they have different saved times)
# Each N has a different timestep so they save at different exact times
max_t_per_N = (
    spectra.groupby("N", as_index=False)["t"].max().rename(columns={"t": "t_max"})
)
plot_df = spectra.merge(max_t_per_N, on="N")
plot_df = plot_df[plot_df["t"] == plot_df["t_max"]].drop(columns=["t_max"])


g = sns.relplot(
    data=plot_df,
    x="k_abs",
    y="abs_u_hat_clipped",
    hue="Treatment",
    col="N",
    kind="scatter",
    alpha=0.6,
)

for ax in g.axes.flat:
    ax.set_yscale("log")
    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$|\hat{u}_k|$")

# Add parameter information to title
N_unique_s = sorted(spectra["N"].unique())
L_val_s = spectra["L"].iloc[0]
t_final = plot_df["t"].max()

if len(N_unique_s) > 1:
    param_text_s = rf"\tiny $N \in [{N_unique_s[0]}, {N_unique_s[-1]}]$, $L = {L_val_s:.1f}$, $t = {t_final:.1f}$"
else:
    param_text_s = (
        rf"\tiny $N = {N_unique_s[0]}$, $L = {L_val_s:.1f}$, $t = {t_final:.1f}$"
    )

g.fig.suptitle(
    "KdV Modal Amplitudes at Final Time" + "\n" + param_text_s,
    y=1.02,
)

g.fig.savefig(RESULTS_DIR / "ex_e_final_spectra.pdf", bbox_inches="tight")


# Conservation plot
# Normalize by initial values to show relative error
conservation_rel = conservation.copy()
for quantity in ["mass", "momentum", "energy"]:
    for scenario in conservation["scenario"].unique():
        mask = conservation_rel["scenario"] == scenario
        initial_val = conservation_rel.loc[mask, quantity].iloc[0]
        conservation_rel.loc[mask, f"{quantity}_rel_error"] = (
            conservation_rel.loc[mask, quantity] - initial_val
        ) / abs(initial_val)

# Melt for facet plot
cons_tidy = conservation_rel.melt(
    id_vars=["t", "scenario", "N", "Treatment"],
    value_vars=["mass_rel_error", "momentum_rel_error", "energy_rel_error"],
    var_name="quantity",
    value_name="relative_error",
)

# Clean up quantity names
cons_tidy["Quantity"] = cons_tidy["quantity"].map(
    {
        "mass_rel_error": "Mass",
        "momentum_rel_error": "Momentum",
        "energy_rel_error": "Energy",
    }
)

g_cons = sns.relplot(
    data=cons_tidy,
    x="t",
    y="relative_error",
    hue="Treatment",
    col="Quantity",
    row="N",
    kind="line",
    facet_kws={"sharex": True, "sharey": False},
    markers=True,
    alpha=0.8,
)

for ax in g_cons.axes.flat:
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Relative Error $(Q(t) - Q_0)/|Q_0|$")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

# Add parameter information to title
N_unique_cons = sorted(conservation["N"].unique())
L_val_cons = L_val  # Use L from spectra data
T_val_cons = conservation["t"].max()

if len(N_unique_cons) > 1:
    param_text_cons = rf"\tiny $N \in [{N_unique_cons[0]}, {N_unique_cons[-1]}]$, $L = {L_val_cons:.1f}$, $T = {T_val_cons:.1f}$"
else:
    param_text_cons = rf"\tiny $N = {N_unique_cons[0]}$, $L = {L_val_cons:.1f}$, $T = {T_val_cons:.1f}$"

g_cons.fig.suptitle(
    "KdV Conservation Errors" + "\n" + param_text_cons,
    y=1.05,
)

g_cons.fig.savefig(RESULTS_DIR / "ex_e_conservation.pdf")


print(f"Figures saved in {RESULTS_DIR}")
