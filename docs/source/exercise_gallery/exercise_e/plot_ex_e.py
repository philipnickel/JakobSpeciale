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

from spectral.utils.plotting import add_parameter_footer, get_repo_root

# Paths

repo_root = get_repo_root()
DATA_DIR = repo_root / "data/A2/ex_e"
RESULTS_DIR = repo_root / "figures/A2/ex_e"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPECTRA_PATH = DATA_DIR / "spectra.parquet"

# Load data

print("Loading spectra for exercise e)...")

spectra = pd.read_parquet(SPECTRA_PATH)

if spectra.empty:
    raise RuntimeError("Spectra dataframe is empty – run ex_e.py first.")

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
spectra["dealias_label"] = np.where(
    spectra["dealias"], "dealiased (3/2-rule)", "aliased"
)
spectra["N_label"] = "N=" + spectra["N"].astype(str)

# Helper functions


def _select_snapshot_times(df: pd.DataFrame, count: int = 4) -> np.ndarray:
    """Pick `count` evenly spaced snapshot times present in the dataframe."""
    unique_times = np.sort(df["t"].unique())
    if unique_times.size <= count:
        return unique_times
    indices = np.linspace(0, unique_times.size - 1, count, dtype=int)
    return unique_times[indices]


def plot_high_band_power(df: pd.DataFrame) -> None:
    """Line plot of upper-band energy vs time for each resolution."""
    band_start = 2.0 / 3.0  # focus on the dealiasing cutoff (2/3 rule)
    with_power = df.assign(band_power=df["abs_u_hat"] ** 2)
    high_band = with_power[with_power["k_norm_raw"] >= band_start]

    if high_band.empty:
        return

    summary = (
        high_band.groupby(["N_label", "dealias_label", "t"], sort=False, observed=True)[
            "band_power"
        ]
        .sum()
        .reset_index()
    )

    g = sns.relplot(
        data=summary,
        x="t",
        y="band_power",
        hue="dealias_label",
        col="N_label",
        kind="line",
        height=3.5,
        aspect=1.3,
        facet_kws={"sharex": True, "sharey": True},
        markers=True,
    )

    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"Energy above $2/3 \, k_{\max}$")

    g._legend.set_title("spectral treatment")
    g.fig.suptitle("High-wavenumber energy (aliasing diagnostic)", y=1.02)

    # Add parameter footer
    N_unique = sorted(df["N"].unique())
    L_val = df["L"].iloc[0] if "L" in df.columns else None
    if L_val and len(N_unique) > 1:
        add_parameter_footer(
            g.fig, rf"$N \in [{N_unique[0]}, {N_unique[-1]}]$, $L = {L_val:.1f}$"
        )
    elif L_val:
        add_parameter_footer(g.fig, rf"$N = {N_unique[0]}$, $L = {L_val:.1f}$")

    g.fig.savefig(RESULTS_DIR / "ex_e_high_band_power.pdf", bbox_inches="tight")


def plot_spectrum_scatter(df: pd.DataFrame) -> None:
    """Scatter |û_k| vs k/k_max for representative times in each run."""
    records: list[pd.DataFrame] = []
    for scen, sub in df.groupby("scenario", sort=False):
        times = _select_snapshot_times(sub, count=4)
        subset = sub[sub["t"].isin(times)].copy()
        subset["time_label"] = subset["t"].map(lambda t_val: f"t = {t_val:.2f}")
        records.append(subset)

    if not records:
        return

    plot_df = pd.concat(records, ignore_index=True)

    g = sns.relplot(
        data=plot_df,
        x="k_norm",
        y="abs_u_hat_clipped",
        hue="time_label",
        col="scenario_pretty",
        kind="scatter",
        height=3.5,
        aspect=1.3,
        facet_kws={"sharex": True, "sharey": True},
    )

    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_xlabel(r"$|k| / k_{\max}$")
        ax.set_ylabel(r"$|\hat{u}_k|$")

    g._legend.set_title("snapshot time")
    g.fig.suptitle("Modal amplitudes at representative times", y=1.02)

    # Add parameter footer
    N_unique_s = sorted(df["N"].unique())
    L_val_s = df["L"].iloc[0] if "L" in df.columns else None
    if L_val_s and len(N_unique_s) > 1:
        add_parameter_footer(
            g.fig, rf"$N \in [{N_unique_s[0]}, {N_unique_s[-1]}]$, $L = {L_val_s:.1f}$"
        )
    elif L_val_s:
        add_parameter_footer(g.fig, rf"$N = {N_unique_s[0]}$, $L = {L_val_s:.1f}$")

    g.fig.savefig(RESULTS_DIR / "ex_e_spectrum_snapshots.pdf", bbox_inches="tight")


# Generate figures

plot_high_band_power(spectra)
plot_spectrum_scatter(spectra)

print(f"Figures saved in {RESULTS_DIR}")
