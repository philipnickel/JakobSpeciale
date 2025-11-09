"""
Aliasing Diagnostics for Fourier KdV Solver
============================================

Runs a handful of coarse KdV simulations (aliased vs dealiased), samples the
FFT spectra at fixed times, and stores the results as tidy Parquet tables for
plotting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, get_time_integrator, two_soliton_initial

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_DIR = Path("data/A2/ex_e")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SPECTRA = DATA_DIR / "spectra.parquet"
OUTPUT_CONSERVATION = DATA_DIR / "conservation.parquet"

L = 20.0  # half domain length (x ∈ [-L, L)) - smaller domain
T_FINAL = 25.0  # longer time to show aliasing error accumulation
SAVE_EVERY = 60  # store every N solver steps
SAFETY = 0.3  # CFL safety factor
INTEGRATOR_NAME = "rk3"

# Resolution / dealias combinations to explore
# Use moderate resolution where aliasing becomes significant
N_VALUES = [70, 100]
DEALIAS_OPTIONS = [False, True]

# Two-soliton collision seed: produces spectral activity near the Nyquist limit
# Very strong amplitudes to force significant high-frequency content
C1, X01 = 3.0, -5.0
C2, X02 = 1.5, 5.0


# --------------------------------------------------------------------------- #
# Helper routines
# --------------------------------------------------------------------------- #


def stable_dt(N: int, L: float, u_max: float, *, dealias: bool) -> float:
    """Return a stable timestep (with safety margin) for each configuration."""
    dt_est = KdVSolver.stable_dt(
        N, L, u_max, integrator_name=INTEGRATOR_NAME, dealiased=dealias
    )
    if not np.isfinite(dt_est):
        return 1e-3
    return SAFETY * float(dt_est)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

spectra_frames: list[pd.DataFrame] = []
conservation_frames: list[pd.DataFrame] = []

for N in N_VALUES:
    # Compute dt once using aliased formula (more conservative)
    x = np.linspace(-L, L, N, endpoint=False)
    u0 = two_soliton_initial(x, C1, X01, C2, X02)
    dt = stable_dt(N, L, float(np.max(np.abs(u0))), dealias=False)

    for use_dealias in DEALIAS_OPTIONS:
        label = f"N{N}_{'dealias' if use_dealias else 'alias'}"
        treatment = "dealiased (3/2-rule)" if use_dealias else "aliased"
        print(
            f"\nRunning {label:>12s} | N={N:3d}, dealias={use_dealias}, dt={dt:.6e}",
            flush=True,
        )

        solver = KdVSolver(N, L, dealias=use_dealias)
        integrator = get_time_integrator(INTEGRATOR_NAME)

        t_saved, u_saved = solver.solve(
            u0.copy(),
            T_FINAL,
            dt,
            save_every=SAVE_EVERY,
            integrator=integrator,
        )

        # Compute conserved quantities: Mass, Momentum, Energy
        dx = solver.dx
        mass = np.sum(u_saved, axis=1) * dx  # M = ∫ u dx
        momentum = np.sum(u_saved**2, axis=1) * dx  # V = ∫ u² dx

        # Energy: E = ∫ (u³ - u_x²) dx
        # Compute u_x in physical space for each time
        energy = np.zeros(len(t_saved))
        for i, u in enumerate(u_saved):
            u_hat = np.fft.fft(u)
            ux_hat = solver.ik * u_hat
            ux = np.fft.ifft(ux_hat).real
            energy[i] = np.sum(u**3 - ux**2) * dx

        # Store conservation data
        conservation_df = pd.DataFrame(
            {
                "t": t_saved,
                "mass": mass,
                "momentum": momentum,
                "energy": energy,
                "scenario": label,
                "N": N,
                "dealias": use_dealias,
                "Treatment": treatment,
            }
        )
        conservation_frames.append(conservation_df)

        idx = np.arange(N // 2 + 1, dtype=int)
        k_vals = pd.Series(solver.k[idx], index=idx)

        fft_vals = np.fft.fft(u_saved, axis=1)[:, idx]
        magnitudes = np.abs(fft_vals)

        df = pd.DataFrame(magnitudes, columns=idx)
        df["t"] = t_saved
        tidy = df.melt(
            id_vars="t", var_name="mode_index", value_name="abs_u_hat"
        ).assign(
            scenario=label,
            N=N,
            dealias=use_dealias,
            Treatment=treatment,
            dt=dt,
            L=L,
        )
        tidy["mode_index"] = tidy["mode_index"].astype(int)
        tidy["k"] = tidy["mode_index"].map(k_vals)
        tidy["k_abs"] = tidy["k"].abs()

        spectra_frames.append(tidy)

spectra_all = pd.concat(spectra_frames, ignore_index=True)
conservation_all = pd.concat(conservation_frames, ignore_index=True)

spectra_all.to_parquet(OUTPUT_SPECTRA, index=False)
conservation_all.to_parquet(OUTPUT_CONSERVATION, index=False)

print(f"\nSaved spectra       → {OUTPUT_SPECTRA}")
print(f"Saved conservation  → {OUTPUT_CONSERVATION}")
