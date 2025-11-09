"""
Two-Soliton KdV Collision Dataset Generation
=============================================

Generates a two-soliton KdV collision dataset for analysis and visualization.
"""

# %% Imports and setup -------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, RK4, two_soliton_initial

# %% Configuration -----------------------------------------------------------
DATA_DIR = Path("data/A2/ex_f")
DATA_DIR.mkdir(parents=True, exist_ok=True)

L = 80.0  # half-domain [-L, L]
N = 160  # grid points (high resolution)
t_final = 120.0
save_every = 200  # save every N steps to control file size
dt_scale = 0.2  # safety factor on stability bound
c1, x01 = 0.5, -40.0
c2, x02 = 0.25, -15.0
output_path = DATA_DIR / "kdv_two_soliton.parquet"
DEALIAS_OPTIONS = (False, True)

print("=" * 60)
print("Exercise f – KdV two-soliton collision")
print("=" * 60)
print(f"Domain: x ∈ [{-L}, {L}], N = {N}")
print(f"Initial solitons: (x0, c) = ({x01}, {c1}) and ({x02}, {c2})")
print(f"Target time interval: t ∈ [0, {t_final}]")

all_frames: list[pd.DataFrame] = []

for dealias in DEALIAS_OPTIONS:
    treatment = "De-aliased (3/2-rule)" if dealias else "Aliased"
    print("-" * 60)
    print(f"Configuration: {treatment} (dealias={dealias})")

    solver = KdVSolver(N, L, dealias=dealias)
    x = solver.x
    dx = solver.dx

    u0 = two_soliton_initial(x, c1, x01, c2, x02)

    # Estimate stable timestep from initial amplitude
    u_max = float(np.max(np.abs(u0)))
    dt_est = KdVSolver.stable_dt(
        N,
        L,
        u_max,
        integrator_name="rk4",
        dealiased=dealias,
    )

    if not np.isfinite(dt_est) or dt_est <= 0.0:
        dt_est = 0.05  # fallback for unexpected values

    dt = dt_scale * dt_est
    integrator = RK4()

    print(f"dx = {dx:.3f}, estimated stable dt = {dt_est:.4e}")
    print(f"Using dt = {dt:.4e}, save_every = {save_every}")

    # Number of time steps and expected saves help gauge workload
    n_steps = int(np.ceil(t_final / dt))
    n_saves = int(np.ceil(n_steps / save_every)) + 1  # include initial snapshot
    print(f"≈ {n_steps} total steps → storing ≈ {n_saves} snapshots")

    # Time integration
    t_saved, u_saved = solver.solve(
        u0.copy(),
        t_final,
        dt,
        save_every=save_every,
        integrator=integrator,
    )

    print(f"Collected {len(t_saved)} snapshots for {treatment}")

    # Build tidy dataframe for this configuration
    records = []
    for t, u in zip(t_saved, u_saved):
        df_t = pd.DataFrame(
            {
                "x": x.astype(np.float64),
                "u": u.astype(np.float64),
            }
        )
        df_t["t"] = float(t)
        records.append(df_t)

    df_config = pd.concat(records, ignore_index=True)
    df_config["dx"] = dx
    df_config["dt"] = dt
    df_config["N"] = N
    df_config["L"] = L
    df_config["save_every"] = save_every
    df_config["c1"] = c1
    df_config["x01"] = x01
    df_config["c2"] = c2
    df_config["x02"] = x02
    df_config["dealias"] = dealias
    df_config["Treatment"] = treatment

    all_frames.append(df_config)

# Combine and persist -------------------------------------------------------
df = pd.concat(all_frames, ignore_index=True)
print(f"Resulting dataframe shape: {df.shape}")

df.to_parquet(output_path, index=False)
print(f"Saved dataset → {output_path}")

print("=" * 60)
print("Done.")
print("=" * 60)
