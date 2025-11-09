"""
Spatial and Temporal Convergence for Fourier KdV Solver
========================================================

The script generates two Parquet tables:

* ``kdv_spatial_convergence.parquet`` – error vs. number of modes (N) for
  aliased/dealiased runs.
* ``kdv_temporal_convergence.parquet`` – error vs. timestep (dt) for the time
  integrators used in the assignment.
"""

# TODO: have a look at L6 - slides 28 and 29
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK4, RK3


# //----------------------------------------------------------------------- #
# Configuration
# //----------------------------------------------------------------------- #

DATA_DIR = Path("data/A2/ex_c")
DATA_DIR.mkdir(parents=True, exist_ok=True)

L_SPATIAL = 40.0
L_TEMPORAL = 40.0  # Larger domain => larger dx => allows larger dt
X0 = 0.0

DEALIAS_OPTIONS = [False, True]
INTEGRATORS = [RK4, RK3]

WAVE_SPEED = 1.00

T_SPATIAL = 0.01  # 2.0e-2
DT_SPATIAL = 1.0e-6  # sufficiently small to suppress temporal error

N_VALUES_SPATIAL = 2 * np.logspace(np.log10(5), np.log10(175), num=20, dtype=int)
# N_VALUES_SPATIAL = 2 * np.logspace(np.log10(5), np.log10(20), num=5, dtype=int)
# Round and force evenness


N_TEMPORAL = 100  # Moderate-high spatial resolution for good accuracy
T_TEMPORAL = 1.00  # Long time to accumulate temporal error
U_MAX_TEMPORAL = 0.5 * WAVE_SPEED  # Peak amplitude of single soliton

# Temporal dt sampling: start from larger dt (aim for min dt ~ 0.01)
# Skip tiny dt values that just hit spatial error floor
DT_FRACTION_MIN = 0.01  # Start from 1% of dt_stable
DT_FRACTION_MAX = 0.90  # Go to 80% of stability limit
N_DT_SUBCRITICAL = 30  # Good sampling to see power law
N_DT_SUPERCRITICAL = 0  # Don't test supercritical (that's just blowup)


def _compute_temporal_dt_table() -> dict[tuple[bool, str], tuple[np.ndarray, float]]:
    """
    Build mapping {(dealias, method_name): (dt_values, dt_stable)} for time study.
    """
    table: dict[tuple[bool, str], tuple[np.ndarray, float]] = {}
    for dealias in DEALIAS_OPTIONS:
        for integrator_class in INTEGRATORS:
            method_name = integrator_class.__name__
            dt_stable = KdVSolver.stable_dt(
                N_TEMPORAL,
                L_TEMPORAL,
                U_MAX_TEMPORAL,
                integrator_name=method_name.lower(),
                dealiased=dealias,
            )
            if not (np.isfinite(dt_stable) and dt_stable > 0.0):
                raise RuntimeError(
                    f"Stable dt could not be estimated for {method_name} "
                    f"(dealias={dealias})."
                )

            # Sample dt from DT_FRACTION_MIN * dt_stable to DT_FRACTION_MAX * dt_stable
            if N_DT_SUPERCRITICAL > 0 and DT_FRACTION_MAX > 1.0:
                # Sample both sub and supercritical
                subcritical = np.geomspace(
                    DT_FRACTION_MIN, 1.0, num=N_DT_SUBCRITICAL, dtype=float
                )
                supercritical = np.geomspace(
                    1.0, DT_FRACTION_MAX, num=N_DT_SUPERCRITICAL, dtype=float
                )[1:]
                fractions = np.concatenate((subcritical, supercritical))
            else:
                # Only sample subcritical range
                fractions = np.geomspace(
                    DT_FRACTION_MIN,
                    min(DT_FRACTION_MAX, 1.0),
                    num=N_DT_SUBCRITICAL,
                    dtype=float,
                )
            table[(dealias, method_name)] = (dt_stable * fractions, float(dt_stable))

    return table


TEMPORAL_DT_TABLE = _compute_temporal_dt_table()
TOTAL_TEMPORAL_CASES = sum(len(info[0]) for info in TEMPORAL_DT_TABLE.values())


# //----------------------------------------------------------------------- #
# Helpers
# //----------------------------------------------------------------------- #


def _reset_multistep_state(integrator: object) -> None:
    """Clear history buffers for multi-step integrators."""
    if hasattr(integrator, "u_history"):
        integrator.u_history = []
    if hasattr(integrator, "f_history"):
        integrator.f_history = []


def _solve_case(
    N: int,
    *,
    dt: float,
    T: float,
    integrator_class: type,
    wave_speed: float,
    dealias: bool,
    half_length: float,
) -> tuple[np.ndarray, float, np.ndarray, float, int]:
    """Integrate soliton and return (grid, dx, final solution, t_end, steps)."""
    solver = KdVSolver(N, half_length, dealias=dealias)
    x = solver.x
    dx = solver.dx

    u0 = soliton(x, 0.0, wave_speed, X0)
    integrator = integrator_class()
    _reset_multistep_state(integrator)

    save_every = max(1, int(np.ceil(T / dt)))

    t_saved, u_hist = solver.solve(
        u0.copy(),
        T,
        dt,
        integrator=integrator,
        save_every=save_every,
    )

    if len(u_hist) == 0:
        raise RuntimeError("Solver returned no states.")

    u_final = u_hist[-1]
    t_end = float(t_saved[-1])
    steps_taken = len(u_hist) - 1
    return x, dx, u_final, t_end, steps_taken


# //----------------------------------------------------------------------- #
# Spatial convergence: exponential drop with number of modes
# //----------------------------------------------------------------------- #

spatial_rows: list[dict[str, object]] = []

for dealias in DEALIAS_OPTIONS:
    dealias_label = "De-aliased" if dealias else "Aliased"
    print(f"\n--- {dealias_label} ---")

    for integrator_class in INTEGRATORS:
        method_name = integrator_class.__name__
        print(f"  Method: {method_name}")

        for N in N_VALUES_SPATIAL:
            current_half_length = L_SPATIAL
            x, dx, u_num, t_end, steps_taken = _solve_case(
                N,
                dt=DT_SPATIAL,
                T=T_SPATIAL,
                integrator_class=integrator_class,
                wave_speed=WAVE_SPEED,
                dealias=dealias,
                half_length=current_half_length,
            )
            u_exact = soliton(x, t_end, WAVE_SPEED, X0)
            diff = u_num - u_exact
            l2 = float(np.sqrt(np.sum(diff**2) * dx))
            linf = float(np.max(np.abs(diff)))

            spatial_rows.append(
                {
                    "N": N,
                    "dt": DT_SPATIAL,
                    "T": T_SPATIAL,
                    "t_end": t_end,
                    "n_steps": steps_taken,
                    "method": method_name,
                    "dealias": dealias_label,
                    "L": current_half_length,
                    "Error": l2,
                }
            )

            print(f"    N={N:3d}: L2={l2:.3e}, L∞={linf:.3e}")

df_spatial = pd.DataFrame(spatial_rows)
df_spatial["method"] = df_spatial["method"].astype("category")

spatial_path = DATA_DIR / "kdv_spatial_convergence.parquet"
df_spatial.to_parquet(spatial_path, index=False)

print("\nSaved spatial convergence data")


# //----------------------------------------------------------------------- #
# Temporal convergence: dt error for explicit/implicit integrators
# //----------------------------------------------------------------------- #

temporal_rows: list[dict[str, object]] = []

print("\n--- Temporal Convergence ---")
print(
    "Testing "
    f"{len(DEALIAS_OPTIONS)} dealias options × {len(INTEGRATORS)} integrators "
    f"× variable timesteps = {TOTAL_TEMPORAL_CASES} cases\n"
)

for dealias in DEALIAS_OPTIONS:
    dealias_label = "De-aliased" if dealias else "Aliased"
    print(f"\n--- {dealias_label} ---")

    for integrator_idx, integrator_class in enumerate(INTEGRATORS, 1):
        method_name = integrator_class.__name__
        dt_values, dt_stable = TEMPORAL_DT_TABLE[(dealias, method_name)]
        print(
            f"[{integrator_idx}/{len(INTEGRATORS)}] Method: {method_name} "
            f"(dt_stable={dt_stable:.3e}, {len(dt_values)} samples)"
        )

        successful_runs = 0
        for dt_idx, dt in enumerate(dt_values, 1):
            try:
                current_half_length = L_TEMPORAL
                x, dx, u_num, t_end, steps_taken = _solve_case(
                    N_TEMPORAL,
                    dt=float(dt),
                    T=T_TEMPORAL,
                    integrator_class=integrator_class,
                    wave_speed=WAVE_SPEED,
                    dealias=dealias,
                    half_length=current_half_length,
                )
                u_exact = soliton(x, t_end, WAVE_SPEED, X0)
                diff = u_num - u_exact
                l2 = float(np.sqrt(np.sum(diff**2) * dx))
                linf = float(np.max(np.abs(diff)))

                # Skip if we got NaN or inf
                if not (np.isfinite(l2) and np.isfinite(linf)):
                    print(
                        f"  [{dt_idx:2d}/{len(dt_values)}] dt={dt:.3e}: SKIPPED (unstable)"
                    )
                    continue

            except Exception as exc:  # pragma: no cover - diagnostic output
                print(f"  [{dt_idx:2d}/{len(dt_values)}] dt={dt:.3e}: FAILED ({exc})")
                continue

            n_timesteps = int(np.round(T_TEMPORAL / dt))
            temporal_rows.append(
                {
                    "dt": float(dt),
                    "N": N_TEMPORAL,
                    "T": T_TEMPORAL,
                    "t_end": t_end,
                    "n_steps": n_timesteps,
                    "method": method_name,
                    "dealias": dealias_label,
                    "L": current_half_length,
                    "Error": l2,
                }
            )

            successful_runs += 1
            print(
                f"  [{dt_idx:2d}/{len(dt_values)}] dt={dt:.3e} "
                f"({n_timesteps:4d} steps): L2={l2:.6e}, L∞={linf:.6e}"
            )

        print(
            f"  → Completed {successful_runs}/{len(dt_values)} runs for {method_name}\n"
        )


df_temporal = pd.DataFrame(temporal_rows)
df_temporal["method"] = df_temporal["method"].astype("category")

temporal_path = DATA_DIR / "kdv_temporal_convergence.parquet"
df_temporal.to_parquet(temporal_path, index=False)

print("\nSaved temporal convergence data")
print("\nConvergence studies completed.")
