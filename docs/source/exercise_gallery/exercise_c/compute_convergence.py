"""
Spatial and Temporal Convergence for Fourier KdV Solver
========================================================

The script generates two Parquet tables:

* ``kdv_spatial_convergence.parquet`` – error vs. number of modes (N) for
  aliased/dealiased runs.
* ``kdv_temporal_convergence.parquet`` – error vs. timestep (dt) for the time
  integrators used in the assignment.
"""

# TODO: Clean up
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from spectral.tdp import KdVSolver, soliton, RK4, RK3


# //----------------------------------------------------------------------- #
# Configuration
# //----------------------------------------------------------------------- #

DATA_DIR = Path("data/A2/ex_c")
DATA_DIR.mkdir(parents=True, exist_ok=True)

L_SPATIAL = 50.0
L_TEMPORAL = 30.0
X0 = 0.0

INTEGRATOR_FACTORIES: dict[str, Callable[[], object]] = {
    "RK4": RK4,
    "RK3": RK3,
}
TEMPORAL_METHODS: tuple[str, ...] = ("RK4", "RK3")


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
    method_name: str,
    wave_speed: float,
    dealias: bool,
    half_length: float,
) -> tuple[np.ndarray, float, np.ndarray, float, int]:
    """Integrate soliton and return (grid, dx, final solution, t_end, steps)."""
    solver = KdVSolver(N, half_length, dealias=dealias)
    x = solver.x
    dx = solver.dx

    u0 = soliton(x, 0.0, wave_speed, X0)
    integrator = INTEGRATOR_FACTORIES[method_name]()
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


def _stability_limited_dt(
    N: int,
    wave_speed: float,
    *,
    method_name: str,
    dealias: bool,
    half_length: float,
) -> float:
    """Return a conservative stable timestep for (N, method, wave)."""
    solver = KdVSolver(N, half_length, dealias=dealias)
    u0 = soliton(solver.x, 0.0, wave_speed, X0)
    u_max = float(np.max(np.abs(u0)))
    dt_est = KdVSolver.stable_dt(
        N,
        half_length,
        u_max,
        integrator_name=method_name.lower(),
        dealiased=dealias,
    )
    if not np.isfinite(dt_est) or dt_est <= 0.0:
        return 1e-3
    return float(dt_est)


# //----------------------------------------------------------------------- #
# Spatial convergence: exponential drop with number of modes
# //----------------------------------------------------------------------- #

print("=" * 70)
print("KdV Soliton – Spatial Convergence")
print("=" * 70)

SPATIAL_WAVE_SPEED = 0.5
T_SPATIAL = 2.0e-2
DT_SPATIAL = 2.0e-6  # sufficiently small to suppress temporal error
N_VALUES_SPATIAL = [32, 64, 100, 150, 200, 250, 300]
DEALIAS_OPTIONS = [False, True]

print(f"Final time T = {T_SPATIAL:g}, dt = {DT_SPATIAL:.2e}")
print(f"N modes: {N_VALUES_SPATIAL}")
print(f"Spatial half-domain L = {L_SPATIAL}")

spatial_rows: list[dict[str, object]] = []

for dealias in DEALIAS_OPTIONS:
    dealias_label = "De-aliased" if dealias else "Aliased"
    print(f"\n--- {dealias_label} ---")

    for method_name in INTEGRATOR_FACTORIES:
        print(f"  Method: {method_name}")

        for N in N_VALUES_SPATIAL:
            current_half_length = L_SPATIAL
            x, dx, u_num, t_end, steps_taken = _solve_case(
                N,
                dt=DT_SPATIAL,
                T=T_SPATIAL,
                method_name=method_name,
                wave_speed=SPATIAL_WAVE_SPEED,
                dealias=dealias,
                half_length=current_half_length,
            )
            u_exact = soliton(x, t_end, SPATIAL_WAVE_SPEED, X0)
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

print(f"\nSaved spatial convergence data → {spatial_path} ({df_spatial.shape})")


# //----------------------------------------------------------------------- #
# Temporal convergence: dt error for explicit/implicit integrators
# //----------------------------------------------------------------------- #

print("\n" + "=" * 70)
print("KdV Soliton – Temporal Convergence")
print("=" * 70)

TEMPORAL_WAVE_SPEED = 2.0
N_TEMPORAL = 128
TEMPORAL_DEALIAS = True
DT_SCALES = np.array(
    [
        0.4,
        0.2,
        0.1,
        0.05,
        0.025,
        0.0125,
        0.00625,
        0.003125,
        0.0015625,
        0.00078125,
        0.000390625,
        0.0001953125,
        0.00009765625,
        0.000048828125,
        0.0000244140625,
        0.00001220703125,
        0.000006103515625,
        0.0000030517578125,
        0.00000152587890625,
        0.000000762939453125,
        0.0000003814697265625,
        0.00000019073486328125,
        0.000000095367431640625,
        0.0000000476837158203125,
        0.00000002384185791015625,
        0.000000011920928955078125,
        0.0000000059604644775390625,
        0.0000000029802322387695312,
        0.0000000014901161193847656,
        0.0000000007450580596923828,
        0.0000000003725290298461914,
        0.0000000001862645149230957,
        0.00000000009313225746154785,
        0.000000000046566128730773926,
        0.000000000023283064365386963,
    ]
)
MAX_STEPS_TEMPORAL = 40000  # skip runs that would be too expensive

print(f"N = {N_TEMPORAL}, dealias = {TEMPORAL_DEALIAS}")

temporal_rows: list[dict[str, object]] = []

for method_name in TEMPORAL_METHODS:
    dt_stable = _stability_limited_dt(
        N_TEMPORAL,
        TEMPORAL_WAVE_SPEED,
        method_name=method_name,
        dealias=TEMPORAL_DEALIAS,
        half_length=L_TEMPORAL,
    )

    dt_values = np.array(dt_stable * DT_SCALES, dtype=float)
    dt_values = dt_values[(dt_values > 0.0) & np.isfinite(dt_values)]

    if dt_values.size == 0:
        print(
            f"  {method_name}: unable to find stable dt values (stable_dt={dt_stable:.3e})"
        )
        continue

    print(f"\n  {method_name}: stable dt ≈ {dt_stable:.3e}")

    for dt in dt_values:
        target_T = float(dt)

        try:
            current_half_length = L_TEMPORAL
            x, dx, u_num, t_end, steps_taken = _solve_case(
                N_TEMPORAL,
                dt=float(dt),
                T=target_T,
                method_name=method_name,
                wave_speed=TEMPORAL_WAVE_SPEED,
                dealias=TEMPORAL_DEALIAS,
                half_length=current_half_length,
            )
            u_exact = soliton(x, target_T, TEMPORAL_WAVE_SPEED, X0)
            diff = u_num - u_exact
            l2 = float(np.sqrt(np.sum(diff**2) * dx))
            linf = float(np.max(np.abs(diff)))
        except Exception as exc:  # pragma: no cover - diagnostic output
            print(f"    dt={dt:.3e}: FAILED ({exc})")
            continue

        if not np.isfinite(l2) or not np.isfinite(linf):
            print(f"    dt={dt:.3e}: discarded (non-finite error)")
            continue

        temporal_rows.append(
            {
                "dt": float(dt),
                "N": N_TEMPORAL,
                "T": target_T,
                "t_end": target_T,
                "n_steps": steps_taken,
                "method": method_name,
                "dealias": "De-aliased" if TEMPORAL_DEALIAS else "Aliased",
                "L": current_half_length,
                "Error": l2,
            }
        )

        print(f"    dt={dt:.3e} (steps={steps_taken}): L2={l2:.3e}, L∞={linf:.3e}")

        if l2 < 1e-12:
            print("    Reached error floor, stopping sweep for this method.")
            break

df_temporal = pd.DataFrame(temporal_rows)
df_temporal["method"] = df_temporal["method"].astype("category")

temporal_path = DATA_DIR / "kdv_temporal_convergence.parquet"
df_temporal.to_parquet(temporal_path, index=False)

print(f"\nSaved temporal convergence data → {temporal_path} ({df_temporal.shape})")
print("\nConvergence studies completed.")
