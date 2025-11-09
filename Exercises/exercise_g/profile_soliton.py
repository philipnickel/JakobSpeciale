"""
Profiling Script: Single Soliton KdV Solver
============================================

Simple script for memory and CPU profiling of KdV solver.
Usage: python profile_soliton.py <N>
"""

import argparse
import sys
import numpy as np
from spectral.tdp import KdVSolver, soliton


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Profile KdV solver with single soliton"
    )
    parser.add_argument("N", type=int, help="Number of grid points")
    parser.add_argument(
        "--T", type=float, default=0.1, help="Final time (default: 0.1)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="RK4",
        choices=["RK3", "RK4"],
        help="Time integrator (default: RK4)",
    )
    args = parser.parse_args()

    N = args.N
    T_FINAL = args.T

    # Domain and problem setup
    L = 40.0
    X0 = 0.0
    WAVE_SPEED = 1.0

    # Compute suitable dt based on CFL condition
    # For KdV: dt ≈ C * (dx)^3 / max(|u|)
    # For stability, use dt ≈ 0.001 * (L/N)^3
    dx = L / N
    dt = 0.001 * dx**3

    # Make sure we have reasonable number of steps
    n_steps = int(np.round(T_FINAL / dt))
    dt = T_FINAL / n_steps  # Adjust to hit T_FINAL exactly

    print("Profiling KdV Solver")
    print(f"N = {N}, dx = {dx:.6f}, dt = {dt:.6e}")
    print(f"T_FINAL = {T_FINAL}, n_steps = {n_steps}")
    print(f"Method: {args.method}")
    print("-" * 60)

    # Initialize solver
    solver = KdVSolver(N, L, dealias=True)
    x = solver.x

    # Initial condition (soliton at t=0)
    u0 = soliton(x, 0.0, WAVE_SPEED, X0)

    # Select integrator
    if args.method == "RK4":
        from spectral.tdp import RK4

        integrator = RK4()
    else:
        from spectral.tdp import RK3

        integrator = RK3()

    # Time integration
    u = u0.copy()
    t = 0.0

    for step in range(n_steps):
        u = integrator.step(solver.rhs, u, t, dt)
        t += dt

        # Print progress every 10%
        if (step + 1) % max(1, n_steps // 10) == 0:
            progress = 100 * (step + 1) / n_steps
            print(f"  Progress: {progress:.0f}% (t = {t:.4f})")

    # Compute final error
    u_exact = soliton(x, t, WAVE_SPEED, X0)
    diff = u - u_exact
    error_l2 = float(np.sqrt(np.sum(diff**2) * dx))
    error_linf = float(np.max(np.abs(diff)))

    print("-" * 60)
    print(f"Final L2 error:   {error_l2:.6e}")
    print(f"Final L∞ error:   {error_linf:.6e}")
    print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
