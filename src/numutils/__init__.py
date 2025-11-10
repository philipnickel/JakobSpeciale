"""
Numerical Utilities Package
============================

A simple package demonstrating basic numerical operations for University projects.

This package provides:
- Basic linear algebra operations
- Numerical integration methods
- Plotting utilities
"""

from numutils.integration import integrate_trapz, integrate_simpson
from numutils.linalg import norm, solve_linear

__all__ = [
    "integrate_trapz",
    "integrate_simpson",
    "norm",
    "solve_linear",
]

__version__ = "0.1.0"
