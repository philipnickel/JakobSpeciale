.. _api_reference:

=============
API Reference
=============

This page provides an overview of the ``spectral`` package API.

.. currentmodule:: spectral

Spectral Basis Classes
======================

Classes for spectral methods on different bases.

.. autosummary::
   :toctree: generated
   :nosignatures:

   LegendreLobattoBasis
   FourierEquispacedBasis

Spectral Differentiation Matrices
==================================

Functions to construct differentiation and mass matrices.

.. autosummary::
   :toctree: generated
   :nosignatures:

   legendre_diff_matrix
   legendre_mass_matrix
   fourier_diff_matrix_cotangent
   fourier_diff_matrix_on_interval

Boundary Value Problems
========================

Solvers for boundary value problems (BVPs).

.. autosummary::
   :toctree: generated
   :nosignatures:

   BvpProblem
   solve_bvp
   solve_legendre_collocation
   solve_legendre_tau
   solve_polar_bvp

Time-Dependent PDEs
===================

Time integrators and PDE solvers.

Time Integrators
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   TimeIntegrator
   get_time_integrator
   RK3
   RK4

KdV Equation Solver
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   KdVSolver
   soliton
   two_soliton_initial

Utilities
=========

Helper functions for I/O, formatting, plotting, and numerical analysis.

.. currentmodule:: spectral.utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   io.ensure_output_dir
   io.load_simulation_data
   io.save_simulation_data
   formatting.extract_metadata
   formatting.format_dt_latex
   formatting.format_parameter_range
   formatting.build_parameter_string
   plotting.add_parameter_footer
   plotting.get_repo_root
   norms.discrete_l2_norm
   norms.discrete_l2_error
   norms.discrete_linf_error

