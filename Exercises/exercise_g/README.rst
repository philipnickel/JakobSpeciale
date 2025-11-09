Exercise G
==========

Nonlinear problems: profiling, scalability, and work-precision analysis for the KdV solver.

.. rubric:: Available scripts

Profiling
---------

* ``compute_profiling_functions.py`` – Function-level profiling using cProfile.
  Saves to ``data/A2/ex_g/cprofile_functions.parquet``.

* ``compute_profiling_lines.py`` – Line-by-line profiling using line_profiler.
  Requires @profile decorators in ``src/spectral/tdp.py``.

  Usage: ``uv run kernprof -l Exercises/exercise_g/compute_profiling_lines.py``
  then ``uv run python Exercises/exercise_g/compute_profiling_lines.py --parse``

* ``plot_profiling_functions.py`` – Function-level profiling visualizations.
  Saves to ``figures/A2/ex_g/function_profiling.pdf``.

* ``plot_profiling_lines.py`` – Line-by-line profiling visualizations.
  Saves to ``figures/A2/ex_g/line_profiling.pdf``.

Performance Analysis
--------------------

* ``compute_scalability.py`` – Measures strong/weak scaling behavior.
  Saves to ``data/A2/ex_g/scalability_timing.parquet``.

* ``compute_work-precision.py`` – Sweeps stable time-step fractions for RK3/RK4 and
  records work vs accuracy metrics. Saves to ``data/A2/ex_g/work_precision.parquet``.

* ``plot_scalability.py`` – Creates scalability analysis plots.
  Saves to ``figures/A2/ex_g/scalability_analysis.pdf``.

* ``plot_work-precision.py`` – Creates work-precision diagrams.
  Saves to ``figures/A2/ex_g/work_precision.pdf``.

Note: Run the corresponding ``compute_*.py`` or ``profile_*.py`` scripts before plotting.
