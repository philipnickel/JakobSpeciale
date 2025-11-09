Assignment 2 – 02689 Advanced Numerical Algorithms
==================================================

**Authors:** Louis Kamp Eskildsen, Aske Funch Schrøder Nielsen, Philip Korsager Nickel

This documentation serves as an appendix to assignment 2 for DTU course 02689 Advanced Numerical Algorithms.

For the full codebase, please visit the `GitHub repository <https://github.com/philipnickel/02689-Advanced-Num>`.

Contents
--------

:doc:`exercise_gallery/index`
   Gallery of all exercises (A-H) with the scripts used for computing data and figures.
:doc:`api_reference`
   Complete reference for our implementation in the form of the ``spectral`` package.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Exercises

   exercise_gallery/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   api_reference

Installation
------------

The package requires Python 3.12+ and uses ``uv`` for dependency management::

    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync

