"""
Plotting Utilities
==================

Helper functions for creating publication-quality plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def get_style_path(style_name: str = "ana") -> Path:
    """
    Get the path to a custom matplotlib style file.

    Parameters
    ----------
    style_name : str, optional
        Name of the style file (without .mplstyle extension)

    Returns
    -------
    Path
        Path to the style file
    """
    return Path(__file__).parent.parent / "styles" / f"{style_name}.mplstyle"


def setup_plotting(style: str = "ana") -> None:
    """
    Configure matplotlib with sensible defaults for scientific plotting.

    Parameters
    ----------
    style : str, optional
        Matplotlib style to use. Can be:
        - 'ana': Custom style for LaTeX reports (default)
        - Any built-in matplotlib style name

    Examples
    --------
    >>> from numutils.utils import setup_plotting
    >>> setup_plotting()  # Uses custom 'ana' style
    >>> setup_plotting('seaborn-v0_8-darkgrid')  # Uses built-in style
    """
    # Try custom style first
    if style == "ana":
        style_path = get_style_path("ana")
        if style_path.exists():
            plt.style.use(str(style_path))
            return

    # Fall back to built-in styles
    try:
        plt.style.use(style)
    except OSError:
        # If style not found, use default matplotlib settings
        pass
