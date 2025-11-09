"""Sphinx configuration for spectral package documentation."""

import importlib.util
import os
import sys

# Add src directory to path for module imports
src_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "src"))
sys.path.insert(0, src_path)

repo_root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
sys.path.insert(0, repo_root)

# -- Project information -----------------------------------------------------

project = "Assignment 2 - 02689 Advanced Numerical Algorithms"
copyright = (
    "2025, Louis Kamp Eskildsen, Aske Funch Schrøder Nielsen, Philip Korsager Nickel"
)
author = "Louis Kamp Eskildsen, Aske Funch Schrøder Nielsen, Philip Korsager Nickel"

# -- General configuration ---------------------------------------------------

# Detect LaTeX build from marker file (set by main.py)
import os

_marker_file = os.path.join(os.path.dirname(__file__), ".building_latex")
_is_latex_build = os.path.exists(_marker_file)

# Extensions - conditionally exclude gallery for LaTeX
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
]

# Only add gallery for non-LaTeX builds
if not _is_latex_build:
    extensions.append("sphinx_gallery.gen_gallery")

# Source files - use different index for LaTeX
root_doc = "index_latex" if _is_latex_build else "index"
source_suffix = {".rst": "restructuredtext"}

# -- Autodoc configuration ---------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

# Mock heavy runtime dependencies
# autodoc_mock_imports = ["numba", "pyarrow", "matplotlib"]  # Disabled - causing import issues

# -- Numpydoc configuration --------------------------------------------------

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "default", "of"}

# Use numpydoc's enhanced autosummary templates
templates_path = ["_templates"]
numpydoc_use_plots = False  # Don't auto-generate plots from Examples

# -- Sphinx Gallery configuration --------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "../../Exercises",  # Path to exercise scripts
    "gallery_dirs": "exercise_gallery",  # Output directory for gallery
    "filename_pattern": "/plot_",  # Pattern to match which scripts to execute
    "download_all_examples": False,  # No download buttons
    "remove_config_comments": True,  # Clean up notebook outputs
    "abort_on_example_error": False,  # Continue if examples fail
    "plot_gallery": True,  # Gallery extension will be removed for LaTeX in setup()
    "capture_repr": ("_repr_html_", "__repr__"),  # Capture output representations
    "matplotlib_animations": True,  # Support matplotlib animations
    # Remove Jupyter cell markers (# %%) from rendered output
    "first_notebook_cell": None,  # Don't add a first cell
    "last_notebook_cell": None,  # Don't add a last cell
    "notebook_images": False,  # Don't embed images in notebooks
    # Cross-referencing: Create "Examples using X" in API docs
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("spectral",),  # Generate backreferences for our package
    "inspect_global_variables": True,  # Detect classes/functions used in examples
    # Make code clickable: Link to API docs when code mentions spectral functions
    "reference_url": {
        "spectral": None,  # None = use local docs (not external URL)
    },
}

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output options -----------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "02689 Advanced Numerical Algorithms - Assignment 2"
html_static_path = ["_static"]
html_show_sourcelink = False  # Hide "Show Source" link
html_css_files = ["custom.css"]  # Custom CSS for hiding download buttons

html_theme_options = {
    "github_url": "https://github.com/philipnickel/02689-Advanced-Num",
    "show_nav_level": 1,  # Only show top-level items expanded in sidebar
    "navigation_depth": 2,  # Allow 2 levels but don't expand by default
    "show_toc_level": 3,  # Show 3 levels in the page TOC (includes subsections)
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "collapse_navigation": True,  # Start with collapsed navigation
    "secondary_sidebar_items": ["page-toc"],  # Only show page TOC, not section nav
}

# -- LaTeX/PDF output options ------------------------------------------------

# Import LaTeX configuration from separate file
spec = importlib.util.spec_from_file_location(
    "latex_conf", os.path.join(os.path.dirname(__file__), "latex_conf.py")
)
latex_conf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latex_conf)

latex_engine = latex_conf.latex_engine
latex_documents = latex_conf.latex_documents
latex_show_urls = latex_conf.latex_show_urls
latex_elements = latex_conf.latex_elements
