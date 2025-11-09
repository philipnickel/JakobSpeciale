"""LaTeX/PDF output configuration for Sphinx.

This file contains all LaTeX-specific settings for generating PDF documentation.
Imported by conf.py when building LaTeX output.
"""

# -- LaTeX output options ----------------------------------------------------

latex_engine = "pdflatex"

latex_documents = [
    ("index_latex", "Apendix_code.tex", "Code Appendix", "", "report"),
]

# Don't show URLs for external references (intersphinx links) as footnotes
latex_show_urls = "no"

latex_elements = {
    # A4 and 11pt, like your report
    "papersize": "a4paper",
    "pointsize": "11pt",
    # Ensure UTF-8 and T1 with pdflatex
    "inputenc": r"\usepackage[utf8]{inputenc}",
    "fontenc": r"\usepackage[T1]{fontenc}",
    # Geometry: use same fractional margins you have
    "sphinxsetup": r"""
        hmargin={0.15\paperwidth,0.15\paperwidth},
        vmargin={0.1111\paperheight,0.1111\paperheight},
        InnerLinkColor={HTML}{000000},
        OuterLinkColor={HTML}{000000},
        % code-block aesthetics
        pre_border-TeXcolor={HTML}{B3B3B3},
        pre_background-TeXcolor={HTML}{F7F7F7},
        pre_border-width=0.5pt,
        pre_padding=3pt
    """,
    # Keep figures anchored
    "figure_align": "H",
    # Add math/formatting packages
    "preamble": r"""
\usepackage{mathtools,amssymb,bm}
\usepackage[protrusion=true,expansion=true]{microtype}
\usepackage{siunitx}
\sisetup{separate-uncertainty=true}

% Match headers/footers
\usepackage{fancyhdr}
\setlength{\headheight}{13.6pt}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancyfoot[L]{\footnotesize Assignment 2}
\fancyfoot[C]{\footnotesize\thepage}
\fancyfoot[R]{\footnotesize Advanced Numerical Algorithms}

% Make chapter titles plain
\makeatletter
\def\@makechapterhead#1{%
  \vspace*{1\baselineskip}%
  {\parindent\z@ \raggedright\normalfont
   \Huge\bfseries #1\par
   \nobreak
   \vspace{0.5\baselineskip}}}
\def\@makeschapterhead#1{%
  \vspace*{1\baselineskip}%
  {\parindent\z@ \raggedright\normalfont
   \Huge\bfseries #1\par
   \nobreak
   \vspace{0.5\baselineskip}}}
\makeatother
""",
}
