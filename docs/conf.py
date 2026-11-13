"""Sphinx configuration for the MANDALA documentation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).parent / "_ext")]

project = "MANDALA"
author = "MANDALA developers"
copyright = "2026, MANDALA developers"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "mandala_docs",
]

exclude_patterns = [
    "_build",
    "*.md",
]
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["mandala.css"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
html_title = "MANDALA documentation"
html_baseurl = "https://mandala-org.github.io/Mandala/"
pygments_style = "friendly"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {}
autodoc_inherit_docstrings = False
autosummary_generate = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True

graphviz_output_format = "svg"

mathjax3_config = {
    "tex": {
        "macros": {
            "bm": ["\\boldsymbol{#1}", 1],
            "Tr": "\\operatorname{Tr}",
            "irr": ["D^{(#1)}", 1],
        }
    }
}

rst_prolog = """
.. role:: python(code)
   :language: python
"""
