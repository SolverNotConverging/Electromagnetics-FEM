"""Sphinx configuration. Build with: sphinx-build -W docs docs/_build/html"""

from importlib.metadata import version as _version

project = "fem"
author = "FEM contributors"
release = _version("electromagnetics-fem")
version = ".".join(release.split(".")[:2])

extensions = ["sphinx.ext.mathjax", "myst_parser"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
myst_heading_anchors = 3

html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "description": "Finite-element electromagnetics",
    "fixed_sidebar": True,
}
