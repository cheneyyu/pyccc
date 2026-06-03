from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "pyccc"
author = "pyccc contributors"
copyright = "2026, pyccc contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

html_theme = "alabaster"
html_static_path = ["_static"]
html_title = "pyccc"

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
myst_enable_extensions = ["colon_fence"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
