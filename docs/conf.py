import os
import sys

# Ensure Sphinx finds the project root
sys.path.insert(0, os.path.abspath("../adversarial_lab"))

# Project info
project = "Adversarial Lab"
author = "Pavan Reddy"
release = "0.0.1rc0"

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",
    "show-inheritance": True
}

# Theme
html_theme = "sphinx_rtd_theme"

# Prevent warnings for missing _static/
html_static_path = []

rst_prolog = """
.. toctree::
   :maxdepth: 100
"""

# -- HTML Output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"  # ReadTheDocs theme

# Prevent warnings for missing _static/
html_static_path = []

# -- Exclude Build Directories -----------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Additional Settings -----------------------------------------------------
todo_include_todos = True  # Include TODOs in output