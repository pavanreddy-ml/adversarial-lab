# Configuration file for Sphinx documentation

import os
import sys

# -- Path Setup --------------------------------------------------------------

# Ensure Sphinx can find the project root
sys.path.insert(0, os.path.abspath(".."))  # Adjust path to locate `adversarial_lab`

# -- Project Information -----------------------------------------------------
project = "Adversarial Lab"
author = "Pavan Reddy"
release = "0.0.1"

# -- General Configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # Parses docstrings
    "sphinx.ext.autosummary",    # Generates summary tables
    "sphinx.ext.napoleon",       # Supports Google-style docstrings
    "sphinx.ext.viewcode",       # Links to source code
    "sphinx.ext.todo"            # Allows TODOs in documentation
]

autosummary_generate = True  # Auto-generate summary pages

# Automatically document members, including private & special methods
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",
    "show-inheritance": True
}

# -- HTML Output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"  # ReadTheDocs theme
html_static_path = ["_static"]

# -- Options for Autodoc & Napoleon ------------------------------------------
napoleon_google_docstring = True  # Enable Google-style docstrings
napoleon_numpy_docstring = False  # Disable NumPy-style docstrings

# -- Suppress Warnings ------------------------------------------------------
suppress_warnings = ["autodoc.import_object"]

# -- Exclude Build Directories -----------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Additional Settings -----------------------------------------------------
todo_include_todos = True  # Include TODOs in output
