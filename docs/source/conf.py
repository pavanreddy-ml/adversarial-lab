# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Adversarial Lab'
copyright = '2025, Pavan Reddy'
author = 'Pavan Reddy'
release = '0.0.3'

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust based on your project structure


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Supports Google-style docstrings
    'sphinx_autodoc_typehints'  # Enables automatic type hint extraction
]

autodoc_typehints = "description"  # Shows type hints inside docstrings


html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 20,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
