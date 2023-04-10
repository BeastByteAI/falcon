# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from os.path import dirname, abspath

d = dirname(dirname(dirname(abspath(__file__))))
print(d)
sys.path.append(d)

project = "Falcon"
copyright = "2022, Oleg Kostromin, Marco Pasini, Iryna Kondrashchenko"
author = "Oleg Kostromin, Marco Pasini, Iryna Kondrashchenko"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_design"
]
napoleon_numpy_docstring = True
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_style = "customStyles.css"
html_static_path = ["_static"]
html_theme_options = {"navigation_depth": 4}
