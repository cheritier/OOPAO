# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# Add the OOPAO package root to sys.path so autodoc can import modules.
# Adjust this path if your repo layout differs.
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information ------------------------------------------------------

project   = 'OOPAO'
copyright = '2024, C.T. Héritier, C. Vérinaud, and contributors'
author    = 'C.T. Héritier, C. Vérinaud, and contributors'

# The short X.Y version and the full version (update as needed)
version = '1.0'
release = '1.0'

# -- General configuration ----------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',       # Pull docstrings from source code
    'sphinx.ext.autosummary',   # Generate summary tables
    'sphinx.ext.napoleon',      # Google / NumPy style docstrings
    'sphinx.ext.viewcode',      # Add [source] links to API pages
    'sphinx.ext.intersphinx',   # Cross-link to NumPy, SciPy, etc.
    'sphinx.ext.mathjax',       # Render LaTeX math
    'sphinx.ext.todo',          # .. todo:: directives
]

# Autosummary: generate stub .rst files automatically
autosummary_generate = True

# Napoleon settings (NumPy-style docstrings used in OOPAO)
napoleon_google_docstring = False
napoleon_numpy_docstring  = True
napoleon_use_param        = True
napoleon_use_rtype        = True
napoleon_preprocess_types = True

# Autodoc defaults
autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,
    'show-inheritance': True,
    'member-order':     'bysource',
}
autodoc_typehints = 'description'

# Intersphinx: link to external package docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3',   None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

# todo extension
todo_include_todos = True

# Source file suffix
source_suffix = '.rst'

# The master toctree document
master_doc = 'index'

# Patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Syntax highlighting
pygments_style = 'sphinx'

# -- Options for HTML output --------------------------------------------------

html_theme = 'furo'          # pip install furo  (clean, modern theme)
# Fallback: 'pydata_sphinx_theme' or the built-in 'alabaster'

html_theme_options = {
    # furo options
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
    'source_repository': 'https://github.com/cheritier/OOPAO',
    'source_branch': 'master',
    'source_directory': 'docs/source/',
    'light_css_variables': {
        'color-brand-primary':    '#0a6fb8',
        'color-brand-content':    '#0a6fb8',
        'color-highlight-on-target': '#fff3cd',
    },
}

html_static_path  = ['_static']
html_title        = 'OOPAO Documentation'
html_short_title  = 'OOPAO'

# Logo (place a logo.png in source/_static/ to enable)
# html_logo = '_static/logo.png'

# Favicon
# html_favicon = '_static/favicon.ico'

# -- Options for LaTeX / PDF output -------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
''',
}

latex_documents = [
    (master_doc,
     'OOPAO.tex',
     'OOPAO Documentation',
     r'C.T.\ Héritier \and C.\ Vérinaud \and contributors',
     'manual'),
]

# -- Options for manual page output -------------------------------------------

man_pages = [
    (master_doc, 'oopao', 'OOPAO Documentation', [author], 1)
]
