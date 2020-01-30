# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

package_root = os.path.abspath('..')
package_name = 'inside_analysis'
sys.path.insert(0, package_root)
sys.path.insert(0, os.path.join(package_root,package_name))


# -- Project information -----------------------------------------------------

project = package_name
copyright = u'2019, Greta Del Nista'
author = u'Greta Del Nista'

# The short X.Y version
from version import __version__
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = extensions = ['sphinxcontrib.autoprogram',
                           'sphinx.ext.napoleon',
                           'sphinx.ext.autodoc',
                           'sphinx.ext.doctest',
                           'sphinx.ext.todo',
                           'sphinx.ext.coverage',
                           'sphinx.ext.mathjax',
                           'sphinx.ext.ifconfig',
                           'sphinx.ext.viewcode'
                           ]

source_suffix ='.rst'

master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'inside_analysisdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#
# 'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#
# 'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#
# 'preamble': '',

# Latex figure (float) alignment
#
# 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
                   (master_doc, 'inside_analysis.tex', u'inside_analysis Documentation',
                    u'Greta Del Nista', 'manual'),
                   ]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
             (master_doc, 'inside_analysis', u'inside_analysis Documentation',
              [author], 1)
             ]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
                     (master_doc, 'inside_analysis', u'inside_analysis Documentation',
                      author, 'inside_analysis', 'One line description of project.',
                      'Miscellaneous'),
                     ]


# -- Extension configuration -------------------------------------------------
