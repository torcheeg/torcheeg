# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import inspect
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import os.path as op
import sys

import pytorch_sphinx_theme
import sphinx_gallery

sys.path.insert(0, os.path.abspath('../../'))

import datetime

import torcheeg

# -- Project information -----------------------------------------------------

project = 'torcheeg'
version = torcheeg.__version__
author = 'TorchEEG Team'
copyright = f'{datetime.datetime.now().year}, {author}'

# The full version, including alpha/beta/rc tags
release = '1.0.10'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax', 'sphinx.ext.napoleon', 'sphinx.ext.autodoc',
    'sphinx.ext.autosummary', 'sphinx.ext.intersphinx', 'sphinx.ext.viewcode',
    'sphinx.ext.linkcode', 'sphinx_gallery.gen_gallery'
]


def linkcode_resolve(domain, info):
    # adapted from https://github.com/braindecode/braindecode/blob/master/docs/conf.py
    import mne
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None
    # deal with our decorators properly
    while hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None
    fn = op.relpath(fn, start=op.dirname(mne.__file__))
    fn = '/'.join(op.normpath(fn).split(os.sep))  # in case on Windows

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    return "http://github.com/tczhangzhi/torcheeg/blob/main/torcheeg/%s%s" % (
        fn, linespec)


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

sphinx_gallery_conf = {
    'examples_dirs': ['../../examples'],
    'gallery_dirs': ['auto_examples'],
    'doc_module': ('torcheeg', 'mne'),
    'backreferences_dir': 'generated',
    'reference_url': dict(torcheeg=None),
}

html_logo = '_static/torcheeg_logo_light.svg'

html_css_files = ['custom.css']

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "pytorch_project": "docs",
    "navigation_with_keys": True
}

html_context = {
    'display_github': True,
    'github_user': 'tczhangzhi',
    'github_repo': 'torcheeg',
    'github_version': 'main/docs/',
}

html_favicon = '_static/favicon.ico'