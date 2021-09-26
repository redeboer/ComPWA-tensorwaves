# type: ignore

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import inspect
import os
import re
import shutil
import subprocess
import sys
from os.path import dirname, realpath
from typing import Optional

import sphobjinv as soi
from git import Repo
from git.exc import InvalidGitRepositoryError
from git.objects import Commit, TagObject
from pkg_resources import get_distribution

# -- Project information -----------------------------------------------------
project = "TensorWaves"
package = "tensorwaves"
REPO_NAME = "tensorwaves"
copyright = "2020, ComPWA"
author = "Common Partial Wave Analysis"

if os.path.exists(f"../src/{package}/version.py"):
    __release = get_distribution(package).version
    version = ".".join(__release.split(".")[:3])

# -- Generate API ------------------------------------------------------------
sys.path.insert(0, os.path.abspath("."))
import abbreviate_signature

shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    " ".join(
        [
            "sphinx-apidoc",
            f"../src/{package}/",
            "-o api/",
            "--force",
            "--no-toc",
            "--templatedir _templates",
            "--separate",
        ]
    ),
    shell=True,
)

# -- Convert sphinx object inventory -----------------------------------------
subprocess.call("sphobjinv convert -o zlib tensorflow.txt", shell=True)


# -- General configuration ---------------------------------------------------
master_doc = "index.md"
source_suffix = {
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
    ".rst": "restructuredtext",
}

# The master toctree document.
master_doc = "index"
modindex_common_prefix = [
    f"{package}.",
]

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_panels",
    "sphinx_thebe",
    "sphinx_togglebutton",
]
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "adr*",
    "tests",
]

# General sphinx settings
add_module_names = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": ", ".join(
        [
            "__call__",
            "__eq__",
        ]
    ),
}
autodoc_insert_signature_linebreaks = False
graphviz_output_format = "svg"
html_copy_source = True  # needed for download notebook button
html_css_files = []
if autodoc_insert_signature_linebreaks:
    html_css_files.append("linebreaks-api.css")
html_favicon = "_static/favicon.ico"
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/ComPWA/{REPO_NAME}",
    "repository_branch": "stable",
    "path_to_docs": "docs",
    "use_download_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "theme_dev_mode": True,
}
html_title = "TensorWaves"
panels_add_bootstrap_css = False  # wider page width with sphinx-panels
pygments_style = "sphinx"
todo_include_todos = False

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"
nitpicky = True  # warn if cross-references are missing
nitpick_ignore = [
    ("py:class", "tensorflow.keras.losses.Loss"),
    ("py:class", "tensorflow.python.keras.losses.Loss"),
    ("py:obj", "Loss"),
]

# Intersphinx settings
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
CONSTRAINTS_PATH = f"../.constraints/py{PYTHON_VERSION}.txt"
with open(CONSTRAINTS_PATH) as stream:
    CONSTRAINTS = stream.read()
RELEASES = dict()
for line in CONSTRAINTS.split("\n"):
    line = line.split("#")[0]  # remove comments
    line = line.strip()
    if not line:
        continue
    package, version = tuple(line.split("=="))
    package = package.strip()
    version = version.strip()
    RELEASES[package] = version

intersphinx_mapping = {
    "ampform": (
        f"https://ampform.readthedocs.io/en/{RELEASES['ampform']}",
        None,
    ),
    "compwa-org": ("https://compwa-org.readthedocs.io/en/stable", None),
    "iminuit": ("https://iminuit.readthedocs.io/en/stable", None),
    "jax": ("https://jax.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pwa": ("https://pwa.readthedocs.io", None),
    "python": ("https://docs.python.org/3", None),
    "qrules": (
        f"https://qrules.readthedocs.io/en/{RELEASES['qrules']}",
        None,
    ),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "tensorflow.inv",
    ),
}

# Linkcode settings
def get_blob_url() -> str:
    try:
        local_repo_path = dirname(dirname(realpath(__file__)))
        repo = Repo(local_repo_path)
        branch = repo.active_branch
        tracking_branch = branch.tracking_branch()
        if tracking_branch:
            remote = repo.remote(tracking_branch.remote_name)
            repo_url = get_repo_url(remote.url)
            if repo_url is not None:
                commit = repo.head.commit
                tag = get_tag(repo, commit)
                if tag is None:
                    commit = str(commit)[:7]
                    return f"{repo_url}/blob/{commit}"
                return f"{repo_url}/tree/{tag}"
    except InvalidGitRepositoryError:
        pass
    return f"https://github.com/ComPWA/{REPO_NAME}/blob/main"


def get_repo_url(ssh_remote: str) -> Optional[str]:
    matches = re.match(r"^git@github.com:(.+)/(.+).git$", ssh_remote)
    if matches is None:
        matches = re.match(
            r"^https://github.com/(.+)/(.+?)(\.git)?$", ssh_remote
        )
        if matches is None:
            return None
    organization = matches[1]
    repo_name = matches[2]
    return f"https://github.com/{organization}/{repo_name}"


def get_tag(repo: Repo, commit: Commit) -> TagObject:
    for tag in repo.tags:
        if commit == repo.commit(tag):
            return tag
    return None


def linkcode_resolve(domain, info):
    """See https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html.

    Based on
    https://github.com/numpy/numpy/blob/0b01b48/doc/source/conf.py#L409-L476
    but links to specific commits if available on the remote.
    """
    if domain != "py":
        return None

    module = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(module)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        line_specification = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        line_specification = ""

    filename = info["module"].replace(".", "/") + ".py"
    blob_url = get_blob_url()
    return f"{blob_url}/src/{filename}{line_specification}"


# Settings for autosectionlabel
autosectionlabel_prefix_document = True

# Settings for copybutton
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest

# Settings for linkcheck
linkcheck_anchors = False

# Settings for myst_nb
execution_timeout = -1
nb_output_stderr = "remove"
nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    )
}
nb_render_priority["doctest"] = nb_render_priority["html"]

jupyter_execute_notebooks = "off"
if "EXECUTE_NB" in os.environ:
    print("\033[93;1mWill run Jupyter notebooks!\033[0m")
    jupyter_execute_notebooks = "force"

# Settings for myst-parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
]
myst_update_mathjax = False

# Settings for Thebe cell output
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}
