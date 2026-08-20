import os
import sys
from pathlib import Path

_DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = Path(
    os.environ.get("PYBSPCOV_SOURCE_ROOT", str(_DEFAULT_REPOSITORY_ROOT))
)
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from pybspcov._version import __version__

project = "pybspcov"
author = "Kyeongwon Lee"
release = __version__
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
html_theme = "furo"
templates_path = ["_templates"]
html_baseurl = "https://kw-lee.github.io/pybspcov/"
html_context = {
    "current_version": None,
    "docs_base_url": html_baseurl,
    "versions": [],
}
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "version-selector.html",
    ]
}

try:
    from sphinx_polyversion.api import LoadError, load
except ImportError:
    pass
else:
    try:
        load(globals())
    except LoadError:
        pass

nitpicky = True
nitpick_ignore_regex = [
    (
        "py:class",
        (
            r"(?:collections\.abc\.Sequence|jax\.Array|"
            r"numpy\.(?:ndarray|bool|number)|"
            r"_DTypeName|_DeviceRequest|_CutoffMethod)"
        ),
    ),
]
nitpick_ignore = [
    ("py:class", "ArrayLike"),
    ("py:class", "DTypeName"),
    ("py:class", "PlotType"),
    ("py:class", "ThresholdMethod"),
    ("py:class", "jaxlib._jax.Device"),
    ("py:class", "pathlib.Path"),
]

exclude_patterns = []
