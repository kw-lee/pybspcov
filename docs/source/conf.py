from importlib.metadata import version as package_version

project = "pybspcov"
author = "Kyeongwon Lee"
release = package_version("pybspcov")
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
html_theme = "furo"
nitpicky = True
nitpick_ignore_regex = [
    (
        "py:class",
        (
            r"(?:jax\.Array|numpy\.(?:ndarray|bool|number)|"
            r"_DTypeName|_DeviceRequest|_CutoffMethod)"
        ),
    ),
]
exclude_patterns = []
