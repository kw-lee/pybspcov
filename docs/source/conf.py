from importlib.metadata import version as package_version

project = "pybspcov"
author = "Kyeongwon Lee"
release = package_version("pybspcov")
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
html_theme = "furo"
nitpicky = True
exclude_patterns = []
