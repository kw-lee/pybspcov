import re
from importlib.metadata import metadata, version
from pathlib import Path


def test_package_version_is_exported() -> None:
    import pybspcov

    assert pybspcov.__version__ == version("pybspcov")


def test_distribution_metadata() -> None:
    package = metadata("pybspcov")
    assert package["Author-email"] is not None
    assert "kwlee1718@gmail.com" in package["Author-email"]
    assert package["License-Expression"] == "GPL-2.0-or-later"


def test_actions_are_pinned_to_full_shas() -> None:
    workflow = Path(".github/workflows/ci.yml")
    assert workflow.is_file()
    for line in workflow.read_text(encoding="utf-8").splitlines():
        if "uses:" in line:
            assert re.search(r"@[0-9a-f]{40}(?:\s|$)", line)
