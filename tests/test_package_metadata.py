from importlib.metadata import metadata, version


def test_package_version_is_exported() -> None:
    import pybspcov

    assert pybspcov.__version__ == version("pybspcov")


def test_distribution_metadata() -> None:
    package = metadata("pybspcov")
    assert package["Author-email"] is not None
    assert "kwlee1718@gmail.com" in package["Author-email"]
    assert package["License-Expression"] == "GPL-2.0-or-later"
