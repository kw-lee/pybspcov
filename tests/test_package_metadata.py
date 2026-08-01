import re
from importlib.metadata import metadata, version
from pathlib import Path

import pytest


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
        uses = re.match(r"^\s*(?:-\s+)?uses:\s*(?P<reference>[^\s#]+)", line)
        if uses is None:
            continue
        reference = uses.group("reference")
        if reference.startswith(("./", "docker://")):
            continue
        assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference), (
            f"external action is not pinned to a full SHA: {reference}"
        )


def _run_action_pin_policy(
    workflow_text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = tmp_path / ".github/workflows/ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(workflow_text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    test_actions_are_pinned_to_full_shas()


def test_action_pin_policy_rejects_sha_only_in_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(AssertionError):
        _run_action_pin_policy(
            "- uses: actions/checkout@v6 "
            "# prior @aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
            tmp_path,
            monkeypatch,
        )


@pytest.mark.parametrize(
    "reference",
    ["./.github/actions/local", "docker://alpine:3.20"],
    ids=["local", "docker"],
)
def test_action_pin_policy_ignores_non_external_references(
    reference: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_action_pin_policy(f"- uses: {reference}\n", tmp_path, monkeypatch)


def test_action_pin_policy_ignores_commented_out_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_action_pin_policy("# - uses: actions/checkout@v6\n", tmp_path, monkeypatch)
