from dataclasses import dataclass

import pytest

from pybspcov import _docs_versioning as versioning


@dataclass(frozen=True)
class StubRef:
    name: str
    kind: str
    remote: str | None = None


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (StubRef("main", "branch", "origin"), "main"),
        (
            StubRef("feat/posterior/plots", "branch", "origin"),
            "branches/feat/posterior/plots",
        ),
        (StubRef("v1.2.3", "tag"), "versions/v1.2.3"),
    ],
)
def test_published_path_separates_main_feature_branches_and_tags(
    ref: StubRef, expected: str
) -> None:
    assert versioning.published_path(ref.name, ref.kind) == expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (StubRef("main", "branch", "origin"), True),
        (StubRef("feat/posterior", "branch", "origin"), True),
        (StubRef("fix/posterior", "branch", "origin"), False),
        (StubRef("feat/posterior", "branch"), False),
        (StubRef("v1.2.3", "tag"), True),
        (StubRef("1.2.3", "tag"), False),
    ],
)
def test_only_public_remote_branches_and_release_tags_are_selected(
    ref: StubRef, expected: bool
) -> None:
    assert versioning.is_published_ref(ref.name, ref.kind, ref.remote) is expected


def test_version_metadata_exposes_stable_labels_and_urls() -> None:
    refs = [
        StubRef("main", "branch", "origin"),
        StubRef("feat/posterior", "branch", "origin"),
        StubRef("v1.2.3", "tag"),
    ]

    assert versioning.version_metadata(refs, refs[1]) == {
        "current_version": {
            "kind": "branch",
            "name": "feat/posterior",
            "path": "branches/feat/posterior",
        },
        "versions": [
            {"kind": "branch", "name": "main", "path": "main"},
            {
                "kind": "branch",
                "name": "feat/posterior",
                "path": "branches/feat/posterior",
            },
            {"kind": "tag", "name": "v1.2.3", "path": "versions/v1.2.3"},
        ],
    }
