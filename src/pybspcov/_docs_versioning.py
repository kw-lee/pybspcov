"""Shared rules for selecting and naming published documentation versions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

RefKind = Literal["branch", "tag"]

_FEATURE_BRANCH = re.compile(r"feat/.+")
_RELEASE_TAG = re.compile(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?")


class RefLike(Protocol):
    """Minimal revision information needed by the publishing rules."""

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def remote(self) -> str | None: ...


@dataclass(frozen=True)
class PublishedRef:
    """A normalized branch or tag selected for documentation publishing."""

    name: str
    kind: RefKind
    remote: str | None = None


class VersionEntry(TypedDict):
    """Template metadata for one published documentation version."""

    kind: RefKind
    name: str
    path: str


class VersionMetadata(TypedDict):
    """Template metadata for a versioned Sphinx build."""

    current_version: VersionEntry
    versions: list[VersionEntry]


def published_path(name: str, kind: str) -> str:
    """Return the stable Pages path for a supported Git reference."""
    if kind == "branch" and name == "main":
        return "main"
    if kind == "branch" and _FEATURE_BRANCH.fullmatch(name):
        return f"branches/{name}"
    if kind == "tag" and _RELEASE_TAG.fullmatch(name):
        return f"versions/{name}"
    raise ValueError(f"Unsupported documentation reference: {kind} {name!r}")


def is_published_ref(name: str, kind: str, remote: str | None) -> bool:
    """Return whether a reference belongs in the public Pages artifact."""
    if kind == "branch":
        return remote == "origin" and (
            name == "main" or _FEATURE_BRANCH.fullmatch(name) is not None
        )
    return kind == "tag" and remote is None and _RELEASE_TAG.fullmatch(name) is not None


def _entry(ref: RefLike) -> VersionEntry:
    if ref.kind == "branch":
        kind: RefKind = "branch"
    elif ref.kind == "tag":
        kind = "tag"
    else:
        raise ValueError(f"Unsupported documentation reference kind: {ref.kind!r}")
    return {"kind": kind, "name": ref.name, "path": published_path(ref.name, kind)}


def _sort_key(ref: RefLike) -> tuple[int, str]:
    if ref.kind == "branch" and ref.name == "main":
        return (0, ref.name)
    if ref.kind == "branch":
        return (1, ref.name)
    return (2, ref.name)


def version_metadata(refs: Sequence[RefLike], current: RefLike) -> VersionMetadata:
    """Create deterministic version-selector data for Sphinx templates."""
    return {
        "current_version": _entry(current),
        "versions": [_entry(ref) for ref in sorted(refs, key=_sort_key)],
    }
