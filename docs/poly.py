"""Build all public documentation revisions into one GitHub Pages artifact."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from sphinx_polyversion.api import apply_overrides
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.environment import Environment
from sphinx_polyversion.git import Git, GitRef, GitRefType, file_predicate
from sphinx_polyversion.sphinx import SphinxBuilder

from pybspcov._docs_versioning import (
    PublishedRef,
    RefKind,
    is_published_ref,
    published_path,
    version_metadata,
)

OUTPUT_DIR = "docs/_build/html"
SOURCE_DIR = Path("docs/source")
MOCK = False
SEQUENTIAL = False
DOCS_BASE_URL = "https://kw-lee.github.io/pybspcov/"

_has_documentation = file_predicate([SOURCE_DIR, SOURCE_DIR / "conf.py"])


def _kind(ref: GitRef) -> RefKind:
    return "tag" if ref.type_ == GitRefType.TAG else "branch"


def _published_ref(ref: GitRef) -> PublishedRef:
    return PublishedRef(ref.name, _kind(ref), ref.remote)


class SourceEnvironment(Environment):
    """Run Sphinx against the source tree checked out for each revision."""

    async def run(
        self,
        *cmd: str,
        decode: bool = True,
        **kwargs: Any,
    ) -> tuple[str | bytes | None, str | bytes | None, int]:
        environment = dict(kwargs.get("env") or os.environ)
        source_path = str(self.path / "src")
        existing_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            path for path in (source_path, existing_path) if path
        )
        environment["PYBSPCOV_SOURCE_ROOT"] = str(self.path)
        kwargs["env"] = environment
        return await super().run(*cmd, decode=decode, **kwargs)


class PublishedGit(Git):
    """Select origin branches and local tags intended for public Pages."""

    async def predicate(self, root: Path, ref: GitRef) -> bool:
        selected = is_published_ref(ref.name, _kind(ref), ref.remote)
        return selected and await _has_documentation(root, ref)


def output_name(ref: GitRef) -> str:
    """Map a Git revision to its stable output directory."""
    return published_path(ref.name, _kind(ref))


def build_data(
    driver: DefaultDriver[GitRef, Environment, object],
    current: GitRef,
    environment: Environment,
) -> dict[str, object]:
    """Return version-selector data for one revision build."""
    del environment
    refs = [_published_ref(ref) for ref in driver.targets]
    return cast(
        dict[str, object],
        {
            **version_metadata(refs, _published_ref(current)),
            "docs_base_url": DOCS_BASE_URL,
        },
    )


def root_data(
    driver: DefaultDriver[GitRef, Environment, object],
) -> dict[str, object]:
    """Return metadata available to root-level templates."""
    return {
        "docs_base_url": DOCS_BASE_URL,
        "versions": [_published_ref(ref) for ref in driver.builds],
    }


apply_overrides(globals())
repository_root = Git.root(Path(__file__).parent)

DefaultDriver(
    repository_root,
    OUTPUT_DIR,
    vcs=PublishedGit(branch_regex=r".*", tag_regex=r".*"),
    builder=SphinxBuilder(
        SOURCE_DIR,
        args=["--keep-going", "-b", "html", "-c", str(repository_root / SOURCE_DIR)],
    ),
    env=SourceEnvironment.factory(),
    namer=output_name,
    data_factory=build_data,
    root_data_factory=root_data,
    template_dir=repository_root / "docs/pages/templates",
    static_dir=repository_root / "docs/pages/static",
).run(MOCK, SEQUENTIAL)
