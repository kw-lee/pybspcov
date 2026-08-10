# Contributing to pybspcov

## Workflow

Use a focused topic branch and one worktree per task. Add tests with behavior
changes, run the focused tests first, and run the complete CPU verification
matrix before requesting review. Changes reach `main` only through a pull
request satisfying `docs/development/workflow.md`.

## Scientific and license review

Compare statistical behavior with independently generated R reference results.
Review copied or adapted material for provenance, compatible licensing, and
required attribution. Preserve `GPL-2.0-or-later` for derived project code.
GitHub Actions must use least privilege and full commit SHAs.

## AI assistance

Disclose material AI assistance in the pull request. Do not commit prompts or
chat logs. Treat generated output as unreviewed: inspect every diff, verify
claims against primary sources, and use independent tests. Never send secrets,
restricted data, unpublished research data, or third-party confidential code
to an external model.

## Checks

Run `uv run pre-commit run --all-files`, `uv run mypy -p pybspcov`,
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest -q`, the Sphinx build, and
package build
checks described in the development workflow.
