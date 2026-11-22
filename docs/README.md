# Documentation Layout

The `docs` tree contains two kinds of documentation.

- `docs/source/` is the Sphinx source directory for published user and API
  documentation. Its `conf.py` and `index.md` define the public documentation
  build.
- `docs/development/` contains maintainer workflow and engineering policy.
- `docs/_build/` contains generated Sphinx output and is ignored by Git.

Sphinx is always invoked with an explicit source directory:

```bash
uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

Because Sphinx reads `docs/source/`, it does not discover or publish maintainer
documents under `docs/development/`. Public development guidance may link to a
curated page under `docs/source/development/`.

Executable examples live in the repository-level `examples/` directory. A
later examples plan will connect them to Sphinx after the estimator API is
stable. Small CPU examples must run in CI; GPU-only performance examples are
kept out of the documentation build.

See [Development Workflow](development/workflow.md) for branch, worktree,
parallel-development, and integration rules.

## Versioned GitHub Pages build

The Pages workflow rebuilds one artifact from the complete Git history whenever
`main`, a `feat/**` branch, or a `v*` tag is pushed. Published documentation is
available at these stable paths:

- `main` at `/pybspcov/main/`
- feature branches at `/pybspcov/branches/feat/<branch-name>/`
- release tags at `/pybspcov/versions/<tag>/`

The site root redirects to the `main` documentation. Each version built with
the current Sphinx configuration also exposes the available branches and tags
in the Furo sidebar.

To reproduce the Pages artifact from a full clone:

```bash
uv sync --locked --all-groups
uv run sphinx-polyversion --sequential docs/poly.py docs/_build/html
```

The multi-version build selects `origin/main`, public `origin/feat/**` branches,
and local semantic-version tags such as `v1.2.3`.
