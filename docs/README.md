# Documentation Layout

The `docs` tree contains three deliberately separate kinds of documentation.

- `docs/source/` is the Sphinx source directory for published user and API
  documentation. Its `conf.py` and `index.md` define the public documentation
  build.
- `docs/development/` contains maintainer workflow and engineering policy.
- `docs/superpowers/specs/` contains approved design records.
- `docs/superpowers/plans/` contains executable implementation plans.
- `docs/_build/` contains generated Sphinx output and is ignored by Git.

Sphinx is always invoked with an explicit source directory:

```bash
uv run sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

Because Sphinx reads `docs/source/`, it does not discover or publish planning
records under `docs/superpowers/`. Public development guidance may link to a
curated page under `docs/source/development/`; internal plans are not added to
the Sphinx toctree.

Executable examples live in the repository-level `examples/` directory. A
later examples plan will connect them to Sphinx after the estimator API is
stable. Small CPU examples must run in CI; GPU-only performance examples are
kept out of the ordinary documentation build.

See [Development Workflow](development/workflow.md) for branch, worktree,
parallel-development, and integration rules.
