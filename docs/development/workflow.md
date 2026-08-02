# Development Workflow

## Scope

This document defines the branch, worktree, parallel-development, review, and
integration rules for `pybspcov`. The bootstrap root commit is the only
exception to the protected-main workflow. The `origin` remote is
`https://github.com/kw-lee/pybspcov.git`. After the bootstrap commit is pushed,
changes to `main` must arrive through reviewed pull requests.

## Protected `main` Branch

Create an active GitHub ruleset targeting the default branch with these rules:

- Require a pull request before merge.
- Require at least one approval from a person other than the most recent pusher.
- Dismiss stale approvals when the diff changes.
- Require all review conversations to be resolved.
- Require the `quality`, `tests`, `docs`, and `build` status checks.
- Require the branch to be up to date before merge.
- Require linear history and use squash merge for normal pull requests.
- Block force pushes and branch deletion.
- Do not grant routine bypass permission to automation or maintainers.

An emergency maintainer bypass must be documented in a follow-up issue and
reviewed after the incident. Signed commits are encouraged but are not a merge
requirement during the initial contributor phase.

## Branch Names

Use short, lowercase, hyphenated topic names:

- `feat/<topic>` for product or algorithm work.
- `fix/<topic>` for defect corrections.
- `test/<topic>` for independent test infrastructure or reference fixtures.
- `docs/<topic>` for documentation-only work.
- `perf/<topic>` for benchmark or optimization experiments.
- `chore/<topic>` for packaging, CI, and repository maintenance.

Do not reuse a merged branch name for unrelated work. A branch should have one
reviewable purpose and should not mix algorithm, formatting, and infrastructure
changes without a direct dependency.

## Worktrees

Use one branch per worktree. The project-local `.worktrees/` directory is the
default fallback when no platform-native worktree mechanism is available. It
must remain ignored by Git.

Create and prepare a worktree from an up-to-date `main`:

```bash
git switch main
git pull --ff-only
git worktree add .worktrees/gig-sampler -b feat/gig-sampler main
cd .worktrees/gig-sampler
uv sync --all-groups --all-extras
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest -q
```

If the baseline fails, stop and report the failure before changing code. Each
worktree has its own `.venv`; the shared `uv` download cache may be reused.

Inspect and remove worktrees with Git rather than deleting their directories:

```bash
git worktree list
git worktree remove .worktrees/gig-sampler
git branch -d feat/gig-sampler
git worktree prune
```

Remove a worktree only after its changes are committed or intentionally
discarded and its pull request is merged or closed. Do not use forced removal
to hide an unclean tree.

### Merged Branch Cleanup

Audit branches and worktrees at the end of every integration wave and before
creating another worktree. Fetch remote deletion state first, then remove only
clean worktrees whose pull requests are merged or closed:

```bash
git fetch --prune origin
git worktree list
git branch --merged main
git worktree remove .worktrees/gig-sampler
git branch -d feat/gig-sampler
git worktree prune
```

Delete the corresponding remote topic branch when the pull request is merged.
Do not remove the base of an active stacked branch until its dependents have
been rebased or retargeted to the updated `main`. Never use forced worktree or
branch deletion as routine cleanup; investigate uncommitted or unmerged work
instead.

## Parallel Work Rules

Parallel work is allowed only when tasks have independent inputs, outputs, and
file ownership. Every parallel task gets its own branch, worktree, test cycle,
and pull request.

| Track | Primary ownership | May start when |
| --- | --- | --- |
| Package scaffolding | `pyproject.toml`, package skeleton, shared CI contracts | Immediately; serial foundation |
| GIG sampler | `src/pybspcov/sampling/`, focused sampler tests | Package skeleton is merged |
| R reference fixtures | `reference/r/`, fixture metadata, parity harness | Package skeleton is merged |
| Validation and estimator contracts | `src/pybspcov/validation/`, estimator API tests | Public API contract is merged |
| Sphinx infrastructure | `docs/source/`, docs build configuration | Package metadata is merged |
| Benchmark harness | `benchmarks/`, benchmark metadata schema | Package skeleton is merged |
| BM kernel | BM kernel and integration tests | GIG sampler and state interfaces are merged |
| SBM screening and kernel | SBM-owned kernel and screening tests | Shared kernel interfaces are merged |
| Examples and tutorials | `examples/`, public guide pages | Relevant estimator API is stable |
| GPU optimization | isolated `perf/` branches | Correctness and parity baselines pass |

Tests for a component belong in the same branch as its implementation. A
separate test track is appropriate only for shared reference fixtures, test
infrastructure, or an independent black-box parity suite.

## Shared-File Ownership

`pyproject.toml`, `uv.lock`, public `__init__.py` exports, common JAX state
types, and CI workflow files are integration hotspots. Assign one active owner
for each hotspot during a parallel wave. Other branches declare needed changes
in their pull-request description and let the owner integrate them.

Branches that depend on unmerged work should be stacked explicitly: branch the
dependent work from its prerequisite, target the prerequisite branch during
review, and retarget to `main` after the prerequisite merges. Do not duplicate
the prerequisite commits by copying patches between worktrees.

## Integration Sequence

1. Rebase or update the topic branch from current `main`.
2. Run focused tests, then the complete CPU suite.
3. Run formatting, type checking, documentation, and package-build checks.
4. Review generated files, dependency changes, and benchmark claims.
5. Open a focused pull request and resolve all review conversations.
6. Merge only after required checks and approval pass.
7. Delete the remote topic branch and remove its local worktree.

GPU correctness jobs may run concurrently when resources permit. Performance
benchmarks must hold an exclusive GPU reservation and record other device
activity so concurrent jobs do not contaminate timing or memory results.

### CI Accelerator Policy

The required pull-request suite runs on CPU and tests accelerator selection and
failure handling with deterministic mocks. It must not initialize CUDA or infer
GPU availability from a generic GitHub-hosted runner. Real CUDA smoke and
correctness checks belong in a separate workflow backed by an explicitly
provisioned GPU runner; performance benchmarks remain manual or scheduled and
must never be a required check on ordinary pull requests.

## Agentic and AI-Assisted Work

An agentic worker receives one bounded task and one worktree. Two workers must
not edit the same hotspot concurrently. Agent output is reviewed as untrusted
code: inspect the diff, verify provenance, run the full integration suite, and
never grant issue-triage or code-generation automation release credentials.

## Documentation Integration

Sphinx uses `docs/source/` as its explicit source directory and
`docs/_build/` as generated output. Engineering records under
`docs/superpowers/` therefore do not enter the public documentation build.
The repository-level `examples/` directory remains executable source; selected
examples may later be rendered into the public guide without copying their
implementation.
