# R Example Parity and Performance Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible R-versus-JAX BM example harness that validates posterior similarity and reports compile, steady-state, and end-to-end timing separately.

**Architecture:** A base-R generator owns the upstream example dataset and R results. A Python runner consumes the same dataset, invokes the public BM chain API, writes the same summary schema, and produces a comparison report. Correctness tests exercise parsers and statistical acceptance independently from wall-clock performance.

**Tech Stack:** R 4.x, bspcov 1.0.3, JAX, Python 3.12+, pytest, CSV, JSON, Markdown

## Global Constraints

- All repository content is English.
- The package remains pure Python with JAX as its only runtime dependency.
- Use float64 for statistical comparison.
- Do not require identical R and JAX random streams.
- Do not use absolute timing thresholds as test assertions.
- Record compilation and steady-state sampling separately.
- Do not modify core sampler production files on this branch.

---

### Task 1: Upstream BM example fixture and R runner

**Files:**
- Create: `benchmarks/r_example/generate_case.R`
- Create: `benchmarks/r_example/run_bspcov.R`
- Create: `benchmarks/r_example/data/bm_example_x.csv`
- Create: `benchmarks/r_example/data/bm_example_truth.csv`
- Create: `benchmarks/r_example/data/bm_example_initial.csv`
- Test: `tests/test_benchmark_r_example_data.py`

**Interfaces:**
- Produces a deterministic data fixture and R summary/timing CSV with explicit metadata.
- The Python runner in Task 2 consumes the same fixture files and summary columns.

- [ ] **Step 1: Write the failing fixture contract test**

Assert literal shapes `(20, 5)`, symmetry and positive definiteness of the truth
and initial covariance matrices, centered columns, and a stable fixture checksum.

- [ ] **Step 2: Run the test to verify RED**

Run: `uv run pytest -q tests/test_benchmark_r_example_data.py`
Expected: FAIL because the fixture files do not exist.

- [ ] **Step 3: Implement the base-R fixture and runner**

Port the documented `bmspcov` example with a fixed seed. The runner must accept
burn-in, retained sample count, and output directory arguments; write posterior
mean, standard deviation, quantiles, RMSE, elapsed seconds, and session metadata.

- [ ] **Step 4: Verify the fixture contract**

Run: `uv run pytest -q tests/test_benchmark_r_example_data.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/r_example tests/test_benchmark_r_example_data.py
git commit -m "bench: add upstream R BM example runner"
```

### Task 2: JAX runner and common comparison schema

**Files:**
- Create: `benchmarks/r_example/run_pybspcov.py`
- Create: `benchmarks/r_example/compare_results.py`
- Create: `tests/test_benchmark_comparison.py`

**Interfaces:**
- Consumes `sample_bm_chain(...) -> BMChainResult` from the core branch.
- Produces JAX summaries matching the R columns and a comparison JSON containing posterior differences and timing categories.

- [ ] **Step 1: Write failing parser and acceptance tests**

Use small literal R/JAX summary fixtures. Assert that posterior differences
inside combined Monte Carlo tolerances pass, differences outside them fail, and
timing fields never affect the statistical verdict.

- [ ] **Step 2: Run the test to verify RED**

Run: `uv run pytest -q tests/test_benchmark_comparison.py`
Expected: FAIL because the comparison module does not exist.

- [ ] **Step 3: Implement the minimal runners**

Compile one JAX call, synchronize with `block_until_ready`, run warmed
repetitions, and write compile-plus-execution, steady-state, and end-to-end
seconds. Compute posterior summaries from retained covariance draws and compare
them without importing R from package code.

- [ ] **Step 4: Verify comparison behavior**

Run: `JAX_ENABLE_X64=1 uv run pytest -q tests/test_benchmark_comparison.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/r_example tests/test_benchmark_comparison.py
git commit -m "bench: add JAX BM parity and timing runner"
```

### Task 3: Reproducible smoke run and report

**Files:**
- Create: `benchmarks/r_example/README.md`
- Create: `benchmarks/r_example/results/.gitkeep`
- Modify: `.gitignore`

**Interfaces:**
- Documents exact R and Python commands and the generated result schema.
- Keeps generated raw results untracked except explicitly curated baseline files.

- [ ] **Step 1: Run short R and JAX smoke benchmarks**

Use `burnin = 10` and `n_samples = 20` only to verify command execution and
schema creation. Do not present these short-run values as performance evidence.

- [ ] **Step 2: Run focused and full verification**

Run: `JAX_ENABLE_X64=1 uv run pytest -q tests/test_benchmark_r_example_data.py tests/test_benchmark_comparison.py`
Expected: PASS.

Run: `uv run pre-commit run --all-files`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add .gitignore benchmarks/r_example
git commit -m "docs: document R example benchmark workflow"
```
