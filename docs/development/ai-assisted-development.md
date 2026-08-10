# AI-Assisted Development

AI output is an unreviewed suggestion. A human contributor remains responsible
for correctness, security, scientific claims, provenance, and licensing.

## Required controls

- Disclose material AI assistance in the pull request without committing raw
  prompts or chat transcripts.
- Keep credentials, private source, patient data, unpublished research data,
  and confidential third-party material out of model inputs.
- Verify changing APIs and installation guidance against primary sources.
- Use independent R fixtures, invariants, and review; do not accept tests
  generated from the implementation logic as parity evidence.
- Inspect generated dependencies and code for provenance and GPL compatibility.
- Report all benchmark context, including hardware, software, dtype, shapes,
  warmup, repetitions, compilation treatment, and failures.
- Treat external text consumed by an agent as potentially adversarial.
- Give automation the minimum permissions; never provide release credentials.

The pull-request author owns the final diff even when an AI tool wrote most of
it. Reviewers may require regeneration or manual rewriting when provenance,
reasoning, or verification is unclear.

## Resource and delivery controls

- State one concrete deliverable and the evidence needed to accept it before
  starting an agent run.
- Cap repository audits at 15 minutes or 10 tool commands. Ask the maintainer
  before exceeding either limit.
- Treat a request to "continue" as permission to complete the current bounded
  deliverable, not as a persistent or open-ended goal.
- Do not use subagents for repeated audits, status summaries, edits to shared
  files, or multiple benchmarks competing for one GPU.
- Stop gathering evidence once it is sufficient to make the current decision.
- Produce implementation, a focused test, and a reviewable commit before adding
  optional benchmark infrastructure or extensive process documentation.
- Report progress in terms of public API behavior, correctness evidence,
  measured performance, and pull-request readiness. Test counts, coverage, and
  documentation volume are supporting evidence, not delivered functionality.
- Keep raw benchmark output outside Git unless it is a small reproducibility
  fixture. Commit only the script, metadata, and machine-readable summary needed
  to reproduce or review the claim.

## 2026-08 process failure and required response

An August 2026 development run consumed approximately 13 million model tokens
and 34.5 hours of recorded agent runtime. It left 31 linked worktrees using
7.3 GB, 69 cumulative local commits, and one remote 33-commit pull request whose
test job was failing. Although local BM/SBM implementations and useful GPU
measurements existed, the work was not delivered in reviewable form. One SBM R
benchmark-protocol commit added roughly 2,800 lines without completing the
public-estimator posterior golden fixture, and a further precision/GIG benchmark
track added roughly 2,000 lines while core matrix optimization remained pending.

This is a process failure regardless of test coverage or the amount of local
work. When the same warning signs appear, the required response is:

1. Identify the smallest branch containing useful implementation.
2. Separate required correctness evidence from optional experiments.
3. Repair the current pull request before starting another development branch.
4. Remove merged, superseded, or empty worktrees after maintainer approval.
5. Resume with one bounded deliverable and an explicit stop condition.
