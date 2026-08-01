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
