# GPU chain-vmap preliminary artifact

This is a versioned, preliminary artifact for the selectable GPU `vmap` mode.
It does not replace or modify `benchmarks/baselines/0.1.0.dev0/`.

## Scope

Each measurement runs independent MCMC chains on the same generated `X` and
the same estimator configuration. It is not a benchmark of unrelated datasets
or a claim that unrelated datasets are batched. `vmap` means XLA vectorization
over independent chain keys and states inside one estimator fit. It is not
Python multithreading.

The benchmark excludes compilation from primary timings by performing one
synchronized warm-up fit. Every timed fit synchronizes public outputs before
the wall clock is stopped. JSONL results record the chain count, total batch
wall time, normalized wall seconds per chain, chains per second, accepted and
rejected sweeps, finite/symmetric/SPD posterior-mean checks, truth-relative
error, fixture SHA-256, seed/PRNG policy, environment, and git provenance.
Normalized wall seconds per chain is throughput normalization, not the latency
of any individual chain.

## Maintainer-supplied preliminary result

The following float32 data is host-specific and preliminary. It was supplied
for an RTX 3090 host running JAX 0.11.1. It is not a multi-host result and it
does not contain float64 measurements or invented repetitions.

### Measured configuration supplied with the table

- Estimator: BM.
- Dimension (`p`): 200.
- Observations (`n`): 600.
- Burn-in sweeps: 50.
- Retained samples: 50.
- Fixture target density: not supplied (BM has no screening-density parameter).
- Dtype: float32.
- Device: NVIDIA GeForce RTX 3090.
- JAX/JAXLIB: 0.11.1.
- Compilation handling: excluded by one synchronized warm-up.
- Timed-output synchronization: public outputs synchronized with
  `block_until_ready`.
- Compared modes: sequential 1-chain baseline and vmap chain counts 2, 4,
  and 8.
- Not supplied: seed; repetition count; fixture hash or raw JSONL; NVIDIA
  driver or CUDA versions; XLA memory preallocation or allocator policy.

These settings and unknowns bind only the supplied preliminary table. The
later reproduction matrix is a proposed future experiment, not missing
metadata inferred for these measurements.

| Mode | Chains | Total batch wall time | Normalized wall per chain | Throughput relative to sequential 1-chain | Peak memory delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| sequential | 1 | 81.1648 s | 81.1648 s/chain | 1.0000x | about 452 MiB |
| vmap | 2 | 156.1529 s | 78.0765 s/chain | 1.0396x | not supplied |
| vmap | 4 | 178.5941 s | 44.6485 s/chain | 1.8180x | about 588 MiB |
| vmap | 8 | 181.8813 s | 22.7352 s/chain | 3.5700x | about 586 MiB |

The observed 8-chain total batch latency was about 72.0 percent below eight
sequential fits. Two chains are near parity, while four and eight chains show
improvement consistent with saturation and fixed-cost amortization. These
figures are not evidence of linear scaling. No float64 recommendation is
justified without float64 measurements.

## Reproduction matrix

Run the full matrix later on the target GPU. This intentionally has not been
launched for this task because it is hour-scale. It measures both execution
modes for float32 and float64, chain counts 1, 2, 4, and 8, with three timed
repetitions per cell.

```sh
for dtype in float32 float64; do
  if [ "$dtype" = float64 ]; then export JAX_ENABLE_X64=1; else export JAX_ENABLE_X64=0; fi
  for execution_mode in sequential vmap; do
    for chains in 1 2 4 8; do
      JAX_PLATFORMS=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run --frozen --extra cuda12 python benchmarks/sbm_public_scaling.py \
        --estimator bm --device gpu --dtype "$dtype" --dimensions 200 --n-factor 3 \
        --density 0.05 --burnin 50 --samples 50 --chains "$chains" \
        --repetitions 3 --seed 20260803 --execution-mode "$execution_mode" \
        --output "benchmarks/baselines/0.2.0.dev0/raw/p200-${dtype}-${execution_mode}-${chains}.jsonl"
    done
  done
done
```

Retain the output JSONL files and record GPU model, driver, CUDA/JAX versions,
host load, and memory configuration with the result. Compare only runs with
matching fixture hashes and seeds. The benchmark derives one warm-up key and
then splits independent repetition and chain keys from the requested seed.

## Limitations

Memory demand can limit useful chain counts and can produce OOM failures.
XLA preallocation, allocator settings, compilation cache state, device cache
state, concurrent host/device load, and asynchronous dispatch can affect
results. Synchronization protects the reported public-fit timing boundary but
does not remove all system-level variability. Re-run after cache warm-up and
with an explicit memory policy before drawing deployment conclusions.
