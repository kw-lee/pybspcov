# Installation

## Development checkout

Create the locked CPU development environment from a repository checkout:

```bash
uv sync --all-groups
```

Verify the selected JAX backend:

```bash
uv run python -c "import jax; print(jax.devices())"
```

## NVIDIA GPU

CUDA-enabled JAX wheels depend on the operating system, GPU, driver, and CUDA
generation. Follow the current
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
and then rerun the device command above. The base lock is CPU-compatible and
does not promise a CUDA runtime.
