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

Run the end-to-end package example:

```bash
uv run python examples/quickstart.py
```

To test the distributable package instead of the editable checkout:

```bash
uv run python -m build
python -m pip install dist/pybspcov-*.whl
```

## NVIDIA GPU

Install the project's CUDA 12 optional dependency and verify the selected
device:

```bash
uv sync --extra cuda12
uv run python -c "import jax; print(jax.devices())"
```

CUDA-enabled JAX wheels depend on the operating system, GPU, driver, and CUDA
generation. Confirm compatibility against the current
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
