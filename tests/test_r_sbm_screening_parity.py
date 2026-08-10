import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pybspcov.kernels import screening
from pybspcov.kernels.screening import (
    correlation_screening_mask,
    fnr_screening_mask,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"


def _load_csv(name: str) -> np.ndarray:
    return np.loadtxt(
        FIXTURE_DIR / name,
        delimiter=",",
        converters=lambda value: np.nan if value == "NA" else float(value),
        dtype=np.float64,
        ndmin=2,
    )


def _load_metadata() -> dict[str, Any]:
    with (FIXTURE_DIR / "sbm_screening_metadata.json").open() as stream:
        return json.load(stream)


def test_fnr_screening_matches_bspcov_1_0_3_eager_and_jit() -> None:
    assert jax.config.x64_enabled
    metadata = _load_metadata()
    scores = jnp.asarray(_load_csv("sbm_screening_pairwise_bf.csv"), dtype=jnp.float64)
    cutoff = jnp.asarray(metadata["fnr"]["cutoff"], dtype=jnp.float64)
    expected = _load_csv("sbm_screening_fnr_active_mask.csv").astype(bool)

    eager_mask = fnr_screening_mask(scores, cutoff)
    compiled_mask = jax.jit(screening._fnr_screening_mask_unchecked)(scores, cutoff)

    np.testing.assert_array_equal(np.asarray(eager_mask), expected)
    np.testing.assert_array_equal(np.asarray(compiled_mask), expected)


def test_correlation_screening_matches_bspcov_1_0_3_eager_and_jit() -> None:
    assert jax.config.x64_enabled
    metadata = _load_metadata()
    x = jnp.asarray(_load_csv("sbm_screening_x.csv"), dtype=jnp.float64)
    retained_fraction = jnp.asarray(metadata["corr"]["thr"], dtype=jnp.float64)
    expected = _load_csv("sbm_screening_corr_active_mask.csv").astype(bool)

    eager_mask = correlation_screening_mask(x, retained_fraction)
    compiled_mask = jax.jit(screening._correlation_screening_mask_unchecked)(
        x, retained_fraction
    )

    np.testing.assert_array_equal(np.asarray(eager_mask), expected)
    np.testing.assert_array_equal(np.asarray(compiled_mask), expected)


def test_correlation_screening_uses_non_degenerate_r_type_7_interpolation() -> None:
    assert jax.config.x64_enabled
    x = jnp.asarray(_load_csv("sbm_screening_x.csv"), dtype=jnp.float64)
    correlations = _load_csv("sbm_screening_correlations.csv")
    lower_absolute = np.abs(correlations[np.tril_indices(4, k=-1)])
    retained_fraction = jnp.asarray(0.3, dtype=jnp.float64)
    expected_type_7_cutoff = 0.37125
    expected = np.array(
        [
            [False, True, False, True],
            [True, False, False, False],
            [False, False, False, False],
            [True, False, False, False],
        ]
    )

    cutoff = np.quantile(lower_absolute, 0.7, method="linear")
    eager_mask = correlation_screening_mask(x, retained_fraction)
    compiled_mask = jax.jit(screening._correlation_screening_mask_unchecked)(
        x, retained_fraction
    )

    np.testing.assert_allclose(cutoff, expected_type_7_cutoff, rtol=0.0, atol=1e-15)
    assert not np.any(np.isclose(lower_absolute, cutoff, rtol=0.0, atol=1e-15))
    np.testing.assert_array_equal(np.asarray(eager_mask), expected)
    np.testing.assert_array_equal(np.asarray(compiled_mask), expected)
