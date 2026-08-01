import json
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"


def _load_csv(name: str) -> np.ndarray:
    return np.loadtxt(
        FIXTURE_DIR / name,
        delimiter=",",
        converters=lambda value: np.nan if value == "NA" else float(value),
        ndmin=2,
    )


def _load_metadata() -> dict[str, Any]:
    with (FIXTURE_DIR / "sbm_screening_metadata.json").open() as stream:
        return json.load(stream)


def test_sbm_screening_fixture_has_expected_schema_and_provenance() -> None:
    metadata = _load_metadata()

    assert metadata["package"] == "bspcov"
    assert metadata["package_version"] == "1.0.3"
    assert metadata["source_tag"] == "1.0.3"
    assert metadata["source_commit"] == "165106c5ab8f6506e6d69b0b8f94ce5bdc99092f"
    assert metadata["index_base"] == 0
    assert metadata["n"] == 12
    assert metadata["p"] == 4
    assert metadata["mask_semantics"] == {
        "excluded": "true_for_upstream_INDzero",
        "active": "true_for_retained_off_diagonal_edge",
    }
    assert metadata["fnr"] == {
        "method": "FNR",
        "seed": 314159,
        "rho": 0.25,
        "FNR": 0.05,
        "nsimdata": 1000,
        "cutoff": metadata["fnr"]["cutoff"],
    }
    assert metadata["corr"] == {
        "method": "corr",
        "thr": 0.2,
        "quantile_probability": 0.8,
        "cutoff": metadata["corr"]["cutoff"],
    }

    x = _load_csv("sbm_screening_x.csv")
    bayes_factors = _load_csv("sbm_screening_pairwise_bf.csv")
    correlations = _load_csv("sbm_screening_correlations.csv")
    assert x.shape == (metadata["n"], metadata["p"])
    assert bayes_factors.shape == (metadata["p"], metadata["p"])
    assert correlations.shape == (metadata["p"], metadata["p"])
    np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(correlations, correlations.T, atol=1e-15)
    np.testing.assert_allclose(np.diag(correlations), 1.0, atol=1e-15)
    assert np.isnan(bayes_factors[np.triu_indices(metadata["p"])]).all()
    assert np.isfinite(bayes_factors[np.tril_indices(metadata["p"], k=-1)]).all()


def test_fnr_fixture_encodes_upstream_strict_bayes_factor_screening() -> None:
    metadata = _load_metadata()
    bayes_factors = _load_csv("sbm_screening_pairwise_bf.csv")
    excluded_mask = _load_csv("sbm_screening_fnr_excluded_mask.csv").astype(np.int64)
    active_mask = _load_csv("sbm_screening_fnr_active_mask.csv").astype(np.int64)
    expected_excluded_mask = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.int64,
    )

    lower_retained = bayes_factors > metadata["fnr"]["cutoff"]
    retained = np.tril(lower_retained, k=-1)
    retained |= retained.T

    np.testing.assert_array_equal(excluded_mask, expected_excluded_mask)
    np.testing.assert_array_equal(active_mask, retained)
    np.testing.assert_array_equal(excluded_mask, (~retained) & ~np.eye(4, dtype=bool))
    np.testing.assert_array_equal(excluded_mask, excluded_mask.T)
    np.testing.assert_array_equal(np.diag(excluded_mask), 0)


def test_correlation_fixture_encodes_upstream_quantile_screening() -> None:
    metadata = _load_metadata()
    correlations = _load_csv("sbm_screening_correlations.csv")
    excluded_mask = _load_csv("sbm_screening_corr_excluded_mask.csv").astype(np.int64)
    active_mask = _load_csv("sbm_screening_corr_active_mask.csv").astype(np.int64)
    expected_excluded_mask = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=np.int64,
    )

    lower = np.abs(correlations[np.tril_indices(4, k=-1)])
    cutoff = np.quantile(lower, metadata["corr"]["quantile_probability"])
    retained = np.abs(correlations) > cutoff
    np.fill_diagonal(retained, False)

    np.testing.assert_allclose(cutoff, metadata["corr"]["cutoff"], atol=1e-15)
    np.testing.assert_array_equal(excluded_mask, expected_excluded_mask)
    np.testing.assert_array_equal(active_mask, retained)
    np.testing.assert_array_equal(excluded_mask, (~retained) & ~np.eye(4, dtype=bool))


def test_screened_covariance_fixtures_zero_only_screened_edges() -> None:
    initial = _load_csv("sbm_screening_initial_covariance.csv")

    for method in ("fnr", "corr"):
        mask = _load_csv(f"sbm_screening_{method}_excluded_mask.csv").astype(bool)
        screened = _load_csv(f"sbm_screening_{method}_covariance.csv")
        expected = initial.copy()
        expected[mask] = 0.0

        np.testing.assert_array_equal(screened, expected)
        np.testing.assert_array_equal(np.diag(screened), np.diag(initial))
        np.testing.assert_array_equal(screened, screened.T)
