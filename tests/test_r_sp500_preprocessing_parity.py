from pathlib import Path

import numpy as np

from pybspcov import preprocess_sp500

FIXTURES = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"


def _case() -> np.ndarray:
    return np.asarray(
        [
            ("AAA", "2020-01-01", 10.0, "Tech"),
            ("AAA", "2020-02-01", 11.0, "Tech"),
            ("AAA", "2020-03-01", 12.1, "Tech"),
            ("BBB", "2020-01-01", 20.0, "Energy"),
            ("BBB", "2020-02-01", 18.0, "Energy"),
            ("BBB", "2020-03-01", 19.8, "Energy"),
        ],
        dtype=[
            ("symbol", "U3"),
            ("date", "datetime64[D]"),
            ("adjusted", "f8"),
            ("sector", "U6"),
        ],
    )


def _matrix(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURES / name, delimiter=",", skiprows=1, ndmin=2)


def test_fixed_factor_sp500_preprocessing_matches_independent_r_fixture() -> None:
    processed = preprocess_sp500(_case(), sectors=["Tech", "Energy"], n_factors=1)

    np.testing.assert_allclose(
        processed.returns, _matrix("sp500_fixed_returns.csv"), atol=1e-14
    )
    np.testing.assert_allclose(
        processed.factorparthat, _matrix("sp500_fixed_factor.csv"), atol=1e-14
    )
    np.testing.assert_allclose(
        processed.Uhat, _matrix("sp500_fixed_residuals.csv"), atol=1e-14
    )
    assert processed.symbols.tolist() == ["BBB", "AAA"]
    assert processed.sectornames.tolist() == ["Energy", "Tech"]
