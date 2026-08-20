from pathlib import Path

import numpy as np

from pybspcov.datasets import load_colon, preprocess_colon


def test_colon_preprocessing_matches_bspcov_1_0_3() -> None:
    reference = np.loadtxt(
        Path(__file__).parent
        / "fixtures"
        / "r"
        / "bspcov-1.0.3"
        / "colon_preprocessed.csv",
        delimiter=",",
        skiprows=1,
    )
    raw = load_colon()
    actual = preprocess_colon(raw.data, raw.target).X

    np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)
