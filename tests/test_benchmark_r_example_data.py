import hashlib
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parents[1] / "benchmarks" / "r_example" / "data"
FIXTURE_NAMES = (
    "bm_example_x.csv",
    "bm_example_truth.csv",
    "bm_example_initial.csv",
)
EXPECTED_FIXTURE_SHA256 = (
    "1da3b680d62bb9bd1b5d9fbc26309ce4be7c5e58f51d3d4b84ce76d132dea22d"
)


def test_upstream_bm_example_fixture_contract() -> None:
    fixture_paths = [DATA_DIR / name for name in FIXTURE_NAMES]
    missing = [str(path) for path in fixture_paths if not path.is_file()]
    assert not missing, f"missing benchmark fixtures: {missing}"

    x, truth, initial = (
        np.loadtxt(path, delimiter=",", dtype=np.float64) for path in fixture_paths
    )
    assert x.shape == (20, 5)
    assert truth.shape == (5, 5)
    assert initial.shape == (5, 5)
    np.testing.assert_allclose(x.mean(axis=0), np.zeros(5), rtol=0.0, atol=1e-14)

    for covariance in (truth, initial):
        np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1e-14)
        assert np.linalg.eigvalsh(covariance).min() > 0.0

    digest = hashlib.sha256()
    for path in fixture_paths:
        digest.update(path.name.encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    assert digest.hexdigest() == EXPECTED_FIXTURE_SHA256
