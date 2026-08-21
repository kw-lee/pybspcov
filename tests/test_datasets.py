import numpy as np

from pybspcov.datasets import (
    DatasetBunch,
    load_colon,
    load_sp500,
    preprocess_colon,
    preprocess_sp500,
)


def test_load_colon_returns_sklearn_style_bunch_and_tuple() -> None:
    dataset = load_colon()
    data, target = load_colon(return_X_y=True)

    assert isinstance(dataset, DatasetBunch)
    assert dataset.data.shape == (62, 2000)
    assert dataset.target.shape == (62,)
    assert dataset["data"] is dataset.data
    assert np.array_equal(data, dataset.data)
    assert np.array_equal(target, dataset.target)
    assert len(dataset.feature_names) == 2000
    assert dataset.version == "bspcov 1.0.3"
    assert len(dataset.sha256) == 64


def test_load_sp500_preserves_all_upstream_rows_and_columns() -> None:
    dataset = load_sp500()

    assert isinstance(dataset, DatasetBunch)
    assert dataset.data.shape == (1_345_751,)
    assert dataset.data.dtype.names == ("symbol", "date", "adjusted", "sector")
    assert dataset.feature_names == ["symbol", "date", "adjusted", "sector"]
    assert str(dataset.data[0]["symbol"]) == "AAPL"
    assert str(dataset.data[0]["date"]) == "2013-01-02"


def test_preprocess_colon_selects_top_50_genes_and_group_indices() -> None:
    raw = load_colon()
    processed = preprocess_colon(raw.data, raw.target)

    assert processed.X.shape == (62, 50)
    assert processed.normal_idx.shape == (22,)
    assert processed.tumor_idx.shape == (40,)
    assert np.array_equal(np.unique(processed.group), np.asarray([1, 2]))
    assert np.all(np.isfinite(processed.X))


def test_preprocess_sp500_returns_factor_residual_contract() -> None:
    records = np.asarray(
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

    processed = preprocess_sp500(records, sectors=["Tech", "Energy"], n_factors=1)

    assert processed.Uhat.shape == (3, 2)
    assert processed.factorparthat.shape == (3, 2)
    assert processed.Khat == 1
    assert processed.sectornames.tolist() == ["Energy", "Tech"]
    np.testing.assert_allclose(
        processed.returns,
        np.asarray([[0.0, -2.0 / 30.0], [-0.1, 1.0 / 30.0], [0.1, 1.0 / 30.0]]),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(processed.Uhat + processed.factorparthat, processed.returns)
