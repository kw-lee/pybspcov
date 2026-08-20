"""Bundled bspcov 1.0.3 example datasets and preprocessing helpers."""

from __future__ import annotations

from hashlib import sha256
from importlib import import_module, resources
from typing import Any

import numpy as np
import numpy.typing as npt


class DatasetBunch(dict[str, Any]):
    """Dictionary whose values are also available as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _resource_path(filename: str) -> resources.abc.Traversable:
    return resources.files("pybspcov").joinpath("_data", filename)


def _description(filename: str) -> str:
    return _resource_path(filename).read_text(encoding="utf-8")


def _checksum(resource: resources.abc.Traversable) -> str:
    return sha256(resource.read_bytes()).hexdigest()


def _pandas() -> Any:
    try:
        return import_module("pandas")
    except ModuleNotFoundError as error:
        if error.name != "pandas":
            raise
        raise ImportError(
            "DataFrame output requires pandas; install pybspcov[data]"
        ) from None


def load_colon(
    *,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> DatasetBunch | tuple[Any, Any]:
    """Load the colon expression data in samples-by-genes orientation."""
    resource = _resource_path("colon.npz")
    with resources.as_file(resource) as path, np.load(path, allow_pickle=False) as raw:
        data = raw["data"].copy()
        target = raw["target"].copy()
        feature_names = raw["feature_names"].astype(str).tolist()
        sample_names = raw["sample_names"].astype(str).tolist()
    frame = None
    if as_frame:
        pandas = _pandas()
        data = pandas.DataFrame(data, columns=feature_names, index=sample_names)
        target = pandas.Series(target, name="tissue", index=sample_names)
        frame = data.assign(tissue=target)
    if return_X_y:
        return data, target
    return DatasetBunch(
        data=data,
        target=target,
        frame=frame,
        feature_names=feature_names,
        target_names=["tumor", "normal"],
        sample_names=sample_names,
        DESCR=_description("colon.rst"),
        source="R bspcov data/colon.rda and data/tissues.rda",
        version="bspcov 1.0.3",
        sha256=_checksum(resource),
    )


def load_sp500(*, as_frame: bool = False) -> DatasetBunch:
    """Load the S&P 500 adjusted-price data distributed by bspcov."""
    resource = _resource_path("sp500.npz")
    with resources.as_file(resource) as path, np.load(path, allow_pickle=False) as raw:
        symbol = raw["symbol"].astype(str)
        date = raw["date"].astype("datetime64[D]")
        adjusted = raw["adjusted"].astype(np.float64)
        sector = raw["sector"].astype(str)
    dtype = np.dtype(
        [
            ("symbol", f"U{max(map(len, symbol))}"),
            ("date", "datetime64[D]"),
            ("adjusted", "f8"),
            ("sector", f"U{max(map(len, sector))}"),
        ]
    )
    records = np.empty(symbol.shape[0], dtype=dtype)
    records["symbol"] = symbol
    records["date"] = date
    records["adjusted"] = adjusted
    records["sector"] = sector
    frame = None
    data: Any = records
    if as_frame:
        pandas = _pandas()
        frame = pandas.DataFrame.from_records(records)
        data = frame
    return DatasetBunch(
        data=data,
        frame=frame,
        feature_names=["symbol", "date", "adjusted", "sector"],
        DESCR=_description("sp500.rst"),
        source="R bspcov data/SP500.rda",
        version="bspcov 1.0.3",
        sha256=_checksum(resource),
    )


def preprocess_colon(
    colon: npt.ArrayLike,
    tissues: npt.ArrayLike,
    *,
    n_features: int = 50,
) -> DatasetBunch:
    """Log-transform and select genes by the absolute Welch statistic."""
    data = np.asarray(colon, dtype=np.float64)
    target = np.asarray(tissues)
    if data.ndim != 2 or target.ndim != 1 or data.shape[0] != target.shape[0]:
        raise ValueError("colon must be samples by genes and match tissues")
    if np.any(~np.isfinite(data)) or np.any(data <= 0):
        raise ValueError("colon must contain finite positive expression values")
    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise TypeError("n_features must be an integer")
    if not 1 <= n_features <= data.shape[1]:
        raise ValueError("n_features must be between 1 and the gene count")
    normal_idx = np.flatnonzero(target > 0)
    tumor_idx = np.flatnonzero(target < 0)
    if normal_idx.size < 2 or tumor_idx.size < 2:
        raise ValueError("tissues must identify at least two samples per group")
    transformed = np.log10(data)
    normal = transformed[normal_idx]
    tumor = transformed[tumor_idx]
    difference = normal.mean(axis=0) - tumor.mean(axis=0)
    standard_error = np.sqrt(
        normal.var(axis=0, ddof=1) / normal.shape[0]
        + tumor.var(axis=0, ddof=1) / tumor.shape[0]
    )
    statistic = np.abs(np.divide(difference, standard_error))
    selected = np.argsort(-statistic, kind="stable")[:n_features]
    return DatasetBunch(
        X=transformed[:, selected],
        normal_idx=normal_idx,
        tumor_idx=tumor_idx,
        group=np.where(target > 0, 1, 2),
        selected_features=selected,
        statistic=statistic[selected],
    )


def _records(data: Any) -> np.ndarray[Any, Any]:
    if isinstance(data, DatasetBunch):
        data = data.data
    if hasattr(data, "to_records"):
        data = data.to_records(index=False)
    records = np.asarray(data)
    expected = {"symbol", "date", "adjusted", "sector"}
    if records.dtype.names is None or set(records.dtype.names) != expected:
        raise ValueError(f"SP500 data must contain fields {sorted(expected)}")
    return records


def preprocess_sp500(
    data: Any,
    sectors: list[str] | tuple[str, ...],
    *,
    n_factors: int | None = None,
) -> DatasetBunch:
    """Compute aligned monthly returns and POET-style factor residuals."""
    records = _records(data)
    requested = set(sectors)
    selected = records[np.isin(records["sector"], list(requested))]
    if selected.size == 0:
        raise ValueError("sectors did not select any rows")
    symbols = np.unique(selected["symbol"])
    returns_by_symbol: dict[str, tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]] = {}
    sector_by_symbol: dict[str, str] = {}
    common_months: set[np.datetime64] | None = None
    for symbol_value in symbols:
        rows = selected[selected["symbol"] == symbol_value]
        order = np.argsort(rows["date"], kind="stable")
        rows = rows[order]
        months = rows["date"].astype("datetime64[M]")
        _, reverse_indices = np.unique(months[::-1], return_index=True)
        last_indices = np.sort(rows.size - 1 - reverse_indices)
        monthly_dates = months[last_indices]
        prices = rows["adjusted"][last_indices].astype(np.float64)
        monthly_returns = prices[1:] / prices[:-1] - 1.0
        return_months = monthly_dates[1:]
        symbol = str(symbol_value)
        returns_by_symbol[symbol] = (return_months, monthly_returns)
        sector_by_symbol[symbol] = str(rows["sector"][0])
        month_set = set(return_months.tolist())
        common_months = month_set if common_months is None else common_months & month_set
    if not common_months:
        raise ValueError("selected symbols have no common monthly returns")
    aligned_months = np.asarray(sorted(common_months), dtype="datetime64[M]")
    matrix_columns: list[npt.NDArray[np.float64]] = []
    kept_symbols: list[str] = []
    for symbol in symbols.astype(str):
        months, values = returns_by_symbol[symbol]
        lookup = {month: value for month, value in zip(months.tolist(), values)}
        column = np.asarray([lookup[month] for month in aligned_months.tolist()])
        if np.all(np.isfinite(column)):
            matrix_columns.append(column)
            kept_symbols.append(symbol)
    returns = np.column_stack(matrix_columns)
    returns = returns - returns.mean(axis=0, keepdims=True)
    maximum = min(returns.shape)
    if n_factors is None:
        singular_values = np.linalg.svd(returns, compute_uv=False)
        if singular_values.size < 2:
            factor_count = 0
        else:
            upper = min(maximum - 1, max(1, int(np.sqrt(maximum))))
            ratios = singular_values[:upper] / np.maximum(
                singular_values[1 : upper + 1], np.finfo(float).eps
            )
            factor_count = int(np.argmax(ratios) + 1)
    else:
        if isinstance(n_factors, bool) or not isinstance(n_factors, int):
            raise TypeError("n_factors must be an integer or None")
        if not 0 <= n_factors <= maximum:
            raise ValueError("n_factors must be between zero and min(n, p)")
        factor_count = n_factors
    if factor_count:
        left, singular_values, right = np.linalg.svd(returns, full_matrices=False)
        factor_part = (left[:, :factor_count] * singular_values[:factor_count]) @ right[
            :factor_count
        ]
    else:
        factor_part = np.zeros_like(returns)
    return DatasetBunch(
        Uhat=returns - factor_part,
        Khat=factor_count,
        factorparthat=factor_part,
        sectornames=np.asarray([sector_by_symbol[symbol] for symbol in kept_symbols]),
        symbols=np.asarray(kept_symbols),
        months=aligned_months,
        returns=returns,
    )
