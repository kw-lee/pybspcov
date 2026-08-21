"""Build deterministic NumPy resources from exported bspcov 1.0.3 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    colon = pd.read_csv(args.export_directory / "colon.csv")
    tissues = pd.read_csv(args.export_directory / "tissues.csv")
    np.savez_compressed(
        args.output_directory / "colon.npz",
        data=colon.to_numpy(dtype=np.float64).T,
        target=tissues["tissues"].to_numpy(dtype=np.int32),
        feature_names=np.asarray(
            [f"gene_{index + 1}" for index in range(colon.shape[0])]
        ),
        sample_names=np.asarray(colon.columns.astype(str).tolist(), dtype=str),
    )

    sp500 = pd.read_csv(args.export_directory / "SP500.csv")
    dates = pd.to_datetime(sp500["date"]).to_numpy(dtype="datetime64[D]")
    np.savez_compressed(
        args.output_directory / "sp500.npz",
        symbol=sp500["symbol"].to_numpy(dtype=str),
        date=dates,
        adjusted=sp500["adjusted"].to_numpy(dtype=np.float64),
        sector=sp500["sector"].to_numpy(dtype=str),
    )


if __name__ == "__main__":
    main()
