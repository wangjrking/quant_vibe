from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TEXT_COLUMNS = {"stock_code", "trade_date", "name", "industry", "act_ent_type"}


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame["trade_date"] = frame["trade_date"].astype(str)
    for col in frame.columns:
        if col in TEXT_COLUMNS:
            continue
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def patch_factor_columns(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame.sort_values(["stock_code", "trade_date"], inplace=True)
    grouped = frame.groupby("stock_code", sort=False)

    frame["post6_close"] = grouped["close"].shift(-6)
    frame["6d_yield_rate"] = (frame["post6_close"] - frame["close"]) / frame["close"]
    frame["open_yield_rate"] = (frame["post_open"] - frame["close"]) / frame["close"]
    frame["next_open_yield_rate"] = frame["open_yield_rate"]
    frame["midpoint"] = grouped["close"].transform(lambda x: (x.rolling(14).max() + x.rolling(14).min()) / 2)
    frame["midprice"] = (
        grouped["high"].transform(lambda x: x.rolling(14).max())
        + grouped["low"].transform(lambda x: x.rolling(14).min())
    ) / 2
    frame = _normalize(frame)
    frame.to_parquet(path, index=False)
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Patch deterministic formula columns in stock_factor_data.parquet.")
    parser.add_argument("--path", default="data_file/stock_factor_data.parquet")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame = patch_factor_columns(args.path)
    print(f"patch_done rows={frame.shape[0]} max={frame['trade_date'].max()} path={Path(args.path)}", flush=True)


if __name__ == "__main__":
    main()
