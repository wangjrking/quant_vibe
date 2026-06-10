from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from leakage_guard import find_leaky_features


KEY_COLUMNS = ["stock_code", "trade_date"]


def _read_ranked_features(score_path: Path, top_n: int) -> list[str]:
    with score_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    features = [row["feature"] for row in rows if row.get("feature")]
    leaky = set(find_leaky_features(features, label="10d_yield_rate"))
    selected = []
    for feature in features:
        if feature in leaky or feature in KEY_COLUMNS:
            continue
        selected.append(feature)
        if len(selected) >= top_n:
            break
    return selected


def add_legacy_columns(current_path: str | Path, legacy_path: str | Path, score_path: str | Path, top_n: int) -> list[str]:
    current_path = Path(current_path)
    legacy_path = Path(legacy_path)
    score_path = Path(score_path)
    features = _read_ranked_features(score_path, top_n)

    current = pd.read_parquet(current_path)
    current["trade_date"] = current["trade_date"].astype(str)
    existing_features = [feature for feature in features if feature in current.columns]
    legacy = pd.read_parquet(legacy_path, columns=KEY_COLUMNS + existing_features)
    legacy["trade_date"] = legacy["trade_date"].astype(str)
    rename_map = {feature: f"legacy_{feature}" for feature in existing_features}
    legacy.rename(columns=rename_map, inplace=True)

    legacy_cols = list(rename_map.values())
    current.drop(columns=[col for col in legacy_cols if col in current.columns], inplace=True)
    merged = current.merge(legacy, on=KEY_COLUMNS, how="left")
    for col in legacy_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.sort_values(KEY_COLUMNS, inplace=True)
    merged.to_parquet(current_path, index=False)
    return legacy_cols


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Add legacy pre-fix factor columns into current stock_factor_data.")
    parser.add_argument("--current", default="../data_file/stock_factor_data.parquet")
    parser.add_argument("--legacy", default="../data_file/stock_factor_data_before_formula_fix_20260606.parquet")
    parser.add_argument("--scores", default="../data_file/feature_ic_scores_10d_yield_rate_pre_2y_oos.csv")
    parser.add_argument("--top-n", type=int, default=160)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cols = add_legacy_columns(args.current, args.legacy, args.scores, args.top_n)
    print(f"legacy_columns_added={len(cols)}")
    print(",".join(cols[:30]))


if __name__ == "__main__":
    main()
