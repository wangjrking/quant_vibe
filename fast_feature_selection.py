from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd

from feature_selection_module import prepare_selection_label
from leakage_guard import find_leaky_features


def _date_filter(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    dates = frame["trade_date"].astype(str)
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= dates >= start
    if end:
        mask &= dates <= end
    return frame.loc[mask]


def _candidate_features(frame: pd.DataFrame, label: str, max_missing_ratio: float) -> list[str]:
    excluded = {"stock_code", "trade_date", "name", "industry", "act_ent_type", label}
    candidates = [col for col in frame.columns if col not in excluded]
    leaky = set(find_leaky_features(candidates, label=label))
    result = []
    for feature in candidates:
        if feature in leaky:
            continue
        if pd.api.types.is_numeric_dtype(frame[feature]) and frame[feature].notna().mean() >= 1.0 - max_missing_ratio:
            result.append(feature)
    return result


def _fold_indices(dates: pd.Series, folds: int) -> list[pd.Series]:
    unique_dates = sorted(dates.astype(str).unique())
    if not unique_dates:
        return []
    fold_size = max(len(unique_dates) // folds, 1)
    result = []
    for idx in range(folds):
        start = idx * fold_size
        end = len(unique_dates) if idx == folds - 1 else (idx + 1) * fold_size
        fold_dates = set(unique_dates[start:end])
        if fold_dates:
            result.append(dates.astype(str).isin(fold_dates))
    return result


def score_features_fast(frame: pd.DataFrame, label: str, start: str | None, end: str | None, top_n: int, min_abs_ic: float, max_missing_ratio: float, folds: int) -> tuple[list[dict], list[str]]:
    frame = prepare_selection_label(frame.copy(), label)
    frame = _date_filter(frame, start, end)
    features = _candidate_features(frame, label, max_missing_ratio)
    label_values = pd.to_numeric(frame[label], errors="coerce")
    valid_label = label_values.notna()
    frame = frame.loc[valid_label]
    label_values = label_values.loc[valid_label]
    dates = frame["trade_date"]

    rows = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        fold_ics = []
        for mask in _fold_indices(dates, folds):
            x = values.loc[mask]
            y = label_values.loc[mask]
            valid = x.notna() & y.notna()
            if valid.sum() < 100 or x.loc[valid].nunique() < 2 or y.loc[valid].nunique() < 2:
                continue
            ic = x.loc[valid].rank().corr(y.loc[valid].rank())
            if ic is not None and not math.isnan(ic):
                fold_ics.append(float(ic))
        if not fold_ics:
            continue
        mean_ic = sum(fold_ics) / len(fold_ics)
        std_ic = pd.Series(fold_ics).std()
        ic_ir = 0.0 if not std_ic or math.isnan(std_ic) else mean_ic / std_ic
        rows.append(
            {
                "feature": feature,
                "mean_ic": mean_ic,
                "abs_mean_ic": abs(mean_ic),
                "ic_ir": ic_ir,
                "fold_count": len(fold_ics),
                "missing_ratio": float(values.isna().mean()),
            }
        )
    rows.sort(key=lambda row: (row["abs_mean_ic"], abs(row["ic_ir"]), -row["missing_ratio"]), reverse=True)
    selected = [row["feature"] for row in rows if row["abs_mean_ic"] >= min_abs_ic][:top_n]
    return rows, selected


def write_score_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fast time-fold IC feature selection.")
    parser.add_argument("--data", default="../data_file/stock_factor_data.parquet")
    parser.add_argument("--label", default="executable_10d_open_return")
    parser.add_argument("--top-n", type=int, default=160)
    parser.add_argument("--min-abs-ic", type=float, default=0.005)
    parser.add_argument("--max-missing-ratio", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", default="../data_file/selected_features_executable_10d_open_return.json")
    parser.add_argument("--score-output", default="../data_file/feature_ic_scores_executable_10d_open_return_fast.csv")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    frame = pd.read_parquet(args.data)
    rows, selected = score_features_fast(
        frame,
        label=args.label,
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        min_abs_ic=args.min_abs_ic,
        max_missing_ratio=args.max_missing_ratio,
        folds=args.folds,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"label": args.label, "features": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_score_csv(rows, Path(args.score_output))
    print(f"scored_features={len(rows)} selected_features={len(selected)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
