from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from feature_selection_module import prepare_selection_label
from leakage_guard import find_leaky_features
from model_asset_route import (
    MODEL_FEATURE_MODE_LEGACY,
    MODEL_FEATURE_MODE_SPLIT,
)


def _date_filter(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    dates = frame["trade_date"].astype(str)
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= dates >= start
    if end:
        mask &= dates <= end
    return frame.loc[mask]


def _candidate_features(
    frame: pd.DataFrame,
    label: str,
    max_missing_ratio: float,
    exclude_prefixes: tuple[str, ...] = (),
) -> list[str]:
    excluded = {"stock_code", "trade_date", "name", "industry", "act_ent_type", label}
    candidates = [col for col in frame.columns if col not in excluded]
    leaky = set(find_leaky_features(candidates, label=label))
    result = []
    for feature in candidates:
        if feature in leaky:
            continue
        if exclude_prefixes and str(feature).startswith(exclude_prefixes):
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


def score_features_fast(
    frame: pd.DataFrame,
    label: str,
    start: str | None,
    end: str | None,
    top_n: int,
    min_abs_ic: float,
    max_missing_ratio: float,
    folds: int,
    exclude_prefixes: tuple[str, ...] = (),
) -> tuple[list[dict], list[str]]:
    frame = prepare_selection_label(frame.copy(), label)
    frame = _date_filter(frame, start, end)
    features = _candidate_features(frame, label, max_missing_ratio, exclude_prefixes=exclude_prefixes)
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


def _label_required_columns(label: str) -> list[str]:
    if label == "risk_adjusted_10d_yield_rate":
        return ["10d_yield_rate", "atr_qfq", "close"]
    if label == "executable_10d_open_return":
        return ["post_open", "post12_open"]
    if label == "executable_5d_open_return":
        return ["post_open", "post6_open"]
    if label == "executable_3d_open_return":
        return ["post_open", "post4_open"]
    if label == "executable_1d_open_return":
        return ["post_open", "post2_open"]
    if label.startswith("executable_") and "_top" in label:
        if "_10d_" in label:
            return ["post_open", "post12_open"]
        if "_5d_" in label:
            return ["post_open", "post6_open"]
        if "_3d_" in label:
            return ["post_open", "post4_open"]
        if "_1d_" in label:
            return ["post_open", "post2_open"]
    if label == "excess_10d_yield_rate":
        return ["adjust_10d_yield_rate"]
    return [label]


def _split_label_required_columns(label: str) -> list[str]:
    if label == "risk_adjusted_10d_yield_rate":
        return ["10d_yield_rate"]
    if label == "excess_10d_yield_rate":
        return ["adjust_10d_yield_rate"]
    if label.startswith("executable_") and "_top" in label:
        if "_10d_" in label:
            return ["executable_10d_open_return"]
        if "_5d_" in label:
            return ["executable_5d_open_return"]
        if "_3d_" in label:
            return ["executable_3d_open_return"]
        if "_1d_" in label:
            return ["executable_1d_open_return"]
    return [label]


def _split_feature_required_columns(label: str) -> list[str]:
    if label == "risk_adjusted_10d_yield_rate":
        return ["atr_qfq", "close"]
    return []


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _stable_parquet_read_target(path: Path) -> Path | list[Path]:
    if not path.is_dir():
        return path
    parts = sorted(part for part in path.glob("*.parquet") if part.is_file())
    if not parts:
        raise FileNotFoundError(f"no stable parquet parts found under {path}")
    return parts


def _parquet_schema_names(path: Path) -> list[str]:
    return list(pq.ParquetDataset(_stable_parquet_read_target(path)).schema.names)


def _read_parquet_date_range(path: Path, columns: list[str], start: str | None, end: str | None) -> pd.DataFrame:
    target = _stable_parquet_read_target(path)
    filters = []
    if start:
        filters.append(("trade_date", ">=", str(start)))
    if end:
        filters.append(("trade_date", "<=", str(end)))
    if filters:
        return pd.read_parquet(target, columns=columns, filters=filters)
    return pd.read_parquet(target, columns=columns)


def score_features_fast_parquet(
    path: str | Path,
    label: str,
    start: str | None,
    end: str | None,
    top_n: int,
    min_abs_ic: float,
    max_missing_ratio: float,
    folds: int,
    exclude_prefixes: tuple[str, ...] = (),
    chunk_size: int = 40,
) -> tuple[list[dict], list[str]]:
    """Score features without loading the whole wide parquet into memory."""
    path = Path(path)
    schema_columns = _parquet_schema_names(path) if path.is_dir() else list(pq.ParquetFile(path).schema_arrow.names)
    base_columns = list(dict.fromkeys(["trade_date", "stock_code"] + _label_required_columns(label)))
    base_columns = [col for col in base_columns if col in schema_columns]
    if "trade_date" not in base_columns:
        raise ValueError("trade_date column is required for feature selection")
    base = _read_parquet_date_range(path, base_columns, start, end)
    base = prepare_selection_label(base, label)
    if label not in base.columns:
        raise ValueError(f"label column not found: {label}")
    label_values = pd.to_numeric(base[label], errors="coerce")
    valid_label = label_values.notna()
    base = base.loc[valid_label]
    label_values = label_values.loc[valid_label]
    dates = base["trade_date"]
    needed = set(base_columns) | {"name", "industry", "act_ent_type", label}
    candidates = [col for col in schema_columns if col not in needed]
    if exclude_prefixes:
        candidates = [col for col in candidates if not str(col).startswith(exclude_prefixes)]
    leaky = set(find_leaky_features(candidates, label=label))
    candidates = [col for col in candidates if col not in leaky]

    rows = []
    total_chunks = math.ceil(len(candidates) / chunk_size) if chunk_size > 0 else 0
    for chunk_idx, feature_chunk in enumerate(_chunks(candidates, chunk_size), start=1):
        print(
            f"feature_chunk {chunk_idx}/{total_chunks} size={len(feature_chunk)} "
            f"first={feature_chunk[0] if feature_chunk else ''} last={feature_chunk[-1] if feature_chunk else ''}",
            flush=True,
        )
        frame = _read_parquet_date_range(path, ["trade_date", *feature_chunk], start, end)
        frame = frame.loc[valid_label]
        for feature in feature_chunk:
            if feature not in frame.columns:
                continue
            if not pd.api.types.is_numeric_dtype(frame[feature]):
                values = pd.to_numeric(frame[feature], errors="coerce")
            else:
                values = frame[feature]
            missing_ratio = float(values.isna().mean())
            if missing_ratio > max_missing_ratio:
                continue
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
                    "missing_ratio": missing_ratio,
                }
            )
        del frame
        print(f"feature_chunk_done {chunk_idx}/{total_chunks} scored={len(rows)}", flush=True)
    rows.sort(key=lambda row: (row["abs_mean_ic"], abs(row["ic_ir"]), -row["missing_ratio"]), reverse=True)
    selected = [row["feature"] for row in rows if row["abs_mean_ic"] >= min_abs_ic][:top_n]
    return rows, selected


def score_features_fast_split(
    path: str | Path,
    *,
    label_path: str | Path,
    label: str,
    start: str | None,
    end: str | None,
    top_n: int,
    min_abs_ic: float,
    max_missing_ratio: float,
    folds: int,
    exclude_prefixes: tuple[str, ...] = (),
    chunk_size: int = 40,
) -> tuple[list[dict], list[str]]:
    feature_path = Path(path)
    label_path = Path(label_path)
    feature_schema = _parquet_schema_names(feature_path)
    label_schema = _parquet_schema_names(label_path)

    feature_base_columns = list(
        dict.fromkeys(["trade_date", "stock_code"] + _split_feature_required_columns(label))
    )
    feature_base_columns = [column for column in feature_base_columns if column in feature_schema]
    label_columns = list(
        dict.fromkeys(
            ["trade_date", "stock_code", "5d_yield_rate", "open6_yield_rate", "10d_yield_rate"]
            + _split_label_required_columns(label)
        )
    )
    label_columns = [column for column in label_columns if column in label_schema]
    base_features = _read_parquet_date_range(feature_path, feature_base_columns, start, end)
    base_labels = _read_parquet_date_range(label_path, label_columns, start, end)
    base = base_features.merge(base_labels, on=["trade_date", "stock_code"], how="inner")
    base = prepare_selection_label(base, label)
    if label not in base.columns:
        raise ValueError(f"label column not found after split merge: {label}")
    label_values = pd.to_numeric(base[label], errors="coerce")
    valid_label = label_values.notna()
    base = base.loc[valid_label].reset_index(drop=True)
    label_values = label_values.loc[valid_label].reset_index(drop=True)
    dates = base["trade_date"].reset_index(drop=True)

    needed = set(feature_base_columns) | {"name", "industry", "act_ent_type", label}
    candidates = [col for col in feature_schema if col not in needed]
    if exclude_prefixes:
        candidates = [col for col in candidates if not str(col).startswith(exclude_prefixes)]
    leaky = set(find_leaky_features(candidates, label=label))
    candidates = [col for col in candidates if col not in leaky]

    base_keys = base[["trade_date", "stock_code"]].reset_index(drop=True)
    rows = []
    total_chunks = math.ceil(len(candidates) / chunk_size) if chunk_size > 0 else 0
    for chunk_idx, feature_chunk in enumerate(_chunks(candidates, chunk_size), start=1):
        print(
            f"feature_chunk {chunk_idx}/{total_chunks} size={len(feature_chunk)} "
            f"first={feature_chunk[0] if feature_chunk else ''} last={feature_chunk[-1] if feature_chunk else ''}",
            flush=True,
        )
        frame = _read_parquet_date_range(feature_path, ["trade_date", "stock_code", *feature_chunk], start, end)
        frame = base_keys.merge(frame, on=["trade_date", "stock_code"], how="left").reset_index(drop=True)
        for feature in feature_chunk:
            if feature not in frame.columns:
                continue
            if not pd.api.types.is_numeric_dtype(frame[feature]):
                values = pd.to_numeric(frame[feature], errors="coerce")
            else:
                values = frame[feature]
            missing_ratio = float(values.isna().mean())
            if missing_ratio > max_missing_ratio:
                continue
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
                    "missing_ratio": missing_ratio,
                }
            )
        del frame
        print(f"feature_chunk_done {chunk_idx}/{total_chunks} scored={len(rows)}", flush=True)
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
    parser.add_argument("--data", default="data_file/production_factor_parts")
    parser.add_argument("--labels", default="data_file/prediction_label_parts")
    parser.add_argument(
        "--feature-source",
        default=MODEL_FEATURE_MODE_SPLIT,
        choices=[MODEL_FEATURE_MODE_SPLIT, MODEL_FEATURE_MODE_LEGACY],
    )
    parser.add_argument("--label", default="executable_10d_open_return")
    parser.add_argument("--top-n", type=int, default=160)
    parser.add_argument("--min-abs-ic", type=float, default=0.005)
    parser.add_argument("--max-missing-ratio", type=float, default=0.35)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--exclude-prefixes", default="")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", default="data_file/selected_features_executable_10d_open_return.json")
    parser.add_argument("--score-output", default="data_file/feature_ic_scores_executable_10d_open_return_fast.csv")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    exclude_prefixes = tuple(prefix.strip() for prefix in str(args.exclude_prefixes).split(",") if prefix.strip())
    if args.feature_source == MODEL_FEATURE_MODE_LEGACY:
        rows, selected = score_features_fast_parquet(
            args.data,
            label=args.label,
            start=args.start,
            end=args.end,
            top_n=args.top_n,
            min_abs_ic=args.min_abs_ic,
            max_missing_ratio=args.max_missing_ratio,
            folds=args.folds,
            exclude_prefixes=exclude_prefixes,
        )
    else:
        rows, selected = score_features_fast_split(
            args.data,
            label_path=args.labels,
            label=args.label,
            start=args.start,
            end=args.end,
            top_n=args.top_n,
            min_abs_ic=args.min_abs_ic,
            max_missing_ratio=args.max_missing_ratio,
            folds=args.folds,
            exclude_prefixes=exclude_prefixes,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"label": args.label, "features": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_score_csv(rows, Path(args.score_output))
    print(f"scored_features={len(rows)} selected_features={len(selected)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
