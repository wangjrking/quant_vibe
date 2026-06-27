from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a prediction asset on model-side ranking metrics.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction-dir", help="Directory containing fold_predictions/*.parquet or parquet files")
    source.add_argument("--prediction-db-path", help="SQLite DB path containing the merged prediction table")
    parser.add_argument("--prediction-table", help="SQLite table name when --prediction-db-path is used")
    parser.add_argument("--label-col", help="Label column name; auto-detected when omitted")
    parser.add_argument("--score-col", default="pred_prob")
    parser.add_argument("--top-k", default="5,10,20")
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    return parser.parse_args(argv)


def _read_parquet_dir(path: Path) -> pd.DataFrame:
    if (path / "fold_predictions").exists():
        files = sorted((path / "fold_predictions").glob("fold*.parquet"))
    else:
        files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet prediction files found under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _read_sqlite_table(db_path: Path, table: str) -> pd.DataFrame:
    if not table:
        raise ValueError("--prediction-table is required with --prediction-db-path")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"select * from '{table}'", conn)


def _auto_label_col(frame: pd.DataFrame, score_col: str) -> str:
    candidates = [col for col in frame.columns if col.startswith("executable_") and col.endswith("_return")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Multiple label columns detected, please specify --label-col: {candidates}")
    for fallback in ("label", "target", "y_true"):
        if fallback in frame.columns and fallback != score_col:
            return fallback
    raise ValueError("Unable to auto-detect label column")


def _to_float(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _split_top_k(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(int(piece))
    if not values:
        raise ValueError("--top-k produced an empty list")
    return values


def _apply_date_filter(frame: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    if date_from in (None, "") and date_to in (None, ""):
        return frame
    if "trade_date" not in frame.columns:
        raise ValueError("Prediction asset must contain trade_date for date filtering")
    filtered = frame.copy()
    filtered["trade_date"] = filtered["trade_date"].astype(str)
    if date_from not in (None, ""):
        filtered = filtered[filtered["trade_date"] >= str(date_from)]
    if date_to not in (None, ""):
        filtered = filtered[filtered["trade_date"] <= str(date_to)]
    return filtered


def evaluate_frame(
    frame: pd.DataFrame,
    *,
    score_col: str,
    label_col: str,
    top_k: Iterable[int],
    quantiles: int,
) -> tuple[dict, pd.DataFrame]:
    if "trade_date" not in frame.columns or "stock_code" not in frame.columns:
        raise ValueError("Prediction asset must contain trade_date and stock_code columns")
    if score_col not in frame.columns:
        raise ValueError(f"Score column {score_col!r} not found")
    if label_col not in frame.columns:
        raise ValueError(f"Label column {label_col!r} not found")

    data = frame.copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data = data[["trade_date", "stock_code", score_col, label_col]].copy()
    total_rows = int(len(data))
    data = data.dropna(subset=[score_col, label_col])
    valid_rows = int(len(data))
    valid_ratio = valid_rows / total_rows if total_rows else 0.0

    top_k = list(sorted(set(int(k) for k in top_k)))
    daily_rows: list[dict] = []
    quantile_labels = [str(i) for i in range(1, quantiles + 1)]

    for trade_date, group in data.groupby("trade_date", sort=True):
        if len(group) < 2:
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        pearson = ordered[score_col].corr(ordered[label_col], method="pearson")
        rank_ic = ordered[score_col].corr(ordered[label_col], method="spearman")
        row = {
            "trade_date": trade_date,
            "rows": int(len(ordered)),
            "pearson_ic": _to_float(pearson),
            "rank_ic": _to_float(rank_ic),
        }
        qbins = pd.qcut(
            ordered[score_col].rank(method="first"),
            q=quantiles,
            labels=quantile_labels,
            duplicates="drop",
        )
        ordered = ordered.assign(_quantile=qbins.astype(str))
        q_means = ordered.groupby("_quantile", sort=True)[label_col].mean().to_dict()
        for q in quantile_labels:
            row[f"q{q}_mean"] = _to_float(q_means.get(q))
        top_bucket = row.get(f"q{quantiles}_mean")
        bottom_bucket = row.get("q1_mean")
        if top_bucket is not None and bottom_bucket is not None:
            row["top_minus_bottom"] = top_bucket - bottom_bucket
        else:
            row["top_minus_bottom"] = None
        for k in top_k:
            row[f"top{k}_mean"] = _to_float(ordered.head(k)[label_col].mean()) if len(ordered) >= k else None
        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise ValueError("No valid daily groups available for evaluation")

    summary = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "valid_ratio": valid_ratio,
        "date_min": str(data["trade_date"].min()) if valid_rows else None,
        "date_max": str(data["trade_date"].max()) if valid_rows else None,
        "trade_days": int(daily["trade_date"].nunique()),
        "daily_pearson_ic_mean": _to_float(daily["pearson_ic"].mean()),
        "daily_pearson_ic_std": _to_float(daily["pearson_ic"].std(ddof=0)),
        "daily_rank_ic_mean": _to_float(daily["rank_ic"].mean()),
        "daily_rank_ic_std": _to_float(daily["rank_ic"].std(ddof=0)),
        "rank_ic_positive_ratio": _to_float((daily["rank_ic"] > 0).mean()),
        "top_decile_mean_return": _to_float(daily[f"q{quantiles}_mean"].mean()),
        "bottom_decile_mean_return": _to_float(daily["q1_mean"].mean()),
        "top_minus_bottom_mean": _to_float(daily["top_minus_bottom"].mean()),
        "top_minus_bottom_positive_ratio": _to_float((daily["top_minus_bottom"] > 0).mean()),
        "q1_to_q10_means": {
            str(i): _to_float(daily[f"q{i}_mean"].mean()) for i in range(1, quantiles + 1)
        },
        "top_k_mean_returns": {
            str(k): _to_float(daily[f"top{k}_mean"].mean()) for k in top_k
        },
    }
    return summary, daily


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.prediction_dir:
        frame = _read_parquet_dir(Path(args.prediction_dir))
        source_desc = str(Path(args.prediction_dir).resolve())
    else:
        frame = _read_sqlite_table(Path(args.prediction_db_path), args.prediction_table)
        source_desc = f"{Path(args.prediction_db_path).resolve()}::{args.prediction_table}"
    frame = _apply_date_filter(frame, args.date_from, args.date_to)
    label_col = args.label_col or _auto_label_col(frame, args.score_col)
    summary, daily = evaluate_frame(
        frame,
        score_col=args.score_col,
        label_col=label_col,
        top_k=_split_top_k(args.top_k),
        quantiles=args.quantiles,
    )
    payload = {
        "source": source_desc,
        "score_col": args.score_col,
        "label_col": label_col,
        "date_filter_from": args.date_from,
        "date_filter_to": args.date_to,
        **summary,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out, index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
