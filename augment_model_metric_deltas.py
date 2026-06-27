from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


def rank_ic_positive_ratio_by_window(
    daily_rank_ic: pd.DataFrame,
    windows: pd.DataFrame,
) -> list[dict]:
    daily = daily_rank_ic.copy()
    daily["trade_date"] = daily["trade_date"].astype(str)
    rows: list[dict] = []
    for _, window in windows.iterrows():
        window_name = str(window["window"])
        start = str(window["eval_date_min"])
        end = str(window["eval_date_max"])
        subset = daily[(daily["trade_date"] >= start) & (daily["trade_date"] <= end)]
        if subset.empty:
            rows.append(
                {
                    "window": window_name,
                    "candidate_rank_ic_positive_ratio": None,
                    "formal_rank_ic_positive_ratio": None,
                    "rank_ic_positive_ratio_delta": None,
                    "rank_ic_positive_ratio_days": 0,
                }
            )
            continue
        candidate_ratio = float((subset["candidate_rank_ic"] > 0).mean())
        formal_ratio = float((subset["formal_rank_ic"] > 0).mean())
        rows.append(
            {
                "window": window_name,
                "candidate_rank_ic_positive_ratio": candidate_ratio,
                "formal_rank_ic_positive_ratio": formal_ratio,
                "rank_ic_positive_ratio_delta": candidate_ratio - formal_ratio,
                "rank_ic_positive_ratio_days": int(len(subset)),
            }
        )
    return rows


def _read_prediction_table(
    conn: sqlite3.Connection,
    *,
    table: str,
    score_alias: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"""
        select trade_date, stock_code, pred_prob as {score_alias}
        from {table}
        where cast(trade_date as text) between ? and ?
        """,
        conn,
        params=(start, end),
    )
    frame["trade_date"] = frame["trade_date"].astype(str).str.strip()
    frame["stock_code"] = frame["stock_code"].astype(str).str.strip()
    return frame


def _read_label_parts(
    label_parts_dir: Path,
    *,
    label_cols: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    columns = ["trade_date", "stock_code", *sorted(set(label_cols))]
    frames: list[pd.DataFrame] = []
    for path in sorted(label_parts_dir.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path, columns=columns)
        except Exception:
            continue
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    labels = pd.concat(frames, ignore_index=True)
    labels["trade_date"] = labels["trade_date"].astype(str).str.strip()
    labels["stock_code"] = labels["stock_code"].astype(str).str.strip()
    return labels


def _daily_rank_ic(frame: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    data = frame.dropna(subset=["formal_score", "candidate_score", label_col]).copy()
    data[label_col] = pd.to_numeric(data[label_col], errors="coerce")
    data = data.dropna(subset=[label_col])
    rows: list[dict] = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        if len(group) < 2:
            continue
        rows.append(
            {
                "trade_date": str(trade_date),
                "formal_rank_ic": group["formal_score"].corr(group[label_col], method="spearman"),
                "candidate_rank_ic": group["candidate_score"].corr(group[label_col], method="spearman"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["trade_date", "formal_rank_ic", "candidate_rank_ic"])
    return pd.DataFrame(rows).dropna()


def augment_metrics(
    metrics: pd.DataFrame,
    *,
    snapshot: dict,
    db_path: Path,
    label_parts_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    augmented = metrics.copy()
    for column in (
        "candidate_rank_ic_positive_ratio",
        "formal_rank_ic_positive_ratio",
        "rank_ic_positive_ratio_delta",
        "rank_ic_positive_ratio_days",
    ):
        if column not in augmented.columns:
            augmented[column] = pd.NA

    all_daily: list[pd.DataFrame] = []
    choices = snapshot.get("choices", {})
    formal_baseline = snapshot.get("formal_baseline", {})
    labels = [item.get("label") for item in choices.values() if item.get("label")]
    min_date = str(metrics["eval_date_min"].min())
    max_date = str(metrics["eval_date_max"].max())
    label_frame = _read_label_parts(label_parts_dir, label_cols=labels, start=min_date, end=max_date)

    with sqlite3.connect(db_path) as conn:
        for horizon, choice in choices.items():
            horizon_key = str(horizon).lower()
            windows = metrics[metrics["horizon"].astype(str).str.lower() == horizon_key][
                ["window", "eval_date_min", "eval_date_max"]
            ]
            if windows.empty:
                continue
            label_col = choice.get("label")
            candidate_table = choice.get("table")
            formal_table = formal_baseline.get(horizon)
            if not label_col or not candidate_table or not formal_table:
                continue
            start = str(windows["eval_date_min"].min())
            end = str(windows["eval_date_max"].max())
            formal = _read_prediction_table(
                conn,
                table=formal_table,
                score_alias="formal_score",
                start=start,
                end=end,
            )
            candidate = _read_prediction_table(
                conn,
                table=candidate_table,
                score_alias="candidate_score",
                start=start,
                end=end,
            )
            horizon_labels = label_frame[["trade_date", "stock_code", label_col]].copy()
            joined = formal.merge(candidate, on=["trade_date", "stock_code"], how="inner").merge(
                horizon_labels,
                on=["trade_date", "stock_code"],
                how="inner",
            )
            daily = _daily_rank_ic(joined, label_col=label_col)
            if daily.empty:
                continue
            daily.insert(0, "horizon", horizon_key)
            all_daily.append(daily)
            for result in rank_ic_positive_ratio_by_window(daily, windows):
                mask = (
                    (augmented["horizon"].astype(str).str.lower() == horizon_key)
                    & (augmented["window"].astype(str) == result["window"])
                )
                for column, value in result.items():
                    if column == "window":
                        continue
                    augmented.loc[mask, column] = value

    daily_output = (
        pd.concat(all_daily, ignore_index=True)
        if all_daily
        else pd.DataFrame(columns=["horizon", "trade_date", "formal_rank_ic", "candidate_rank_ic"])
    )
    return augmented, daily_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment model metric CSV with RankIC positive-ratio deltas.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--snapshot-json", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--label-parts-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--daily-output-csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = pd.read_csv(args.metrics_csv)
    snapshot = json.loads(Path(args.snapshot_json).read_text(encoding="utf-8"))
    augmented, daily = augment_metrics(
        metrics,
        snapshot=snapshot,
        db_path=Path(args.db_path),
        label_parts_dir=Path(args.label_parts_dir),
    )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output, index=False, encoding="utf-8-sig")
    if args.daily_output_csv:
        daily_output = Path(args.daily_output_csv)
        daily_output.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(daily_output, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "output_csv": str(output.resolve()),
                "daily_output_csv": str(Path(args.daily_output_csv).resolve()) if args.daily_output_csv else None,
                "rows": int(len(augmented)),
                "daily_rows": int(len(daily)),
                "boundaries": {
                    "no_training": True,
                    "no_prediction": True,
                    "no_production_manifest_change": True,
                    "no_signal": True,
                    "no_backtest": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
