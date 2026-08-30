"""Evaluate L4 prediction assets under the four-year observation policy.

This script is model-layer only. It reads prediction tables, computes model
quality metrics, and writes research evidence. It does not generate signals,
run backtests, or modify manifests.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
TOP_NS = [1, 3, 5, 10, 20]


def _parse_baseline(values: Iterable[str]) -> dict[str, str]:
    baselines: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"baseline must use name=table format: {value}")
        name, table = value.split("=", 1)
        name = name.strip()
        table = table.strip()
        if not name or not table:
            raise ValueError(f"invalid baseline: {value}")
        baselines[name] = table
    return baselines


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "select count(*) from sqlite_master where type='table' and name=?",
        (table,),
    ).fetchone()
    return bool(row and row[0])


def _read_asset(con: sqlite3.Connection, table: str, label: str) -> pd.DataFrame:
    if not _table_exists(con, table):
        raise ValueError(f"missing table: {table}")
    columns = {row[1] for row in con.execute(f"pragma table_info({table})")}
    if "trade_date" not in columns or "stock_code" not in columns or "pred_prob" not in columns:
        raise ValueError(f"table lacks required prediction columns: {table}")
    if label not in columns:
        raise ValueError(f"table lacks label column {label}: {table}")
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob, {label} as label from {table}",
        con,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def _quality(con: sqlite3.Connection, table: str) -> dict[str, object]:
    row = con.execute(
        f"""
        select
            count(*),
            min(trade_date),
            max(trade_date),
            count(distinct trade_date),
            sum(case when pred_prob is null then 1 else 0 end)
        from {table}
        """
    ).fetchone()
    dup = con.execute(
        f"""
        select count(*) from (
            select trade_date, stock_code, count(*) c
            from {table}
            group by trade_date, stock_code
            having c > 1
        )
        """
    ).fetchone()[0]
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
    }


def _daily_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid_frame = frame.dropna(subset=["pred_prob", "label"]).copy()
    for trade_date, group in valid_frame.groupby("trade_date", sort=True):
        ordered = group.sort_values("pred_prob", ascending=False)
        if len(ordered) >= 2:
            rank_ic = ordered["pred_prob"].rank(method="average").corr(
                ordered["label"].rank(method="average")
            )
        else:
            rank_ic = np.nan
        row: dict[str, object] = {
            "trade_date": str(trade_date),
            "n": int(len(ordered)),
            "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
        }
        for top_n in TOP_NS:
            row[f"top{top_n}"] = float(ordered.head(top_n)["label"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize(daily: pd.DataFrame) -> dict[str, object]:
    if daily.empty:
        return {"trade_days": 0, "min_trade_date": None, "max_trade_date": None}
    out: dict[str, object] = {
        "trade_days": int(len(daily)),
        "min_trade_date": str(daily["trade_date"].min()),
        "max_trade_date": str(daily["trade_date"].max()),
    }
    for metric in METRICS:
        out[metric] = float(pd.to_numeric(daily[metric], errors="coerce").mean())
    return out


def _period_table(daily: pd.DataFrame, period_len: int) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["period", *METRICS, "trade_days"])
    frame = daily.copy()
    frame["period"] = frame["trade_date"].str.slice(0, period_len)
    grouped = frame.groupby("period", as_index=False)[METRICS].mean()
    grouped["trade_days"] = frame.groupby("period").size().values
    return grouped


def _compare(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    windows: list[int],
) -> tuple[dict[str, object], pd.DataFrame]:
    joined = candidate.rename(
        columns={"pred_prob": "candidate_pred", "label": "candidate_label"}
    ).merge(
        baseline.rename(columns={"pred_prob": "baseline_pred", "label": "baseline_label"}),
        on=["trade_date", "stock_code"],
        how="inner",
    )
    joined = joined.dropna(subset=["candidate_pred", "baseline_pred", "candidate_label"])
    candidate_eval = joined[
        ["trade_date", "stock_code", "candidate_pred", "candidate_label"]
    ].rename(columns={"candidate_pred": "pred_prob", "candidate_label": "label"})
    baseline_eval = joined[
        ["trade_date", "stock_code", "baseline_pred", "candidate_label"]
    ].rename(columns={"baseline_pred": "pred_prob", "candidate_label": "label"})
    candidate_daily = _daily_metrics(candidate_eval)
    baseline_daily = _daily_metrics(baseline_eval)
    candidate_summary = _summarize(candidate_daily)
    baseline_summary = _summarize(baseline_daily)
    comparison: dict[str, object] = {
        "joined_rows": int(len(joined)),
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "delta": {
            metric: float(candidate_summary[metric] - baseline_summary[metric])
            for metric in METRICS
        },
    }
    for window in windows:
        candidate_window = _summarize(candidate_daily.tail(window))
        baseline_window = _summarize(baseline_daily.tail(window))
        comparison[f"recent{window}_delta"] = {
            metric: float(candidate_window[metric] - baseline_window[metric])
            for metric in METRICS
        }

    merged_daily = candidate_daily.merge(
        baseline_daily,
        on="trade_date",
        suffixes=("_candidate", "_baseline"),
    )
    for metric in METRICS:
        merged_daily[f"{metric}_delta"] = (
            merged_daily[f"{metric}_candidate"] - merged_daily[f"{metric}_baseline"]
        )
    return comparison, merged_daily


def evaluate(
    *,
    db_path: Path,
    table: str,
    label: str,
    output_dir: Path,
    asset_name: str,
    baselines: dict[str, str],
    recent_windows: list[int],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        candidate = _read_asset(con, table, label)
        candidate_daily = _daily_metrics(candidate)
        result: dict[str, object] = {
            "asset_name": asset_name,
            "label": label,
            "table": table,
            "db_path": str(db_path),
            "quality": _quality(con, table),
            "full": _summarize(candidate_daily),
            "recent_windows": {
                f"recent{window}": _summarize(candidate_daily.tail(window))
                for window in recent_windows
            },
            "comparisons": [],
            "boundaries": {
                "no_signal": True,
                "no_backtest": True,
                "no_manifest_change": True,
                "model_layer_metrics_only": True,
            },
        }
        candidate_daily.to_csv(
            output_dir / f"{asset_name}_daily_eval.csv",
            index=False,
            encoding="utf-8",
        )
        _period_table(candidate_daily, 6).to_csv(
            output_dir / f"{asset_name}_monthly_eval.csv",
            index=False,
            encoding="utf-8",
        )
        _period_table(candidate_daily, 4).to_csv(
            output_dir / f"{asset_name}_annual_eval.csv",
            index=False,
            encoding="utf-8",
        )
        for baseline_name, baseline_table in baselines.items():
            if not _table_exists(con, baseline_table):
                result["comparisons"].append(
                    {
                        "baseline_name": baseline_name,
                        "baseline_table": baseline_table,
                        "error": "missing table",
                    }
                )
                continue
            baseline = _read_asset(con, baseline_table, label)
            comparison, merged_daily = _compare(
                candidate,
                baseline,
                windows=recent_windows,
            )
            comparison["baseline_name"] = baseline_name
            comparison["baseline_table"] = baseline_table
            result["comparisons"].append(comparison)
            merged_daily.to_csv(
                output_dir / f"{asset_name}_vs_{baseline_name}_daily.csv",
                index=False,
                encoding="utf-8",
            )
            delta_cols = ["trade_date", *[f"{metric}_delta" for metric in METRICS]]
            monthly_delta = _period_table(
                merged_daily[delta_cols].rename(
                    columns={f"{metric}_delta": metric for metric in METRICS}
                ),
                6,
            )
            monthly_delta.to_csv(
                output_dir / f"{asset_name}_vs_{baseline_name}_monthly_delta.csv",
                index=False,
                encoding="utf-8",
            )
    summary_path = output_dir / f"{asset_name}_eval_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-name", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--recent-window", action="append", type=int, default=[20, 63, 126])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        db_path=Path(args.db_path),
        table=args.table,
        label=args.label,
        output_dir=Path(args.output_dir),
        asset_name=args.asset_name,
        baselines=_parse_baseline(args.baseline),
        recent_windows=list(dict.fromkeys(args.recent_window)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
