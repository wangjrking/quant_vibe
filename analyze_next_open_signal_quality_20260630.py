from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from stock_daily_data_route import resolve_stock_daily_duckdb_path


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"

SIGNALS = {
    "dyn_mild_09_12_15": REPORT_DIR / "dynamic_target_signals" / "dyn_mild_09_12_15.csv",
    "hp_new70_turn80_top5_s170_cap27_h2m3": REPORT_DIR
    / "soft_newstock_highpos_signals"
    / "hp_new70_turn80_top5_s170_cap27_h2m3.csv",
    "hp_new70_gap_rerank_penalty_1p5": REPORT_DIR
    / "open_gap_frequency_signals"
    / "hp_new70_turn80_top5_s170_cap27_h2m3__gap_rerank_penalty_1p5.csv",
}

OUT_DETAIL = REPORT_DIR / "next_open_signal_quality_detail_20260630.csv"
OUT_BUCKET = REPORT_DIR / "next_open_signal_quality_buckets_20260630.csv"
OUT_JSON = REPORT_DIR / "next_open_signal_quality_summary_20260630.json"


FEATURES: list[dict[str, Any]] = [
    {"name": "rank", "direction": "low_good", "q": 5},
    {"name": "pred_prob", "direction": "high_good", "q": 5},
    {"name": "rank_3d", "direction": "high_good", "q": 5},
    {"name": "rank_5d", "direction": "high_good", "q": 5},
    {"name": "rank_10d", "direction": "high_good", "q": 5},
    {"name": "amount", "direction": "high_good", "q": 5},
    {"name": "turnover_rate", "direction": "middle_good", "q": 5},
    {"name": "total_mv", "direction": "high_good", "q": 5},
    {"name": "atr_qfq", "direction": "low_good", "q": 5},
    {"name": "pct_chg", "direction": "middle_good", "q": 5},
    {"name": "prev_pct_chg", "direction": "middle_good", "q": 5},
    {"name": "two_day_ret", "direction": "middle_good", "q": 5},
    {"name": "target_pct", "direction": "middle_good", "q": 5},
    {"name": "buy_open_gap", "direction": "low_good", "q": 5},
]


def _read_signal(path: Path, strategy: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["strategy"] = strategy
    df["buy_date"] = df["buy_date"].fillna("").astype(str)
    df = df[df["buy_date"].str.len() == 8].copy()
    for column in [
        "rank",
        "pred_prob",
        "rank_3d",
        "rank_5d",
        "rank_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "pct_chg",
        "prev_pct_chg",
        "two_day_ret",
        "target_pct",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _trade_calendar(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [
        row[0]
        for row in con.execute(
            """
            SELECT DISTINCT trade_date
            FROM STOCK_DAILY_DATA
            WHERE trade_date BETWEEN '20220601' AND '20260630'
            ORDER BY trade_date
            """
        ).fetchall()
    ]


def _market_frame(con: duckdb.DuckDBPyConnection, signals: pd.DataFrame, next_map: dict[str, str | None]) -> pd.DataFrame:
    codes = sorted(set(signals["stock_code"].dropna().astype(str)))
    dates = set(signals["buy_date"].dropna().astype(str))
    for date in list(dates):
        next_date = next_map.get(date)
        if next_date:
            dates.add(next_date)
    code_df = pd.DataFrame({"stock_code": codes})
    date_df = pd.DataFrame({"trade_date": sorted(dates)})
    con.register("tmp_codes", code_df)
    con.register("tmp_dates", date_df)
    try:
        return con.execute(
            """
            SELECT m.stock_code, m.trade_date, m.open, m.close, m.pre_close, m.pct_chg
            FROM STOCK_DAILY_DATA m
            JOIN tmp_codes c USING (stock_code)
            JOIN tmp_dates d USING (trade_date)
            """
        ).fetchdf()
    finally:
        con.unregister("tmp_codes")
        con.unregister("tmp_dates")


def _attach_forward_returns(signals: pd.DataFrame) -> pd.DataFrame:
    market_duckdb_path = resolve_stock_daily_duckdb_path(require_exists=True)
    con = duckdb.connect(str(market_duckdb_path), read_only=True)
    try:
        trade_days = _trade_calendar(con)
        next_map = {day: trade_days[i + 1] if i + 1 < len(trade_days) else None for i, day in enumerate(trade_days)}
        market = _market_frame(con, signals, next_map)
    finally:
        con.close()

    market_buy = market.rename(
        columns={
            "trade_date": "buy_date",
            "open": "buy_open",
            "close": "buy_close",
            "pre_close": "buy_pre_close",
            "pct_chg": "buy_pct_chg",
        }
    )
    market_next = market.rename(
        columns={
            "trade_date": "next_trade_date",
            "open": "next_open",
            "close": "next_close",
            "pre_close": "next_pre_close",
            "pct_chg": "next_pct_chg",
        }
    )

    out = signals.copy()
    out["next_trade_date"] = out["buy_date"].map(next_map)
    out = out.merge(market_buy, on=["stock_code", "buy_date"], how="left")
    out = out.merge(market_next[["stock_code", "next_trade_date", "next_open", "next_close"]], on=["stock_code", "next_trade_date"], how="left")
    out["buy_open_gap"] = out["buy_open"] / out["buy_pre_close"] - 1.0
    out["buy_open_to_close"] = out["buy_close"] / out["buy_open"] - 1.0
    out["buy_open_to_next_open"] = out["next_open"] / out["buy_open"] - 1.0
    out["weighted_next_open_ret"] = out["buy_open_to_next_open"] * out["target_pct"].fillna(0.0)
    out["period"] = "train"
    out.loc[out["signal_date"] >= "20250601", "period"] = "recent"
    out.loc[out["signal_date"] >= "20260101", "period"] = "ytd2026"
    return out


def _summary_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (strategy, period), grp in df.groupby(["strategy", "period"], dropna=False):
        valid = grp.dropna(subset=["buy_open_to_next_open"])
        if valid.empty:
            continue
        rows.append(
            {
                "strategy": strategy,
                "feature": "__overall__",
                "bucket": "all",
                "period": period,
                "rows": int(len(valid)),
                "mean_next_open_ret": float(valid["buy_open_to_next_open"].mean()),
                "median_next_open_ret": float(valid["buy_open_to_next_open"].median()),
                "weighted_next_open_ret": float(valid["weighted_next_open_ret"].sum() / valid["target_pct"].sum()),
                "win_rate": float((valid["buy_open_to_next_open"] > 0).mean()),
                "avg_open_gap": float(valid["buy_open_gap"].mean()),
                "avg_target_pct": float(valid["target_pct"].mean()),
            }
        )
    return rows


def _bucket_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy, sdf in df.groupby("strategy"):
        for item in FEATURES:
            feature = item["name"]
            if feature not in sdf.columns:
                continue
            feature_values = pd.to_numeric(sdf[feature], errors="coerce")
            valid_feature = sdf[feature_values.notna()].copy()
            if valid_feature[feature].nunique(dropna=True) < 4:
                continue
            try:
                valid_feature["_bucket"] = pd.qcut(valid_feature[feature], int(item["q"]), duplicates="drop")
            except ValueError:
                continue
            for period, pdf in valid_feature.groupby("period", dropna=False):
                for bucket, grp in pdf.groupby("_bucket", observed=True):
                    valid = grp.dropna(subset=["buy_open_to_next_open"])
                    if valid.empty:
                        continue
                    target_sum = valid["target_pct"].sum()
                    rows.append(
                        {
                            "strategy": strategy,
                            "feature": feature,
                            "direction": item["direction"],
                            "bucket": str(bucket),
                            "period": period,
                            "rows": int(len(valid)),
                            "mean_next_open_ret": float(valid["buy_open_to_next_open"].mean()),
                            "median_next_open_ret": float(valid["buy_open_to_next_open"].median()),
                            "weighted_next_open_ret": float(valid["weighted_next_open_ret"].sum() / target_sum) if target_sum else None,
                            "win_rate": float((valid["buy_open_to_next_open"] > 0).mean()),
                            "avg_feature": float(valid[feature].mean()),
                            "avg_open_gap": float(valid["buy_open_gap"].mean()),
                            "avg_target_pct": float(valid["target_pct"].mean()),
                        }
                    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def main() -> None:
    frames = [_read_signal(path, strategy) for strategy, path in SIGNALS.items()]
    signals = pd.concat(frames, ignore_index=True)
    detail = _attach_forward_returns(signals)
    detail.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    rows = _summary_rows(detail) + _bucket_rows(detail)
    _write_csv(OUT_BUCKET, rows)

    bucket_df = pd.DataFrame(rows)
    recent = bucket_df[bucket_df["period"].isin(["recent", "ytd2026"]) & (bucket_df["feature"] != "__overall__")].copy()
    recent["weighted_next_open_ret"] = pd.to_numeric(recent["weighted_next_open_ret"], errors="coerce")
    worst = recent.sort_values("weighted_next_open_ret").head(20).to_dict("records")
    best = recent.sort_values("weighted_next_open_ret", ascending=False).head(20).to_dict("records")
    summary = {
        "status": "completed_research_only",
        "note": "forward next-open return is used only as diagnostic target; live rules must use signal-date or buy-open observable features.",
        "detail_file": str(OUT_DETAIL),
        "bucket_file": str(OUT_BUCKET),
        "row_count": int(len(detail)),
        "valid_forward_rows": int(detail["buy_open_to_next_open"].notna().sum()),
        "worst_recent_buckets": worst,
        "best_recent_buckets": best,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
