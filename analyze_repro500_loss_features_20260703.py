from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
ARTIFACT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "postrank_open_filter_candidates"
    / "qfq_rerank_refill_candidates"
    / "juejin_runs"
    / "artifacts_reconstruct_500_p435"
)
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "repro500_loss_feature_analysis"


FEATURES = [
    "rank",
    "pred_prob",
    "pred_1d",
    "pred_3d",
    "pred_5d",
    "pred_10d",
    "amount",
    "turnover_rate",
    "total_mv",
    "atr_qfq",
    "pct_chg",
    "buy_open_gap_pct",
]


def _safe_qcut(series: pd.Series, q: int = 5) -> pd.Series:
    try:
        return pd.qcut(series, q=q, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(series), index=series.index)


def _bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURES:
        if col not in df.columns:
            continue
        tmp = df[[col, "ret", "pnl"]].dropna().copy()
        if tmp.empty:
            continue
        tmp["bucket"] = _safe_qcut(tmp[col].astype(float), 5)
        for bucket, g in tmp.groupby("bucket", observed=True):
            rows.append(
                {
                    "feature": col,
                    "bucket": str(bucket),
                    "count": int(len(g)),
                    "avg_ret": float(g["ret"].mean()),
                    "median_ret": float(g["ret"].median()),
                    "win_rate": float((g["ret"] > 0).mean()),
                    "sum_pnl": float(g["pnl"].sum()),
                    "min_ret": float(g["ret"].min()),
                    "max_ret": float(g["ret"].max()),
                }
            )
    return pd.DataFrame(rows)


def _threshold_scan(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = {
        "pct_chg_max": ("pct_chg", "<=", [-8, -6, -5, -4, -3, -2.5, -2.0, -1.75]),
        "buy_gap_max": ("buy_open_gap_pct", "<=", [-5, -4, -3, -2, -1, 0, 0.5, 1.0, 1.5]),
        "buy_gap_min": ("buy_open_gap_pct", ">=", [-8, -7, -6, -5, -4, -3, -2, -1]),
        "pred_1d_min": ("pred_1d", ">=", [-0.0025, -0.002, -0.0015, -0.001, -0.0005, 0]),
        "pred_10d_min": ("pred_10d", ">=", [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]),
        "amount_min": ("amount", ">=", [80000, 100000, 150000, 200000, 300000, 500000]),
        "turnover_min": ("turnover_rate", ">=", [1, 2, 3, 5, 8, 10]),
        "total_mv_min": ("total_mv", ">=", [200000, 300000, 500000, 800000, 1000000, 2000000]),
        "atr_max": ("atr_qfq", "<=", [0.5, 1, 1.5, 2, 3, 5, 8]),
    }
    for name, (col, op, vals) in specs.items():
        if col not in df.columns:
            continue
        for v in vals:
            mask = df[col].astype(float) <= v if op == "<=" else df[col].astype(float) >= v
            g = df[mask].copy()
            if len(g) < 30:
                continue
            daily = g.groupby("buy_date")["ret"].mean().sort_index()
            std = daily.std(ddof=1)
            rows.append(
                {
                    "rule": f"{col} {op} {v}",
                    "feature": col,
                    "op": op,
                    "threshold": v,
                    "trade_count": int(len(g)),
                    "buy_days": int(daily.size),
                    "avg_ret": float(g["ret"].mean()),
                    "win_rate": float((g["ret"] > 0).mean()),
                    "daily_sharpe_proxy": float(daily.mean() / std * np.sqrt(252)) if std else None,
                    "sum_pnl": float(g["pnl"].sum()),
                    "removed_trades": int(len(df) - len(g)),
                    "removed_losing_trades": int(((~mask) & (df["ret"] < 0)).sum()),
                    "removed_winning_trades": int(((~mask) & (df["ret"] > 0)).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["daily_sharpe_proxy", "avg_ret"], ascending=False)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ARTIFACT_DIR / "paired_trades_with_signal_features.csv", dtype={"signal_date": str, "buy_date": str})
    df["buy_month"] = df["buy_date"].str.slice(0, 6)
    df["buy_year"] = df["buy_date"].str.slice(0, 4)
    df["weighted_ret"] = df["ret"] * df["target_pct"]

    bucket = _bucket_summary(df)
    bucket.to_csv(OUT_DIR / "feature_bucket_summary.csv", index=False, encoding="utf-8-sig")

    scan = _threshold_scan(df)
    scan.to_csv(OUT_DIR / "single_threshold_trade_scan.csv", index=False, encoding="utf-8-sig")

    month = (
        df.groupby("buy_month")
        .agg(
            trades=("ret", "size"),
            avg_ret=("ret", "mean"),
            win_rate=("ret", lambda s: float((s > 0).mean())),
            sum_pnl=("pnl", "sum"),
            min_ret=("ret", "min"),
            max_ret=("ret", "max"),
        )
        .reset_index()
        .sort_values("sum_pnl")
    )
    month.to_csv(OUT_DIR / "month_contribution.csv", index=False, encoding="utf-8-sig")

    stock = (
        df.groupby(["stock_code", "name"])
        .agg(
            trades=("ret", "size"),
            avg_ret=("ret", "mean"),
            win_rate=("ret", lambda s: float((s > 0).mean())),
            sum_pnl=("pnl", "sum"),
            min_ret=("ret", "min"),
            max_ret=("ret", "max"),
        )
        .reset_index()
        .sort_values("sum_pnl")
    )
    stock.to_csv(OUT_DIR / "stock_contribution.csv", index=False, encoding="utf-8-sig")

    worst = df.sort_values("pnl").head(40)
    worst.to_csv(OUT_DIR / "worst_40_trades.csv", index=False, encoding="utf-8-sig")

    summary = {
        "trade_count": int(len(df)),
        "avg_ret": float(df["ret"].mean()),
        "win_rate": float((df["ret"] > 0).mean()),
        "sum_pnl": float(df["pnl"].sum()),
        "worst_months": month.head(10).to_dict("records"),
        "best_single_thresholds": scan.head(20).to_dict("records"),
    }
    (OUT_DIR / "loss_feature_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
