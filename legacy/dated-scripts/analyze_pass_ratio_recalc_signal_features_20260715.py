from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
)
SIGNAL = REPORT_DIR / "signals" / "recalc_hardgate_scale" / "rh_top1_s120_cap065_full.csv"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
OUT_TRADES = REPORT_DIR / "pass_ratio_recalc_signal_feature_trades_20260715.csv"
OUT_BUCKETS = REPORT_DIR / "pass_ratio_recalc_signal_feature_buckets_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_recalc_signal_feature_buckets_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_recalc_signal_feature_buckets_20260715.md"

FEATURES = [
    "exec_open_gap_pct",
    "signal_pct_chg_raw",
    "amount",
    "turnover_rate",
    "total_mv",
    "atr_qfq",
    "pred_1d",
    "pred_5d",
    "pred_10d",
    "target_pct",
]


def next_trade_returns(signal: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    con.register("sig", signal[["buy_date", "stock_code"]])
    market = con.execute(
        """
        WITH cal AS (
            SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
            FROM (SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA)
        )
        SELECT
            s.buy_date,
            s.stock_code,
            c.next_trade_date,
            b.open AS buy_open_raw,
            n.open AS next_open_raw,
            n.open / NULLIF(b.open, 0) - 1 AS next_open_ret
        FROM sig s
        JOIN cal c ON c.trade_date = s.buy_date
        LEFT JOIN STOCK_DAILY_DATA b ON b.trade_date = s.buy_date AND b.stock_code = s.stock_code
        LEFT JOIN STOCK_DAILY_DATA n ON n.trade_date = c.next_trade_date AND n.stock_code = s.stock_code
        """
    ).fetchdf()
    con.close()
    return market


def assign_periods(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = sorted(out["buy_date"].dropna().astype(str).unique())
    recent60 = set(dates[-60:])
    out["period"] = "full"
    out.loc[out["buy_date"].astype(str) >= "20250102", "period_from2025"] = "from_202501"
    out.loc[out["buy_date"].astype(str).isin(recent60), "period_recent60"] = "recent60"
    return out


def bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    periods = [("full", frame), ("from_202501", frame[frame["buy_date"].astype(str) >= "20250102"]), ("recent60", frame[frame["buy_date"].astype(str).isin(sorted(frame["buy_date"].dropna().astype(str).unique())[-60:])])]
    for feature in FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.notna().sum() < 8:
            continue
        try:
            labels = pd.qcut(values.rank(method="first"), 4, labels=["q0_low", "q1", "q2", "q3_high"])
        except ValueError:
            continue
        work = frame.copy()
        work["bucket"] = labels.astype(str)
        for period_name, period_df in periods:
            part = work.loc[period_df.index]
            for bucket, group in part.groupby("bucket"):
                ret = pd.to_numeric(group["next_open_ret"], errors="coerce")
                weight = pd.to_numeric(group["target_pct"], errors="coerce").fillna(0)
                valid = ret.notna()
                if valid.sum() == 0:
                    continue
                weighted = (ret[valid] * weight[valid]).sum() / weight[valid].sum() if weight[valid].sum() else ret[valid].mean()
                rows.append(
                    {
                        "feature": feature,
                        "period": period_name,
                        "bucket": bucket,
                        "rows": int(len(group)),
                        "mean_feature": float(pd.to_numeric(group[feature], errors="coerce").mean()),
                        "mean_ret": float(ret[valid].mean()),
                        "weighted_ret": float(weighted),
                        "win_rate": float((ret[valid] > 0).mean()),
                        "mean_target_pct": float(weight.mean()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    signal = pd.read_csv(SIGNAL, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in FEATURES:
        if col in signal.columns:
            signal[col] = pd.to_numeric(signal[col], errors="coerce")
    ret = next_trade_returns(signal)
    merged = signal.merge(ret, on=["buy_date", "stock_code"], how="left")
    merged.to_csv(OUT_TRADES, index=False, encoding="utf-8-sig")
    buckets = bucket_summary(merged)
    buckets.to_csv(OUT_BUCKETS, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(buckets.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 最新硬门槛 Top1 信号特征分层诊断 20260715",
        "",
        "## 说明",
        "",
        "- 输入为 `rh_top1_s120_cap065_full.csv`，即最新 L2 未复权开盘硬门槛重算后的当前候选。",
        "- 收益诊断使用买入日开盘到下一交易日开盘的未复权收益，只用于研究分层，不替代掘金回测。",
        "",
        "## recent60 分层摘要",
        "",
        "| 特征 | 分桶 | 行数 | 特征均值 | 次开均值 | 加权次开 | 胜率 | 平均仓位 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    recent = buckets[buckets["period"] == "recent60"].copy()
    for row in recent.sort_values(["feature", "bucket"]).to_dict("records"):
        lines.append(
            f"| `{row['feature']}` | {row['bucket']} | {row['rows']} | {row['mean_feature']:.4f} | "
            f"{row['mean_ret'] * 100:.2f}% | {row['weighted_ret'] * 100:.2f}% | "
            f"{row['win_rate'] * 100:.2f}% | {row['mean_target_pct'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 逐笔诊断：`{OUT_TRADES}`",
            f"- 分桶 CSV：`{OUT_BUCKETS}`",
            f"- 分桶 JSON：`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
