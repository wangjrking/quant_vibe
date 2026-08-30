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
    / "direct_l4_recent_window_probe"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DB_1D = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_1d_open_return_formal.duckdb"
DB_5D = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_5d_open_return_formal.duckdb"
DB_10D = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
DB_L2 = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
T1 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate"
T5 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate"
T10 = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"

OUT_CSV = REPORT_DIR / "direct_l4_recent_probe_pool_20260715.csv"
SUMMARY_CSV = REPORT_DIR / "direct_l4_recent_probe_summary_20260715.csv"
REPORT_MD = REPORT_DIR / "direct_l4_recent_probe_report_20260715.md"


def load_recent_pool() -> pd.DataFrame:
    if OUT_CSV.exists():
        return pd.read_csv(OUT_CSV, encoding="utf-8-sig", dtype={"trade_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"ATTACH '{DB_1D.as_posix()}' AS d1 (READ_ONLY)")
    con.execute(f"ATTACH '{DB_5D.as_posix()}' AS d5 (READ_ONLY)")
    con.execute(f"ATTACH '{DB_10D.as_posix()}' AS d10 (READ_ONLY)")
    con.execute(f"ATTACH '{DB_L2.as_posix()}' AS l2 (READ_ONLY)")
    sql = f"""
    WITH cal AS (
        SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
               lead(trade_date, 2) OVER (ORDER BY trade_date) AS next_trade_date
        FROM (SELECT DISTINCT trade_date FROM l2.STOCK_DAILY_DATA WHERE trade_date >= '20250102')
    ),
    t10f AS (
        SELECT trade_date, stock_code, pred_prob AS pred_10d
        FROM d10.{T10}
        WHERE trade_date >= '20250102'
          AND pred_prob >= 0.88
          AND stock_code NOT LIKE '%.BJ'
    )
    SELECT
        t10f.trade_date,
        cal.buy_date,
        cal.next_trade_date,
        t10f.stock_code,
        t1.pred_prob AS pred_1d,
        t5.pred_prob AS pred_5d,
        t10f.pred_10d,
        t1.pred_prob AS rank_1d,
        t5.pred_prob AS rank_5d,
        t10f.pred_10d AS rank_10d,
        m.name,
        m.amount,
        m.turnover_rate,
        m.total_mv,
        m.atr_qfq,
        m.pct_chg AS signal_pct_chg_raw,
        (b.open / NULLIF(b.pre_close, 0) - 1.0) * 100.0 AS exec_open_gap_pct,
        n.open / NULLIF(b.open, 0) - 1.0 AS next_open_ret
    FROM t10f
    JOIN d5.{T5} t5 ON t5.trade_date = t10f.trade_date AND t5.stock_code = t10f.stock_code
    JOIN d1.{T1} t1 ON t1.trade_date = t10f.trade_date AND t1.stock_code = t10f.stock_code
    JOIN cal ON cal.trade_date = t10f.trade_date
    JOIN l2.STOCK_DAILY_DATA m ON m.trade_date = t10f.trade_date AND m.stock_code = t10f.stock_code
    JOIN l2.STOCK_DAILY_DATA b ON b.trade_date = cal.buy_date AND b.stock_code = t10f.stock_code
    JOIN l2.STOCK_DAILY_DATA n ON n.trade_date = cal.next_trade_date AND n.stock_code = t10f.stock_code
    WHERE cal.buy_date IS NOT NULL
      AND cal.next_trade_date IS NOT NULL
      AND m.pct_chg <= -1.0
      AND m.amount >= 80000
      AND m.total_mv >= 150000
      AND m.atr_qfq IS NOT NULL
      AND (b.open / NULLIF(b.pre_close, 0) - 1.0) * 100.0 BETWEEN -8.0 AND 2.0
      AND b.open < b.pre_close * (1.0 + CASE WHEN t10f.stock_code LIKE '300%' OR t10f.stock_code LIKE '301%' OR t10f.stock_code LIKE '688%' THEN 0.20 ELSE 0.10 END) * 0.995
      AND coalesce(m.ST_TYPE, '') IN ('', '0', '0.0')
      AND coalesce(b.ST_TYPE, '') IN ('', '0', '0.0')
    """
    df = con.execute(sql).fetchdf()
    con.close()
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return df


def score(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "r10":
        return df["rank_10d"]
    if mode == "r10_5":
        return 0.75 * df["rank_10d"] + 0.25 * df["rank_5d"]
    if mode == "r10_low5":
        return 0.85 * df["rank_10d"] + 0.15 * (1 - df["rank_5d"])
    if mode == "r10_mid1":
        return 0.80 * df["rank_10d"] - 0.12 * (df["rank_1d"] - 0.82).abs()
    raise ValueError(mode)


def main() -> None:
    df = load_recent_pool()
    for col in ["rank_1d", "rank_5d", "rank_10d", "signal_pct_chg_raw", "exec_open_gap_pct", "next_open_ret"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    for mode in ["r10", "r10_5", "r10_low5", "r10_mid1"]:
        df["_score"] = score(df, mode)
        for pct_max in [-1.5, -2.5, -4.0, -6.0]:
            for gap_high in [0.5, 1.0, 1.5]:
                for gap_low in [-8.0, -3.0, -1.0]:
                    for r10_min in [0.90, 0.95, 0.98]:
                        part = df[
                            (df["signal_pct_chg_raw"] <= pct_max)
                            & (df["exec_open_gap_pct"] >= gap_low)
                            & (df["exec_open_gap_pct"] <= gap_high)
                            & (df["rank_10d"] >= r10_min)
                        ].copy()
                        if part["buy_date"].nunique() < 40:
                            continue
                        sel = part.sort_values(["buy_date", "_score", "stock_code"], ascending=[True, False, True]).groupby("buy_date", as_index=False).head(1)
                        dates = sorted(sel["buy_date"].astype(str).unique())
                        recent60 = sel[sel["buy_date"].astype(str).isin(dates[-60:])]
                        for label, sub in [("from2025", sel), ("recent60", recent60)]:
                            ret = sub["next_open_ret"].astype(float)
                            rows.append(
                                {
                                    "case": f"{mode}_p{pct_max}_gl{gap_low}_gh{gap_high}_r10{r10_min}".replace("-", "m").replace(".", "p"),
                                    "mode": mode,
                                    "period": label,
                                    "pct_max": pct_max,
                                    "gap_low": gap_low,
                                    "gap_high": gap_high,
                                    "r10_min": r10_min,
                                    "buy_days": int(sub["buy_date"].nunique()),
                                    "rows": int(len(sub)),
                                    "max_buy_date": str(sub["buy_date"].max()),
                                    "mean_next_open_ret": float(ret.mean()),
                                    "win_rate": float((ret > 0).mean()),
                                    "median_next_open_ret": float(ret.median()),
                                }
                            )
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    best = out[out["period"] == "recent60"].sort_values(["mean_next_open_ret", "buy_days"], ascending=False).head(30)
    lines = [
        "# 直接 L4 近期窗口买入逻辑探针 20260715",
        "",
        "## 说明",
        "",
        "- 本轮只用 2025 以来窗口做轻量探针，避免全量三模型 join 长时间占用。",
        "- 指标是买入日开盘到下一交易日开盘收益均值，用于判断是否存在明显优于当前策略的近期买入逻辑。",
        "",
        "| 候选 | 买入日 | 最新买入日 | 平均次开收益 | 中位次开收益 | 胜率 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {int(row['buy_days'])} | {row['max_buy_date']} | "
            f"{row['mean_next_open_ret']*100:.2f}% | {row['median_next_open_ret']*100:.2f}% | {row['win_rate']*100:.2f}% |"
        )
    lines.extend(["", "## 证据路径", "", f"- 近期候选池：`{OUT_CSV}`", f"- 探针汇总：`{SUMMARY_CSV}`"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pool_rows": len(df), "summary_rows": len(out), "report": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
