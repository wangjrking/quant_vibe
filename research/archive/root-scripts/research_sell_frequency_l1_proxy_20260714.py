from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
L1_DAILY_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l1_raw_tables" / "daily_data.duckdb"

OUT_CSV = REPORT_DIR / "sell_frequency_l1_proxy_results_20260714.csv"
OUT_JSON = REPORT_DIR / "sell_frequency_l1_proxy_results_20260714.json"
REPORT_MD = REPORT_DIR / "sell_frequency_l1_proxy_review_20260714.md"

SIGNALS = {
    "conditional_full": REPORT_DIR / "signals" / "ogd_conditional_reweight" / "ogd_mildhi20_deep00_cap90_full.csv",
    "conditional_202501": REPORT_DIR / "signals" / "ogd_conditional_reweight" / "ogd_mildhi20_deep00_cap90_from_202501.csv",
    "downscale_full": REPORT_DIR / "signals" / "latest_core_downscale" / "ogd_deep8_x0p85_cap75_full.csv",
    "cpo_lift_202501": REPORT_DIR / "signals" / "latest_candidate_position_lift" / "cpo_low60_x3p0_cap90_from_202501.csv",
    "ogd_lift_202501": REPORT_DIR / "signals" / "latest_candidate_position_lift" / "ogd_deep8_x2p5_cap90_from_202501.csv",
}

HOLD_DAYS = [1, 2, 3, 5]


def load_l1_open() -> pd.DataFrame:
    con = duckdb.connect(str(L1_DAILY_DB), read_only=True)
    try:
        return con.execute(
            """
            WITH cal AS (
                SELECT DISTINCT trade_date
                FROM daily_data
            ),
            cal2 AS (
                SELECT
                    trade_date,
                    lead(trade_date, 1) OVER (ORDER BY trade_date) AS d1,
                    lead(trade_date, 2) OVER (ORDER BY trade_date) AS d2,
                    lead(trade_date, 3) OVER (ORDER BY trade_date) AS d3,
                    lead(trade_date, 5) OVER (ORDER BY trade_date) AS d5
                FROM cal
            )
            SELECT
                d.ts_code AS stock_code,
                d.trade_date,
                d.open,
                c.d1,
                n1.open AS open_d1,
                c.d2,
                n2.open AS open_d2,
                c.d3,
                n3.open AS open_d3,
                c.d5,
                n5.open AS open_d5
            FROM daily_data d
            JOIN cal2 c ON c.trade_date = d.trade_date
            LEFT JOIN daily_data n1 ON n1.trade_date = c.d1 AND n1.ts_code = d.ts_code
            LEFT JOIN daily_data n2 ON n2.trade_date = c.d2 AND n2.ts_code = d.ts_code
            LEFT JOIN daily_data n3 ON n3.trade_date = c.d3 AND n3.ts_code = d.ts_code
            LEFT JOIN daily_data n5 ON n5.trade_date = c.d5 AND n5.ts_code = d.ts_code
            """
        ).fetchdf()
    finally:
        con.close()


def annualize(daily_returns: pd.Series, periods: int = 244) -> float:
    clean = daily_returns.dropna()
    if clean.empty:
        return 0.0
    equity = (1.0 + clean).prod()
    years = len(clean) / periods
    if years <= 0:
        return 0.0
    return float(equity ** (1.0 / years) - 1.0)


def sharpe(daily_returns: pd.Series, periods: int = 244) -> float:
    clean = daily_returns.dropna()
    std = clean.std(ddof=1)
    if clean.empty or not std or pd.isna(std):
        return 0.0
    return float(clean.mean() / std * (periods ** 0.5))


def max_drawdown(daily_returns: pd.Series) -> float:
    clean = daily_returns.dropna()
    if clean.empty:
        return 0.0
    equity = (1.0 + clean).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(-dd.min())


def evaluate_signal(name: str, path: Path, market: pd.DataFrame) -> list[dict]:
    if not path.exists():
        return [{"case": name, "missing_signal": str(path)}]
    sig = pd.read_csv(path, encoding="utf-8-sig", dtype={"buy_date": str, "stock_code": str})
    sig["target_pct"] = pd.to_numeric(sig["target_pct"], errors="coerce").fillna(0.0)
    merged = sig.merge(
        market,
        left_on=["buy_date", "stock_code"],
        right_on=["trade_date", "stock_code"],
        how="left",
    )
    rows = []
    for hold in HOLD_DAYS:
        exit_col = f"open_d{hold}"
        date_col = f"d{hold}"
        part = merged.copy()
        part["trade_ret"] = pd.to_numeric(part[exit_col], errors="coerce") / pd.to_numeric(part["open"], errors="coerce") - 1.0
        part["weighted_ret"] = part["trade_ret"] * part["target_pct"]
        daily = part.groupby("buy_date")["weighted_ret"].sum().sort_index()
        rows.append(
            {
                "case": name,
                "signal_file": str(path),
                "hold_days_proxy": hold,
                "rows": int(len(part)),
                "buy_days": int(part["buy_date"].nunique()),
                "stock_count": int(part["stock_code"].nunique()),
                "min_buy": str(part["buy_date"].min()),
                "max_buy": str(part["buy_date"].max()),
                "missing_entry_open": int(part["open"].isna().sum()),
                "missing_exit_open": int(part[exit_col].isna().sum()),
                "mean_daily_target_sum": float(part.groupby("buy_date")["target_pct"].sum().mean()),
                "proxy_total_return": float((1.0 + daily.fillna(0.0)).prod() - 1.0),
                "proxy_annual_return": annualize(daily),
                "proxy_sharpe": sharpe(daily),
                "proxy_max_drawdown": max_drawdown(daily),
                "proxy_win_day_ratio": float((daily > 0).mean()) if len(daily) else 0.0,
                "exit_date_col": date_col,
            }
        )
    return rows


def write_report(frame: pd.DataFrame) -> None:
    lines = [
        "# 卖出频率 L1 代理验证",
        "",
        "## 说明",
        "",
        "- 本报告使用 L1 原始日线 `open` 做开盘到开盘代理测算。",
        "- 这是研究筛选，不是掘金正式回测结论。",
        "- 目的只是筛出 L2 写锁释放后优先补跑的卖出频率参数。",
        "",
        "## 结果",
        "",
        "| 信号 | 代理持有天数 | 代理年化 | 代理 Sharpe | 代理最大回撤 | 买入日 | 平均目标仓位 | 缺失退出价 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    show = frame.sort_values(["proxy_annual_return", "proxy_sharpe"], ascending=[False, False])
    for _, row in show.iterrows():
        lines.append(
            f"| {row.get('case')} | {int(row.get('hold_days_proxy', 0))} | "
            f"{float(row.get('proxy_annual_return', 0.0)) * 100:.2f}% | "
            f"{float(row.get('proxy_sharpe', 0.0)):.3f} | "
            f"{float(row.get('proxy_max_drawdown', 0.0)) * 100:.2f}% | "
            f"{int(row.get('buy_days', 0))} | "
            f"{float(row.get('mean_daily_target_sum', 0.0)) * 100:.2f}% | "
            f"{int(row.get('missing_exit_open', 0))} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 汇总 CSV：`{OUT_CSV}`",
            f"- 汇总 JSON：`{OUT_JSON}`",
            f"- L1 日线库：`{L1_DAILY_DB}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    market = load_l1_open()
    rows: list[dict] = []
    for name, path in SIGNALS.items():
        rows.extend(evaluate_signal(name, path, market))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(frame)
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "report": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
