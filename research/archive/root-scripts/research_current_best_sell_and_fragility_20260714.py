from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE = REPORT_DIR / "signals" / "open_gap_deep_rebalance" / "ogd_gap1x50_deep8up110.csv"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SIGNAL_DIR = REPORT_DIR / "signals" / "current_best_sell_and_fragility"
LOG_DIR = REPORT_DIR / "logs" / "current_best_sell_and_fragility"
OUT_CSV = REPORT_DIR / "current_best_sell_and_fragility_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "current_best_sell_and_fragility_juejin_results_20260714.json"
PROXY_CSV = REPORT_DIR / "current_best_proxy_contribution_20260714.csv"
SUMMARY_JSON = REPORT_DIR / "current_best_sell_and_fragility_summary_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SELL_CASES = [
    {"case": "sell_h1_exit098_cont102", "max_holding_days": 1, "exit": 0.98, "cont": 1.02},
    {"case": "sell_h2_exit098_cont102", "max_holding_days": 2, "exit": 0.98, "cont": 1.02},
    {"case": "sell_h3_exit098_cont102", "max_holding_days": 3, "exit": 0.98, "cont": 1.02},
    {"case": "sell_h4_exit098_cont102", "max_holding_days": 4, "exit": 0.98, "cont": 1.02},
    {"case": "sell_h5_exit098_cont102", "max_holding_days": 5, "exit": 0.98, "cont": 1.02},
    {"case": "sell_h3_exit096_cont102", "max_holding_days": 3, "exit": 0.96, "cont": 1.02},
    {"case": "sell_h3_exit100_cont102", "max_holding_days": 3, "exit": 1.00, "cont": 1.02},
    {"case": "sell_h3_exit098_cont100", "max_holding_days": 3, "exit": 0.98, "cont": 1.00},
    {"case": "sell_h3_exit098_cont105", "max_holding_days": 3, "exit": 0.98, "cont": 1.05},
    {"case": "sell_h2_exit100_cont102", "max_holding_days": 2, "exit": 1.00, "cont": 1.02},
    {"case": "sell_h4_exit096_cont102", "max_holding_days": 4, "exit": 0.96, "cont": 1.02},
]


def load_signal() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    return df


def add_proxy_returns(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"ATTACH '{L2_DB.as_posix()}' AS l2")
    prices = con.execute(
        """
        with cal as (
          select trade_date,
                 lead(trade_date, 1) over(order by trade_date) as d1,
                 lead(trade_date, 2) over(order by trade_date) as d2,
                 lead(trade_date, 3) over(order by trade_date) as d3
          from (select distinct trade_date from l2.STOCK_DAILY_DATA order by trade_date)
        )
        select b.stock_code, b.trade_date as buy_date, b.open as buy_open,
               e1.open as open_d1, e2.open as open_d2, e3.open as open_d3
        from l2.STOCK_DAILY_DATA b
        left join cal c on c.trade_date=b.trade_date
        left join l2.STOCK_DAILY_DATA e1 on e1.trade_date=c.d1 and e1.stock_code=b.stock_code
        left join l2.STOCK_DAILY_DATA e2 on e2.trade_date=c.d2 and e2.stock_code=b.stock_code
        left join l2.STOCK_DAILY_DATA e3 on e3.trade_date=c.d3 and e3.stock_code=b.stock_code
        where b.trade_date between '20220607' and '20260714'
        """
    ).fetchdf()
    prices["buy_date"] = prices["buy_date"].astype(str)
    out = df.merge(prices, on=["stock_code", "buy_date"], how="left")
    for h in [1, 2, 3]:
        out[f"ret_h{h}"] = out[f"open_d{h}"] / out["buy_open"] - 1.0
        out[f"weighted_ret_h{h}"] = out["target_pct"] * out[f"ret_h{h}"]
    out["month"] = out["buy_date"].str.slice(0, 6)
    out["year"] = out["buy_date"].str.slice(0, 4)
    out.to_csv(PROXY_CSV, index=False, encoding="utf-8-sig")
    return out


def top_fragility_cases(proxy: pd.DataFrame) -> list[dict]:
    cases: list[dict] = []
    month_contrib = (
        proxy.groupby("month", as_index=False)["weighted_ret_h2"]
        .sum()
        .sort_values("weighted_ret_h2", ascending=False)
    )
    stock_contrib = (
        proxy.groupby("stock_code", as_index=False)["weighted_ret_h2"]
        .sum()
        .sort_values("weighted_ret_h2", ascending=False)
    )
    top_months = month_contrib.head(3)["month"].astype(str).tolist()
    top_stocks = stock_contrib.head(3)["stock_code"].astype(str).tolist()
    cases.append({"case": "frag_remove_top1_month", "remove_months": top_months[:1], "remove_stocks": []})
    cases.append({"case": "frag_remove_top3_month", "remove_months": top_months, "remove_stocks": []})
    cases.append({"case": "frag_remove_top1_stock", "remove_months": [], "remove_stocks": top_stocks[:1]})
    cases.append({"case": "frag_remove_top3_stock", "remove_months": [], "remove_stocks": top_stocks})
    cases.append({"case": "frag_remove_2024", "remove_years": ["2024"], "remove_months": [], "remove_stocks": []})
    return cases


def write_sell_variant(source: pd.DataFrame, case: dict) -> dict:
    df = source.copy()
    df["holding_days"] = 1
    df["max_holding_days"] = int(case["max_holding_days"])
    df["score_exit_entry_ratio"] = f"{float(case['exit']):.5f}"
    df["score_continue_entry_ratio"] = f"{float(case['cont']):.5f}"
    df["min_holding_days_before_score_exit"] = 1
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "case_type": "sell_rule",
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
        "max_holding_days": int(case["max_holding_days"]),
        "score_exit_entry_ratio": float(case["exit"]),
        "score_continue_entry_ratio": float(case["cont"]),
    }


def write_fragility_variant(source: pd.DataFrame, case: dict) -> dict:
    df = source.copy()
    if case.get("remove_months"):
        df = df[~df["buy_date"].str.slice(0, 6).isin(case["remove_months"])].copy()
    if case.get("remove_years"):
        df = df[~df["buy_date"].str.slice(0, 4).isin(case["remove_years"])].copy()
    if case.get("remove_stocks"):
        df = df[~df["stock_code"].isin(case["remove_stocks"])].copy()
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "case_type": "fragility",
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()) if len(df) else 0.0,
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()) if len(df) else 0.0,
        "remove_months": ",".join(case.get("remove_months", [])),
        "remove_years": ",".join(case.get("remove_years", [])),
        "remove_stocks": ",".join(case.get("remove_stocks", [])),
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    return base_mod.run_juejin(row)


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = load_signal()
    proxy = add_proxy_returns(source)
    month_top = (
        proxy.groupby("month", as_index=False)["weighted_ret_h2"]
        .sum()
        .sort_values("weighted_ret_h2", ascending=False)
        .head(10)
        .to_dict("records")
    )
    stock_top = (
        proxy.groupby(["stock_code", "name"], as_index=False)["weighted_ret_h2"]
        .sum()
        .sort_values("weighted_ret_h2", ascending=False)
        .head(10)
        .to_dict("records")
    )
    manifest = [write_sell_variant(source, case) for case in SELL_CASES]
    manifest += [write_fragility_variant(source, case) for case in top_fragility_cases(proxy)]
    results = []
    for row in manifest:
        result = run_or_parse(row)
        results.append(result)
        pd.DataFrame(results).sort_values(
            ["case_type", "sharp_ratio", "pnl_ratio_annual"], ascending=[True, False, False], na_position="last"
        ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "type": result.get("case_type"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results)
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    summary = {
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "proxy_csv": str(PROXY_CSV),
        "top_proxy_months": month_top,
        "top_proxy_stocks": stock_top,
        "cases": int(len(results)),
        "target_hits": int(len(hits)),
        "best_by_sharpe": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).head(8).to_dict("records"),
        "target_hits_table": hits.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
