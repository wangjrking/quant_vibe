from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
CANDIDATE_PARQUET = REPORT_DIR / "active_l4_multilabel_pool_candidates_20260714.parquet"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SIGNAL_DIR = REPORT_DIR / "signals" / "independent_alpha_pockets"
LOG_DIR = REPORT_DIR / "logs" / "independent_alpha_pockets"
PROXY_CSV = REPORT_DIR / "independent_alpha_pockets_proxy_20260714.csv"
RESULT_CSV = REPORT_DIR / "independent_alpha_pockets_juejin_results_20260714.csv"
RESULT_JSON = REPORT_DIR / "independent_alpha_pockets_juejin_results_20260714.json"
SUMMARY_JSON = REPORT_DIR / "independent_alpha_pockets_summary_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


WEIGHTS = {
    "p10": (1.00, 0.00, 0.00, 0.00),
    "p10p1": (0.75, 0.00, 0.00, 0.25),
    "p10p5": (0.75, 0.25, 0.00, 0.00),
    "p10p5p1": (0.60, 0.20, 0.00, 0.20),
    "p5p1": (0.00, 0.55, 0.00, 0.45),
}


def load_candidates() -> pd.DataFrame:
    df = pd.read_parquet(CANDIDATE_PARQUET)
    for col in [
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "buy_open_gap_raw_pct",
        "buy_open_gap_pct",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_returns() -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        ret = con.execute(
            """
            with cal as (
              select trade_date,
                     lead(trade_date, 1) over(order by trade_date) as d1,
                     lead(trade_date, 2) over(order by trade_date) as d2,
                     lead(trade_date, 3) over(order by trade_date) as d3
              from (select distinct trade_date from STOCK_DAILY_DATA order by trade_date)
            )
            select
              b.stock_code,
              b.trade_date as buy_date,
              b.open as buy_open,
              e1.open as open_d1,
              e2.open as open_d2,
              e3.open as open_d3
            from STOCK_DAILY_DATA b
            join cal c on c.trade_date=b.trade_date
            left join STOCK_DAILY_DATA e1 on e1.trade_date=c.d1 and e1.stock_code=b.stock_code
            left join STOCK_DAILY_DATA e2 on e2.trade_date=c.d2 and e2.stock_code=b.stock_code
            left join STOCK_DAILY_DATA e3 on e3.trade_date=c.d3 and e3.stock_code=b.stock_code
            where b.trade_date between '20220607' and '20260714'
            """
        ).fetchdf()
    finally:
        con.close()
    ret["buy_date"] = ret["buy_date"].astype(str)
    for h in [1, 2, 3]:
        ret[f"ret_h{h}"] = ret[f"open_d{h}"] / ret["buy_open"] - 1.0
    return ret[["stock_code", "buy_date", "ret_h1", "ret_h2", "ret_h3"]]


def make_cases() -> list[dict]:
    cases = []
    pct_windows = [
        ("deep", -20.0, -8.0),
        ("mid", -8.0, -3.0),
        ("pull", -5.0, -1.5),
        ("broad", -12.0, -1.5),
    ]
    idx = 0
    for (w_name, weights), top_n, p10_min, p5_min, p1_min, pct, gap_max, amount_min, atr_max, hold in itertools.product(
        WEIGHTS.items(),
        [1, 2],
        [0.96, 0.98, 0.99],
        [0.00, 0.80],
        [0.00, 0.75, 0.90],
        pct_windows,
        [0.5, 1.5],
        [120000, 250000],
        [8.0, 12.0],
        [1, 2],
    ):
        if w_name == "p5p1" and p5_min < 0.80:
            continue
        idx += 1
        cases.append(
            {
                "case": f"iap_{idx:05d}_{w_name}_{pct[0]}_t{top_n}_h{hold}",
                "weights": weights,
                "top_n": top_n,
                "p10_min": p10_min,
                "p5_min": p5_min,
                "p1_min": p1_min,
                "pct_min": pct[1],
                "pct_max": pct[2],
                "gap_min": -8.0,
                "gap_max": gap_max,
                "amount_min": amount_min,
                "mv_min": 200000,
                "atr_max": atr_max,
                "hold": hold,
                "cap": 0.96,
                "base_target": 0.46 if top_n == 2 else 0.62,
                "high_target": 0.68,
                "high_score": 0.985,
            }
        )
    return cases


def build_signal(df: pd.DataFrame, case: dict) -> pd.DataFrame:
    work = df.copy()
    w10, w5, w3, w1 = case["weights"]
    work["entry_score"] = (
        w10 * work["pred_10d"].fillna(0.0)
        + w5 * work["pred_5d"].fillna(0.0)
        + w3 * work["pred_3d"].fillna(0.0)
        + w1 * work["pred_1d"].fillna(0.0)
    )
    mask = (
        (work["pred_10d"] >= case["p10_min"])
        & (work["pred_5d"] >= case["p5_min"])
        & (work["pred_1d"] >= case["p1_min"])
        & (work["signal_pct_chg_raw"] >= case["pct_min"])
        & (work["signal_pct_chg_raw"] <= case["pct_max"])
        & (work["buy_open_gap_raw_pct"] >= case["gap_min"])
        & (work["buy_open_gap_raw_pct"] <= case["gap_max"])
        & (work["amount"] >= case["amount_min"])
        & (work["total_mv"] >= case["mv_min"])
        & (work["atr_qfq"] <= case["atr_max"])
    )
    work = work.loc[mask].copy()
    if work.empty:
        return work
    work["sort_score"] = work["entry_score"] + 0.01 * work["amount"].rank(pct=True)
    work = (
        work.sort_values(["buy_date", "sort_score", "pred_10d"], ascending=[True, False, False])
        .groupby("buy_date", group_keys=False)
        .head(int(case["top_n"]))
        .copy()
    )
    work["target_pct"] = float(case["base_target"])
    high = (work["entry_score"] >= case["high_score"]) & (work["signal_pct_chg_raw"] <= -5.0)
    work.loc[high, "target_pct"] = float(case["high_target"])
    daily_sum = work.groupby("buy_date")["target_pct"].transform("sum")
    work["target_pct"] = work["target_pct"] * (float(case["cap"]) / daily_sum).clip(upper=1.0)
    return work


def proxy_score(signal: pd.DataFrame, case: dict) -> dict | None:
    if signal.empty:
        return None
    if signal["buy_date"].nunique() < 80:
        return None
    ret_col = f"ret_h{case['hold']}"
    signal = signal.dropna(subset=[ret_col]).copy()
    if signal.empty:
        return None
    signal["weighted_ret"] = signal["target_pct"] * signal[ret_col]
    daily = signal.groupby("buy_date", as_index=False).agg(
        daily_ret=("weighted_ret", "sum"),
        target_sum=("target_pct", "sum"),
        names=("stock_code", "count"),
    )
    daily["year"] = daily["buy_date"].str.slice(0, 4)
    by_year = daily.groupby("year")["daily_ret"].sum().to_dict()
    active_years = sum(1 for v in by_year.values() if abs(float(v)) > 0.01)
    if active_years < 3:
        return None
    total = float(daily["daily_ret"].sum())
    if total <= 0:
        return None
    std = daily["daily_ret"].std()
    sharpe = float(daily["daily_ret"].mean() / std * (244 ** 0.5)) if std else 0.0
    monthly = daily.assign(month=daily["buy_date"].str.slice(0, 6)).groupby("month")["daily_ret"].sum()
    top_month_share = float(monthly.max() / max(total, 1e-9))
    year_share = max(abs(float(v)) for v in by_year.values()) / max(abs(total), 1e-9)
    min_year = min(float(v) for v in by_year.values())
    recent60 = float(daily.tail(60)["daily_ret"].sum()) if len(daily) >= 60 else float(daily["daily_ret"].sum())
    objective = total + 0.8 * sharpe + 1.5 * min_year + 0.8 * recent60 - 3.0 * max(0, top_month_share - 0.25) - 2.0 * max(0, year_share - 0.55)
    return {
        "case": case["case"],
        "rows": int(len(signal)),
        "buy_days": int(daily["buy_date"].nunique()),
        "stock_count": int(signal["stock_code"].nunique()),
        "avg_names": float(daily["names"].mean()),
        "mean_target_sum": float(daily["target_sum"].mean()),
        "max_target_sum": float(daily["target_sum"].max()),
        "proxy_total_ret": total,
        "proxy_sharpe": sharpe,
        "proxy_recent60": recent60,
        "proxy_min_year": min_year,
        "proxy_top_month_share": top_month_share,
        "proxy_max_year_share": year_share,
        "objective": objective,
        **{k: v for k, v in case.items() if k != "weights"},
        "w10": case["weights"][0],
        "w5": case["weights"][1],
        "w3": case["weights"][2],
        "w1": case["weights"][3],
    }


def write_signal(signal: pd.DataFrame, case: dict) -> dict:
    df = signal.copy()
    df["rank"] = df.groupby("buy_date")["sort_score"].rank(method="first", ascending=False).astype(int)
    df["pred_prob"] = df["entry_score"]
    df["holding_days"] = int(case["hold"])
    df["max_holding_days"] = max(int(case["hold"]) + 1, 3)
    df["score_exit_entry_ratio"] = "0.98000"
    df["min_holding_days_before_score_exit"] = 1
    df["score_continue_entry_ratio"] = "1.02000"
    df["signal_stop_loss_pct"] = 0.05
    df["signal_take_profit_pct"] = 0.08
    df["strategy_variant"] = case["case"]
    df["source_strategy_variant"] = "active_formal_l4_independent_alpha_pocket"
    df["filter_name"] = case["case"]
    df["entry_weight_name"] = json.dumps({"w10": case["weights"][0], "w5": case["weights"][1], "w3": case["weights"][2], "w1": case["weights"][3]}, ensure_ascii=False)
    df["dynamic_hold_name"] = f"h{case['hold']}m{max(int(case['hold']) + 1, 3)}_exit098_cont102"
    df["buy_day_market_available"] = df["buy_open_raw"].notna()
    df["buy_day_hard_gate_complete"] = df["buy_open_raw"].notna()
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["hybrid_source"] = "active_formal_l4_independent_alpha"
    df["feature_weight_scale"] = 1.0
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "amount", "turnover_rate", "total_mv", "atr_qfq",
        "signal_pct_chg_raw", "target_pct", "holding_days", "max_holding_days", "score_exit_entry_ratio",
        "min_holding_days_before_score_exit", "score_continue_entry_ratio", "signal_stop_loss_pct",
        "signal_take_profit_pct", "strategy_variant", "source_strategy_variant", "filter_name",
        "entry_weight_name", "dynamic_hold_name", "buy_day_market_available", "buy_day_hard_gate_complete",
        "buy_day_st_rejected", "buy_day_open_limit_up_rejected", "buy_open_gap_pct", "hybrid_source",
        "buy_open_gap_raw_pct", "feature_weight_scale", "daily_target_sum_after_cap", "sort_score",
    ]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
        "max_positions": int(case["top_n"]),
        "holding_days": int(case["hold"]),
    }


def run_juejin(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}_mp{row['max_positions']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(base_mod.RUNNER),
        "--strategy-dir", str(base_mod.STRATEGY_DIR),
        "--signal-file", row["signal_file"],
        "--log-file", str(log),
        "--score-db", str(base_mod.SCORE_DB),
        "--score-table", base_mod.SCORE_TABLE,
        "--market-db", str(base_mod.MARKET_DB),
        "--max-positions", str(row["max_positions"]),
        "--holding-days", str(row["holding_days"]),
        "--max-holding-days", str(max(int(row["holding_days"]) + 1, 3)),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    ind = None
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
            ind = payload.get("indicator") or {}
            if isinstance(ind, str):
                ind = base_mod.parse_indicator_text(ind)
        except Exception:
            ind = base_mod.extract_indicator(text)
    else:
        ind = base_mod.extract_indicator(text)
    out = dict(row)
    out["returncode"] = proc.returncode
    out["log_file"] = str(log)
    if isinstance(ind, dict):
        out.update(ind)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def main() -> None:
    candidates = load_candidates()
    returns = load_returns()
    df = candidates.merge(returns, on=["stock_code", "buy_date"], how="left")
    proxy_rows = []
    signal_cache: dict[str, pd.DataFrame] = {}
    case_cache: dict[str, dict] = {}
    for case in make_cases():
        sig = build_signal(df, case)
        score = proxy_score(sig, case)
        if score is None:
            continue
        proxy_rows.append(score)
        signal_cache[case["case"]] = sig
        case_cache[case["case"]] = case
    proxy = pd.DataFrame(proxy_rows).sort_values("objective", ascending=False)
    proxy.to_csv(PROXY_CSV, index=False, encoding="utf-8-sig")
    selected = proxy.head(10)
    results = []
    for _, row in selected.iterrows():
        case = case_cache[str(row["case"])]
        sig = signal_cache[str(row["case"])]
        manifest = write_signal(sig, case)
        result = run_juejin(manifest)
        result.update({f"proxy_{k}": row[k] for k in ["objective", "proxy_total_ret", "proxy_sharpe", "proxy_recent60", "proxy_top_month_share", "proxy_max_year_share"]})
        results.append(result)
        pd.DataFrame(results).sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
        RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"case": result.get("case"), "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "open_count": result.get("open_count")}, ensure_ascii=False), flush=True)
    summary = {
        "proxy_csv": str(PROXY_CSV),
        "result_csv": str(RESULT_CSV),
        "result_json": str(RESULT_JSON),
        "proxy_rows": int(len(proxy)),
        "juejin_cases": int(len(results)),
        "best": pd.DataFrame(results).sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
