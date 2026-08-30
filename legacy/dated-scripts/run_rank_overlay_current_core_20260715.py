from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_full_refill_scale_smooth_20260715.py"

spec = importlib.util.spec_from_file_location("smooth_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SRC_REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
CANDIDATE_PARQUET = SRC_REPORT_DIR / "active_l4_multilabel_pool_candidates_20260714.parquet"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
CORE_SIGNALS = {
    "core_ogd_deep8": SRC_REPORT_DIR / "signals" / "latest_core_plus_tiny_refill" / "ogd_deep8_x2p0_rf03_empty_full.csv",
    "core_frs65": REPORT_DIR / "signals" / "full_refill_scale_smooth" / "frs_scale065_cap070_full.csv",
}
SIGNAL_DIR = REPORT_DIR / "signals" / "rank_overlay_current_core"
LOG_DIR = REPORT_DIR / "logs" / "rank_overlay_current_core_20260715"
PROXY_CSV = REPORT_DIR / "rank_overlay_current_core_proxy_20260715.csv"
OUT_CSV = REPORT_DIR / "rank_overlay_current_core_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "rank_overlay_current_core_juejin_20260715.json"
OUT_MD = REPORT_DIR / "rank_overlay_current_core_review_20260715.md"

OVERLAY_CASES = [
    {"name": "r10_990_r1_900_broad_t1", "top_n": 1, "r10": 0.990, "r1": 0.900, "pct_min": -12.0, "pct_max": -1.5, "gap_max": 0.5, "amount_min": 120000, "atr_max": 12.0, "target": 0.03},
    {"name": "r10_985_r1_900_broad_t1", "top_n": 1, "r10": 0.985, "r1": 0.900, "pct_min": -12.0, "pct_max": -1.5, "gap_max": 0.5, "amount_min": 120000, "atr_max": 12.0, "target": 0.03},
    {"name": "r10_980_r1_850_broad_t1", "top_n": 1, "r10": 0.980, "r1": 0.850, "pct_min": -12.0, "pct_max": -1.5, "gap_max": 1.0, "amount_min": 120000, "atr_max": 12.0, "target": 0.03},
    {"name": "r10_975_r1_800_broad_t1", "top_n": 1, "r10": 0.975, "r1": 0.800, "pct_min": -12.0, "pct_max": -1.5, "gap_max": 1.5, "amount_min": 120000, "atr_max": 12.0, "target": 0.03},
    {"name": "r10_990_r1_000_broad_t2", "top_n": 2, "r10": 0.990, "r1": 0.000, "pct_min": -12.0, "pct_max": -1.5, "gap_max": 1.5, "amount_min": 120000, "atr_max": 12.0, "target": 0.025},
    {"name": "r10_985_r1_000_broad_t2", "top_n": 2, "r10": 0.985, "r1": 0.000, "pct_min": -12.0, "pct_max": -1.5, "gap_max": 1.5, "amount_min": 120000, "atr_max": 12.0, "target": 0.025},
    {"name": "r10_990_r1_900_mid_t1", "top_n": 1, "r10": 0.990, "r1": 0.900, "pct_min": -8.0, "pct_max": -3.0, "gap_max": 0.5, "amount_min": 120000, "atr_max": 12.0, "target": 0.04},
    {"name": "r10_985_r1_850_mid_t1", "top_n": 1, "r10": 0.985, "r1": 0.850, "pct_min": -8.0, "pct_max": -3.0, "gap_max": 1.0, "amount_min": 120000, "atr_max": 12.0, "target": 0.04},
]


def to_symbol(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    return {"SH": "SHSE", "SZ": "SZSE"}.get(suffix, suffix) + "." + code


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
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["pred_1d", "pred_5d", "pred_10d"]:
        df[f"{col}_rank"] = df.groupby("signal_date")[col].rank(method="average", pct=True)
    return df


def load_returns() -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        ret = con.execute(
            """
            with cal as (
              select trade_date,
                     lead(trade_date, 1) over(order by trade_date) as d1,
                     lead(trade_date, 2) over(order by trade_date) as d2
              from (select distinct trade_date from STOCK_DAILY_DATA order by trade_date)
            )
            select b.stock_code, b.trade_date as buy_date, b.open as buy_open,
                   e1.open as open_d1, e2.open as open_d2
            from STOCK_DAILY_DATA b
            join cal c on c.trade_date=b.trade_date
            left join STOCK_DAILY_DATA e1 on e1.trade_date=c.d1 and e1.stock_code=b.stock_code
            left join STOCK_DAILY_DATA e2 on e2.trade_date=c.d2 and e2.stock_code=b.stock_code
            where b.trade_date between '20220607' and '20260714'
            """
        ).fetchdf()
    finally:
        con.close()
    ret["buy_date"] = ret["buy_date"].astype(str)
    ret["ret_h1"] = ret["open_d1"] / ret["buy_open"] - 1.0
    ret["ret_h2"] = ret["open_d2"] / ret["buy_open"] - 1.0
    return ret[["stock_code", "buy_date", "ret_h1", "ret_h2"]]


def build_overlay(candidates: pd.DataFrame, case: dict) -> pd.DataFrame:
    df = candidates[
        (candidates["pred_10d_rank"] >= float(case["r10"]))
        & (candidates["pred_1d_rank"] >= float(case["r1"]))
        & (candidates["signal_pct_chg_raw"] >= float(case["pct_min"]))
        & (candidates["signal_pct_chg_raw"] <= float(case["pct_max"]))
        & (candidates["buy_open_gap_raw_pct"] >= -8.0)
        & (candidates["buy_open_gap_raw_pct"] <= float(case["gap_max"]))
        & (candidates["amount"] >= float(case["amount_min"]))
        & (candidates["total_mv"] >= 200000)
        & (candidates["atr_qfq"] <= float(case["atr_max"]))
    ].copy()
    if df.empty:
        return df
    df["entry_score"] = 0.75 * df["pred_10d_rank"] + 0.25 * df["pred_1d_rank"]
    df["sort_score"] = df["entry_score"] + 0.01 * df["amount"].rank(method="average", pct=True)
    df = (
        df.sort_values(["buy_date", "sort_score", "pred_10d_rank"], ascending=[True, False, False])
        .groupby("buy_date", group_keys=False)
        .head(int(case["top_n"]))
        .copy()
    )
    df["target_pct"] = float(case["target"])
    df["symbol"] = df["stock_code"].map(to_symbol)
    return df


def score_overlay(overlay: pd.DataFrame, returns: pd.DataFrame) -> dict:
    if overlay.empty:
        return {"overlay_rows": 0, "overlay_buy_days": 0, "proxy_ret": 0.0, "proxy_sharpe": 0.0, "latest": None}
    df = overlay.merge(returns, on=["stock_code", "buy_date"], how="left")
    df["weighted_ret"] = df["target_pct"] * df["ret_h2"]
    daily = df.dropna(subset=["weighted_ret"]).groupby("buy_date", as_index=False).agg(daily_ret=("weighted_ret", "sum"))
    if daily.empty:
        return {"overlay_rows": int(len(overlay)), "overlay_buy_days": int(overlay["buy_date"].nunique()), "proxy_ret": 0.0, "proxy_sharpe": 0.0, "latest": str(overlay["buy_date"].max())}
    std = daily["daily_ret"].std()
    return {
        "overlay_rows": int(len(overlay)),
        "overlay_buy_days": int(overlay["buy_date"].nunique()),
        "proxy_ret": float(daily["daily_ret"].sum()),
        "proxy_sharpe": float(daily["daily_ret"].mean() / std * (244 ** 0.5)) if std else 0.0,
        "proxy_recent60": float(daily.tail(60)["daily_ret"].sum()),
        "latest": str(overlay["buy_date"].max()),
    }


def prepare_signal_columns(df: pd.DataFrame, case_name: str) -> pd.DataFrame:
    out = df.copy()
    out["rank"] = out.groupby("buy_date")["target_pct"].rank(method="first", ascending=False).astype(int)
    out["pred_prob"] = out.get("entry_score", out.get("pred_prob", out.get("pred_10d_rank", 0.0)))
    out["entry_score"] = out["pred_prob"]
    out["holding_days"] = 1
    out["max_holding_days"] = 3
    out["score_exit_entry_ratio"] = "0.98000"
    out["min_holding_days_before_score_exit"] = 1
    out["score_continue_entry_ratio"] = "1.02000"
    out["signal_stop_loss_pct"] = 0.05
    out["signal_take_profit_pct"] = 0.08
    out["strategy_variant"] = case_name
    out["source_strategy_variant"] = "current_core_plus_rank_overlay"
    out["filter_name"] = case_name
    out["entry_weight_name"] = "core_first_rank_overlay"
    out["dynamic_hold_name"] = "h1_mh3_exit098"
    out["buy_day_market_available"] = True
    out["buy_day_hard_gate_complete"] = True
    out["buy_day_st_rejected"] = False
    out["buy_day_open_limit_up_rejected"] = False
    out["buy_open_gap_pct"] = out["buy_open_gap_raw_pct"]
    out["hybrid_source"] = "active_formal_l4_rank_overlay"
    out["feature_weight_scale"] = 1.0
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    cols = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "amount", "turnover_rate", "total_mv", "atr_qfq",
        "signal_pct_chg_raw", "target_pct", "holding_days", "max_holding_days", "score_exit_entry_ratio",
        "min_holding_days_before_score_exit", "score_continue_entry_ratio", "signal_stop_loss_pct",
        "signal_take_profit_pct", "strategy_variant", "source_strategy_variant", "filter_name",
        "entry_weight_name", "dynamic_hold_name", "buy_day_market_available", "buy_day_hard_gate_complete",
        "buy_day_st_rejected", "buy_day_open_limit_up_rejected", "buy_open_gap_pct", "hybrid_source",
        "buy_open_gap_raw_pct", "feature_weight_scale", "daily_target_sum_after_cap",
    ]
    for col in cols:
        if col not in out.columns:
            out[col] = None
    return out[cols].copy()


def build_combined(core_name: str, core_path: Path, overlay_name: str, overlay: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict]:
    core = pd.read_csv(core_path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    core["layer"] = "core"
    overlay = overlay.copy()
    overlay["layer"] = "overlay"
    core_days = set(core["buy_date"].astype(str))
    if mode == "empty":
        overlay = overlay[~overlay["buy_date"].astype(str).isin(core_days)].copy()
    elif mode == "lowsum":
        core_sum = core.groupby("buy_date")["target_pct"].sum().rename("core_sum")
        overlay = overlay.merge(core_sum, on="buy_date", how="left")
        overlay["core_sum"] = overlay["core_sum"].fillna(0.0)
        overlay = overlay[overlay["core_sum"] < 0.45].copy()
    core_keys = set(zip(core["buy_date"], core["stock_code"]))
    if len(overlay):
        overlay = overlay[~overlay.apply(lambda r: (r["buy_date"], r["stock_code"]) in core_keys, axis=1)].copy()
    combined = pd.concat([core, overlay], ignore_index=True, sort=False)
    case_name = f"{core_name}_{overlay_name}_{mode}"
    combined = prepare_signal_columns(combined, case_name)
    daily = combined.groupby("buy_date")["target_pct"].sum()
    meta = {
        "case": case_name,
        "core": core_name,
        "overlay": overlay_name,
        "mode": mode,
        "rows": int(len(combined)),
        "core_rows": int(len(core)),
        "overlay_rows": int(len(overlay)),
        "buy_days": int(combined["buy_date"].nunique()),
        "overlay_buy_days": int(overlay["buy_date"].nunique()) if len(overlay) else 0,
        "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
        "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
        "max_positions": int(max(combined.groupby("buy_date")["stock_code"].count().max(), 1)),
        "max_buy_date": str(combined["buy_date"].max()) if len(combined) else None,
    }
    return combined, meta


def run_juejin(signal_file: Path, log_file: Path, max_positions: int) -> dict:
    cmd = [
        sys.executable,
        str(base.RUNNER),
        "--strategy-dir", str(base.STRATEGY_DIR),
        "--signal-file", str(signal_file),
        "--log-file", str(log_file),
        "--max-positions", str(max_positions),
        "--holding-days", "1",
        "--max-holding-days", "3",
        "--score-db", str(base.SCORE_DB),
        "--score-table", base.SCORE_TABLE,
        "--market-db", str(base.MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    out = {"returncode": int(proc.returncode), "log_file": str(log_file)}
    out.update(base.parse_metrics(text))
    return out


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def num(value: object, default: float = 0.0) -> float:
    if value is None or value == "" or pd.isna(value):
        return default
    return float(value)


def integer(value: object, default: int = 0) -> int:
    if value is None or value == "" or pd.isna(value):
        return default
    return int(float(value))


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    returns = load_returns()
    overlay_rows = []
    overlays = {}
    for case in OVERLAY_CASES:
        overlay = build_overlay(candidates, case)
        score = score_overlay(overlay, returns)
        score.update(case)
        overlay_rows.append(score)
        overlays[case["name"]] = overlay
    proxy = pd.DataFrame(overlay_rows).sort_values(["proxy_sharpe", "proxy_ret"], ascending=[False, False])
    proxy.to_csv(PROXY_CSV, index=False, encoding="utf-8-sig")
    selected_names = list(proxy.head(4)["name"])

    results = []
    for core_name, core_path in CORE_SIGNALS.items():
        for overlay_name in selected_names:
            for mode in ["empty", "lowsum"]:
                combined, meta = build_combined(core_name, core_path, overlay_name, overlays[overlay_name], mode)
                out_path = SIGNAL_DIR / f"{meta['case']}.csv"
                combined.to_csv(out_path, index=False, encoding="utf-8-sig")
                log_file = LOG_DIR / f"{meta['case']}_mp{meta['max_positions']}.log"
                result = run_juejin(out_path, log_file, int(meta["max_positions"]))
                result.update(meta)
                result["signal_file"] = str(out_path)
                results.append(result)
                print(json.dumps({"case": meta["case"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "buy_days": meta["buy_days"]}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    best = frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 当前核心叠加排名阈值 overlay 验证 20260715",
        "",
        "## 口径",
        "",
        "- 修正旧 overlay 的绝对分数阈值，改为每日 rank 阈值。",
        "- 核心信号使用当前覆盖到 20260714 的 full 信号。",
        "- 本轮为 research-only，正式指标以掘金日志为准。",
        "",
        "## 结果",
        "",
        "| 版本 | 年化 | Sharpe | 最大回撤 | 买入日 | overlay行 | 开仓 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row.get('pnl_ratio_annual'))} | {num(row.get('sharp_ratio')):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {integer(row.get('buy_days'))} | {integer(row.get('overlay_rows'))} | "
            f"{integer(row.get('open_count'))} | {row.get('max_buy_date')} |"
        )
    lines.extend(["", "## 证据路径", "", f"- overlay代理：`{PROXY_CSV}`", f"- 掘金结果：`{OUT_CSV}`", f"- 结果 JSON：`{OUT_JSON}`", f"- 信号目录：`{SIGNAL_DIR}`", f"- 掘金日志目录：`{LOG_DIR}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
