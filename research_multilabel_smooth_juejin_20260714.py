from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
PROXY_CSV = REPORT_DIR / "multilabel_smooth_proxy_search_20260714.csv"
CANDIDATE_PARQUET = REPORT_DIR / "active_l4_multilabel_pool_candidates_20260714.parquet"
SIGNAL_DIR = REPORT_DIR / "signals" / "multilabel_smooth_juejin"
LOG_DIR = REPORT_DIR / "logs" / "multilabel_smooth_juejin"
LOG_DIR_MP = REPORT_DIR / "logs" / "multilabel_smooth_juejin_maxpos"
OUT_CSV = REPORT_DIR / "multilabel_smooth_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "multilabel_smooth_juejin_results_20260714.json"
OUT_MP_CSV = REPORT_DIR / "multilabel_smooth_juejin_maxpos_results_20260714.csv"
OUT_MP_JSON = REPORT_DIR / "multilabel_smooth_juejin_maxpos_results_20260714.json"
SUMMARY_JSON = REPORT_DIR / "multilabel_smooth_juejin_summary_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


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


def case_from_row(row: pd.Series) -> dict:
    return {
        "case": str(row["case"]),
        "weights": {
            "p10": float(row["w10"]),
            "p5": float(row["w5"]),
            "p3": float(row["w3"]),
            "p1": float(row["w1"]),
        },
        "top_n": int(row["top_n"]),
        "p10_min": float(row["p10_min"]),
        "p5_min": float(row["p5_min"]),
        "p1_min": float(row["p1_min"]),
        "pct_min": float(row["pct_min"]),
        "pct_max": float(row["pct_max"]),
        "gap_min": float(row["gap_min"]),
        "gap_max": float(row["gap_max"]),
        "amount_min": float(row["amount_min"]),
        "mv_min": float(row["mv_min"]),
        "atr_max": float(row["atr_max"]),
        "hold": int(row["hold"]),
        "liq_bonus": float(row["liq_bonus"]),
        "cap": float(row["cap"]),
        "base_target": float(row["base_target"]),
        "high_score": float(row["high_score"]),
        "deep_pct": float(row["deep_pct"]),
        "high_target": float(row["high_target"]),
    }


def write_signal(candidates: pd.DataFrame, case: dict) -> dict:
    df = candidates.copy()
    w = case["weights"]
    df["entry_score"] = (
        float(w["p10"]) * df["pred_10d"].fillna(0.0)
        + float(w["p5"]) * df["pred_5d"].fillna(0.0)
        + float(w["p3"]) * df["pred_3d"].fillna(0.0)
        + float(w["p1"]) * df["pred_1d"].fillna(0.0)
    )
    mask = (
        (df["pred_10d"] >= case["p10_min"])
        & (df["pred_5d"] >= case["p5_min"])
        & (df["pred_1d"] >= case["p1_min"])
        & (df["signal_pct_chg_raw"] <= case["pct_max"])
        & (df["signal_pct_chg_raw"] >= case["pct_min"])
        & (df["amount"] >= case["amount_min"])
        & (df["total_mv"] >= case["mv_min"])
        & (df["atr_qfq"] <= case["atr_max"])
        & (df["buy_open_gap_raw_pct"] <= case["gap_max"])
        & (df["buy_open_gap_raw_pct"] >= case["gap_min"])
    )
    df = df.loc[mask].copy()
    df["liquidity_score"] = df["amount"].rank(pct=True) + df["total_mv"].rank(pct=True)
    df["sort_score"] = df["entry_score"] + case["liq_bonus"] * df["liquidity_score"]
    df = (
        df.sort_values(["buy_date", "sort_score", "pred_10d"], ascending=[True, False, False])
        .groupby("buy_date", group_keys=False)
        .head(case["top_n"])
        .copy()
    )
    base_target = min(case["cap"] / case["top_n"], case["base_target"])
    df["target_pct"] = base_target
    high = (df["entry_score"] >= case["high_score"]) & (df["signal_pct_chg_raw"] <= case["deep_pct"])
    df.loc[high, "target_pct"] = case["high_target"]
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    df["target_pct"] = df["target_pct"] * (case["cap"] / daily_sum).clip(upper=1.0)
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
    df["source_strategy_variant"] = "active_formal_l4_multilabel_smooth_proxy"
    df["filter_name"] = case["case"]
    df["entry_weight_name"] = json.dumps(case["weights"], ensure_ascii=False)
    df["dynamic_hold_name"] = f"h{case['hold']}m{max(int(case['hold']) + 1, 3)}_exit098_cont102"
    df["buy_day_market_available"] = df["buy_open_raw"].notna()
    df["buy_day_hard_gate_complete"] = df["buy_open_raw"].notna()
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["hybrid_source"] = "active_formal_l4_multilabel"
    df["feature_weight_scale"] = 1.0
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    out_cols = [
        "signal_date",
        "buy_date",
        "symbol",
        "stock_code",
        "name",
        "rank",
        "pred_prob",
        "entry_score",
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "target_pct",
        "holding_days",
        "max_holding_days",
        "score_exit_entry_ratio",
        "min_holding_days_before_score_exit",
        "score_continue_entry_ratio",
        "signal_stop_loss_pct",
        "signal_take_profit_pct",
        "strategy_variant",
        "source_strategy_variant",
        "filter_name",
        "entry_weight_name",
        "dynamic_hold_name",
        "buy_day_market_available",
        "buy_day_hard_gate_complete",
        "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected",
        "buy_open_gap_pct",
        "hybrid_source",
        "buy_open_gap_raw_pct",
        "feature_weight_scale",
        "daily_target_sum_after_cap",
        "sort_score",
    ]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(out, index=False, encoding="utf-8-sig")
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


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR_MP / f"{row['case']}_mp{row['max_positions']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(base_mod.RUNNER),
        "--strategy-dir",
        str(base_mod.STRATEGY_DIR),
        "--signal-file",
        row["signal_file"],
        "--log-file",
        str(log),
        "--score-db",
        str(base_mod.SCORE_DB),
        "--score-table",
        base_mod.SCORE_TABLE,
        "--market-db",
        str(base_mod.MARKET_DB),
        "--max-positions",
        str(row["max_positions"]),
        "--holding-days",
        str(row["holding_days"]),
        "--max-holding-days",
        str(max(int(row["holding_days"]) + 1, 3)),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    result = dict(row)
    result["returncode"] = proc.returncode
    result["log_file"] = str(log)
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
    if isinstance(ind, dict):
        result.update(ind)
    else:
        result["indicator_error"] = "missing"
        result["stdout"] = proc.stdout[-1000:]
        result["stderr"] = proc.stderr[-1000:]
    return result


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR_MP
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR_MP.mkdir(parents=True, exist_ok=True)
    proxy = pd.read_csv(PROXY_CSV, encoding="utf-8-sig").sort_values("objective", ascending=False)
    top = proxy.head(8)
    candidates = load_candidates()
    manifest = [write_signal(candidates, case_from_row(row)) for _, row in top.iterrows()]
    results = []
    for row in manifest:
        result = run_or_parse(row)
        results.append(result)
        pd.DataFrame(results).sort_values(
            ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
        ).to_csv(OUT_MP_CSV, index=False, encoding="utf-8-sig")
        OUT_MP_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
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
    summary = {
        "csv": str(OUT_MP_CSV),
        "json": str(OUT_MP_JSON),
        "cases": len(results),
        "best": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).to_dict("records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
