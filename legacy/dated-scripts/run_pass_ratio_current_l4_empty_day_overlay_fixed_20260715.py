from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_open_verified_scale_fine_20260715.py"

spec = importlib.util.spec_from_file_location("scale_fine", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SOURCE_REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
CORE_SIGNAL = REPORT_DIR / "signals" / "recalc_hardgate_scale" / "rh_top1_s120_cap065_full.csv"
CANDIDATE_POOL = SOURCE_REPORT_DIR / "active_l4_multilabel_pool_candidates_20260714.parquet"

OUT_DIR = REPORT_DIR / "signals" / "current_l4_empty_day_overlay_fixed"
LOG_DIR = REPORT_DIR / "logs" / "current_l4_empty_day_overlay_fixed_20260715"
OUT_CSV = REPORT_DIR / "current_l4_empty_day_overlay_fixed_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "current_l4_empty_day_overlay_fixed_juejin_20260715.json"
OUT_MD = REPORT_DIR / "current_l4_empty_day_overlay_fixed_review_20260715.md"

TRADE_DAYS = 997

CASES = [
    {"case": "edo_fixed_broad_pos05", "target": 0.05, "top_n": 1, "p10_min": 0.94, "amount_min": 120000, "turn_min": 0.0, "pct_lo": -12.0, "pct_hi": -1.5, "gap_lo": -8.0, "gap_hi": 1.5, "atr_max": 15.0},
    {"case": "edo_fixed_broad_pos08", "target": 0.08, "top_n": 1, "p10_min": 0.94, "amount_min": 120000, "turn_min": 0.0, "pct_lo": -12.0, "pct_hi": -1.5, "gap_lo": -8.0, "gap_hi": 1.5, "atr_max": 15.0},
    {"case": "edo_fixed_broad_pos12", "target": 0.12, "top_n": 1, "p10_min": 0.94, "amount_min": 120000, "turn_min": 0.0, "pct_lo": -12.0, "pct_hi": -1.5, "gap_lo": -8.0, "gap_hi": 1.5, "atr_max": 15.0},
    {"case": "edo_fixed_liq_pos08", "target": 0.08, "top_n": 1, "p10_min": 0.94, "amount_min": 250000, "turn_min": 1.5, "pct_lo": -12.0, "pct_hi": -1.5, "gap_lo": -8.0, "gap_hi": 1.5, "atr_max": 15.0},
    {"case": "edo_fixed_liq_pos12", "target": 0.12, "top_n": 1, "p10_min": 0.94, "amount_min": 250000, "turn_min": 1.5, "pct_lo": -12.0, "pct_hi": -1.5, "gap_lo": -8.0, "gap_hi": 1.5, "atr_max": 15.0},
    {"case": "edo_fixed_strict_pos08", "target": 0.08, "top_n": 1, "p10_min": 0.97, "amount_min": 150000, "turn_min": 1.5, "pct_lo": -8.0, "pct_hi": -1.5, "gap_lo": -5.0, "gap_hi": 0.5, "atr_max": 12.0},
    {"case": "edo_fixed_strict_pos12", "target": 0.12, "top_n": 1, "p10_min": 0.97, "amount_min": 150000, "turn_min": 1.5, "pct_lo": -8.0, "pct_hi": -1.5, "gap_lo": -5.0, "gap_hi": 0.5, "atr_max": 12.0},
    {"case": "edo_fixed_top2_pos05", "target": 0.05, "top_n": 2, "p10_min": 0.94, "amount_min": 250000, "turn_min": 1.5, "pct_lo": -12.0, "pct_hi": -1.5, "gap_lo": -8.0, "gap_hi": 1.5, "atr_max": 15.0},
]


def to_symbol(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".", 1)
    return {"SH": "SHSE", "SZ": "SZSE"}.get(suffix, suffix) + "." + code


def load_core() -> pd.DataFrame:
    core = pd.read_csv(CORE_SIGNAL, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    core["target_pct"] = pd.to_numeric(core["target_pct"], errors="coerce").fillna(0.0)
    core["coverage_layer"] = "core"
    return core


def load_pool(core_dates: set[str]) -> pd.DataFrame:
    pool = pd.read_parquet(CANDIDATE_POOL)
    pool["signal_date"] = pool["signal_date"].astype(str)
    pool["buy_date"] = pool["buy_date"].astype(str)
    pool = pool[~pool["buy_date"].isin(core_dates)].copy()
    pool = pool[~pool["stock_code"].astype(str).str.endswith(".BJ")].copy()
    pool = pool[pool["ST_TYPE"].fillna("").astype(str).isin(["", "0", "0.0", "None", "nan"])].copy()
    pool = pool[~pool["name"].fillna("").astype(str).str.startswith(("ST", "*ST", "退"))].copy()
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
        pool[col] = pd.to_numeric(pool[col], errors="coerce")
    pool["pred_1d_rank"] = pool.groupby("signal_date")["pred_1d"].rank(method="average", pct=True)
    pool["pred_10d_rank"] = pool.groupby("signal_date")["pred_10d"].rank(method="average", pct=True)
    pool["overlay_score"] = 0.75 * pool["pred_10d_rank"] + 0.25 * pool["pred_1d_rank"]
    return pool


def build_overlay(pool: pd.DataFrame, case: dict) -> pd.DataFrame:
    mask = (
        (pool["pred_10d"] >= float(case["p10_min"]))
        & (pool["amount"] >= float(case["amount_min"]))
        & (pool["turnover_rate"] >= float(case["turn_min"]))
        & (pool["atr_qfq"] <= float(case["atr_max"]))
        & (pool["signal_pct_chg_raw"] >= float(case["pct_lo"]))
        & (pool["signal_pct_chg_raw"] <= float(case["pct_hi"]))
        & (pool["buy_open_gap_raw_pct"] >= float(case["gap_lo"]))
        & (pool["buy_open_gap_raw_pct"] <= float(case["gap_hi"]))
    )
    selected = pool.loc[mask].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(["buy_date", "overlay_score", "pred_10d", "amount"], ascending=[True, False, False, False])
    selected = selected.groupby("buy_date", group_keys=False).head(int(case["top_n"])).copy()
    selected["rank"] = selected.groupby("buy_date").cumcount() + 1
    selected["target_pct"] = float(case["target"])
    selected["coverage_layer"] = "empty_day_overlay"
    selected["symbol"] = selected["stock_code"].map(to_symbol)
    selected["pred_prob"] = selected["overlay_score"]
    selected["entry_score"] = selected["overlay_score"]
    selected["holding_days"] = 1
    selected["max_holding_days"] = 3
    selected["score_exit_entry_ratio"] = "0.98000"
    selected["min_holding_days_before_score_exit"] = 1
    selected["score_continue_entry_ratio"] = "1.02000"
    selected["signal_stop_loss_pct"] = 0.05
    selected["signal_take_profit_pct"] = 0.08
    selected["source_strategy_variant"] = "active_l4_empty_day_overlay_fixed"
    selected["entry_weight_name"] = "0.75_10d_rank_plus_0.25_1d_rank"
    selected["dynamic_hold_name"] = "h1_mh3_score_exit098"
    selected["buy_day_market_available"] = True
    selected["buy_day_hard_gate_complete"] = True
    selected["buy_day_st_rejected"] = False
    selected["buy_day_open_limit_up_rejected"] = False
    selected["latest_market_date"] = "20260714"
    selected["buy_open_gap_pct"] = selected["buy_open_gap_raw_pct"]
    selected["hybrid_source"] = "current_active_formal_l4_empty_day_overlay"
    selected["feature_weight_scale"] = 1.0
    selected["daily_target_sum_after_cap"] = selected.groupby("buy_date")["target_pct"].transform("sum")
    return selected


def build_signal(core: pd.DataFrame, overlay: pd.DataFrame, case: dict) -> pd.DataFrame:
    core_norm = core.copy()
    overlay_norm = overlay.copy()
    combined = pd.concat([core_norm, overlay_norm], ignore_index=True, sort=False)
    combined["strategy_variant"] = case["case"]
    combined["filter_name"] = case["case"]
    combined = combined.sort_values(["buy_date", "coverage_layer", "rank"]).copy()
    combined["rank"] = combined.groupby("buy_date").cumcount() + 1
    combined["daily_target_sum_after_cap"] = combined.groupby("buy_date")["target_pct"].transform("sum")
    cols = [
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
        "latest_market_date",
        "buy_open_gap_pct",
        "hybrid_source",
        "buy_open_gap_raw_pct",
        "feature_weight_scale",
        "daily_target_sum_after_cap",
        "coverage_layer",
    ]
    for col in cols:
        if col not in combined.columns:
            combined[col] = None
    return combined[cols].copy()


def slice_frame(frame: pd.DataFrame, start: str | None) -> pd.DataFrame:
    if start == "__recent60__":
        dates = sorted(frame["buy_date"].dropna().astype(str).unique())
        keep = set(dates[-60:])
        return frame[frame["buy_date"].astype(str).isin(keep)].copy()
    if start:
        return frame[frame["buy_date"].astype(str) >= start].copy()
    return frame.copy()


def run_juejin(path: Path, case: str, slice_name: str, max_positions: int) -> dict:
    log_file = LOG_DIR / f"{case}_{slice_name}.log"
    cmd = [
        sys.executable,
        str(base.RUNNER),
        "--strategy-dir",
        str(base.STRATEGY_DIR),
        "--signal-file",
        str(path),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(max_positions),
        "--score-db",
        str(base.SCORE_DB),
        "--score-table",
        base.SCORE_TABLE,
        "--market-db",
        str(base.MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    metrics = base.parse_metrics(text)
    return {
        "case": case,
        "slice": slice_name,
        "signal_file": str(path),
        "log_file": str(log_file),
        "returncode": proc.returncode,
        **metrics,
    }


def pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    core = load_core()
    pool = load_pool(set(core["buy_date"].astype(str)))
    rows: list[dict] = []
    for case in CASES:
        overlay = build_overlay(pool, case)
        signal = build_signal(core, overlay, case)
        max_positions = int(signal.groupby("buy_date")["stock_code"].count().max()) if len(signal) else 1
        for slice_name, start in base.SLICES:
            sliced = slice_frame(signal, start)
            path = OUT_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(path, index=False, encoding="utf-8-sig")
            result = run_juejin(path, case["case"], slice_name, max_positions)
            result.update(
                {
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "trade_day_coverage": int(sliced["buy_date"].nunique()) / TRADE_DAYS if slice_name == "full" else None,
                    "overlay_rows": int((sliced["coverage_layer"] == "empty_day_overlay").sum()) if len(sliced) else 0,
                    "overlay_buy_days": int(sliced.loc[sliced["coverage_layer"] == "empty_day_overlay", "buy_date"].nunique()) if len(sliced) else 0,
                    "latest_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                    "max_positions": max_positions,
                    **case,
                }
            )
            rows.append(result)
            print(json.dumps({"case": case["case"], "slice": slice_name, "buy_days": result["buy_days"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    full = frame[frame["slice"] == "full"].sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=False, na_position="last")
    lines = [
        "# 当前 L4 空窗覆盖层固定邻域复核 20260715",
        "",
        "## 结论表",
        "",
        "| 版本 | 覆盖率 | 买入日 | 覆盖层买入日 | 年化 | Sharpe | 最大回撤 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in full.iterrows():
        lines.append(
            f"| `{row['case']}` | {pct(row.get('trade_day_coverage'))} | {int(row.get('buy_days', 0))} | "
            f"{int(row.get('overlay_buy_days', 0))} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | {row.get('latest_buy_date')} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 掘金结果 CSV：`{OUT_CSV}`",
            f"- 掘金结果 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{OUT_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
