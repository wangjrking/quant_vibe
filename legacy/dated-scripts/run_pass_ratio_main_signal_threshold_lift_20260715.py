from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SOURCE_REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
POOL_PATH = SOURCE_REPORT_DIR / "active_l4_wide_buy_quality_base_pool_20260714.parquet"
SIGNAL_DIR = REPORT_DIR / "signals" / "main_signal_threshold_lift"
LOG_DIR = REPORT_DIR / "logs" / "main_signal_threshold_lift_20260715"
OUT_CSV = REPORT_DIR / "main_signal_threshold_lift_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "main_signal_threshold_lift_juejin_20260715.json"
OUT_MD = REPORT_DIR / "main_signal_threshold_lift_review_20260715.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {
        "case": "ms_core_p175_p10_70_gap15_pos65",
        "pct_max": -1.75,
        "pred10_min": 0.70,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.65,
        "mild_pos": 0.35,
        "deep_pos": 0.65,
    },
    {
        "case": "ms_lift_p150_p10_90_gap15_pos55",
        "pct_max": -1.50,
        "pred10_min": 0.90,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.55,
        "mild_pos": 0.28,
        "deep_pos": 0.55,
    },
    {
        "case": "ms_lift_p125_p10_95_gap15_pos50",
        "pct_max": -1.25,
        "pred10_min": 0.95,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.50,
        "mild_pos": 0.22,
        "deep_pos": 0.50,
    },
    {
        "case": "ms_lift_p100_p10_98_gap15_pos42",
        "pct_max": -1.00,
        "pred10_min": 0.98,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.42,
        "mild_pos": 0.16,
        "deep_pos": 0.42,
    },
    {
        "case": "ms_lift_p075_p10_99_gap15_pos35",
        "pct_max": -0.75,
        "pred10_min": 0.99,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.35,
        "mild_pos": 0.12,
        "deep_pos": 0.35,
    },
    {
        "case": "ms_lift_p125_p10_95_gap25_pos45",
        "pct_max": -1.25,
        "pred10_min": 0.95,
        "gap_max": 2.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.45,
        "mild_pos": 0.18,
        "deep_pos": 0.45,
    },
    {
        "case": "ms_liq_p125_p10_95_gap15_amt20_pos48",
        "pct_max": -1.25,
        "pred10_min": 0.95,
        "gap_max": 1.5,
        "amount_min": 200000,
        "mv_min": 300000,
        "pos_base": 0.48,
        "mild_pos": 0.20,
        "deep_pos": 0.48,
    },
    {
        "case": "ms_soft_p050_p10_995_gap10_pos25",
        "pct_max": -0.50,
        "pred10_min": 0.995,
        "gap_max": 1.0,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.25,
        "mild_pos": 0.08,
        "deep_pos": 0.25,
    },
    {
        "case": "ms_mid_p250_p10_99_p1_75_gap15_pos50",
        "pct_max": -2.50,
        "pred10_min": 0.99,
        "pred1_min": 0.75,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.50,
        "mild_pos": 0.22,
        "deep_pos": 0.50,
    },
    {
        "case": "ms_mid_p200_p10_995_gap15_pos45",
        "pct_max": -2.00,
        "pred10_min": 0.995,
        "pred1_min": None,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.45,
        "mild_pos": 0.18,
        "deep_pos": 0.45,
    },
    {
        "case": "ms_mid_p175_p10_995_p1_75_gap15_pos42",
        "pct_max": -1.75,
        "pred10_min": 0.995,
        "pred1_min": 0.75,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.42,
        "mild_pos": 0.16,
        "deep_pos": 0.42,
    },
    {
        "case": "ms_mid_p125_p10_99_p1_99_gap15_pos32",
        "pct_max": -1.25,
        "pred10_min": 0.99,
        "pred1_min": 0.99,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.32,
        "mild_pos": 0.10,
        "deep_pos": 0.32,
    },
    {
        "case": "ms_mid_p075_p10_995_p1_90_gap15_pos30",
        "pct_max": -0.75,
        "pred10_min": 0.995,
        "pred1_min": 0.90,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.30,
        "mild_pos": 0.09,
        "deep_pos": 0.30,
    },
    {
        "case": "ms_rank_p175_r10_95_gap15_pos45",
        "pct_max": -1.75,
        "pred10_min": None,
        "pred10_rank_min": 0.95,
        "pred1_rank_min": None,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.45,
        "mild_pos": 0.18,
        "deep_pos": 0.45,
    },
    {
        "case": "ms_rank_p125_r10_97_gap15_pos38",
        "pct_max": -1.25,
        "pred10_min": None,
        "pred10_rank_min": 0.97,
        "pred1_rank_min": None,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.38,
        "mild_pos": 0.14,
        "deep_pos": 0.38,
    },
    {
        "case": "ms_rank_p100_r10_98_r1_60_gap15_pos32",
        "pct_max": -1.00,
        "pred10_min": None,
        "pred10_rank_min": 0.98,
        "pred1_rank_min": 0.60,
        "gap_max": 1.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.32,
        "mild_pos": 0.10,
        "deep_pos": 0.32,
    },
    {
        "case": "ms_rank_p075_r10_99_r1_70_gap10_pos25",
        "pct_max": -0.75,
        "pred10_min": None,
        "pred10_rank_min": 0.99,
        "pred1_rank_min": 0.70,
        "gap_max": 1.0,
        "amount_min": 120000,
        "mv_min": 200000,
        "pos_base": 0.25,
        "mild_pos": 0.08,
        "deep_pos": 0.25,
    },
]


def to_symbol(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    return {"SH": "SHSE", "SZ": "SZSE"}.get(suffix, suffix) + "." + code


def add_scores(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    for col in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        out[f"{col}_rank"] = out.groupby("signal_date")[col].rank(method="average", pct=True)
    out["blend_score"] = (
        0.05 * out["pred_1d_rank"]
        + 0.10 * out["pred_3d_rank"]
        + 0.20 * out["pred_5d_rank"]
        + 0.65 * out["pred_10d_rank"]
    )
    return out


def build_signal(pool: pd.DataFrame, case: dict) -> dict:
    df = pool[
        (pool["signal_pct_chg_raw"] <= float(case["pct_max"]))
        & (pool["buy_open_gap_raw_pct"] <= float(case["gap_max"]))
        & (pool["buy_open_gap_raw_pct"] >= -8.0)
        & (pool["amount"] >= float(case["amount_min"]))
        & (pool["total_mv"] >= float(case["mv_min"]))
        & (~pool["buy_open_limit_up_rejected"])
    ].copy()
    if case.get("pred10_min") is not None:
        df = df[df["pred_10d"] >= float(case["pred10_min"])].copy()
    if case.get("pred10_rank_min") is not None:
        df = df[df["pred_10d_rank"] >= float(case["pred10_rank_min"])].copy()
    if case.get("pred1_min") is not None:
        df = df[df["pred_1d"] >= float(case["pred1_min"])].copy()
    if case.get("pred1_rank_min") is not None:
        df = df[df["pred_1d_rank"] >= float(case["pred1_rank_min"])].copy()
    df = df.sort_values(["buy_date", "blend_score", "pred_10d", "amount"], ascending=[True, False, False, False])
    selected = df.groupby("buy_date", group_keys=False).head(1).copy()
    selected["rank"] = 1
    pct = pd.to_numeric(selected["signal_pct_chg_raw"], errors="coerce")
    gap = pd.to_numeric(selected["buy_open_gap_raw_pct"], errors="coerce")
    target = pd.Series(float(case["pos_base"]), index=selected.index)
    target.loc[pct > -1.75] = float(case["mild_pos"])
    target.loc[pct <= -5.0] = float(case["deep_pos"])
    target.loc[gap > 1.0] *= 0.75
    selected["target_pct"] = target.clip(lower=0.0, upper=0.65)
    selected = selected[selected["target_pct"] > 0].copy()

    selected["symbol"] = selected["stock_code"].map(to_symbol)
    selected["pred_prob"] = selected["blend_score"]
    selected["entry_score"] = selected["blend_score"]
    selected["atr_qfq"] = None
    selected["holding_days"] = 1
    selected["max_holding_days"] = 3
    selected["score_exit_entry_ratio"] = "0.98000"
    selected["min_holding_days_before_score_exit"] = 1
    selected["score_continue_entry_ratio"] = "1.02000"
    selected["signal_stop_loss_pct"] = 0.05
    selected["signal_take_profit_pct"] = 0.08
    selected["strategy_variant"] = case["case"]
    selected["source_strategy_variant"] = "active_l4_formal_main_signal_threshold_lift"
    selected["filter_name"] = case["case"]
    selected["entry_weight_name"] = "rank_blend_active_l4_top1_no_refill"
    selected["dynamic_hold_name"] = "h1_mh3_score_continue"
    selected["buy_day_market_available"] = True
    selected["buy_day_hard_gate_complete"] = True
    selected["buy_day_st_rejected"] = False
    selected["buy_day_open_limit_up_rejected"] = False
    selected["latest_market_date"] = "20260714"
    selected["buy_open_gap_pct"] = selected["buy_open_gap_raw_pct"]
    selected["hybrid_source"] = "active_l4_formal_duckdb_main_signal_no_refill"
    selected["feature_weight_scale"] = 1.0
    selected["daily_target_sum_after_cap"] = selected["target_pct"]

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
    ]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    selected[cols].to_csv(out, index=False, encoding="utf-8-sig")
    return {
        **case,
        "signal_file": str(out),
        "rows": int(len(selected)),
        "buy_days": int(selected["buy_date"].nunique()) if len(selected) else 0,
        "stock_count": int(selected["stock_code"].nunique()) if len(selected) else 0,
        "mean_daily_target_sum": float(selected["target_pct"].mean()) if len(selected) else 0.0,
        "max_buy": str(selected["buy_date"].max()) if len(selected) else None,
    }


def run_or_parse(row: dict) -> dict:
    old_signal_dir = base_mod.SIGNAL_DIR
    old_log_dir = base_mod.LOG_DIR
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    try:
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
    finally:
        base_mod.SIGNAL_DIR = old_signal_dir
        base_mod.LOG_DIR = old_log_dir


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(POOL_PATH)
    pool = add_scores(pool)
    all_buy_days = int(pool["buy_date"].nunique())
    manifest = [build_signal(pool, case) for case in CASES]
    results = []
    for row in manifest:
        result = run_or_parse(row)
        result["all_buy_days"] = all_buy_days
        result["buy_day_coverage"] = result["buy_days"] / all_buy_days if all_buy_days else None
        results.append(result)
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "buy_days": result.get("buy_days"),
                    "coverage": result.get("buy_day_coverage"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    )
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 主信号通过比例提升复核 20260715",
        "",
        "## 口径",
        "",
        "- 只使用 active formal L4 DuckDB 与 L2 未复权执行行情派生池。",
        "- 不做候选补位；每天最多从主排序里取 1 只，通过就是信号，不通过就是无信号。",
        "- 调参只放宽信号日通过条件：`pct_chg`、`pred_10d`、买入日开盘缺口、流动性门槛和轻信号仓位。",
        "- 本轮为 research-only，不修改生产策略。",
        "",
        "## 掘金结果",
        "",
        "| 版本 | 买入日 | 覆盖率 | 年化 | Sharpe | 最大回撤 | 开仓 | 平仓 | 均值仓位 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {int(row['buy_days'])} | {pct(row.get('buy_day_coverage'))} | "
            f"{pct(row.get('pnl_ratio_annual'))} | {float(row.get('sharp_ratio') or 0):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {int(row.get('open_count') or 0)} | "
            f"{int(row.get('close_count') or 0)} | {pct(row.get('mean_daily_target_sum'))} | {row.get('max_buy')} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 覆盖率可以从核心版约 21% 提高到更高区间，但收益和 Sharpe 会快速下降。",
            "- 在不补位、只提高主信号通过率的前提下，当前还没有找到同时满足 500% 年化、Sharpe 4、低偶然性的放宽版本。",
            "- 更宽通过率版本可作为平滑化研究，但不能替代当前高收益核心版。",
            "",
            "## 证据路径",
            "",
            f"- 结果 CSV：`{OUT_CSV}`",
            f"- 结果 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
