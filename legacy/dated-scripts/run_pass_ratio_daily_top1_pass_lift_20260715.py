from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_open_verified_scale_fine_20260715.py"

spec = importlib.util.spec_from_file_location("juejin_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
POOL_PATH = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
    / "active_l4_wide_buy_quality_base_pool_20260714.parquet"
)
SIGNAL_DIR = REPORT_DIR / "signals" / "daily_top1_pass_lift"
LOG_DIR = REPORT_DIR / "logs" / "daily_top1_pass_lift_20260715"
OUT_CSV = REPORT_DIR / "daily_top1_pass_lift_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "daily_top1_pass_lift_juejin_20260715.json"
OUT_MD = REPORT_DIR / "daily_top1_pass_lift_review_20260715.md"


CASES = [
    {
        "case": "dt1_core_p175_s115",
        "pct_max": -1.75,
        "base_pos": 0.55,
        "mild_pos": 0.00,
        "weak_pos": 0.00,
        "deep_mult": 1.15,
        "gap_soft": 0.50,
    },
    {
        "case": "dt1_lift_p125_s095",
        "pct_max": -1.25,
        "base_pos": 0.42,
        "mild_pos": 0.16,
        "weak_pos": 0.00,
        "deep_mult": 1.10,
        "gap_soft": 0.50,
    },
    {
        "case": "dt1_lift_p075_s080",
        "pct_max": -0.75,
        "base_pos": 0.34,
        "mild_pos": 0.12,
        "weak_pos": 0.00,
        "deep_mult": 1.05,
        "gap_soft": 0.50,
    },
    {
        "case": "dt1_lift_p000_s060",
        "pct_max": 0.00,
        "base_pos": 0.28,
        "mild_pos": 0.09,
        "weak_pos": 0.00,
        "deep_mult": 1.00,
        "gap_soft": 0.45,
    },
    {
        "case": "dt1_lift_p075pos_s045",
        "pct_max": 0.75,
        "base_pos": 0.24,
        "mild_pos": 0.07,
        "weak_pos": 0.03,
        "deep_mult": 1.00,
        "gap_soft": 0.40,
    },
    {
        "case": "dt1_lift_p150pos_s035",
        "pct_max": 1.50,
        "base_pos": 0.20,
        "mild_pos": 0.06,
        "weak_pos": 0.025,
        "deep_mult": 1.00,
        "gap_soft": 0.40,
    },
    {
        "case": "dt1_all_tiny_s020",
        "pct_max": None,
        "base_pos": 0.16,
        "mild_pos": 0.045,
        "weak_pos": 0.02,
        "deep_mult": 1.00,
        "gap_soft": 0.35,
    },
    {
        "case": "dt1_all_rankguard_s030",
        "pct_max": None,
        "base_pos": 0.22,
        "mild_pos": 0.06,
        "weak_pos": 0.02,
        "deep_mult": 1.00,
        "gap_soft": 0.35,
        "rank10_min": 0.985,
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


def build_daily_top1(pool: pd.DataFrame) -> pd.DataFrame:
    base_pool = pool[
        (pool["buy_open_gap_raw_pct"] <= 1.5)
        & (pool["buy_open_gap_raw_pct"] >= -8.0)
        & (pool["amount"] >= 120000)
        & (pool["total_mv"] >= 200000)
        & (~pool["buy_open_limit_up_rejected"])
    ].copy()
    base_pool = base_pool.sort_values(
        ["buy_date", "blend_score", "pred_10d", "amount"],
        ascending=[True, False, False, False],
    )
    top1 = base_pool.groupby("buy_date", group_keys=False).head(1).copy()
    top1["rank"] = 1
    return top1


def build_signal(top1: pd.DataFrame, case: dict) -> pd.DataFrame:
    df = top1.copy()
    if case.get("pct_max") is not None:
        df = df[df["signal_pct_chg_raw"] <= float(case["pct_max"])].copy()
    if case.get("rank10_min") is not None:
        df = df[df["pred_10d_rank"] >= float(case["rank10_min"])].copy()

    pct = pd.to_numeric(df["signal_pct_chg_raw"], errors="coerce")
    gap = pd.to_numeric(df["buy_open_gap_raw_pct"], errors="coerce")
    target = pd.Series(float(case["base_pos"]), index=df.index)
    target.loc[pct > -1.75] = float(case["mild_pos"])
    target.loc[pct > 0.0] = float(case["weak_pos"])
    target.loc[pct <= -5.0] *= float(case["deep_mult"])
    target.loc[gap > 1.0] *= float(case["gap_soft"])
    df["target_pct"] = target.clip(lower=0.0, upper=0.65)
    df = df[df["target_pct"] > 0].copy()

    df["symbol"] = df["stock_code"].map(to_symbol)
    df["pred_prob"] = df["blend_score"]
    df["entry_score"] = df["blend_score"]
    df["atr_qfq"] = None
    df["holding_days"] = 1
    df["max_holding_days"] = 3
    df["score_exit_entry_ratio"] = "0.98000"
    df["min_holding_days_before_score_exit"] = 1
    df["score_continue_entry_ratio"] = "1.02000"
    df["signal_stop_loss_pct"] = 0.05
    df["signal_take_profit_pct"] = 0.08
    df["strategy_variant"] = case["case"]
    df["source_strategy_variant"] = "active_l4_formal_daily_top1_pass_lift_no_refill"
    df["filter_name"] = case["case"]
    df["entry_weight_name"] = "rank_blend_active_l4_daily_top1_no_refill"
    df["dynamic_hold_name"] = "h1_mh3_score_continue"
    df["buy_day_market_available"] = True
    df["buy_day_hard_gate_complete"] = True
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["latest_market_date"] = "20260714"
    df["buy_open_gap_pct"] = df["buy_open_gap_raw_pct"]
    df["hybrid_source"] = "active_l4_formal_duckdb_daily_top1_no_refill"
    df["feature_weight_scale"] = 1.0
    df["daily_target_sum_after_cap"] = df["target_pct"]

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
    return df[cols].copy()


def run_juejin(signal_file: Path, case: str, slice_name: str) -> dict:
    old_log_dir = base.LOG_DIR
    base.LOG_DIR = LOG_DIR
    try:
        return base.run_juejin(signal_file, case, slice_name)
    finally:
        base.LOG_DIR = old_log_dir


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(POOL_PATH)
    pool = add_scores(pool)
    top1 = build_daily_top1(pool)
    total_buy_days = int(top1["buy_date"].nunique())

    rows: list[dict] = []
    for case in CASES:
        signal = build_signal(top1, case)
        for slice_name, start in base.SLICES:
            sliced = base.slice_frame(signal, start)
            out_path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(out_path, index=False, encoding="utf-8-sig")
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result = run_juejin(out_path, case["case"], slice_name)
            result.update(
                {
                    **case,
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "all_top1_buy_days": total_buy_days,
                    "pass_ratio": (float(sliced["buy_date"].nunique()) / total_buy_days) if total_buy_days else None,
                    "stock_count": int(sliced["stock_code"].nunique()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                    "max_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                }
            )
            rows.append(result)
            print(
                json.dumps(
                    {
                        "case": result["case"],
                        "slice": slice_name,
                        "pass_ratio": result["pass_ratio"],
                        "annual": result.get("pnl_ratio_annual"),
                        "sharpe": result.get("sharp_ratio"),
                        "mdd": result.get("max_drawdown"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    full = frame[frame["slice"].eq("full")].sort_values(
        ["sharp_ratio", "pnl_ratio_annual"],
        ascending=[False, False],
        na_position="last",
    )
    lines = [
        "# 每日 Top1 主信号通过率提升验证 20260715",
        "",
        "## 口径",
        "",
        "- 输入：最新 active formal L4 + L2 未复权执行行情派生池。",
        "- 不做候补/补位：每个买入日只按主排序取 Top1，Top1 不通过则无信号。",
        "- 本轮只调整 Top1 的信号日涨跌幅通过阈值和低质量轻仓，不扩大到 Top2/Top5。",
        "- 结果为 research-only，正式收益以掘金日志为准。",
        "",
        "## full 结果",
        "",
        "| 版本 | 通过率 | 买入日 | 年化 | Sharpe | 最大回撤 | 胜率 | 均值仓位 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in full.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row.get('pass_ratio'))} | {int(row['buy_days'])} | "
            f"{pct(row.get('pnl_ratio_annual'))} | {float(row.get('sharp_ratio') or 0):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {pct(row.get('win_ratio'))} | "
            f"{pct(row.get('mean_daily_target_sum'))} | {row.get('max_buy_date')} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 如果只提高主 Top1 通过比例，不做候补，覆盖率可以从约 21% 提到 99% 以上。",
            "- 但放宽后的多数交易日不是高收益样本，必须用降仓保护；是否能准入以本轮掘金结果为准。",
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
