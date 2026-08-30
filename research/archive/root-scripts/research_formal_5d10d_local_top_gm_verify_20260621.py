from __future__ import annotations

import importlib
from pathlib import Path


base = importlib.import_module("research_formal_5d10d_noncalendar_quality_refine_20260621")

ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_local_top_gm_verify_20260621"
)


RANK3_40 = {1: 0.40, 2: 0.34, 3: 0.24}
RANK4_30 = {1: 0.30, 2: 0.26, 3: 0.23, 4: 0.20}
RANK5_25 = {1: 0.25, 2: 0.23, 3: 0.21, 4: 0.18, 5: 0.15}


def v(name: str, **kwargs: object) -> dict:
    return base._best_variant(name, **kwargs)


NO_DD = {"GM_EQUITY_DD_RISK_MODE": "0", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.00"}
PROD_DD = {
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.12",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.22",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.90",
    "GM_EQUITY_DD_HARD_SCALE": "0.70",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
}


VARIANTS = [
    v("gm_local_r3_h7_mv150_r10_099", rank_max=3, max_positions=3, holding_days=7, max_holding_days=10, max_total_mv=150000, min_rank_10d=0.99, post_filter_avg_pred_min=1.90, max_target_pct=0.40, rank_target_pct=RANK3_40, primary_count_min=1, extra_env=NO_DD),
    v("gm_local_r3_h7_mv150_r10_099_dd", rank_max=3, max_positions=3, holding_days=7, max_holding_days=10, max_total_mv=150000, min_rank_10d=0.99, post_filter_avg_pred_min=1.90, max_target_pct=0.40, rank_target_pct=RANK3_40, primary_count_min=1, extra_env=PROD_DD),
    v("gm_local_r3_h5_mv150", rank_max=3, max_positions=3, holding_days=5, max_holding_days=8, max_total_mv=150000, post_filter_avg_pred_min=1.90, max_target_pct=0.40, rank_target_pct=RANK3_40, primary_count_min=1, extra_env=NO_DD),
    v("gm_local_r3_h5_mv150_avg205", rank_max=3, max_positions=3, holding_days=5, max_holding_days=8, max_total_mv=150000, post_filter_avg_pred_min=2.05, max_target_pct=0.40, rank_target_pct=RANK3_40, primary_count_min=1, extra_env=NO_DD),
    v("gm_local_r3_h5_all_avg205", rank_max=3, max_positions=3, holding_days=5, max_holding_days=8, post_filter_avg_pred_min=2.05, max_target_pct=0.40, rank_target_pct=RANK3_40, primary_count_min=1, extra_env=NO_DD),
    v("gm_local_r5_h10_turnlow", rank_max=5, max_positions=5, holding_days=10, max_holding_days=15, max_turnover=1.7, post_filter_avg_pred_min=1.90, max_target_pct=0.25, rank_target_pct=RANK5_25, primary_count_min=1, extra_env=NO_DD),
    v("gm_local_r5_h10_turnlow_avg205", rank_max=5, max_positions=5, holding_days=10, max_holding_days=15, max_turnover=1.7, post_filter_avg_pred_min=2.05, max_target_pct=0.25, rank_target_pct=RANK5_25, primary_count_min=1, extra_env=NO_DD),
    v("gm_local_r5_h10_turnlow_dd", rank_max=5, max_positions=5, holding_days=10, max_holding_days=15, max_turnover=1.7, post_filter_avg_pred_min=1.90, max_target_pct=0.25, rank_target_pct=RANK5_25, primary_count_min=1, extra_env=PROD_DD),
    v("gm_local_r4_h7_mv150_r10_099", rank_max=4, max_positions=4, holding_days=7, max_holding_days=10, max_total_mv=150000, min_rank_10d=0.99, post_filter_avg_pred_min=1.90, max_target_pct=0.30, rank_target_pct=RANK4_30, primary_count_min=1, extra_env=NO_DD),
    v("gm_local_r4_h7_mv150_r10_099_dd", rank_max=4, max_positions=4, holding_days=7, max_holding_days=10, max_total_mv=150000, min_rank_10d=0.99, post_filter_avg_pred_min=1.90, max_target_pct=0.30, rank_target_pct=RANK4_30, primary_count_min=1, extra_env=PROD_DD),
    v("gm_local_r3_h15_mv150_r10_099", rank_max=3, max_positions=3, holding_days=15, max_holding_days=20, max_total_mv=150000, min_rank_10d=0.99, post_filter_avg_pred_min=1.90, max_target_pct=0.40, rank_target_pct=RANK3_40, primary_count_min=1, extra_env=NO_DD),
    v("gm_local_r5_h10_pred5_mid", rank_max=5, max_positions=5, holding_days=10, max_holding_days=15, min_pred_5d=0.0057, max_target_pct=0.25, rank_target_pct=RANK5_25, post_filter_avg_pred_min=1.90, primary_count_min=1, extra_env=NO_DD),
]


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        if not signal_file.exists():
            base._write_variant_signal(variant, signal_file)
        returncode = base._run_backtest(variant, signal_file, log_file)
        indicator = base._extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "rank_max": variant.get("rank_max"),
            "primary_count_min": variant.get("primary_count_min"),
            "min_rank_10d": variant.get("min_rank_10d"),
            "max_turnover": variant.get("max_turnover"),
            "max_total_mv": variant.get("max_total_mv"),
            "min_pred_5d": variant.get("min_pred_5d"),
            "post_filter_avg_pred_min": variant.get("post_filter_avg_pred_min"),
            "max_positions": variant.get("max_positions"),
            "holding_days": variant.get("holding_days"),
            "max_holding_days": variant.get("max_holding_days", variant.get("holding_days")),
            "max_target_pct": variant.get("max_target_pct", variant.get("target_pct")),
            "rank_target_pct": variant.get("rank_target_pct"),
            "extra_env": variant.get("extra_env"),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(base._signal_stats(signal_file))
        row.update(base._exposure_stats(log_file))
        results.append(row)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"avg_inv={row.get('avg_invested_pct')} signals={row.get('signal_count')}"
        )
    base._write_rows(REPORT_DIR / "summary.csv", results)
    base._write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=base._sort_by_annual, reverse=True))
    base._write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=base._sort_by_sharpe, reverse=True))
    target = [
        row
        for row in results
        if (row.get("annual") is not None and float(row["annual"]) >= 3.0)
        and (row.get("sharpe") is not None and float(row["sharpe"]) >= 4.0)
        and (row.get("avg_invested_pct") is not None and float(row["avg_invested_pct"]) >= 0.80)
    ]
    base._write_rows(REPORT_DIR / "summary_target_hits.csv", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
