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
    / "formal_5d10d_concentrated_high_exposure_20260621"
)


RANK2_50 = {1: 0.50, 2: 0.49}
RANK2_45 = {1: 0.45, 2: 0.44}
RANK3_34 = {1: 0.34, 2: 0.33, 3: 0.32}
RANK3_40 = {1: 0.40, 2: 0.34, 3: 0.25}
RANK4_28 = {1: 0.28, 2: 0.26, 3: 0.24, 4: 0.21}
RANK5_23 = {1: 0.23, 2: 0.22, 3: 0.20, 4: 0.18, 5: 0.17}


def v(name: str, **kwargs: object) -> dict:
    return base._best_variant(name, **kwargs)


NO_DD = {"GM_EQUITY_DD_RISK_MODE": "0", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.00"}
NO_EARLY_EXIT = {
    "GM_EQUITY_DD_RISK_MODE": "0",
    "GM_OPEN_DAILY_SCORE_EXIT": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.80",
    "GM_MAX_DAILY_SELLS": "0",
}
LOOSE_DD = {
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.16",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.28",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.06",
    "GM_EQUITY_DD_SOFT_SCALE": "0.95",
    "GM_EQUITY_DD_HARD_SCALE": "0.80",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.00",
}


VARIANTS = [
    v(
        "conc2_all_h7_no_dd",
        rank_max=2,
        max_positions=2,
        primary_count_min=1,
        max_target_pct=0.50,
        rank_target_pct=RANK2_50,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_DD,
    ),
    v(
        "conc2_all_h10_no_dd",
        rank_max=2,
        max_positions=2,
        primary_count_min=1,
        holding_days=10,
        max_holding_days=15,
        max_target_pct=0.50,
        rank_target_pct=RANK2_50,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_DD,
    ),
    v(
        "conc2_strong_h7_no_dd",
        rank_max=2,
        max_positions=2,
        max_target_pct=0.45,
        rank_target_pct=RANK2_45,
        post_filter_avg_pred_min=2.10,
        extra_env=NO_DD,
    ),
    v(
        "conc2_strong_h10_loose_dd",
        rank_max=2,
        max_positions=2,
        holding_days=10,
        max_holding_days=15,
        max_target_pct=0.45,
        rank_target_pct=RANK2_45,
        post_filter_avg_pred_min=2.10,
        extra_env=LOOSE_DD,
    ),
    v(
        "conc3_all_h7_no_dd",
        rank_max=3,
        max_positions=3,
        primary_count_min=1,
        max_target_pct=0.34,
        rank_target_pct=RANK3_34,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_DD,
    ),
    v(
        "conc3_all_h10_no_dd",
        rank_max=3,
        max_positions=3,
        primary_count_min=1,
        holding_days=10,
        max_holding_days=15,
        max_target_pct=0.34,
        rank_target_pct=RANK3_34,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_DD,
    ),
    v(
        "conc3_topheavy_h7_no_dd",
        rank_max=3,
        max_positions=3,
        primary_count_min=1,
        max_target_pct=0.40,
        rank_target_pct=RANK3_40,
        post_filter_avg_pred_min=1.95,
        extra_env=NO_DD,
    ),
    v(
        "conc3_topheavy_h10_loose_dd",
        rank_max=3,
        max_positions=3,
        holding_days=10,
        max_holding_days=15,
        max_target_pct=0.40,
        rank_target_pct=RANK3_40,
        post_filter_avg_pred_min=1.95,
        extra_env=LOOSE_DD,
    ),
    v(
        "conc3_strong_h7_no_dd",
        rank_max=3,
        max_positions=3,
        max_target_pct=0.34,
        rank_target_pct=RANK3_34,
        post_filter_avg_pred_min=2.10,
        extra_env=NO_DD,
    ),
    v(
        "conc3_strong_h10_loose_dd",
        rank_max=3,
        max_positions=3,
        holding_days=10,
        max_holding_days=15,
        max_target_pct=0.34,
        rank_target_pct=RANK3_34,
        post_filter_avg_pred_min=2.10,
        extra_env=LOOSE_DD,
    ),
    v(
        "conc4_all_h7_no_dd",
        rank_max=4,
        max_positions=4,
        primary_count_min=1,
        max_target_pct=0.28,
        rank_target_pct=RANK4_28,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_DD,
    ),
    v(
        "conc4_all_h10_loose_dd",
        rank_max=4,
        max_positions=4,
        primary_count_min=1,
        holding_days=10,
        max_holding_days=15,
        max_target_pct=0.28,
        rank_target_pct=RANK4_28,
        post_filter_avg_pred_min=1.90,
        extra_env=LOOSE_DD,
    ),
    v(
        "conc4_strong_h7_no_dd",
        rank_max=4,
        max_positions=4,
        max_target_pct=0.28,
        rank_target_pct=RANK4_28,
        post_filter_avg_pred_min=2.10,
        extra_env=NO_DD,
    ),
    v(
        "conc4_strong_h10_loose_dd",
        rank_max=4,
        max_positions=4,
        holding_days=10,
        max_holding_days=15,
        max_target_pct=0.28,
        rank_target_pct=RANK4_28,
        post_filter_avg_pred_min=2.10,
        extra_env=LOOSE_DD,
    ),
    v(
        "conc5_all_h10_no_dd",
        rank_max=5,
        max_positions=5,
        primary_count_min=1,
        holding_days=10,
        max_holding_days=15,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_DD,
    ),
    v(
        "conc5_strong_h10_loose_dd",
        rank_max=5,
        max_positions=5,
        holding_days=10,
        max_holding_days=15,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        post_filter_avg_pred_min=2.10,
        extra_env=LOOSE_DD,
    ),
    v(
        "conc3_all_h15_no_early_exit",
        rank_max=3,
        max_positions=3,
        primary_count_min=1,
        holding_days=15,
        max_holding_days=25,
        max_target_pct=0.34,
        rank_target_pct=RANK3_34,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_EARLY_EXIT,
    ),
    v(
        "conc3_all_h20_no_early_exit",
        rank_max=3,
        max_positions=3,
        primary_count_min=1,
        holding_days=20,
        max_holding_days=30,
        max_target_pct=0.34,
        rank_target_pct=RANK3_34,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_EARLY_EXIT,
    ),
    v(
        "conc4_all_h15_no_early_exit",
        rank_max=4,
        max_positions=4,
        primary_count_min=1,
        holding_days=15,
        max_holding_days=25,
        max_target_pct=0.28,
        rank_target_pct=RANK4_28,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_EARLY_EXIT,
    ),
    v(
        "conc4_all_h20_no_early_exit",
        rank_max=4,
        max_positions=4,
        primary_count_min=1,
        holding_days=20,
        max_holding_days=30,
        max_target_pct=0.28,
        rank_target_pct=RANK4_28,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_EARLY_EXIT,
    ),
    v(
        "conc5_all_h15_no_early_exit",
        rank_max=5,
        max_positions=5,
        primary_count_min=1,
        holding_days=15,
        max_holding_days=25,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_EARLY_EXIT,
    ),
    v(
        "conc5_all_h20_no_early_exit",
        rank_max=5,
        max_positions=5,
        primary_count_min=1,
        holding_days=20,
        max_holding_days=30,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        post_filter_avg_pred_min=1.90,
        extra_env=NO_EARLY_EXIT,
    ),
    v(
        "conc5_strong_h15_no_early_exit",
        rank_max=5,
        max_positions=5,
        holding_days=15,
        max_holding_days=25,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        post_filter_avg_pred_min=2.10,
        extra_env=NO_EARLY_EXIT,
    ),
    v(
        "conc5_strong_h20_no_early_exit",
        rank_max=5,
        max_positions=5,
        holding_days=20,
        max_holding_days=30,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        post_filter_avg_pred_min=2.10,
        extra_env=NO_EARLY_EXIT,
    ),
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
