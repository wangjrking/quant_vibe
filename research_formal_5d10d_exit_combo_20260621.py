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
    / "formal_5d10d_exit_combo_20260621"
)


RANK5_21 = {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19}
RANK5_23 = {1: 0.23, 2: 0.22, 3: 0.20, 4: 0.18, 5: 0.17}


def v(name: str, **kwargs: object) -> dict:
    return base._best_variant(name, **kwargs)


VARIANTS = [
    v(
        "exit_prod_minhold1",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        min_holding_days_before_score_exit=1,
        extra_env={"GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_prod_minhold2",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        min_holding_days_before_score_exit=2,
        extra_env={"GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_prod_entry098",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        score_exit_entry_ratio=0.98,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.98", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_prod_entry101",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        score_exit_entry_ratio=1.01,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "1.01", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_prod_daydrop098",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.98", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_prod_daydrop099",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.99", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_prod_rank085",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={"GM_SCORE_EXIT_RANK": "0.85", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_prod_rank090",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={"GM_SCORE_EXIT_RANK": "0.90", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_prod_maxsell2",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        max_daily_sells=2,
        extra_env={"GM_MAX_DAILY_SELLS": "2", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_prod_maxsell0",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        max_daily_sells=0,
        extra_env={"GM_MAX_DAILY_SELLS": "0", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "exit_highannual_base",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "exit_highannual_minhold2",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        min_holding_days_before_score_exit=2,
        extra_env={"GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "exit_highannual_entry101",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        score_exit_entry_ratio=1.01,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "1.01", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "exit_highannual_daydrop099",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        extra_env={"GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.99", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "exit_prod_intraday_stop05",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={
            "GM_INTRADAY_RISK_MODE": "1",
            "GM_STOP_LOSS_PCT": "0.05",
            "GM_TAKE_PROFIT_PCT": "none",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
        },
    ),
    v(
        "exit_prod_intraday_stop06_tp12",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={
            "GM_INTRADAY_RISK_MODE": "1",
            "GM_STOP_LOSS_PCT": "0.06",
            "GM_TAKE_PROFIT_PCT": "0.12",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
        },
    ),
    v(
        "exit_prod_intraday_half_stop06",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={
            "GM_INTRADAY_RISK_MODE": "1",
            "GM_STOP_LOSS_PCT": "0.06",
            "GM_INTRADAY_RISK_SELL_FRACTION": "0.5",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
        },
    ),
    v(
        "exit_prod_no_stop",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={"GM_STOP_LOSS_PCT": "none", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
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
            "post_filter_avg_pred_min": variant.get("post_filter_avg_pred_min"),
            "max_positions": variant.get("max_positions"),
            "holding_days": variant.get("holding_days"),
            "max_holding_days": variant.get("max_holding_days", variant.get("holding_days")),
            "max_target_pct": variant.get("max_target_pct", variant.get("target_pct")),
            "rank_target_pct": variant.get("rank_target_pct"),
            "score_exit_entry_ratio": variant.get("score_exit_entry_ratio", 1.0),
            "min_holding_days_before_score_exit": variant.get("min_holding_days_before_score_exit", 3),
            "max_daily_sells": variant.get("max_daily_sells", 1),
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
