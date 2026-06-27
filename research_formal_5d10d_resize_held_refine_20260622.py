from __future__ import annotations

import csv
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
    / "formal_5d10d_resize_held_refine_20260622"
)


RANK_PROD = {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19}
RANK_HIGH = {1: 0.23, 2: 0.22, 3: 0.20, 4: 0.18, 5: 0.17}
RANK_FRONT = {1: 0.25, 2: 0.23, 3: 0.20, 4: 0.17, 5: 0.15}


def resize_env(mult: float, min_delta: float = 0.015, continue_ratio: float = 1.02) -> dict[str, str]:
    return {
        "GM_RESIZE_HELD_ON_SIGNAL": "1",
        "GM_RESIZE_HELD_TARGET_MULT": f"{mult:.4f}",
        "GM_RESIZE_HELD_MIN_DELTA_PCT": f"{min_delta:.4f}",
        "GM_SCORE_CONTINUE_ENTRY_RATIO": f"{continue_ratio:.4f}",
    }


def v(name: str, **kwargs: object) -> dict:
    return base._best_variant(name, **kwargs)


VARIANTS = [
    v(
        "resize_prod_m110_cap25",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK_PROD,
        extra_env=resize_env(1.10),
    ),
    v(
        "resize_prod_m120_cap26",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.26,
        rank_target_pct=RANK_PROD,
        extra_env=resize_env(1.20),
    ),
    v(
        "resize_prod_m130_cap28",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.28,
        rank_target_pct=RANK_PROD,
        extra_env=resize_env(1.30),
    ),
    v(
        "resize_prod_m150_cap32",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.32,
        rank_target_pct=RANK_PROD,
        extra_env=resize_env(1.50),
    ),
    v(
        "resize_prod_m130_cap28_lowdelta",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.28,
        rank_target_pct=RANK_PROD,
        extra_env=resize_env(1.30, min_delta=0.005),
    ),
    v(
        "resize_prod_m150_cap32_lowdelta",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.32,
        rank_target_pct=RANK_PROD,
        extra_env=resize_env(1.50, min_delta=0.005),
    ),
    v(
        "resize_high_m110_cap26",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.26,
        rank_target_pct=RANK_HIGH,
        extra_env=resize_env(1.10, continue_ratio=1.01),
    ),
    v(
        "resize_high_m120_cap28",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.28,
        rank_target_pct=RANK_HIGH,
        extra_env=resize_env(1.20, continue_ratio=1.01),
    ),
    v(
        "resize_high_m130_cap30",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.30,
        rank_target_pct=RANK_HIGH,
        extra_env=resize_env(1.30, continue_ratio=1.01),
    ),
    v(
        "resize_high_m140_cap32",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.32,
        rank_target_pct=RANK_HIGH,
        extra_env=resize_env(1.40, continue_ratio=1.01),
    ),
    v(
        "resize_front_m110_cap28",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.28,
        rank_target_pct=RANK_FRONT,
        extra_env=resize_env(1.10, continue_ratio=1.01),
    ),
    v(
        "resize_front_m120_cap30",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.30,
        rank_target_pct=RANK_FRONT,
        extra_env=resize_env(1.20, continue_ratio=1.01),
    ),
    v(
        "resize_front_m130_cap33",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.33,
        rank_target_pct=RANK_FRONT,
        extra_env=resize_env(1.30, continue_ratio=1.01),
    ),
    v(
        "resize_high_m120_cap28_noeqdd",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.28,
        rank_target_pct=RANK_HIGH,
        extra_env={**resize_env(1.20, continue_ratio=1.01), "GM_EQUITY_DD_RISK_MODE": "0"},
    ),
    v(
        "resize_high_m130_cap30_noeqdd",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.30,
        rank_target_pct=RANK_HIGH,
        extra_env={**resize_env(1.30, continue_ratio=1.01), "GM_EQUITY_DD_RISK_MODE": "0"},
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
