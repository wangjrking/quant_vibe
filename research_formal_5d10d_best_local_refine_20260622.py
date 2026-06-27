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
    / "formal_5d10d_best_local_refine_20260622"
)


def v(name: str, **kwargs: object) -> dict:
    return base._best_variant(name, **kwargs)


RANK_2121201919 = {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19}
RANK_213211200190186 = {1: 0.213, 2: 0.211, 3: 0.200, 4: 0.190, 5: 0.186}
RANK_215212200188185 = {1: 0.215, 2: 0.212, 3: 0.200, 4: 0.188, 5: 0.185}
RANK_208208200192192 = {1: 0.208, 2: 0.208, 3: 0.200, 4: 0.192, 5: 0.192}


def env(**items: str) -> dict[str, str]:
    return dict(items)


VARIANTS = [
    v(
        "local_best_repro",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
    ),
    v(
        "local_best_exit099",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        score_exit_entry_ratio=0.99,
        extra_env=env(GM_SCORE_EXIT_ENTRY_RATIO="0.99"),
    ),
    v(
        "local_best_exit101",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        score_exit_entry_ratio=1.01,
        extra_env=env(GM_SCORE_EXIT_ENTRY_RATIO="1.01"),
    ),
    v(
        "local_best_minh2",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        min_holding_days_before_score_exit=2,
        extra_env=env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="2"),
    ),
    v(
        "local_best_minh4",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        min_holding_days_before_score_exit=4,
        extra_env=env(GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT="4"),
    ),
    v(
        "local_best_sell2",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        max_daily_sells=2,
        extra_env=env(GM_MAX_DAILY_SELLS="2"),
    ),
    v(
        "local_best_sell0",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        max_daily_sells=0,
        extra_env=env(GM_MAX_DAILY_SELLS="0"),
    ),
    v(
        "local_best_c101",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        extra_env=env(GM_SCORE_CONTINUE_ENTRY_RATIO="1.01"),
    ),
    v(
        "local_best_c103",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        extra_env=env(GM_SCORE_CONTINUE_ENTRY_RATIO="1.03"),
    ),
    v(
        "local_best_cash995",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        extra_env=env(GM_CASH_BUFFER="0.995"),
    ),
    v(
        "local_best_cash985",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        extra_env=env(GM_CASH_BUFFER="0.985"),
    ),
    v(
        "local_best_dd_loose",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        extra_env=env(GM_EQUITY_DD_SOFT_TRIGGER="0.14", GM_EQUITY_DD_HARD_TRIGGER="0.24"),
    ),
    v(
        "local_best_dd_tight",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        extra_env=env(GM_EQUITY_DD_SOFT_TRIGGER="0.10", GM_EQUITY_DD_HARD_TRIGGER="0.20"),
    ),
    v(
        "local_best_rw213",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.213,
        rank_target_pct=RANK_213211200190186,
    ),
    v(
        "local_best_rw215",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.215,
        rank_target_pct=RANK_215212200188185,
    ),
    v(
        "local_best_rw208_balanced",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.208,
        rank_target_pct=RANK_208208200192192,
    ),
    v(
        "local_best_avg203",
        post_filter_avg_pred_min=2.03,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
    ),
    v(
        "local_best_avg207",
        post_filter_avg_pred_min=2.07,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
    ),
    v(
        "local_best_h6_mh9",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        holding_days=6,
        max_holding_days=9,
    ),
    v(
        "local_best_h8_mh11",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK_2121201919,
        holding_days=8,
        max_holding_days=11,
    ),
]


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
            "score_exit_entry_ratio": variant.get("score_exit_entry_ratio"),
            "min_holding_days_before_score_exit": variant.get("min_holding_days_before_score_exit"),
            "max_daily_sells": variant.get("max_daily_sells"),
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
            f"avg_inv={row.get('avg_invested_pct')} ge80={row.get('ge80_ratio')}"
        )

    _write_rows(REPORT_DIR / "summary.csv", results)
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=base._sort_by_annual, reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=base._sort_by_sharpe, reverse=True))
    target = [
        row
        for row in results
        if float(row.get("annual") or -999.0) >= 3.0
        and float(row.get("sharpe") or -999.0) >= 4.0
        and float(row.get("avg_invested_pct") or -999.0) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_target_hits.csv", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
