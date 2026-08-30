from __future__ import annotations

from typing import Any

import run_latest_l4_newstock_softscale_probe_20260630 as soft


REPORT_DIR = soft.REPORT_DIR

soft.base.SCORE_DB = REPORT_DIR / "scores" / "price_action_frequency_scores.duckdb"
soft.base.OUT_SIGNAL_DIR = REPORT_DIR / "price_action_frequency_signals"
soft.base.OUT_LOG_DIR = REPORT_DIR / "price_action_frequency_logs"
soft.base.OUT_CSV = REPORT_DIR / "price_action_frequency_probe_20260630.csv"
soft.base.OUT_JSON = REPORT_DIR / "price_action_frequency_probe_20260630.json"


soft.base.CASES = [
    {"name": "pa_abs12_prev12_2d15_new80_s190_cap30_h2m3", "topn": 5, "abs_pct_cap": 12.0, "abs_prev_cap": 12.0, "two_day_abs_cap": 0.15, "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "pa_abs10_prev10_2d12_new80_s190_cap30_h2m3", "topn": 5, "abs_pct_cap": 10.0, "abs_prev_cap": 10.0, "two_day_abs_cap": 0.12, "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "pa_abs15_prev15_2d18_new80_s190_cap30_h2m3", "topn": 5, "abs_pct_cap": 15.0, "abs_prev_cap": 15.0, "two_day_abs_cap": 0.18, "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "pa_abs12_prev12_2d15_new90_s190_cap30_h2m3", "topn": 5, "abs_pct_cap": 12.0, "abs_prev_cap": 12.0, "two_day_abs_cap": 0.15, "new_scale": 0.90, "turnover_high": 80.0, "turn_scale": 0.95, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "pa_abs12_prev12_2d15_new80_s210_cap33_h2m3", "topn": 5, "abs_pct_cap": 12.0, "abs_prev_cap": 12.0, "two_day_abs_cap": 0.15, "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 2.10, "cap": 0.33, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "pa_abs12_prev12_2d15_new80_s190_cap30_h1m2", "topn": 5, "abs_pct_cap": 12.0, "abs_prev_cap": 12.0, "two_day_abs_cap": 0.15, "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.90, "cap": 0.30, "holding_days": 1, "max_holding_days": 2, "score_exit": 0.99, "score_continue": 0.995, "max_daily_sells": 1},
]


def _extra_filter(case: dict[str, Any]) -> str:
    abs_pct = float(case["abs_pct_cap"])
    abs_prev = float(case["abs_prev_cap"])
    two_day_abs = float(case["two_day_abs_cap"])
    return f"""
        abs(coalesce(try_cast(md.pct_chg AS DOUBLE), 0.0)) <= {abs_pct}
        AND abs(coalesce(try_cast(md.prev_pct_chg AS DOUBLE), 0.0)) <= {abs_prev}
        AND abs(coalesce(((1 + try_cast(md.pct_chg AS DOUBLE) / 100.0)
             * (1 + try_cast(md.prev_pct_chg AS DOUBLE) / 100.0) - 1.0), 0.0)) <= {two_day_abs}
    """


soft.base._extra_filter = _extra_filter


if __name__ == "__main__":
    soft.base.main()

