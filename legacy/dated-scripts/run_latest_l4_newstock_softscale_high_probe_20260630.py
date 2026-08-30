from __future__ import annotations

import run_latest_l4_newstock_softscale_probe_20260630 as soft


REPORT_DIR = soft.REPORT_DIR

soft.base.SCORE_DB = REPORT_DIR / "scores" / "newstock_softscale_high_scores.duckdb"
soft.base.OUT_SIGNAL_DIR = REPORT_DIR / "newstock_softscale_high_signals"
soft.base.OUT_LOG_DIR = REPORT_DIR / "newstock_softscale_high_logs"
soft.base.OUT_CSV = REPORT_DIR / "newstock_softscale_high_probe_20260630.csv"
soft.base.OUT_JSON = REPORT_DIR / "newstock_softscale_high_probe_20260630.json"

soft.base.CASES = [
    {"name": "soft_new60_turn50_top5_s120_h2m3", "new_scale": 0.60, "turnover_high": 50.0, "turn_scale": 0.70, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new70_turn50_top5_s120_h2m3", "new_scale": 0.70, "turnover_high": 50.0, "turn_scale": 0.80, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new80_turn50_top5_s120_h2m3", "new_scale": 0.80, "turnover_high": 50.0, "turn_scale": 0.85, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new60_turn50_top5_s130_h2m3", "new_scale": 0.60, "turnover_high": 50.0, "turn_scale": 0.70, "target_scale": 1.30, "cap": 0.195, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new70_turn50_top5_s130_h2m3", "new_scale": 0.70, "turnover_high": 50.0, "turn_scale": 0.80, "target_scale": 1.30, "cap": 0.195, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new80_turn50_top5_s130_h2m3", "new_scale": 0.80, "turnover_high": 50.0, "turn_scale": 0.85, "target_scale": 1.30, "cap": 0.195, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
]


if __name__ == "__main__":
    soft.base.main()

