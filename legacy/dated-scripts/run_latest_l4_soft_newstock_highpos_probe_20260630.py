from __future__ import annotations

import run_latest_l4_newstock_softscale_probe_20260630 as soft


REPORT_DIR = soft.REPORT_DIR

soft.base.SCORE_DB = REPORT_DIR / "scores" / "soft_newstock_highpos_scores.duckdb"
soft.base.OUT_SIGNAL_DIR = REPORT_DIR / "soft_newstock_highpos_signals"
soft.base.OUT_LOG_DIR = REPORT_DIR / "soft_newstock_highpos_logs"
soft.base.OUT_CSV = REPORT_DIR / "soft_newstock_highpos_probe_20260630.csv"
soft.base.OUT_JSON = REPORT_DIR / "soft_newstock_highpos_probe_20260630.json"


soft.base.CASES = [
    {"name": "hp_new70_turn80_top5_s150_cap24_h2m3", "new_scale": 0.70, "turnover_high": 80.0, "turn_scale": 0.85, "target_scale": 1.50, "cap": 0.24, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new70_turn80_top5_s170_cap27_h2m3", "new_scale": 0.70, "turnover_high": 80.0, "turn_scale": 0.85, "target_scale": 1.70, "cap": 0.27, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new70_turn80_top5_s190_cap30_h2m3", "new_scale": 0.70, "turnover_high": 80.0, "turn_scale": 0.85, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new80_turn80_top5_s150_cap24_h2m3", "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.50, "cap": 0.24, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new80_turn80_top5_s170_cap27_h2m3", "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.70, "cap": 0.27, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new80_turn80_top5_s190_cap30_h2m3", "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new90_turn80_top5_s150_cap24_h2m3", "new_scale": 0.90, "turnover_high": 80.0, "turn_scale": 0.95, "target_scale": 1.50, "cap": 0.24, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new90_turn80_top5_s170_cap27_h2m3", "new_scale": 0.90, "turnover_high": 80.0, "turn_scale": 0.95, "target_scale": 1.70, "cap": 0.27, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new90_turn80_top5_s190_cap30_h2m3", "new_scale": 0.90, "turnover_high": 80.0, "turn_scale": 0.95, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new80_turn80_top4_s190_cap30_h2m3", "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "hp_new80_turn80_top5_s170_cap27_h1m2", "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.70, "cap": 0.27, "holding_days": 1, "max_holding_days": 2, "score_exit": 0.99, "score_continue": 0.995, "max_daily_sells": 1},
    {"name": "hp_new80_turn80_top5_s170_cap27_h1m3_ms2", "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.70, "cap": 0.27, "holding_days": 1, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 2},
]


if __name__ == "__main__":
    soft.base.main()

