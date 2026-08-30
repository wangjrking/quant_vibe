from __future__ import annotations

import run_latest_l4_newstock_softscale_probe_20260630 as soft


REPORT_DIR = soft.REPORT_DIR

soft.base.SCORE_DB = REPORT_DIR / "scores" / "soft_newstock_topn_scores.duckdb"
soft.base.OUT_SIGNAL_DIR = REPORT_DIR / "soft_newstock_topn_signals"
soft.base.OUT_LOG_DIR = REPORT_DIR / "soft_newstock_topn_logs"
soft.base.OUT_CSV = REPORT_DIR / "soft_newstock_topn_probe_20260630.csv"
soft.base.OUT_JSON = REPORT_DIR / "soft_newstock_topn_probe_20260630.json"


soft.base.CASES = [
    {"name": "top4_new70_s170_cap27_h2m3", "topn": 4, "new_scale": 0.70, "turnover_high": 80.0, "turn_scale": 0.85, "target_scale": 1.70, "cap": 0.27, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "top4_new70_s190_cap30_h2m3", "topn": 4, "new_scale": 0.70, "turnover_high": 80.0, "turn_scale": 0.85, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "top4_new80_s170_cap27_h2m3", "topn": 4, "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.70, "cap": 0.27, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "top4_new80_s190_cap30_h2m3", "topn": 4, "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.90, "cap": 0.30, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "top3_new70_s190_cap34_h2m3", "topn": 3, "new_scale": 0.70, "turnover_high": 80.0, "turn_scale": 0.85, "target_scale": 1.90, "cap": 0.34, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "top3_new80_s190_cap34_h2m3", "topn": 3, "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 1.90, "cap": 0.34, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "top3_new70_s210_cap36_h2m3", "topn": 3, "new_scale": 0.70, "turnover_high": 80.0, "turn_scale": 0.85, "target_scale": 2.10, "cap": 0.36, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "top3_new80_s210_cap36_h2m3", "topn": 3, "new_scale": 0.80, "turnover_high": 80.0, "turn_scale": 0.90, "target_scale": 2.10, "cap": 0.36, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
]


if __name__ == "__main__":
    soft.base.main()

