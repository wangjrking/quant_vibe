from __future__ import annotations

from pathlib import Path

import run_latest_l4_joint_frequency_probe_20260630 as joint


REPORT_DIR = joint.REPORT_DIR

joint.OUT_SIGNAL_DIR = REPORT_DIR / "joint_neighborhood_signals"
joint.OUT_LOG_DIR = REPORT_DIR / "joint_neighborhood_logs"
joint.OUT_CSV = REPORT_DIR / "joint_neighborhood_probe_20260630.csv"
joint.OUT_JSON = REPORT_DIR / "joint_neighborhood_probe_20260630.json"

joint.CASES = [
    {"name": "nb_scale50_top5_s110_h2m3_e098_c099_ms1", "base": "scale50", "topn": 5, "target_scale": 1.10, "cap": 0.165, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "nb_scale50_top5_s115_h2m3_e098_c099_ms1", "base": "scale50", "topn": 5, "target_scale": 1.15, "cap": 0.1725, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "nb_scale50_top5_s120_h2m3_e098_c099_ms1", "base": "scale50", "topn": 5, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "nb_scale50_top5_s125_h2m3_e098_c099_ms1", "base": "scale50", "topn": 5, "target_scale": 1.25, "cap": 0.1875, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "nb_scale50_top5_s130_h2m3_e098_c099_ms1", "base": "scale50", "topn": 5, "target_scale": 1.30, "cap": 0.195, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "nb_scale50_top5_s120_h2m3_e097_c098_ms1", "base": "scale50", "topn": 5, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.97, "score_continue": 0.98, "max_daily_sells": 1},
    {"name": "nb_scale50_top5_s120_h2m3_e099_c0995_ms1", "base": "scale50", "topn": 5, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.99, "score_continue": 0.995, "max_daily_sells": 1},
    {"name": "nb_scale50_top5_s120_h2m4_e098_c099_ms1", "base": "scale50", "topn": 5, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 4, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "nb_scale50_top5_s120_h1m3_e098_c099_ms1", "base": "scale50", "topn": 5, "target_scale": 1.20, "cap": 0.18, "holding_days": 1, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "nb_scale50_top5_s120_h2m3_e098_c099_ms2", "base": "scale50", "topn": 5, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 2},
    {"name": "nb_scale50_top4_s120_h2m3_e098_c099_ms1", "base": "scale50", "topn": 4, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "nb_scale50_top4_s130_h2m3_e098_c099_ms1", "base": "scale50", "topn": 4, "target_scale": 1.30, "cap": 0.195, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
]


if __name__ == "__main__":
    joint.main()
