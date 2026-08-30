from __future__ import annotations

from typing import Any

import run_latest_l4_newstock_guard_probe_20260630 as base


REPORT_DIR = base.REPORT_DIR

base.SCORE_DB = REPORT_DIR / "scores" / "newstock_softscale_scores.duckdb"
base.OUT_SIGNAL_DIR = REPORT_DIR / "newstock_softscale_signals"
base.OUT_LOG_DIR = REPORT_DIR / "newstock_softscale_logs"
base.OUT_CSV = REPORT_DIR / "newstock_softscale_probe_20260630.csv"
base.OUT_JSON = REPORT_DIR / "newstock_softscale_probe_20260630.json"


base.CASES = [
    {"name": "soft_new10_turn50_top5_s120_h2m3", "new_scale": 0.10, "turnover_high": 50.0, "turn_scale": 0.50, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new20_turn50_top5_s120_h2m3", "new_scale": 0.20, "turnover_high": 50.0, "turn_scale": 0.50, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new30_turn50_top5_s120_h2m3", "new_scale": 0.30, "turnover_high": 50.0, "turn_scale": 0.50, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new40_turn50_top5_s120_h2m3", "new_scale": 0.40, "turnover_high": 50.0, "turn_scale": 0.60, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new20_turn30_top5_s120_h2m3", "new_scale": 0.20, "turnover_high": 30.0, "turn_scale": 0.50, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new30_turn30_top5_s120_h2m3", "new_scale": 0.30, "turnover_high": 30.0, "turn_scale": 0.50, "target_scale": 1.20, "cap": 0.18, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new30_turn50_top5_s130_h2m3", "new_scale": 0.30, "turnover_high": 50.0, "turn_scale": 0.50, "target_scale": 1.30, "cap": 0.195, "holding_days": 2, "max_holding_days": 3, "score_exit": 0.98, "score_continue": 0.99, "max_daily_sells": 1},
    {"name": "soft_new30_turn50_top5_s120_h1m2", "new_scale": 0.30, "turnover_high": 50.0, "turn_scale": 0.50, "target_scale": 1.20, "cap": 0.18, "holding_days": 1, "max_holding_days": 2, "score_exit": 0.99, "score_continue": 0.995, "max_daily_sells": 1},
]


def _extra_filter(_: dict[str, Any]) -> str:
    return "TRUE"


def _target_expr(case: dict[str, Any]) -> str:
    scale = float(case["target_scale"])
    cap = float(case["cap"])
    new_scale = float(case["new_scale"])
    turnover_high = float(case["turnover_high"])
    turn_scale = float(case["turn_scale"])
    risk_scale = f"""
        CASE
          WHEN try_cast(atr_qfq AS DOUBLE) IS NULL OR list_age_days < 60 THEN {new_scale}
          WHEN try_cast(turnover_rate AS DOUBLE) >= {turnover_high} THEN {turn_scale}
          ELSE 1.0
        END
    """
    base_target = """
        CASE
          WHEN rank_1d < 0.50 THEN (
            CASE
              WHEN rank_3d < 0.45 OR rank_5d < 0.45 THEN 0.09
              WHEN rank_3d < 0.65 OR rank_5d < 0.65 THEN 0.12
              ELSE 0.15
            END
          ) * 0.4
          ELSE (
            CASE
              WHEN rank_3d < 0.45 OR rank_5d < 0.45 THEN 0.09
              WHEN rank_3d < 0.65 OR rank_5d < 0.65 THEN 0.12
              ELSE 0.15
            END
          )
        END
    """
    return f"least({cap}, ({base_target}) * {scale} * ({risk_scale}))"


base._extra_filter = _extra_filter
base._target_expr = _target_expr


if __name__ == "__main__":
    base.main()

