from __future__ import annotations

from pathlib import Path

import optimize_latest_formal_prod_dynamic_top1_20260625 as base


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA = ROOT / "quant" / "data_file"

base.REPORT_DIR = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_opt_round2_20260626"
base.SCORE_DB = base.REPORT_DIR / "scores" / "grid_scores.db"

base.ENTRY_CASES = [
    {
        "case_name": "w82_10_08_amt95_mv20",
        "weight_name": "w82_5d10_3d08",
        "w10d": 0.82,
        "w5d": 0.10,
        "w3d": 0.08,
        "amount_min": 95000.0,
        "total_mv_min": 200000.0,
    },
    {
        "case_name": "w83_10_07_amt95_mv20",
        "weight_name": "w83_5d10_3d07",
        "w10d": 0.83,
        "w5d": 0.10,
        "w3d": 0.07,
        "amount_min": 95000.0,
        "total_mv_min": 200000.0,
    },
    {
        "case_name": "w84_09_07_amt95_mv20",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 95000.0,
        "total_mv_min": 200000.0,
    },
    {
        "case_name": "w83_11_06_amt95_mv20",
        "weight_name": "w83_5d11_3d06",
        "w10d": 0.83,
        "w5d": 0.11,
        "w3d": 0.06,
        "amount_min": 95000.0,
        "total_mv_min": 200000.0,
    },
    {
        "case_name": "w82_10_08_amt90_mv20",
        "weight_name": "w82_5d10_3d08",
        "w10d": 0.82,
        "w5d": 0.10,
        "w3d": 0.08,
        "amount_min": 90000.0,
        "total_mv_min": 200000.0,
    },
]

base.EXEC_CASES = [
    {
        "case_name": "exec_c097_pos90",
        "target_position_pct": 0.90,
        "score_continue_entry_ratio": 0.97,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 3,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.08,
        "dd_hard_trigger": 0.115,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
    },
    {
        "case_name": "exec_c0965_pos90",
        "target_position_pct": 0.90,
        "score_continue_entry_ratio": 0.965,
        "score_exit_entry_ratio": 0.968,
        "holding_days": 2,
        "max_holding_days": 3,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.08,
        "dd_hard_trigger": 0.115,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
    },
    {
        "case_name": "exec_c097_pos92",
        "target_position_pct": 0.92,
        "score_continue_entry_ratio": 0.97,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 3,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.08,
        "dd_hard_trigger": 0.115,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.75,
        "dd_hard_scale": 0.55,
    },
    {
        "case_name": "exec_c097_pos90_dd12",
        "target_position_pct": 0.90,
        "score_continue_entry_ratio": 0.97,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 3,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.085,
        "dd_hard_trigger": 0.12,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.78,
        "dd_hard_scale": 0.58,
    },
]


if __name__ == "__main__":
    base.main()
