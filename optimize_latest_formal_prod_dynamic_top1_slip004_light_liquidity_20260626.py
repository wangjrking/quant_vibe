from __future__ import annotations

from pathlib import Path

import optimize_latest_formal_prod_dynamic_top1_20260625 as base


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA = ROOT / "quant" / "data_file"

base.REPORT_DIR = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_slip004_light_liquidity_20260626"
base.SCORE_DB = base.REPORT_DIR / "scores" / "grid_scores.db"
base.BACKTEST_SLIPPAGE_RATIO = "0.0040"

base.ENTRY_CASES = [
    {
        "case_name": "w84_09_07_amt90_mv20",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 90000.0,
        "total_mv_min": 200000.0,
    },
    {
        "case_name": "w84_09_07_amt100_mv30",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 100000.0,
        "total_mv_min": 300000.0,
    },
    {
        "case_name": "w84_09_07_amt110_mv30",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 110000.0,
        "total_mv_min": 300000.0,
    },
    {
        "case_name": "w85_08_07_amt100_mv30",
        "weight_name": "w85_5d08_3d07",
        "w10d": 0.85,
        "w5d": 0.08,
        "w3d": 0.07,
        "amount_min": 100000.0,
        "total_mv_min": 300000.0,
    },
]

base.EXEC_CASES = [
    {
        "case_name": "h2_m3_c097_e097_pos90_dd12",
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
    {
        "case_name": "h2_m4_c0975_e097_pos90_dd12",
        "target_position_pct": 0.90,
        "score_continue_entry_ratio": 0.975,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 4,
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
