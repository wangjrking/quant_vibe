from __future__ import annotations

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
    / "formal_5d10d_target300_s4_avg80_20260621"
)


def v(name: str, **kwargs: object) -> dict:
    return base._best_variant(name, **kwargs)


RANK5_21 = {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19}
RANK5_23 = {1: 0.23, 2: 0.22, 3: 0.20, 4: 0.18, 5: 0.17}
RANK5_25 = {1: 0.25, 2: 0.23, 3: 0.21, 4: 0.19, 5: 0.17}
RANK5_27 = {1: 0.27, 2: 0.24, 3: 0.21, 4: 0.18, 5: 0.15}
RANK4_28 = {1: 0.28, 2: 0.26, 3: 0.24, 4: 0.20}
RANK8_14 = {1: 0.14, 2: 0.135, 3: 0.13, 4: 0.125, 5: 0.12, 6: 0.115, 7: 0.11, 8: 0.105}
RANK10_12 = {1: 0.12, 2: 0.115, 3: 0.11, 4: 0.105, 5: 0.10, 6: 0.095, 7: 0.09, 8: 0.085, 9: 0.08, 10: 0.075}
RANK12_10 = {1: 0.10, 2: 0.098, 3: 0.096, 4: 0.094, 5: 0.092, 6: 0.09, 7: 0.088, 8: 0.086, 9: 0.084, 10: 0.082, 11: 0.08, 12: 0.078}


VARIANTS = [
    v(
        "goal_best_rank5_exit100_h7_mh10",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_best_rank5_exit101_h7_mh10",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        score_exit_entry_ratio=1.01,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "1.01", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_best_rank5_exit102_h7_mh10",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        score_exit_entry_ratio=1.02,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "1.02", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_best_rank5_sell2_exit100",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        max_daily_sells=2,
        extra_env={"GM_MAX_DAILY_SELLS": "2", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_best_rank5_stop06",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={"GM_STOP_LOSS_PCT": "0.06", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_best_rank5_stopnone",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={"GM_STOP_LOSS_PCT": "none", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_best_rank5_light04",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={"GM_LIGHT_STOP_LOSS_PCT": "0.04", "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "2", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_best_rank5_eqdd_strict",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.16",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
            "GM_EQUITY_DD_SOFT_SCALE": "0.80",
            "GM_EQUITY_DD_HARD_SCALE": "0.55",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
        },
    ),
    v(
        "goal_best_rank5_eqdd_resize",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.21,
        rank_target_pct=RANK5_21,
        extra_env={
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
            "GM_EQUITY_DD_SOFT_SCALE": "0.85",
            "GM_EQUITY_DD_HARD_SCALE": "0.65",
            "GM_EQUITY_DD_RESIZE_EXISTING": "1",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
        },
    ),
    v(
        "goal_highannual_rank5_23_exit100",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "goal_highannual_rank5_23_exit101",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        score_exit_entry_ratio=1.01,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "1.01", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "goal_highannual_rank5_23_noeqdd",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.23,
        rank_target_pct=RANK5_23,
        extra_env={"GM_EQUITY_DD_RISK_MODE": "0"},
    ),
    v(
        "goal_highannual_rank5_25_avg205",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_25,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_highannual_rank5_25_avg200",
        post_filter_avg_pred_min=2.00,
        max_target_pct=0.25,
        rank_target_pct=RANK5_25,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.01"},
    ),
    v(
        "goal_highannual_rank5_27_avg205",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.27,
        rank_target_pct=RANK5_27,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_highannual_rank4_28_avg205",
        rank_max=4,
        max_positions=4,
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.28,
        rank_target_pct=RANK4_28,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_highannual_rank5_25_idxrisk",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_25,
        extra_env={
            "GM_INDEX_RISK_EXIT_MODE": "1",
            "GM_INDEX_RISK_CC_THRESHOLD": "-0.025",
            "GM_INDEX_RISK_INTRADAY_THRESHOLD": "-0.025",
            "GM_INDEX_RISK_BUY_SCALE": "0.70",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
        },
    ),
    v(
        "goal_highannual_rank5_25_eqdd_loose",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_25,
        extra_env={
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.12",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.22",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
            "GM_EQUITY_DD_SOFT_SCALE": "0.90",
            "GM_EQUITY_DD_HARD_SCALE": "0.70",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
        },
    ),
    v(
        "goal_highannual_rank5_25_eqdd_resize",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_25,
        extra_env={
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
            "GM_EQUITY_DD_SOFT_SCALE": "0.85",
            "GM_EQUITY_DD_HARD_SCALE": "0.65",
            "GM_EQUITY_DD_RESIZE_EXISTING": "1",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
        },
    ),
    v(
        "goal_highannual_rank5_25_exit_rank8",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_25,
        extra_env={"GM_SCORE_EXIT_RANK": "8", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_highannual_rank5_25_exit_rank5",
        post_filter_avg_pred_min=2.05,
        max_target_pct=0.25,
        rank_target_pct=RANK5_25,
        extra_env={"GM_SCORE_EXIT_RANK": "5", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02"},
    ),
    v(
        "goal_avg80_rank8_exit100",
        rank_max=8,
        max_positions=8,
        post_filter_avg_pred_min=1.95,
        max_target_pct=0.14,
        rank_target_pct=RANK8_14,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0"},
    ),
    v(
        "goal_avg80_rank8_exit102",
        rank_max=8,
        max_positions=8,
        post_filter_avg_pred_min=1.95,
        max_target_pct=0.14,
        rank_target_pct=RANK8_14,
        score_exit_entry_ratio=1.02,
        extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "1.02", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0"},
    ),
    v(
        "goal_avg80_rank10_exit100",
        rank_max=10,
        max_positions=10,
        post_filter_avg_pred_min=1.90,
        max_target_pct=0.12,
        rank_target_pct=RANK10_12,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0"},
    ),
    v(
        "goal_avg80_rank10_sell0",
        rank_max=10,
        max_positions=10,
        post_filter_avg_pred_min=1.90,
        max_target_pct=0.12,
        rank_target_pct=RANK10_12,
        max_daily_sells=0,
        extra_env={"GM_MAX_DAILY_SELLS": "0", "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0"},
    ),
    v(
        "goal_avg80_rank12_exit100",
        rank_max=12,
        max_positions=12,
        post_filter_avg_pred_min=1.85,
        max_target_pct=0.10,
        rank_target_pct=RANK12_10,
        extra_env={"GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0"},
    ),
    v(
        "goal_avg80_rank12_no_score_exit",
        rank_max=12,
        max_positions=12,
        post_filter_avg_pred_min=1.85,
        max_target_pct=0.10,
        rank_target_pct=RANK12_10,
        extra_env={"GM_OPEN_DAILY_SCORE_EXIT": "0"},
    ),
]


def main() -> int:
    base.REPORT_DIR = REPORT_DIR
    base.VARIANTS = VARIANTS
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
