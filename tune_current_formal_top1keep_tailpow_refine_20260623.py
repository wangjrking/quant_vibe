from __future__ import annotations

import tune_current_formal_top1keep_exit_transform_20260623 as prev


base = prev.base

base.REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "current_formal_top1keep_tailpow_refine"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_top1keep_tailpow_refine.db"
base.SCORE_DB = base.REPORT_DIR / "scores_top1keep_tailpow_refine.db"
prev._RANK10D_UPPER = None


def _cfg(
    name: str,
    *,
    gamma: float,
    others_exit_ratio: float,
    top1_exit_ratio: float = 0.91,
    top1_holding_days: int = 7,
) -> dict:
    return {
        "name": name,
        "w10": 0.80,
        "w5": 0.20,
        "w3": 0.00,
        "top_k": 4,
        "target_pct_each": 0.3305,
        "cap": 0.50,
        "holding_days": 5,
        "top1_rule": {
            "score_exit_entry_ratio": top1_exit_ratio,
            "holding_days": top1_holding_days,
        },
        "exit_transform": "tailpow",
        "exit_param": gamma,
        "env": {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": str(others_exit_ratio),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
        },
    }


GAMMAS = [1.18, 1.22, 1.26, 1.28, 1.30, 1.32, 1.34, 1.38]
OTHERS = [0.928, 0.932, 0.935, 0.938, 0.942]

base.VARIANTS = [
    _cfg(
        f"tp_g{str(gamma).replace('.', 'p')}_o{str(others).replace('.', 'p')}",
        gamma=gamma,
        others_exit_ratio=others,
    )
    for gamma in GAMMAS
    for others in OTHERS
]


if __name__ == "__main__":
    raise SystemExit(base.main())
