from __future__ import annotations

import tune_current_formal_top1keep_exit_transform_20260623 as prev


base = prev.base

base.REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "current_formal_top1keep_tailpow_peak"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_top1keep_tailpow_peak.db"
base.SCORE_DB = base.REPORT_DIR / "scores_top1keep_tailpow_peak.db"
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


GAMMAS = [1.36, 1.38, 1.40, 1.42, 1.44]
OTHERS = [0.940, 0.942, 0.944, 0.946]

base.VARIANTS = [
    _cfg(
        f"tp_peak_g{str(gamma).replace('.', 'p')}_o{str(others).replace('.', 'p')}",
        gamma=gamma,
        others_exit_ratio=others,
    )
    for gamma in GAMMAS
    for others in OTHERS
]


if __name__ == "__main__":
    raise SystemExit(base.main())
