from __future__ import annotations

import copy
from pathlib import Path

import tune_current_formal_top5_grid2_20260623 as base


REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "current_formal_top5_refine"
)


def _cfg(
    name: str,
    *,
    w10: float = 0.80,
    scale: float = 1.12,
    cap: float = 0.30,
    stop_loss: str = "0.08",
    take_profit: str = "none",
    score_exit: bool = False,
    exit_ratio: str = "0.95",
    min_exit_hold: int = 2,
    light_stop: str = "none",
    min_light_hold: int = 2,
) -> dict:
    cfg = base._variant(
        name,
        w10=w10,
        top_k=5,
        scale=scale,
        cap=cap,
        holding_days=5,
        cash_buffer="0.99",
    )
    env = copy.deepcopy(cfg["env"])
    env["GM_STOP_LOSS_PCT"] = stop_loss
    env["GM_TAKE_PROFIT_PCT"] = take_profit
    env["GM_LIGHT_STOP_LOSS_PCT"] = light_stop
    env["GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP"] = str(min_light_hold)
    env["GM_OPEN_DAILY_SCORE_EXIT"] = "1" if score_exit else "0"
    env["GM_SCORE_EXIT_ENTRY_RATIO"] = exit_ratio if score_exit else "none"
    env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(min_exit_hold)
    env["GM_MAX_DAILY_SELLS"] = "5"
    cfg["env"] = env
    return cfg


VARIANTS = [
    _cfg("w78_s112_c30_h5_noexit", w10=0.78),
    _cfg("w79_s112_c30_h5_noexit", w10=0.79),
    _cfg("w80_s112_c30_h5_noexit", w10=0.80),
    _cfg("w81_s112_c30_h5_noexit", w10=0.81),
    _cfg("w82_s112_c30_h5_noexit", w10=0.82),
    _cfg("w80_s111_c30_h5_noexit", scale=1.11),
    _cfg("w80_s113_c30_h5_noexit", scale=1.13),
    _cfg("w80_s114_c30_h5_noexit", scale=1.14),
    _cfg("w80_s115_c30_h5_noexit", scale=1.15),
    _cfg("w80_s112_c31_h5_noexit", cap=0.31),
    _cfg("w80_s112_c32_h5_noexit", cap=0.32),
    _cfg("w80_s114_c32_h5_noexit", scale=1.14, cap=0.32),
    _cfg("w80_s116_c32_h5_noexit", scale=1.16, cap=0.32),
    _cfg("w80_s112_c30_h5_stop06", stop_loss="0.06"),
    _cfg("w80_s112_c30_h5_stop10", stop_loss="0.10"),
    _cfg("w80_s112_c30_h5_nostop", stop_loss="none"),
    _cfg("w80_s112_c30_h5_tp20", take_profit="0.20"),
    _cfg("w80_s112_c30_h5_tp30", take_profit="0.30"),
    _cfg("w80_s112_c30_h5_lstop04", light_stop="0.04", min_light_hold=2),
    _cfg("w80_s112_c30_h5_exit092_mh2", score_exit=True, exit_ratio="0.92", min_exit_hold=2),
    _cfg("w80_s112_c30_h5_exit095_mh2", score_exit=True, exit_ratio="0.95", min_exit_hold=2),
    _cfg("w80_s112_c30_h5_exit098_mh2", score_exit=True, exit_ratio="0.98", min_exit_hold=2),
    _cfg("w80_s112_c30_h5_exit095_mh3", score_exit=True, exit_ratio="0.95", min_exit_hold=3),
]


def main() -> int:
    base.REPORT_DIR = REPORT_DIR
    base.SCORE_DB = REPORT_DIR / "top5_refine_scores.db"
    base.VARIANTS = VARIANTS
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
