from __future__ import annotations

import copy

import tune_current_formal_top5_grid2_20260623 as base


REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "current_formal_concentrated"
)

base.BASE_TARGETS[1] = {1: 0.98}
base.BASE_TARGETS[2] = {1: 0.55, 2: 0.43}
base.BASE_TARGETS[3] = {1: 0.38, 2: 0.32, 3: 0.27}


def _cfg(
    name: str,
    *,
    w10: float,
    top_k: int,
    scale: float = 1.0,
    cap: float = 0.98,
    holding_days: int = 5,
    score_exit: bool = False,
    exit_ratio: str = "0.95",
    min_exit_hold: int = 2,
) -> dict:
    cfg = base._variant(
        name,
        w10=w10,
        top_k=top_k,
        scale=scale,
        cap=cap,
        holding_days=holding_days,
        cash_buffer="0.99",
    )
    env = copy.deepcopy(cfg["env"])
    env["GM_MAX_DAILY_SELLS"] = str(top_k)
    env["GM_OPEN_DAILY_SCORE_EXIT"] = "1" if score_exit else "0"
    env["GM_SCORE_EXIT_ENTRY_RATIO"] = exit_ratio if score_exit else "none"
    env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(min_exit_hold)
    env["GM_STOP_LOSS_PCT"] = "0.08"
    env["GM_TAKE_PROFIT_PCT"] = "none"
    env["GM_LIGHT_STOP_LOSS_PCT"] = "none"
    cfg["env"] = env
    return cfg


VARIANTS = [
    _cfg("top1_w75_h3", w10=0.75, top_k=1, holding_days=3),
    _cfg("top1_w80_h3", w10=0.80, top_k=1, holding_days=3),
    _cfg("top1_w85_h3", w10=0.85, top_k=1, holding_days=3),
    _cfg("top1_w80_h5", w10=0.80, top_k=1, holding_days=5),
    _cfg("top1_w80_h3_exit095", w10=0.80, top_k=1, holding_days=3, score_exit=True, exit_ratio="0.95", min_exit_hold=1),
    _cfg("top2_w75_h3", w10=0.75, top_k=2, holding_days=3),
    _cfg("top2_w80_h3", w10=0.80, top_k=2, holding_days=3),
    _cfg("top2_w85_h3", w10=0.85, top_k=2, holding_days=3),
    _cfg("top2_w80_h5", w10=0.80, top_k=2, holding_days=5),
    _cfg("top3_w75_h3", w10=0.75, top_k=3, holding_days=3),
    _cfg("top3_w80_h3", w10=0.80, top_k=3, holding_days=3),
    _cfg("top3_w85_h3", w10=0.85, top_k=3, holding_days=3),
]


def main() -> int:
    base.REPORT_DIR = REPORT_DIR
    base.SCORE_DB = REPORT_DIR / "concentrated_scores.db"
    base.VARIANTS = VARIANTS
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
