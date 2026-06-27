from __future__ import annotations

import sqlite3

import tune_current_formal_3d5d10d_20260623 as base


OLD_FUSION_DB = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "latest_formal_5d10d_stclean_refill"
    / "fusion_5d10d_latest_20260622.db"
)

base.REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "current_formal_top4_weight_refine"
)
base.FUSION_DB = base.REPORT_DIR / "fusion_top4_weight_refine.db"
base.SCORE_DB = base.REPORT_DIR / "scores_top4_weight_refine.db"

base.BASE_TARGETS[4] = {1: 0.27, 2: 0.25, 3: 0.22, 4: 0.18}


def _cfg(name: str, *, w10: float, w5: float) -> dict:
    return {
        "name": name,
        "w10": w10,
        "w5": w5,
        "w3": 0.00,
        "top_k": 4,
        "scale": 1.29,
        "cap": 0.415,
        "holding_days": 5,
        "env": {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.94",
            "GM_SCORE_EXIT_RANK": "0.94",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
        },
    }


base.VARIANTS = [
    _cfg("top4_w95_05_s129_exit094", w10=0.95, w5=0.05),
    _cfg("top4_w90_10_s129_exit094", w10=0.90, w5=0.10),
    _cfg("top4_w85_15_s129_exit094", w10=0.85, w5=0.15),
    _cfg("top4_w80_20_s129_exit094", w10=0.80, w5=0.20),
    _cfg("top4_w75_25_s129_exit094", w10=0.75, w5=0.25),
    _cfg("top4_w70_30_s129_exit094", w10=0.70, w5=0.30),
]


def _build_fusion_db() -> None:
    base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if base.FUSION_DB.exists():
        return
    conn = sqlite3.connect(base.FUSION_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS old", (str(OLD_FUSION_DB),))
        conn.executescript(
            """
            CREATE TABLE fusion_rank_base AS
            SELECT
                trade_date,
                stock_code,
                0.0 AS pred_3d,
                pred_5d,
                pred_10d,
                0.0 AS rank_3d,
                rank_5d,
                rank_10d,
                name,
                pre_close,
                open,
                close,
                amount,
                turnover_rate,
                total_mv,
                atr_qfq,
                limit_times,
                st_type,
                st_type_name
            FROM old.fusion_rank_base;
            CREATE INDEX idx_fusion_trade_stock ON fusion_rank_base(trade_date, stock_code);
            CREATE INDEX idx_fusion_trade ON fusion_rank_base(trade_date);
            """
        )
        conn.commit()
    finally:
        conn.close()


base._build_fusion_db = _build_fusion_db


if __name__ == "__main__":
    raise SystemExit(base.main())
