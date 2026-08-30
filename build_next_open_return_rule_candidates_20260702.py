from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
SRC_DIR = BASE_DIR / "postrank_open_filter_candidates" / "qfq_rerank_refill_candidates"
SOURCE_SIGNAL_FILE = SRC_DIR / "liq_refill_top3_p435_h1_sigpct_le_m175_bog_le1p5_s96_p10d70_fine" / "signals.csv"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "next_open_return_rule_candidates"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "nor_ret_p44_gap_le0_sigpct_le_m250_amt_hi",
        "target_pct": "0.44000",
        "topn": 3,
        "primary": {
            "gap_low": -0.09,
            "gap_high": 0.0,
            "pct_chg_max": -2.50,
            "amount_rank_min": 0.45,
            "turnover_min": 3.0,
            "mv_rank_max": 0.88,
        },
        "refill": {
            "gap_low": -0.10,
            "gap_high": 0.012,
            "pct_chg_max": -1.75,
            "amount_rank_min": 0.30,
            "turnover_min": 2.0,
            "mv_rank_max": 0.92,
        },
        "rank_expr": "entry_score + 0.040 * amount_rank + 0.025 * turnover_rank - 0.030 * mv_rank - 0.020 * gap_rank",
        "exit_ratio": "0.96000",
    },
    {
        "name": "nor_ret_p45_deepgap_refill",
        "target_pct": "0.45000",
        "topn": 3,
        "primary": {
            "gap_low": -0.09,
            "gap_high": -0.002,
            "pct_chg_max": -2.00,
            "amount_rank_min": 0.25,
            "turnover_min": 2.0,
            "mv_rank_max": 0.92,
        },
        "refill": {
            "gap_low": -0.10,
            "gap_high": 0.006,
            "pct_chg_max": -1.75,
            "amount_rank_min": 0.20,
            "turnover_min": 1.5,
            "mv_rank_max": 0.95,
        },
        "rank_expr": "entry_score + 0.030 * amount_rank + 0.020 * turnover_rank - 0.020 * mv_rank - 0.030 * gap_rank",
        "exit_ratio": "0.96000",
    },
    {
        "name": "nor_ret_p42_winrate_bias",
        "target_pct": "0.42000",
        "topn": 3,
        "primary": {
            "gap_low": -0.08,
            "gap_high": -0.003,
            "pct_chg_max": -3.00,
            "amount_rank_min": 0.10,
            "turnover_min": 3.5,
            "mv_rank_max": 0.85,
        },
        "refill": {
            "gap_low": -0.09,
            "gap_high": 0.0,
            "pct_chg_max": -2.20,
            "amount_rank_min": 0.10,
            "turnover_min": 2.5,
            "mv_rank_max": 0.90,
        },
        "rank_expr": "entry_score + 0.020 * amount_rank + 0.040 * turnover_rank - 0.030 * mv_rank - 0.035 * gap_rank",
        "exit_ratio": "0.97000",
    },
    {
        "name": "nor_ret_p47_elastic",
        "target_pct": "0.47000",
        "topn": 3,
        "primary": {
            "gap_low": -0.10,
            "gap_high": 0.006,
            "pct_chg_max": -1.75,
            "amount_rank_min": 0.20,
            "turnover_min": 2.0,
            "mv_rank_max": 0.95,
        },
        "refill": {
            "gap_low": -0.11,
            "gap_high": 0.015,
            "pct_chg_max": -1.50,
            "amount_rank_min": 0.10,
            "turnover_min": 1.5,
            "mv_rank_max": 0.97,
        },
        "rank_expr": "entry_score + 0.025 * amount_rank + 0.020 * turnover_rank - 0.020 * mv_rank - 0.025 * gap_rank",
        "exit_ratio": "0.95500",
    },
    {
        "name": "nor_ret_top4_p34_balanced",
        "target_pct": "0.34000",
        "topn": 4,
        "primary": {
            "gap_low": -0.09,
            "gap_high": 0.0,
            "pct_chg_max": -2.00,
            "amount_rank_min": 0.25,
            "turnover_min": 2.0,
            "mv_rank_max": 0.92,
        },
        "refill": {
            "gap_low": -0.10,
            "gap_high": 0.010,
            "pct_chg_max": -1.75,
            "amount_rank_min": 0.10,
            "turnover_min": 1.5,
            "mv_rank_max": 0.95,
        },
        "rank_expr": "entry_score + 0.030 * amount_rank + 0.020 * turnover_rank - 0.020 * mv_rank - 0.025 * gap_rank",
        "exit_ratio": "0.96000",
    },
]


def _mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return (
        (df["buy_open_gap_qfq"] >= float(cfg["gap_low"]))
        & (df["buy_open_gap_qfq"] <= float(cfg["gap_high"]))
        & (df["signal_pct_chg"] <= float(cfg["pct_chg_max"]))
        & (df["amount_rank"] >= float(cfg["amount_rank_min"]))
        & (df["turnover_rate"] >= float(cfg["turnover_min"]))
        & (df["mv_rank"] <= float(cfg["mv_rank_max"]))
    )


def _local_stats(df: pd.DataFrame) -> dict:
    ret = df.groupby("signal_date")["ret_open_to_next_open_qfq"].mean().sort_index()
    nav = (1.0 + ret).cumprod()
    peak = nav.cummax()
    std = ret.std(ddof=1)
    return {
        "local_days": int(ret.size),
        "local_annual_oo": float((1.0 + ret.mean()) ** 252 - 1.0) if ret.size else 0.0,
        "local_sharpe_oo": float(ret.mean() / std * (252 ** 0.5)) if std else None,
        "local_mdd_oo": float(-(nav / peak - 1.0).min()) if ret.size else 0.0,
        "local_win": float((ret > 0).mean()) if ret.size else 0.0,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SOURCE_SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        con.register("sig", sig)
        enriched = con.execute(
            """
            WITH md AS (
                SELECT
                    trade_date,
                    stock_code,
                    open_qfq,
                    close_qfq,
                    amount,
                    turnover_rate,
                    total_mv,
                    pct_chg
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            ),
            cal AS (
                SELECT
                    trade_date,
                    lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
                FROM (SELECT DISTINCT trade_date FROM md)
            ),
            e AS (
                SELECT
                    sig.*,
                    sigmd.pct_chg AS signal_pct_chg,
                    buy.open_qfq / NULLIF(sigmd.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                    sellref.open_qfq / NULLIF(buy.open_qfq, 0) - 1 AS ret_open_to_next_open_qfq,
                    percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.amount) AS amount_rank,
                    percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.total_mv) AS mv_rank,
                    percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.turnover_rate) AS turnover_rank,
                    percent_rank() OVER (
                        PARTITION BY sig.signal_date
                        ORDER BY buy.open_qfq / NULLIF(sigmd.close_qfq, 0) - 1
                    ) AS gap_rank
                FROM sig
                JOIN cal ON cal.trade_date = sig.buy_date
                JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
                JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
                JOIN md sellref ON sellref.trade_date = cal.next_trade_date AND sellref.stock_code = sig.stock_code
            )
            SELECT * FROM e
            """
        ).fetchdf()
    finally:
        con.close()

    manifest_rows = []
    for case in CASES:
        x = enriched.copy()
        x["rerank_score"] = x.eval(case["rank_expr"])
        primary = x[_mask(x, case["primary"])].copy()
        primary["tier"] = 0
        refill = x[_mask(x, case["refill"])].copy()
        refill["tier"] = 1
        combined = pd.concat([primary, refill], ignore_index=True)
        combined = combined.sort_values(
            ["signal_date", "tier", "rerank_score", "stock_code"],
            ascending=[True, True, False, True],
        )
        combined = combined.drop_duplicates(["signal_date", "stock_code"], keep="first")
        combined = combined.groupby("signal_date", group_keys=False).head(int(case["topn"]))
        combined["rank"] = combined.groupby("signal_date").cumcount() + 1
        combined["target_pct"] = case["target_pct"]
        combined["holding_days"] = 1
        combined["max_holding_days"] = 1
        combined["score_exit_entry_ratio"] = case["exit_ratio"]
        combined["min_holding_days_before_score_exit"] = 1
        combined["score_continue_entry_ratio"] = "9.99000"
        combined["strategy_variant"] = case["name"]
        combined["filter_name"] = case["name"]
        combined["dynamic_hold_name"] = "h1m1_next_open_return_rule"

        stats = _local_stats(combined)
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        drop_cols = [
            "signal_pct_chg",
            "buy_open_gap_qfq",
            "ret_open_to_next_open_qfq",
            "amount_rank",
            "mv_rank",
            "turnover_rank",
            "gap_rank",
            "rerank_score",
            "tier",
        ]
        combined.drop(columns=[c for c in drop_cols if c in combined.columns]).to_csv(
            signal_file, index=False, encoding="utf-8"
        )
        counts = combined.groupby("signal_date").size()
        manifest_rows.append(
            {
                "name": case["name"],
                "signal_file": str(signal_file),
                "rows": int(len(combined)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()) if not counts.empty else 0.0,
                "days_below_target": int((counts < int(case["topn"])).sum()) if not counts.empty else 0,
                "topn": int(case["topn"]),
                "target_pct": case["target_pct"],
                **stats,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
