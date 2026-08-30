from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
SRC_DIR = BASE_DIR / "postrank_open_filter_candidates" / "qfq_rerank_refill_candidates"
SOURCE_SIGNAL_FILE = SRC_DIR / "liq_refill_top3_p435_h1_sigpct_le_m175_bog_le1p5_s96_p10d70_fine" / "signals.csv"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "next_open_dynamic_sizing_candidates"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    {
        "name": "nods_v1_strong55_mid44_weak30",
        "cap": 0.55,
        "strong": 0.55,
        "good": 0.48,
        "base": 0.44,
        "weak": 0.30,
        "very_weak": 0.20,
        "exit_ratio": "0.96000",
    },
    {
        "name": "nods_v2_strong52_mid435_weak34",
        "cap": 0.52,
        "strong": 0.52,
        "good": 0.47,
        "base": 0.435,
        "weak": 0.34,
        "very_weak": 0.24,
        "exit_ratio": "0.96500",
    },
    {
        "name": "nods_v3_strong58_mid45_weak28",
        "cap": 0.58,
        "strong": 0.58,
        "good": 0.50,
        "base": 0.45,
        "weak": 0.28,
        "very_weak": 0.18,
        "exit_ratio": "0.95500",
    },
    {
        "name": "nods_v4_sharpe_bias50_42_25",
        "cap": 0.50,
        "strong": 0.50,
        "good": 0.45,
        "base": 0.42,
        "weak": 0.25,
        "very_weak": 0.15,
        "exit_ratio": "0.97000",
    },
]


def _assign_target(row: pd.Series, case: dict) -> float:
    pct_chg = float(row["signal_pct_chg"])
    gap_pct = float(row["buy_open_gap_pct"])
    amount_rank = float(row["amount_rank"])
    turnover = float(row["turnover_rate"])

    deep_signal = pct_chg <= -5.50
    good_signal = pct_chg <= -3.90
    weak_signal = pct_chg > -2.32
    deep_or_flat_open = gap_pct <= -0.95
    not_up_open = gap_pct <= 0.25
    weak_up_open = gap_pct > 0.25
    liquid = amount_rank >= 0.35 or turnover >= 5.0

    if deep_signal and deep_or_flat_open and liquid:
        return float(case["strong"])
    if good_signal and not_up_open:
        return float(case["good"])
    if weak_signal or weak_up_open:
        return float(case["weak"] if amount_rank >= 0.30 else case["very_weak"])
    return float(case["base"])


def _local_stats(df: pd.DataFrame) -> dict:
    weighted = df.copy()
    weighted["weighted_ret"] = weighted["ret_open_to_next_open_qfq"] * weighted["target_pct_float"]
    ret = weighted.groupby("signal_date")["weighted_ret"].sum().sort_index()
    nav = (1.0 + ret).cumprod()
    peak = nav.cummax()
    std = ret.std(ddof=1)
    return {
        "local_days": int(ret.size),
        "local_annual_weighted_oo": float((1.0 + ret.mean()) ** 252 - 1.0) if ret.size else 0.0,
        "local_sharpe_weighted_oo": float(ret.mean() / std * (252 ** 0.5)) if std else None,
        "local_mdd_weighted_oo": float(-(nav / peak - 1.0).min()) if ret.size else 0.0,
        "local_win": float((ret > 0).mean()) if ret.size else 0.0,
        "avg_target_sum": float(weighted.groupby("signal_date")["target_pct_float"].sum().mean()),
        "max_target_sum": float(weighted.groupby("signal_date")["target_pct_float"].sum().max()),
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
                SELECT trade_date, stock_code, open_qfq, close_qfq, amount, pct_chg
                FROM STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            ),
            cal AS (
                SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
                FROM (SELECT DISTINCT trade_date FROM md)
            )
            SELECT
                sig.*,
                sigmd.pct_chg AS signal_pct_chg,
                buy.open_qfq / NULLIF(sigmd.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                sellref.open_qfq / NULLIF(buy.open_qfq, 0) - 1 AS ret_open_to_next_open_qfq,
                percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.amount) AS amount_rank
            FROM sig
            JOIN cal ON cal.trade_date = sig.buy_date
            JOIN md sigmd ON sigmd.trade_date = sig.signal_date AND sigmd.stock_code = sig.stock_code
            JOIN md buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            JOIN md sellref ON sellref.trade_date = cal.next_trade_date AND sellref.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    manifest_rows = []
    for case in CASES:
        out = enriched.copy()
        out["target_pct_float"] = out.apply(lambda row: _assign_target(row, case), axis=1)
        out["target_pct_float"] = out["target_pct_float"].clip(lower=0.01, upper=float(case["cap"]))
        out["target_pct"] = out["target_pct_float"].map(lambda value: f"{value:.5f}")
        out["score_exit_entry_ratio"] = case["exit_ratio"]
        out["min_holding_days_before_score_exit"] = 1
        out["score_continue_entry_ratio"] = "9.99000"
        out["strategy_variant"] = case["name"]
        out["filter_name"] = case["name"]
        out["dynamic_hold_name"] = "h1m1_next_open_dynamic_sizing"

        stats = _local_stats(out)
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        drop_cols = [
            "signal_pct_chg",
            "buy_open_gap_qfq",
            "ret_open_to_next_open_qfq",
            "amount_rank",
            "target_pct_float",
        ]
        out.drop(columns=[c for c in drop_cols if c in out.columns]).to_csv(signal_file, index=False, encoding="utf-8")
        counts = out.groupby("signal_date").size()
        manifest_rows.append(
            {
                "name": case["name"],
                "signal_file": str(signal_file),
                "rows": int(len(out)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()) if not counts.empty else 0.0,
                "target_cap": float(case["cap"]),
                **stats,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
