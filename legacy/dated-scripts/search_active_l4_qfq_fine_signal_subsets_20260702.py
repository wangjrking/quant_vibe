from __future__ import annotations

import itertools
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
BASE_DIR = REPORT_DIR / "active_l4_qfq_rebuild"
SOURCE_SIGNAL_FILE = BASE_DIR / "active_l4_qfq_best_rebuild_p435" / "signals.csv"
OUT_DIR = REPORT_DIR / "active_l4_qfq_fine_subset_search"
MARKET_DB_COPY = REPORT_DIR / "active_l4_qfq_event_search" / "runtime_l2_stock_daily_data_copy.duckdb"


def _annualized(mean_daily: float) -> float:
    return (1.0 + mean_daily) ** 252 - 1.0


def _sharpe(ret: pd.Series) -> float | None:
    std = float(ret.std(ddof=1))
    if not std:
        return None
    return float(ret.mean()) / std * (252 ** 0.5)


def _mdd(ret: pd.Series) -> float:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    peak = nav.cummax()
    return float(-(nav / peak - 1.0).min())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(SOURCE_SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    if not MARKET_DB_COPY.exists():
        raise SystemExit(f"L2 runtime copy not found: {MARKET_DB_COPY}")

    con = duckdb.connect(str(MARKET_DB_COPY), read_only=True)
    try:
        con.register("sig", sig)
        df = con.execute(
            """
            WITH cal AS (
                SELECT trade_date, lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
                FROM (SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA)
            )
            SELECT
                sig.*,
                sellref.open_qfq / NULLIF(buy.open_qfq, 0) - 1 AS ret_open_to_next_open_qfq,
                percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.amount) AS amount_rank,
                percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.turnover_rate) AS turnover_rank,
                percent_rank() OVER (PARTITION BY sig.signal_date ORDER BY sig.total_mv) AS mv_rank
            FROM sig
            JOIN cal ON cal.trade_date = sig.buy_date
            JOIN STOCK_DAILY_DATA buy ON buy.trade_date = sig.buy_date AND buy.stock_code = sig.stock_code
            JOIN STOCK_DAILY_DATA sellref ON sellref.trade_date = cal.next_trade_date AND sellref.stock_code = sig.stock_code
            """
        ).fetchdf()
    finally:
        con.close()

    rows = []
    for pct_max, gap_high, p10_min, score_min, amount_min, turn_min, mv_max, topn, target in itertools.product(
        [-1.75, -4.0, -6.0],
        [0.0, 0.015],
        [0.70, 0.90],
        [0.45, 0.75],
        [0.0, 0.20],
        [0.0, 4.0],
        [0.90, 1.0],
        [1, 3],
        [0.50, 0.70],
    ):
        x = df[
            (df["pct_chg"] <= pct_max)
            & ((df["buy_open_gap_pct"] / 100.0) <= gap_high)
            & (df["pred_10d"] >= p10_min)
            & (df["entry_score"] >= score_min)
            & (df["amount_rank"] >= amount_min)
            & (df["turnover_rate"] >= turn_min)
            & (df["mv_rank"] <= mv_max)
        ].copy()
        if x.empty:
            continue
        x["rank_score"] = x["entry_score"] + 0.02 * x["amount_rank"] + 0.02 * x["turnover_rank"] - 0.01 * x["mv_rank"]
        x = x.sort_values(["signal_date", "rank_score", "stock_code"], ascending=[True, False, True])
        x = x.groupby("signal_date", group_keys=False).head(topn)
        grouped = x.groupby("signal_date").agg(
            daily_ret=("ret_open_to_next_open_qfq", lambda values: float(values.mean()) * target),
            names=("stock_code", "count"),
        )
        if len(grouped) < 60 or float(grouped["names"].mean()) < min(topn, 1.5):
            continue
        ret = grouped["daily_ret"].astype(float)
        annual = _annualized(float(ret.mean()))
        sharpe = _sharpe(ret)
        mdd = _mdd(ret)
        rows.append(
            {
                "name": f"fine_subset_p{str(target).replace('.','p')}_t{topn}_pc{str(abs(pct_max)).replace('.','p')}_g{str(gap_high).replace('.','p')}_p10{int(p10_min*100)}_s{int(score_min*100)}_a{int(amount_min*100)}_tr{int(turn_min)}_mv{int(mv_max*100)}",
                "pct_max": pct_max,
                "gap_high": gap_high,
                "p10_min": p10_min,
                "score_min": score_min,
                "amount_min": amount_min,
                "turn_min": turn_min,
                "mv_max": mv_max,
                "topn": topn,
                "target_pct": target,
                "days": int(len(grouped)),
                "avg_names": float(grouped["names"].mean()),
                "local_annual": annual,
                "local_sharpe": sharpe,
                "local_mdd": mdd,
                "recent60": _annualized(float(ret.tail(60).mean())),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("no local subset candidates")
    summary = summary.sort_values(["local_sharpe", "local_annual"], ascending=[False, False])
    summary.to_csv(OUT_DIR / "local_subset_summary.csv", index=False, encoding="utf-8-sig")
    selected = summary[
        (summary["local_annual"] >= 3.0)
        & (summary["local_sharpe"] >= 3.0)
        & (summary["local_mdd"] <= 0.40)
        & (summary["recent60"] > 0)
    ].head(6)
    if selected.empty:
        selected = summary.head(6)
    selected.to_csv(OUT_DIR / "selected_for_juejin.csv", index=False, encoding="utf-8-sig")

    manifest_rows = []
    for row in selected.to_dict("records"):
        x = df[
            (df["pct_chg"] <= row["pct_max"])
            & ((df["buy_open_gap_pct"] / 100.0) <= row["gap_high"])
            & (df["pred_10d"] >= row["p10_min"])
            & (df["entry_score"] >= row["score_min"])
            & (df["amount_rank"] >= row["amount_min"])
            & (df["turnover_rate"] >= row["turn_min"])
            & (df["mv_rank"] <= row["mv_max"])
        ].copy()
        x["rank_score"] = x["entry_score"] + 0.02 * x["amount_rank"] + 0.02 * x["turnover_rank"] - 0.01 * x["mv_rank"]
        x = x.sort_values(["signal_date", "rank_score", "stock_code"], ascending=[True, False, True])
        x = x.groupby("signal_date", group_keys=False).head(int(row["topn"]))
        x["rank"] = x.groupby("signal_date").cumcount() + 1
        x["target_pct"] = f"{float(row['target_pct']):.5f}"
        x["strategy_variant"] = row["name"]
        x["filter_name"] = "active_l4_qfq_fine_subset"
        x["dynamic_hold_name"] = "h1m1_active_qfq_fine_subset"
        out_dir = OUT_DIR / row["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        drop_cols = ["ret_open_to_next_open_qfq", "amount_rank", "turnover_rank", "mv_rank", "rank_score"]
        x.drop(columns=[c for c in drop_cols if c in x.columns]).to_csv(signal_file, index=False, encoding="utf-8")
        counts = x.groupby("signal_date").size()
        manifest_rows.append(
            {
                "name": row["name"],
                "signal_file": str(signal_file),
                "score_db": str(BASE_DIR / "score.duckdb"),
                "score_table": "score",
                "rows": int(len(x)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()),
                "days_below_topn": int((counts < int(row["topn"])).sum()),
                **row,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "juejin_candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
