from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
sys.path.insert(0, str(MAIN))

from prediction_manifest import load_prediction_source_manifest  # noqa: E402


REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "active_l4_rawpred_open_event_search"
)
MARKET_DB_COPY = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "active_l4_qfq_event_search"
    / "runtime_l2_stock_daily_data_copy.duckdb"
)
MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _annualized(mean_daily: float) -> float:
    return (1.0 + mean_daily) ** 252 - 1.0


def _sharpe(ret: pd.Series) -> float | None:
    std = float(ret.std(ddof=1))
    if not std:
        return None
    return float(ret.mean()) / std * (252**0.5)


def _mdd(ret: pd.Series) -> float:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    return float(-(nav / nav.cummax() - 1.0).min())


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not MARKET_DB_COPY.exists():
        raise SystemExit(f"runtime market db copy not found: {MARKET_DB_COPY}")

    sources = {
        label: load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)
        for label, path in MANIFESTS.items()
    }
    for label, source in sources.items():
        if source["source_type"] != "duckdb_table":
            raise SystemExit(f"{label} is not duckdb_table: {source}")

    con = duckdb.connect()
    try:
        for label, source in sources.items():
            con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{MARKET_DB_COPY.as_posix()}' AS marketdb (READ_ONLY)")
        tables = {label: source["table"] for label, source in sources.items()}

        # Keep this base deliberately narrow. Earlier full-rank searches were slow and
        # produced fragile candidates; here we filter on raw formal predictions first,
        # then optimize only the buy-day open event and sell frequency.
        base = con.execute(
            f"""
            WITH cal AS (
                SELECT
                    trade_date,
                    lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                    lead(trade_date, 2) OVER (ORDER BY trade_date) AS sell_ref_date
                FROM (
                    SELECT DISTINCT trade_date
                    FROM marketdb.STOCK_DAILY_DATA
                    WHERE trade_date BETWEEN '20220606' AND '20260702'
                )
            ),
            preds AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p1.pred_prob AS pred_1d,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM l4_10d."{tables['10d']}" p10
                JOIN l4_5d."{tables['5d']}" p5 USING (trade_date, stock_code)
                JOIN l4_3d."{tables['3d']}" p3 USING (trade_date, stock_code)
                JOIN l4_1d."{tables['1d']}" p1 USING (trade_date, stock_code)
                WHERE p10.trade_date BETWEEN '20220606' AND '20260630'
                  AND p10.stock_code NOT LIKE '%.BJ'
                  AND p10.pred_prob >= 0.93
            ),
            md AS (
                SELECT
                    trade_date,
                    stock_code,
                    name,
                    market,
                    industry,
                    open_qfq,
                    close_qfq,
                    high_qfq,
                    low_qfq,
                    pre_close_qfq,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    ST_TYPE,
                    ST_TYPE_name
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260702'
            )
            SELECT
                p.trade_date AS signal_date,
                cal.buy_date,
                p.stock_code,
                md.name,
                md.market,
                md.industry,
                p.pred_1d,
                p.pred_3d,
                p.pred_5d,
                p.pred_10d,
                (
                    0.10 * p.pred_1d
                    + 0.15 * p.pred_3d
                    + 0.25 * p.pred_5d
                    + 0.50 * p.pred_10d
                )::DOUBLE AS entry_score,
                md.amount,
                md.turnover_rate,
                md.total_mv,
                md.atr_qfq,
                md.pct_chg,
                buy.open_qfq / NULLIF(md.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                sellref.open_qfq / NULLIF(buy.open_qfq, 0) - 1 AS ret_open_to_next_open_qfq,
                md.atr_qfq / NULLIF(md.close_qfq, 0) AS atr_pct_qfq
            FROM preds p
            JOIN cal ON cal.trade_date = p.trade_date
            JOIN md ON md.trade_date = p.trade_date AND md.stock_code = p.stock_code
            JOIN md buy ON buy.trade_date = cal.buy_date AND buy.stock_code = p.stock_code
            JOIN md sellref ON sellref.trade_date = cal.sell_ref_date AND sellref.stock_code = p.stock_code
            WHERE cal.buy_date IS NOT NULL
              AND cal.sell_ref_date IS NOT NULL
              AND md.close_qfq IS NOT NULL
              AND buy.open_qfq IS NOT NULL
              AND sellref.open_qfq IS NOT NULL
              AND coalesce(md.ST_TYPE, '') IN ('', '0')
              AND coalesce(md.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(md.name, '') NOT LIKE 'ST%'
              AND coalesce(md.name, '') NOT LIKE '*ST%'
              AND md.amount >= 150000
              AND md.total_mv >= 200000
              AND md.turnover_rate >= 1.0
              AND md.turnover_rate IS NOT NULL
              AND md.atr_qfq IS NOT NULL
              AND buy.open_qfq / NULLIF(md.close_qfq, 0) - 1 BETWEEN -0.08 AND 0.03
            """
        ).fetchdf()
    finally:
        con.close()

    if base.empty:
        raise SystemExit("empty base")

    base["amount_rank"] = base.groupby("signal_date")["amount"].rank(pct=True)
    base["turnover_rank"] = base.groupby("signal_date")["turnover_rate"].rank(pct=True)
    base["mv_rank"] = base.groupby("signal_date")["total_mv"].rank(pct=True)
    base["atr_rank"] = base.groupby("signal_date")["atr_pct_qfq"].rank(pct=True)
    base.to_parquet(REPORT_DIR / "rawpred_open_event_base.parquet", index=False)

    rows = []
    grid = itertools.product(
        [0.93, 0.95, 0.97, 0.985],
        [None, 0.50, 0.70],
        [None, 0.55],
        [-0.08, -0.04, 0.0],
        [0.005, 0.015, 0.030],
        [0.0, 0.30, 0.55],
        [0.0, 2.0, 4.0],
        [0.60, 0.80, 1.00],
        [1, 2, 3, 5],
        [0.20, 0.35, 0.50, 0.70],
    )
    for p10_min, p5_min, p1_min, gap_low, gap_high, amount_rank_min, turn_min, atr_rank_max, topn, target in grid:
        if target * topn > 1.05:
            continue
        x = base[
            (base["pred_10d"] >= p10_min)
            & (base["buy_open_gap_qfq"] >= gap_low)
            & (base["buy_open_gap_qfq"] <= gap_high)
            & (base["amount_rank"] >= amount_rank_min)
            & (base["turnover_rate"] >= turn_min)
            & (base["atr_rank"] <= atr_rank_max)
        ].copy()
        if p5_min is not None:
            x = x[x["pred_5d"] >= p5_min]
        if p1_min is not None:
            x = x[x["pred_1d"] >= p1_min]
        if x.empty:
            continue
        x["rank_score"] = (
            x["entry_score"]
            + 0.035 * x["amount_rank"]
            + 0.020 * x["turnover_rank"]
            - 0.020 * x["atr_rank"]
            - 0.010 * x["mv_rank"]
        )
        x = x.sort_values(["signal_date", "rank_score", "stock_code"], ascending=[True, False, True])
        x = x.groupby("signal_date", group_keys=False).head(topn)
        daily = x.groupby("signal_date").agg(
            daily_ret=("ret_open_to_next_open_qfq", lambda s: float(s.mean()) * target),
            names=("stock_code", "count"),
        )
        if len(daily) < 120:
            continue
        if float(daily["names"].mean()) < min(topn, 1.4):
            continue
        ret = daily["daily_ret"].astype(float)
        annual = _annualized(float(ret.mean()))
        sharpe = _sharpe(ret)
        mdd = _mdd(ret)
        rows.append(
            {
                "name": (
                    f"rawp_p10{int(p10_min*100)}"
                    f"_p5{('x' if p5_min is None else int(p5_min*100))}"
                    f"_p1{('x' if p1_min is None else int(p1_min*100))}"
                    f"_g{int(gap_low*1000)}to{int(gap_high*1000)}"
                    f"_a{int(amount_rank_min*100)}_tr{int(turn_min)}"
                    f"_atr{int(atr_rank_max*100)}_top{topn}_pos{int(target*100)}"
                ),
                "p10_min": p10_min,
                "p5_min": p5_min,
                "p1_min": p1_min,
                "gap_low": gap_low,
                "gap_high": gap_high,
                "amount_rank_min": amount_rank_min,
                "turn_min": turn_min,
                "atr_rank_max": atr_rank_max,
                "topn": topn,
                "target_pct": target,
                "days": int(len(daily)),
                "avg_names": float(daily["names"].mean()),
                "local_annual": annual,
                "local_sharpe": sharpe,
                "local_mdd": mdd,
                "recent60": _annualized(float(ret.tail(60).mean())) if len(ret) >= 60 else None,
                "recent120": _annualized(float(ret.tail(120).mean())) if len(ret) >= 120 else None,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("no candidates")
    summary = summary.sort_values(["local_sharpe", "local_annual"], ascending=[False, False])
    summary.to_csv(REPORT_DIR / "local_rawpred_open_event_summary.csv", index=False, encoding="utf-8-sig")
    selected = summary[
        (summary["local_annual"] >= 3.0)
        & (summary["local_sharpe"] >= 3.0)
        & (summary["local_mdd"] <= 0.40)
        & (summary["recent60"] > 0)
        & (summary["recent120"] > 0)
    ].head(8)
    if selected.empty:
        selected = summary.head(8)
    selected.to_csv(REPORT_DIR / "selected_for_juejin.csv", index=False, encoding="utf-8-sig")

    score_db = REPORT_DIR / "score.duckdb"
    if score_db.exists():
        score_db.unlink()
    con = duckdb.connect(str(score_db))
    try:
        con.register("base", base[["signal_date", "stock_code", "entry_score"]])
        con.execute(
            """
            CREATE TABLE score AS
            SELECT signal_date AS trade_date, stock_code, entry_score AS pred_prob
            FROM base
            """
        )
        con.execute("CHECKPOINT")
    finally:
        con.close()

    manifest_rows = []
    for row in selected.to_dict("records"):
        x = base[
            (base["pred_10d"] >= row["p10_min"])
            & (base["buy_open_gap_qfq"] >= row["gap_low"])
            & (base["buy_open_gap_qfq"] <= row["gap_high"])
            & (base["amount_rank"] >= row["amount_rank_min"])
            & (base["turnover_rate"] >= row["turn_min"])
            & (base["atr_rank"] <= row["atr_rank_max"])
        ].copy()
        if pd.notna(row["p5_min"]):
            x = x[x["pred_5d"] >= row["p5_min"]]
        if pd.notna(row["p1_min"]):
            x = x[x["pred_1d"] >= row["p1_min"]]
        x["rank_score"] = (
            x["entry_score"]
            + 0.035 * x["amount_rank"]
            + 0.020 * x["turnover_rank"]
            - 0.020 * x["atr_rank"]
            - 0.010 * x["mv_rank"]
        )
        x = x.sort_values(["signal_date", "rank_score", "stock_code"], ascending=[True, False, True])
        x = x.groupby("signal_date", group_keys=False).head(int(row["topn"]))
        x["rank"] = x.groupby("signal_date").cumcount() + 1
        x.insert(2, "symbol", x["stock_code"].map(_symbol))
        x["pred_prob"] = x["entry_score"]
        x["target_pct"] = f"{float(row['target_pct']):.5f}"
        x["holding_days"] = 1
        x["max_holding_days"] = 1
        x["score_exit_entry_ratio"] = "0.96000"
        x["min_holding_days_before_score_exit"] = 1
        x["score_continue_entry_ratio"] = "9.99000"
        x["signal_stop_loss_pct"] = "0.05000"
        x["signal_take_profit_pct"] = "0.08000"
        x["strategy_variant"] = row["name"]
        x["filter_name"] = "active_l4_rawpred_open_event"
        x["entry_weight_name"] = "rawpred_10d_dominant"
        x["dynamic_hold_name"] = "h1m1_rawpred_open_event"
        x["buy_day_market_available"] = True
        x["buy_day_hard_gate_complete"] = True
        x["buy_day_st_rejected"] = False
        x["buy_day_open_limit_up_rejected"] = False
        x["latest_market_date"] = "20260701"
        x["buy_open_gap_pct"] = x["buy_open_gap_qfq"] * 100.0
        out_cols = [
            "signal_date",
            "buy_date",
            "symbol",
            "stock_code",
            "name",
            "rank",
            "pred_prob",
            "entry_score",
            "pred_1d",
            "pred_3d",
            "pred_5d",
            "pred_10d",
            "amount",
            "turnover_rate",
            "total_mv",
            "atr_qfq",
            "pct_chg",
            "buy_open_gap_pct",
            "target_pct",
            "holding_days",
            "max_holding_days",
            "score_exit_entry_ratio",
            "min_holding_days_before_score_exit",
            "score_continue_entry_ratio",
            "signal_stop_loss_pct",
            "signal_take_profit_pct",
            "strategy_variant",
            "filter_name",
            "entry_weight_name",
            "dynamic_hold_name",
            "buy_day_market_available",
            "buy_day_hard_gate_complete",
            "buy_day_st_rejected",
            "buy_day_open_limit_up_rejected",
            "latest_market_date",
        ]
        out_dir = REPORT_DIR / row["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        x[out_cols].to_csv(signal_file, index=False, encoding="utf-8")
        counts = x.groupby("signal_date").size()
        manifest_rows.append(
            {
                "name": row["name"],
                "signal_file": str(signal_file),
                "score_db": str(score_db),
                "score_table": "score",
                "rows": int(len(x)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()),
                "days_below_topn": int((counts < int(row["topn"])).sum()),
                **row,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(REPORT_DIR / "juejin_candidate_manifest.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "l4_manifests": {k: str(v) for k, v in MANIFESTS.items()},
                "sources": sources,
                "market_db_copy": str(MARKET_DB_COPY),
                "score_db": str(score_db),
                "base_rows": int(len(base)),
                "candidate_count": int(len(summary)),
                "boundary": "research-only; Juejin validation required before any admission claim",
                "qfq_semantics": {
                    "buy_open_gap": "buy.open_qfq / signal.close_qfq - 1",
                    "local_return": "buy.open_qfq to next trade open_qfq",
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
