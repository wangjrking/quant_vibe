from __future__ import annotations

import itertools
import json
import shutil
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
sys.path.insert(0, str(MAIN))

from prediction_manifest import load_prediction_source_manifest  # noqa: E402
from stock_daily_data_route import resolve_stock_daily_duckdb_path  # noqa: E402


REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "active_l4_qfq_event_search"
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
    return float(ret.mean()) / std * (252 ** 0.5)


def _mdd(ret: pd.Series) -> float:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    peak = nav.cummax()
    return float(-(nav / peak - 1.0).min())


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        label: load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)
        for label, path in MANIFESTS.items()
    }
    market_db = Path(resolve_stock_daily_duckdb_path(require_exists=True))
    market_db_copy = REPORT_DIR / "runtime_l2_stock_daily_data_copy.duckdb"
    if not market_db_copy.exists():
        shutil.copy2(market_db, market_db_copy)

    con = duckdb.connect()
    try:
        for label, source in sources.items():
            con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{market_db_copy.as_posix()}' AS marketdb (READ_ONLY)")
        tables = {label: sources[label]["table"] for label in sources}
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base AS
            WITH cal AS (
                SELECT
                    trade_date,
                    lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                    lead(trade_date, 2) OVER (ORDER BY trade_date) AS sell_ref_date
                FROM (
                    SELECT DISTINCT trade_date
                    FROM marketdb.STOCK_DAILY_DATA
                    WHERE trade_date BETWEEN '20220606' AND '20260701'
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
            ),
            ranked AS (
                SELECT
                    *,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_1d) AS r1,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS r3,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS r5,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS r10
                FROM preds
            ),
            md AS (
                SELECT
                    trade_date,
                    stock_code,
                    name,
                    open_qfq,
                    close_qfq,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    ST_TYPE,
                    ST_TYPE_name
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                r.*,
                (0.25 * r1 + 0.25 * r3 + 0.50 * r10)::DOUBLE AS entry_score,
                md.name,
                md.amount,
                md.turnover_rate,
                md.total_mv,
                md.atr_qfq,
                md.pct_chg,
                buy.open_qfq / NULLIF(md.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                md.atr_qfq / NULLIF(md.close_qfq, 0) AS atr_pct_qfq,
                sellref.open_qfq / NULLIF(buy.open_qfq, 0) - 1 AS ret_open_to_next_open_qfq,
                cal.buy_date,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY md.amount) AS amount_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY md.turnover_rate) AS turnover_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY md.total_mv) AS mv_rank
            FROM ranked r
            JOIN cal ON cal.trade_date = r.trade_date
            JOIN md ON md.trade_date = r.trade_date AND md.stock_code = r.stock_code
            JOIN md buy ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
            JOIN md sellref ON sellref.trade_date = cal.sell_ref_date AND sellref.stock_code = r.stock_code
            WHERE cal.buy_date IS NOT NULL
              AND cal.sell_ref_date IS NOT NULL
              AND r.stock_code NOT LIKE '%.BJ'
              AND r.pred_10d >= 0.70
              AND md.pct_chg <= -1.75
              AND buy.open_qfq / NULLIF(md.close_qfq, 0) - 1 BETWEEN -0.12 AND 0.015
              AND coalesce(md.ST_TYPE, '') IN ('', '0')
              AND coalesce(md.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(md.name, '') NOT LIKE 'ST%'
              AND coalesce(md.name, '') NOT LIKE '*ST%'
              AND md.close_qfq IS NOT NULL
              AND buy.open_qfq IS NOT NULL
              AND sellref.open_qfq IS NOT NULL
            """
        )

        base_df = con.execute("SELECT * FROM base").fetchdf()
        rows = []
        for pct_max, gap_low, gap_high, p10_min, score_min, amount_rank_min, turn_min, mv_max, topn, target in itertools.product(
            [-1.75, -4.0, -6.0],
            [-0.12, -0.08],
            [0.0, 0.015],
            [0.70, 0.90],
            [0.45, 0.75],
            [0.0, 0.2],
            [0.0, 2.0],
            [0.90, 1.0],
            [1, 3],
            [0.435, 0.58],
        ):
            if topn == 1 and target < 0.50:
                continue
            x = base_df[
                (base_df["pct_chg"] <= pct_max)
                & (base_df["buy_open_gap_qfq"] >= gap_low)
                & (base_df["buy_open_gap_qfq"] <= gap_high)
                & (base_df["pred_10d"] >= p10_min)
                & (base_df["entry_score"] >= score_min)
                & (base_df["amount_rank"] >= amount_rank_min)
                & (base_df["turnover_rate"] >= turn_min)
                & (base_df["mv_rank"] <= mv_max)
            ].copy()
            if x.empty:
                continue
            x["rank_score"] = x["entry_score"] + 0.020 * x["amount_rank"] + 0.020 * x["turnover_rank"] - 0.015 * x["mv_rank"]
            x = x.sort_values(["trade_date", "rank_score", "stock_code"], ascending=[True, False, True])
            x = x.groupby("trade_date", group_keys=False).head(topn)
            grouped = x.groupby("trade_date").agg(
                daily_ret=("ret_open_to_next_open_qfq", lambda values: float(values.mean()) * target),
                names=("stock_code", "count"),
            ).reset_index()
            days = len(grouped)
            avg_names = float(grouped["names"].mean())
            if days < 80 or avg_names < min(topn, 2):
                continue
            ret = grouped["daily_ret"].astype(float)
            row = {
                "name": f"evt_p{str(target).replace('.','p')}_t{topn}_pc{str(abs(pct_max)).replace('.','p')}_g{str(gap_high).replace('.','p')}_p10{int(p10_min*100)}_s{int(score_min*100)}_a{int(amount_rank_min*100)}_tr{int(turn_min)}_mv{int(mv_max*100)}",
                "pct_max": pct_max,
                "gap_low": gap_low,
                "gap_high": gap_high,
                "p10_min": p10_min,
                "score_min": score_min,
                "amount_rank_min": amount_rank_min,
                "turn_min": turn_min,
                "mv_max": mv_max,
                "topn": topn,
                "target_pct": target,
                "days": days,
                "avg_names": avg_names,
                "local_annual": _annualized(float(ret.mean())),
                "local_sharpe": _sharpe(ret),
                "local_mdd": _mdd(ret),
                "recent60": _annualized(float(ret.tail(60).mean())) if days >= 60 else None,
            }
            rows.append(row)

        local = pd.DataFrame(rows)
        local = local.sort_values(["local_sharpe", "local_annual"], ascending=[False, False])
        local.to_csv(REPORT_DIR / "local_event_grid_summary.csv", index=False, encoding="utf-8-sig")
        selected = local[
            (local["local_annual"] >= 3.0)
            & (local["local_sharpe"] >= 3.0)
            & (local["local_mdd"] <= 0.40)
            & (local["recent60"] > 0)
        ].head(6)
        if selected.empty:
            selected = local.head(6)
        selected.to_csv(REPORT_DIR / "selected_for_juejin.csv", index=False, encoding="utf-8-sig")

        # Shared score table for score-exit.
        score_db = REPORT_DIR / "score.duckdb"
        if score_db.exists():
            score_db.unlink()
        con.execute(f"ATTACH '{score_db.as_posix()}' AS scoreout")
        con.execute(
            """
            CREATE TABLE scoreout.score AS
            SELECT trade_date, stock_code, entry_score AS pred_prob FROM base
            """
        )
        con.execute("DETACH scoreout")

        manifest_rows = []
        for row in selected.to_dict("records"):
            name = row["name"]
            signal_df = base_df[
                (base_df["pct_chg"] <= row["pct_max"])
                & (base_df["buy_open_gap_qfq"] >= row["gap_low"])
                & (base_df["buy_open_gap_qfq"] <= row["gap_high"])
                & (base_df["pred_10d"] >= row["p10_min"])
                & (base_df["entry_score"] >= row["score_min"])
                & (base_df["amount_rank"] >= row["amount_rank_min"])
                & (base_df["turnover_rate"] >= row["turn_min"])
                & (base_df["mv_rank"] <= row["mv_max"])
            ].copy()
            signal_df["rank_score"] = (
                signal_df["entry_score"]
                + 0.020 * signal_df["amount_rank"]
                + 0.020 * signal_df["turnover_rank"]
                - 0.015 * signal_df["mv_rank"]
            )
            signal_df = signal_df.sort_values(["trade_date", "rank_score", "stock_code"], ascending=[True, False, True])
            signal_df = signal_df.groupby("trade_date", group_keys=False).head(int(row["topn"]))
            signal_df["rank"] = signal_df.groupby("trade_date").cumcount() + 1
            signal_df = signal_df.rename(columns={"trade_date": "signal_date"})
            signal_df = signal_df[
                [
                    "signal_date",
                    "buy_date",
                    "stock_code",
                    "name",
                    "rank",
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
                    "buy_open_gap_qfq",
                ]
            ]
            signal_df["pred_prob"] = signal_df["entry_score"]
            signal_df = signal_df[
                [
                    "signal_date",
                    "buy_date",
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
                    "buy_open_gap_qfq",
                ]
            ]
            signal_df.insert(2, "symbol", signal_df["stock_code"].map(_symbol))
            signal_df["target_pct"] = f"{float(row['target_pct']):.5f}"
            signal_df["holding_days"] = 1
            signal_df["max_holding_days"] = 1
            signal_df["score_exit_entry_ratio"] = "0.96000"
            signal_df["min_holding_days_before_score_exit"] = 1
            signal_df["score_continue_entry_ratio"] = "9.99000"
            signal_df["signal_stop_loss_pct"] = "0.05000"
            signal_df["signal_take_profit_pct"] = "0.08000"
            signal_df["strategy_variant"] = name
            signal_df["filter_name"] = "active_l4_qfq_event"
            signal_df["entry_weight_name"] = "w25_25_00_50"
            signal_df["dynamic_hold_name"] = "h1m1_active_qfq_event"
            signal_df["buy_day_market_available"] = True
            signal_df["buy_day_hard_gate_complete"] = True
            signal_df["buy_day_st_rejected"] = False
            signal_df["buy_day_open_limit_up_rejected"] = False
            signal_df["latest_market_date"] = "20260701"
            signal_df["buy_open_gap_pct"] = signal_df["buy_open_gap_qfq"] * 100.0
            signal_df = signal_df.drop(columns=["buy_open_gap_qfq"])
            out_dir = REPORT_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            signal_file = out_dir / "signals.csv"
            signal_df.to_csv(signal_file, index=False, encoding="utf-8")
            counts = signal_df.groupby("signal_date").size()
            manifest_rows.append(
                {
                    "name": name,
                    "signal_file": str(signal_file),
                    "score_db": str(score_db),
                    "score_table": "score",
                    "rows": int(len(signal_df)),
                    "signal_days": int(counts.size),
                    "avg_names": float(counts.mean()) if not counts.empty else 0.0,
                    "days_below_topn": int((counts < int(row["topn"])).sum()) if not counts.empty else 0,
                    **row,
                }
            )
        pd.DataFrame(manifest_rows).to_csv(REPORT_DIR / "juejin_candidate_manifest.csv", index=False, encoding="utf-8-sig")
        (REPORT_DIR / "run_manifest.json").write_text(
            json.dumps(
                {
                    "source_manifests": {label: str(path) for label, path in MANIFESTS.items()},
                    "loaded_sources": sources,
                    "market_db": str(market_db),
                    "market_db_runtime_copy": str(market_db_copy),
                    "score_db": str(score_db),
                    "price_fields": {
                        "buy_open_gap": "buy.open_qfq / signal.close_qfq - 1",
                        "local_eval": "buy.open_qfq -> next_trade_open_qfq",
                    },
                    "boundary": "research-only; Juejin validation required",
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(pd.DataFrame(manifest_rows).to_string(index=False))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
