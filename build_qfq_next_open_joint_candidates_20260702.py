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
from stock_daily_data_route import resolve_stock_daily_duckdb_path  # noqa: E402


REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_latest_l4_weight_candidates_20260702"
    / "qfq_next_open_joint_candidates"
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


def _max_drawdown(ret: pd.Series) -> float:
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    peak = nav.cummax()
    return float(-(nav / peak - 1.0).min())


def _target_pct(topn: int, gross: float) -> str:
    return f"{min(gross / topn, 0.34):.5f}"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        label: load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)
        for label, path in MANIFESTS.items()
    }
    market_db = resolve_stock_daily_duckdb_path(require_exists=True)

    con = duckdb.connect()
    try:
        for label, source in sources.items():
            con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{Path(market_db).as_posix()}' AS marketdb (READ_ONLY)")
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
                md.name,
                md.amount,
                md.turnover_rate,
                md.total_mv,
                md.atr_qfq,
                md.pct_chg,
                md.close_qfq AS signal_close_qfq,
                buy.open_qfq AS buy_open_qfq,
                buy.close_qfq AS buy_close_qfq,
                sellref.open_qfq AS sell_ref_open_qfq,
                cal.buy_date,
                cal.sell_ref_date,
                buy.open_qfq / NULLIF(md.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                sellref.open_qfq / NULLIF(buy.open_qfq, 0) - 1 AS ret_open_to_next_open_qfq,
                md.atr_qfq / NULLIF(md.close_qfq, 0) AS atr_pct_qfq
            FROM ranked r
            JOIN cal ON cal.trade_date = r.trade_date
            JOIN md ON md.trade_date = r.trade_date AND md.stock_code = r.stock_code
            JOIN md buy ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
            JOIN md sellref ON sellref.trade_date = cal.sell_ref_date AND sellref.stock_code = r.stock_code
            WHERE cal.buy_date IS NOT NULL
              AND cal.sell_ref_date IS NOT NULL
              AND r.stock_code NOT LIKE '%.BJ'
              AND coalesce(md.ST_TYPE, '') IN ('', '0')
              AND coalesce(md.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(md.name, '') NOT LIKE 'ST%'
              AND coalesce(md.name, '') NOT LIKE '*ST%'
              AND buy.open_qfq IS NOT NULL
              AND sellref.open_qfq IS NOT NULL
              AND md.close_qfq IS NOT NULL
            """
        )

        weights = [
            ("w50_25_00_25", {"1d": 0.50, "3d": 0.25, "5d": 0.00, "10d": 0.25}),
            ("w40_30_00_30", {"1d": 0.40, "3d": 0.30, "5d": 0.00, "10d": 0.30}),
            ("w30_30_10_30", {"1d": 0.30, "3d": 0.30, "5d": 0.10, "10d": 0.30}),
            ("w25_25_00_50", {"1d": 0.25, "3d": 0.25, "5d": 0.00, "10d": 0.50}),
            ("w20_20_10_50", {"1d": 0.20, "3d": 0.20, "5d": 0.10, "10d": 0.50}),
        ]
        filters = [
            ("flat_m6_p2_pct12", -0.06, 0.02, 12.0, 0.12, 90_000.0, 200_000.0),
            ("flat_m5_p1_pct10", -0.05, 0.01, 10.0, 0.10, 150_000.0, 300_000.0),
            ("lowflat_m5_0_pct12", -0.05, 0.00, 12.0, 0.12, 90_000.0, 200_000.0),
            ("lowflat_m4_0_pct10", -0.04, 0.00, 10.0, 0.10, 150_000.0, 300_000.0),
            ("weakup_m2_p2_pct8", -0.02, 0.02, 8.0, 0.10, 150_000.0, 300_000.0),
        ]
        topns = [3, 4, 5, 6]
        gross_by_topn = {3: 0.75, 4: 0.90, 5: 0.90, 6: 1.00}

        local_rows = []
        candidate_defs = []
        for (wname, w), (fname, gap_low, gap_high, pct_cap, atr_cap, amount_min, mv_min), topn in itertools.product(
            weights, filters, topns
        ):
            gross = gross_by_topn[topn]
            score_expr = f"({w['1d']} * r1 + {w['3d']} * r3 + {w['5d']} * r5 + {w['10d']} * r10)"
            df = con.execute(
                f"""
                WITH scored AS (
                    SELECT
                        *,
                        {score_expr} AS entry_score
                    FROM base
                    WHERE buy_open_gap_qfq BETWEEN {gap_low} AND {gap_high}
                      AND abs(pct_chg) <= {pct_cap}
                      AND atr_pct_qfq <= {atr_cap}
                      AND amount >= {amount_min}
                      AND total_mv >= {mv_min}
                ),
                picked AS (
                    SELECT
                        *,
                        row_number() OVER (PARTITION BY trade_date ORDER BY entry_score DESC, stock_code ASC) AS rn
                    FROM scored
                )
                SELECT
                    trade_date,
                    avg(ret_open_to_next_open_qfq) AS daily_ret,
                    count(*) AS names
                FROM picked
                WHERE rn <= {topn}
                GROUP BY trade_date
                ORDER BY trade_date
                """
            ).fetchdf()
            if df.empty:
                continue
            coverage = len(df) / 985.0
            avg_names = float(df["names"].mean())
            if coverage < 0.85 or avg_names < min(topn, 3):
                continue
            ret = df["daily_ret"].astype(float)
            ann = _annualized(float(ret.mean()))
            shp = _sharpe(ret)
            mdd = _max_drawdown(ret)
            recent60 = _annualized(float(ret.tail(60).mean()))
            recent120 = _annualized(float(ret.tail(120).mean()))
            case_name = f"{wname}_{fname}_top{topn}_gross{int(gross*100)}"
            local_rows.append(
                {
                    "case_name": case_name,
                    "weight_name": wname,
                    "filter_name": fname,
                    "topn": topn,
                    "gross_target": gross,
                    "target_pct": _target_pct(topn, gross),
                    "coverage": coverage,
                    "avg_names": avg_names,
                    "annual_local_oo": ann,
                    "sharpe_local_oo": shp,
                    "max_drawdown_local_oo": mdd,
                    "recent60_annual_local_oo": recent60,
                    "recent120_annual_local_oo": recent120,
                    "gap_low": gap_low,
                    "gap_high": gap_high,
                    "pct_cap": pct_cap,
                    "atr_cap": atr_cap,
                    "amount_min": amount_min,
                    "mv_min": mv_min,
                    "weights": json.dumps(w, ensure_ascii=False),
                }
            )
            candidate_defs.append((case_name, wname, w, fname, gap_low, gap_high, pct_cap, atr_cap, amount_min, mv_min, topn, gross))

        local = pd.DataFrame(local_rows).sort_values(
            ["annual_local_oo", "sharpe_local_oo", "max_drawdown_local_oo"],
            ascending=[False, False, True],
        )
        local.to_csv(REPORT_DIR / "local_open_to_open_candidate_summary.csv", index=False, encoding="utf-8-sig")
        stable = local[
            (local["max_drawdown_local_oo"] <= 0.40)
            & (local["recent60_annual_local_oo"] > 0)
            & (local["recent120_annual_local_oo"] > 0)
        ].copy()
        stable.to_csv(REPORT_DIR / "local_stable_candidate_summary.csv", index=False, encoding="utf-8-sig")

        selected = pd.concat([local.head(4), stable.head(4)]).drop_duplicates("case_name").head(8)
        selected.to_csv(REPORT_DIR / "selected_for_juejin.csv", index=False, encoding="utf-8-sig")
        selected_names = set(selected["case_name"].tolist())

        manifest_rows = []
        for case_name, wname, w, fname, gap_low, gap_high, pct_cap, atr_cap, amount_min, mv_min, topn, gross in candidate_defs:
            if case_name not in selected_names:
                continue
            score_expr = f"({w['1d']} * r1 + {w['3d']} * r3 + {w['5d']} * r5 + {w['10d']} * r10)"
            signal_df = con.execute(
                f"""
                WITH scored AS (
                    SELECT
                        *,
                        {score_expr} AS entry_score
                    FROM base
                    WHERE buy_open_gap_qfq BETWEEN {gap_low} AND {gap_high}
                      AND abs(pct_chg) <= {pct_cap}
                      AND atr_pct_qfq <= {atr_cap}
                      AND amount >= {amount_min}
                      AND total_mv >= {mv_min}
                ),
                picked AS (
                    SELECT
                        *,
                        row_number() OVER (PARTITION BY trade_date ORDER BY entry_score DESC, stock_code ASC) AS rn
                    FROM scored
                )
                SELECT
                    trade_date AS signal_date,
                    buy_date,
                    stock_code,
                    name,
                    rn AS rank,
                    entry_score AS pred_prob,
                    entry_score,
                    pred_1d,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg
                FROM picked
                WHERE rn <= {topn}
                ORDER BY signal_date, rn
                """
            ).fetchdf()
            signal_df.insert(2, "symbol", signal_df["stock_code"].map(_symbol))
            signal_df["target_pct"] = _target_pct(topn, gross)
            signal_df["holding_days"] = 1
            signal_df["max_holding_days"] = 1
            signal_df["score_exit_entry_ratio"] = "0.98000"
            signal_df["min_holding_days_before_score_exit"] = 1
            signal_df["score_continue_entry_ratio"] = "9.99000"
            signal_df["signal_stop_loss_pct"] = "0.05000"
            signal_df["signal_take_profit_pct"] = "0.08000"
            signal_df["strategy_variant"] = case_name
            signal_df["filter_name"] = fname
            signal_df["entry_weight_name"] = wname
            signal_df["dynamic_hold_name"] = "h1m1_e098_c999_min1_qfq_open"
            signal_df["buy_day_market_available"] = True
            signal_df["buy_day_hard_gate_complete"] = True
            signal_df["buy_day_st_rejected"] = False
            signal_df["buy_day_open_limit_up_rejected"] = False
            signal_df["latest_market_date"] = "20260701"

            out_dir = REPORT_DIR / case_name
            out_dir.mkdir(parents=True, exist_ok=True)
            signal_file = out_dir / "signals.csv"
            signal_df.to_csv(signal_file, index=False, encoding="utf-8")
            count_by_day = signal_df.groupby("signal_date").size()
            manifest_rows.append(
                {
                    "case_name": case_name,
                    "signal_file": str(signal_file),
                    "rows": int(len(signal_df)),
                    "signal_days": int(count_by_day.size),
                    "avg_names": float(count_by_day.mean()) if not count_by_day.empty else 0.0,
                    "topn": topn,
                    "target_pct": _target_pct(topn, gross),
                }
            )

        pd.DataFrame(manifest_rows).to_csv(REPORT_DIR / "juejin_candidate_manifest.csv", index=False, encoding="utf-8-sig")
        (REPORT_DIR / "run_manifest.json").write_text(
            json.dumps(
                {
                    "manifests": {label: str(path) for label, path in MANIFESTS.items()},
                    "sources": sources,
                    "market_db": str(market_db),
                    "price_fields": {
                        "filter_open": "buy_open_qfq",
                        "filter_close": "signal_close_qfq",
                        "local_eval": "buy_open_qfq -> next_trade_open_qfq",
                    },
                    "boundary": "research-only; local open-to-open is for candidate selection; Juejin validation required",
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print("selected")
        print(selected.to_string(index=False))
        print("manifest")
        print(pd.DataFrame(manifest_rows).to_string(index=False))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
