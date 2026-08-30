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


REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_open_condition_search_20260702"
MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


WEIGHTS = [
    ("w25_25_00_50", {"1d": 0.25, "3d": 0.25, "5d": 0.00, "10d": 0.50}),
]

GAPS = [
    ("gap_m8_p3", -0.08, 0.03),
    ("gap_m5_p2", -0.05, 0.02),
    ("gap_m3_p2", -0.03, 0.02),
    ("gap_m8_0", -0.08, 0.00),
    ("gap_0_p3", 0.00, 0.03),
]
AMOUNT_MINS = [0.0, 90_000.0, 150_000.0]
MV_MINS = [0.0, 200_000.0]
ATR_PCTS = [None, 0.09]
PCT_CAPS = [None, 12.0]
TOPNS = [8, 12]


def _annualized(mean_daily: float) -> float:
    return (1.0 + mean_daily) ** 252 - 1.0


def _sharpe(mean_daily: float, std_daily: float) -> float | None:
    if not std_daily or pd.isna(std_daily):
        return None
    return mean_daily / std_daily * (252 ** 0.5)


def _max_drawdown(daily_returns: pd.Series) -> float:
    nav = (1.0 + daily_returns.fillna(0.0)).cumprod()
    peak = nav.cummax()
    return float(-(nav / peak - 1.0).min())


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
                    lead(trade_date, 2) OVER (ORDER BY trade_date) AS next_date
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
                    open,
                    close,
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
                md.close AS signal_close,
                buy.open AS buy_open,
                buy.close AS buy_close,
                nxt.open AS next_open,
                c.buy_date,
                c.next_date,
                buy.open / NULLIF(md.close, 0) - 1 AS open_gap,
                nxt.open / NULLIF(buy.open, 0) - 1 AS ret_oo,
                buy.close / NULLIF(buy.open, 0) - 1 AS ret_oc,
                md.atr_qfq / NULLIF(md.close, 0) AS atr_pct
            FROM ranked r
            JOIN cal c ON c.trade_date = r.trade_date
            JOIN md ON md.trade_date = r.trade_date AND md.stock_code = r.stock_code
            JOIN md buy ON buy.trade_date = c.buy_date AND buy.stock_code = r.stock_code
            LEFT JOIN md nxt ON nxt.trade_date = c.next_date AND nxt.stock_code = r.stock_code
            WHERE c.buy_date IS NOT NULL
              AND c.next_date IS NOT NULL
              AND r.stock_code NOT LIKE '%.BJ'
              AND coalesce(md.ST_TYPE, '') IN ('', '0')
              AND coalesce(md.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(md.name, '') NOT LIKE 'ST%'
              AND coalesce(md.name, '') NOT LIKE '*ST%'
              AND buy.open IS NOT NULL
              AND nxt.open IS NOT NULL
            """
        )

        rows: list[dict] = []
        for (wname, w), (gname, gl, gh), amount_min, mv_min, atr_cap, pct_cap, topn in itertools.product(
            WEIGHTS, GAPS, AMOUNT_MINS, MV_MINS, ATR_PCTS, PCT_CAPS, TOPNS
        ):
            where = [
                f"open_gap BETWEEN {gl} AND {gh}",
                f"amount >= {amount_min}",
                f"total_mv >= {mv_min}",
            ]
            if atr_cap is not None:
                where.append(f"(atr_pct IS NOT NULL AND atr_pct <= {atr_cap})")
            if pct_cap is not None:
                where.append(f"abs(pct_chg) <= {pct_cap}")
            where_sql = " AND ".join(where)
            score_expr = f"({w['1d']} * r1 + {w['3d']} * r3 + {w['5d']} * r5 + {w['10d']} * r10)"
            df = con.execute(
                f"""
                WITH scored AS (
                    SELECT
                        trade_date,
                        stock_code,
                        ret_oo,
                        ret_oc,
                        {score_expr} AS entry_score
                    FROM base
                    WHERE {where_sql}
                ),
                picked AS (
                    SELECT
                        *,
                        row_number() OVER (PARTITION BY trade_date ORDER BY entry_score DESC, stock_code ASC) AS rn
                    FROM scored
                )
                SELECT trade_date, avg(ret_oo) AS daily_oo, avg(ret_oc) AS daily_oc, count(*) AS names
                FROM picked
                WHERE rn <= {topn}
                GROUP BY trade_date
                ORDER BY trade_date
                """
            ).fetchdf()
            if df.empty:
                continue
            coverage = len(df) / 986.0
            avg_names = float(df["names"].mean())
            if coverage < 0.70 or avg_names < max(2.5, topn * 0.75):
                continue
            mean_oo = float(df["daily_oo"].mean())
            std_oo = float(df["daily_oo"].std())
            recent60 = df.tail(60)
            recent120 = df.tail(120)
            rows.append(
                {
                    "case_name": f"{wname}_{gname}_amt{int(amount_min)}_mv{int(mv_min)}_atr{atr_cap or 'none'}_pct{pct_cap or 'none'}_top{topn}",
                    "weight_name": wname,
                    "gap_name": gname,
                    "gap_low": gl,
                    "gap_high": gh,
                    "amount_min": amount_min,
                    "mv_min": mv_min,
                    "atr_cap": atr_cap,
                    "pct_cap": pct_cap,
                    "topn": topn,
                    "signal_days": len(df),
                    "coverage": coverage,
                    "avg_names": avg_names,
                    "mean_daily_oo": mean_oo,
                    "annual_oo": _annualized(mean_oo),
                    "sharpe_oo": _sharpe(mean_oo, std_oo),
                    "max_drawdown_oo": _max_drawdown(df["daily_oo"]),
                    "recent60_annual_oo": _annualized(float(recent60["daily_oo"].mean())),
                    "recent120_annual_oo": _annualized(float(recent120["daily_oo"].mean())),
                    "annual_oc": _annualized(float(df["daily_oc"].mean())),
                }
            )
        out = pd.DataFrame(rows)
        out = out.sort_values(["annual_oo", "sharpe_oo"], ascending=False)
        out.to_csv(REPORT_DIR / "open_condition_grid_summary.csv", index=False, encoding="utf-8")
        stable = out[
            (out["max_drawdown_oo"] <= 0.45)
            & (out["recent60_annual_oo"] > 0)
            & (out["recent120_annual_oo"] > 0)
            & (out["coverage"] >= 0.90)
        ].copy()
        stable.to_csv(REPORT_DIR / "open_condition_stable_candidates.csv", index=False, encoding="utf-8")
        (REPORT_DIR / "run_manifest.json").write_text(
            json.dumps(
                {
                    "manifests": {k: str(v) for k, v in MANIFESTS.items()},
                    "market_db": str(market_db),
                    "note": "local open-to-open screening only; Juejin validation required",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("top")
        print(out.head(20).to_string(index=False))
        print("stable")
        print(stable.head(20).to_string(index=False))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
