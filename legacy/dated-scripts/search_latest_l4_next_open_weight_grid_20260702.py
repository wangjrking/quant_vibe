from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(r"D:\work\quant\quant_mcp") / "quant" / "main"))
from prediction_manifest import load_prediction_source_manifest
from stock_daily_data_route import resolve_stock_daily_duckdb_path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_next_open_weight_search_20260702"
MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def _annualized(mean_daily: float) -> float:
    return (1.0 + mean_daily) ** 252 - 1.0


def _sharpe(mean_daily: float, std_daily: float) -> float | None:
    if not std_daily or pd.isna(std_daily):
        return None
    return mean_daily / std_daily * (252 ** 0.5)


def _max_drawdown(daily_returns: pd.Series) -> float:
    nav = (1.0 + daily_returns.fillna(0.0)).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(-dd.min())


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        sources = {label: load_prediction_source_manifest(path, require_approved=True, allow_legacy=False) for label, path in MANIFESTS.items()}
        market_db = resolve_stock_daily_duckdb_path(require_exists=True)
        for label, source in sources.items():
            con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{Path(market_db).as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE base AS
            WITH preds AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p1.pred_prob AS pred_1d,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM l4_10d."{table10}" p10
                JOIN l4_5d."{table5}" p5 USING (trade_date, stock_code)
                JOIN l4_3d."{table3}" p3 USING (trade_date, stock_code)
                JOIN l4_1d."{table1}" p1 USING (trade_date, stock_code)
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
                    stock_code,
                    trade_date,
                    name,
                    open,
                    close,
                    amount,
                    total_mv,
                    ST_TYPE,
                    ST_TYPE_name,
                    lead(trade_date) OVER (PARTITION BY stock_code ORDER BY trade_date) AS next_date,
                    lead(open) OVER (PARTITION BY stock_code ORDER BY trade_date) AS next_open,
                    lead(close) OVER (PARTITION BY stock_code ORDER BY trade_date) AS next_close,
                    lead(open, 2) OVER (PARTITION BY stock_code ORDER BY trade_date) AS next2_open
                FROM marketdb.STOCK_DAILY_DATA
                WHERE trade_date BETWEEN '20220606' AND '20260701'
            )
            SELECT
                r.trade_date,
                r.stock_code,
                md.name,
                md.close,
                md.next_date,
                md.next_open,
                md.next_close,
                md.next2_open,
                (md.next_close / nullif(md.next_open, 0) - 1) AS next_open_to_close,
                (md.next2_open / nullif(md.next_open, 0) - 1) AS next_open_to_next_open,
                (md.next_open / nullif(md.close, 0) - 1) AS open_gap,
                r.r1,
                r.r3,
                r.r5,
                r.r10
            FROM ranked r
            JOIN md ON r.trade_date = md.trade_date AND r.stock_code = md.stock_code
            WHERE md.next_open IS NOT NULL
              AND md.next_close IS NOT NULL
              AND md.next2_open IS NOT NULL
              AND r.stock_code NOT LIKE '%.BJ'
              AND substr(r.stock_code, 1, 1) NOT IN ('4', '8')
              AND NOT (
                  upper(coalesce(md.name, '')) LIKE 'ST%'
                  OR upper(coalesce(md.name, '')) LIKE '*ST%'
                  OR coalesce(md.ST_TYPE, '') NOT IN ('', '0')
                  OR coalesce(md.ST_TYPE_name, '') <> ''
              )
              AND md.amount >= 150000
              AND md.total_mv >= 300000
              AND md.open <= 150
            """.format(
                table1=sources["1d"]["table"],
                table3=sources["3d"]["table"],
                table5=sources["5d"]["table"],
                table10=sources["10d"]["table"],
            )
        )

        total_days = con.execute("SELECT count(DISTINCT trade_date) FROM base").fetchone()[0]
        weight_grid: list[tuple[float, float, float, float]] = []
        weight_unit = 4
        for w1_i, w3_i, w5_i in itertools.product(range(0, weight_unit + 1), repeat=3):
            w10_i = weight_unit - w1_i - w3_i - w5_i
            if w10_i < 0:
                continue
            weight_grid.append((w1_i / weight_unit, w3_i / weight_unit, w5_i / weight_unit, w10_i / weight_unit))

        gap_rules = [
            ("nogap", -0.20, 0.20),
            ("gap_m8_p3", -0.08, 0.03),
            ("gap_0_p3", 0.0, 0.03),
        ]
        topns = [3, 5, 8]
        rows: list[dict[str, object]] = []

        for w1, w3, w5, w10 in weight_grid:
            score_expr = f"({w1} * r1 + {w3} * r3 + {w5} * r5 + {w10} * r10)"
            for gap_name, gap_low, gap_high in gap_rules:
                for topn in topns:
                    sql = f"""
                    WITH ranked AS (
                        SELECT
                            trade_date,
                            stock_code,
                            next_open_to_next_open,
                            next_open_to_close,
                            row_number() OVER (
                                PARTITION BY trade_date
                                ORDER BY {score_expr} DESC, stock_code ASC
                            ) AS rn
                        FROM base
                        WHERE open_gap BETWEEN {gap_low} AND {gap_high}
                    ),
                    picked AS (
                        SELECT * FROM ranked WHERE rn <= {topn}
                    ),
                    daily AS (
                        SELECT
                            trade_date,
                            count(*) AS cnt,
                            avg(next_open_to_next_open) AS ret_oo,
                            avg(next_open_to_close) AS ret_oc
                        FROM picked
                        GROUP BY trade_date
                    )
                    SELECT trade_date, cnt, ret_oo, ret_oc
                    FROM daily
                    ORDER BY trade_date
                    """
                    df = con.execute(sql).fetchdf()
                    if df.empty:
                        continue
                    ret = df["ret_oo"].astype(float)
                    recent60 = ret.tail(60)
                    recent120 = ret.tail(120)
                    mean_daily = float(ret.mean())
                    std_daily = float(ret.std(ddof=1))
                    rows.append(
                        {
                            "case_name": f"w1_{w1:.1f}_w3_{w3:.1f}_w5_{w5:.1f}_w10_{w10:.1f}_{gap_name}_top{topn}",
                            "w1": w1,
                            "w3": w3,
                            "w5": w5,
                            "w10": w10,
                            "gap_name": gap_name,
                            "gap_low": gap_low,
                            "gap_high": gap_high,
                            "topn": topn,
                            "signal_days": int(df["trade_date"].nunique()),
                            "coverage_ratio": float(df["trade_date"].nunique() / total_days),
                            "avg_names": float(df["cnt"].mean()),
                            "mean_daily_oo": mean_daily,
                            "annual_oo": _annualized(mean_daily),
                            "sharpe_oo": _sharpe(mean_daily, std_daily),
                            "max_drawdown_oo": _max_drawdown(ret),
                            "recent60_annual_oo": _annualized(float(recent60.mean())) if len(recent60) else None,
                            "recent120_annual_oo": _annualized(float(recent120.mean())) if len(recent120) else None,
                            "mean_daily_oc": float(df["ret_oc"].mean()),
                            "annual_oc": _annualized(float(df["ret_oc"].mean())),
                        }
                    )

        out = pd.DataFrame(rows)
        out = out.sort_values(["annual_oo", "sharpe_oo", "coverage_ratio"], ascending=[False, False, False])
        out.to_csv(REPORT_DIR / "weight_grid_next_open_summary.csv", index=False, encoding="utf-8-sig")

        stable = out[
            (out["coverage_ratio"] >= 0.95)
            & (out["avg_names"] >= 3)
            & (out["max_drawdown_oo"] <= 0.40)
            & (out["recent60_annual_oo"] > 0)
            & (out["recent120_annual_oo"] > 0)
        ].copy()
        stable.to_csv(REPORT_DIR / "weight_grid_next_open_stable_candidates.csv", index=False, encoding="utf-8-sig")

        manifest = {
            "report_dir": str(REPORT_DIR),
            "manifests": {label: str(path) for label, path in MANIFESTS.items()},
            "sources": {
                label: {key: str(value) for key, value in source.items()}
                for label, source in sources.items()
            },
            "market_db": str(market_db),
            "total_days": int(total_days),
            "weight_step": 0.25,
            "objective": "next_open_to_next_open",
            "rows": int(len(out)),
            "stable_rows": int(len(stable)),
        }
        (REPORT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(out.head(20).to_string(index=False))
        print("stable")
        print(stable.head(20).to_string(index=False))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
