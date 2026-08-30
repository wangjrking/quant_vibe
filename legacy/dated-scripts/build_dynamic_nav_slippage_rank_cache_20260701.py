from __future__ import annotations

import json
from pathlib import Path

import duckdb

from l1_raw_data_route import resolve_l1_raw_duckdb_path
from prediction_manifest import load_prediction_source_manifest
from stock_daily_data_route import resolve_stock_daily_duckdb_path


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_dynamic_nav_slippage_20260701"
OUTPUT_DB = REPORT_DIR / "rank_cache.duckdb"
MAIN = ROOT / "quant" / "main"
MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _duckdb_literal(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def _load_formal_source(label: str) -> dict[str, object]:
    manifest_path = MANIFESTS[label]
    source = load_prediction_source_manifest(manifest_path, require_approved=True, allow_legacy=False)
    if source["source_type"] != "duckdb_table":
        raise RuntimeError(f"{manifest_path} must be duckdb_table, got {source['source_type']}")
    active_market = resolve_stock_daily_duckdb_path(require_exists=True).resolve()
    manifest_market = source.get("market_db_path")
    if manifest_market and Path(str(manifest_market)).resolve() != active_market:
        raise RuntimeError(
            f"{manifest_path} market_db_path does not match current active L2 DuckDB route: "
            f"{manifest_market} != {active_market}"
        )
    return {
        "manifest_path": manifest_path,
        "db_path": Path(str(source["db_path"])).resolve(),
        "table": str(source["table"]),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    formal_sources = {label: _load_formal_source(label) for label in ("1d", "3d", "5d", "10d")}
    market_db = resolve_stock_daily_duckdb_path(require_exists=True).resolve()
    raw_daily_db = resolve_l1_raw_duckdb_path("daily_data", require_exists=True).resolve()

    con = duckdb.connect(str(OUTPUT_DB))
    try:
        alias_by_path: dict[Path, str] = {}

        def ensure_alias(path: Path, prefix: str) -> str:
            resolved = path.resolve()
            alias = alias_by_path.get(resolved)
            if alias:
                return alias
            alias = f"{prefix}_{len(alias_by_path)}"
            con.execute(f"ATTACH {_duckdb_literal(resolved)} AS {_quote_ident(alias)} (READ_ONLY)")
            alias_by_path[resolved] = alias
            return alias

        source_refs: dict[str, str] = {}
        for label, source in formal_sources.items():
            alias = ensure_alias(Path(str(source["db_path"])), f"pred_{label}")
            source_refs[label] = f'{_quote_ident(alias)}.{_quote_ident(str(source["table"]))}'

        market_alias = ensure_alias(market_db, "market")
        raw_alias = ensure_alias(raw_daily_db, "raw")
        market_ref = f'{_quote_ident(market_alias)}."STOCK_DAILY_DATA"'
        raw_daily_ref = f'{_quote_ident(raw_alias)}."daily_data"'

        con.execute(
            f"""
            CREATE TABLE rank_cache AS
            WITH market_with_prev AS (
                SELECT
                    *,
                    lag(pct_chg, 1) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_pct_chg
                FROM {market_ref}
            ),
            market_dates AS (
                SELECT DISTINCT trade_date FROM {market_ref}
            ),
            date_map AS (
                SELECT
                    trade_date AS signal_date,
                    lead(trade_date, 1) OVER (ORDER BY trade_date) AS buy_date,
                    lead(trade_date, 2) OVER (ORDER BY trade_date) AS sell1,
                    lead(trade_date, 3) OVER (ORDER BY trade_date) AS sell2,
                    lead(trade_date, 4) OVER (ORDER BY trade_date) AS sell3,
                    lead(trade_date, 6) OVER (ORDER BY trade_date) AS sell5
                FROM market_dates
            ),
            scores AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p1.pred_prob AS pred_1d,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM {source_refs["10d"]} p10
                INNER JOIN {source_refs["1d"]} p1
                  ON p10.trade_date = p1.trade_date
                 AND p10.stock_code = p1.stock_code
                INNER JOIN {source_refs["3d"]} p3
                  ON p10.trade_date = p3.trade_date
                 AND p10.stock_code = p3.stock_code
                INNER JOIN {source_refs["5d"]} p5
                  ON p10.trade_date = p5.trade_date
                 AND p10.stock_code = p5.stock_code
            ),
            ranked AS (
                SELECT
                    *,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_1d) AS r1,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS r3,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS r5,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS r10
                FROM scores
            ),
            joined AS (
                SELECT
                    r.trade_date,
                    r.stock_code,
                    r.pred_1d,
                    r.pred_3d,
                    r.pred_5d,
                    r.pred_10d,
                    r.r1,
                    r.r3,
                    r.r5,
                    r.r10,
                    dm.buy_date,
                    dm.sell1,
                    dm.sell2,
                    dm.sell3,
                    dm.sell5,
                    md.name,
                    try_cast(md.amount AS DOUBLE) AS amount,
                    try_cast(md.total_mv AS DOUBLE) AS total_mv,
                    try_cast(md.turnover_rate AS DOUBLE) AS turnover_rate,
                    try_cast(md.close AS DOUBLE) AS close,
                    try_cast(md.pct_chg AS DOUBLE) AS pct_chg,
                    try_cast(md.prev_pct_chg AS DOUBLE) AS prev_pct_chg,
                    ((1 + try_cast(md.pct_chg AS DOUBLE) / 100.0)
                     * (1 + try_cast(md.prev_pct_chg AS DOUBLE) / 100.0) - 1.0) AS two_day_ret,
                    try_cast(braw.open AS DOUBLE) AS buy_open,
                    try_cast(braw.pre_close AS DOUBLE) AS buy_pre_close,
                    try_cast(braw.open AS DOUBLE) / nullif(try_cast(braw.pre_close AS DOUBLE), 0) - 1.0 AS buy_open_gap,
                    try_cast(braw.open AS DOUBLE) / nullif(try_cast(md.close AS DOUBLE), 0) - 1.0 AS buy_open_vs_signal_close,
                    try_cast(s1raw.open AS DOUBLE) AS sell1_open,
                    try_cast(s2raw.open AS DOUBLE) AS sell2_open,
                    try_cast(s3raw.open AS DOUBLE) AS sell3_open,
                    try_cast(s5raw.open AS DOUBLE) AS sell5_open,
                    CASE
                        WHEN dm.buy_date IS NULL THEN 0
                        WHEN bmd.stock_code IS NULL THEN 0
                        WHEN braw.ts_code IS NULL THEN 0
                        WHEN upper(coalesce(md.name, '')) LIKE 'ST%' THEN 0
                        WHEN upper(coalesce(md.name, '')) LIKE '*ST%' THEN 0
                        WHEN upper(coalesce(bmd.name, '')) LIKE 'ST%' THEN 0
                        WHEN upper(coalesce(bmd.name, '')) LIKE '*ST%' THEN 0
                        WHEN coalesce(md.ST_TYPE_name, '') LIKE '%风险%' THEN 0
                        WHEN coalesce(bmd.ST_TYPE_name, '') LIKE '%风险%' THEN 0
                        WHEN try_cast(md.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(md.ST_TYPE AS DOUBLE) <> 0 THEN 0
                        WHEN try_cast(bmd.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(bmd.ST_TYPE AS DOUBLE) <> 0 THEN 0
                        WHEN instr(coalesce(md.name, ''), '退市') > 0 THEN 0
                        WHEN instr(coalesce(bmd.name, ''), '退市') > 0 THEN 0
                        WHEN coalesce(try_cast(md.limit_times AS DOUBLE), 0.0) > 0.0 THEN 0
                        WHEN coalesce(try_cast(bmd.limit_times AS DOUBLE), 0.0) > 0.0 THEN 0
                        WHEN try_cast(braw.pre_close AS DOUBLE) IS NULL OR try_cast(braw.open AS DOUBLE) IS NULL THEN 0
                        WHEN try_cast(braw.pre_close AS DOUBLE) <= 0 OR try_cast(braw.open AS DOUBLE) <= 0 THEN 0
                        WHEN try_cast(braw.open AS DOUBLE) >= try_cast(braw.pre_close AS DOUBLE) * (
                            1.0 + CASE WHEN substr(r.stock_code, 1, 3) IN ('300', '301', '688') THEN 0.20 ELSE 0.10 END
                        ) * 0.995 THEN 0
                        ELSE 1
                    END AS buy_ok
                FROM ranked r
                LEFT JOIN market_with_prev md
                  ON r.trade_date = md.trade_date
                 AND r.stock_code = md.stock_code
                LEFT JOIN date_map dm
                  ON r.trade_date = dm.signal_date
                LEFT JOIN {market_ref} bmd
                  ON dm.buy_date = bmd.trade_date
                 AND r.stock_code = bmd.stock_code
                LEFT JOIN {raw_daily_ref} braw
                  ON dm.buy_date = braw.trade_date
                 AND r.stock_code = braw.ts_code
                LEFT JOIN {raw_daily_ref} s1raw
                  ON dm.sell1 = s1raw.trade_date
                 AND r.stock_code = s1raw.ts_code
                LEFT JOIN {raw_daily_ref} s2raw
                  ON dm.sell2 = s2raw.trade_date
                 AND r.stock_code = s2raw.ts_code
                LEFT JOIN {raw_daily_ref} s3raw
                  ON dm.sell3 = s3raw.trade_date
                 AND r.stock_code = s3raw.ts_code
                LEFT JOIN {raw_daily_ref} s5raw
                  ON dm.sell5 = s5raw.trade_date
                 AND r.stock_code = s5raw.ts_code
                WHERE NOT (r.stock_code LIKE '%.BJ' OR substr(r.stock_code, 1, 1) IN ('4', '8'))
            )
            SELECT * FROM joined
            """
        )
        con.execute("CREATE INDEX idx_rank_cache_date_code ON rank_cache(trade_date, stock_code)")
        con.execute("CREATE INDEX idx_rank_cache_date ON rank_cache(trade_date)")
        summary = con.execute(
            """
            SELECT
                min(trade_date) AS min_trade_date,
                max(trade_date) AS max_trade_date,
                count(*) AS row_count,
                count(distinct trade_date) AS trade_days,
                sum(CASE WHEN buy_ok = 1 THEN 1 ELSE 0 END) AS buy_ok_rows,
                sum(CASE WHEN buy_open IS NULL THEN 1 ELSE 0 END) AS missing_buy_open_rows
            FROM rank_cache
            """
        ).fetchone()
    finally:
        con.close()
    payload = {
        "rank_cache": str(OUTPUT_DB),
        "active_market_db": str(market_db),
        "raw_daily_db": str(raw_daily_db),
        "summary": {
            "min_trade_date": summary[0],
            "max_trade_date": summary[1],
            "row_count": summary[2],
            "trade_days": summary[3],
            "buy_ok_rows": summary[4],
            "missing_buy_open_rows": summary[5],
        },
        "formal_sources": {
            label: {
                "manifest_path": str(source["manifest_path"]),
                "db_path": str(source["db_path"]),
                "table": source["table"],
            }
            for label, source in formal_sources.items()
        },
    }
    (REPORT_DIR / "rank_cache_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
