from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


QFQ_PRICES = {"open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"}


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def validate(target_path: Path, l1_dir: Path, trade_date: str) -> dict[str, object]:
    with duckdb.connect(str(target_path), read_only=True) as conn:
        for alias, name in (("dd", "daily_data"), ("af", "adj_factor"), ("sf", "stk_factor")):
            source = (l1_dir / f"{name}.duckdb").as_posix().replace("'", "''")
            conn.execute(f"ATTACH '{source}' AS {alias} (READ_ONLY)")

        full = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT stock_code),
                   COUNT(*) FILTER (WHERE stock_code LIKE '%.BJ'),
                   MIN(trade_date), MAX(trade_date)
            FROM STOCK_DAILY_DATA
            """
        ).fetchone()
        target = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT stock_code),
                   COUNT(*) FILTER (WHERE stock_code LIKE '%.BJ')
            FROM STOCK_DAILY_DATA WHERE trade_date = ?
            """,
            [trade_date],
        ).fetchone()
        duplicate_groups = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT stock_code, trade_date
                FROM STOCK_DAILY_DATA GROUP BY 1, 2 HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        raw_price_mismatch = conn.execute(
            """
            SELECT COUNT(*)
            FROM STOCK_DAILY_DATA AS s
            JOIN dd.daily_data AS d
              ON s.stock_code = d.ts_code AND s.trade_date = d.trade_date
            WHERE s.trade_date = ?
              AND (s.open IS DISTINCT FROM d.open
                OR s.high IS DISTINCT FROM d.high
                OR s.low IS DISTINCT FROM d.low
                OR s.close IS DISTINCT FROM d.close
                OR s.pre_close IS DISTINCT FROM d.pre_close)
            """,
            [trade_date],
        ).fetchone()[0]
        qfq_formula_mismatch = conn.execute(
            """
            WITH latest AS (
                SELECT ts_code, ARG_MAX(adj_factor, trade_date) AS latest_adj
                FROM af.adj_factor GROUP BY 1
            )
            SELECT COUNT(*)
            FROM STOCK_DAILY_DATA AS s
            JOIN af.adj_factor AS a
              ON a.ts_code = s.stock_code AND a.trade_date = s.trade_date
            JOIN latest AS l ON l.ts_code = s.stock_code
            WHERE s.trade_date = ?
              AND (ABS(s.open_qfq - s.open * a.adj_factor / l.latest_adj) > 1e-8
                OR ABS(s.high_qfq - s.high * a.adj_factor / l.latest_adj) > 1e-8
                OR ABS(s.low_qfq - s.low * a.adj_factor / l.latest_adj) > 1e-8
                OR ABS(s.close_qfq - s.close * a.adj_factor / l.latest_adj) > 1e-8
                OR ABS(s.pre_close_qfq - s.pre_close * a.adj_factor / l.latest_adj) > 1e-8)
            """,
            [trade_date],
        ).fetchone()[0]
        qfq_indicators = [
            row[0]
            for row in conn.execute("DESCRIBE STOCK_DAILY_DATA").fetchall()
            if "qfq" in row[0].lower() and row[0] not in QFQ_PRICES
        ]
        mismatch_predicate = " OR ".join(
            f"s.{quote_ident(column)} IS DISTINCT FROM f.{quote_ident(column)}"
            for column in qfq_indicators
        )
        indicator_mismatch = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM STOCK_DAILY_DATA AS s
            JOIN sf.stk_factor AS f
              ON s.stock_code = f.ts_code AND s.trade_date = f.trade_date
            WHERE s.trade_date = ? AND ({mismatch_predicate})
            """,
            [trade_date],
        ).fetchone()[0]
        source_gap = conn.execute(
            """
            SELECT COUNT(*)
            FROM STOCK_DAILY_DATA AS s
            LEFT JOIN sf.stk_factor AS f
              ON s.stock_code = f.ts_code AND s.trade_date = f.trade_date
            WHERE s.trade_date = ? AND f.ts_code IS NULL
            """,
            [trade_date],
        ).fetchone()[0]

    passed = all(
        value == 0
        for value in (
            int(full[2]),
            int(target[2]),
            int(duplicate_groups),
            int(raw_price_mismatch),
            int(qfq_formula_mismatch),
            int(indicator_mismatch),
        )
    )
    return {
        "status": "passed" if passed else "failed",
        "target_trade_date": trade_date,
        "target_path": str(target_path.resolve()),
        "full": {
            "row_count": int(full[0]),
            "stock_count": int(full[1]),
            "bj_rows": int(full[2]),
            "min_trade_date": str(full[3]),
            "max_trade_date": str(full[4]),
        },
        "target": {
            "row_count": int(target[0]),
            "stock_count": int(target[1]),
            "bj_rows": int(target[2]),
        },
        "duplicate_key_groups": int(duplicate_groups),
        "raw_price_mismatch": int(raw_price_mismatch),
        "qfq_formula_mismatch": int(qfq_formula_mismatch),
        "qfq_indicator_field_count": len(qfq_indicators),
        "qfq_indicator_mismatch_rows": int(indicator_mismatch),
        "stk_factor_source_gap_rows": int(source_gap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-path", required=True, type=Path)
    parser.add_argument("--l1-dir", required=True, type=Path)
    parser.add_argument("--target-trade-date", required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = validate(args.target_path, args.l1_dir, args.target_trade_date)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
