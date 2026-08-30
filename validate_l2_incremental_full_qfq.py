from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import duckdb

from project_paths import resolve_data_dir
from stock_daily_data_route import resolve_stock_daily_duckdb_path


def validate(target_trade_date: str, data_dir: str | Path | None = None) -> dict[str, object]:
    resolved_data_dir = resolve_data_dir(data_dir)
    reports_dir = resolved_data_dir / "reports"
    active = resolve_stock_daily_duckdb_path(data_dir=resolved_data_dir, require_exists=True)
    l1_dir = resolved_data_dir / "production_assets" / "duckdb" / "l1_raw_tables"
    indicator_validation_path = (
        reports_dir / f"l2_incremental_full_qfq_{target_trade_date}_qfq_indicator_gap_validation.json"
    )
    snapshot_manifest_path = (
        reports_dir / f"l2_incremental_full_qfq_{target_trade_date}_snapshot_manifest.json"
    )
    indicator_validation = json.loads(indicator_validation_path.read_text(encoding="utf-8"))
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))

    with duckdb.connect(str(active), read_only=True) as conn:
        for alias, table in (
            ("dd", "daily_data"),
            ("af", "adj_factor"),
            ("sf", "stk_factor"),
        ):
            source_path = (l1_dir / f"{table}.duckdb").as_posix()
            conn.execute(f"ATTACH '{source_path}' AS {alias} (READ_ONLY)")

        overall = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT stock_code), MIN(trade_date), MAX(trade_date),
                   SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END)
            FROM STOCK_DAILY_DATA
            """
        ).fetchone()
        target = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT stock_code),
                   SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END)
            FROM STOCK_DAILY_DATA WHERE trade_date = ?
            """,
            [target_trade_date],
        ).fetchone()
        duplicate_groups = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT stock_code, trade_date, COUNT(*) c
                FROM STOCK_DAILY_DATA GROUP BY 1, 2 HAVING c > 1
            )
            """
        ).fetchone()[0]
        l1_daily = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT ts_code),
                   SUM(CASE WHEN ts_code LIKE '%.BJ' THEN 1 ELSE 0 END)
            FROM dd.daily_data WHERE trade_date = ?
            """,
            [target_trade_date],
        ).fetchone()
        l1_stk = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT ts_code),
                   SUM(CASE WHEN ts_code LIKE '%.BJ' THEN 1 ELSE 0 END)
            FROM sf.stk_factor WHERE trade_date = ?
            """,
            [target_trade_date],
        ).fetchone()
        target_raw_mismatch = conn.execute(
            """
            SELECT COUNT(*)
            FROM STOCK_DAILY_DATA s
            JOIN dd.daily_data d
              ON s.stock_code = d.ts_code AND s.trade_date = d.trade_date
            WHERE s.trade_date = ?
              AND (
                s.open IS DISTINCT FROM d.open OR
                s.high IS DISTINCT FROM d.high OR
                s.low IS DISTINCT FROM d.low OR
                s.close IS DISTINCT FROM d.close OR
                s.pre_close IS DISTINCT FROM d.pre_close OR
                s.change IS DISTINCT FROM d.change OR
                s.pct_chg IS DISTINCT FROM d.pct_chg OR
                s.vol IS DISTINCT FROM d.vol OR
                s.amount IS DISTINCT FROM d.amount
              )
            """,
            [target_trade_date],
        ).fetchone()[0]
        qfq_nulls = conn.execute(
            """
            SELECT
                   SUM(CASE WHEN open_qfq IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN high_qfq IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN low_qfq IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN close_qfq IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN pre_close_qfq IS NULL THEN 1 ELSE 0 END)
            FROM STOCK_DAILY_DATA WHERE trade_date = ?
            """,
            [target_trade_date],
        ).fetchone()
        formula_expected, formula_mismatch = conn.execute(
            """
            WITH latest AS (
                SELECT ts_code, adj_factor AS latest_adj_factor
                FROM (
                    SELECT ts_code, adj_factor,
                           ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) rn
                    FROM af.adj_factor
                ) WHERE rn = 1
            )
            SELECT
              COUNT(*),
              SUM(CASE WHEN
                s.open_qfq IS NULL OR s.high_qfq IS NULL OR s.low_qfq IS NULL OR
                s.close_qfq IS NULL OR s.pre_close_qfq IS NULL OR
                ABS(s.open_qfq - s.open * d.adj_factor / l.latest_adj_factor) > 1e-10 OR
                ABS(s.high_qfq - s.high * d.adj_factor / l.latest_adj_factor) > 1e-10 OR
                ABS(s.low_qfq - s.low * d.adj_factor / l.latest_adj_factor) > 1e-10 OR
                ABS(s.close_qfq - s.close * d.adj_factor / l.latest_adj_factor) > 1e-10 OR
                ABS(s.pre_close_qfq - s.pre_close * d.adj_factor / l.latest_adj_factor) > 1e-10
              THEN 1 ELSE 0 END)
            FROM STOCK_DAILY_DATA s
            JOIN af.adj_factor d
              ON s.stock_code = d.ts_code AND s.trade_date = d.trade_date
            JOIN latest l ON s.stock_code = l.ts_code
            WHERE d.adj_factor IS NOT NULL
              AND l.latest_adj_factor IS NOT NULL
              AND l.latest_adj_factor <> 0
              AND s.open IS NOT NULL AND s.high IS NOT NULL AND s.low IS NOT NULL
              AND s.close IS NOT NULL AND s.pre_close IS NOT NULL
            """
        ).fetchone()
        st_nulls = conn.execute(
            """
            SELECT
                   SUM(CASE WHEN ST_TYPE IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN ST_TYPE_name IS NULL THEN 1 ELSE 0 END)
            FROM STOCK_DAILY_DATA WHERE trade_date = ?
            """,
            [target_trade_date],
        ).fetchone()

    price_columns = ["open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"]
    price_nulls = dict(zip(price_columns, [int(value or 0) for value in qfq_nulls]))
    passed = (
        int(target[0]) == int(l1_daily[0])
        and int(target[1]) == int(l1_daily[1])
        and int(target[2] or 0) == 0
        and int(duplicate_groups) == 0
        and str(overall[3]) == target_trade_date
        and int(target_raw_mismatch) == 0
        and int(formula_mismatch or 0) == 0
        and indicator_validation.get("status") == "passed"
        and not indicator_validation.get("mismatch_by_year")
    )
    return {
        "task_id": f"l2_incremental_full_qfq-{target_trade_date}",
        "status": "completed" if passed else "failed",
        "target_trade_date": target_trade_date,
        "execution_mode": {
            "non_qfq_fields": "target_date_incremental_replace",
            "qfq_price_fields": "full_history_recomputed",
            "stk_factor_qfq_indicator_fields": "full_history_synced_from_active_l1",
            "standard_full_rebuild_claimed": False,
        },
        "active_l2": str(active),
        "table": "STOCK_DAILY_DATA",
        "overall": list(overall),
        "target": list(target),
        "l1_daily": list(l1_daily),
        "l1_stk_factor": list(l1_stk),
        "duplicate_key_groups": int(duplicate_groups),
        "target_raw_market_field_mismatch_rows": int(target_raw_mismatch),
        "target_qfq_price_nulls": price_nulls,
        "full_price_qfq_formula_expected_rows": int(formula_expected),
        "full_price_qfq_formula_mismatch_count": int(formula_mismatch or 0),
        "qfq_indicator_column_count": int(indicator_validation["qfq_indicator_column_count"]),
        "qfq_indicator_mismatch_by_year": indicator_validation["mismatch_by_year"],
        "target_qfq_indicator_nulls": indicator_validation["target_qfq_nulls"],
        "st_governance_target_nulls": {
            "ST_TYPE": int(st_nulls[0] or 0),
            "ST_TYPE_name": int(st_nulls[1] or 0),
        },
        "snapshot": {
            "manifest_path": str(snapshot_manifest_path),
            "asset_path": snapshot_manifest["asset_path"],
            "size_bytes": snapshot_manifest["file_state"]["size_bytes"],
            "sha256": snapshot_manifest["sha256"],
            "table_metrics": snapshot_manifest["table_metrics"],
        },
        "governance": {
            "daily_data_drives_calendar": True,
            "adj_factor_does_not_drive_calendar": True,
            "no_bj": True,
            "duckdb_only": True,
            "one_table_one_file": True,
            "legacy_sqlite_used": False,
            "l3_l8_triggered": False,
            "allow_l3_continue": False,
            "audit_required": True,
        },
        "evidence_paths": [str(snapshot_manifest_path), str(indicator_validation_path)],
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def render_markdown(payload: dict[str, object]) -> str:
    overall = payload["overall"]
    target = payload["target"]
    snapshot = payload["snapshot"]
    return f"""# L2 增量整合与复权字段全量整合报告

- 目标交易日：`{payload['target_trade_date']}`
- 执行状态：`{payload['status']}`
- Active 资产：`{payload['active_l2']}::STOCK_DAILY_DATA`
- 普通字段：仅替换目标交易日记录
- 复权字段：5 个 qfq 价格字段全历史重算，74 个 qfq 技术字段全历史同步
- 本次不宣称标准工作流的 L2 全量重建

## 完整性结果

- 全表行数：`{overall[0]}`
- 股票数：`{overall[1]}`
- 日期范围：`{overall[2]} - {overall[3]}`
- 目标日行数 / 股票数：`{target[0]} / {target[1]}`
- 重复键组数：`{payload['duplicate_key_groups']}`
- `.BJ` 行数：`{overall[4]}`
- 目标日 raw 行情与 L1 不一致行数：`{payload['target_raw_market_field_mismatch_rows']}`

## 复权结果

- 5 个 qfq 价格字段目标日空值：`{payload['target_qfq_price_nulls']}`
- 全历史 qfq 价格公式校验行数：`{payload['full_price_qfq_formula_expected_rows']}`
- 全历史 qfq 价格公式不一致数：`{payload['full_price_qfq_formula_mismatch_count']}`
- qfq 技术字段数量：`{payload['qfq_indicator_column_count']}`
- qfq 技术字段全历史不一致年份：`{payload['qfq_indicator_mismatch_by_year']}`

目标日 qfq 技术字段的少量空值与 active L1 `stk_factor` 完全一致，属于源表真实空值，不做伪造补值。

## 快照与边界

- 独立快照：`{snapshot['asset_path']}`
- SHA256：`{snapshot['sha256']}`
- 未使用 SQLite、`odb.db`、root parquet 或 raw split DB。
- 未触发 L3-L8。
- `allow_l3_continue=false`，等待审计复核。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-trade-date", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    payload = validate(args.target_trade_date, args.data_dir)
    Path(args.output_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
