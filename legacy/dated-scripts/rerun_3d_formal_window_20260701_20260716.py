from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

import incremental_formal_l4_duckdb_mainline as l4


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_formal_window_rerun_20260701_20260716"
REPORT_JSON = REPORT_DIR / "l4_3d_formal_window_rerun_report.json"
REPORT_MD = REPORT_DIR / "l4_3d_formal_window_rerun_report.md"
VALIDATION_JSON = REPORT_DIR / "l4_3d_formal_window_distribution_validation.json"
HANDOFF_JSON = REPORT_DIR / "l4_3d_formal_window_audit_handoff.json"
ROLLBACK_DB = REPORT_DIR / "l4_3d_formal_window_before_rerun.duckdb"
ROLLBACK_TABLE = "l4_3d_formal_window_before_rerun"

TARGET_DATES = [
    "20260701",
    "20260702",
    "20260703",
    "20260706",
    "20260707",
    "20260708",
    "20260709",
    "20260710",
    "20260713",
    "20260714",
    "20260715",
    "20260716",
]
SOURCE_LIMITED_DATES = {"20260706", "20260707"}
SOURCE_LIMITED_FIELDS = [
    "index_2000_open",
    "index_2000_high",
    "index_2000_low",
    "index_2000_amount",
]
REPAIRED_FIELDS = ["turnover_rate", "turnover_rate_f", "total_mv"]
REQUIRED_3D_FIELDS = [*SOURCE_LIMITED_FIELDS, *REPAIRED_FIELDS]

EXPECTED_FACTOR_PATH = DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
EXPECTED_FACTOR_TABLE = "prod_l3_production_factor_parts_20260625"
EXPECTED_LABEL_PATH = DATA_DIR / "production_assets" / "duckdb" / "l3_label_current.duckdb"
EXPECTED_LABEL_TABLE = "prod_l3_prediction_label_parts_current"
TARGET_DB = DATA_DIR / "production_assets" / "duckdb" / "l4_executable_3d_open_return_formal.duckdb"
TARGET_TABLE = l4.FORMAL_TABLES["3d"]
MANIFEST_PATH = (
    ROOT
    / "quant"
    / "main"
    / "config"
    / "prediction_manifests"
    / "executable_3d_open_return_l4_formal_20260617.json"
)
UPSTREAM_HANDOFF = (
    DATA_DIR
    / "reports"
    / "l3_required_feature_window_repair_active_switch_20260719"
    / "l3_to_l4_handoff_contract_required_feature_window_repair_active_20260719.json"
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def placeholders(count: int) -> str:
    return ", ".join(["?"] * count)


def assert_active_inputs() -> tuple[Path, str, Path, str]:
    factor_path, factor_table = l4.factor_input_asset()
    label_path, label_table = l4.label_input_asset()
    if factor_path.resolve() != EXPECTED_FACTOR_PATH.resolve() or factor_table != EXPECTED_FACTOR_TABLE:
        raise RuntimeError(
            "active L3 feature route is not the audited repaired asset: "
            f"{factor_path}::{factor_table}"
        )
    if label_path.resolve() != EXPECTED_LABEL_PATH.resolve() or label_table != EXPECTED_LABEL_TABLE:
        raise RuntimeError(
            "active L3 label route is not the audited active label asset: "
            f"{label_path}::{label_table}"
        )
    handoff = json.loads(UPSTREAM_HANDOFF.read_text(encoding="utf-8"))
    checks = handoff.get("contract", {}).get("gate_checks", [])
    failed = [item.get("name") for item in checks if not item.get("passed")]
    if failed:
        raise RuntimeError(f"upstream L3 handoff has failed gate checks: {failed}")
    return factor_path, factor_table, label_path, label_table


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [str(row[1]) for row in con.execute(f"PRAGMA table_info({quote(table)})").fetchall()]


def row_fingerprint(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    where_sql: str,
    params: list[str],
) -> dict[str, Any]:
    columns = table_columns(con, table)
    hash_expr = f"hash({', '.join(quote(column) for column in columns)})"
    row = con.execute(
        f"SELECT count(*), coalesce(sum({hash_expr}), 0), coalesce(bit_xor({hash_expr}), 0) "
        f"FROM {quote(table)} WHERE {where_sql}",
        params,
    ).fetchone()
    return {
        "row_count": int(row[0]),
        "hash_sum": str(row[1]),
        "hash_xor": str(row[2]),
    }


def date_stats(con: duckdb.DuckDBPyConnection, table: str, trade_date: str) -> dict[str, Any]:
    row = con.execute(
        f"""
        SELECT count(*), count(DISTINCT stock_code), count(DISTINCT pred_prob),
               coalesce(stddev_pop(pred_prob), 0), min(pred_prob), max(pred_prob),
               count(*) FILTER (WHERE pred_prob IS NULL),
               count(*) FILTER (WHERE upper(stock_code) LIKE '%.BJ')
        FROM {quote(table)} WHERE trade_date = ?
        """,
        [trade_date],
    ).fetchone()
    dup_groups = con.execute(
        f"""
        SELECT count(*) FROM (
          SELECT trade_date, stock_code
          FROM {quote(table)}
          WHERE trade_date = ?
          GROUP BY 1, 2 HAVING count(*) > 1
        )
        """,
        [trade_date],
    ).fetchone()[0]
    return {
        "trade_date": trade_date,
        "row_count": int(row[0]),
        "stock_count": int(row[1]),
        "distinct_pred_prob": int(row[2]),
        "std_pred_prob": float(row[3]),
        "min_pred_prob": None if row[4] is None else float(row[4]),
        "max_pred_prob": None if row[5] is None else float(row[5]),
        "null_pred_prob": int(row[6]),
        "bj_rows": int(row[7]),
        "duplicate_key_groups": int(dup_groups),
    }


def feature_null_semantics(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    trade_date: str,
) -> dict[str, Any]:
    expressions = ["count(*)"] + [
        f"count(*) FILTER (WHERE {quote(column)} IS NULL)" for column in REQUIRED_3D_FIELDS
    ]
    row = con.execute(
        f"SELECT {', '.join(expressions)} FROM {quote(table)} WHERE trade_date = ?",
        [trade_date],
    ).fetchone()
    result = {"row_count": int(row[0])}
    result.update({f"{column}_nulls": int(row[index + 1]) for index, column in enumerate(REQUIRED_3D_FIELDS)})
    expected_source_limited = trade_date in SOURCE_LIMITED_DATES
    for column in SOURCE_LIMITED_FIELDS:
        expected = result["row_count"] if expected_source_limited else 0
        if result[f"{column}_nulls"] != expected:
            raise RuntimeError(
                f"unexpected source-limited semantics for {trade_date} {column}: "
                f"expected {expected}, got {result[f'{column}_nulls']}"
            )
    for column in REPAIRED_FIELDS:
        if result[f"{column}_nulls"] != 0:
            raise RuntimeError(f"repaired L3 field remains null for {trade_date}: {column}")
    result["source_limited_semantics"] = (
        "active_l2_source_limited_null_preserved"
        if expected_source_limited
        else "required_3d_fields_complete"
    )
    return result


def prepare_frames(
    factor_con: duckdb.DuckDBPyConnection,
    label_con: duckdb.DuckDBPyConnection,
    *,
    factor_table: str,
    label_table: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = l4.MODEL_SPECS["3d"]
    frames: list[pd.DataFrame] = []
    validation: dict[str, Any] = {}
    for trade_date in TARGET_DATES:
        semantics = feature_null_semantics(factor_con, table=factor_table, trade_date=trade_date)
        scored, model_meta = l4.predict_savedmodel_frame(
            factor_con,
            factor_table=factor_table,
            trade_date=trade_date,
            metadata_path=Path(spec["metadata_path"]),
        )
        stock_codes = scored["stock_code"].tolist()
        label_frame = l4.load_label_frame(
            label_con,
            label_table=label_table,
            label_column=str(spec["label"]),
            trade_date=trade_date,
            stock_codes=stock_codes,
        )
        formal_frame = l4.build_formal_frame(
            trade_date=trade_date,
            label_column=str(spec["label"]),
            scored=scored,
            label_frame=label_frame,
        )
        stats = {
            "row_count": int(len(formal_frame)),
            "stock_count": int(formal_frame["stock_code"].nunique()),
            "distinct_pred_prob": int(formal_frame["pred_prob"].nunique(dropna=True)),
            "std_pred_prob": float(formal_frame["pred_prob"].std(ddof=0)),
            "null_pred_prob": int(formal_frame["pred_prob"].isna().sum()),
            "bj_rows": int(formal_frame["stock_code"].str.upper().str.endswith(".BJ").sum()),
            "duplicate_key_groups": int(formal_frame.duplicated(["trade_date", "stock_code"], keep=False).sum()),
        }
        if stats["row_count"] != semantics["row_count"] or stats["stock_count"] != stats["row_count"]:
            raise RuntimeError(f"3D scoring coverage mismatch for {trade_date}: {stats}")
        if stats["null_pred_prob"] or stats["bj_rows"] or stats["duplicate_key_groups"]:
            raise RuntimeError(f"3D scoring quality gate failed for {trade_date}: {stats}")
        if stats["distinct_pred_prob"] < 20 or stats["std_pred_prob"] <= 1e-5:
            raise RuntimeError(f"3D scoring remains constant or near-constant for {trade_date}: {stats}")
        validation[trade_date] = {
            **stats,
            "feature_null_semantics": semantics,
            "model_metadata_path": model_meta["metadata_path"],
            "model_path": model_meta["model_path"],
        }
        frames.append(formal_frame)
    return pd.concat(frames, ignore_index=True), validation


def create_rollback_snapshot() -> None:
    if ROLLBACK_DB.exists():
        ROLLBACK_DB.unlink()
    source_path = str(TARGET_DB).replace("'", "''")
    with duckdb.connect(str(ROLLBACK_DB), read_only=False) as rollback_con:
        rollback_con.execute(f"ATTACH '{source_path}' AS source_db (READ_ONLY)")
        rollback_con.execute(
            f"CREATE TABLE {quote(ROLLBACK_TABLE)} AS "
            f"SELECT * FROM source_db.{quote(TARGET_TABLE)} "
            f"WHERE trade_date IN ({placeholders(len(TARGET_DATES))}) ORDER BY trade_date, stock_code",
            TARGET_DATES,
        )
        rollback_con.execute("DETACH source_db")


def write_window(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> None:
    con.register("_rerun_frame", frame)
    columns = table_columns(con, TARGET_TABLE)
    frame_columns = [str(column) for column in frame.columns]
    if columns != frame_columns:
        raise RuntimeError(f"target/frame schema mismatch: target={columns}, frame={frame_columns}")
    column_sql = ", ".join(quote(column) for column in columns)
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(
            f"DELETE FROM {quote(TARGET_TABLE)} WHERE trade_date IN ({placeholders(len(TARGET_DATES))})",
            TARGET_DATES,
        )
        con.execute(
            f"INSERT INTO {quote(TARGET_TABLE)} ({column_sql}) SELECT {column_sql} FROM _rerun_frame"
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("_rerun_frame")


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 3D formal 受影响窗口重刷报告",
        "",
        "## 结论",
        "",
        f"- 执行状态：`{report['status']}`",
        "- 仅重刷 `20260701-20260716` 内的 12 个交易日，未改写 `20260717` 或其他日期。",
        "- 未训练、未调参、未修改其他期限、未生成信号、未运行回测。",
        "",
        "## 输入与输出",
        "",
        f"- L3 特征：`{report['inputs']['factor_asset']}`",
        f"- L3 标签：`{report['inputs']['label_asset']}`",
        f"- L4 3D formal：`{report['output']['asset']}`",
        f"- manifest：`{report['output']['manifest']}`（本次只读，未修改）",
        "",
        "## 逐日验证",
        "",
        "| 交易日 | 行数 | 股票数 | 不同分数 | 分数标准差 | 空分数 | 重复键 | 北交所 | 输入语义 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trade_date in TARGET_DATES:
        item = report["validation"][trade_date]
        semantics = item["feature_null_semantics"]["source_limited_semantics"]
        lines.append(
            f"| {trade_date} | {item['row_count']} | {item['stock_count']} | "
            f"{item['distinct_pred_prob']} | {item['std_pred_prob']:.8f} | "
            f"{item['null_pred_prob']} | {item['duplicate_key_groups']} | {item['bj_rows']} | {semantics} |"
        )
    lines.extend(
        [
            "",
            "## 源头受限说明",
            "",
            "- `20260706`、`20260707` 的 `index_2000_open/high/low/amount` 按 active L2 源头受限语义保持全空。",
            "- 未对上述字段伪造补值；模型按已归档 XGBoost 对缺失值的既有处理生成分数。",
            "- 两日的 `turnover_rate`、`turnover_rate_f`、`total_mv` 已完整恢复，输出分布通过非退化门禁。",
            "",
            "## 下游边界",
            "",
            "- `ready_for_audit_review=true`",
            "- `allow_next_layer_continue=false`",
            "- 等待审计智能体只读复核，不推进 L5-L8。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(*, apply: bool) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    factor_path, factor_table, label_path, label_table = assert_active_inputs()
    spec = l4.MODEL_SPECS["3d"]

    with duckdb.connect(str(factor_path), read_only=True) as factor_con, duckdb.connect(
        str(label_path), read_only=True
    ) as label_con:
        label_max = str(label_con.execute(f"SELECT max(trade_date) FROM {quote(label_table)}").fetchone()[0])
        if label_max != "20260616":
            raise RuntimeError(f"unexpected label maturity boundary: {label_max}")
        frame, prewrite_validation = prepare_frames(
            factor_con,
            label_con,
            factor_table=factor_table,
            label_table=label_table,
        )

    target_where = f"trade_date IN ({placeholders(len(TARGET_DATES))})"
    non_target_where = f"trade_date NOT IN ({placeholders(len(TARGET_DATES))})"
    with duckdb.connect(str(TARGET_DB), read_only=True) as con:
        pre_target = row_fingerprint(
            con, table=TARGET_TABLE, where_sql=target_where, params=TARGET_DATES
        )
        pre_non_target = row_fingerprint(
            con, table=TARGET_TABLE, where_sql=non_target_where, params=TARGET_DATES
        )
        pre_20260717 = row_fingerprint(
            con, table=TARGET_TABLE, where_sql="trade_date = ?", params=["20260717"]
        )

    if apply:
        create_rollback_snapshot()
        with duckdb.connect(str(TARGET_DB), read_only=False) as con:
            write_window(con, frame)

    with duckdb.connect(str(TARGET_DB), read_only=True) as con:
        post_target = row_fingerprint(
            con, table=TARGET_TABLE, where_sql=target_where, params=TARGET_DATES
        )
        post_non_target = row_fingerprint(
            con, table=TARGET_TABLE, where_sql=non_target_where, params=TARGET_DATES
        )
        post_20260717 = row_fingerprint(
            con, table=TARGET_TABLE, where_sql="trade_date = ?", params=["20260717"]
        )
        max_trade_date = str(
            con.execute(f"SELECT max(trade_date) FROM {quote(TARGET_TABLE)}").fetchone()[0]
        )
        post_stats = {trade_date: date_stats(con, TARGET_TABLE, trade_date) for trade_date in TARGET_DATES}

    if apply and pre_non_target != post_non_target:
        raise RuntimeError("non-target rows changed during controlled 3D window rerun")
    if apply and pre_20260717 != post_20260717:
        raise RuntimeError("20260717 rows changed during controlled 3D window rerun")
    if apply:
        for trade_date, stats in post_stats.items():
            expected = prewrite_validation[trade_date]
            if stats["row_count"] != expected["row_count"] or stats["stock_count"] != expected["stock_count"]:
                raise RuntimeError(f"post-write coverage mismatch for {trade_date}: {stats}")
            if stats["null_pred_prob"] or stats["bj_rows"] or stats["duplicate_key_groups"]:
                raise RuntimeError(f"post-write quality gate failed for {trade_date}: {stats}")
            if stats["distinct_pred_prob"] < 20 or stats["std_pred_prob"] <= 1e-5:
                raise RuntimeError(f"post-write distribution remains degenerate for {trade_date}: {stats}")

    reported_stats = post_stats if apply else {
        trade_date: {
            "trade_date": trade_date,
            **{
                key: value
                for key, value in prewrite_validation[trade_date].items()
                if key
                in {
                    "row_count",
                    "stock_count",
                    "distinct_pred_prob",
                    "std_pred_prob",
                    "null_pred_prob",
                    "bj_rows",
                    "duplicate_key_groups",
                }
            },
            "min_pred_prob": None,
            "max_pred_prob": None,
        }
        for trade_date in TARGET_DATES
    }

    report = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "task_id": "incremental-trading-signal-20260717-L4-3d-formal-window-rerun-20260701-20260716",
        "status": "completed_waiting_for_audit" if apply else "dry_run_passed",
        "risk_level": "P2_source_limited_input_disclosed",
        "date_scope": {
            "target_dates": TARGET_DATES,
            "forbidden_rewrite_date": "20260717",
            "controller_date_context": "2026-07-19",
        },
        "inputs": {
            "factor_asset": f"{factor_path}::{factor_table}",
            "label_asset": f"{label_path}::{label_table}",
            "label_max_trade_date": label_max,
            "upstream_handoff": str(UPSTREAM_HANDOFF),
            "model_metadata": str(spec["metadata_path"]),
        },
        "output": {
            "asset": f"{TARGET_DB}::{TARGET_TABLE}",
            "manifest": str(MANIFEST_PATH),
            "manifest_modified": False,
            "max_trade_date": max_trade_date,
            "rollback_snapshot": str(ROLLBACK_DB) if apply else None,
        },
        "validation": {
            trade_date: {**reported_stats[trade_date], "feature_null_semantics": prewrite_validation[trade_date]["feature_null_semantics"]}
            for trade_date in TARGET_DATES
        },
        "fingerprints": {
            "target_before": pre_target,
            "target_after": post_target,
            "non_target_before": pre_non_target,
            "non_target_after": post_non_target,
            "non_target_unchanged": pre_non_target == post_non_target,
            "date_20260717_before": pre_20260717,
            "date_20260717_after": post_20260717,
            "date_20260717_unchanged": pre_20260717 == post_20260717,
        },
        "source_limited_disclosure": {
            "dates": sorted(SOURCE_LIMITED_DATES),
            "fields": SOURCE_LIMITED_FIELDS,
            "semantics": "active_l2_source_limited_null_preserved_without_imputation",
            "model_handling": "archived_xgboost_native_missing_value_handling",
        },
        "ready_for_audit_review": bool(apply),
        "allow_next_layer_continue": False,
        "boundaries": {
            "no_training": True,
            "no_tuning": True,
            "no_other_horizon_write": True,
            "no_20260717_write": True,
            "no_signal": True,
            "no_backtest": True,
            "no_l5_l8": True,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="受控重刷 3D formal 的 20260701-20260716 窗口")
    parser.add_argument("--apply", action="store_true", help="通过全部门禁后写入 3D formal 窗口")
    args = parser.parse_args()
    report = run(apply=args.apply)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    VALIDATION_JSON.write_text(
        json.dumps(
            {
                "generated_at": report["generated_at"],
                "task_id": report["task_id"],
                "validation": report["validation"],
                "fingerprints": report["fingerprints"],
                "source_limited_disclosure": report["source_limited_disclosure"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    HANDOFF_JSON.write_text(
        json.dumps(
            {
                "generated_at": report["generated_at"],
                "task_id": report["task_id"],
                "status": report["status"],
                "ready_for_audit_review": report["ready_for_audit_review"],
                "allow_next_layer_continue": report["allow_next_layer_continue"],
                "report_path": str(REPORT_JSON),
                "validation_path": str(VALIDATION_JSON),
                "output_asset": report["output"]["asset"],
                "residual_risk": report["risk_level"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    REPORT_MD.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
