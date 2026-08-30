from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb

import incremental_formal_l4_duckdb_mainline as l4


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MAIN_DIR = ROOT / "quant" / "main"
MANIFEST = MAIN_DIR / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json"
MANIFEST_ARCHIVE = MANIFEST.with_name(
    "executable_3d_open_return_l4_formal_20260617_archive_before_20260720_3d_window_metadata_remediation.json"
)
FORMAL_DB = DATA_DIR / "production_assets" / "duckdb" / "l4_executable_3d_open_return_formal.duckdb"
FORMAL_TABLE = l4.FORMAL_TABLES["3d"]
FEATURE_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
LABEL_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
WINDOW_REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_formal_window_rerun_20260701_20260716"
WINDOW_REPORT = WINDOW_REPORT_DIR / "l4_3d_formal_window_rerun_report.json"
WINDOW_REPORT_MD = WINDOW_REPORT_DIR / "l4_3d_formal_window_rerun_report.md"
WINDOW_VALIDATION = WINDOW_REPORT_DIR / "l4_3d_formal_window_distribution_validation.json"
ROLLBACK_DB = WINDOW_REPORT_DIR / "l4_3d_formal_window_before_rerun.duckdb"
ROLLBACK_TABLE = "l4_3d_formal_window_before_rerun"
UPSTREAM_SWITCH_REPORT = (
    DATA_DIR
    / "reports"
    / "l3_required_feature_window_repair_active_switch_20260719"
    / "l3_required_feature_window_repair_active_switch_20260719.json"
)
UPSTREAM_HANDOFF = (
    DATA_DIR
    / "reports"
    / "l3_required_feature_window_repair_active_switch_20260719"
    / "l3_to_l4_handoff_contract_required_feature_window_repair_active_20260719.json"
)
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_formal_manifest_metadata_remediation_20260720"
REPORT_JSON = REPORT_DIR / "metadata_remediation_report.json"
REPORT_MD = REPORT_DIR / "metadata_remediation_report.md"
VALIDATION_JSON = REPORT_DIR / "metadata_consistency_validation.json"
AUDIT_HANDOFF_JSON = REPORT_DIR / "metadata_remediation_audit_handoff.json"

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
FEATURE_DIGEST_COLUMNS = [
    "trade_date",
    "stock_code",
    "index_2000_open",
    "index_2000_high",
    "index_2000_low",
    "index_2000_amount",
    "turnover_rate",
    "turnover_rate_f",
    "total_mv",
]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_state(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": file_sha256(path) if include_sha256 else None,
    }


def ordered_digest(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    columns: list[str],
    where_sql: str | None = None,
    params: list[str] | None = None,
) -> str:
    digest = hashlib.sha256()
    sql = f"SELECT {', '.join(quote(column) for column in columns)} FROM {quote(table)}"
    if where_sql:
        sql += f" WHERE {where_sql}"
    sql += " ORDER BY trade_date, stock_code"
    cursor = con.execute(sql, params or [])
    while True:
        rows = cursor.fetchmany(100_000)
        if not rows:
            break
        for row in rows:
            encoded = "|".join("<NULL>" if value is None else format(value, ".17g") if isinstance(value, float) else str(value) for value in row)
            digest.update(encoded.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def formal_state() -> dict[str, Any]:
    with duckdb.connect(str(FORMAL_DB), read_only=True) as con:
        stats = l4.duckdb_stats(con, FORMAL_TABLE, latest_date="20260717")
        window_digest = ordered_digest(
            con,
            table=FORMAL_TABLE,
            columns=["trade_date", "stock_code", "pred_prob"],
            where_sql=f"trade_date IN ({', '.join(['?'] * len(TARGET_DATES))})",
            params=TARGET_DATES,
        )
    return {
        "asset": f"{FORMAL_DB}::{FORMAL_TABLE}",
        "file_state": file_state(FORMAL_DB),
        "table_stats": stats,
        "window_pred_prob_sha256": window_digest,
    }


def rollback_state() -> dict[str, Any]:
    with duckdb.connect(str(ROLLBACK_DB), read_only=True) as con:
        row = con.execute(
            f"SELECT count(*), min(trade_date), max(trade_date), count(distinct trade_date) FROM {quote(ROLLBACK_TABLE)}"
        ).fetchone()
        digest = ordered_digest(
            con,
            table=ROLLBACK_TABLE,
            columns=["trade_date", "stock_code", "pred_prob"],
        )
    return {
        "asset": f"{ROLLBACK_DB}::{ROLLBACK_TABLE}",
        "file_state": file_state(ROLLBACK_DB),
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "pred_prob_sha256": digest,
    }


def active_l3_state() -> dict[str, Any]:
    switch = json.loads(UPSTREAM_SWITCH_REPORT.read_text(encoding="utf-8"))
    audited_feature = switch["active_feature_after"]["file_state"]
    audited_label = switch["pre_switch_fingerprints"]["active_label"]
    feature_current = file_state(FEATURE_DB, include_sha256=False)
    label_current = file_state(LABEL_DB, include_sha256=False)
    for name, current, audited in [
        ("feature", feature_current, audited_feature),
        ("label", label_current, audited_label),
    ]:
        if current["size_bytes"] != int(audited["size_bytes"]) or current["mtime_ns"] != int(audited["mtime_ns"]):
            raise RuntimeError(f"active L3 {name} file state drifted from audited fingerprint")

    with duckdb.connect(str(FEATURE_DB), read_only=True) as feature_con:
        feature_window_digest = ordered_digest(
            feature_con,
            table=FEATURE_TABLE,
            columns=FEATURE_DIGEST_COLUMNS,
            where_sql=f"trade_date IN ({', '.join(['?'] * len(TARGET_DATES))})",
            params=TARGET_DATES,
        )
        feature_row = feature_con.execute(
            f"SELECT count(*), min(trade_date), max(trade_date) FROM {quote(FEATURE_TABLE)}"
        ).fetchone()
    with duckdb.connect(str(LABEL_DB), read_only=True) as label_con:
        label_row = label_con.execute(
            f"SELECT count(*), min(trade_date), max(trade_date) FROM {quote(LABEL_TABLE)}"
        ).fetchone()

    return {
        "feature": {
            "asset": f"{FEATURE_DB}::{FEATURE_TABLE}",
            "file_state_current": feature_current,
            "audited_file_sha256": str(audited_feature["sha256"]),
            "audited_fingerprint_source": str(UPSTREAM_SWITCH_REPORT),
            "current_stat_matches_audited_fingerprint": True,
            "target_window_required_feature_sha256": feature_window_digest,
            "row_count": int(feature_row[0]),
            "min_trade_date": str(feature_row[1]),
            "max_trade_date": str(feature_row[2]),
        },
        "label": {
            "asset": f"{LABEL_DB}::{LABEL_TABLE}",
            "file_state_current": label_current,
            "audited_file_sha256": str(audited_label["sha256"]),
            "audited_fingerprint_source": str(UPSTREAM_SWITCH_REPORT),
            "current_stat_matches_audited_fingerprint": True,
            "row_count": int(label_row[0]),
            "min_trade_date": str(label_row[1]),
            "max_trade_date": str(label_row[2]),
        },
    }


def build_markdown(report: dict[str, Any]) -> str:
    formal = report["formal_after"]
    lines = [
        "# 3D formal manifest metadata 一致性整改报告",
        "",
        "## 结论",
        "",
        "- 本次仅修复 metadata、lineage 与 digest；没有重新预测，也没有改写正式表任何行。",
        f"- 全表 `pred_prob_sha256` 已更新为 `{formal['table_stats']['pred_prob_sha256']}`。",
        "- 正式 DuckDB 文件 SHA256、全表逻辑摘要和目标窗口摘要在整改前后完全一致。",
        "- 当前仍为待审计状态，`allow_next_layer_continue=false`。",
        "",
        "## 绑定资产",
        "",
        f"- 正式 3D：`{formal['asset']}`",
        f"- rollback：`{report['rollback']['asset']}`",
        f"- active L3 feature：`{report['active_l3']['feature']['asset']}`",
        f"- active L3 label：`{report['active_l3']['label']['asset']}`",
        f"- 旧 manifest 归档：`{report['manifest']['archive_path']}`",
        "",
        "## 窗口修复 lineage",
        "",
        "- 窗口：`20260701-20260716`，共 12 个交易日。",
        "- 方法：使用已归档 3D 生产模型，对已审计修复后的 active L3 特征执行受控重刷。",
        "- `20260706/07` 的 `index_2000_open/high/low/amount` 保留 active L2 源头受限空值语义，未补值。",
        "- `20260717` 及其他非目标日期未改写。",
        "",
        "## 边界",
        "",
        "- 未重新预测",
        "- 未训练或调参",
        "- 未修改 1D/5D/10D/compat",
        "- 未生成信号或运行回测",
        "- 未推进 L5-L8",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    required_paths = [MANIFEST, FORMAL_DB, FEATURE_DB, LABEL_DB, WINDOW_REPORT, WINDOW_VALIDATION, ROLLBACK_DB, UPSTREAM_SWITCH_REPORT, UPSTREAM_HANDOFF]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required remediation evidence is missing: {missing}")

    formal_before = formal_state()
    rollback = rollback_state()
    active_l3 = active_l3_state()
    old_manifest_bytes = MANIFEST.read_bytes()
    old_manifest_sha256 = hashlib.sha256(old_manifest_bytes).hexdigest()
    if MANIFEST_ARCHIVE.exists():
        if MANIFEST_ARCHIVE.read_bytes() != old_manifest_bytes:
            raise RuntimeError("existing manifest archive does not match current pre-remediation manifest")
    else:
        shutil.copy2(MANIFEST, MANIFEST_ARCHIVE)

    window_report = json.loads(WINDOW_REPORT.read_text(encoding="utf-8"))
    if not window_report.get("ready_for_audit_review") or window_report.get("allow_next_layer_continue"):
        raise RuntimeError("3D window rerun report is not in the expected waiting-for-audit state")

    generated_at = now_iso()
    payload = json.loads(old_manifest_bytes.decode("utf-8"))
    stats = formal_before["table_stats"]
    payload.update(
        {
            "generated_at": generated_at,
            "row_count": stats["row_count"],
            "trade_days": stats["trade_days"],
            "stock_count": stats["stock_count"],
            "min_trade_date": stats["min_trade_date"],
            "max_trade_date": stats["max_trade_date"],
            "latest_day_rows": stats["latest_day_rows"],
            "latest_day_stock_count": stats["latest_day_stock_count"],
            "duplicate_keys": stats["duplicate_key_groups"],
            "null_pred_prob": stats["null_pred_prob"],
            "pred_prob_sha256": stats["pred_prob_sha256"],
            "metadata_consistency_status": "remediated_pending_read_only_reaudit",
            "allow_next_layer_continue": False,
            "latest_metadata_remediation": {
                "task_id": "incremental-trading-signal-20260717-L4-3d-formal-manifest-metadata-remediation",
                "remediated_at": generated_at,
                "metadata_only": True,
                "formal_values_recomputed": False,
                "formal_table_rows_rewritten": False,
                "window": {"start": "20260701", "end": "20260716", "trade_days": 12},
                "report_path": "../../../data_file/reports/model_agent_3d_formal_manifest_metadata_remediation_20260720/metadata_remediation_report.json",
                "validation_path": "../../../data_file/reports/model_agent_3d_formal_manifest_metadata_remediation_20260720/metadata_consistency_validation.json",
                "audit_status": "pending_read_only_reaudit",
            },
            "window_repair_lineage": {
                "task_id": "incremental-trading-signal-20260717-L4-3d-formal-window-rerun-20260701-20260716",
                "window": {"start": "20260701", "end": "20260716", "trade_dates": TARGET_DATES},
                "method": "saved_3d_production_model_on_audited_repaired_active_l3_feature",
                "formal_asset": formal_before["asset"],
                "formal_file_sha256": formal_before["file_state"]["sha256"],
                "formal_full_table_pred_prob_sha256": stats["pred_prob_sha256"],
                "formal_window_pred_prob_sha256": formal_before["window_pred_prob_sha256"],
                "rollback_asset": rollback["asset"],
                "rollback_file_sha256": rollback["file_state"]["sha256"],
                "rollback_window_pred_prob_sha256": rollback["pred_prob_sha256"],
                "active_l3_feature": active_l3["feature"],
                "active_l3_label": active_l3["label"],
                "source_limited_semantics": {
                    "dates": ["20260706", "20260707"],
                    "fields": ["index_2000_open", "index_2000_high", "index_2000_low", "index_2000_amount"],
                    "handling": "active_l2_source_limited_null_preserved_without_imputation",
                },
                "rerun_report": "../../../data_file/reports/model_agent_3d_formal_window_rerun_20260701_20260716/l4_3d_formal_window_rerun_report.json",
                "distribution_validation": "../../../data_file/reports/model_agent_3d_formal_window_rerun_20260701_20260716/l4_3d_formal_window_distribution_validation.json",
                "non_target_rows_unchanged": True,
                "date_20260717_unchanged": True,
            },
            "previous_manifest_archive_before_3d_window_metadata_remediation": MANIFEST_ARCHIVE.name,
        }
    )
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    formal_after = formal_state()
    formal_numeric_unchanged = (
        formal_before["file_state"]["sha256"] == formal_after["file_state"]["sha256"]
        and formal_before["table_stats"]["pred_prob_sha256"] == formal_after["table_stats"]["pred_prob_sha256"]
        and formal_before["window_pred_prob_sha256"] == formal_after["window_pred_prob_sha256"]
    )
    if not formal_numeric_unchanged:
        raise RuntimeError("formal 3D numeric asset changed during metadata-only remediation")

    sys.path.insert(0, str(MAIN_DIR))
    from prediction_manifest import load_prediction_source_manifest

    loaded = load_prediction_source_manifest(str(MANIFEST), require_approved=True, allow_legacy=False)
    manifest_after_sha256 = file_sha256(MANIFEST)
    archive_sha256 = file_sha256(MANIFEST_ARCHIVE)
    if archive_sha256 != old_manifest_sha256:
        raise RuntimeError("manifest archive fingerprint mismatch")
    if payload["pred_prob_sha256"] != formal_after["table_stats"]["pred_prob_sha256"]:
        raise RuntimeError("manifest pred_prob_sha256 does not match current formal table")

    report = {
        "generated_at": generated_at,
        "actor": "model-agent",
        "task_id": "incremental-trading-signal-20260717-L4-3d-formal-manifest-metadata-remediation",
        "status": "completed_waiting_for_reaudit",
        "date_scope": {"window_start": "20260701", "window_end": "20260716", "trade_days": 12},
        "metadata_only": True,
        "formal_values_recomputed": False,
        "formal_table_rows_rewritten": False,
        "formal_before": formal_before,
        "formal_after": formal_after,
        "formal_numeric_unchanged": formal_numeric_unchanged,
        "rollback": rollback,
        "active_l3": active_l3,
        "manifest": {
            "path": str(MANIFEST),
            "archive_path": str(MANIFEST_ARCHIVE),
            "before_sha256": old_manifest_sha256,
            "archive_sha256": archive_sha256,
            "after_sha256": manifest_after_sha256,
            "pred_prob_sha256_matches_formal_table": True,
            "loader_validation": {
                "approval_status": loaded["approval_status"],
                "source_type": loaded["source_type"],
                "db_path": str(loaded["db_path"]),
                "table": loaded["table"],
            },
        },
        "window_rerun_report": str(WINDOW_REPORT),
        "window_distribution_validation": str(WINDOW_VALIDATION),
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "residual_risk": [
            "20260706/20260707 four index_2000 fields remain source-limited null by active L2 semantics",
            "L5-L8 remain blocked until audit-agent read-only re-audit passes and commander closes the gate",
        ],
        "boundaries": {
            "no_prediction": True,
            "no_formal_numeric_write": True,
            "no_training": True,
            "no_tuning": True,
            "no_other_horizon_change": True,
            "no_signal": True,
            "no_backtest": True,
            "no_l5_l8": True,
        },
    }

    validation = {
        "generated_at": generated_at,
        "task_id": report["task_id"],
        "checks": {
            "formal_file_sha256_unchanged": formal_before["file_state"]["sha256"] == formal_after["file_state"]["sha256"],
            "formal_full_table_pred_prob_sha256_unchanged": formal_before["table_stats"]["pred_prob_sha256"] == formal_after["table_stats"]["pred_prob_sha256"],
            "formal_window_pred_prob_sha256_unchanged": formal_before["window_pred_prob_sha256"] == formal_after["window_pred_prob_sha256"],
            "manifest_pred_prob_sha256_matches_formal": payload["pred_prob_sha256"] == formal_after["table_stats"]["pred_prob_sha256"],
            "manifest_archive_matches_pre_remediation": archive_sha256 == old_manifest_sha256,
            "active_feature_stat_matches_audited_fingerprint": active_l3["feature"]["current_stat_matches_audited_fingerprint"],
            "active_label_stat_matches_audited_fingerprint": active_l3["label"]["current_stat_matches_audited_fingerprint"],
            "manifest_loader_approved_for_l5_duckdb_table": loaded["approval_status"] == "approved_for_l5" and loaded["source_type"] == "duckdb_table",
            "formal_rows_unchanged": True,
            "prediction_not_called": True,
        },
        "digests": {
            "formal_file_sha256": formal_after["file_state"]["sha256"],
            "formal_full_table_pred_prob_sha256": formal_after["table_stats"]["pred_prob_sha256"],
            "formal_window_pred_prob_sha256": formal_after["window_pred_prob_sha256"],
            "rollback_file_sha256": rollback["file_state"]["sha256"],
            "rollback_window_pred_prob_sha256": rollback["pred_prob_sha256"],
            "active_feature_audited_file_sha256": active_l3["feature"]["audited_file_sha256"],
            "active_feature_target_window_required_feature_sha256": active_l3["feature"]["target_window_required_feature_sha256"],
            "active_label_audited_file_sha256": active_l3["label"]["audited_file_sha256"],
            "manifest_before_sha256": old_manifest_sha256,
            "manifest_after_sha256": manifest_after_sha256,
        },
    }
    if not all(validation["checks"].values()):
        raise RuntimeError(f"metadata remediation validation failed: {validation['checks']}")

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    VALIDATION_JSON.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT_HANDOFF_JSON.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "task_id": report["task_id"],
                "status": report["status"],
                "ready_for_audit_review": True,
                "allow_next_layer_continue": False,
                "report_path": str(REPORT_JSON),
                "validation_path": str(VALIDATION_JSON),
                "manifest_path": str(MANIFEST),
                "manifest_archive_path": str(MANIFEST_ARCHIVE),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(build_markdown(report), encoding="utf-8")

    window_report["metadata_remediation"] = {
        "status": "completed_waiting_for_reaudit",
        "metadata_only": True,
        "formal_values_recomputed": False,
        "formal_table_rows_rewritten": False,
        "formal_full_table_pred_prob_sha256": formal_after["table_stats"]["pred_prob_sha256"],
        "formal_file_sha256": formal_after["file_state"]["sha256"],
        "rollback_file_sha256": rollback["file_state"]["sha256"],
        "report_path": str(REPORT_JSON),
        "validation_path": str(VALIDATION_JSON),
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    WINDOW_REPORT.write_text(json.dumps(window_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with WINDOW_REPORT_MD.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Metadata 一致性整改\n\n"
            f"- 已完成 metadata-only 整改，正式表数值未重算、未改写。\n"
            f"- 全表 `pred_prob_sha256`：`{formal_after['table_stats']['pred_prob_sha256']}`。\n"
            f"- 整改报告：`{REPORT_JSON}`。\n"
            "- 当前仍等待审计复核，`allow_next_layer_continue=false`。\n"
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
