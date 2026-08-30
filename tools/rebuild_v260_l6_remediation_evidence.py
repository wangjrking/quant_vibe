from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "quant" / "data_file"
MAIN = ROOT / "quant" / "main"
STRATEGY_ID = "prod_v260_10d_regime_warmup_all4key_v20260724"
REPORT_DIR = (
    DATA
    / "reports"
    / "strategy_agent_v260_all4key_l5_l6_latest_20260727_p1_remediation"
)
ROLLBACK_DIR = REPORT_DIR / "rollback"
VALIDATION = (
    MAIN
    / "strategy_library"
    / "production"
    / STRATEGY_ID
    / "validation.json"
)
L6_CURRENT = (
    DATA
    / "production_assets"
    / "duckdb"
    / "production"
    / "l6"
    / "prod_l6_strategy_validation_current.duckdb"
)
LATEST_CSV = DATA / "production_signals" / f"{STRATEGY_ID}_latest.csv"
LATEST_STATUS = DATA / "production_signals" / f"{STRATEGY_ID}_latest_status.json"
L7_ASSETS = {
    "signal_rows": (
        DATA
        / "production_assets"
        / "duckdb"
        / "production"
        / "l7"
        / "prod_l7_signal_rows_current.duckdb",
        "prod_l7_signal_rows_current",
    ),
    "signal_status": (
        DATA
        / "production_assets"
        / "duckdb"
        / "production"
        / "l7"
        / "prod_l7_signal_status_current.duckdb",
        "prod_l7_signal_status_current",
    ),
    "signal_files": (
        DATA
        / "production_assets"
        / "duckdb"
        / "production"
        / "l7"
        / "prod_l7_signal_files_current.duckdb",
        "prod_l7_signal_files_current",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def read_l6(path: Path) -> dict[str, Any]:
    with duckdb.connect(str(path), read_only=True) as connection:
        row = connection.execute(
            "SELECT latest_signal_date, latest_buy_date, "
            "buy_day_hard_gate_complete, validation_payload_json "
            'FROM "prod_l6_strategy_validation_current"'
        ).fetchone()
    payload = json.loads(row[3])
    return {
        "latest_signal_date": str(row[0]),
        "latest_buy_date": str(row[1]),
        "buy_day_hard_gate_complete": row[2],
        "embedded_latest_signal_status": payload.get("latest_signal_status"),
    }


def read_l7(path: Path, table: str) -> dict[str, Any]:
    with duckdb.connect(str(path), read_only=True) as connection:
        schema = connection.execute(f'DESCRIBE "{table}"').fetchdf()
        rows = connection.execute(f'SELECT * FROM "{table}"').fetchdf()
    records = safe(rows.to_dict(orient="records"))
    return {
        "path": str(path),
        "table": table,
        "sha256": sha256(path),
        "schema": safe(schema.to_dict(orient="records")),
        "row_count": len(records),
        "strategy_ids": sorted(
            {str(row["strategy_id"]) for row in records if row.get("strategy_id")}
        ),
        "signal_dates": sorted(
            {
                str(row.get("signal_date") or row.get("latest_signal_date"))
                for row in records
                if row.get("signal_date") or row.get("latest_signal_date")
            }
        ),
        "buy_dates": sorted(
            {
                str(row.get("buy_date") or row.get("latest_buy_date"))
                for row in records
                if row.get("buy_date") or row.get("latest_buy_date")
            }
        ),
        "source_paths": sorted(
            {str(row["source_path"]) for row in records if row.get("source_path")}
        ),
    }


def main() -> None:
    original_report = json.loads(
        (REPORT_DIR / "remediation_report.json").read_text(encoding="utf-8")
    )
    status = json.loads(LATEST_STATUS.read_text(encoding="utf-8"))
    current = read_l6(L6_CURRENT)
    rollback = read_l6(ROLLBACK_DIR / L6_CURRENT.name)
    l7 = {
        name: read_l7(path, table)
        for name, (path, table) in L7_ASSETS.items()
    }
    checks = {
        "before_embedded_batch_was_20260724_20260727": (
            rollback["embedded_latest_signal_status"]["signal_date"] == "20260724"
            and rollback["embedded_latest_signal_status"]["buy_date"] == "20260727"
        ),
        "l6_top_level_batch_is_20260727_20260728": (
            current["latest_signal_date"] == "20260727"
            and current["latest_buy_date"] == "20260728"
        ),
        "l6_embedded_batch_is_20260727_20260728": (
            current["embedded_latest_signal_status"]["signal_date"] == "20260727"
            and current["embedded_latest_signal_status"]["buy_date"] == "20260728"
        ),
        "latest_status_matches_l6": (
            status["signal_date"] == current["latest_signal_date"]
            and status["buy_date"] == current["latest_buy_date"]
        ),
        "pending_hard_gate_preserved": (
            status["status"] == "pending_buy_day_hard_gate"
            and status["buy_day_hard_gate_complete"] is False
            and status["l7_execution_allowed"] is False
            and current["embedded_latest_signal_status"]["l7_execution_allowed"]
            is False
        ),
        "latest_csv_matches_pre_change_hash": (
            sha256(LATEST_CSV)
            == "55E8C596AB179256461369B3E794EE95290FA9E732ACB09485076B14178A751B"
        ),
        "latest_status_matches_pre_change_hash": (
            sha256(LATEST_STATUS)
            == "8E25E8682AA9BD1E056EE127DFD962E5242904ECE5BE9BC25345254647ECE653"
        ),
        "l7_unchanged_during_remediation": all(
            original_report["before"]["l7"][name]["file_sha256"]
            == original_report["after"]["l7"][name]["file_sha256"]
            for name in L7_ASSETS
        ),
    }
    report = {
        "schema_version": 2,
        "task_id": (
            "incremental-trading-signal-20260727-L5-L6-"
            "v260-all4key-latest-P1-remediation"
        ),
        "strategy_id": STRATEGY_ID,
        "scope": "仅验证已完成的 L6 元数据纠偏；未再次写入业务资产",
        "before": {
            "validation_json_sha256": sha256(ROLLBACK_DIR / VALIDATION.name),
            "l6_current_sha256": sha256(ROLLBACK_DIR / L6_CURRENT.name),
            "l6_payload": rollback,
        },
        "after": {
            "validation_json_sha256": sha256(VALIDATION),
            "l6_current_sha256": sha256(L6_CURRENT),
            "latest_csv_sha256": sha256(LATEST_CSV),
            "latest_status_sha256": sha256(LATEST_STATUS),
            "l6_payload": current,
        },
        "l7_attribution": {
            "conclusion": (
                "本轮纠偏执行期间，L7 三个文件的前后哈希一致，"
                "当时属于 20260723/20260724 批次。纠偏完成后，"
                "L7 被外部后续流程更新为 20260727/20260728 批次；"
                "本任务未执行该更新，也未删除、覆盖或生成 L7。"
            ),
            "during_remediation": {
                name: {
                    "before_sha256": original_report["before"]["l7"][name][
                        "file_sha256"
                    ],
                    "after_sha256": original_report["after"]["l7"][name][
                        "file_sha256"
                    ],
                    "row_count": original_report["after"]["l7"][name][
                        "row_count"
                    ],
                }
                for name in L7_ASSETS
            },
            "current_read_only_probe": {
                "note": "纠偏完成后的外部更新，仅作只读归属说明",
                "assets": l7,
            },
        },
        "checks": checks,
        "rollback": {
            "validation_json": str(ROLLBACK_DIR / VALIDATION.name),
            "l6_current": str(ROLLBACK_DIR / L6_CURRENT.name),
        },
        "ready_for_audit_review": all(checks.values()),
        "allow_next_layer_continue": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    strict_path = REPORT_DIR / "remediation_report_strict.json"
    strict_path.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    handoff = {
        "schema_version": 2,
        "from_agent": "strategy-agent",
        "to_agent": "audit-agent",
        "task_id": report["task_id"],
        "status": "ready_for_audit_review",
        "evidence": [
            str(strict_path),
            str(REPORT_DIR / "L6内嵌状态纠偏说明.md"),
            str(ROLLBACK_DIR / VALIDATION.name),
            str(ROLLBACK_DIR / L6_CURRENT.name),
            str(LATEST_CSV),
            str(LATEST_STATUS),
        ],
        "ready_for_audit_review": all(checks.values()),
        "allow_next_layer_continue": False,
    }
    (REPORT_DIR / "audit_handoff_strict.json").write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = (
        "# L6 内嵌状态纠偏说明\n\n"
        f"- 策略：`{STRATEGY_ID}`\n"
        "- 本轮批次：`signal_date=20260727`，`buy_date=20260728`。\n"
        "- 纠偏内容：L6 current 内嵌 `latest_signal_status` 从旧批次 "
        "`20260724/20260727` 对齐为 `20260727/20260728`。\n"
        "- 未修改：策略参数、输入契约、评分、正式 latest CSV/status、"
        "production current、L7/L8。\n"
        "- 当前门禁：`pending_buy_day_hard_gate`，"
        "`buy_day_hard_gate_complete=false`，`l7_execution_allowed=false`。\n"
        "- L7 归属：纠偏执行期间仍为 `20260723/20260724` 旧批次，"
        "三个 L7 DuckDB 前后哈希一致；纠偏完成后由外部流程更新为 "
        "`20260727/20260728`，本任务未执行该更新。\n"
        "- 结论：P1 元数据日期不一致已纠正，等待审计独立复核。\n"
    )
    (REPORT_DIR / "L6内嵌状态纠偏说明.md").write_text(
        summary, encoding="utf-8"
    )
    if not all(checks.values()):
        raise RuntimeError(
            "证据自检失败："
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    print(strict_path)


if __name__ == "__main__":
    main()
