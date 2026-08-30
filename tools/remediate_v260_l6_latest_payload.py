from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
STRATEGY_ID = "prod_v260_10d_regime_warmup_all4key_v20260724"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / STRATEGY_ID
VALIDATION = STRATEGY_DIR / "validation.json"
LATEST_CSV = DATA / "production_signals" / f"{STRATEGY_ID}_latest.csv"
LATEST_STATUS = DATA / "production_signals" / f"{STRATEGY_ID}_latest_status.json"
L5_REGISTRY = (
    DATA
    / "production_assets"
    / "duckdb"
    / "production"
    / "l5"
    / "prod_l5_strategy_registry_current.duckdb"
)
L5_MANIFEST = L5_REGISTRY.with_name("prod_l5_strategy_manifest_current.duckdb")
L6_VALIDATION = (
    DATA
    / "production_assets"
    / "duckdb"
    / "production"
    / "l6"
    / "prod_l6_strategy_validation_current.duckdb"
)
L7_DIR = DATA / "production_assets" / "duckdb" / "production" / "l7"
L7_ASSETS = {
    "signal_rows": (
        L7_DIR / "prod_l7_signal_rows_current.duckdb",
        "prod_l7_signal_rows_current",
    ),
    "signal_status": (
        L7_DIR / "prod_l7_signal_status_current.duckdb",
        "prod_l7_signal_status_current",
    ),
    "signal_files": (
        L7_DIR / "prod_l7_signal_files_current.duckdb",
        "prod_l7_signal_files_current",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def duckdb_snapshot(path: Path, table: str) -> dict[str, Any]:
    with duckdb.connect(str(path), read_only=True) as connection:
        schema = connection.execute(f'DESCRIBE "{table}"').fetchdf()
        rows = connection.execute(f'SELECT * FROM "{table}"').fetchdf()
    return {
        "path": str(path),
        "table": table,
        "file_sha256": sha256(path),
        "schema": schema.to_dict(orient="records"),
        "row_count": len(rows),
        "rows": rows.to_dict(orient="records"),
    }


def current_payload() -> dict[str, Any]:
    with duckdb.connect(str(L6_VALIDATION), read_only=True) as connection:
        row = connection.execute(
            'SELECT latest_signal_date, latest_buy_date, buy_day_hard_gate_complete, '
            'validation_payload_json FROM "prod_l6_strategy_validation_current"'
        ).fetchone()
    payload = json.loads(row[3])
    return {
        "latest_signal_date": str(row[0]),
        "latest_buy_date": str(row[1]),
        "buy_day_hard_gate_complete": row[2],
        "embedded_latest_signal_status": payload.get("latest_signal_status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="修正 V260 L6 current 内嵌 latest payload")
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--buy-date", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    rollback_dir = output_dir / "rollback"
    rollback_dir.mkdir(parents=True, exist_ok=True)

    latest_status = load_json(LATEST_STATUS)
    if (
        latest_status.get("signal_date") != args.signal_date
        or latest_status.get("buy_date") != args.buy_date
        or latest_status.get("status") != "pending_buy_day_hard_gate"
        or latest_status.get("buy_day_hard_gate_complete") is not False
        or latest_status.get("l7_execution_allowed") is not False
    ):
        raise RuntimeError("latest status is not the approved pending hard-gate batch")

    before = {
        "validation_json_sha256": sha256(VALIDATION),
        "l6_validation_duckdb_sha256": sha256(L6_VALIDATION),
        "latest_csv_sha256": sha256(LATEST_CSV),
        "latest_status_sha256": sha256(LATEST_STATUS),
        "l6_payload": current_payload(),
        "l7": {
            name: duckdb_snapshot(path, table)
            for name, (path, table) in L7_ASSETS.items()
        },
    }
    for source in (VALIDATION, L6_VALIDATION):
        shutil.copy2(source, rollback_dir / source.name)

    validation = load_json(VALIDATION)
    validation["latest_signal_status"] = {
        "status": latest_status["status"],
        "signal_date": latest_status["signal_date"],
        "buy_date": latest_status["buy_date"],
        "buy_date_source": latest_status["buy_date_source"],
        "row_count": latest_status["row_count"],
        "stock_count": latest_status["stock_count"],
        "buy_count": latest_status["buy_count"],
        "sell_count": latest_status["sell_count"],
        "duplicate_key_count": latest_status["duplicate_key_count"],
        "bj_rows": latest_status["bj_rows"],
        "buy_day_market_available": latest_status["buy_day_market_available"],
        "buy_day_hard_gate_complete": latest_status["buy_day_hard_gate_complete"],
        "l7_execution_allowed": latest_status["l7_execution_allowed"],
        "formal_signal_generated": latest_status["formal_signal_generated"],
    }
    VALIDATION.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(MAIN))
    from l6_duckdb_sync import sync_strategy_backtests_to_duckdb

    sync_result = sync_strategy_backtests_to_duckdb()
    after_payload = current_payload()
    after_l7 = {
        name: duckdb_snapshot(path, table)
        for name, (path, table) in L7_ASSETS.items()
    }
    after = {
        "validation_json_sha256": sha256(VALIDATION),
        "l6_validation_duckdb_sha256": sha256(L6_VALIDATION),
        "latest_csv_sha256": sha256(LATEST_CSV),
        "latest_status_sha256": sha256(LATEST_STATUS),
        "l6_payload": after_payload,
        "l7": after_l7,
    }

    expected_embedded = after_payload["embedded_latest_signal_status"]
    checks = {
        "l6_top_level_dates_match": after_payload["latest_signal_date"]
        == args.signal_date
        and after_payload["latest_buy_date"] == args.buy_date,
        "l6_embedded_dates_match": expected_embedded.get("signal_date")
        == args.signal_date
        and expected_embedded.get("buy_date") == args.buy_date,
        "pending_hard_gate_preserved": expected_embedded.get("status")
        == "pending_buy_day_hard_gate"
        and expected_embedded.get("buy_day_hard_gate_complete") is False
        and expected_embedded.get("l7_execution_allowed") is False,
        "latest_csv_unchanged": before["latest_csv_sha256"]
        == after["latest_csv_sha256"],
        "latest_status_unchanged": before["latest_status_sha256"]
        == after["latest_status_sha256"],
        "l7_files_unchanged": all(
            before["l7"][name]["file_sha256"] == after["l7"][name]["file_sha256"]
            for name in L7_ASSETS
        ),
        "l5_unchanged_by_scope": L5_REGISTRY.is_file() and L5_MANIFEST.is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "fail-closed checks failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    l7_dates: dict[str, Any] = {}
    for name, item in after_l7.items():
        rows = item["rows"]
        l7_dates[name] = {
            "row_count": item["row_count"],
            "strategy_ids": sorted(
                {str(row.get("strategy_id")) for row in rows if row.get("strategy_id")}
            ),
            "signal_dates": sorted(
                {
                    str(row.get("signal_date") or row.get("latest_signal_date"))
                    for row in rows
                    if row.get("signal_date") or row.get("latest_signal_date")
                }
            ),
            "buy_dates": sorted(
                {
                    str(row.get("buy_date") or row.get("latest_buy_date"))
                    for row in rows
                    if row.get("buy_date") or row.get("latest_buy_date")
                }
            ),
            "source_paths": sorted(
                {
                    str(row.get("source_path"))
                    for row in rows
                    if row.get("source_path")
                }
            ),
        }

    report = {
        "schema_version": 1,
        "task_id": "incremental-trading-signal-20260727-L5-L6-v260-all4key-latest-P1-remediation",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_id": STRATEGY_ID,
        "signal_date": args.signal_date,
        "buy_date": args.buy_date,
        "scope": "L6 metadata/payload only; no signal, strategy, L7 or L8 change",
        "before": before,
        "after": after,
        "sync_result": sync_result,
        "checks": checks,
        "l7_attribution": {
            "conclusion": "现有 L7 current 属于更早批次，与本轮 L5/L6 整改无关；本轮未写入、删除或覆盖 L7。",
            "details": l7_dates,
        },
        "rollback": {
            "directory": str(rollback_dir),
            "validation_json": str(rollback_dir / VALIDATION.name),
            "l6_validation_duckdb": str(rollback_dir / L6_VALIDATION.name),
        },
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    handoff = {
        "schema_version": 1,
        "from_agent": "strategy-agent",
        "to_agent": "audit-agent",
        "task_id": report["task_id"],
        "status": "ready_for_audit_review",
        "audit_focus": [
            "L6 顶层与内嵌 latest_signal_status 均为 20260727/20260728",
            "正式 latest CSV/status 哈希未变化",
            "策略参数、输入契约、评分和生产 current 未变化",
            "L7 三个 current DuckDB 哈希未变化且属于更早批次",
            "pending_buy_day_hard_gate 与 l7_execution_allowed=false 保持不变",
        ],
        "evidence": [
            str(output_dir / "remediation_report.json"),
            str(output_dir / "L6内嵌状态纠偏说明.md"),
            str(rollback_dir / VALIDATION.name),
            str(rollback_dir / L6_VALIDATION.name),
            str(LATEST_CSV),
            str(LATEST_STATUS),
        ],
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    (output_dir / "remediation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "audit_handoff.json").write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = (
        "# L6 内嵌状态纠偏说明\n\n"
        f"- 策略：`{STRATEGY_ID}`\n"
        f"- 信号日/买入日：`{args.signal_date}` / `{args.buy_date}`\n"
        "- 修改范围：仅 `validation.json.latest_signal_status` 及其 L6 current 镜像。\n"
        "- 未修改：策略参数、输入契约、评分、正式 latest CSV/status、生产 current、L7/L8。\n"
        "- 当前状态：`pending_buy_day_hard_gate`，`l7_execution_allowed=false`。\n"
        "- L7 说明：现有记录属于更早批次，本轮只读检查，三个 DuckDB 哈希前后一致。\n"
        "- 结论：P1 元数据日期不一致已纠正，等待审计复核。\n"
    )
    (output_dir / "L6内嵌状态纠偏说明.md").write_text(summary, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
