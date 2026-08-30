from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb


TABLES = (
    "prod_l7_signal_rows_current",
    "prod_l7_signal_status_current",
    "prod_l7_signal_files_current",
)
EXPECTED_ROW_COUNTS = {
    "prod_l7_signal_rows_current": 2,
    "prod_l7_signal_status_current": 1,
    "prod_l7_signal_files_current": 2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def table_snapshot(path: Path, table: str) -> dict[str, Any]:
    with duckdb.connect(str(path), read_only=True) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        ]
        schema = [list(row) for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()]
        row_count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        strategy_ids = [
            row[0]
            for row in connection.execute(
                f'SELECT DISTINCT strategy_id FROM "{table}" ORDER BY strategy_id'
            ).fetchall()
        ]
        source_paths = [
            row[0]
            for row in connection.execute(
                f'SELECT DISTINCT source_path FROM "{table}" ORDER BY source_path'
            ).fetchall()
            if row[0]
        ]
        status_values: list[str] = []
        if table == "prod_l7_signal_status_current":
            status_values = [
                row[0]
                for row in connection.execute(
                    f'SELECT DISTINCT status FROM "{table}" ORDER BY status'
                ).fetchall()
            ]
    return {
        "path": str(path),
        "table": table,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "tables": tables,
        "schema": schema,
        "row_count": row_count,
        "strategy_ids": strategy_ids,
        "source_paths": source_paths,
        "source_path_exists": {value: Path(value).is_file() for value in source_paths},
        "status_values": status_values,
    }


def validate_registry(registry: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    assets = [asset for asset in registry.get("assets", []) if asset.get("layer") == "L7_trading_delivery"]
    if len(assets) != len(TABLES):
        raise RuntimeError(f"expected exactly three L7 registry assets, got {len(assets)}")
    by_table: dict[str, dict[str, Any]] = {}
    for asset in assets:
        raw_path = str(asset.get("asset_path") or "")
        if "::" not in raw_path:
            raise RuntimeError(f"L7 registry asset is not a DuckDB table route: {raw_path}")
        file_part, table = raw_path.rsplit("::", 1)
        path = (root / file_part).resolve()
        expected_path = (
            root
            / "quant"
            / "data_file"
            / "production_assets"
            / "duckdb"
            / "production"
            / "l7"
            / f"{table}.duckdb"
        ).resolve()
        if table not in TABLES or path != expected_path:
            raise RuntimeError(f"unexpected L7 registry route: {raw_path}")
        if asset.get("asset_type") != "duckdb_table" or asset.get("owner_agent") != "trading-agent":
            raise RuntimeError(f"unexpected L7 registry contract for {table}")
        by_table[table] = asset
    if set(by_table) != set(TABLES):
        raise RuntimeError(f"L7 registry table set mismatch: {sorted(by_table)}")
    return [by_table[table] for table in TABLES]


def validate_before(
    snapshots: dict[str, dict[str, Any]], expected_strategy_id: str
) -> None:
    for table, snapshot in snapshots.items():
        if snapshot["tables"] != [table]:
            raise RuntimeError(f"unexpected table set in {table}: {snapshot['tables']}")
        expected_rows = EXPECTED_ROW_COUNTS[table]
        if snapshot["row_count"] != expected_rows:
            raise RuntimeError(
                f"baseline drift in {table}: expected {expected_rows}, got {snapshot['row_count']}"
            )
        if snapshot["strategy_ids"] != [expected_strategy_id]:
            raise RuntimeError(f"unexpected strategy IDs in {table}: {snapshot['strategy_ids']}")
        if any(snapshot["source_path_exists"].values()):
            raise RuntimeError(f"an L7 current source still exists for {table}; stop for manual review")
    statuses = snapshots["prod_l7_signal_status_current"]["status_values"]
    if statuses != ["pending_buy_day_hard_gate"]:
        raise RuntimeError(f"unexpected L7 current status baseline: {statuses}")


def validate_after(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]], expected_strategy_id: str
) -> None:
    for table in TABLES:
        if after[table]["row_count"] != 0:
            raise RuntimeError(f"L7 current cleanup failed for {table}")
        if after[table]["schema"] != before[table]["schema"]:
            raise RuntimeError(f"schema changed during L7 cleanup for {table}")
        if after[table]["tables"] != [table]:
            raise RuntimeError(f"table identity changed during L7 cleanup for {table}")
        if expected_strategy_id in after[table]["strategy_ids"]:
            raise RuntimeError(f"withdrawn strategy remains in {table}")
        if after[table]["source_paths"]:
            raise RuntimeError(f"source path residue remains in {table}")
    if after["prod_l7_signal_status_current"]["status_values"]:
        raise RuntimeError("L7 current status residue remains after cleanup")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# L7 已撤销策略 current 资产受控清理报告",
        "",
        "## 结论",
        "",
        f"- 状态：`{payload['status']}`",
        "- 本次仅清空 L7 current 三表数据，保留 schema、表名、一表一文件和 registry 路径。",
        "- 未生成新信号、未生成新交付包、未下单、未推进 L8。",
        f"- `ready_for_audit_review={str(payload['ready_for_audit_review']).lower()}`",
        f"- `allow_next_layer_continue={str(payload['allow_next_layer_continue']).lower()}`",
        "",
        "## 清理结果",
        "",
        "| 表 | 清理前行数 | 清理后行数 | 清理前 SHA256 | 清理后 SHA256 | schema 未变 |",
        "|---|---:|---:|---|---|---|",
    ]
    for table in TABLES:
        before = payload["before"]["tables"][table]
        after = payload["after"]["tables"][table]
        lines.append(
            f"| `{table}` | {before['row_count']} | {after['row_count']} | "
            f"`{before['sha256']}` | `{after['sha256']}` | "
            f"{str(before['schema'] == after['schema']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Registry 与回滚",
            "",
            f"- registry 清理前 SHA256：`{payload['before']['registry_sha256']}`",
            f"- registry 清理后 SHA256：`{payload['after']['registry_sha256']}`",
            f"- registry 未修改：`{str(payload['registry_unchanged']).lower()}`",
            f"- rollback 目录：`{payload['rollback_dir']}`",
            "- rollback 中三张 DuckDB 与 registry 均已逐文件校验，哈希与清理前一致。",
            "",
            "## 硬边界",
            "",
            "- 历史策略归档和项目回收站证据保持只读未删除。",
            "- 当前不存在 `production.current`，L7 生成与手工交付入口必须继续 fail-closed。",
            "- L8 继续冻结，等待审计智能体只读复核。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="受控清空已撤销策略遗留的 L7 current 三表。")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-strategy-id", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rollback_dir = output_dir / "rollback"
    l7_dir = root / "quant" / "data_file" / "production_assets" / "duckdb" / "production" / "l7"
    registry_path = root / "quant" / "data_file" / "asset_registry" / "production_assets.json"
    strategy_registry_path = root / "quant" / "main" / "strategy_library" / "registry.json"
    withdrawal_report = (
        root
        / "quant"
        / "data_file"
        / "reports"
        / "strategy_agent_prod_high_return_frs_withdrawal_20260721"
        / "withdrawal_execution_report.json"
    )

    strategy_registry = load_json(strategy_registry_path)
    production = strategy_registry.get("production", {})
    if str(production.get("current") or "").strip():
        raise RuntimeError("production.current is not empty; cleanup is not allowed")
    if production.get("state") != "no_available_production_strategy":
        raise RuntimeError(f"unexpected production state: {production.get('state')}")
    if production.get("strategies"):
        raise RuntimeError("production strategy list is not empty")
    if not withdrawal_report.is_file():
        raise FileNotFoundError(f"withdrawal evidence is missing: {withdrawal_report}")
    withdrawal = load_json(withdrawal_report)
    if (
        withdrawal.get("user_choice") != "B_withdrawal"
        or withdrawal.get("strategy_id") != args.expected_strategy_id
        or not withdrawal.get("ready_for_audit_review")
    ):
        raise RuntimeError("withdrawal evidence does not authorize this exact cleanup")

    registry = load_json(registry_path)
    registry_assets = validate_registry(registry, root)
    table_paths = {table: l7_dir / f"{table}.duckdb" for table in TABLES}
    before_tables = {table: table_snapshot(path, table) for table, path in table_paths.items()}
    validate_before(before_tables, args.expected_strategy_id)
    registry_before_hash = sha256(registry_path)
    before = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "registry_path": str(registry_path),
        "registry_sha256": registry_before_hash,
        "registry_assets": registry_assets,
        "tables": before_tables,
    }
    write_json(output_dir / "before_cleanup_baseline.json", before)

    if not args.execute:
        print(json.dumps({"status": "dry_run_passed", "before": before}, ensure_ascii=False, indent=2))
        return 0

    if rollback_dir.exists():
        raise FileExistsError(f"rollback directory already exists: {rollback_dir}")
    rollback_dir.mkdir(parents=True)
    rollback_hashes: dict[str, str] = {}
    for table, source in table_paths.items():
        destination = rollback_dir / source.name
        shutil.copy2(source, destination)
        if sha256(destination) != before_tables[table]["sha256"]:
            raise RuntimeError(f"rollback hash mismatch for {table}")
        rollback_hashes[destination.name] = sha256(destination)
    registry_snapshot = rollback_dir / registry_path.name
    shutil.copy2(registry_path, registry_snapshot)
    if sha256(registry_snapshot) != registry_before_hash:
        raise RuntimeError("registry rollback hash mismatch")
    rollback_hashes[registry_snapshot.name] = sha256(registry_snapshot)
    write_json(rollback_dir / "rollback_manifest.json", {"files": rollback_hashes, "before": before})

    try:
        for table, path in table_paths.items():
            with duckdb.connect(str(path)) as connection:
                connection.execute("BEGIN TRANSACTION")
                connection.execute(f'DELETE FROM "{table}"')
                connection.execute("COMMIT")
                connection.execute("CHECKPOINT")
        after_tables = {table: table_snapshot(path, table) for table, path in table_paths.items()}
        validate_after(before_tables, after_tables, args.expected_strategy_id)
        registry_after_hash = sha256(registry_path)
        if registry_after_hash != registry_before_hash:
            raise RuntimeError("production asset registry changed during L7 cleanup")
    except Exception:
        for table, destination in table_paths.items():
            shutil.copy2(rollback_dir / destination.name, destination)
        if sha256(registry_path) != registry_before_hash:
            shutil.copy2(registry_snapshot, registry_path)
        restored = {table: table_snapshot(path, table) for table, path in table_paths.items()}
        for table in TABLES:
            if restored[table]["sha256"] != before_tables[table]["sha256"]:
                raise RuntimeError(f"cleanup failed and rollback verification failed for {table}")
        raise

    after = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "registry_path": str(registry_path),
        "registry_sha256": registry_after_hash,
        "registry_assets": validate_registry(load_json(registry_path), root),
        "tables": after_tables,
    }
    payload = {
        "schema_version": 1,
        "task_id": "incremental-trading-signal-20260720-L7-withdrawn-strategy-current-cleanup",
        "actor": "trading-agent",
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "cleanup_completed_waiting_for_audit",
        "expected_strategy_id": args.expected_strategy_id,
        "before": before,
        "after": after,
        "registry_unchanged": registry_before_hash == registry_after_hash,
        "rollback_dir": str(rollback_dir),
        "rollback_hashes": rollback_hashes,
        "withdrawal_evidence": str(withdrawal_report),
        "historical_archive_preserved": True,
        "new_signal_generated": False,
        "new_delivery_package_generated": False,
        "trade_triggered": False,
        "l8_modified": False,
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    write_json(output_dir / "cleanup_result.json", payload)
    (output_dir / "L7已撤销策略current资产受控清理报告.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
