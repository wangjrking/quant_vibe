from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "quant/data_file"
SOURCE_ROOT = DATA / "experimental_assets/tushare_stock_risk_events_v1"
WORK_ROOT = (
    DATA
    / "runtime/agent_workspaces/data-ingestion-agent/work"
    / "tushare_stock_risk_events_production_promotion_20260822_r1"
)
ACTIVE_DB_ROOT = DATA / "production_assets/duckdb/l1_raw_tables"
REPORT_ROOT = DATA / "reports"

SPECS = {
    "stk_shock": {
        "source": "l1_stk_shock.duckdb",
        "date": "trade_date",
        "keys": ("ts_code", "trade_date", "reason", "period"),
    },
    "stk_high_shock": {
        "source": "l1_stk_high_shock.duckdb",
        "date": "trade_date",
        "keys": ("ts_code", "trade_date", "reason", "period"),
    },
    "stk_alert": {
        "source": "l1_stk_alert.duckdb",
        "date": "start_date",
        "keys": ("ts_code", "start_date", "end_date", "type"),
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_single_table(path: Path, table: str) -> pd.DataFrame:
    with duckdb.connect(str(path), read_only=True) as connection:
        tables = [row[0] for row in connection.execute("SHOW TABLES").fetchall()]
        if tables != [table]:
            raise RuntimeError(f"{path} must contain exactly {table}, got {tables}")
        return connection.execute(f'SELECT * FROM "{table}"').fetchdf()


def canonicalize(frame: pd.DataFrame, keys: tuple[str, ...], date_column: str) -> pd.DataFrame:
    missing = sorted(({"ts_code", date_column} | set(keys)) - set(frame.columns))
    if missing:
        raise RuntimeError(f"candidate schema missing columns: {missing}")
    value = frame.copy()
    value["ts_code"] = value["ts_code"].astype("string").str.strip().str.upper()
    value = value[~value["ts_code"].str.endswith(".BJ", na=False)].copy()
    value = value.drop_duplicates(subset=list(keys), keep="last")
    value = value.sort_values([date_column, "ts_code", *[key for key in keys if key not in {"ts_code", date_column}]], na_position="last")
    return value.reset_index(drop=True)


def quality(frame: pd.DataFrame, keys: tuple[str, ...], date_column: str) -> dict[str, Any]:
    descriptive_columns = [key for key in keys if key not in {"ts_code", date_column}]
    return {
        "rows": int(len(frame)),
        "stocks": int(frame["ts_code"].nunique()),
        "min_date": None if frame.empty else str(frame[date_column].min()),
        "max_date": None if frame.empty else str(frame[date_column].max()),
        "duplicate_key_groups": int(frame.duplicated(list(keys), keep=False).sum()),
        "bj_rows": int(frame["ts_code"].str.endswith(".BJ", na=False).sum()),
        "required_key_nulls": int(frame[["ts_code", date_column]].isna().sum().sum()),
        "descriptive_key_nulls": int(frame[descriptive_columns].isna().sum().sum()),
    }


def write_duckdb(path: Path, table: str, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with duckdb.connect(str(temporary)) as connection:
        connection.register("payload", frame)
        connection.execute(f'CREATE TABLE "{table}" AS SELECT * FROM payload')
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def build_candidate() -> dict[str, Any]:
    candidate_root = WORK_ROOT / "candidate"
    result: dict[str, Any] = {
        "schema_version": 1,
        "task_id": "tushare-stock-risk-events-l1-production-promotion-20260822",
        "status": "candidate_ready_for_audit",
        "created_at": now_iso(),
        "source": "Tushare Pro official SDK candidate asset",
        "scope": list(SPECS),
        "production_modified": False,
        "l2_modified": False,
        "tables": {},
    }
    for table, spec in SPECS.items():
        source_path = SOURCE_ROOT / str(spec["source"])
        frame = canonicalize(
            read_single_table(source_path, table),
            spec["keys"],
            str(spec["date"]),
        )
        parquet_path = candidate_root / "raw" / f"{table}.parquet"
        db_path = candidate_root / "duckdb" / f"{table}.duckdb"
        write_parquet(parquet_path, frame)
        write_duckdb(db_path, table, frame)
        parquet_frame = pd.read_parquet(parquet_path)
        db_frame = read_single_table(db_path, table)
        expected = quality(frame, spec["keys"], str(spec["date"]))
        parquet_quality = quality(parquet_frame, spec["keys"], str(spec["date"]))
        duckdb_quality = quality(db_frame, spec["keys"], str(spec["date"]))
        if expected != parquet_quality or expected != duckdb_quality:
            raise RuntimeError(f"source/parquet/DuckDB mismatch for {table}")
        if expected["duplicate_key_groups"] or expected["bj_rows"] or expected["required_key_nulls"]:
            raise RuntimeError(f"quality gate failed for {table}: {expected}")
        result["tables"][table] = {
            "natural_key": list(spec["keys"]),
            "date_column": spec["date"],
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "source_after_no_bj": expected,
            "candidate_parquet": str(parquet_path),
            "candidate_parquet_sha256": sha256_file(parquet_path),
            "candidate_duckdb": str(db_path),
            "candidate_duckdb_sha256": sha256_file(db_path),
            "candidate_duckdb_tables": [table],
        }
    result["ready_for_audit_review"] = True
    result["allow_production_cutover"] = False
    report_path = WORK_ROOT / "candidate_report.json"
    atomic_json(report_path, result)
    atomic_json(REPORT_ROOT / "l1_tushare_stock_risk_events_production_candidate_20260822.json", result)
    return result


def publish(approval_path: Path) -> dict[str, Any]:
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("audit_passed") is not True or approval.get("allow_production_cutover") is not True:
        raise RuntimeError("audit approval must set audit_passed and allow_production_cutover true")
    report_path = WORK_ROOT / "candidate_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("ready_for_audit_review") is not True:
        raise RuntimeError("candidate report is not ready")
    backup_root = WORK_ROOT / "rollback" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root.mkdir(parents=True, exist_ok=False)
    rollback_assets: list[dict[str, Any]] = []
    published: dict[str, Any] = {}
    for table, item in report["tables"].items():
        candidate_parquet = Path(item["candidate_parquet"])
        candidate_db = Path(item["candidate_duckdb"])
        if sha256_file(candidate_parquet) != item["candidate_parquet_sha256"]:
            raise RuntimeError(f"candidate parquet fingerprint drift: {table}")
        if sha256_file(candidate_db) != item["candidate_duckdb_sha256"]:
            raise RuntimeError(f"candidate DuckDB fingerprint drift: {table}")
        active_parquet = DATA / f"{table}.parquet"
        active_db = ACTIVE_DB_ROOT / f"{table}.duckdb"
        for active in (active_parquet, active_db):
            if active.exists():
                backup = backup_root / active.relative_to(DATA)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(active, backup)
                rollback_assets.append(
                    {
                        "active_path": str(active),
                        "pre_publish_state": "present",
                        "backup_path": str(backup),
                        "backup_sha256": sha256_file(backup),
                        "rollback_action": "atomic_restore_backup",
                    }
                )
            else:
                rollback_assets.append(
                    {
                        "active_path": str(active),
                        "pre_publish_state": "absent",
                        "backup_path": None,
                        "backup_sha256": None,
                        "rollback_action": "remove_newly_published_asset",
                    }
                )
        active_parquet.parent.mkdir(parents=True, exist_ok=True)
        active_db.parent.mkdir(parents=True, exist_ok=True)
        parquet_tmp = active_parquet.with_name(f".{active_parquet.name}.{uuid.uuid4().hex}.tmp")
        db_tmp = active_db.with_name(f".{active_db.name}.{uuid.uuid4().hex}.tmp")
        shutil.copy2(candidate_parquet, parquet_tmp)
        shutil.copy2(candidate_db, db_tmp)
        os.replace(parquet_tmp, active_parquet)
        os.replace(db_tmp, active_db)
        published[table] = {
            "parquet": str(active_parquet),
            "parquet_sha256": sha256_file(active_parquet),
            "duckdb": str(active_db),
            "duckdb_sha256": sha256_file(active_db),
            "quality": item["source_after_no_bj"],
        }
    rollback_manifest = {
        "schema_version": 1,
        "task_id": report["task_id"],
        "created_at": now_iso(),
        "assets": rollback_assets,
        "rollback_order": "restore or remove DuckDB and parquet assets, then run --verify-active only after a subsequent approved publish",
    }
    atomic_json(backup_root / "rollback_manifest.json", rollback_manifest)
    result = {
        "schema_version": 1,
        "task_id": report["task_id"],
        "status": "production_l1_published",
        "published_at": now_iso(),
        "audit_approval": str(approval_path),
        "tables": published,
        "rollback_root": str(backup_root),
        "rollback_manifest": str(backup_root / "rollback_manifest.json"),
        "production_modified": True,
        "l2_modified": False,
        "allow_next_layer_continue": False,
    }
    atomic_json(REPORT_ROOT / "l1_tushare_stock_risk_events_production_publish_20260822.json", result)
    return result


def verify_active() -> dict[str, Any]:
    publish_report_path = REPORT_ROOT / "l1_tushare_stock_risk_events_production_publish_20260822.json"
    publish_report = json.loads(publish_report_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "schema_version": 1,
        "task_id": publish_report["task_id"],
        "status": "active_verified",
        "verified_at": now_iso(),
        "tables": {},
        "production_route": str(ACTIVE_DB_ROOT),
        "duckdb_only": True,
        "one_table_one_file": True,
        "l2_modified": False,
    }
    rollback_root = Path(publish_report["rollback_root"])
    rollback_manifest = rollback_root / "rollback_manifest.json"
    if not rollback_manifest.is_file():
        raise RuntimeError(f"rollback manifest is missing: {rollback_manifest}")
    rollback_payload = json.loads(rollback_manifest.read_text(encoding="utf-8"))
    if len(rollback_payload.get("assets", [])) != len(SPECS) * 2:
        raise RuntimeError("rollback manifest must cover parquet and DuckDB for every table")
    result["rollback"] = {
        "root": str(rollback_root),
        "manifest": str(rollback_manifest),
        "manifest_sha256": sha256_file(rollback_manifest),
        "asset_count": len(rollback_payload["assets"]),
        "valid": True,
    }
    for table, spec in SPECS.items():
        active_parquet = DATA / f"{table}.parquet"
        active_db = ACTIVE_DB_ROOT / f"{table}.duckdb"
        parquet_frame = pd.read_parquet(active_parquet)
        db_frame = read_single_table(active_db, table)
        parquet_quality = quality(parquet_frame, spec["keys"], str(spec["date"]))
        db_quality = quality(db_frame, spec["keys"], str(spec["date"]))
        expected = publish_report["tables"][table]["quality"]
        if parquet_quality != expected or db_quality != expected:
            raise RuntimeError(f"published quality mismatch for {table}")
        if sha256_file(active_parquet) != publish_report["tables"][table]["parquet_sha256"]:
            raise RuntimeError(f"published parquet fingerprint drift: {table}")
        if sha256_file(active_db) != publish_report["tables"][table]["duckdb_sha256"]:
            raise RuntimeError(f"published DuckDB fingerprint drift: {table}")
        result["tables"][table] = {
            "parquet": str(active_parquet),
            "duckdb": str(active_db),
            "quality": expected,
            "parquet_sha256": sha256_file(active_parquet),
            "duckdb_sha256": sha256_file(active_db),
            "duckdb_tables": [table],
        }
    result["passed"] = True
    result["ready_for_post_publish_audit"] = True
    result["allow_next_layer_continue"] = False
    atomic_json(REPORT_ROOT / "l1_tushare_stock_risk_events_production_validation_20260822.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--verify-active", action="store_true")
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    if args.verify_active:
        result = verify_active()
    elif args.publish:
        if args.approval is None:
            raise SystemExit("--approval is required with --publish")
        result = publish(args.approval)
    else:
        result = build_candidate()
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
