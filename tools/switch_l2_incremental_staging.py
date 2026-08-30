from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import duckdb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(path: Path, trade_date: str) -> dict[str, object]:
    with duckdb.connect(str(path), read_only=True) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT stock_code), MIN(trade_date), MAX(trade_date),
                   COUNT(*) FILTER (WHERE stock_code LIKE '%.BJ')
            FROM STOCK_DAILY_DATA
            """
        ).fetchone()
        target = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT stock_code)
            FROM STOCK_DAILY_DATA WHERE trade_date = ?
            """,
            [trade_date],
        ).fetchone()
        duplicates = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT stock_code, trade_date FROM STOCK_DAILY_DATA
                GROUP BY 1, 2 HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    return {
        "row_count": int(row[0]),
        "stock_count": int(row[1]),
        "min_trade_date": str(row[2]),
        "max_trade_date": str(row[3]),
        "bj_rows": int(row[4]),
        "target_rows": int(target[0]),
        "target_stocks": int(target[1]),
        "duplicate_key_groups": int(duplicates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", required=True, type=Path)
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--validation-json", required=True, type=Path)
    parser.add_argument("--target-trade-date", required=True)
    parser.add_argument("--expected-active-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    active = args.active.resolve()
    staging = args.staging.resolve()
    snapshot = args.snapshot.resolve()
    if len({active, staging, snapshot}) != 3:
        raise RuntimeError("active, staging and snapshot paths must be distinct")
    validation = json.loads(args.validation_json.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise RuntimeError("staging validation did not pass")

    before_stat = active.stat()
    active_hash = sha256(active)
    staging_hash = sha256(staging)
    expected = args.expected_active_sha256.lower()
    if active_hash != expected:
        raise RuntimeError("active no longer matches approved baseline")
    if snapshot.exists():
        raise RuntimeError("snapshot path already exists; a fresh independent path is required")

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot_temp = snapshot.with_name(snapshot.name + ".copying")
    if snapshot_temp.exists():
        raise RuntimeError("snapshot temporary path already exists")
    with duckdb.connect(str(active)) as conn:
        conn.execute("SELECT 1").fetchone()
        conn.execute("CHECKPOINT")
    shutil.copy2(active, snapshot_temp)
    os.replace(snapshot_temp, snapshot)

    snapshot_hash = sha256(snapshot)
    if snapshot_hash != expected:
        raise RuntimeError("independent snapshot does not match approved baseline")
    metrics(snapshot, args.target_trade_date)

    # Close the DuckDB handle before Windows os.replace, then recheck the
    # active fingerprint immediately before the atomic switch.
    if sha256(active) != expected:
        raise RuntimeError("active baseline drifted after snapshot creation")
    with duckdb.connect(str(active)) as conn:
        conn.execute("SELECT 1").fetchone()
    before_metrics = metrics(active, args.target_trade_date)
    staging_metrics = metrics(staging, args.target_trade_date)
    if staging_metrics["max_trade_date"] != args.target_trade_date:
        raise RuntimeError("staging max trade date mismatch")
    if staging_metrics["bj_rows"] or staging_metrics["duplicate_key_groups"]:
        raise RuntimeError("staging no-BJ or duplicate gate failed")

    switch_started_at = datetime.now().astimezone().isoformat()
    os.replace(staging, active)
    switch_completed_at = datetime.now().astimezone().isoformat()
    after_hash = sha256(active)
    if after_hash != staging_hash:
        raise RuntimeError("active hash does not match validated staging after replace")
    after_metrics = metrics(active, args.target_trade_date)

    result = {
        "status": "completed_waiting_for_audit",
        "operation": "atomic_incremental_staging_to_active_replace",
        "target_trade_date": args.target_trade_date,
        "active_path": str(active),
        "staging_path_before_replace": str(staging),
        "staging_exists_after_replace": staging.exists(),
        "snapshot_path": str(snapshot),
        "switch_started_at": switch_started_at,
        "switch_completed_at": switch_completed_at,
        "before": {
            "sha256": active_hash,
            "size_bytes": before_stat.st_size,
            "mtime_ns": before_stat.st_mtime_ns,
            "metrics": before_metrics,
        },
        "snapshot": {"sha256": snapshot_hash, "size_bytes": snapshot.stat().st_size},
        "staging": {"sha256": staging_hash, "metrics": staging_metrics},
        "after": {"sha256": after_hash, "metrics": after_metrics},
        "active_written": True,
        "legacy_input_used": False,
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
