from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
import sys

if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from duckdb_asset_route import normalize_duckdb_zone, require_duckdb_installed, resolve_duckdb_zone_path


CN_TZ = timezone(timedelta(hours=8))


def initialize_duckdb_zone(
    zone: str,
    *,
    data_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    require_duckdb_installed()
    import duckdb

    normalized = normalize_duckdb_zone(zone)
    db_path = resolve_duckdb_zone_path(normalized, data_dir=data_dir, require_exists=False)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not overwrite:
        action = "existing"
    else:
        action = "created" if not db_path.exists() else "overwritten"

    created_at = datetime.now(CN_TZ).isoformat(timespec="seconds")
    with duckdb.connect(str(db_path)) as conn:
        if overwrite:
            conn.execute("DROP TABLE IF EXISTS duckdb_zone_metadata")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS duckdb_zone_metadata (
                zone VARCHAR,
                db_file VARCHAR,
                created_at VARCHAR,
                purpose VARCHAR
            )
            """
        )
        if overwrite:
            conn.execute("DELETE FROM duckdb_zone_metadata")
        if conn.execute("SELECT COUNT(*) FROM duckdb_zone_metadata").fetchone()[0] == 0:
            conn.execute(
                """
                INSERT INTO duckdb_zone_metadata(zone, db_file, created_at, purpose)
                VALUES (?, ?, ?, ?)
                """,
                [
                    normalized,
                    db_path.name,
                    created_at,
                    f"{normalized}_duckdb_zone",
                ],
            )
        row_count = int(conn.execute("SELECT COUNT(*) FROM duckdb_zone_metadata").fetchone()[0])

    return {
        "status": "ok",
        "zone": normalized,
        "db_path": str(db_path),
        "action": action,
        "metadata_table": "duckdb_zone_metadata",
        "metadata_rows": row_count,
        "created_at": created_at,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Initialize a named DuckDB zone file and metadata table.")
    parser.add_argument("--zone", required=True, choices=["production", "experiment", "archive"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = initialize_duckdb_zone(
        args.zone,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
