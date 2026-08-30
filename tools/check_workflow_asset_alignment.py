"""Validate that local active data routes match the declared workflow contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


MAIN_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_DIR.parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "quant" / "data_file"
CONTRACT_PATH = MAIN_DIR / "config" / "workflow_asset_contract_v1_20260830.json"


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _single_table(connection: duckdb.DuckDBPyConnection) -> str:
    tables = [row[0] for row in connection.execute("SHOW TABLES").fetchall()]
    if len(tables) != 1:
        raise ValueError(f"expected one business table, found {tables}")
    return tables[0]


def _date_bounds(path: Path, table: str | None = None) -> tuple[int, str | None, str | None]:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        table = table or _single_table(connection)
        columns = {row[0] for row in connection.execute(f"DESCRIBE {_quote(table)}").fetchall()}
        if "trade_date" not in columns:
            return int(connection.execute(f"SELECT count(*) FROM {_quote(table)}").fetchone()[0]), None, None
        count, minimum, maximum = connection.execute(
            f"SELECT count(*), min(trade_date), max(trade_date) FROM {_quote(table)}"
        ).fetchone()
        return int(count), str(minimum), str(maximum)
    finally:
        connection.close()


def check_alignment(data_dir: Path, contract: dict[str, Any]) -> dict[str, Any]:
    duckdb_root = data_dir / "production_assets" / "duckdb"
    l1_root = duckdb_root / "l1_raw_tables"
    errors: list[str] = []
    observations: dict[str, Any] = {}

    l2_path = duckdb_root / "l2_stock_daily_data.duckdb"
    l3_feature_path = duckdb_root / "l3_feature_current.duckdb"
    l3_label_path = duckdb_root / "l3_label_current.duckdb"
    for path in (l2_path, l3_feature_path, l3_label_path):
        if not path.exists():
            errors.append(f"missing active asset: {path}")
    if errors:
        return {"valid": False, "errors": errors, "observations": observations}

    _, _, l2_max = _date_bounds(l2_path, "STOCK_DAILY_DATA")
    _, _, feature_max = _date_bounds(l3_feature_path)
    _, _, label_max = _date_bounds(l3_label_path)
    observations["l2_max_trade_date"] = l2_max
    observations["l3_feature_max_trade_date"] = feature_max
    observations["l3_label_mature_max_trade_date"] = label_max
    if feature_max != l2_max:
        errors.append(f"L3 feature max {feature_max} does not match L2 max {l2_max}")
    if label_max and l2_max and label_max > l2_max:
        errors.append(f"L3 label max {label_max} exceeds L2 max {l2_max}")

    required_tables = contract["incremental"]["target_date_tables"]
    table_bounds: dict[str, Any] = {}
    for table in required_tables:
        path = l1_root / f"{table}.duckdb"
        if not path.exists():
            errors.append(f"missing required L1 table: {table}")
            continue
        count, minimum, maximum = _date_bounds(path, table)
        table_bounds[table] = {"rows": count, "min_trade_date": minimum, "max_trade_date": maximum}
        if maximum != l2_max:
            errors.append(f"L1 {table} max {maximum} does not match L2 max {l2_max}")
    observations["incremental_l1_tables"] = table_bounds

    for table in contract["incremental"]["risk_event_companion_tables"]:
        path = l1_root / f"{table}.duckdb"
        if not path.exists():
            errors.append(f"missing risk-event companion table: {table}")
    observations["full_history_note"] = (
        "Current active L1 auxiliary tables may be rolling windows. Full-history initialization requires a separate "
        "official-source candidate and cannot infer historical completeness from this active incremental snapshot."
    )
    return {"valid": not errors, "errors": errors, "observations": observations}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = check_alignment(args.data_dir, contract)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
