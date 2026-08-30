import json
from pathlib import Path

import duckdb

from tools.check_workflow_asset_alignment import check_alignment


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "config" / "workflow_asset_contract_v1_20260830.json").read_text(encoding="utf-8"))


def _write_table(path: Path, table: str, dates: list[str]) -> None:
    connection = duckdb.connect(str(path))
    connection.execute(f'CREATE TABLE "{table}" (trade_date VARCHAR, stock_code VARCHAR)')
    connection.executemany(f'INSERT INTO "{table}" VALUES (?, ?)', [(date, "000001.SZ") for date in dates])
    connection.close()


def test_alignment_accepts_matching_incremental_routes(tmp_path: Path) -> None:
    root = tmp_path / "production_assets" / "duckdb"
    l1 = root / "l1_raw_tables"
    l1.mkdir(parents=True)
    for table in CONTRACT["incremental"]["target_date_tables"]:
        _write_table(l1 / f"{table}.duckdb", table, ["20260828"])
    for table in CONTRACT["incremental"]["risk_event_companion_tables"]:
        _write_table(l1 / f"{table}.duckdb", table, ["20260828"])
    _write_table(root / "l2_stock_daily_data.duckdb", "STOCK_DAILY_DATA", ["20260828"])
    _write_table(root / "l3_feature_current.duckdb", "features", ["20260828"])
    _write_table(root / "l3_label_current.duckdb", "labels", ["20260827"])

    result = check_alignment(tmp_path, CONTRACT)

    assert result["valid"] is True
    assert result["observations"]["l2_max_trade_date"] == "20260828"


def test_alignment_rejects_target_date_drift(tmp_path: Path) -> None:
    root = tmp_path / "production_assets" / "duckdb"
    l1 = root / "l1_raw_tables"
    l1.mkdir(parents=True)
    for table in CONTRACT["incremental"]["target_date_tables"]:
        _write_table(l1 / f"{table}.duckdb", table, ["20260827"])
    for table in CONTRACT["incremental"]["risk_event_companion_tables"]:
        _write_table(l1 / f"{table}.duckdb", table, ["20260828"])
    _write_table(root / "l2_stock_daily_data.duckdb", "STOCK_DAILY_DATA", ["20260828"])
    _write_table(root / "l3_feature_current.duckdb", "features", ["20260828"])
    _write_table(root / "l3_label_current.duckdb", "labels", ["20260827"])

    result = check_alignment(tmp_path, CONTRACT)

    assert result["valid"] is False
    assert any("does not match L2 max" in error for error in result["errors"])
