from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from l5_duckdb_sync import sync_strategy_registry_to_duckdb
from l6_duckdb_sync import sync_strategy_backtests_to_duckdb
from l7_duckdb_sync import sync_production_signal_artifacts_to_duckdb
from strategy_asset_route import (
    load_current_production_strategy_context,
    load_strategy_validation_payload,
    resolve_strategy_signal_duckdb_path,
    resolve_strategy_validation_duckdb_path,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_split_production_assets(data_dir: Path) -> None:
    adjustment_semantics = {
        "explicit_front_adjusted_marker_required": True,
        "front_adjusted_marker": "qfq",
        "front_adjusted_column_suffix": "_qfq",
        "front_adjusted_indicator_suffix": "_qfq",
        "front_adjusted_factor_marker_required": True,
        "forbidden_implicit_front_adjusted_columns": ["open", "high", "low", "close", "pre_close"],
        "contract_rule": "temp split validation",
    }
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "temp_l5_audit.md",
        "temp_l6_audit.md",
        "temp_l7_audit.md",
        "temp_l5_manifest.json",
        "temp_l6_manifest.json",
        "temp_l7_manifest.json",
    ]:
        path = reports_dir / name
        path.write_text("{}\n" if path.suffix == ".json" else "temp\n", encoding="utf-8")

    _write_json(
        data_dir / "asset_registry" / "production_assets.json",
        {
            "assets": [
                {
                    "asset_id": "prod_l5_split_temp",
                    "track": "production",
                    "layer": "L5_strategy_signal",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(
                        data_dir
                        / "production_assets"
                        / "duckdb"
                        / "production"
                        / "l5"
                        / "prod_l5_strategy_registry_current.duckdb"
                    )
                    + "::prod_l5_strategy_registry_current",
                    "audit_record": str(reports_dir / "temp_l5_audit.md"),
                    "migration_source_asset": "legacy_temp_l5",
                    "duckdb_migration_manifest": str(reports_dir / "temp_l5_manifest.json"),
                    "adjustment_semantics": adjustment_semantics,
                },
                {
                    "asset_id": "prod_l6_split_temp",
                    "track": "production",
                    "layer": "L6_backtest",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(
                        data_dir
                        / "production_assets"
                        / "duckdb"
                        / "production"
                        / "l6"
                        / "prod_l6_strategy_validation_current.duckdb"
                    )
                    + "::prod_l6_strategy_validation_current",
                    "audit_record": str(reports_dir / "temp_l6_audit.md"),
                    "migration_source_asset": "legacy_temp_l6",
                    "duckdb_migration_manifest": str(reports_dir / "temp_l6_manifest.json"),
                    "adjustment_semantics": adjustment_semantics,
                },
                {
                    "asset_id": "prod_l7_split_temp",
                    "track": "production",
                    "layer": "L7_trading_delivery",
                    "asset_type": "duckdb_table",
                    "status": "production_active",
                    "allowed_for_main_workflow": True,
                    "asset_path": str(
                        data_dir
                        / "production_assets"
                        / "duckdb"
                        / "production"
                        / "l7"
                        / "prod_l7_signal_rows_current.duckdb"
                    )
                    + "::prod_l7_signal_rows_current",
                    "audit_record": str(reports_dir / "temp_l7_audit.md"),
                    "migration_source_asset": "legacy_temp_l7",
                    "duckdb_migration_manifest": str(reports_dir / "temp_l7_manifest.json"),
                    "adjustment_semantics": adjustment_semantics,
                },
            ]
        },
    )


def _seed_initial_project(root: Path) -> tuple[Path, Path, Path]:
    project_dir = root / "quant" / "main"
    data_dir = root / "quant" / "data_file"
    strategy_dir = project_dir / "strategy_library" / "production" / "prod_a"
    signal_dir = data_dir / "production_signals"
    backtest_dir = strategy_dir / "backtests"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    signal_dir.mkdir(parents=True, exist_ok=True)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        project_dir / "strategy_library" / "registry.json",
        {
            "production": {
                "current": "prod_a",
                "strategies": [
                    {
                        "strategy_id": "prod_a",
                        "name": "Prod A",
                        "status": "production",
                        "path": "strategy_library/production/prod_a",
                        "published_at": "2026-06-28T09:00:00+08:00",
                    }
                ],
            }
        },
    )
    _write_json(
        strategy_dir / "strategy_manifest.json",
        {
            "strategy_id": "prod_a",
            "name": "Prod A",
            "status": "production",
            "production_version": "v1",
            "published_at": "2026-06-28T09:00:00+08:00",
            "current_signal": {
                "latest_file": "prod_a_latest.csv",
                "latest_signal_date": "20260628",
                "latest_buy_date": "20260630",
            },
            "input_contract": {"formal_manifest": "manifest_a.json"},
            "validation": {"annual_return": 0.11, "sharpe": 1.21, "max_drawdown": -0.08},
        },
    )
    _write_json(
        strategy_dir / "validation.json",
        {
            "status": "production",
            "validation_platform": "gm",
            "metrics": {
                "annual_return": 0.11,
                "pnl_ratio": 1.5,
                "sharpe": 1.21,
                "max_drawdown": -0.08,
                "avg_invested_pct": 0.92,
                "max_active_positions": 1,
            },
            "latest_signal_status": {
                "signal_date": "20260628",
                "buy_date": "20260630",
                "buy_day_hard_gate_complete": True,
                "selected_stock_code": "000001.SZ",
            },
            "hard_gate_audit": {
                "failed_files": 0,
                "signal_rows": 1,
            },
        },
    )
    _write_json(
        backtest_dir / "summary.json",
        {"status": "ok", "annual": 0.11},
    )
    _write_csv(
        backtest_dir / "time_slices.csv",
        ["trade_date", "annual", "sharpe", "max_drawdown", "signal_file", "log_file"],
        [
            {
                "trade_date": "20260628",
                "annual": "0.11",
                "sharpe": "1.21",
                "max_drawdown": "-0.08",
                "signal_file": "prod_a_latest.csv",
                "log_file": "slice.log",
            }
        ],
    )
    _write_csv(
        signal_dir / "prod_a_latest.csv",
        ["signal_date", "buy_date", "stock_code", "symbol", "name", "rank", "pred_prob", "target_pct"],
        [
            {
                "signal_date": "20260628",
                "buy_date": "20260630",
                "stock_code": "000001.SZ",
                "symbol": "SZSE.000001",
                "name": "骞冲畨閾惰",
                "rank": "1",
                "pred_prob": "0.91",
                "target_pct": "0.98",
            }
        ],
    )
    _write_json(
        signal_dir / "prod_a_status.json",
        {
            "status": "published",
            "latest_signal_date": "20260628",
            "latest_buy_date": "20260630",
            "published_at": "2026-06-28T09:00:00+08:00",
        },
    )
    _seed_split_production_assets(data_dir)
    return project_dir, data_dir, strategy_dir


def _refresh_project(strategy_dir: Path, data_dir: Path, project_dir: Path) -> None:
    _write_json(
        project_dir / "strategy_library" / "registry.json",
        {
            "production": {
                "current": "prod_a",
                "strategies": [
                    {
                        "strategy_id": "prod_a",
                        "name": "Prod A Updated",
                        "status": "production",
                        "path": "strategy_library/production/prod_a",
                        "published_at": "2026-06-28T10:00:00+08:00",
                    }
                ],
            }
        },
    )
    _write_json(
        strategy_dir / "strategy_manifest.json",
        {
            "strategy_id": "prod_a",
            "name": "Prod A Updated",
            "status": "production",
            "production_version": "v2",
            "published_at": "2026-06-28T10:00:00+08:00",
            "current_signal": {
                "latest_file": "prod_a_latest.csv",
                "latest_signal_date": "20260701",
                "latest_buy_date": "20260702",
            },
            "input_contract": {"formal_manifest": "manifest_b.json"},
            "validation": {"annual_return": 0.15, "sharpe": 1.45, "max_drawdown": -0.07},
        },
    )
    _write_json(
        strategy_dir / "validation.json",
        {
            "status": "production",
            "validation_platform": "gm",
            "metrics": {
                "annual_return": 0.15,
                "pnl_ratio": 1.8,
                "sharpe": 1.45,
                "max_drawdown": -0.07,
                "avg_invested_pct": 0.95,
                "max_active_positions": 1,
            },
            "latest_signal_status": {
                "signal_date": "20260701",
                "buy_date": "20260702",
                "buy_day_hard_gate_complete": True,
                "selected_stock_code": "000002.SZ",
            },
            "hard_gate_audit": {
                "failed_files": 0,
                "signal_rows": 2,
            },
        },
    )
    _write_json(
        strategy_dir / "backtests" / "summary.json",
        {"status": "ok", "annual": 0.15},
    )
    _write_csv(
        strategy_dir / "backtests" / "time_slices.csv",
        ["trade_date", "annual", "sharpe", "max_drawdown", "signal_file", "log_file"],
        [
            {
                "trade_date": "20260628",
                "annual": "0.11",
                "sharpe": "1.21",
                "max_drawdown": "-0.08",
                "signal_file": "prod_a_latest.csv",
                "log_file": "slice.log",
            },
            {
                "trade_date": "20260701",
                "annual": "0.15",
                "sharpe": "1.45",
                "max_drawdown": "-0.07",
                "signal_file": "prod_a_latest.csv",
                "log_file": "slice2.log",
            },
        ],
    )
    _write_csv(
        data_dir / "production_signals" / "prod_a_latest.csv",
        ["signal_date", "buy_date", "stock_code", "symbol", "name", "rank", "pred_prob", "target_pct"],
        [
            {
                "signal_date": "20260701",
                "buy_date": "20260702",
                "stock_code": "000001.SZ",
                "symbol": "SZSE.000001",
                "name": "骞冲畨閾惰",
                "rank": "1",
                "pred_prob": "0.93",
                "target_pct": "0.50",
            },
            {
                "signal_date": "20260701",
                "buy_date": "20260702",
                "stock_code": "000002.SZ",
                "symbol": "SZSE.000002",
                "name": "涓囩A",
                "rank": "2",
                "pred_prob": "0.88",
                "target_pct": "0.48",
            },
        ],
    )
    _write_json(
        data_dir / "production_signals" / "prod_a_status.json",
        {
            "status": "published",
            "latest_signal_date": "20260701",
            "latest_buy_date": "20260702",
            "published_at": "2026-06-28T10:00:00+08:00",
        },
    )


def _duckdb_summary(l5_path: Path, l6_path: Path, l7_path: Path) -> dict:
    import duckdb

    with duckdb.connect(str(l5_path), read_only=True) as conn:
        l5 = {
            "registry_rows": int(conn.execute('SELECT COUNT(*) FROM "prod_l5_strategy_registry_current"').fetchone()[0]),
            "manifest_rows": int(conn.execute('SELECT COUNT(*) FROM "prod_l5_strategy_manifest_current"').fetchone()[0]),
            "registry_name": conn.execute('SELECT name FROM "prod_l5_strategy_registry_current" LIMIT 1').fetchone()[0],
            "manifest_version": conn.execute('SELECT production_version FROM "prod_l5_strategy_manifest_current" LIMIT 1').fetchone()[0],
        }
    with duckdb.connect(str(l6_path), read_only=True) as conn:
        l6 = {
            "validation_rows": int(conn.execute('SELECT COUNT(*) FROM "prod_l6_strategy_validation_current"').fetchone()[0]),
            "backtest_csv_rows": int(conn.execute('SELECT COUNT(*) FROM "prod_l6_backtest_csv_rows_current"').fetchone()[0]),
            "validation_annual_return": conn.execute('SELECT annual_return FROM "prod_l6_strategy_validation_current" LIMIT 1').fetchone()[0],
        }
    with duckdb.connect(str(l7_path), read_only=True) as conn:
        l7 = {
            "signal_rows": int(conn.execute('SELECT COUNT(*) FROM "prod_l7_signal_rows_current"').fetchone()[0]),
            "signal_status_rows": int(conn.execute('SELECT COUNT(*) FROM "prod_l7_signal_status_current"').fetchone()[0]),
            "latest_signal_date": conn.execute('SELECT max(signal_date) FROM "prod_l7_signal_rows_current"').fetchone()[0],
        }
    return {"l5": l5, "l6": l6, "l7": l7}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate L5/L6/L7 DuckDB update contract with isolated temp data.")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    tmp_root = Path(tempfile.mkdtemp(prefix="l5_l6_l7_duckdb_contract_"))
    root = tmp_root / "workspace"
    project_dir, data_dir, strategy_dir = _seed_initial_project(root)
    l5_duckdb_path = data_dir / "production_assets" / "duckdb" / "production" / "l5" / "prod_l5_strategy_registry_current.duckdb"
    l6_duckdb_path = data_dir / "production_assets" / "duckdb" / "production" / "l6" / "prod_l6_strategy_validation_current.duckdb"
    l7_duckdb_path = data_dir / "production_assets" / "duckdb" / "production" / "l7" / "prod_l7_signal_rows_current.duckdb"
    registry_path = project_dir / "strategy_library" / "registry.json"

    initial_l5 = sync_strategy_registry_to_duckdb(project_dir=project_dir, data_dir=data_dir, registry_path=registry_path, duckdb_path=l5_duckdb_path)
    initial_l6 = sync_strategy_backtests_to_duckdb(project_dir=project_dir, data_dir=data_dir, registry_path=registry_path, duckdb_path=l6_duckdb_path)
    initial_l7 = sync_production_signal_artifacts_to_duckdb(project_dir=project_dir, data_dir=data_dir, registry_path=registry_path, duckdb_path=l7_duckdb_path)
    initial_summary = _duckdb_summary(l5_duckdb_path, l6_duckdb_path, l7_duckdb_path)

    _refresh_project(strategy_dir, data_dir, project_dir)

    refresh_l5 = sync_strategy_registry_to_duckdb(project_dir=project_dir, data_dir=data_dir, registry_path=registry_path, duckdb_path=l5_duckdb_path)
    refresh_l6 = sync_strategy_backtests_to_duckdb(project_dir=project_dir, data_dir=data_dir, registry_path=registry_path, duckdb_path=l6_duckdb_path)
    refresh_l7 = sync_production_signal_artifacts_to_duckdb(project_dir=project_dir, data_dir=data_dir, registry_path=registry_path, duckdb_path=l7_duckdb_path)
    refresh_summary = _duckdb_summary(l5_duckdb_path, l6_duckdb_path, l7_duckdb_path)
    current_context = load_current_production_strategy_context(
        registry_file=registry_path,
        strategy_root=project_dir / "strategy_library" / "production",
        signal_dir=data_dir / "production_signals",
        data_dir=data_dir,
        backend="duckdb",
    )
    validation_payload = load_strategy_validation_payload(
        "prod_a",
        data_dir=data_dir,
        backend="duckdb",
    )

    checks = {
        "initial_l5_rows_ok": initial_summary["l5"]["registry_rows"] == 1 and initial_summary["l5"]["manifest_rows"] == 1,
        "initial_l6_rows_ok": initial_summary["l6"]["validation_rows"] == 1 and initial_summary["l6"]["backtest_csv_rows"] == 1,
        "initial_l7_rows_ok": initial_summary["l7"]["signal_rows"] == 1 and initial_summary["l7"]["signal_status_rows"] == 1,
        "refresh_l5_payload_updated": refresh_summary["l5"]["registry_name"] == "Prod A Updated" and refresh_summary["l5"]["manifest_version"] == "v2",
        "refresh_l6_payload_updated": float(refresh_summary["l6"]["validation_annual_return"]) == 0.15 and refresh_summary["l6"]["backtest_csv_rows"] == 2,
        "refresh_l7_payload_updated": refresh_summary["l7"]["signal_rows"] == 2 and str(refresh_summary["l7"]["latest_signal_date"]) == "20260701",
        "duckdb_route_reads_current_context": len(current_context["signal_rows"]) == 2,
        "duckdb_route_reads_validation": float(validation_payload["metrics"]["annual_return"]) == 0.15,
    }

    result = {
        "status": "ok" if all(checks.values()) else "failed",
        "checks": checks,
        "initial_sync": {
            "l5": initial_l5,
            "l6": initial_l6,
            "l7": initial_l7,
            "duckdb_summary": initial_summary,
        },
        "refresh_sync": {
            "l5": refresh_l5,
            "l6": refresh_l6,
            "l7": refresh_l7,
            "duckdb_summary": refresh_summary,
        },
        "route_validation": {
            "strategy_id": current_context["strategy_entry"]["strategy_id"],
            "signal_rows": len(current_context["signal_rows"]),
            "validation_annual_return": validation_payload["metrics"]["annual_return"],
            "resolved_paths": {
                "l6": str(resolve_strategy_validation_duckdb_path(data_dir=data_dir, require_exists=True)),
                "l7": str(resolve_strategy_signal_duckdb_path(data_dir=data_dir, require_exists=True)),
            },
        },
        "boundary": {
            "uses_isolated_temp_data": True,
            "edits_production_registry": False,
            "switches_mainline_routes": False,
            "runs_business_pipeline": False,
            "touches_real_l5_l6_l7_assets": False,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(report_path))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

