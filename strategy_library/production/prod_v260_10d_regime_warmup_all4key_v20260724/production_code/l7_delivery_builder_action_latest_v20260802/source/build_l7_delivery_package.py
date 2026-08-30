from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from export_latest_signal_from_strategy_archive import export_latest_signal
from l7_buy_day_hard_gate import finalize_platform_signals, load_market_snapshot_rows
from l7_duckdb_sync import (
    require_current_production_strategy_id,
    sync_production_signal_artifacts_to_duckdb,
)
from project_paths import resolve_data_path, resolve_project_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    output_fieldnames = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in output_fieldnames:
                output_fieldnames.append(key)
    if not output_fieldnames:
        raise ValueError(f"cannot write CSV without fieldnames: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in output_fieldnames} for row in rows])


def _resolve_maybe_abs(value: str | Path | None, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _derive_symbol(stock_code: str) -> str:
    code = str(stock_code or "").strip().upper()
    if code.endswith(".SZ"):
        return f"SZSE.{code[:6]}"
    if code.endswith(".SH"):
        return f"SHSE.{code[:6]}"
    return ""


def _build_snapshot_availability_payload(
    *,
    strategy_id: str,
    signal_date: str,
    buy_date: str,
    market_snapshot_path: Path | None,
) -> dict[str, Any]:
    if market_snapshot_path and market_snapshot_path.exists():
        snapshot_rows = list(load_market_snapshot_rows(market_snapshot_path).values())
        for row in snapshot_rows:
            row["requested_buy_date"] = buy_date
            row["snapshot_trade_date_matches_buy_date"] = str(row.get("trade_date") or "") == str(buy_date)
        return {
            "strategy_id": strategy_id,
            "signal_date": signal_date,
            "buy_date": buy_date,
            "market_snapshot_path": str(market_snapshot_path),
            "snapshot_rows": snapshot_rows,
            "all_snapshot_trade_dates_match_buy_date": bool(snapshot_rows)
            and all(bool(row.get("snapshot_trade_date_matches_buy_date")) for row in snapshot_rows),
            "note": "已提供买入日实时/平台行情快照；是否可放行以硬门控结果为准。",
        }
    return {
        "strategy_id": strategy_id,
        "signal_date": signal_date,
        "buy_date": buy_date,
        "market_snapshot_path": None,
        "snapshot_rows": [],
        "all_snapshot_trade_dates_match_buy_date": False,
        "note": "未提供买入日实时/平台行情快照，当前仅完成 L7 交付打包，状态维持 pending_buy_day_hard_gate。",
    }


def _require_structured_no_signal_status(
    status: dict[str, Any],
    *,
    strategy_id: str,
    signal_date: str | None = None,
    buy_date: str | None = None,
) -> tuple[str, str]:
    expected_true = ("no_signal", "hold_only", "formal_batch_generated")
    expected_false = (
        "l7_execution_allowed",
        "execution_allowed",
        "approved_for_execution",
        "auto_execution_allowed",
        "live_execution_allowed",
    )
    expected_zero = ("action_count", "row_count", "buy_count", "sell_count")

    if str(status.get("strategy_id") or "") != strategy_id:
        raise ValueError("no-signal status strategy_id does not match production.current")
    if str(status.get("status") or "") != "pending_buy_day_hard_gate":
        raise ValueError("no-signal status must remain pending_buy_day_hard_gate")
    if str(status.get("signal_semantics") or "") != "no_signal_hold_only":
        raise ValueError("empty latest requires signal_semantics=no_signal_hold_only")
    if any(status.get(key) is not True for key in expected_true):
        raise ValueError("no-signal status must explicitly set no_signal/hold_only/formal_batch_generated=true")
    if any(status.get(key) is not False for key in expected_false):
        raise ValueError("no-signal status must explicitly keep every execution flag false")
    if any(int(status.get(key, -1)) != 0 for key in expected_zero):
        raise ValueError("no-signal status must explicitly declare all action counts as zero")

    selected_signal_date = str(signal_date or status.get("signal_date") or "")
    selected_buy_date = str(buy_date or status.get("buy_date") or "")
    if not selected_signal_date or not selected_buy_date:
        raise ValueError("no-signal status must include signal_date and buy_date")
    if selected_signal_date != str(status.get("signal_date") or ""):
        raise ValueError("requested signal_date does not match no-signal status")
    if selected_buy_date != str(status.get("buy_date") or ""):
        raise ValueError("requested buy_date does not match no-signal status")
    return selected_signal_date, selected_buy_date


def _require_action_signal_status(
    status: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    strategy_id: str,
    signal_date: str | None = None,
    buy_date: str | None = None,
) -> tuple[str, str]:
    """Validate a formal action batch when no archive export path is present."""
    expected_false = (
        "no_signal",
        "hold_only",
        "l7_execution_allowed",
        "execution_allowed",
        "approved_for_execution",
        "auto_execution_allowed",
        "live_execution_allowed",
    )
    if str(status.get("strategy_id") or "") != strategy_id:
        raise ValueError("action status strategy_id does not match production.current")
    if str(status.get("status") or "") != "pending_buy_day_hard_gate":
        raise ValueError("action status must remain pending_buy_day_hard_gate")
    if str(status.get("signal_semantics") or "") != "trade_actions_pending_buy_day_hard_gate":
        raise ValueError("action latest requires trade_actions_pending_buy_day_hard_gate semantics")
    if status.get("formal_batch_generated") is not True:
        raise ValueError("action status must explicitly set formal_batch_generated=true")
    if any(status.get(key) is not False for key in expected_false):
        raise ValueError("action status must explicitly keep no-signal and execution flags false")
    if int(status.get("row_count", -1)) != len(rows) or int(status.get("action_count", -1)) != len(rows):
        raise ValueError("action status row_count/action_count does not match latest CSV")

    selected_signal_date = str(signal_date or status.get("signal_date") or "")
    selected_buy_date = str(buy_date or status.get("buy_date") or "")
    if not selected_signal_date or not selected_buy_date:
        raise ValueError("action status must include signal_date and buy_date")
    if selected_signal_date != str(status.get("signal_date") or ""):
        raise ValueError("requested signal_date does not match action status")
    if selected_buy_date != str(status.get("buy_date") or ""):
        raise ValueError("requested buy_date does not match action status")
    for row in rows:
        if str(row.get("strategy_id") or "") != strategy_id:
            raise ValueError("action latest CSV strategy_id does not match production.current")
        if str(row.get("signal_date") or "") != selected_signal_date:
            raise ValueError("action latest CSV signal_date does not match status")
        if str(row.get("buy_date") or "") != selected_buy_date:
            raise ValueError("action latest CSV buy_date does not match status")
        if str(row.get("action") or "").upper() not in {"BUY", "SELL"}:
            raise ValueError("action latest CSV contains an unsupported action")
    return selected_signal_date, selected_buy_date


def _build_integrity_checks(
    rows: list[dict[str, Any]],
    *,
    no_signal_hold_only: bool = False,
) -> dict[str, bool]:
    keys = {(str(row.get("signal_date") or ""), str(row.get("stock_code") or "")) for row in rows}
    return {
        "row_count_positive": len(rows) > 0,
        "row_count_valid": len(rows) > 0 or no_signal_hold_only,
        "structured_no_signal_hold_only": no_signal_hold_only,
        "duplicate_keys_is_0": len(keys) == len(rows),
        "all_symbol_mapping_valid": all(
            not str(row.get("symbol") or "").strip()
            or str(row.get("symbol") or "").strip() == _derive_symbol(str(row.get("stock_code") or ""))
            for row in rows
        ),
        "all_name_fields_nonempty": all(bool(str(row.get("name") or "").strip()) for row in rows),
    }


def _write_handoff_markdown(
    path: Path,
    *,
    strategy_id: str,
    signal_date: str,
    buy_date: str,
    latest_status: dict[str, Any],
    rows: list[dict[str, Any]],
    hard_gate_summary: dict[str, Any],
    no_signal_hold_only: bool = False,
) -> None:
    top_lines = [
        "# L7 交易交付包",
        "",
        "## 当前状态",
        f"- 策略 ID：`{strategy_id}`",
        f"- 信号日：`{signal_date}`",
        f"- 买入日：`{buy_date}`",
        f"- 正式状态：`{latest_status.get('status')}`",
        f"- 硬门控状态：`{hard_gate_summary.get('status')}`",
        f"- 信号条数：`{len(rows)}`",
        "",
        "## 当前结论",
    ]
    if no_signal_hold_only:
        top_lines.append("- 本批为结构化无交易信号批次，不包含任何 BUY、SELL 或伪造 HOLD 行。")
        top_lines.append("- 当前语义为继续持有且不可执行；所有执行、审批、自动化和实盘标志均为 false。")
    elif hard_gate_summary.get("ready_for_human_confirmation_execution"):
        top_lines.append("- 买入日硬门控已通过，可进入人工确认执行。")
    else:
        top_lines.append("- 当前仅完成 L7 交付打包，买入日硬门控尚未通过。")
        top_lines.append("- 在实时/平台行情快照缺失或未通过前，不允许解释为可自动执行。")
    top_lines.extend(
        [
            "",
            "## 标的列表",
        ]
    )
    for row in rows:
        top_lines.append(
            f"- `{row.get('stock_code')}` `{row.get('name')}` -> `{row.get('symbol')}` "
            f"`target_pct={row.get('target_pct')}`"
        )
    if not rows:
        top_lines.append("- 无动作行。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(top_lines) + "\n", encoding="utf-8")


def build_l7_delivery_package(
    *,
    strategy_dir: str | Path,
    production_signal_dir: str | Path,
    delivery_root: str | Path,
    market_db_path: str | Path,
    market_snapshot_path: str | Path | None = None,
    stock_daily_table: str = "STOCK_DAILY_DATA",
    signal_date: str | None = None,
    buy_date: str | None = None,
    require_fresh: bool = True,
    sync_duckdb: bool = True,
    no_signal_upstream_audit_approved: bool = False,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    strategy_path = Path(strategy_dir).resolve()
    production_signal_path = Path(production_signal_dir).resolve()
    delivery_root_path = Path(delivery_root).resolve()
    market_db = Path(market_db_path).resolve()
    market_snapshot = Path(market_snapshot_path).resolve() if market_snapshot_path else None

    manifest = _load_json(strategy_path / "strategy_manifest.json")
    strategy_id = str(manifest.get("strategy_id") or strategy_path.name)
    strategy_registry_path = strategy_path.parents[1] / "registry.json"
    strategy_project_dir = strategy_path.parents[2]
    current_strategy_id = require_current_production_strategy_id(
        registry_path=strategy_registry_path,
        project_dir=strategy_project_dir,
    )
    if strategy_id != current_strategy_id:
        raise ValueError(
            "L7 delivery strategy must exactly match production.current: "
            f"current={current_strategy_id}, requested={strategy_id}"
        )
    current_signal = manifest.get("current_signal", {}) if isinstance(manifest.get("current_signal"), dict) else {}

    latest_csv_path = production_signal_path / f"{strategy_id}_latest.csv"
    latest_status_path = production_signal_path / f"{strategy_id}_latest_status.json"
    archive_latest_signal = _resolve_maybe_abs(manifest.get("latest_signal_file"), strategy_path)
    archive_rows = (
        _read_csv_rows(archive_latest_signal)
        if archive_latest_signal is not None and archive_latest_signal.exists()
        else []
    )
    no_signal_hold_only = False

    if archive_rows:
        selected_signal_date = str(
            signal_date or current_signal.get("signal_date") or archive_rows[0].get("signal_date") or ""
        )
        selected_buy_date = str(
            buy_date or current_signal.get("buy_date") or archive_rows[0].get("buy_date") or ""
        )
        if not selected_signal_date or not selected_buy_date:
            raise ValueError(f"missing signal_date/buy_date for {strategy_id}")
        export_latest_signal(
            strategy_dir=strategy_path,
            output=latest_csv_path,
            status_output=latest_status_path,
            signal_date=selected_signal_date,
            buy_date=selected_buy_date,
            require_fresh=require_fresh,
        )
        latest_rows = _read_csv_rows(latest_csv_path)
        latest_status = _load_json(latest_status_path)
    else:
        if not latest_csv_path.is_file() or not latest_status_path.is_file():
            raise FileNotFoundError(
                "empty or missing archive latest is only valid when structured formal latest CSV/status exist"
            )
        latest_rows = _read_csv_rows(latest_csv_path)
        latest_status = _load_json(latest_status_path)
        if latest_rows:
            selected_signal_date, selected_buy_date = _require_action_signal_status(
                latest_status,
                latest_rows,
                strategy_id=strategy_id,
                signal_date=signal_date,
                buy_date=buy_date,
            )
        else:
            selected_signal_date, selected_buy_date = _require_structured_no_signal_status(
                latest_status,
                strategy_id=strategy_id,
                signal_date=signal_date,
                buy_date=buy_date,
            )
            no_signal_hold_only = True
        required_columns = {
            "strategy_id",
            "signal_date",
            "buy_date",
            "action",
            "stock_code",
            "name",
            "market",
            "strategy_score",
            "score_rank",
            "score_denominator",
            "target_pct",
            "reason",
            "execution_open_raw",
            "status",
        }
        latest_fieldnames = set(_read_csv_fieldnames(latest_csv_path))
        if not required_columns.issubset(latest_fieldnames):
            missing = sorted(required_columns - latest_fieldnames)
            raise ValueError(f"no-signal latest CSV missing required columns: {missing}")
        if no_signal_hold_only and sync_duckdb and not no_signal_upstream_audit_approved:
            raise ValueError(
                "structured no-signal L7 current sync requires explicit upstream audit approval"
            )

    delivery_dir = delivery_root_path / f"{strategy_id}_{selected_signal_date}_for_{selected_buy_date}"
    platform_signals_path = delivery_dir / "l7_platform_signals.csv"
    _write_csv_rows(
        platform_signals_path,
        latest_rows,
        fieldnames=_read_csv_fieldnames(latest_csv_path),
    )

    snapshot_availability_path = delivery_dir / "buy_day_market_snapshot_availability.json"
    snapshot_payload = _build_snapshot_availability_payload(
        strategy_id=strategy_id,
        signal_date=selected_signal_date,
        buy_date=selected_buy_date,
        market_snapshot_path=market_snapshot,
    )
    _write_json(snapshot_availability_path, snapshot_payload)

    hard_gate_dir = delivery_dir / "buy_day_hard_gate_current_check"
    hard_gate_summary = finalize_platform_signals(
        platform_signals_path=platform_signals_path,
        market_db_path=market_db,
        output_dir=hard_gate_dir,
        stock_daily_table=stock_daily_table,
        market_snapshot_path=market_snapshot,
        no_signal_hold_only=no_signal_hold_only,
    )

    duplicate_keys = len(latest_rows) - len(
        {(str(row.get("signal_date") or ""), str(row.get("stock_code") or "")) for row in latest_rows}
    )
    summary = {
        "strategy_id": strategy_id,
        "signal_date": selected_signal_date,
        "buy_date": selected_buy_date,
        "status": "pending_buy_day_hard_gate",
        "row_count": len(latest_rows),
        "stock_count": len({str(row.get("stock_code") or "") for row in latest_rows}),
        "duplicate_signal_stock_keys": duplicate_keys,
        "target_pct_sum": round(sum(float(row.get("target_pct") or 0.0) for row in latest_rows), 10),
        "latest_status": latest_status,
        "signal_semantics": "no_signal_hold_only" if no_signal_hold_only else "action_signals",
        "no_signal": no_signal_hold_only,
        "hold_only": no_signal_hold_only,
        "action_count": len(latest_rows),
        "ready_for_human_confirmation_execution": False,
        "pending_user_approval": False,
        "auto_executable": False,
        "live_trading_ready": False,
        "l7_execution_allowed": False,
        "execution_allowed": False,
        "approved_for_execution": False,
        "auto_execution_allowed": False,
        "live_execution_allowed": False,
        "integrity_checks": _build_integrity_checks(
            latest_rows,
            no_signal_hold_only=no_signal_hold_only,
        ),
        "source_assets": {
            "latest_csv": str(latest_csv_path),
            "latest_status_json": str(latest_status_path),
            "archive_latest_signal_csv": str(archive_latest_signal or ""),
            "archive_latest_status_json": str(
                _resolve_maybe_abs(
                    manifest.get("latest_signal_status_file")
                    or ((manifest.get("current_signal") or {}).get("status_file")),
                    strategy_path,
                )
                or ""
            ),
        },
        "snapshot_availability_path": str(snapshot_availability_path),
        "buy_day_hard_gate": {
            "status": hard_gate_summary.get("status"),
            "ready_for_human_confirmation_execution": hard_gate_summary.get(
                "ready_for_human_confirmation_execution"
            ),
            "buy_day_hard_gate_complete": all(
                bool(item.get("buy_day_hard_gate_complete")) for item in hard_gate_summary.get("gate_results", [])
            )
            if hard_gate_summary.get("gate_results")
            else False,
            "summary_path": str(hard_gate_dir / "buy_day_hard_gate_summary.json"),
        },
        "pending_reason": (
            "买入日实时/平台行情快照缺失或未通过，当前仅允许维持 pending_buy_day_hard_gate。"
            if hard_gate_summary.get("status") != "ready_for_human_confirmation_execution"
            else ""
        ),
    }
    summary_path = delivery_dir / "l7_delivery_summary.json"
    _write_json(summary_path, summary)

    _write_handoff_markdown(
        delivery_dir / "handoff.md",
        strategy_id=strategy_id,
        signal_date=selected_signal_date,
        buy_date=selected_buy_date,
        latest_status=latest_status,
        rows=latest_rows,
        hard_gate_summary=hard_gate_summary,
        no_signal_hold_only=no_signal_hold_only,
    )

    sync_result: dict[str, Any] | None = None
    if sync_duckdb:
        effective_data_dir = Path(data_dir).resolve() if data_dir else production_signal_path.parent
        sync_result = sync_production_signal_artifacts_to_duckdb(
            data_dir=effective_data_dir,
            signal_dir=production_signal_path,
            project_dir=strategy_project_dir,
            registry_path=strategy_registry_path,
            strategy_ids={strategy_id},
            status_payload_overrides=(
                {
                    "signal_semantics": "no_signal_hold_only",
                    "ready_for_human_confirmation_execution": False,
                    "pending_user_approval": False,
                    "auto_executable": False,
                    "live_trading_ready": False,
                    "l7_execution_allowed": False,
                    "execution_allowed": False,
                    "approved_for_execution": False,
                    "auto_execution_allowed": False,
                    "live_execution_allowed": False,
                }
                if no_signal_hold_only
                else None
            ),
        )

    return {
        "strategy_id": strategy_id,
        "signal_date": selected_signal_date,
        "buy_date": selected_buy_date,
        "latest_csv_path": str(latest_csv_path),
        "latest_status_path": str(latest_status_path),
        "delivery_dir": str(delivery_dir),
        "summary_path": str(summary_path),
        "hard_gate_summary_path": str(hard_gate_dir / "buy_day_hard_gate_summary.json"),
        "sync_result": sync_result,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the current L7 delivery package from latest production signals.")
    parser.add_argument("--strategy-dir", required=True)
    parser.add_argument("--production-signal-dir", default="quant/data_file/production_signals")
    parser.add_argument("--delivery-root", default="quant/data_file/runtime/trading_agent/delivery_packages")
    parser.add_argument("--market-db", required=True)
    parser.add_argument("--market-snapshot")
    parser.add_argument("--stock-daily-table", default="STOCK_DAILY_DATA")
    parser.add_argument("--signal-date")
    parser.add_argument("--buy-date")
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument("--no-sync-duckdb", action="store_true")
    parser.add_argument(
        "--no-signal-upstream-audit-approved",
        action="store_true",
        help="Required before a structured no-signal batch may update L7 current.",
    )
    parser.add_argument("--data-dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_l7_delivery_package(
        strategy_dir=resolve_project_path(args.strategy_dir),
        production_signal_dir=resolve_project_path(args.production_signal_dir),
        delivery_root=resolve_project_path(args.delivery_root),
        market_db_path=resolve_project_path(args.market_db),
        market_snapshot_path=resolve_project_path(args.market_snapshot) if args.market_snapshot else None,
        stock_daily_table=args.stock_daily_table,
        signal_date=args.signal_date,
        buy_date=args.buy_date,
        require_fresh=not args.allow_stale,
        sync_duckdb=not args.no_sync_duckdb,
        no_signal_upstream_audit_approved=args.no_signal_upstream_audit_approved,
        data_dir=resolve_data_path(args.data_dir) if args.data_dir else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
