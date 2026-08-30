from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from adjustment_semantics import validate_strategy_output_field_names
from project_paths import resolve_project_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    validate_strategy_output_field_names(fieldnames, context="export_latest_signal_from_strategy_archive.output_rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def _resolve_maybe_abs(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _formal_max_trade_date(manifest_path: Path) -> str:
    manifest = _load_json(manifest_path)
    if manifest.get("approval_status") != "approved_for_l5":
        raise RuntimeError(f"manifest is not approved_for_l5: {manifest_path}")
    max_trade_date = str(manifest.get("max_trade_date") or "")
    if not max_trade_date:
        raise RuntimeError(f"manifest missing max_trade_date: {manifest_path}")
    return max_trade_date


def _require_fresh_archive_signal(manifest: dict[str, Any], latest_signal_date: str) -> dict[str, Any]:
    contract = manifest.get("input_contract", {}) if isinstance(manifest.get("input_contract"), dict) else {}
    formal_paths = [
        contract.get("formal_manifest_3d"),
        contract.get("formal_manifest_5d"),
        contract.get("formal_manifest_10d"),
    ]
    checked: list[dict[str, str]] = []
    for raw_path in formal_paths:
        if raw_path in (None, ""):
            continue
        manifest_path = _resolve_maybe_abs(str(raw_path), Path.cwd())
        max_trade_date = _formal_max_trade_date(manifest_path)
        checked.append({"manifest_path": str(manifest_path), "max_trade_date": max_trade_date})
        if str(max_trade_date) > str(latest_signal_date):
            raise RuntimeError(
                "archive signal is stale: "
                f"latest_signal_date={latest_signal_date}, manifest_max_trade_date={max_trade_date}, "
                f"manifest={manifest_path}"
            )
    return {"latest_signal_date": latest_signal_date, "formal_manifest_checks": checked}


def export_latest_signal(
    *,
    strategy_dir: Path,
    output: Path,
    status_output: Path | None = None,
    signal_date: str | None = None,
    buy_date: str | None = None,
    require_fresh: bool = True,
) -> dict[str, Any]:
    manifest = _load_json(strategy_dir / "strategy_manifest.json")
    source_raw = manifest.get("full_history_signal_file")
    if not source_raw:
        raise RuntimeError(f"strategy_manifest missing full_history_signal_file: {strategy_dir}")
    source_file = _resolve_maybe_abs(str(source_raw), strategy_dir)
    rows = _read_csv(source_file)
    if not rows:
        raise RuntimeError(f"archive signal file is empty: {source_file}")

    selected_signal_date = str(signal_date or max(str(row.get("signal_date") or "") for row in rows))
    latest_rows = [dict(row) for row in rows if str(row.get("signal_date") or "") == selected_signal_date]
    if not latest_rows:
        raise RuntimeError(f"no rows for signal_date={selected_signal_date}: {source_file}")
    latest_rows.sort(key=lambda row: int(row.get("rank") or "999999"))

    current_signal = manifest.get("current_signal", {}) if isinstance(manifest.get("current_signal"), dict) else {}
    manifest_gate_complete = bool(current_signal.get("buy_day_hard_gate_complete"))
    selected_buy_date = str(buy_date or current_signal.get("buy_date") or latest_rows[0].get("buy_date") or "")
    source_buy_dates = {str(row.get("buy_date") or "") for row in latest_rows}
    buy_date_overridden = bool(selected_buy_date) and source_buy_dates != {selected_buy_date}
    if selected_buy_date:
        for row in latest_rows:
            row["buy_date"] = selected_buy_date
            if buy_date_overridden and not manifest_gate_complete:
                row["buy_day_market_available"] = "False"
                row["buy_day_hard_gate_complete"] = "False"
                row["buy_day_st_rejected"] = "False"
                row["buy_day_open_limit_up_rejected"] = "False"
    for row in latest_rows:
        row["strategy_variant"] = manifest.get("strategy_id") or row.get("strategy_variant") or ""

    freshness: dict[str, Any] | None = None
    if require_fresh:
        freshness = _require_fresh_archive_signal(manifest, selected_signal_date)

    _write_csv(output, latest_rows)

    duplicate_keys = len(latest_rows) - len(
        {(row.get("signal_date"), row.get("stock_code")) for row in latest_rows}
    )
    hard_gate_complete = (
        manifest_gate_complete
        if buy_date_overridden
        else all(str(row.get("buy_day_hard_gate_complete")).lower() == "true" for row in latest_rows)
    )
    status_value = "ready_for_human_confirmation_execution" if hard_gate_complete else "pending_buy_day_hard_gate"
    status = {
        "strategy_id": manifest.get("strategy_id"),
        "status": status_value,
        "signal_date": selected_signal_date,
        "buy_date": selected_buy_date,
        "latest_signal_date": selected_signal_date,
        "latest_buy_date": selected_buy_date,
        "source_file": str(source_file),
        "output_file": str(output),
        "row_count": len(latest_rows),
        "stock_count": len({row.get("stock_code") for row in latest_rows}),
        "duplicate_signal_stock_keys": duplicate_keys,
        "buy_day_hard_gate_complete": hard_gate_complete,
        "buy_day_realtime_checks_required": True,
        "freshness": freshness,
    }
    if status_output is not None:
        _write_json(status_output, status)
    return {"rows": latest_rows, "status": status}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latest signal rows from a production strategy archive.")
    parser.add_argument("--strategy-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status-output")
    parser.add_argument("--signal-date")
    parser.add_argument("--buy-date")
    parser.add_argument("--allow-stale", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_latest_signal(
        strategy_dir=resolve_project_path(args.strategy_dir),
        output=resolve_project_path(args.output),
        status_output=resolve_project_path(args.status_output) if args.status_output else None,
        signal_date=args.signal_date,
        buy_date=args.buy_date,
        require_fresh=not args.allow_stale,
    )
    print(json.dumps(result["status"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
