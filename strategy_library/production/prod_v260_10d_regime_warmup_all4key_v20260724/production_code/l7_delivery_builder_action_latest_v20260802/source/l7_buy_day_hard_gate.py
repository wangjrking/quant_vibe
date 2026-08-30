from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _is_missing(value: Any) -> bool:
    if value in (None, "", "None", "NONE"):
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _is_st_like(row: dict[str, Any] | None) -> bool:
    if not row:
        return True
    name = str(_value(row, "name") or "").strip().upper()
    if name.startswith("ST") or name.startswith("*ST"):
        return True
    st_type_name = str(_value(row, "ST_TYPE_name", "st_type_name") or "").strip()
    if "风险" in st_type_name:
        return True
    st_type = _value(row, "ST_TYPE", "st_type")
    if _is_missing(st_type):
        return False
    text = str(st_type).strip().upper()
    if text in {"0", "0.0", "FALSE", "NONE", "NAN"}:
        return False
    try:
        return float(st_type) != 0.0
    except (TypeError, ValueError):
        return text in {"ST", "*ST", "S"}


def _is_limit_buy(row: dict[str, Any] | None) -> bool:
    if not row:
        return True
    limit_times = _as_float(_value(row, "limit_times"), 0.0)
    if limit_times and limit_times > 0:
        return True
    pre_close = _as_float(_value(row, "pre_close"))
    open_price = _as_float(_value(row, "open"))
    if pre_close is None or open_price is None or pre_close <= 0 or open_price <= 0:
        return False
    code = str(_value(row, "stock_code") or "")
    pct = 0.20 if code.startswith(("300", "301", "688")) else 0.10
    return open_price >= pre_close * (1.0 + pct) * 0.995


def _market_data_blockers(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return ["buy_day_market_row_missing"]
    blockers: list[str] = []
    pre_close = _as_float(_value(row, "pre_close"))
    open_price = _as_float(_value(row, "open"))
    if pre_close is None or pre_close <= 0:
        blockers.append("buy_day_pre_close_missing")
    if open_price is None or open_price <= 0:
        blockers.append("buy_day_open_price_missing")
    return blockers


def evaluate_buy_day_hard_gate(
    *,
    signal_row: dict[str, Any],
    market_row: dict[str, Any] | None,
) -> dict[str, Any]:
    stock_code = str(_value(signal_row, "stock_code") or "").strip()
    buy_date = str(_value(signal_row, "buy_date") or "").strip()
    result: dict[str, Any] = {
        "stock_code": stock_code,
        "buy_date": buy_date,
        "buy_day_market_available": bool(market_row),
        "buy_day_hard_gate_complete": False,
        "buy_day_st_rejected": False,
        "buy_day_open_limit_up_rejected": False,
        "ready_for_human_confirmation_execution": False,
        "status": "pending_buy_day_hard_gate",
        "blockers": [],
    }
    market_blockers = _market_data_blockers(market_row)
    if market_blockers:
        result["blockers"].extend(market_blockers)
        if market_row:
            result["market_row"] = market_row
        return result

    st_rejected = _is_st_like(market_row)
    limit_rejected = _is_limit_buy(market_row)
    result.update(
        {
            "buy_day_hard_gate_complete": True,
            "buy_day_st_rejected": st_rejected,
            "buy_day_open_limit_up_rejected": limit_rejected,
            "market_row": market_row,
        }
    )
    if st_rejected:
        result["blockers"].append("buy_day_st_or_risk_warning")
    if limit_rejected:
        result["blockers"].append("buy_day_open_limit_up")
    if result["blockers"]:
        result["status"] = "buy_day_hard_gate_rejected"
        return result

    result["status"] = "buy_day_hard_gate_passed"
    result["ready_for_human_confirmation_execution"] = True
    return result


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _read_csv_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file).fieldnames or [])


def _write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    fields = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        raise ValueError(f"cannot write CSV without fieldnames: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def load_market_row(
    *,
    db_path: Path,
    table: str,
    trade_date: str,
    stock_code: str,
) -> dict[str, Any] | None:
    sql = f"""
        SELECT *
        FROM {_quote_ident(table)}
        WHERE trade_date = ? AND stock_code = ?
        LIMIT 1
    """
    if db_path.suffix.lower() != ".duckdb":
        raise ValueError("L7 buy-day hard gate is DuckDB-only; market_db_path must point to a .duckdb file")

    import duckdb

    with duckdb.connect(str(db_path), read_only=True) as conn:
        result = conn.execute(sql, [trade_date, stock_code])
        row = result.fetchone()
        if row is None:
            return None
        columns = [item[0] for item in result.description]
        return dict(zip(columns, row))


def _normalize_market_snapshot_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    code = _value(row, "stock_code", "ts_code", "symbol", "code")
    if code is not None:
        code_text = str(code).strip().upper()
        if "." not in code_text and len(code_text) == 6:
            if code_text.startswith(("0", "3")):
                code_text = f"{code_text}.SZ"
            elif code_text.startswith(("6", "9")):
                code_text = f"{code_text}.SH"
        normalized["stock_code"] = code_text
    trade_date = _value(row, "trade_date", "buy_date", "date")
    if trade_date is not None:
        normalized["trade_date"] = str(trade_date).strip()
    open_price = _value(row, "open", "openPrice", "open_price")
    if open_price is not None:
        normalized["open"] = open_price
    pre_close = _value(row, "pre_close", "lastClose", "preClose", "last_close", "prev_close")
    if pre_close is not None:
        normalized["pre_close"] = pre_close
    name = _value(row, "name", "stock_name", "instrument_name")
    if name is not None:
        normalized["name"] = name
    return normalized


def _read_market_snapshot(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            if isinstance(payload.get("rows"), list):
                rows = payload["rows"]
            elif isinstance(payload.get("ticks"), list):
                rows = payload["ticks"]
            elif isinstance(payload.get("data"), list):
                rows = payload["data"]
            else:
                rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        return [_normalize_market_snapshot_row(row) for row in rows if isinstance(row, dict)]
    return [_normalize_market_snapshot_row(row) for row in _read_csv_rows(path)]


def _snapshot_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(_value(row, "trade_date") or "").strip(), str(_value(row, "stock_code") or "").strip().upper())


def load_market_snapshot_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = _read_market_snapshot(path)
    return {_snapshot_key(row): row for row in rows if all(_snapshot_key(row))}


def finalize_platform_signals(
    *,
    platform_signals_path: Path,
    market_db_path: Path,
    output_dir: Path,
    stock_daily_table: str = "STOCK_DAILY_DATA",
    market_snapshot_path: Path | None = None,
    no_signal_hold_only: bool = False,
) -> dict[str, Any]:
    rows = _read_csv_rows(platform_signals_path)
    if no_signal_hold_only and rows:
        raise ValueError("no_signal_hold_only cannot contain platform action rows")
    snapshot_rows = load_market_snapshot_rows(market_snapshot_path) if market_snapshot_path else {}
    finalized_rows: list[dict[str, Any]] = []
    gate_results: list[dict[str, Any]] = []
    for row in rows:
        buy_date = str(row.get("buy_date") or "")
        stock_code = str(row.get("stock_code") or "").strip().upper()
        market_row = snapshot_rows.get((buy_date, stock_code))
        market_source = "market_snapshot" if market_row else "stock_daily_table"
        if market_row is None:
            market_row = load_market_row(
                db_path=market_db_path,
                table=stock_daily_table,
                trade_date=buy_date,
                stock_code=stock_code,
            )
        gate = evaluate_buy_day_hard_gate(signal_row=row, market_row=market_row)
        gate["market_source"] = market_source
        gate_results.append(gate)
        finalized = dict(row)
        finalized.update(
            {
                "status": gate["status"],
                "buy_day_market_available": str(gate["buy_day_market_available"]),
                "buy_day_hard_gate_complete": str(gate["buy_day_hard_gate_complete"]),
                "buy_day_st_rejected": str(gate["buy_day_st_rejected"]),
                "buy_day_open_limit_up_rejected": str(gate["buy_day_open_limit_up_rejected"]),
                "ready_for_human_confirmation_execution": str(gate["ready_for_human_confirmation_execution"]),
                "hard_gate_blockers": ";".join(gate["blockers"]),
                "hard_gate_market_source": market_source,
            }
        )
        finalized_rows.append(finalized)

    all_ready = bool(gate_results) and all(item["ready_for_human_confirmation_execution"] for item in gate_results)
    any_rejected = any(item["status"] == "buy_day_hard_gate_rejected" for item in gate_results)
    status = "ready_for_human_confirmation_execution" if all_ready else "pending_buy_day_hard_gate"
    if any_rejected:
        status = "buy_day_hard_gate_rejected"

    output_dir.mkdir(parents=True, exist_ok=True)
    finalized_csv = output_dir / "buy_day_hard_gate_platform_signals.csv"
    summary_json = output_dir / "buy_day_hard_gate_summary.json"
    report_md = output_dir / "buy_day_hard_gate_report.md"
    _write_csv_rows(
        finalized_csv,
        finalized_rows,
        fieldnames=_read_csv_fieldnames(platform_signals_path),
    )
    summary = {
        "status": status,
        "signal_semantics": "no_signal_hold_only" if no_signal_hold_only else "action_signals",
        "no_signal": no_signal_hold_only,
        "hold_only": no_signal_hold_only,
        "buy_day_hard_gate_applicable": not no_signal_hold_only,
        "buy_day_hard_gate_complete": False if no_signal_hold_only else all_ready,
        "ready_for_human_confirmation_execution": all_ready,
        "l7_execution_allowed": False,
        "execution_allowed": False,
        "approved_for_execution": False,
        "auto_execution_allowed": False,
        "live_execution_allowed": False,
        "signal_count": len(rows),
        "market_db_path": str(market_db_path),
        "stock_daily_table": stock_daily_table,
        "market_snapshot_path": str(market_snapshot_path) if market_snapshot_path else None,
        "source_platform_signals": str(platform_signals_path),
        "finalized_platform_signals": str(finalized_csv),
        "report_path": str(report_md),
        "gate_results": gate_results,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(
        "\n".join(
            [
                "# L7 买入日硬门控复核",
                "",
                f"- 状态：`{status}`",
                f"- 信号数量：`{len(rows)}`",
                f"- 是否可人工确认执行：`{all_ready}`",
                f"- 信号语义：`{summary['signal_semantics']}`",
                f"- 是否允许 L7 执行：`{summary['l7_execution_allowed']}`",
                f"- 实时/平台行情快照：`{market_snapshot_path}`",
                f"- 明细 CSV：`{finalized_csv}`",
                f"- 摘要 JSON：`{summary_json}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize L7 buy-day hard gate for platform delivery signals.")
    parser.add_argument("--platform-signals", required=True)
    parser.add_argument("--market-db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stock-daily-table", default="STOCK_DAILY_DATA")
    parser.add_argument(
        "--market-snapshot",
        default=None,
        help="Optional buy-day live/platform market snapshot CSV or JSON. This is checked before STOCK_DAILY_DATA.",
    )
    parser.add_argument("--exit-nonzero-when-not-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = finalize_platform_signals(
        platform_signals_path=Path(args.platform_signals),
        market_db_path=Path(args.market_db),
        output_dir=Path(args.output_dir),
        stock_daily_table=args.stock_daily_table,
        market_snapshot_path=Path(args.market_snapshot) if args.market_snapshot else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.exit_nonzero_when_not_ready and not summary["ready_for_human_confirmation_execution"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
