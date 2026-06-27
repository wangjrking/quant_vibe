from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from prediction_manifest import load_prediction_source_manifest
from project_paths import resolve_project_path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


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


def _to_gm_symbol(stock_code: str) -> str:
    code = str(stock_code or "").strip().upper()
    if code.endswith(".SH"):
        return f"SHSE.{code[:6]}"
    if code.endswith(".SZ"):
        return f"SZSE.{code[:6]}"
    if code.endswith(".BJ"):
        return f"BJSE.{code[:6]}"
    if code.startswith(("6", "9")):
        return f"SHSE.{code[:6]}"
    if code.startswith(("8", "4")):
        return f"BJSE.{code[:6]}"
    return f"SZSE.{code[:6]}"


def _is_bj(stock_code: str) -> bool:
    text = str(stock_code or "").strip().upper()
    return text.endswith(".BJ") or text.startswith("BJSE.") or text.startswith(("8", "4"))


def _is_delisting_name(name: Any) -> bool:
    text = str(name or "")
    return "退市" in text or text.startswith("退") or text.endswith("退")


def _is_st_like(row: dict[str, Any] | None) -> bool:
    if not row:
        return True
    name = str(row.get("name") or "").strip().upper()
    if name.startswith("ST") or name.startswith("*ST"):
        return True
    st_type_name = str(row.get("st_type_name") or "").strip()
    if "风险" in st_type_name:
        return True
    st_type = row.get("st_type")
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
    limit_times = _as_float(row.get("limit_times"), 0.0)
    if limit_times and limit_times > 0:
        return True
    pre_close = _as_float(row.get("pre_close"))
    open_price = _as_float(row.get("open"))
    if pre_close is None or open_price is None or pre_close <= 0 or open_price <= 0:
        return True
    code = str(row.get("stock_code") or "")
    pct = 0.20 if code.startswith(("300", "301", "688")) else 0.10
    return open_price >= pre_close * (1.0 + pct) * 0.995


def _load_manifest(path_str: str) -> dict[str, Any]:
    source = load_prediction_source_manifest(
        path_str,
        require_approved=True,
        allow_legacy=False,
    )
    return {
        "manifest_path": source["manifest_path"],
        "db_path": source["db_path"],
        "table": source["table"],
        "market_db_path": source["market_db_path"],
    }


def _load_strategy(strategy_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _load_json(strategy_dir / "strategy_manifest.json")
    rules = _load_json(strategy_dir / "trading_rules.json")
    contract = manifest["input_contract"]
    sources = {
        "3d": _load_manifest(contract["formal_manifest_3d"]),
        "5d": _load_manifest(contract["formal_manifest_5d"]),
        "10d": _load_manifest(contract["formal_manifest_10d"]),
    }
    return manifest, rules, sources


def _common_latest_trade_date(sources: dict[str, dict[str, Any]]) -> str:
    latest_dates: list[str] = []
    for item in sources.values():
        conn = sqlite3.connect(str(item["db_path"]))
        try:
            latest = conn.execute(
                f"SELECT MAX(trade_date) FROM {_quote_ident(item['table'])}"
            ).fetchone()[0]
        finally:
            conn.close()
        if not latest:
            raise RuntimeError(f"prediction table has no trade_date: {item['table']}")
        latest_dates.append(str(latest))
    return min(latest_dates)


def _next_trade_date(market_db: Path, signal_date: str) -> tuple[str | None, str]:
    conn = sqlite3.connect(str(market_db))
    try:
        latest_market_date = str(conn.execute("SELECT MAX(trade_date) FROM STOCK_DAILY_DATA").fetchone()[0] or "")
        next_row = conn.execute(
            """
            SELECT MIN(trade_date)
            FROM STOCK_DAILY_DATA
            WHERE trade_date > ?
            """,
            (str(signal_date),),
        ).fetchone()[0]
    finally:
        conn.close()
    return (str(next_row) if next_row else None), latest_market_date


def _load_buy_day_market_row(market_db: Path, buy_date: str | None, stock_code: str) -> dict[str, Any] | None:
    if not buy_date:
        return None
    conn = sqlite3.connect(str(market_db))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT trade_date, stock_code, name, pre_close, open, close, limit_times,
                   ST_TYPE AS st_type, ST_TYPE_name AS st_type_name
            FROM STOCK_DAILY_DATA
            WHERE trade_date = ? AND stock_code = ?
            """,
            (str(buy_date), str(stock_code)),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _build_candidates(
    source_3d: dict[str, Any],
    source_5d: dict[str, Any],
    source_10d: dict[str, Any],
    market_db: Path,
    signal_date: str,
) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(source_10d["db_path"]))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ATTACH DATABASE ? AS pred3d", (str(source_3d["db_path"]),))
        conn.execute("ATTACH DATABASE ? AS pred5d", (str(source_5d["db_path"]),))
        conn.execute("ATTACH DATABASE ? AS market", (str(market_db),))
        rows = conn.execute(
            f"""
            SELECT
                t10.trade_date,
                t10.stock_code,
                t3.pred_prob AS pred_3d,
                t5.pred_prob AS pred_5d,
                t10.pred_prob AS pred_10d,
                PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t3.pred_prob) AS rank_3d,
                PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t5.pred_prob) AS rank_5d,
                PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t10.pred_prob) AS rank_10d,
                m.name,
                m.pre_close,
                m.open,
                m.close,
                m.amount,
                m.turnover_rate,
                m.total_mv,
                m.atr_qfq,
                m.limit_times,
                m.ST_TYPE AS st_type,
                m.ST_TYPE_name AS st_type_name
            FROM {_quote_ident(source_10d['table'])} t10
            JOIN pred5d.{_quote_ident(source_5d['table'])} t5
              ON t10.trade_date = t5.trade_date AND t10.stock_code = t5.stock_code
            JOIN pred3d.{_quote_ident(source_3d['table'])} t3
              ON t10.trade_date = t3.trade_date AND t10.stock_code = t3.stock_code
            LEFT JOIN market.STOCK_DAILY_DATA m
              ON t10.trade_date = m.trade_date AND t10.stock_code = m.stock_code
            WHERE t10.trade_date = ?
            """,
            (str(signal_date),),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _apply_filters(rows: list[dict[str, Any]], rules: dict[str, Any]) -> list[dict[str, Any]]:
    selection = rules["selection_rule"]
    weights = rules["model_input"]["entry_weights"]
    filtered: list[dict[str, Any]] = []
    for row in rows:
        stock_code = str(row.get("stock_code") or "")
        if selection.get("exclude_bj", True) and _is_bj(stock_code):
            continue
        if selection.get("exclude_st_risk_warning", True) and _is_st_like(row):
            continue
        if selection.get("exclude_delist", True) and _is_delisting_name(row.get("name")):
            continue
        if selection.get("skip_open_limit_up_buy", True) and _as_float(row.get("limit_times"), 0.0) not in (None, 0.0):
            continue
        amount = _as_float(row.get("amount"))
        total_mv = _as_float(row.get("total_mv"))
        turnover_rate = _as_float(row.get("turnover_rate"))
        if amount is None or amount < float(selection["amount_min"]):
            continue
        if total_mv is None or total_mv < float(selection["total_mv_min"]):
            continue
        max_total_mv = selection.get("total_mv_max")
        if max_total_mv not in (None, "") and total_mv > float(max_total_mv):
            continue
        min_turnover = selection.get("turnover_min")
        if min_turnover not in (None, "") and (turnover_rate is None or turnover_rate < float(min_turnover)):
            continue
        entry_score = (
            float(weights["10d"]) * float(row["rank_10d"])
            + float(weights["5d"]) * float(row["rank_5d"])
            + float(weights["3d"]) * float(row["rank_3d"])
        )
        out = dict(row)
        out["entry_score"] = entry_score
        filtered.append(out)
    filtered.sort(key=lambda item: (-float(item["entry_score"]), str(item["stock_code"])))
    return filtered


def _build_signal_row(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    rules: dict[str, Any],
    signal_date: str,
    buy_date: str | None,
    buy_day_row: dict[str, Any] | None,
    latest_market_date: str,
) -> dict[str, Any]:
    position = rules["position_rule"]
    holding = rules["holding_rule"]
    risk = rules["risk_rule"]
    return {
        "signal_date": signal_date,
        "buy_date": buy_date or "",
        "symbol": _to_gm_symbol(str(candidate["stock_code"])),
        "stock_code": str(candidate["stock_code"]),
        "name": candidate.get("name"),
        "rank": 1,
        "pred_prob": candidate.get("entry_score"),
        "entry_score": candidate.get("entry_score"),
        "pred_3d": candidate.get("pred_3d"),
        "pred_5d": candidate.get("pred_5d"),
        "pred_10d": candidate.get("pred_10d"),
        "rank_3d": candidate.get("rank_3d"),
        "rank_5d": candidate.get("rank_5d"),
        "rank_10d": candidate.get("rank_10d"),
        "amount": candidate.get("amount"),
        "turnover_rate": candidate.get("turnover_rate"),
        "total_mv": candidate.get("total_mv"),
        "atr_qfq": candidate.get("atr_qfq"),
        "target_pct": f"{float(position['target_position_pct']):.5f}",
        "holding_days": int(holding["holding_days"]),
        "max_holding_days": int(holding["max_holding_days"]),
        "score_exit_entry_ratio": f"{float(holding['score_exit_entry_ratio']):.5f}",
        "min_holding_days_before_score_exit": int(holding["min_holding_days_before_score_exit"]),
        "score_continue_entry_ratio": f"{float(holding['score_continue_entry_ratio']):.5f}",
        "signal_stop_loss_pct": f"{float(risk['intraday_stop_loss_pct']):.5f}",
        "signal_take_profit_pct": f"{float(risk['take_profit_pct']):.5f}",
        "strategy_variant": manifest["strategy_id"],
        "filter_name": rules["selection_rule"]["filter_name"],
        "entry_weight_name": rules["model_input"]["weight_name"],
        "dynamic_hold_name": rules["holding_rule"]["holding_name"],
        "buy_day_market_available": bool(buy_day_row),
        "buy_day_hard_gate_complete": bool(buy_day_row),
        "buy_day_st_rejected": bool(buy_day_row and _is_st_like(buy_day_row)),
        "buy_day_open_limit_up_rejected": bool(buy_day_row and _is_limit_buy(buy_day_row)),
        "latest_market_date": latest_market_date,
    }


def export_signals(
    strategy_dir: Path,
    output: Path,
    status_output: Path | None = None,
    signal_date: str | None = None,
    buy_date: str | None = None,
) -> dict[str, Any]:
    manifest, rules, sources = _load_strategy(strategy_dir)
    market_db = Path(str(resolve_project_path(manifest["input_contract"]["market_db_path"])))
    actual_signal_date = str(signal_date or _common_latest_trade_date(sources))
    auto_buy_date, latest_market_date = _next_trade_date(market_db, actual_signal_date)
    actual_buy_date = str(buy_date or auto_buy_date or "")

    candidates = _apply_filters(
        _build_candidates(sources["3d"], sources["5d"], sources["10d"], market_db, actual_signal_date),
        rules,
    )
    if not candidates:
        raise RuntimeError(f"no candidate left after filters for signal_date={actual_signal_date}")

    skipped: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    chosen_buy_day: dict[str, Any] | None = None
    for candidate in candidates:
        buy_day_row = _load_buy_day_market_row(market_db, actual_buy_date or None, str(candidate["stock_code"]))
        if buy_day_row and (_is_st_like(buy_day_row) or _is_limit_buy(buy_day_row)):
            skipped.append(
                {
                    "stock_code": str(candidate["stock_code"]),
                    "name": candidate.get("name"),
                    "reason": "buy_day_hard_gate_rejected",
                }
            )
            continue
        chosen = candidate
        chosen_buy_day = buy_day_row
        break
    if chosen is None:
        raise RuntimeError(f"no candidate passed buy-day hard gate for signal_date={actual_signal_date}")

    row = _build_signal_row(
        chosen,
        manifest,
        rules,
        actual_signal_date,
        actual_buy_date or None,
        chosen_buy_day,
        latest_market_date,
    )
    rows = [row]
    _write_csv(output, rows)

    status = {
        "strategy_id": manifest["strategy_id"],
        "signal_date": actual_signal_date,
        "buy_date": actual_buy_date,
        "latest_market_date": latest_market_date,
        "buy_day_market_available": bool(chosen_buy_day),
        "buy_day_hard_gate_complete": bool(chosen_buy_day),
        "candidate_count_after_filters": len(candidates),
        "skipped_candidates": skipped[:20],
        "selected_stock_code": row["stock_code"],
        "selected_name": row["name"],
    }
    if status_output is not None:
        _write_json(status_output, status)
    return {"rows": rows, "status": status}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latest formal dynamic top1 signals.")
    parser.add_argument("--strategy-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status-output")
    parser.add_argument("--signal-date")
    parser.add_argument("--buy-date")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = export_signals(
        strategy_dir=resolve_project_path(args.strategy_dir),
        output=Path(str(resolve_project_path(args.output))),
        status_output=Path(str(resolve_project_path(args.status_output))) if args.status_output else None,
        signal_date=args.signal_date,
        buy_date=args.buy_date,
    )
    print(json.dumps(result["status"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
