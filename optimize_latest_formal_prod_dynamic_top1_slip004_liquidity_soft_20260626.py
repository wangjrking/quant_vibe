from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import optimize_latest_formal_prod_dynamic_top1_20260625 as base
from export_dynamic_top1_formal_signals import (
    _build_candidates,
    _build_signal_row,
    _common_latest_trade_date,
    _is_bj,
    _is_delisting_name,
    _is_limit_buy,
    _is_st_like,
    _load_buy_day_market_row,
    _load_strategy,
    _next_trade_date,
)


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA = ROOT / "quant" / "data_file"
MARKET_DB = DATA / "STOCK_DAILY_DATA.db"

base.REPORT_DIR = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_slip004_liquidity_soft_20260626"
base.SCORE_DB = base.REPORT_DIR / "scores" / "grid_scores.db"
base.BACKTEST_SLIPPAGE_RATIO = "0.0040"

ENTRY_CASES = [
    {
        "case_name": "raw_base_amt90_mv20",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 90000.0,
        "total_mv_min": 200000.0,
        "amount_bonus": 0.00,
        "mv_bonus": 0.00,
        "atr_penalty": 0.00,
    },
    {
        "case_name": "liqsoft_a1_mv05_amt90_mv20",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 90000.0,
        "total_mv_min": 200000.0,
        "amount_bonus": 0.010,
        "mv_bonus": 0.005,
        "atr_penalty": 0.000,
    },
    {
        "case_name": "liqsoft_a15_mv10_amt90_mv20",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 90000.0,
        "total_mv_min": 200000.0,
        "amount_bonus": 0.015,
        "mv_bonus": 0.010,
        "atr_penalty": 0.000,
    },
    {
        "case_name": "liqsoft_a15_mv10_atr05_amt90_mv20",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 90000.0,
        "total_mv_min": 200000.0,
        "amount_bonus": 0.015,
        "mv_bonus": 0.010,
        "atr_penalty": 0.005,
    },
    {
        "case_name": "liqsoft_a20_mv10_atr10_amt90_mv20",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 90000.0,
        "total_mv_min": 200000.0,
        "amount_bonus": 0.020,
        "mv_bonus": 0.010,
        "atr_penalty": 0.010,
    },
    {
        "case_name": "liqsoft_a20_mv15_atr05_amt120_mv30",
        "weight_name": "w84_5d09_3d07",
        "w10d": 0.84,
        "w5d": 0.09,
        "w3d": 0.07,
        "amount_min": 120000.0,
        "total_mv_min": 300000.0,
        "amount_bonus": 0.020,
        "mv_bonus": 0.015,
        "atr_penalty": 0.005,
    },
]

EXEC_CASES = [
    {
        "case_name": "h2_m3_c097_e097_pos90_dd12",
        "target_position_pct": 0.90,
        "score_continue_entry_ratio": 0.97,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 3,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.085,
        "dd_hard_trigger": 0.12,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.78,
        "dd_hard_scale": 0.58,
    },
    {
        "case_name": "h2_m4_c0975_e097_pos90_dd12",
        "target_position_pct": 0.90,
        "score_continue_entry_ratio": 0.975,
        "score_exit_entry_ratio": 0.97,
        "holding_days": 2,
        "max_holding_days": 4,
        "min_holding_days_before_score_exit": 1,
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.07,
        "dd_soft_trigger": 0.085,
        "dd_hard_trigger": 0.12,
        "dd_recover_trigger": 0.03,
        "dd_soft_scale": 0.78,
        "dd_hard_scale": 0.58,
    },
]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _percent_ranks(values: list[float | None]) -> list[float]:
    indexed = [(index, value) for index, value in enumerate(values) if value is not None and not math.isnan(float(value))]
    if not indexed:
        return [0.0] * len(values)
    indexed.sort(key=lambda item: float(item[1]))
    n = len(indexed)
    out = [0.0] * len(values)
    if n == 1:
        out[indexed[0][0]] = 1.0
        return out
    for rank, (index, _) in enumerate(indexed):
        out[index] = rank / (n - 1)
    return out


def _base_score(row: dict[str, Any], entry_case: dict[str, Any]) -> float:
    return (
        float(entry_case["w10d"]) * float(row["rank_10d"])
        + float(entry_case["w5d"]) * float(row["rank_5d"])
        + float(entry_case["w3d"]) * float(row["rank_3d"])
    )


def _apply_filters_soft(rows: list[dict[str, Any]], entry_case: dict[str, Any]) -> list[dict[str, Any]]:
    prelim: list[dict[str, Any]] = []
    for row in rows:
        stock_code = str(row.get("stock_code") or "")
        if _is_bj(stock_code):
            continue
        if _is_st_like(row):
            continue
        if _is_delisting_name(row.get("name")):
            continue
        limit_times = row.get("limit_times")
        if limit_times not in (None, "", 0, 0.0, "0", "0.0"):
            continue
        amount = row.get("amount")
        total_mv = row.get("total_mv")
        if amount is None or float(amount) < float(entry_case["amount_min"]):
            continue
        if total_mv is None or float(total_mv) < float(entry_case["total_mv_min"]):
            continue
        prelim.append(dict(row))

    amount_ranks = _percent_ranks([float(row["amount"]) if row.get("amount") is not None else None for row in prelim])
    mv_ranks = _percent_ranks([float(row["total_mv"]) if row.get("total_mv") is not None else None for row in prelim])
    atr_ranks = _percent_ranks([float(row["atr_qfq"]) if row.get("atr_qfq") not in (None, "") else None for row in prelim])

    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(prelim):
        raw_score = _base_score(row, entry_case)
        liquidity_score = (
            float(entry_case["amount_bonus"]) * amount_ranks[index]
            + float(entry_case["mv_bonus"]) * mv_ranks[index]
            - float(entry_case["atr_penalty"]) * atr_ranks[index]
        )
        out = dict(row)
        out["raw_entry_score"] = raw_score
        out["liquidity_score"] = liquidity_score
        out["entry_score"] = raw_score + liquidity_score
        out["amount_rank_soft"] = amount_ranks[index]
        out["mv_rank_soft"] = mv_ranks[index]
        out["atr_rank_soft"] = atr_ranks[index]
        enriched.append(out)

    enriched.sort(key=lambda item: (-float(item["entry_score"]), str(item["stock_code"])))
    return enriched


def _build_case_rules(base_rules: dict[str, Any], entry_case: dict[str, Any], exec_case: dict[str, Any]) -> dict[str, Any]:
    rules = json.loads(json.dumps(base_rules, ensure_ascii=False))
    rules["model_input"]["weight_name"] = entry_case["weight_name"]
    rules["model_input"]["entry_weights"] = {
        "10d": entry_case["w10d"],
        "5d": entry_case["w5d"],
        "3d": entry_case["w3d"],
    }
    rules["selection_rule"]["filter_name"] = entry_case["case_name"]
    rules["selection_rule"]["amount_min"] = entry_case["amount_min"]
    rules["selection_rule"]["total_mv_min"] = entry_case["total_mv_min"]
    rules["position_rule"]["target_position_pct"] = exec_case["target_position_pct"]
    rules["holding_rule"]["score_continue_entry_ratio"] = exec_case["score_continue_entry_ratio"]
    rules["holding_rule"]["score_exit_entry_ratio"] = exec_case["score_exit_entry_ratio"]
    rules["holding_rule"]["holding_days"] = exec_case["holding_days"]
    rules["holding_rule"]["max_holding_days"] = exec_case["max_holding_days"]
    rules["holding_rule"]["min_holding_days_before_score_exit"] = exec_case["min_holding_days_before_score_exit"]
    rules["risk_rule"]["intraday_stop_loss_pct"] = exec_case["stop_loss_pct"]
    rules["risk_rule"]["take_profit_pct"] = exec_case["take_profit_pct"]
    rules["risk_rule"]["dd_soft_trigger"] = exec_case["dd_soft_trigger"]
    rules["risk_rule"]["dd_hard_trigger"] = exec_case["dd_hard_trigger"]
    rules["risk_rule"]["dd_recover_trigger"] = exec_case["dd_recover_trigger"]
    rules["risk_rule"]["dd_soft_scale"] = exec_case["dd_soft_scale"]
    rules["risk_rule"]["dd_hard_scale"] = exec_case["dd_hard_scale"]
    return rules


def _case_key(entry_case: dict[str, Any], exec_case: dict[str, Any]) -> str:
    return f"{entry_case['case_name']}__{exec_case['case_name']}"


def _signal_dates(sources: dict[str, dict[str, Any]], latest_signal_date: str) -> list[str]:
    conn = sqlite3.connect(str(sources["10d"]["db_path"]))
    try:
        rows = conn.execute(
            f"""
            SELECT DISTINCT trade_date
            FROM "{sources['10d']['table']}"
            WHERE trade_date <= ?
            ORDER BY trade_date
            """,
            (str(latest_signal_date),),
        ).fetchall()
    finally:
        conn.close()
    return [str(row[0]) for row in rows]


def _build_signal_and_scores(
    manifest: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    case_rules: dict[str, Any],
    entry_case: dict[str, Any],
    case_key: str,
) -> dict[str, Any]:
    latest_signal_date = _common_latest_trade_date(sources)
    signal_dates = _signal_dates(sources, latest_signal_date)
    signal_rows: list[dict[str, Any]] = []
    score_rows: list[tuple[str, str, float]] = []

    for signal_date in signal_dates:
        buy_date, latest_market_date = _next_trade_date(MARKET_DB, signal_date)
        daily_rows = _build_candidates(sources["3d"], sources["5d"], sources["10d"], MARKED_DB := MARKET_DB, signal_date)
        filtered = _apply_filters_soft(daily_rows, entry_case)
        for row in filtered:
            score_rows.append((signal_date, str(row["stock_code"]), float(row["entry_score"])))
        if not filtered:
            continue

        chosen: dict[str, Any] | None = None
        chosen_buy_day: dict[str, Any] | None = None
        for candidate in filtered:
            buy_day_row = _load_buy_day_market_row(MARKED_DB, buy_date, str(candidate["stock_code"]))
            if buy_day_row and (_is_st_like(buy_day_row) or _is_limit_buy(buy_day_row)):
                continue
            chosen = candidate
            chosen_buy_day = buy_day_row
            break
        if chosen is None:
            continue
        row = _build_signal_row(
            chosen,
            manifest,
            case_rules,
            signal_date,
            buy_date,
            chosen_buy_day,
            latest_market_date,
        )
        row["strategy_variant"] = case_key
        row["raw_entry_score"] = chosen.get("raw_entry_score")
        row["liquidity_score"] = chosen.get("liquidity_score")
        row["amount_rank_soft"] = chosen.get("amount_rank_soft")
        row["mv_rank_soft"] = chosen.get("mv_rank_soft")
        row["atr_rank_soft"] = chosen.get("atr_rank_soft")
        signal_rows.append(row)

    signal_file = base.REPORT_DIR / "signals" / f"{case_key}.csv"
    score_table = f"score_{case_key}".replace("-", "_").replace(".", "_")
    _write_rows(signal_file, signal_rows)
    return {
        "signal_file": signal_file,
        "signal_rows": len(signal_rows),
        "score_rows": score_rows,
        "score_table": score_table,
        "latest_signal_date": latest_signal_date,
    }


def _summary_row(
    entry_case: dict[str, Any],
    exec_case: dict[str, Any],
    case_meta: dict[str, Any],
    slice_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_slice = {row["slice"]: row for row in slice_rows}
    full = by_slice["full"]
    recent120 = by_slice["recent120"]
    recent60 = by_slice["recent60"]
    ytd = by_slice["ytd2026"]
    return {
        "case_key": _case_key(entry_case, exec_case),
        "entry_case": entry_case["case_name"],
        "exec_case": exec_case["case_name"],
        "weight_name": entry_case["weight_name"],
        "w10d": entry_case["w10d"],
        "w5d": entry_case["w5d"],
        "w3d": entry_case["w3d"],
        "amount_min": entry_case["amount_min"],
        "total_mv_min": entry_case["total_mv_min"],
        "amount_bonus": entry_case["amount_bonus"],
        "mv_bonus": entry_case["mv_bonus"],
        "atr_penalty": entry_case["atr_penalty"],
        "target_position_pct": exec_case["target_position_pct"],
        "score_continue_entry_ratio": exec_case["score_continue_entry_ratio"],
        "score_exit_entry_ratio": exec_case["score_exit_entry_ratio"],
        "holding_days": exec_case["holding_days"],
        "max_holding_days": exec_case["max_holding_days"],
        "signal_rows": case_meta["signal_rows"],
        "full_annual": full["annual"],
        "full_sharpe": full["sharpe"],
        "full_max_drawdown": full["max_drawdown"],
        "recent120_annual": recent120["annual"],
        "recent60_annual": recent60["annual"],
        "ytd2026_annual": ytd["annual"],
    }


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    def _f(key: str) -> float:
        value = row.get(key)
        return float(value) if value not in (None, "") else float("-inf")

    dd = row.get("full_max_drawdown")
    dd_value = float(dd) if dd not in (None, "") else float("inf")
    return (
        _f("full_annual"),
        _f("recent60_annual"),
        _f("full_sharpe"),
        -dd_value,
    )


def main() -> None:
    manifest, base_rules, sources = _load_strategy(base.STRATEGY_DIR)
    base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = base.REPORT_DIR / "detail.csv"
    summary_path = base.REPORT_DIR / "summary.csv"
    detail_rows = base._read_rows(detail_path)
    summary_rows = base._read_rows(summary_path)
    completed = {row["case_key"] for row in summary_rows if row.get("case_key")}
    if not completed and base.SCORE_DB.exists():
        base.SCORE_DB.unlink()

    for entry_case in ENTRY_CASES:
        for exec_case in EXEC_CASES:
            case_key = _case_key(entry_case, exec_case)
            if case_key in completed:
                continue
            case_rules = _build_case_rules(base_rules, entry_case, exec_case)
            case_meta = _build_signal_and_scores(manifest, sources, case_rules, entry_case, case_key)
            base._append_score_table(case_meta["score_table"], case_meta["score_rows"])
            slice_rows = [
                base._run_backtest(case_key, case_rules, case_meta["signal_file"], case_meta["score_table"], tag, start, end)
                for tag, start, end in base.TIME_SLICES
            ]
            detail_rows.extend(slice_rows)
            summary_rows.append(_summary_row(entry_case, exec_case, case_meta, slice_rows))
            summary_rows.sort(key=_sort_key, reverse=True)
            _write_rows(detail_path, detail_rows)
            _write_rows(summary_path, summary_rows)
            _write_json(base.REPORT_DIR / "summary.json", summary_rows)

    summary_rows.sort(key=_sort_key, reverse=True)
    _write_rows(detail_path, detail_rows)
    _write_rows(summary_path, summary_rows)
    _write_json(base.REPORT_DIR / "summary.json", summary_rows)
    print(json.dumps({"report_dir": str(base.REPORT_DIR), "cases": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
