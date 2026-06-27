from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from prediction_manifest import load_prediction_source_manifest
from project_paths import resolve_project_path
from selection_module import SelectionConfig


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _matches_rule(value: float | None, rule: dict[str, Any]) -> bool:
    if value is None:
        return False
    op = str(rule.get("op") or "").strip()
    if op == "lt":
        return value < float(rule["threshold"])
    if op == "le":
        return value <= float(rule["threshold"])
    if op == "gt":
        return value > float(rule["threshold"])
    if op == "ge":
        return value >= float(rule["threshold"])
    if op == "between":
        return value >= float(rule["lower"]) and value < float(rule["upper"])
    if op == "eq":
        return abs(value - float(rule["threshold"])) < 1e-12
    raise ValueError(f"unsupported target adjustment op: {op}")


def _target_adjusted_pct(
    base_target: float,
    feature_row: dict[str, Any],
    target_adjustment: dict[str, Any] | None,
    max_single_position_pct: float,
) -> tuple[float, float | None, float]:
    if not target_adjustment:
        return base_target, None, 1.0
    pred_5d = _to_float(feature_row.get("pred_5d"))
    pred_10d = _to_float(feature_row.get("pred_10d"))
    pred_gap = abs(pred_10d - pred_5d) if pred_5d is not None and pred_10d is not None else None
    values = {
        "pred_gap": pred_gap,
        "pred_gap_abs_10d_minus_5d": pred_gap,
        "pred_5d": pred_5d,
        "pred_10d": pred_10d,
    }
    scale = 1.0
    for rule in target_adjustment.get("rules_structured", []):
        if _matches_rule(_to_float(values.get(str(rule.get("field")))), rule):
            scale *= float(rule["scale"])
    cap = float(target_adjustment.get("max_single_position_pct_after_adjustment") or max_single_position_pct)
    return max(0.0, min(base_target * scale, cap)), pred_gap, scale


def _cap_day_target_sum(rows: list[dict[str, Any]], cap: float) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("signal_date") or ""), []).append(row)
    for day_rows in grouped.values():
        total = sum(_to_float(row.get("target_pct"), 0.0) or 0.0 for row in day_rows)
        if total <= cap or total <= 0:
            continue
        multiplier = cap / total
        for row in day_rows:
            target = (_to_float(row.get("target_pct"), 0.0) or 0.0) * multiplier
            row["target_pct"] = f"{target:.5f}"
            row["day_target_cap_multiplier"] = f"{multiplier:.10f}"


def _is_missing(value: Any) -> bool:
    if value in (None, "", "None"):
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _is_bj(row: dict[str, Any]) -> bool:
    stock_code = str(row.get("stock_code") or "").strip().upper()
    return stock_code.endswith(".BJ") or stock_code.startswith("BJSE.")


def _is_st(row: dict[str, Any]) -> bool:
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


def _is_delisting(row: dict[str, Any]) -> bool:
    name = str(row.get("name") or "")
    return "\u9000\u5e02" in name or name.startswith("\u9000") or name.endswith("\u9000")


def _is_current_limit(row: dict[str, Any]) -> bool:
    value = row.get("limit_times")
    if _is_missing(value):
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return True


def _pre_filter_rows(rows: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if bool(selection["exclude_bj"]) and _is_bj(row):
            continue
        if bool(selection["exclude_st"]) and _is_st(row):
            continue
        if bool(selection["exclude_delisting"]) and _is_delisting(row):
            continue
        if bool(selection["exclude_current_limit"]) and _is_current_limit(row):
            continue
        filtered.append(row)
    return filtered


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if "." in table:
        schema, table_name = table.split(".", 1)
        sql = f"PRAGMA {_quote_ident(schema)}.table_info({_quote_ident(table_name)})"
    else:
        sql = f"PRAGMA table_info({_quote_ident(table)})"
    return {str(row[1]) for row in conn.execute(sql).fetchall()}


def _optional_column(columns: set[str], alias: str, column: str, output: str | None = None) -> str:
    output_name = output or column
    if column in columns:
        return f"{alias}.{_quote_ident(column)} AS {_quote_ident(output_name)}"
    return f"NULL AS {_quote_ident(output_name)}"


def _pool_formula_sql(pool: dict[str, Any]) -> str:
    weight_10d = float(pool["rank_weight_10d"])
    weight_5d = float(pool["rank_weight_5d"])
    return f"(rank_10d * {weight_10d:.12g}) + (rank_5d * {weight_5d:.12g})"


def _pool_condition_sql(alias: str, pool: dict[str, Any], columns: set[str] | None = None) -> tuple[str, list[object]]:
    columns = columns or set()
    checks = [
        f"{alias}.trade_date >= ?",
        f"{alias}.trade_date <= ?",
        f"{alias}.stock_code NOT LIKE '%.BJ'",
        f"COALESCE({alias}.name, '') NOT LIKE 'ST%'",
        f"COALESCE({alias}.name, '') NOT LIKE '*ST%'",
        f"COALESCE({alias}.name, '') NOT LIKE ?",
        f"COALESCE({alias}.name, '') NOT LIKE ?",
        f"COALESCE({alias}.name, '') NOT LIKE ?",
        f"({alias}.limit_times IS NULL OR {alias}.limit_times = '' OR {alias}.limit_times = 'None' OR CAST({alias}.limit_times AS REAL) = 0)",
        f"{alias}.rank_5d IS NOT NULL",
        f"{alias}.rank_10d IS NOT NULL",
        f"{alias}.total_mv IS NOT NULL AND {alias}.total_mv <= ?",
        f"{alias}.amount IS NOT NULL AND {alias}.amount >= ?",
        f"{alias}.turnover_rate IS NOT NULL AND {alias}.turnover_rate >= ?",
        f"{alias}.close IS NOT NULL",
    ]
    values: list[object] = [
        "%\u9000\u5e02%",
        "\u9000%",
        "%\u9000",
        float(pool["max_total_mv"]),
        float(pool["min_amount"]),
        float(pool["min_turnover_rate"]),
    ]
    if "st_type_name" in columns:
        checks.insert(8, f"(COALESCE({alias}.st_type_name, '') NOT LIKE '%风险%')")
    if "st_type" in columns:
        checks.insert(
            8,
            f"({alias}.st_type IS NULL OR {alias}.st_type = '' OR {alias}.st_type = 'None' OR UPPER(CAST({alias}.st_type AS TEXT)) IN ('0', '0.0', 'FALSE', 'NONE', 'NAN'))",
        )
    return " AND ".join(checks), values


def _load_pool_rows(
    fusion_db: Path,
    pool: dict[str, Any],
    start: str,
    end: str,
    tier_offset: float,
) -> list[dict[str, Any]]:
    formula = _pool_formula_sql(pool)
    conn = sqlite3.connect(fusion_db)
    conn.row_factory = sqlite3.Row
    try:
        columns = _table_columns(conn, "fusion_rank_base")
        where, condition_values = _pool_condition_sql("b", pool, columns)
        st_type_select = _optional_column(columns, "b", "st_type")
        st_type_name_select = _optional_column(columns, "b", "st_type_name")
        rows = conn.execute(
            f"""
            SELECT *
            FROM (
                SELECT
                    b.trade_date,
                    b.stock_code,
                    {tier_offset:.12g} + ({formula}) AS pred_prob,
                    b.pred_10d,
                    b.pred_5d,
                    b.name,
                    b.pre_close,
                    b.open,
                    b.close,
                    b.amount,
                    b.turnover_rate,
                    b.total_mv,
                    b.atr_qfq,
                    b.limit_times,
                    {st_type_select},
                    {st_type_name_select},
                    b.rank_5d,
                    b.rank_10d,
                    ROW_NUMBER() OVER (
                        PARTITION BY b.trade_date
                        ORDER BY ({formula}) DESC, b.stock_code
                    ) AS rn
                FROM fusion_rank_base b
                WHERE {where}
            )
            WHERE rn <= ?
            ORDER BY trade_date, pred_prob DESC, stock_code
            """,
            [str(start), str(end), *condition_values, int(pool["limit"])],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _group_rows_by_date(rows: list[dict[str, Any]], date_key: str = "trade_date") -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        date = str(row.get(date_key) or "")
        if date:
            grouped.setdefault(date, []).append(row)
    return grouped


def _merge_pool_rows_for_dates(
    primary_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    dates: set[str],
) -> list[dict[str, Any]]:
    primary_by_date = _group_rows_by_date(primary_rows)
    fallback_by_date = _group_rows_by_date(fallback_rows)
    selected: list[dict[str, Any]] = []
    for trade_date in sorted(dates):
        seen: set[tuple[Any, Any]] = set()
        for row in [*primary_by_date.get(trade_date, []), *fallback_by_date.get(trade_date, [])]:
            key = (row.get("trade_date"), row.get("stock_code"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    return selected


def _build_pool_signals(
    rows: list[dict[str, Any]],
    market_rows: dict[str, dict[str, dict]],
    holding_days: int,
    max_positions: int,
    target_pct: float,
) -> list[dict[str, Any]]:
    return build_gm_signal_rows(
        rows,
        config=SelectionConfig(
            top_k=5,
            pred_col="pred_prob",
            min_pred_prob=None,
            min_pred_quantile=None,
            max_atr_ratio=None,
            min_amount=None,
            min_turnover_rate=None,
            max_total_mv=None,
            max_per_industry=999999,
            exclude_bj=True,
            exclude_st=True,
            exclude_delisting=True,
            exclude_current_limit=True,
        ),
        market_rows_by_trade_date=market_rows,
        holding_days=holding_days,
        max_positions=max_positions,
        weight_mode="equal",
        target_total_pct=target_pct * float(max_positions),
    )


def _build_base_signals_from_pool_rule(
    fusion_db: Path,
    pool_rule: dict[str, Any],
    start: str,
    end: str,
    market_rows: dict[str, dict[str, dict]],
    holding_days: int,
    max_positions: int,
) -> list[dict[str, Any]]:
    target_pct = float(pool_rule.get("base_target_pct", 0.10925))
    primary_pool = dict(pool_rule["primary_pool"])
    normal_fallback_pool = dict(pool_rule["normal_fallback_pool"])
    weak_fallback_pool = dict(pool_rule.get("weak_fallback_pool", normal_fallback_pool))
    primary_offset = float(pool_rule.get("primary_tier_offset", 2.0))
    fallback_offset = float(pool_rule.get("fallback_tier_offset", 1.0))

    primary_rows = _load_pool_rows(fusion_db, primary_pool, start, end, primary_offset)
    normal_fallback_rows = _load_pool_rows(fusion_db, normal_fallback_pool, start, end, fallback_offset)
    weak_fallback_rows = _load_pool_rows(fusion_db, weak_fallback_pool, start, end, fallback_offset)

    all_dates = set(_group_rows_by_date(normal_fallback_rows))
    baseline_rows = _merge_pool_rows_for_dates(primary_rows, normal_fallback_rows, all_dates)
    baseline_signals = _build_pool_signals(
        baseline_rows,
        market_rows,
        holding_days=holding_days,
        max_positions=max_positions,
        target_pct=target_pct,
    )
    baseline_features = _signal_day_features(baseline_signals)
    weak_rule = pool_rule.get("weak_day_rule", {})
    weak_mode = str(weak_rule.get("mode", "none"))
    if weak_mode == "none":
        weak_dates: set[str] = set()
    elif weak_mode == "primary_count_le1":
        weak_dates = {
            signal_date
            for signal_date, item in baseline_features.items()
            if int(item.get("primary_count") or 0) <= 1
        }
    else:
        raise ValueError(f"unsupported weak_day_rule.mode: {weak_mode}")

    normal_dates = all_dates - weak_dates
    selected_rows = [
        *_merge_pool_rows_for_dates(primary_rows, normal_fallback_rows, normal_dates),
        *_merge_pool_rows_for_dates(primary_rows, weak_fallback_rows, weak_dates),
    ]
    base_signals = _build_pool_signals(
        selected_rows,
        market_rows,
        holding_days=holding_days,
        max_positions=max_positions,
        target_pct=target_pct,
    )
    for signal in base_signals:
        signal["target_pct"] = f"{target_pct:.5f}"
        signal["score_exit_entry_ratio"] = "1.0"
        signal["min_holding_days_before_score_exit"] = "3"
        signal["weak_mode"] = weak_mode
    return base_signals


def _resolve_start_for_lookback(
    score_db: Path,
    score_table: str,
    requested_start: str,
    end: str,
    lookback_trade_days: int | None,
) -> str:
    if lookback_trade_days is None or lookback_trade_days <= 0:
        return str(requested_start)
    conn = sqlite3.connect(score_db)
    try:
        rows = conn.execute(
            f"""
            SELECT DISTINCT trade_date
            FROM {_quote_ident(score_table)}
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
            """,
            (str(requested_start), str(end)),
        ).fetchall()
    finally:
        conn.close()
    dates = [str(row[0]) for row in rows]
    if not dates:
        return str(requested_start)
    index = max(0, len(dates) - int(lookback_trade_days) - 1)
    return dates[index]


def _load_rows(score_db: Path, score_table: str, fusion_db: Path, start: str, end: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(score_db)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ATTACH DATABASE ? AS fusion", (str(fusion_db),))
        columns = _table_columns(conn, "fusion.fusion_rank_base")
        st_type_select = _optional_column(columns, "f", "st_type")
        st_type_name_select = _optional_column(columns, "f", "st_type_name")
        query = f"""
            SELECT
                s.trade_date,
                s.stock_code,
                s.pred_prob,
                f.pred_10d,
                f.pred_5d,
                f.name,
                f.pre_close,
                f.open,
                f.close,
                f.amount,
                f.turnover_rate,
                f.total_mv,
                f.atr_qfq,
                f.limit_times,
                {st_type_select},
                {st_type_name_select},
                f.rank_5d,
                f.rank_10d
            FROM {_quote_ident(score_table)} s
            LEFT JOIN fusion.fusion_rank_base f
              ON s.trade_date = f.trade_date AND s.stock_code = f.stock_code
            WHERE s.trade_date >= ? AND s.trade_date <= ?
            ORDER BY s.trade_date, s.pred_prob DESC, s.stock_code
        """
        return [dict(row) for row in conn.execute(query, (str(start), str(end))).fetchall()]
    finally:
        conn.close()


def _signal_day_features(signals: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in signals:
        grouped.setdefault(str(row.get("signal_date") or ""), []).append(row)
    features: dict[str, dict[str, float | int | None]] = {}
    for signal_date, rows in grouped.items():
        preds = [_to_float(row.get("pred_prob"), 0.0) or 0.0 for row in rows]
        features[signal_date] = {
            "count": len(rows),
            "primary_count": sum(1 for value in preds if value >= 2.0),
            "avg_pred": sum(preds) / len(preds) if preds else None,
        }
    return features


def _post_filter_day_features(signals: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    return _signal_day_features(signals)


def _feature_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("trade_date") or ""), str(row.get("stock_code") or "")): row
        for row in rows
        if row.get("trade_date") and row.get("stock_code")
    }


def _build_signals(
    strategy_dir: Path,
    start: str,
    end: str,
    market_db_override: str | None = None,
    lookback_trade_days: int | None = None,
) -> list[dict[str, Any]]:
    manifest = _load_json(strategy_dir / "strategy_manifest.json")
    rules = _load_json(strategy_dir / "trading_rules.json")
    input_contract = manifest["input_contract"]

    load_prediction_source_manifest(
        input_contract["formal_manifest_5d"],
        require_approved=True,
        allow_legacy=False,
    )
    load_prediction_source_manifest(
        input_contract["formal_manifest_10d"],
        require_approved=True,
        allow_legacy=False,
    )

    asset_base_dir = strategy_dir
    score_asset = rules["score_rule"]["score_asset"]
    fusion_asset = rules["score_rule"]["fusion_asset"]
    execution = rules["execution_rule"]
    selection = rules["selection_rule"]
    position = rules["position_rule"]
    holding = rules["holding_rule"]

    score_db = _resolve_path(score_asset["db_path"], asset_base_dir)
    fusion_db = _resolve_path(fusion_asset["db_path"], asset_base_dir)
    market_db = _resolve_path(
        market_db_override or execution["market_db_path"],
        asset_base_dir,
    )

    start = _resolve_start_for_lookback(score_db, score_asset["table"], start, end, lookback_trade_days)
    market_rows = load_market_rows_by_trade_date(market_db, start, end)
    rows = _pre_filter_rows(_load_rows(score_db, score_asset["table"], fusion_db, start, end), selection)
    feature_by_key = _feature_lookup(rows)
    pool_rule = rules["score_rule"].get("pool_generation_rule")
    if pool_rule:
        base_signals = _build_base_signals_from_pool_rule(
            fusion_db,
            pool_rule,
            start,
            end,
            market_rows,
            holding_days=int(holding["holding_days"]),
            max_positions=int(position["max_positions"]),
        )
    else:
        base_signals = build_gm_signal_rows(
            rows,
            config=SelectionConfig(
                top_k=int(selection["top_k_after_base_pool"]),
                pred_col="pred_prob",
                min_pred_prob=None,
                min_pred_quantile=None,
                max_atr_ratio=None,
                min_amount=None,
                min_turnover_rate=None,
                max_total_mv=None,
                max_per_industry=999999,
                exclude_bj=bool(selection["exclude_bj"]),
                exclude_st=bool(selection["exclude_st"]),
                exclude_delisting=bool(selection["exclude_delisting"]),
                exclude_current_limit=bool(selection["exclude_current_limit"]),
            ),
            market_rows_by_trade_date=market_rows,
            holding_days=int(holding["holding_days"]),
            max_positions=int(position["max_positions"]),
            weight_mode="equal",
            target_total_pct=1.0,
        )

    features = _signal_day_features(base_signals)
    rank_targets = {int(k): float(v) for k, v in position["rank_target_pct"].items()}
    target_adjustment = rules["score_rule"].get("target_adjustment")
    max_single_position_pct = float(position.get("max_single_position_pct") or max(rank_targets.values()))
    pre_avg_filtered: list[dict[str, Any]] = []
    for row in base_signals:
        signal_date = str(row.get("signal_date") or "")
        day_features = features.get(signal_date, {})
        rank = int(float(row.get("rank") or 999999))
        feature_row = feature_by_key.get((signal_date, str(row.get("stock_code") or "")), {})
        pred_10d = _to_float(feature_row.get("pred_10d"))
        if int(day_features.get("primary_count") or 0) < int(selection["primary_count_min"]):
            continue
        if rank > int(selection["rank_max"]):
            continue
        if pred_10d is None or pred_10d < float(selection["min_pred_10d"]):
            continue
        output_row = dict(row)
        target_pct, pred_gap, target_adjustment_scale = _target_adjusted_pct(
            rank_targets.get(rank, 0.0),
            feature_row,
            target_adjustment,
            max_single_position_pct,
        )
        output_row["target_pct"] = f"{target_pct:.5f}"
        output_row["target_adjustment_scale"] = f"{target_adjustment_scale:.6f}"
        output_row["score_exit_entry_ratio"] = str(holding["score_exit_entry_ratio"])
        output_row["min_holding_days_before_score_exit"] = str(holding["min_holding_days_before_score_exit"])
        output_row["max_daily_sells"] = str(holding["max_daily_sells"])
        output_row["score_continue_entry_ratio"] = str(holding["score_continue_entry_ratio"])
        output_row["weak_mode"] = rules["score_rule"]["base_pool"]["weak_day_rule"]["mode"]
        output_row["pred_10d"] = feature_row.get("pred_10d")
        output_row["pred_5d"] = feature_row.get("pred_5d")
        output_row["pred_gap"] = "" if pred_gap is None else f"{pred_gap:.10f}"
        pre_avg_filtered.append(output_row)
    post_features = _post_filter_day_features(pre_avg_filtered)
    filtered: list[dict[str, Any]] = []
    for row in pre_avg_filtered:
        avg_pred = _to_float(post_features.get(str(row.get("signal_date") or ""), {}).get("avg_pred"))
        if avg_pred is None or avg_pred < float(selection["post_filter_avg_pred_min"]):
            continue
        filtered.append(row)
    if target_adjustment:
        _cap_day_target_sum(filtered, float(position.get("cash_buffer") or 0.99))
    return filtered


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export formal 5D+10D L5 production strategy signals.")
    parser.add_argument("--strategy-dir", required=True)
    parser.add_argument("--market-db")
    parser.add_argument("--start", default="20240604")
    parser.add_argument("--end", default="20991231")
    parser.add_argument("--lookback-trade-days", type=int)
    parser.add_argument("--output", default="../data_file/production_signals/prod_formal_5d10d_gap_v20260622_latest.csv")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    strategy_dir = resolve_project_path(args.strategy_dir)
    signals = _build_signals(
        strategy_dir,
        args.start,
        args.end,
        args.market_db,
        lookback_trade_days=args.lookback_trade_days,
    )
    dates = sorted({str(row.get("buy_date") or "") for row in signals if row.get("buy_date")})
    print(f"signals: {len(signals)}")
    print(f"buy_days: {len(dates)}")
    if dates:
        print(f"buy_date_range: {dates[0]}-{dates[-1]}")
    if args.validate_only:
        return 0
    write_gm_signals_csv(signals, args.output)
    print(f"output: {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
