from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas_market_calendars as mcal

from adjustment_semantics import (
    MARKET_FIELD_SCHEMA_KIND_STRATEGY,
    default_market_field_semantics,
    same_adjustment_semantics,
    same_market_field_semantics,
    validate_strategy_output_field_names,
    validate_adjustment_semantics,
    validate_market_field_semantics,
)
from prediction_manifest import load_prediction_source_manifest
from stock_daily_data_route import resolve_stock_daily_duckdb_path


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
PRODUCTION_SIGNALS = DATA / "production_signals"
REPORTS = DATA / "reports"
STRATEGY_ID = "prod_dyn_mild_09_12_15_v20260630"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / STRATEGY_ID
SIGNAL_DIR = STRATEGY_DIR / "signals"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    fields = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _validate_strategy_output_rows(rows: list[dict[str, Any]], *, context: str) -> None:
    if not rows:
        return
    validate_strategy_output_field_names(rows[0].keys(), context=context)


def _stock_to_symbol(stock_code: str) -> str:
    if stock_code.endswith(".SZ"):
        return f"SZSE.{stock_code[:6]}"
    if stock_code.endswith(".SH"):
        return f"SHSE.{stock_code[:6]}"
    if stock_code.endswith(".BJ"):
        return f"BJSE.{stock_code[:6]}"
    return stock_code


def _next_trade_date(signal_date: str) -> str:
    calendar = mcal.get_calendar("SSE")
    start = dt.datetime.strptime(signal_date, "%Y%m%d").date()
    end = start + dt.timedelta(days=14)
    schedule = calendar.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    days = [idx.strftime("%Y%m%d") for idx in schedule.index]
    for day in days:
        if day > signal_date:
            return day
    raise RuntimeError(f"cannot resolve next SSE trade date after {signal_date}")


def _target_pct(rank_3d: float, rank_5d: float) -> float:
    agreement = min(rank_3d, rank_5d)
    if agreement >= 0.65:
        return 0.15
    if agreement >= 0.45:
        return 0.12
    return 0.09


def _load_sources(strategy_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = strategy_manifest.get("input_contract", {})
    contract_semantics = validate_adjustment_semantics(
        contract.get("adjustment_semantics"),
        context="strategy_manifest.input_contract",
    )
    contract_market_semantics = validate_market_field_semantics(
        contract.get("market_field_semantics"),
        context="strategy_manifest.input_contract",
        expected_schema_kind=MARKET_FIELD_SCHEMA_KIND_STRATEGY,
    )
    raw_paths = {
        "3d": contract.get("formal_manifest_3d"),
        "5d": contract.get("formal_manifest_5d"),
        "10d": contract.get("formal_manifest_10d"),
    }
    out: dict[str, dict[str, Any]] = {}
    for label, raw_path in raw_paths.items():
        if not raw_path:
            raise RuntimeError(f"missing formal manifest for {label}")
        source = load_prediction_source_manifest(raw_path, require_approved=True, allow_legacy=False)
        if source["source_type"] != "duckdb_table":
            raise RuntimeError(f"{label} manifest must be duckdb_table, got {source['source_type']}")
        manifest_semantics = source.get("adjustment_semantics")
        if not same_adjustment_semantics(contract_semantics, manifest_semantics):
            raise RuntimeError(
                f"{label} manifest adjustment_semantics does not match strategy input contract"
            )
        manifest_market_semantics = source.get("market_field_semantics")
        if not same_market_field_semantics(
            contract_market_semantics,
            manifest_market_semantics,
            expected_schema_kind=MARKET_FIELD_SCHEMA_KIND_STRATEGY,
        ):
            raise RuntimeError(
                f"{label} manifest market_field_semantics does not match strategy input contract"
            )
        out[label] = source
    return out


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _duckdb_literal(value: str | Path) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def _resolve_market_duckdb_path(strategy_manifest: dict[str, Any], sources: dict[str, dict[str, Any]]) -> Path:
    contract = strategy_manifest.get("input_contract", {})
    contract_market = contract.get("market_db_path")
    active_market_path = resolve_stock_daily_duckdb_path(require_exists=False).resolve()
    if contract_market not in (None, ""):
        resolved_contract_market = Path(str(contract_market)).resolve()
        if resolved_contract_market != active_market_path:
            raise RuntimeError(
                "strategy input_contract.market_db_path must match current active L2 DuckDB route: "
                f"{resolved_contract_market} != {active_market_path}"
            )
        return resolved_contract_market
    manifest_paths = {
        Path(str(source["market_db_path"])).resolve()
        for source in sources.values()
        if source.get("market_db_path")
    }
    if len(manifest_paths) > 1:
        raise RuntimeError(f"formal manifests disagree on market_db_path: {sorted(str(item) for item in manifest_paths)}")
    if manifest_paths:
        manifest_market_path = next(iter(manifest_paths))
        if manifest_market_path != active_market_path:
            raise RuntimeError(
                "formal manifests market_db_path must match current active L2 DuckDB route: "
                f"{manifest_market_path} != {active_market_path}"
            )
        return manifest_market_path
    return active_market_path


def _attach_duckdb_sources(
    con: duckdb.DuckDBPyConnection,
    *,
    sources: dict[str, dict[str, Any]],
    market_db_path: Path,
) -> tuple[dict[str, str], str]:
    alias_by_path: dict[Path, str] = {}

    def ensure_alias(path: Path, prefix: str) -> str:
        resolved = path.resolve()
        alias = alias_by_path.get(resolved)
        if alias:
            return alias
        alias = f"{prefix}_{len(alias_by_path)}"
        con.execute(f"ATTACH {_duckdb_literal(resolved)} AS {_quote_ident(alias)} (READ_ONLY)")
        alias_by_path[resolved] = alias
        return alias

    source_refs: dict[str, str] = {}
    for label, source in sources.items():
        alias = ensure_alias(Path(source["db_path"]), f"pred_{label}")
        source_refs[label] = f'{_quote_ident(alias)}.{_quote_ident(str(source["table"]))}'

    market_alias = ensure_alias(market_db_path, "market")
    market_table_ref = f'{_quote_ident(market_alias)}."STOCK_DAILY_DATA"'
    return source_refs, market_table_ref


def _table_check(
    con: duckdb.DuckDBPyConnection,
    source: dict[str, Any],
    signal_date: str,
    *,
    table_ref: str,
) -> dict[str, Any]:
    table = source["table"]
    row = con.execute(
        f"""
        SELECT
          min(trade_date) AS min_trade_date,
          max(trade_date) AS max_trade_date,
          count(*) AS row_count,
          count(DISTINCT trade_date) AS trade_days,
          sum(CASE WHEN trade_date = ? THEN 1 ELSE 0 END) AS signal_date_rows,
          count(DISTINCT CASE WHEN trade_date = ? THEN stock_code END) AS signal_date_stock_count,
          sum(CASE WHEN trade_date = ? AND pred_prob IS NULL THEN 1 ELSE 0 END) AS signal_date_null_pred,
          (
            SELECT count(*)
            FROM (
              SELECT trade_date, stock_code
              FROM {table_ref}
              WHERE trade_date = ?
              GROUP BY trade_date, stock_code
              HAVING count(*) > 1
            )
          ) AS duplicate_keys
        FROM {table_ref}
        """,
        [signal_date, signal_date, signal_date, signal_date],
    ).fetchone()
    return {
        "manifest_path": str(source["manifest_path"]),
        "source_type": source["source_type"],
        "db_path": str(source["db_path"]),
        "table": table,
        "approval_status": source["approval_status"],
        "min_trade_date": row[0],
        "max_trade_date": row[1],
        "row_count": int(row[2]),
        "trade_days": int(row[3]),
        "signal_date_rows": int(row[4] or 0),
        "signal_date_stock_count": int(row[5] or 0),
        "signal_date_null_pred": int(row[6] or 0),
        "duplicate_keys": int(row[7] or 0),
    }


def _query_rows(
    con: duckdb.DuckDBPyConnection,
    table_refs: dict[str, str],
    market_table_ref: str,
    signal_date: str,
) -> list[dict[str, Any]]:
    t3 = table_refs["3d"]
    t5 = table_refs["5d"]
    t10 = table_refs["10d"]
    sql = f"""
    WITH
    p3 AS (
      SELECT
        trade_date,
        stock_code,
        pred_prob AS pred_3d,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_prob) AS rank_3d
      FROM {t3}
      WHERE trade_date = ?
    ),
    p5 AS (
      SELECT
        trade_date,
        stock_code,
        pred_prob AS pred_5d,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_prob) AS rank_5d
      FROM {t5}
      WHERE trade_date = ?
    ),
    p10 AS (
      SELECT
        trade_date,
        stock_code,
        pred_prob AS pred_10d,
        percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_prob) AS rank_10d
      FROM {t10}
      WHERE trade_date = ?
    ),
    market_lagged AS (
      SELECT
        stock_code,
        trade_date,
        name,
        open AS open_raw,
        close AS close_raw,
        pre_close AS pre_close_raw,
        pct_chg,
        lag(pct_chg) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_pct_chg,
        amount,
        vol,
        turnover_rate,
        total_mv,
        atr_qfq,
        list_date,
        limit_times,
        ST_TYPE,
        ST_TYPE_name
      FROM {market_table_ref}
      WHERE trade_date <= ?
    ),
    m AS (
      SELECT * FROM market_lagged WHERE trade_date = ?
    )
    SELECT
      p10.trade_date AS signal_date,
      p10.stock_code,
      m.name,
      p3.pred_3d,
      p5.pred_5d,
      p10.pred_10d,
      p3.rank_3d,
      p5.rank_5d,
      p10.rank_10d,
      0.72 * p10.rank_10d + 0.23 * p5.rank_5d + 0.05 * p3.rank_3d AS entry_score,
      m.amount,
      m.vol,
      m.turnover_rate,
      m.total_mv,
      m.atr_qfq,
      m.open_raw,
      m.close_raw,
      m.pre_close_raw,
      m.pct_chg,
      m.prev_pct_chg,
      ((1 + coalesce(m.pct_chg, 0) / 100.0) * (1 + coalesce(m.prev_pct_chg, 0) / 100.0) - 1) AS two_day_ret,
      m.list_date,
      m.limit_times,
      m.ST_TYPE,
      m.ST_TYPE_name,
      CASE WHEN p10.stock_code LIKE '%.BJ' THEN 1 ELSE 0 END AS is_bj,
      CASE
        WHEN m.name IS NULL THEN 0
        WHEN instr(m.name, '退') > 0 THEN 1
        ELSE 0
      END AS is_delist,
      CASE
        WHEN m.ST_TYPE IS NOT NULL AND trim(CAST(m.ST_TYPE AS VARCHAR)) NOT IN ('', '0', '无', '正常', 'None', 'none') THEN 1
        WHEN m.ST_TYPE_name IS NOT NULL AND trim(CAST(m.ST_TYPE_name AS VARCHAR)) NOT IN ('', '0', '无', '正常', 'None', 'none') THEN 1
        WHEN m.name LIKE 'ST%' OR m.name LIKE '*ST%' THEN 1
        ELSE 0
      END AS is_st_risk_warning,
      CASE
        WHEN m.open_raw IS NULL OR m.close_raw IS NULL OR coalesce(m.vol, 0) <= 0 OR coalesce(m.amount, 0) <= 0 THEN 1
        ELSE 0
      END AS is_suspended_or_untradable,
      CASE WHEN m.list_date = ? THEN 1 ELSE 0 END AS is_listing_first_day
    FROM p10
    JOIN p5 ON p10.trade_date = p5.trade_date AND p10.stock_code = p5.stock_code
    JOIN p3 ON p10.trade_date = p3.trade_date AND p10.stock_code = p3.stock_code
    LEFT JOIN m ON p10.stock_code = m.stock_code
    """
    df = con.execute(sql, [signal_date, signal_date, signal_date, signal_date, signal_date, signal_date]).fetchdf()
    return df.to_dict("records")


def _filter_rows(rows: list[dict[str, Any]], rules: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = rules["selection_rule"]
    steps: list[dict[str, Any]] = []
    current = list(rows)

    def apply_step(name: str, predicate) -> None:
        nonlocal current
        before = len(current)
        rejected = [row for row in current if not predicate(row)]
        current = [row for row in current if predicate(row)]
        steps.append({"step": name, "before": before, "rejected": len(rejected), "after": len(current)})

    steps.append({"step": "raw_common_3d_5d_10d_with_market_left_join", "before": len(rows), "rejected": 0, "after": len(rows)})
    apply_step("exclude_bj", lambda r: int(r.get("is_bj") or 0) == 0)
    apply_step("exclude_st_risk_warning", lambda r: int(r.get("is_st_risk_warning") or 0) == 0)
    apply_step("exclude_delist", lambda r: int(r.get("is_delist") or 0) == 0)
    apply_step("exclude_listing_first_day", lambda r: int(r.get("is_listing_first_day") or 0) == 0)
    apply_step("exclude_suspended_or_untradable_signal_date", lambda r: int(r.get("is_suspended_or_untradable") or 0) == 0)
    amount_min = float(selection["amount_min"])
    total_mv_min = float(selection["total_mv_min"])
    price_cap = float(selection["price_cap_signal_close"])
    two_day_cap = float(selection["two_day_cap"])
    apply_step("amount_min", lambda r: float(r.get("amount") or 0) >= amount_min)
    apply_step("total_mv_min", lambda r: float(r.get("total_mv") or 0) >= total_mv_min)
    apply_step("price_cap_signal_close", lambda r: float(r.get("close_raw") or 1e18) <= price_cap)
    apply_step("two_day_cap", lambda r: float(r.get("two_day_ret") or 0) < two_day_cap)

    current.sort(key=lambda row: (-float(row["entry_score"]), row["stock_code"]))
    return current, {"filter_steps": steps}


def _format_signal_row(row: dict[str, Any], rank: int, buy_date: str, latest_market_date: str) -> dict[str, Any]:
    rank_3d = float(row["rank_3d"])
    rank_5d = float(row["rank_5d"])
    target_pct = _target_pct(rank_3d, rank_5d)
    stock_code = str(row["stock_code"])
    return {
        "signal_date": str(row["signal_date"]),
        "buy_date": buy_date,
        "symbol": _stock_to_symbol(stock_code),
        "stock_code": stock_code,
        "name": str(row.get("name") or ""),
        "rank": str(rank),
        "pred_prob": f"{float(row['entry_score']):.16g}",
        "entry_score": f"{float(row['entry_score']):.16g}",
        "pred_3d": f"{float(row['pred_3d']):.16g}",
        "pred_5d": f"{float(row['pred_5d']):.16g}",
        "pred_10d": f"{float(row['pred_10d']):.16g}",
        "rank_3d": f"{rank_3d:.16g}",
        "rank_5d": f"{rank_5d:.16g}",
        "rank_10d": f"{float(row['rank_10d']):.16g}",
        "amount": f"{float(row.get('amount') or 0):.10g}",
        "turnover_rate": f"{float(row.get('turnover_rate') or 0):.10g}",
        "total_mv": f"{float(row.get('total_mv') or 0):.10g}",
        "signal_open_raw": f"{float(row.get('open_raw') or 0):.10g}",
        "signal_close_raw": f"{float(row.get('close_raw') or 0):.10g}",
        "signal_pre_close_raw": f"{float(row.get('pre_close_raw') or 0):.10g}",
        "atr_qfq": f"{float(row.get('atr_qfq') or 0):.10g}",
        "signal_pct_chg_raw": f"{float(row.get('pct_chg') or 0):.10g}",
        "signal_prev_pct_chg_raw": f"{float(row.get('prev_pct_chg') or 0):.10g}",
        "signal_two_day_ret_raw": f"{float(row.get('two_day_ret') or 0):.16g}",
        "target_pct": f"{target_pct:.5f}",
        "holding_days": "2",
        "max_holding_days": "3",
        "score_exit_entry_ratio": "0.97000",
        "min_holding_days_before_score_exit": "1",
        "score_continue_entry_ratio": "0.98000",
        "signal_stop_loss_pct": "0.05000",
        "signal_take_profit_pct": "0.08000",
        "strategy_variant": "dyn_mild_09_12_15",
        "filter_name": "cool2d10",
        "entry_weight_name": "w72_23_05_amt150_mv30",
        "dynamic_hold_name": "h2m3_e097_c098_daydrop99",
        "buy_day_market_available": "False",
        "buy_day_hard_gate_complete": "False",
        "buy_day_st_rejected": "False",
        "buy_day_open_limit_up_rejected": "False",
        "latest_market_date": latest_market_date,
    }


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["selected_rows"]
    lines = [
        "# L5/L6 20260630 增量信号自审报告",
        "",
        f"- 策略：`{payload['strategy_id']}`",
        f"- signal_date：`{payload['signal_date']}`",
        f"- buy_date：`{payload['buy_date']}`",
        f"- 状态：`{payload['status']}`",
        f"- 最终信号数：`{payload['signal_row_count']}`",
        f"- 买入日硬门禁：`{payload['buy_day_hard_gate_complete']}`",
        "",
        "## 最终信号",
        "",
        "| rank | stock_code | name | target_pct | entry_score |",
        "|---:|---|---|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['rank']} | {row['stock_code']} | {row['name']} | {row['target_pct']} | {row['entry_score']} |"
        )
    lines.extend(
        [
            "",
            "## 过滤统计",
            "",
            "| step | before | rejected | after |",
            "|---|---:|---:|---:|",
        ]
    )
    for step in payload["filter_audit"]["filter_steps"]:
        lines.append(f"| {step['step']} | {step['before']} | {step['rejected']} | {step['after']} |")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 本次只生成 L5/L6 增量信号资产，不触发交易、不跑回测、不调参。",
            "- 输入为 active formal DuckDB manifests，未读取 legacy/research-only 预测资产。",
            "- `buy_date=20260701` 的买入日实时硬门禁仍需 L7/交易侧在当日行情可用后执行。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_incremental_signal(signal_date: str) -> dict[str, Any]:
    buy_date = _next_trade_date(signal_date)
    registry = _load_json(MAIN / "strategy_library" / "registry.json")
    current_id = registry.get("production", {}).get("current")
    if current_id != STRATEGY_ID:
        raise RuntimeError(f"registry current strategy mismatch: {current_id} != {STRATEGY_ID}")

    strategy_manifest_path = STRATEGY_DIR / "strategy_manifest.json"
    trading_rules_path = STRATEGY_DIR / "trading_rules.json"
    strategy_manifest = _load_json(strategy_manifest_path)
    trading_rules = _load_json(trading_rules_path)
    sources = _load_sources(strategy_manifest)

    market_duckdb_path = _resolve_market_duckdb_path(strategy_manifest, sources)

    with duckdb.connect() as con:
        table_refs, market_table_ref = _attach_duckdb_sources(
            con,
            sources=sources,
            market_db_path=market_duckdb_path,
        )
        table_checks = {
            label: _table_check(con, source, signal_date, table_ref=table_refs[label])
            for label, source in sources.items()
        }
        rows = _query_rows(con, table_refs, market_table_ref, signal_date)
        latest_market_date = con.execute(f"SELECT max(trade_date) FROM {market_table_ref}").fetchone()[0]
        buy_day_market_rows = con.execute(
            f"SELECT count(*) FROM {market_table_ref} WHERE trade_date = ?",
            [buy_date],
        ).fetchone()[0]
    filtered, filter_audit = _filter_rows(rows, trading_rules)
    exclude_bj_step = next(
        (step for step in filter_audit["filter_steps"] if step.get("step") == "exclude_bj"),
        None,
    )
    position = trading_rules["position_rule"]
    topn = int(position["topn"])
    selected = filtered[:topn]
    signal_rows = [
        _format_signal_row(row, idx, buy_date=buy_date, latest_market_date=str(latest_market_date))
        for idx, row in enumerate(selected, start=1)
    ]

    raw_score_path = REPORTS / f"strategy_agent_incremental_signal_{signal_date}_blended_scores.csv"
    candidate_path = REPORTS / f"strategy_agent_incremental_signal_{signal_date}_candidates.csv"
    latest_path = PRODUCTION_SIGNALS / f"{STRATEGY_ID}_latest.csv"
    latest_status_path = PRODUCTION_SIGNALS / f"{STRATEGY_ID}_latest_status.json"
    archive_latest_path = SIGNAL_DIR / f"latest_signal_{signal_date}_for_{buy_date}.csv"
    archive_status_path = SIGNAL_DIR / f"latest_signal_{signal_date}_for_{buy_date}_status.json"
    archive_auto_status_path = SIGNAL_DIR / "latest_signal_status_auto.json"
    archive_signals_latest_path = SIGNAL_DIR / "signals_latest.csv"
    full_history_path = Path(strategy_manifest["full_history_signal_file"])
    self_audit_path = REPORTS / f"strategy_agent_incremental_signal_{signal_date}_l5_l6_self_audit.json"
    handoff_path = REPORTS / f"strategy_agent_incremental_signal_{signal_date}_l5_l6_handoff.md"

    score_fields = [
        "signal_date",
        "stock_code",
        "name",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "rank_3d",
        "rank_5d",
        "rank_10d",
        "entry_score",
        "amount",
        "turnover_rate",
        "total_mv",
        "open_raw",
        "close_raw",
        "pre_close_raw",
        "atr_qfq",
        "pct_chg",
        "prev_pct_chg",
        "two_day_ret",
        "is_bj",
        "is_st_risk_warning",
        "is_delist",
        "is_listing_first_day",
        "is_suspended_or_untradable",
    ]
    _write_csv(raw_score_path, rows, score_fields)
    _write_csv(candidate_path, filtered, score_fields)
    _validate_strategy_output_rows(signal_rows, context="export_prod_dyn_mild_incremental_signal.signal_rows")
    _write_csv(latest_path, signal_rows)
    _write_csv(archive_latest_path, signal_rows)
    _write_csv(archive_signals_latest_path, signal_rows)

    existing_rows = _read_csv(full_history_path)
    history_fields = list(existing_rows[0].keys()) if existing_rows else list(signal_rows[0].keys())
    existing_rows = [row for row in existing_rows if str(row.get("signal_date") or "") != signal_date]
    _write_csv(full_history_path, existing_rows + signal_rows, history_fields)

    duplicate_signal_stock_keys = len(signal_rows) - len(
        {(row["signal_date"], row["stock_code"]) for row in signal_rows}
    )
    shortage_audit = {
        "target_topn": topn,
        "actual_signal_rows": len(signal_rows),
        "candidate_pool_after_all_filters": len(filtered),
        "shortage": len(signal_rows) < topn,
    }
    status = {
        "strategy_id": STRATEGY_ID,
        "status": "pending_buy_day_hard_gate",
        "signal_date": signal_date,
        "buy_date": buy_date,
        "latest_signal_date": signal_date,
        "latest_buy_date": buy_date,
        "source_file": str(full_history_path),
        "archive_file": str(archive_latest_path),
        "latest_file": str(latest_path),
        "row_count": len(signal_rows),
        "stock_count": len({row["stock_code"] for row in signal_rows}),
        "duplicate_signal_stock_keys": duplicate_signal_stock_keys,
        "buy_day_market_available": bool(buy_day_market_rows),
        "buy_day_hard_gate_complete": False,
        "buy_day_realtime_checks_required": True,
        "selected_rows": signal_rows,
        "shortage_audit": shortage_audit,
        "note": "L5/L6 生产增量信号已生成；买入日实时硬门禁待 L7/交易侧在 buy_date 行情可用后确认。",
    }
    _write_json(latest_status_path, status)
    _write_json(archive_status_path, status)
    _write_json(archive_auto_status_path, status)

    strategy_manifest["current_signal"] = {
        "signal_date": signal_date,
        "buy_date": buy_date,
        "archive_file": str(archive_latest_path),
        "latest_file": str(latest_path),
        "status_file": str(archive_status_path),
        "buy_day_hard_gate_complete": False,
        "buy_day_realtime_checks_required": True,
    }
    _write_json(strategy_manifest_path, strategy_manifest)

    payload = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "strategy_id": STRATEGY_ID,
        "registry_current": current_id,
        "signal_date": signal_date,
        "buy_date": buy_date,
        "calendar_basis": "pandas_market_calendars.SSE",
        "target_trade_date_from_commander": signal_date,
        "status": status["status"],
        "strategy_manifest_path": str(strategy_manifest_path),
        "trading_rules_path": str(trading_rules_path),
        "validation_path": str(STRATEGY_DIR / "validation.json"),
        "input_manifests": {label: str(source["manifest_path"]) for label, source in sources.items()},
        "prediction_tables": {label: source["table"] for label, source in sources.items()},
        "market_field_semantics": default_market_field_semantics(),
        "table_checks": table_checks,
        "filter_audit": filter_audit,
        "target_pct_rule": {
            "mode": "dynamic_mild_09_12_15",
            "formula": "target_pct = 0.15 if min(rank_3d, rank_5d) >= 0.65; 0.12 if >= 0.45; else 0.09",
        },
        "raw_score_row_count": len(rows),
        "candidate_row_count": len(filtered),
        "signal_row_count": len(signal_rows),
        "signal_stock_count": len({row["stock_code"] for row in signal_rows}),
        "duplicate_signal_stock_keys": duplicate_signal_stock_keys,
        "shortage_audit": shortage_audit,
        "selected_rows": signal_rows,
        "no_bj_policy_check": {
            "policy": "exclude_bj_at_strategy_input",
            "filter_step": exclude_bj_step,
            "bj_rows_after_filter": sum(1 for row in filtered if str(row.get("stock_code") or "").endswith(".BJ")),
        },
        "buy_day_market_rows": int(buy_day_market_rows or 0),
        "buy_day_market_available": bool(buy_day_market_rows),
        "buy_day_hard_gate_complete": False,
        "old_chain_read_check": {
            "allow_legacy": False,
            "source_types": {label: source["source_type"] for label, source in sources.items()},
            "db_paths": {label: str(source["db_path"]) for label, source in sources.items()},
            "legacy_odb_used": False,
            "model_predictions_sqlite_used": False,
            "research_only_used": False,
        },
        "output_paths": {
            "blended_scores_csv": str(raw_score_path),
            "candidates_csv": str(candidate_path),
            "latest_signal_csv": str(latest_path),
            "latest_status_json": str(latest_status_path),
            "archive_latest_signal_csv": str(archive_latest_path),
            "archive_status_json": str(archive_status_path),
            "full_history_signal_csv": str(full_history_path),
            "self_audit_json": str(self_audit_path),
            "handoff_md": str(handoff_path),
        },
        "l7_handoff": {
            "allowed_now": False,
            "reason": "本任务需先提交审计复核；且 buy_date 买入日硬门禁未完成。",
            "allowed_after": "审计通过且 L7/交易侧完成 buy_date 实时硬门禁。",
        },
        "residual_risks": [
            "20260701 买入日行情尚未落入当前 DuckDB 日线库，涨停/ST/停牌/开盘价上限等买入日硬门禁需要 L7 当日实时确认。",
            "本次未做研究回测，不能把 20260630 inference 信号表述为已成熟标签验证。",
        ],
    }
    _write_json(self_audit_path, payload)
    _write_markdown_report(handoff_path, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export incremental signal for current dyn_mild production strategy.")
    parser.add_argument("--signal-date", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = export_incremental_signal(args.signal_date)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
