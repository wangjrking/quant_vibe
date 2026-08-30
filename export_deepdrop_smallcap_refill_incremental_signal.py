from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
sys.path.insert(0, str(MAIN))

from l5_duckdb_sync import sync_strategy_registry_to_duckdb  # noqa: E402
from l6_duckdb_sync import sync_strategy_backtests_to_duckdb  # noqa: E402
from prediction_manifest import load_prediction_source_manifest  # noqa: E402
from stock_daily_data_route import resolve_stock_daily_duckdb_path  # noqa: E402


STRATEGY_ID = "prod_deepdrop_smallcap_refill_v20260704"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / STRATEGY_ID
SIGNAL_DIR = STRATEGY_DIR / "signals"
PRODUCTION_SIGNAL_DIR = ROOT / "quant" / "data_file" / "production_signals"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports"

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}

FULL_HISTORY = SIGNAL_DIR / "full_history_deepdrop_smallcap_refill.csv"

COLUMNS = [
    "signal_date",
    "buy_date",
    "symbol",
    "stock_code",
    "name",
    "rank",
    "pred_prob",
    "entry_score",
    "pred_1d",
    "pred_3d",
    "pred_5d",
    "pred_10d",
    "amount",
    "turnover_rate",
    "total_mv",
    "atr_qfq",
    "pct_chg",
    "target_pct",
    "holding_days",
    "max_holding_days",
    "score_exit_entry_ratio",
    "min_holding_days_before_score_exit",
    "score_continue_entry_ratio",
    "signal_stop_loss_pct",
    "signal_take_profit_pct",
    "strategy_variant",
    "source_strategy_variant",
    "filter_name",
    "entry_weight_name",
    "dynamic_hold_name",
    "buy_day_market_available",
    "buy_day_hard_gate_complete",
    "buy_day_st_rejected",
    "buy_day_open_limit_up_rejected",
    "latest_market_date",
    "buy_open_gap_pct",
    "hybrid_source",
    "buy_open_gap_raw_pct",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _next_weekday_yyyymmdd(value: str) -> str:
    day = datetime.strptime(value, "%Y%m%d").date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.strftime("%Y%m%d")


def _resolve_next_trade_date(con: duckdb.DuckDBPyConnection, signal_date: str) -> tuple[str, bool]:
    row = con.execute(
        """
        SELECT min(trade_date)
        FROM market.STOCK_DAILY_DATA
        WHERE trade_date > ?
        """,
        [signal_date],
    ).fetchone()
    if row and row[0]:
        return str(row[0]), True
    return _next_weekday_yyyymmdd(signal_date), False


def _is_bad_status_expr(prefix: str) -> str:
    return (
        f"coalesce(cast({prefix}.ST_TYPE AS varchar), '') IN ('', '0') "
        f"AND coalesce(cast({prefix}.ST_TYPE_name AS varchar), '') NOT LIKE '%ST%' "
        f"AND coalesce(cast({prefix}.name AS varchar), '') NOT LIKE 'ST%' "
        f"AND coalesce(cast({prefix}.name AS varchar), '') NOT LIKE '*ST%' "
        f"AND coalesce(cast({prefix}.name AS varchar), '') NOT LIKE '%退%'"
    )


def _attach_sources(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, Any]]:
    sources = {
        label: load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)
        for label, path in MANIFESTS.items()
    }
    for label, source in sources.items():
        if source["source_type"] != "duckdb_table":
            raise RuntimeError(f"L4 manifest is not DuckDB-only: {source['manifest_path']}")
        con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
    market_db = resolve_stock_daily_duckdb_path(require_exists=True)
    con.execute(f"ATTACH '{Path(market_db).as_posix()}' AS market (READ_ONLY)")
    sources["_market"] = {"db_path": market_db, "table": "STOCK_DAILY_DATA"}
    return sources


def _latest_common_signal_date(con: duckdb.DuckDBPyConnection, sources: dict[str, dict[str, Any]]) -> str:
    parts = []
    for label in ("1d", "3d", "5d", "10d"):
        parts.append(f"SELECT max(trade_date) AS max_date FROM l4_{label}.\"{sources[label]['table']}\"")
    sql = "SELECT min(max_date) FROM (" + " UNION ALL ".join(parts) + ")"
    value = con.execute(sql).fetchone()[0]
    if not value:
        raise RuntimeError("cannot resolve common L4 max trade_date")
    return str(value)


def _build_latest_candidates(
    con: duckdb.DuckDBPyConnection,
    sources: dict[str, dict[str, Any]],
    signal_date: str,
    buy_date: str,
    buy_day_market_available: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tables = {label: sources[label]["table"] for label in ("1d", "3d", "5d", "10d")}
    buy_join = (
        "LEFT JOIN market.STOCK_DAILY_DATA buy ON buy.trade_date = ? AND buy.stock_code = sig.stock_code"
        if buy_day_market_available
        else "LEFT JOIN market.STOCK_DAILY_DATA buy ON 1=0"
    )
    buy_params = [buy_date] if buy_day_market_available else []
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE latest_universe AS
        WITH preds AS (
            SELECT
                p10.trade_date,
                p10.stock_code,
                p1.pred_prob AS raw_pred_1d,
                p3.pred_prob AS raw_pred_3d,
                p5.pred_prob AS raw_pred_5d,
                p10.pred_prob AS raw_pred_10d
            FROM l4_10d."{tables['10d']}" p10
            JOIN l4_5d."{tables['5d']}" p5 USING (trade_date, stock_code)
            JOIN l4_3d."{tables['3d']}" p3 USING (trade_date, stock_code)
            JOIN l4_1d."{tables['1d']}" p1 USING (trade_date, stock_code)
            WHERE p10.trade_date = ?
              AND p10.stock_code NOT LIKE '%.BJ'
        ),
        ranked AS (
            SELECT
                *,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY raw_pred_1d) AS rank_1d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY raw_pred_3d) AS rank_3d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY raw_pred_5d) AS rank_5d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY raw_pred_10d) AS rank_10d
            FROM preds
        )
        SELECT
            r.trade_date AS signal_date,
            r.stock_code,
            sig.name,
            sig.market,
            r.raw_pred_1d,
            r.raw_pred_3d,
            r.raw_pred_5d,
            r.raw_pred_10d,
            r.rank_1d,
            r.rank_3d,
            r.rank_5d,
            r.rank_10d,
            (0.25 * r.rank_1d + 0.25 * r.rank_3d + 0.50 * r.rank_10d) AS base_entry_score,
            (0.10 * r.rank_3d + 0.20 * r.rank_5d + 0.70 * r.rank_10d) AS refill_entry_score,
            sig.amount,
            sig.turnover_rate,
            sig.total_mv,
            sig.atr_qfq,
            sig.pct_chg,
            sig.close,
            sig.close_qfq,
            sig.open,
            sig.pre_close,
            sig.ST_TYPE,
            sig.ST_TYPE_name,
            buy.open AS buy_open_raw,
            buy.pre_close AS buy_pre_close_raw,
            buy.ST_TYPE AS buy_ST_TYPE,
            buy.ST_TYPE_name AS buy_ST_TYPE_name,
            buy.name AS buy_name,
            CASE
                WHEN buy.open IS NOT NULL AND buy.pre_close IS NOT NULL AND buy.pre_close > 0
                THEN ((buy.open / buy.pre_close) - 1.0) * 100.0
                ELSE NULL
            END AS buy_open_gap_raw_pct
        FROM ranked r
        JOIN market.STOCK_DAILY_DATA sig ON sig.trade_date = r.trade_date AND sig.stock_code = r.stock_code
        {buy_join}
        WHERE {_is_bad_status_expr('sig')}
          AND sig.amount >= 90000
          AND sig.total_mv >= 200000
        """,
        [signal_date, *buy_params],
    )

    universe_count = con.execute("SELECT count(*) FROM latest_universe").fetchone()[0]
    signal_day_rejects = {
        "after_l4_join_and_signal_day_filters": int(universe_count or 0),
        "bj_rows_after_l4_join": int(con.execute("SELECT count(*) FROM latest_universe WHERE stock_code LIKE '%.BJ'").fetchone()[0] or 0),
    }

    buy_gate = ""
    if buy_day_market_available:
        buy_gate = f"""
          AND {_is_bad_status_expr('buy')}
          AND buy_open_gap_raw_pct <= 1.5
          AND buy_open_raw / NULLIF(buy_pre_close_raw, 0) - 1.0 < CASE
              WHEN stock_code LIKE '300%' OR stock_code LIKE '301%' OR stock_code LIKE '688%' THEN 0.195
              ELSE 0.095
          END
        """

    base_df = con.execute(
        f"""
        WITH eligible AS (
            SELECT *
            FROM latest_universe
            WHERE pct_chg <= -1.75
              AND rank_10d >= 0.70
              AND raw_pred_1d >= -0.002
              {buy_gate}
        ),
        picked AS (
            SELECT
                *,
                row_number() OVER (ORDER BY base_entry_score DESC, stock_code) AS pick_rank
            FROM eligible
        )
        SELECT * FROM picked WHERE pick_rank <= 3
        """
    ).fetchdf()

    selected_codes = set(base_df["stock_code"].astype(str).tolist()) if not base_df.empty else set()
    selected_filter = ",".join("'" + code.replace("'", "''") + "'" for code in selected_codes) or "''"
    refill_need = max(0, 3 - len(selected_codes))
    refill_df = pd.DataFrame()
    if refill_need > 0:
        refill_df = con.execute(
            f"""
            WITH eligible AS (
                SELECT *
                FROM latest_universe
                WHERE pct_chg <= -8.20237
                  AND total_mv <= 453773
                  AND refill_entry_score >= 0.995
                  AND stock_code NOT IN ({selected_filter})
                  {buy_gate}
            ),
            picked AS (
                SELECT
                    *,
                    row_number() OVER (ORDER BY refill_entry_score DESC, stock_code) AS pick_rank
                FROM eligible
            )
            SELECT * FROM picked WHERE pick_rank <= {refill_need}
            """
        ).fetchdf()

    rows: list[dict[str, Any]] = []
    for _, row in base_df.iterrows():
        rows.append(_row_to_signal(row, buy_date, buy_day_market_available, len(rows) + 1, "base_production"))
    for _, row in refill_df.iterrows():
        rows.append(_row_to_signal(row, buy_date, buy_day_market_available, len(rows) + 1, "formal_l4_refill"))
    latest = pd.DataFrame(rows, columns=COLUMNS)
    audit = {
        "universe_count": int(universe_count or 0),
        "base_selected": int(len(base_df)),
        "refill_selected": int(len(refill_df)),
        "target_topn": 3,
        "actual_rows": int(len(latest)),
        "shortage": int(max(0, 3 - len(latest))),
        "signal_day_rejects": signal_day_rejects,
    }
    return latest, audit


def _row_to_signal(row: pd.Series, buy_date: str, buy_day_market_available: bool, rank: int, source: str) -> dict[str, Any]:
    is_base = source == "base_production"
    entry_score = float(row["base_entry_score"] if is_base else row["refill_entry_score"])
    return {
        "signal_date": str(row["signal_date"]),
        "buy_date": buy_date,
        "symbol": _symbol(str(row["stock_code"])),
        "stock_code": str(row["stock_code"]),
        "name": row["name"],
        "rank": rank,
        "pred_prob": entry_score,
        "entry_score": entry_score,
        "pred_1d": float(row["rank_1d"]),
        "pred_3d": float(row["rank_3d"]),
        "pred_5d": float(row["rank_5d"]),
        "pred_10d": float(row["rank_10d"]),
        "amount": row["amount"],
        "turnover_rate": row["turnover_rate"],
        "total_mv": row["total_mv"],
        "atr_qfq": row["atr_qfq"],
        "pct_chg": row["pct_chg"],
        "target_pct": 0.435 if is_base else 0.1,
        "holding_days": 1,
        "max_holding_days": 1,
        "score_exit_entry_ratio": 0.96,
        "min_holding_days_before_score_exit": 1,
        "score_continue_entry_ratio": 9.99,
        "signal_stop_loss_pct": 0.05,
        "signal_take_profit_pct": 0.08,
        "strategy_variant": STRATEGY_ID,
        "source_strategy_variant": "reconstruct_500_from_p44_plus_1dge_m002"
        if is_base
        else "score70_top3_s995_pctm175_gap15",
        "filter_name": "sigpct_le_m175_bog_le1p5_p10d70_1dge_m002"
        if is_base and buy_day_market_available
        else (
            "sigpct_le_m175_rank10d70_1draw_ge_m002_pending_buy_gate"
            if is_base
            else "refill_mv45_deep82_pending_buy_gate"
        ),
        "entry_weight_name": "w25_25_00_50_reconstructed"
        if is_base
        else "score_rank_10d70_5d20_3d10_top3_pctm1_gap15",
        "dynamic_hold_name": "h1m1_reconstructed" if is_base else "h1m1_score_threshold",
        "buy_day_market_available": bool(buy_day_market_available),
        "buy_day_hard_gate_complete": bool(buy_day_market_available),
        "buy_day_st_rejected": False,
        "buy_day_open_limit_up_rejected": False,
        "latest_market_date": None,
        "buy_open_gap_pct": None,
        "hybrid_source": source,
        "buy_open_gap_raw_pct": row["buy_open_gap_raw_pct"] if buy_day_market_available else None,
    }


def _append_full_history(latest: pd.DataFrame) -> dict[str, Any]:
    if not FULL_HISTORY.is_file():
        raise FileNotFoundError(FULL_HISTORY)
    history = pd.read_csv(FULL_HISTORY, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    before = len(history)
    if not latest.empty:
        latest = latest.copy()
        latest["signal_date"] = latest["signal_date"].astype(str)
        latest["buy_date"] = latest["buy_date"].astype(str)
        dates = set(latest["signal_date"].astype(str))
        history = history[~history["signal_date"].astype(str).isin(dates)]
        history = pd.concat([history, latest], ignore_index=True, sort=False)
        history = history.reindex(columns=COLUMNS)
        history = history.sort_values(["signal_date", "rank", "stock_code"], kind="stable")
        history.to_csv(FULL_HISTORY, index=False, encoding="utf-8-sig")
    return {
        "before_rows": int(before),
        "after_rows": int(len(history)),
        "signal_days": int(history["signal_date"].astype(str).nunique()) if len(history) else 0,
        "buy_days": int(history["buy_date"].astype(str).nunique()) if len(history) else 0,
        "full_history": str(FULL_HISTORY),
    }


def _update_archive_files(
    latest: pd.DataFrame,
    signal_date: str,
    buy_date: str,
    buy_day_market_available: bool,
    sources: dict[str, dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Path]:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCTION_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    latest_file = SIGNAL_DIR / f"latest_signal_{signal_date}_for_{buy_date}.csv"
    prod_file = PRODUCTION_SIGNAL_DIR / f"{STRATEGY_ID}_latest.csv"
    latest.to_csv(latest_file, index=False, encoding="utf-8-sig")
    latest.to_csv(prod_file, index=False, encoding="utf-8-sig")

    status = {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "status": "pending_buy_day_hard_gate"
        if not buy_day_market_available
        else "latest_signal_generated",
        "signal_date": signal_date,
        "buy_date": buy_date,
        "latest_signal_date": signal_date,
        "latest_buy_date": buy_date,
        "source_file": str(latest_file),
        "output_file": str(prod_file),
        "row_count": int(len(latest)),
        "stock_count": int(latest["stock_code"].nunique()) if not latest.empty else 0,
        "duplicate_signal_stock_keys": int(
            latest.duplicated(["signal_date", "stock_code"]).sum()
        )
        if not latest.empty
        else 0,
        "buy_day_market_available": bool(buy_day_market_available),
        "buy_day_hard_gate_complete": bool(buy_day_market_available),
        "l7_execution_allowed": False,
        "note": "buy_day_market_not_available; L5 candidate signal only; L7 execution is blocked."
        if not buy_day_market_available
        else "buy_day_hard_gate_complete; audit review is still required before L7 handoff.",
        "audit": audit,
    }
    status_file = SIGNAL_DIR / f"latest_signal_{signal_date}_for_{buy_date}_status.json"
    auto_status = SIGNAL_DIR / "latest_signal_status_auto.json"
    prod_status = PRODUCTION_SIGNAL_DIR / f"{STRATEGY_ID}_latest_status.json"
    for path in (status_file, auto_status, prod_status):
        _write_json(path, status)

    manifest_path = STRATEGY_DIR / "strategy_manifest.json"
    manifest = _load_json(manifest_path)
    manifest["latest_signal_file"] = str(latest_file)
    manifest["latest_signal_status"] = status["status"]
    manifest["latest_signal_date"] = signal_date
    manifest["latest_buy_date"] = buy_date
    manifest["latest_signal_status_file"] = str(status_file)
    current_signal = {
        "latest_file": str(prod_file),
        "archive_latest_file": str(latest_file),
        "signal_date": signal_date,
        "buy_date": buy_date,
        "latest_signal_date": signal_date,
        "latest_buy_date": buy_date,
        "buy_day_market_available": bool(buy_day_market_available),
        "buy_day_hard_gate_complete": bool(buy_day_market_available),
        "l7_execution_allowed": False,
        "status": status["status"],
    }
    if len(latest) == 1:
        current_signal["selected_stock_code"] = str(latest.iloc[0]["stock_code"])
    manifest["current_signal"] = current_signal
    _write_json(manifest_path, manifest)

    validation_path = STRATEGY_DIR / "validation.json"
    validation = _load_json(validation_path)
    validation["latest_signal_status"] = status
    validation.setdefault("date_coverage", {})["signal_date_max"] = signal_date
    validation.setdefault("date_coverage", {})["buy_date_max"] = buy_date
    signal_audit = validation.setdefault("signal_audit", {})
    history_audit = audit.get("full_history_update", {})
    if history_audit:
        signal_audit["signal_rows"] = int(history_audit.get("after_rows") or 0)
        signal_audit["signal_days"] = int(history_audit.get("signal_days") or 0)
        signal_audit["buy_days"] = int(history_audit.get("buy_days") or 0)
    signal_audit["latest_signal_rows"] = int(len(latest))
    signal_audit["duplicate_key_groups"] = status["duplicate_signal_stock_keys"]
    signal_audit["bj_rows"] = int(
        latest["stock_code"].astype(str).str.endswith(".BJ").sum()
    ) if not latest.empty else 0
    _write_json(validation_path, validation)

    report_json = REPORT_DIR / f"strategy_agent_{STRATEGY_ID}_incremental_signal_{signal_date}.json"
    report_md = REPORT_DIR / f"strategy_agent_{STRATEGY_ID}_incremental_signal_{signal_date}.md"
    source_summary = {
        label: {
            "manifest": str(sources[label]["manifest_path"]),
            "db_path": str(sources[label]["db_path"]),
            "table": sources[label]["table"],
            "approval_status": sources[label]["approval_status"],
        }
        for label in ("1d", "3d", "5d", "10d")
    }
    payload = {
        "strategy_id": STRATEGY_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_day_market_available": buy_day_market_available,
        "l7_execution_allowed": False,
        "input_manifests": source_summary,
        "market_db": str(sources["_market"]["db_path"]),
        "outputs": {
            "archive_latest_signal": str(latest_file),
            "production_latest_signal": str(prod_file),
            "archive_status": str(status_file),
            "production_status": str(prod_status),
            "full_history": str(FULL_HISTORY),
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
        "latest_rows": latest.to_dict("records"),
        "audit": audit,
        "status": status,
    }
    _write_json(report_json, payload)
    report_md.write_text(_render_report(payload), encoding="utf-8")
    return {
        "latest_file": latest_file,
        "production_file": prod_file,
        "status_file": status_file,
        "production_status": prod_status,
        "report_json": report_json,
        "report_md": report_md,
        "manifest": manifest_path,
        "validation": validation_path,
    }


def _render_report(payload: dict[str, Any]) -> str:
    rows = payload["latest_rows"]
    lines = [
        f"# {payload['strategy_id']} 增量信号自审报告",
        "",
        "## 结论",
        f"- 信号日：`{payload['signal_date']}`",
        f"- 买入日：`{payload['buy_date']}`",
        f"- 信号数量：`{len(rows)}`",
        f"- 买入日行情是否已落地：`{payload['buy_day_market_available']}`",
        "- L7 交易放行：`False`，原因是本报告只补齐 L5/L6 资产，仍需审计和买入日硬门校验。",
        "",
        "## 最新信号",
    ]
    if rows:
        lines.append("| rank | stock_code | name | target_pct | source | entry_score | pct_chg | total_mv | amount |")
        lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['rank']} | {row['stock_code']} | {row['name']} | {row['target_pct']} | "
                f"{row['hybrid_source']} | {float(row['entry_score']):.6f} | "
                f"{float(row['pct_chg']):.4f} | {float(row['total_mv']):.2f} | {float(row['amount']):.2f} |"
            )
    else:
        lines.append("无信号。")
    lines.extend(
        [
            "",
            "## 关键约束",
            "- 输入仅使用 active formal DuckDB L4 manifests，`require_approved=True`、`allow_legacy=False`。",
            "- `.BJ` 不进入候选，ST/风险警示/退市名称在信号日过滤。",
            "- 买入日硬门使用裸 `open/pre_close` 原始价格；当前买入日行情未落地时只标记 pending，不使用前复权价格替代。",
            "- `atr_qfq` 和模型/因子派生排名保留 qfq 显式语义。",
            "",
            "## 证据路径",
            f"- 生产 latest CSV：`{payload['outputs']['production_latest_signal']}`",
            f"- 归档 latest CSV：`{payload['outputs']['archive_latest_signal']}`",
            f"- 状态文件：`{payload['outputs']['production_status']}`",
            f"- JSON 自审：`{payload['outputs'].get('report_json', '')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-date", default=None)
    args = parser.parse_args()

    con = duckdb.connect()
    try:
        sources = _attach_sources(con)
        signal_date = str(args.signal_date or _latest_common_signal_date(con, sources))
        buy_date, buy_day_market_available = _resolve_next_trade_date(con, signal_date)
        latest, audit = _build_latest_candidates(con, sources, signal_date, buy_date, buy_day_market_available)
    finally:
        con.close()

    history_audit = _append_full_history(latest)
    audit["full_history_update"] = history_audit
    paths = _update_archive_files(latest, signal_date, buy_date, buy_day_market_available, sources, audit)
    paths["report_json"].write_text(
        json.dumps(
            {
                **json.loads(paths["report_json"].read_text(encoding="utf-8")),
                "outputs": {
                    **json.loads(paths["report_json"].read_text(encoding="utf-8"))["outputs"],
                    "report_json": str(paths["report_json"]),
                    "report_md": str(paths["report_md"]),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    sync_l5 = sync_strategy_registry_to_duckdb(project_dir=MAIN)
    sync_l6 = sync_strategy_backtests_to_duckdb(project_dir=MAIN)

    print(
        json.dumps(
            {
                "strategy_id": STRATEGY_ID,
                "signal_date": signal_date,
                "buy_date": buy_date,
                "rows": int(len(latest)),
                "buy_day_market_available": buy_day_market_available,
                "l7_execution_allowed": False,
                "paths": {key: str(value) for key, value in paths.items()},
                "sync_l5": sync_l5,
                "sync_l6": sync_l6,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
