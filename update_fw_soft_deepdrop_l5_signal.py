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


STRATEGY_ID = "prod_fw_soft_deepdrop_weight_v20260706"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / STRATEGY_ID
SIGNAL_DIR = STRATEGY_DIR / "signals"
FULL_HISTORY = SIGNAL_DIR / "full_history_fw_soft_deepdrop_weight.csv"
PRODUCTION_SIGNAL_DIR = ROOT / "quant" / "data_file" / "production_signals"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports"

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}

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
    "signal_pct_chg_raw",
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
    "feature_weight_scale",
    "daily_target_sum_after_cap",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def next_weekday(value: str) -> str:
    day = datetime.strptime(value, "%Y%m%d").date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.strftime("%Y%m%d")


def clean_status_columns_expr(st_type: str, st_type_name: str, name: str) -> str:
    return (
        f"coalesce(cast({st_type} AS varchar), '') IN ('', '0', '0.0', 'None', 'NONE') "
        f"AND coalesce(cast({st_type_name} AS varchar), '') NOT LIKE '%ST%' "
        f"AND coalesce(cast({name} AS varchar), '') NOT LIKE 'ST%' "
        f"AND coalesce(cast({name} AS varchar), '') NOT LIKE '*ST%' "
        f"AND coalesce(cast({name} AS varchar), '') NOT LIKE '%退%'"
    )


def clean_status_expr(alias: str) -> str:
    return clean_status_columns_expr(
        f"{alias}.ST_TYPE",
        f"{alias}.ST_TYPE_name",
        f"{alias}.name",
    )


def attach_sources(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for label, manifest_path in MANIFESTS.items():
        source = load_prediction_source_manifest(manifest_path, require_approved=True, allow_legacy=False)
        if source["source_type"] != "duckdb_table":
            raise RuntimeError(f"L4 manifest is not DuckDB-only: {source['manifest_path']}")
        sources[label] = source
        con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
    market_db = resolve_stock_daily_duckdb_path(require_exists=True)
    con.execute(f"ATTACH '{Path(market_db).as_posix()}' AS market (READ_ONLY)")
    sources["_market"] = {"db_path": str(market_db), "table": "STOCK_DAILY_DATA"}
    return sources


def latest_common_signal_date(con: duckdb.DuckDBPyConnection, sources: dict[str, dict[str, Any]]) -> str:
    parts = [
        f'SELECT max(trade_date) AS max_date FROM l4_{label}."{sources[label]["table"]}"'
        for label in ("1d", "3d", "5d", "10d")
    ]
    value = con.execute("SELECT min(max_date) FROM (" + " UNION ALL ".join(parts) + ")").fetchone()[0]
    if not value:
        raise RuntimeError("cannot resolve common L4 signal_date")
    return str(value)


def resolve_buy_date(con: duckdb.DuckDBPyConnection, signal_date: str) -> tuple[str, bool, str]:
    row = con.execute(
        "SELECT min(trade_date) FROM market.STOCK_DAILY_DATA WHERE trade_date > ?",
        [signal_date],
    ).fetchone()
    if row and row[0]:
        return str(row[0]), True, "l2_market_calendar"
    return next_weekday(signal_date), False, "weekday_estimate_l2_market_not_available"


def l4_coverage(con: duckdb.DuckDBPyConnection, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("1d", "3d", "5d", "10d"):
        table = sources[label]["table"]
        row = con.execute(
            f'''
            SELECT min(trade_date), max(trade_date), count(*), count(distinct trade_date),
                   sum(CASE WHEN pred_prob IS NULL THEN 1 ELSE 0 END)
            FROM l4_{label}."{table}"
            '''
        ).fetchone()
        latest = con.execute(
            f'''
            SELECT trade_date, count(*), count(distinct stock_code)
            FROM l4_{label}."{table}"
            GROUP BY trade_date
            ORDER BY trade_date DESC
            LIMIT 1
            '''
        ).fetchone()
        dup = con.execute(
            f'''
            SELECT count(*)
            FROM (
                SELECT trade_date, stock_code, count(*) AS c
                FROM l4_{label}."{table}"
                GROUP BY 1, 2
                HAVING count(*) > 1
            )
            '''
        ).fetchone()[0]
        result[label] = {
            "manifest": str(sources[label]["manifest_path"]),
            "db_path": str(sources[label]["db_path"]),
            "table": table,
            "approval_status": sources[label].get("approval_status"),
            "source_type": sources[label].get("source_type"),
            "min_trade_date": row[0],
            "max_trade_date": row[1],
            "row_count": int(row[2]),
            "trade_days": int(row[3]),
            "null_pred_prob": int(row[4] or 0),
            "latest_trade_date": latest[0],
            "latest_rows": int(latest[1]),
            "latest_stock_count": int(latest[2]),
            "duplicate_key_groups": int(dup or 0),
        }
    return result


def build_latest(
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
        else "LEFT JOIN market.STOCK_DAILY_DATA buy ON 1 = 0"
    )
    buy_params: list[Any] = [buy_date] if buy_day_market_available else []
    con.execute(
        f'''
        CREATE OR REPLACE TEMP TABLE latest_universe AS
        WITH preds AS (
            SELECT
                p10.trade_date,
                p10.stock_code,
                p1.pred_prob AS raw_pred_1d,
                p3.pred_prob AS raw_pred_3d,
                p5.pred_prob AS raw_pred_5d,
                p10.pred_prob AS raw_pred_10d
            FROM l4_10d."{tables["10d"]}" p10
            JOIN l4_5d."{tables["5d"]}" p5 USING (trade_date, stock_code)
            JOIN l4_3d."{tables["3d"]}" p3 USING (trade_date, stock_code)
            JOIN l4_1d."{tables["1d"]}" p1 USING (trade_date, stock_code)
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
        JOIN market.STOCK_DAILY_DATA sig
          ON sig.trade_date = r.trade_date AND sig.stock_code = r.stock_code
        {buy_join}
        WHERE {clean_status_expr("sig")}
          AND sig.amount >= 90000
          AND sig.total_mv >= 200000
        ''',
        [signal_date, *buy_params],
    )
    universe_count = int(con.execute("SELECT count(*) FROM latest_universe").fetchone()[0] or 0)
    l4_join_count = int(
        con.execute(
            '''
            SELECT count(*)
            FROM l4_10d."{}" p10
            JOIN l4_5d."{}" p5 USING (trade_date, stock_code)
            JOIN l4_3d."{}" p3 USING (trade_date, stock_code)
            JOIN l4_1d."{}" p1 USING (trade_date, stock_code)
            WHERE p10.trade_date = ?
              AND p10.stock_code NOT LIKE '%.BJ'
            '''.format(tables["10d"], tables["5d"], tables["3d"], tables["1d"]),
            [signal_date],
        ).fetchone()[0]
        or 0
    )
    market_rows = int(
        con.execute("SELECT count(*) FROM market.STOCK_DAILY_DATA WHERE trade_date = ?", [signal_date]).fetchone()[0]
        or 0
    )

    buy_gate = ""
    if buy_day_market_available:
        buy_gate = f'''
          AND {clean_status_columns_expr("buy_ST_TYPE", "buy_ST_TYPE_name", "buy_name")}
          AND buy_open_gap_raw_pct <= 1.5
          AND buy_open_raw / NULLIF(buy_pre_close_raw, 0) - 1.0 < CASE
              WHEN stock_code LIKE '300%' OR stock_code LIKE '301%' OR stock_code LIKE '688%' THEN 0.195
              ELSE 0.095
          END
        '''

    base_df = con.execute(
        f'''
        WITH eligible AS (
            SELECT *
            FROM latest_universe
            WHERE pct_chg <= -1.75
              AND rank_10d >= 0.70
              AND rank_1d >= 0.00
              {buy_gate}
        ),
        picked AS (
            SELECT *, row_number() OVER (ORDER BY base_entry_score DESC, stock_code) AS pick_rank
            FROM eligible
        )
        SELECT * FROM picked WHERE pick_rank <= 3
        '''
    ).fetchdf()

    selected = set(base_df["stock_code"].astype(str).tolist()) if not base_df.empty else set()
    selected_sql = ",".join("'" + code.replace("'", "''") + "'" for code in selected) or "''"
    refill_need = max(0, 3 - len(selected))
    refill_df = pd.DataFrame()
    if refill_need:
        refill_df = con.execute(
            f'''
            WITH eligible AS (
                SELECT *
                FROM latest_universe
                WHERE pct_chg <= -8.20237
                  AND total_mv <= 453773
                  AND refill_entry_score >= 0.995
                  AND stock_code NOT IN ({selected_sql})
                  {buy_gate}
            ),
            picked AS (
                SELECT *, row_number() OVER (ORDER BY refill_entry_score DESC, stock_code) AS pick_rank
                FROM eligible
            )
            SELECT * FROM picked WHERE pick_rank <= {refill_need}
            '''
        ).fetchdf()

    rows: list[dict[str, Any]] = []
    for _, row in base_df.iterrows():
        rows.append(to_signal_row(row, buy_date, buy_day_market_available, len(rows) + 1, "base_production"))
    for _, row in refill_df.iterrows():
        rows.append(to_signal_row(row, buy_date, buy_day_market_available, len(rows) + 1, "formal_l4_refill"))

    latest = pd.DataFrame(rows, columns=COLUMNS)
    latest = apply_feature_weight(latest)
    audit = {
        "l4_join_count_no_bj": l4_join_count,
        "signal_day_market_rows": market_rows,
        "after_signal_day_filters": universe_count,
        "base_selected": int(len(base_df)),
        "refill_selected": int(len(refill_df)),
        "target_topn": 3,
        "actual_rows": int(len(latest)),
        "shortage": int(max(0, 3 - len(latest))),
        "bj_rows": int(latest["stock_code"].astype(str).str.endswith(".BJ").sum()) if not latest.empty else 0,
        "duplicate_signal_stock_keys": int(latest.duplicated(["signal_date", "stock_code"]).sum()) if not latest.empty else 0,
        "latest_target_pct_sum": float(latest["target_pct"].sum()) if not latest.empty else 0.0,
    }
    return latest, audit


def to_signal_row(
    row: pd.Series,
    buy_date: str,
    buy_day_market_available: bool,
    rank: int,
    source: str,
) -> dict[str, Any]:
    is_base = source == "base_production"
    entry_score = float(row["base_entry_score"] if is_base else row["refill_entry_score"])
    return {
        "signal_date": str(row["signal_date"]),
        "buy_date": buy_date,
        "symbol": symbol(str(row["stock_code"])),
        "stock_code": str(row["stock_code"]),
        "name": row["name"],
        "rank": rank,
        "pred_prob": entry_score,
        "entry_score": entry_score,
        "pred_1d": float(row["rank_1d"]),
        "pred_3d": float(row["rank_3d"]),
        "pred_5d": float(row["rank_5d"]),
        "pred_10d": float(row["rank_10d"]),
        "amount": float(row["amount"]) if pd.notna(row["amount"]) else None,
        "turnover_rate": float(row["turnover_rate"]) if pd.notna(row["turnover_rate"]) else None,
        "total_mv": float(row["total_mv"]) if pd.notna(row["total_mv"]) else None,
        "atr_qfq": float(row["atr_qfq"]) if pd.notna(row["atr_qfq"]) else None,
        "signal_pct_chg_raw": float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
        "target_pct": 0.435 if is_base else 0.10,
        "holding_days": 1,
        "max_holding_days": 1,
        "score_exit_entry_ratio": 0.96,
        "min_holding_days_before_score_exit": 1,
        "score_continue_entry_ratio": 9.99,
        "signal_stop_loss_pct": 0.05,
        "signal_take_profit_pct": 0.08,
        "strategy_variant": STRATEGY_ID,
        "source_strategy_variant": "fw_soft_rebuilt_from_current_production_full_history",
        "filter_name": "sigpct_le_m175_bog_le1p5_rank10d70_1draw_ge_m002"
        if is_base and buy_day_market_available
        else (
            "sigpct_le_m175_rank10d70_1draw_ge_m002_pending_buy_gate"
            if is_base
            else "refill_mv45_deep82_pending_buy_gate"
        ),
        "entry_weight_name": "w25_25_00_50_reconstructed"
        if is_base
        else "score_rank_10d70_5d20_3d10_refill",
        "dynamic_hold_name": "h1m1_reconstructed",
        "buy_day_market_available": bool(buy_day_market_available),
        "buy_day_hard_gate_complete": bool(buy_day_market_available),
        "buy_day_st_rejected": False,
        "buy_day_open_limit_up_rejected": False,
        "latest_market_date": None,
        "buy_open_gap_pct": None,
        "hybrid_source": source,
        "buy_open_gap_raw_pct": float(row["buy_open_gap_raw_pct"])
        if buy_day_market_available and pd.notna(row["buy_open_gap_raw_pct"])
        else None,
        "feature_weight_scale": 1.0,
        "daily_target_sum_after_cap": None,
    }


def apply_feature_weight(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reindex(columns=COLUMNS)
    out = df.copy()
    pct = out["signal_pct_chg_raw"].astype(float)
    gap = out["buy_open_gap_raw_pct"].fillna(0).astype(float)
    turnover = out["turnover_rate"].astype(float)
    weak = (pct >= -2.5) | (gap >= 0)
    strong = (pct <= -5.0) & (gap <= -0.8) & (turnover >= 4.5)
    out["feature_weight_scale"] = 1.0
    out.loc[weak, "feature_weight_scale"] = 0.85
    out.loc[strong, "feature_weight_scale"] = 1.05
    out["target_pct"] = out["target_pct"].astype(float) * out["feature_weight_scale"]
    daily_sum = out.groupby("buy_date")["target_pct"].transform("sum")
    scale = (0.91 / daily_sum).clip(upper=1.0)
    out["target_pct"] = out["target_pct"] * scale
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    return out.reindex(columns=COLUMNS)


def append_full_history(latest: pd.DataFrame) -> dict[str, Any]:
    history = pd.read_csv(FULL_HISTORY, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    before = len(history)
    if not latest.empty:
        dates = set(latest["signal_date"].astype(str))
        history = history[~history["signal_date"].astype(str).isin(dates)]
        history = pd.concat([history, latest], ignore_index=True, sort=False)
        history = history.reindex(columns=COLUMNS)
        history = history.sort_values(["signal_date", "rank", "stock_code"], kind="stable")
        history.to_csv(FULL_HISTORY, index=False, encoding="utf-8-sig")
    return {
        "before_rows": int(before),
        "after_rows": int(len(history)),
        "signal_days": int(history["signal_date"].astype(str).nunique()),
        "buy_days": int(history["buy_date"].astype(str).nunique()),
        "duplicate_buy_stock_keys": int(history.duplicated(["buy_date", "stock_code"]).sum()),
        "full_history": str(FULL_HISTORY),
    }


def update_json_contracts(
    latest: pd.DataFrame,
    signal_date: str,
    buy_date: str,
    buy_day_market_available: bool,
    buy_date_source: str,
    audit: dict[str, Any],
    l4_summary: dict[str, Any],
    paths: dict[str, str],
) -> dict[str, Any]:
    status = {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "status": "pending_buy_day_hard_gate" if not buy_day_market_available else "latest_signal_generated",
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_date_source": buy_date_source,
        "latest_signal_date": signal_date,
        "latest_buy_date": buy_date,
        "source_file": paths["archive_latest_signal"],
        "output_file": paths["production_latest_signal"],
        "row_count": int(len(latest)),
        "stock_count": int(latest["stock_code"].nunique()) if not latest.empty else 0,
        "duplicate_signal_stock_keys": int(latest.duplicated(["signal_date", "stock_code"]).sum()) if not latest.empty else 0,
        "buy_day_market_available": bool(buy_day_market_available),
        "buy_day_hard_gate_complete": bool(buy_day_market_available),
        "l7_execution_allowed": False,
        "note": "L2 has not landed buy-date market data; this is a L5 signal asset with pending open hard gate."
        if not buy_day_market_available
        else "Buy-date hard gate completed; handoff still requires audit review.",
        "audit": audit,
        "l4_coverage": l4_summary,
    }

    manifest_path = STRATEGY_DIR / "strategy_manifest.json"
    validation_path = STRATEGY_DIR / "validation.json"
    manifest = load_json(manifest_path)
    manifest["latest_signal_file"] = paths["archive_latest_signal"]
    manifest["latest_signal_status"] = status["status"]
    manifest["latest_signal_date"] = signal_date
    manifest["latest_buy_date"] = buy_date
    manifest["latest_signal_status_file"] = paths["archive_status"]
    manifest["current_signal"] = {
        "latest_file": paths["production_latest_signal"],
        "archive_latest_file": paths["archive_latest_signal"],
        "signal_date": signal_date,
        "buy_date": buy_date,
        "latest_signal_date": signal_date,
        "latest_buy_date": buy_date,
        "buy_day_market_available": bool(buy_day_market_available),
        "buy_day_hard_gate_complete": bool(buy_day_market_available),
        "l7_execution_allowed": False,
        "status": status["status"],
    }
    write_json(manifest_path, manifest)

    validation = load_json(validation_path)
    validation.setdefault("date_coverage", {})["signal_date_max"] = signal_date
    validation.setdefault("date_coverage", {})["buy_date_max"] = buy_date
    signal_audit = validation.setdefault("signal_audit", {})
    fh = audit.get("full_history_update", {})
    signal_audit["signal_rows"] = int(fh.get("after_rows", signal_audit.get("signal_rows", 0)))
    signal_audit["signal_days"] = int(fh.get("signal_days", signal_audit.get("signal_days", 0)))
    signal_audit["buy_days"] = int(fh.get("buy_days", signal_audit.get("buy_days", 0)))
    signal_audit["latest_signal_rows"] = int(len(latest))
    signal_audit["duplicate_key_groups"] = int(fh.get("duplicate_buy_stock_keys", 0))
    signal_audit["bj_rows"] = int(audit.get("bj_rows", 0))
    validation["latest_signal_status"] = status
    write_json(validation_path, validation)
    return status


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {STRATEGY_ID} L5 增量信号自审报告",
        "",
        "## 当前结论",
        f"- 信号日：`{payload['signal_date']}`",
        f"- 买入执行日：`{payload['buy_date']}`",
        f"- 买入执行日来源：`{payload['buy_date_source']}`",
        f"- 最新信号数量：`{payload['status']['row_count']}`",
        f"- L7 是否允许继续：`{payload['status']['l7_execution_allowed']}`",
        f"- 状态：`{payload['status']['status']}`",
        "",
        "## 最新信号",
    ]
    rows = payload.get("latest_rows", [])
    if rows:
        lines.append("| 排名 | 代码 | 名称 | 目标仓位 | 分数 | 信号日涨跌幅 | 成交额 | 总市值 |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['rank']} | {row['stock_code']} | {row['name']} | "
                f"{float(row['target_pct']):.6f} | {float(row['entry_score']):.6f} | "
                f"{float(row['signal_pct_chg_raw']):.4f} | {float(row['amount']):.2f} | "
                f"{float(row['total_mv']):.2f} |"
            )
    else:
        lines.append("本次无信号。")
    lines.extend(
        [
            "",
            "## 自审要点",
            "- L4 输入：仅使用 active formal DuckDB manifests，`require_approved=True`，`allow_legacy=False`。",
            "- 交易执行价格：涨停/开盘缺口硬门禁只允许使用 L2 未复权 `open/pre_close`。",
            "- qfq 口径：`atr_qfq` 及模型/因子派生排名保留显式 qfq 语义，不混用裸价格字段。",
            "- 买入日硬门禁：若 L2 尚无买入日行情，本次只更新 L5 资产，L7 继续阻断。",
            "",
            "## 证据路径",
            f"- 生产 latest CSV：`{payload['outputs']['production_latest_signal']}`",
            f"- 归档 latest CSV：`{payload['outputs']['archive_latest_signal']}`",
            f"- 生产 status：`{payload['outputs']['production_status']}`",
            f"- JSON 自审：`{payload['outputs']['report_json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-date", default=None)
    args = parser.parse_args()

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCTION_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        sources = attach_sources(con)
        signal_date = str(args.signal_date or latest_common_signal_date(con, sources))
        buy_date, buy_day_market_available, buy_date_source = resolve_buy_date(con, signal_date)
        l4_summary = l4_coverage(con, sources)
        latest, audit = build_latest(con, sources, signal_date, buy_date, buy_day_market_available)
    finally:
        con.close()

    archive_latest = SIGNAL_DIR / f"latest_signal_{signal_date}_for_{buy_date}.csv"
    production_latest = PRODUCTION_SIGNAL_DIR / f"{STRATEGY_ID}_latest.csv"
    archive_status = SIGNAL_DIR / f"latest_signal_{signal_date}_for_{buy_date}_status.json"
    production_status = PRODUCTION_SIGNAL_DIR / f"{STRATEGY_ID}_latest_status.json"
    auto_status = SIGNAL_DIR / "latest_signal_status_auto.json"
    report_json = REPORT_DIR / f"strategy_agent_{STRATEGY_ID}_incremental_signal_{signal_date}.json"
    report_md = REPORT_DIR / f"strategy_agent_{STRATEGY_ID}_incremental_signal_{signal_date}.md"

    latest.to_csv(archive_latest, index=False, encoding="utf-8-sig")
    latest.to_csv(production_latest, index=False, encoding="utf-8-sig")
    history_update = append_full_history(latest)
    audit["full_history_update"] = history_update

    paths = {
        "archive_latest_signal": str(archive_latest),
        "production_latest_signal": str(production_latest),
        "archive_status": str(archive_status),
        "production_status": str(production_status),
        "auto_status": str(auto_status),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "full_history": str(FULL_HISTORY),
    }
    status = update_json_contracts(
        latest,
        signal_date,
        buy_date,
        buy_day_market_available,
        buy_date_source,
        audit,
        l4_summary,
        paths,
    )
    for path in (archive_status, production_status, auto_status):
        write_json(path, status)

    payload = {
        "strategy_id": STRATEGY_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_date_source": buy_date_source,
        "outputs": paths,
        "latest_rows": latest.to_dict("records"),
        "audit": audit,
        "status": status,
        "l4_coverage": l4_summary,
    }
    write_json(report_json, payload)
    report_md.write_text(render_report(payload), encoding="utf-8")

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
                "outputs": paths,
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
