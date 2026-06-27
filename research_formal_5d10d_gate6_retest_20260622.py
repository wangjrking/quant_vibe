from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
MANIFEST_5D = MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json"
MANIFEST_10D = MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json"
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260622"
    / "formal_5d10d_gate6_retest_20260622"
)
FUSION_DB = REPORT_DIR / "formal_5d10d_gate6_fusion_scores.db"
FUSION_TABLE = "fusion_rank_base"
SCORE_TABLE = "score_rank_10d90_5d10"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
START_DATE = "20240604"
END_DATE = "20260618"


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_SCORE_EXIT_ENTRY_RATIO": "1.0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "4",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_STOP_LOSS_PCT": "0.08",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_SCORE_EXIT_RANK": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.10",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.18",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.05",
    "GM_EQUITY_DD_SOFT_SCALE": "0.85",
    "GM_EQUITY_DD_HARD_SCALE": "0.60",
}


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("approval_status") != "approved_for_l5":
        raise RuntimeError(f"manifest is not approved_for_l5: {path}")
    if manifest.get("source_type") != "sqlite_table":
        raise RuntimeError(f"manifest is not sqlite_table: {path}")
    return manifest


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _rank_pct_sql(column: str) -> str:
    return f"""(
        CAST(RANK() OVER (PARTITION BY trade_date ORDER BY {column} ASC) AS REAL)
        + (CAST(COUNT(*) OVER (PARTITION BY trade_date, {column}) AS REAL) - 1.0) / 2.0
    ) / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL)"""


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "None"):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _safe(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("/", "_").replace("None", "none")


M5 = _load_manifest(MANIFEST_5D)
M10 = _load_manifest(MANIFEST_10D)
PRED_DB_5D = _resolve_manifest_path(MANIFEST_5D, M5["db_path"])
PRED_DB_10D = _resolve_manifest_path(MANIFEST_10D, M10["db_path"])
MARKET_DB = _resolve_manifest_path(MANIFEST_10D, M10["market_db_path"])
TABLE_5D = str(M5["table"])
TABLE_10D = str(M10["table"])

if PRED_DB_5D != PRED_DB_10D:
    raise RuntimeError(f"5D and 10D manifests resolve to different DBs: {PRED_DB_5D} vs {PRED_DB_10D}")


def build_fusion_db() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if FUSION_DB.exists():
        return
    conn = sqlite3.connect(FUSION_DB)
    try:
        conn.execute(f"ATTACH DATABASE {_quote_literal(PRED_DB_5D)} AS model")
        conn.execute(f"ATTACH DATABASE {_quote_literal(MARKET_DB)} AS market")
        t5 = _quote_ident(TABLE_5D)
        t10 = _quote_ident(TABLE_10D)
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS fusion_rank_base;
            CREATE TABLE fusion_rank_base AS
            WITH joined AS (
                SELECT
                    d10.trade_date AS trade_date,
                    d10.stock_code AS stock_code,
                    d10.pred_prob AS pred_10d,
                    d5.pred_prob AS pred_5d,
                    m.name AS name,
                    d10.close AS close,
                    d10.pre_close AS pre_close,
                    m.open AS open,
                    d10.amount AS amount,
                    d10.turnover_rate AS turnover_rate,
                    d10.turnover_rate_f AS turnover_rate_f,
                    d10.total_mv AS total_mv,
                    d10.circ_mv AS circ_mv,
                    d10.volume_ratio AS volume_ratio,
                    d10.close_rate AS close_rate,
                    d10.atr_qfq AS atr_qfq,
                    m.limit_times AS limit_times,
                    m.st_type AS st_type
                FROM model.{t10} d10
                INNER JOIN model.{t5} d5
                    ON d10.trade_date = d5.trade_date
                   AND d10.stock_code = d5.stock_code
                LEFT JOIN market.STOCK_DAILY_DATA m
                    ON d10.trade_date = m.trade_date
                   AND d10.stock_code = m.stock_code
                WHERE d10.trade_date >= '{START_DATE}'
                  AND d10.trade_date <= '{END_DATE}'
                  AND d10.pred_prob IS NOT NULL
                  AND d5.pred_prob IS NOT NULL
            ),
            ranked AS (
                SELECT
                    joined.*,
                    {_rank_pct_sql("pred_5d")} AS rank_5d,
                    {_rank_pct_sql("pred_10d")} AS rank_10d
                FROM joined
            )
            SELECT * FROM ranked;

            CREATE INDEX idx_fusion_trade_stock ON fusion_rank_base(trade_date, stock_code);
            CREATE INDEX idx_fusion_trade_rank10 ON fusion_rank_base(trade_date, rank_10d DESC);
            CREATE INDEX idx_fusion_trade_rank5 ON fusion_rank_base(trade_date, rank_5d DESC);

            DROP TABLE IF EXISTS {SCORE_TABLE};
            CREATE TABLE {SCORE_TABLE} AS
            SELECT
                trade_date,
                stock_code,
                (rank_10d * 0.9 + rank_5d * 0.1) AS pred_prob,
                pred_10d,
                pred_5d,
                rank_10d,
                rank_5d,
                name,
                close,
                pre_close,
                open,
                amount,
                turnover_rate,
                turnover_rate_f,
                total_mv,
                circ_mv,
                volume_ratio,
                close_rate,
                atr_qfq,
                limit_times,
                st_type
            FROM fusion_rank_base;
            CREATE INDEX idx_score_trade_stock ON {SCORE_TABLE}(trade_date, stock_code);
            CREATE INDEX idx_score_trade_pred ON {SCORE_TABLE}(trade_date, pred_prob DESC);
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS manifest (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        manifest = {
            "manifest_5d": str(MANIFEST_5D),
            "manifest_10d": str(MANIFEST_10D),
            "table_5d": TABLE_5D,
            "table_10d": TABLE_10D,
            "score_table": SCORE_TABLE,
            "start_date": START_DATE,
            "end_date": END_DATE,
        }
        conn.executemany(
            "INSERT OR REPLACE INTO manifest(key, value) VALUES(?, ?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in manifest.items()],
        )
        conn.commit()
    finally:
        conn.close()


def _fetch_pool_rows(
    pool: dict[str, Any],
    tier_offset: float,
    start: str = START_DATE,
    end: str = END_DATE,
) -> list[dict[str, Any]]:
    w10 = float(pool.get("rank_weight_10d", 0.9))
    w5 = float(pool.get("rank_weight_5d", 0.1))
    formula = f"(rank_10d * {w10:.12g}) + (rank_5d * {w5:.12g})"
    where = [
        "trade_date >= ?",
        "trade_date <= ?",
        "stock_code NOT LIKE '%.BJ'",
        "COALESCE(name, '') NOT LIKE 'ST%'",
        "COALESCE(name, '') NOT LIKE '*ST%'",
        "COALESCE(name, '') NOT LIKE ?",
        "COALESCE(name, '') NOT LIKE ?",
        "COALESCE(name, '') NOT LIKE ?",
        "(limit_times IS NULL OR limit_times = '' OR limit_times = 'None' OR CAST(limit_times AS REAL) = 0)",
        "rank_5d IS NOT NULL",
        "rank_10d IS NOT NULL",
        "total_mv IS NOT NULL AND total_mv <= ?",
        "amount IS NOT NULL AND amount >= ?",
        "turnover_rate IS NOT NULL AND turnover_rate >= ?",
        "close IS NOT NULL",
    ]
    params: list[Any] = [
        start,
        end,
        "%退市%",
        "退%",
        "%退",
        float(pool["max_total_mv"]),
        float(pool["min_amount"]),
        float(pool["min_turnover_rate"]),
        int(pool["limit"]),
    ]
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM (
                SELECT
                    *,
                    {float(tier_offset):.12g} + {formula} AS pred_prob,
                    ROW_NUMBER() OVER (
                        PARTITION BY trade_date
                        ORDER BY {formula} DESC, stock_code
                    ) AS rn
                FROM fusion_rank_base
                WHERE {" AND ".join(where)}
            )
            WHERE rn <= ?
            ORDER BY trade_date, pred_prob DESC, stock_code
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _group_by_date(rows: list[dict[str, Any]], key: str = "trade_date") -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trade_date = str(row.get(key) or "")
        if trade_date:
            grouped.setdefault(trade_date, []).append(row)
    return grouped


def _merge_pool_rows(primary: list[dict[str, Any]], fallback: list[dict[str, Any]], dates: set[str]) -> list[dict[str, Any]]:
    primary_by_date = _group_by_date(primary)
    fallback_by_date = _group_by_date(fallback)
    out: list[dict[str, Any]] = []
    for trade_date in sorted(dates):
        seen: set[str] = set()
        for role, day_rows in (("strong", primary_by_date.get(trade_date, [])), ("weak", fallback_by_date.get(trade_date, []))):
            for row in day_rows:
                code = str(row.get("stock_code") or "")
                if code in seen:
                    continue
                item = dict(row)
                item["pool_role"] = role
                seen.add(code)
                out.append(item)
    return out


def _pool_signals(rows: list[dict[str, Any]], market_rows: dict[str, dict[str, dict]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
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
        holding_days=int(cfg["holding_days"]),
        max_positions=int(cfg["max_positions"]),
        weight_mode="equal",
        target_total_pct=float(cfg["base_target_pct"]) * float(cfg["max_positions"]),
    )


def _signal_features(signals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = _group_by_date(signals, "signal_date")
    result: dict[str, dict[str, Any]] = {}
    for signal_date, rows in grouped.items():
        preds = [_to_float(row.get("pred_prob"), 0.0) or 0.0 for row in rows]
        result[signal_date] = {
            "count": len(rows),
            "primary_count": sum(1 for value in preds if value >= 2.0),
            "avg_pred": sum(preds) / len(preds) if preds else None,
        }
    return result


def build_base_signals(cfg: dict[str, Any], market_rows: dict[str, dict[str, dict]]) -> list[dict[str, Any]]:
    primary = _fetch_pool_rows(cfg["primary_pool"], tier_offset=2.0)
    normal_fallback = _fetch_pool_rows(cfg["normal_fallback_pool"], tier_offset=1.0)
    weak_fallback = _fetch_pool_rows(cfg["weak_fallback_pool"], tier_offset=1.0)
    all_dates = set(_group_by_date(normal_fallback))
    baseline_rows = _merge_pool_rows(primary, normal_fallback, all_dates)
    baseline_signals = _pool_signals(baseline_rows, market_rows, cfg)
    baseline_features = _signal_features(baseline_signals)
    weak_dates = {
        date
        for date, features in baseline_features.items()
        if int(features.get("primary_count") or 0) <= int(cfg.get("weak_primary_count_max", 1))
    }
    selected_rows = [
        *_merge_pool_rows(primary, normal_fallback, all_dates - weak_dates),
        *_merge_pool_rows(primary, weak_fallback, weak_dates),
    ]
    signals = _pool_signals(selected_rows, market_rows, cfg)
    features = _signal_features(signals)
    filtered: list[dict[str, Any]] = []
    row_by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in selected_rows}
    for signal in signals:
        signal_date = str(signal.get("signal_date") or "")
        source = row_by_key.get((signal_date, str(signal.get("stock_code") or "")), {})
        rank = int(float(signal.get("rank") or 999999))
        if int(features.get(signal_date, {}).get("primary_count") or 0) < int(cfg["primary_count_min"]):
            continue
        if rank > int(cfg["rank_max"]):
            continue
        pred_10d = _to_float(source.get("pred_10d"))
        if pred_10d is None or pred_10d < float(cfg.get("min_pred_10d", 0.0)):
            continue
        out = dict(signal)
        pred_5d = _to_float(source.get("pred_5d"))
        pred_gap = abs(pred_10d - pred_5d) if pred_5d is not None else None
        target = float(cfg["rank_targets"].get(rank, cfg["base_target_pct"]))
        if pred_gap is not None:
            if pred_gap < float(cfg.get("gap_boost_lt", -1.0)):
                target *= float(cfg.get("gap_boost_scale", 1.0))
            if float(cfg.get("gap_cut_lower", 999.0)) <= pred_gap < float(cfg.get("gap_cut_upper", 999.0)):
                target *= float(cfg.get("gap_cut_scale", 1.0))
        target = min(target, float(cfg["max_single_position_pct"]))
        out["target_pct"] = f"{target:.5f}"
        out["pred_10d"] = pred_10d
        out["pred_5d"] = pred_5d
        out["rank_10d"] = source.get("rank_10d")
        out["rank_5d"] = source.get("rank_5d")
        out["pred_gap"] = "" if pred_gap is None else f"{pred_gap:.10f}"
        out["amount"] = source.get("amount")
        out["turnover_rate"] = source.get("turnover_rate")
        out["total_mv"] = source.get("total_mv")
        out["pool_role"] = source.get("pool_role")
        out["score_exit_entry_ratio"] = str(cfg["score_exit_entry_ratio"])
        out["min_holding_days_before_score_exit"] = str(cfg["min_holding_days_before_score_exit"])
        out["score_continue_entry_ratio"] = str(cfg["score_continue_entry_ratio"])
        filtered.append(out)
    post_features = _signal_features(filtered)
    return [
        row
        for row in filtered
        if (_to_float(post_features.get(str(row.get("signal_date") or ""), {}).get("avg_pred")) or 0.0)
        >= float(cfg["post_filter_avg_pred_min"])
    ]


def _variant(name: str, **kwargs: Any) -> dict[str, Any]:
    base = {
        "name": name,
        "holding_days": 7,
        "max_positions": 5,
        "base_target_pct": 0.10925,
        "rank_targets": {1: 0.21, 2: 0.21, 3: 0.20, 4: 0.19, 5: 0.19},
        "max_single_position_pct": 0.28,
        "primary_count_min": 2,
        "rank_max": 5,
        "min_pred_10d": 0.0,
        "post_filter_avg_pred_min": 2.05,
        "gap_boost_lt": 0.03,
        "gap_boost_scale": 1.08,
        "gap_cut_lower": 0.05,
        "gap_cut_upper": 0.10,
        "gap_cut_scale": 0.88,
        "score_exit_entry_ratio": 1.0,
        "min_holding_days_before_score_exit": 3,
        "score_continue_entry_ratio": 1.02,
        "weak_primary_count_max": 1,
        "primary_pool": {
            "rank_weight_10d": 0.9,
            "rank_weight_5d": 0.1,
            "limit": 2,
            "max_total_mv": 150000.0,
            "min_amount": 10000.0,
            "min_turnover_rate": 0.3,
        },
        "normal_fallback_pool": {
            "rank_weight_10d": 0.6,
            "rank_weight_5d": 0.4,
            "limit": 5,
            "max_total_mv": 200000.0,
            "min_amount": 10000.0,
            "min_turnover_rate": 0.3,
        },
        "weak_fallback_pool": {
            "rank_weight_10d": 0.6,
            "rank_weight_5d": 0.4,
            "limit": 5,
            "max_total_mv": 200000.0,
            "min_amount": 20000.0,
            "min_turnover_rate": 0.5,
        },
        "env": dict(BASE_ENV),
    }
    base.update(kwargs)
    base["env"] = {**BASE_ENV, **dict(kwargs.get("env") or {})}
    return base


VARIANTS = [
    _variant("prod_like_new5d"),
    _variant("prod_like_post200", post_filter_avg_pred_min=2.00),
    _variant("prod_like_post210", post_filter_avg_pred_min=2.10),
    _variant("pos80_post200", rank_targets={1: 0.28, 2: 0.26, 3: 0.22, 4: 0.20, 5: 0.18}, max_single_position_pct=0.32, post_filter_avg_pred_min=2.00),
    _variant("pos80_post205", rank_targets={1: 0.28, 2: 0.26, 3: 0.22, 4: 0.20, 5: 0.18}, max_single_position_pct=0.32),
    _variant("turn6_down85", rank_targets={1: 0.32, 2: 0.30, 3: 0.24, 4: 0.22, 5: 0.20}, max_single_position_pct=0.42, post_filter_avg_pred_min=1.95, env={"GM_EQUITY_DD_SOFT_SCALE": "0.82", "GM_EQUITY_DD_HARD_SCALE": "0.58"}),
    _variant("h6_turn6_down85", holding_days=6, rank_targets={1: 0.32, 2: 0.30, 3: 0.24, 4: 0.22, 5: 0.20}, max_single_position_pct=0.42, post_filter_avg_pred_min=1.95, env={"GM_EQUITY_DD_SOFT_SCALE": "0.82", "GM_EQUITY_DD_HARD_SCALE": "0.58"}),
    _variant("score_exit098_mh3", score_exit_entry_ratio=0.98, min_holding_days_before_score_exit=3),
    _variant("score_exit102_mh2", score_exit_entry_ratio=1.02, min_holding_days_before_score_exit=2),
    _variant("no_score_exit", env={"GM_OPEN_DAILY_SCORE_EXIT": "0"}),
]


def _extract_indicator(log_file: Path) -> dict[str, Any] | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _exposure_stats(log_file: Path) -> dict[str, Any]:
    values: list[float] = []
    active: list[int] = []
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active) if active else None,
        "exposure_points": len(values),
    }


def _signal_stats(signal_file: Path) -> dict[str, Any]:
    if not signal_file.exists():
        return {"signal_count": 0, "buy_days": 0, "avg_signal_target_pct": None}
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
    }


def _write_signal(cfg: dict[str, Any], signal_file: Path, market_rows: dict[str, dict[str, dict]]) -> None:
    signals = build_base_signals(cfg, market_rows)
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    write_gm_signals_csv(signals, signal_file)


def _run_backtest(cfg: dict[str, Any], signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in cfg["env"].items()})
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(int(cfg["max_positions"])),
        "--holding-days",
        str(int(cfg["holding_days"])),
        "--max-holding-days",
        str(int(cfg["holding_days"])),
        "--target-position-pct",
        "0.196",
        "--score-db",
        str(FUSION_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        "2024-06-05 09:00:00",
        "--backtest-end",
        "2026-06-18 15:30:00",
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _summarize(cfg: dict[str, Any], signal_file: Path, log_file: Path, returncode: int) -> dict[str, Any]:
    indicator = _extract_indicator(log_file) or {}
    row = {
        "name": cfg["name"],
        "returncode": returncode,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
        "max_drawdown": indicator.get("max_drawdown"),
        "cum_return": indicator.get("pnl_ratio"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "holding_days": cfg["holding_days"],
        "post_filter_avg_pred_min": cfg["post_filter_avg_pred_min"],
        "score_exit_entry_ratio": cfg["score_exit_entry_ratio"],
        "min_holding_days_before_score_exit": cfg["min_holding_days_before_score_exit"],
        "env_json": json.dumps(cfg["env"], ensure_ascii=False, sort_keys=True),
    }
    row.update(_signal_stats(signal_file))
    row.update(_exposure_stats(log_file))
    return row


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
        writer.writerows(rows)


def _metric(row: dict[str, Any], key: str) -> float:
    return _to_float(row.get(key), float("-inf")) or float("-inf")


def _write_report(rows: list[dict[str, Any]]) -> None:
    best_annual = max(rows, key=lambda row: _metric(row, "annual_return"))
    best_sharpe = max(rows, key=lambda row: _metric(row, "sharpe"))
    target_hits = [
        row
        for row in rows
        if _metric(row, "annual_return") >= 3.0
        and _metric(row, "sharpe") >= 4.0
        and _metric(row, "avg_invested_pct") >= 0.80
    ]
    report = f"""# formal 5D+10D gate6 新 5D 回测结论

## 结论

本轮只使用当前 approved formal L4 资产：`{MANIFEST_5D}` 与 `{MANIFEST_10D}`。策略侧未使用 1D、行业限制、月份过滤、日期答案过滤或外部数据；候选过滤保留北交所排除、ST/退市排除、涨停不买入。

- 候选数量：{len(rows)}
- 达到 `年化 >= 3.0`、`夏普 >= 4.0`、`平均持仓 >= 0.80` 的候选数量：{len(target_hits)}
- 是否建议进入 L5 替换评审：{"是" if target_hits else "否"}

## 最优候选

- 最高年化：`{best_annual["name"]}`，年化 `{_metric(best_annual, "annual_return"):.6f}`，夏普 `{_metric(best_annual, "sharpe"):.6f}`，最大回撤 `{_metric(best_annual, "max_drawdown"):.6f}`，平均持仓 `{_metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高夏普：`{best_sharpe["name"]}`，年化 `{_metric(best_sharpe, "annual_return"):.6f}`，夏普 `{_metric(best_sharpe, "sharpe"):.6f}`，最大回撤 `{_metric(best_sharpe, "max_drawdown"):.6f}`，平均持仓 `{_metric(best_sharpe, "avg_invested_pct"):.6f}`。

## 证据

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 按夏普排序：`{REPORT_DIR / "summary_by_sharpe.csv"}`
- 按年化排序：`{REPORT_DIR / "summary_by_annual.csv"}`
- 研究 score/fusion 库：`{FUSION_DB}`
- 信号目录：`{REPORT_DIR / "signals"}`
- 掘金日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "formal_5d10d_gate6新5D回测结论.md").write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    build_fusion_db()
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    rows: list[dict[str, Any]] = []
    for cfg in VARIANTS:
        name = str(cfg["name"])
        signal_file = REPORT_DIR / "signals" / f"{_safe(name)}.csv"
        log_file = REPORT_DIR / "logs" / f"{_safe(name)}.log"
        if not (signal_file.exists() and log_file.exists() and _extract_indicator(log_file)):
            _write_signal(cfg, signal_file, market_rows)
        returncode = _run_backtest(cfg, signal_file, log_file)
        result = _summarize(cfg, signal_file, log_file, returncode)
        rows.append(result)
        _write_csv(REPORT_DIR / "summary.csv", rows)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    _write_csv(REPORT_DIR / "summary_by_sharpe.csv", sorted(rows, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_csv(REPORT_DIR / "summary_by_annual.csv", sorted(rows, key=lambda row: _metric(row, "annual_return"), reverse=True))
    _write_report(rows)
    print(json.dumps({"report_dir": str(REPORT_DIR), "variants": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
