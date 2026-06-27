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

from gm_signal_module import to_gm_symbol


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
REPORT_DIR = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623" / "current_formal_3d5d10d"
FUSION_DB = REPORT_DIR / "fusion_3d5d10d.db"
SCORE_DB = REPORT_DIR / "scores_3d5d10d.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

START_DATE = "20240604"
END_DATE = "20260618"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-23 15:30:00"

TABLE_3D = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
TABLE_5D = "stock_predict_data_model_agent_5d_gate6_balanced_bonus_20260622_executable_5d_open_return_research"
TABLE_10D = "stock_predict_data_model_agent_10d_topgate_fusion_20260622_executable_10d_open_return_research"

BASE_TARGETS = {
    2: {1: 0.55, 2: 0.43},
    5: {1: 0.24, 2: 0.22, 3: 0.19, 4: 0.16, 5: 0.14},
}

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "0",
    "GM_SCORE_EXIT_ENTRY_RATIO": "none",
    "GM_SCORE_EXIT_RANK": "none",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "5",
    "GM_MAX_DAILY_SELLS": "5",
    "GM_STOP_LOSS_PCT": "0.08",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "0",
}


VARIANTS = [
    {"name": "w10_80_w5_20_w3_00_top5_s112_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "w10_75_w5_20_w3_05_top5_s112_h5", "w10": 0.75, "w5": 0.20, "w3": 0.05, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "w10_70_w5_20_w3_10_top5_s112_h5", "w10": 0.70, "w5": 0.20, "w3": 0.10, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "w10_65_w5_20_w3_15_top5_s112_h5", "w10": 0.65, "w5": 0.20, "w3": 0.15, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "w10_60_w5_20_w3_20_top5_s112_h5", "w10": 0.60, "w5": 0.20, "w3": 0.20, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "w10_70_w5_10_w3_20_top5_s112_h5", "w10": 0.70, "w5": 0.10, "w3": 0.20, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "w10_60_w5_30_w3_10_top5_s112_h5", "w10": 0.60, "w5": 0.30, "w3": 0.10, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "w10_50_w5_30_w3_20_top5_s112_h5", "w10": 0.50, "w5": 0.30, "w3": 0.20, "top_k": 5, "scale": 1.12, "cap": 0.30, "holding_days": 5},
    {"name": "w10_75_w5_20_w3_05_top5_s115_h5", "w10": 0.75, "w5": 0.20, "w3": 0.05, "top_k": 5, "scale": 1.15, "cap": 0.32, "holding_days": 5},
    {"name": "w10_70_w5_20_w3_10_top5_s115_h5", "w10": 0.70, "w5": 0.20, "w3": 0.10, "top_k": 5, "scale": 1.15, "cap": 0.32, "holding_days": 5},
    {"name": "w10_80_w5_20_w3_00_top2_h5", "w10": 0.80, "w5": 0.20, "w3": 0.00, "top_k": 2, "scale": 1.00, "cap": 0.98, "holding_days": 5},
    {"name": "w10_75_w5_20_w3_05_top2_h5", "w10": 0.75, "w5": 0.20, "w3": 0.05, "top_k": 2, "scale": 1.00, "cap": 0.98, "holding_days": 5},
    {"name": "w10_70_w5_20_w3_10_top2_h5", "w10": 0.70, "w5": 0.20, "w3": 0.10, "top_k": 2, "scale": 1.00, "cap": 0.98, "holding_days": 5},
]


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _is_missing(value) -> bool:
    if value in (None, "", "None", "NONE", "nan", "NaN"):
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _market_rows() -> dict[str, dict[str, dict]]:
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, stock_code, name, pre_close, open, close,
                   limit_times, ST_TYPE AS st_type, ST_TYPE_name AS st_type_name
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= ? AND trade_date <= ?
            """,
            (START_DATE, END_DATE),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        out.setdefault(str(row["trade_date"]), {})[str(row["stock_code"])] = dict(row)
    return out


def _date_map() -> dict[str, str]:
    conn = sqlite3.connect(FUSION_DB)
    try:
        dates = [str(row[0]) for row in conn.execute("SELECT DISTINCT trade_date FROM fusion_rank_base WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date", (START_DATE, END_DATE))]
    finally:
        conn.close()
    return {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}


def _is_st_like(row: dict | None) -> bool:
    if not row:
        return True
    name = str(row.get("name") or "")
    st_type = row.get("st_type")
    st_name = str(row.get("st_type_name") or "")
    if name.startswith(("ST", "*ST")) or "退" in name:
        return True
    if "风险" in st_name or "退市" in st_name:
        return True
    if _is_missing(st_type):
        return False
    text = str(st_type).strip().upper()
    return text not in {"0", "0.0", "FALSE", "NONE", "NAN"}


def _is_limit_buy(row: dict | None) -> bool:
    if not row:
        return True
    limit_times = row.get("limit_times")
    if not _is_missing(limit_times) and (_to_float(limit_times, 0.0) or 0.0) > 0.0:
        return True
    pre_close = _to_float(row.get("pre_close"))
    open_price = _to_float(row.get("open"))
    if pre_close is None or open_price is None or pre_close <= 0 or open_price <= 0:
        return True
    code = str(row.get("stock_code") or "")
    pct = 0.20 if code.startswith(("300", "301", "688")) else 0.10
    return open_price >= pre_close * (1.0 + pct) * 0.995


def _build_fusion_db() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if FUSION_DB.exists():
        return
    conn = sqlite3.connect(FUSION_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS model", (str(MODEL_DB),))
        conn.execute("ATTACH DATABASE ? AS market", (str(MARKET_DB),))
        conn.executescript(
            f"""
            CREATE TABLE fusion_rank_base AS
            SELECT
                t10.trade_date,
                t10.stock_code,
                t3.pred_prob AS pred_3d,
                t5.pred_prob AS pred_5d,
                t10.pred_prob AS pred_10d,
                PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t3.pred_prob) AS rank_3d,
                PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t5.pred_prob) AS rank_5d,
                PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t10.pred_prob) AS rank_10d,
                m.name, m.pre_close, m.open, m.close, m.amount, m.turnover_rate,
                m.total_mv, m.atr_qfq, m.limit_times, m.ST_TYPE AS st_type, m.ST_TYPE_name AS st_type_name
            FROM model.{_quote(TABLE_10D)} t10
            JOIN model.{_quote(TABLE_5D)} t5
              ON t10.trade_date = t5.trade_date AND t10.stock_code = t5.stock_code
            JOIN model.{_quote(TABLE_3D)} t3
              ON t10.trade_date = t3.trade_date AND t10.stock_code = t3.stock_code
            LEFT JOIN market.STOCK_DAILY_DATA m
              ON t10.trade_date = m.trade_date AND t10.stock_code = m.stock_code;
            CREATE INDEX idx_fusion_trade_stock ON fusion_rank_base(trade_date, stock_code);
            CREATE INDEX idx_fusion_trade ON fusion_rank_base(trade_date);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _expr(cfg: dict) -> str:
    return (
        f"(rank_10d * {float(cfg['w10']):.12g}) + "
        f"(rank_5d * {float(cfg['w5']):.12g}) + "
        f"(rank_3d * {float(cfg['w3']):.12g})"
    )


def _score_table(cfg: dict) -> str:
    return "score_" + re.sub(r"[^A-Za-z0-9_]+", "_", cfg["name"])


def _build_score_table(cfg: dict) -> str:
    table = _score_table(cfg)
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SCORE_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS fusion", (str(FUSION_DB),))
        expr = _expr(cfg)
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS "{table}";
            CREATE TABLE "{table}" AS
            SELECT trade_date, stock_code, {expr} AS pred_prob
            FROM fusion.fusion_rank_base
            WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}';
            CREATE INDEX idx_{table}_trade_stock ON "{table}"(trade_date, stock_code);
            CREATE INDEX idx_{table}_trade_pred ON "{table}"(trade_date, pred_prob DESC);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return table


def _build_signals(cfg: dict) -> list[dict]:
    market = _market_rows()
    next_date = _date_map()
    expr = _expr(cfg)
    conn = sqlite3.connect(FUSION_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT *
            FROM (
                SELECT
                    trade_date, stock_code, name, pred_3d, pred_5d, pred_10d,
                    rank_3d, rank_5d, rank_10d, amount, turnover_rate, total_mv,
                    atr_qfq, close, limit_times, st_type, st_type_name,
                    {expr} AS score,
                    ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY {expr} DESC, stock_code) AS rn
                FROM fusion_rank_base
                WHERE trade_date >= ? AND trade_date <= ?
                  AND stock_code NOT LIKE '%.BJ'
                  AND COALESCE(name, '') NOT LIKE 'ST%'
                  AND COALESCE(name, '') NOT LIKE '*ST%'
                  AND COALESCE(name, '') NOT LIKE '%退%'
                  AND (st_type IS NULL OR st_type = '' OR st_type = 'None' OR UPPER(CAST(st_type AS TEXT)) IN ('0', '0.0', 'FALSE', 'NONE', 'NAN'))
                  AND COALESCE(st_type_name, '') NOT LIKE '%风险%'
                  AND (limit_times IS NULL OR limit_times = '' OR limit_times = 'None' OR CAST(limit_times AS REAL) = 0)
                  AND rank_3d IS NOT NULL AND rank_5d IS NOT NULL AND rank_10d IS NOT NULL
                  AND close IS NOT NULL
            )
            WHERE rn <= ?
            ORDER BY trade_date, rn
            """,
            (START_DATE, END_DATE, int(cfg["top_k"])),
        ).fetchall()
    finally:
        conn.close()
    signals = []
    targets = BASE_TARGETS[int(cfg["top_k"])]
    for row in rows:
        item = dict(row)
        signal_date = str(item["trade_date"])
        buy_date = next_date.get(signal_date)
        if not buy_date:
            continue
        buy_market = market.get(buy_date, {}).get(str(item["stock_code"]))
        if _is_st_like(buy_market) or _is_limit_buy(buy_market):
            continue
        rank = int(item["rn"])
        raw_target = float(targets.get(rank, 0.0))
        target = min(raw_target * float(cfg["scale"]), float(cfg["cap"]))
        signals.append(
            {
                "signal_date": signal_date,
                "buy_date": buy_date,
                "symbol": to_gm_symbol(str(item["stock_code"])),
                "stock_code": item["stock_code"],
                "name": item.get("name"),
                "rank": rank,
                "pred_prob": item["score"],
                "pred_3d": item.get("pred_3d"),
                "pred_5d": item.get("pred_5d"),
                "pred_10d": item.get("pred_10d"),
                "rank_3d": item.get("rank_3d"),
                "rank_5d": item.get("rank_5d"),
                "rank_10d": item.get("rank_10d"),
                "target_pct": f"{target:.5f}",
                "holding_days": int(cfg["holding_days"]),
                "max_holding_days": int(cfg["holding_days"]),
            }
        )
    return signals


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


def _extract_indicator(log_file: Path) -> dict | None:
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


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
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


def _stats(rows: list[dict]) -> dict:
    day_sums: dict[str, float] = {}
    for row in rows:
        day = str(row.get("buy_date") or "")
        day_sums[day] = day_sums.get(day, 0.0) + (_to_float(row.get("target_pct"), 0.0) or 0.0)
    return {
        "signal_count": len(rows),
        "buy_days": len(day_sums),
        "avg_day_target_sum": sum(day_sums.values()) / len(day_sums) if day_sums else None,
        "min_day_target_sum": min(day_sums.values()) if day_sums else None,
        "max_day_target_sum": max(day_sums.values()) if day_sums else None,
    }


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def _run_backtest(cfg: dict, signal_file: Path, log_file: Path, score_table: str) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update({str(k): str(v) for k, v in cfg.get("env", {}).items()})
    env["GM_MAX_DAILY_SELLS"] = str(cfg["top_k"])
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
        str(cfg["top_k"]),
        "--holding-days",
        str(cfg["holding_days"]),
        "--max-holding-days",
        str(cfg["holding_days"]),
        "--target-position-pct",
        str(float(cfg["cap"])),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        score_table,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        BACKTEST_START,
        "--backtest-end",
        BACKTEST_END,
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


def main() -> int:
    _build_fusion_db()
    results = []
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    for index, cfg in enumerate(VARIANTS, start=1):
        score_table = _build_score_table(cfg)
        signals = _build_signals(cfg)
        signal_file = REPORT_DIR / "signals" / f"{cfg['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{cfg['name']}.log"
        _write_rows(signal_file, signals)
        returncode = _run_backtest(cfg, signal_file, log_file, score_table) if signals else 2
        indicator = _extract_indicator(log_file) or {}
        row = {
            "name": cfg["name"],
            "returncode": returncode,
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            "score_table": score_table,
            "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            **_stats(signals),
            **_exposure_stats(log_file),
        }
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {cfg['name']} annual={row.get('annual')} "
            f"sharpe={row.get('sharpe')} maxdd={row.get('max_drawdown')} signals={row.get('signal_count')}",
            flush=True,
        )
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(REPORT_DIR / "summary_target_hits.csv", [row for row in results if _metric(row, "annual") >= 5.0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
