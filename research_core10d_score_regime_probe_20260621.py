from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path

from gm_signal_module import to_gm_symbol
from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "core10d_score_regime_probe_20260621"
)
TABLE_10D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618"
TABLE_5D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_5d_open_return_score_20240604_20260618"

CONFIGS = [
    {"name": "baseline_rebuild", "mode": "baseline"},
    {"name": "exclude_gt0p5_always", "mode": "exclude_gt0p5_always"},
    {"name": "exclude_gt0p5_highday", "mode": "exclude_gt0p5_highday"},
    {"name": "exclude_gt0p5_highday_lowlt0p02_lowday", "mode": "combined"},
    {"name": "band_0p02_0p5_highday", "mode": "band_highday"},
]


def _is_open_limit_up(stock_code: str, pre_close, open_price) -> bool:
    try:
        pre_close = float(pre_close)
        open_price = float(open_price)
    except (TypeError, ValueError):
        return False
    if pre_close <= 0 or open_price <= 0:
        return False
    pct = 0.20 if str(stock_code).startswith(("300", "301", "688")) else 0.10
    return open_price >= pre_close * (1.0 + pct) * 0.995


def _prepare_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(PRED_DB), timeout=120)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(f"ATTACH DATABASE '{MARKET_DB.as_posix()}' AS market")
    conn.execute("DROP TABLE IF EXISTS temp.rank5d_ok")
    conn.execute(
        f"""
        CREATE TEMP TABLE rank5d_ok AS
        WITH ranked AS (
            SELECT trade_date, stock_code,
                   1.0 - ((ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) - 1.0)
                   / NULLIF(COUNT(*) OVER (PARTITION BY trade_date) - 1.0, 0.0)) AS rank5d
            FROM "{TABLE_5D}"
            WHERE pred_prob IS NOT NULL
        )
        SELECT trade_date, stock_code, rank5d
        FROM ranked
        WHERE rank5d >= 0.5
        """
    )
    conn.execute("CREATE INDEX idx_rank5d_ok_date_code ON rank5d_ok(trade_date, stock_code)")
    return conn


def _trade_dates(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(f'SELECT DISTINCT trade_date FROM "{TABLE_10D}" ORDER BY trade_date').fetchall()
    ]


def _select_day(conn: sqlite3.Connection, signal_date: str, buy_date: str, mode: str) -> list[sqlite3.Row]:
    sql = f"""
        SELECT t.trade_date, t.stock_code, t.pred_prob,
               COALESCE(m.name, n.name, '') AS name,
               n.open AS next_open, n.pre_close AS next_pre_close
        FROM "{TABLE_10D}" t
        JOIN rank5d_ok r
          ON r.trade_date = t.trade_date AND r.stock_code = t.stock_code
        LEFT JOIN market.STOCK_DAILY_DATA m
          ON m.trade_date = t.trade_date AND m.stock_code = t.stock_code
        LEFT JOIN market.STOCK_DAILY_DATA n
          ON n.trade_date = ? AND n.stock_code = t.stock_code
        WHERE t.trade_date = ?
          AND t.pred_prob IS NOT NULL
          AND t.stock_code NOT LIKE '%.BJ'
          AND t.close_rate BETWEEN 0.965 AND 1.085
          AND t.total_mv IS NOT NULL AND t.total_mv <= 200000
          AND (m.name IS NULL OR (m.name NOT LIKE 'ST%' AND m.name NOT LIKE '*ST%' AND m.name NOT LIKE '退%' AND m.name NOT LIKE '%退市%'))
          AND (m.ST_TYPE IS NULL OR m.ST_TYPE IN ('', '0', '0.0', 'None', 'NONE'))
          AND (m.limit_times IS NULL OR m.limit_times IN ('', 'None', 'NONE'))
          AND (n.limit_times IS NULL OR n.limit_times IN ('', 'None', 'NONE'))
        ORDER BY t.pred_prob DESC, t.stock_code
        LIMIT 800
    """
    candidates = []
    for row in conn.execute(sql, (buy_date, signal_date)):
        if _is_open_limit_up(row["stock_code"], row["next_pre_close"], row["next_open"]):
            continue
        candidates.append(row)
    baseline_values = [float(row["pred_prob"]) for row in candidates[:5]]
    top_score = max(baseline_values) if baseline_values else 0.0
    mean_score = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0
    selected = []
    for row in candidates:
        score = float(row["pred_prob"])
        ok = True
        if mode == "exclude_gt0p5_always" and score > 0.5:
            ok = False
        elif mode == "exclude_gt0p5_highday" and top_score > 0.5 and score > 0.5:
            ok = False
        elif mode == "combined":
            if top_score > 0.5 and score > 0.5:
                ok = False
            if mean_score < 0.05 and score < 0.02:
                ok = False
        elif mode == "band_highday" and top_score > 0.5 and not (0.02 <= score <= 0.5):
            ok = False
        if ok:
            selected.append(row)
        if len(selected) >= 5:
            break
    return selected


def build_signal(config: dict, conn: sqlite3.Connection) -> Path:
    signal_file = REPORT_DIR / "signals" / f"{config['name']}.csv"
    if signal_file.exists() and signal_file.stat().st_size > 0:
        return signal_file
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    dates = _trade_dates(conn)
    rows = []
    for idx, signal_date in enumerate(dates[:-1]):
        buy_date = dates[idx + 1]
        selected = _select_day(conn, signal_date, buy_date, config["mode"])
        for rank, row in enumerate(selected, start=1):
            rows.append(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "symbol": to_gm_symbol(row["stock_code"]),
                    "stock_code": row["stock_code"],
                    "name": row["name"],
                    "rank": rank,
                    "pred_prob": row["pred_prob"],
                    "atr_ratio": "",
                    "holding_days": 5,
                    "target_pct": 0.196,
                }
            )
    with signal_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "signal_date",
                "buy_date",
                "symbol",
                "stock_code",
                "name",
                "rank",
                "pred_prob",
                "atr_ratio",
                "holding_days",
                "target_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return signal_file


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
    active_positions = []
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        active_positions.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active_positions) if active_positions else None,
        "exposure_points": len(values),
    }


def run_backtest(config: dict, signal_file: Path) -> tuple[int, Path]:
    log_file = REPORT_DIR / "logs" / f"{config['name']}.log"
    if log_file.exists() and _extract_indicator(log_file):
        return 0, log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_MAX_DAILY_SELLS": "1",
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
        }
    )
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
        "5",
        "--holding-days",
        "5",
        "--max-holding-days",
        "5",
        "--target-position-pct",
        "0.196",
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    return proc.returncode, log_file


def main() -> None:
    rows = []
    conn = _prepare_connection()
    try:
        for config in CONFIGS:
            signal_file = build_signal(config, conn)
            returncode, log_file = run_backtest(config, signal_file)
            indicator = _extract_indicator(log_file)
            signal_count = sum(1 for _ in csv.DictReader(signal_file.open(encoding="utf-8-sig", newline="")))
            row = {
                "name": config["name"],
                "mode": config["mode"],
                "returncode": returncode,
                "signal_count": signal_count,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                **_exposure_stats(log_file),
            }
            if indicator:
                row.update(
                    {
                        "annual": indicator.get("pnl_ratio_annual"),
                        "sharpe": indicator.get("sharp_ratio"),
                        "max_drawdown": indicator.get("max_drawdown"),
                        "open_count": indicator.get("open_count"),
                        "close_count": indicator.get("close_count"),
                        "win_ratio": indicator.get("win_ratio"),
                        "calmar_ratio": indicator.get("calmar_ratio"),
                    }
                )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, default=str))
    finally:
        conn.close()
    fieldnames = list(rows[0].keys())
    with (REPORT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    qualified = [
        row
        for row in rows
        if row.get("annual") is not None
        and float(row["annual"]) > 2.0
        and row.get("avg_invested_pct") is not None
        and float(row["avg_invested_pct"]) >= 0.80
    ]
    with (REPORT_DIR / "summary_qualified_annual2_avg80_by_sharpe.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(qualified, key=lambda row: float(row["sharpe"]), reverse=True))


if __name__ == "__main__":
    main()
