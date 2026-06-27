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
from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, STRATEGY_DIR
from research_rebuild_standard_fusion_signals_20260621 import FUSION_DB, build_fusion_db


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "standard_fusion_atr_filter_probe_20260621"
)

CONFIGS = []
for atr_max in (0.04, 0.05, 0.06, 0.08, 0.10):
    CONFIGS.append(
        {
            "fusion_table": "fusion_min_consensus",
            "top_k": 3,
            "holding_days": 15,
            "max_positions": 3,
            "target_pct": 0.3267,
            "atr_max": atr_max,
            "min_close_rate": 0.965,
            "max_close_rate": 1.085,
            "open_daily_score_exit": 1,
            "score_exit_entry_ratio": 0.95,
            "max_daily_sells": 0,
        }
    )
    CONFIGS.append(
        {
            "fusion_table": "fusion_min_consensus",
            "top_k": 5,
            "holding_days": 5,
            "max_positions": 5,
            "target_pct": 0.196,
            "atr_max": atr_max,
            "min_close_rate": 0.965,
            "max_close_rate": 1.085,
            "open_daily_score_exit": 0,
            "score_exit_entry_ratio": None,
            "max_daily_sells": 1,
        }
    )
for table in ("fusion_eq_3d5d10d", "fusion_10d60_5d30_3d10", "fusion_10d80_5d20"):
    for atr_max in (0.05, 0.08):
        CONFIGS.append(
            {
                "fusion_table": table,
                "top_k": 5,
                "holding_days": 5,
                "max_positions": 5,
                "target_pct": 0.196,
                "atr_max": atr_max,
                "min_close_rate": 0.965,
                "max_close_rate": 1.085,
                "open_daily_score_exit": 0,
                "score_exit_entry_ratio": None,
                "max_daily_sells": 1,
            }
        )


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none")


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


def build_signal(config: dict) -> Path:
    signal_file = REPORT_DIR / "signals" / (
        f"{config['fusion_table']}_tk{config['top_k']}_h{config['holding_days']}"
        f"_atr{_safe(config['atr_max'])}_cr{_safe(config['min_close_rate'])}_{_safe(config['max_close_rate'])}"
        f"_exit{config['open_daily_score_exit']}.csv"
    )
    if signal_file.exists() and signal_file.stat().st_size > 0:
        return signal_file
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(FUSION_DB))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"ATTACH DATABASE '{MARKET_DB.as_posix()}' AS market")
        trade_dates = [
            str(row[0])
            for row in conn.execute(
                f'SELECT DISTINCT trade_date FROM "{config["fusion_table"]}" ORDER BY trade_date'
            )
        ]
        rows = []
        for idx, signal_date in enumerate(trade_dates[:-1]):
            buy_date = trade_dates[idx + 1]
            sql = f"""
                SELECT f.trade_date, f.stock_code, f.pred_prob, f.atr_qfq, f.close,
                       COALESCE(m.name, n.name, '') AS name,
                       n.open AS next_open, n.pre_close AS next_pre_close
                FROM "{config['fusion_table']}" f
                LEFT JOIN market.STOCK_DAILY_DATA m
                  ON m.trade_date = f.trade_date AND m.stock_code = f.stock_code
                LEFT JOIN market.STOCK_DAILY_DATA n
                  ON n.trade_date = ? AND n.stock_code = f.stock_code
                WHERE f.trade_date = ?
                  AND f.stock_code NOT LIKE '%.BJ'
                  AND f.pred_prob IS NOT NULL
                  AND f.atr_qfq IS NOT NULL
                  AND f.close IS NOT NULL
                  AND f.close > 0
                  AND f.atr_qfq / f.close <= ?
                  AND f.close_rate >= ?
                  AND f.close_rate <= ?
                  AND (m.name IS NULL OR (m.name NOT LIKE 'ST%' AND m.name NOT LIKE '*ST%' AND m.name NOT LIKE '退%' AND m.name NOT LIKE '%退市%'))
                  AND (m.ST_TYPE IS NULL OR m.ST_TYPE IN ('', '0', '0.0', 'None', 'NONE'))
                  AND (m.limit_times IS NULL OR m.limit_times IN ('', 'None', 'NONE'))
                  AND (n.limit_times IS NULL OR n.limit_times IN ('', 'None', 'NONE'))
                ORDER BY f.pred_prob DESC, f.stock_code
                LIMIT 200
            """
            selected = []
            for row in conn.execute(
                sql,
                (
                    buy_date,
                    signal_date,
                    config["atr_max"],
                    config["min_close_rate"],
                    config["max_close_rate"],
                ),
            ):
                if _is_open_limit_up(row["stock_code"], row["next_pre_close"], row["next_open"]):
                    continue
                selected.append(row)
                if len(selected) >= int(config["top_k"]):
                    break
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
                        "atr_ratio": float(row["atr_qfq"]) / float(row["close"]),
                        "holding_days": config["holding_days"],
                        "target_pct": config["target_pct"],
                    }
                )
    finally:
        conn.close()
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
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
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
    log_file = REPORT_DIR / "logs" / f"{signal_file.stem}.log"
    if log_file.exists() and _extract_indicator(log_file):
        return 0, log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(config["open_daily_score_exit"]),
            "GM_MAX_DAILY_SELLS": str(config["max_daily_sells"]),
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
        str(config["max_positions"]),
        "--holding-days",
        str(config["holding_days"]),
        "--max-holding-days",
        str(config["holding_days"]),
        "--target-position-pct",
        str(config["target_pct"]),
        "--score-db",
        str(FUSION_DB),
        "--score-table",
        config["fusion_table"],
        "--market-db",
        str(MARKET_DB),
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    if config["score_exit_entry_ratio"] is not None:
        command.extend(["--score-exit-entry-ratio", str(config["score_exit_entry_ratio"])])
    proc = subprocess.run(command, cwd=str(MAIN), env=env, capture_output=True, text=True)
    return proc.returncode, log_file


def main() -> None:
    build_fusion_db()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in CONFIGS:
        signal_file = build_signal(config)
        returncode, log_file = run_backtest(config, signal_file)
        indicator = _extract_indicator(log_file)
        row = {
            "name": signal_file.stem,
            "returncode": returncode,
            **config,
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
