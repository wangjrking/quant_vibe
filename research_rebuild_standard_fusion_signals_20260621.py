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
from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "standard_fusion_rebuild_probe_20260621"
)
FUSION_DB = REPORT_DIR / "standard_fusion_scores.db"

TABLE_3D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_3d_open_return_score_20240604_20260618"
TABLE_5D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_5d_open_return_score_20240604_20260618"
TABLE_10D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618"

FUSION_TABLES = {
    "fusion_min_consensus": "MIN(rank_3d, rank_5d, rank_10d)",
    "fusion_eq_3d5d10d": "(rank_3d + rank_5d + rank_10d) / 3.0",
    "fusion_10d60_5d30_3d10": "rank_10d * 0.60 + rank_5d * 0.30 + rank_3d * 0.10",
    "fusion_10d70_5d20_3d10": "rank_10d * 0.70 + rank_5d * 0.20 + rank_3d * 0.10",
    "fusion_10d80_5d20": "rank_10d * 0.80 + rank_5d * 0.20",
}

SIGNAL_CONFIGS = [
    {
        "fusion_table": "fusion_min_consensus",
        "top_k": 5,
        "holding_days": 5,
        "max_positions": 5,
        "target_pct": 0.196,
        "max_total_mv": None,
        "min_close_rate": 0.965,
        "max_close_rate": 1.085,
        "min_amount": None,
        "open_daily_score_exit": 0,
        "score_exit_entry_ratio": None,
        "min_hold_before_exit": 2,
        "max_daily_sells": 1,
    },
    {
        "fusion_table": "fusion_min_consensus",
        "top_k": 3,
        "holding_days": 15,
        "max_positions": 3,
        "target_pct": 0.3267,
        "max_total_mv": None,
        "min_close_rate": 0.965,
        "max_close_rate": 1.085,
        "min_amount": None,
        "open_daily_score_exit": 1,
        "score_exit_entry_ratio": 0.95,
        "min_hold_before_exit": 2,
        "max_daily_sells": 0,
    },
    {
        "fusion_table": "fusion_min_consensus",
        "top_k": 5,
        "holding_days": 10,
        "max_positions": 5,
        "target_pct": 0.196,
        "max_total_mv": None,
        "min_close_rate": 0.965,
        "max_close_rate": 1.085,
        "min_amount": None,
        "open_daily_score_exit": 1,
        "score_exit_entry_ratio": 0.95,
        "min_hold_before_exit": 2,
        "max_daily_sells": 0,
    },
    {
        "fusion_table": "fusion_10d60_5d30_3d10",
        "top_k": 5,
        "holding_days": 5,
        "max_positions": 5,
        "target_pct": 0.196,
        "max_total_mv": 300000,
        "min_close_rate": 0.965,
        "max_close_rate": 1.085,
        "min_amount": None,
        "open_daily_score_exit": 0,
        "score_exit_entry_ratio": None,
        "min_hold_before_exit": 2,
        "max_daily_sells": 1,
    },
    {
        "fusion_table": "fusion_10d70_5d20_3d10",
        "top_k": 5,
        "holding_days": 5,
        "max_positions": 5,
        "target_pct": 0.196,
        "max_total_mv": 300000,
        "min_close_rate": 0.965,
        "max_close_rate": 1.085,
        "min_amount": None,
        "open_daily_score_exit": 0,
        "score_exit_entry_ratio": None,
        "min_hold_before_exit": 2,
        "max_daily_sells": 1,
    },
    {
        "fusion_table": "fusion_10d80_5d20",
        "top_k": 5,
        "holding_days": 5,
        "max_positions": 5,
        "target_pct": 0.196,
        "max_total_mv": 250000,
        "min_close_rate": 0.965,
        "max_close_rate": 1.085,
        "min_amount": None,
        "open_daily_score_exit": 0,
        "score_exit_entry_ratio": None,
        "min_hold_before_exit": 2,
        "max_daily_sells": 1,
    },
    {
        "fusion_table": "fusion_eq_3d5d10d",
        "top_k": 5,
        "holding_days": 5,
        "max_positions": 5,
        "target_pct": 0.196,
        "max_total_mv": 300000,
        "min_close_rate": 0.965,
        "max_close_rate": 1.085,
        "min_amount": None,
        "open_daily_score_exit": 0,
        "score_exit_entry_ratio": None,
        "min_hold_before_exit": 2,
        "max_daily_sells": 1,
    },
]


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none")


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def build_fusion_db() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = _connect(FUSION_DB)
    try:
        existing = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if set(FUSION_TABLES).issubset(existing):
            return
        conn.execute(f"ATTACH DATABASE '{PRED_DB.as_posix()}' AS pred")
        conn.execute("DROP TABLE IF EXISTS fusion_rank_base")
        conn.execute(
            f"""
            CREATE TABLE fusion_rank_base AS
            WITH
            r3 AS (
                SELECT trade_date, stock_code, pred_prob AS pred_3d,
                       1.0 - ((ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) - 1.0)
                       / NULLIF(COUNT(*) OVER (PARTITION BY trade_date) - 1.0, 0.0)) AS rank_3d
                FROM pred."{TABLE_3D}"
                WHERE pred_prob IS NOT NULL
            ),
            r5 AS (
                SELECT trade_date, stock_code, pred_prob AS pred_5d,
                       1.0 - ((ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) - 1.0)
                       / NULLIF(COUNT(*) OVER (PARTITION BY trade_date) - 1.0, 0.0)) AS rank_5d
                FROM pred."{TABLE_5D}"
                WHERE pred_prob IS NOT NULL
            ),
            r10 AS (
                SELECT trade_date, stock_code, pred_prob AS pred_10d,
                       close, pre_close, industry, industry_encode, atr_qfq, close_rate,
                       amount, turnover_rate, turnover_rate_f, circ_mv, total_mv, volume_ratio,
                       1.0 - ((ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code) - 1.0)
                       / NULLIF(COUNT(*) OVER (PARTITION BY trade_date) - 1.0, 0.0)) AS rank_10d
                FROM pred."{TABLE_10D}"
                WHERE pred_prob IS NOT NULL
            )
            SELECT r10.trade_date, r10.stock_code,
                   r3.pred_3d, r5.pred_5d, r10.pred_10d,
                   r3.rank_3d, r5.rank_5d, r10.rank_10d,
                   r10.close, r10.pre_close, r10.industry, r10.industry_encode,
                   r10.atr_qfq, r10.close_rate, r10.amount, r10.turnover_rate,
                   r10.turnover_rate_f, r10.circ_mv, r10.total_mv, r10.volume_ratio
            FROM r10
            JOIN r5 ON r5.trade_date = r10.trade_date AND r5.stock_code = r10.stock_code
            JOIN r3 ON r3.trade_date = r10.trade_date AND r3.stock_code = r10.stock_code
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fusion_rank_base_date_score ON fusion_rank_base(trade_date, rank_10d)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fusion_rank_base_code_date ON fusion_rank_base(stock_code, trade_date)")
        for table, expression in FUSION_TABLES.items():
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(
                f"""
                CREATE TABLE {table} AS
                SELECT trade_date, stock_code,
                       ({expression}) AS pred_prob,
                       pred_3d, pred_5d, pred_10d,
                       rank_3d, rank_5d, rank_10d,
                       close, pre_close, industry, industry_encode, atr_qfq, close_rate,
                       amount, turnover_rate, turnover_rate_f, circ_mv, total_mv, volume_ratio
                FROM fusion_rank_base
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_date_score ON {table}(trade_date, pred_prob)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_code_date ON {table}(stock_code, trade_date)")
        conn.commit()
    finally:
        conn.close()


def _to_float(value):
    if value in (None, "", "None"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _is_st_or_delisting(name: str | None, st_type: str | None) -> bool:
    text = str(name or "")
    if text.startswith(("ST", "*ST", "退")) or "退市" in text:
        return True
    st_text = str(st_type or "").strip().upper()
    if st_text in {"", "0", "0.0", "NONE", "NULL", "NAN", "FALSE"}:
        return False
    return True


def _limit_up_pct(stock_code: str, name: str | None = None, st_type: str | None = None) -> float:
    if _is_st_or_delisting(name, st_type):
        return 0.05
    if str(stock_code or "").startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def _is_open_limit_up(stock_code: str, name: str | None, st_type: str | None, pre_close, open_price) -> bool:
    pre_close_value = _to_float(pre_close)
    open_value = _to_float(open_price)
    if pre_close_value in (None, 0) or open_value in (None, 0):
        return False
    return open_value >= pre_close_value * (1.0 + _limit_up_pct(stock_code, name, st_type)) * 0.995


def _load_trade_dates(conn: sqlite3.Connection, table: str) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(f'SELECT DISTINCT trade_date FROM "{table}" ORDER BY trade_date').fetchall()
    ]


def build_signal(config: dict) -> Path:
    output = REPORT_DIR / "signals" / (
        f"{config['fusion_table']}_tk{config['top_k']}_h{config['holding_days']}"
        f"_pos{config['max_positions']}_mv{_safe(config['max_total_mv'])}"
        f"_cr{_safe(config['min_close_rate'])}_{_safe(config['max_close_rate'])}"
        f"_amt{_safe(config['min_amount'])}_exit{config['open_daily_score_exit']}"
        f"_ratio{_safe(config['score_exit_entry_ratio'])}.csv"
    )
    if output.exists() and output.stat().st_size > 0:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(FUSION_DB))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"ATTACH DATABASE '{MARKET_DB.as_posix()}' AS market")
        trade_dates = _load_trade_dates(conn, config["fusion_table"])
        next_date_by_signal_date = {
            trade_dates[idx]: trade_dates[idx + 1]
            for idx in range(len(trade_dates) - 1)
        }
        rows = []
        for signal_date, buy_date in next_date_by_signal_date.items():
            params = {
                "signal_date": signal_date,
                "buy_date": buy_date,
                "min_close_rate": config["min_close_rate"],
                "max_close_rate": config["max_close_rate"],
            }
            conditions = [
                "f.trade_date = :signal_date",
                "f.pred_prob IS NOT NULL",
                "f.stock_code NOT LIKE '%.BJ'",
                "(m.name IS NULL OR (m.name NOT LIKE 'ST%' AND m.name NOT LIKE '*ST%' AND m.name NOT LIKE '退%' AND m.name NOT LIKE '%退市%'))",
                "(m.ST_TYPE IS NULL OR m.ST_TYPE IN ('', '0', '0.0', 'None', 'NONE'))",
                "(m.limit_times IS NULL OR m.limit_times IN ('', 'None', 'NONE'))",
                "(n.limit_times IS NULL OR n.limit_times IN ('', 'None', 'NONE'))",
                "f.close_rate >= :min_close_rate",
                "f.close_rate <= :max_close_rate",
            ]
            if config["max_total_mv"] is not None:
                conditions.append("m.total_mv IS NOT NULL AND m.total_mv <= :max_total_mv")
                params["max_total_mv"] = config["max_total_mv"]
            if config["min_amount"] is not None:
                conditions.append("m.amount IS NOT NULL AND m.amount >= :min_amount")
                params["min_amount"] = config["min_amount"]
            sql = f"""
                SELECT f.trade_date, f.stock_code, f.pred_prob, f.atr_qfq,
                       COALESCE(m.name, n.name, '') AS name,
                       m.close AS signal_close, m.pre_close AS signal_pre_close,
                       m.amount, m.turnover_rate, m.total_mv, f.close_rate,
                       n.open AS next_open, n.pre_close AS next_pre_close, n.name AS next_name, n.ST_TYPE AS next_st_type
                FROM "{config['fusion_table']}" f
                LEFT JOIN market.STOCK_DAILY_DATA m
                  ON m.trade_date = f.trade_date AND m.stock_code = f.stock_code
                LEFT JOIN market.STOCK_DAILY_DATA n
                  ON n.trade_date = :buy_date AND n.stock_code = f.stock_code
                WHERE {" AND ".join(conditions)}
                ORDER BY f.pred_prob DESC, f.stock_code
                LIMIT 200
            """
            selected = []
            for row in conn.execute(sql, params):
                stock_code = str(row["stock_code"])
                if _is_open_limit_up(
                    stock_code,
                    row["next_name"],
                    row["next_st_type"],
                    row["next_pre_close"],
                    row["next_open"],
                ):
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
                        "atr_ratio": "",
                        "holding_days": config["holding_days"],
                        "target_pct": config["target_pct"],
                    }
                )
        with output.open("w", encoding="utf-8-sig", newline="") as file:
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
    finally:
        conn.close()
    return output


def _extract_indicator(log_file: Path) -> dict | None:
    marker = "GM_BACKTEST_INDICATOR:"
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
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
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
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
    name = signal_file.stem
    log_file = REPORT_DIR / "logs" / f"{name}.log"
    if log_file.exists() and log_file.stat().st_size > 0 and _extract_indicator(log_file):
        return 0, log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(config["open_daily_score_exit"]),
            "GM_MAX_DAILY_SELLS": str(config["max_daily_sells"]),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(config["min_hold_before_exit"]),
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
    rows = []
    for config in SIGNAL_CONFIGS:
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
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
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
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(qualified, key=lambda row: float(row["sharpe"]), reverse=True))


if __name__ == "__main__":
    main()
