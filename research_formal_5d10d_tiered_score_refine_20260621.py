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

import research_formal_5d10d_tiered_fill_20260621 as tiered


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
SOURCE_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
)
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_tiered_score_refine_20260621"
)
SCORE_DB = REPORT_DIR / "tiered_scores.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


BASES = [
    {
        "name": "p90_f50",
        "primary": {"asset": "rank_10d90_5d10", "max_total_mv": 150000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3},
        "fallback": {"asset": "rank_10d50_5d50", "max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5},
        "primary_limit": 2,
    },
    {
        "name": "p90_f60",
        "primary": {"asset": "rank_10d90_5d10", "max_total_mv": 150000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3},
        "fallback": {"asset": "rank_10d60_5d40", "max_total_mv": 200000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3},
        "primary_limit": 2,
    },
]

EXEC_VARIANTS = [
    {"max_positions": 10, "holding_days": 4, "open_score_exit": 0, "score_exit_ratio": 0.95, "min_hold_score_exit": 2, "max_daily_sells": 1},
    {"max_positions": 10, "holding_days": 4, "open_score_exit": 1, "score_exit_ratio": 0.95, "min_hold_score_exit": 2, "max_daily_sells": 1},
    {"max_positions": 10, "holding_days": 4, "open_score_exit": 1, "score_exit_ratio": 0.90, "min_hold_score_exit": 2, "max_daily_sells": 1},
    {"max_positions": 10, "holding_days": 4, "open_score_exit": 1, "score_exit_ratio": 0.95, "min_hold_score_exit": 1, "max_daily_sells": 1},
    {"max_positions": 10, "holding_days": 4, "open_score_exit": 1, "score_exit_ratio": 0.95, "min_hold_score_exit": 2, "max_daily_sells": 2},
    {"max_positions": 10, "holding_days": 5, "open_score_exit": 1, "score_exit_ratio": 0.95, "min_hold_score_exit": 2, "max_daily_sells": 1},
    {"max_positions": 8, "holding_days": 4, "open_score_exit": 1, "score_exit_ratio": 0.95, "min_hold_score_exit": 2, "max_daily_sells": 1},
]


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


def _condition(alias: str, pool: dict) -> str:
    checks = [
        f"{alias}.stock_code NOT LIKE '%.BJ'",
        f"COALESCE({alias}.name, '') NOT LIKE 'ST%'",
        f"COALESCE({alias}.name, '') NOT LIKE '*ST%'",
        f"COALESCE({alias}.name, '') NOT LIKE '%退市%'",
        f"COALESCE({alias}.name, '') NOT LIKE '退%'",
        f"({alias}.limit_times IS NULL OR {alias}.limit_times = '' OR {alias}.limit_times = 'None' OR CAST({alias}.limit_times AS REAL) = 0)",
        f"{alias}.total_mv IS NOT NULL AND {alias}.total_mv <= {float(pool['max_total_mv'])}",
        f"{alias}.amount IS NOT NULL AND {alias}.amount >= {float(pool['min_amount'])}",
        f"{alias}.turnover_rate IS NOT NULL AND {alias}.turnover_rate >= {float(pool['min_turnover_rate'])}",
    ]
    return " AND ".join(checks)


def _formula(asset: str) -> str:
    formulas = {
        "rank_10d90_5d10": "(rank_10d * 0.90) + (rank_5d * 0.10)",
        "rank_10d80_5d20": "(rank_10d * 0.80) + (rank_5d * 0.20)",
        "rank_10d70_5d30": "(rank_10d * 0.70) + (rank_5d * 0.30)",
        "rank_10d60_5d40": "(rank_10d * 0.60) + (rank_5d * 0.40)",
        "rank_10d50_5d50": "(rank_10d * 0.50) + (rank_5d * 0.50)",
    }
    if asset not in formulas:
        raise ValueError(f"unsupported tiered score asset: {asset}")
    return formulas[asset]


def _table_name(base_cfg: dict) -> str:
    return (
        f"tier_{base_cfg['name']}_plim{base_cfg['primary_limit']}"
        f"_pmv{_safe(base_cfg['primary']['max_total_mv'])}_fmv{_safe(base_cfg['fallback']['max_total_mv'])}"
    )


def _build_score_table(base_cfg: dict, asset_map: dict) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table_name = _table_name(base_cfg)
    if SCORE_DB.exists() and SCORE_DB.stat().st_size == 0:
        SCORE_DB.unlink()
        journal = SCORE_DB.with_name(SCORE_DB.name + "-journal")
        if journal.exists():
            journal.unlink()
    if SCORE_DB.exists():
        conn = sqlite3.connect(SCORE_DB)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if exists:
                return table_name
        finally:
            conn.close()
    source_db = Path(asset_map[base_cfg["fallback"]["asset"]]["db_path"])
    primary_expr = _formula(base_cfg["primary"]["asset"])
    fallback_expr = _formula(base_cfg["fallback"]["asset"])
    conn = sqlite3.connect(SCORE_DB)
    try:
        conn.execute(f"ATTACH DATABASE {_quote_literal(str(source_db))} AS source")
        pcond = _condition("b", base_cfg["primary"])
        fcond = _condition("b", base_cfg["fallback"])
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS {_quote_ident(table_name)};
            CREATE TABLE {_quote_ident(table_name)} AS
            SELECT
                b.trade_date,
                b.stock_code,
                CASE
                    WHEN {pcond} THEN 2.0 + ({primary_expr})
                    WHEN {fcond} THEN 1.0 + ({fallback_expr})
                    ELSE ({fallback_expr})
                END AS pred_prob
            FROM source.fusion_rank_base b
            WHERE b.trade_date >= '20240604' AND b.trade_date <= '20260618'
              AND b.rank_5d IS NOT NULL
              AND b.rank_10d IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_{table_name}_trade_stock ON {_quote_ident(table_name)}(trade_date, stock_code);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return table_name


def _slug(base_cfg: dict, variant: dict) -> str:
    return (
        f"{base_cfg['name']}_plim{base_cfg['primary_limit']}"
        f"_mp{variant['max_positions']}_h{variant['holding_days']}"
        f"_exit{variant['open_score_exit']}_ser{_safe(variant['score_exit_ratio'])}"
        f"_mhse{variant['min_hold_score_exit']}_sells{variant['max_daily_sells']}"
    )


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


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {"signal_count": len(rows), "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")})}


def _run_backtest(base_cfg: dict, variant: dict, table_name: str, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(variant["open_score_exit"]),
            "GM_SCORE_EXIT_ENTRY_RATIO": str(variant["score_exit_ratio"]),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(variant["min_hold_score_exit"]),
            "GM_MAX_DAILY_SELLS": str(variant["max_daily_sells"]),
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.0",
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
        str(variant["max_positions"]),
        "--holding-days",
        str(variant["holding_days"]),
        "--max-holding-days",
        str(variant["holding_days"]),
        "--target-position-pct",
        str(0.98 / float(int(variant["max_positions"]))),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        table_name,
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


def _sort_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row.get("sharpe") or -999.0),
        float(row.get("annual") or -999.0),
        float(row.get("avg_invested_pct") or -999.0),
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    asset_map = tiered._load_asset_map()
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    grid = [(base_cfg, variant) for base_cfg in BASES for variant in EXEC_VARIANTS]
    for index, (base_cfg, variant) in enumerate(grid, start=1):
        table_name = _build_score_table(base_cfg, asset_map)
        slug = _slug(base_cfg, variant)
        signal_file = REPORT_DIR / "signals" / f"{slug}.csv"
        log_file = REPORT_DIR / "logs" / f"{slug}.log"
        if not signal_file.exists():
            params = {
                "asset_map": asset_map,
                "primary": base_cfg["primary"],
                "fallback": base_cfg["fallback"],
                "primary_limit": base_cfg["primary_limit"],
                "holding_days": variant["holding_days"],
                "max_positions": variant["max_positions"],
                "open_score_exit": variant["open_score_exit"],
                "max_daily_sells": variant["max_daily_sells"],
            }
            tiered._write_signal(params, signal_file)
        returncode = _run_backtest(base_cfg, variant, table_name, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            "base_name": base_cfg["name"],
            "score_table": table_name,
            "primary_asset": base_cfg["primary"]["asset"],
            "primary_limit": base_cfg["primary_limit"],
            "fallback_asset": base_cfg["fallback"]["asset"],
            **variant,
            "returncode": returncode,
            "signal_file": str(signal_file),
            "score_db": str(SCORE_DB),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual"),
            "sharpe": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
            **_signal_stats(signal_file),
            **_exposure_stats(log_file),
        }
        results.append(result)
        _write_rows(REPORT_DIR / "summary.csv", results)
        _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=_sort_key, reverse=True))
        print(
            f"[{index}/{len(grid)}] {slug} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}",
            flush=True,
        )
    qualified = [
        row
        for row in sorted(results, key=_sort_key, reverse=True)
        if float(row.get("annual") or -999) >= 2.0
        and float(row.get("sharpe") or -999) >= 3.0
        and float(row.get("avg_invested_pct") or -999) >= 0.80
    ]
    _write_rows(REPORT_DIR / "summary_qualified_target.csv", qualified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
