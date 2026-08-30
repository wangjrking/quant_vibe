from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import duckdb

from export_dynamic_top1_formal_signals import _load_strategy


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_top1_pos60_ddloose_v20260629"
CODE_SNAPSHOT_DIR = STRATEGY_DIR / "code_snapshot"
REPORT_DIR = DATA / "reports" / "strategy_agent_adaptive70w_fouryear_no_intraday_grid_20260629"
SCORE_DB = REPORT_DIR / "scores" / "grid_scores.duckdb"

TIME_SLICES = [
    ("full", "2022-06-07 09:00:00", "2026-06-26 15:30:00"),
    ("recent120", "2025-12-24 09:00:00", "2026-06-26 15:30:00"),
    ("recent60", "2026-03-25 09:00:00", "2026-06-26 15:30:00"),
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-26 15:30:00"),
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "0",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_FORCE_SELL_MARKET_ORDER": "0",
    "GM_FORCE_BUY_MARKET_ORDER": "0",
    "GM_INTRADAY_RISK_MODE": "0",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
    "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
    "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
    "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
    "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
    "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
    "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
    "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
}

ENTRY_CASES = [
    {"name": "w70_25_05_amt150_mv30", "w10d": 0.70, "w5d": 0.25, "w3d": 0.05, "amount_min": 150000.0, "total_mv_min": 300000.0},
    {"name": "w72_23_05_amt150_mv30", "w10d": 0.72, "w5d": 0.23, "w3d": 0.05, "amount_min": 150000.0, "total_mv_min": 300000.0},
    {"name": "w75_20_05_amt150_mv30", "w10d": 0.75, "w5d": 0.20, "w3d": 0.05, "amount_min": 150000.0, "total_mv_min": 300000.0},
    {"name": "w72_23_05_amt200_mv50", "w10d": 0.72, "w5d": 0.23, "w3d": 0.05, "amount_min": 200000.0, "total_mv_min": 500000.0},
]

EXEC_CASES = [
    {
        "name": "h3m5_c097_e096_ddbase_pos40",
        "holding_days": 3,
        "max_holding_days": 5,
        "continue_ratio": 0.97,
        "exit_ratio": 0.96,
        "target_pct": 0.40,
        "stop_loss": 0.05,
        "take_profit": 0.08,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "h3m5_c097_e096_ddbase_pos50",
        "holding_days": 3,
        "max_holding_days": 5,
        "continue_ratio": 0.97,
        "exit_ratio": 0.96,
        "target_pct": 0.50,
        "stop_loss": 0.05,
        "take_profit": 0.08,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "h3m5_c097_e096_ddbase_pos60",
        "holding_days": 3,
        "max_holding_days": 5,
        "continue_ratio": 0.97,
        "exit_ratio": 0.96,
        "target_pct": 0.60,
        "stop_loss": 0.05,
        "take_profit": 0.08,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "h2m3_c097_e097_ddbase_pos50",
        "holding_days": 2,
        "max_holding_days": 3,
        "continue_ratio": 0.97,
        "exit_ratio": 0.97,
        "target_pct": 0.50,
        "stop_loss": 0.05,
        "take_profit": 0.08,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
    {
        "name": "h1m2_c098_e097_ddbase_pos50",
        "holding_days": 1,
        "max_holding_days": 2,
        "continue_ratio": 0.98,
        "exit_ratio": 0.97,
        "target_pct": 0.50,
        "stop_loss": 0.05,
        "take_profit": 0.08,
        "dd_soft": 0.06,
        "dd_hard": 0.10,
        "dd_soft_scale": 0.65,
        "dd_hard_scale": 0.45,
    },
]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


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


def _case_key(entry: dict[str, Any], exe: dict[str, Any]) -> str:
    return f"{entry['name']}__{exe['name']}"


def _score_table(case_key: str) -> str:
    return "score_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in case_key)


def _build_case_assets(
    manifest: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    entry: dict[str, Any],
    exe: dict[str, Any],
) -> dict[str, Any]:
    case_key = _case_key(entry, exe)
    score_table = _score_table(case_key)
    signal_file = REPORT_DIR / "signals" / f"{case_key}.csv"
    if signal_file.exists():
        rows = list(csv.DictReader(signal_file.open("r", encoding="utf-8-sig", newline="")))
        return {"case_key": case_key, "signal_file": signal_file, "score_table": score_table, "signal_rows": len(rows)}

    source_10d = sources["10d"]
    source_5d = sources["5d"]
    source_3d = sources["3d"]
    duckdb_path_10d = Path(str(source_10d["db_path"])).resolve()
    duckdb_path_5d = Path(str(source_5d["db_path"])).resolve()
    duckdb_path_3d = Path(str(source_3d["db_path"])).resolve()
    market_duckdb_path = Path(str(source_10d["market_db_path"])).resolve()
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{duckdb_path_10d.as_posix()}' AS l4_10d (READ_ONLY)")
        con.execute(f"ATTACH '{duckdb_path_5d.as_posix()}' AS l4_5d (READ_ONLY)")
        con.execute(f"ATTACH '{duckdb_path_3d.as_posix()}' AS l4_3d (READ_ONLY)")
        con.execute(f"ATTACH '{market_duckdb_path.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(f"ATTACH '{SCORE_DB.as_posix()}' AS scoredb")
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW joined_scores AS
            WITH base AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM l4_10d."{source_10d['table']}" p10
                INNER JOIN l4_5d."{source_5d['table']}" p5
                    ON p10.trade_date = p5.trade_date
                   AND p10.stock_code = p5.stock_code
                INNER JOIN l4_3d."{source_3d['table']}" p3
                    ON p10.trade_date = p3.trade_date
                   AND p10.stock_code = p3.stock_code
            )
            SELECT
                trade_date,
                stock_code,
                pred_3d,
                pred_5d,
                pred_10d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS rank_3d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS rank_5d,
                percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS rank_10d,
                {float(entry['w10d'])} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d)
              + {float(entry['w5d'])} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d)
              + {float(entry['w3d'])} * percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS entry_score
            FROM base
            """
        )
        con.execute(f'DROP TABLE IF EXISTS scoredb."{score_table}"')
        con.execute(
            f"""
            CREATE TABLE scoredb."{score_table}" AS
            SELECT trade_date, stock_code, entry_score AS pred_prob
            FROM joined_scores
            """
        )
        selected_sql = f"""
        WITH market_dates AS (
            SELECT DISTINCT trade_date
            FROM marketdb.STOCK_DAILY_DATA
        ),
        next_dates AS (
            SELECT
                trade_date AS signal_date,
                lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                max(trade_date) OVER () AS latest_market_date
            FROM market_dates
        ),
        scored AS (
            SELECT
                js.trade_date AS signal_date,
                nd.buy_date,
                nd.latest_market_date,
                js.stock_code,
                md.name,
                md.amount,
                md.turnover_rate,
                md.total_mv,
                md.atr_qfq,
                js.pred_3d,
                js.pred_5d,
                js.pred_10d,
                js.rank_3d,
                js.rank_5d,
                js.rank_10d,
                js.entry_score
            FROM joined_scores js
            LEFT JOIN marketdb.STOCK_DAILY_DATA md
                ON js.trade_date = md.trade_date
               AND js.stock_code = md.stock_code
            LEFT JOIN next_dates nd
                ON js.trade_date = nd.signal_date
            WHERE
                NOT (js.stock_code LIKE '%.BJ' OR substr(js.stock_code, 1, 1) IN ('4', '8'))
                AND NOT (
                    upper(coalesce(md.name, '')) LIKE 'ST%%'
                    OR upper(coalesce(md.name, '')) LIKE '*ST%%'
                    OR coalesce(md.ST_TYPE_name, '') LIKE '%风险%'
                    OR (try_cast(md.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(md.ST_TYPE AS DOUBLE) <> 0)
                )
                AND NOT (
                    instr(coalesce(md.name, ''), '退市') > 0
                    OR coalesce(md.name, '') LIKE '退%'
                    OR coalesce(md.name, '') LIKE '%退'
                )
                AND coalesce(try_cast(md.limit_times AS DOUBLE), 0.0) = 0.0
                AND try_cast(md.amount AS DOUBLE) >= {float(entry['amount_min'])}
                AND try_cast(md.total_mv AS DOUBLE) >= {float(entry['total_mv_min'])}
        ),
        with_buy_checks AS (
            SELECT
                s.*,
                CASE
                    WHEN s.buy_date IS NULL THEN TRUE
                    WHEN bm.stock_code IS NULL THEN FALSE
                    WHEN upper(coalesce(bm.name, '')) LIKE 'ST%%' THEN FALSE
                    WHEN upper(coalesce(bm.name, '')) LIKE '*ST%%' THEN FALSE
                    WHEN coalesce(bm.ST_TYPE_name, '') LIKE '%风险%' THEN FALSE
                    WHEN try_cast(bm.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(bm.ST_TYPE AS DOUBLE) <> 0 THEN FALSE
                    WHEN coalesce(try_cast(bm.limit_times AS DOUBLE), 0.0) > 0.0 THEN FALSE
                    WHEN try_cast(bm.pre_close AS DOUBLE) IS NULL OR try_cast(bm.open AS DOUBLE) IS NULL THEN FALSE
                    WHEN try_cast(bm.pre_close AS DOUBLE) <= 0 OR try_cast(bm.open AS DOUBLE) <= 0 THEN FALSE
                    WHEN try_cast(bm.open AS DOUBLE) >= try_cast(bm.pre_close AS DOUBLE) * (
                        1.0 + CASE WHEN substr(s.stock_code, 1, 3) IN ('300', '301', '688') THEN 0.20 ELSE 0.10 END
                    ) * 0.995 THEN FALSE
                    ELSE TRUE
                END AS buy_day_ok
            FROM scored s
            LEFT JOIN marketdb.STOCK_DAILY_DATA bm
                ON s.buy_date = bm.trade_date
               AND s.stock_code = bm.stock_code
        )
        SELECT
            signal_date,
            coalesce(buy_date, '') AS buy_date,
            CASE
                WHEN stock_code LIKE '%.SH' THEN 'SHSE.' || substr(stock_code, 1, 6)
                WHEN stock_code LIKE '%.SZ' THEN 'SZSE.' || substr(stock_code, 1, 6)
                WHEN stock_code LIKE '%.BJ' THEN 'BJSE.' || substr(stock_code, 1, 6)
                WHEN substr(stock_code, 1, 1) IN ('6', '9') THEN 'SHSE.' || substr(stock_code, 1, 6)
                WHEN substr(stock_code, 1, 1) IN ('8', '4') THEN 'BJSE.' || substr(stock_code, 1, 6)
                ELSE 'SZSE.' || substr(stock_code, 1, 6)
            END AS symbol,
            stock_code,
            name,
            1 AS rank,
            entry_score AS pred_prob,
            entry_score,
            pred_3d,
            pred_5d,
            pred_10d,
            rank_3d,
            rank_5d,
            rank_10d,
            amount,
            turnover_rate,
            total_mv,
            atr_qfq,
            '{float(exe['target_pct']):.5f}' AS target_pct,
            {int(exe['holding_days'])} AS holding_days,
            {int(exe['max_holding_days'])} AS max_holding_days,
            '{float(exe['exit_ratio']):.5f}' AS score_exit_entry_ratio,
            1 AS min_holding_days_before_score_exit,
            '{float(exe['continue_ratio']):.5f}' AS score_continue_entry_ratio,
            '{float(exe['stop_loss']):.5f}' AS signal_stop_loss_pct,
            '{float(exe['take_profit']):.5f}' AS signal_take_profit_pct,
            '{case_key}' AS strategy_variant,
            'amt{int(entry['amount_min'])}_mv{int(entry['total_mv_min'])}' AS filter_name,
            '{entry['name']}' AS entry_weight_name,
            '{exe['name']}' AS dynamic_hold_name,
            buy_day_ok AS buy_day_market_available,
            buy_day_ok AS buy_day_hard_gate_complete,
            false AS buy_day_st_rejected,
            false AS buy_day_open_limit_up_rejected,
            latest_market_date
        FROM with_buy_checks
        WHERE buy_day_ok
        QUALIFY row_number() OVER (PARTITION BY signal_date ORDER BY entry_score DESC, stock_code ASC) = 1
        ORDER BY signal_date
        """
        rows = con.execute(selected_sql).fetchdf().to_dict("records")
        _write_rows(signal_file, rows)
    finally:
        con.close()

    conn = duckdb.connect(str(SCORE_DB))
    try:
        conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{score_table}_date_code" ON "{score_table}" (trade_date, stock_code)')
    finally:
        conn.close()
    return {"case_key": case_key, "signal_file": signal_file, "score_table": score_table, "signal_rows": len(rows)}


def _run_backtest(case: dict[str, Any], exe: dict[str, Any], tag: str, start: str, end: str) -> dict[str, Any]:
    case_key = case["case_key"]
    log_file = REPORT_DIR / "logs" / f"{case_key}__{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_EQUITY_DD_SOFT_TRIGGER"] = str(exe["dd_soft"])
        env["GM_EQUITY_DD_HARD_TRIGGER"] = str(exe["dd_hard"])
        env["GM_EQUITY_DD_RECOVER_TRIGGER"] = "0.03"
        env["GM_EQUITY_DD_SOFT_SCALE"] = str(exe["dd_soft_scale"])
        env["GM_EQUITY_DD_HARD_SCALE"] = str(exe["dd_hard_scale"])
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(exe["exit_ratio"])
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = "1"
        env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(exe["continue_ratio"])
        command = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(CODE_SNAPSHOT_DIR),
            "--signal-file",
            str(case["signal_file"]),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            str(exe["holding_days"]),
            "--max-holding-days",
            str(exe["max_holding_days"]),
            "--target-position-pct",
            str(exe["target_pct"]),
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            str(case["score_table"]),
            "--market-db",
            str(Path(str(sources["10d"]["market_db_path"])).resolve()),
            "--backtest-start",
            start,
            "--backtest-end",
            end,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0",
            "--stop-loss-pct",
            str(exe["stop_loss"]),
            "--take-profit-pct",
            str(exe["take_profit"]),
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    return {
        "case_key": case_key,
        "slice": tag,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
    }


def _summary_row(entry: dict[str, Any], exe: dict[str, Any], case: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_slice = {row["slice"]: row for row in rows}
    return {
        "case_key": case["case_key"],
        "entry": entry["name"],
        "exe": exe["name"],
        "w10d": entry["w10d"],
        "w5d": entry["w5d"],
        "w3d": entry["w3d"],
        "amount_min": entry["amount_min"],
        "total_mv_min": entry["total_mv_min"],
        "holding_days": exe["holding_days"],
        "max_holding_days": exe["max_holding_days"],
        "continue_ratio": exe["continue_ratio"],
        "exit_ratio": exe["exit_ratio"],
        "target_pct": exe["target_pct"],
        "signal_rows": case["signal_rows"],
        "signal_file": str(case["signal_file"]),
        "full_annual": by_slice["full"]["annual"],
        "full_sharpe": by_slice["full"]["sharpe"],
        "full_max_drawdown": by_slice["full"]["max_drawdown"],
        "full_win_ratio": by_slice["full"]["win_ratio"],
        "full_open_count": by_slice["full"]["open_count"],
        "recent120_annual": by_slice["recent120"]["annual"],
        "recent120_sharpe": by_slice["recent120"]["sharpe"],
        "recent60_annual": by_slice["recent60"]["annual"],
        "recent60_sharpe": by_slice["recent60"]["sharpe"],
        "ytd2026_annual": by_slice["ytd2026"]["annual"],
    }


def _sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    def f(name: str, default: float = -9999.0) -> float:
        try:
            value = row.get(name)
            return float(value) if value not in (None, "") else default
        except Exception:
            return default

    return (
        f("recent120_annual"),
        f("recent60_annual"),
        f("ytd2026_annual"),
        f("full_sharpe"),
        -f("full_max_drawdown", default=9999.0),
        f("full_annual"),
    )


def main() -> None:
    manifest, _, sources = _load_strategy(STRATEGY_DIR)
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for entry in ENTRY_CASES:
        for exe in EXEC_CASES:
            case = _build_case_assets(manifest, sources, entry, exe)
            rows = [_run_backtest(case, exe, tag, start, end) for tag, start, end in TIME_SLICES]
            detail_rows.extend(rows)
            summary = _summary_row(entry, exe, case, rows)
            summary_rows.append(summary)
            _write_rows(REPORT_DIR / "no_intraday_detail.csv", detail_rows)
            _write_rows(REPORT_DIR / "no_intraday_summary.csv", summary_rows)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    ranked = sorted(summary_rows, key=_sort_key, reverse=True)
    (REPORT_DIR / "top_candidates.json").write_text(
        json.dumps(ranked[:10], ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
