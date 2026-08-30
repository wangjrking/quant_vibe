from __future__ import annotations

import ast
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_correct_exec_search_20260630"
RANK_CACHE = REPORT_DIR / "rank_cache.duckdb"
STRATEGY_DIR = REPORT_DIR / "code_snapshot_correct_exec"
EXEC_MARKET_DB = REPORT_DIR / "exec_market_raw_open.db"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
SCORE_DB = REPORT_DIR / "scores" / "correct_exec_verify_scores.duckdb"
SIGNAL_DIR = REPORT_DIR / "verify_signals"
LOG_DIR = REPORT_DIR / "verify_logs"
REPRO_DIR = REPORT_DIR / "repro_packages"


CANDIDATES = [
    {
        "name": "ce_10only_top10_h3_pos10",
        "score": "r10",
        "score_table": "score_ce_10only",
        "filter": "amount>=100000 AND total_mv>=200000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 10,
        "target": 0.10,
        "holding_days": 3,
    },
    {
        "name": "ce_10only_top10_h5_pos10",
        "score": "r10",
        "score_table": "score_ce_10only",
        "filter": "amount>=100000 AND total_mv>=200000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 10,
        "target": 0.10,
        "holding_days": 5,
    },
    {
        "name": "ce_10only_top10_h5_amt20_mv50_pos10",
        "score": "r10",
        "score_table": "score_ce_10only",
        "filter": "amount>=200000 AND total_mv>=500000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 10,
        "target": 0.10,
        "holding_days": 5,
    },
    {
        "name": "ce_10only_top10_h5_amt50_mv100_pos10",
        "score": "r10",
        "score_table": "score_ce_10only",
        "filter": "amount>=500000 AND total_mv>=1000000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 10,
        "target": 0.10,
        "holding_days": 5,
    },
    {
        "name": "ce_10only_top10_h5_amt100_mv200_pos10",
        "score": "r10",
        "score_table": "score_ce_10only",
        "filter": "amount>=1000000 AND total_mv>=2000000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 10,
        "target": 0.10,
        "holding_days": 5,
    },
    {
        "name": "ce_10only_top10_h5_amt200_mv300_pos10",
        "score": "r10",
        "score_table": "score_ce_10only",
        "filter": "amount>=2000000 AND total_mv>=3000000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 10,
        "target": 0.10,
        "holding_days": 5,
    },
    {
        "name": "ce_10only_top8_h5_amt20_mv50_pos12",
        "score": "r10",
        "score_table": "score_ce_10only",
        "filter": "amount>=200000 AND total_mv>=500000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 8,
        "target": 0.12,
        "holding_days": 5,
    },
    {
        "name": "ce_1d20_10d80_top10_h3_pos10",
        "score": "0.2*r1+0.8*r10",
        "score_table": "score_ce_1d20_10d80",
        "filter": "amount>=100000 AND total_mv>=200000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 10,
        "target": 0.10,
        "holding_days": 3,
    },
    {
        "name": "ce_1d20_10d50_5d30_top10_h3_pos10",
        "score": "0.2*r1+0.5*r10+0.3*r5",
        "score_table": "score_ce_1d20_10d50_5d30",
        "filter": "amount>=100000 AND total_mv>=200000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 10,
        "target": 0.10,
        "holding_days": 3,
    },
    {
        "name": "ce_3only_top10_h1_pos10",
        "score": "r3",
        "score_table": "score_ce_3only",
        "filter": "amount>=100000 AND total_mv>=200000 AND close<=150 AND coalesce(two_day_ret,-999)<0.20",
        "topn": 10,
        "target": 0.10,
        "holding_days": 1,
    },
]


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "0",
    "GM_MAX_DAILY_SELLS": "0",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_STOP_LOSS_PCT": "none",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "1",
    "GM_FORCE_SELL_MARKET_ORDER": "0",
    "GM_FORCE_BUY_MARKET_ORDER": "0",
    "GM_INTRADAY_RISK_MODE": "0",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_EQUITY_DD_RISK_MODE": "0",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "0",
    "GM_INDEX_RISK_EXIT_MODE": "0",
    "GM_BREADTH_RISK_EXIT_MODE": "0",
    "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
    "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
    "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
    "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
    "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
    "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
    "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
    "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.25",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def extract_indicator(log_file: Path) -> dict[str, Any] | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": dt})
    return None


def ensure_score_table(candidate: dict[str, Any]) -> None:
    SCORE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{RANK_CACHE.as_posix()}' AS rc (READ_ONLY)")
        con.execute(f"ATTACH '{SCORE_DB.as_posix()}' AS scoredb")
        table = candidate["score_table"]
        con.execute(f'DROP TABLE IF EXISTS scoredb."{table}"')
        con.execute(
            f"""
            CREATE TABLE scoredb."{table}" AS
            SELECT trade_date, stock_code, ({candidate['score']}) AS pred_prob
            FROM rc.rank_cache
            """
        )
    finally:
        con.close()
    with duckdb.connect(str(SCORE_DB)) as score_conn:
        score_conn.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{candidate["score_table"]}" ON "{candidate["score_table"]}" (trade_date, stock_code)'
        )


def make_signal(candidate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    output = SIGNAL_DIR / f"{candidate['name']}.csv"
    if output.exists():
        rows = list(csv.DictReader(output.open("r", encoding="utf-8-sig", newline="")))
    else:
        con = duckdb.connect(str(RANK_CACHE), read_only=True)
        try:
            rows = con.execute(
                f"""
                WITH selected AS (
                    SELECT
                        *,
                        ({candidate['score']}) AS entry_score,
                        row_number() OVER (PARTITION BY trade_date ORDER BY ({candidate['score']}) DESC, stock_code) AS rn
                    FROM rank_cache
                    WHERE buy_ok = 1
                      AND buy_open IS NOT NULL
                      AND {candidate['filter']}
                )
                SELECT
                    trade_date AS signal_date,
                    buy_date,
                    CASE
                        WHEN stock_code LIKE '%.SH' THEN 'SHSE.' || substr(stock_code, 1, 6)
                        WHEN stock_code LIKE '%.SZ' THEN 'SZSE.' || substr(stock_code, 1, 6)
                        WHEN substr(stock_code, 1, 1) IN ('6', '9') THEN 'SHSE.' || substr(stock_code, 1, 6)
                        ELSE 'SZSE.' || substr(stock_code, 1, 6)
                    END AS symbol,
                    stock_code,
                    name,
                    rn AS rank,
                    entry_score AS pred_prob,
                    entry_score,
                    pred_1d,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    r1 AS rank_1d,
                    r3 AS rank_3d,
                    r5 AS rank_5d,
                    r10 AS rank_10d,
                    amount,
                    turnover_rate,
                    total_mv,
                    pct_chg,
                    prev_pct_chg,
                    two_day_ret,
                    buy_open AS buy_raw_open,
                    buy_pre_close AS buy_raw_pre_close,
                    printf('%.5f', {candidate['target']}) AS target_pct,
                    {candidate['holding_days']} AS holding_days,
                    {candidate['holding_days']} AS max_holding_days,
                    '0.00000' AS score_exit_entry_ratio,
                    99 AS min_holding_days_before_score_exit,
                    '999.00000' AS score_continue_entry_ratio,
                    'none' AS signal_stop_loss_pct,
                    'none' AS signal_take_profit_pct,
                    '{candidate['name']}' AS strategy_variant,
                    '{candidate['filter']}' AS filter_name,
                    '{candidate['score']}' AS entry_weight_name,
                    'fixed_h{candidate['holding_days']}' AS dynamic_hold_name,
                    TRUE AS buy_day_market_available,
                    TRUE AS buy_day_hard_gate_complete,
                    FALSE AS buy_day_st_rejected,
                    FALSE AS buy_day_open_limit_up_rejected,
                    '20260629' AS latest_market_date
                FROM selected
                WHERE rn <= {candidate['topn']}
                ORDER BY signal_date, rn
                """
            ).fetchdf().to_dict("records")
        finally:
            con.close()
        write_rows(output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["signal_date"])] = counts.get(str(row["signal_date"]), 0) + 1
    return output, {
        "signal_rows": len(rows),
        "signal_days": len(counts),
        "days_below_target": sum(1 for value in counts.values() if value < int(candidate["topn"])),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "signal_sha256": sha256(output),
    }


def run_candidate(candidate: dict[str, Any], signal_file: Path, suffix: str = "main") -> dict[str, Any]:
    log_file = LOG_DIR / f"{candidate['name']}__{suffix}.log"
    indicator = extract_indicator(log_file)
    returncode = 0
    cmd = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(candidate["topn"]),
        "--holding-days",
        str(candidate["holding_days"]),
        "--max-holding-days",
        str(candidate["holding_days"]),
        "--target-position-pct",
        str(candidate["target"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        str(candidate["score_table"]),
        "--market-db",
        str(EXEC_MARKET_DB),
        "--score-exit-entry-ratio",
        "0",
        "--score-continue-entry-ratio",
        "999",
        "--min-holding-days-before-score-exit",
        "99",
        "--backtest-start",
        "2022-06-07 09:00:00",
        "--backtest-end",
        "2026-06-29 15:30:00",
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0",
    ]
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (LOG_DIR / f"{candidate['name']}__{suffix}.wrapper_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        returncode = proc.returncode
        indicator = extract_indicator(log_file)
    return {
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
        "command": " ".join(cmd),
        "env_delta_json": json.dumps(BASE_ENV, ensure_ascii=False, sort_keys=True),
    }


def sell_skip_count(log_file: str) -> int:
    path = Path(log_file)
    if not path.exists():
        return -1
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if "SELL_SKIP" in line)


def same_metric(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-10
    except Exception:
        return left == right


def package_repro(candidate: dict[str, Any], signal_file: Path, result: dict[str, Any], repeat_result: dict[str, Any]) -> dict[str, Any]:
    pkg = REPRO_DIR / candidate["name"]
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STRATEGY_DIR / "main.py", pkg / "main.py")
    shutil.copy2(MAIN / "run_juejin_signal_backtest.py", pkg / "run_juejin_signal_backtest.py")
    shutil.copy2(MAIN / "run_correct_exec_verify_candidates_20260630.py", pkg / "run_correct_exec_verify_candidates_20260630.py")
    shutil.copy2(signal_file, pkg / signal_file.name)
    shutil.copy2(Path(result["log_file"]), pkg / Path(result["log_file"]).name)
    shutil.copy2(Path(repeat_result["log_file"]), pkg / Path(repeat_result["log_file"]).name)
    shutil.copy2(SCORE_DB, pkg / SCORE_DB.name)
    shutil.copy2(EXEC_MARKET_DB, pkg / "exec_market_raw_open.db")
    manifest = {
        "candidate": candidate,
        "result": result,
        "repeat_result": repeat_result,
        "base_env": BASE_ENV,
        "files": {},
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "research_repro_package_not_production",
    }
    for path in pkg.iterdir():
        if path.is_file():
            manifest["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (pkg / "repro_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_sha = sha256(pkg / "repro_manifest.json")
    return {"repro_package_path": str(pkg), "repro_manifest": str(pkg / "repro_manifest.json"), "repro_manifest_sha256": manifest_sha}


def main() -> None:
    for path in [JUEJIN_PYTHON, RANK_CACHE, STRATEGY_DIR / "main.py", EXEC_MARKET_DB]:
        if not path.exists():
            raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        ensure_score_table(candidate)
        signal_file, signal_meta = make_signal(candidate)
        main_result = run_candidate(candidate, signal_file, "main")
        repeat_result = run_candidate(candidate, signal_file, "repeat")
        repeat_match = all(
            same_metric(main_result.get(key), repeat_result.get(key))
            for key in ["annual", "pnl_ratio", "sharpe", "max_drawdown", "win_ratio", "open_count", "close_count"]
        )
        repro = package_repro(candidate, signal_file, main_result, repeat_result)
        row = {
            "name": candidate["name"],
            **candidate,
            **signal_meta,
            "annual": main_result.get("annual"),
            "pnl_ratio": main_result.get("pnl_ratio"),
            "sharpe": main_result.get("sharpe"),
            "max_drawdown": main_result.get("max_drawdown"),
            "win_ratio": main_result.get("win_ratio"),
            "open_count": main_result.get("open_count"),
            "close_count": main_result.get("close_count"),
            "sell_skip_count": sell_skip_count(str(main_result.get("log_file"))),
            "main_log_file": main_result.get("log_file"),
            "repeat_log_file": repeat_result.get("log_file"),
            "repeat_match": repeat_match,
            "command": main_result.get("command"),
            "env_delta_json": main_result.get("env_delta_json"),
            **repro,
        }
        row["admission_pass"] = bool(
            row["annual"] is not None
            and float(row["annual"]) >= 1.0
            and float(row["sharpe"] or 0) >= 1.0
            and float(row["max_drawdown"] or 9) <= 0.40
            and int(row["open_count"] or 0) >= 100
            and int(row["sell_skip_count"] or 999) < 100
            and row["repeat_match"]
            and row["repro_manifest_sha256"]
        )
        rows.append(row)
        write_rows(REPORT_DIR / "correct_exec_verify_candidates_summary.csv", rows)
        (REPORT_DIR / "correct_exec_verify_candidates_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
