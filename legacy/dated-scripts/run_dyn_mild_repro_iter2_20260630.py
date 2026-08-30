from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
PROD_DIR = MAIN / "strategy_library" / "production" / "prod_dyn_mild_09_12_15_v20260630"
STRATEGY_DIR = PROD_DIR / "code_snapshot"
BASE_SIGNAL = PROD_DIR / "signals" / "full_history_dyn_mild_09_12_15.csv"
REPORT_DIR = DATA / "reports" / "strategy_agent_dyn_mild_repro_iter2_20260630"
SIGNAL_DIR = REPORT_DIR / "signals"
LOG_DIR = REPORT_DIR / "juejin_logs"
WRAPPER_DIR = REPORT_DIR / "wrapper_stdout"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630" / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"
OUT_CSV = REPORT_DIR / "dyn_mild_repro_iter2_summary.csv"
OUT_JSON = REPORT_DIR / "dyn_mild_repro_iter2_summary.json"
OUT_MANIFEST = REPORT_DIR / "dyn_mild_repro_iter2_manifest.json"
OUT_REPORT = REPORT_DIR / "dyn_mild_repro_iter2_report.md"


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "1",
    "GM_FORCE_SELL_MARKET_ORDER": "0",
    "GM_FORCE_BUY_MARKET_ORDER": "0",
    "GM_INTRADAY_RISK_MODE": "0",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
    "GM_RESIZE_HELD_ON_SIGNAL": "0",
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
    "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
}


CASES: list[dict[str, Any]] = [
    {
        "name": "baseline_repro_cap15",
        "target_scale": 1.00,
        "target_cap": 0.15,
        "holding_days": 2,
        "max_holding_days": 3,
        "exit_ratio": 0.97,
        "continue_ratio": 0.98,
        "day_drop_ratio": None,
        "max_daily_sells": 0,
        "dd_mode": "prod_loose",
        "sell_slip_mult": 0.0,
    },
    {
        "name": "base_plus_daydrop995",
        "target_scale": 1.00,
        "target_cap": 0.15,
        "holding_days": 2,
        "max_holding_days": 3,
        "exit_ratio": 0.98,
        "continue_ratio": 0.99,
        "day_drop_ratio": 0.995,
        "max_daily_sells": 1,
        "dd_mode": "prod_loose",
        "sell_slip_mult": 0.0,
    },
    {
        "name": "raise_pos_110_cap165",
        "target_scale": 1.10,
        "target_cap": 0.165,
        "holding_days": 2,
        "max_holding_days": 3,
        "exit_ratio": 0.98,
        "continue_ratio": 0.99,
        "day_drop_ratio": 0.995,
        "max_daily_sells": 1,
        "dd_mode": "prod_loose",
        "sell_slip_mult": 0.0,
    },
    {
        "name": "raise_pos_120_cap18",
        "target_scale": 1.20,
        "target_cap": 0.18,
        "holding_days": 2,
        "max_holding_days": 3,
        "exit_ratio": 0.98,
        "continue_ratio": 0.99,
        "day_drop_ratio": 0.995,
        "max_daily_sells": 1,
        "dd_mode": "prod_loose",
        "sell_slip_mult": 0.0,
    },
    {
        "name": "raise_pos_120_cap18_riskcut",
        "target_scale": 1.20,
        "target_cap": 0.18,
        "riskcut": True,
        "holding_days": 2,
        "max_holding_days": 3,
        "exit_ratio": 0.98,
        "continue_ratio": 0.99,
        "day_drop_ratio": 0.995,
        "max_daily_sells": 1,
        "dd_mode": "prod_loose",
        "sell_slip_mult": 0.0,
    },
    {
        "name": "raise_pos_120_cap18_ddmid",
        "target_scale": 1.20,
        "target_cap": 0.18,
        "holding_days": 2,
        "max_holding_days": 3,
        "exit_ratio": 0.98,
        "continue_ratio": 0.99,
        "day_drop_ratio": 0.995,
        "max_daily_sells": 1,
        "dd_mode": "mid",
        "sell_slip_mult": 0.0,
    },
    {
        "name": "faster_exit_cap15",
        "target_scale": 1.00,
        "target_cap": 0.15,
        "holding_days": 1,
        "max_holding_days": 2,
        "exit_ratio": 0.985,
        "continue_ratio": 0.995,
        "day_drop_ratio": 0.997,
        "max_daily_sells": 2,
        "dd_mode": "prod_loose",
        "sell_slip_mult": 0.0,
    },
    {
        "name": "conservative_sellslip025",
        "target_scale": 1.00,
        "target_cap": 0.15,
        "holding_days": 2,
        "max_holding_days": 3,
        "exit_ratio": 0.98,
        "continue_ratio": 0.99,
        "day_drop_ratio": 0.995,
        "max_daily_sells": 1,
        "dd_mode": "prod_loose",
        "sell_slip_mult": 0.25,
    },
]


DD_ENV = {
    "off": {
        "GM_EQUITY_DD_RISK_MODE": "0",
        "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
        "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
        "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
        "GM_EQUITY_DD_SOFT_SCALE": "0.80",
        "GM_EQUITY_DD_HARD_SCALE": "0.60",
    },
    "prod_loose": {
        "GM_EQUITY_DD_RISK_MODE": "1",
        "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
        "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
        "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
        "GM_EQUITY_DD_SOFT_SCALE": "0.80",
        "GM_EQUITY_DD_HARD_SCALE": "0.60",
    },
    "mid": {
        "GM_EQUITY_DD_RISK_MODE": "1",
        "GM_EQUITY_DD_SOFT_TRIGGER": "0.07",
        "GM_EQUITY_DD_HARD_TRIGGER": "0.12",
        "GM_EQUITY_DD_RECOVER_TRIGGER": "0.035",
        "GM_EQUITY_DD_SOFT_SCALE": "0.72",
        "GM_EQUITY_DD_HARD_SCALE": "0.50",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out


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


def _target_pct(row: dict[str, Any], case: dict[str, Any]) -> float:
    base = _as_float(row.get("target_pct")) or 0.0
    target = min(base * float(case["target_scale"]), float(case["target_cap"]))
    if case.get("riskcut"):
        pct_chg = _as_float(row.get("pct_chg"))
        prev_pct_chg = _as_float(row.get("prev_pct_chg"))
        two_day_ret = _as_float(row.get("two_day_ret"))
        turnover = _as_float(row.get("turnover_rate"))
        if pct_chg is not None and pct_chg >= 10.0:
            target *= 0.55
        if two_day_ret is not None and two_day_ret >= 0.08 and turnover is not None and turnover >= 12.0:
            target *= 0.70
        if prev_pct_chg is not None and prev_pct_chg <= -8.0 and pct_chg is not None and pct_chg >= 6.0:
            target *= 0.75
    return max(0.0, target)


def _make_signal(case: dict[str, Any], base_rows: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        item = dict(row)
        item["target_pct"] = f"{_target_pct(row, case):.5f}"
        item["holding_days"] = str(case["holding_days"])
        item["max_holding_days"] = str(case["max_holding_days"])
        item["score_exit_entry_ratio"] = f"{float(case['exit_ratio']):.5f}"
        item["score_continue_entry_ratio"] = f"{float(case['continue_ratio']):.5f}"
        item["min_holding_days_before_score_exit"] = "1"
        item["strategy_variant"] = str(case["name"])
        item["dynamic_hold_name"] = str(case["name"])
        rows.append(item)
    rows.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    signal_file = SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(signal_file, rows)
    counts = Counter(row["signal_date"] for row in rows)
    return signal_file, {
        "signal_rows": len(rows),
        "signal_days": len(counts),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "days_below_5": sum(1 for value in counts.values() if value < 5),
        "signal_sha256": _sha256(signal_file),
    }


def _command(case: dict[str, Any], signal_file: Path, log_file: Path) -> list[str]:
    return [
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
        str(case["holding_days"]),
        "--max-holding-days",
        str(case["max_holding_days"]),
        "--target-position-pct",
        str(case["target_cap"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        str(case["exit_ratio"]),
        "--score-continue-entry-ratio",
        str(case["continue_ratio"]),
        "--min-holding-days-before-score-exit",
        "1",
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
        "--stop-loss-pct",
        "0.05",
        "--take-profit-pct",
        "0.08",
    ]


def _env_for(case: dict[str, Any]) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update(DD_ENV[str(case["dd_mode"])])
    env["GM_MAX_DAILY_SELLS"] = str(case["max_daily_sells"])
    env["GM_LIGHT_STOP_LOSS_PCT"] = "none"
    env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(case["exit_ratio"])
    env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(case["continue_ratio"])
    if case.get("day_drop_ratio") is None:
        env["GM_SCORE_STOP_LOSS_DAY_DROP_RATIO"] = "none"
    else:
        env["GM_SCORE_STOP_LOSS_DAY_DROP_RATIO"] = str(case["day_drop_ratio"])
    env["GM_ADAPTIVE_SELL_SLIPPAGE_MULT"] = str(case.get("sell_slip_mult", 0.0))
    return env


def _run(case: dict[str, Any], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    signal_file, signal_meta = _make_signal(case, base_rows)
    log_file = LOG_DIR / f"{case['name']}.log"
    wrapper_file = WRAPPER_DIR / f"{case['name']}.json"
    cmd = _command(case, signal_file, log_file)
    env_delta = _env_for(case)
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(env_delta)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        wrapper_file.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wrapper_file.write_text(proc.stdout or "", encoding="utf-8")
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    row = {
        "name": case["name"],
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "target_scale": case["target_scale"],
        "target_cap": case["target_cap"],
        "holding_days": case["holding_days"],
        "max_holding_days": case["max_holding_days"],
        "exit_ratio": case["exit_ratio"],
        "continue_ratio": case["continue_ratio"],
        "day_drop_ratio": case["day_drop_ratio"],
        "max_daily_sells": case["max_daily_sells"],
        "dd_mode": case["dd_mode"],
        "riskcut": bool(case.get("riskcut", False)),
        "sell_slip_mult": case.get("sell_slip_mult", 0.0),
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "wrapper_stdout_file": str(wrapper_file),
        "command": " ".join(cmd),
        "env_delta_json": json.dumps(env_delta, ensure_ascii=False, sort_keys=True),
        **signal_meta,
    }
    return row


def _write_report(rows: list[dict[str, Any]]) -> None:
    sorted_rows = sorted(
        [row for row in rows if row.get("returncode") == 0 and row.get("annual") is not None],
        key=lambda row: (float(row["annual"]), float(row["sharpe"] or 0.0)),
        reverse=True,
    )
    lines = [
        "# dyn_mild 鍙鐜拌凯浠ｄ簩鎶ュ憡",
        "",
        "## 缁撹鍙ｅ緞",
        "",
        "- 鏈疆涓?research-only锛屼笉淇敼鐢熶骇鍙傛暟锛屼笉鐢熸垚浜ゆ槗浜や粯銆?",
        "- 杈撳叆鍥哄畾涓哄綋鍓嶇敓浜у綊妗ｅ叏鍘嗗彶淇″彿锛屼粎娲剧敓浠撲綅鍜屽崠鍑哄弬鏁板€欓€夈€?",
        "- 鎵€鏈夊€欓€夊潎淇濆瓨淇″彿 hash銆佷唬鐮?hash銆乻core_db hash銆佽繍琛屽懡浠ゃ€乪nv delta 鍜屾帢閲戞棩蹇椼€?",
        "",
        "## 鍊欓€夌粨鏋?",
        "",
        "| 鍚嶇О | 骞村寲 | Sharpe | 鏈€澶у洖鎾?| 鑳滅巼 | 寮€浠?| 鏃ュ潎淇″彿 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_rows:
        signal_days = float(row.get("signal_days") or 0)
        signal_rows = float(row.get("signal_rows") or 0)
        avg_names = signal_rows / signal_days if signal_days else 0.0
        lines.append(
            "| {name} | {annual:.2%} | {sharpe:.3f} | {mdd:.2%} | {win:.2%} | {open_count} | {avg_names:.2f} |".format(
                name=row["name"],
                annual=float(row["annual"]),
                sharpe=float(row["sharpe"] or 0.0),
                mdd=float(row["max_drawdown"] or 0.0),
                win=float(row["win_ratio"] or 0.0),
                open_count=row.get("open_count", ""),
                avg_names=avg_names,
            )
        )
    if sorted_rows:
        best = sorted_rows[0]
        lines.extend(
            [
                "",
                "## 褰撳墠鏈€浣?",
                "",
                f"- best candidate: `{best['name']}`",
                f"- annual: {float(best['annual']):.2%}",
                f"- Sharpe: {float(best['sharpe'] or 0.0):.3f}",
                f"- max drawdown: {float(best['max_drawdown'] or 0.0):.2%}",
                f"- signal file: `{best['signal_file']}`",
                f"- log file: `{best['log_file']}`",
            ]
        )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [JUEJIN_PYTHON, STRATEGY_DIR / "main.py", BASE_SIGNAL, MARKET_DB, SCORE_DB]:
        if not path.exists():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base_rows = list(csv.DictReader(BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="")))
    manifest = {
        "schema_version": 1,
        "generated_at": datetime_module.datetime.now().isoformat(timespec="seconds"),
        "status": "research_only_no_production_change",
        "base_signal": str(BASE_SIGNAL),
        "base_signal_sha256": _sha256(BASE_SIGNAL),
        "strategy_dir": str(STRATEGY_DIR),
        "strategy_main_sha256": _sha256(STRATEGY_DIR / "main.py"),
        "score_db": str(SCORE_DB),
        "score_db_sha256": _sha256(SCORE_DB),
        "score_table": SCORE_TABLE,
        "market_db": str(MARKET_DB),
        "juejin_python": str(JUEJIN_PYTHON),
        "cases": CASES,
        "base_env": BASE_ENV,
        "dd_env": DD_ENV,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case, base_rows)
        rows.append(row)
        _write_rows(OUT_CSV, rows)
        OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_report(rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

