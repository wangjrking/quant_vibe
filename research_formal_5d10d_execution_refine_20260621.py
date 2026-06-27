from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
BASE_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_l5_candidate_grid_20260621"
)
REPORT_DIR = (
    DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_execution_refine_20260621"
)
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


BASE_SIGNALS = [
    {
        "base_name": "rank_10d80_5d20",
        "signal_file": BASE_DIR
        / "signals"
        / "rank_10d80_5d20_top_tk5_h5_mp8_mv200000p0_amt20000p0_turn0p5_atrnone.csv",
        "score_db": BASE_DIR / "fusion_5d10d.db",
        "score_table": "rank_10d80_5d20",
    },
    {
        "base_name": "rank_10d50_5d50",
        "signal_file": BASE_DIR
        / "signals"
        / "rank_10d50_5d50_top_tk5_h5_mp8_mv200000p0_amt20000p0_turn0p5_atrnone.csv",
        "score_db": BASE_DIR / "fusion_5d10d.db",
        "score_table": "rank_10d50_5d50",
    },
    {
        "base_name": "rank_10d60_5d40",
        "signal_file": BASE_DIR
        / "signals"
        / "rank_10d60_5d40_top_tk5_h5_mp10_mv200000p0_amt20000p0_turn0p5_atrnone.csv",
        "score_db": BASE_DIR / "fusion_5d10d.db",
        "score_table": "rank_10d60_5d40",
    },
]


EXEC_VARIANTS = [
    {"max_positions": 6, "holding_days": 5, "max_daily_sells": 1, "open_score_exit": 0, "continue_ratio": 1.0},
    {"max_positions": 7, "holding_days": 5, "max_daily_sells": 1, "open_score_exit": 0, "continue_ratio": 1.0},
    {"max_positions": 8, "holding_days": 4, "max_daily_sells": 1, "open_score_exit": 0, "continue_ratio": 1.0},
    {"max_positions": 8, "holding_days": 5, "max_daily_sells": 2, "open_score_exit": 0, "continue_ratio": 1.0},
    {"max_positions": 8, "holding_days": 5, "max_daily_sells": 0, "open_score_exit": 0, "continue_ratio": 1.0},
    {"max_positions": 8, "holding_days": 5, "max_daily_sells": 1, "open_score_exit": 1, "continue_ratio": 1.0, "score_exit_ratio": 0.95},
    {"max_positions": 8, "holding_days": 5, "max_daily_sells": 1, "open_score_exit": 0, "continue_ratio": 0.98, "max_holding_days": 7},
    {"max_positions": 10, "holding_days": 4, "max_daily_sells": 1, "open_score_exit": 0, "continue_ratio": 1.0},
]


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


def _slug(base: dict, variant: dict) -> str:
    return (
        f"{base['base_name']}_mp{variant['max_positions']}"
        f"_h{variant['holding_days']}_sells{variant['max_daily_sells']}"
        f"_exit{variant['open_score_exit']}_cont{_safe(variant['continue_ratio'])}"
        f"_mh{variant.get('max_holding_days', variant['holding_days'])}"
    )


def _write_scaled_signal(source: Path, dest: Path, variant: dict) -> None:
    target_pct = 0.98 / float(int(variant["max_positions"]))
    with source.open("r", encoding="utf-8-sig", newline="") as src:
        rows = list(csv.DictReader(src))
        fieldnames = list(rows[0].keys())
    for row in rows:
        row["target_pct"] = f"{target_pct:.10f}"
        row["holding_days"] = str(int(variant["holding_days"]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _run_backtest(base: dict, variant: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(variant["open_score_exit"]),
            "GM_MAX_DAILY_SELLS": str(variant["max_daily_sells"]),
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": str(variant["continue_ratio"]),
        }
    )
    if "score_exit_ratio" in variant:
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(variant["score_exit_ratio"])
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
        str(variant.get("max_holding_days", variant["holding_days"])),
        "--target-position-pct",
        str(0.98 / float(int(variant["max_positions"]))),
        "--score-db",
        str(base["score_db"]),
        "--score-table",
        str(base["score_table"]),
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
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    total = len(BASE_SIGNALS) * len(EXEC_VARIANTS)
    index = 0
    for base in BASE_SIGNALS:
        for variant in EXEC_VARIANTS:
            index += 1
            slug = _slug(base, variant)
            signal_file = REPORT_DIR / "signals" / f"{slug}.csv"
            log_file = REPORT_DIR / "logs" / f"{slug}.log"
            if not signal_file.exists():
                _write_scaled_signal(Path(base["signal_file"]), signal_file, variant)
            returncode = _run_backtest(base, variant, signal_file, log_file)
            indicator = _extract_indicator(log_file) or {}
            result = {
                **base,
                **variant,
                "score_db": str(base["score_db"]),
                "source_signal_file": str(base["signal_file"]),
                "scaled_signal_file": str(signal_file),
                "log_file": str(log_file),
                "returncode": returncode,
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
                f"[{index}/{total}] {slug} annual={result.get('annual')} sharpe={result.get('sharpe')} avg={result.get('avg_invested_pct')}",
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
