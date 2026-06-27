from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

import research_formal_5d10d_l5_candidate_grid_20260621 as base


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
    / "formal_5d10d_selection_refine_20260621"
)
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-30 15:30:00"


ASSETS = [
    "rank_10d90_5d10",
    "rank_10d80_5d20",
    "rank_10d70_5d30",
    "rank_10d60_5d40",
    "rank_10d50_5d50",
]

FILTERS = [
    {"max_total_mv": 100000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3, "top_k": 5, "holding_days": 4, "max_positions": 8},
    {"max_total_mv": 150000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3, "top_k": 5, "holding_days": 4, "max_positions": 8},
    {"max_total_mv": 150000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "top_k": 5, "holding_days": 4, "max_positions": 8},
    {"max_total_mv": 200000.0, "min_amount": 10000.0, "min_turnover_rate": 0.3, "top_k": 5, "holding_days": 4, "max_positions": 8},
    {"max_total_mv": 200000.0, "min_amount": 20000.0, "min_turnover_rate": 0.5, "top_k": 5, "holding_days": 4, "max_positions": 8},
    {"max_total_mv": 250000.0, "min_amount": 30000.0, "min_turnover_rate": 0.8, "top_k": 5, "holding_days": 4, "max_positions": 8},
    {"max_total_mv": 300000.0, "min_amount": 50000.0, "min_turnover_rate": 1.0, "top_k": 8, "holding_days": 4, "max_positions": 10},
    {"max_total_mv": 150000.0, "min_amount": 10000.0, "min_turnover_rate": 0.5, "top_k": 3, "holding_days": 4, "max_positions": 6},
]


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace(" ", "")


def _slug(params: dict) -> str:
    return (
        f"{params['asset']}_{params['direction']}_tk{params['top_k']}_h{params['holding_days']}"
        f"_mp{params['max_positions']}_mv{_safe(params['max_total_mv'])}"
        f"_amt{_safe(params['min_amount'])}_turn{_safe(params['min_turnover_rate'])}"
        f"_exit{params['open_score_exit']}_ser{_safe(params['score_exit_entry_ratio'])}"
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


def _run_backtest(params: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(params["open_score_exit"]),
            "GM_MAX_DAILY_SELLS": str(params["max_daily_sells"]),
            "GM_SCORE_EXIT_ENTRY_RATIO": str(params["score_exit_entry_ratio"]),
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": "0",
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": str(params["score_continue_entry_ratio"]),
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
        str(params["max_positions"]),
        "--holding-days",
        str(params["holding_days"]),
        "--max-holding-days",
        str(params["holding_days"]),
        "--target-position-pct",
        str(float(params["target_total_pct"]) / max(1, int(params["max_positions"]))),
        "--score-db",
        str(params["db_path"]),
        "--score-table",
        str(params["table"]),
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


def _grid() -> list[dict]:
    manifest = json.loads((SOURCE_DIR / "fusion_5d10d_manifest.json").read_text(encoding="utf-8"))
    asset_map = {
        combo["combo_name"]: {
            "db_path": Path(manifest["output_db"]),
            "table": combo["table"],
            "manifest_path": SOURCE_DIR / "fusion_5d10d_manifest.json",
        }
        for combo in manifest["combos"]
    }
    runs = []
    for asset in ASSETS:
        for filters in FILTERS:
            runs.append(
                {
                    **asset_map[asset],
                    **filters,
                    "asset": asset,
                    "direction": "top",
                    "weight_mode": "equal",
                    "target_total_pct": 0.98,
                    "max_atr_ratio": None,
                    "open_score_exit": 1,
                    "score_exit_entry_ratio": 0.95,
                    "max_daily_sells": 1,
                    "score_continue_entry_ratio": 1.0,
                }
            )
    return runs


def main() -> int:
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    results = []
    grid = _grid()
    for index, params in enumerate(grid, start=1):
        slug = _slug(params)
        signal_file = REPORT_DIR / "signals" / f"{slug}.csv"
        log_file = REPORT_DIR / "logs" / f"{slug}.log"
        if not signal_file.exists():
            base._write_signal(params, base._load_rows(params), signal_file)
        returncode = _run_backtest(params, signal_file, log_file)
        indicator = _extract_indicator(log_file) or {}
        result = {
            **{key: value for key, value in params.items() if key not in {"db_path", "manifest_path"}},
            "db_path": str(params["db_path"]),
            "manifest_path": str(params["manifest_path"]),
            "returncode": returncode,
            "signal_file": str(signal_file),
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
