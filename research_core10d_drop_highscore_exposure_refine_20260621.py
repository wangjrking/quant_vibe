from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, PRED_DB, STRATEGY_DIR


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_model_application_20260621"
SOURCE_DIR = REPORT_ROOT / "core10d_original_signal_score_cut_20260621"
REPORT_DIR = REPORT_ROOT / "core10d_drop_highscore_exposure_refine_20260621"
SIGNAL_FILE = SOURCE_DIR / "signals" / "drop_gt0p5_all.csv"
TABLE_10D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_10d_open_return_score_20240604_20260618"


CONFIGS: list[dict] = []
for target_pct in (0.196, 0.21, 0.22, 0.24, 0.26):
    CONFIGS.append(
        {
            "name": f"dropgt0p5_t{str(target_pct).replace('.', 'p')}_h5_mh5_sync0",
            "target_pct": target_pct,
            "holding_days": 5,
            "max_holding_days": 5,
            "sync_positions": 0,
            "max_positions": 5,
        }
    )
for holding_days in (6, 7):
    for target_pct in (0.196, 0.21, 0.22):
        CONFIGS.append(
            {
                "name": f"dropgt0p5_t{str(target_pct).replace('.', 'p')}_h{holding_days}_mh{holding_days}_sync0",
                "target_pct": target_pct,
                "holding_days": holding_days,
                "max_holding_days": holding_days,
                "sync_positions": 0,
                "max_positions": 5,
            }
        )
for holding_days, max_holding_days in ((5, 6), (5, 7), (6, 8)):
    for target_pct in (0.196, 0.21):
        CONFIGS.append(
            {
                "name": (
                    f"dropgt0p5_t{str(target_pct).replace('.', 'p')}"
                    f"_h{holding_days}_mh{max_holding_days}_sync0"
                ),
                "target_pct": target_pct,
                "holding_days": holding_days,
                "max_holding_days": max_holding_days,
                "sync_positions": 0,
                "max_positions": 5,
            }
        )
for target_pct in (0.196, 0.21, 0.22):
    CONFIGS.append(
        {
            "name": f"dropgt0p5_t{str(target_pct).replace('.', 'p')}_h5_mh5_sync1",
            "target_pct": target_pct,
            "holding_days": 5,
            "max_holding_days": 5,
            "sync_positions": 1,
            "max_positions": 5,
        }
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
    active_positions = []
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
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


def run_backtest(config: dict) -> tuple[int, Path]:
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
            "GM_SYNC_POSITIONS": str(config["sync_positions"]),
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
        str(SIGNAL_FILE),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(config["max_positions"]),
        "--holding-days",
        str(config["holding_days"]),
        "--max-holding-days",
        str(config["max_holding_days"]),
        "--target-position-pct",
        str(config["target_pct"]),
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
        "0.0001",
    ]
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode, log_file


def _count_signals() -> int:
    with SIGNAL_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def main() -> None:
    if not SIGNAL_FILE.exists():
        raise SystemExit(f"Signal file not found: {SIGNAL_FILE}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    signal_count = _count_signals()
    rows = []
    for config in CONFIGS:
        code, log_file = run_backtest(config)
        indicator = _extract_indicator(log_file) or {}
        exposure = _exposure_stats(log_file)
        row = {
            **config,
            "returncode": code,
            "signal_file": str(SIGNAL_FILE),
            "signal_count": signal_count,
            "log_file": str(log_file),
            "pnl_ratio": indicator.get("pnl_ratio"),
            "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
            "sharp_ratio": indicator.get("sharp_ratio"),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            **exposure,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    summary_file = REPORT_DIR / "summary.csv"
    with summary_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    qualified = [
        row
        for row in rows
        if (row.get("pnl_ratio_annual") or 0) >= 2.0
        and (row.get("sharp_ratio") or 0) >= 3.0
        and (row.get("avg_invested_pct") or 0) >= 0.8
    ]
    with (REPORT_DIR / "qualified.json").open("w", encoding="utf-8") as file:
        json.dump(qualified, file, ensure_ascii=False, indent=2)
    print(f"summary={summary_file}")
    print(f"qualified_count={len(qualified)}")


if __name__ == "__main__":
    main()
