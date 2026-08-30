from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import subprocess
from pathlib import Path

from research_list_age_current_best_probe_20260620 import JUEJIN_PYTHON, MAIN, MARKET_DB, STRATEGY_DIR


ROOT = Path(__file__).resolve().parents[2]
BASE_REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_model_application_20260620"
REPORT_DIR = BASE_REPORT_DIR / "fusion_frontier_runtime_refine_20260620"
FUSION_DB = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260618"
    / "published_asset_fusions"
    / "fusion_combos.db"
)
FUSION_TABLE = "combo_rank_min_consensus"
SIGNAL_DIR = BASE_REPORT_DIR / "sc_rz1_weight_fine" / "signals"

SIGNALS = [
    ("sc0p4_rz0p6", SIGNAL_DIR / "sc0p4_rz0p6.csv"),
    ("sc0p45_rz0p55", SIGNAL_DIR / "sc0p45_rz0p55.csv"),
]

CONFIGS = []
for signal_name, signal_file in SIGNALS:
    for target_cap in (0.3267, 0.38, 0.45, 0.98):
        CONFIGS.append(
            {
                "signal_name": signal_name,
                "signal_file": signal_file,
                "target_cap": target_cap,
                "open_daily_score_exit": 0,
                "score_exit_entry_ratio": None,
                "min_hold": 2,
            }
        )
        for ratio in (0.88, 0.92, 0.95, 0.98, 1.02):
            for min_hold in (1, 2, 3):
                CONFIGS.append(
                    {
                        "signal_name": signal_name,
                        "signal_file": signal_file,
                        "target_cap": target_cap,
                        "open_daily_score_exit": 1,
                        "score_exit_entry_ratio": ratio,
                        "min_hold": min_hold,
                    }
                )


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none")


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


def _run(config: dict, log_file: Path) -> int:
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(config["open_daily_score_exit"]),
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": str(config["min_hold"]),
            "GM_MAX_DAILY_SELLS": "0",
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
        str(config["signal_file"]),
        "--log-file",
        str(log_file),
        "--max-positions",
        "3",
        "--holding-days",
        "5",
        "--max-holding-days",
        "15",
        "--target-position-pct",
        str(config["target_cap"]),
        "--score-db",
        str(FUSION_DB),
        "--score-table",
        FUSION_TABLE,
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
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return int(payload.get("returncode", proc.returncode))
        except Exception:
            pass
    return proc.returncode


def main() -> None:
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for config in CONFIGS:
        if not Path(config["signal_file"]).exists():
            raise SystemExit(f"Missing signal file: {config['signal_file']}")
        name = (
            f"{config['signal_name']}_cap{_safe(config['target_cap'])}_"
            f"exit{config['open_daily_score_exit']}_ratio{_safe(config['score_exit_entry_ratio'])}_"
            f"mh{config['min_hold']}"
        )
        log_file = REPORT_DIR / "logs" / f"{name}.log"
        returncode = 0 if log_file.exists() and log_file.stat().st_size > 0 else _run(config, log_file)
        indicator = _extract_indicator(log_file)
        row = {
            "name": name,
            "returncode": returncode,
            "signal_name": config["signal_name"],
            "signal_file": str(config["signal_file"]),
            "target_cap": config["target_cap"],
            "open_daily_score_exit": config["open_daily_score_exit"],
            "score_exit_entry_ratio": config["score_exit_entry_ratio"],
            "min_hold": config["min_hold"],
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
    qualifying = [
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
        writer.writerows(sorted(qualifying, key=lambda row: float(row["sharpe"]), reverse=True))


if __name__ == "__main__":
    main()
