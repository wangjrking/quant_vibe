from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import statistics
import subprocess
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_ROOT = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623"
OUT_DIR = REPORT_ROOT / "current_formal_top1keep_low_path_dependency"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")

BASE_SIGNAL = REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "signals" / "tp_peak_g1p38_o0p942.csv"
SCORE_DB = REPORT_ROOT / "current_formal_top1keep_tailpow_peak" / "scores_top1keep_tailpow_peak.db"
SCORE_TABLE = "score_tp_peak_g1p38_o0p942"
BACKTEST_END = "2026-06-23 15:30:00"

STARTS = [
    ("full_20240605", "20240605", "2024-06-05 09:00:00", False),
    ("late_20250701", "20250701", "2025-07-01 09:00:00", True),
    ("late_20251009", "20251009", "2025-10-09 09:00:00", True),
    ("late_20260105", "20260105", "2026-01-05 09:00:00", True),
]

CANDIDATES = [
    {"name": "lp_top2_h1_eq", "top_k": 2, "holding_days": 1, "target_each": 0.49},
    {"name": "lp_top3_h1_eq", "top_k": 3, "holding_days": 1, "target_each": 0.326},
    {"name": "lp_top4_h1_eq", "top_k": 4, "holding_days": 1, "target_each": 0.245},
    {"name": "lp_top2_h2_eq", "top_k": 2, "holding_days": 2, "target_each": 0.49},
    {"name": "lp_top3_h2_eq", "top_k": 3, "holding_days": 2, "target_each": 0.326},
    {"name": "lp_top4_h2_eq", "top_k": 4, "holding_days": 2, "target_each": 0.245},
    {"name": "lp_top2_h3_eq", "top_k": 2, "holding_days": 3, "target_each": 0.49},
    {"name": "lp_top3_h3_eq", "top_k": 3, "holding_days": 3, "target_each": 0.326},
    {"name": "lp_top4_h3_eq", "top_k": 4, "holding_days": 3, "target_each": 0.245},
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "0",
    "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "99",
    "GM_MAX_DAILY_SELLS": "0",
    "GM_STOP_LOSS_PCT": "none",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "none",
    "GM_EQUITY_DD_RISK_MODE": "0",
}


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _write_candidate_signal(fields: list[str], rows: list[dict[str, str]], candidate: dict) -> Path:
    output = OUT_DIR / "signals" / f"{candidate['name']}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    keep_fields = list(fields)
    for field in ("holding_days", "max_holding_days", "target_pct", "score_exit_entry_ratio"):
        if field not in keep_fields:
            keep_fields.append(field)
    adjusted = []
    for row in rows:
        rank = int(float(row.get("rank") or 999999))
        if rank > int(candidate["top_k"]):
            continue
        item = dict(row)
        item["target_pct"] = f"{float(candidate['target_each']):.5f}"
        item["holding_days"] = str(int(candidate["holding_days"]))
        item["max_holding_days"] = str(int(candidate["holding_days"]))
        item["score_exit_entry_ratio"] = ""
        adjusted.append({key: item.get(key, "") for key in keep_fields})
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keep_fields)
        writer.writeheader()
        writer.writerows(adjusted)
    return output


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


def _run_case(candidate: dict, signal_file: Path, start_name: str, backtest_start: str) -> dict:
    log_file = OUT_DIR / "logs" / f"{candidate['name']}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
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
            str(int(candidate["top_k"])),
            "--holding-days",
            str(int(candidate["holding_days"])),
            "--max-holding-days",
            str(int(candidate["holding_days"])),
            "--target-position-pct",
            "0.5",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            backtest_start,
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
        indicator = _extract_indicator(log_file)
        returncode = proc.returncode
    else:
        returncode = 0
    return {
        "candidate": candidate["name"],
        "top_k": candidate["top_k"],
        "holding_days": candidate["holding_days"],
        "target_each": candidate["target_each"],
        "start_name": start_name,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
    }


def _summarize(rows: list[dict]) -> list[dict]:
    by_candidate: dict[str, list[dict]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate"], []).append(row)
    summary = []
    for name, items in by_candidate.items():
        late = [row for row in items if row["start_name"].startswith("late_")]
        late_annual = [float(row["annual"]) for row in late if row["annual"] is not None]
        all_annual = [float(row["annual"]) for row in items if row["annual"] is not None]
        full = next(row for row in items if row["start_name"] == "full_20240605")
        summary.append(
            {
                "candidate": name,
                "top_k": full["top_k"],
                "holding_days": full["holding_days"],
                "target_each": full["target_each"],
                "full_annual": full["annual"],
                "full_sharpe": full["sharpe"],
                "full_max_drawdown": full["max_drawdown"],
                "late_min_annual": min(late_annual),
                "late_median_annual": statistics.median(late_annual),
                "late_mean_annual": statistics.mean(late_annual),
                "late_max_drawdown_worst": max(float(row["max_drawdown"]) for row in late if row["max_drawdown"] is not None),
                "all_min_annual": min(all_annual),
                "score": min(late_annual) * 0.55 + statistics.median(late_annual) * 0.35 + float(full["annual"]) * 0.10,
            }
        )
    summary.sort(key=lambda row: (row["score"], row["late_min_annual"], row["late_median_annual"]), reverse=True)
    return summary


def main() -> int:
    fields, base_rows = _load_rows(BASE_SIGNAL)
    rows = []
    for candidate in CANDIDATES:
        signal_file = _write_candidate_signal(fields, base_rows, candidate)
        for start_name, _start_date, backtest_start, _is_late in STARTS:
            row = _run_case(candidate, signal_file, start_name, backtest_start)
            rows.append(row)
            print(
                f"{candidate['name']} {start_name} annual={row['annual']} "
                f"sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "low_path_cases.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows)
    with (OUT_DIR / "low_path_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (OUT_DIR / "low_path_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
