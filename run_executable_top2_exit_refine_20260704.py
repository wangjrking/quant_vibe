from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("D:/work/quant/quant_mcp")
MAIN = ROOT / "quant/main"
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DIR = REPORT_DIR / "production_only_juejin_20260704"
RUN_DIR = OUT_DIR / "executable_top2_exit_refine_grid"
SUMMARY = OUT_DIR / "executable_top2_exit_refine_summary.csv"
STRATEGY_DIR = OUT_DIR / "code_snapshot_sell_available_safe"
SIGNAL_FILE = OUT_DIR / "executable_top2_open_gap_market_signals/top2_s150_cap82_w62.csv"
RUNNER = MAIN / "run_juejin_signal_backtest.py"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


CASES = []
for hold, max_hold in [(1, 1), (1, 2), (2, 3), (2, 4), (3, 5)]:
    for score_exit in [0.88, 0.90, 0.92, 0.95, 9.99]:
        for stop in [0.05, 0.08, 0.10]:
            for take in [None, 0.08, 0.12]:
                name = (
                    f"h{hold}m{max_hold}_e{str(score_exit).replace('.', 'p')}"
                    f"_sl{str(stop).replace('.', 'p')}"
                    f"_tp{('off' if take is None else str(take).replace('.', 'p'))}"
                )
                CASES.append((name, hold, max_hold, score_exit, stop, take))


def parse_indicator(stdout: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except Exception:
        return {}
    indicator = payload.get("indicator")
    if isinstance(indicator, dict):
        return indicator
    if not isinstance(indicator, str):
        return {}
    parsed: dict[str, object] = {}
    for key in [
        "pnl_ratio_annual",
        "sharp_ratio",
        "max_drawdown",
        "pnl_ratio",
        "open_count",
        "close_count",
        "win_ratio",
        "calmar_ratio",
    ]:
        match = re.search(rf"'{key}': ([0-9eE+\-.]+)", indicator)
        if match:
            value = float(match.group(1))
            parsed[key] = int(value) if key.endswith("_count") else value
    if parsed:
        return parsed
    text = re.sub(r"datetime\.datetime\(.*?\)", "'datetime'", indicator)
    try:
        parsed_any = ast.literal_eval(text)
    except Exception:
        return {}
    return parsed_any if isinstance(parsed_any, dict) else {}


def run_case(case: tuple[str, int, int, float, float, float | None]) -> dict[str, object]:
    name, hold, max_hold, score_exit, stop, take = case
    log_file = RUN_DIR / f"{name}.log"
    runner_json = RUN_DIR / f"{name}.runner.json"
    if runner_json.exists() and log_file.exists():
        stdout = runner_json.read_text(encoding="utf-8", errors="ignore")
        returncode = 0
    else:
        cmd = [
            sys.executable,
            str(RUNNER),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(SIGNAL_FILE),
            "--log-file",
            str(log_file),
            "--max-positions",
            "2",
            "--holding-days",
            str(hold),
            "--target-position-pct",
            "0.82",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            "score",
            "--market-db",
            str(MARKET_DB),
            "--score-exit-entry-ratio",
            str(score_exit),
            "--min-holding-days-before-score-exit",
            str(hold),
            "--score-continue-entry-ratio",
            "1.000",
            "--max-holding-days",
            str(max_hold),
            "--light-stop-loss-pct",
            str(stop),
            "--min-holding-days-before-light-stop",
            "1",
            "--backtest-adjust",
            "none",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        if take is not None:
            cmd.extend(["--take-profit-pct", str(take)])
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout = proc.stdout
        returncode = proc.returncode
        runner_json.write_text(stdout, encoding="utf-8")
    indicator = parse_indicator(stdout)
    return {
        "name": name,
        "hold": hold,
        "max_hold": max_hold,
        "score_exit": score_exit,
        "stop": stop,
        "take": take,
        "returncode": returncode,
        "signal_file": str(SIGNAL_FILE),
        "log_file": str(log_file),
        "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
        "sharp_ratio": indicator.get("sharp_ratio"),
        "max_drawdown": indicator.get("max_drawdown"),
        "pnl_ratio": indicator.get("pnl_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
        "win_ratio": indicator.get("win_ratio"),
        "calmar_ratio": indicator.get("calmar_ratio"),
    }


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for case in CASES:
        print("RUN", case[0], flush=True)
        rows.append(run_case(case))
    rows.sort(
        key=lambda r: (
            float(r.get("sharp_ratio") or -999),
            float(r.get("pnl_ratio_annual") or -999),
        ),
        reverse=True,
    )
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows[:20], ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
