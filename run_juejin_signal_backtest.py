from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a Juejin official backtest against a signal file and save the log.")
    parser.add_argument("--strategy-dir", required=True)
    parser.add_argument("--signal-file", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--holding-days", type=int)
    parser.add_argument("--target-position-pct", type=float)
    parser.add_argument("--stop-loss-pct", type=float)
    parser.add_argument("--take-profit-pct", type=float)
    parser.add_argument("--score-db")
    parser.add_argument("--score-table")
    parser.add_argument("--market-db")
    parser.add_argument("--score-stop-loss-pred", type=float)
    parser.add_argument("--score-take-profit-pred", type=float)
    parser.add_argument("--score-stop-loss-ratio", type=float)
    parser.add_argument("--score-take-profit-ratio", type=float)
    parser.add_argument("--score-stop-loss-rank", type=float)
    parser.add_argument("--score-take-profit-rank", type=float)
    parser.add_argument("--score-stop-loss-day-drop-ratio", type=float)
    parser.add_argument("--score-take-profit-day-drop-ratio", type=float)
    parser.add_argument("--score-exit-entry-ratio", type=float)
    parser.add_argument("--score-exit-rank", type=float)
    parser.add_argument("--min-holding-days-before-score-exit", type=int)
    parser.add_argument("--score-continue-entry-ratio", type=float)
    parser.add_argument("--max-holding-days", type=int)
    parser.add_argument("--light-stop-loss-pct", type=float)
    parser.add_argument("--min-holding-days-before-light-stop", type=int)
    parser.add_argument("--backtest-start")
    parser.add_argument("--backtest-end")
    parser.add_argument("--backtest-adjust")
    parser.add_argument("--backtest-initial-cash", type=float)
    parser.add_argument("--backtest-slippage-ratio", type=float)
    return parser.parse_args(argv)


def extract_indicator(log_text: str):
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_text.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            try:
                return ast.literal_eval(payload)
            except Exception:
                return payload
    return None


def infer_backtest_window(signal_file: Path, holding_days: int | None) -> tuple[str, str]:
    buy_dates: list[datetime] = []
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            value = str(row.get("buy_date") or "").strip()
            if not value:
                continue
            buy_dates.append(datetime.strptime(value, "%Y%m%d"))
    if not buy_dates:
        raise SystemExit(f"No buy_date rows found in signal file: {signal_file}")
    min_buy = min(buy_dates)
    max_buy = max(buy_dates)
    buffer_days = max(int(holding_days or 1) * 3, 10)
    start = min_buy.strftime("%Y-%m-%d 09:00:00")
    end = (max_buy + timedelta(days=buffer_days)).strftime("%Y-%m-%d 15:30:00")
    return start, end


def main(argv=None):
    args = parse_args(argv)
    strategy_dir = Path(args.strategy_dir).resolve()
    signal_file = Path(args.signal_file).resolve()
    log_file = Path(args.log_file).resolve()
    if not strategy_dir.joinpath("main.py").exists():
        raise SystemExit(f"Strategy main.py not found under {strategy_dir}")
    if not signal_file.exists():
        raise SystemExit(f"Signal file not found: {signal_file}")

    env = os.environ.copy()
    env["GM_SIGNAL_FILE"] = str(signal_file)
    if args.max_positions is not None:
        env["GM_MAX_POSITIONS"] = str(args.max_positions)
    if args.holding_days is not None:
        env["GM_HOLDING_DAYS"] = str(args.holding_days)
    if args.target_position_pct is not None:
        env["GM_TARGET_POSITION_PCT"] = str(args.target_position_pct)
    if args.stop_loss_pct is not None:
        env["GM_STOP_LOSS_PCT"] = str(args.stop_loss_pct)
    if args.take_profit_pct is not None:
        env["GM_TAKE_PROFIT_PCT"] = str(args.take_profit_pct)
    if args.score_db:
        env["GM_SCORE_DB"] = str(Path(args.score_db).resolve())
    if args.score_table:
        env["GM_SCORE_TABLE"] = str(args.score_table)
    if args.market_db:
        env["GM_MARKET_DB"] = str(Path(args.market_db).resolve())
    if args.score_stop_loss_pred is not None:
        env["GM_SCORE_STOP_LOSS_PRED"] = str(args.score_stop_loss_pred)
    if args.score_take_profit_pred is not None:
        env["GM_SCORE_TAKE_PROFIT_PRED"] = str(args.score_take_profit_pred)
    if args.score_stop_loss_ratio is not None:
        env["GM_SCORE_STOP_LOSS_RATIO"] = str(args.score_stop_loss_ratio)
    if args.score_take_profit_ratio is not None:
        env["GM_SCORE_TAKE_PROFIT_RATIO"] = str(args.score_take_profit_ratio)
    if args.score_stop_loss_rank is not None:
        env["GM_SCORE_STOP_LOSS_RANK"] = str(args.score_stop_loss_rank)
    if args.score_take_profit_rank is not None:
        env["GM_SCORE_TAKE_PROFIT_RANK"] = str(args.score_take_profit_rank)
    if args.score_stop_loss_day_drop_ratio is not None:
        env["GM_SCORE_STOP_LOSS_DAY_DROP_RATIO"] = str(args.score_stop_loss_day_drop_ratio)
    if args.score_take_profit_day_drop_ratio is not None:
        env["GM_SCORE_TAKE_PROFIT_DAY_DROP_RATIO"] = str(args.score_take_profit_day_drop_ratio)
    if args.score_exit_entry_ratio is not None:
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(args.score_exit_entry_ratio)
    if args.score_exit_rank is not None:
        env["GM_SCORE_EXIT_RANK"] = str(args.score_exit_rank)
    if args.min_holding_days_before_score_exit is not None:
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(args.min_holding_days_before_score_exit)
    if args.score_continue_entry_ratio is not None:
        env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(args.score_continue_entry_ratio)
    if args.max_holding_days is not None:
        env["GM_MAX_HOLDING_DAYS"] = str(args.max_holding_days)
    if args.light_stop_loss_pct is not None:
        env["GM_LIGHT_STOP_LOSS_PCT"] = str(args.light_stop_loss_pct)
    if args.min_holding_days_before_light_stop is not None:
        env["GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP"] = str(args.min_holding_days_before_light_stop)
    backtest_start, backtest_end = infer_backtest_window(signal_file, args.holding_days)
    if args.backtest_start:
        backtest_start = args.backtest_start
    if args.backtest_end:
        backtest_end = args.backtest_end
    env["GM_BACKTEST_START"] = backtest_start
    env["GM_BACKTEST_END"] = backtest_end
    if args.backtest_adjust:
        env["GM_BACKTEST_ADJUST"] = str(args.backtest_adjust)
    if args.backtest_initial_cash is not None:
        env["GM_BACKTEST_INITIAL_CASH"] = str(args.backtest_initial_cash)
    if args.backtest_slippage_ratio is not None:
        env["GM_BACKTEST_SLIPPAGE_RATIO"] = str(args.backtest_slippage_ratio)

    python_exe = sys.executable
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [python_exe, "main.py"],
            cwd=str(strategy_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    indicator = extract_indicator(text)
    summary = {
        "returncode": proc.returncode,
        "strategy_dir": str(strategy_dir),
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "backtest_start": backtest_start,
        "backtest_end": backtest_end,
        "indicator": indicator,
    }
    print(json.dumps(summary, ensure_ascii=False, default=str))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
