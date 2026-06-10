from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export gm signals from a prediction table, then run the official Juejin backtest and save the log."
    )
    parser.add_argument("--db", default="../data_file/odb.db")
    parser.add_argument("--table", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--stock-pool")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--min-pred", default="none")
    parser.add_argument("--max-atr-ratio", type=float, required=True)
    parser.add_argument("--weight-mode", default="equal", choices=["equal", "rank", "score"])
    parser.add_argument("--target-total-pct", type=float, default=0.98)
    parser.add_argument("--signal-output", required=True)
    parser.add_argument("--strategy-dir", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--max-positions", type=int, required=True)
    parser.add_argument("--holding-days", type=int, required=True)
    parser.add_argument("--target-position-pct", type=float)
    parser.add_argument("--stop-loss-pct", type=float)
    parser.add_argument("--take-profit-pct", type=float)
    parser.add_argument("--score-db")
    parser.add_argument("--score-table")
    parser.add_argument("--score-stop-loss-pred", type=float)
    parser.add_argument("--score-take-profit-pred", type=float)
    parser.add_argument("--score-stop-loss-ratio", type=float)
    parser.add_argument("--score-take-profit-ratio", type=float)
    parser.add_argument("--score-stop-loss-rank", type=float)
    parser.add_argument("--score-take-profit-rank", type=float)
    parser.add_argument("--score-stop-loss-day-drop-ratio", type=float)
    parser.add_argument("--score-take-profit-day-drop-ratio", type=float)
    parser.add_argument("--score-exit-entry-ratio", type=float)
    parser.add_argument("--min-holding-days-before-score-exit", type=int)
    parser.add_argument("--score-continue-entry-ratio", type=float)
    parser.add_argument("--max-holding-days", type=int)
    parser.add_argument("--light-stop-loss-pct", type=float)
    parser.add_argument("--min-holding-days-before-light-stop", type=int)
    return parser.parse_args(argv)


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    python_exe = sys.executable
    script_dir = Path(__file__).resolve().parent
    signal_output = Path(args.signal_output).resolve()
    log_file = Path(args.log_file).resolve()

    export_cmd = [
        python_exe,
        str((script_dir / "export_gm_signals.py").resolve()),
        "--db",
        str(Path(args.db).resolve()),
        "--table",
        args.table,
        "--start",
        args.start,
        "--end",
        args.end,
        "--top-k",
        str(args.top_k),
        "--min-pred",
        str(args.min_pred),
        "--max-atr-ratio",
        str(args.max_atr_ratio),
        "--weight-mode",
        args.weight_mode,
        "--target-total-pct",
        str(args.target_total_pct),
        "--output",
        str(signal_output),
    ]
    if args.stock_pool:
        export_cmd.extend(["--stock-pool", str(Path(args.stock_pool).resolve())])
    run(export_cmd, script_dir)

    juejin_cmd = [
        python_exe,
        str((script_dir / "run_juejin_signal_backtest.py").resolve()),
        "--strategy-dir",
        str(Path(args.strategy_dir).resolve()),
        "--signal-file",
        str(signal_output),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(args.max_positions),
        "--holding-days",
        str(args.holding_days),
    ]
    if args.target_position_pct is not None:
        juejin_cmd.extend(["--target-position-pct", str(args.target_position_pct)])
    if args.stop_loss_pct is not None:
        juejin_cmd.extend(["--stop-loss-pct", str(args.stop_loss_pct)])
    if args.take_profit_pct is not None:
        juejin_cmd.extend(["--take-profit-pct", str(args.take_profit_pct)])
    if args.score_db:
        juejin_cmd.extend(["--score-db", str(Path(args.score_db).resolve())])
    if args.score_table:
        juejin_cmd.extend(["--score-table", str(args.score_table)])
    if args.score_stop_loss_pred is not None:
        juejin_cmd.extend(["--score-stop-loss-pred", str(args.score_stop_loss_pred)])
    if args.score_take_profit_pred is not None:
        juejin_cmd.extend(["--score-take-profit-pred", str(args.score_take_profit_pred)])
    if args.score_stop_loss_ratio is not None:
        juejin_cmd.extend(["--score-stop-loss-ratio", str(args.score_stop_loss_ratio)])
    if args.score_take_profit_ratio is not None:
        juejin_cmd.extend(["--score-take-profit-ratio", str(args.score_take_profit_ratio)])
    if args.score_stop_loss_rank is not None:
        juejin_cmd.extend(["--score-stop-loss-rank", str(args.score_stop_loss_rank)])
    if args.score_take_profit_rank is not None:
        juejin_cmd.extend(["--score-take-profit-rank", str(args.score_take_profit_rank)])
    if args.score_stop_loss_day_drop_ratio is not None:
        juejin_cmd.extend(["--score-stop-loss-day-drop-ratio", str(args.score_stop_loss_day_drop_ratio)])
    if args.score_take_profit_day_drop_ratio is not None:
        juejin_cmd.extend(["--score-take-profit-day-drop-ratio", str(args.score_take_profit_day_drop_ratio)])
    if args.score_exit_entry_ratio is not None:
        juejin_cmd.extend(["--score-exit-entry-ratio", str(args.score_exit_entry_ratio)])
    if args.min_holding_days_before_score_exit is not None:
        juejin_cmd.extend(["--min-holding-days-before-score-exit", str(args.min_holding_days_before_score_exit)])
    if args.score_continue_entry_ratio is not None:
        juejin_cmd.extend(["--score-continue-entry-ratio", str(args.score_continue_entry_ratio)])
    if args.max_holding_days is not None:
        juejin_cmd.extend(["--max-holding-days", str(args.max_holding_days)])
    if args.light_stop_loss_pct is not None:
        juejin_cmd.extend(["--light-stop-loss-pct", str(args.light_stop_loss_pct)])
    if args.min_holding_days_before_light_stop is not None:
        juejin_cmd.extend(["--min-holding-days-before-light-stop", str(args.min_holding_days_before_light_stop)])
    run(juejin_cmd, script_dir)
    print(f"signal_output={signal_output}")
    print(f"log_file={log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
