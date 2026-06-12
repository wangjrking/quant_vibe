"""Daily update pipeline for the stock selection strategy.

The daily pipeline is intentionally stage-aware: it only recomputes expensive
factor and prediction artifacts when the upstream data has advanced.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_LABEL = "10d_yield_rate"


@dataclass
class DataState:
    source_date: str | None
    factor_date: str | None
    prediction_date: str | None


@dataclass
class StageResult:
    name: str
    status: str
    detail: str = ""


def _max_date(values) -> str | None:
    if values is None:
        return None
    series = pd.Series(values).dropna().astype(str)
    if series.empty:
        return None
    return series.max()


def parquet_max_date(path: Path, column: str = "trade_date") -> str | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=[column])
    return _max_date(frame[column])


def sqlite_table_max_date(db_path: Path, table: str, column: str = "trade_date") -> str | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f'SELECT MAX({column}) FROM "{table}"').fetchone()
    return None if row is None else (str(row[0]) if row[0] is not None else None)


def load_data_state(data_dir: Path) -> DataState:
    return load_data_state_for_label(data_dir, DEFAULT_LABEL)


def load_data_state_for_label(data_dir: Path, label: str) -> DataState:
    return DataState(
        source_date=sqlite_table_max_date(data_dir / "odb.db", "STOCK_DAILY_DATA"),
        factor_date=parquet_max_date(data_dir / "stock_factor_data.parquet"),
        prediction_date=sqlite_table_max_date(data_dir / "odb.db", f"stock_predict_data_{label}"),
    )


def should_update_factors(state: DataState) -> bool:
    return bool(state.source_date and (state.factor_date is None or state.factor_date < state.source_date))


def should_update_predictions(state: DataState, force: bool = False) -> bool:
    return bool(
        force
        or (
            state.source_date
            and (state.prediction_date is None or state.prediction_date < state.source_date)
        )
    )


def run_command(command: list[str] | str, cwd: Path, log_path: Path) -> StageResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(command, str):
        command_text = command
        shell = True
    else:
        command_text = " ".join(shlex.quote(part) for part in command)
        shell = False

    started = datetime.now()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {command_text}\n")
        log.write(f"started_at={started.isoformat(timespec='seconds')}\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ended = datetime.now()
        log.write(f"ended_at={ended.isoformat(timespec='seconds')}\n")
        log.write(f"returncode={proc.returncode}\n")

    status = "ok" if proc.returncode == 0 else "failed"
    detail = f"returncode={proc.returncode}; log={log_path}"
    return StageResult(log_path.stem, status, detail)


def run_selection(
    python_exe: str,
    project_dir: Path,
    data_dir: Path,
    output_path: Path | None,
    top_k: int,
    min_pred: float,
    max_atr_ratio: float,
    label: str = DEFAULT_LABEL,
) -> StageResult:
    output_path = output_path or data_dir / "selection_default_latest.csv"
    command = [
        python_exe,
        "selection_module.py",
        "--db",
        str(data_dir / "odb.db"),
        "--table",
        f"stock_predict_data_{label}",
        "--output",
        str(output_path),
        "--top-k",
        str(top_k),
        "--min-pred",
        str(min_pred),
        "--max-atr-ratio",
        str(max_atr_ratio),
    ]
    return run_command(command, project_dir, data_dir / "logs" / "daily_selection.log")


def run_daily(args) -> dict:
    project_dir = Path(args.project_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    python_exe = args.python
    label = getattr(args, "label", DEFAULT_LABEL)
    results: list[StageResult] = []

    if args.source_command:
        results.append(
            run_command(
                args.source_command,
                project_dir,
                data_dir / "logs" / "daily_source_update.log",
            )
        )
        if results[-1].status != "ok":
            return {"state": asdict(load_data_state_for_label(data_dir, label)), "results": [asdict(item) for item in results]}

    state = load_data_state_for_label(data_dir, label)
    if state.source_date is None:
        results.append(StageResult("state_check", "failed", "STOCK_DAILY_DATA has no trade_date"))
        return {"state": asdict(state), "results": [asdict(item) for item in results]}

    if should_update_factors(state):
        results.append(
            run_command(
                [python_exe, "run_incremental_cdb_update.py"],
                project_dir,
                data_dir / "logs" / "daily_incremental_cdb.log",
            )
        )
        if results[-1].status != "ok":
            return {"state": asdict(load_data_state_for_label(data_dir, label)), "results": [asdict(item) for item in results]}

        results.append(
            run_command(
                [python_exe, "run_repair_factor_types.py"],
                project_dir,
                data_dir / "logs" / "daily_repair_factor_types.log",
            )
        )
        if results[-1].status != "ok":
            return {"state": asdict(load_data_state_for_label(data_dir, label)), "results": [asdict(item) for item in results]}
    else:
        results.append(StageResult("factor_update", "skipped", "factor table already matches source date"))

    state = load_data_state_for_label(data_dir, label)
    if should_update_predictions(state, force=args.force_prediction):
        results.append(
            run_command(
                [python_exe, "run_pdb_update.py", "--label", label, "--data-dir", str(data_dir)],
                project_dir,
                data_dir / "logs" / "daily_pdb_update.log",
            )
        )
        if results[-1].status != "ok":
            return {"state": asdict(load_data_state_for_label(data_dir, label)), "results": [asdict(item) for item in results]}
    else:
        results.append(StageResult("prediction_update", "skipped", "prediction table already matches source date"))

    final_state = load_data_state_for_label(data_dir, label)
    selection_output = args.output
    if not selection_output and final_state.prediction_date:
        selection_output = str(data_dir / f"selection_daily_{label}_{final_state.prediction_date}.csv")
    results.append(
        run_selection(
            python_exe,
            project_dir,
            data_dir,
            Path(selection_output) if selection_output else None,
            args.top_k,
            args.min_pred,
            args.max_atr_ratio,
            label,
        )
    )

    return {"state": asdict(load_data_state_for_label(data_dir, label)), "results": [asdict(item) for item in results]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the daily stock selection strategy.")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--data-dir", default="data_file")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--source-command",
        help="Optional command to refresh raw/source data before factor and prediction stages.",
    )
    parser.add_argument("--force-prediction", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-pred", type=float, default=0.01)
    parser.add_argument("--max-atr-ratio", type=float, default=0.10)
    parser.add_argument("--output")
    parser.add_argument("--summary-output", default="data_file/daily_strategy_summary.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_daily(args)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] in {"ok", "skipped"} for item in summary["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
