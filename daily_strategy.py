"""Legacy mixed-asset daily stock selection pipeline.

This entry still stitches together the historical wide-factor parquet and
legacy `odb.db.stock_predict_data_*` prediction tables. The current standard
L5 automation entrypoint is `run_production_tasks.py` with an approved L4
prediction manifest. This legacy path therefore requires an explicit opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from model_asset_route import resolve_model_feature_path
from stock_daily_data_route import resolve_stock_daily_db_path


DEFAULT_LABEL = "10d_yield_rate"
LEGACY_ENTRY_NOTICE = """\
daily_strategy.py is a legacy mixed-asset orchestration entry.

It still checks:
- L3 legacy factor asset: data_file/stock_factor_data.parquet
- L4 legacy prediction asset: data_file/odb.db::stock_predict_data_*

Current standard layered defaults are:
- L1: quant/data_file/production_assets/duckdb/l1_raw_tables/[table].duckdb
- L2: quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb::STOCK_DAILY_DATA
- L3 features: quant/data_file/production_assets/duckdb/l3_feature_current.duckdb::<active_table>
- L3 labels: quant/data_file/production_assets/duckdb/l3_label_current.duckdb::<active_table>
- L5 automation entry: quant/main/run_production_tasks.py

To run this historical path intentionally, pass --allow-legacy-asset-chain
or set QUANT_ALLOW_LEGACY_DAILY_STRATEGY=1.
"""


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


def legacy_asset_chain_opted_in(args) -> bool:
    return bool(getattr(args, "allow_legacy_asset_chain", False)) or os.environ.get(
        "QUANT_ALLOW_LEGACY_DAILY_STRATEGY"
    ) == "1"


def require_legacy_asset_chain_opt_in(args) -> None:
    if legacy_asset_chain_opted_in(args):
        return
    raise RuntimeError(LEGACY_ENTRY_NOTICE)


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


def parquet_asset_max_date(path: Path, column: str = "trade_date") -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return parquet_max_date(path, column=column)

    max_date: str | None = None
    for parquet_path in sorted(path.glob("*.parquet")):
        table = pq.read_table(parquet_path, columns=[column])
        value = _max_date(table.column(column).to_pylist())
        if value is not None and (max_date is None or value > max_date):
            max_date = value
    return max_date


def sqlite_table_max_date(db_path: Path, table: str, column: str = "trade_date") -> str | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f'SELECT MAX({column}) FROM "{table}"').fetchone()
    return None if row is None else (str(row[0]) if row[0] is not None else None)


def load_data_state(data_dir: Path) -> DataState:
    return load_data_state_for_label(data_dir, DEFAULT_LABEL)


def load_data_state_for_label(data_dir: Path, label: str) -> DataState:
    stock_daily_db = resolve_stock_daily_db_path(data_dir=data_dir)
    feature_path = resolve_model_feature_path(data_dir=data_dir)
    return DataState(
        source_date=sqlite_table_max_date(stock_daily_db, "STOCK_DAILY_DATA"),
        factor_date=parquet_asset_max_date(feature_path),
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
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy mixed-asset daily stock selection strategy. "
            "Current production/default L5 automation should use run_production_tasks.py."
        )
    )
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
    parser.add_argument(
        "--allow-legacy-asset-chain",
        action="store_true",
        help="Explicitly allow this legacy mixed-asset entry to run.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        require_legacy_asset_chain_opt_in(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summary = run_daily(args)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] in {"ok", "skipped"} for item in summary["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
