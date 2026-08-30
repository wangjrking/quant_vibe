from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "quant" / "main"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_command(job: dict[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(MAIN_DIR / "run_parallel_expanding2010_folds.py"),
        "--data-file-url",
        str(job["data_file_url"]),
        "--data-start",
        str(job["data_start"]),
        "--first-test",
        str(job["first_test"]),
        "--final-test",
        str(job["final_test"]),
        "--label",
        str(job["label"]),
        "--model-type",
        "reg",
        "--train-years",
        str(job["train_years"]),
        "--train-mode",
        str(job["train_mode"]),
        "--test-months",
        str(job["test_months"]),
        "--step-months",
        str(job["step_months"]),
        "--embargo-days",
        str(job["embargo_days"]),
        "--selected-features-path",
        str(job["selected_features_path"]),
        "--output-table",
        str(job["output_table"]),
        "--output-dir",
        str(job["output_dir"]),
        "--experiment-name",
        str(job["experiment_name"]),
        "--max-workers",
        str(job["max_workers"]),
        "--xgb-n-estimators",
        str(job["xgb"]["n_estimators"]),
        "--xgb-learning-rate",
        str(job["xgb"]["learning_rate"]),
        "--xgb-max-depth",
        str(job["xgb"]["max_depth"]),
        "--xgb-reg-lambda",
        str(job["xgb"]["reg_lambda"]),
        "--xgb-reg-alpha",
        str(job["xgb"]["reg_alpha"]),
        "--xgb-subsample",
        str(job["xgb"]["subsample"]),
        "--xgb-colsample-bytree",
        str(job["xgb"]["colsample_bytree"]),
        "--xgb-early-stopping-rounds",
        str(job["xgb"]["early_stopping_rounds"]),
        "--xgb-device",
        str(job["xgb"]["device"]),
        "--xgb-n-jobs",
        str(job["xgb"]["n_jobs"]),
        "--feature-source",
        "production_split",
        "--prediction-output-mode",
        "independent",
    ]
    if job.get("use_light_factor_data"):
        command.append("--use-light-factor-data")
    return command


def build_env(job: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in (job.get("env") or {}).items():
        env[str(key)] = str(value)
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one research full-retrain label job.")
    parser.add_argument("--job-spec", required=True)
    args = parser.parse_args(argv)

    job_path = Path(args.job_spec).resolve()
    job = read_json(job_path)
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "job_status.json"
    stdout_path = output_dir / "launcher_stdout.log"
    stderr_path = output_dir / "launcher_stderr.log"
    command = build_command(job)

    write_json(
        status_path,
        {
            "status": "running",
            "started_at": now_iso(),
            "job_spec": str(job_path),
            "command": command,
        },
    )

    env = build_env(job)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            command,
            cwd=str(MAIN_DIR),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

    payload = {
        "status": "completed" if proc.returncode == 0 else "failed",
        "started_at": read_json(status_path).get("started_at"),
        "finished_at": now_iso(),
        "job_spec": str(job_path),
        "command": command,
        "returncode": int(proc.returncode),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    write_json(status_path, payload)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
