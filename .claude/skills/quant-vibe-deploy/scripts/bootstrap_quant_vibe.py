from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


SKILL_SCRIPT = Path(__file__).resolve()
REPO_ROOT = SKILL_SCRIPT.parents[4]
DEFAULT_VENV = REPO_ROOT / ".venv"
DEFAULT_DATA_DIR = REPO_ROOT.parent / "data_file"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a Quant Vibe clone.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--check", action="store_true", help="Validate prerequisites without running the full chain.")
    parser.add_argument("--skip-pip", action="store_true", help="Skip requirements installation.")
    parser.add_argument("--require-token", action="store_true", help="Fail when TUSHARE_TOKEN is missing.")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--target-date", default=None, help="Factor target date; defaults to --end-date.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--venv", default=str(DEFAULT_VENV))
    parser.add_argument("--limit-stocks", type=int, default=None, help="Limit raw update stock count for smoke runs.")
    parser.add_argument("--full-refresh", action="store_true", help="Pass --full-refresh to raw data update.")
    parser.add_argument("--skip-agent-session", action="store_true", help="Do not create local agent session files.")
    return parser.parse_args(argv)


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def is_tracked_script(name: str) -> bool:
    path = REPO_ROOT / name
    if not path.exists():
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", name],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.returncode == 0


def run_step(label: str, command: list[str] | None, *, dry_run: bool, cwd: Path = REPO_ROOT) -> None:
    print(f"[step] {label}")
    if command:
        print(f"       {command_text(command)}")
    if dry_run or command is None:
        return
    completed = subprocess.run(command, cwd=str(cwd), text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def ensure_layout(data_dir: Path, *, dry_run: bool) -> None:
    config = REPO_ROOT / "config.json"
    example = REPO_ROOT / "config.example.json"
    for path in [
        data_dir,
        data_dir / "reports",
        data_dir / "model_predictions",
        data_dir / "production_signals",
        REPO_ROOT / "logs",
    ]:
        print(f"[step] ensure directory {path}")
        if not dry_run:
            path.mkdir(parents=True, exist_ok=True)
    print("[step] copy config.example.json to config.json when missing")
    if not dry_run and not config.exists() and example.exists():
        shutil.copy2(example, config)


def write_agent_session(*, data_dir: Path, python_path: Path, dry_run: bool) -> Path:
    sessions_dir = data_dir / "runtime" / "agent_sessions"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"bootstrap_{timestamp}_{uuid.uuid4().hex[:8]}"
    session_path = sessions_dir / f"{session_id}.json"
    current_path = sessions_dir / "current_session.json"
    print("[step] initialize agent session")
    print(f"       session_file={session_path}")
    if dry_run:
        return session_path

    sessions_dir.mkdir(parents=True, exist_ok=True)
    session = {
        "session_id": session_id,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "data_dir": str(data_dir),
        "python": str(python_path),
        "skill": "quant-vibe-deploy",
        "deployment_phase": "initialized",
        "agents": [
            {"id": "commander", "role": "deployment orchestration and approvals"},
            {"id": "data-agent", "role": "raw data download and data quality checks"},
            {"id": "model-agent", "role": "factor, label, training, and prediction assets"},
            {"id": "strategy-agent", "role": "strategy signal generation and production task handoff"},
            {"id": "audit-agent", "role": "reproducibility, leakage, and deployment evidence checks"},
        ],
        "next_actions": [
            "Verify TUSHARE_TOKEN is present before real data download.",
            "Run raw data update.",
            "Build factors and labels through the clone-safe tracked chain.",
            "Train or update model predictions.",
            "Generate strategy signals only after model assets are ready.",
        ],
    }
    session_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    current_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "session_file": str(session_path),
                "updated_at": session["created_at"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return session_path


def check_prerequisites(args: argparse.Namespace, py: Path, data_dir: Path) -> int:
    failures: list[str] = []
    if sys.version_info >= (3, 13):
        failures.append("Python 3.13 is not supported by this project.")
    if not (REPO_ROOT / "requirements.txt").exists():
        failures.append("requirements.txt is missing.")
    if not (REPO_ROOT / "config.example.json").exists():
        failures.append("config.example.json is missing.")
    if args.require_token and not os.environ.get("TUSHARE_TOKEN"):
        failures.append("TUSHARE_TOKEN is required for data download.")

    print(f"repo_root={REPO_ROOT}")
    print(f"python={sys.executable}")
    print(f"planned_runtime_python={py}")
    print(f"data_dir={data_dir}")
    print(f"token_present={bool(os.environ.get('TUSHARE_TOKEN'))}")

    if failures:
        for failure in failures:
            print(f"[error] {failure}", file=sys.stderr)
        return 2
    return 0


def build_steps(args: argparse.Namespace, py: Path, data_dir: Path) -> list[tuple[str, list[str] | None]]:
    target_date = args.target_date or args.end_date
    raw_update = [
        str(py),
        "run_all_a_raw_update.py",
        "--end",
        args.end_date,
        "--data-dir",
        str(data_dir),
        "--sync-sqlite",
    ]
    if args.full_refresh:
        raw_update.append("--full-refresh")
    if args.limit_stocks is not None:
        raw_update.extend(["--limit-stocks", str(args.limit_stocks)])

    steps: list[tuple[str, list[str] | None]] = [
        ("create virtual environment", [sys.executable, "-m", "venv", str(args.venv)]),
        ("install requirements", [str(py), "-m", "pip", "install", "-r", "requirements.txt"]),
        ("copy config.example.json to config.json when needed", None),
        ("download/update raw data", raw_update),
    ]
    if is_tracked_script("incremental_factor_update_target_date.py") and is_tracked_script(
        "build_prediction_label_parts.py"
    ):
        steps.extend(
            [
                (
                    "build production factors",
                    [
                        str(py),
                        "incremental_factor_update_target_date.py",
                        "--target-date",
                        target_date,
                        "--data-dir",
                        str(data_dir),
                    ],
                ),
                (
                    "build prediction labels",
                    [str(py), "build_prediction_label_parts.py", "--data-dir", str(data_dir)],
                ),
            ]
        )
    else:
        factor_script = "run_cdb_update.py" if args.full_refresh else "run_incremental_cdb_update.py"
        steps.extend(
            [
                (
                    "build production factors",
                    [str(py), factor_script],
                ),
                (
                    "build prediction labels from compatibility factor table",
                    None,
                ),
            ]
        )
    steps.extend(
        [
            (
                "train/update model predictions",
                [str(py), "run_pdb_update.py", "--data-dir", str(data_dir)],
            ),
            (
                "generate strategy signals",
                [str(py), "run_production_tasks.py", "--config", "config/production_tasks.example.json", "--dry-run"],
            ),
        ]
    )
    return steps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    venv = Path(args.venv).expanduser().resolve()
    py = venv_python(venv)

    check_code = check_prerequisites(args, py, data_dir)
    if args.check or check_code:
        return check_code

    ensure_layout(data_dir, dry_run=args.dry_run)
    if not args.skip_agent_session:
        write_agent_session(data_dir=data_dir, python_path=py, dry_run=args.dry_run)
    for label, command in build_steps(args, py, data_dir):
        if args.skip_pip and label == "install requirements":
            print("[step] install requirements")
            print("       skipped by --skip-pip")
            continue
        if py.exists() and label == "create virtual environment":
            print("[step] create virtual environment")
            print(f"       already exists: {venv}")
            continue
        run_step(label, command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
