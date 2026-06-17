from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
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
    return parser.parse_args(argv)


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


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

    return [
        ("create virtual environment", [sys.executable, "-m", "venv", str(args.venv)]),
        ("install requirements", [str(py), "-m", "pip", "install", "-r", "requirements.txt"]),
        ("copy config.example.json to config.json when needed", None),
        ("download/update raw data", raw_update),
        (
            "build production factors",
            [str(py), "incremental_factor_update_target_date.py", "--target-date", target_date, "--data-dir", str(data_dir)],
        ),
        (
            "build prediction labels",
            [str(py), "build_prediction_label_parts.py", "--data-dir", str(data_dir)],
        ),
        (
            "train/update model predictions",
            [str(py), "run_pdb_update.py", "--data-dir", str(data_dir)],
        ),
        (
            "generate strategy signals",
            [str(py), "run_production_tasks.py", "--config", "config/production_tasks.example.json", "--dry-run"],
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    venv = Path(args.venv).expanduser().resolve()
    py = venv_python(venv)

    check_code = check_prerequisites(args, py, data_dir)
    if args.check or check_code:
        return check_code

    ensure_layout(data_dir, dry_run=args.dry_run)
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
