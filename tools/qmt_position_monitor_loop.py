"""Run the QMT position monitor in a local background loop.

This is a local process runner, not a Windows scheduled task. It repeatedly
invokes qmt_position_monitor.py during A-share trading windows and records
status files so missed runs are observable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time as time_module
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


MAIN_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "quant" / "data_file"
DEFAULT_RUNTIME_DIR = DATA_DIR / "runtime" / "trading_agent" / "position_monitor_loop"
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def is_trading_window(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    current = now.time()
    return time(9, 30) <= current <= time(11, 30) or time(13, 0) <= current <= time(15, 0)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_monitor(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    command = [
        sys.executable,
        str(MAIN_DIR / "tools" / "qmt_position_monitor.py"),
        "--account-id",
        args.account_id,
        "--account-type",
        args.account_type,
        "--format",
        "markdown",
    ]
    if args.force_run:
        command.append("--force")
    if args.strategy_id:
        command.extend(["--strategy-id", args.strategy_id])
    if args.userdata_path:
        command.extend(["--userdata-path", args.userdata_path])

    started_at = datetime.now(SHANGHAI_TZ)
    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_seconds,
    )
    finished_at = datetime.now(SHANGHAI_TZ)
    return {
        "event": "monitor_run",
        "scheduled_at": now.isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Loop qmt_position_monitor.py during trading hours.")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--account-type", default="STOCK")
    parser.add_argument("--userdata-path", default="")
    parser.add_argument("--strategy-id", default="")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--outside-sleep-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--once", action="store_true", help="Run one loop iteration and exit.")
    parser.add_argument("--force-run", action="store_true", help="Run monitor even outside trading windows.")
    args = parser.parse_args()

    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pid_path = runtime_dir / "monitor_loop.pid"
    status_path = runtime_dir / "status.json"
    log_path = runtime_dir / f"monitor_loop_{datetime.now(SHANGHAI_TZ).strftime('%Y%m%d')}.jsonl"

    startup = {
        "event": "startup",
        "pid": os.getpid(),
        "started_at": datetime.now(SHANGHAI_TZ).isoformat(),
        "account_id": args.account_id,
        "interval_seconds": args.interval_seconds,
        "runtime_dir": str(runtime_dir),
    }
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    write_json(status_path, startup)
    append_jsonl(log_path, startup)

    while True:
        now = datetime.now(SHANGHAI_TZ)
        in_window = is_trading_window(now)
        if in_window or args.force_run:
            try:
                event = run_monitor(args, now)
            except Exception as exc:  # pragma: no cover - runtime guard
                event = {
                    "event": "monitor_error",
                    "at": datetime.now(SHANGHAI_TZ).isoformat(),
                    "error": repr(exc),
                }
            write_json(status_path, event)
            append_jsonl(log_path, event)
            sleep_seconds = args.interval_seconds
        else:
            event = {
                "event": "outside_trading_window",
                "at": now.isoformat(),
                "next_check_seconds": args.outside_sleep_seconds,
            }
            write_json(status_path, event)
            append_jsonl(log_path, event)
            sleep_seconds = args.outside_sleep_seconds

        if args.once:
            return 0
        time_module.sleep(max(sleep_seconds, 1))


if __name__ == "__main__":
    raise SystemExit(main())
