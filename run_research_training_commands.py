from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now().isoformat(timespec="seconds"), **payload}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run prepared research training commands sequentially.")
    parser.add_argument("--launch-json", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--status-path", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    launch_json = Path(args.launch_json)
    log_dir = Path(args.log_dir)
    status_path = Path(args.status_path)
    payload = json.loads(launch_json.read_text(encoding="utf-8"))
    commands = payload.get("commands") or []
    _write_status(
        status_path,
        {
            "event": "runner_start",
            "launch_json": str(launch_json),
            "command_count": len(commands),
        },
    )
    for command_spec in commands:
        name = f"{command_spec.get('horizon')}_{command_spec.get('variant')}"
        stdout_path = log_dir / f"{name}.stdout.log"
        stderr_path = log_dir / f"{name}.stderr.log"
        _write_status(
            status_path,
            {
                "event": "experiment_start",
                "name": name,
                "horizon": command_spec.get("horizon"),
                "variant": command_spec.get("variant"),
                "output_table": command_spec.get("output_table"),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
        )
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(command_spec["command"], shell=True, stdout=stdout, stderr=stderr, text=True)
        _write_status(
            status_path,
            {
                "event": "experiment_end",
                "name": name,
                "horizon": command_spec.get("horizon"),
                "variant": command_spec.get("variant"),
                "output_table": command_spec.get("output_table"),
                "exit_code": proc.returncode,
            },
        )
        if proc.returncode != 0:
            _write_status(status_path, {"event": "runner_stop_on_failure", "name": name, "exit_code": proc.returncode})
            return proc.returncode
    _write_status(status_path, {"event": "runner_complete", "command_count": len(commands)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
