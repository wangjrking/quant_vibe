from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from project_paths import PROJECT_ROOT, resolve_project_path


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "production_tasks.example.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def production_strategy_ids(registry: dict[str, Any]) -> set[str]:
    strategies = registry.get("production", {}).get("strategies", [])
    return {
        str(item["strategy_id"])
        for item in strategies
        if item.get("status") == "production" and item.get("strategy_id")
    }


def strategy_lookup(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["strategy_id"]): item
        for item in registry.get("production", {}).get("strategies", [])
        if item.get("strategy_id")
    }


def render_token(value: str, context: dict[str, str]) -> str:
    rendered = value
    for key, replacement in context.items():
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


def render_command(command: list[str], context: dict[str, str]) -> list[str]:
    return [render_token(str(part), context) for part in command]


def run_step(command: list[str], cwd: Path, log_path: Path, dry_run: bool = False) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    command_text = " ".join(command)

    if dry_run:
        return {
            "status": "dry_run",
            "command": command,
            "log_file": str(log_path),
            "started_at": started.isoformat(timespec="seconds"),
            "ended_at": started.isoformat(timespec="seconds"),
            "returncode": 0,
        }

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text}\n")
        log.write(f"started_at={started.isoformat(timespec='seconds')}\n")
        log.flush()
        proc = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
        ended = datetime.now()
        log.write(f"ended_at={ended.isoformat(timespec='seconds')}\n")
        log.write(f"returncode={proc.returncode}\n")

    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "command": command,
        "log_file": str(log_path),
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "returncode": proc.returncode,
    }


def run_production_tasks(config_path: Path, dry_run: bool = False) -> dict[str, Any]:
    config = load_json(config_path)
    defaults = config.get("defaults", {})
    project_dir = resolve_project_path(defaults.get("project_dir", "."))
    data_dir = resolve_project_path(defaults.get("data_dir", "data_file"))
    log_dir = resolve_project_path(defaults.get("log_dir", "logs/production_tasks"))
    signal_dir = resolve_project_path(defaults.get("signal_dir", "data_file/production_signals"))
    registry_path = resolve_project_path(defaults.get("registry_file", "strategy_library/registry.json"))
    registry = load_json(registry_path)

    production_ids = production_strategy_ids(registry)
    registry_by_id = strategy_lookup(registry)
    stop_on_failure = bool(defaults.get("stop_on_failure", True))
    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    signal_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_at": now.isoformat(timespec="seconds"),
        "config_file": str(config_path),
        "registry_file": str(registry_path),
        "production_strategies": sorted(production_ids),
        "results": [],
    }

    base_context = {
        "python": str(defaults.get("python", sys.executable)),
        "project_dir": str(project_dir),
        "data_dir": str(data_dir),
        "log_dir": str(log_dir),
        "signal_dir": str(signal_dir),
        "run_id": run_id,
    }

    for task in config.get("strategies", []):
        strategy_id = str(task.get("strategy_id", ""))
        if not task.get("enabled", True):
            continue
        if strategy_id not in production_ids:
            summary["results"].append(
                {
                    "strategy_id": strategy_id,
                    "status": "skipped",
                    "detail": "strategy is not registered as production",
                }
            )
            continue

        strategy = registry_by_id[strategy_id]
        context = {
            **base_context,
            "strategy_id": strategy_id,
            "strategy_name": str(strategy.get("name", strategy_id)),
        }
        strategy_result: dict[str, Any] = {
            "strategy_id": strategy_id,
            "strategy_name": strategy.get("name"),
            "status": "ok",
            "steps": [],
        }

        for step in task.get("steps", []):
            step_name = str(step.get("name", "step"))
            command = render_command(step.get("command", []), context)
            if not command:
                step_result = {"name": step_name, "status": "failed", "detail": "empty command"}
            else:
                log_path = log_dir / strategy_id / f"{run_id}_{step_name}.log"
                step_result = {
                    "name": step_name,
                    **run_step(command, project_dir, log_path, dry_run=dry_run),
                }
            strategy_result["steps"].append(step_result)
            if step_result["status"] == "failed":
                strategy_result["status"] = "failed"
                if stop_on_failure:
                    break

        summary["results"].append(strategy_result)
        if strategy_result["status"] == "failed" and stop_on_failure:
            break

    summary["ended_at"] = datetime.now().isoformat(timespec="seconds")
    summary_path = log_dir / f"production_tasks_{run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_file"] = str(summary_path)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run automation tasks for registered production strategies only.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_production_tasks(resolve_project_path(args.config), dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = any(item.get("status") == "failed" for item in summary["results"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
