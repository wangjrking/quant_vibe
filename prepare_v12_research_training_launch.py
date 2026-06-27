from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


READY_STATUS = "ready_for_research_only_training_execution"


def prepare_launch(*, plan: dict[str, Any], execution_gate: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if execution_gate.get("ok") is not True or execution_gate.get("status") != READY_STATUS:
        errors.append("execution gate is not ready")
        errors.extend(str(item) for item in execution_gate.get("errors", []))
        return {
            "ok": False,
            "status": "blocked",
            "commands": [],
            "errors": errors,
            "warnings": warnings,
            "execution_gate_status": execution_gate.get("status"),
            "boundaries": {
                "no_training_executed_by_launcher": True,
                "no_prediction_generated_by_launcher": True,
                "no_production_manifest_change": True,
                "no_signal": True,
                "no_backtest": True,
            },
        }

    commands: list[dict[str, Any]] = []
    for item in plan.get("experiments", []):
        command = str(item.get("command") or "")
        if not command:
            errors.append(f"{item.get('horizon')} command is missing")
            continue
        commands.append(
            {
                "horizon": item.get("horizon"),
                "variant": item.get("variant"),
                "output_table": item.get("output_table"),
                "command": command,
            }
        )

    return {
        "ok": not errors,
        "status": "ready_commands_prepared_not_executed" if not errors else "not_ready",
        "commands": commands if not errors else [],
        "errors": errors,
        "warnings": warnings,
        "execution_gate_status": execution_gate.get("status"),
        "boundaries": {
            "no_training_executed_by_launcher": True,
            "no_prediction_generated_by_launcher": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare V12 research training launch commands after execution gate.")
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--execution-gate-json", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = prepare_launch(
        plan=_load_json(args.plan_json),
        execution_gate=_load_json(args.execution_gate_json),
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
