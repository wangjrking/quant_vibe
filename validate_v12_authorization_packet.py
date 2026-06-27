from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_BOUNDARIES = [
    "no_training_executed_by_packet_generation",
    "no_prediction_generated_by_packet_generation",
    "no_production_manifest_change",
    "no_signal",
    "no_backtest",
]

REQUIRED_NOT_ALLOWED_TERMS = [
    "production manifest change",
    "formal L4 or L5 promotion",
    "trading signal generation",
    "strategy backtest",
    "legacy feature or prediction asset usage",
]

ALLOWED_APPROVAL_STATUSES = {
    "waiting_user_or_supervisor_approval",
    "approved_by_user_or_supervisor",
    "research_area_auto_approved_by_user_policy",
}


def _experiment_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("horizon")), str(item.get("output_table")))


def validate_authorization_packet(
    *,
    packet: dict[str, Any],
    plan: dict[str, Any],
    plan_validation: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if packet.get("approval_status") not in ALLOWED_APPROVAL_STATUSES:
        errors.append(
            "approval_status must be waiting_user_or_supervisor_approval, "
            "approved_by_user_or_supervisor, or research_area_auto_approved_by_user_policy"
        )

    boundaries = packet.get("boundaries", {})
    for key in REQUIRED_BOUNDARIES:
        if boundaries.get(key) is not True:
            errors.append(f"boundary {key} must be true")

    packet_experiments = packet.get("experiments", [])
    plan_experiments = plan.get("experiments", [])
    if len(packet_experiments) != len(plan_experiments):
        errors.append("authorization packet experiment count does not match experiment plan")

    requested = packet.get("requested_authorization", {})
    if int(requested.get("experiments_count") or -1) != len(plan_experiments):
        errors.append("requested_authorization.experiments_count does not match experiment plan")

    packet_keys = [_experiment_key(item) for item in packet_experiments]
    plan_keys = [_experiment_key(item) for item in plan_experiments]
    if packet_keys != plan_keys:
        errors.append("authorization packet experiments do not match experiment plan")

    not_allowed = set(str(item) for item in requested.get("explicitly_not_allowed", []))
    for term in REQUIRED_NOT_ALLOWED_TERMS:
        if term not in not_allowed:
            errors.append(f"requested_authorization.explicitly_not_allowed missing {term}")

    if plan_validation.get("ok") is not True:
        errors.append("experiment plan validation must be ok")
        errors.extend(str(item) for item in plan_validation.get("errors", []))
    if int(plan_validation.get("experiments_checked") or -1) != len(plan_experiments):
        errors.append("plan_validation.experiments_checked does not match experiment plan")

    pre_validation = packet.get("pre_execution_validation", {})
    if pre_validation.get("plan_validation_ok") is not True:
        errors.append("pre_execution_validation.plan_validation_ok must be true")
    if int(pre_validation.get("experiments_checked") or -1) != len(plan_experiments):
        errors.append("pre_execution_validation.experiments_checked does not match experiment plan")
    if int(pre_validation.get("parse_errors") or 0) != 0:
        errors.append("pre_execution_validation.parse_errors must be 0")

    for horizon, table in packet_keys:
        if "_research" not in table:
            errors.append(f"{horizon} output_table must contain _research")
        if "formal" in table or "production" in table:
            errors.append(f"{horizon} output_table must not contain formal or production")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "experiments_checked": len(packet_experiments),
        "boundaries": {
            "no_training": True,
            "no_prediction": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate V12 research training authorization packet.")
    parser.add_argument("--packet-json", required=True)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--plan-validation-json", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = validate_authorization_packet(
        packet=_load_json(args.packet_json),
        plan=_load_json(args.plan_json),
        plan_validation=_load_json(args.plan_validation_json),
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
