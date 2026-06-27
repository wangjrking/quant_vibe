from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


APPROVED_STATUS = "approved_by_user_or_supervisor"
RESEARCH_AREA_AUTO_APPROVED_STATUS = "research_area_auto_approved_by_user_policy"
EXECUTION_ALLOWED_STATUSES = {APPROVED_STATUS, RESEARCH_AREA_AUTO_APPROVED_STATUS}


def validate_execution_gate(*, packet: dict[str, Any], packet_validation: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if packet_validation.get("ok") is not True:
        errors.append("authorization packet validation must be ok")
        errors.extend(str(item) for item in packet_validation.get("errors", []))

    approval_status = str(packet.get("approval_status", ""))
    if approval_status not in EXECUTION_ALLOWED_STATUSES:
        errors.append("authorization packet is not approved")

    requested = packet.get("requested_authorization", {})
    experiments_authorized = int(requested.get("experiments_count") or 0)
    experiments_checked = int(packet_validation.get("experiments_checked") or 0)
    if experiments_authorized <= 0:
        errors.append("requested_authorization.experiments_count must be positive")
    if experiments_checked != experiments_authorized:
        errors.append("packet validation experiments_checked must match authorized experiments_count")

    boundaries = packet.get("boundaries", {})
    for key in [
        "no_training_executed_by_packet_generation",
        "no_prediction_generated_by_packet_generation",
        "no_production_manifest_change",
        "no_signal",
        "no_backtest",
    ]:
        if boundaries.get(key) is not True:
            errors.append(f"boundary {key} must be true")

    if errors and approval_status not in EXECUTION_ALLOWED_STATUSES:
        status = "not_ready_waiting_for_user_or_supervisor_approval"
    elif errors:
        status = "not_ready"
    else:
        status = "ready_for_research_only_training_execution"

    return {
        "ok": not errors,
        "status": status,
        "approval_status": approval_status,
        "experiments_authorized": experiments_authorized,
        "errors": errors,
        "warnings": warnings,
        "boundaries": {
            "no_training_executed_by_gate_validation": True,
            "no_prediction_generated_by_gate_validation": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate V12 research training execution gate.")
    parser.add_argument("--packet-json", required=True)
    parser.add_argument("--packet-validation-json", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = validate_execution_gate(
        packet=_load_json(args.packet_json),
        packet_validation=_load_json(args.packet_validation_json),
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
