from __future__ import annotations

from copy import deepcopy
from typing import Any


SCHEMA_VERSION = 1
CONTRACT_TYPE = "workflow_layer_handoff"
WORKFLOW_TEMPLATE_STANDARD_INCREMENTAL = "standard_incremental_trading_signal_l1_l8"
LAYER_ORDER = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")
LAYER_OWNER = {
    "L1": "data-ingestion-agent",
    "L2": "data-integration-agent",
    "L3": "factor-agent",
    "L4": "model-agent",
    "L5": "strategy-agent",
    "L6": "strategy-agent",
    "L7": "trading-agent",
    "L8": "mcp-agent",
}
TERMINAL_STATUSES = {
    "blocked",
    "source_not_ready",
    "pending",
    "in_progress",
    "reviewing",
    "ready_for_audit_review",
    "audit_passed",
    "completed",
    "pending_buy_day_hard_gate",
    "ready_for_human_confirmation_execution",
    "pending_user_approval",
}
DEFAULT_HARD_RULES = [
    "no-BJ",
    "DuckDB-only",
    "one-table-one-file",
    "explicit-qfq",
    "fail-closed",
]


def next_layer_for(layer: str) -> str | None:
    if layer not in LAYER_ORDER:
        return None
    idx = LAYER_ORDER.index(layer)
    if idx + 1 >= len(LAYER_ORDER):
        return None
    return LAYER_ORDER[idx + 1]


def _ensure_list(value: list[Any] | None) -> list[Any]:
    if value is None:
        return []
    return list(value)


def build_layer_handoff_contract(
    *,
    workflow_run_id: str,
    layer: str,
    target_trade_date: str,
    status: str,
    ready_for_audit_review: bool,
    allow_next_layer_continue: bool,
    active_input_assets: list[str] | None = None,
    active_output_assets: list[str] | None = None,
    gate_checks: list[dict[str, Any]] | None = None,
    handoff_constraints: list[str] | None = None,
    evidence_paths: list[str] | None = None,
    residual_risk: list[dict[str, Any]] | None = None,
    boundaries: dict[str, Any] | None = None,
    layer_payload: dict[str, Any] | None = None,
    owner_agent: str | None = None,
    workflow_template: str = WORKFLOW_TEMPLATE_STANDARD_INCREMENTAL,
    hard_rules: list[str] | None = None,
) -> dict[str, Any]:
    if layer not in LAYER_ORDER:
        raise ValueError(f"unsupported layer: {layer}")
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unsupported contract status: {status}")
    agent = owner_agent or LAYER_OWNER[layer]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_type": CONTRACT_TYPE,
        "workflow_template": workflow_template,
        "workflow_run_id": workflow_run_id,
        "layer": layer,
        "owner_agent": agent,
        "target_trade_date": str(target_trade_date),
        "status": status,
        "ready_for_audit_review": bool(ready_for_audit_review),
        "allow_next_layer_continue": bool(allow_next_layer_continue),
        "next_layer": next_layer_for(layer),
        "hard_rules": list(hard_rules or DEFAULT_HARD_RULES),
        "contract": {
            "active_input_assets": _ensure_list(active_input_assets),
            "active_output_assets": _ensure_list(active_output_assets),
            "gate_checks": _ensure_list(gate_checks),
            "handoff_constraints": _ensure_list(handoff_constraints),
            "evidence_paths": _ensure_list(evidence_paths),
            "residual_risk": _ensure_list(residual_risk),
            "boundaries": deepcopy(boundaries or {}),
        },
        "layer_payload": deepcopy(layer_payload or {}),
    }


def validate_layer_handoff_contract(
    payload: dict[str, Any],
    *,
    expected_layer: str | None = None,
    expected_owner_agent: str | None = None,
) -> list[str]:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return ["payload must be a dict"]
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("contract_type") != CONTRACT_TYPE:
        errors.append(f"contract_type must be {CONTRACT_TYPE}")

    layer = str(payload.get("layer") or "")
    if layer not in LAYER_ORDER:
        errors.append(f"layer must be one of {', '.join(LAYER_ORDER)}")
    if expected_layer and layer != expected_layer:
        errors.append(f"layer must be {expected_layer}")

    owner_agent = str(payload.get("owner_agent") or "")
    if layer in LAYER_OWNER and owner_agent != LAYER_OWNER[layer]:
        errors.append(f"owner_agent must be {LAYER_OWNER[layer]} for {layer}")
    if expected_owner_agent and owner_agent != expected_owner_agent:
        errors.append(f"owner_agent must be {expected_owner_agent}")

    for key in ("workflow_template", "workflow_run_id", "target_trade_date", "status"):
        if not str(payload.get(key) or "").strip():
            errors.append(f"{key} must be non-empty")

    if str(payload.get("status") or "") not in TERMINAL_STATUSES:
        errors.append("status is not in allowed terminal statuses")

    for key in ("ready_for_audit_review", "allow_next_layer_continue"):
        if not isinstance(payload.get(key), bool):
            errors.append(f"{key} must be boolean")

    next_layer = payload.get("next_layer")
    expected_next = next_layer_for(layer) if layer in LAYER_ORDER else None
    if next_layer != expected_next:
        errors.append(f"next_layer must be {expected_next!r} for {layer}")

    hard_rules = payload.get("hard_rules")
    if not isinstance(hard_rules, list) or not hard_rules:
        errors.append("hard_rules must be a non-empty list")
    else:
        missing_rules = [rule for rule in DEFAULT_HARD_RULES if rule not in hard_rules]
        if missing_rules:
            errors.append(f"hard_rules missing required items: {missing_rules}")

    contract = payload.get("contract")
    if not isinstance(contract, dict):
        errors.append("contract must be a dict")
        return errors

    list_fields = (
        "active_input_assets",
        "active_output_assets",
        "gate_checks",
        "handoff_constraints",
        "evidence_paths",
        "residual_risk",
    )
    for field in list_fields:
        if not isinstance(contract.get(field), list):
            errors.append(f"contract.{field} must be a list")
    if not isinstance(contract.get("boundaries"), dict):
        errors.append("contract.boundaries must be a dict")

    gate_checks = contract.get("gate_checks") or []
    if isinstance(gate_checks, list):
        for idx, item in enumerate(gate_checks):
            if not isinstance(item, dict):
                errors.append(f"contract.gate_checks[{idx}] must be a dict")
                continue
            for required in ("name", "passed"):
                if required not in item:
                    errors.append(f"contract.gate_checks[{idx}] missing {required}")
            if "passed" in item and not isinstance(item["passed"], bool):
                errors.append(f"contract.gate_checks[{idx}].passed must be boolean")

    layer_payload = payload.get("layer_payload")
    if not isinstance(layer_payload, dict):
        errors.append("layer_payload must be a dict")

    return errors
