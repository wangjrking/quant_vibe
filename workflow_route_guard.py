"""Machine-checkable route contract for the standard incremental workflow.

This module is deliberately side-effect free.  It validates the route before a
workflow monitor is created, so a natural-language request for an incremental
run cannot silently select a full-history entry point.
"""

from __future__ import annotations

from typing import Any


TARGET_DATE_INCREMENTAL_SCOPE = "target_trade_date_only"
ROUTINE_INCREMENTAL_PROFILE = "routine_target_date_incremental_fast_path_v1"
ROUTINE_AUTHORIZATION_POLICY = "owner_approved_once_auto_advance_v1"

# These entry points are valid for other explicitly approved workflows, but
# must never appear in the standard target-date incremental route.
FORBIDDEN_INCREMENTAL_ENTRYPOINTS = frozenset(
    {
        "rebuild_l2_stock_daily_duckdb_mainline.py",
        "rebuild_l3_full_duckdb_mainline.py",
        "refresh_l3_active_duckdb_full_delivery.py",
        "rebuild_factor_data_batched.py",
        "gtja_alpha_workflow.py",
    }
)

REQUIRED_INCREMENTAL_ENTRYPOINTS = {
    "L1": "run_all_a_raw_update.py",
    "L2": "integrate_l2_latest_and_recompute_qfq.py",
    "L3": "deliver_l3_target_date_duckdb_mainline.py",
    "L4": "incremental_formal_l4_duckdb_mainline.py",
    "L5": "run_production_tasks.py",
    "L6": "run_production_tasks.py",
    "L7": "build_l7_delivery_package.py",
    "L8": "workflow_monitor_manage.py",
}
REQUIRED_INCREMENTAL_COMPANIONS = {
    "L2": ("integrate_l2_stock_risk_events_daily.py",),
}


def build_target_date_incremental_route(target_trade_date: str) -> dict[str, Any]:
    """Return the only supported route for a target-date signal run."""

    target_trade_date = str(target_trade_date).strip()
    if len(target_trade_date) != 8 or not target_trade_date.isdigit():
        raise ValueError("target_trade_date must be YYYYMMDD")
    stages = []
    for layer, entrypoint in REQUIRED_INCREMENTAL_ENTRYPOINTS.items():
        stage = {
                "layer": layer,
                "entrypoint": entrypoint,
                "scope": TARGET_DATE_INCREMENTAL_SCOPE,
                "target_trade_date": target_trade_date,
                "full_history_rebuild": False,
            }
        if layer in REQUIRED_INCREMENTAL_COMPANIONS:
            stage["companion_entrypoints"] = list(
                REQUIRED_INCREMENTAL_COMPANIONS[layer]
            )
        stages.append(stage)
    return {
        "route_id": "standard_target_date_incremental_l1_l8_v1",
        "execution_profile": ROUTINE_INCREMENTAL_PROFILE,
        "execution_scope": TARGET_DATE_INCREMENTAL_SCOPE,
        "target_trade_date": target_trade_date,
        "full_history_rebuild": False,
        "requires_candidate_grant": False,
        "requires_architect_review": False,
        "authorization_policy": {
            "policy_id": ROUTINE_AUTHORIZATION_POLICY,
            "owner_approval_at_workflow_start": True,
            "per_layer_commander_grant": False,
            "audit_pass_auto_dispatches_next_layer": True,
            "control_thread_timeout_is_blocking": False,
            "business_execution_retry_limit": 1,
            "retryable_failure_classes": [
                "launcher_identity",
                "stale_lease",
                "control_thread_timeout",
                "bounded_stage_timeout",
            ],
            "non_retryable_failure_classes": [
                "data_quality",
                "source_not_ready",
                "schema_drift",
                "future_leakage",
                "unknown_writer",
                "asset_fingerprint_drift",
            ],
        },
        "evidence_policy": {
            "single_run_manifest": True,
            "single_startup_preflight": True,
            "audit_each_layer_handoff": True,
            "repeat_package_parser_checks": False,
        },
        "stages": stages,
    }


def validate_incremental_route(route: dict[str, Any]) -> list[str]:
    """Return blocking errors; an empty list means the route is safe to use."""

    errors: list[str] = []
    if route.get("execution_profile") != ROUTINE_INCREMENTAL_PROFILE:
        errors.append(f"execution_profile must be {ROUTINE_INCREMENTAL_PROFILE}")
    if route.get("execution_scope") != TARGET_DATE_INCREMENTAL_SCOPE:
        errors.append("execution_scope must be target_trade_date_only")
    if route.get("full_history_rebuild") is not False:
        errors.append("full_history_rebuild must be false")
    if route.get("requires_candidate_grant") is not False:
        errors.append("routine incremental route must not require a candidate grant")
    if route.get("requires_architect_review") is not False:
        errors.append("routine incremental route must not require an architect review")
    authorization_policy = route.get("authorization_policy")
    if not isinstance(authorization_policy, dict):
        errors.append("authorization_policy must be an object")
    else:
        expected_policy = {
            "policy_id": ROUTINE_AUTHORIZATION_POLICY,
            "owner_approval_at_workflow_start": True,
            "per_layer_commander_grant": False,
            "audit_pass_auto_dispatches_next_layer": True,
            "control_thread_timeout_is_blocking": False,
            "business_execution_retry_limit": 1,
        }
        for key, value in expected_policy.items():
            if authorization_policy.get(key) != value:
                errors.append(f"authorization_policy.{key} must be {value!r}")
        if not authorization_policy.get("retryable_failure_classes"):
            errors.append("authorization_policy.retryable_failure_classes must be non-empty")
        if not authorization_policy.get("non_retryable_failure_classes"):
            errors.append("authorization_policy.non_retryable_failure_classes must be non-empty")
    evidence_policy = route.get("evidence_policy")
    if not isinstance(evidence_policy, dict):
        errors.append("evidence_policy must be an object")
    else:
        for key in ("single_run_manifest", "single_startup_preflight", "audit_each_layer_handoff"):
            if evidence_policy.get(key) is not True:
                errors.append(f"evidence_policy.{key} must be true")
        if evidence_policy.get("repeat_package_parser_checks") is not False:
            errors.append("evidence_policy.repeat_package_parser_checks must be false")
    target_trade_date = str(route.get("target_trade_date") or "")
    if len(target_trade_date) != 8 or not target_trade_date.isdigit():
        errors.append("target_trade_date must be YYYYMMDD")

    stages = route.get("stages")
    if not isinstance(stages, list):
        return errors + ["stages must be a list"]
    seen_layers: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            errors.append("each stage must be an object")
            continue
        layer = str(stage.get("layer") or "")
        entrypoint = str(stage.get("entrypoint") or "")
        seen_layers.add(layer)
        expected = REQUIRED_INCREMENTAL_ENTRYPOINTS.get(layer)
        if expected is None:
            errors.append(f"unsupported layer: {layer}")
        elif entrypoint != expected:
            errors.append(f"{layer} must use {expected}, got {entrypoint}")
        if stage.get("scope") != TARGET_DATE_INCREMENTAL_SCOPE:
            errors.append(f"{layer} scope must be target_trade_date_only")
        if str(stage.get("target_trade_date") or "") != target_trade_date:
            errors.append(f"{layer} target_trade_date mismatch")
        if stage.get("full_history_rebuild") is not False:
            errors.append(f"{layer} full_history_rebuild must be false")
        if entrypoint in FORBIDDEN_INCREMENTAL_ENTRYPOINTS:
            errors.append(f"forbidden full-history entrypoint in {layer}: {entrypoint}")
        expected_companions = set(REQUIRED_INCREMENTAL_COMPANIONS.get(layer, ()))
        actual_companions = set(stage.get("companion_entrypoints") or [])
        missing_companions = expected_companions - actual_companions
        errors.extend(
            f"{layer} missing companion entrypoint: {entrypoint}"
            for entrypoint in sorted(missing_companions)
        )
    missing = set(REQUIRED_INCREMENTAL_ENTRYPOINTS) - seen_layers
    errors.extend(f"missing layer: {layer}" for layer in sorted(missing))
    return errors


def validate_template_execution_contract(template: dict[str, Any]) -> list[str]:
    """Validate a monitor template's embedded route contract."""

    route = template.get("execution_route")
    if not isinstance(route, dict):
        return ["template is missing execution_route"]
    return validate_incremental_route(route)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate an incremental workflow route.")
    parser.add_argument("--target-trade-date", required=True)
    args = parser.parse_args()
    route = build_target_date_incremental_route(args.target_trade_date)
    errors = validate_incremental_route(route)
    print(json.dumps({"valid": not errors, "route": route, "errors": errors}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 2)
