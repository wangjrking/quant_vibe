"""Deterministic state transitions for the routine target-date L1-L8 chain.

The module performs no business I/O.  It converts verified layer/audit events
into dispatch decisions so a commander response is not required between every
routine layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from workflow_contract import LAYER_ORDER, next_layer_for


RETRYABLE_FAILURES = frozenset(
    {
        "launcher_identity",
        "stale_lease",
        "control_thread_timeout",
        "bounded_stage_timeout",
    }
)
HARD_STOP_FAILURES = frozenset(
    {
        "data_quality",
        "source_not_ready",
        "schema_drift",
        "future_leakage",
        "unknown_writer",
        "asset_fingerprint_drift",
    }
)


@dataclass(frozen=True)
class AdvanceDecision:
    action: str
    layer: str
    next_layer: str | None
    workflow_status: str
    consume_business_retry: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "layer": self.layer,
            "next_layer": self.next_layer,
            "workflow_status": self.workflow_status,
            "consume_business_retry": self.consume_business_retry,
            "reason": self.reason,
        }


def decide_next_action(
    *,
    layer: str,
    event: str,
    contract_valid: bool = False,
    audit_passed: bool = False,
    failure_class: str | None = None,
    retries_used: int = 0,
    retry_limit: int = 1,
) -> AdvanceDecision:
    if layer not in LAYER_ORDER:
        raise ValueError(f"unsupported layer: {layer}")
    downstream = next_layer_for(layer)

    if event == "layer_completed":
        if not contract_valid:
            return AdvanceDecision("hard_stop", layer, None, "failed", reason="invalid_handoff_contract")
        return AdvanceDecision("request_audit", layer, downstream, "waiting_for_audit")

    if event == "audit_completed":
        if not audit_passed:
            return AdvanceDecision("hard_stop", layer, None, "failed", reason="audit_failed")
        if downstream is None:
            return AdvanceDecision("complete_workflow", layer, None, "completed")
        return AdvanceDecision("dispatch_next_layer", layer, downstream, "in_progress")

    if event == "control_thread_timeout":
        return AdvanceDecision(
            "redispatch_control_message",
            layer,
            layer,
            "in_progress",
            consume_business_retry=False,
            reason="control_thread_timeout_is_not_a_business_failure",
        )

    if event == "stage_failed":
        kind = str(failure_class or "")
        if kind in RETRYABLE_FAILURES and retries_used < retry_limit:
            return AdvanceDecision(
                "remediate_and_retry_same_layer",
                layer,
                layer,
                "in_progress",
                consume_business_retry=True,
                reason=kind,
            )
        reason = kind or "unclassified_failure"
        if kind in RETRYABLE_FAILURES:
            reason = f"retry_exhausted:{kind}"
        return AdvanceDecision("hard_stop", layer, None, "failed", reason=reason)

    raise ValueError(f"unsupported event: {event}")
