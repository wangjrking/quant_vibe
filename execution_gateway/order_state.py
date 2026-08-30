from __future__ import annotations


ORDER_STATE_TRANSITIONS: dict[str, set[str]] = {
    "proposed": {"risk_rejected", "approved"},
    "risk_rejected": set(),
    "approved": {"submitting", "risk_rejected"},
    "submitting": {"acknowledged", "reconcile_exception", "cancel_pending"},
    "acknowledged": {"partial_filled", "filled", "cancel_pending", "cancelled", "reconcile_exception"},
    "partial_filled": {"filled", "cancel_pending", "reconcile_exception"},
    "filled": set(),
    "cancel_pending": {"cancelled", "partial_filled", "filled", "reconcile_exception"},
    "cancelled": set(),
    "reconcile_exception": set(),
}


def assert_valid_transition(current_state: str, next_state: str) -> None:
    if current_state not in ORDER_STATE_TRANSITIONS:
        raise ValueError(f"unsupported current order state: {current_state}")
    if next_state not in ORDER_STATE_TRANSITIONS[current_state]:
        raise ValueError(f"invalid order state transition: {current_state} -> {next_state}")


def is_terminal_state(state: str) -> bool:
    if state not in ORDER_STATE_TRANSITIONS:
        raise ValueError(f"unsupported order state: {state}")
    return not ORDER_STATE_TRANSITIONS[state]
