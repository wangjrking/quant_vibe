from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from .contracts import ExecutionAuthorization


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_policy_hash(payload: dict[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("policy_hash", None)
    return sha256(_canonical_json(normalized).encode("utf-8")).hexdigest().upper()


def load_execution_authorization(path: str | Path) -> ExecutionAuthorization:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    expected_hash = compute_policy_hash(payload)
    if payload.get("policy_hash") in (None, ""):
        payload["policy_hash"] = expected_hash
    elif str(payload.get("policy_hash")).upper() != expected_hash:
        raise ValueError(
            "execution authorization policy_hash mismatch: "
            f"expected {expected_hash}, got {payload.get('policy_hash')}"
        )
    authorization = ExecutionAuthorization.from_dict(payload)
    if not authorization.authorization_id:
        raise ValueError("execution authorization missing authorization_id")
    if not authorization.account_id:
        raise ValueError("execution authorization missing account_id")
    if not authorization.strategy_id:
        raise ValueError("execution authorization missing strategy_id")
    if not authorization.strategy_version:
        raise ValueError("execution authorization missing strategy_version")
    if not authorization.owner_approval_id:
        raise ValueError("execution authorization missing owner_approval_id")
    if authorization.max_orders_per_batch <= 0:
        raise ValueError("execution authorization max_orders_per_batch must be positive")
    if authorization.max_order_notional <= 0:
        raise ValueError("execution authorization max_order_notional must be positive")
    return authorization
