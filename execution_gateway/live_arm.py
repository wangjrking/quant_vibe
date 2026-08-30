from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ExecutionAuthorization


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class LiveArmState:
    enabled: bool = False
    arm_id: str = ""
    owner_approval_id: str = ""
    account_id: str = ""
    strategy_id: str = ""
    stage: str = "live"
    armed_by: str = ""
    reason: str = ""
    created_at: str = ""
    expires_at: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiveArmState":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            arm_id=str(payload.get("arm_id") or "").strip(),
            owner_approval_id=str(payload.get("owner_approval_id") or "").strip(),
            account_id=str(payload.get("account_id") or "").strip(),
            strategy_id=str(payload.get("strategy_id") or "").strip(),
            stage=str(payload.get("stage") or "live").strip().lower(),
            armed_by=str(payload.get("armed_by") or "").strip(),
            reason=str(payload.get("reason") or "").strip(),
            created_at=str(payload.get("created_at") or "").strip(),
            expires_at=str(payload.get("expires_at") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "arm_id": self.arm_id,
            "owner_approval_id": self.owner_approval_id,
            "account_id": self.account_id,
            "strategy_id": self.strategy_id,
            "stage": self.stage,
            "armed_by": self.armed_by,
            "reason": self.reason,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


def load_live_arm_state(path: str | Path) -> LiveArmState:
    file_path = Path(path)
    if not file_path.exists():
        return LiveArmState()
    return LiveArmState.from_dict(json.loads(file_path.read_text(encoding="utf-8-sig")))


def validate_live_arm_state(
    *,
    arm_state: LiveArmState,
    authorization: ExecutionAuthorization,
    now: datetime | None = None,
) -> tuple[bool, tuple[str, ...]]:
    blockers: list[str] = []
    now_value = now or datetime.now(timezone.utc)

    if authorization.stage != "live":
        blockers.append("authorization_stage_not_live")
    if not arm_state.enabled:
        blockers.append("live_arm_disabled")
    if arm_state.stage and arm_state.stage != "live":
        blockers.append("live_arm_stage_not_live")
    if not arm_state.arm_id:
        blockers.append("live_arm_id_missing")
    if not arm_state.owner_approval_id:
        blockers.append("live_arm_owner_approval_missing")
    if arm_state.owner_approval_id != authorization.owner_approval_id:
        blockers.append("live_arm_owner_approval_mismatch")
    if arm_state.account_id != authorization.account_id:
        blockers.append("live_arm_account_mismatch")
    if arm_state.strategy_id != authorization.strategy_id:
        blockers.append("live_arm_strategy_mismatch")

    expiry = _parse_ts(arm_state.expires_at)
    if expiry is None:
        blockers.append("live_arm_expiry_missing_or_invalid")
    elif now_value > expiry:
        blockers.append("live_arm_expired")

    created = _parse_ts(arm_state.created_at)
    if created is None:
        blockers.append("live_arm_created_at_missing_or_invalid")

    return (not blockers), tuple(blockers)


def save_live_arm_state(path: str | Path, state: LiveArmState) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_dict()
    payload.setdefault("created_at", _utc_iso_now())
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
