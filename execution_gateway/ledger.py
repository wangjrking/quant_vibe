from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    event_type: str
    created_at: str
    batch_id: str
    client_order_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LedgerEvent":
        return cls(
            event_id=str(payload.get("event_id") or ""),
            event_type=str(payload.get("event_type") or ""),
            created_at=str(payload.get("created_at") or ""),
            batch_id=str(payload.get("batch_id") or ""),
            client_order_id=str(payload.get("client_order_id") or ""),
            payload=dict(payload.get("payload") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "created_at": self.created_at,
            "batch_id": self.batch_id,
            "client_order_id": self.client_order_id,
            "payload": self.payload,
        }


class ExecutionLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        batch_id: str = "",
        client_order_id: str = "",
    ) -> LedgerEvent:
        event = LedgerEvent(
            event_id=uuid4().hex,
            event_type=event_type,
            created_at=_utc_iso_now(),
            batch_id=batch_id,
            client_order_id=client_order_id,
            payload=payload,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
        return event

    def read_events(self) -> list[LedgerEvent]:
        if not self.path.exists():
            return []
        events: list[LedgerEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                events.append(LedgerEvent.from_dict(json.loads(line)))
        return events

    def latest_events_by_client_order(self) -> dict[str, LedgerEvent]:
        latest: dict[str, LedgerEvent] = {}
        for event in self.read_events():
            if event.client_order_id:
                latest[event.client_order_id] = event
        return latest

    def has_submission(self, client_order_id: str) -> bool:
        for event in self.read_events():
            if event.client_order_id == client_order_id and event.event_type in {
                "order_submitted",
                "order_shadowed",
                "order_paper_acknowledged",
            }:
                return True
        return False
