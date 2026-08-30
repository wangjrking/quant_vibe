from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class KillSwitchState:
    global_blocked: bool = False
    blocked_accounts: dict[str, str] = field(default_factory=dict)
    blocked_strategies: dict[str, str] = field(default_factory=dict)
    blocked_symbols: dict[str, str] = field(default_factory=dict)
    updated_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "KillSwitchState":
        return cls(
            global_blocked=bool(payload.get("global_blocked", False)),
            blocked_accounts=dict(payload.get("blocked_accounts") or {}),
            blocked_strategies=dict(payload.get("blocked_strategies") or {}),
            blocked_symbols=dict(payload.get("blocked_symbols") or {}),
            updated_at=str(payload.get("updated_at") or _now_iso()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "global_blocked": self.global_blocked,
            "blocked_accounts": self.blocked_accounts,
            "blocked_strategies": self.blocked_strategies,
            "blocked_symbols": self.blocked_symbols,
            "updated_at": self.updated_at,
        }

    def blockers_for(self, account_id: str, strategy_id: str, symbol: str) -> list[str]:
        blockers: list[str] = []
        if self.global_blocked:
            blockers.append("kill_switch_global")
        if account_id and account_id in self.blocked_accounts:
            blockers.append(f"kill_switch_account:{self.blocked_accounts[account_id]}")
        if strategy_id and strategy_id in self.blocked_strategies:
            blockers.append(f"kill_switch_strategy:{self.blocked_strategies[strategy_id]}")
        if symbol and symbol in self.blocked_symbols:
            blockers.append(f"kill_switch_symbol:{self.blocked_symbols[symbol]}")
        return blockers


def load_kill_switch_state(path: str | Path) -> KillSwitchState:
    file_path = Path(path)
    if not file_path.exists():
        return KillSwitchState()
    return KillSwitchState.from_dict(json.loads(file_path.read_text(encoding="utf-8")))


def save_kill_switch_state(path: str | Path, state: KillSwitchState) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = _now_iso()
    file_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
