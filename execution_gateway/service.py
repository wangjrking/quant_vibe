from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .contracts import AccountSnapshot, ExecutionAuthorization, MarketSnapshot, RiskDecision
from .kill_switch import KillSwitchState
from .pretrade_risk import evaluate_order_intent
from .signal_loader import build_order_intents_from_latest


class ExecutionGatewayService:
    def __init__(
        self,
        *,
        authorization: ExecutionAuthorization,
        kill_switch_state: KillSwitchState | None = None,
    ) -> None:
        self.authorization = authorization
        self.kill_switch_state = kill_switch_state or KillSwitchState()

    def evaluate_latest_batch(
        self,
        *,
        latest_csv_path: str,
        latest_status_path: str,
        workflow_run_id: str,
        account: AccountSnapshot,
        market_by_symbol: dict[str, MarketSnapshot],
    ) -> dict[str, Any]:
        intents, latest_status = build_order_intents_from_latest(
            latest_csv_path=latest_csv_path,
            latest_status_path=latest_status_path,
            workflow_run_id=workflow_run_id,
            account_id=account.account_id,
            strategy_version=self.authorization.strategy_version,
        )
        if not intents:
            return {
                "batch_status": "no_signal_hold_only",
                "signal_status": latest_status,
                "authorization": self.authorization.to_dict(),
                "intent_count": 0,
                "approved_count": 0,
                "decisions": [],
            }

        if len(intents) > self.authorization.max_orders_per_batch:
            return {
                "batch_status": "risk_rejected",
                "signal_status": latest_status,
                "authorization": self.authorization.to_dict(),
                "intent_count": len(intents),
                "approved_count": 0,
                "decisions": [],
                "batch_blockers": ["batch_order_count_limit_exceeded"],
            }

        decisions: list[RiskDecision] = []
        for intent in intents:
            market = market_by_symbol.get(intent.symbol)
            if market is None:
                decisions.append(
                    RiskDecision(
                        intent=intent,
                        status="risk_rejected",
                        approved=False,
                        blockers=("market_snapshot_missing",),
                    )
                )
                continue
            decisions.append(
                evaluate_order_intent(
                    intent=intent,
                    authorization=self.authorization,
                    account=account,
                    market=market,
                    kill_switch_state=self.kill_switch_state,
                )
            )

        approved_count = sum(1 for item in decisions if item.approved)
        statuses = {item.status for item in decisions}
        batch_status = "approved"
        if "risk_rejected" in statuses:
            batch_status = "risk_rejected"
        elif statuses == {"shadow_only"}:
            batch_status = "shadow_only"
        return {
            "batch_status": batch_status,
            "signal_status": latest_status,
            "authorization": self.authorization.to_dict(),
            "intent_count": len(intents),
            "approved_count": approved_count,
            "decisions": [item.to_dict() for item in decisions],
        }
