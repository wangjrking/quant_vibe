from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any

from .broker_adapter import BrokerAdapter, BrokerOrderAck, BrokerOrderRequest
from .contracts import AccountSnapshot, MarketSnapshot, RiskDecision
from .ledger import ExecutionLedger


A_SHARE_BUY_LOT = 100


@dataclass(frozen=True)
class PreparedOrder:
    request: BrokerOrderRequest
    risk_decision: RiskDecision
    sizing_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "risk_decision": self.risk_decision.to_dict(),
            "sizing_notes": list(self.sizing_notes),
        }


def _buy_quantity_from_target_pct(
    *,
    target_pct: float,
    total_asset: float,
    unit_price: float,
) -> int:
    if target_pct <= 0 or total_asset <= 0 or unit_price <= 0:
        return 0
    raw_shares = (target_pct * total_asset) / unit_price
    return max(floor(raw_shares / A_SHARE_BUY_LOT), 0) * A_SHARE_BUY_LOT


def _sell_quantity_from_account(
    *,
    symbol: str,
    explicit_quantity: int | None,
    account: AccountSnapshot,
) -> int:
    if explicit_quantity and explicit_quantity > 0:
        return explicit_quantity
    position = account.positions.get(symbol)
    if not position:
        return 0
    return max(position.sellable_shares, 0)


def build_order_request(
    *,
    decision: RiskDecision,
    account: AccountSnapshot,
    market: MarketSnapshot,
    price_mode: str = "LIMIT",
) -> PreparedOrder:
    if not decision.approved:
        raise ValueError("cannot build order request from non-approved risk decision")

    intent = decision.intent
    unit_price = intent.limit_price or market.ask_price_1 or market.bid_price_1 or market.last_price
    notes: list[str] = []

    if intent.side == "BUY":
        quantity = intent.quantity or _buy_quantity_from_target_pct(
            target_pct=intent.target_pct,
            total_asset=account.total_asset,
            unit_price=unit_price,
        )
        if intent.quantity is None:
            notes.append("quantity_derived_from_target_pct")
        if quantity < A_SHARE_BUY_LOT:
            raise ValueError("derived buy quantity fell below one board lot")
    else:
        quantity = _sell_quantity_from_account(
            symbol=intent.symbol,
            explicit_quantity=intent.quantity,
            account=account,
        )
        if intent.quantity is None:
            notes.append("quantity_derived_from_sellable_shares")
        if quantity <= 0:
            raise ValueError("sell quantity resolved to zero")

    request = BrokerOrderRequest(
        intent=intent,
        client_order_id=intent.idempotency_key,
        quantity=quantity,
        price_mode=price_mode,
        limit_price=unit_price if price_mode == "LIMIT" else None,
        strategy_name=intent.strategy_id,
        order_remark=intent.idempotency_key,
        metadata={
            "workflow_run_id": intent.workflow_run_id,
            "signal_date": intent.signal_date,
            "buy_date": intent.buy_date,
            "target_pct": intent.target_pct,
        },
    )
    return PreparedOrder(request=request, risk_decision=decision, sizing_notes=tuple(notes))


def submit_prepared_orders(
    *,
    prepared_orders: list[PreparedOrder],
    adapter: BrokerAdapter,
    ledger: ExecutionLedger,
    batch_id: str,
) -> list[BrokerOrderAck]:
    acknowledgements: list[BrokerOrderAck] = []
    for prepared in prepared_orders:
        request = prepared.request
        if ledger.has_submission(request.client_order_id):
            raise RuntimeError(f"duplicate client_order_id detected in ledger: {request.client_order_id}")
        ledger.append(
            event_type="order_prepared",
            batch_id=batch_id,
            client_order_id=request.client_order_id,
            payload=prepared.to_dict(),
        )
        ack = adapter.submit_order(request)
        ledger.append(
            event_type=_ack_event_type(ack),
            batch_id=batch_id,
            client_order_id=request.client_order_id,
            payload=ack.to_dict(),
        )
        acknowledgements.append(ack)
    return acknowledgements


def _ack_event_type(ack: BrokerOrderAck) -> str:
    if ack.state == "shadow_only":
        return "order_shadowed"
    if ack.state == "acknowledged":
        return "order_paper_acknowledged"
    if ack.submitted:
        return "order_submitted"
    return "order_submission_rejected"
