from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .broker_adapter import BrokerOrderUpdate
from .executor import PreparedOrder
from .ledger import LedgerEvent


@dataclass(frozen=True)
class ReconciliationSummary:
    expected_client_order_ids: tuple[str, ...]
    ledger_client_order_ids: tuple[str, ...]
    broker_client_order_ids: tuple[str, ...]
    missing_in_ledger: tuple[str, ...]
    missing_at_broker: tuple[str, ...]
    unknown_broker_orders: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "expected_client_order_ids": list(self.expected_client_order_ids),
            "ledger_client_order_ids": list(self.ledger_client_order_ids),
            "broker_client_order_ids": list(self.broker_client_order_ids),
            "missing_in_ledger": list(self.missing_in_ledger),
            "missing_at_broker": list(self.missing_at_broker),
            "unknown_broker_orders": list(self.unknown_broker_orders),
        }


def reconcile_orders(
    *,
    prepared_orders: Iterable[PreparedOrder],
    ledger_events: Iterable[LedgerEvent],
    broker_orders: Iterable[BrokerOrderUpdate],
) -> ReconciliationSummary:
    ledger_event_list = list(ledger_events)
    expected_ids = tuple(item.request.client_order_id for item in prepared_orders)
    ledger_ids = tuple(
        event.client_order_id
        for event in ledger_event_list
        if event.client_order_id and event.event_type in {
            "order_prepared",
            "order_submitted",
            "order_shadowed",
            "order_paper_acknowledged",
            "order_submission_rejected",
        }
    )
    broker_ids = tuple(item.client_order_id for item in broker_orders if item.client_order_id)

    expected_set = set(expected_ids)
    ledger_set = set(ledger_ids)
    broker_set = set(broker_ids)
    shadow_ids = {
        event.client_order_id
        for event in ledger_event_list
        if event.client_order_id and event.event_type == "order_shadowed"
    }
    non_broker_expected = shadow_ids

    missing_in_ledger = tuple(sorted(expected_set - ledger_set))
    missing_at_broker = tuple(sorted((expected_set - broker_set) - non_broker_expected))
    unknown_broker_orders = tuple(sorted(broker_set - expected_set))

    return ReconciliationSummary(
        expected_client_order_ids=expected_ids,
        ledger_client_order_ids=ledger_ids,
        broker_client_order_ids=broker_ids,
        missing_in_ledger=missing_in_ledger,
        missing_at_broker=missing_at_broker,
        unknown_broker_orders=unknown_broker_orders,
    )
