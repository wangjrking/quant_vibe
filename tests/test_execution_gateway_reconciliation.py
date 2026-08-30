import sys
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.broker_adapter import BrokerOrderUpdate
from execution_gateway.contracts import OrderIntent, RiskDecision
from execution_gateway.executor import PreparedOrder
from execution_gateway.ledger import LedgerEvent
from execution_gateway.reconciliation import reconcile_orders


def _prepared_order(client_order_id: str) -> PreparedOrder:
    from execution_gateway.broker_adapter import BrokerOrderRequest

    intent = OrderIntent(
        workflow_run_id="wf_1",
        strategy_id="prod_v260",
        strategy_version="v1",
        signal_date="20260730",
        buy_date="20260731",
        account_id="acct",
        symbol="603155.SH",
        side="BUY",
        rebalance_version="latest",
        target_pct=0.05,
    )
    return PreparedOrder(
        request=BrokerOrderRequest(
            intent=intent,
            client_order_id=client_order_id,
            quantity=100,
            price_mode="LIMIT",
            limit_price=20.0,
        ),
        risk_decision=RiskDecision(intent=intent, status="approved", approved=True, blockers=()),
        sizing_notes=(),
    )


class ExecutionGatewayReconciliationTests(unittest.TestCase):
    def test_reconcile_orders_flags_missing_and_unknown_records(self):
        prepared = [_prepared_order("client_1"), _prepared_order("client_2")]
        ledger_events = [
            LedgerEvent(
                event_id="evt_1",
                event_type="order_prepared",
                created_at="2026-07-30T10:00:00+08:00",
                batch_id="wf_1",
                client_order_id="client_1",
                payload={},
            )
        ]
        broker_orders = [
            BrokerOrderUpdate(
                broker_order_id="broker_1",
                account_id="acct",
                symbol="603155.SH",
                side="BUY",
                state="acknowledged",
                client_order_id="client_1",
            ),
            BrokerOrderUpdate(
                broker_order_id="broker_2",
                account_id="acct",
                symbol="688486.SH",
                side="BUY",
                state="acknowledged",
                client_order_id="unknown_client",
            ),
        ]
        result = reconcile_orders(
            prepared_orders=prepared,
            ledger_events=ledger_events,
            broker_orders=broker_orders,
        )

        self.assertEqual(result.missing_in_ledger, ("client_2",))
        self.assertEqual(result.missing_at_broker, ("client_2",))
        self.assertEqual(result.unknown_broker_orders, ("unknown_client",))

    def test_shadowed_orders_are_not_treated_as_missing_at_broker(self):
        prepared = [_prepared_order("client_1")]
        ledger_events = [
            LedgerEvent(
                event_id="evt_1",
                event_type="order_shadowed",
                created_at="2026-07-30T10:00:00+08:00",
                batch_id="wf_1",
                client_order_id="client_1",
                payload={},
            )
        ]
        result = reconcile_orders(
            prepared_orders=prepared,
            ledger_events=ledger_events,
            broker_orders=[],
        )
        self.assertEqual(result.missing_at_broker, ())


if __name__ == "__main__":
    unittest.main()
