import sys
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.broker_adapter import (
    BrokerOrderRequest,
    PaperBrokerAdapter,
    ShadowBrokerAdapter,
)
from execution_gateway.contracts import OrderIntent


def _intent() -> OrderIntent:
    return OrderIntent(
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


class ExecutionGatewayBrokerAdapterTests(unittest.TestCase):
    def test_shadow_adapter_keeps_order_non_submittable(self):
        adapter = ShadowBrokerAdapter()
        request = BrokerOrderRequest(
            intent=_intent(),
            client_order_id="client_1",
            quantity=100,
            price_mode="LIMIT",
            limit_price=20.5,
        )
        ack = adapter.submit_order(request)
        self.assertFalse(ack.submitted)
        self.assertEqual(ack.state, "shadow_only")

    def test_paper_adapter_acknowledges_and_can_query_orders(self):
        adapter = PaperBrokerAdapter()
        request = BrokerOrderRequest(
            intent=_intent(),
            client_order_id="client_1",
            quantity=100,
            price_mode="LIMIT",
            limit_price=20.5,
        )
        ack = adapter.submit_order(request)
        self.assertTrue(ack.submitted)
        self.assertEqual(ack.state, "acknowledged")

        orders = adapter.query_orders("acct")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].client_order_id, "client_1")


if __name__ == "__main__":
    unittest.main()
