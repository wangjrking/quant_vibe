import sys
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.contracts import AccountPosition, AccountSnapshot, MarketSnapshot, OrderIntent, RiskDecision
from execution_gateway.executor import build_order_request


class ExecutionGatewayExecutorTests(unittest.TestCase):
    def test_buy_request_derives_board_lot_quantity(self):
        decision = RiskDecision(
            intent=OrderIntent(
                workflow_run_id="wf_1",
                strategy_id="prod_v260",
                strategy_version="v1",
                signal_date="20260730",
                buy_date="20260731",
                account_id="acct",
                symbol="603155.SH",
                side="BUY",
                rebalance_version="latest",
                target_pct=0.055,
            ),
            status="approved",
            approved=True,
            blockers=(),
        )
        prepared = build_order_request(
            decision=decision,
            account=AccountSnapshot(account_id="acct", cash=100000.0, total_asset=100000.0, positions={}),
            market=MarketSnapshot(
                symbol="603155.SH",
                quote_ts="2026-07-30T09:31:00+08:00",
                last_price=20.0,
                ask_price_1=20.0,
                stock_status="NORMAL",
                is_st=False,
            ),
        )
        self.assertEqual(prepared.request.quantity, 200)

    def test_sell_request_defaults_to_sellable_shares(self):
        decision = RiskDecision(
            intent=OrderIntent(
                workflow_run_id="wf_1",
                strategy_id="prod_v260",
                strategy_version="v1",
                signal_date="20260730",
                buy_date="20260731",
                account_id="acct",
                symbol="603155.SH",
                side="SELL",
                rebalance_version="latest",
            ),
            status="approved",
            approved=True,
            blockers=(),
        )
        prepared = build_order_request(
            decision=decision,
            account=AccountSnapshot(
                account_id="acct",
                cash=10000.0,
                total_asset=100000.0,
                positions={
                    "603155.SH": AccountPosition(
                        symbol="603155.SH",
                        shares=500,
                        sellable_shares=350,
                        market_value=8000.0,
                    )
                },
            ),
            market=MarketSnapshot(
                symbol="603155.SH",
                quote_ts="2026-07-30T09:31:00+08:00",
                last_price=20.0,
                bid_price_1=19.9,
                stock_status="NORMAL",
                is_st=False,
            ),
        )
        self.assertEqual(prepared.request.quantity, 350)


if __name__ == "__main__":
    unittest.main()
