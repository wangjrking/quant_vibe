import sys
import tempfile
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.state_store import ExecutionStateStore


class ExecutionGatewayStateStoreTests(unittest.TestCase):
    def test_upsert_batch_and_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ExecutionStateStore(Path(tmpdir) / "state.sqlite3")
            store.upsert_batch(
                batch_id="wf_1",
                mode="paper",
                workflow_run_id="wf_1",
                account_id="acct",
                strategy_id="prod_v260",
                strategy_version="v1",
                signal_date="20260730",
                buy_date="20260731",
                batch_status="approved",
                intent_count=1,
                approved_count=1,
                created_at="2026-07-30T10:00:00+08:00",
                payload={"hello": "world"},
            )
            summary = store.batch_summary("wf_1")
            counts = store.counts()

        self.assertEqual(summary, {"hello": "world"})
        self.assertEqual(counts["execution_batches"], 1)

    def test_upsert_prepared_order_and_ack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ExecutionStateStore(Path(tmpdir) / "state.sqlite3")
            store.upsert_batch(
                batch_id="wf_1",
                mode="paper",
                workflow_run_id="wf_1",
                account_id="acct",
                strategy_id="prod_v260",
                strategy_version="v1",
                signal_date="20260730",
                buy_date="20260731",
                batch_status="approved",
                intent_count=1,
                approved_count=1,
                created_at="2026-07-30T10:00:00+08:00",
                payload={},
            )
            store.upsert_prepared_order(
                client_order_id="client_1",
                batch_id="wf_1",
                symbol="603155.SH",
                side="BUY",
                quantity=100,
                price_mode="LIMIT",
                limit_price=20.0,
                created_at="2026-07-30T10:00:01+08:00",
                request_payload={"quantity": 100},
            )
            store.upsert_broker_ack(
                client_order_id="client_1",
                broker_order_id="PAPER-1",
                state="acknowledged",
                submitted=True,
                acknowledged_at="2026-07-30T10:00:02+08:00",
                ack_payload={"state": "acknowledged"},
            )
            counts = store.counts()

        self.assertEqual(counts["prepared_orders"], 1)
        self.assertEqual(counts["broker_acknowledgements"], 1)


if __name__ == "__main__":
    unittest.main()
