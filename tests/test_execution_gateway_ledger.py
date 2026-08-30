import sys
import tempfile
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.ledger import ExecutionLedger


class ExecutionGatewayLedgerTests(unittest.TestCase):
    def test_append_and_read_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = ExecutionLedger(Path(tmpdir) / "ledger.jsonl")
            ledger.append(
                event_type="order_prepared",
                batch_id="wf_1",
                client_order_id="client_1",
                payload={"symbol": "603155.SH"},
            )
            ledger.append(
                event_type="order_paper_acknowledged",
                batch_id="wf_1",
                client_order_id="client_1",
                payload={"broker_order_id": "PAPER-1"},
            )
            events = ledger.read_events()

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "order_prepared")
        self.assertEqual(events[1].event_type, "order_paper_acknowledged")

    def test_has_submission_detects_written_ack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = ExecutionLedger(Path(tmpdir) / "ledger.jsonl")
            ledger.append(
                event_type="order_shadowed",
                batch_id="wf_1",
                client_order_id="client_1",
                payload={},
            )
            self.assertTrue(ledger.has_submission("client_1"))
            self.assertFalse(ledger.has_submission("missing"))


if __name__ == "__main__":
    unittest.main()
