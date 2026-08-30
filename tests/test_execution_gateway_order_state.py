import sys
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.order_state import assert_valid_transition, is_terminal_state


class ExecutionGatewayOrderStateTests(unittest.TestCase):
    def test_valid_transition_passes(self):
        assert_valid_transition("proposed", "approved")
        assert_valid_transition("acknowledged", "partial_filled")

    def test_invalid_transition_raises(self):
        with self.assertRaisesRegex(ValueError, "invalid order state transition"):
            assert_valid_transition("filled", "approved")

    def test_terminal_state_detection(self):
        self.assertTrue(is_terminal_state("filled"))
        self.assertFalse(is_terminal_state("approved"))


if __name__ == "__main__":
    unittest.main()
