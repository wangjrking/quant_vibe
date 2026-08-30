import sys
import tempfile
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.kill_switch import KillSwitchState, load_kill_switch_state, save_kill_switch_state


class ExecutionGatewayKillSwitchTests(unittest.TestCase):
    def test_missing_file_defaults_to_unblocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = load_kill_switch_state(Path(tmpdir) / "missing.json")
        self.assertFalse(state.global_blocked)
        self.assertEqual(state.blockers_for("acct", "strategy", "000001.SZ"), [])

    def test_blockers_for_scope_are_reported(self):
        state = KillSwitchState(
            blocked_accounts={"acct": "owner_pause"},
            blocked_strategies={"strategy": "audit_hold"},
            blocked_symbols={"000001.SZ": "manual_blacklist"},
        )
        blockers = state.blockers_for("acct", "strategy", "000001.SZ")
        self.assertEqual(
            blockers,
            [
                "kill_switch_account:owner_pause",
                "kill_switch_strategy:audit_hold",
                "kill_switch_symbol:manual_blacklist",
            ],
        )

    def test_save_and_reload_preserves_state(self):
        state = KillSwitchState(global_blocked=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kill_switch.json"
            save_kill_switch_state(path, state)
            restored = load_kill_switch_state(path)
        self.assertTrue(restored.global_blocked)


if __name__ == "__main__":
    unittest.main()
