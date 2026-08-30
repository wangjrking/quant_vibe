import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.contracts import ExecutionAuthorization
from execution_gateway.live_arm import (
    LiveArmState,
    load_live_arm_state,
    save_live_arm_state,
    validate_live_arm_state,
)


def _authorization() -> ExecutionAuthorization:
    return ExecutionAuthorization(
        authorization_id="auth_1",
        account_id="acct",
        stage="live",
        strategy_id="prod_v260",
        strategy_version="v1",
        owner_approval_id="approval_1",
        policy_hash="HASH",
        enabled=True,
        expires_at="2099-12-31T16:00:00+08:00",
        allowed_sides=("BUY", "SELL"),
        allowed_symbols=(),
        max_order_notional=100000.0,
        max_symbol_position_pct=0.1,
        max_total_position_pct=0.6,
        max_orders_per_batch=10,
        quote_max_age_seconds=10,
        kill_switch_required=True,
    )


class ExecutionGatewayLiveArmTests(unittest.TestCase):
    def test_missing_file_defaults_to_disabled_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = load_live_arm_state(Path(tmpdir) / "missing.json")
        self.assertFalse(state.enabled)

    def test_valid_live_arm_passes_validation(self):
        state = LiveArmState(
            enabled=True,
            arm_id="arm_1",
            owner_approval_id="approval_1",
            account_id="acct",
            strategy_id="prod_v260",
            stage="live",
            armed_by="owner",
            reason="open small live window",
            created_at="2026-07-30T09:25:00+08:00",
            expires_at="2026-07-30T15:05:00+08:00",
        )
        valid, blockers = validate_live_arm_state(
            arm_state=state,
            authorization=_authorization(),
            now=datetime(2026, 7, 30, 2, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(valid)
        self.assertEqual(blockers, ())

    def test_invalid_live_arm_reports_specific_blockers(self):
        state = LiveArmState(
            enabled=False,
            arm_id="",
            owner_approval_id="wrong",
            account_id="other",
            strategy_id="other_strategy",
            stage="paper",
            created_at="bad-ts",
            expires_at="2026-07-29T15:05:00+08:00",
        )
        valid, blockers = validate_live_arm_state(
            arm_state=state,
            authorization=_authorization(),
            now=datetime(2026, 7, 30, 2, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(valid)
        self.assertIn("live_arm_disabled", blockers)
        self.assertIn("live_arm_owner_approval_mismatch", blockers)
        self.assertIn("live_arm_expired", blockers)

    def test_save_and_reload_round_trips(self):
        state = LiveArmState(
            enabled=True,
            arm_id="arm_1",
            owner_approval_id="approval_1",
            account_id="acct",
            strategy_id="prod_v260",
            stage="live",
            armed_by="owner",
            reason="window",
            created_at="2026-07-30T09:25:00+08:00",
            expires_at="2026-07-30T15:05:00+08:00",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "live_arm.json"
            save_live_arm_state(path, state)
            restored = load_live_arm_state(path)
        self.assertEqual(restored.arm_id, "arm_1")
        self.assertEqual(restored.strategy_id, "prod_v260")


if __name__ == "__main__":
    unittest.main()
