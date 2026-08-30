import json
import sys
import tempfile
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.policy import compute_policy_hash, load_execution_authorization


class ExecutionGatewayPolicyTests(unittest.TestCase):
    def test_load_execution_authorization_fills_missing_policy_hash(self):
        payload = {
            "authorization_id": "auth_1",
            "enabled": False,
            "stage": "paper",
            "account_id": "acct",
            "strategy_id": "prod_v260",
            "strategy_version": "v1",
            "owner_approval_id": "approval_1",
            "expires_at": "2026-12-31T16:00:00+08:00",
            "allowed_sides": ["BUY", "SELL"],
            "allowed_symbols": [],
            "max_order_notional": 10000,
            "max_symbol_position_pct": 0.1,
            "max_total_position_pct": 0.5,
            "max_orders_per_batch": 5,
            "quote_max_age_seconds": 10,
            "kill_switch_required": True,
            "policy_hash": "",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            authorization = load_execution_authorization(path)

        self.assertEqual(authorization.authorization_id, "auth_1")
        self.assertEqual(authorization.policy_hash, compute_policy_hash(payload))

    def test_load_execution_authorization_rejects_hash_drift(self):
        payload = {
            "authorization_id": "auth_1",
            "enabled": False,
            "stage": "paper",
            "account_id": "acct",
            "strategy_id": "prod_v260",
            "strategy_version": "v1",
            "owner_approval_id": "approval_1",
            "expires_at": "2026-12-31T16:00:00+08:00",
            "allowed_sides": ["BUY"],
            "allowed_symbols": [],
            "max_order_notional": 10000,
            "max_symbol_position_pct": 0.1,
            "max_total_position_pct": 0.5,
            "max_orders_per_batch": 5,
            "quote_max_age_seconds": 10,
            "kill_switch_required": True,
            "policy_hash": "BADHASH",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "policy_hash mismatch"):
                load_execution_authorization(path)

    def test_load_execution_authorization_accepts_utf8_sig_json(self):
        payload = {
            "authorization_id": "auth_1",
            "enabled": False,
            "stage": "paper",
            "account_id": "acct",
            "strategy_id": "prod_v260",
            "strategy_version": "v1",
            "owner_approval_id": "approval_1",
            "expires_at": "2026-12-31T16:00:00+08:00",
            "allowed_sides": ["BUY", "SELL"],
            "allowed_symbols": [],
            "max_order_notional": 10000,
            "max_symbol_position_pct": 0.1,
            "max_total_position_pct": 0.5,
            "max_orders_per_batch": 5,
            "quote_max_age_seconds": 10,
            "kill_switch_required": True,
            "policy_hash": "",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
            authorization = load_execution_authorization(path)

        self.assertEqual(authorization.account_id, "acct")


if __name__ == "__main__":
    unittest.main()
