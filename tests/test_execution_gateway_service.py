import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from execution_gateway.contracts import AccountSnapshot, MarketSnapshot
from execution_gateway.kill_switch import KillSwitchState
from execution_gateway.policy import load_execution_authorization
from execution_gateway.service import ExecutionGatewayService


class ExecutionGatewayServiceTests(unittest.TestCase):
    def _write_policy(self, root: Path, *, stage: str = "paper", enabled: bool = True) -> Path:
        payload = {
            "authorization_id": "auth_1",
            "enabled": enabled,
            "stage": stage,
            "account_id": "acct",
            "strategy_id": "prod_v260",
            "strategy_version": "v1",
            "owner_approval_id": "approval_1",
            "expires_at": "2099-12-31T16:00:00+08:00",
            "allowed_sides": ["BUY", "SELL"],
            "allowed_symbols": [],
            "max_order_notional": 50000,
            "max_symbol_position_pct": 0.1,
            "max_total_position_pct": 0.6,
            "max_orders_per_batch": 5,
            "quote_max_age_seconds": 999999999,
            "kill_switch_required": True,
            "policy_hash": "",
        }
        path = root / "policy.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _write_latest_files(self, root: Path, *, no_signal: bool = False) -> tuple[Path, Path]:
        csv_path = root / "latest.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "strategy_id",
                    "signal_date",
                    "buy_date",
                    "action",
                    "stock_code",
                    "name",
                    "market",
                    "strategy_score",
                    "score_rank",
                    "score_denominator",
                    "target_pct",
                    "reason",
                    "execution_open_raw",
                    "status",
                ],
            )
            writer.writeheader()
            if not no_signal:
                writer.writerow(
                    {
                        "strategy_id": "prod_v260",
                        "signal_date": "20260729",
                        "buy_date": "20260730",
                        "action": "BUY",
                        "stock_code": "603155.SH",
                        "name": "新亚强",
                        "market": "SH",
                        "strategy_score": "0.99",
                        "score_rank": "1",
                        "score_denominator": "5194",
                        "target_pct": "0.055",
                        "reason": "top ranked",
                        "execution_open_raw": "",
                        "status": "pending_buy_day_hard_gate",
                    }
                )

        status_path = root / "latest_status.json"
        status_path.write_text(
            json.dumps(
                {
                    "strategy_id": "prod_v260",
                    "signal_date": "20260729",
                    "buy_date": "20260730",
                    "status": "pending_buy_day_hard_gate",
                    "signal_semantics": "no_signal_hold_only" if no_signal else "trade_actions_pending_buy_day_hard_gate",
                    "no_signal": no_signal,
                    "hold_only": no_signal,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return csv_path, status_path

    def test_no_signal_batch_returns_structured_no_signal_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy = load_execution_authorization(self._write_policy(root, enabled=False))
            csv_path, status_path = self._write_latest_files(root, no_signal=True)
            service = ExecutionGatewayService(authorization=policy, kill_switch_state=KillSwitchState())
            result = service.evaluate_latest_batch(
                latest_csv_path=str(csv_path),
                latest_status_path=str(status_path),
                workflow_run_id="wf_1",
                account=AccountSnapshot(account_id="acct", cash=100000.0, total_asset=100000.0, positions={}),
                market_by_symbol={},
            )

        self.assertEqual(result["batch_status"], "no_signal_hold_only")
        self.assertEqual(result["intent_count"], 0)

    def test_paper_batch_approves_when_risk_checks_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy = load_execution_authorization(self._write_policy(root, enabled=True))
            csv_path, status_path = self._write_latest_files(root, no_signal=False)
            service = ExecutionGatewayService(authorization=policy, kill_switch_state=KillSwitchState())
            result = service.evaluate_latest_batch(
                latest_csv_path=str(csv_path),
                latest_status_path=str(status_path),
                workflow_run_id="wf_1",
                account=AccountSnapshot(account_id="acct", cash=100000.0, total_asset=100000.0, positions={}),
                market_by_symbol={
                    "603155.SH": MarketSnapshot(
                        symbol="603155.SH",
                        quote_ts="2026-07-30T09:31:00+08:00",
                        last_price=20.0,
                        ask_price_1=20.1,
                        bid_price_1=20.0,
                        stock_status="NORMAL",
                        is_st=False,
                    )
                },
            )

        self.assertEqual(result["batch_status"], "approved")
        self.assertEqual(result["approved_count"], 1)

    def test_shadow_batch_stays_non_submittable_even_when_checks_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy = load_execution_authorization(self._write_policy(root, enabled=True, stage="shadow"))
            csv_path, status_path = self._write_latest_files(root, no_signal=False)
            service = ExecutionGatewayService(authorization=policy, kill_switch_state=KillSwitchState())
            result = service.evaluate_latest_batch(
                latest_csv_path=str(csv_path),
                latest_status_path=str(status_path),
                workflow_run_id="wf_1",
                account=AccountSnapshot(account_id="acct", cash=100000.0, total_asset=100000.0, positions={}),
                market_by_symbol={
                    "603155.SH": MarketSnapshot(
                        symbol="603155.SH",
                        quote_ts="2026-07-30T09:31:00+08:00",
                        last_price=20.0,
                        ask_price_1=20.1,
                        bid_price_1=20.0,
                        stock_status="NORMAL",
                        is_st=False,
                    )
                },
            )

        self.assertEqual(result["batch_status"], "shadow_only")
        self.assertEqual(result["decisions"][0]["status"], "shadow_only")

    def test_batch_is_rejected_when_kill_switch_blocks_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy = load_execution_authorization(self._write_policy(root, enabled=True))
            csv_path, status_path = self._write_latest_files(root, no_signal=False)
            service = ExecutionGatewayService(
                authorization=policy,
                kill_switch_state=KillSwitchState(blocked_symbols={"603155.SH": "manual_pause"}),
            )
            result = service.evaluate_latest_batch(
                latest_csv_path=str(csv_path),
                latest_status_path=str(status_path),
                workflow_run_id="wf_1",
                account=AccountSnapshot(account_id="acct", cash=100000.0, total_asset=100000.0, positions={}),
                market_by_symbol={
                    "603155.SH": MarketSnapshot(
                        symbol="603155.SH",
                        quote_ts="2026-07-30T09:31:00+08:00",
                        last_price=20.0,
                        ask_price_1=20.1,
                        bid_price_1=20.0,
                        stock_status="NORMAL",
                        is_st=False,
                    )
                },
            )

        self.assertEqual(result["batch_status"], "risk_rejected")
        self.assertIn("kill_switch_symbol:manual_pause", result["decisions"][0]["blockers"])


if __name__ == "__main__":
    unittest.main()
