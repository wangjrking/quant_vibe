import json
import sys
import unittest
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from tools.select_agent_model import DEFAULT_POLICY, select_model


class SelectAgentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))

    def test_complex_critical_prompt_uses_project_fixed_terra_low(self):
        result = select_model(
            self.policy,
            complexity=5,
            speed="normal",
            risk="critical",
            remaining_tokens=60000,
            token_budget=100000,
            remaining_seconds=6000,
            time_budget_seconds=10000,
        )
        self.assertEqual(result["selected_model"], "gpt-5.6-terra")
        self.assertEqual(result["thinking"], "low")
        self.assertEqual(result["status"], "selected_fixed_dispatch")

    def test_fixed_dispatch_ignores_budget_bands(self):
        result = select_model(
            self.policy,
            complexity=5,
            speed="immediate",
            risk="critical",
            remaining_tokens=20000,
            token_budget=100000,
            remaining_seconds=2000,
            time_budget_seconds=10000,
        )
        self.assertEqual(result["decision"], "dispatch")
        self.assertEqual(result["selected_model"], "gpt-5.6-terra")
        self.assertEqual(result["thinking"], "low")

    def test_standard_development_uses_terra(self):
        result = select_model(
            self.policy,
            complexity=3,
            speed="normal",
            risk="normal",
            remaining_tokens=40000,
            token_budget=100000,
            remaining_seconds=4000,
            time_budget_seconds=10000,
        )
        self.assertEqual(result["selected_model"], "gpt-5.6-terra")
        self.assertEqual(result["thinking"], "low")

    def test_fast_low_risk_prompt_uses_project_fixed_terra(self):
        result = select_model(
            self.policy,
            complexity=2,
            speed="immediate",
            risk="low",
            remaining_tokens=50000,
            token_budget=100000,
            remaining_seconds=5000,
            time_budget_seconds=10000,
        )
        self.assertEqual(result["selected_model"], "gpt-5.6-terra")
        self.assertEqual(result["thinking"], "low")

    def test_low_resource_normal_prompt_uses_project_fixed_terra(self):
        result = select_model(
            self.policy,
            complexity=3,
            speed="fast",
            risk="normal",
            remaining_tokens=15000,
            token_budget=100000,
            remaining_seconds=1500,
            time_budget_seconds=10000,
        )
        self.assertEqual(result["selected_model"], "gpt-5.6-terra")

    def test_less_than_ten_percent_uses_project_fixed_terra(self):
        result = select_model(
            self.policy,
            complexity=1,
            speed="immediate",
            risk="low",
            remaining_tokens=900,
            token_budget=10000,
            remaining_seconds=90,
            time_budget_seconds=1000,
        )
        self.assertEqual(result["decision"], "dispatch")
        self.assertEqual(result["selected_model"], "gpt-5.6-terra")

    def test_unknown_budget_uses_project_fixed_terra_and_reports_unknown(self):
        result = select_model(
            self.policy,
            complexity=4,
            speed="deep",
            risk="high",
        )
        self.assertEqual(result["selected_model"], "gpt-5.6-terra")
        self.assertEqual(result["thinking"], "low")
        self.assertEqual(result["budget"]["unknown_dimensions"], ["tokens", "time"])

    def test_non_terra_policy_is_rejected_before_dispatch(self):
        policy = json.loads(json.dumps(self.policy))
        policy["models"]["luna"] = {
            "model": "gpt-5.6-luna",
            "allowed_thinking": ["low"],
        }
        with self.assertRaisesRegex(ValueError, "terra_default_policy"):
            select_model(policy, complexity=1, speed="normal", risk="low")


if __name__ == "__main__":
    unittest.main()
