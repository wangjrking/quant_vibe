from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from research_v260_risk_event_overlay import (
    RULE,
    build_block_matrix,
    build_event_masks,
    compose_overlay_masks,
    entry_ranked_without_events,
    maintenance_buy_allowed,
    score_tiebreak_order,
    validate_frozen_contract,
)


class Fixed10RiskEventOverlayTests(unittest.TestCase):
    def test_only_severe_event_blocks_the_exact_stock_date(self) -> None:
        dates = np.asarray(["20250102", "20250103"])
        stocks = np.asarray(["000001.SZ", "600000.SH", "300001.SZ"])
        features = pd.DataFrame(
            [
                {"signal_date": "20250102", "stock_code": "000001.SZ", "shock_count_1d": 1, "high_shock_count_1d": 0, "high_shock_count_5d": 0, "alert_active_count": 0},
                {"signal_date": "20250102", "stock_code": "600000.SH", "shock_count_1d": 0, "high_shock_count_1d": 1, "high_shock_count_5d": 1, "alert_active_count": 0},
                {"signal_date": "20250103", "stock_code": "300001.SZ", "shock_count_1d": 0, "high_shock_count_1d": 0, "high_shock_count_5d": 0, "alert_active_count": 1},
            ]
        )
        block, audit = build_block_matrix(dates, stocks, features)
        self.assertEqual(int(block.sum()), 1)
        self.assertFalse(block[0, 0])
        self.assertTrue(block[0, 1])
        self.assertFalse(block[1, 2])
        self.assertEqual(audit["component_stock_dates"]["ordinary_shock_1d"], 1)
        self.assertEqual(audit["component_stock_dates"]["severe_shock_5d"], 1)
        self.assertEqual(audit["component_stock_dates"]["exchange_alert_active"], 1)

    def test_entry_filter_does_not_mutate_exit_ranking(self) -> None:
        ranked_exit = [0, 1, 2]
        entry_ranked = entry_ranked_without_events(
            ranked_exit, np.asarray([True, False, True])
        )
        self.assertEqual(entry_ranked, [1])
        self.assertEqual(ranked_exit, [0, 1, 2])

    def test_maintenance_helper_honors_only_the_explicit_mask(self) -> None:
        blocked = np.asarray([True, False])
        self.assertFalse(maintenance_buy_allowed(0, blocked))
        self.assertTrue(maintenance_buy_allowed(1, blocked))
        self.assertIn("never forces a sale", RULE["held_position_exit"])

    def test_shared_overlay_implements_all_three_frozen_rules(self) -> None:
        dates = np.asarray(["20250102"])
        stocks = np.asarray(["000001.SZ", "600000.SH", "300001.SZ"])
        features = pd.DataFrame([
            {"signal_date": "20250102", "stock_code": "000001.SZ", "shock_count_1d": 1, "high_shock_count_1d": 0, "high_shock_count_5d": 0, "alert_active_count": 0},
            {"signal_date": "20250102", "stock_code": "600000.SH", "shock_count_1d": 0, "high_shock_count_1d": 1, "high_shock_count_5d": 1, "alert_active_count": 0},
            {"signal_date": "20250102", "stock_code": "300001.SZ", "shock_count_1d": 0, "high_shock_count_1d": 0, "high_shock_count_5d": 0, "alert_active_count": 1},
        ])
        components, audit = build_event_masks(dates, stocks, features)
        entry, maintenance = compose_overlay_masks(components)
        self.assertEqual(entry.tolist(), [[True, True, True]])
        self.assertEqual(maintenance.tolist(), [[False, True, True]])
        self.assertEqual(audit["ordinary_1d_stock_dates"], 1)
        self.assertEqual(audit["severe_5d_stock_dates"], 1)
        self.assertEqual(audit["alert_active_stock_dates"], 1)

    def test_shared_overlay_rejects_invalid_counts_and_shape_drift(self) -> None:
        invalid = pd.DataFrame([
            {"signal_date": "20250102", "stock_code": "000001.SZ", "shock_count_1d": -1, "high_shock_count_5d": 0, "alert_active_count": 0},
        ])
        with self.assertRaises(ValueError):
            build_event_masks(
                np.asarray(["20250102"]), np.asarray(["000001.SZ"]), invalid
            )
        with self.assertRaises(ValueError):
            compose_overlay_masks({
                "ordinary_1d": np.zeros((1, 1), dtype=np.bool_),
                "severe_5d": np.zeros((2, 1), dtype=np.bool_),
                "alert_active": np.zeros((1, 1), dtype=np.bool_),
            })

    def test_shared_implementation_validates_frozen_contract(self) -> None:
        contract = {
            "diagnostic_coverage": {
                "source_event_min_dates": {
                    "stk_shock": "20260303",
                    "stk_high_shock": "20260209",
                    "stk_alert": "20260210",
                },
                "ordinary_first_visible_session": "20260304",
                "severe_first_visible_session": "20260210",
                "exchange_alert_first_visible_session": "20260211",
                "event_metric_start": "20260210",
                "precoverage_state": "unknown_not_zero",
            },
            "rules": {
                "ordinary_abnormal_volatility": {
                    "action": "observe_only",
                    "forced_exit": False,
                },
                "severe_abnormal_volatility": {
                    "action": "new_entry_score_tiebreak",
                    "event_window_sessions": 1,
                    "margin_source": "base_candidate.replacement_advantage",
                    "changes_eligibility": False,
                    "forced_exit": False,
                },
                "exchange_focus_security": {
                    "action": "observe_only",
                    "forced_exit": False,
                },
                "missing_or_unknown_event_state": "fail_closed_for_diagnostic_input",
            }
        }
        validate_frozen_contract(contract)
        contract["rules"]["severe_abnormal_volatility"]["event_window_sessions"] = 4
        with self.assertRaises(ValueError):
            validate_frozen_contract(contract)

    def test_duplicate_event_keys_fail_closed(self) -> None:
        duplicate = pd.DataFrame(
            [
                {"signal_date": "20250102", "stock_code": "000001.SZ", "shock_count_1d": 1, "high_shock_count_1d": 0, "high_shock_count_5d": 0, "alert_active_count": 0},
                {"signal_date": "20250102", "stock_code": "000001.SZ", "shock_count_1d": 0, "high_shock_count_1d": 1, "high_shock_count_5d": 1, "alert_active_count": 0},
            ]
        )
        with self.assertRaises(ValueError):
            build_block_matrix(
                np.asarray(["20250102"]), np.asarray(["000001.SZ"]), duplicate
            )

    def test_severe_event_uses_score_tiebreak_without_changing_eligibility(self) -> None:
        score = np.asarray([[0.95, 0.91, 0.89]])
        order = np.asarray([[0, 1, 2]])
        components = {"severe_1d": np.asarray([[True, False, False]])}
        actual = score_tiebreak_order(score, order, components, 0.05)
        self.assertEqual(actual.tolist(), [[1, 0, 2]])
        self.assertEqual(sorted(actual[0].tolist()), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
