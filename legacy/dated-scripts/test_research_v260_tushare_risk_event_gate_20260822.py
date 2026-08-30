from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_v260_tushare_risk_event_gate_20260822 as module


class TushareRiskEventStrategyTests(unittest.TestCase):
    def test_fixed_gate_uses_exact_date_and_stock_keys(self):
        dates = np.asarray(["20260820", "20260821"])
        stocks = np.asarray(["000001.SZ", "600000.SH"])
        features = pd.DataFrame(
            [
                {"signal_date": "20260820", "stock_code": "000001.SZ", "shock_count_1d": 1, "high_shock_count_5d": 0, "alert_active_count": 0},
                {"signal_date": "20260820", "stock_code": "999999.SZ", "shock_count_1d": 1, "high_shock_count_5d": 0, "alert_active_count": 0},
                {"signal_date": "20260822", "stock_code": "600000.SH", "shock_count_1d": 0, "high_shock_count_5d": 1, "alert_active_count": 0},
            ]
        )
        gate, audit = module.fixed_gate_matrix(dates, stocks, features)
        self.assertTrue(gate[0, 0])
        self.assertEqual(int(gate.sum()), 1)
        self.assertEqual(audit["blocked_stock_dates"], 1)

    def test_all_three_components_can_block(self):
        dates = np.asarray(["20260820"])
        stocks = np.asarray(["000001.SZ", "000002.SZ", "000003.SZ"])
        features = pd.DataFrame(
            [
                {"signal_date": "20260820", "stock_code": "000001.SZ", "shock_count_1d": 1, "high_shock_count_5d": 0, "alert_active_count": 0},
                {"signal_date": "20260820", "stock_code": "000002.SZ", "shock_count_1d": 0, "high_shock_count_5d": 1, "alert_active_count": 0},
                {"signal_date": "20260820", "stock_code": "000003.SZ", "shock_count_1d": 0, "high_shock_count_5d": 0, "alert_active_count": 1},
            ]
        )
        gate, audit = module.fixed_gate_matrix(dates, stocks, features)
        self.assertEqual(int(gate.sum()), 3)
        self.assertEqual(audit["component_stock_dates"]["ordinary_shock_1d"], 1)
        self.assertEqual(audit["component_stock_dates"]["severe_shock_5d"], 1)
        self.assertEqual(audit["component_stock_dates"]["exchange_alert_active"], 1)

    def test_missing_feature_contract_fails_closed(self):
        with self.assertRaises(ValueError):
            module.fixed_gate_matrix(
                np.asarray(["20260820"]),
                np.asarray(["000001.SZ"]),
                pd.DataFrame([{"signal_date": "20260820", "stock_code": "000001.SZ"}]),
            )


if __name__ == "__main__":
    unittest.main()
