import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.scan_l5_l6_l7_write_bypass import writes_current_scope_assets


class ScanL5L6L7WriteBypassTests(unittest.TestCase):
    def test_detects_production_signal_write_scope(self):
        text = """
signal_dir = data_dir / "production_signals"
output_path = signal_dir / "prod_a_latest.csv"
output_path.write_text("ok", encoding="utf-8")
"""
        self.assertTrue(writes_current_scope_assets(text))

    def test_detects_strategy_library_write_scope(self):
        text = """
strategy_dir = project_dir / "strategy_library" / "production" / "prod_a"
(strategy_dir / "validation.json").write_text("{}", encoding="utf-8")
"""
        self.assertTrue(writes_current_scope_assets(text))

    def test_ignores_generic_json_write_outside_current_scope(self):
        text = """
report_path = output_dir / "summary.json"
report_path.write_text("{}", encoding="utf-8")
"""
        self.assertFalse(writes_current_scope_assets(text))


if __name__ == "__main__":
    unittest.main()
