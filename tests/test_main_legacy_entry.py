import unittest
from pathlib import Path


class MainLegacyEntryTests(unittest.TestCase):
    def test_main_py_requires_explicit_legacy_opt_in(self):
        source = Path(__file__).resolve().parents[1] / "main.py"
        text = source.read_text(encoding="utf-8")

        self.assertIn("QUANT_ALLOW_LEGACY_MAIN", text)
        self.assertIn("legacy / historical mixed-db", text)
        self.assertIn("STOCK_DAILY_DATA.db::STOCK_DAILY_DATA", text)
        self.assertIn("raise SystemExit(2)", text)


if __name__ == "__main__":
    unittest.main()
