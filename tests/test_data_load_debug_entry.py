import unittest
from pathlib import Path


class DataLoadDebugEntryTests(unittest.TestCase):
    def test_debug_entry_does_not_write_legacy_odb_test_table(self):
        source = Path(__file__).resolve().parents[1] / "data_load_module.py"
        text = source.read_text(encoding="utf-8")
        main_block = text.split("if __name__ == '__main__':", 1)[1]

        self.assertNotIn("adj_factor_test", main_block)
        self.assertNotIn("odb.db", main_block)


if __name__ == "__main__":
    unittest.main()
