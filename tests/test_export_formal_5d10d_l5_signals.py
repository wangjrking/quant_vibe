import unittest

from export_formal_5d10d_l5_signals import _is_delisting


class ExportFormal5d10dL5SignalsTests(unittest.TestCase):
    def test_delisting_filter_matches_suffix_retired_name(self):
        self.assertTrue(_is_delisting({"name": "\u5929\u9f99\u9000"}))


if __name__ == "__main__":
    unittest.main()
