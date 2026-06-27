import unittest
from pathlib import Path

from run_incremental_factor_update_chunked import (
    build_chunk_command,
    chunk_ranges,
    parse_args,
    should_run_new_stock_fill,
)


class RunIncrementalFactorUpdateChunkedTests(unittest.TestCase):
    def test_chunk_ranges_splits_contiguously(self):
        self.assertEqual(chunk_ranges(list(range(630, 636)), 2), [(630, 631), (632, 633), (634, 635)])

    def test_chunk_ranges_requires_positive_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_ranges([1, 2, 3], 0)

    def test_parse_args_defaults(self):
        args = parse_args(["--target-date", "20260622"])
        self.assertEqual(args.chunk_size, 130)
        self.assertEqual(args.workers, 4)

    def test_build_chunk_command_contains_part_bounds(self):
        args = parse_args(["--target-date", "20260622", "--python-executable", "python"])
        command = build_chunk_command(args, 630, 760, Path("report.json"))
        self.assertIn("--start-part", command)
        self.assertIn("--end-part", command)
        self.assertEqual(command[0], "python")
        self.assertIn("630", command)
        self.assertIn("760", command)

    def test_new_stock_fill_only_runs_for_full_range(self):
        self.assertTrue(should_run_new_stock_fill(None, None))
        self.assertFalse(should_run_new_stock_fill(0, None))
        self.assertFalse(should_run_new_stock_fill(None, 100))
        self.assertFalse(should_run_new_stock_fill(0, 100))


if __name__ == "__main__":
    unittest.main()
