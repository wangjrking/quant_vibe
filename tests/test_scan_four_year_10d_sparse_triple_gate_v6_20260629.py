from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "quant" / "main"))

import scan_four_year_10d_sparse_triple_gate_v6_20260629 as target


class SparseTripleGateV6Tests(unittest.TestCase):
    def test_choose_decision_requires_hard_pass(self):
        self.assertEqual(
            target.choose_decision(0),
            "continue_research_no_sparse_triple_hard_pass",
        )
        self.assertEqual(
            target.choose_decision(1),
            "sparse_triple_gate_v6_has_hard_pass_candidate_needs_review",
        )


if __name__ == "__main__":
    unittest.main()
