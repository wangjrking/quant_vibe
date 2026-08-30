import unittest

from feature_contract_v2_candidate import (
    FeatureContractV2Error,
    SEMANTIC_FUTURE_DERIVED_COLUMNS,
    audit_candidate_feature_schema,
    build_feature_contract_v2_candidate,
    candidate_production_raw_columns,
    validate_future_lineage,
)


class L3FeatureContractV2CandidateTests(unittest.TestCase):
    def _valid_schema(self):
        qfq_prices = ["open_qfq", "high_qfq", "low_qfq", "close_qfq", "pre_close_qfq"]
        qfq_technical = [f"technical_{index:03d}_qfq" for index in range(71)]
        qfq_technical.extend(["ema_qfq_10", "ma_qfq_20", "rsi_qfq_12", "macdsignal_qfq", "macdhist_qfq"])
        gtja_qfq = [f"gtja_alpha{index:03d}_qfq" for index in range(1, 192)]
        base = ["trade_date", "stock_code", "industry", "industry_encode", "cci"]
        base.extend(f"base_{index:03d}" for index in range(566 - len(base)))
        return [*base, *qfq_prices, *qfq_technical, *gtja_qfq]

    def test_candidate_raw_allowlist_excludes_known_future_field(self):
        columns = [
            "trade_date",
            "stock_code",
            "open_qfq",
            "high_qfq",
            "low_qfq",
            "close_qfq",
            "pre_close_qfq",
            "index_2000_post10_close",
            "cci",
        ]
        selected = candidate_production_raw_columns(columns)
        self.assertNotIn("index_2000_post10_close", selected)
        self.assertIn("cci", selected)

    def test_negative_shift_lineage_is_explicitly_classified(self):
        result = validate_future_lineage(
            {"index_2000_post10_close": "group_data['index_2000_close'].shift(-10)"}
        )
        self.assertEqual(result["negative_shift_fields"][0]["field"], "index_2000_post10_close")

    def test_unknown_negative_shift_fails_closed(self):
        with self.assertRaisesRegex(FeatureContractV2Error, "unregistered negative shift"):
            validate_future_lineage({"unknown_factor": "frame['close'].shift(-1)"})

    def test_v2_schema_has_expected_qfq_and_no_future_columns(self):
        report = audit_candidate_feature_schema(self._valid_schema())
        self.assertEqual(report["column_count"], 838)
        self.assertEqual(report["qfq_price_count"], 5)
        self.assertEqual(report["qfq_technical_count"], 76)
        self.assertEqual(report["l2_source_qfq_technical_count"], 74)
        self.assertEqual(report["production_feature_qfq_technical_count"], 76)
        self.assertEqual(report["gtja_qfq_count"], 191)
        self.assertEqual(report["future_or_label_columns"], [])

    def test_v2_schema_rejects_future_and_naked_aliases(self):
        columns = self._valid_schema()
        columns[-1] = "gtja_alpha191"
        columns.append("index_2000_post10_close")
        with self.assertRaisesRegex(FeatureContractV2Error, "naked_gtja_columns"):
            audit_candidate_feature_schema(columns)

    def test_candidate_contract_is_non_mutating_and_not_ready_for_next_layer(self):
        payload = build_feature_contract_v2_candidate(
            workflow_run_id="l3-feature-contract-v2-candidate-20260804",
            target_trade_date="20260804",
            base_active_feature={"sha256": "base"},
            candidate_feature={"sha256": "candidate", "column_count": 838},
            candidate_columns=self._valid_schema(),
            lineage={"index_2000_post10_close": "shift(-10)"},
            evidence_paths=["candidate.json"],
        )
        self.assertTrue(payload["ready_for_audit_review"])
        self.assertFalse(payload["allow_next_layer_continue"])
        self.assertFalse(payload["active_mutation"]["active_feature_modified"])
        self.assertEqual(payload["production_raw_allowlist"]["semantic_future_denylist"], sorted(SEMANTIC_FUTURE_DERIVED_COLUMNS))


if __name__ == "__main__":
    unittest.main()
