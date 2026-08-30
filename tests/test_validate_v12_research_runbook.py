import tempfile
import unittest
from pathlib import Path

from validate_v12_research_runbook import validate_runbook


class ValidateV12ResearchRunbookTest(unittest.TestCase):
    def test_valid_runbook_matches_job_specs_and_required_entrypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entrypoint = root / "run_parallel_expanding2010_folds.py"
            entrypoint.write_text("print('ok')\n", encoding="utf-8")
            formula_entrypoint = root / "validate_formula_score_asset.py"
            formula_entrypoint.write_text("print('ok')\n", encoding="utf-8")
            runbook = {
                "approval_status": "research_plan_not_approved_for_execution",
                "standard_chain": {
                    "feature_input": str(root / "production_factor_parts"),
                    "label_input": str(root / "prediction_label_parts"),
                    "prediction_asset_root": str(root / "production_assets" / "duckdb"),
                },
                "entrypoints": {
                    "parallel_folds": str(entrypoint),
                    "formula_asset_validation": str(formula_entrypoint),
                },
                "v12_execution_order": [
                    {
                        "horizon": "3d",
                        "label": "executable_3d_open_return",
                        "job_type": "retrain_or_feature_search",
                        "failed_constraint": "full_daily_rank_ic_delta_floor",
                        "requires_training_authorization": True,
                        "research_table_template": "stock_predict_data_v12_3d_research",
                    },
                    {
                        "horizon": "10d",
                        "label": "executable_10d_open_return",
                        "job_type": "audit_and_strategy_research_validation",
                        "failed_constraint": "",
                        "requires_training_authorization": False,
                        "formula_asset_validation_required": True,
                    },
                ],
                "required_artifacts": [
                    "models/model_foldXX.json",
                    "models/model_foldXX_metadata.json",
                    "prediction_manifest.json",
                ],
                "boundaries": {
                    "no_training": True,
                    "no_prediction": True,
                    "no_production_manifest_change": True,
                    "no_signal": True,
                    "no_backtest": True,
                },
            }
            job_specs = {
                "job_specs": [
                    {
                        "horizon": "3d",
                        "label": "executable_3d_open_return",
                        "job_type": "retrain_or_feature_search",
                        "failed_constraints": "full_daily_rank_ic_delta_floor",
                        "requires_training_authorization": True,
                    },
                    {
                        "horizon": "10d",
                        "label": "executable_10d_open_return",
                        "job_type": "audit_and_strategy_research_validation",
                        "failed_constraints": "",
                        "requires_training_authorization": False,
                    },
                ]
            }

            result = validate_runbook(runbook, job_specs)

            self.assertTrue(result["ok"])
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["checked_horizons"], ["3d", "10d"])

    def test_validation_only_10d_requires_formula_asset_validation_entrypoint(self):
        runbook = {
            "approval_status": "research_plan_not_approved_for_execution",
            "standard_chain": {
                "feature_input": "quant/data_file/production_factor_parts/",
                "label_input": "quant/data_file/prediction_label_parts/",
                "prediction_asset_root": "quant/data_file/production_assets/duckdb/",
            },
            "entrypoints": {},
            "v12_execution_order": [
                {
                    "horizon": "10d",
                    "label": "executable_10d_open_return",
                    "job_type": "audit_and_strategy_research_validation",
                    "failed_constraint": "",
                    "requires_training_authorization": False,
                }
            ],
            "required_artifacts": [
                "models/model_foldXX.json",
                "models/model_foldXX_metadata.json",
                "prediction_manifest.json",
            ],
            "boundaries": {
                "no_training": True,
                "no_prediction": True,
                "no_production_manifest_change": True,
                "no_signal": True,
                "no_backtest": True,
            },
        }
        job_specs = {
            "job_specs": [
                {
                    "horizon": "10d",
                    "label": "executable_10d_open_return",
                    "job_type": "audit_and_strategy_research_validation",
                    "failed_constraints": "",
                    "requires_training_authorization": False,
                }
            ]
        }

        result = validate_runbook(runbook, job_specs)

        self.assertFalse(result["ok"])
        self.assertIn("10d validation-only item must require formula asset validation", result["errors"])
        self.assertIn("entrypoint formula_asset_validation is required for validation-only 10d", result["errors"])

    def test_research_training_item_rejects_non_research_table_template(self):
        runbook = {
            "approval_status": "research_plan_not_approved_for_execution",
            "standard_chain": {
                "feature_input": "quant/data_file/production_factor_parts/",
                "label_input": "quant/data_file/prediction_label_parts/",
                "prediction_asset_root": "quant/data_file/production_assets/duckdb/",
            },
            "entrypoints": {},
            "v12_execution_order": [
                {
                    "horizon": "5d",
                    "label": "executable_5d_open_return",
                    "job_type": "topn_constrained_retraining",
                    "failed_constraint": "full_top10_delta_floor",
                    "requires_training_authorization": True,
                    "research_table_template": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
                }
            ],
            "required_artifacts": ["models/model_foldXX.json"],
            "boundaries": {
                "no_training": True,
                "no_prediction": True,
                "no_production_manifest_change": True,
                "no_signal": True,
                "no_backtest": True,
            },
        }
        job_specs = {
            "job_specs": [
                {
                    "horizon": "5d",
                    "label": "executable_5d_open_return",
                    "job_type": "topn_constrained_retraining",
                    "failed_constraints": "full_top10_delta_floor",
                    "requires_training_authorization": True,
                }
            ]
        }

        result = validate_runbook(runbook, job_specs)

        self.assertFalse(result["ok"])
        self.assertIn("5d research_table_template must contain _research", result["errors"])


if __name__ == "__main__":
    unittest.main()
