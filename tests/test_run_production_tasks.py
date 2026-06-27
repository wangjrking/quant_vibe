import json
import tempfile
import unittest
from pathlib import Path

from run_production_tasks import production_strategy_ids, run_production_tasks


class RunProductionTasksTests(unittest.TestCase):
    def test_production_strategy_ids_only_returns_registered_production(self):
        registry = {
            "production": {
                "strategies": [
                    {"strategy_id": "prod_a", "status": "production"},
                    {"strategy_id": "exp_a", "status": "exploration"},
                    {"strategy_id": "missing_status"},
                ]
            }
        }

        self.assertEqual(production_strategy_ids(registry), {"prod_a"})

    def test_dry_run_executes_enabled_production_and_skips_others(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_path = root / "registry.json"
            config_path = root / "tasks.json"
            manifest_path = root / "prediction_manifest.json"
            log_dir = root / "logs"
            signal_dir = root / "signals"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "sqlite_table",
                        "db_path": "predictions.db",
                        "table": "l4_prediction_table",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "strategies": [
                                {"strategy_id": "prod_a", "name": "Prod A", "status": "production"}
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "python": "python",
                            "project_dir": str(root),
                            "data_dir": str(root / "data"),
                            "registry_file": str(registry_path),
                            "log_dir": str(log_dir),
                            "signal_dir": str(signal_dir),
                        },
                        "strategies": [
                            {
                                "strategy_id": "prod_a",
                                "enabled": True,
                                "prediction_manifest": str(manifest_path),
                                "steps": [
                                    {
                                        "name": "echo",
                                        "command": ["{python}", "-c", "print('{strategy_id}')"],
                                    }
                                ],
                            },
                            {
                                "strategy_id": "exp_a",
                                "enabled": True,
                                "steps": [{"name": "bad", "command": ["should-not-run"]}],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = run_production_tasks(config_path, dry_run=True)

        self.assertEqual(summary["production_strategies"], ["prod_a"])
        self.assertEqual(summary["results"][0]["strategy_id"], "prod_a")
        self.assertEqual(summary["results"][0]["status"], "ok")
        self.assertEqual(summary["results"][0]["steps"][0]["status"], "dry_run")
        self.assertEqual(summary["results"][0]["steps"][0]["command"][-1], "print('prod_a')")
        self.assertEqual(summary["results"][1]["strategy_id"], "exp_a")
        self.assertEqual(summary["results"][1]["status"], "skipped")

    def test_dry_run_rejects_missing_prediction_manifest_for_production(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_path = root / "registry.json"
            config_path = root / "tasks.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "strategies": [
                                {"strategy_id": "prod_a", "name": "Prod A", "status": "production"}
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "python": "python",
                            "project_dir": str(root),
                            "data_dir": str(root / "data"),
                            "registry_file": str(registry_path),
                            "log_dir": str(root / "logs"),
                            "signal_dir": str(root / "signals"),
                        },
                        "strategies": [
                            {
                                "strategy_id": "prod_a",
                                "enabled": True,
                                "steps": [
                                    {
                                        "name": "echo",
                                        "command": ["{python}", "-c", "print('prod')"],
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = run_production_tasks(config_path, dry_run=True)

        self.assertEqual(summary["results"][0]["status"], "failed")
        self.assertIn("prediction_manifest", summary["results"][0]["detail"])

    def test_dry_run_rejects_unapproved_prediction_manifest_for_production(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_path = root / "registry.json"
            config_path = root / "tasks.json"
            manifest_path = root / "prediction_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "pending_model_agent_confirmation",
                        "source_type": "sqlite_table",
                        "db_path": "predictions.db",
                        "table": "l4_prediction_table",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "strategies": [
                                {"strategy_id": "prod_a", "name": "Prod A", "status": "production"}
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "python": "python",
                            "project_dir": str(root),
                            "data_dir": str(root / "data"),
                            "registry_file": str(registry_path),
                            "log_dir": str(root / "logs"),
                            "signal_dir": str(root / "signals"),
                        },
                        "strategies": [
                            {
                                "strategy_id": "prod_a",
                                "enabled": True,
                                "prediction_manifest": str(manifest_path),
                                "steps": [
                                    {
                                        "name": "echo",
                                        "command": ["{python}", "-c", "print('prod')"],
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = run_production_tasks(config_path, dry_run=True)

        self.assertEqual(summary["results"][0]["status"], "failed")
        self.assertIn("approved_for_l5", summary["results"][0]["detail"])

    def test_dry_run_uses_approved_prediction_manifest_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_path = root / "registry.json"
            config_path = root / "tasks.json"
            manifest_path = root / "prediction_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "asset_role": "l4_formal_prediction_asset",
                        "approval_status": "approved_for_l5",
                        "source_type": "sqlite_table",
                        "db_path": "predictions.db",
                        "table": "l4_prediction_table",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            registry_path.write_text(
                json.dumps(
                    {
                        "production": {
                            "strategies": [
                                {"strategy_id": "prod_a", "name": "Prod A", "status": "production"}
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "python": "python",
                            "project_dir": str(root),
                            "data_dir": str(root / "data"),
                            "registry_file": str(registry_path),
                            "log_dir": str(root / "logs"),
                            "signal_dir": str(root / "signals"),
                        },
                        "strategies": [
                            {
                                "strategy_id": "prod_a",
                                "enabled": True,
                                "prediction_manifest": str(manifest_path),
                                "steps": [
                                    {
                                        "name": "echo",
                                        "command": ["{python}", "-c", "print('{prediction_manifest}')"],
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = run_production_tasks(config_path, dry_run=True)

        self.assertEqual(summary["results"][0]["status"], "ok")
        self.assertEqual(summary["results"][0]["steps"][0]["status"], "dry_run")
        self.assertEqual(summary["results"][0]["steps"][0]["command"][-1], f"print('{manifest_path}')")


if __name__ == "__main__":
    unittest.main()
