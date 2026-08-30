import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_production_tasks import (
    production_strategy_ids,
    run_production_tasks,
    validate_current_production_task_alignment,
)


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

    def test_current_production_task_alignment_rejects_stale_enabled_strategy(self):
        registry = {
            "production": {
                "current": "prod_current",
                "strategies": [
                    {"strategy_id": "prod_current", "status": "production"},
                ],
            }
        }
        config = {
            "strategies": [
                {"strategy_id": "prod_stale", "enabled": True},
            ]
        }

        with self.assertRaisesRegex(ValueError, "exactly match registry current"):
            validate_current_production_task_alignment(config, registry)

    def test_current_production_task_alignment_accepts_current_strategy(self):
        registry = {
            "production": {
                "current": "prod_current",
                "strategies": [
                    {"strategy_id": "prod_current", "status": "production"},
                ],
            }
        }
        config = {
            "strategies": [
                {"strategy_id": "prod_current", "enabled": True},
            ]
        }

        self.assertEqual(
            validate_current_production_task_alignment(config, registry),
            "prod_current",
        )

    def test_current_production_task_alignment_rejects_missing_current(self):
        registry = {
            "production": {
                "current": "",
                "strategies": [],
            }
        }
        config = {"strategies": []}

        with self.assertRaisesRegex(ValueError, "production.current missing"):
            validate_current_production_task_alignment(config, registry)

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

    @patch("run_production_tasks.sync_production_signal_artifacts_to_duckdb")
    @patch("run_production_tasks.sync_strategy_backtests_to_duckdb")
    @patch("run_production_tasks.sync_strategy_registry_to_duckdb")
    @patch("run_production_tasks.load_production_strategy_registry")
    @patch("run_production_tasks.run_step")
    def test_non_dry_run_triggers_l5_l7_duckdb_sync_hooks(
        self,
        fake_run_step,
        fake_load_production_strategy_registry,
        fake_sync_strategy_registry_to_duckdb,
        fake_sync_strategy_backtests_to_duckdb,
        fake_sync_production_signal_artifacts_to_duckdb,
    ):
        fake_run_step.return_value = {
            "status": "ok",
            "command": ["python", "-c", "print('ok')"],
            "log_file": "dummy.log",
            "started_at": "2026-06-28T00:00:00",
            "ended_at": "2026-06-28T00:00:01",
            "returncode": 0,
        }
        fake_sync_strategy_registry_to_duckdb.return_value = {"registry_rows": 1}
        fake_sync_strategy_backtests_to_duckdb.return_value = {"validation_row_count": 1}
        fake_sync_production_signal_artifacts_to_duckdb.return_value = {"signal_row_count": 1}
        fake_load_production_strategy_registry.return_value = {
            "production": {
                "current": "prod_a",
                "strategies": [{"strategy_id": "prod_a", "name": "Prod A", "status": "production"}],
            }
        }

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
                            "current": "prod_a",
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

            summary = run_production_tasks(config_path, dry_run=False)

        self.assertEqual(summary["results"][0]["status"], "ok")
        self.assertTrue(fake_sync_strategy_registry_to_duckdb.called)
        self.assertTrue(fake_sync_strategy_backtests_to_duckdb.called)
        self.assertTrue(fake_sync_production_signal_artifacts_to_duckdb.called)
        self.assertTrue(fake_load_production_strategy_registry.called)


if __name__ == "__main__":
    unittest.main()
