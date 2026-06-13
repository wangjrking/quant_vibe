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
            log_dir = root / "logs"
            signal_dir = root / "signals"
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


if __name__ == "__main__":
    unittest.main()
