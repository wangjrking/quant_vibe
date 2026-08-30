import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from tools.apply_production_asset_pair_change import (
    execute_pair_registry_change,
    main,
    prepare_pair_registry_payload,
    validate_pair_change_plan,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(asset_id, layer, path, active=True):
    return {
        "asset_id": asset_id,
        "layer": layer,
        "status": "production_active" if active else "retired_legacy_reference",
        "allowed_for_main_workflow": active,
        "asset_path": path,
    }


def _database(path: Path, table: str, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as conn:
        conn.execute(f'CREATE TABLE "{table}"(trade_date VARCHAR, stock_code VARCHAR, value INTEGER)')
        conn.execute(f'INSERT INTO "{table}" VALUES (?, ?, ?)', ["20260717", "000001.SZ", value])


class ApplyProductionAssetPairChangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry_path = self.root / "quant/data_file/asset_registry/production_assets.json"
        self.registry_path.parent.mkdir(parents=True)
        self.old_feature = self.root / "quant/data_file/production_assets/duckdb/l3_feature_current.duckdb"
        self.old_label = self.root / "quant/data_file/production_assets/duckdb/l3_label_current.duckdb"
        self.feature_candidate = self.root / "quant/data_file/runtime/work/feature_candidate.duckdb"
        self.label_candidate = self.root / "quant/data_file/runtime/work/label_candidate.duckdb"
        _database(self.old_feature, "feature", 1)
        _database(self.old_label, "label", 2)
        _database(self.feature_candidate, "feature", 3)
        _database(self.label_candidate, "label", 4)
        self.registry = {
            "assets": [
                _asset("old_feature", "L3_features", "quant/data_file/production_assets/duckdb/l3_feature_current.duckdb::feature"),
                _asset("old_label", "L3_labels", "quant/data_file/production_assets/duckdb/l3_label_current.duckdb::label"),
            ]
        }
        self.registry_path.write_text(json.dumps(self.registry), encoding="utf-8")
        self.change_path = self.root / "pair.json"
        self.audit_path = self.root / "audit.json"
        self.quarantine_dir = self.root / "quarantine"
        self.change = self._change()
        self.change_path.write_text(json.dumps(self.change), encoding="utf-8")
        self.audit_path.write_text(json.dumps({
            "approved_for_pair_change": True,
            "pair_transaction_id": "pair-1",
            "risk_level": "P1",
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _change(self):
        return {
            "pair_transaction_id": "pair-1",
            "prepared_at": "2026-07-18T00:00:00+08:00",
            "expected_registry_sha256": _sha256(self.registry_path),
            "changes": [
                {
                    "asset_before_id": "old_feature",
                    "asset_before_expected": {
                        "asset_path": self.registry["assets"][0]["asset_path"],
                        "table": "feature",
                        "sha256": _sha256(self.old_feature),
                    },
                    "candidate": {"path": str(self.feature_candidate), "table": "feature", "sha256": _sha256(self.feature_candidate)},
                    "asset_after": _asset(
                        "new_feature", "L3_features",
                        "quant/data_file/production_assets/duckdb/l3_feature_pair_1.duckdb::feature",
                    ),
                },
                {
                    "asset_before_id": "old_label",
                    "asset_before_expected": {
                        "asset_path": self.registry["assets"][1]["asset_path"],
                        "table": "label",
                        "sha256": _sha256(self.old_label),
                    },
                    "candidate": {"path": str(self.label_candidate), "table": "label", "sha256": _sha256(self.label_candidate)},
                    "asset_after": _asset(
                        "new_label", "L3_labels",
                        "quant/data_file/production_assets/duckdb/l3_label_pair_1.duckdb::label",
                    ),
                },
            ],
        }

    def test_prepare_pair_change_keeps_feature_label_consistent(self):
        result = prepare_pair_registry_payload(self.registry, self.change)
        active = [asset for asset in result["assets"] if asset["allowed_for_main_workflow"]]
        self.assertEqual({asset["asset_id"] for asset in active}, {"new_feature", "new_label"})
        self.assertEqual(result["last_pair_transaction_id"], "pair-1")

    def test_dry_run_validates_old_candidate_and_future_routes(self):
        result = validate_pair_change_plan(self.root, self.registry_path, self.change)
        self.assertEqual(result["status"], "dry_run_validated")
        self.assertEqual(set(result["candidates"]), {"l3_features", "l3_labels"})
        self.assertFalse(result["active_switch_called"])

    def test_cli_defaults_to_dry_run_and_keeps_registry_unchanged(self):
        before = self.registry_path.read_bytes()
        self.assertEqual(main([
            "--project-root", str(self.root), "--registry", str(self.registry_path),
            "--pair-change", str(self.change_path),
        ]), 0)
        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_execute_requires_approved_audit_record(self):
        with self.assertRaisesRegex(PermissionError, "approved-audit-record"):
            main([
                "--project-root", str(self.root), "--registry", str(self.registry_path),
                "--pair-change", str(self.change_path), "--execute",
            ])

    def test_atomic_pair_commit_and_formal_route_revalidation(self):
        result = execute_pair_registry_change(
            self.root, self.registry_path, self.change_path, self.audit_path, self.quarantine_dir
        )
        self.assertEqual(result["status"], "pair_change_committed")
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        active = [asset for asset in registry["assets"] if asset["allowed_for_main_workflow"]]
        self.assertEqual({asset["asset_id"] for asset in active}, {"new_feature", "new_label"})
        self.assertTrue((self.root / "quant/data_file/production_assets/duckdb/l3_feature_pair_1.duckdb").is_file())
        self.assertTrue((self.root / "quant/data_file/production_assets/duckdb/l3_label_pair_1.duckdb").is_file())

    def test_post_replace_failure_restores_registry_and_quarantines_pair(self):
        before = self.registry_path.read_bytes()

        def fail_after_replace(_registry):
            raise RuntimeError("formal route probe failed")

        with self.assertRaisesRegex(RuntimeError, "formal route probe failed"):
            execute_pair_registry_change(
                self.root,
                self.registry_path,
                self.change_path,
                self.audit_path,
                self.quarantine_dir,
                post_replace_validate=fail_after_replace,
            )
        self.assertEqual(self.registry_path.read_bytes(), before)
        quarantined = list((self.quarantine_dir / "pair-1").glob("*.duckdb"))
        self.assertEqual(len(quarantined), 4)
        self.assertFalse(self.feature_candidate.exists())
        self.assertFalse(self.label_candidate.exists())

    def test_old_registry_hash_drift_fails_closed(self):
        change = self._change()
        change["expected_registry_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "old registry SHA256 mismatch"):
            validate_pair_change_plan(self.root, self.registry_path, change)

    def test_candidate_hash_drift_fails_closed(self):
        change = self._change()
        change["changes"][0]["candidate"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "DuckDB SHA256 mismatch"):
            validate_pair_change_plan(self.root, self.registry_path, change)

    def test_old_path_or_table_drift_fails_closed(self):
        change = self._change()
        change["changes"][0]["asset_before_expected"]["asset_path"] = "wrong.duckdb::feature"
        with self.assertRaisesRegex(RuntimeError, "old active asset path drift"):
            validate_pair_change_plan(self.root, self.registry_path, change)

    def test_candidate_table_drift_fails_closed(self):
        change = self._change()
        change["changes"][0]["candidate"]["table"] = "wrong_feature"
        with self.assertRaisesRegex(RuntimeError, "candidate/after table mismatch"):
            validate_pair_change_plan(self.root, self.registry_path, change)

    def test_partial_destination_replace_never_commits_registry(self):
        before = self.registry_path.read_bytes()
        replace_calls = 0

        def fail_second_destination(source, target):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 2:
                raise OSError("second destination replace failed")
            return __import__("os").replace(source, target)

        with self.assertRaisesRegex(OSError, "second destination replace failed"):
            execute_pair_registry_change(
                self.root,
                self.registry_path,
                self.change_path,
                self.audit_path,
                self.quarantine_dir,
                replace=fail_second_destination,
            )
        self.assertEqual(self.registry_path.read_bytes(), before)
        self.assertFalse((self.root / "quant/data_file/production_assets/duckdb/l3_feature_pair_1.duckdb").exists())
        self.assertFalse((self.root / "quant/data_file/production_assets/duckdb/l3_label_pair_1.duckdb").exists())
        self.assertGreaterEqual(len(list((self.quarantine_dir / "pair-1").glob("*.duckdb"))), 3)

    def test_single_asset_change_is_rejected(self):
        change = self._change()
        change["changes"] = change["changes"][:1]
        with self.assertRaisesRegex(ValueError, "exactly two"):
            prepare_pair_registry_payload(self.registry, change)


if __name__ == "__main__":
    unittest.main()
