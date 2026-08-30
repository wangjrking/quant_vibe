from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path

import pandas as pd


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import export_v260_all4key_production_signals as exporter


class V260NoSignalContractTest(unittest.TestCase):
    def test_empty_intent_frame_keeps_formal_schema(self) -> None:
        frame = pd.DataFrame([], columns=exporter.INTENT_COLUMNS)

        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), exporter.INTENT_COLUMNS)
        self.assertEqual(
            int(
                frame.duplicated(
                    ["signal_date", "buy_date", "action", "stock_code"]
                ).sum()
            ),
            0,
        )
        self.assertEqual(
            int(frame["stock_code"].astype(str).str.endswith(".BJ").sum()),
            0,
        )

    def test_no_signal_contract_does_not_create_hold_action(self) -> None:
        frame = pd.DataFrame([], columns=exporter.INTENT_COLUMNS)

        self.assertEqual(len(frame), 0)
        self.assertNotIn("HOLD", set(frame["action"].astype(str)))

    def test_capture_prewrite_snapshot_materializes_manifest_and_copies_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            sources = {}
            for name in (
                "latest_csv",
                "latest_status",
                "strategy_manifest",
                "validation_json",
                "current_actions",
                "l5_registry_duckdb",
                "l5_manifest_duckdb",
                "l6_validation_duckdb",
            ):
                path = root / f"{name}.bin"
                path.write_bytes(f"{name}-payload".encode("ascii"))
                sources[name] = path

            payload = exporter.capture_prewrite_snapshot(
                report_dir,
                "20260730",
                "20260731",
                sources=sources,
            )

            manifest_path = report_dir / "rollback_snapshot_manifest.json"
            self.assertTrue(manifest_path.is_file())
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["signal_date"], "20260730")
            self.assertEqual(set(loaded["sources"]), set(sources))
            for name, source in sources.items():
                entry = payload["sources"][name]
                snapshot_path = Path(entry["snapshot_path"])
                self.assertTrue(snapshot_path.is_file())
                self.assertEqual(snapshot_path.read_bytes(), source.read_bytes())
                self.assertEqual(entry["source_state"]["sha256"], entry["snapshot_state"]["sha256"])

            reused = exporter.capture_prewrite_snapshot(
                report_dir,
                "20260730",
                "20260731",
                sources=sources,
            )
            self.assertEqual(reused["sources"]["latest_csv"]["source_state"]["sha256"], payload["sources"]["latest_csv"]["source_state"]["sha256"])

    def test_capture_prewrite_snapshot_fails_closed_when_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            sources = {"latest_csv": root / "missing.csv"}

            with self.assertRaisesRegex(RuntimeError, "prewrite rollback snapshot requires existing active sources"):
                exporter.capture_prewrite_snapshot(
                    report_dir,
                    "20260730",
                    "20260731",
                    sources=sources,
                )


if __name__ == "__main__":
    unittest.main()
