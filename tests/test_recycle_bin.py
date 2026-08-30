from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import recycle_bin


class RecycleBinToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.project_root = self.root / "project"
        self.project_root.mkdir()
        self.bin_root = self.project_root / "quant" / "data_file" / "runtime" / "recycle_bin"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_file(self, relative_path: str, content: str = "data") -> Path:
        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_purge_selected_batch_removes_payload_and_appends_manifest_record(self) -> None:
        source = self._write_file("quant/data_file/legacy.db")
        moved = recycle_bin.move_to_recycle_bin(
            [str(source)],
            actor="tester",
            reason="trash legacy db",
            bin_root=self.bin_root,
            project_root=self.project_root,
            retention_days=30,
        )

        batch_id = moved[0]["batch_id"]
        trashed_path = Path(moved[0]["trashed_path"])
        self.assertTrue(trashed_path.exists())

        purged = recycle_bin.purge_recycle_bin(
            self.bin_root,
            actor="tester",
            reason="hard purge retired payload",
            batch_ids={batch_id},
            dry_run=False,
        )

        self.assertEqual(1, len(purged))
        self.assertFalse(trashed_path.exists())
        manifest_rows = [
            json.loads(line)
            for line in recycle_bin.manifest_path(self.bin_root).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(["trashed", "purged"], [row["status"] for row in manifest_rows])
        self.assertEqual("hard purge retired payload", manifest_rows[-1]["reason"])

    def test_purge_by_original_prefix_dry_run_preserves_payload(self) -> None:
        source_a = self._write_file("quant/data_file/old/a.sqlite")
        source_b = self._write_file("quant/data_file/keep/b.sqlite")
        recycle_bin.move_to_recycle_bin(
            [str(source_a), str(source_b)],
            actor="tester",
            reason="trash legacy files",
            bin_root=self.bin_root,
            project_root=self.project_root,
            retention_days=30,
        )

        purged = recycle_bin.purge_recycle_bin(
            self.bin_root,
            actor="tester",
            reason="preview purge",
            original_prefixes=("quant/data_file/old",),
            dry_run=True,
        )

        self.assertEqual(1, len(purged))
        self.assertEqual("true", purged[0]["dry_run"])
        self.assertTrue(Path(purged[0]["trashed_path"]).exists())


if __name__ == "__main__":
    unittest.main()
