import sys
import tempfile
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from tools.build_retention_registry import lifecycle_for, owner_for, sample_fingerprint


def test_sample_fingerprint_changes_with_tail() -> None:
    with tempfile.TemporaryDirectory(dir=MAIN_DIR) as directory:
        path = Path(directory) / "asset.duckdb"
        path.write_bytes(b"a" * 32)
        first = sample_fingerprint(path, sample_bytes=8)
        path.write_bytes(b"a" * 31 + b"b")
        second = sample_fingerprint(path, sample_bytes=8)
        assert first != second


def test_owner_and_lifecycle_are_conservative() -> None:
    assert owner_for(Path("reports/l3_delivery/asset.duckdb")) == "factor-agent"
    assert lifecycle_for(Path("reports/l3_delivery/rollback/asset.duckdb")) == "rollback_review"
    assert lifecycle_for(Path("runtime/agent_workspaces/model/candidate.duckdb")) == "runtime_candidate_review"
