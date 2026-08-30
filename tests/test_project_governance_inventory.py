import sys
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from tools.project_governance_inventory import (
    classify_root_python,
    format_bytes,
    is_dated_script,
    root_python_stats,
    workspace_shell_stats,
)


def test_is_dated_script_requires_python_and_full_date_token() -> None:
    assert is_dated_script("research_signal_20260822.py")
    assert not is_dated_script("research_signal_202608.py")
    assert not is_dated_script("research_signal_20260822.json")


def test_classify_root_python_uses_stable_prefixes() -> None:
    assert classify_root_python("research_signal_20260822.py") == "research"
    assert classify_root_python("run_pipeline.py") == "run"
    assert classify_root_python("prediction_manifest.py") == "other"


def test_format_bytes_is_binary_and_stable() -> None:
    assert format_bytes(0) == "0.000 B"
    assert format_bytes(1024) == "1.000 KiB"
    assert format_bytes(1024**3) == "1.000 GiB"


def test_dated_script_registry_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "research_signal_20260822.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "runtime.py").write_text("pass\n", encoding="utf-8")

    stats = root_python_stats(tmp_path)

    assert stats["dated_python_files"] == 1
    assert stats["dated_python_registry"][0]["path"] == "research_signal_20260822.py"
    assert stats["dated_python_registry"][0]["lifecycle_state"] == "reference_review_required"
    assert stats["dated_python_registry"][0]["deletion_allowed"] is False


def test_workspace_shell_files_are_reference_review_only(tmp_path: Path) -> None:
    (tmp_path / "entry.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "repo").mkdir()

    stats = workspace_shell_stats(tmp_path)

    assert stats["python_files"] == 1
    assert stats["directories"] == ["repo"]
    assert stats["files"][0]["deletion_allowed"] is False
