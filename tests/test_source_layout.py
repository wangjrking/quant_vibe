import json
from pathlib import Path

from tools.check_source_layout import validate_layout


def policy() -> dict:
    return {
        "root_python_file_limit": 3,
        "required_root_entrypoints": ["entry.py"],
        "required_directories": ["tools", "legacy/dated-scripts"],
        "allowed_root_dated_python": ["compat_20260830.py"],
        "root_research_prefix_exception_only": True,
    }


def make_main(tmp_path: Path) -> Path:
    main = tmp_path / "quant" / "main"
    (main / "tools").mkdir(parents=True)
    (main / "legacy" / "dated-scripts").mkdir(parents=True)
    (main / "entry.py").write_text("pass\n", encoding="utf-8")
    return main


def test_layout_accepts_registered_compatibility_script(tmp_path: Path) -> None:
    main = make_main(tmp_path)
    (main / "compat_20260830.py").write_text("pass\n", encoding="utf-8")
    assert validate_layout(main, policy()) == []


def test_layout_rejects_unregistered_dated_and_research_scripts(tmp_path: Path) -> None:
    main = make_main(tmp_path)
    (main / "research_probe_20260830.py").write_text("pass\n", encoding="utf-8")
    errors = validate_layout(main, policy())
    assert any("unexpected dated root scripts" in error for error in errors)
    assert any("unexpected root research scripts" in error for error in errors)


def test_layout_rejects_root_file_limit_and_missing_entry(tmp_path: Path) -> None:
    main = make_main(tmp_path)
    (main / "a.py").write_text("pass\n", encoding="utf-8")
    (main / "b.py").write_text("pass\n", encoding="utf-8")
    (main / "c.py").write_text("pass\n", encoding="utf-8")
    (main / "d.py").write_text("pass\n", encoding="utf-8")
    (main / "entry.py").unlink()
    errors = validate_layout(main, policy())
    assert any("root python file limit exceeded" in error for error in errors)
    assert any("missing required root entrypoint" in error for error in errors)
