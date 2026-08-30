from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from l3_process_gate import L3_MUTATION_ENTRY_REGISTRY


MAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MAIN_DIR.parents[1]
ACTIVE_ROUTE_HINTS = (
    "l3_feature_current.duckdb",
    "l3_label_current.duckdb",
    "resolve_l3_feature_duckdb_path",
    "resolve_l3_label_duckdb_path",
    '_active_registry_asset("L3_features")',
    '_active_registry_asset("L3_labels")',
    "sync_feature_parts_full_to_duckdb",
    "sync_feature_parts_target_date_to_duckdb",
    "sync_label_parts_full_to_duckdb",
    "sync_label_parts_target_date_to_duckdb",
)
MUTATION_HINTS = (
    "os.replace(",
    "CREATE OR REPLACE",
    "_connect_writable(",
    "execute_pair_registry_change(",
    "apply_change(",
)
DIRECT_SHARED_LEASE_SCRIPTS = {
    "deliver_l3_target_date_duckdb_mainline.py",
    "refresh_l3_active_duckdb_full_delivery.py",
    "apply_production_asset_pair_change.py",
    "apply_production_asset_change.py",
}
SINK_GUARDED_SCRIPTS = {
    "l3_duckdb_sync.py",
    "build_production_factor_parts.py",
    "build_production_factor_raw_gtja_parts.py",
    "build_prediction_label_parts.py",
    "incremental_factor_update_target_date.py",
    "incremental_prediction_label_update_target_date.py",
    "run_incremental_factor_update_chunked.py",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_path(script: str) -> Path:
    tool_scripts = {"apply_production_asset_pair_change.py", "apply_production_asset_change.py", "production_asset_gate.py"}
    return (MAIN_DIR / "tools" / script) if script in tool_scripts else (MAIN_DIR / script)


def _line_matches(text: str, hints: tuple[str, ...]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        found = [hint for hint in hints if hint.lower() in lowered]
        if found:
            matches.append({"line": line_number, "hints": found, "text": line.strip()[:500]})
    return matches


def _active_path_symbols(text: str) -> set[str]:
    symbols: set[str] = set()
    for line in text.splitlines():
        if "l3_feature_current.duckdb" not in line.lower() and "l3_label_current.duckdb" not in line.lower():
            continue
        match = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
        if match:
            symbols.add(match.group(1).lower())
    return symbols


def _direct_active_mutation_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    active_symbols = _active_path_symbols(text)
    sync_call = re.compile(
        r"\b(?:sync_feature_parts_(?:full|target_date)_to_duckdb|"
        r"sync_label_parts_(?:full|target_date)_to_duckdb)\s*\("
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        reasons: list[str] = []
        if sync_call.search(line):
            reasons.append("calls active L3 DuckDB sync sink")
        if "_connect_writable(" in lowered:
            reasons.append("opens a writable L3 DuckDB sink")
        if "os.replace(" in lowered and (
            any(
                token in lowered
                for token in ("l3_feature_current", "l3_label_current", "active_path", "registry_path")
            )
            or any(re.search(rf"\b{re.escape(symbol)}\b", lowered) for symbol in active_symbols)
        ):
            reasons.append("atomically replaces an active L3 or registry path")
        if (
            "duckdb.connect(" in lowered
            and "read_only=true" not in lowered
            and (
                "l3_feature_current" in lowered
                or "l3_label_current" in lowered
                or re.search(r"\b(?:feature_db|label_db)\b", lowered)
            )
        ):
            reasons.append("opens an active-looking L3 DuckDB path without read_only=True")
        if any(token in lowered for token in ("write_text(", "write_bytes(")) and "production_assets.json" in lowered:
            reasons.append("writes the production registry path")
        if reasons:
            matches.append({"line": line_number, "reasons": reasons, "text": line.strip()[:500]})
    return matches


def scan_l3_write_bypass(main_dir: Path = MAIN_DIR) -> dict[str, Any]:
    entrypoints: list[dict[str, Any]] = []
    missing_sources: list[str] = []
    missing_lease_contract: list[dict[str, Any]] = []
    registered_names = set(L3_MUTATION_ENTRY_REGISTRY)
    for script, spec in sorted(L3_MUTATION_ENTRY_REGISTRY.items()):
        source = _source_path(script)
        exists = source.is_file()
        if not exists and spec.get("policy") != "readonly_observer":
            missing_sources.append(str(source))
        text = source.read_text(encoding="utf-8", errors="replace") if exists else ""
        active_policy = spec.get("active_write_policy")
        direct_lease_markers = (
            "acquire_active_l3_writer_lease" in text
            and "release_active_l3_writer_lease" in text
        )
        sink_guard_marker = (
            "require_active_l3_writer_lease_for_path" in text
            if script == "l3_duckdb_sync.py"
            else "l3_duckdb_sync" in text or "sync_feature_parts" in text or "sync_label_parts" in text
        )
        if script in DIRECT_SHARED_LEASE_SCRIPTS and not direct_lease_markers:
            missing_lease_contract.append({"script": script, "reason": "direct shared lease acquire/release markers missing"})
        if script in SINK_GUARDED_SCRIPTS and not sink_guard_marker:
            missing_lease_contract.append({"script": script, "reason": "shared l3_duckdb_sync sink guard marker missing"})
        if script == "rebuild_l3_full_duckdb_mainline.py" and active_policy != "candidate_only_no_active_write":
            missing_lease_contract.append({"script": script, "reason": "candidate-only full rebuild contract missing"})
        entrypoints.append({
            "script": script,
            "path": str(source),
            "exists": exists,
            "sha256": _sha256(source) if exists else None,
            "size_bytes": source.stat().st_size if exists else None,
            "policy": spec.get("policy"),
            "active_write_policy": active_policy,
            "mutation_scope": spec.get("mutation_scope"),
            "direct_shared_lease_markers": direct_lease_markers,
            "sink_guard_marker": sink_guard_marker,
            "active_route_matches": _line_matches(text, ACTIVE_ROUTE_HINTS),
            "mutation_matches": _line_matches(text, MUTATION_HINTS),
        })

    potential_sites: list[dict[str, Any]] = []
    unregistered_bypasses: list[dict[str, Any]] = []
    for source in sorted(main_dir.rglob("*.py")):
        relative_parts = {part.lower() for part in source.relative_to(main_dir).parts}
        if relative_parts & {"tests", "strategy_library", "__pycache__"}:
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        route_matches = _line_matches(text, ACTIVE_ROUTE_HINTS)
        direct_mutation_matches = _direct_active_mutation_matches(text)
        if not direct_mutation_matches or (not route_matches and source.name not in registered_names):
            continue
        script = source.name
        registered = script in registered_names
        item = {
            "script": script,
            "path": str(source),
            "registered": registered,
            "active_route_matches": route_matches,
            "direct_active_mutation_matches": direct_mutation_matches,
        }
        potential_sites.append(item)
        if not registered and script not in {"l3_active_writer_lease.py", "scan_l3_write_bypass.py"}:
            unregistered_bypasses.append(item)

    passed = not missing_sources and not missing_lease_contract and not unregistered_bypasses
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed_closed",
        "generated_at": _now(),
        "main_dir": str(main_dir.resolve()),
        "entrypoint_count": len(entrypoints),
        "entrypoints": entrypoints,
        "potential_active_mutation_sites": potential_sites,
        "missing_sources": missing_sources,
        "missing_lease_contract": missing_lease_contract,
        "unregistered_bypasses": unregistered_bypasses,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan all registered and textual L3 active write entrypoints for lease bypasses.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = scan_l3_write_bypass()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"write bypass scan output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output)
    print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
