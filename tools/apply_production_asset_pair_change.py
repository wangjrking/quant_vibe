from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from l3_active_writer_lease import (
    acquire_active_l3_writer_lease,
    release_active_l3_writer_lease,
    require_active_l3_writer_lease_for_path,
)

DEFAULT_REGISTRY = PROJECT_ROOT / "quant" / "data_file" / "asset_registry" / "production_assets.json"
REQUIRED_LAYERS = {"l3_features", "l3_labels"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, content: bytes, *, replace: Callable = os.replace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        replace(str(temp_path), str(path))
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: dict[str, Any], *, replace: Callable = os.replace) -> None:
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _write_bytes_atomic(path, content, replace=replace)


def _split_asset_path(asset_path: str) -> tuple[str, str]:
    value = str(asset_path or "").strip()
    if "::" not in value:
        raise ValueError("asset_path must identify a DuckDB table with path::table")
    path, table = value.rsplit("::", 1)
    if not path or not table:
        raise ValueError("asset_path path and table must both be non-empty")
    return path, table


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_under(path: Path, parent: Path, *, context: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as error:
        raise ValueError(f"{context} must stay under {parent}: {path}") from error


def _duckdb_state(path: Path, table: str, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = _sha256(path)
    if actual_hash != str(expected_sha256).lower():
        raise RuntimeError(f"DuckDB SHA256 mismatch: {path} expected={expected_sha256} actual={actual_hash}")
    with closing(duckdb.connect(str(path), read_only=True)) as conn:
        tables = [str(row[0]) for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY 1"
        ).fetchall()]
        if tables != [table]:
            raise RuntimeError(f"one-table-one-file gate failed for {path}: expected={[table]} actual={tables}")
        row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{table.replace(chr(34), chr(34) * 2)}"').fetchone()[0])
    return {"path": str(path), "table": table, "sha256": actual_hash, "row_count": row_count}


def _active_routes(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    active: dict[str, list[dict[str, Any]]] = {layer: [] for layer in REQUIRED_LAYERS}
    for asset in registry.get("assets", []):
        layer = str(asset.get("layer") or "").strip().lower()
        if layer in active and asset.get("allowed_for_main_workflow") is True and asset.get("status") == "production_active":
            active[layer].append(asset)
    bad = {layer: len(items) for layer, items in active.items() if len(items) != 1}
    if bad:
        raise RuntimeError(f"pair registry must have exactly one active asset per L3 layer: {bad}")
    return {layer: items[0] for layer, items in active.items()}


def prepare_pair_registry_payload(registry: dict[str, Any], pair_change: dict[str, Any]) -> dict[str, Any]:
    transaction_id = str(pair_change.get("pair_transaction_id") or "").strip()
    changes = pair_change.get("changes")
    if not transaction_id:
        raise ValueError("pair_transaction_id is required")
    if not isinstance(changes, list) or len(changes) != 2:
        raise ValueError("pair change must contain exactly two asset changes")

    changed_layers: set[str] = set()
    assets = [deepcopy(item) for item in registry.get("assets", [])]
    by_id = {str(item.get("asset_id")): idx for idx, item in enumerate(assets)}
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("each pair change item must be an object")
        before_id = str(change.get("asset_before_id") or "").strip()
        after = deepcopy(change.get("asset_after") or {})
        after_id = str(after.get("asset_id") or "").strip()
        layer = str(after.get("layer") or "").strip().lower()
        if not before_id or before_id not in by_id:
            raise ValueError(f"asset_before_id not found: {before_id}")
        if not after_id:
            raise ValueError("asset_after.asset_id is required")
        if layer not in REQUIRED_LAYERS:
            raise ValueError(f"unexpected pair layer: {layer}")
        if layer in changed_layers:
            raise ValueError(f"duplicate pair layer: {layer}")
        if after.get("allowed_for_main_workflow") is not True or after.get("status") != "production_active":
            raise ValueError("asset_after must be production_active and allowed_for_main_workflow=true")
        _split_asset_path(str(after.get("asset_path") or ""))

        previous = deepcopy(assets[by_id[before_id]])
        previous["status"] = "retired_legacy_reference"
        previous["allowed_for_main_workflow"] = False
        previous["superseded_by"] = after_id
        previous["pair_transaction_id"] = transaction_id
        assets[by_id[before_id]] = previous

        existing_idx = by_id.get(after_id)
        if existing_idx is None:
            by_id[after_id] = len(assets)
            assets.append(after)
        else:
            assets[existing_idx] = after
        changed_layers.add(layer)

    if changed_layers != REQUIRED_LAYERS:
        raise ValueError(f"pair change must cover {sorted(REQUIRED_LAYERS)}")
    result = deepcopy(registry)
    result["assets"] = assets
    result["updated_at"] = str(pair_change.get("prepared_at") or result.get("updated_at") or "")
    result["last_pair_transaction_id"] = transaction_id
    _active_routes(result)
    return result


def validate_pair_change_plan(
    root: Path,
    registry_path: Path,
    pair_change: dict[str, Any],
) -> dict[str, Any]:
    root = root.resolve()
    registry_path = registry_path.resolve()
    registry_bytes = registry_path.read_bytes()
    actual_registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    expected_registry_hash = str(pair_change.get("expected_registry_sha256") or "").lower()
    if not expected_registry_hash or actual_registry_hash != expected_registry_hash:
        raise RuntimeError(
            f"old registry SHA256 mismatch: expected={expected_registry_hash or '<missing>'} actual={actual_registry_hash}"
        )
    registry = json.loads(registry_bytes.decode("utf-8"))
    current_routes = _active_routes(registry)
    updated = prepare_pair_registry_payload(registry, pair_change)
    future_routes = _active_routes(updated)
    candidate_states: dict[str, dict[str, Any]] = {}
    old_states: dict[str, dict[str, Any]] = {}
    destination_states: dict[str, dict[str, str]] = {}

    for change in pair_change["changes"]:
        after = change["asset_after"]
        layer = str(after["layer"]).lower()
        before_expected = change.get("asset_before_expected") or {}
        candidate = change.get("candidate") or {}
        current = current_routes[layer]
        if current.get("asset_id") != change.get("asset_before_id"):
            raise RuntimeError(f"old active asset-id drift for {layer}")
        if str(current.get("asset_path")) != str(before_expected.get("asset_path")):
            raise RuntimeError(f"old active asset path drift for {layer}")
        old_path_text, old_table = _split_asset_path(str(current.get("asset_path")))
        if old_table != str(before_expected.get("table")):
            raise RuntimeError(f"old active table drift for {layer}")
        old_path = _resolve_path(root, old_path_text)
        old_states[layer] = _duckdb_state(old_path, old_table, str(before_expected.get("sha256") or ""))

        candidate_path = _resolve_path(root, str(candidate.get("path") or ""))
        candidate_table = str(candidate.get("table") or "")
        if not candidate_table or candidate_table != str(after.get("asset_path")).rsplit("::", 1)[-1]:
            raise RuntimeError(f"candidate/after table mismatch for {layer}")
        candidate_states[layer] = _duckdb_state(candidate_path, candidate_table, str(candidate.get("sha256") or ""))

        destination_text, destination_table = _split_asset_path(str(after.get("asset_path")))
        destination_path = _resolve_path(root, destination_text)
        production_duckdb_dir = root / "quant" / "data_file" / "production_assets" / "duckdb"
        _require_under(destination_path, production_duckdb_dir, context=f"{layer} destination")
        if destination_path.exists():
            raise FileExistsError(f"destination already exists; fail closed: {destination_path}")
        if future_routes[layer].get("asset_id") != after.get("asset_id"):
            raise RuntimeError(f"temporary registry active route mismatch for {layer}")
        if destination_table != candidate_table:
            raise RuntimeError(f"destination table mismatch for {layer}")
        destination_states[layer] = {"path": str(destination_path), "table": destination_table}

    return {
        "status": "dry_run_validated",
        "pair_transaction_id": pair_change["pair_transaction_id"],
        "registry_path": str(registry_path),
        "registry_sha256": actual_registry_hash,
        "old_assets": old_states,
        "candidates": candidate_states,
        "destinations": destination_states,
        "updated_registry": updated,
        "active_switch_called": False,
    }


def _validate_audit_record(path: Path, transaction_id: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("approved_for_pair_change") is not True:
        raise PermissionError("audit record must set approved_for_pair_change=true")
    if str(payload.get("pair_transaction_id") or "") != transaction_id:
        raise PermissionError("audit record pair_transaction_id mismatch")
    if str(payload.get("risk_level") or "").upper() not in {"P0", "P1", "P2", "P3"}:
        raise PermissionError("audit record risk_level is required")
    return payload


def _quarantine_paths(paths: list[Path], quarantine_dir: Path) -> list[str]:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        target = quarantine_dir / path.name
        if target.exists():
            target = quarantine_dir / f"{path.stem}.{len(moved):02d}{path.suffix}"
        os.replace(path, target)
        moved.append(str(target))
    return moved


def execute_pair_registry_change(
    root: Path,
    registry_path: Path,
    pair_change_path: Path,
    audit_record_path: Path,
    quarantine_dir: Path,
    *,
    replace: Callable = os.replace,
    post_replace_validate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    require_active_l3_writer_lease_for_path(registry_path)
    pair_change = _read_json(pair_change_path)
    dry_run = validate_pair_change_plan(root, registry_path, pair_change)
    transaction_id = str(pair_change["pair_transaction_id"])
    _validate_audit_record(audit_record_path, transaction_id)
    original_bytes = registry_path.read_bytes()
    copied_temp_paths: list[Path] = []
    finalized_paths: list[Path] = []
    candidate_paths: list[Path] = []
    registry_replaced = False
    try:
        for change in pair_change["changes"]:
            layer = str(change["asset_after"]["layer"]).lower()
            candidate_path = Path(dry_run["candidates"][layer]["path"])
            destination_path = Path(dry_run["destinations"][layer]["path"])
            candidate_paths.append(candidate_path)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = destination_path.with_suffix(destination_path.suffix + f".{transaction_id}.tmp")
            temp_path.unlink(missing_ok=True)
            shutil.copy2(candidate_path, temp_path)
            if _sha256(temp_path) != dry_run["candidates"][layer]["sha256"]:
                raise RuntimeError(f"copied candidate SHA256 mismatch for {layer}")
            copied_temp_paths.append(temp_path)

        for change, temp_path in zip(pair_change["changes"], copied_temp_paths):
            layer = str(change["asset_after"]["layer"]).lower()
            destination_path = Path(dry_run["destinations"][layer]["path"])
            replace(str(temp_path), str(destination_path))
            finalized_paths.append(destination_path)

        _write_json_atomic(registry_path, dry_run["updated_registry"], replace=replace)
        registry_replaced = True
        committed = _read_json(registry_path)
        committed_routes = _active_routes(committed)
        for change in pair_change["changes"]:
            after = change["asset_after"]
            layer = str(after["layer"]).lower()
            if committed_routes[layer].get("asset_id") != after.get("asset_id"):
                raise RuntimeError(f"formal route revalidation failed for {layer}")
            destination = dry_run["destinations"][layer]
            _duckdb_state(Path(destination["path"]), destination["table"], dry_run["candidates"][layer]["sha256"])
        if post_replace_validate is not None:
            post_replace_validate(committed)
        return {
            "status": "pair_change_committed",
            "pair_transaction_id": transaction_id,
            "registry_path": str(registry_path),
            "registry_sha256": _sha256(registry_path),
            "active_routes": {layer: committed_routes[layer]["asset_path"] for layer in sorted(REQUIRED_LAYERS)},
            "audit_record": str(audit_record_path),
        }
    except Exception:
        if registry_replaced or registry_path.read_bytes() != original_bytes:
            _write_bytes_atomic(registry_path, original_bytes, replace=replace)
            restored = registry_path.read_bytes()
            if restored != original_bytes:
                raise RuntimeError("registry rollback verification failed")
            _active_routes(json.loads(restored.decode("utf-8")))
        _quarantine_paths(
            [*candidate_paths, *finalized_paths, *copied_temp_paths],
            quarantine_dir / transaction_id,
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run or atomically commit an audited L3 feature/label registry pair change."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--pair-change", required=True)
    parser.add_argument("--dry-run-output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved-audit-record")
    parser.add_argument("--quarantine-dir")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    registry_path = _resolve_path(root, args.registry)
    pair_path = _resolve_path(root, args.pair_change)
    pair_change = _read_json(pair_path)
    if not args.execute:
        result = validate_pair_change_plan(root, registry_path, pair_change)
        output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.dry_run_output:
            _write_json_atomic(_resolve_path(root, args.dry_run_output), result)
        print(output, end="")
        return 0

    if not args.approved_audit_record:
        raise PermissionError("--execute requires --approved-audit-record")
    if not args.quarantine_dir:
        raise ValueError("--execute requires --quarantine-dir")
    audit_record_path = _resolve_path(root, args.approved_audit_record)
    quarantine_dir = _resolve_path(root, args.quarantine_dir)
    pair_transaction_id = str(pair_change.get("pair_transaction_id") or "unknown")
    lease = acquire_active_l3_writer_lease(
        workflow_run_id=f"l3-pair-change-{pair_transaction_id}",
        workspace=quarantine_dir,
        report_dir=audit_record_path.parent,
        process_role="apply_production_asset_pair_change",
    )
    try:
        result = execute_pair_registry_change(
            root,
            registry_path,
            pair_path,
            audit_record_path,
            quarantine_dir,
        )
    finally:
        release_active_l3_writer_lease(lease)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
