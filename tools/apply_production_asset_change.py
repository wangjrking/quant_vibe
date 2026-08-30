from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = PROJECT_ROOT / "quant" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from tools.production_asset_gate import check_change_file, check_registry, load_json, registry_path
from l3_active_writer_lease import (
    acquire_active_l3_writer_lease,
    release_active_l3_writer_lease,
    require_active_l3_writer_lease_for_path,
)


def project_root() -> Path:
    return PROJECT_ROOT


def _load_change(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("apply_production_asset_change expects exactly one change object")
        return payload[0]
    if "production_asset_changes" in payload:
        changes = payload.get("production_asset_changes") or []
        if len(changes) != 1:
            raise ValueError("apply_production_asset_change expects exactly one change object")
        return changes[0]
    return payload


def _load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_or_append_asset(assets: list[dict[str, Any]], asset_after: dict[str, Any]) -> list[dict[str, Any]]:
    asset_id = asset_after.get("asset_id")
    updated: list[dict[str, Any]] = []
    replaced = False
    for asset in assets:
        if asset.get("asset_id") == asset_id:
            updated.append(asset_after)
            replaced = True
        else:
            updated.append(asset)
    if not replaced:
        updated.append(asset_after)
    return updated


def apply_change(root: Path, change_file: Path) -> Path:
    gate_errors = check_change_file(root, change_file)
    if gate_errors:
        raise RuntimeError("change file failed production_asset_gate: " + "; ".join(gate_errors))

    reg_path = registry_path(root)
    require_active_l3_writer_lease_for_path(reg_path)
    registry = _load_registry(reg_path)
    assets = list(registry.get("assets", []))
    change = _load_change(change_file)
    before = dict(change.get("asset_before") or {})
    after = dict(change.get("asset_after") or {})
    before_id = before.get("asset_id")
    after_id = after.get("asset_id")
    if not before_id or not after_id:
        raise ValueError("change file must contain asset_before.asset_id and asset_after.asset_id")

    found_before = False
    updated_assets: list[dict[str, Any]] = []
    for asset in assets:
        if asset.get("asset_id") != before_id:
            updated_assets.append(asset)
            continue
        found_before = True
        previous = dict(asset)
        previous["status"] = "retired_legacy_reference"
        previous["allowed_for_main_workflow"] = False
        previous["superseded_by"] = after_id
        updated_assets.append(previous)

    if not found_before:
        raise RuntimeError(f"asset_before not found in registry: {before_id}")

    updated_assets = _replace_or_append_asset(updated_assets, after)
    registry["assets"] = updated_assets
    reg_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    registry_errors = check_registry(root)
    if registry_errors:
        raise RuntimeError("updated registry failed production_asset_gate: " + "; ".join(registry_errors))
    return reg_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply an approved production asset change to production_assets.json.")
    parser.add_argument("--project-root", default=str(project_root()))
    parser.add_argument("--change-file", required=True)
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    change_file = Path(args.change_file)
    if not change_file.is_absolute():
        change_file = root / change_file
    if not change_file.is_file():
        raise FileNotFoundError(f"change file not found: {change_file}")

    lease = acquire_active_l3_writer_lease(
        workflow_run_id=f"l3-single-asset-change-{change_file.stem}",
        workspace=change_file.parent,
        report_dir=change_file.parent,
        process_role="apply_production_asset_change",
    )
    try:
        path = apply_change(root, change_file)
    finally:
        release_active_l3_writer_lease(lease)
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
