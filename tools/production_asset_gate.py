"""Fail-closed guard for production asset changes.

The tool validates proposed production asset changes before they are allowed to
enter the main workflow. It is a governance guard only and does not mutate
business assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_CHANGE_FIELDS = {
    "change_id",
    "change_type",
    "layer",
    "owner_agent",
    "asset_after",
    "audit_required",
    "audit_status",
    "audit_record",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def registry_path(root: Path) -> Path:
    return root / "quant" / "data_file" / "asset_registry" / "production_assets.json"


def check_change(root: Path, change: dict[str, Any], errors: list[str]) -> None:
    missing = REQUIRED_CHANGE_FIELDS - set(change)
    change_id = change.get("change_id", "<unknown>")
    if missing:
        errors.append(f"change {change_id} missing fields: {', '.join(sorted(missing))}")

    if change.get("audit_required") is not True:
        errors.append(f"change {change_id} must set audit_required=true")
    if change.get("audit_status") not in {"passed", "approved", "approved_for_production"}:
        errors.append(f"change {change_id} audit_status is not passed/approved")

    audit_record = resolve(root, change.get("audit_record"))
    if audit_record is None:
        errors.append(f"change {change_id} missing audit_record")
    elif not audit_record.is_file():
        errors.append(f"change {change_id} audit_record not found: {audit_record}")

    asset_after = change.get("asset_after")
    if isinstance(asset_after, dict):
        if asset_after.get("track") != "production":
            errors.append(f"change {change_id} asset_after.track must be production")
        if asset_after.get("allowed_for_main_workflow") is not True:
            errors.append(f"change {change_id} asset_after must be allowed_for_main_workflow")
        if not asset_after.get("audit_record"):
            errors.append(f"change {change_id} asset_after missing audit_record")


def check_change_file(root: Path, path: Path) -> list[str]:
    data = load_json(path)
    changes = data if isinstance(data, list) else data.get("production_asset_changes", [data])
    errors: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            errors.append("change entry must be object")
            continue
        check_change(root, change, errors)
    return errors


def check_registry(root: Path) -> list[str]:
    data = load_json(registry_path(root))
    errors: list[str] = []
    for asset in data.get("assets", []):
        asset_id = asset.get("asset_id", "<unknown>")
        if asset.get("track") != "production":
            errors.append(f"asset {asset_id} is not production track")
        if asset.get("allowed_for_main_workflow") is not True:
            errors.append(f"asset {asset_id} not allowed for main workflow")
        audit_record = resolve(root, asset.get("audit_record"))
        if audit_record is None or not audit_record.is_file():
            errors.append(f"asset {asset_id} audit_record not found: {asset.get('audit_record')}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate production asset change gate.")
    parser.add_argument(
        "--project-root",
        default=str(project_root()),
        help="Path to D:/work/quant/quant_mcp. Defaults to inferred project root.",
    )
    parser.add_argument("--change-file", default=None, help="JSON change request to validate.")
    parser.add_argument("--check-registry", action="store_true", help="Validate current production registry.")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    errors: list[str] = []
    if args.check_registry:
        errors.extend(check_registry(root))
    if args.change_file:
        path = Path(args.change_file)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"change file not found: {path}")
        else:
            errors.extend(check_change_file(root, path))
    if not args.check_registry and not args.change_file:
        errors.append("must pass --check-registry or --change-file")

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("PASS production_asset_gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
