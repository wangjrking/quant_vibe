from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MAIN_DIR = ROOT / "quant" / "main"
MANIFEST_DIR = MAIN_DIR / "config" / "prediction_manifests"
REGISTRY_PATH = DATA_DIR / "asset_registry" / "production_assets.json"
PRODUCTION_MODELS_DIR = DATA_DIR / "production_assets" / "production_models" / "l4"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_production_model_asset_registration_20260802"

MODEL_SOURCES = {
    "1d": {
        "label": "executable_1d_open_return",
        "manifest": "executable_1d_open_return_l4_formal_20260619.json",
        "metadata": DATA_DIR / "reports" / "model_agent_standard_chain_tune_20260617" / "xgb_reg1d_d2_l5_fs80_standardchain_fold1fixed_20260617" / "models" / "model_fold09_metadata.json",
    },
    "3d": {
        "label": "executable_3d_open_return",
        "manifest": "executable_3d_open_return_l4_formal_20260617.json",
        "metadata": DATA_DIR / "reports" / "model_agent_3d_fixed4y_latestfold_retrain_v1_20260628" / "fixed4y_fs100_k3_t126_m8_lr003_d2_l8" / "fold15_17" / "models" / "model_fold17_metadata.json",
    },
    "5d": {
        "label": "executable_5d_open_return",
        "manifest": "executable_5d_open_return_l4_formal_20260620.json",
        "compat_manifest": "prod_liq_prime_one_v20260612_l4_formal.json",
        "metadata": DATA_DIR / "reports" / "model_agent_research_5d_fixed4y_fold08_fs120_k10_structure_scan_20260626" / "d3_l8_a01_lr004_n5000" / "fold08" / "models" / "model_fold08_metadata.json",
    },
    "10d": {
        "label": "executable_10d_open_return",
        "manifest": "executable_10d_open_return_l4_formal_20260617.json",
        "metadata": DATA_DIR / "reports" / "model_agent_research_10d_fixed4y_fold08_fs40_structure_scan_20260626" / "fs40_d3_l8" / "fold08" / "models" / "model_fold08_metadata.json",
    },
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_model_path(metadata_path: Path, metadata: dict) -> Path:
    raw = Path(str(metadata["model_path"]))
    if raw.is_absolute():
        return raw
    adjacent = (metadata_path.parent / raw).resolve()
    return adjacent if adjacent.exists() else (MAIN_DIR / raw).resolve()


def copy_with_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def atomic_write_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with temp.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)


def copy_bundle(label_key: str, metadata_source: Path, model_source: Path) -> dict:
    target_dir = PRODUCTION_MODELS_DIR / label_key
    stage_dir = PRODUCTION_MODELS_DIR / ".staging" / f"{label_key}-{uuid.uuid4().hex}"
    stage_dir.mkdir(parents=True, exist_ok=False)
    staged_metadata = stage_dir / "model_metadata.json"
    staged_model = stage_dir / "model.json"
    try:
        copy_with_fsync(metadata_source, staged_metadata)
        copy_with_fsync(model_source, staged_model)
        source_metadata_sha = file_sha256(metadata_source)
        source_model_sha = file_sha256(model_source)
        if file_sha256(staged_metadata) != source_metadata_sha or file_sha256(staged_model) != source_model_sha:
            raise RuntimeError(f"staged production model hash mismatch: {label_key}")

        if target_dir.exists():
            target_metadata = target_dir / "model_metadata.json"
            target_model = target_dir / "model.json"
            if not target_metadata.is_file() or not target_model.is_file():
                raise RuntimeError(f"production model target is incomplete: {target_dir}")
            if file_sha256(target_metadata) != source_metadata_sha or file_sha256(target_model) != source_model_sha:
                raise RuntimeError(f"production model target conflicts with immutable source: {target_dir}")
            shutil.rmtree(stage_dir)
        else:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_dir, target_dir)

        target_metadata = target_dir / "model_metadata.json"
        target_model = target_dir / "model.json"
        if file_sha256(metadata_source) != source_metadata_sha or file_sha256(model_source) != source_model_sha:
            raise RuntimeError(f"source changed during production promotion: {label_key}")
        if file_sha256(target_metadata) != source_metadata_sha or file_sha256(target_model) != source_model_sha:
            raise RuntimeError(f"production copy verification failed: {label_key}")
        return {
            "metadata_source": metadata_source,
            "model_source": model_source,
            "metadata_destination": target_metadata,
            "model_destination": target_model,
            "metadata_sha256": source_metadata_sha,
            "model_sha256": source_model_sha,
            "metadata_size": target_metadata.stat().st_size,
            "model_size": target_model.stat().st_size,
        }
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)


def archive_manifest(path: Path) -> Path:
    archive = path.with_name(path.stem + "_archive_before_20260802_production_model_binding.json")
    if not archive.exists():
        shutil.copy2(path, archive)
    return archive


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    registry_before_sha = file_sha256(REGISTRY_PATH)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    before_manifest_sha: dict[str, str] = {}
    promotion: dict[str, dict] = {}
    manifest_archives: dict[str, str] = {}

    # Preflight every source before writing registry or manifest state.
    prepared: dict[str, dict] = {}
    for key, spec in MODEL_SOURCES.items():
        metadata_path = Path(spec["metadata"])
        if not metadata_path.is_file():
            raise RuntimeError(f"production source metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_path = resolve_model_path(metadata_path, metadata)
        if not model_path.is_file():
            raise RuntimeError(f"production source model is missing: {model_path}")
        prepared[key] = {**spec, "metadata_path": metadata_path, "model_path": model_path}
        for name in (spec["manifest"], *([spec["compat_manifest"]] if spec.get("compat_manifest") else [])):
            manifest_path = MANIFEST_DIR / name
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("approval_status") != "approved_for_l5" or manifest.get("asset_role") != "l4_formal_prediction_asset":
                raise RuntimeError(f"not an approved formal manifest: {manifest_path}")
            before_manifest_sha[name] = file_sha256(manifest_path)

    for key, spec in prepared.items():
        copied = copy_bundle(key, spec["metadata_path"], spec["model_path"])
        asset_id = f"prod_l4_model_{key}_immutable_20260802"
        promotion[key] = {**copied, "asset_id": asset_id, "label": spec["label"]}

        registry_asset = {
            "asset_id": asset_id,
            "track": "production",
            "layer": "L4_model_artifact",
            "asset_type": "xgboost_model_bundle",
            "owner_agent": "model-agent",
            "consumer_agents": ["model-agent", "audit-agent"],
            "status": "production_active",
            "allowed_for_main_workflow": True,
            "asset_path": relative_to_root(copied["metadata_destination"]),
            "model_metadata_path": relative_to_root(copied["metadata_destination"]),
            "model_path": relative_to_root(copied["model_destination"]),
            "model_metadata_sha256": copied["metadata_sha256"],
            "model_sha256": copied["model_sha256"],
            "model_metadata_size": copied["metadata_size"],
            "model_size": copied["model_size"],
            "source_metadata_path": relative_to_root(copied["metadata_source"]),
            "source_model_path": relative_to_root(copied["model_source"]),
            "source_before_after_sha256_verified": True,
            "lineage": {
                "label": spec["label"],
                "model_type": "xgboost_booster",
                "copy_method": "chunked_copy_fsync_source_before_after_sha256_atomic_promote",
                "production_copy_is_content_identical": True,
            },
            "created_at": now_iso(),
        }
        registry["assets"] = [item for item in registry.get("assets", []) if item.get("asset_id") != asset_id]
        registry["assets"].append(registry_asset)

    registry["updated_at"] = now_iso()
    atomic_write_json(REGISTRY_PATH, registry)

    for key, spec in prepared.items():
        binding = promotion[key]
        manifest_names = [spec["manifest"]]
        if spec.get("compat_manifest"):
            manifest_names.append(spec["compat_manifest"])
        for name in manifest_names:
            manifest_path = MANIFEST_DIR / name
            archive = archive_manifest(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(
                {
                    "production_model_asset_id": binding["asset_id"],
                    "production_model_metadata_path": relative_to_root(binding["metadata_destination"]),
                    "production_model_path": relative_to_root(binding["model_destination"]),
                    "production_model_metadata_sha256": binding["metadata_sha256"],
                    "production_model_sha256": binding["model_sha256"],
                    "production_model_binding": {
                        "binding_type": "immutable_production_model_copy",
                        "source_before_after_sha256_verified": True,
                        "copy_method": "chunked_copy_fsync_source_before_after_sha256_atomic_promote",
                        "bound_at": now_iso(),
                    },
                }
            )
            atomic_write_json(manifest_path, manifest)
            manifest_archives[name] = str(archive)

    result = {
        "task": "production_model_asset_registration_and_formal_manifest_binding",
        "generated_at": now_iso(),
        "registry_path": str(REGISTRY_PATH),
        "registry_before_sha256": registry_before_sha,
        "registry_after_sha256": file_sha256(REGISTRY_PATH),
        "source_and_production_assets": {
            key: {
                **{field: (str(value) if isinstance(value, Path) else value) for field, value in value.items()},
                "metadata_destination_sha256": file_sha256(value["metadata_destination"]),
                "model_destination_sha256": file_sha256(value["model_destination"]),
            }
            for key, value in promotion.items()
        },
        "manifest_before_sha256": before_manifest_sha,
        "manifest_after_sha256": {name: file_sha256(MANIFEST_DIR / name) for name in before_manifest_sha},
        "manifest_archives": manifest_archives,
        "prediction_tables_modified": False,
        "model_file_contents_modified": False,
        "strategy_parameters_modified": False,
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    atomic_write_json(REPORT_DIR / "production_model_asset_registration_report.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
