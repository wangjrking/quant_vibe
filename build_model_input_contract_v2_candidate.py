"""Materialize candidate-only L4 input contracts without touching active assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from model_input_contract_v2_candidate import build_contract, canonical_json_sha256, validate_contract


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "quant/main/config/prediction_manifests"
L3_FEATURE_DB = ROOT / "quant/data_file/production_assets/duckdb/l3_feature_current.duckdb"
L3_FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
MANIFESTS = {
    "1d": "executable_1d_open_return_l4_formal_20260619.json",
    "3d": "executable_3d_open_return_l4_formal_20260617.json",
    "5d": "executable_5d_open_return_l4_formal_20260620.json",
    "10d": "executable_10d_open_return_l4_formal_20260617.json",
}


def root_path(raw: str) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate-only L4 model input contracts.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with duckdb.connect(str(L3_FEATURE_DB), read_only=True) as connection:
        available = [row[1] for row in connection.execute(f"PRAGMA table_info('{L3_FEATURE_TABLE}')").fetchall()]

    contracts = {}
    for label_key, manifest_name in MANIFESTS.items():
        manifest_path = MANIFEST_DIR / manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = build_contract(
            label_key=label_key,
            production_model_asset_id=str(manifest["production_model_asset_id"]),
            metadata_path=root_path(manifest["production_model_metadata_path"]),
            model_path=root_path(manifest["production_model_path"]),
            available_l3_columns=available,
        )
        validate_contract(contract)
        contracts[label_key] = contract

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for label_key, contract in contracts.items():
        (output_dir / f"{label_key}_model_input_contract_v2_candidate.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "candidate_only": True,
        "approved_for_l5": False,
        "allow_next_layer_continue": False,
        "l3_feature_path": str(L3_FEATURE_DB),
        "l3_feature_table": L3_FEATURE_TABLE,
        "contract_hashes": {
            label_key: canonical_json_sha256(contract) for label_key, contract in contracts.items()
        },
        "input_dimensions": {label_key: contract["input_dimension"] for label_key, contract in contracts.items()},
    }
    (output_dir / "model_input_contract_v2_candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
