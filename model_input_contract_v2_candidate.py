"""Candidate-only v2 contract for production L4 model inputs.

This module is intentionally not imported by the active L4 refresh entrypoint.
It defines the validation that must pass before a later, separately approved
integration may read a wide L3 feature table for a production model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import xgboost as xgb


CONTRACT_VERSION = "model_input_contract_v2_candidate_20260804"
FORBIDDEN_FUTURE_COLUMNS = ("index_2000_post10_close",)
QFQ_ALIAS_BASES = {
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "atr",
    "macd",
    "macdsignal",
    "macdhist",
    "kdj",
    "wr",
    "mfi",
    "rsi",
    "tema",
    "dema",
    "dema_10",
    "t3",
}


class ModelInputContractError(RuntimeError):
    """Raised when an L4 model input contract cannot be proven safe."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_strings(values: Iterable[Any], *, field_name: str) -> list[str]:
    ordered = [str(value) for value in values]
    if not ordered:
        raise ModelInputContractError(f"{field_name} must not be empty")
    if len(ordered) != len(set(ordered)):
        raise ModelInputContractError(f"{field_name} contains duplicate columns")
    return ordered


def resolve_explicit_qfq_source(model_feature: str, available_columns: set[str]) -> str | None:
    """Map legacy saved-model names only to their explicit *_qfq L3 columns."""
    if model_feature in available_columns:
        return model_feature
    if model_feature.startswith("gtja_alpha"):
        qfq_name = f"{model_feature}_qfq"
        return qfq_name if qfq_name in available_columns else None
    if model_feature in QFQ_ALIAS_BASES:
        qfq_name = f"{model_feature}_qfq"
        return qfq_name if qfq_name in available_columns else None
    return None


def build_contract(
    *,
    label_key: str,
    production_model_asset_id: str,
    metadata_path: Path,
    model_path: Path,
    available_l3_columns: Iterable[str],
) -> dict[str, Any]:
    """Build a frozen, exact-order model input contract from production files."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_columns = _ordered_strings(metadata.get("feature_columns", []), field_name="metadata.feature_columns")

    forbidden = set(FORBIDDEN_FUTURE_COLUMNS)
    forbidden_hits = sorted(forbidden.intersection(metadata_columns))
    if forbidden_hits:
        raise ModelInputContractError(f"forbidden future columns enter model input: {forbidden_hits}")

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    if booster.feature_names is None:
        raise ModelInputContractError("Booster.feature_names is required for three-way feature verification")
    booster_columns = _ordered_strings(booster.feature_names, field_name="Booster.feature_names")
    if metadata_columns != booster_columns:
        raise ModelInputContractError("metadata.feature_columns and Booster.feature_names differ in order or content")

    available = {str(column) for column in available_l3_columns}
    bindings: list[dict[str, str]] = []
    unresolved: list[str] = []
    for feature in metadata_columns:
        source = resolve_explicit_qfq_source(feature, available)
        if source is None:
            unresolved.append(feature)
        else:
            bindings.append({"model_column": feature, "source_column": source})
    if unresolved:
        raise ModelInputContractError(f"active L3 schema cannot satisfy ordered model input: {unresolved[:20]}")

    ordered_sha = canonical_json_sha256(metadata_columns)
    return {
        "contract_version": CONTRACT_VERSION,
        "candidate_only": True,
        "approved_for_l5": False,
        "allow_next_layer_continue": False,
        "label_key": label_key,
        "production_model_asset_id": production_model_asset_id,
        "metadata_path": str(metadata_path),
        "model_path": str(model_path),
        "metadata_sha256": file_sha256(metadata_path),
        "model_sha256": file_sha256(model_path),
        "forbidden_future_columns": list(FORBIDDEN_FUTURE_COLUMNS),
        "allowed_columns_ordered": metadata_columns,
        "allowed_columns_sha256": ordered_sha,
        "metadata_feature_columns_sha256": canonical_json_sha256(metadata_columns),
        "booster_feature_names_sha256": canonical_json_sha256(booster_columns),
        "input_column_bindings": bindings,
        "input_dimension": len(metadata_columns),
        "read_policy": {
            "projection_only": True,
            "select_star_forbidden": True,
            "wide_frame_then_drop_columns_forbidden": True,
            "matrix_columns_must_match_allowed_order_exactly": True,
        },
    }


def validate_contract(contract: dict[str, Any]) -> None:
    """Fail closed unless the contract's three feature identities agree exactly."""
    allowed = _ordered_strings(contract.get("allowed_columns_ordered", []), field_name="contract.allowed_columns_ordered")
    expected = canonical_json_sha256(allowed)
    for field_name in (
        "allowed_columns_sha256",
        "metadata_feature_columns_sha256",
        "booster_feature_names_sha256",
    ):
        if contract.get(field_name) != expected:
            raise ModelInputContractError(f"three-way feature hash mismatch: {field_name}")
    forbidden_hits = sorted(set(contract.get("forbidden_future_columns", [])).intersection(allowed))
    if forbidden_hits:
        raise ModelInputContractError(f"forbidden future columns enter contract allowlist: {forbidden_hits}")
    bindings = contract.get("input_column_bindings")
    if not isinstance(bindings, list) or [item.get("model_column") for item in bindings] != allowed:
        raise ModelInputContractError("input bindings do not preserve the exact model-column order")
    if contract.get("input_dimension") != len(allowed):
        raise ModelInputContractError("input dimension does not match ordered allowlist")


def build_projection_sql(*, factor_table: str, contract: dict[str, Any]) -> str:
    """Return the only permitted feature query: explicit projection in model order."""
    validate_contract(contract)
    columns = ['"trade_date"', '"stock_code"']
    for binding in contract["input_column_bindings"]:
        source = str(binding["source_column"]).replace('"', '""')
        model = str(binding["model_column"]).replace('"', '""')
        columns.append(f'"{source}" AS "{model}"')
    table = str(factor_table).replace('"', '""')
    return f"SELECT {', '.join(columns)} FROM \"{table}\" WHERE \"trade_date\" = ? ORDER BY \"stock_code\""


def validate_runtime_matrix_columns(columns: Iterable[str], contract: dict[str, Any]) -> None:
    """Reject a SELECT * frame, drop-column frame, or any order/dimension drift."""
    validate_contract(contract)
    expected = list(contract["allowed_columns_ordered"])
    actual = [str(column) for column in columns]
    if actual != expected:
        raise ModelInputContractError(
            "runtime matrix columns must equal the ordered allowlist exactly; wide-frame projection is forbidden"
        )
