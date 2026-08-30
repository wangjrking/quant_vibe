"""Candidate-only L3 feature contract v2 governance.

This module is deliberately not wired into the active delivery entrypoint. It
provides the allowlist, lineage, and schema gates for audit before any active
feature rebuild or switch is authorized.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from adjustment_semantics import (
    FRONT_ADJUSTED_MARKET_PRICE_COLUMNS,
    prefer_explicit_qfq_columns,
    require_explicit_qfq_market_columns,
    validate_strategy_output_field_names,
)
from build_production_factor_parts import (
    KEY_COLUMNS,
    is_future_or_label_column,
    is_raw_factor_column,
)
from gtja_alpha_workflow import GTJA_ALPHA_COLUMNS


FEATURE_CONTRACT_V2 = "l3_feature_contract_v2"
PRODUCTION_RAW_ALLOWLIST_VERSION = "l3-production-raw-v2"
SEMANTIC_FUTURE_DERIVED_COLUMNS = frozenset({"index_2000_post10_close"})
PRODUCTION_FEATURE_COLUMN_COUNT_V2 = 838
PRODUCTION_FEATURE_QFQ_TECHNICAL_COUNT_V2 = 76
NEGATIVE_SHIFT_RE = re.compile(r"\.shift\(\s*-\s*\d+\s*\)")


class FeatureContractV2Error(ValueError):
    """Raised when a candidate violates the v2 feature contract."""


def _is_explicit_qfq_column(column: str) -> bool:
    """Recognize both suffix and period-qualified qfq names."""

    return column.endswith("_qfq") or "_qfq_" in column


def candidate_production_raw_columns(raw_columns: Sequence[str]) -> list[str]:
    """Select raw inputs while explicitly excluding semantic future fields."""

    normalized = list(dict.fromkeys(str(column) for column in raw_columns))
    require_explicit_qfq_market_columns(normalized, context="feature contract v2 raw schema")
    preferred = prefer_explicit_qfq_columns(normalized)
    selected = [
        column
        for column in preferred
        if column not in SEMANTIC_FUTURE_DERIVED_COLUMNS
        and column not in GTJA_ALPHA_COLUMNS
        and is_raw_factor_column(column)
    ]
    validate_strategy_output_field_names(selected, context="feature contract v2 raw schema output")
    forbidden = sorted(set(selected) & SEMANTIC_FUTURE_DERIVED_COLUMNS)
    if forbidden:
        raise FeatureContractV2Error(
            "feature contract v2 raw allowlist contains semantic future fields: "
            + ", ".join(forbidden)
        )
    return selected


def validate_future_lineage(lineage: Mapping[str, str]) -> dict:
    """Classify negative-shift lineage and fail closed for unknown fields."""

    excluded: list[dict[str, str]] = []
    errors: list[str] = []
    for field, expression in lineage.items():
        field_name = str(field)
        expression_text = str(expression)
        if not NEGATIVE_SHIFT_RE.search(expression_text):
            continue
        if field_name in SEMANTIC_FUTURE_DERIVED_COLUMNS or is_future_or_label_column(field_name):
            excluded.append({"field": field_name, "expression": expression_text})
        else:
            errors.append(
                f"unregistered negative shift lineage must be denied: {field_name}: {expression_text}"
            )
    if errors:
        raise FeatureContractV2Error("; ".join(errors))
    return {"negative_shift_fields": excluded, "errors": []}


def audit_candidate_feature_schema(
    columns: Sequence[str],
    *,
    expected_column_count: int = PRODUCTION_FEATURE_COLUMN_COUNT_V2,
) -> dict:
    """Validate the candidate output schema without touching any asset."""

    normalized = [str(column) for column in columns]
    duplicate_columns = sorted({column for column in normalized if normalized.count(column) > 1})
    naked_prices = sorted(set(normalized) & {"open", "high", "low", "close", "pre_close"})
    naked_gtja = sorted(set(normalized) & set(GTJA_ALPHA_COLUMNS))
    future_columns = sorted(
        {
            column
            for column in normalized
            if column in SEMANTIC_FUTURE_DERIVED_COLUMNS or is_future_or_label_column(column)
        }
    )
    qfq_prices = sorted(set(normalized) & set(FRONT_ADJUSTED_MARKET_PRICE_COLUMNS))
    gtja_qfq = sorted(
        set(normalized) & {f"{column}_qfq" for column in GTJA_ALPHA_COLUMNS}
    )
    qfq_technical = sorted(
        column
        for column in normalized
        if _is_explicit_qfq_column(column)
        and column not in FRONT_ADJUSTED_MARKET_PRICE_COLUMNS
        and column not in gtja_qfq
    )
    errors: list[str] = []
    if len(normalized) != expected_column_count:
        errors.append(f"column_count={len(normalized)} expected={expected_column_count}")
    if duplicate_columns:
        errors.append(f"duplicate_columns={duplicate_columns}")
    if naked_prices:
        errors.append(f"naked_price_columns={naked_prices}")
    if naked_gtja:
        errors.append(f"naked_gtja_columns={naked_gtja}")
    if future_columns:
        errors.append(f"future_or_label_columns={future_columns}")
    if qfq_prices != sorted(FRONT_ADJUSTED_MARKET_PRICE_COLUMNS):
        errors.append(f"qfq_price_count={len(qfq_prices)} expected=5")
    if len(qfq_technical) != PRODUCTION_FEATURE_QFQ_TECHNICAL_COUNT_V2:
        errors.append(
            f"qfq_technical_count={len(qfq_technical)} "
            f"expected={PRODUCTION_FEATURE_QFQ_TECHNICAL_COUNT_V2}"
        )
    if len(gtja_qfq) != 191:
        errors.append(f"gtja_qfq_count={len(gtja_qfq)} expected=191")
    if errors:
        raise FeatureContractV2Error("; ".join(errors))
    return {
        "contract_id": FEATURE_CONTRACT_V2,
        "allowlist_version": PRODUCTION_RAW_ALLOWLIST_VERSION,
        "column_count": len(normalized),
        "qfq_price_count": len(qfq_prices),
        "qfq_technical_count": len(qfq_technical),
        "l2_source_qfq_technical_count": 74,
        "production_feature_qfq_technical_count": len(qfq_technical),
        "gtja_qfq_count": len(gtja_qfq),
        "naked_price_columns": [],
        "naked_gtja_columns": [],
        "future_or_label_columns": [],
        "duplicate_columns": [],
        "excluded_semantic_future_columns": sorted(SEMANTIC_FUTURE_DERIVED_COLUMNS),
    }


def build_feature_contract_v2_candidate(
    *,
    workflow_run_id: str,
    target_trade_date: str,
    base_active_feature: dict,
    candidate_feature: dict,
    candidate_columns: Sequence[str],
    lineage: Mapping[str, str],
    evidence_paths: Sequence[str],
) -> dict:
    """Build an audit-only contract; it has no active mutation capability."""

    lineage_audit = validate_future_lineage(lineage)
    schema_audit = audit_candidate_feature_schema(candidate_columns)
    return {
        "schema_version": 2,
        "contract_id": FEATURE_CONTRACT_V2,
        "workflow_run_id": workflow_run_id,
        "layer": "L3",
        "target_trade_date": str(target_trade_date),
        "status": "candidate_only_pending_audit",
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "active_mutation": {
            "active_feature_modified": False,
            "active_label_modified": False,
            "registry_modified": False,
            "pair_change_called": False,
            "training_called": False,
        },
        "base_active_feature": base_active_feature,
        "candidate_feature": candidate_feature,
        "production_raw_allowlist": {
            "version": PRODUCTION_RAW_ALLOWLIST_VERSION,
            "semantic_future_denylist": sorted(SEMANTIC_FUTURE_DERIVED_COLUMNS),
            "negative_shift_policy": "deny shift(-n), n>0, from production feature output",
        },
        "lineage_audit": lineage_audit,
        "schema_audit": schema_audit,
        "evidence_paths": list(evidence_paths),
        "apply_prerequisites": [
            "audit-agent read-only approval",
            "model-agent feature manifest compatibility review",
            "fresh target-date and full-history candidate rebuild as required",
            "real rollback snapshot and atomic active switch authorization",
        ],
    }
