"""Candidate-only contract for a future read-only L3 canonical-L2 probe.

This module deliberately does not open the canonical L2 path.  It builds and
validates a pre-registered probe plan; a later, separately authorized runner
must perform the actual read-only probe.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CANONICAL_L2_PATH = (
    "D:/work/quant/quant_mcp/quant/data_file/production_assets/duckdb/"
    "l2_stock_daily_data.duckdb"
)
CANONICAL_L2_TABLE = "STOCK_DAILY_DATA"
EXPECTED_L2_SHA256 = "34a88582b2e0e53fa5ececadf8500673ad44259dd7d2bc8e44d3c689c7905e4b"
TARGET_DATE_WINDOW = ("20260803", "20260804")
HASH_BUCKET_COUNT = 16
HASH_BUCKET_ID = 3
FUTURE_DENYLIST = {"index_2000_post10_close"}
NEGATIVE_SHIFT_RE = re.compile(r"shift\s*\(\s*-\d+\s*\)", re.IGNORECASE)
FORBIDDEN_INPUT_TOKENS = ("quarantine", "legacy", "sqlite", "odb.db", ".parquet", "snapshot")


@dataclass(frozen=True)
class ProbeScope:
    run_id: str
    workspace: str
    staging_dir: str
    report_dir: str
    target_date_window: tuple[str, str] = TARGET_DATE_WINDOW
    hash_bucket_count: int = HASH_BUCKET_COUNT
    hash_bucket_id: int = HASH_BUCKET_ID
    l2_path: str = CANONICAL_L2_PATH
    l2_table: str = CANONICAL_L2_TABLE
    expected_l2_sha256: str = EXPECTED_L2_SHA256


def build_probe_scope(root: str | Path, run_id: str) -> ProbeScope:
    root_path = Path(root).resolve()
    return ProbeScope(
        run_id=run_id,
        workspace=str(root_path / "workspace"),
        staging_dir=str(root_path / "staging"),
        report_dir=str(root_path / "reports"),
    )


def validate_scope(scope: ProbeScope) -> dict[str, Any]:
    errors: list[str] = []
    if scope.l2_path != CANONICAL_L2_PATH:
        errors.append("non_canonical_l2_path")
    if scope.l2_table != CANONICAL_L2_TABLE:
        errors.append("non_canonical_l2_table")
    if scope.expected_l2_sha256 != EXPECTED_L2_SHA256:
        errors.append("l2_sha256_not_locked")
    if not 0 <= scope.hash_bucket_id < scope.hash_bucket_count:
        errors.append("invalid_hash_bucket")
    if scope.target_date_window[0] > scope.target_date_window[1]:
        errors.append("invalid_trade_date_window")
    for value in (scope.workspace, scope.staging_dir, scope.report_dir):
        lowered = value.lower().replace("\\", "/")
        if any(token in lowered for token in FORBIDDEN_INPUT_TOKENS):
            errors.append("forbidden_input_or_quarantine_path")
    return {"passed": not errors, "errors": errors}


def build_read_only_sql(scope: ProbeScope) -> str:
    """Return the registered SQL text without opening or checking L2."""
    validate = validate_scope(scope)
    if not validate["passed"]:
        raise RuntimeError(f"probe scope rejected: {validate['errors']}")
    start, end = scope.target_date_window
    return (
        "SELECT * FROM STOCK_DAILY_DATA "
        f"WHERE trade_date BETWEEN DATE '{start[:4]}-{start[4:6]}-{start[6:]}' "
        f"AND DATE '{end[:4]}-{end[4:6]}-{end[6:]}' "
        f"AND MOD(ABS(HASH(stock_code)), {scope.hash_bucket_count}) = {scope.hash_bucket_id}"
    )


def validate_projection_columns(columns: list[str]) -> dict[str, Any]:
    rejected: list[dict[str, str]] = []
    for column in columns:
        lowered = column.lower()
        if column in FUTURE_DENYLIST:
            rejected.append({"column": column, "reason": "explicit_future_denylist"})
        elif NEGATIVE_SHIFT_RE.search(lowered):
            rejected.append({"column": column, "reason": "negative_shift"})
        elif "future" in lowered or "label" in lowered or "_post" in lowered:
            rejected.append({"column": column, "reason": "future_or_label_name"})
    qfq = [c for c in columns if "_qfq" in c.lower()]
    return {
        "passed": not rejected,
        "rejected": rejected,
        "qfq_column_count_observed": len(qfq),
        "required_qfq_contract": {
            "l2_source_technical": 74,
            "production_feature_technical": 76,
            "derived": 2,
            "gtja": 191,
        },
    }


def validate_probe_contract(
    scope: ProbeScope,
    *,
    runtime_passed: bool,
    process_gate_passed: bool,
    wal_gate_passed: bool,
    lease_gate_passed: bool,
    source_before: dict[str, Any] | None,
    source_after: dict[str, Any] | None,
    output_columns: list[str],
) -> dict[str, Any]:
    scope_result = validate_scope(scope)
    projection = validate_projection_columns(output_columns)
    errors = list(scope_result["errors"])
    if not runtime_passed:
        errors.append("runtime_provenance_gate_failed")
    if not process_gate_passed:
        errors.append("process_gate_failed")
    if not wal_gate_passed:
        errors.append("wal_gate_failed")
    if not lease_gate_passed:
        errors.append("writer_lease_gate_failed")
    if not projection["passed"]:
        errors.append("future_lineage_gate_failed")
    if source_before is None or source_after is None or source_before != source_after:
        errors.append("canonical_l2_fingerprint_drift_or_missing")
    return {
        "passed": not errors,
        "errors": errors,
        "scope": asdict(scope),
        "projection": projection,
        "source_fingerprint_equal": source_before is not None and source_before == source_after,
        "read_only": True,
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
    }


def build_candidate_handoff(scope: ProbeScope, code_sha256: str) -> dict[str, Any]:
    return {
        "status": "candidate_only_ready_for_audit",
        "run_id": scope.run_id,
        "candidate_code_sha256": code_sha256,
        "canonical_l2_path": CANONICAL_L2_PATH,
        "canonical_l2_sha256": EXPECTED_L2_SHA256,
        "probe_started": False,
        "business_inputs_read": False,
        "candidate_written": False,
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
        "allow_probe_only": False,
        "allow_full_rebuild": False,
        "allow_l3": False,
        "allow_next_layer_continue": False,
        "ready_for_audit_review": True,
        "rollback": "No business asset was opened or changed; any future probe failure quarantines its isolated workspace.",
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == "__main__":  # pragma: no cover - contract module has no execution mode
    raise SystemExit("candidate-only module: no probe execution entry is available")
