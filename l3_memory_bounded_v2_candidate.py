"""Candidate-only planner and gates for a memory-bounded L3 v2 rebuild.

This module plans work and validates fragment metadata. It intentionally does
not open production assets for writing and does not execute a rebuild.
"""

from __future__ import annotations

import hashlib
import math
import sys
import sysconfig
import unittest as _unittest
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


MEMORY_BOUNDED_V2 = "l3-memory-bounded-v2"
DEFAULT_DATE_CHUNK_DAYS = 20
DEFAULT_STOCK_BUCKETS = 64
DEFAULT_MEMORY_BUDGET_BYTES = 4 * 1024**3
PROJECT_PYTHON = "D:/work/quant/quant_mcp/runtime_candidates/my_quant_copy_20260804/python.exe"
PROJECT_VENV = "D:/work/quant/quant_mcp/runtime_candidates/my_quant_copy_20260804"
SYSTEM_HARD_STOP_BYTES = 20 * 1024**3
STARTUP_AVAILABLE_BYTES = 64 * 1024**3
PARENT_BUDGET_BYTES = 4 * 1024**3
WORKER_BUDGET_BYTES = 24 * 1024**3
RESERVE_BYTES = 16 * 1024**3
EXECUTOR_STAGES = (
    "precheck",
    "raw_fragment",
    "raw_fanin_merge",
    "gtja_fragment",
    "gtja_fanin_merge",
    "feature_ctas",
    "label_ctas",
    "global_validation",
    "quarantine",
)


@dataclass(frozen=True)
class MemoryBudgetContract:
    workers: int = 1
    system_hard_stop_bytes: int = SYSTEM_HARD_STOP_BYTES
    startup_available_bytes: int = STARTUP_AVAILABLE_BYTES
    parent_budget_bytes: int = PARENT_BUDGET_BYTES
    worker_budget_bytes: int = WORKER_BUDGET_BYTES
    reserve_bytes: int = RESERVE_BYTES

    @property
    def dispatch_min_available_bytes(self) -> int:
        return (
            self.system_hard_stop_bytes
            + self.parent_budget_bytes
            + self.workers * self.worker_budget_bytes
            + self.reserve_bytes
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dispatch_min_available_bytes"] = self.dispatch_min_available_bytes
        payload["formula"] = "system_hard_stop + parent_budget + workers*worker_budget + reserve"
        return payload


def validate_runtime_contract(
    *,
    executable: str | None = None,
    prefix: str | None = None,
    base_prefix: str | None = None,
    stdlib_path: str | None = None,
    unittest_path: str | None = None,
    expected_executable: str = PROJECT_PYTHON,
    expected_prefix: str = PROJECT_VENV,
    expected_base_prefix: str = PROJECT_VENV,
) -> dict[str, Any]:
    """Reject conda/launcher and mixed-standard-library runtimes."""

    actual_executable = str(executable or sys.executable).replace("\\", "/")
    expected = str(expected_executable).replace("\\", "/")
    actual_prefix = str(prefix or sys.prefix).replace("\\", "/")
    actual_base_prefix = str(base_prefix or getattr(sys, "base_prefix", "")).replace("\\", "/")
    actual_stdlib_path = str(stdlib_path or sysconfig.get_paths().get("stdlib", "")).replace("\\", "/")
    actual_unittest_path = str(unittest_path or getattr(_unittest, "__file__", "")).replace("\\", "/")
    if actual_executable.lower() != expected.lower():
        raise RuntimeError(f"project runtime mismatch: expected={expected}, actual={actual_executable}")
    if actual_prefix.lower() != str(expected_prefix).replace("\\", "/").lower():
        raise RuntimeError(f"project prefix mismatch: {actual_prefix}")
    if actual_base_prefix.lower() != str(expected_base_prefix).replace("\\", "/").lower():
        raise RuntimeError(f"project base_prefix mismatch: {actual_base_prefix}")
    if not actual_stdlib_path.lower().startswith(PROJECT_VENV.lower()):
        raise RuntimeError(f"project stdlib provenance mismatch: {actual_stdlib_path}")
    if not actual_unittest_path.lower().startswith(PROJECT_VENV.lower()):
        raise RuntimeError(f"project unittest provenance mismatch: {actual_unittest_path}")
    return {
        "sys_executable": actual_executable,
        "sys_prefix": actual_prefix,
        "sys_base_prefix": actual_base_prefix,
        "stdlib_path": actual_stdlib_path,
        "unittest_path": actual_unittest_path,
        "expected_executable": expected,
        "expected_prefix": str(expected_prefix).replace("\\", "/"),
        "expected_base_prefix": str(expected_base_prefix).replace("\\", "/"),
        "status": "passed",
    }


def validate_memory_budget_contract(contract: MemoryBudgetContract) -> dict[str, Any]:
    if contract.workers != 1:
        raise ValueError("memory-bounded v2 executor requires workers=1")
    if contract.startup_available_bytes != STARTUP_AVAILABLE_BYTES:
        raise ValueError("startup gate must remain 64GiB")
    if contract.system_hard_stop_bytes != SYSTEM_HARD_STOP_BYTES:
        raise ValueError("system hard stop must remain 20GiB")
    if contract.dispatch_min_available_bytes != STARTUP_AVAILABLE_BYTES:
        raise ValueError("memory budget formula must equal the 64GiB startup gate")
    return {"status": "passed", "contract": contract.as_dict()}


def validate_worker_recovery(
    metrics: Mapping[str, Any],
    *,
    contract: MemoryBudgetContract | None = None,
) -> dict[str, Any]:
    """Fail closed when a short-lived worker crossed or failed a resource gate."""

    budget = contract or MemoryBudgetContract()
    errors: list[str] = []
    if int(metrics.get("rss_peak_bytes", -1)) > budget.worker_budget_bytes:
        errors.append("worker_rss_budget_exceeded")
    if int(metrics.get("system_available_before_bytes", -1)) < budget.dispatch_min_available_bytes:
        errors.append("dispatch_available_below_budget")
    if int(metrics.get("system_available_min_bytes", -1)) < budget.system_hard_stop_bytes:
        errors.append("system_hard_stop_crossed")
    if int(metrics.get("recovery_timeout_seconds", -1)) > 120:
        errors.append("recovery_timeout_exceeded")
    if metrics.get("recovery_completed") is not True:
        errors.append("worker_recovery_incomplete")
    if int(metrics.get("system_available_after_exit_bytes", -1)) < budget.startup_available_bytes:
        errors.append("post_exit_available_not_recovered")
    if errors:
        raise ValueError("worker recovery gate failed: " + ", ".join(errors))
    return {"status": "passed", "errors": [], "memory_gate_status": "passed"}


def validate_short_lived_worker_lifecycle(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the fixture state machine without starting a process."""

    expected = ("owned", "launching", "running", "completed", "joined", "validated")
    actual = tuple(str(event.get("state")) for event in events)
    if actual != expected:
        raise ValueError(f"worker lifecycle gate failed: expected={expected}, actual={actual}")
    pids = {int(event.get("pid", -1)) for event in events}
    if len(pids) != 1 or -1 in pids:
        raise ValueError("worker lifecycle gate failed: pid identity is not stable")
    if any(event.get("residual") is True for event in events):
        raise ValueError("worker lifecycle gate failed: residual process")
    return {"status": "passed", "pid": next(iter(pids)), "states": list(actual)}


def validate_stage_transition(previous: str | None, current: str) -> dict[str, Any]:
    if current not in EXECUTOR_STAGES:
        raise ValueError(f"unknown executor stage: {current}")
    if previous is not None:
        if previous not in EXECUTOR_STAGES:
            raise ValueError(f"unknown previous executor stage: {previous}")
        if EXECUTOR_STAGES.index(current) < EXECUTOR_STAGES.index(previous):
            raise ValueError(f"stage regression: {previous} -> {current}")
    return {"status": "passed", "previous": previous, "current": current}


def validate_candidate_stage_state(stage: str, state: dict[str, Any]) -> dict[str, Any]:
    validate_stage_transition(None, stage)
    errors: list[str] = []
    if state.get("candidate_only") is not True:
        errors.append("candidate_only_required")
    if state.get("active_write") is True:
        errors.append("active_write_forbidden")
    if state.get("label_write") is True:
        errors.append("label_write_forbidden")
    if state.get("registry_write") is True:
        errors.append("registry_write_forbidden")
    if state.get("production_paths_opened") is True:
        errors.append("production_paths_opened")
    if state.get("wal_present") is True:
        errors.append("wal_present")
    if state.get("unknown_writer") is True:
        errors.append("unknown_writer")
    if errors:
        raise ValueError("candidate stage gate failed: " + ", ".join(errors))
    return {"status": "passed", "stage": stage, "errors": []}


def validate_fanin_equivalence(
    *,
    source_rows: int,
    fragment_rows: int,
    merged_rows: int,
    source_key_hash: str,
    merged_key_hash: str,
    duplicate_key_groups: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if not (source_rows == fragment_rows == merged_rows):
        errors.append("row_count_mismatch")
    if source_key_hash != merged_key_hash:
        errors.append("key_hash_mismatch")
    if duplicate_key_groups != 0:
        errors.append("duplicate_key_groups_nonzero")
    if errors:
        raise ValueError("fan-in equivalence gate failed: " + ", ".join(errors))
    return {"status": "passed", "errors": [], "approved_for_next_stage": True}


def build_fragment_metadata(
    *,
    fragment_id: str,
    source_rows: int,
    output_rows: int,
    columns: Sequence[str],
    key_hash: str,
    duplicate_key_groups: int,
    bj_rows: int,
    future_or_label_columns: Sequence[str],
    shard_sha256: str,
    size_bytes: int,
    mtime_ns: int,
) -> dict[str, Any]:
    """Build deterministic fixture metadata; never opens the shard path."""

    if not fragment_id or not columns or not key_hash or not shard_sha256:
        raise ValueError("fragment metadata requires id, columns, key hash and shard hash")
    if int(size_bytes) <= 0 or int(mtime_ns) <= 0:
        raise ValueError("fragment metadata requires positive size and mtime")
    schema_hash = hashlib.sha256("\n".join(str(value) for value in columns).encode("utf-8")).hexdigest()
    metadata = {
        "fragment_id": fragment_id,
        "source_rows": int(source_rows),
        "output_rows": int(output_rows),
        "column_count": len(columns),
        "schema_hash": schema_hash,
        "key_hash": key_hash,
        "duplicate_key_groups": int(duplicate_key_groups),
        "bj_rows": int(bj_rows),
        "future_or_label_columns": list(future_or_label_columns),
        "shard_sha256": shard_sha256,
        "size_bytes": int(size_bytes),
        "mtime_ns": int(mtime_ns),
        "candidate_only": True,
        "approved_for_candidate": False,
        "approved_for_active": False,
        "approved_for_downstream": False,
    }
    return metadata


def validate_fragment_metadata(metadata: Mapping[str, Any], *, expected_columns: Sequence[str]) -> dict[str, Any]:
    """Validate fixture metadata before a fragment is eligible for fan-in."""

    expected_schema_hash = hashlib.sha256("\n".join(str(value) for value in expected_columns).encode("utf-8")).hexdigest()
    errors: list[str] = []
    if metadata.get("candidate_only") is not True:
        errors.append("candidate_only_required")
    if int(metadata.get("source_rows", -1)) != int(metadata.get("output_rows", -2)):
        errors.append("source_output_rows_mismatch")
    if int(metadata.get("duplicate_key_groups", -1)) != 0:
        errors.append("duplicate_key_groups_nonzero")
    if int(metadata.get("bj_rows", -1)) != 0:
        errors.append("bj_rows_nonzero")
    if list(metadata.get("future_or_label_columns", [])):
        errors.append("future_or_label_columns_present")
    if str(metadata.get("schema_hash")) != expected_schema_hash:
        errors.append("schema_hash_mismatch")
    if int(metadata.get("column_count", -1)) != len(expected_columns):
        errors.append("column_count_mismatch")
    if not metadata.get("shard_sha256") or int(metadata.get("size_bytes", 0)) <= 0 or int(metadata.get("mtime_ns", 0)) <= 0:
        errors.append("shard_identity_incomplete")
    if errors:
        raise ValueError("fragment metadata gate failed: " + ", ".join(errors))
    return {"status": "passed", "approved_for_merge": True, "errors": []}


def validate_fanin_merge_contract(
    *,
    fan_in: int,
    child_count: int,
    source_rows: int,
    merged_rows: int,
    source_key_hash: str,
    merged_key_hash: str,
    duplicate_key_groups: int,
) -> dict[str, Any]:
    """Validate fixed 8/16-way hierarchical merge metadata."""

    if fan_in not in (8, 16):
        raise ValueError("fan-in must be exactly 8 or 16")
    if child_count <= 0 or child_count > fan_in:
        raise ValueError("fan-in child count is outside the fixed merge bound")
    return validate_fanin_equivalence(
        source_rows=source_rows,
        fragment_rows=source_rows,
        merged_rows=merged_rows,
        source_key_hash=source_key_hash,
        merged_key_hash=merged_key_hash,
        duplicate_key_groups=duplicate_key_groups,
    ) | {"fan_in": fan_in, "child_count": child_count}


def validate_gtja_chunk_contract(
    *,
    date_start: str,
    date_end: str,
    lookback_start: str,
    alpha_count: int,
    output_dates: int,
    expected_output_dates: int | None = None,
    each_alpha_date_once: bool = True,
    boundary_matches_reference: bool = True,
) -> dict[str, Any]:
    if lookback_start > date_start:
        raise ValueError("GTJA lookback_start cannot be after date_start")
    if alpha_count != 191:
        raise ValueError("GTJA chunk must retain all 191 qfq alphas")
    if output_dates <= 0 or date_end < date_start:
        raise ValueError("invalid GTJA chunk date bounds")
    if expected_output_dates is not None and output_dates != expected_output_dates:
        raise ValueError("GTJA output date count mismatch")
    if each_alpha_date_once is not True:
        raise ValueError("GTJA alpha/date multiplicity gate failed")
    if boundary_matches_reference is not True:
        raise ValueError("GTJA boundary output differs from single-block reference")
    return {
        "status": "passed",
        "date_start": date_start,
        "date_end": date_end,
        "lookback_start": lookback_start,
        "alpha_count": alpha_count,
        "output_dates": output_dates,
        "each_alpha_date_once": each_alpha_date_once,
        "boundary_matches_reference": boundary_matches_reference,
    }


def build_quarantine_manifest(*, run_id: str, workspace: str, reason: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workspace": workspace,
        "status": "quarantined",
        "reason": reason,
        "approved_for_candidate": False,
        "approved_for_active": False,
        "approved_for_downstream": False,
        "reuse_prohibited": True,
    }


def build_executor_contract(*, run_id: str, workspace: str, report_dir: str) -> dict[str, Any]:
    budget = MemoryBudgetContract()
    validate_memory_budget_contract(budget)
    return {
        "contract_id": MEMORY_BOUNDED_V2,
        "run_id": run_id,
        "workspace": workspace,
        "report_dir": report_dir,
        "execution_started": False,
        "probe_started": False,
        "full_rebuild_started": False,
        "business_asset_written": False,
        "production_paths_opened": False,
        "business_input_opened": False,
        "fixture_only": True,
        "active_switch": False,
        "label_write": False,
        "registry_change": False,
        "runtime": {
            "expected_python": PROJECT_PYTHON,
            "expected_prefix": PROJECT_VENV,
            "expected_base_prefix": PROJECT_VENV,
            "stdlib_and_unittest_must_be_under": PROJECT_VENV,
            "workers": 1,
            "short_lived_process": True,
            "conda_provenance_fail_closed": True,
        },
        "memory_budget": budget.as_dict(),
        "stages": list(EXECUTOR_STAGES),
        "raw_partition": {"date_chunk_days": 20, "stock_buckets": 64},
        "gtja": {"date_chunks_preserve_lookback": True, "alpha_count": 191, "alpha_suffix": "_qfq"},
        "candidate_only": True,
        "label_read_only": True,
        "failure_policy": "stop_dispatch_join_quarantine_reuse_prohibited",
        "global_gates": {
            "rows_and_key_domain_equal": True,
            "schema_hash_equal": True,
            "duplicate_zero": True,
            "bj_zero": True,
            "future_label_zero": True,
            "qfq_5_74_76_191": True,
            "label_maturity_not_forward_filled": True,
            "label_write_false": True,
            "active_label_registry_fingerprints_unchanged": True,
            "wal_unknown_writer_fail_closed": True,
        },
    }


def stable_stock_bucket(stock_code: str, bucket_count: int) -> int:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    digest = hashlib.sha256(str(stock_code).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % bucket_count


def build_partition_plan(
    trade_dates: Sequence[str],
    stock_codes: Sequence[str],
    *,
    date_chunk_days: int = DEFAULT_DATE_CHUNK_DAYS,
    stock_buckets: int = DEFAULT_STOCK_BUCKETS,
) -> list[dict[str, Any]]:
    """Return disjoint date-block × stock-bucket work units."""

    if date_chunk_days <= 0 or stock_buckets <= 0:
        raise ValueError("date_chunk_days and stock_buckets must be positive")
    dates = sorted({str(value) for value in trade_dates})
    codes = sorted({str(value) for value in stock_codes if not str(value).endswith(".BJ")})
    plan: list[dict[str, Any]] = []
    for offset in range(0, len(dates), date_chunk_days):
        block = dates[offset : offset + date_chunk_days]
        for bucket in range(stock_buckets):
            bucket_codes = [code for code in codes if stable_stock_bucket(code, stock_buckets) == bucket]
            if not bucket_codes:
                continue
            plan.append(
                {
                    "partition_id": f"date_{offset:05d}_{offset + len(block) - 1:05d}_bucket_{bucket:03d}",
                    "date_start": block[0],
                    "date_end": block[-1],
                    "stock_bucket": bucket,
                    "stock_bucket_count": stock_buckets,
                    "stock_count": len(bucket_codes),
                    "source_codes_sha256": hashlib.sha256("\n".join(bucket_codes).encode("utf-8")).hexdigest(),
                }
            )
    return plan


def estimate_peak_memory_bytes(
    *,
    max_rows_per_fragment: int,
    selected_column_count: int,
    object_multiplier: float = 3.0,
    working_set_multiplier: float = 2.0,
) -> int:
    """Conservative planning estimate; not a substitute for runtime gates."""

    if min(max_rows_per_fragment, selected_column_count) <= 0:
        raise ValueError("row and column counts must be positive")
    if object_multiplier < 1 or working_set_multiplier < 1:
        raise ValueError("memory multipliers must be >= 1")
    return math.ceil(
        max_rows_per_fragment
        * selected_column_count
        * 8
        * object_multiplier
        * working_set_multiplier
    )


def validate_fragment_metrics(
    metrics: dict[str, Any],
    *,
    expected_schema_hash: str,
    expected_columns: int,
) -> dict[str, Any]:
    """Fail closed on a fragment before it can enter the read-only merge."""

    errors: list[str] = []
    if int(metrics.get("source_rows", -1)) != int(metrics.get("output_rows", -2)):
        errors.append("source_output_rows_mismatch")
    if int(metrics.get("duplicate_key_groups", -1)) != 0:
        errors.append("duplicate_key_groups_nonzero")
    if int(metrics.get("bj_rows", -1)) != 0:
        errors.append("bj_rows_nonzero")
    if str(metrics.get("schema_hash")) != str(expected_schema_hash):
        errors.append("schema_hash_mismatch")
    if int(metrics.get("column_count", -1)) != int(expected_columns):
        errors.append("column_count_mismatch")
    if metrics.get("future_or_label_columns"):
        errors.append("future_or_label_columns_present")
    if metrics.get("write_active") is True or metrics.get("write_registry") is True:
        errors.append("active_write_flag_present")
    if errors:
        raise ValueError("memory-bounded fragment gate failed: " + ", ".join(errors))
    return {"status": "passed", "errors": [], "approved_for_merge": True}


def validate_fingerprint_stability(fingerprints: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Require unchanged active/label/registry identity across candidate work."""

    required = ("active_feature", "active_label", "production_registry")
    errors: list[str] = []
    for name in required:
        item = fingerprints.get(name, {})
        before = item.get("before")
        after = item.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            errors.append(f"{name}_fingerprint_missing")
            continue
        for field in ("sha256", "size_bytes", "mtime_ns"):
            if not before.get(field) or before.get(field) != after.get(field):
                errors.append(f"{name}_{field}_changed")
    if errors:
        raise ValueError("asset fingerprint stability gate failed: " + ", ".join(errors))
    return {"status": "passed", "errors": [], "assets_unchanged": list(required)}


def validate_global_candidate_metrics(
    metrics: dict[str, Any],
    *,
    fingerprints: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate global key-domain and schema gates after read-only merge."""

    required_zero = (
        "duplicate_key_groups",
        "bj_rows",
        "future_or_label_column_count",
        "key_domain_missing",
        "key_domain_extra",
    )
    errors = [name for name in required_zero if int(metrics.get(name, -1)) != 0]
    if int(metrics.get("column_count", 0)) != 838:
        errors.append("column_count_not_838")
    if int(metrics.get("qfq_price_count", 0)) != 5:
        errors.append("qfq_price_count_not_5")
    if int(metrics.get("l2_source_qfq_technical_count", 0)) != 74:
        errors.append("l2_source_qfq_technical_count_not_74")
    if int(metrics.get("production_feature_qfq_technical_count", 0)) != 76:
        errors.append("production_feature_qfq_technical_count_not_76")
    if int(metrics.get("derived_qfq_technical_count", 0)) != 2:
        errors.append("derived_qfq_technical_count_not_2")
    if int(metrics.get("gtja_qfq_count", 0)) != 191:
        errors.append("gtja_qfq_count_not_191")
    if metrics.get("label_write_called") is True:
        errors.append("label_write_called")
    if metrics.get("active_switch_called") is True:
        errors.append("active_switch_called")
    if metrics.get("registry_change") is True:
        errors.append("registry_change")
    if metrics.get("production_paths_opened") is True:
        errors.append("production_paths_opened")
    if metrics.get("wal_present") is True:
        errors.append("wal_present")
    if metrics.get("unknown_writer") is True:
        errors.append("unknown_writer")
    if int(metrics.get("label_target_rows", 0)) != 0:
        errors.append("label_target_rows_nonzero")
    maturity_max = metrics.get("label_maturity_max_trade_date")
    label_max = metrics.get("label_max_trade_date")
    if maturity_max is None or label_max != maturity_max:
        errors.append("label_maturity_contract_mismatch")
    if fingerprints is None:
        errors.append("asset_fingerprints_required")
    if errors:
        raise ValueError("memory-bounded global gate failed: " + ", ".join(errors))
    stability = validate_fingerprint_stability(fingerprints)
    return {"status": "passed", "errors": [], "approved_for_audit": True, "fingerprints": stability}
