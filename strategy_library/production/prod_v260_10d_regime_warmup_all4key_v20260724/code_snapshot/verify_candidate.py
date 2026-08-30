from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ARCHIVE = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[6]
MAIN = PROJECT / "quant/main"
EXPECTED_ACTION_HASH = (
    "D02BB8B6384BE1B586FA39D2676CAE2E15495D35A39132C29279F65A173D516F"
)
EXPECTED_ACTION_ROWS = 1563
MANIFESTS = {
    "1d": MAIN
    / "config/prediction_manifests/"
    "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN
    / "config/prediction_manifests/"
    "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN
    / "config/prediction_manifests/"
    "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN
    / "config/prediction_manifests/"
    "executable_10d_open_return_l4_formal_20260617.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def resolve_source(path: Path, payload: dict) -> tuple[Path, str]:
    db_path = (path.parent / str(payload["db_path"])).resolve()
    if db_path.suffix.lower() != ".duckdb" or not db_path.is_file():
        raise RuntimeError(f"formal DuckDB source is unavailable: {path}")
    return db_path, str(payload["table"])


def verify_sources() -> int:
    mismatches = []
    snapshot_dir = ARCHIVE / "code_snapshot/source_modules"
    snapshots = sorted(snapshot_dir.glob("*.py"))
    for snapshot in snapshots:
        active = MAIN / snapshot.name
        if not active.exists() or sha256(active) != sha256(snapshot):
            mismatches.append(snapshot.name)
    if mismatches:
        raise RuntimeError(
            "active source differs from frozen snapshot: " + ", ".join(mismatches)
        )
    return len(snapshots)


def verify_manifests() -> dict[str, dict]:
    payloads = {}
    max_dates = set()
    for label, path in MANIFESTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("approval_status") != "approved_for_l5"
            or payload.get("source_type") != "duckdb_table"
        ):
            raise RuntimeError(f"{label} formal manifest is not approved DuckDB")
        serialized = json.dumps(payload, ensure_ascii=False).lower()
        if (
            "sqlite" in serialized
            or "odb.db" in serialized
            or "research_only" in serialized
        ):
            raise RuntimeError(f"legacy or research-only input detected: {label}")
        db_path, table = resolve_source(path, payload)
        with duckdb.connect(str(db_path), read_only=True) as con:
            row_count, stock_count, min_date, max_date, duplicate_keys, null_scores = (
                con.execute(
                    f"""
                    SELECT
                      COUNT(*),
                      COUNT(DISTINCT stock_code),
                      MIN(trade_date),
                      MAX(trade_date),
                      COUNT(*) - COUNT(DISTINCT trade_date || '|' || stock_code),
                      SUM(CASE WHEN pred_prob IS NULL THEN 1 ELSE 0 END)
                    FROM "{table}"
                    """
                ).fetchone()
            )
        if int(duplicate_keys) != 0 or int(null_scores or 0) != 0:
            raise RuntimeError(f"{label} formal table key/score quality failed")
        if str(max_date) != str(payload.get("max_trade_date")):
            raise RuntimeError(f"{label} manifest/table max date mismatch")
        max_dates.add(str(max_date))
        payloads[label] = {
            "manifest_path": str(path),
            "manifest_sha256": sha256(path),
            "db_path": str(db_path),
            "db_sha256": sha256(db_path),
            "table": table,
            "row_count": int(row_count),
            "stock_count": int(stock_count),
            "min_trade_date": str(min_date),
            "max_trade_date": str(max_date),
            "duplicate_keys": int(duplicate_keys),
            "null_scores": int(null_scores or 0),
        }
    if len(max_dates) != 1:
        raise RuntimeError(f"formal manifest max dates are not aligned: {max_dates}")
    return payloads


def verify_exact_actions() -> dict:
    sys.path.insert(0, str(MAIN))
    import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
    import research_active_l4_v258_position_warmup_v260_20260723 as v260

    frozen_path = ARCHIVE / "signals/full_history_actions.csv"
    if sha256(frozen_path) != EXPECTED_ACTION_HASH:
        raise RuntimeError("frozen action hash mismatch")
    frozen = pd.read_csv(
        frozen_path,
        dtype={"signal_date": str, "buy_date": str, "stock_code": str},
    )
    if len(frozen) != EXPECTED_ACTION_ROWS:
        raise RuntimeError("frozen action row count mismatch")

    protocol = json.loads(
        (ARCHIVE / "inputs/preregistered_protocol.json").read_text(encoding="utf-8")
    )
    arrays = v260.v258.v252.fresh_arrays()
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definition = v260.definition_for(protocol, 50)
    _, rebuilt = v260.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        "20260721",
        record_actions=True,
    )
    if list(rebuilt.columns) != list(frozen.columns):
        raise RuntimeError("rebuilt action schema mismatch")
    if len(rebuilt) != len(frozen):
        raise RuntimeError(
            f"rebuilt action row mismatch: expected {len(frozen)}, got {len(rebuilt)}"
        )
    for column in ("signal_date", "buy_date", "action", "stock_code"):
        if not frozen[column].astype(str).equals(rebuilt[column].astype(str)):
            raise RuntimeError(f"rebuilt action key mismatch: {column}")
    for column in ("target_pct", "execution_open_raw"):
        if not np.allclose(
            frozen[column].to_numpy(dtype=float),
            rebuilt[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
        ):
            raise RuntimeError(f"rebuilt action value mismatch: {column}")
    return {
        "frozen_action_rows": len(frozen),
        "frozen_action_sha256": sha256(frozen_path),
        "rebuilt_action_rows": len(rebuilt),
        "extra_action_keys": 0,
        "missing_action_keys": 0,
        "key_domain": "inner_intersection_1d_3d_5d_10d",
        "score_weights": {"1d": 0.0, "3d": 0.0, "5d": 0.0, "10d": 1.0},
    }


def main() -> None:
    module_count = verify_sources()
    manifests = verify_manifests()
    actions = verify_exact_actions()
    print(
        json.dumps(
            {
                "status": "verified",
                "strategy_id": "prod_v260_10d_regime_warmup_all4key_v20260724",
                "source_module_count": module_count,
                "source_mismatches": [],
                "formal_manifests": manifests,
                "exact_action_reproduction": actions,
                "production_cutover_performed": False,
                "formal_signal_generated": False,
                "l7_execution_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
