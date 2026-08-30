from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARCHIVE = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[6]
MAIN = PROJECT / "quant/main"
EXPECTED_ACTION_HASH = (
    "D02BB8B6384BE1B586FA39D2676CAE2E15495D35A39132C29279F65A173D516F"
)
MANIFEST = (
    MAIN
    / "config/prediction_manifests/"
    "executable_10d_open_return_l4_formal_20260617.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    action_file = ARCHIVE / "signals/full_history_actions.csv"
    action_hash = sha256(action_file)
    if action_hash != EXPECTED_ACTION_HASH:
        raise RuntimeError("frozen action hash mismatch")

    source_mismatches = []
    snapshot_dir = ARCHIVE / "code_snapshot/source_modules"
    for snapshot in sorted(snapshot_dir.glob("*.py")):
        active = MAIN / snapshot.name
        if not active.exists() or sha256(active) != sha256(snapshot):
            source_mismatches.append(snapshot.name)
    if source_mismatches:
        raise RuntimeError(
            "active source differs from frozen snapshot: "
            + ", ".join(source_mismatches)
        )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("approval_status") != "approved_for_l5"
        or manifest.get("source_type") != "duckdb_table"
    ):
        raise RuntimeError("10D formal manifest is not approved DuckDB")
    source_text = json.dumps(manifest, ensure_ascii=False).lower()
    if "sqlite" in source_text or "odb.db" in source_text:
        raise RuntimeError("legacy production input detected")

    print(
        json.dumps(
            {
                "status": "verified",
                "strategy_id": "prod_v260_10d_regime_warmup_v20260724",
                "action_sha256": action_hash,
                "source_modules": len(list(snapshot_dir.glob("*.py"))),
                "source_mismatches": source_mismatches,
                "manifest_approval_status": manifest["approval_status"],
                "manifest_source_type": manifest["source_type"],
                "production_cutover_performed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
