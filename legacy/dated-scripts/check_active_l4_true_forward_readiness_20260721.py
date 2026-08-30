from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_true_forward_monitor_20260721"
PROTOCOL_PATH = REPORT_DIR / "forward_protocol.json"
STATUS_PATH = REPORT_DIR / "forward_readiness_status.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_manifest(path: Path) -> tuple[dict, Path, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, (path.parent / payload["db_path"]).resolve(), str(payload["table"])


def table_probe(db_path: Path, table: str, anchor: str) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        max_date, forward_days, bj_rows, duplicate_groups = con.execute(
            f"""
            SELECT
              (SELECT max(trade_date) FROM \"{table}\"),
              (SELECT count(DISTINCT trade_date) FROM \"{table}\" WHERE trade_date > ?),
              (SELECT count(*) FROM \"{table}\" WHERE trade_date > ? AND stock_code LIKE '%.BJ'),
              (SELECT count(*) FROM (
                   SELECT trade_date, stock_code
                   FROM \"{table}\"
                   WHERE trade_date > ?
                   GROUP BY trade_date, stock_code
                   HAVING count(*) > 1
               ))
            """,
            [anchor, anchor, anchor],
        ).fetchone()
        dates = [str(row[0]) for row in con.execute(f'SELECT DISTINCT trade_date FROM "{table}" WHERE trade_date > ? ORDER BY trade_date', [anchor]).fetchall()]
        return {"max_trade_date": str(max_date), "forward_days": int(forward_days), "bj_rows": int(bj_rows), "duplicate_key_groups": int(duplicate_groups), "forward_dates": dates}
    finally:
        con.close()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    anchor = protocol["anchor_last_seen_date"]
    probes = {}
    common_dates = None
    blockers = []
    candidate_probes = []
    for candidate in protocol["candidates"]:
        source = ROOT / candidate["source"]
        actual_hash = sha256(source)
        if actual_hash != candidate["source_sha256"]:
            blockers.append(f"candidate_source_hash_drift:{candidate['case_id']}")
        source_payload = json.loads(source.read_text(encoding="utf-8"))
        matching = [row for row in source_payload.get("profiles", []) if row.get("case_id") == candidate["case_id"]]
        parameters_match = len(matching) == 1 and all(matching[0].get(key) == value for key, value in candidate["parameters"].items())
        if not parameters_match:
            blockers.append(f"candidate_parameters_drift:{candidate['case_id']}")
        candidate_probes.append({
            "case_id": candidate["case_id"],
            "source": candidate["source"],
            "expected_sha256": candidate["source_sha256"],
            "actual_sha256": actual_hash,
            "source_hash_match": actual_hash == candidate["source_sha256"],
            "parameters_match": parameters_match,
        })
    for rel in protocol["formal_manifests"]:
        manifest_path = ROOT / rel
        manifest, db_path, table = resolve_manifest(manifest_path)
        if manifest.get("approval_status") != "approved_for_l5":
            blockers.append(f"manifest_not_approved:{manifest_path.name}")
        if manifest.get("source_type") != "duckdb_table":
            blockers.append(f"source_not_duckdb_table:{manifest_path.name}")
        probe = table_probe(db_path, table, anchor)
        probe.update({"manifest": rel, "manifest_sha256": sha256(manifest_path), "db_path": str(db_path), "table": table})
        probes[manifest_path.name] = probe
        common_dates = set(probe["forward_dates"]) if common_dates is None else common_dates & set(probe["forward_dates"])
        if probe["bj_rows"]:
            blockers.append(f"bj_rows_positive:{manifest_path.name}")
        if probe["duplicate_key_groups"]:
            blockers.append(f"duplicate_keys_positive:{manifest_path.name}")
    l2_path = ROOT / protocol["l2"]["path"]
    l2_probe = table_probe(l2_path, protocol["l2"]["table"], anchor)
    probes["L2"] = {**l2_probe, "db_path": str(l2_path), "table": protocol["l2"]["table"]}
    common_dates = (common_dates or set()) & set(l2_probe["forward_dates"])
    if l2_probe["bj_rows"]:
        blockers.append("bj_rows_positive:L2")
    if l2_probe["duplicate_key_groups"]:
        blockers.append("duplicate_keys_positive:L2")
    common_dates = sorted(common_dates)
    interim = int(protocol["interim_observation_trade_days"])
    admission = int(protocol["first_admission_review_trade_days"])
    if len(common_dates) < interim:
        blockers.append(f"common_forward_days_below_{interim}")
    payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "anchor_last_seen_date": anchor,
        "common_forward_days": len(common_dates),
        "common_forward_min": common_dates[0] if common_dates else None,
        "common_forward_max": common_dates[-1] if common_dates else None,
        "interim_observation_allowed": len(common_dates) >= interim and not blockers,
        "first_admission_review_allowed": len(common_dates) >= admission and not [item for item in blockers if not item.startswith("common_forward_days_below_")],
        "status": "ready_for_interim_observation" if len(common_dates) >= interim and not blockers else "blocked_waiting_for_true_forward_data",
        "blockers": blockers,
        "candidate_probes": candidate_probes,
        "probes": probes,
        "production_changed": False,
        "signal_generated": False
    }
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
