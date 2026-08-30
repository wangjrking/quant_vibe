"""Recompute formal L4 target-day evidence with a documented byte contract.

The command is read-only with respect to L4 prediction tables and manifests.
It exists so an auditor can replay the target-day digest without relying on
implicit database string formatting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "quant/main/config/prediction_manifests"
DIGEST_ALGORITHM_ID = "l4_target_day_utf8_trade_date_stock_code_float17g_newline_v1"
LEGACY_ALGORITHM_ID = "duckdb_string_agg_stock_code_pipe_cast_pred_prob_v0"


def quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def resolve_manifest_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path.resolve()
    return (MANIFEST_DIR / path).resolve()


def canonical_digest(connection: duckdb.DuckDBPyConnection, table: str, trade_date: str) -> str:
    """Hash exact UTF-8 records: trade_date|stock_code|float17g\\n, ordered by key."""
    digest = hashlib.sha256()
    rows = connection.execute(
        f"SELECT trade_date, stock_code, pred_prob FROM {quote(table)} "
        "WHERE trade_date = ? ORDER BY trade_date, stock_code",
        [trade_date],
    ).fetchall()
    for row_trade_date, stock_code, pred_prob in rows:
        if pred_prob is None:
            value = "<NULL>"
        else:
            value = format(float(pred_prob), ".17g")
        digest.update(f"{row_trade_date}|{stock_code}|{value}\n".encode("utf-8"))
    return digest.hexdigest()


def legacy_digest(connection: duckdb.DuckDBPyConnection, table: str, trade_date: str) -> str | None:
    """Replay the undocumented v0 digest only for historical-evidence comparison."""
    return connection.execute(
        f"SELECT sha256(string_agg(stock_code || '|' || cast(pred_prob AS VARCHAR), '' "
        f"ORDER BY stock_code)) FROM {quote(table)} WHERE trade_date = ?",
        [trade_date],
    ).fetchone()[0]


def probe_manifest(manifest_path: Path, trade_date: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    db_path = (manifest_path.parent / str(manifest["db_path"])).resolve()
    table = str(manifest["table"])
    with duckdb.connect(str(db_path), read_only=True) as connection:
        stats = connection.execute(
            f"SELECT count(*), count(distinct stock_code), "
            f"sum(case when stock_code like '%.BJ' then 1 else 0 end), "
            f"sum(case when pred_prob is null then 1 else 0 end), "
            f"sum(case when pred_prob is not null and not isfinite(pred_prob) then 1 else 0 end), "
            f"count(distinct pred_prob), stddev_pop(pred_prob) "
            f"FROM {quote(table)} WHERE trade_date = ?",
            [trade_date],
        ).fetchone()
        duplicate_groups = connection.execute(
            f"SELECT count(*) FROM (SELECT stock_code FROM {quote(table)} "
            "WHERE trade_date = ? GROUP BY stock_code HAVING count(*) > 1)",
            [trade_date],
        ).fetchone()[0]
        return {
            "manifest": str(manifest_path),
            "db_path": str(db_path),
            "table": table,
            "source_type": manifest.get("source_type"),
            "approval_status": manifest.get("approval_status"),
            "rows": stats[0],
            "stocks": stats[1],
            "bj_rows": int(stats[2] or 0),
            "null_pred_prob": int(stats[3] or 0),
            "nonfinite_pred_prob": int(stats[4] or 0),
            "distinct_scores": stats[5],
            "score_std": stats[6],
            "duplicate_key_groups": duplicate_groups,
            "canonical_digest_algorithm": DIGEST_ALGORITHM_ID,
            "canonical_digest_sha256": canonical_digest(connection, table, trade_date),
            "legacy_digest_algorithm": LEGACY_ALGORITHM_ID,
            "legacy_digest_sha256": legacy_digest(connection, table, trade_date),
            "valid": bool(
                stats[0]
                and stats[0] == stats[1]
                and not int(stats[2] or 0)
                and not int(stats[3] or 0)
                and not int(stats[4] or 0)
                and not duplicate_groups
                and int(stats[5] or 0) > 1
                and float(stats[6] or 0.0) > 0.0
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only formal L4 target-day digest recomputation.")
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = {
        "digest_contract": {
            "algorithm_id": DIGEST_ALGORITHM_ID,
            "sort_order": ["trade_date ASC", "stock_code ASC"],
            "record_fields": ["trade_date", "stock_code", "pred_prob"],
            "record_template": "{trade_date}|{stock_code}|{format(float(pred_prob), '.17g')}\\n",
            "encoding": "utf-8",
            "null_token": "<NULL>",
            "float_serialization": "Python format(float(pred_prob), '.17g')",
        },
        "target_trade_date": str(args.target_date),
        "read_only": True,
        "outputs": {},
    }
    common_keys: set[str] | None = None
    for raw_manifest in args.manifest:
        manifest_path = resolve_manifest_path(raw_manifest)
        probe = probe_manifest(manifest_path, str(args.target_date))
        result["outputs"][manifest_path.name] = probe
        if manifest_path.name != "prod_liq_prime_one_v20260612_l4_formal.json":
            with duckdb.connect(probe["db_path"], read_only=True) as connection:
                keys = {
                    row[0]
                    for row in connection.execute(
                        f"SELECT stock_code FROM {quote(probe['table'])} "
                        "WHERE trade_date = ?",
                        [str(args.target_date)],
                    ).fetchall()
                }
            common_keys = keys if common_keys is None else common_keys & keys
    result["all4_common_key_count"] = len(common_keys or set())
    result["valid"] = bool(
        common_keys
        and all(probe["valid"] for probe in result["outputs"].values())
        and len(common_keys) == next(iter(result["outputs"].values()))["stocks"]
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
