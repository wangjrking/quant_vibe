"""Read-only postwrite validation for a target-date L4 formal refresh."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "quant" / "main" / "config" / "prediction_manifests"
MANIFEST_NAMES = (
    "executable_1d_open_return_l4_formal_20260619.json",
    "executable_3d_open_return_l4_formal_20260617.json",
    "executable_5d_open_return_l4_formal_20260620.json",
    "executable_10d_open_return_l4_formal_20260617.json",
    "prod_liq_prime_one_v20260612_l4_formal.json",
)
ALL4_MANIFEST_NAMES = MANIFEST_NAMES[:4]
DIGEST_CONTRACT = {
    "algorithm_id": "l4_target_day_utf8_trade_date_stock_code_float17g_newline_v1",
    "sort_order": ["trade_date ASC", "stock_code ASC"],
    "record_fields": ["trade_date", "stock_code", "pred_prob"],
    "record_template": "{trade_date}|{stock_code}|{format(float(pred_prob), '.17g')}\\n",
    "encoding": "utf-8",
    "float_serialization": "Python format(float(pred_prob), '.17g')",
    "null_token": "<NULL>",
}


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def canonical_digest(conn: duckdb.DuckDBPyConnection, table: str, target_date: str) -> str:
    digest = hashlib.sha256()
    query = (
        f"SELECT trade_date, stock_code, pred_prob FROM {quote_identifier(table)} "
        "WHERE CAST(trade_date AS VARCHAR) = ? ORDER BY trade_date, stock_code"
    )
    for trade_date, stock_code, pred_prob in conn.execute(query, [target_date]).fetchall():
        value = "<NULL>" if pred_prob is None else format(float(pred_prob), ".17g")
        digest.update(f"{trade_date}|{stock_code}|{value}\n".encode("utf-8"))
    return digest.hexdigest()


def legacy_digest(conn: duckdb.DuckDBPyConnection, table: str, target_date: str) -> str:
    digest = hashlib.sha256()
    query = (
        f"SELECT stock_code, pred_prob FROM {quote_identifier(table)} "
        "WHERE CAST(trade_date AS VARCHAR) = ? ORDER BY stock_code"
    )
    for stock_code, pred_prob in conn.execute(query, [target_date]).fetchall():
        digest.update(f"{stock_code}|{pred_prob}\n".encode("utf-8"))
    return digest.hexdigest()


def load_manifest(name: str) -> dict[str, Any]:
    path = MANIFEST_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["_manifest_path"] = str(path)
    return data


def validate_output(
    manifest_name: str,
    target_date: str,
    expected_rows: int,
    probe_stock_code: str | None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_name)
    db_path = (MANIFEST_DIR / manifest["db_path"]).resolve()
    table = manifest["table"]
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        table_q = quote_identifier(table)
        stats = conn.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT stock_code) AS stocks,
                SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END) AS bj_rows,
                SUM(CASE WHEN pred_prob IS NULL THEN 1 ELSE 0 END) AS null_pred_prob,
                SUM(CASE WHEN pred_prob IS NOT NULL AND NOT isfinite(pred_prob) THEN 1 ELSE 0 END) AS nonfinite_pred_prob,
                COUNT(DISTINCT pred_prob) AS distinct_scores,
                STDDEV_POP(pred_prob) AS score_std
            FROM {table_q}
            WHERE CAST(trade_date AS VARCHAR) = ?
            """,
            [target_date],
        ).fetchone()
        duplicate_key_groups = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT trade_date, stock_code
                FROM {table_q}
                WHERE CAST(trade_date AS VARCHAR) = ?
                GROUP BY trade_date, stock_code
                HAVING COUNT(*) > 1
            )
            """,
            [target_date],
        ).fetchone()[0]
        probe = None
        if probe_stock_code:
            probe_row = conn.execute(
                f"SELECT COUNT(*) AS row_count, MIN(pred_prob) AS pred_prob "
                f"FROM {table_q} WHERE CAST(trade_date AS VARCHAR) = ? AND stock_code = ?",
                [target_date, probe_stock_code],
            ).fetchone()
            probe = {"stock_code": probe_stock_code, "rows": probe_row[0], "pred_prob": probe_row[1]}
        rows, stocks, bj_rows, nulls, nonfinite, distinct, score_std = stats
        score_std = None if score_std is None else float(score_std)
        valid = (
            manifest.get("source_type") == "duckdb_table"
            and manifest.get("approval_status") == "approved_for_l5"
            and rows == expected_rows
            and stocks == expected_rows
            and bj_rows == 0
            and nulls == 0
            and nonfinite == 0
            and duplicate_key_groups == 0
            and distinct is not None
            and distinct > 1
            and score_std is not None
            and math.isfinite(score_std)
            and score_std > 0
            and (probe is None or probe["rows"] == 1 and probe["pred_prob"] is not None)
        )
        return {
            "manifest": manifest_name,
            "db_path": str(db_path),
            "table": table,
            "source_type": manifest.get("source_type"),
            "approval_status": manifest.get("approval_status"),
            "rows": rows,
            "stocks": stocks,
            "bj_rows": bj_rows,
            "null_pred_prob": nulls,
            "nonfinite_pred_prob": nonfinite,
            "duplicate_key_groups": duplicate_key_groups,
            "distinct_scores": distinct,
            "score_std": score_std,
            "canonical_digest_algorithm": DIGEST_CONTRACT["algorithm_id"],
            "canonical_digest_sha256": canonical_digest(conn, table, target_date),
            "legacy_digest_algorithm": "stock_code_pipe_duckdb_python_string_v0",
            "legacy_digest_sha256": legacy_digest(conn, table, target_date),
            "stock_probe": probe,
            "valid": valid,
        }
    finally:
        conn.close()


def all4_common_key_count(target_date: str) -> int:
    key_sets: list[set[str]] = []
    for manifest_name in ALL4_MANIFEST_NAMES:
        manifest = load_manifest(manifest_name)
        db_path = (MANIFEST_DIR / manifest["db_path"]).resolve()
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = conn.execute(
                f"SELECT stock_code FROM {quote_identifier(manifest['table'])} "
                "WHERE CAST(trade_date AS VARCHAR) = ?",
                [target_date],
            ).fetchall()
            key_sets.append({row[0] for row in rows})
        finally:
            conn.close()
    return len(set.intersection(*key_sets)) if key_sets else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe-stock-code")
    args = parser.parse_args()

    outputs = {
        name: validate_output(name, args.target_date, args.expected_rows, args.probe_stock_code)
        for name in MANIFEST_NAMES
    }
    common_keys = all4_common_key_count(args.target_date)
    valid = common_keys == args.expected_rows and all(item["valid"] for item in outputs.values())
    result = {
        "report_type": "l4_formal_incremental_postwrite_validation",
        "read_only": True,
        "target_trade_date": args.target_date,
        "expected_rows_and_stocks": args.expected_rows,
        "digest_contract": DIGEST_CONTRACT,
        "outputs": outputs,
        "all4_common_key_count": common_keys,
        "valid": valid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write("\n")
    if not valid:
        raise SystemExit("postwrite validation failed")


if __name__ == "__main__":
    main()
