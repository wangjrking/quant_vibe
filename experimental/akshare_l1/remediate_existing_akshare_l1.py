from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

import akshare_l1_experimental as module


ROOT = module.DEFAULT_OUTPUT_ROOT
REPORT_ROOT = module.DEFAULT_REPORT_ROOT
ORIGINAL_JSON = REPORT_ROOT / "investment_platform_akshare_l1_gap_fill_20260714.json"
ORIGINAL_MD = ORIGINAL_JSON.with_suffix(".md")
REMEDIATION_JSON = REPORT_ROOT / "investment_platform_akshare_l1_gap_fill_20260714_remediation.json"
REMEDIATION_MD = REMEDIATION_JSON.with_suffix(".md")
HASH_EVIDENCE = REPORT_ROOT / "investment_platform_akshare_l1_gap_fill_20260714_remediation_hashes.json"
MANIFEST = ROOT / "manifest.json"
ASSETS = {
    "company_announcements": (["announcement_id"], "announcement_date"),
    "stock_news": (["news_id"], "published_at"),
    "social_sentiment_snapshot": (["snapshot_at", "rank"], "snapshot_at"),
    "stock_minutes_1m": (["ts_code", "trade_time"], "trade_time"),
    "financial_indicator": (["ts_code", "report_date"], "report_date"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def db_path(table: str) -> Path:
    return ROOT / table / f"{table}.duckdb"


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def schema(connection: duckdb.DuckDBPyConnection, table: str) -> list[dict[str, Any]]:
    return [
        {"ordinal": int(row[0]), "name": row[1], "type": row[2], "not_null": bool(row[3])}
        for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    ]


def fingerprint(connection: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> dict[str, Any]:
    expression = ", ".join(quote(column) for column in columns)
    row_count, checksum = connection.execute(
        f"SELECT COUNT(*), bit_xor(hash({expression})) FROM {quote(table)}"
    ).fetchone()
    return {"rows": int(row_count), "bit_xor_hash": str(checksum)}


def migrate_financial() -> dict[str, Any]:
    path = db_path("financial_indicator")
    before_hash = sha256(path)
    connection = duckdb.connect(str(path))
    try:
        before_schema = schema(connection, "financial_indicator")
        before_columns = [item["name"] for item in before_schema]
        required = {"report_date", "REPORT_DATE_1"}
        if not required.issubset(before_columns):
            raise RuntimeError(f"unexpected financial schema before remediation: {before_columns}")
        mismatch = int(
            connection.execute(
                "SELECT COUNT(*) FROM financial_indicator "
                "WHERE report_date IS DISTINCT FROM strftime(try_cast(REPORT_DATE_1 AS TIMESTAMP), '%Y%m%d')"
            ).fetchone()[0]
        )
        if mismatch != 0:
            raise RuntimeError(f"canonical/source report date mismatch rows: {mismatch}")
        retained_columns = [column for column in before_columns if column != "REPORT_DATE_1"]
        before_fingerprint = fingerprint(connection, "financial_indicator", retained_columns)
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute('ALTER TABLE financial_indicator DROP COLUMN "REPORT_DATE_1"')
            after_schema = schema(connection, "financial_indicator")
            after_columns = [item["name"] for item in after_schema]
            if after_columns != retained_columns:
                raise RuntimeError("financial schema order/value contract changed unexpectedly")
            after_fingerprint = fingerprint(connection, "financial_indicator", after_columns)
            if before_fingerprint != after_fingerprint:
                raise RuntimeError("financial retained-column fingerprint changed")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()
    return {
        "before_sha256": before_hash,
        "after_sha256": sha256(path),
        "before_schema": before_schema,
        "after_schema": after_schema,
        "removed_columns": ["REPORT_DATE_1"],
        "canonical_date_field": "report_date",
        "date_equivalence_mismatch_rows": mismatch,
        "before_retained_values_fingerprint": before_fingerprint,
        "after_retained_values_fingerprint": after_fingerprint,
        "lossless": True,
    }


def migrate_minutes_lineage() -> dict[str, Any]:
    path = db_path("stock_minutes_1m")
    before_hash = sha256(path)
    connection = duckdb.connect(str(path))
    try:
        columns = [item["name"] for item in schema(connection, "stock_minutes_1m")]
        retained_columns = [column for column in columns if column != "source"]
        before_fingerprint = fingerprint(connection, "stock_minutes_1m", retained_columns)
        before_sources = {str(row[0]): int(row[1]) for row in connection.execute("SELECT source, COUNT(*) FROM stock_minutes_1m GROUP BY source").fetchall()}
        if set(before_sources) - {"AKShare.minute_api", "AKShare.stock_zh_a_minute"}:
            raise RuntimeError(f"unexpected minute lineage source values: {before_sources}")
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                "UPDATE stock_minutes_1m SET source='AKShare.stock_zh_a_minute' "
                "WHERE source='AKShare.minute_api'"
            )
            after_sources = {str(row[0]): int(row[1]) for row in connection.execute("SELECT source, COUNT(*) FROM stock_minutes_1m GROUP BY source").fetchall()}
            if after_sources != {"AKShare.stock_zh_a_minute": 714}:
                raise RuntimeError(f"minute lineage remediation failed: {after_sources}")
            after_fingerprint = fingerprint(connection, "stock_minutes_1m", retained_columns)
            if before_fingerprint != after_fingerprint:
                raise RuntimeError("minute business-value fingerprint changed")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()
    return {
        "before_sha256": before_hash,
        "after_sha256": sha256(path),
        "before_sources": before_sources,
        "after_sources": after_sources,
        "source": "AKShare.stock_zh_a_minute",
        "fallback_from": "AKShare.stock_zh_a_hist_min_em",
        "adjustment": module.ASSET_CONTRACTS["stock_minutes_1m"]["adjustment"],
        "sample_scope": module.ASSET_CONTRACTS["stock_minutes_1m"]["sample_scope"],
        "source_parameters": module.ASSET_CONTRACTS["stock_minutes_1m"]["source_parameters"],
        "before_business_values_fingerprint": before_fingerprint,
        "after_business_values_fingerprint": after_fingerprint,
        "business_values_unchanged": True,
    }


def inspect_asset(table: str, mapping_status: str) -> dict[str, Any]:
    path = db_path(table)
    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY table_name"
            ).fetchall()
        ]
        if tables != [table]:
            raise RuntimeError(f"one-table-one-file contract failed for {path}: {tables}")
        frame = connection.execute(f"SELECT * FROM {quote(table)}").fetchdf()
    finally:
        connection.close()
    keys, date_field = ASSETS[table]
    entry = module.build_asset_entry(table, path, frame, keys, date_field, mapping_status)
    entry["physical_schema"] = [str(column) for column in frame.columns]
    if entry["duplicate_key_groups"] != 0:
        raise RuntimeError(f"duplicate key groups remain in {table}")
    return entry


def render_remediation(report: dict[str, Any]) -> str:
    lines = [
        "# AKShare experimental/test L1 remediation",
        "",
        f"- Status: `{report['status']}`",
        f"- Completed at: `{report['completed_at']}`",
        "- Scope: existing five-table experimental sample only; no source refetch",
        "- Production assets/route/registry touched: `false`; L2-L8 triggered: `false`; legacy odb used: `false`",
        "",
        "## Asset contracts",
        "",
        "| Table | Source | Rows | Stocks | Date range | Key | Duplicates | Status |",
        "|---|---|---:|---:|---|---|---:|---|",
    ]
    for table, asset in report["assets"].items():
        lines.append(
            f"| `{table}` | `{asset['source']}` | {asset['rows']} | {asset.get('stock_coverage')} | "
            f"{asset.get('min_date')} to {asset.get('max_date')} | `{asset['natural_key']}` | "
            f"{asset['duplicate_key_groups']} | `{asset['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Schema remediation",
            "",
            "- `financial_indicator`: removed physical `REPORT_DATE_1`; canonical field is `report_date`.",
            f"- Date equivalence mismatch rows: `{report['financial_schema_remediation']['date_equivalence_mismatch_rows']}`.",
            "- Retained-column row count and bit-xor hash are unchanged.",
            "",
            "## Minute lineage",
            "",
            "- Actual source: `AKShare.stock_zh_a_minute`.",
            "- Fallback from: `AKShare.stock_zh_a_hist_min_em` after ProxyError.",
            "- Adjustment: empty `adjust`, verified as unadjusted/raw prices in AKShare 1.18.64.",
            "- Scope: 3 stocks, 20260713 only, 09:31-15:00; not full market and not full history.",
            "",
            "## Tests",
            "",
            f"- Command: `{report['tests']['command']}`",
            f"- Tests: `{report['tests']['count']}`; success: `{str(report['tests']['success']).lower()}`.",
            "",
            "This package is ready only for read-only re-audit. It is not approved for production or website financial activation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    module.validate_output_root(ROOT)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    tracked_before = {str(path): sha256(path) for path in [MANIFEST, *(db_path(table) for table in ASSETS)]}
    original_report = json.loads(ORIGINAL_JSON.read_text(encoding="utf-8"))
    mapping_status = {table: original_report["assets"][table]["mapping_status"] for table in ASSETS}

    financial = migrate_financial()
    minutes = migrate_minutes_lineage()

    assets = {table: inspect_asset(table, mapping_status[table]) for table in ASSETS}
    for table in ASSETS:
        path = db_path(table)
        if table not in {"financial_indicator", "stock_minutes_1m"} and sha256(path) != tracked_before[str(path)]:
            raise RuntimeError(f"concurrent modification detected for untouched asset: {path}")

    completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    remediation_path = str(REMEDIATION_JSON)
    original_report.update(
        {
            "assets": assets,
            "production_assets_touched": False,
            "production_route_or_registry_touched": False,
            "l2_l8_triggered": False,
            "legacy_odb_used": False,
            "source_ready_for_production": False,
            "experimental_audit_ready": True,
            "remediation": {
                "task_id": "investment-platform-akshare-test-l1-gap-fill-20260714-remediation",
                "status": "completed_ready_for_read_only_reaudit",
                "started_at": started_at,
                "completed_at": completed_at,
                "source_refetched": False,
                "data_scope_expanded": False,
                "financial_schema": financial,
                "minute_lineage": minutes,
                "remediation_report_path": remediation_path,
            },
        }
    )
    original_report["interface_contracts"] = module.build_interface_contracts(original_report)
    encoded = json.dumps(original_report, ensure_ascii=False, indent=2, default=str)
    MANIFEST.write_text(encoded, encoding="utf-8")
    ORIGINAL_JSON.write_text(encoded, encoding="utf-8")
    ORIGINAL_MD.write_text(module.render_markdown(original_report), encoding="utf-8")

    command = f'{sys.executable} -m unittest -v test_akshare_l1_experimental.py'
    test_run = subprocess.run(
        [sys.executable, "-m", "unittest", "-v", "test_akshare_l1_experimental.py"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    test_output = (test_run.stdout + "\n" + test_run.stderr).strip()
    count_match = re.search(r"Ran (\d+) tests?", test_output)
    test_count = int(count_match.group(1)) if count_match else 0
    if test_run.returncode != 0:
        raise RuntimeError(f"remediation tests failed:\n{test_output}")

    remediation_report: dict[str, Any] = {
        "task_id": "investment-platform-akshare-test-l1-gap-fill-20260714-remediation",
        "original_task_id": "investment-platform-akshare-test-l1-gap-fill-20260714",
        "status": "completed_ready_for_read_only_reaudit",
        "started_at": started_at,
        "completed_at": completed_at,
        "source_refetched": False,
        "data_scope_expanded": False,
        "akshare_version": original_report["akshare_version"],
        "probe_time": original_report["probe_time"],
        "assets": assets,
        "financial_schema_remediation": financial,
        "stock_minutes_lineage_remediation": minutes,
        "before_file_sha256": tracked_before,
        "tests": {"command": command, "count": test_count, "success": True, "output": test_output},
        "production_assets_touched": False,
        "production_route_or_registry_touched": False,
        "l2_l8_triggered": False,
        "legacy_odb_used": False,
        "production_approved": False,
        "website_financial_activation_allowed": False,
        "requires_read_only_reaudit": True,
        "residual_risks": [
            "all non-announcement assets remain small experimental samples",
            "social sentiment remains name-only without a security code",
            "minute fallback history is limited and not full-market coverage",
            "financial sample covers only three stocks and remains disabled for website use",
        ],
    }
    REMEDIATION_JSON.write_text(json.dumps(remediation_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    REMEDIATION_MD.write_text(render_remediation(remediation_report), encoding="utf-8")

    modified_files = [
        Path(module.__file__),
        Path(__file__),
        Path(__file__).parent / "test_akshare_l1_experimental.py",
        MANIFEST,
        ORIGINAL_JSON,
        ORIGINAL_MD,
        REMEDIATION_JSON,
        REMEDIATION_MD,
        *(db_path(table) for table in ASSETS),
    ]
    hashes = {str(path): sha256(path) for path in modified_files}
    HASH_EVIDENCE.write_text(
        json.dumps(
            {
                "task_id": remediation_report["task_id"],
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "algorithm": "SHA256",
                "files": hashes,
                "note": "This evidence file intentionally does not self-hash.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": remediation_report["status"], "tests": remediation_report["tests"], "hash_evidence": str(HASH_EVIDENCE)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
