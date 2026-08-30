from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MANIFEST_DIR = ROOT / "quant" / "main" / "config" / "prediction_manifests"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_experiment_replace_production_release_20260715"
ARCHIVE_SUFFIX = "archive_before_20260715_experiment_replacement"


TARGETS: dict[str, dict[str, Any]] = {
    "1d": {
        "manifest": MANIFEST_DIR / "executable_1d_open_return_l4_formal_20260619.json",
        "formal_db": DATA_DIR / "production_assets" / "duckdb" / "l4_executable_1d_open_return_formal.duckdb",
        "formal_table": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate",
        "candidate_db": DATA_DIR
        / "experimental_assets"
        / "model-agent"
        / "l4_predictions"
        / "l4_1d_strict_train_selected_rank_blend_20260714.duckdb",
        "candidate_table": "stock_predict_data_model_agent_four_year_1d_strict_train_selected_rank_blend_20260714_research",
        "candidate_id": "research_1d_strict_train_selected_rank_blend_20260714",
        "lineage": "strict_train_selected_rank_blend_20260714",
        "formula": "rank_pct(0.99*rank_1d + 0.01*rank_10d)",
        "risk_note": "Recent20/63 RankIC slightly negative while TopN improves.",
    },
    "5d": {
        "manifest": MANIFEST_DIR / "executable_5d_open_return_l4_formal_20260620.json",
        "formal_db": DATA_DIR / "production_assets" / "duckdb" / "l4_executable_5d_open_return_formal.duckdb",
        "formal_table": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate",
        "candidate_db": DATA_DIR
        / "experimental_assets"
        / "model-agent"
        / "l4_predictions"
        / "l4_5d_front_combo_calibrator_20260714.duckdb",
        "candidate_table": "stock_predict_data_model_agent_four_year_5d_front_combo_calibrator_20260714_research",
        "candidate_id": "research_5d_front_combo_calibrator_20260714",
        "lineage": "front_combo_calibrator_20260714",
        "formula": "rank_pct(pred_5d_rank + 0.01*(front_top3_rank-0.5) when pred_5d_rank>=0.90 else pred_5d_rank)",
        "risk_note": "Improvement is mainly TopN; RankIC is nearly flat.",
    },
    "10d": {
        "manifest": MANIFEST_DIR / "executable_10d_open_return_l4_formal_20260617.json",
        "formal_db": DATA_DIR / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb",
        "formal_table": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate",
        "candidate_db": DATA_DIR
        / "experimental_assets"
        / "model-agent"
        / "l4_predictions"
        / "l4_10d_front_combo_calibrator_20260714.duckdb",
        "candidate_table": "stock_predict_data_model_agent_four_year_10d_front_combo_calibrator_20260714_research",
        "candidate_id": "research_10d_front_combo_calibrator_20260714",
        "lineage": "front_combo_calibrator_20260714",
        "formula": "rank_pct(pred_10d_rank + 0.03*(front_top5_rank-0.5) - 0.01*(front_top3_rank-0.5) when pred_10d_rank>=0.90 else pred_10d_rank)",
        "risk_note": "Full RankIC is slightly lower; Top5/Top10 and recent TopN improve.",
    },
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def table_quality(db: Path, table: str) -> dict[str, Any]:
    with duckdb.connect(str(db), read_only=True) as con:
        row = con.execute(
            f"""
            SELECT
                count(*) AS row_count,
                min(trade_date) AS min_trade_date,
                max(trade_date) AS max_trade_date,
                count(distinct trade_date) AS trade_days,
                count(distinct stock_code) AS stock_count,
                sum(case when pred_prob is null then 1 else 0 end) AS null_pred_prob,
                count(*) - count(distinct trade_date || '|' || stock_code) AS duplicate_key_groups,
                sum(case when stock_code like '%.BJ' then 1 else 0 end) AS bj_rows
            FROM {quote_ident(table)}
            """
        ).fetchone()
        latest_date = row[2]
        latest = con.execute(
            f"""
            SELECT count(*) AS latest_day_rows, count(distinct stock_code) AS latest_day_stocks
            FROM {quote_ident(table)}
            WHERE trade_date = ?
            """,
            [latest_date],
        ).fetchone()
        digest = hashlib.sha256()
        for trade_date, stock_code, pred_prob in con.execute(
            f"""
            SELECT trade_date, stock_code, pred_prob
            FROM {quote_ident(table)}
            ORDER BY trade_date, stock_code
            """
        ).fetchall():
            digest.update(f"{trade_date}|{stock_code}|{float(pred_prob):.17g}\n".encode("utf-8"))
    return {
        "row_count": int(row[0]),
        "min_trade_date": row[1],
        "max_trade_date": row[2],
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "null_pred_prob": int(row[5] or 0),
        "duplicate_key_groups": int(row[6] or 0),
        "bj_rows": int(row[7] or 0),
        "latest_day_rows": int(latest[0]),
        "latest_day_stocks": int(latest[1]),
        "pred_prob_sha256": digest.hexdigest(),
    }


def ensure_candidate_quality(label: str, spec: dict[str, Any]) -> dict[str, Any]:
    quality = table_quality(spec["candidate_db"], spec["candidate_table"])
    errors: list[str] = []
    if quality["max_trade_date"] != "20260714":
        errors.append(f"{label} candidate max_trade_date is {quality['max_trade_date']}, expected 20260714")
    if quality["latest_day_rows"] != 5197 or quality["latest_day_stocks"] != 5197:
        errors.append(f"{label} candidate latest day is not 5197/5197")
    for field in ["null_pred_prob", "duplicate_key_groups", "bj_rows"]:
        if quality[field] != 0:
            errors.append(f"{label} candidate {field}={quality[field]}, expected 0")
    if errors:
        raise RuntimeError("; ".join(errors))
    return quality


def publish_target(label: str, spec: dict[str, Any], audit_record: str, execute: bool) -> dict[str, Any]:
    before_quality = table_quality(spec["formal_db"], spec["formal_table"])
    candidate_quality = ensure_candidate_quality(label, spec)
    archive_table = f"{spec['formal_table']}_{ARCHIVE_SUFFIX}"
    archive_manifest = spec["manifest"].with_name(
        spec["manifest"].stem + f"_{ARCHIVE_SUFFIX}" + spec["manifest"].suffix
    )
    result = {
        "label": label,
        "candidate_id": spec["candidate_id"],
        "formal_db": str(spec["formal_db"]),
        "formal_table": spec["formal_table"],
        "candidate_db": str(spec["candidate_db"]),
        "candidate_table": spec["candidate_table"],
        "archive_table": archive_table,
        "archive_manifest": str(archive_manifest),
        "before_quality": before_quality,
        "candidate_quality": candidate_quality,
        "executed": execute,
    }
    if not execute:
        return result

    manifest = load_json(spec["manifest"])
    archive_payload = dict(manifest)
    archive_payload["archive_status"] = "rollback_archive_only"
    archive_payload["archive_reason"] = "archived_before_20260715_experiment_replacement"
    archive_payload["archived_at"] = now_iso()
    archive_payload["active_replaced_by_candidate"] = spec["candidate_id"]
    write_json(archive_manifest, archive_payload)

    with duckdb.connect(str(spec["formal_db"])) as con:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {quote_ident(archive_table)} AS
            SELECT * FROM {quote_ident(spec["formal_table"])}
            """
        )
        con.execute(f"ATTACH '{spec['candidate_db'].as_posix()}' AS cand (READ_ONLY)")
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {quote_ident(spec["formal_table"])} AS
            SELECT * FROM cand.{quote_ident(spec["candidate_table"])}
            """
        )

    after_quality = table_quality(spec["formal_db"], spec["formal_table"])
    archive_quality = table_quality(spec["formal_db"], archive_table)
    manifest["approval_status"] = "approved_for_l5"
    manifest["asset_role"] = "l4_formal_prediction_asset"
    manifest["source_type"] = "duckdb_table"
    manifest["db_path"] = "../../../data_file/production_assets/duckdb/" + spec["formal_db"].name
    manifest["table"] = spec["formal_table"]
    manifest["latest_formal_release"] = {
        "released_at": now_iso(),
        "release_id": "experiment_replacement_20260715",
        "candidate_id": spec["candidate_id"],
        "lineage": spec["lineage"],
        "formula": spec["formula"],
        "audit_record": audit_record,
        "user_authorization": "用户在模型线程明确要求：实验版替代生产版",
        "rollback_manifest": str(archive_manifest),
        "rollback_table": archive_table,
        "risk_note": spec["risk_note"],
        "quality": after_quality,
        "no_training": True,
        "no_signal": True,
        "no_backtest": True,
    }
    manifest["audit_record"] = audit_record
    manifest["approval_basis"] = "experiment_replacement_20260715_user_authorized_audit_approved"
    manifest["governance_status"] = "approved_l5_consumable_experiment_replacement_20260715"
    write_json(spec["manifest"], manifest)
    result["after_quality"] = after_quality
    result["archive_quality"] = archive_quality
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audit-record", default="")
    args = parser.parse_args()
    if args.execute and not args.audit_record:
        raise SystemExit("--audit-record is required when --execute is used")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        label: publish_target(label, spec, args.audit_record, args.execute)
        for label, spec in TARGETS.items()
    }
    report = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "publish_experiment_replacement_l4_20260715",
        "executed": args.execute,
        "audit_record": args.audit_record,
        "published_labels": list(TARGETS),
        "unchanged_labels": ["3d"],
        "results": results,
        "boundaries": {
            "no_training": True,
            "no_signal": True,
            "no_backtest": True,
            "no_l5_l7": True,
        },
    }
    path = REPORT_DIR / ("formal_release_report.json" if args.execute else "dry_run_report.json")
    write_json(path, report)
    print(json.dumps({"report": str(path), "executed": args.execute}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
