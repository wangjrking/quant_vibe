from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Iterable


RESEARCH_APPROVAL_STATUSES = {
    "research_only_not_approved_for_l4_or_l5",
    "research_only_not_for_l5",
}

SUCCESS_STATUSES = {"ok", "skipped_existing", "skipped"}


def _fold_suffix(fold: int) -> str:
    return f"{int(fold):02d}"


def _glob_one(root: Path, pattern: str) -> bool:
    return any(root.glob(pattern))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _validate_fold_results(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return errors
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    except Exception as exc:
        return [f"fold_results.csv is not readable: {exc}"]
    for row in rows:
        fold = str(row.get("fold", "")).strip() or "unknown"
        status = str(row.get("status", "")).strip().lower()
        returncode = str(row.get("returncode", "")).strip()
        if status and status not in SUCCESS_STATUSES:
            errors.append(f"fold_results contains failed fold {fold}")
        if returncode not in ("", "0"):
            errors.append(f"fold_results contains nonzero returncode for fold {fold}")
    return errors


def _quote_sql_name(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _validate_prediction_table(db_path: str | Path, table: str) -> tuple[list[str], dict]:
    errors: list[str] = []
    summary: dict = {}
    path = Path(db_path)
    if not path.exists():
        return [f"prediction db does not exist: {path}"], summary
    if not table:
        return ["prediction_table is required when prediction_db_path is provided"], summary

    conn = sqlite3.connect(path)
    try:
        exists = conn.execute(
            "select 1 from sqlite_master where type='table' and name=?",
            (str(table),),
        ).fetchone()
        if not exists:
            return [f"prediction table does not exist: {table}"], summary
        quoted = _quote_sql_name(table)
        row = conn.execute(
            f"""
            select count(*),
                   min(trade_date),
                   max(trade_date),
                   count(distinct trade_date),
                   count(distinct stock_code),
                   sum(case when pred_prob is null then 1 else 0 end)
            from {quoted}
            """
        ).fetchone()
        dup = conn.execute(
            f"""
            select count(*) from (
              select trade_date, stock_code, count(*) c
              from {quoted}
              group by trade_date, stock_code
              having c > 1
            )
            """
        ).fetchone()[0]
    finally:
        conn.close()

    summary = {
        "row_count": int(row[0] or 0),
        "min_trade_date": row[1],
        "max_trade_date": row[2],
        "trade_days": int(row[3] or 0),
        "stock_count": int(row[4] or 0),
        "null_pred_prob": int(row[5] or 0),
        "duplicate_key_groups": int(dup or 0),
    }
    if summary["row_count"] <= 0:
        errors.append("prediction table row_count must be positive")
    if summary["null_pred_prob"] != 0:
        errors.append("prediction table null pred_prob count must be 0")
    if summary["duplicate_key_groups"] != 0:
        errors.append("prediction table duplicate key groups must be 0")
    return errors, summary


def validate_research_run(
    run_dir: str | Path,
    *,
    label: str,
    expected_folds: Iterable[int],
    prediction_db_path: str | Path | None = None,
    prediction_table: str | None = None,
) -> dict:
    root = Path(run_dir)
    folds = [int(fold) for fold in expected_folds]
    errors: list[str] = []
    warnings: list[str] = []

    if not root.exists():
        errors.append(f"run_dir does not exist: {root}")
        return {"ok": False, "errors": errors, "warnings": warnings, "folds_checked": folds}

    for rel in [
        "merge_meta.json",
        "prediction_manifest.json",
        "fold_results.csv",
        "evaluation_summary.json",
        "evaluation_daily.csv",
    ]:
        if not (root / rel).exists():
            errors.append(f"missing {rel}")

    errors.extend(_validate_fold_results(root / "fold_results.csv"))

    for fold in folds:
        suffix = _fold_suffix(fold)
        required_files = [
            f"fold_predictions/fold{suffix}.parquet",
            f"fold_logs/fold{suffix}.log",
            f"fold_summaries/fold{suffix}.csv",
            f"models/model_fold{suffix}.json",
            f"models/model_fold{suffix}_metadata.json",
        ]
        for rel in required_files:
            if not (root / rel).exists():
                errors.append(f"missing {rel}")
        selected_pattern = f"feature_scores/**/selected_features_{label}_rolling_fold{fold}.json"
        score_pattern = f"feature_scores/**/feature_ic_scores_{label}_rolling_fold{fold}.csv"
        if not _glob_one(root, selected_pattern):
            errors.append(f"missing selected features for fold {fold}")
        if not _glob_one(root, score_pattern):
            errors.append(f"missing feature IC scores for fold {fold}")

    manifest_path = root / "prediction_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        approval_status = str(manifest.get("approval_status", ""))
        if approval_status not in RESEARCH_APPROVAL_STATUSES:
            errors.append("prediction_manifest approval_status must be research-only")
        table = str(
            manifest.get("prediction_table")
            or manifest.get("table")
            or manifest.get("prediction_table_name")
            or ""
        )
        if table and "_research" not in table:
            errors.append("prediction table must contain _research")
        asset_role = str(manifest.get("asset_role", ""))
        if asset_role and "formal" in asset_role:
            errors.append("prediction_manifest asset_role must not be formal")
        if prediction_table is None and table:
            prediction_table = table
    else:
        warnings.append("prediction_manifest not available for status validation")

    prediction_table_summary = {}
    if prediction_db_path is not None:
        table_errors, prediction_table_summary = _validate_prediction_table(prediction_db_path, prediction_table or "")
        errors.extend(table_errors)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "folds_checked": folds,
        "prediction_table_summary": prediction_table_summary,
    }


def _parse_folds(value: str) -> list[int]:
    folds = []
    for piece in str(value).split(","):
        piece = piece.strip()
        if piece:
            folds.append(int(piece))
    return folds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate research rolling-training run artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-folds", required=True, help="Comma-separated fold ids, for example 1,2,3")
    parser.add_argument("--prediction-db-path")
    parser.add_argument("--prediction-table")
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = validate_research_run(
        args.run_dir,
        label=args.label,
        expected_folds=_parse_folds(args.expected_folds),
        prediction_db_path=args.prediction_db_path,
        prediction_table=args.prediction_table,
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
