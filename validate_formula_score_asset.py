from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


RESEARCH_APPROVAL_STATUSES = {
    "research_only_not_approved_for_l4_or_l5",
    "research_only_not_for_l5",
}


def _quote_sql_name(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_prediction_table(db_path: str | Path, table: str) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    summary: dict[str, Any] = {}
    path = Path(db_path)
    if not path.exists():
        return [f"prediction db does not exist: {path}"], summary
    if not table:
        return ["target_table is required for prediction db validation"], summary

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
        duplicate_key_groups = conn.execute(
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
        "duplicate_key_groups": int(duplicate_key_groups or 0),
    }
    if summary["row_count"] <= 0:
        errors.append("prediction table row_count must be positive")
    if summary["null_pred_prob"] != 0:
        errors.append("prediction table null pred_prob count must be 0")
    if summary["duplicate_key_groups"] != 0:
        errors.append("prediction table duplicate key groups must be 0")
    return errors, summary


def validate_formula_score_asset(
    report_json: str | Path,
    *,
    prediction_db_path: str | Path,
    min_full_rank_ic_delta: float = 0.0,
    min_full_top5_delta: float = 0.0,
    min_full_top10_delta: float = 0.0,
) -> dict[str, Any]:
    report = _read_json(report_json)
    errors: list[str] = []
    warnings: list[str] = []

    approval_status = str(report.get("approval_status", ""))
    if approval_status not in RESEARCH_APPROVAL_STATUSES:
        errors.append("approval_status must be research-only")

    asset_role = str(report.get("asset_role", ""))
    if asset_role.startswith("l4_formal") or asset_role == "formal_prediction_asset":
        errors.append("asset_role must not be formal")

    target_table = str(report.get("target_table") or report.get("table") or "")
    if not target_table:
        errors.append("target_table is required")
    elif "_research" not in target_table:
        errors.append("target_table must contain _research")

    sources = report.get("sources")
    if not isinstance(sources, dict) or not sources:
        errors.append("sources must be a non-empty object")
    else:
        if not sources.get("formal"):
            errors.append("sources.formal is required")
        if len([value for value in sources.values() if value]) < 2:
            errors.append("formula score asset must declare at least two source assets")

    best_params = report.get("best_params")
    if not isinstance(best_params, dict) or not best_params:
        errors.append("best_params must be a non-empty object")
    elif "objective" not in best_params:
        warnings.append("best_params.objective is missing")

    governance = report.get("governance") or {}
    if governance.get("no_training") is not True:
        errors.append("governance.no_training must be true")
    if governance.get("no_production_manifest_change") is not True:
        errors.append("governance.no_production_manifest_change must be true")
    if governance.get("no_signal") is not True:
        errors.append("governance.no_signal must be true")
    if governance.get("no_backtest") is not True:
        errors.append("governance.no_backtest must be true")

    deltas = report.get("target_delta_vs_formal") or {}
    full_delta = deltas.get("full") or {}
    metric_checks = [
        ("full delta_rank_ic", full_delta.get("delta_rank_ic"), min_full_rank_ic_delta),
        ("full delta_top5", full_delta.get("delta_top5"), min_full_top5_delta),
        ("full delta_top10", full_delta.get("delta_top10"), min_full_top10_delta),
    ]
    for name, value, floor in metric_checks:
        if value is None:
            errors.append(f"{name} is required")
        elif _as_float(value) < float(floor):
            errors.append(f"{name} below floor {floor}")

    table_errors, prediction_table_summary = _validate_prediction_table(prediction_db_path, target_table)
    errors.extend(table_errors)

    return {
        "ok": not errors,
        "status": "formula_score_asset_ready_for_model_side_audit" if not errors else "not_ready",
        "label": report.get("label"),
        "target_table": target_table,
        "sources": sources,
        "best_params": best_params,
        "metric_floors": {
            "min_full_rank_ic_delta": min_full_rank_ic_delta,
            "min_full_top5_delta": min_full_top5_delta,
            "min_full_top10_delta": min_full_top10_delta,
        },
        "prediction_table_summary": prediction_table_summary,
        "errors": errors,
        "warnings": warnings,
        "boundaries": {
            "no_training": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate a research-only formula/blend score asset.")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--prediction-db-path", required=True)
    parser.add_argument("--min-full-rank-ic-delta", type=float, default=0.0)
    parser.add_argument("--min-full-top5-delta", type=float, default=0.0)
    parser.add_argument("--min-full-top10-delta", type=float, default=0.0)
    parser.add_argument("--output-json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = validate_formula_score_asset(
        args.report_json,
        prediction_db_path=args.prediction_db_path,
        min_full_rank_ic_delta=args.min_full_rank_ic_delta,
        min_full_top5_delta=args.min_full_top5_delta,
        min_full_top10_delta=args.min_full_top10_delta,
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
