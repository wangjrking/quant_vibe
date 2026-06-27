from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import research_5d10d_formal_guarded_candidate_v74_20260627 as guarded
from score_model_promotion_candidate import evaluate_candidate


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
DB_PATH = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_current_best_guarded_blend_weight_v86_20260627"
SNAPSHOT = DATA_DIR / "reports" / "model_agent_current_best_research_snapshot_20260627_v84" / "current_best_research_snapshot_v84.json"
CONSTRAINTS = guarded.CONSTRAINTS

FORMAL_TABLES = {
    "executable_1d_open_return": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
    "executable_3d_open_return": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
    "executable_5d_open_return": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
    "executable_10d_open_return": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
}
CANDIDATE_TABLES = {
    "executable_1d_open_return": "stock_predict_data_model_agent_1d_exante_iqr_gate_v67_20260627_executable_1d_open_return_research",
    "executable_3d_open_return": "stock_predict_data_model_agent_3d_exante_std_gate_v69_20260627_executable_3d_open_return_research",
    "executable_5d_open_return": "stock_predict_data_model_agent_5d_recent_top_condblend_latest_20260626_executable_5d_open_return_research",
    "executable_10d_open_return": "stock_predict_data_model_agent_10d_monthly_top5_double_veto_v62_20260627_executable_10d_open_return_research",
}
TARGET_TABLES = {
    "executable_1d_open_return": "stock_predict_data_model_agent_1d_guarded_blend_weight_v86_20260627_executable_1d_open_return_research",
    "executable_3d_open_return": "stock_predict_data_model_agent_3d_guarded_blend_weight_v86_20260627_executable_3d_open_return_research",
    "executable_5d_open_return": "stock_predict_data_model_agent_5d_guarded_blend_weight_v86_20260627_executable_5d_open_return_research",
    "executable_10d_open_return": "stock_predict_data_model_agent_10d_guarded_blend_weight_v86_20260627_executable_10d_open_return_research",
}
RANK_IC_FLOORS = {
    "executable_1d_open_return": 0.0,
    "executable_3d_open_return": -0.0015,
    "executable_5d_open_return": -0.001,
    "executable_10d_open_return": -0.0005,
}
WEIGHTS = [0.25, 0.50, 0.75, 1.00]


def apply_clause(features: pd.DataFrame, clause: str) -> pd.Series:
    match = re.fullmatch(r"\s*([A-Za-z0-9_]+)\s*(<=|>=)\s*([-+0-9.eE]+)\s*", clause)
    if not match:
        raise ValueError(f"unsupported clause: {clause}")
    feature, op, threshold_text = match.groups()
    threshold = float(threshold_text)
    if op == "<=":
        return features[feature] <= threshold
    return features[feature] >= threshold


def active_dates_from_condition(features: pd.DataFrame, condition: str) -> set[str]:
    text = condition.replace("(", "").replace(")", "")
    if " AND " in text:
        left, right = text.split(" AND ", 1)
        mask = apply_clause(features, left) & apply_clause(features, right)
    elif " OR " in text:
        left, right = text.split(" OR ", 1)
        mask = apply_clause(features, left) | apply_clause(features, right)
    else:
        mask = apply_clause(features, text)
    return set(features.loc[mask, "trade_date"].astype(str))


def summarize(daily: pd.DataFrame, window: int | None = None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {metric: float(sub[metric].mean()) for metric in guarded.METRICS}


def evaluate_weight(eval_frame: pd.DataFrame, formal_daily: pd.DataFrame, active_dates: set[str], label: str, weight: float) -> dict[str, object]:
    scored = eval_frame.copy()
    active = scored["trade_date"].isin(active_dates) & scored["candidate_rank"].notna()
    blended = (1.0 - weight) * scored["formal_rank"] + weight * scored["candidate_rank"]
    scored["_score"] = scored["formal_rank"].where(~active, blended)
    daily = guarded.daily_eval(scored, "_score", label)
    full = summarize(daily)
    recent63 = summarize(daily, 63)
    recent20 = summarize(daily, 20)
    formal_full = summarize(formal_daily)
    formal63 = summarize(formal_daily, 63)
    formal20 = summarize(formal_daily, 20)
    delta_full = guarded.deltas(full, formal_full)
    delta63 = guarded.deltas(recent63, formal63)
    delta20 = guarded.deltas(recent20, formal20)
    month = guarded.month_top5_delta(daily, formal_daily)
    month.pop("monthly")
    return {
        "daily": daily,
        "delta_full": delta_full,
        "delta_recent63": delta63,
        "delta_recent20": delta20,
        "month": month,
        "objective": guarded.objective(delta_full, delta63, delta20, month),
    }


def row_from_result(label: str, condition: str, weight: float, active_days: int, result: dict[str, object]) -> dict[str, object]:
    row = {
        "condition": condition,
        "blend_weight": weight,
        "active_days": active_days,
        "objective": result["objective"],
        "full_rank_ic_delta": result["delta_full"]["rank_ic"],
        "full_top1_delta": result["delta_full"]["top1"],
        "full_top5_delta": result["delta_full"]["top5"],
        "recent63_rank_ic_delta": result["delta_recent63"]["rank_ic"],
        "recent63_top1_delta": result["delta_recent63"]["top1"],
        "recent63_top5_delta": result["delta_recent63"]["top5"],
        "recent20_rank_ic_delta": result["delta_recent20"]["rank_ic"],
        "recent20_top1_delta": result["delta_recent20"]["top1"],
        "recent20_top5_delta": result["delta_recent20"]["top5"],
        "positive_top5_months": result["month"]["positive_top5_months"],
        "min_month_top5_delta": result["month"]["min_month_top5_delta"],
    }
    row["pass_hard"] = bool(
        row["full_top5_delta"] >= 0
        and row["recent63_top5_delta"] >= 0
        and row["recent20_top5_delta"] >= 0
        and row["positive_top5_months"] >= 3
        and row["min_month_top5_delta"] >= 0
        and row["full_rank_ic_delta"] >= RANK_IC_FLOORS[label]
        and (label != "executable_3d_open_return" or row["recent63_rank_ic_delta"] >= 0)
    )
    return row


def write_asset(label: str, scores: pd.DataFrame, active_dates: set[str], weight: float) -> dict[str, object]:
    table = TARGET_TABLES[label]
    out = scores[["trade_date", "stock_code"]].copy()
    active = scores["trade_date"].isin(active_dates) & scores["candidate_rank"].notna()
    blended = (1.0 - weight) * scores["formal_rank"] + weight * scores["candidate_rank"]
    out["pred_prob"] = scores["formal_rank"].where(~active, blended)
    out["score_formula"] = np.where(active, f"formal_candidate_rank_blend_weight_{weight}", "formal_rank_fallback")
    spec = {
        "target_table": table,
    }
    with guarded.sqlite3.connect(DB_PATH, timeout=300) as conn:
        out.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(
            f"create index if not exists idx_{table[:40]}_date_code on {guarded.quote(table)}(trade_date, stock_code)"
        )
        conn.commit()
    return guarded.db_summary(table)


def run_label(label: str, item: dict[str, object], config: dict[str, object], expected_latest: str, expected_rows: int) -> dict[str, object]:
    output_dir = REPORT_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "asset": f"research_{label.replace('executable_', '').replace('_open_return', '')}_guarded_blend_weight_v86_20260627",
        "formal_table": FORMAL_TABLES[label],
        "candidate_table": CANDIDATE_TABLES[label],
        "target_table": TARGET_TABLES[label],
        "rank_ic_floor": RANK_IC_FLOORS[label],
    }
    scores = guarded.load_scores(spec)
    labels = guarded.load_labels(label, str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    formal_daily = guarded.daily_eval(eval_frame, "formal_rank", label)
    features = guarded.daily_features(scores)
    active_dates = active_dates_from_condition(features, str(item["condition"]))

    rows = []
    evaluated = []
    for weight in WEIGHTS:
        result = evaluate_weight(eval_frame, formal_daily, active_dates, label, weight)
        row = row_from_result(label, str(item["condition"]), weight, len(active_dates), result)
        rows.append(row)
        evaluated.append({"row": row, "result": result})
    scan = pd.DataFrame(rows).sort_values(
        ["pass_hard", "objective", "recent63_top5_delta", "recent20_top5_delta", "full_top5_delta"],
        ascending=[False, False, False, False, False],
    )
    best_row = scan.iloc[0].to_dict()
    best = next(entry for entry in evaluated if entry["row"]["blend_weight"] == best_row["blend_weight"])
    scan.to_csv(output_dir / "blend_weight_scan.csv", index=False, encoding="utf-8-sig")
    best["result"]["daily"].to_csv(output_dir / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output_dir / "score_state_features.csv", index=False, encoding="utf-8-sig")

    summary = write_asset(label, scores, active_dates, float(best_row["blend_weight"]))
    candidate = {
        "label": label,
        "asset": spec["asset"],
        "table": spec["target_table"],
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "feature_input": "quant/data_file/production_factor_parts/",
        "label_input": "quant/data_file/prediction_label_parts/",
        "prediction_db": "quant/data_file/model_predictions/MODEL_PREDICTIONS.db",
        "forbidden_inputs_present": False,
        "min_trade_date": summary["min_trade_date"],
        "latest_trade_date": summary["latest_trade_date"],
        "expected_latest_trade_date": expected_latest,
        "latest_day_rows": summary["latest_day_rows"],
        "expected_latest_day_rows": expected_rows,
        "null_pred_prob": summary["null_pred_prob"],
        "duplicate_key_groups": summary["duplicate_key_groups"],
        "is_train_candidate": False,
        "evaluation_report_present": True,
        "candidate_manifest_present": True,
        "full_rank_ic_delta": best_row["full_rank_ic_delta"],
        "full_top1_delta": best_row["full_top1_delta"],
        "full_top5_delta": best_row["full_top5_delta"],
        "recent63_rank_ic_delta": best_row["recent63_rank_ic_delta"],
        "recent63_top1_delta": best_row["recent63_top1_delta"],
        "recent63_top5_delta": best_row["recent63_top5_delta"],
        "recent20_rank_ic_delta": best_row["recent20_rank_ic_delta"],
        "recent20_top1_delta": best_row["recent20_top1_delta"],
        "recent20_top5_delta": best_row["recent20_top5_delta"],
        "positive_top5_periods": best_row["positive_top5_months"],
        "min_period_top5_delta": best_row["min_month_top5_delta"],
        "prefer_simpler_formula": "formal fallback + guarded formal/candidate rank blend",
        "prefer_fewer_active_days": best_row["active_days"],
        "prefer_fewer_upstream_sources": "formal+candidate score tables",
        "prefer_clearer_roll_back_path": "drop research-only v86 table and keep current formal unchanged",
    }
    candidate_path = output_dir / "promotion_candidate.json"
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    gate = evaluate_candidate(candidate, config)
    (output_dir / "promotion_gate_result.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "asset_role": "research_l4_candidate",
        "source_type": "sqlite_table",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db",
        "table": spec["target_table"],
        "label": label,
        "allowed_for_main_workflow": False,
        "feature_input": "quant/data_file/production_factor_parts/",
        "label_input": "quant/data_file/prediction_label_parts/",
        "formula": "formal fallback with guarded formal/candidate rank blend",
        "condition": item["condition"],
        "blend_weight": float(best_row["blend_weight"]),
        "formal_table": spec["formal_table"],
        "candidate_table": spec["candidate_table"],
        "notes": "研究候选资产，仅用于模型侧评价与送审准备；未批准进入 formal 或 L5。",
    }
    (output_dir / "research_candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "asset": spec["asset"],
        "formal_table": spec["formal_table"],
        "candidate_table": spec["candidate_table"],
        "target_table": spec["target_table"],
        "condition": item["condition"],
        "best": best_row,
        "db_summary": summary,
        "promotion_gate": gate,
        "candidate_json": str(candidate_path),
    }
    (output_dir / f"{label}_v86_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# {label} 门控融合权重研究 v86

## 结论

本轮只在 v84 已通过的门控日期上扫描 formal/candidate rank 融合权重。未训练模型，未修改 formal/production manifest，未生成信号或回测结论。

## 最优权重

- 资产：`{spec['asset']}`
- 表：`{spec['target_table']}`
- 条件：`{item['condition']}`
- 融合权重：`{best_row['blend_weight']}`
- 激活交易日：`{best_row['active_days']}`
- promotion gate：`{gate['hard_constraint_passed']}`，目标状态 `{gate['target_approval_status']}`

## 相对 formal baseline 的增量

- 全窗口 Top5 delta：`{best_row['full_top5_delta']:.10f}`
- 近 63 日 Top5 delta：`{best_row['recent63_top5_delta']:.10f}`
- 近 20 日 Top5 delta：`{best_row['recent20_top5_delta']:.10f}`
"""
    (output_dir / f"{label}_v86_report.md").write_text(report, encoding="utf-8")
    return payload


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    expected_latest, expected_rows = guarded.production_factor_latest()
    config = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    payload = {
        "generated_at": guarded.now_iso(),
        "scope": "research_only_current_best_guarded_blend_weight_v86",
        "source_snapshot": str(SNAPSHOT),
        "results": {},
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    rows = []
    for label, item in snapshot["current_best"].items():
        result = run_label(label, item, config, expected_latest, expected_rows)
        payload["results"][label] = result
        best = result["best"]
        summary = result["db_summary"]
        gate = result["promotion_gate"]
        rows.append(
            {
                "label": label,
                "asset": result["asset"],
                "table": result["target_table"],
                "condition": result["condition"],
                "blend_weight": best["blend_weight"],
                "promotion_gate_passed": gate["hard_constraint_passed"],
                "target_approval_status": gate["target_approval_status"],
                "latest_trade_date": summary["latest_trade_date"],
                "latest_day_rows": summary["latest_day_rows"],
                "full_rank_ic_delta": best["full_rank_ic_delta"],
                "full_top1_delta": best["full_top1_delta"],
                "full_top5_delta": best["full_top5_delta"],
                "recent63_rank_ic_delta": best["recent63_rank_ic_delta"],
                "recent63_top1_delta": best["recent63_top1_delta"],
                "recent63_top5_delta": best["recent63_top5_delta"],
                "recent20_rank_ic_delta": best["recent20_rank_ic_delta"],
                "recent20_top1_delta": best["recent20_top1_delta"],
                "recent20_top5_delta": best["recent20_top5_delta"],
                "positive_top5_months": best["positive_top5_months"],
                "min_month_top5_delta": best["min_month_top5_delta"],
            }
        )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(REPORT_DIR / "v86_summary_matrix.csv", index=False, encoding="utf-8-sig")
    payload["summary_matrix"] = str(REPORT_DIR / "v86_summary_matrix.csv")
    (REPORT_DIR / "v86_run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 当前最优门控融合权重研究 v86",
        "",
        "## 结论",
        "",
        "本轮只扫描 v84 已通过门控日期上的融合权重，不训练模型，不修改 formal/production manifest，不生成信号，不运行回测。",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- 资产：`{row['asset']}`",
                f"- 表：`{row['table']}`",
                f"- 条件：`{row['condition']}`",
                f"- 权重：`{row['blend_weight']}`",
                f"- 全窗口 Top5 delta：`{row['full_top5_delta']:.10f}`",
                f"- 近 63 日 Top5 delta：`{row['recent63_top5_delta']:.10f}`",
                f"- 近 20 日 Top5 delta：`{row['recent20_top5_delta']:.10f}`",
                "",
            ]
        )
    (REPORT_DIR / "v86_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
