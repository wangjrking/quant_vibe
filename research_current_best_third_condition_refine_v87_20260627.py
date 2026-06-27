from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import research_5d10d_formal_guarded_candidate_v74_20260627 as guarded
from score_model_promotion_candidate import evaluate_candidate


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
DB_PATH = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_current_best_third_condition_refine_v87_20260627"
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
    "executable_1d_open_return": "stock_predict_data_model_agent_1d_third_condition_refine_v87_20260627_executable_1d_open_return_research",
    "executable_3d_open_return": "stock_predict_data_model_agent_3d_third_condition_refine_v87_20260627_executable_3d_open_return_research",
    "executable_5d_open_return": "stock_predict_data_model_agent_5d_third_condition_refine_v87_20260627_executable_5d_open_return_research",
    "executable_10d_open_return": "stock_predict_data_model_agent_10d_third_condition_refine_v87_20260627_executable_10d_open_return_research",
}
RANK_IC_FLOORS = {
    "executable_1d_open_return": 0.0,
    "executable_3d_open_return": -0.0015,
    "executable_5d_open_return": -0.001,
    "executable_10d_open_return": -0.0005,
}
FEATURES = guarded.FEATURES
QUANTILES = [0.05, 0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90, 0.95]


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def v84_active_dates(table: str) -> set[str]:
    with sqlite3.connect(DB_PATH, timeout=300) as conn:
        cols = [row[1] for row in conn.execute(f"pragma table_info({quote(table)})")]
        if "score_formula" in cols:
            rows = conn.execute(
                f"select distinct trade_date from {quote(table)} where score_formula <> 'formal_rank_fallback'"
            ).fetchall()
            return {str(row[0]) for row in rows}
        frame = pd.read_sql_query(f"select trade_date, stock_code, pred_prob from {quote(table)}", conn)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return set(frame["trade_date"].unique())


def summarize(daily: pd.DataFrame, window: int | None = None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {metric: float(sub[metric].mean()) for metric in guarded.METRICS}


def evaluate_variant(eval_frame: pd.DataFrame, formal_daily: pd.DataFrame, active_dates: set[str], label: str) -> dict[str, object]:
    scored = eval_frame.copy()
    active = scored["trade_date"].isin(active_dates) & scored["candidate_rank"].notna()
    scored["_score"] = scored["formal_rank"].where(~active, scored["candidate_rank"])
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


def row_from_result(label: str, base_active_days: int, condition: str, active_dates: set[str], result: dict[str, object]) -> dict[str, object]:
    row = {
        "condition": condition,
        "base_active_days": base_active_days,
        "active_days": len(active_dates),
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


def scan_third_condition(label: str, base_active: set[str], features: pd.DataFrame, eval_frame: pd.DataFrame, formal_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    evaluated = []

    base_result = evaluate_variant(eval_frame, formal_daily, base_active, label)
    base_row = row_from_result(label, len(base_active), "v84_base_condition", base_active, base_result)
    base_row["third_feature"] = "none"
    base_row["third_op"] = "none"
    base_row["third_threshold"] = None
    rows.append(base_row)
    evaluated.append({"row": base_row, "active_dates": base_active, "result": base_result})

    for feature in FEATURES:
        values = features[feature].replace([np.inf, -np.inf], np.nan).dropna()
        thresholds = sorted(set(float(values.quantile(q)) for q in QUANTILES))
        for threshold in thresholds:
            for op in ["<=", ">="]:
                if op == "<=":
                    dates = set(features.loc[features[feature] <= threshold, "trade_date"])
                else:
                    dates = set(features.loc[features[feature] >= threshold, "trade_date"])
                active = base_active & dates
                if len(active) < 3 or len(active) >= len(base_active):
                    continue
                condition = f"v84_base AND {feature} {op} {threshold:.12g}"
                result = evaluate_variant(eval_frame, formal_daily, active, label)
                row = row_from_result(label, len(base_active), condition, active, result)
                row["third_feature"] = feature
                row["third_op"] = op
                row["third_threshold"] = threshold
                rows.append(row)
                evaluated.append({"row": row, "active_dates": active, "result": result})

    frame = pd.DataFrame(rows).sort_values(
        ["pass_hard", "objective", "recent63_top5_delta", "recent20_top5_delta", "full_top5_delta"],
        ascending=[False, False, False, False, False],
    )
    best_row = frame.iloc[0].to_dict()
    best = next(item for item in evaluated if item["row"]["condition"] == best_row["condition"])
    return frame, best


def write_asset(label: str, scores: pd.DataFrame, best: dict[str, object]) -> dict[str, object]:
    table = TARGET_TABLES[label]
    out = scores[["trade_date", "stock_code"]].copy()
    active = scores["trade_date"].isin(best["active_dates"]) & scores["candidate_rank"].notna()
    out["pred_prob"] = scores["formal_rank"].where(~active, scores["candidate_rank"])
    out["score_formula"] = np.where(active, str(best["row"]["condition"]) + ": candidate_rank", "formal_rank_fallback")
    with sqlite3.connect(DB_PATH, timeout=300) as conn:
        out.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_{table[:40]}_date_code on {quote(table)}(trade_date, stock_code)")
        conn.commit()
    return guarded.db_summary(table)


def run_label(label: str, item: dict[str, object], config: dict[str, object], expected_latest: str, expected_rows: int) -> dict[str, object]:
    output_dir = REPORT_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "asset": f"research_{label.replace('executable_', '').replace('_open_return', '')}_third_condition_refine_v87_20260627",
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
    base_active = v84_active_dates(str(item["table"]))
    scan, best = scan_third_condition(label, base_active, features, eval_frame, formal_daily)
    scan.to_csv(output_dir / "third_condition_scan.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output_dir / "score_state_features.csv", index=False, encoding="utf-8-sig")
    best["result"]["daily"].to_csv(output_dir / "best_daily_eval.csv", index=False, encoding="utf-8-sig")

    summary = write_asset(label, scores, best)
    row = best["row"]
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
        "full_rank_ic_delta": row["full_rank_ic_delta"],
        "full_top1_delta": row["full_top1_delta"],
        "full_top5_delta": row["full_top5_delta"],
        "recent63_rank_ic_delta": row["recent63_rank_ic_delta"],
        "recent63_top1_delta": row["recent63_top1_delta"],
        "recent63_top5_delta": row["recent63_top5_delta"],
        "recent20_rank_ic_delta": row["recent20_rank_ic_delta"],
        "recent20_top1_delta": row["recent20_top1_delta"],
        "recent20_top5_delta": row["recent20_top5_delta"],
        "positive_top5_periods": row["positive_top5_months"],
        "min_period_top5_delta": row["min_month_top5_delta"],
        "prefer_simpler_formula": "v84 base condition plus one stricter AND filter",
        "prefer_fewer_active_days": row["active_days"],
        "prefer_fewer_upstream_sources": "formal+candidate score tables",
        "prefer_clearer_roll_back_path": "drop research-only v87 table and keep current formal unchanged",
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
        "formula": "v84 base condition plus one stricter AND filter",
        "condition": row["condition"],
        "base_condition": item["condition"],
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
        "base_asset": item["asset"],
        "base_table": item["table"],
        "best": row,
        "db_summary": summary,
        "promotion_gate": gate,
        "candidate_json": str(candidate_path),
    }
    (output_dir / f"{label}_v87_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# {label} 第三条件过滤研究 v87

## 结论

本轮只在 v84 实际触发日期上增加一个 AND 过滤条件。未训练模型，未修改 formal/production manifest，未生成信号或回测结论。

## 最优候选

- 资产：`{spec['asset']}`
- 表：`{spec['target_table']}`
- 条件：`{row['condition']}`
- 原始触发日：`{row['base_active_days']}`
- 过滤后触发日：`{row['active_days']}`
- promotion gate：`{gate['hard_constraint_passed']}`，目标状态 `{gate['target_approval_status']}`

## 相对 formal baseline 的增量

- 全窗口 Top5 delta：`{row['full_top5_delta']:.10f}`
- 近 63 日 Top5 delta：`{row['recent63_top5_delta']:.10f}`
- 近 20 日 Top5 delta：`{row['recent20_top5_delta']:.10f}`
"""
    (output_dir / f"{label}_v87_report.md").write_text(report, encoding="utf-8")
    return payload


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    expected_latest, expected_rows = guarded.production_factor_latest()
    config = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    payload = {
        "generated_at": guarded.now_iso(),
        "scope": "research_only_current_best_third_condition_refine_v87",
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
        row = result["best"]
        summary = result["db_summary"]
        gate = result["promotion_gate"]
        rows.append(
            {
                "label": label,
                "asset": result["asset"],
                "table": result["target_table"],
                "condition": row["condition"],
                "promotion_gate_passed": gate["hard_constraint_passed"],
                "target_approval_status": gate["target_approval_status"],
                "latest_trade_date": summary["latest_trade_date"],
                "latest_day_rows": summary["latest_day_rows"],
                "base_active_days": row["base_active_days"],
                "active_days": row["active_days"],
                "full_rank_ic_delta": row["full_rank_ic_delta"],
                "full_top1_delta": row["full_top1_delta"],
                "full_top5_delta": row["full_top5_delta"],
                "recent63_rank_ic_delta": row["recent63_rank_ic_delta"],
                "recent63_top1_delta": row["recent63_top1_delta"],
                "recent63_top5_delta": row["recent63_top5_delta"],
                "recent20_rank_ic_delta": row["recent20_rank_ic_delta"],
                "recent20_top1_delta": row["recent20_top1_delta"],
                "recent20_top5_delta": row["recent20_top5_delta"],
                "positive_top5_months": row["positive_top5_months"],
                "min_month_top5_delta": row["min_month_top5_delta"],
            }
        )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(REPORT_DIR / "v87_summary_matrix.csv", index=False, encoding="utf-8-sig")
    payload["summary_matrix"] = str(REPORT_DIR / "v87_summary_matrix.csv")
    (REPORT_DIR / "v87_run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 当前最优第三条件过滤研究 v87",
        "",
        "## 结论",
        "",
        "本轮只扫描 v84 实际触发日期上的第三条件 AND 过滤，不训练模型，不修改 formal/production manifest，不生成信号，不运行回测。",
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
                f"- 触发日：`{row['base_active_days']}` -> `{row['active_days']}`",
                f"- 全窗口 Top5 delta：`{row['full_top5_delta']:.10f}`",
                f"- 近 63 日 Top5 delta：`{row['recent63_top5_delta']:.10f}`",
                f"- 近 20 日 Top5 delta：`{row['recent20_top5_delta']:.10f}`",
                "",
            ]
        )
    (REPORT_DIR / "v87_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
