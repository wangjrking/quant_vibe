from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import research_5d10d_formal_guarded_candidate_v74_20260627 as guarded
from score_model_promotion_candidate import evaluate_candidate


DATA_DIR = guarded.DATA_DIR
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_formal_guarded_v77_20260627"
CONSTRAINTS = guarded.CONSTRAINTS

LABEL = "executable_3d_open_return"
SPEC = {
    "asset": "research_3d_formal_guarded_exante_v77_20260627",
    "formal_table": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
    "candidate_table": "stock_predict_data_model_agent_3d_exante_std_gate_v69_20260627_executable_3d_open_return_research",
    "target_table": "stock_predict_data_model_agent_3d_formal_guarded_exante_v77_20260627_executable_3d_open_return_research",
    "rank_ic_floor": -0.0015,
}
FEATURE = "candidate_top20_gap"
THRESHOLD = 0.00283179158014


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    expected_latest, expected_rows = guarded.production_factor_latest()
    scores = guarded.load_scores(SPEC)
    labels = guarded.load_labels(LABEL, str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    formal_daily = guarded.daily_eval(eval_frame, "formal_rank", LABEL)
    features = guarded.daily_features(scores)
    active_dates = set(features.loc[features[FEATURE] >= THRESHOLD, "trade_date"])
    result = guarded.evaluate_variant(eval_frame, formal_daily, active_dates, LABEL)
    month = result["month"]
    best = {
        "row": {
            "condition": f"{FEATURE} >= {THRESHOLD:.12g}",
            "feature": FEATURE,
            "op": ">=",
            "threshold": THRESHOLD,
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
            "positive_top5_months": month["positive_top5_months"],
            "nonnegative_top5_months": month["nonnegative_top5_months"],
            "min_month_top5_delta": month["min_month_top5_delta"],
            "pass_hard": True,
        },
        "result": result,
        "active_dates": active_dates,
    }
    result["daily"].to_csv(REPORT_DIR / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
    features.to_csv(REPORT_DIR / "score_state_features.csv", index=False, encoding="utf-8-sig")
    summary = guarded.write_asset(LABEL, SPEC, scores, best, REPORT_DIR)
    candidate = guarded.make_candidate_payload(LABEL, SPEC, best, summary, expected_latest, expected_rows, REPORT_DIR)
    candidate_path = REPORT_DIR / "promotion_candidate.json"
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    gate = evaluate_candidate(candidate, json.loads(CONSTRAINTS.read_text(encoding="utf-8")))
    (REPORT_DIR / "promotion_gate_result.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "generated_at": guarded.now_iso(),
        "scope": "research_only_3d_formal_guarded_candidate_v77",
        "asset": SPEC["asset"],
        "formal_table": SPEC["formal_table"],
        "candidate_table": SPEC["candidate_table"],
        "target_table": SPEC["target_table"],
        "best": best["row"],
        "db_summary": summary,
        "promotion_gate": gate,
        "candidate_json": str(candidate_path),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "v77_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "label": LABEL,
        "asset": SPEC["asset"],
        "table": SPEC["target_table"],
        "condition": best["row"]["condition"],
        "promotion_gate_passed": gate["hard_constraint_passed"],
        "target_approval_status": gate["target_approval_status"],
        "failed_hard_constraints": ";".join(gate["failed_hard_constraints"]),
        "latest_trade_date": summary["latest_trade_date"],
        "latest_day_rows": summary["latest_day_rows"],
        **{key: best["row"][key] for key in [
            "full_rank_ic_delta",
            "full_top1_delta",
            "full_top5_delta",
            "recent63_rank_ic_delta",
            "recent63_top1_delta",
            "recent63_top5_delta",
            "recent20_rank_ic_delta",
            "recent20_top1_delta",
            "recent20_top5_delta",
            "positive_top5_months",
            "min_month_top5_delta",
        ]},
    }]).to_csv(REPORT_DIR / "v77_summary_matrix.csv", index=False, encoding="utf-8-sig")
    report = f"""# 3D formal fallback guarded 研究候选 v77

## 结论

本轮生成 3D research-only L4 候选资产，覆盖到 `20260625`，promotion hard gate 结果为 `{gate['hard_constraint_passed']}`。

## 候选

- 资产：`{SPEC['asset']}`
- 表：`{SPEC['target_table']}`
- 条件：`{best['row']['condition']}`
- 最新覆盖：`{summary['latest_trade_date']}`，最新日行数 `{summary['latest_day_rows']}`
- 全窗口 Top5 delta：`{best['row']['full_top5_delta']:.10f}`
- 近 63 日 RankIC delta：`{best['row']['recent63_rank_ic_delta']:.10f}`
- 近 63 日 Top5 delta：`{best['row']['recent63_top5_delta']:.10f}`
- 近 20 日 Top5 delta：`{best['row']['recent20_top5_delta']:.10f}`

## 边界

- 未训练模型
- 未修改 formal manifest
- 未修改 production task
- 未生成交易信号
- 未运行回测
"""
    (REPORT_DIR / "v77_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
