from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import research_3d_formal_guarded_two_condition_v81_20260627 as two_gate
import research_5d10d_formal_guarded_candidate_v74_20260627 as guarded
from score_model_promotion_candidate import evaluate_candidate


DATA_DIR = guarded.DATA_DIR
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d5d10d_formal_guarded_two_condition_v83_20260627"
CONSTRAINTS = guarded.CONSTRAINTS

SPECS = {
    "executable_1d_open_return": {
        "asset": "research_1d_formal_guarded_two_condition_v83_20260627",
        "formal_table": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
        "candidate_table": "stock_predict_data_model_agent_1d_exante_iqr_gate_v67_20260627_executable_1d_open_return_research",
        "target_table": "stock_predict_data_model_agent_1d_formal_guarded_two_condition_v83_20260627_executable_1d_open_return_research",
        "rank_ic_floor": 0.0,
    },
    "executable_5d_open_return": {
        "asset": "research_5d_formal_guarded_two_condition_v83_20260627",
        "formal_table": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
        "candidate_table": "stock_predict_data_model_agent_5d_recent_top_condblend_latest_20260626_executable_5d_open_return_research",
        "target_table": "stock_predict_data_model_agent_5d_formal_guarded_two_condition_v83_20260627_executable_5d_open_return_research",
        "rank_ic_floor": -0.001,
    },
    "executable_10d_open_return": {
        "asset": "research_10d_formal_guarded_two_condition_v83_20260627",
        "formal_table": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
        "candidate_table": "stock_predict_data_model_agent_10d_monthly_top5_double_veto_v62_20260627_executable_10d_open_return_research",
        "target_table": "stock_predict_data_model_agent_10d_formal_guarded_two_condition_v83_20260627_executable_10d_open_return_research",
        "rank_ic_floor": -0.0005,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build research-only two-condition formal-guarded model candidates.")
    parser.add_argument("--labels", default=",".join(SPECS), help="Comma separated labels to run.")
    return parser.parse_args()


def run_label(label: str, spec: dict[str, object], config: dict[str, object], expected_latest: str, expected_rows: int) -> dict[str, object]:
    output_dir = REPORT_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = guarded.load_scores(spec)
    labels = guarded.load_labels(label, str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")

    formal_daily = guarded.daily_eval(eval_frame, "formal_rank", label)
    candidate_daily = guarded.daily_eval(eval_frame, "candidate_rank", label)
    pair = formal_daily.merge(
        candidate_daily,
        on="trade_date",
        suffixes=("_formal", "_candidate"),
        validate="one_to_one",
    )
    features = guarded.daily_features(scores)

    original_spec = two_gate.SPEC
    try:
        two_gate.SPEC = spec
        scan_frame, best = two_gate.scan_variants(features, pair, formal_daily)
    finally:
        two_gate.SPEC = original_spec

    scan_frame.to_csv(output_dir / "scan_results.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output_dir / "score_state_features.csv", index=False, encoding="utf-8-sig")
    best["result"]["daily"].to_csv(output_dir / "best_daily_eval.csv", index=False, encoding="utf-8-sig")

    summary = guarded.write_asset(label, spec, scores, best, output_dir)
    candidate = guarded.make_candidate_payload(label, spec, best, summary, expected_latest, expected_rows, output_dir)
    candidate["prefer_simpler_formula"] = "formal fallback + two score-state gates"
    candidate["evidence"]["scan_summary"] = str(output_dir / f"{label}_v83_summary.json")
    candidate_path = output_dir / "promotion_candidate.json"
    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")

    gate = evaluate_candidate(candidate, config)
    (output_dir / "promotion_gate_result.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    result_payload = {
        "asset": spec["asset"],
        "formal_table": spec["formal_table"],
        "candidate_table": spec["candidate_table"],
        "target_table": spec["target_table"],
        "best": best["row"],
        "db_summary": summary,
        "promotion_gate": gate,
        "candidate_json": str(candidate_path),
    }
    (output_dir / f"{label}_v83_summary.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# {label} 双条件门控研究候选 v83

## 结论

本轮只生成 research-only L4 候选资产，没有训练模型，没有修改 formal/production manifest，没有生成信号或回测结论。

## 候选摘要

- 资产：`{spec['asset']}`
- 表：`{spec['target_table']}`
- 条件：`{best['row']['condition']}`
- 类型：`{best['row']['kind']}`
- 激活交易日：`{best['row']['active_days']}`
- 最新覆盖：`{summary['latest_trade_date']}`，最新日行数 `{summary['latest_day_rows']}`
- promotion gate：`{gate['hard_constraint_passed']}`，目标状态 `{gate['target_approval_status']}`

## 相对当前 formal baseline 的增量

- 全窗口 RankIC delta：`{best['row']['full_rank_ic_delta']:.10f}`
- 全窗口 Top1 delta：`{best['row']['full_top1_delta']:.10f}`
- 全窗口 Top5 delta：`{best['row']['full_top5_delta']:.10f}`
- 近 63 日 Top1 delta：`{best['row']['recent63_top1_delta']:.10f}`
- 近 63 日 Top5 delta：`{best['row']['recent63_top5_delta']:.10f}`
- 近 20 日 Top1 delta：`{best['row']['recent20_top1_delta']:.10f}`
- 近 20 日 Top5 delta：`{best['row']['recent20_top5_delta']:.10f}`
"""
    (output_dir / f"{label}_v83_report.md").write_text(report, encoding="utf-8")
    return result_payload


def main() -> int:
    args = parse_args()
    selected_labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    unknown = [label for label in selected_labels if label not in SPECS]
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    expected_latest, expected_rows = guarded.production_factor_latest()
    config = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    payload = {
        "generated_at": guarded.now_iso(),
        "scope": "research_only_1d5d10d_formal_guarded_two_condition_v83",
        "expected_latest_trade_date": expected_latest,
        "expected_latest_day_rows": expected_rows,
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
    for label in selected_labels:
        result = run_label(label, SPECS[label], config, expected_latest, expected_rows)
        payload["results"][label] = result
        best = result["best"]
        summary = result["db_summary"]
        gate = result["promotion_gate"]
        rows.append(
            {
                "label": label,
                "asset": result["asset"],
                "table": result["target_table"],
                "condition": best["condition"],
                "kind": best["kind"],
                "promotion_gate_passed": gate["hard_constraint_passed"],
                "target_approval_status": gate["target_approval_status"],
                "failed_hard_constraints": ";".join(gate["failed_hard_constraints"]),
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
    matrix.to_csv(REPORT_DIR / "v83_summary_matrix.csv", index=False, encoding="utf-8-sig")
    payload["summary_matrix"] = str(REPORT_DIR / "v83_summary_matrix.csv")
    (REPORT_DIR / "v83_run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 1D/5D/10D 双条件门控研究候选 v83",
        "",
        "## 结论",
        "",
        "本轮只生成 research-only L4 候选资产；未训练模型，未修改 formal/production manifest，未生成信号，未运行回测。",
        "",
        "## 摘要",
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
                f"- gate：`{row['promotion_gate_passed']}`，目标状态 `{row['target_approval_status']}`",
                f"- 全窗口 Top5 delta：`{row['full_top5_delta']:.10f}`",
                f"- 近 63 日 Top5 delta：`{row['recent63_top5_delta']:.10f}`",
                f"- 近 20 日 Top5 delta：`{row['recent20_top5_delta']:.10f}`",
                "",
            ]
        )
    (REPORT_DIR / "v83_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
