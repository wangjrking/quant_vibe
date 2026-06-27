from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import research_5d10d_formal_guarded_candidate_v74_20260627 as guarded
from score_model_promotion_candidate import evaluate_candidate


DATA_DIR = guarded.DATA_DIR
MAIN_DIR = guarded.MAIN_DIR
DB_PATH = guarded.DB_PATH
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d3d_formal_guarded_v76_20260627"
CONSTRAINTS = guarded.CONSTRAINTS


SPECS = {
    "executable_1d_open_return": {
        "asset": "research_1d_formal_guarded_exante_v76_20260627",
        "formal_table": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
        "candidate_table": "stock_predict_data_model_agent_1d_exante_iqr_gate_v67_20260627_executable_1d_open_return_research",
        "target_table": "stock_predict_data_model_agent_1d_formal_guarded_exante_v76_20260627_executable_1d_open_return_research",
        "rank_ic_floor": 0.0,
    },
    "executable_3d_open_return": {
        "asset": "research_3d_formal_guarded_exante_v76_20260627",
        "formal_table": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
        "candidate_table": "stock_predict_data_model_agent_3d_exante_std_gate_v69_20260627_executable_3d_open_return_research",
        "target_table": "stock_predict_data_model_agent_3d_formal_guarded_exante_v76_20260627_executable_3d_open_return_research",
        "rank_ic_floor": -0.0015,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build research-only formal-guarded 1D/3D model candidates.")
    parser.add_argument("--labels", default=",".join(SPECS), help="Comma separated labels to run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    unknown = [label for label in selected_labels if label not in SPECS]
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    expected_latest, expected_rows = guarded.production_factor_latest()
    config = json.loads(CONSTRAINTS.read_text(encoding="utf-8"))
    all_rows = []
    payload = {
        "generated_at": guarded.now_iso(),
        "scope": "research_only_1d3d_formal_guarded_candidate_v76",
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
    for label in selected_labels:
        spec = SPECS[label]
        output_dir = REPORT_DIR / label
        output_dir.mkdir(parents=True, exist_ok=True)
        scores = guarded.load_scores(spec)
        labels = guarded.load_labels(label, str(scores["trade_date"].min()), str(scores["trade_date"].max()))
        scan_frame, best, features = guarded.scan(label, scores, labels, float(spec["rank_ic_floor"]))
        scan_frame.to_csv(output_dir / "scan_results.csv", index=False, encoding="utf-8-sig")
        features.to_csv(output_dir / "score_state_features.csv", index=False, encoding="utf-8-sig")
        best["result"]["daily"].to_csv(output_dir / "best_daily_eval.csv", index=False, encoding="utf-8-sig")
        summary = guarded.write_asset(label, spec, scores, best, output_dir)
        candidate = guarded.make_candidate_payload(label, spec, best, summary, expected_latest, expected_rows, output_dir)
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
        (output_dir / f"{label}_v76_summary.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["results"][label] = result_payload
        all_rows.append(
            {
                "label": label,
                "asset": spec["asset"],
                "table": spec["target_table"],
                "condition": best["row"]["condition"],
                "pass_scan_hard": best["row"]["pass_hard"],
                "promotion_gate_passed": gate["hard_constraint_passed"],
                "target_approval_status": gate["target_approval_status"],
                "failed_hard_constraints": ";".join(gate["failed_hard_constraints"]),
                "latest_trade_date": summary["latest_trade_date"],
                "latest_day_rows": summary["latest_day_rows"],
                "full_rank_ic_delta": best["row"]["full_rank_ic_delta"],
                "full_top1_delta": best["row"]["full_top1_delta"],
                "full_top5_delta": best["row"]["full_top5_delta"],
                "recent63_rank_ic_delta": best["row"]["recent63_rank_ic_delta"],
                "recent63_top1_delta": best["row"]["recent63_top1_delta"],
                "recent63_top5_delta": best["row"]["recent63_top5_delta"],
                "recent20_rank_ic_delta": best["row"]["recent20_rank_ic_delta"],
                "recent20_top1_delta": best["row"]["recent20_top1_delta"],
                "recent20_top5_delta": best["row"]["recent20_top5_delta"],
                "positive_top5_months": best["row"]["positive_top5_months"],
                "min_month_top5_delta": best["row"]["min_month_top5_delta"],
            }
        )

    matrix = pd.DataFrame(all_rows)
    matrix.to_csv(REPORT_DIR / "v76_summary_matrix.csv", index=False, encoding="utf-8-sig")
    payload["summary_matrix"] = str(REPORT_DIR / "v76_summary_matrix.csv")
    (REPORT_DIR / "v76_run_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 1D/3D formal fallback guarded 研究候选 v76",
        "",
        "## 结论",
        "",
        "本轮只生成 research-only L4 候选资产；未训练模型，未修改 formal manifest，未修改 production task，未生成信号，未运行回测。",
        "",
        "## 摘要",
        "",
    ]
    for row in all_rows:
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- 资产：`{row['asset']}`",
                f"- 表：`{row['table']}`",
                f"- 条件：`{row['condition']}`",
                f"- 最新覆盖：`{row['latest_trade_date']}`，最新日行数 `{row['latest_day_rows']}`",
                f"- promotion gate：`{row['promotion_gate_passed']}`，失败项：`{row['failed_hard_constraints'] or '无'}`",
                f"- 全窗口 Top5 delta：`{row['full_top5_delta']:.10f}`",
                f"- 近 63 日 Top5 delta：`{row['recent63_top5_delta']:.10f}`",
                f"- 近 20 日 Top5 delta：`{row['recent20_top5_delta']:.10f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 证据",
            "",
            "- `v76_run_summary.json`",
            "- `v76_summary_matrix.csv`",
            "- 每个标签目录下的 `promotion_candidate.json`、`promotion_gate_result.json`、`scan_results.csv`、`best_daily_eval.csv`。",
        ]
    )
    (REPORT_DIR / "v76_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
