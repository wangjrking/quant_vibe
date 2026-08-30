from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


CN_TZ = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_10d_new_axis_plan_20260629"

FRONTIER_V19 = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_research_frontier_status_v19_20260629"
    / "research_frontier_status_v19.json"
)

DEFAULT_XGB_EARLY_STOPPING_ROUNDS = 300


EXPERIMENTS = [
    {
        "variant": "fixed4y_fs40_d3_l8_alpha01_topn_balance",
        "feature_top_n": 40,
        "max_depth": 3,
        "learning_rate": 0.004,
        "n_estimators": 5000,
        "reg_lambda": 8,
        "reg_alpha": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "purpose": "复核 fs40 小特征集在 10D 四年窗口里的 TopN 稳定性。",
    },
    {
        "variant": "fixed4y_fs80_d2_l12_alpha05_stability",
        "feature_top_n": 80,
        "max_depth": 2,
        "learning_rate": 0.003,
        "n_estimators": 7000,
        "reg_lambda": 12,
        "reg_alpha": 0.5,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "purpose": "用浅树和更强正则压缩 10D 月度负尾。",
    },
    {
        "variant": "fixed4y_fs120_d3_l6_alpha02_rankic_topn",
        "feature_top_n": 120,
        "max_depth": 3,
        "learning_rate": 0.003,
        "n_estimators": 6000,
        "reg_lambda": 6,
        "reg_alpha": 0.2,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "purpose": "保留较宽特征集，验证 RankIC 与 Top5 是否能同时超过当前 bestset。",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug_float(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _experiment_name(run_id: str, item: dict[str, Any]) -> str:
    return (
        f"{run_id}_{item['variant']}_"
        f"fs{item['feature_top_n']}_d{item['max_depth']}_"
        f"lr{_slug_float(item['learning_rate'])}_n{item['n_estimators']}"
    )


def _command(run_id: str, item: dict[str, Any]) -> str:
    python_exe = ROOT / ".venv" / "Scripts" / "python.exe"
    runner = MAIN / "run_parallel_expanding2010_folds.py"
    experiment_name = _experiment_name(run_id, item)
    output_table = (
        f"stock_predict_data_model_agent_{run_id}_{item['variant']}_"
        "executable_10d_open_return_research"
    )
    output_dir = REPORT_DIR / "research_training" / experiment_name
    parts = [
        str(python_exe),
        str(runner),
        "--data-file-url",
        str(DATA_DIR),
        "--data-start",
        "20100101",
        "--first-test",
        "20220606",
        "--final-test",
        "20260626",
        "--label",
        "executable_10d_open_return",
        "--model-type",
        "reg",
        "--train-mode",
        "fixed",
        "--train-years",
        "4",
        "--test-months",
        "3",
        "--step-months",
        "3",
        "--embargo-days",
        "10",
        "--fold-feature-top-n",
        str(item["feature_top_n"]),
        "--fold-feature-min-abs-ic",
        "0",
        "--fold-feature-max-missing-ratio",
        "0.30",
        "--fold-feature-folds",
        "8",
        "--feature-source",
        "production_split",
        "--prediction-output-mode",
        "independent",
        "--output-table",
        output_table,
        "--output-dir",
        str(output_dir),
        "--experiment-name",
        experiment_name,
        "--max-workers",
        "1",
        "--xgb-n-estimators",
        str(item["n_estimators"]),
        "--xgb-learning-rate",
        str(item["learning_rate"]),
        "--xgb-max-depth",
        str(item["max_depth"]),
        "--xgb-reg-lambda",
        str(item["reg_lambda"]),
        "--xgb-reg-alpha",
        str(item["reg_alpha"]),
        "--xgb-subsample",
        str(item["subsample"]),
        "--xgb-colsample-bytree",
        str(item["colsample_bytree"]),
        "--xgb-early-stopping-rounds",
        str(DEFAULT_XGB_EARLY_STOPPING_ROUNDS),
        "--xgb-device",
        "cuda",
        "--xgb-n-jobs",
        "0",
        "--skip-existing",
    ]
    return " ".join(parts)


def build_plan(frontier: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    current_10d = frontier["current_bestset"]["10d"]
    experiments: list[dict[str, Any]] = []
    for item in EXPERIMENTS:
        experiment_name = _experiment_name(run_id, item)
        experiments.append(
            {
                "variant": item["variant"],
                "purpose": item["purpose"],
                "label": "executable_10d_open_return",
                "experiment_name": experiment_name,
                "output_table": (
                    f"stock_predict_data_model_agent_{run_id}_{item['variant']}_"
                    "executable_10d_open_return_research"
                ),
                "params": {
                    "train_mode": "fixed",
                    "train_years": 4,
                    "test_months": 3,
                    "step_months": 3,
                    "embargo_days": 10,
                    "feature_top_n": item["feature_top_n"],
                    "max_depth": item["max_depth"],
                    "learning_rate": item["learning_rate"],
                    "n_estimators": item["n_estimators"],
                    "reg_lambda": item["reg_lambda"],
                    "reg_alpha": item["reg_alpha"],
                    "subsample": item["subsample"],
                    "colsample_bytree": item["colsample_bytree"],
                    "early_stopping_rounds": DEFAULT_XGB_EARLY_STOPPING_ROUNDS,
                },
                "command": _command(run_id, item),
            }
        )

    hard_gate = {
        "must_improve_against_current_bestset": True,
        "full_rank_ic_delta_vs_current_bestset": "> 0",
        "full_top5_delta_vs_current_bestset": "> 0",
        "recent63_top5_delta_vs_current_bestset": ">= 0",
        "recent20_top5_delta_vs_current_bestset": ">= 0",
        "month_min_top5_delta_vs_current_bestset": ">= 0",
        "table_quality": {
            "null_pred_prob": 0,
            "duplicate_key_groups": 0,
        },
    }

    return {
        "generated_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
        "scope": "research_only_four_year_10d_new_model_or_feature_axis_plan",
        "run_id": run_id,
        "current_10d_bestset": {
            "asset": current_10d["asset"],
            "table": current_10d["table"],
            "eval_min_trade_date": current_10d["eval_min_trade_date"],
            "eval_max_trade_date": current_10d["eval_max_trade_date"],
            "eval_trade_days": current_10d["eval_trade_days"],
            "full_rank_ic": current_10d["full_rank_ic"],
            "full_top5": current_10d["full_top5"],
            "recent63_abs_top5": current_10d["recent63_abs_top5"],
            "recent20_abs_top5": current_10d["recent20_abs_top5"],
            "month_top5_negative": current_10d["month_top5_negative"],
            "month_min_top5_delta": current_10d["month_min_top5_delta"],
        },
        "why_new_axis": {
            "frontier_decision": frontier["next_focus"],
            "failed_score_reuse_evidence": frontier["recent_progress"],
        },
        "experiments": experiments,
        "hard_gate_for_entering_production_candidate_discussion": hard_gate,
        "required_artifacts_after_execution": {
            "fold_model_files": True,
            "fold_selected_features": True,
            "fold_train_test_window_meta": True,
            "fold_results_csv": True,
            "merged_research_prediction_table": True,
            "four_year_eval_summary": True,
            "monthly_and_annual_stability": True,
        },
        "boundaries": {
            "research_only": True,
            "no_training_executed_by_plan": True,
            "no_prediction_generated_by_plan": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
            "production_release_requires_user_authorization": True,
        },
    }


def render_markdown(plan: dict[str, Any]) -> str:
    current = plan["current_10d_bestset"]
    lines = [
        "# 10D 四年新模型/特征轴研究计划",
        "",
        "## 当前结论",
        "",
        "现有 10D score-reuse / gate / blend 路线已连续未通过四年硬门槛，下一步转向新的 fixed4y 模型参数和特征数量实验。",
        "",
        "本文件只定义 research-only 实验计划，不训练、不生成预测、不修改 formal manifest、不切生产。",
        "",
        "## 当前 10D bestset",
        "",
        f"- 资产：`{current['asset']}`",
        f"- 表：`{current['table']}`",
        f"- 评价窗口：`{current['eval_min_trade_date']}` 到 `{current['eval_max_trade_date']}`，`{current['eval_trade_days']}` 个交易日",
        f"- Full RankIC：`{current['full_rank_ic']:.6f}`",
        f"- Full Top5：`{current['full_top5']:.6f}`",
        f"- Recent63 Top5：`{current['recent63_abs_top5']:.6f}`",
        f"- Recent20 Top5：`{current['recent20_abs_top5']:.6f}`",
        f"- 月度 Top5 负月数：`{current['month_top5_negative']}`",
        "",
        "## 实验清单",
        "",
        "| 变体 | 特征数 | 深度 | 学习率 | 树数 | 正则 | 目的 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for exp in plan["experiments"]:
        params = exp["params"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{exp['variant']}`",
                    f"`{params['feature_top_n']}`",
                    f"`{params['max_depth']}`",
                    f"`{params['learning_rate']}`",
                    f"`{params['n_estimators']}`",
                    f"`lambda={params['reg_lambda']}, alpha={params['reg_alpha']}`",
                    exp["purpose"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## 执行命令", ""])
    for exp in plan["experiments"]:
        lines.extend(
            [
                f"### {exp['variant']}",
                "",
                "```powershell",
                exp["command"],
                "```",
                "",
            ]
        )
    gate = plan["hard_gate_for_entering_production_candidate_discussion"]
    lines.extend(
        [
            "## 进入生产候选讨论硬门槛",
            "",
            f"- Full RankIC 相对当前 bestset：`{gate['full_rank_ic_delta_vs_current_bestset']}`",
            f"- Full Top5 相对当前 bestset：`{gate['full_top5_delta_vs_current_bestset']}`",
            f"- Recent63 Top5 相对当前 bestset：`{gate['recent63_top5_delta_vs_current_bestset']}`",
            f"- Recent20 Top5 相对当前 bestset：`{gate['recent20_top5_delta_vs_current_bestset']}`",
            f"- 月度最差 Top5 delta：`{gate['month_min_top5_delta_vs_current_bestset']}`",
            "- 表质量：`pred_prob` 空值为 0，`trade_date + stock_code` 无重复键。",
            "",
            "## 边界",
            "",
            "- 本计划不执行训练。",
            "- 本计划不生成预测表。",
            "- 本计划不修改 formal manifest 或 production manifest。",
            "- 后续即使实验通过，发布生产仍必须先给出建议并获得主人明确授权。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-only 10D new-axis experiment plan.")
    parser.add_argument("--frontier-json", default=str(FRONTIER_V19))
    parser.add_argument("--run-id", default="four_year_10d_newaxis_v1_20260629")
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(_load_json(Path(args.frontier_json)), run_id=args.run_id)
    json_path = output_dir / "10d_new_axis_research_plan.json"
    md_path = output_dir / "10d_new_axis_research_plan.md"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(plan), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "experiments": len(plan["experiments"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
