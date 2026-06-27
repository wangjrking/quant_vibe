from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


CN_TZ = timezone(timedelta(hours=8))


def _pick(values: list[Any], index: str) -> Any:
    if not values:
        return None
    if index == "first":
        return values[0]
    if index == "last":
        return values[-1]
    return values[len(values) // 2]


def _slug_float(value: Any) -> str:
    text = str(value)
    return text.replace(".", "p").replace("-", "m")


def _experiment_params(item: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    feature_top_n = item.get("suggested_feature_top_n") or []
    max_depth = item.get("suggested_max_depth") or []
    learning_rate = item.get("suggested_learning_rate") or []
    n_estimators = item.get("suggested_n_estimators") or []
    reg_lambda = item.get("suggested_reg_lambda") or []
    reg_alpha = item.get("suggested_reg_alpha") or []

    return [
        (
            "stability_first",
            {
                "feature_top_n": _pick(feature_top_n, "first"),
                "max_depth": _pick(max_depth, "first"),
                "learning_rate": _pick(learning_rate, "first"),
                "n_estimators": _pick(n_estimators, "first"),
                "reg_lambda": _pick(reg_lambda, "last"),
                "reg_alpha": _pick(reg_alpha, "last"),
            },
        ),
        (
            "objective_repair",
            {
                "feature_top_n": _pick(feature_top_n, "middle"),
                "max_depth": _pick(max_depth, "last"),
                "learning_rate": _pick(learning_rate, "last"),
                "n_estimators": _pick(n_estimators, "last"),
                "reg_lambda": _pick(reg_lambda, "middle"),
                "reg_alpha": _pick(reg_alpha, "first"),
            },
        ),
    ]


def _embargo_days(horizon: str) -> int:
    digits = "".join(ch for ch in str(horizon) if ch.isdigit())
    return int(digits or 1)


def _command(
    *,
    python_exe: str,
    parallel_folds: str,
    data_file_url: str,
    label: str,
    horizon: str,
    output_table: str,
    output_dir: str,
    experiment_name: str,
    params: dict[str, Any],
    max_workers: int,
) -> str:
    return " ".join(
        [
            python_exe,
            parallel_folds,
            "--data-file-url",
            data_file_url,
            "--data-start",
            "20100101",
            "--first-test",
            "20240604",
            "--final-test",
            "20260623",
            "--label",
            label,
            "--model-type",
            "reg",
            "--train-years",
            "5",
            "--test-months",
            "3",
            "--step-months",
            "3",
            "--embargo-days",
            str(_embargo_days(horizon)),
            "--fold-feature-top-n",
            str(params["feature_top_n"]),
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
            output_dir,
            "--experiment-name",
            experiment_name,
            "--max-workers",
            str(max_workers),
            "--xgb-n-estimators",
            str(params["n_estimators"]),
            "--xgb-learning-rate",
            str(params["learning_rate"]),
            "--xgb-max-depth",
            str(params["max_depth"]),
            "--xgb-reg-lambda",
            str(params["reg_lambda"]),
            "--xgb-reg-alpha",
            str(params["reg_alpha"]),
            "--xgb-subsample",
            "0.90",
            "--xgb-colsample-bytree",
            "0.80",
            "--xgb-device",
            "cuda",
            "--xgb-n-jobs",
            "0",
            "--skip-existing",
        ]
    )


def build_experiment_plan(
    *,
    runbook: dict[str, Any],
    status: dict[str, Any],
    run_id: str,
    python_exe: str = "D:/work/quant/quant_mcp/.venv/Scripts/python.exe",
    report_root: str = "D:/work/quant/quant_mcp/quant/data_file/reports",
) -> dict[str, Any]:
    standard_chain = runbook.get("standard_chain", {})
    entrypoints = runbook.get("entrypoints", {})
    parallel_folds = str(entrypoints.get("parallel_folds", "D:/work/quant/quant_mcp/quant/main/run_parallel_expanding2010_folds.py"))
    data_file_url = str(Path(str(standard_chain.get("prediction_db", ""))).parent.parent).replace("\\", "/")

    experiments: list[dict[str, Any]] = []
    skipped_validation_only: list[str] = []
    horizon_status = status.get("horizons", {})

    for item in runbook.get("v12_execution_order", []):
        horizon = str(item.get("horizon"))
        if item.get("requires_training_authorization") is not True:
            skipped_validation_only.append(horizon)
            continue
        for variant, params in _experiment_params(item):
            experiment_name = (
                f"{run_id}_{horizon}_{variant}_"
                f"fs{params['feature_top_n']}_d{params['max_depth']}_"
                f"lr{_slug_float(params['learning_rate'])}_n{params['n_estimators']}"
            )
            output_table = (
                f"stock_predict_data_model_agent_{run_id}_{horizon}_{variant}_"
                f"{item.get('label')}_research"
            )
            output_dir = f"{report_root}/model_agent_{run_id}_research_training/{experiment_name}"
            experiments.append(
                {
                    "horizon": horizon,
                    "label": item.get("label"),
                    "priority": item.get("priority"),
                    "variant": variant,
                    "failed_constraint": item.get("failed_constraint"),
                    "weighted_score": (horizon_status.get(horizon) or {}).get("weighted_score"),
                    "latest_mature_label_date": (horizon_status.get(horizon) or {}).get("latest_mature_label_date"),
                    "requires_training_authorization": True,
                    "params": params,
                    "output_table": output_table,
                    "output_dir": output_dir,
                    "experiment_name": experiment_name,
                    "command": _command(
                        python_exe=python_exe,
                        parallel_folds=parallel_folds,
                        data_file_url=data_file_url,
                        label=str(item.get("label")),
                        horizon=horizon,
                        output_table=output_table,
                        output_dir=output_dir,
                        experiment_name=experiment_name,
                        params=params,
                        max_workers=int(item.get("initial_max_workers") or 1),
                    ),
                }
            )

    return {
        "generated_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
        "actor": "model-agent",
        "run_id": run_id,
        "asset_role": "research_experiment_plan_not_executed",
        "approval_status": "research_plan_not_approved_for_execution",
        "experiments": experiments,
        "skipped_validation_only": skipped_validation_only,
        "boundaries": {
            "no_training_executed_by_plan_generation": True,
            "no_prediction_generated_by_plan_generation": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# V12 研究训练实验计划",
        "",
        "## 当前结论",
        "",
        "本计划只生成 research-only 训练实验命令，不执行训练、不生成预测、不修改生产入口。",
        "",
        f"- 实验数量：`{len(plan.get('experiments', []))}`",
        f"- 跳过 validation-only 标签：`{', '.join(plan.get('skipped_validation_only', [])) or '无'}`",
        "",
        "## 实验清单",
        "",
        "| 标签 | 变体 | 失败约束 | 特征数 | 深度 | 学习率 | 树数 | 输出表 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for exp in plan.get("experiments", []):
        params = exp["params"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{exp['horizon']}`",
                    f"`{exp['variant']}`",
                    f"`{exp.get('failed_constraint')}`",
                    f"`{params.get('feature_top_n')}`",
                    f"`{params.get('max_depth')}`",
                    f"`{params.get('learning_rate')}`",
                    f"`{params.get('n_estimators')}`",
                    f"`{exp.get('output_table')}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## 执行命令", ""])
    for exp in plan.get("experiments", []):
        lines.extend(
            [
                f"### {exp['horizon']} / {exp['variant']}",
                "",
                "```powershell",
                exp["command"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 边界",
            "",
            "上述命令只有在用户或主管明确批准 research-only 训练后才允许执行。输出必须保持 `_research` 表和 research-only manifest，不得切换 production manifest。",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build V12 research-only training experiment plan.")
    parser.add_argument("--runbook-json", required=True)
    parser.add_argument("--status-json", required=True)
    parser.add_argument("--run-id", default="v12_20260624")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    plan = build_experiment_plan(
        runbook=_load_json(args.runbook_json),
        status=_load_json(args.status_json),
        run_id=args.run_id,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(plan), encoding="utf-8")
    print(json.dumps({"ok": True, "output_json": str(out_json), "output_md": str(args.output_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
