from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


CN_TZ = timezone(timedelta(hours=8))


def _primary_gap(gap_item: dict[str, Any]) -> tuple[int, float]:
    gap = gap_item.get("gap")
    if gap is None:
        return (1, 999.0)
    return (0, float(gap))


def _acceptance_metrics(gaps: list[dict[str, Any]]) -> list[str]:
    metrics: list[str] = []
    for item in gaps:
        key = item.get("metric_key")
        if key and key not in metrics:
            metrics.append(str(key))
    return metrics


def build_acceptance_matrix(*, plan: dict[str, Any], gap_report: dict[str, Any]) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    horizons = gap_report.get("horizons", {})

    for exp in plan.get("experiments", []):
        horizon = str(exp.get("horizon"))
        horizon_gap = horizons.get(horizon, {})
        gaps = list(horizon_gap.get("constraint_gaps", []))
        failed_gaps = [item for item in gaps if item.get("passed") is False]
        failed_gaps.sort(key=_primary_gap)
        primary = failed_gaps[0] if failed_gaps else (gaps[0] if gaps else {})
        experiments.append(
            {
                "horizon": horizon,
                "variant": exp.get("variant"),
                "output_table": exp.get("output_table"),
                "gate_status": horizon_gap.get("gate_status"),
                "primary_failed_constraint": primary.get("constraint"),
                "primary_metric_key": primary.get("metric_key"),
                "primary_actual": primary.get("actual"),
                "primary_floor": primary.get("floor"),
                "primary_gap": primary.get("gap"),
                "acceptance_metrics": _acceptance_metrics(gaps),
                "acceptance_rule": "candidate must satisfy all V12 objective constraints and keep research-only asset governance",
            }
        )

    return {
        "generated_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
        "actor": "model-agent",
        "asset_role": "research_experiment_acceptance_matrix_not_formal",
        "approval_status": "research_report_only",
        "summary": {
            "experiments": len(experiments),
            "skipped_validation_only": plan.get("skipped_validation_only", []),
        },
        "experiments": experiments,
        "boundaries": {
            "no_training": True,
            "no_prediction": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def render_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# V12 实验验收矩阵",
        "",
        "## 当前结论",
        "",
        f"- 训练实验数：`{matrix['summary'].get('experiments')}`",
        f"- 跳过 validation-only 标签：`{', '.join(matrix['summary'].get('skipped_validation_only') or []) or '无'}`",
        "",
        "## 验收矩阵",
        "",
        "| 标签 | 变体 | 主缺口约束 | 指标 | 实际值 | 门槛 | 缺口 | 验收指标 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in matrix.get("experiments", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.get('horizon')}`",
                    f"`{item.get('variant')}`",
                    f"`{item.get('primary_failed_constraint')}`",
                    f"`{item.get('primary_metric_key')}`",
                    f"`{item.get('primary_actual')}`",
                    f"`{item.get('primary_floor')}`",
                    f"`{item.get('primary_gap')}`",
                    "`" + ", ".join(item.get("acceptance_metrics") or []) + "`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本矩阵只定义研究实验完成后的模型侧验收指标，未训练模型、未生成预测、未修改 production manifest、未生成交易信号、未运行回测。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(matrix: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "horizon",
                "variant",
                "output_table",
                "gate_status",
                "primary_failed_constraint",
                "primary_metric_key",
                "primary_actual",
                "primary_floor",
                "primary_gap",
                "acceptance_metrics",
                "acceptance_rule",
            ],
        )
        writer.writeheader()
        for item in matrix.get("experiments", []):
            row = dict(item)
            row["acceptance_metrics"] = ",".join(item.get("acceptance_metrics") or [])
            writer.writerow(row)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build V12 experiment acceptance matrix.")
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--gap-report-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    matrix = build_acceptance_matrix(
        plan=_load_json(args.plan_json),
        gap_report=_load_json(args.gap_report_json),
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(matrix), encoding="utf-8")
    write_csv(matrix, args.output_csv)
    print(json.dumps({"ok": True, "output_json": str(out_json), "output_md": args.output_md, "output_csv": args.output_csv}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
