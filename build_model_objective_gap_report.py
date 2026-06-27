from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


CN_TZ = timezone(timedelta(hours=8))

CONSTRAINT_METRIC_MAP = {
    "full_daily_rank_ic_delta_floor": "full.daily_rank_ic_mean_delta",
    "recent63_daily_rank_ic_delta_floor": "recent63.daily_rank_ic_mean_delta",
    "recent20_daily_rank_ic_delta_floor": "recent20.daily_rank_ic_mean_delta",
    "full_top5_delta_floor": "full.top5_delta",
    "full_top10_delta_floor": "full.top10_delta",
    "recent63_top5_delta_floor": "recent63.top5_delta",
    "recent63_top10_delta_floor": "recent63.top10_delta",
    "recent20_top5_delta_floor": "recent20.top5_delta",
    "recent20_top10_delta_floor": "recent20.top10_delta",
}


def _term_values(result: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in result.get("weighted_terms", []):
        key = f"{item.get('window')}.{item.get('metric')}"
        try:
            values[key] = float(item.get("value"))
        except (TypeError, ValueError):
            continue
    return values


def _constraint_gap(constraint: str, floor: Any, values: dict[str, float]) -> dict[str, Any]:
    metric_key = CONSTRAINT_METRIC_MAP.get(str(constraint))
    if metric_key is None:
        return {
            "constraint": constraint,
            "metric_key": None,
            "actual": None,
            "floor": floor,
            "gap": None,
            "passed": False,
            "note": "metric mapping missing",
        }
    actual = values.get(metric_key)
    if actual is None:
        return {
            "constraint": constraint,
            "metric_key": metric_key,
            "actual": None,
            "floor": floor,
            "gap": None,
            "passed": False,
            "note": "metric value missing",
        }
    floor_value = float(floor)
    gap = float(actual) - floor_value
    return {
        "constraint": constraint,
        "metric_key": metric_key,
        "actual": float(actual),
        "floor": floor_value,
        "gap": gap,
        "passed": gap >= 0,
        "note": "",
    }


def build_gap_report(*, objective_score: dict[str, Any], objective_config: dict[str, Any]) -> dict[str, Any]:
    config_by_horizon = objective_config.get("horizon_objectives", {})
    horizons: dict[str, Any] = {}
    failed_horizons: list[str] = []
    worst_gaps: list[dict[str, Any]] = []

    for result in objective_score.get("results", []):
        horizon = str(result.get("horizon"))
        config = config_by_horizon.get(horizon, {})
        constraints = config.get("additional_constraints", {})
        values = _term_values(result)
        constraint_gaps = [
            _constraint_gap(name, floor, values)
            for name, floor in constraints.items()
            if isinstance(floor, (int, float))
        ]
        constraint_gaps.sort(key=lambda item: (item["gap"] is None, item["gap"] if item["gap"] is not None else 999))
        if result.get("gate_status") == "failed":
            failed_horizons.append(horizon)
        for gap in constraint_gaps:
            if gap["gap"] is not None and gap["gap"] < 0:
                worst_gaps.append({"horizon": horizon, **gap})

        horizons[horizon] = {
            "label": result.get("label"),
            "priority": result.get("priority") or config.get("priority"),
            "gate_status": result.get("gate_status"),
            "weighted_score": result.get("weighted_score"),
            "failed_constraints": result.get("failed_constraints", []),
            "passed_constraints": result.get("passed_constraints", []),
            "optimization_style": result.get("optimization_style") or config.get("optimization_style"),
            "constraint_gaps": constraint_gaps,
        }

    worst_gaps.sort(key=lambda item: item["gap"])
    return {
        "generated_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
        "actor": "model-agent",
        "asset_role": "research_objective_gap_report_not_formal",
        "approval_status": "research_report_only",
        "summary": {
            "failed_horizons": failed_horizons,
            "worst_negative_gaps": worst_gaps[:10],
        },
        "horizons": horizons,
        "boundaries": {
            "no_training": True,
            "no_prediction": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V12 模型目标缺口报告",
        "",
        "## 当前结论",
        "",
        f"- 未通过标签：`{', '.join(report['summary'].get('failed_horizons', [])) or '无'}`",
        "",
        "## 负向缺口排序",
        "",
        "| 标签 | 约束 | 指标 | 实际值 | 门槛 | 缺口 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in report["summary"].get("worst_negative_gaps", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['horizon']}`",
                    f"`{item['constraint']}`",
                    f"`{item['metric_key']}`",
                    f"{item['actual']:.8f}",
                    f"{item['floor']:.8f}",
                    f"{item['gap']:.8f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## 分标签约束", ""])
    for horizon, item in report["horizons"].items():
        lines.extend([f"### {horizon}", "", f"- gate：`{item.get('gate_status')}`", f"- weighted_score：`{item.get('weighted_score')}`", ""])
        for gap in item.get("constraint_gaps", []):
            lines.append(
                f"- `{gap['constraint']}`：actual=`{gap['actual']}`，floor=`{gap['floor']}`，gap=`{gap['gap']}`，passed=`{gap['passed']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## 边界",
            "",
            "本报告只量化模型目标缺口，未训练模型、未生成预测、未修改 production manifest、未生成交易信号、未运行回测。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_gap_csv(report: dict[str, Any], path: str | Path) -> None:
    rows = []
    for horizon, item in report["horizons"].items():
        for gap in item.get("constraint_gaps", []):
            rows.append({"horizon": horizon, **gap})
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["horizon", "constraint", "metric_key", "actual", "floor", "gap", "passed", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build model objective constraint gap report.")
    parser.add_argument("--objective-score-json", required=True)
    parser.add_argument("--objective-config-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_gap_report(
        objective_score=_load_json(args.objective_score_json),
        objective_config=_load_json(args.objective_config_json),
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    write_gap_csv(report, args.output_csv)
    print(json.dumps({"ok": True, "output_json": str(out_json), "output_md": args.output_md, "output_csv": args.output_csv}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
