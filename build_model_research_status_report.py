from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


CN_TZ = timezone(timedelta(hours=8))


def _index_by(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(item.get(key)): item for item in items}


def _coverage_by_horizon(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _index_by(snapshot.get("coverage", []), "horizon")


def _next_step(item: dict[str, Any], acceptance_status: str | None) -> str:
    horizon = str(item.get("horizon"))
    if item.get("requires_training_authorization") is True:
        return "需要 research-only 训练授权后重训/特征搜索"
    if horizon == "10d" and acceptance_status == "formula_score_asset_ready_for_model_side_audit":
        return "可交审计做 research-only 公式型评分资产复核"
    if item.get("status") == "passed_objective_gate":
        return "需要补齐候选验收或审计复核"
    return "需要继续模型侧研究"


def build_status_report(
    *,
    queue: dict[str, Any],
    snapshot: dict[str, Any],
    readiness: dict[str, Any],
    formula_validations: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    formula_validations = formula_validations or {}
    formal_baseline = snapshot.get("formal_baseline", {})
    coverage = _coverage_by_horizon(snapshot)
    mature_labels = readiness.get("label_asset", {}).get("mature_label_dates", {})
    readiness_decision = readiness.get("readiness_decision", {})
    training_order = list(readiness_decision.get("recommended_order") or readiness_decision.get("training_authorization_required_for") or [])

    horizons: dict[str, Any] = {}
    for item in sorted(queue.get("queue", []), key=lambda row: int(row.get("queue_rank") or 999)):
        horizon = str(item.get("horizon"))
        label = str(item.get("label"))
        formula_validation = formula_validations.get(horizon)
        acceptance_status = None
        if formula_validation:
            acceptance_status = str(formula_validation.get("status") if formula_validation.get("ok") else "not_ready")

        horizons[horizon] = {
            "label": label,
            "priority": item.get("priority"),
            "queue_rank": item.get("queue_rank"),
            "objective_status": item.get("status"),
            "weighted_score": item.get("weighted_score"),
            "failed_constraints": item.get("failed_constraints") or "",
            "next_action_type": item.get("next_action_type"),
            "requires_training_authorization": bool(item.get("requires_training_authorization")),
            "formal_baseline_table": formal_baseline.get(horizon),
            "current_best_research_table": item.get("current_best_research_table"),
            "coverage": coverage.get(horizon, {}),
            "latest_mature_label_date": (mature_labels.get(label) or {}).get("latest_non_null_date"),
            "acceptance_status": acceptance_status or "not_validated",
            "next_step": _next_step(item, acceptance_status),
        }

    return {
        "generated_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
        "actor": "model-agent",
        "asset_role": "research_status_report_not_formal",
        "approval_status": "research_report_only",
        "summary": {
            "feature_latest_trade_date": readiness.get("factor_asset", {}).get("latest_trade_date"),
            "feature_latest_day_rows": readiness.get("factor_asset", {}).get("latest_day_rows"),
            "training_authorization_required_for": readiness_decision.get("training_authorization_required_for", []),
            "validation_only_for": readiness_decision.get("validation_only_for", []),
            "next_training_order": training_order,
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
        "# 模型研究状态汇总",
        "",
        "## 当前结论",
        "",
        f"- 生产因子最新日期：`{report['summary'].get('feature_latest_trade_date')}`，最新日行数：`{report['summary'].get('feature_latest_day_rows')}`",
        f"- 需要训练授权的标签：`{', '.join(report['summary'].get('training_authorization_required_for') or []) or '无'}`",
        f"- 建议训练顺序：`{', '.join(report['summary'].get('next_training_order') or []) or '无'}`",
        f"- 仅验证标签：`{', '.join(report['summary'].get('validation_only_for') or []) or '无'}`",
        "",
        "## 分标签状态",
        "",
        "| 标签 | 目标状态 | 验收状态 | 最新可评价标签日 | 当前研究表 | 下一步 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for horizon, item in report["horizons"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{horizon}`",
                    f"`{item.get('objective_status')}`",
                    f"`{item.get('acceptance_status')}`",
                    f"`{item.get('latest_mature_label_date')}`",
                    f"`{item.get('current_best_research_table')}`",
                    str(item.get("next_step")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本报告只做模型研究状态汇总，未训练模型、未生成新预测、未修改 production manifest、未生成交易信号、未运行回测。",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a current model research status report.")
    parser.add_argument("--queue-json", required=True)
    parser.add_argument("--snapshot-json", required=True)
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--formula-validation-json", action="append", default=[], help="HORIZON=PATH")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    formula_validations: dict[str, dict[str, Any]] = {}
    for item in args.formula_validation_json:
        horizon, path = str(item).split("=", 1)
        formula_validations[horizon] = _load_json(path)

    report = build_status_report(
        queue=_load_json(args.queue_json),
        snapshot=_load_json(args.snapshot_json),
        readiness=_load_json(args.readiness_json),
        formula_validations=formula_validations,
    )
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "output_json": str(out_json), "output_md": str(args.output_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
