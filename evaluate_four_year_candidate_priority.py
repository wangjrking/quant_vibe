from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _gte(left: Any, right: Any) -> bool:
    try:
        return float(left) >= float(right)
    except (TypeError, ValueError):
        return False


def _label_key_from_label(label: str) -> str:
    if "1d" in label:
        return "1d"
    if "3d" in label:
        return "3d"
    if "5d" in label:
        return "5d"
    if "10d" in label:
        return "10d"
    raise KeyError(f"unsupported label: {label}")


def evaluate_priority(
    candidate_status: dict[str, Any],
    comparison: dict[str, Any],
    stability: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stability_by_label = {row["label"]: row for row in stability.get("rows", [])}
    labels = candidate_status.get("current_best_candidates", {})
    default_rules = policy.get("default_rules", {})
    label_rules = policy.get("labels", {})

    for label, record in labels.items():
        label_key = _label_key_from_label(label)
        comp = comparison["labels"][label_key]
        stable = stability_by_label.get(label, {})
        rules = label_rules.get(label, {})

        discussion_checks = {
            "promotable_formal_candidate": record.get("decision") == "promotable_formal_candidate",
            "full_top5_delta_positive": _is_positive(record.get("full_top5_delta")),
            "recent63_top5_delta_positive": _is_positive(record.get("recent63_top5_delta")),
            "recent20_top5_delta_positive": _is_positive(record.get("recent20_top5_delta")),
            "recent63_abs_top5_positive": _is_positive(comp.get("recent63_candidate_abs", {}).get("top5")),
            "recent20_abs_top5_positive": _is_positive(comp.get("recent20_candidate_abs", {}).get("top5")),
        }
        discussion_passed = all(
            discussion_checks.get(name, False)
            for name in default_rules.get("discussion_requires", [])
        )

        replacement_checks = {
            "current_full_rank_ic_delta_gte_replaced": _gte(
                record.get("full_rank_ic_delta"),
                record.get("replaced_full_rank_ic_delta"),
            ),
            "current_full_top5_delta_gte_replaced": _gte(
                record.get("full_top5_delta"),
                record.get("replaced_full_top5_delta"),
            ),
        }
        clear_replacement = all(
            replacement_checks.get(name, False)
            for name in default_rules.get("clear_replacement_requires", [])
        )

        if discussion_passed and clear_replacement:
            priority_status = default_rules.get("clear_case_status")
        elif discussion_passed:
            priority_status = default_rules.get("mixed_case_status")
        else:
            priority_status = "research_only_not_ready_for_candidate_discussion"

        formal_precheck_ready = clear_replacement
        requires_formal_explanation = discussion_passed and not clear_replacement

        explanation = []
        if discussion_passed:
            explanation.append("已满足四年候选讨论硬门槛。")
        else:
            explanation.append("尚未满足四年候选讨论硬门槛。")
        if clear_replacement:
            explanation.append("相对被替换候选，当前方案在 Full RankIC 与 Full Top5 上均不弱。")
        elif requires_formal_explanation:
            explanation.append("相对被替换候选存在指标分化，formal 送审前必须补充选择依据。")
        if label == "executable_10d_open_return" and not _is_positive(comp.get("recent20_candidate_abs", {}).get("rank_ic")):
            explanation.append("10D 的 Recent20 RankIC 接近零或为负，需解释 RankIC 与 TopN 分化。")

        row = {
            "label": label,
            "asset": record.get("asset"),
            "table": record.get("table"),
            "selection_axis": rules.get("selection_axis"),
            "discussion_passed": discussion_passed,
            "clear_replacement": clear_replacement,
            "priority_status": priority_status,
            "formal_precheck_ready": formal_precheck_ready,
            "requires_formal_explanation": requires_formal_explanation,
            "full_rank_ic_delta": record.get("full_rank_ic_delta"),
            "full_top5_delta": record.get("full_top5_delta"),
            "replaced_full_rank_ic_delta": record.get("replaced_full_rank_ic_delta"),
            "replaced_full_top5_delta": record.get("replaced_full_top5_delta"),
            "recent63_abs_top5": comp.get("recent63_candidate_abs", {}).get("top5"),
            "recent20_abs_top5": comp.get("recent20_candidate_abs", {}).get("top5"),
            "recent63_top5_delta": record.get("recent63_top5_delta"),
            "recent20_top5_delta": record.get("recent20_top5_delta"),
            "recent20_abs_rank_ic": comp.get("recent20_candidate_abs", {}).get("rank_ic"),
            "stability_score": stable.get("stability_score"),
            "optimize_priority": stable.get("optimize_priority"),
            "discussion_checks": discussion_checks,
            "replacement_checks": replacement_checks,
            "note": rules.get("note"),
            "explanation": " ".join(explanation),
        }
        rows.append(row)

    return {
        "generated_at": policy.get("generated_at"),
        "policy_config_id": policy.get("config_id"),
        "scope": "four_year_candidate_priority_review",
        "rows": rows,
        "boundaries": policy.get("boundaries", {}),
    }


def build_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# 四年候选优先级评审",
        "",
        f"策略配置：`{review['policy_config_id']}`",
        "",
        "## 当前结论",
        "",
    ]
    for row in review["rows"]:
        lines.append(
            f"- `{row['label']}`：`{row['priority_status']}`；"
            f"discussion_passed={row['discussion_passed']}；"
            f"clear_replacement={row['clear_replacement']}；"
            f"formal_precheck_ready={row['formal_precheck_ready']}。"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
        ]
    )
    for row in review["rows"]:
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- 当前候选：`{row['asset']}`",
                f"- 选择轴：`{row['selection_axis']}`",
                f"- Full RankIC Delta：`{row['full_rank_ic_delta']}`；被替换候选：`{row['replaced_full_rank_ic_delta']}`",
                f"- Full Top5 Delta：`{row['full_top5_delta']}`；被替换候选：`{row['replaced_full_top5_delta']}`",
                f"- Recent63 Abs Top5：`{row['recent63_abs_top5']}`；Recent20 Abs Top5：`{row['recent20_abs_top5']}`",
                f"- Recent63 Top5 Delta：`{row['recent63_top5_delta']}`；Recent20 Top5 Delta：`{row['recent20_top5_delta']}`",
                f"- 解释：{row['explanation']}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether four-year candidates are discussion-only or clearly superior to replaced candidates.")
    parser.add_argument("--candidate-status-json", required=True)
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument("--stability-json", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    review = evaluate_priority(
        _load_json(args.candidate_status_json),
        _load_json(args.comparison_json),
        _load_json(args.stability_json),
        _load_json(args.policy_json),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(review), encoding="utf-8")
    print(json.dumps(review, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
