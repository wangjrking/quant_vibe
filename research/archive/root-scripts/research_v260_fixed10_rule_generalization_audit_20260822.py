from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


REPORTS = REPO / "quant/data_file/reports"
SOURCE = (
    REPORTS
    / "strategy_agent_v260_fixed10_current_rule_ablation_20260822/"
    "rule_ablation.json"
)
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_rule_generalization_audit_20260822"
)
YEARS = ("2022", "2023", "2024", "2025")


def support_summary(full: dict, removed: dict) -> dict:
    full_metrics = full["metrics_0_30pct"]
    removed_metrics = removed["metrics"]["metrics_0_30pct"]
    yearly_gain = {
        year: float(
            full_metrics["annual_returns"][year]
            - removed_metrics["annual_returns"][year]
        )
        for year in YEARS
    }
    positive_years = [year for year, value in yearly_gain.items() if value > 0.0]
    nonnegative_years = [year for year, value in yearly_gain.items() if value >= 0.0]
    overall_gain = float(
        full_metrics["cumulative_return"] - removed_metrics["cumulative_return"]
    )
    stress_gain = float(
        full["metrics_0_65pct"]["cumulative_return"]
        - removed["metrics"]["metrics_0_65pct"]["cumulative_return"]
    )
    train_gain = float(
        full["train_2022_2024"]["cumulative_return"]
        - removed["metrics"]["train_2022_2024"]["cumulative_return"]
    )
    holdout_gain = float(
        full["holdout_2025"]["cumulative_return"]
        - removed["metrics"]["holdout_2025"]["cumulative_return"]
    )
    supported = min(overall_gain, stress_gain, train_gain, holdout_gain) > 0.0
    if supported and len(positive_years) >= 3:
        classification = "robust_core_rule"
    elif supported and len(nonnegative_years) == len(YEARS):
        classification = "secondary_fixed_rule_not_independent_tuning_knob"
    else:
        classification = "insufficient_generalization_support"
    return {
        "classification": classification,
        "yearly_return_gain_with_rule": yearly_gain,
        "positive_year_count": len(positive_years),
        "nonnegative_year_count": len(nonnegative_years),
        "overall_cumulative_return_gain": overall_gain,
        "stress_cumulative_return_gain": stress_gain,
        "train_2022_2024_cumulative_return_gain": train_gain,
        "holdout_2025_cumulative_return_gain": holdout_gain,
    }


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("validation_2026_opened") is not False:
        raise PermissionError("rule audit cannot consume 2026 validation")
    if source.get("production_modified") is not False:
        raise PermissionError("rule audit source modified production")
    rules = {
        name: support_summary(source["full_current"], item)
        for name, item in source["ablations"].items()
    }
    unsupported = [
        name
        for name, item in rules.items()
        if item["classification"] == "insufficient_generalization_support"
    ]
    result = {
        "status": "pre2026_rule_generalization_audit_complete",
        "method": "leave-one-rule-out annual/train/holdout/stress support",
        "rules": rules,
        "unsupported_rules": unsupported,
        "decision": (
            "retain frozen candidate; no independent maintenance top-up threshold "
            "search; reuse the fixed 0.80 renewal threshold"
        ),
        "candidate_changed": False,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "rule_generalization_audit.json", result)
    print(json.dumps({
        "status": result["status"],
        "classifications": {
            name: item["classification"] for name, item in rules.items()
        },
        "candidate_changed": False,
        "validation_2026_opened": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
