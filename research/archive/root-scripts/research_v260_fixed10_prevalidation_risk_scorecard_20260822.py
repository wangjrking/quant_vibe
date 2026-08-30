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
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_prevalidation_risk_scorecard_20260822"
)
CHECKPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
ROLLING = (
    REPORTS
    / "strategy_agent_v260_fixed10_rolling_relative_robustness_20260822/"
    "rolling_relative_robustness.json"
)
BOOTSTRAP = (
    REPORTS
    / "strategy_agent_v260_fixed10_paired_block_bootstrap_20260822/"
    "paired_block_bootstrap.json"
)
FRONTIER = (
    REPORTS
    / "strategy_agent_v260_fixed10_pre2026_candidate_frontier_20260822/"
    "candidate_frontier.json"
)
PROTOCOL = (
    REPORTS
    / "strategy_agent_v260_fixed10_one_shot_2026_validation_protocol_20260822/"
    "one_shot_validation_protocol.json"
)


def classify_validation_result(
    candidate: dict,
    production: dict,
    frozen_gates_passed: bool,
) -> str:
    if not frozen_gates_passed:
        return "reject"
    return_higher = candidate["cumulative_return"] > production["cumulative_return"]
    if not return_higher:
        return "reject"
    risk_adjusted = (
        candidate["sharpe"] >= production["sharpe"]
        and candidate["max_drawdown"] <= production["max_drawdown"]
    )
    return (
        "risk_adjusted_upgrade"
        if risk_adjusted
        else "return_upgrade_with_risk_tradeoff"
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sources = {
        "checkpoint": json.loads(CHECKPOINT.read_text(encoding="utf-8")),
        "rolling": json.loads(ROLLING.read_text(encoding="utf-8")),
        "bootstrap": json.loads(BOOTSTRAP.read_text(encoding="utf-8")),
        "frontier": json.loads(FRONTIER.read_text(encoding="utf-8")),
        "protocol": json.loads(PROTOCOL.read_text(encoding="utf-8")),
    }
    if any(item.get("validation_2026_opened") is not False for item in sources.values()):
        raise PermissionError("2026 validation is not closed in all scorecard sources")
    checkpoint = sources["checkpoint"]
    candidate = checkpoint["current_best_equalweight"]
    production = checkpoint["production_baseline"]
    rolling = sources["rolling"]
    bootstrap = sources["bootstrap"]["comparisons"]["current_weak2_vs_production"]
    frontier = sources["frontier"]
    if not frontier["current_candidate"]["utility_rank_is_first"]:
        raise RuntimeError("current candidate is no longer first among eligible variants")

    delta = {
        key: float(candidate[key] - production[key])
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    annual = rolling["annual"]
    positive_annual = [row["period"] for row in annual if row["excess_return"] > 0]
    negative_annual = [row["period"] for row in annual if row["excess_return"] <= 0]
    payload = {
        "status": "prevalidation_risk_scorecard_complete_2026_not_opened",
        "candidate": checkpoint["selected_candidate"],
        "pre2026_metrics": {
            "candidate": candidate,
            "production": production,
            "delta_candidate_minus_production": delta,
        },
        "time_breadth": {
            "annual_outperformance_frequency": rolling[
                "annual_outperformance_frequency"
            ],
            "positive_excess_years": positive_annual,
            "nonpositive_excess_years": negative_annual,
            "quarterly_outperformance_frequency": rolling[
                "quarterly_outperformance_frequency"
            ],
            "rolling_21d_outperformance_frequency": rolling["rolling_windows"][
                "21"
            ]["fixed10_outperformance_frequency"],
            "rolling_63d_outperformance_frequency": rolling["rolling_windows"][
                "63"
            ]["fixed10_outperformance_frequency"],
            "rolling_126d_outperformance_frequency": rolling["rolling_windows"][
                "126"
            ]["fixed10_outperformance_frequency"],
            "rolling_252d_outperformance_frequency": rolling["rolling_windows"][
                "252"
            ]["fixed10_outperformance_frequency"],
        },
        "paired_block_bootstrap": {
            block: {
                "probability_return_higher": values[
                    "probability_annualized_log_return_higher"
                ],
                "probability_sharpe_higher": values["probability_sharpe_higher"],
                "probability_drawdown_lower": values[
                    "probability_max_drawdown_lower"
                ],
            }
            for block, values in bootstrap.items()
        },
        "pre2026_classification": classify_validation_result(
            candidate, production, frozen_gates_passed=True
        ),
        "one_shot_result_labels": {
            "reject": "frozen base-candidate gate fails or return does not beat production",
            "return_upgrade_with_risk_tradeoff": (
                "return beats production and frozen gates pass, but Sharpe or maximum "
                "drawdown does not both match production"
            ),
            "risk_adjusted_upgrade": (
                "return beats production, Sharpe is not lower, maximum drawdown is not "
                "higher, and all frozen gates pass"
            ),
        },
        "decision": {
            "candidate_role": "aggressive_return_candidate",
            "production_role": "lower_drawdown_reference",
            "interpretation": (
                "pre-2026 evidence supports a return-seeking alternative, not a proven "
                "risk-adjusted replacement; the one-shot 2026 result must preserve this "
                "distinction"
            ),
            "no_new_thresholds_fitted": True,
            "candidate_changed": False,
        },
        "source_protocol_id": sources["protocol"]["protocol_id"],
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "prevalidation_risk_scorecard.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "classification": payload["pre2026_classification"],
        "time_breadth": payload["time_breadth"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
