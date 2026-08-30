from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_pressure_exit_attribution_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
ROBUSTNESS_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_robustness_20260822/"
    "development_result.json"
)


def period_log_contribution(frame: pd.DataFrame, frequency: str) -> dict[str, float]:
    dates = pd.to_datetime(frame["date"], format="%Y%m%d")
    if frequency == "year":
        labels = dates.dt.year.astype(str)
    elif frequency == "half":
        labels = dates.dt.year.astype(str) + "H" + ((dates.dt.month > 6).astype(int) + 1).astype(str)
    elif frequency == "quarter":
        labels = dates.dt.to_period("Q").astype(str)
    else:
        raise ValueError(f"unsupported frequency: {frequency}")
    return {
        str(label): float(values.sum())
        for label, values in frame["log_excess"].groupby(labels)
    }


def action_keys(actions: pd.DataFrame, buy_dates: set[str]) -> set[tuple]:
    scoped = actions[actions["buy_date"].astype(str).isin(buy_dates)]
    return {
        (str(row.buy_date), str(row.action), str(row.stock_code))
        for row in scoped.itertuples(index=False)
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    robustness = json.loads(ROBUSTNESS_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"] or robustness["validation_2026_opened"]:
        raise PermissionError("2026 data entered pressure attribution")
    candidate_policy = copy.deepcopy(checkpoint["selected_policy"])
    baseline_policy = copy.deepcopy(candidate_policy)
    baseline_policy["score_sell_pressure_limit"] = 1
    context = harness.load_context(candidate_policy)
    candidate_result, candidate_daily, candidate_actions = harness.run_policy(
        context, candidate_policy
    )
    baseline_result, baseline_daily, baseline_actions = harness.run_policy(
        context, baseline_policy
    )

    joined = baseline_daily[["date", "return"]].merge(
        candidate_daily[["date", "return"]], on="date", suffixes=("_baseline", "_candidate"), validate="one_to_one"
    )
    joined["log_excess"] = np.log1p(joined["return_candidate"].astype(float)) - np.log1p(
        joined["return_baseline"].astype(float)
    )
    trigger_records = robustness["results"]["control_trigger4_limit2"][
        "pressure_diagnostics"
    ]["records"]
    trigger_buy_dates = {str(item["buy_date"]) for item in trigger_records}
    candidate_keys = action_keys(candidate_actions, trigger_buy_dates)
    baseline_keys = action_keys(baseline_actions, trigger_buy_dates)
    candidate_only = sorted(candidate_keys - baseline_keys)
    baseline_only = sorted(baseline_keys - candidate_keys)
    intervention_dates = sorted(
        {item[0] for item in candidate_only} | {item[0] for item in baseline_only}
    )

    quarters = period_log_contribution(joined, "quarter")
    positive_quarters = {key: value for key, value in quarters.items() if value > 0.0}
    positive_total = float(sum(positive_quarters.values()))
    max_positive_share = (
        float(max(positive_quarters.values()) / positive_total)
        if positive_total > 0.0
        else None
    )
    result = {
        "status": "pressure_exit_attribution_complete_2026_not_opened",
        "comparison": "trigger4_limit2 versus same policy with pressure limit1",
        "trigger_days": len(trigger_buy_dates),
        "direct_intervention_days": len(intervention_dates),
        "direct_intervention_years": sorted({date[:4] for date in intervention_dates}),
        "candidate_only_action_count_on_trigger_dates": len(candidate_only),
        "baseline_only_action_count_on_trigger_dates": len(baseline_only),
        "candidate_only_actions_on_trigger_dates": candidate_only,
        "baseline_only_actions_on_trigger_dates": baseline_only,
        "log_excess_by_year": period_log_contribution(joined, "year"),
        "log_excess_by_half_year": period_log_contribution(joined, "half"),
        "log_excess_by_quarter": quarters,
        "maximum_positive_quarter_share": max_positive_share,
        "candidate_metrics": candidate_result["metrics_0_30pct"],
        "baseline_metrics": baseline_result["metrics_0_30pct"],
        "diagnostic_summary": {
            "interventions_not_single_event": len(intervention_dates) >= 12,
            "interventions_span_at_least_three_years": len(
                {date[:4] for date in intervention_dates}
            )
            >= 3,
            "positive_log_excess_not_dominated_by_one_quarter": (
                max_positive_share is not None and max_positive_share <= 0.60
            ),
        },
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "attribution.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
