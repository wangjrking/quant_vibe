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

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_vol_priority_action_concentration_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
PENALTIES = (0.0, 0.05, 0.075)
ACTION_KEY = ("signal_date", "buy_date", "action", "stock_code")


def normalized_action_keys(actions: pd.DataFrame) -> set[tuple[str, ...]]:
    if actions.empty:
        return set()
    missing = set(ACTION_KEY) - set(actions.columns)
    if missing:
        raise ValueError(f"action columns missing: {sorted(missing)}")
    selected = actions.loc[:, ACTION_KEY].astype(str)
    if selected.duplicated().any():
        duplicates = selected.loc[selected.duplicated(keep=False)].head(10)
        raise ValueError(f"duplicate action keys: {duplicates.to_dict('records')}")
    return set(selected.itertuples(index=False, name=None))


def action_difference(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict:
    candidate_keys = normalized_action_keys(candidate)
    reference_keys = normalized_action_keys(reference)
    candidate_only = candidate_keys - reference_keys
    reference_only = reference_keys - candidate_keys
    changed_dates = {
        key[1] for key in candidate_only | reference_only
    }
    return {
        "candidate_action_count": len(candidate_keys),
        "reference_action_count": len(reference_keys),
        "candidate_only_action_count": len(candidate_only),
        "reference_only_action_count": len(reference_only),
        "symmetric_difference_count": len(candidate_only) + len(reference_only),
        "changed_execution_date_count": len(changed_dates),
        "changed_execution_dates": sorted(changed_dates),
    }


def daily_log_excess(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "return"}
    if not required.issubset(candidate.columns) or not required.issubset(reference.columns):
        raise ValueError("daily frames require date and return")
    left = candidate.loc[:, ["date", "return"]].copy()
    right = reference.loc[:, ["date", "return"]].copy()
    left["date"] = left["date"].astype(str)
    right["date"] = right["date"].astype(str)
    if left["date"].duplicated().any() or right["date"].duplicated().any():
        raise ValueError("daily frame contains duplicate dates")
    merged = left.merge(
        right,
        on="date",
        how="outer",
        suffixes=("_candidate", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("daily calendars do not align")
    candidate_return = merged["return_candidate"].to_numpy(dtype=np.float64)
    reference_return = merged["return_reference"].to_numpy(dtype=np.float64)
    if np.any(candidate_return <= -1.0) or np.any(reference_return <= -1.0):
        raise ValueError("daily return cannot be at or below -100%")
    merged["log_excess"] = np.log1p(candidate_return) - np.log1p(reference_return)
    return merged.loc[:, ["date", "log_excess"]].sort_values("date").reset_index(drop=True)


def concentration_summary(frame: pd.DataFrame) -> dict:
    values = frame["log_excess"].to_numpy(dtype=np.float64)
    positive = np.sort(values[values > 1e-15])[::-1]
    negative = values[values < -1e-15]
    absolute = np.sort(np.abs(values[np.abs(values) > 1e-15]))[::-1]
    gross_positive = float(positive.sum())
    gross_absolute = float(absolute.sum())
    total = float(values.sum())

    def share(items: np.ndarray, count: int, denominator: float) -> float | None:
        if denominator <= 0.0:
            return None
        return float(items[:count].sum() / denominator)

    monthly = frame.assign(month=frame["date"].str[:6]).groupby(
        "month", sort=True
    )["log_excess"].sum()
    yearly = frame.assign(year=frame["date"].str[:4]).groupby(
        "year", sort=True
    )["log_excess"].sum()
    return {
        "total_log_excess": total,
        "equivalent_relative_wealth_gain": float(np.expm1(total)),
        "active_day_count": int(np.count_nonzero(np.abs(values) > 1e-15)),
        "positive_day_count": int(len(positive)),
        "negative_day_count": int(len(negative)),
        "top1_positive_share": share(positive, 1, gross_positive),
        "top5_positive_share": share(positive, 5, gross_positive),
        "top10_absolute_share": share(absolute, 10, gross_absolute),
        "log_excess_after_removing_top1_positive_day": float(
            total - (positive[:1].sum() if len(positive) else 0.0)
        ),
        "log_excess_after_removing_top5_positive_days": float(
            total - positive[:5].sum()
        ),
        "positive_month_fraction": float((monthly > 0.0).mean()),
        "monthly_log_excess": {str(key): float(value) for key, value in monthly.items()},
        "yearly_log_excess": {str(key): float(value) for key, value in yearly.items()},
    }


def build_priority_matrix(context, penalty: float) -> np.ndarray:
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    return priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, penalty
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered volatility-priority concentration audit")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    extra_age = age_boundary.pressure_age_schedule(strong, 10)

    runs = {}
    for penalty in PENALTIES:
        daily, actions = age_guard.run_policy_at_cost(
            context,
            policy,
            extra_age,
            round1.BASELINE_COST,
            score_sell_priority_override=build_priority_matrix(context, penalty),
        )
        runs[f"{penalty:.3f}"] = {"daily": daily, "actions": actions}

    reference = runs["0.000"]
    comparisons = {}
    for penalty in ("0.050", "0.075"):
        candidate = runs[penalty]
        excess = daily_log_excess(candidate["daily"], reference["daily"])
        comparisons[f"penalty_{penalty}_vs_0"] = {
            "action_difference": action_difference(
                candidate["actions"], reference["actions"]
            ),
            "daily_log_excess_concentration": concentration_summary(excess),
        }

    between = daily_log_excess(runs["0.050"]["daily"], runs["0.075"]["daily"])
    comparisons["penalty_0.050_vs_0.075"] = {
        "action_difference": action_difference(
            runs["0.050"]["actions"], runs["0.075"]["actions"]
        ),
        "daily_log_excess_concentration": concentration_summary(between),
    }

    current = comparisons["penalty_0.050_vs_0"]["daily_log_excess_concentration"]
    broad_support = bool(
        current["log_excess_after_removing_top5_positive_days"] > 0.0
        and current["positive_month_fraction"] >= 0.50
        and current["top5_positive_share"] is not None
        and current["top5_positive_share"] <= 0.50
    )
    result = {
        "status": "vol_priority_action_concentration_complete_2026_not_opened",
        "purpose": (
            "diagnose whether the frozen 0.05 weak-market volatility sell-priority "
            "increment is broad or driven by a few path-dependent actions"
        ),
        "comparisons": comparisons,
        "broad_incremental_support": broad_support,
        "decision": (
            "retain_0.05_with_broad_pre2026_action_support"
            if broad_support
            else "retain_0.05_but_flag_path_concentration_complexity_risk"
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "action_concentration.json", result)
    print(json.dumps({
        "status": result["status"],
        "current_vs_zero": comparisons["penalty_0.050_vs_0"],
        "decision": result["decision"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
