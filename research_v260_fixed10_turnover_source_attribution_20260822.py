from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_weak2_drawdown_position_attribution_20260822 as source
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_turnover_source_attribution_20260822"
)
CATEGORIES = (
    "replacement_exit",
    "new_entry",
    "equalweight_trim",
    "equalweight_topup",
)
CANDIDATE_ID = "fixed10_age10_confirmed_weak2_vol_priority005"


def classify_trade(
    action: str,
    target_pct: float,
    shares_before: float,
    shares_after: float,
) -> str:
    side = str(action).upper()
    before = float(shares_before)
    after = float(shares_after)
    target = float(target_pct)
    if min(before, after, target) < 0:
        raise ValueError("trade classification inputs must be non-negative")
    if side == "SELL":
        if not after < before:
            raise ValueError("SELL must reduce shares")
        return "replacement_exit" if after == 0.0 else "equalweight_trim"
    if side == "BUY":
        if not after > before:
            raise ValueError("BUY must increase shares")
        return "new_entry" if before == 0.0 else "equalweight_topup"
    raise ValueError(f"unsupported action: {action}")


def years_in_frame(daily: pd.DataFrame) -> float:
    if daily.empty:
        raise ValueError("daily frame is empty")
    dates = pd.to_datetime(daily["date"].astype(str), format="%Y%m%d")
    return max(
        float((dates.iloc[-1] - dates.iloc[0]).days / 365.25),
        float(len(daily) / 252.0),
    )


def attribute_turnover(
    daily: pd.DataFrame,
    actions: pd.DataFrame,
    observations: list[dict],
) -> dict:
    daily_by_date = daily.assign(date=daily["date"].astype(str)).set_index("date")
    actions_by_date = {
        str(date): group.copy()
        for date, group in actions.groupby(actions["buy_date"].astype(str), sort=False)
    }
    observation_by_date = {str(item["buy_date"]): item for item in observations}
    if len(observation_by_date) != len(observations):
        raise ValueError("duplicate observation date")
    if set(actions_by_date) - set(observation_by_date):
        raise ValueError("action date is missing its position observation")

    records: list[dict] = []
    daily_reconciliation: list[dict] = []
    category_turnover_by_year: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for date, observation in observation_by_date.items():
        if date not in daily_by_date.index:
            raise ValueError("observation date is missing from daily frame")
        before = {
            str(item["stock_code"]): float(item["shares_before"])
            for item in observation["mark_records"]
        }
        after = {
            str(code): float(item["shares"])
            for code, item in observation["positions_after_trades"].items()
        }
        day_actions = actions_by_date.get(date)
        reconstructed = 0.0
        if day_actions is not None:
            duplicates = day_actions["stock_code"].astype(str).duplicated()
            if bool(duplicates.any()):
                raise ValueError("multiple same-day actions for one stock are ambiguous")
            for row in day_actions.itertuples(index=False):
                code = str(row.stock_code)
                shares_before = before.get(code, 0.0)
                shares_after = after.get(code, 0.0)
                category = classify_trade(
                    row.action,
                    row.target_pct,
                    shares_before,
                    shares_after,
                )
                gross_notional = abs(shares_after - shares_before) * float(
                    row.execution_open_raw
                )
                equity_before = float(observation["equity_before_trades"])
                turnover = gross_notional / max(equity_before, 1.0)
                reconstructed += turnover
                category_turnover_by_year[date[:4]][category] += turnover
                records.append(
                    {
                        "buy_date": date,
                        "stock_code": code,
                        "action": str(row.action),
                        "category": category,
                        "shares_before": shares_before,
                        "shares_after": shares_after,
                        "gross_notional": gross_notional,
                        "turnover": turnover,
                    }
                )
        reported = float(daily_by_date.loc[date, "turnover"])
        daily_reconciliation.append(
            {
                "buy_date": date,
                "reported_turnover": reported,
                "reconstructed_turnover": reconstructed,
                "difference": reconstructed - reported,
            }
        )

    record_frame = pd.DataFrame(records)
    reconciliation = pd.DataFrame(daily_reconciliation)
    max_difference = float(reconciliation["difference"].abs().max())
    if max_difference > 1e-12:
        raise RuntimeError("turnover attribution does not reconcile to simulator output")

    years = years_in_frame(daily)
    total_turnover = float(daily["turnover"].sum())
    by_category = {}
    for category in CATEGORIES:
        selected = record_frame[record_frame["category"] == category]
        turnover_sum = float(selected["turnover"].sum()) if not selected.empty else 0.0
        by_category[category] = {
            "action_count": int(len(selected)),
            "active_day_count": int(selected["buy_date"].nunique()) if not selected.empty else 0,
            "gross_notional": float(selected["gross_notional"].sum()) if not selected.empty else 0.0,
            "turnover_sum": turnover_sum,
            "turnover_share": turnover_sum / total_turnover if total_turnover > 0 else 0.0,
            "turnover_annualized": turnover_sum / years,
        }
    by_year = {
        year: {
            category: float(values.get(category, 0.0))
            for category in CATEGORIES
        }
        for year, values in sorted(category_turnover_by_year.items())
    }
    maintenance_share = (
        by_category["equalweight_trim"]["turnover_share"]
        + by_category["equalweight_topup"]["turnover_share"]
    )
    replacement_share = (
        by_category["replacement_exit"]["turnover_share"]
        + by_category["new_entry"]["turnover_share"]
    )
    return {
        "total_action_count": int(len(record_frame)),
        "total_turnover_sum": total_turnover,
        "total_turnover_annualized": total_turnover / years,
        "by_category": by_category,
        "by_year_turnover_sum": by_year,
        "replacement_turnover_share": replacement_share,
        "equalweight_maintenance_turnover_share": maintenance_share,
        "reconciliation": {
            "days": int(len(reconciliation)),
            "max_absolute_difference": max_difference,
            "passed": True,
        },
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(source.CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for turnover attribution")
    policy = checkpoint["selected_policy"]
    context = harness.load_context(policy)
    daily, actions, observations, _, _, _ = source.run_with_observer(
        context, policy
    )
    metric_daily = research_base.interval(
        daily, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    metric_dates = set(metric_daily["date"].astype(str))
    metric_actions = actions[
        actions["buy_date"].astype(str).isin(metric_dates)
    ].copy()
    metric_observations = [
        item for item in observations if str(item["buy_date"]) in metric_dates
    ]
    attribution = attribute_turnover(
        metric_daily, metric_actions, metric_observations
    )
    expected = checkpoint["current_best_equalweight"]
    actual = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    checkpoint_equivalence = {
        key: bool(np.isclose(actual[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("turnover attribution drifted from the current checkpoint")

    maintenance_share = attribution["equalweight_maintenance_turnover_share"]
    if maintenance_share >= 0.25:
        next_test = "test_one_bounded_equalweight_maintenance_simplification"
    else:
        next_test = "do_not_tune_equalweight_maintenance; replacement turnover dominates"
    result = {
        "status": "pre2026_turnover_source_attribution_complete",
        "source_strategy": context.rules["strategy_id"],
        "candidate": CANDIDATE_ID,
        "development_boundary": context.access,
        "metric_window": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "turnover_attribution": attribution,
        "decision_rule": (
            "Only test an equalweight-maintenance simplification when maintenance "
            "contributes at least 25% of total turnover; otherwise retain the current "
            "20-session maintenance rule and avoid a low-impact parameter search."
        ),
        "next_test": next_test,
        "checkpoint_equivalence": checkpoint_equivalence,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "turnover_source_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
