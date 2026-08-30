from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


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
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_return_concentration_diagnostic_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def observation_hash(observations: list[dict]) -> str:
    raw = json.dumps(
        observations, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run_current(context, policy: dict):
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )
    observations: list[dict] = []
    daily, actions = age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        round1.BASELINE_COST,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
        position_observer=observations.append,
    )
    return daily, actions, observations


def daily_concentration(daily) -> dict:
    frame = daily.loc[
        (daily["date"].astype(str) >= research_base.FIRST_BUY)
        & (daily["date"].astype(str) <= round1.DEVELOPMENT_END)
    ].copy()
    values = frame["return"].to_numpy(dtype=np.float64)
    positive = np.clip(values, 0.0, None)
    positive_sum = float(positive.sum())
    order = np.argsort(values)[::-1]
    removed = {}
    for count in (1, 5, 10, 20):
        adjusted = values.copy()
        adjusted[order[:count]] = 0.0
        removed[str(count)] = {
            "cumulative_return_without_top_days": float(np.prod(1.0 + adjusted) - 1.0),
            "top_days_share_of_positive_simple_returns": float(
                positive[order[:count]].sum() / positive_sum
            ) if positive_sum > 0 else None,
            "dates": frame.iloc[order[:count]]["date"].astype(str).tolist(),
        }
    return {
        "day_count": int(len(frame)),
        "positive_day_count": int((values > 0).sum()),
        "negative_day_count": int((values < 0).sum()),
        "top_day_return": float(values[order[0]]),
        "without_top_positive_days": removed,
    }


def monthly_concentration(daily) -> dict:
    frame = daily.loc[
        (daily["date"].astype(str) >= research_base.FIRST_BUY)
        & (daily["date"].astype(str) <= round1.DEVELOPMENT_END)
    ].copy()
    frame["month"] = frame["date"].astype(str).str[:6]
    monthly = (
        frame.groupby("month", sort=True)["return"]
        .apply(lambda values: float(np.prod(1.0 + values.to_numpy(dtype=np.float64)) - 1.0))
    )
    values = monthly.to_numpy(dtype=np.float64)
    labels = monthly.index.astype(str).to_numpy()
    positive = np.clip(values, 0.0, None)
    positive_sum = float(positive.sum())
    order = np.argsort(values)[::-1]
    removed = {}
    for count in (1, 3, 5):
        adjusted = values.copy()
        adjusted[order[:count]] = 0.0
        removed[str(count)] = {
            "cumulative_return_without_top_months": float(
                np.prod(1.0 + adjusted) - 1.0
            ),
            "top_months_share_of_positive_simple_returns": (
                float(positive[order[:count]].sum() / positive_sum)
                if positive_sum > 0.0
                else None
            ),
            "months": labels[order[:count]].tolist(),
        }
    positive_shares = positive[positive > 0.0] / positive_sum if positive_sum > 0.0 else np.array([])
    return {
        "month_count": int(len(values)),
        "positive_month_count": int((values > 0.0).sum()),
        "negative_month_count": int((values < 0.0).sum()),
        "positive_month_return_hhi": (
            float(np.square(positive_shares).sum()) if len(positive_shares) else None
        ),
        "without_top_positive_months": removed,
    }


def stock_mark_concentration(observations: list[dict]) -> dict:
    pnl = defaultdict(float)
    for observation in observations:
        buy_date = str(observation.get("buy_date", ""))
        if buy_date < research_base.FIRST_BUY or buy_date > round1.DEVELOPMENT_END:
            continue
        for mark in observation.get("mark_records", []):
            pnl[str(mark["stock_code"])] += float(mark["mark_pnl"])
    ordered = sorted(pnl.items(), key=lambda item: (-item[1], item[0]))
    positive_total = float(sum(max(value, 0.0) for _, value in ordered))
    negative_total = float(sum(min(value, 0.0) for _, value in ordered))
    positive_shares = np.array(
        [max(value, 0.0) / positive_total for _, value in ordered], dtype=np.float64
    ) if positive_total > 0 else np.array([], dtype=np.float64)
    return {
        "stock_count": int(len(ordered)),
        "gross_positive_mark_pnl": positive_total,
        "gross_negative_mark_pnl": negative_total,
        "net_mark_pnl": float(sum(value for _, value in ordered)),
        "positive_pnl_hhi": float(np.square(positive_shares).sum())
        if len(positive_shares) else None,
        "top1_share_of_positive_mark_pnl": float(positive_shares[:1].sum())
        if len(positive_shares) else None,
        "top5_share_of_positive_mark_pnl": float(positive_shares[:5].sum())
        if len(positive_shares) else None,
        "top10_share_of_positive_mark_pnl": float(positive_shares[:10].sum())
        if len(positive_shares) else None,
        "top10_positive_contributors": [
            {"stock_code": code, "mark_pnl": float(value)}
            for code, value in ordered[:10]
        ],
        "bottom10_contributors": [
            {"stock_code": code, "mark_pnl": float(value)}
            for code, value in sorted(pnl.items(), key=lambda item: (item[1], item[0]))[:10]
        ],
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered return-concentration diagnostic")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    daily, actions, observations = run_current(context, policy)
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    expected = checkpoint["current_best_equalweight"]
    equivalent = {
        key: bool(np.isclose(metrics[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalent.values()):
        raise RuntimeError("return-concentration baseline drifted")
    production_daily, _ = research_base.run_shell(
        context.harness,
        context.arrays,
        context.protocol,
        context.score,
        context.order,
        context.definition,
        "production_shell",
        round1.DEVELOPMENT_END,
        actions=True,
    )
    _, _, repeated_observations = run_current(context, policy)
    deterministic = observation_hash(observations) == observation_hash(
        repeated_observations
    )
    if not deterministic:
        raise RuntimeError("position observations are not deterministic")
    result = {
        "status": "return_concentration_diagnostic_complete_2026_not_opened",
        "method": (
            "attribute gross daily held-position mark PnL by stock and remove the "
            "largest portfolio-return days without altering any rule"
        ),
        "fixed10_metrics": metrics,
        "fixed10_daily_concentration": daily_concentration(daily),
        "production_daily_concentration": daily_concentration(production_daily),
        "fixed10_monthly_concentration": monthly_concentration(daily),
        "production_monthly_concentration": monthly_concentration(production_daily),
        "fixed10_stock_mark_concentration": stock_mark_concentration(observations),
        "baseline_equivalence": equivalent,
        "deterministic_observation_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "return_concentration.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
