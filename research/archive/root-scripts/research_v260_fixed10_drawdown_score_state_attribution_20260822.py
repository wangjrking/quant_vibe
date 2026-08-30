from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_attribution_20260822 as drawdown
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_weak2_drawdown_position_attribution_20260822 as source
import research_v260_fixed10_weak2_score_decay_sell_priority_20260822 as decay
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_drawdown_score_state_attribution_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
EXISTING_SELL_THRESHOLD = 0.85
HIGH_VOLATILITY_PERCENTILE = 0.75


def classify_row(
    score: float,
    score_change_3d: float,
    volatility_percentile: float,
    confirmed_weak: bool,
) -> dict[str, bool]:
    return {
        "score_below_existing_exit_threshold": bool(
            np.isfinite(score) and score < EXISTING_SELL_THRESHOLD
        ),
        "score_declining_3d": bool(
            np.isfinite(score_change_3d) and score_change_3d < 0.0
        ),
        "high_volatility": bool(
            np.isfinite(volatility_percentile)
            and volatility_percentile >= HIGH_VOLATILITY_PERCENTILE
        ),
        "confirmed_weak_market": bool(confirmed_weak),
        "confirmed_weak_and_high_volatility": bool(
            confirmed_weak
            and np.isfinite(volatility_percentile)
            and volatility_percentile >= HIGH_VOLATILITY_PERCENTILE
        ),
    }


def build_position_rows(
    observations: list[dict],
    context,
    volatility_rank: np.ndarray,
    confirmed_weak: np.ndarray,
) -> list[dict]:
    dates = [str(value) for value in context.arrays["dates"]]
    stocks = [str(value) for value in context.arrays["stocks"]]
    date_index = {value: index for index, value in enumerate(dates)}
    stock_index = {value: index for index, value in enumerate(stocks)}
    if len(date_index) != len(dates) or len(stock_index) != len(stocks):
        raise ValueError("context keys must be unique")
    change_3d = decay.score_change(context.score, 3)
    rows: list[dict] = []
    for observation in observations:
        signal_date = str(observation["signal_date"])
        buy_date = str(observation["buy_date"])
        t = date_index.get(signal_date)
        if t is None:
            raise KeyError(f"signal date absent from context: {signal_date}")
        for mark in observation["mark_records"]:
            stock_code = str(mark["stock_code"])
            idx = stock_index.get(stock_code)
            if idx is None:
                raise KeyError(f"stock absent from context: {stock_code}")
            prior_value = float(mark["shares_before"]) * float(mark["prior_price"])
            if not np.isfinite(prior_value) or prior_value <= 0.0:
                raise ValueError("held-position prior value must be positive and finite")
            mark_return = float(mark["mark_pnl"]) / prior_value
            score = float(context.score[t, idx])
            score_delta = float(change_3d[t, idx])
            volatility = float(volatility_rank[t, idx])
            flags = classify_row(
                score,
                score_delta,
                volatility,
                bool(confirmed_weak[t]),
            )
            rows.append(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "year": buy_date[:4],
                    "stock_code": stock_code,
                    "prior_value": prior_value,
                    "mark_pnl": float(mark["mark_pnl"]),
                    "mark_return": mark_return,
                    "score": score,
                    "score_change_3d": score_delta,
                    "volatility_percentile": volatility,
                    **flags,
                }
            )
    return rows


def summarize_subset(rows: list[dict]) -> dict:
    if not rows:
        return {
            "position_days": 0,
            "gross_exposure": 0.0,
            "net_mark_pnl": 0.0,
            "gross_negative_mark_pnl": 0.0,
            "mean_mark_return": None,
            "negative_position_day_ratio": None,
        }
    exposure = float(sum(float(row["prior_value"]) for row in rows))
    negative = float(sum(min(float(row["mark_pnl"]), 0.0) for row in rows))
    return {
        "position_days": int(len(rows)),
        "gross_exposure": exposure,
        "net_mark_pnl": float(sum(float(row["mark_pnl"]) for row in rows)),
        "gross_negative_mark_pnl": negative,
        "mean_mark_return": float(np.mean([row["mark_return"] for row in rows])),
        "negative_position_day_ratio": float(
            np.mean([float(row["mark_pnl"]) < 0.0 for row in rows])
        ),
    }


def summarize_flags(rows: list[dict], flags: tuple[str, ...]) -> dict:
    total = summarize_subset(rows)
    total_negative = float(total["gross_negative_mark_pnl"])
    total_exposure = float(total["gross_exposure"])
    result = {}
    for flag in flags:
        selected = [row for row in rows if bool(row[flag])]
        summary = summarize_subset(selected)
        summary["share_of_rows"] = (
            float(summary["position_days"] / total["position_days"])
            if total["position_days"]
            else 0.0
        )
        summary["share_of_gross_exposure"] = (
            float(summary["gross_exposure"] / total_exposure)
            if total_exposure > 0.0
            else 0.0
        )
        summary["share_of_gross_negative"] = (
            float(summary["gross_negative_mark_pnl"] / total_negative)
            if total_negative < 0.0
            else 0.0
        )
        result[flag] = summary
    return {"all_positions": total, "flags": result}


def annual_direction(rows: list[dict], flag: str) -> dict:
    result = {}
    for year in sorted({str(row["year"]) for row in rows}):
        annual = [row for row in rows if str(row["year"]) == year]
        selected = [row for row in annual if bool(row[flag])]
        complement = [row for row in annual if not bool(row[flag])]
        selected_summary = summarize_subset(selected)
        complement_summary = summarize_subset(complement)
        selected_mean = selected_summary["mean_mark_return"]
        complement_mean = complement_summary["mean_mark_return"]
        result[year] = {
            "flagged_mean_mark_return": selected_mean,
            "unflagged_mean_mark_return": complement_mean,
            "flagged_minus_unflagged": (
                float(selected_mean - complement_mean)
                if selected_mean is not None and complement_mean is not None
                else None
            ),
            "flagged_position_days": selected_summary["position_days"],
        }
    return result


def broad_support(
    full_rows: list[dict], episode_rows: list[dict], flag: str
) -> dict:
    full = summarize_flags(full_rows, (flag,))["flags"][flag]
    episode = summarize_flags(episode_rows, (flag,))["flags"][flag]
    yearly = annual_direction(full_rows, flag)
    comparable = [
        value["flagged_minus_unflagged"]
        for value in yearly.values()
        if value["flagged_minus_unflagged"] is not None
    ]
    negative_years = int(sum(value < 0.0 for value in comparable))
    loss_enrichment = (
        float(episode["share_of_gross_negative"] / episode["share_of_gross_exposure"])
        if episode["share_of_gross_exposure"] > 0.0
        else 0.0
    )
    supported = bool(
        full["mean_mark_return"] is not None
        and full["mean_mark_return"] < 0.0
        and negative_years >= 3
        and loss_enrichment >= 1.25
        and episode["share_of_gross_negative"] >= 0.30
    )
    return {
        "full_sample": full,
        "maximum_drawdown_episode": episode,
        "annual_direction": yearly,
        "negative_flagged_minus_unflagged_year_count": negative_years,
        "drawdown_loss_enrichment": loss_enrichment,
        "broad_support_for_bounded_rule_test": supported,
    }


def normalized_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            key: (
                None
                if isinstance(value, (float, np.floating))
                and not np.isfinite(value)
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for score-state attribution")
    policy = checkpoint["selected_policy"]
    context = harness.load_context(policy)
    daily, actions, observations, volatility_rank, _, confirmed_weak = (
        source.run_with_observer(context, policy)
    )
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    expected = checkpoint["current_best_equalweight"]
    equivalence = {
        key: bool(np.isclose(metrics[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("score-state attribution baseline drifted")
    rows = build_position_rows(observations, context, volatility_rank, confirmed_weak)
    episode = drawdown.maximum_drawdown_episode(daily)
    episode_rows = [
        row
        for row in rows
        if str(episode["peak_date"]) < str(row["buy_date"])
        <= str(episode["trough_date"])
    ]
    flags = (
        "score_below_existing_exit_threshold",
        "score_declining_3d",
        "high_volatility",
        "confirmed_weak_market",
        "confirmed_weak_and_high_volatility",
    )
    support = {flag: broad_support(rows, episode_rows, flag) for flag in flags}
    supported = [
        flag
        for flag, value in support.items()
        if value["broad_support_for_bounded_rule_test"]
    ]
    repeated_rows = build_position_rows(
        observations, context, volatility_rank, confirmed_weak
    )
    deterministic = normalized_rows(rows) == normalized_rows(repeated_rows)
    if not deterministic:
        raise RuntimeError("score-state attribution replay failed")
    result = {
        "status": "drawdown_score_state_attribution_complete_2026_not_opened",
        "maximum_drawdown_episode": episode,
        "row_count": len(rows),
        "episode_row_count": len(episode_rows),
        "support": support,
        "supported_bounded_rule_tests": supported,
        "decision": (
            "test only the supported one-dimensional bounded rules"
            if supported
            else "do not add another score, decay, volatility or market-state hard rule"
        ),
        "checkpoint_equivalence": equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "score_state_attribution.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "episode": episode,
                "supported": supported,
                "support": {
                    flag: {
                        "episode_loss_share": value["maximum_drawdown_episode"][
                            "share_of_gross_negative"
                        ],
                        "episode_exposure_share": value["maximum_drawdown_episode"][
                            "share_of_gross_exposure"
                        ],
                        "loss_enrichment": value["drawdown_loss_enrichment"],
                        "negative_years": value[
                            "negative_flagged_minus_unflagged_year_count"
                        ],
                        "supported": value[
                            "broad_support_for_bounded_rule_test"
                        ],
                    }
                    for flag, value in support.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
