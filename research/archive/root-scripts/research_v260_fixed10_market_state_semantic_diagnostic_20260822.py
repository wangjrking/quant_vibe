from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_market_state_semantic_diagnostic_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
WINDOWS = (5, 10, 20)
MAX_DRAWDOWN_START = "20231205"
MAX_DRAWDOWN_END = "20240206"


def cross_sectional_median_return(close_qfq: np.ndarray) -> np.ndarray:
    close = np.asarray(close_qfq, dtype=np.float64)
    returns = np.full(close.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(close[1:])
        & np.isfinite(close[:-1])
        & (close[1:] > 0.0)
        & (close[:-1] > 0.0)
    )
    adjacent = np.full_like(close[1:], np.nan)
    adjacent[valid] = close[1:][valid] / close[:-1][valid] - 1.0
    returns[1:] = adjacent
    market = np.full(len(close), np.nan, dtype=np.float64)
    observed = np.any(np.isfinite(returns), axis=1)
    market[observed] = np.nanmedian(returns[observed], axis=1)
    return market


def trailing_compound_return(daily_return: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(daily_return, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    for end in range(window - 1, len(values)):
        sample = values[end - window + 1 : end + 1]
        if np.all(np.isfinite(sample)):
            result[end] = float(np.prod(1.0 + sample) - 1.0)
    return result


def summarize_mask(mask: np.ndarray, returns: dict[int, np.ndarray]) -> dict:
    count = int(np.sum(mask))
    result = {"days": count}
    for window, values in returns.items():
        observed = values[mask & np.isfinite(values)]
        result[f"trailing_{window}d_observed_days"] = int(len(observed))
        result[f"trailing_{window}d_median"] = (
            float(np.median(observed)) if len(observed) else None
        )
        result[f"trailing_{window}d_negative_days"] = int(np.sum(observed < 0.0))
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered market-state diagnostic")
    context = harness.load_context(checkpoint["selected_policy"])
    dates = np.asarray(context.arrays["dates"], dtype=str)
    production_strong = regime.strong_market_mask(context.score, context.protocol)
    production_confirmed_weak = confirmed.confirmed_weak_mask(production_strong, 2)
    market_daily = cross_sectional_median_return(context.arrays["close_qfq"])
    trailing = {
        window: trailing_compound_return(market_daily, window) for window in WINDOWS
    }
    episode = (dates >= MAX_DRAWDOWN_START) & (dates <= MAX_DRAWDOWN_END)
    masks = {
        "all_days": np.ones(len(dates), dtype=np.bool_),
        "production_strong": production_strong,
        "production_weak": ~production_strong,
        "production_confirmed_weak2": production_confirmed_weak,
        "maximum_drawdown_episode": episode,
        "drawdown_production_strong": episode & production_strong,
        "drawdown_production_weak": episode & ~production_strong,
    }
    contradictions = {}
    for window, values in trailing.items():
        valid = np.isfinite(values)
        contradictions[str(window)] = {
            "production_strong_but_market_trend_negative": int(
                np.sum(production_strong & valid & (values < 0.0))
            ),
            "drawdown_strong_but_market_trend_negative": int(
                np.sum(episode & production_strong & valid & (values < 0.0))
            ),
            "production_weak_but_market_trend_nonnegative": int(
                np.sum((~production_strong) & valid & (values >= 0.0))
            ),
        }
    result = {
        "status": "market_state_semantic_diagnostic_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "production_state_semantics": (
            "cross-sectional count of model scores above the production breadth threshold"
        ),
        "price_state_semantics": (
            "PIT trailing compounded return of the daily cross-sectional median qfq return"
        ),
        "summaries": {
            name: summarize_mask(mask, trailing) for name, mask in masks.items()
        },
        "semantic_contradictions": contradictions,
        "maximum_drawdown_boundary": [MAX_DRAWDOWN_START, MAX_DRAWDOWN_END],
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "market_state_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
