from __future__ import annotations

import numpy as np

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority


def test_future_score_rows_cannot_change_prior_decisions() -> None:
    rng = np.random.default_rng(20260822)
    arrays = {
        "rank_5d": rng.random((60, 12), dtype=np.float32),
        "rank_10d": rng.random((60, 12), dtype=np.float32),
    }
    cutoff = 37
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    poisoned = {name: value.copy() for name, value in arrays.items()}
    poisoned["rank_5d"][cutoff + 1 :] = rng.normal(0, 100, (22, 12))
    poisoned["rank_10d"][cutoff + 1 :] = rng.normal(0, 100, (22, 12))
    poisoned_score, poisoned_order = v95.score_pair(poisoned, 0.0, 7, 0.1)
    np.testing.assert_array_equal(score[: cutoff + 1], poisoned_score[: cutoff + 1])
    np.testing.assert_array_equal(order[: cutoff + 1], poisoned_order[: cutoff + 1])


def test_future_rows_cannot_change_regime_or_confirmation_prefix() -> None:
    rng = np.random.default_rng(7)
    score = rng.random((50, 20))
    protocol = {
        "breadth": {"score_threshold": 0.5},
        "breadth_state": {"breadth_count_threshold": 10, "high_when": "above"},
    }
    cutoff = 31
    strong = regime.strong_market_mask(score, protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    poisoned = score.copy()
    poisoned[cutoff + 1 :] = 1.0 - poisoned[cutoff + 1 :]
    poisoned_strong = regime.strong_market_mask(poisoned, protocol)
    poisoned_weak2 = confirmed.confirmed_weak_mask(poisoned_strong, 2)
    np.testing.assert_array_equal(strong[: cutoff + 1], poisoned_strong[: cutoff + 1])
    np.testing.assert_array_equal(weak2[: cutoff + 1], poisoned_weak2[: cutoff + 1])


def test_future_prices_cannot_change_prior_volatility_exit_priority() -> None:
    rng = np.random.default_rng(11)
    close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (70, 15)), axis=0))
    score = rng.random((70, 15))
    confirmed_weak = rng.random(70) > 0.5
    cutoff = 44
    volatility = defensive.trailing_log_volatility(close, 20)
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    sell_priority = priority.sell_priority_matrix(
        score, volatility_rank, ~confirmed_weak, 0.05
    )
    poisoned = close.copy()
    poisoned[cutoff + 1 :] *= np.exp(rng.normal(0, 2.0, poisoned[cutoff + 1 :].shape))
    poisoned_volatility = defensive.trailing_log_volatility(poisoned, 20)
    poisoned_rank = percentile.cross_sectional_percent_rank(poisoned_volatility)
    poisoned_priority = priority.sell_priority_matrix(
        score, poisoned_rank, ~confirmed_weak, 0.05
    )
    np.testing.assert_array_equal(
        volatility[: cutoff + 1], poisoned_volatility[: cutoff + 1]
    )
    np.testing.assert_array_equal(
        sell_priority[: cutoff + 1], poisoned_priority[: cutoff + 1]
    )
