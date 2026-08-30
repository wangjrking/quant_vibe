from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import research_v260_fixed10_rank_sizing_mechanism_attribution_20260823 as module


def test_target_bucket_is_soft_equalweight_classification() -> None:
    assert module.target_bucket(0.11) == "above_10pct"
    assert module.target_bucket(0.10) == "equal_10pct"
    assert module.target_bucket(0.09) == "below_10pct"
    with pytest.raises(ValueError):
        module.target_bucket(0.0)


def test_same_holdings_weight_effect_uses_prior_day_targets() -> None:
    actions = pd.DataFrame(
        [
            {"buy_date": "20220104", "action": "BUY", "stock_code": "A", "target_pct": 0.11},
            {"buy_date": "20220104", "action": "BUY", "stock_code": "B", "target_pct": 0.09},
            {"buy_date": "20220105", "action": "SELL", "stock_code": "A", "target_pct": 0.10},
            {"buy_date": "20220106", "action": "SELL", "stock_code": "A", "target_pct": 0.0},
        ]
    )
    records = [
        {"buy_date": "20220104", "mark_records": []},
        {
            "buy_date": "20220105",
            "mark_records": [
                {"stock_code": "A", "shares_before": 11.0, "prior_price": 10.0, "mark_pnl": 11.0},
                {"stock_code": "B", "shares_before": 9.0, "prior_price": 10.0, "mark_pnl": -9.0},
            ],
        },
        {
            "buy_date": "20220106",
            "mark_records": [
                {"stock_code": "A", "shares_before": 11.0, "prior_price": 11.0, "mark_pnl": 0.0},
                {"stock_code": "B", "shares_before": 9.0, "prior_price": 9.0, "mark_pnl": 0.0},
            ],
        },
    ]
    result = module.summarize_mechanism(records, actions)
    effect = result["same_holdings_weight_effect"]
    assert effect["days"] == 2
    assert effect["compounded_relative_weight_effect"] > 0.0
    assert result["bucket_summary"]["above_10pct"]["gross_mark_pnl"] == 11.0
    assert result["bucket_summary"]["below_10pct"]["gross_mark_pnl"] == -9.0
    assert result["bucket_summary"]["above_10pct"]["position_day_observations"] == 2
    assert result["bucket_summary"]["equal_10pct"]["position_day_observations"] == 0


def test_unknown_held_position_fails_attribution() -> None:
    actions = pd.DataFrame(columns=["buy_date", "action", "stock_code", "target_pct"])
    records = [
        {
            "buy_date": "20220105",
            "mark_records": [
                {"stock_code": "A", "shares_before": 1.0, "prior_price": 10.0, "mark_pnl": 1.0}
            ],
        }
    ]
    with pytest.raises(RuntimeError, match="no prior BUY target"):
        module.summarize_mechanism(records, actions)


def test_target_tolerance_is_deterministic() -> None:
    assert module.target_bucket(0.1 + np.finfo(float).eps) == "equal_10pct"


def test_paired_path_attribution_reconciles_components() -> None:
    candidate = [
        {
            "buy_date": "20220105",
            "previous_equity": 100.0,
            "equity_after_trades": 103.0,
            "mark_records": [
                {"stock_code": "A", "mark_pnl": 2.0},
                {"stock_code": "B", "mark_pnl": 2.0},
            ],
        }
    ]
    equal = [
        {
            "buy_date": "20220105",
            "previous_equity": 100.0,
            "equity_after_trades": 101.0,
            "mark_records": [
                {"stock_code": "A", "mark_pnl": 1.0},
                {"stock_code": "C", "mark_pnl": 1.0},
            ],
        }
    ]
    result = module.paired_path_attribution(candidate, equal)
    assert result["arithmetic_daily_return_delta_sum"] == pytest.approx(0.02)
    assert result["component_sum"] == pytest.approx(0.02)
    assert result["components"]["common_holdings_mark_delta"] == pytest.approx(0.01)
    assert result["components"]["candidate_only_holdings_mark_contribution"] == pytest.approx(0.02)
    assert result["components"]["minus_equalweight_only_holdings_mark_contribution"] == pytest.approx(-0.01)
