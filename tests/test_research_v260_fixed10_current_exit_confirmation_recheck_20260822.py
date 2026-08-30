from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_current_exit_confirmation_recheck_20260822 as subject


def test_candidate_budget_is_one_bounded_change():
    base = {
        "sell_confirmation_days": 1,
        "sell_score_below": 0.85,
        "replacement_advantage": 0.05,
    }
    policies = subject.build_policies(base)
    assert set(policies) == {subject.BASELINE_ID, subject.CANDIDATE_ID}
    assert policies[subject.BASELINE_ID]["sell_confirmation_days"] == 1
    assert policies[subject.CANDIDATE_ID]["sell_confirmation_days"] == 2
    for key in ("sell_score_below", "replacement_advantage"):
        assert policies[subject.BASELINE_ID][key] == policies[subject.CANDIDATE_ID][key]


def test_selection_requires_every_gate():
    baseline = {
        "cost_metrics": {
            "0.0030": {
                "cumulative_return": 1.0,
                "sharpe": 1.0,
                "max_drawdown": 0.2,
                "turnover_annualized": 10.0,
            },
            "0.0065": {"cumulative_return": 0.5},
        },
        "train_2022_2024": {"cagr": 0.2, "sharpe": 1.0},
        "holdout_2025": {"cumulative_return": 0.2},
    }
    candidate = {
        "cost_metrics": {
            "0.0030": {
                "cumulative_return": 1.1,
                "sharpe": 1.1,
                "max_drawdown": 0.19,
                "turnover_annualized": 9.0,
                "annual_returns": {"2022": 0.1},
                "full_10_position_ratio": 1.0,
            },
            "0.0065": {"cumulative_return": 0.6},
        },
        "train_2022_2024": {"cagr": 0.21, "sharpe": 1.01},
        "holdout_2025": {"cumulative_return": 0.21},
        "start_offset_metrics": {"0": {"cumulative_return": 1.1}},
    }
    gates = subject.selection_gates(candidate, baseline)
    assert all(gates.values())
    candidate["holdout_2025"]["cumulative_return"] = 0.19
    gates = subject.selection_gates(candidate, baseline)
    assert gates["forward_2025_not_worse"] is False
    assert not all(gates.values())
