import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_exit_threshold_recheck_20260822 import (
    THRESHOLDS,
    candidate_policy,
    selection_key,
)


def test_threshold_budget_is_coarse_and_bounded():
    assert THRESHOLDS == (0.80, 0.85, 0.90)


def test_candidate_only_changes_threshold():
    base = {
        "sell_score_below": 0.85,
        "score_sell_pressure_trigger": 4,
        "score_sell_pressure_limit": 2,
    }
    actual = candidate_policy(base, 0.90)
    assert actual["sell_score_below"] == 0.90
    assert actual["score_sell_pressure_trigger"] == 4
    assert actual["score_sell_pressure_limit"] == 2
    assert base["sell_score_below"] == 0.85


def test_selection_prioritizes_training_sharpe():
    def item(sharpe, cagr):
        return {
            "train_2022_2024": {
                "sharpe": sharpe,
                "cagr": cagr,
                "max_drawdown": 0.30,
                "turnover_annualized": 10.0,
            }
        }

    assert selection_key(item(1.0, 0.2)) > selection_key(item(0.9, 0.3))
