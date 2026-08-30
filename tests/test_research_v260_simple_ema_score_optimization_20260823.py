import sys
from pathlib import Path

import numpy as np


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from research_v260_simple_ema_score_optimization_20260823 import (
    causal_ema_score,
    selection_decision,
    stable_order,
)


def test_causal_ema_has_no_future_dependency() -> None:
    raw = np.array([[0.2], [0.6], [0.9]], dtype=np.float32)
    poisoned = raw.copy()
    poisoned[2, 0] = -999.0
    assert np.array_equal(
        causal_ema_score(raw)[:2], causal_ema_score(poisoned)[:2]
    )


def test_missing_current_score_is_not_carried_into_output() -> None:
    raw = np.array([[0.2], [np.nan], [0.6]], dtype=np.float32)
    actual = causal_ema_score(raw)
    assert np.isnan(actual[1, 0])
    assert np.isclose(actual[2, 0], 0.3)


def test_stable_order_uses_original_column_order_for_ties() -> None:
    score = np.array([[0.8, 0.8, 0.7]], dtype=np.float32)
    assert stable_order(score).tolist() == [[0, 1, 2]]


def test_selection_requires_profit_and_robustness_together() -> None:
    def item(cumulative, sharpe=1.0, drawdown=0.2):
        return {
            "cumulative_return": cumulative,
            "cagr": cumulative,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "annual_returns": {"2022": 0.1, "2023": 0.1},
        }

    baseline = {"0.003": item(1.0), "0.0065": item(0.5)}
    candidate = {"0.003": item(1.1), "0.0065": item(0.6)}
    assert selection_decision(baseline, candidate)[
        "selected_for_future_validation"
    ]
    candidate["0.0065"] = item(0.4)
    assert not selection_decision(baseline, candidate)[
        "selected_for_future_validation"
    ]
