import pandas as pd
import pytest

from quant.main.research_v260_fixed10_pressure_exit_attribution_20260822 import (
    action_keys,
    period_log_contribution,
)


def test_period_log_contribution_groups_calendar_halves():
    frame = pd.DataFrame(
        {
            "date": ["20230103", "20230703"],
            "log_excess": [0.1, 0.2],
        }
    )
    assert period_log_contribution(frame, "half") == {
        "2023H1": pytest.approx(0.1),
        "2023H2": pytest.approx(0.2),
    }


def test_action_keys_are_exact_date_action_code_tuples():
    actions = pd.DataFrame(
        {
            "buy_date": ["20230103", "20230104"],
            "action": ["SELL", "BUY"],
            "stock_code": ["000001.SZ", "000002.SZ"],
        }
    )
    assert action_keys(actions, {"20230103"}) == {
        ("20230103", "SELL", "000001.SZ")
    }
