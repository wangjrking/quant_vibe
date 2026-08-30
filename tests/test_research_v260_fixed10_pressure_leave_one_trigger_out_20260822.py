import sys
from pathlib import Path

import numpy as np
import pytest


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_pressure_leave_one_trigger_out_20260822 import (
    concentration_summary,
    leave_one_out_schedules,
)


def test_leave_one_out_schedules_suppress_only_named_date():
    schedules = leave_one_out_schedules(
        np.array(["20240102", "20240103", "20240104"]),
        ["20240102", "20240104"],
    )
    assert schedules["20240102"].tolist() == [1, 2, 2]
    assert schedules["20240104"].tolist() == [2, 2, 1]


def test_leave_one_out_rejects_unknown_or_duplicate_dates():
    dates = np.array(["20240102"])
    with pytest.raises(ValueError, match="missing"):
        leave_one_out_schedules(dates, ["20240103"])
    with pytest.raises(ValueError, match="unique"):
        leave_one_out_schedules(dates, ["20240102", "20240102"])


def test_concentration_summary_reports_largest_positive_share():
    full = {"cumulative_return": 3.0}
    variants = {
        "20240102": {"cumulative_return": 2.8},
        "20240103": {"cumulative_return": 2.9},
        "20240104": {"cumulative_return": 3.1},
    }
    summary = concentration_summary(full, variants)
    assert summary["beneficial_trigger_count"] == 2
    assert summary["harmful_trigger_count"] == 1
    assert summary["largest_positive_effect_share"] == pytest.approx(2.0 / 3.0)
