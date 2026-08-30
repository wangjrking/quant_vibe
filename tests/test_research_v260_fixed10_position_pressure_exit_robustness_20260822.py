import sys
from pathlib import Path


MAIN = Path(__file__).resolve().parents[1]
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from research_v260_fixed10_position_pressure_exit_robustness_20260822 import (
    summarize_pressure,
)


def test_pressure_summary_counts_exact_trigger_days_and_years():
    records = [
        {"signal_date": "20231229", "score_sell_candidate_count": 4},
        {"signal_date": "20240102", "score_sell_candidate_count": 5},
        {"signal_date": "20250102", "score_sell_candidate_count": 7},
    ]
    summary = summarize_pressure(records, 5)
    assert summary["triggered_days"] == 2
    assert summary["triggered_years"] == ["2024", "2025"]
    assert summary["maximum_simultaneous_score_sell_candidates"] == 7


def test_baseline_has_no_pressure_trigger_records():
    records = [{"signal_date": "20240102", "score_sell_candidate_count": 9}]
    summary = summarize_pressure(records, None)
    assert summary["triggered_days"] == 0
    assert summary["records"] == []
