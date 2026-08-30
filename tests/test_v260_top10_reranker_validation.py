from __future__ import annotations

import numpy as np
import pandas as pd

from v260_top10_reranker_validation import evaluate_validation


def daily(mean: float, *, turnover: float = 0.1, days: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days).strftime("%Y%m%d")
    returns = np.full(days, mean, dtype=float)
    returns[::10] -= 0.0002
    return pd.DataFrame({"date": dates, "return": returns, "turnover": turnover})


def cases(candidate_mean: float, stress_candidate_mean: float | None = None) -> dict:
    stress_mean = candidate_mean if stress_candidate_mean is None else stress_candidate_mean
    return {
        "0.0030": {"baseline": daily(0.0010), "candidate": daily(candidate_mean)},
        "0.0065": {"baseline": daily(0.0005), "candidate": daily(stress_mean)},
    }


def test_strictly_better_candidate_passes() -> None:
    result = evaluate_validation(cases(0.0012, 0.0007), deterministic_replay=True)
    assert result["decision"] == "validation_passed_candidate_for_production_review"
    assert all(result["gates"].values())


def test_stress_underperformance_rejects() -> None:
    result = evaluate_validation(cases(0.0012, 0.0003), deterministic_replay=True)
    assert result["decision"] == "validation_rejected"
    assert not result["gates"]["stress_return_not_lower"]


def test_date_domain_mismatch_fails_closed() -> None:
    payload = cases(0.0012)
    payload["0.0030"]["candidate"] = payload["0.0030"]["candidate"].iloc[1:].copy()
    try:
        evaluate_validation(payload, deterministic_replay=True)
    except RuntimeError as exc:
        assert "date domain" in str(exc)
    else:
        raise AssertionError("date mismatch was accepted")


def test_nondeterministic_replay_rejects() -> None:
    result = evaluate_validation(cases(0.0012, 0.0007), deterministic_replay=False)
    assert result["decision"] == "validation_rejected"
    assert not result["gates"]["deterministic_replay"]
