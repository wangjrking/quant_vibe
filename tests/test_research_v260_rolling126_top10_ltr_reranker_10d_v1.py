from __future__ import annotations

import research_v260_rolling126_top10_ltr_reranker_10d_v1_20260817 as candidate


def test_single_fixed_window_contract() -> None:
    assert candidate.core.WINDOW == 126
    assert "rolling126" in candidate.core.CONTRACT.as_posix()


def test_reuses_audited_rolling_reranker_core() -> None:
    assert callable(candidate.core.admitted_dates)
    assert callable(candidate.core.build_replay)
