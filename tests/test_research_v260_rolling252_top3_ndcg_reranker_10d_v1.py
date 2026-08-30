from __future__ import annotations

import numpy as np

from research_v260_rolling252_top3_ndcg_reranker_10d_v1_20260817 import deterministic_relevance


def test_relevance_is_zero_to_nine_per_group() -> None:
    y = np.asarray([0.2, -0.1, 0.3, 0.0, 0.1, 0.5, 0.4, -0.2, 0.6, 0.7] * 2)
    qid = np.asarray([0] * 10 + [1] * 10)
    result = deterministic_relevance(y, qid)
    assert set(result[:10]) == set(range(10))
    assert set(result[10:]) == set(range(10))
    assert result[9] == 9


def test_equal_targets_keep_input_order() -> None:
    y = np.ones(10)
    qid = np.zeros(10, dtype=np.int32)
    assert np.array_equal(deterministic_relevance(y, qid), np.arange(10))
