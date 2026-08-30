from __future__ import annotations

import numpy as np
import pandas as pd

from research_v260_top10_continuous_ltr_reranker_10d_v1_20260817 import (
    apply_rerank,
    top10_in_model_order,
)


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20220104"] * 12,
            "stock_code": [f"{index:06d}.SZ" for index in range(12)],
            "baseline_oof_score_current": np.arange(12, 0, -1, dtype=float),
            "target": np.linspace(-0.1, 0.1, 12),
        }
    )


def test_identity_warmup_preserves_order() -> None:
    output = apply_rerank(fixture(), None).sort_values("baseline_rank")
    assert output["final_rank"].tolist() == list(range(1, 13))


def test_rerank_changes_order_but_not_top10_membership() -> None:
    source = fixture()
    output = apply_rerank(source, np.arange(10, dtype=float))
    assert set(output.nsmallest(10, "baseline_rank")["stock_code"]) == set(output.nsmallest(10, "final_rank")["stock_code"])
    assert output.loc[output["baseline_rank"] > 10, "rank_changed"].sum() == 0
    assert output["top1_changed"].iloc[0] == 1


def test_prediction_count_mismatch_fails_closed() -> None:
    try:
        apply_rerank(fixture(), np.arange(9, dtype=float))
    except (RuntimeError, ValueError):
        return
    raise AssertionError("prediction count mismatch was accepted")


def test_model_order_matches_projection_decode_order_for_shuffled_input() -> None:
    source = fixture().sample(frac=1.0, random_state=7)
    ordered = top10_in_model_order(source)
    assert ordered["stock_code"].tolist() == [f"{index:06d}.SZ" for index in range(10)]
    margins = np.arange(10, 0, -1, dtype=float)
    output = apply_rerank(source, margins)
    assert output.nsmallest(10, "final_rank")["stock_code"].tolist() == [f"{index:06d}.SZ" for index in range(10)]
