from __future__ import annotations

import numpy as np
import pandas as pd

from research_v260_cost_aware_top10_pairwise_residual_10d_v1_20260817 import (
    make_adjacent_pairs,
    project_top10,
)


FEATURES = ["feature"]


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20220104"] * 12,
            "stock_code": [f"{index:06d}.SZ" for index in range(12)],
            "feature": np.arange(12, dtype=float),
            "target": np.asarray([0.05, 0.04, 0.03, 0.02, 0.01, 0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06]),
            "baseline_state_score": np.arange(12, 0, -1, dtype=float),
            "baseline_oof_score_current": np.arange(12, 0, -1, dtype=float),
        }
    )


def test_pair_rows_are_symmetrized_and_ambiguous_pairs_are_dropped() -> None:
    pairs, summary = make_adjacent_pairs(frame(), FEATURES, hurdle=0.003)
    assert summary["symmetrized_training_rows"] == 18
    assert pairs["label"].value_counts().to_dict() == {0: 9, 1: 9}


def test_low_confidence_is_identity() -> None:
    source = frame()
    output = project_top10(source, FEATURES, lambda matrix: np.full(len(matrix), 0.5), 0.75)
    assert output.sort_values("baseline_rank")["final_rank"].tolist() == list(range(1, 13))


def test_high_confidence_changes_only_top10_and_preserves_membership() -> None:
    source = frame()
    output = project_top10(source, FEATURES, lambda matrix: np.full(len(matrix), 0.9), 0.75)
    baseline_top10 = set(output.nsmallest(10, "baseline_rank")["stock_code"])
    candidate_top10 = set(output.nsmallest(10, "final_rank")["stock_code"])
    assert baseline_top10 == candidate_top10
    assert output.loc[output["baseline_rank"] > 10, "rank_changed"].sum() == 0
    assert (output["final_rank"] - output["baseline_rank"]).abs().max() <= 2


def test_projection_is_deterministic() -> None:
    source = frame()
    predictor = lambda matrix: (matrix[:, 0] > 0).astype(float)
    first = project_top10(source, FEATURES, predictor, 0.75)
    second = project_top10(source, FEATURES, predictor, 0.75)
    assert first[["stock_code", "final_rank"]].equals(second[["stock_code", "final_rank"]])
