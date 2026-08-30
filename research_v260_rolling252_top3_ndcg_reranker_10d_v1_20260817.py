"""Build one fixed rolling Top3-aligned NDCG reranker."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import research_v260_rolling252_top10_ltr_reranker_10d_v1_20260817 as core


CONTRACT = Path(
    "quant/data_file/runtime/agent_workspaces/strategy-agent/work/"
    "v260_rolling252_top3_ndcg_reranker_20260817/training_contract.json"
)
OUT = Path(
    "quant/data_file/reports/model_agent_v260_rolling252_top3_ndcg_reranker_10d_v1_20260817_r1"
)


def deterministic_relevance(y: np.ndarray, qid: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "position": np.arange(len(y), dtype=np.int64),
            "target": np.asarray(y, dtype=np.float64),
            "qid": np.asarray(qid, dtype=np.int32),
        }
    )
    frame = frame.sort_values(["qid", "target", "position"], kind="mergesort")
    frame["relevance"] = frame.groupby("qid", sort=False).cumcount().astype(np.int32)
    result = frame.sort_values("position", kind="mergesort")["relevance"].to_numpy(dtype=np.int32)
    group_sizes = pd.Series(qid).value_counts(sort=False).to_numpy()
    if len(result) != len(y) or np.any(group_sizes != 10) or result.min() != 0 or result.max() != 9:
        raise RuntimeError("invalid deterministic relevance groups")
    return result


class Top3NDCGRanker(xgb.XGBRanker):
    def fit(self, X, y, *, qid, **kwargs):
        relevance = deterministic_relevance(np.asarray(y), np.asarray(qid))
        return super().fit(X, relevance, qid=qid, **kwargs)


core.CONTRACT = CONTRACT
core.OUT = OUT
core.WINDOW = 252
core.xgb.XGBRanker = Top3NDCGRanker


if __name__ == "__main__":
    raise SystemExit(core.main())
