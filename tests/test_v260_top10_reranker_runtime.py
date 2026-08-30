from __future__ import annotations

import numpy as np
import pandas as pd

from v260_top10_reranker_runtime import (
    admitted_dates,
    complete_top10_training_dates,
    predict_monthly_margins,
    strict_baseline_order,
)


PARAMS = {
    "objective": "rank:pairwise",
    "n_estimators": 2,
    "learning_rate": 0.1,
    "max_depth": 1,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": 0,
}


def fixture_frame() -> pd.DataFrame:
    rows = []
    for day_index, date in enumerate(["20240102", "20240103", "20240201"]):
        for stock_index in range(10):
            rows.append(
                {
                    "trade_date": date,
                    "stock_code": f"{stock_index:06d}.SZ",
                    "baseline_oof_score_current": float(10 - stock_index),
                    "target": float(stock_index - 5 + day_index),
                    "f1": float(stock_index + day_index),
                }
            )
    return pd.DataFrame(rows)


def test_admitted_dates_excludes_immature_and_current_rows() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103", "20240104"],
            "stock_code": ["000001.SZ"] * 3,
        }
    )
    end_map = {"20240102": "20240103", "20240103": "20240105", "20240104": "20240106"}
    assert admitted_dates(frame, end_map, "20240105") == ["20240102"]


def test_strict_order_is_score_then_stock_code_and_rejects_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102"] * 3,
            "stock_code": ["000003.SZ", "000002.SZ", "000001.SZ"],
            "baseline_oof_score_current": [1.0, 2.0, 2.0],
        }
    )
    result = strict_baseline_order(frame)
    assert result["stock_code"].tolist() == ["000001.SZ", "000002.SZ", "000003.SZ"]
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    try:
        strict_baseline_order(duplicate)
    except RuntimeError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate key was accepted")


def test_prediction_target_is_not_part_of_ordering_schema() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240102"],
            "stock_code": ["000001.SZ", "000002.SZ"],
            "baseline_oof_score_current": [0.2, 0.1],
            "target": [np.nan, 999.0],
        }
    )
    assert strict_baseline_order(frame)["stock_code"].tolist() == ["000001.SZ", "000002.SZ"]


def test_training_date_requires_complete_finite_production_top10() -> None:
    frame = fixture_frame()
    frame.loc[
        frame["trade_date"].eq("20240103") & frame["stock_code"].eq("000000.SZ"), "target"
    ] = np.nan
    complete, incomplete = complete_top10_training_dates(
        frame, ["20240102", "20240103", "20240201"]
    )
    assert complete == ["20240102", "20240201"]
    assert incomplete == ["20240103"]


def test_incomplete_training_date_is_skipped_without_partial_group() -> None:
    frame = fixture_frame()
    frame.loc[
        frame["trade_date"].eq("20240103") & frame["stock_code"].eq("000000.SZ"), "target"
    ] = np.nan
    end_map = {"20240102": "20240110", "20240103": "20240111", "20240201": "20240220"}
    _, models = predict_monthly_margins(
        frame,
        features=["f1"],
        params=PARAMS,
        label_end_by_date=end_map,
        prediction_start="20240201",
        prediction_end="20240201",
        window_signal_dates=1,
    )
    assert models[0]["train_date_count"] == 1
    assert models[0]["excluded_incomplete_top10_date_count"] == 1


def test_future_and_immature_target_poison_cannot_change_prediction() -> None:
    frame = fixture_frame()
    end_map = {"20240102": "20240110", "20240103": "20240210", "20240201": "20240220"}
    expected, _ = predict_monthly_margins(
        frame,
        features=["f1"],
        params=PARAMS,
        label_end_by_date=end_map,
        prediction_start="20240201",
        prediction_end="20240201",
        window_signal_dates=1,
    )
    poisoned = frame.copy()
    poisoned.loc[poisoned["trade_date"].isin(["20240103", "20240201"]), "target"] = 1e9
    actual, _ = predict_monthly_margins(
        poisoned,
        features=["f1"],
        params=PARAMS,
        label_end_by_date=end_map,
        prediction_start="20240201",
        prediction_end="20240201",
        window_signal_dates=1,
    )
    assert np.array_equal(expected["reranker_margin"].to_numpy(), actual["reranker_margin"].to_numpy())


def test_cold_start_is_exact_baseline_identity() -> None:
    frame = fixture_frame()
    result, _ = predict_monthly_margins(
        frame,
        features=["f1"],
        params=PARAMS,
        label_end_by_date={"20240102": "20240110", "20240103": "20240120"},
        prediction_start="20240201",
        prediction_end="20240201",
        window_signal_dates=3,
    )
    expected = strict_baseline_order(frame.loc[frame["trade_date"].eq("20240201")])
    merged = expected.merge(result, on=["trade_date", "stock_code"], validate="one_to_one")
    assert (merged["reranker_active"] == 0).all()
    assert np.array_equal(
        merged["baseline_oof_score_current"].to_numpy(dtype=float),
        merged["reranker_margin"].to_numpy(dtype=float),
    )
