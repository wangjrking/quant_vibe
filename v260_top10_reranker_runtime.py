"""Single runtime core for the frozen monthly rolling production-Top10 reranker."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


KEYS = ["trade_date", "stock_code"]


def model_hash(model: xgb.XGBRanker) -> str:
    return hashlib.sha256(model.get_booster().save_raw(raw_format="json")).hexdigest()


def strict_baseline_order(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*KEYS, "baseline_oof_score_current"}
    if not required.issubset(frame.columns):
        raise RuntimeError("reranker runtime schema mismatch")
    if frame.duplicated(KEYS).any():
        raise RuntimeError("duplicate reranker runtime key")
    ordered = frame.copy()
    ordered["trade_date"] = ordered["trade_date"].astype(str)
    ordered["stock_code"] = ordered["stock_code"].astype(str)
    ordered = ordered.sort_values(
        ["trade_date", "baseline_oof_score_current", "stock_code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ordered["baseline_rank"] = ordered.groupby("trade_date", sort=False).cumcount() + 1
    return ordered


def top10_in_model_order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = strict_baseline_order(frame)
    return ordered.loc[ordered["baseline_rank"] <= 10].sort_values(
        ["trade_date", "baseline_rank"], kind="mergesort"
    )


def admitted_dates(frame: pd.DataFrame, label_end_by_date: dict[str, str], prediction_start: str) -> list[str]:
    dates = sorted(frame["trade_date"].astype(str).unique())
    return [
        date
        for date in dates
        if date < prediction_start and label_end_by_date.get(date, "99999999") < prediction_start
    ]


def complete_top10_training_dates(frame: pd.DataFrame, dates: list[str]) -> tuple[list[str], list[str]]:
    """Keep mature dates whose frozen production Top10 has ten finite targets."""
    normalized_dates = [str(date) for date in dates]
    if not normalized_dates:
        return [], []
    selected = frame.loc[frame["trade_date"].astype(str).isin(normalized_dates)].copy()
    top10 = top10_in_model_order(selected)
    top10["_finite_target"] = np.isfinite(
        pd.to_numeric(top10["target"], errors="coerce").to_numpy(dtype="float64")
    )
    daily = top10.groupby("trade_date", sort=False).agg(
        row_count=("stock_code", "size"),
        finite_target_count=("_finite_target", "sum"),
    )
    complete = [
        date
        for date in normalized_dates
        if date in daily.index
        and int(daily.at[date, "row_count"]) == 10
        and int(daily.at[date, "finite_target_count"]) == 10
    ]
    complete_set = set(complete)
    incomplete = [date for date in normalized_dates if date not in complete_set]
    return complete, incomplete


def predict_monthly_margins(
    frame: pd.DataFrame,
    *,
    features: list[str],
    params: dict[str, Any],
    label_end_by_date: dict[str, str],
    prediction_start: str,
    prediction_end: str,
    window_signal_dates: int = 252,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required = {*KEYS, "target", "baseline_oof_score_current", *features}
    if not required.issubset(frame.columns):
        raise RuntimeError("reranker feature/target schema mismatch")
    if len(features) != len(set(features)) or not features:
        raise RuntimeError("invalid reranker feature contract")
    work = frame.copy()
    work["trade_date"] = work["trade_date"].astype(str)
    work["stock_code"] = work["stock_code"].astype(str)
    if work.duplicated(KEYS).any():
        raise RuntimeError("duplicate reranker input key")
    prediction = work.loc[
        work["trade_date"].between(str(prediction_start), str(prediction_end), inclusive="both")
    ].copy()
    if prediction.empty:
        raise RuntimeError("empty reranker prediction domain")
    prediction["month"] = prediction["trade_date"].str[:6]
    outputs = []
    models = []
    for month in sorted(prediction["month"].unique()):
        test = prediction.loc[prediction["month"] == month].drop(columns="month").copy()
        month_start = str(test["trade_date"].min())
        mature_dates = admitted_dates(work, label_end_by_date, month_start)
        dates, incomplete_dates = complete_top10_training_dates(work, mature_dates)
        if len(dates) < int(window_signal_dates):
            margin_frame = test[KEYS].copy()
            margin_frame["reranker_margin"] = test["baseline_oof_score_current"].astype("float64")
            margin_frame["reranker_active"] = np.int8(0)
            model_sha256 = "IDENTITY_COLD_START"
            train_date_count = len(dates)
        else:
            selected_dates = dates[-int(window_signal_dates) :]
            train = top10_in_model_order(work.loc[work["trade_date"].isin(selected_dates)].copy())
            if not np.isfinite(train["target"].to_numpy(dtype="float64")).all():
                raise RuntimeError("nonfinite mature training target")
            test_ordered = strict_baseline_order(test).sort_values(
                ["trade_date", "baseline_rank"], kind="mergesort"
            )
            qid = pd.factorize(train["trade_date"], sort=True)[0].astype("int32")
            model = xgb.XGBRanker(**params)
            model.fit(
                train[features].astype("float32"),
                train["target"].astype("float32"),
                qid=qid,
                base_margin=train["baseline_oof_score_current"].astype("float32"),
                verbose=False,
            )
            margins = model.predict(
                test_ordered[features].astype("float32"),
                output_margin=True,
                base_margin=test_ordered["baseline_oof_score_current"].astype("float32"),
            ).astype("float64")
            margin_frame = test_ordered[KEYS].copy()
            margin_frame["reranker_margin"] = margins
            margin_frame["reranker_active"] = np.int8(1)
            model_sha256 = model_hash(model)
            train_date_count = len(selected_dates)
        margin_frame["train_date_count"] = int(train_date_count)
        outputs.append(margin_frame)
        models.append(
            {
                "month": str(month),
                "prediction_start": month_start,
                "train_date_count": int(train_date_count),
                "excluded_incomplete_top10_date_count": len(incomplete_dates),
                "model_sha256": model_sha256,
            }
        )
    result = pd.concat(outputs, ignore_index=True).sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    if len(result) != len(prediction) or result.duplicated(KEYS).any():
        raise RuntimeError("reranker output key-domain mismatch")
    return result, models
