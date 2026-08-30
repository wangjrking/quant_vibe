from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_drawdown_position_attribution_20260822 as positions
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_position_correlation_diagnostic_20260822"
)
LOOKBACK = 60
MIN_OBSERVATIONS = 40


def returns_from_close(close_qfq: np.ndarray) -> np.ndarray:
    close = np.asarray(close_qfq, dtype=np.float64)
    returns = np.full(close.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(close[1:])
        & np.isfinite(close[:-1])
        & (close[1:] > 0)
        & (close[:-1] > 0)
    )
    adjacent = np.full_like(close[1:], np.nan)
    adjacent[valid] = close[1:][valid] / close[:-1][valid] - 1.0
    returns[1:] = adjacent
    return returns


def pairwise_correlations(
    returns: np.ndarray,
    end_index: int,
    stock_indices: list[int],
    lookback: int,
    min_observations: int,
) -> list[float]:
    start = max(0, int(end_index) - int(lookback) + 1)
    window = returns[start : int(end_index) + 1]
    values = []
    for left_pos, left in enumerate(stock_indices):
        for right in stock_indices[left_pos + 1 :]:
            pair = window[:, [left, right]]
            valid = np.isfinite(pair).all(axis=1)
            if int(valid.sum()) < int(min_observations):
                continue
            left_values, right_values = pair[valid, 0], pair[valid, 1]
            if np.std(left_values) <= 0 or np.std(right_values) <= 0:
                continue
            values.append(float(np.corrcoef(left_values, right_values)[0, 1]))
    return values


def summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("empty correlation diagnostic")
    return {
        "days": int(len(frame)),
        "mean_pairwise_correlation": float(frame["mean_pairwise_correlation"].mean()),
        "median_pairwise_correlation": float(
            frame["mean_pairwise_correlation"].median()
        ),
        "p95_mean_pairwise_correlation": float(
            frame["mean_pairwise_correlation"].quantile(0.95)
        ),
        "mean_max_pairwise_correlation": float(
            frame["max_pairwise_correlation"].mean()
        ),
        "maximum_pairwise_correlation": float(frame["max_pairwise_correlation"].max()),
        "days_mean_correlation_above_050": int(
            (frame["mean_pairwise_correlation"] > 0.50).sum()
        ),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    daily, _, observations, context = positions.run_with_observer()
    _, _, _, _, arrays, access = round1.load_arrays(round1.DEVELOPMENT_END)
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered correlation diagnostic")
    stock_index = {str(code): idx for idx, code in enumerate(arrays["stocks"])}
    date_index = {str(date): idx for idx, date in enumerate(arrays["dates"])}
    returns = returns_from_close(arrays["close_qfq"])
    rows = []
    for observation in observations:
        day = date_index[observation["buy_date"]]
        indices = [
            stock_index[code] for code in observation["positions_after_trades"]
        ]
        correlations = pairwise_correlations(
            returns, day, indices, LOOKBACK, MIN_OBSERVATIONS
        )
        if not correlations:
            continue
        rows.append(
            {
                "buy_date": observation["buy_date"],
                "pair_count": len(correlations),
                "mean_pairwise_correlation": float(np.mean(correlations)),
                "max_pairwise_correlation": float(np.max(correlations)),
            }
        )
    frame = pd.DataFrame(rows)
    episode = attribution.maximum_drawdown_episode(daily)
    drawdown = frame[
        (frame["buy_date"] >= episode["peak_date"])
        & (frame["buy_date"] <= episode["trough_date"])
    ].reset_index(drop=True)
    result = {
        "status": "diagnostic_complete_2026_not_opened",
        "source_strategy": context["rules"]["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "lookback_sessions": LOOKBACK,
        "minimum_pair_observations": MIN_OBSERVATIONS,
        "all_period": summarize(frame),
        "maximum_drawdown_episode": episode,
        "maximum_drawdown_correlation": summarize(drawdown),
        "highest_correlation_days": frame.sort_values(
            ["mean_pairwise_correlation", "buy_date"], ascending=[False, True]
        )
        .head(20)
        .to_dict("records"),
        "data_access": access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "position_correlation_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
