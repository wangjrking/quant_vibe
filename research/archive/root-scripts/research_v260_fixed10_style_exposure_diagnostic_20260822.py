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
    / "quant/data_file/reports/strategy_agent_v260_fixed10_style_exposure_diagnostic_20260822"
)
STYLE_FIELDS = ("total_mv", "turnover_rate", "amount")


def percentile_for_indices(
    values: np.ndarray,
    universe_mask: np.ndarray,
    indices: list[int],
) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(universe_mask, dtype=np.bool_) & np.isfinite(data)
    reference = np.sort(data[mask], kind="mergesort")
    if reference.size < 2:
        raise ValueError("style reference universe is too small")
    result = []
    for index in indices:
        value = float(data[index])
        if not np.isfinite(value):
            raise ValueError("held position has missing style value")
        left = int(np.searchsorted(reference, value, side="left"))
        right = int(np.searchsorted(reference, value, side="right"))
        percentile = (left + right - 1) / (2 * (len(reference) - 1))
        result.append(float(np.clip(percentile, 0.0, 1.0)))
    return result


def summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("empty style exposure frame")
    result = {"days": int(len(frame))}
    for field in STYLE_FIELDS:
        column = f"mean_{field}_percentile"
        values = frame[column].astype(float)
        result[field] = {
            "mean_percentile": float(values.mean()),
            "median_percentile": float(values.median()),
            "p10_percentile": float(values.quantile(0.10)),
            "p90_percentile": float(values.quantile(0.90)),
            "days_below_025": int((values < 0.25).sum()),
            "days_above_075": int((values > 0.75).sum()),
        }
    return result


def build_daily_exposure(
    observations: list[dict],
    arrays: dict,
) -> pd.DataFrame:
    stock_index = {str(code): idx for idx, code in enumerate(arrays["stocks"])}
    date_index = {str(date): idx for idx, date in enumerate(arrays["dates"])}
    rows = []
    for observation in observations:
        signal_date = str(observation["signal_date"])
        day = date_index[signal_date]
        indices = [
            stock_index[str(code)] for code in observation["positions_after_trades"]
        ]
        if not indices:
            continue
        universe = np.asarray(arrays["signal_clean"][day], dtype=np.bool_)
        row = {
            "signal_date": signal_date,
            "buy_date": str(observation["buy_date"]),
            "position_count": len(indices),
            "universe_count": int(universe.sum()),
        }
        for field in STYLE_FIELDS:
            finite_indices = [
                index
                for index in indices
                if np.isfinite(float(arrays[field][day, index]))
            ]
            row[f"observed_{field}_positions"] = len(finite_indices)
            row[f"missing_{field}_positions"] = len(indices) - len(finite_indices)
            if not finite_indices:
                row[f"mean_{field}_percentile"] = np.nan
                row[f"minimum_{field}_percentile"] = np.nan
                row[f"maximum_{field}_percentile"] = np.nan
                continue
            percentiles = percentile_for_indices(
                arrays[field][day], universe, finite_indices
            )
            row[f"mean_{field}_percentile"] = float(np.mean(percentiles))
            row[f"minimum_{field}_percentile"] = float(np.min(percentiles))
            row[f"maximum_{field}_percentile"] = float(np.max(percentiles))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    daily, _, observations, context = positions.run_with_observer()
    _, _, _, _, arrays, access = round1.load_arrays(round1.DEVELOPMENT_END)
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered style exposure diagnostic")
    exposure = build_daily_exposure(observations, arrays)
    episode = attribution.maximum_drawdown_episode(daily)
    drawdown = exposure[
        (exposure["buy_date"] >= episode["peak_date"])
        & (exposure["buy_date"] <= episode["trough_date"])
    ].reset_index(drop=True)
    all_period = summarize(exposure)
    drawdown_period = summarize(drawdown)
    annual = {
        str(year): summarize(group.reset_index(drop=True))
        for year, group in exposure.groupby(exposure["buy_date"].str[:4], sort=True)
    }
    deltas = {
        field: float(
            drawdown_period[field]["mean_percentile"]
            - all_period[field]["mean_percentile"]
        )
        for field in STYLE_FIELDS
    }
    result = {
        "status": "diagnostic_complete_2026_not_opened",
        "source_strategy": context["rules"]["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "percentile_reference": (
            "same-signal-date signal_clean universe; ties use average percentile"
        ),
        "all_period": all_period,
        "maximum_drawdown_episode": episode,
        "maximum_drawdown_style_exposure": drawdown_period,
        "annual_style_exposure": annual,
        "drawdown_minus_all_mean_percentile": deltas,
        "held_style_value_coverage": {
            field: {
                "observed": int(exposure[f"observed_{field}_positions"].sum()),
                "missing": int(exposure[f"missing_{field}_positions"].sum()),
                "coverage_ratio": float(
                    exposure[f"observed_{field}_positions"].sum()
                    / max(
                        exposure[f"observed_{field}_positions"].sum()
                        + exposure[f"missing_{field}_positions"].sum(),
                        1,
                    )
                ),
            }
            for field in STYLE_FIELDS
        },
        "lowest_liquidity_days": exposure.sort_values(
            ["mean_turnover_rate_percentile", "buy_date"], ascending=[True, True]
        )
        .head(20)
        .to_dict("records"),
        "smallest_size_days": exposure.sort_values(
            ["mean_total_mv_percentile", "buy_date"], ascending=[True, True]
        )
        .head(20)
        .to_dict("records"),
        "data_access": access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "style_exposure_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
