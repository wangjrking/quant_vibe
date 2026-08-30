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
import research_v260_fixed10_entry_beta_guard_20260822 as beta_guard
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_beta_exposure_diagnostic_20260822"
)


def summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("empty beta exposure frame")
    values = frame["mean_market_beta"].dropna().astype(float)
    if values.empty:
        raise ValueError("beta exposure has no observed values")
    return {
        "days": int(len(frame)),
        "observed_days": int(len(values)),
        "mean_market_beta": float(values.mean()),
        "median_market_beta": float(values.median()),
        "p10_market_beta": float(values.quantile(0.10)),
        "p90_market_beta": float(values.quantile(0.90)),
        "days_mean_beta_above_120": int((values > 1.20).sum()),
        "days_mean_beta_above_150": int((values > 1.50).sum()),
    }


def build_exposure(observations: list[dict], arrays: dict, beta: np.ndarray) -> pd.DataFrame:
    stock_index = {str(code): idx for idx, code in enumerate(arrays["stocks"])}
    date_index = {str(date): idx for idx, date in enumerate(arrays["dates"])}
    rows = []
    for item in observations:
        day = date_index[str(item["signal_date"])]
        indices = [
            stock_index[str(code)] for code in item["positions_after_trades"]
        ]
        values = np.asarray(beta[day, indices], dtype=np.float64)
        observed = values[np.isfinite(values)]
        rows.append(
            {
                "signal_date": str(item["signal_date"]),
                "buy_date": str(item["buy_date"]),
                "position_count": len(indices),
                "observed_beta_positions": int(len(observed)),
                "missing_beta_positions": int(len(indices) - len(observed)),
                "mean_market_beta": (
                    float(np.mean(observed)) if len(observed) else np.nan
                ),
                "maximum_market_beta": (
                    float(np.max(observed)) if len(observed) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    daily, _, observations, context = positions.run_with_observer()
    _, _, _, _, arrays, access = round1.load_arrays(round1.DEVELOPMENT_END)
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered beta exposure diagnostic")
    beta = beta_guard.trailing_market_beta(
        arrays["close_qfq"],
        beta_guard.BETA_LOOKBACK,
        beta_guard.BETA_MIN_OBSERVATIONS,
    )
    exposure = build_exposure(observations, arrays, beta)
    episode = attribution.maximum_drawdown_episode(daily)
    drawdown = exposure[
        (exposure["buy_date"] >= episode["peak_date"])
        & (exposure["buy_date"] <= episode["trough_date"])
    ].reset_index(drop=True)
    result = {
        "status": "diagnostic_complete_2026_not_opened",
        "source_strategy": context["rules"]["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "beta_definition": {
            "lookback_sessions": beta_guard.BETA_LOOKBACK,
            "minimum_observations": beta_guard.BETA_MIN_OBSERVATIONS,
            "market_return": "same-day cross-sectional median qfq close return",
        },
        "all_period": summarize(exposure),
        "annual": {
            str(year): summarize(group.reset_index(drop=True))
            for year, group in exposure.groupby(exposure["buy_date"].str[:4], sort=True)
        },
        "maximum_drawdown_episode": episode,
        "maximum_drawdown_beta_exposure": summarize(drawdown),
        "coverage": {
            "observed": int(exposure["observed_beta_positions"].sum()),
            "missing": int(exposure["missing_beta_positions"].sum()),
        },
        "data_access": access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "beta_exposure_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
