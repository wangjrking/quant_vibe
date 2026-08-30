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
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_board_risk_exposure_diagnostic_20260822"
)


def summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("empty board exposure frame")
    values = frame["high_limit_board_share"].astype(float)
    return {
        "days": int(len(frame)),
        "mean_high_limit_board_share": float(values.mean()),
        "median_high_limit_board_share": float(values.median()),
        "p90_high_limit_board_share": float(values.quantile(0.90)),
        "days_share_at_least_050": int((values >= 0.50).sum()),
        "days_share_at_least_070": int((values >= 0.70).sum()),
        "maximum_high_limit_board_share": float(values.max()),
    }


def build_exposure(observations: list[dict]) -> pd.DataFrame:
    rows = []
    for item in observations:
        stocks = sorted(str(code) for code in item["positions_after_trades"])
        if not stocks:
            continue
        rates = runtime.core.board_limit_rate(np.asarray(stocks, dtype=str))
        high_count = int((np.asarray(rates, dtype=float) > 0.15).sum())
        rows.append(
            {
                "signal_date": str(item["signal_date"]),
                "buy_date": str(item["buy_date"]),
                "position_count": len(stocks),
                "high_limit_board_positions": high_count,
                "high_limit_board_share": float(high_count / len(stocks)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    daily, _, observations, context = positions.run_with_observer()
    exposure = build_exposure(observations)
    episode = attribution.maximum_drawdown_episode(daily)
    drawdown = exposure[
        (exposure["buy_date"] >= episode["peak_date"])
        & (exposure["buy_date"] <= episode["trough_date"])
    ].reset_index(drop=True)
    result = {
        "status": "diagnostic_complete_2026_not_opened",
        "source_strategy": context["rules"]["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "high_limit_board_definition": (
            "board_limit_rate greater than 15 percent, separating 20 percent "
            "boards from 10 percent main-board storage tolerance"
        ),
        "all_period": summarize(exposure),
        "annual": {
            str(year): summarize(group.reset_index(drop=True))
            for year, group in exposure.groupby(exposure["buy_date"].str[:4], sort=True)
        },
        "maximum_drawdown_episode": episode,
        "maximum_drawdown_board_exposure": summarize(drawdown),
        "data_access": context["access"],
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "board_risk_exposure_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
