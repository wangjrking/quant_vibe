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

import research_v260_fixed10_drawdown_attribution_20260822 as drawdown
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_style_exposure_diagnostic_20260822 as style
import research_v260_fixed10_weak2_drawdown_position_attribution_20260822 as current_run
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_size_regime_return_diagnostic_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"days": 0}
    values = frame["return"].to_numpy(dtype=np.float64)
    return {
        "days": int(len(frame)),
        "compound_return_on_selected_days": float(np.prod(1.0 + values) - 1.0),
        "total_log_return_contribution": float(np.log1p(values).sum()),
        "mean_daily_return": float(np.mean(values)),
        "positive_day_ratio": float(np.mean(values > 0.0)),
        "mean_size_percentile": float(frame["mean_total_mv_percentile"].mean()),
    }


def grouped(frame: pd.DataFrame, key: str) -> dict:
    return {
        str(value): summarize(group.reset_index(drop=True))
        for value, group in frame.groupby(key, sort=True)
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered size-regime return diagnostic")
    context = harness.load_context(checkpoint["selected_policy"])
    daily, actions, observations, _, _, confirmed_weak = current_run.run_with_observer(
        context, checkpoint["selected_policy"]
    )
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    if not np.isclose(
        metrics["cumulative_return"],
        checkpoint["current_best_equalweight"]["cumulative_return"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("size-regime diagnostic baseline drifted")
    exposure = style.build_daily_exposure(observations, context.arrays)
    dates = np.asarray(context.arrays["dates"], dtype=str)
    date_index = {date: idx for idx, date in enumerate(dates)}
    exposure["market_state"] = [
        "confirmed_weak"
        if bool(confirmed_weak[date_index[str(signal_date)]])
        else "strong_or_first_weak"
        for signal_date in exposure["signal_date"]
    ]
    exposure["size_band"] = pd.cut(
        exposure["mean_total_mv_percentile"],
        bins=[-np.inf, 0.20, 0.35, np.inf],
        labels=["small_below_020", "middle_020_035", "larger_at_least_035"],
        right=False,
    ).astype(str)
    daily_frame = daily[["date", "return"]].copy()
    daily_frame["date"] = daily_frame["date"].astype(str)
    joined = daily_frame.merge(
        exposure[
            [
                "buy_date",
                "mean_total_mv_percentile",
                "market_state",
                "size_band",
            ]
        ],
        left_on="date",
        right_on="buy_date",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(daily):
        raise RuntimeError("size-regime diagnostic date alignment drifted")
    joined["year"] = joined["date"].str[:4]
    joined["joint_state"] = joined["market_state"] + "__" + joined["size_band"]
    episode = drawdown.maximum_drawdown_episode(daily)
    episode_frame = joined[
        (joined["date"] >= episode["peak_date"])
        & (joined["date"] <= episode["trough_date"])
    ].reset_index(drop=True)
    by_year_size = {
        year: grouped(group.reset_index(drop=True), "size_band")
        for year, group in joined.groupby("year", sort=True)
    }
    result = {
        "status": "current_size_regime_return_diagnostic_complete_2026_not_opened",
        "method": (
            "attribute unchanged strategy daily returns to contemporaneous held-portfolio "
            "mean market-cap percentile and confirmed weak state"
        ),
        "overall": summarize(joined),
        "by_size_band": grouped(joined, "size_band"),
        "by_market_state": grouped(joined, "market_state"),
        "by_joint_state": grouped(joined, "joint_state"),
        "by_year_and_size": by_year_size,
        "maximum_drawdown_episode": episode,
        "maximum_drawdown_episode_summary": summarize(episode_frame),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "size_regime_return_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
