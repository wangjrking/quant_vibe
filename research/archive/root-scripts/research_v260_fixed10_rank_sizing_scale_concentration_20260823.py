from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_return_concentration_diagnostic_20260822 as concentration


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_scale_concentration_20260823"
)
CAPITALS = (350_000.0, 700_000.0, 1_400_000.0, 7_000_000.0)


def context_with_initial_cash(context, initial_cash: float):
    value = float(initial_cash)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("initial cash must be finite and positive")
    protocol = copy.deepcopy(context.protocol)
    protocol["execution"]["initial_cash"] = value
    return replace(context, protocol=protocol)


def paired_excess_concentration(
    dates: pd.Series,
    candidate_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict:
    if len(dates) != len(candidate_returns) or len(dates) != len(baseline_returns):
        raise ValueError("paired concentration inputs do not align")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates.astype(str), format="%Y%m%d"),
            "log_excess": np.log1p(candidate_returns) - np.log1p(baseline_returns),
        }
    )

    def removals(values: np.ndarray, counts: tuple[int, ...]) -> dict:
        order = np.argsort(values)[::-1]
        result = {}
        for count in counts:
            adjusted = values.copy()
            adjusted[order[:count]] = 0.0
            result[str(count)] = {
                "remaining_log_excess": float(adjusted.sum()),
                "remaining_compounded_excess": float(np.exp(adjusted.sum()) - 1.0),
            }
        return result

    daily = frame["log_excess"].to_numpy(dtype=np.float64)
    monthly = (
        frame.groupby(frame["date"].dt.to_period("M"))["log_excess"]
        .sum()
        .to_numpy(dtype=np.float64)
    )
    return {
        "total_log_excess": float(daily.sum()),
        "total_compounded_excess": float(np.exp(daily.sum()) - 1.0),
        "daily_removed_best": removals(daily, (1, 5, 10, 20)),
        "monthly_removed_best": removals(monthly, (1, 3, 5)),
    }


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered scale/concentration diagnostic")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    base_context = sizing.width_tools.harness.load_context(policy)
    capital_results = {}
    reference_frames = None
    for cash in CAPITALS:
        context = context_with_initial_cash(base_context, cash)
        equal_daily, equal_actions = sizing.run_case(
            context, policy, sizing.BASELINE_COST, None, None
        )
        rank_daily, rank_actions = sizing.run_case(
            context,
            policy,
            sizing.BASELINE_COST,
            sizing.TOP_WEIGHT_MULTIPLIER,
            sizing.BOTTOM_WEIGHT_MULTIPLIER,
        )
        equal_metrics = sizing.evaluate(equal_daily, equal_actions)
        rank_metrics = sizing.evaluate(rank_daily, rank_actions)
        capital_results[str(int(cash))] = {
            "initial_cash": cash,
            "equalweight": equal_metrics,
            "rank_sizing": rank_metrics,
            "rank_minus_equalweight": sizing.checkpoint_tools.metric_delta(
                rank_metrics, equal_metrics
            ),
        }
        if cash == 700_000.0:
            reference_frames = (equal_daily, rank_daily)

    if reference_frames is None:
        raise RuntimeError("reference capital was not evaluated")
    equal_daily, rank_daily = reference_frames
    expected = checkpoint["metrics_0_30pct"]
    reference_equivalent = all(
        np.isclose(
            capital_results["700000"]["rank_sizing"][key],
            expected[key],
            rtol=0.0,
            atol=1e-12,
        )
        for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
    )
    if not reference_equivalent:
        raise RuntimeError("700k rank sizing replay drifted")
    paired = paired_excess_concentration(
        equal_daily["date"],
        rank_daily["return"].to_numpy(dtype=np.float64),
        equal_daily["return"].to_numpy(dtype=np.float64),
    )
    result = {
        "status": "rank_sizing_scale_concentration_complete_2026_not_opened",
        "role": "robustness_diagnostic_not_a_new_selection_gate",
        "capital_results": capital_results,
        "rank_beats_equalweight_at_every_capital": all(
            item["rank_minus_equalweight"]["cumulative_return"] > 0.0
            for item in capital_results.values()
        ),
        "reference_700k_equivalence": reference_equivalent,
        "rank_sizing_daily_concentration": concentration.daily_concentration(
            rank_daily
        ),
        "equalweight_daily_concentration": concentration.daily_concentration(
            equal_daily
        ),
        "rank_sizing_monthly_concentration": concentration.monthly_concentration(
            rank_daily
        ),
        "equalweight_monthly_concentration": concentration.monthly_concentration(
            equal_daily
        ),
        "paired_excess_concentration": paired,
        "data_access": base_context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "scale_concentration.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rank_beats_equalweight_at_every_capital": result[
                    "rank_beats_equalweight_at_every_capital"
                ],
                "paired_excess_concentration": paired,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
