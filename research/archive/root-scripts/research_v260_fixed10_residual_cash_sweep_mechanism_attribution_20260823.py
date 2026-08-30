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

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_rank_sizing_residual_cash_sweep_20260823 as sweep


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_residual_cash_sweep_mechanism_attribution_20260823"
)


def _action_keys(actions: pd.DataFrame) -> set[tuple[str, str, str]]:
    required = {"buy_date", "action", "stock_code"}
    if not required.issubset(actions.columns):
        raise ValueError("action columns are incomplete")
    if actions.duplicated(["buy_date", "action", "stock_code"]).any():
        raise ValueError("action keys must be unique")
    return {
        (str(row.buy_date), str(row.action), str(row.stock_code))
        for row in actions.itertuples(index=False)
    }


def mechanism_attribution(
    current_daily: pd.DataFrame,
    current_actions: pd.DataFrame,
    sweep_daily: pd.DataFrame,
    sweep_actions: pd.DataFrame,
) -> dict:
    daily_columns = {"date", "return", "turnover", "invested_ratio"}
    if not daily_columns.issubset(current_daily.columns) or not daily_columns.issubset(
        sweep_daily.columns
    ):
        raise ValueError("daily columns are incomplete")
    joined = (
        current_daily[list(daily_columns)]
        .rename(
            columns={
                "return": "current_return",
                "turnover": "current_turnover",
                "invested_ratio": "current_invested",
            }
        )
        .merge(
            sweep_daily[list(daily_columns)].rename(
                columns={
                    "return": "sweep_return",
                    "turnover": "sweep_turnover",
                    "invested_ratio": "sweep_invested",
                }
            ),
            on="date",
            validate="one_to_one",
        )
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    numeric = joined.drop(columns="date").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("daily attribution inputs must be finite")
    if (joined[["current_return", "sweep_return"]] <= -1.0).any().any():
        raise ValueError("daily returns must be greater than -100%")

    current_keys = _action_keys(current_actions)
    sweep_keys = _action_keys(sweep_actions)
    extra_buy_keys = sorted(
        key for key in sweep_keys - current_keys if key[1] == "BUY"
    )
    extra_buy_dates = {key[0] for key in extra_buy_keys}
    joined["extra_buy_day"] = joined["date"].astype(str).isin(extra_buy_dates)
    joined["invested_delta"] = joined["sweep_invested"] - joined["current_invested"]
    joined["turnover_delta"] = joined["sweep_turnover"] - joined["current_turnover"]
    joined["log_excess"] = np.log1p(joined["sweep_return"]) - np.log1p(
        joined["current_return"]
    )
    joined["next_session_log_excess"] = joined["log_excess"].shift(-1)
    exposure_days = joined["invested_delta"] > 1e-12

    def segment(mask: pd.Series, column: str = "log_excess") -> dict:
        values = joined.loc[mask, column].dropna().to_numpy(dtype=np.float64)
        if not len(values):
            return {"days": 0, "compound_excess": 0.0, "positive_fraction": 0.0}
        return {
            "days": int(len(values)),
            "compound_excess": float(np.expm1(values.sum())),
            "positive_fraction": float(np.mean(values > 0.0)),
        }

    return {
        "additional_buy_action_keys": int(len(extra_buy_keys)),
        "additional_buy_action_days": int(len(extra_buy_dates)),
        "mean_invested_ratio_delta": float(joined["invested_delta"].mean()),
        "median_invested_ratio_delta": float(joined["invested_delta"].median()),
        "positive_invested_delta_days": int(exposure_days.sum()),
        "annualized_turnover_delta": float(joined["turnover_delta"].sum() / len(joined) * 252.0),
        "all_days": segment(pd.Series(True, index=joined.index)),
        "extra_buy_days": segment(joined["extra_buy_day"]),
        "sessions_after_extra_buy": segment(
            joined["extra_buy_day"], "next_session_log_excess"
        ),
        "higher_investment_days": segment(exposure_days),
        "other_days": segment(~exposure_days),
    }


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered cash-sweep mechanism attribution")
    policy = checkpoint["selected_policy"]
    context = sizing.width_tools.harness.load_context(policy)
    current_daily, current_actions = sizing.run_case(
        context,
        policy,
        sizing.BASELINE_COST,
        sizing.TOP_WEIGHT_MULTIPLIER,
        sizing.BOTTOM_WEIGHT_MULTIPLIER,
    )
    sweep_daily, sweep_actions = sweep.run_with_runtime(
        context, policy, sizing.BASELINE_COST, sweep=True
    )
    result = {
        "status": "residual_cash_sweep_mechanism_attribution_complete_2026_not_opened",
        "role": "economic_attribution_only_no_strategy_gate",
        "attribution": mechanism_attribution(
            current_daily, current_actions, sweep_daily, sweep_actions
        ),
        "interpretation": (
            "The report describes where the return and exposure differences occurred. "
            "No statistic is a strategy rejection condition."
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "mechanism_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
