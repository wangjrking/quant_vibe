from __future__ import annotations

import copy
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822 as checkpoint_tools
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_paired_block_bootstrap_20260822 as bootstrap
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_soft_width_profit_optimization_20260822 as width_tools
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_entry_rank_sizing_profit_optimization_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
BASELINE_COST = 0.0030
STRESS_COST = 0.0065
TOP_WEIGHT_MULTIPLIER = 1.10
BOTTOM_WEIGHT_MULTIPLIER = 0.90
TARGET_POSITIONS = 10
BLOCK_LENGTHS = (5, 20, 60)
BOOTSTRAP_SEED = 2_602_026_082_2


def entry_rank_multipliers(
    order: np.ndarray,
    positions: int = 10,
    top_multiplier: float = TOP_WEIGHT_MULTIPLIER,
    bottom_multiplier: float = BOTTOM_WEIGHT_MULTIPLIER,
) -> np.ndarray:
    if order.ndim != 2:
        raise ValueError("order must be a two-dimensional ranking matrix")
    if positions <= 1 or positions > order.shape[1]:
        raise ValueError("positions must be between two and the stock count")
    result = np.ones(order.shape, dtype=np.float64)
    if top_multiplier < bottom_multiplier or bottom_multiplier <= 0.0:
        raise ValueError("rank sizing bounds must be positive and descending")
    weights = np.linspace(top_multiplier, bottom_multiplier, positions, dtype=np.float64)
    if not np.isclose(weights.sum(), float(positions), rtol=0.0, atol=1e-12):
        raise RuntimeError("rank sizing multipliers do not preserve target gross")
    rows = np.arange(order.shape[0])[:, None]
    result[rows, order[:, :positions]] = weights[None, :]
    return result


def run_case(
    context,
    policy: dict,
    cost: float,
    top_multiplier: float | None,
    bottom_multiplier: float | None,
):
    extra_age, sell_priority = width_tools.build_overrides(context, policy)
    simulator = runtime.simulate
    if top_multiplier is not None:
        if bottom_multiplier is None:
            raise ValueError("bottom multiplier is required for rank sizing")
        multipliers = entry_rank_multipliers(
            context.order,
            TARGET_POSITIONS,
            top_multiplier,
            bottom_multiplier,
        )
        simulator = functools.partial(
            runtime.simulate,
            candidate_target_multiplier_override=multipliers,
        )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        simulator=simulator,
        score_sell_priority_override=sell_priority,
        target_positions_override=TARGET_POSITIONS,
        target_gross_override=1.0,
    )


def evaluate(daily, actions) -> dict:
    return round1.evaluate_run(
        daily,
        actions,
        research_base.FIRST_BUY,
        round1.DEVELOPMENT_END,
        target_positions=TARGET_POSITIONS,
    )


def action_keys(actions) -> set[tuple[str, str, str, str]]:
    return {
        (
            str(row.signal_date),
            str(row.buy_date),
            str(row.action),
            str(row.stock_code),
        )
        for row in actions.itertuples(index=False)
    }


def realized_buy_target_diagnostics(actions: pd.DataFrame) -> dict:
    buys = actions.loc[actions["action"].eq("BUY")].copy()
    values = buys["target_pct"].to_numpy(dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("BUY target weights must be present and finite")
    rounded = np.round(values, 10)
    unique, counts = np.unique(rounded, return_counts=True)
    return {
        "buy_actions": int(len(values)),
        "minimum_target_pct": float(values.min()),
        "maximum_target_pct": float(values.max()),
        "mean_target_pct": float(values.mean()),
        "equalweight_10pct_actions": int(np.isclose(values, 0.10).sum()),
        "non_equalweight_actions": int((~np.isclose(values, 0.10)).sum()),
        "target_pct_counts": {
            f"{value:.10f}": int(count)
            for value, count in zip(unique, counts)
        },
    }


def relative_path_diagnostics(
    dates: pd.Series,
    tilted_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict:
    if len(dates) != len(tilted_returns) or len(dates) != len(baseline_returns):
        raise ValueError("relative path inputs do not align")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates.astype(str), format="%Y%m%d"),
            "log_excess": np.log1p(tilted_returns) - np.log1p(baseline_returns),
        }
    )

    def period_summary(freq: str) -> dict:
        grouped = frame.groupby(frame["date"].dt.to_period(freq))["log_excess"].sum()
        return {
            "periods": int(len(grouped)),
            "positive_periods": int((grouped > 0.0).sum()),
            "positive_fraction": float((grouped > 0.0).mean()),
            "median_log_excess": float(grouped.median()),
        }

    rolling = frame["log_excess"].rolling(252).sum().dropna()
    offsets = {}
    for offset in (0, 5, 20, 60):
        candidate = float(np.prod(1.0 + tilted_returns[offset:]) - 1.0)
        baseline = float(np.prod(1.0 + baseline_returns[offset:]) - 1.0)
        offsets[str(offset)] = {
            "tilted_cumulative_return": candidate,
            "equalweight_cumulative_return": baseline,
            "tilted_minus_equalweight": candidate - baseline,
        }
    return {
        "monthly": period_summary("M"),
        "quarterly": period_summary("Q"),
        "annual": period_summary("Y"),
        "rolling_252": {
            "windows": int(len(rolling)),
            "positive_fraction": float((rolling > 0.0).mean()),
            "median_log_excess": float(rolling.median()),
        },
        "start_offsets": offsets,
    }


def invested_exposure_attribution(
    tilted_returns: np.ndarray,
    baseline_returns: np.ndarray,
    tilted_invested: np.ndarray,
    baseline_invested: np.ndarray,
    close_exposure_tolerance: float = 0.0025,
) -> dict:
    arrays = (
        tilted_returns,
        baseline_returns,
        tilted_invested,
        baseline_invested,
    )
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("exposure attribution inputs do not align")
    if close_exposure_tolerance < 0.0:
        raise ValueError("close exposure tolerance must be non-negative")
    log_excess = np.log1p(tilted_returns) - np.log1p(baseline_returns)
    invested_delta = tilted_invested - baseline_invested
    close = np.abs(invested_delta) <= close_exposure_tolerance
    total_log_excess = float(log_excess.sum())
    close_log_excess = float(log_excess[close].sum())
    correlation = (
        float(np.corrcoef(log_excess, invested_delta)[0, 1])
        if np.std(invested_delta) > 0.0
        else 0.0
    )
    design = np.column_stack([np.ones(len(invested_delta)), invested_delta])
    intercept, slope = np.linalg.lstsq(design, log_excess, rcond=None)[0]
    return {
        "close_exposure_tolerance": close_exposure_tolerance,
        "days": int(len(log_excess)),
        "close_exposure_days": int(close.sum()),
        "close_exposure_fraction": float(close.mean()),
        "total_log_excess": total_log_excess,
        "close_exposure_log_excess": close_log_excess,
        "close_exposure_share_of_total": (
            close_log_excess / total_log_excess if total_log_excess != 0.0 else None
        ),
        "mean_invested_ratio_delta": float(invested_delta.mean()),
        "log_excess_invested_delta_correlation": correlation,
        "ols_daily_log_excess_intercept": float(intercept),
        "ols_invested_delta_slope": float(slope),
        "ols_intercept_annualized": float(intercept * 252.0),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered entry rank sizing development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 development boundary drift")

    cases = {}
    frames = {}
    designs = {
        "equalweight_entry": (None, None, "selectable_baseline"),
        "global_rank11_to_9_entry": (1.10, 0.90, "single_selectable_candidate"),
        "rank12_to_8_control": (1.20, 0.80, "robustness_control_not_selectable"),
        "rank13_to_7_control": (1.30, 0.70, "robustness_control_not_selectable"),
    }
    for name, (top, bottom, role) in designs.items():
        daily, actions = run_case(context, policy, BASELINE_COST, top, bottom)
        stress_daily, stress_actions = run_case(
            context, policy, STRESS_COST, top, bottom
        )
        cases[name] = {
            "role": role,
            "entry_target_semantics": (
                "10pct_each"
                if top is None
                else (
                    f"global_score_top10_linear_multiplier_{top:.2f}_to_"
                    f"{bottom:.2f}_at_new_entry_only_refill_outside_top10_is_1.00"
                )
            ),
            "metrics_0_30pct": evaluate(daily, actions),
            "metrics_0_65pct": evaluate(stress_daily, stress_actions),
        }
        frames[name] = (daily, actions)

    selectable = ("equalweight_entry", "global_rank11_to_9_entry")
    selected = max(
        selectable,
        key=lambda name: (
            cases[name]["metrics_0_30pct"]["cumulative_return"],
            cases[name]["metrics_0_30pct"]["sharpe"],
        ),
    )
    selected_top, selected_bottom, _ = designs[selected]
    repeat_daily, repeat_actions = run_case(
        context, policy, BASELINE_COST, selected_top, selected_bottom
    )
    deterministic = {
        "daily": round1.frame_hash(frames[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(frames[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("entry rank sizing replay is not deterministic")

    baseline_daily, baseline_actions = frames["equalweight_entry"]
    tilted_daily, tilted_actions = frames["global_rank11_to_9_entry"]
    if not baseline_daily["date"].astype(str).equals(
        tilted_daily["date"].astype(str)
    ):
        raise RuntimeError("entry rank sizing dates do not align")
    baseline_returns = baseline_daily["return"].to_numpy(dtype=np.float64)
    tilted_returns = tilted_daily["return"].to_numpy(dtype=np.float64)
    paired_bootstrap = {
        str(block): bootstrap.paired_bootstrap(
            tilted_returns,
            baseline_returns,
            block,
            BOOTSTRAP_SEED + block,
        )
        for block in BLOCK_LENGTHS
    }
    baseline_keys = action_keys(baseline_actions)
    tilted_keys = action_keys(tilted_actions)
    action_overlap = {
        "baseline_action_keys": len(baseline_keys),
        "tilted_action_keys": len(tilted_keys),
        "common_action_keys": len(baseline_keys & tilted_keys),
        "baseline_only_action_keys": len(baseline_keys - tilted_keys),
        "tilted_only_action_keys": len(tilted_keys - baseline_keys),
    }
    path_diagnostics = relative_path_diagnostics(
        baseline_daily["date"], tilted_returns, baseline_returns
    )
    exposure_attribution = invested_exposure_attribution(
        tilted_returns,
        baseline_returns,
        tilted_daily["invested_ratio"].to_numpy(dtype=np.float64),
        baseline_daily["invested_ratio"].to_numpy(dtype=np.float64),
    )

    result = {
        "status": "entry_rank_sizing_profit_optimization_complete_2026_not_opened",
        "only_change": (
            "new-entry target size follows one fixed linear score-rank schedule; "
            "selection, exits, T+1, costs and data remain unchanged"
        ),
        "selection_rule": "maximize_pre2026_cumulative_return",
        "equalweight_is_soft_diagnostic": True,
        "candidates": cases,
        "selectable_candidates": list(selectable),
        "robustness_controls_not_selectable": [
            "rank12_to_8_control",
            "rank13_to_7_control",
        ],
        "selected_candidate": selected,
        "selected_minus_equalweight": checkpoint_tools.metric_delta(
            cases[selected]["metrics_0_30pct"],
            cases["equalweight_entry"]["metrics_0_30pct"],
        ),
        "annual_return_delta_tilted_minus_equalweight": {
            year: float(
                cases["global_rank11_to_9_entry"]["metrics_0_30pct"][
                    "annual_returns"
                ][year]
                - cases["equalweight_entry"]["metrics_0_30pct"][
                    "annual_returns"
                ][year]
            )
            for year in ("2022", "2023", "2024", "2025")
        },
        "neighbor_direction_support": {
            name: {
                "baseline_cumulative_delta": float(
                    cases[name]["metrics_0_30pct"]["cumulative_return"]
                    - cases["equalweight_entry"]["metrics_0_30pct"][
                        "cumulative_return"
                    ]
                ),
                "stress_cumulative_delta": float(
                    cases[name]["metrics_0_65pct"]["cumulative_return"]
                    - cases["equalweight_entry"]["metrics_0_65pct"][
                        "cumulative_return"
                    ]
                ),
            }
            for name in ("rank12_to_8_control", "rank13_to_7_control")
        },
        "paired_block_bootstrap": paired_bootstrap,
        "action_overlap": action_overlap,
        "realized_buy_target_diagnostics": {
            "equalweight_entry": realized_buy_target_diagnostics(
                baseline_actions
            ),
            "global_rank_entry": realized_buy_target_diagnostics(
                tilted_actions
            ),
        },
        "relative_path_diagnostics": path_diagnostics,
        "invested_exposure_attribution": exposure_attribution,
        "deterministic_replay": deterministic,
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_candidate": selected,
                "baseline_cumulative_return": cases["equalweight_entry"]
                ["metrics_0_30pct"]["cumulative_return"],
            "tilted_cumulative_return": cases["global_rank11_to_9_entry"]
                ["metrics_0_30pct"]["cumulative_return"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
