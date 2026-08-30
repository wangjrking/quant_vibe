from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_abnormal_announcement_proxy_20260822 as proxy
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_severe_event_temporal_robustness_20260822"
)
HALF_YEAR_SEGMENTS = {
    "2022H2": ("20220607", "20221231"),
    "2023H1": ("20230101", "20230630"),
    "2023H2": ("20230701", "20231231"),
    "2024H1": ("20240101", "20240630"),
    "2024H2": ("20240701", "20241231"),
    "2025H1": ("20250101", "20250630"),
    "2025H2": ("20250701", "20251231"),
}


def aligned_log_return_attribution(
    baseline_daily: pd.DataFrame,
    candidate_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    required = {"date", "return"}
    for name, frame in (("baseline", baseline_daily), ("candidate", candidate_daily)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} daily missing fields: {missing}")
        if frame["date"].astype(str).duplicated().any():
            raise ValueError(f"{name} daily dates must be unique")
    baseline = baseline_daily[["date", "return"]].copy()
    candidate = candidate_daily[["date", "return"]].copy()
    baseline["date"] = baseline["date"].astype(str)
    candidate["date"] = candidate["date"].astype(str)
    aligned = baseline.merge(
        candidate,
        on="date",
        how="outer",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
        indicator=True,
    )
    if not (aligned["_merge"] == "both").all():
        raise ValueError("baseline and candidate daily dates do not align")
    aligned = aligned.drop(columns="_merge").sort_values("date").reset_index(drop=True)
    for column in ("return_baseline", "return_candidate"):
        values = aligned[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= -1.0):
            raise ValueError(f"invalid daily return in {column}")
    aligned["log_excess"] = np.log1p(aligned["return_candidate"]) - np.log1p(
        aligned["return_baseline"]
    )
    positive = np.sort(
        aligned.loc[aligned["log_excess"] > 0.0, "log_excess"].to_numpy(dtype=np.float64)
    )[::-1]
    positive_sum = float(positive.sum())

    def share(count: int) -> float:
        return float(positive[:count].sum() / positive_sum) if positive_sum > 0.0 else 0.0

    total_log_excess = float(aligned["log_excess"].sum())
    terminal_ratio = float(
        np.prod(1.0 + aligned["return_candidate"].to_numpy(dtype=np.float64))
        / np.prod(1.0 + aligned["return_baseline"].to_numpy(dtype=np.float64))
    )
    return aligned, {
        "changed_return_days": int((np.abs(aligned["log_excess"]) > 1e-15).sum()),
        "positive_log_excess_days": int((aligned["log_excess"] > 0.0).sum()),
        "negative_log_excess_days": int((aligned["log_excess"] < 0.0).sum()),
        "total_log_excess": total_log_excess,
        "terminal_wealth_ratio_candidate_over_baseline": terminal_ratio,
        "log_identity_error": float(abs(np.log(terminal_ratio) - total_log_excess)),
        "top1_positive_day_share": share(1),
        "top3_positive_day_share": share(3),
        "top5_positive_day_share": share(5),
    }


def blocked_baseline_buys(
    actions: pd.DataFrame,
    dates: np.ndarray,
    stocks: np.ndarray,
    severe_block: np.ndarray,
) -> pd.DataFrame:
    if severe_block.shape != (len(dates), len(stocks)):
        raise ValueError("severe block matrix does not align")
    if actions.empty:
        return actions.copy()
    required = {"signal_date", "buy_date", "action", "stock_code"}
    missing = sorted(required.difference(actions.columns))
    if missing:
        raise ValueError(f"actions missing fields: {missing}")
    date_index = {str(value): index for index, value in enumerate(dates)}
    stock_index = {str(value): index for index, value in enumerate(stocks)}
    rows = []
    for row in actions.loc[actions["action"] == "BUY"].itertuples(index=False):
        d_idx = date_index.get(str(row.signal_date))
        s_idx = stock_index.get(str(row.stock_code))
        if d_idx is None or s_idx is None:
            raise ValueError("action key falls outside score arrays")
        if bool(severe_block[d_idx, s_idx]):
            rows.append(row._asdict())
    return pd.DataFrame(rows, columns=actions.columns)


def action_set_diagnostics(
    baseline_actions: pd.DataFrame,
    candidate_actions: pd.DataFrame,
) -> dict:
    key_columns = ["signal_date", "buy_date", "action", "stock_code"]
    for name, frame in (("baseline", baseline_actions), ("candidate", candidate_actions)):
        missing = sorted(set(key_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} actions missing fields: {missing}")
    baseline_keys = set(map(tuple, baseline_actions[key_columns].astype(str).to_numpy()))
    candidate_keys = set(map(tuple, candidate_actions[key_columns].astype(str).to_numpy()))
    changed = baseline_keys.symmetric_difference(candidate_keys)
    changed_dates = sorted({key[0] for key in changed})
    return {
        "baseline_action_count": len(baseline_keys),
        "candidate_action_count": len(candidate_keys),
        "symmetric_difference_count": len(changed),
        "changed_signal_date_count": len(changed_dates),
        "changed_signal_dates": changed_dates,
    }


def segment_metrics(
    baseline_daily: pd.DataFrame,
    baseline_actions: pd.DataFrame,
    candidate_daily: pd.DataFrame,
    candidate_actions: pd.DataFrame,
) -> dict:
    result = {}
    for segment, (start, end) in HALF_YEAR_SEGMENTS.items():
        baseline = round1.evaluate_run(baseline_daily, baseline_actions, start, end)
        candidate = round1.evaluate_run(candidate_daily, candidate_actions, start, end)
        result[segment] = {
            "baseline_cumulative_return": baseline["cumulative_return"],
            "candidate_cumulative_return": candidate["cumulative_return"],
            "cumulative_return_delta": (
                candidate["cumulative_return"] - baseline["cumulative_return"]
            ),
            "baseline_sharpe": baseline["sharpe"],
            "candidate_sharpe": candidate["sharpe"],
            "sharpe_delta": candidate["sharpe"] - baseline["sharpe"],
            "baseline_round_trips": baseline["round_trips"],
            "candidate_round_trips": candidate["round_trips"],
        }
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/"
            "pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(checkpoint["selected_policy"])
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered severe-event robustness diagnostic")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    quality_block = quality.maintenance_quality_block(score, True)
    severe_block, severe_audit = proxy.announcement_block_matrix(
        arrays["dates"],
        arrays["stocks"],
        proxy.load_features(),
        ordinary_window=0,
        severe_window=5,
        warning_window=0,
    )
    common = dict(
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        maintenance_buy_block_mask_override=quality_block,
    )
    baseline_daily, baseline_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        entry_block_mask_override=np.zeros_like(severe_block),
        **common,
    )
    candidate_daily, candidate_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        entry_block_mask_override=severe_block,
        **common,
    )
    baseline_metrics = round1.evaluate_run(
        baseline_daily, baseline_actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    candidate_metrics = round1.evaluate_run(
        candidate_daily, candidate_actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(baseline_metrics[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        drift = {
            key: {
                "actual": baseline_metrics[key],
                "expected": expected[key],
                "delta": baseline_metrics[key] - expected[key],
            }
            for key, matches in checkpoint_equivalence.items()
            if not matches
        }
        raise RuntimeError(f"robustness diagnostic baseline drifted: {drift}")

    aligned, return_attribution = aligned_log_return_attribution(
        baseline_daily, candidate_daily
    )
    blocked_buys = blocked_baseline_buys(
        baseline_actions, arrays["dates"], arrays["stocks"], severe_block
    )
    blocked_buys = blocked_buys.sort_values(
        ["signal_date", "stock_code"], kind="stable"
    ).reset_index(drop=True)
    affected_years = sorted(set(blocked_buys["signal_date"].astype(str).str[:4]))
    segments = segment_metrics(
        baseline_daily, baseline_actions, candidate_daily, candidate_actions
    )
    annual_deltas = {
        year: float(candidate_metrics["annual_returns"][year] - value)
        for year, value in baseline_metrics["annual_returns"].items()
    }
    positive_years = [year for year, value in annual_deltas.items() if value > 1e-12]
    negative_years = [year for year, value in annual_deltas.items() if value < -1e-12]
    robustness_gates = {
        "at_least_four_direct_interventions": len(blocked_buys) >= 4,
        "interventions_span_at_least_two_years": len(affected_years) >= 2,
        "positive_delta_in_at_least_two_years": len(positive_years) >= 2,
        "no_negative_calendar_year_delta": len(negative_years) == 0,
        "top1_positive_day_share_le_50pct": (
            return_attribution["top1_positive_day_share"] <= 0.50
        ),
        "top3_positive_day_share_le_80pct": (
            return_attribution["top3_positive_day_share"] <= 0.80
        ),
        "exact_log_return_identity": return_attribution["log_identity_error"] <= 1e-12,
    }
    supported = bool(all(robustness_gates.values()))
    action_diagnostics = action_set_diagnostics(baseline_actions, candidate_actions)
    result = {
        "status": (
            "temporal_robustness_supported_pre2026_2026_not_opened"
            if supported
            else "temporal_robustness_not_supported_pre2026_2026_not_opened"
        ),
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "rule_under_test": {
            "source": "historical PIT announcement proxy",
            "event": "severe_abnormal_volatility",
            "action": "pause_new_entries_only",
            "window_official_sessions": 5,
            "existing_position_exit_changed": False,
            "maintenance_topup_changed": False,
        },
        "baseline_metrics_0_30pct": baseline_metrics,
        "candidate_metrics_0_30pct": candidate_metrics,
        "annual_return_deltas": annual_deltas,
        "positive_delta_years": positive_years,
        "negative_delta_years": negative_years,
        "half_year_segments": segments,
        "return_attribution": return_attribution,
        "direct_interventions": {
            "blocked_baseline_buy_count": int(len(blocked_buys)),
            "distinct_signal_date_count": int(blocked_buys["signal_date"].nunique()),
            "distinct_stock_count": int(blocked_buys["stock_code"].nunique()),
            "affected_years": affected_years,
            "records": blocked_buys.to_dict(orient="records"),
        },
        "action_set_diagnostics": action_diagnostics,
        "robustness_gates": robustness_gates,
        "supported": supported,
        "checkpoint_equivalence": checkpoint_equivalence,
        "severe_event_proxy_audit": severe_audit,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    aligned.to_csv(
        OUTPUT_ROOT / "daily_log_return_attribution.csv", index=False, encoding="utf-8-sig"
    )
    blocked_buys.to_csv(
        OUTPUT_ROOT / "directly_blocked_baseline_buys.csv", index=False, encoding="utf-8-sig"
    )
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
