from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
HARNESS_PATH = (
    REPO
    / "quant/data_file/runtime/agent_workspaces/strategy-agent/work"
    / "prod_v260_fixed10_equalweight_development_comparison_20260812_r2"
    / "observation_attempt_v1/run_comparison.py"
)
OUTPUT_ROOT = REPO / "quant/data_file/reports/strategy_agent_v260_lowrisk_score_tuning_20260821"
SOURCE_MIN = "20220606"
FIRST_BUY = "20220607"
TRAIN_END = "20241231"
PRE2026_END = "20251231"
VALIDATION_START = "20260105"
VALIDATION_END = "20260820"


def load_harness():
    spec = importlib.util.spec_from_file_location("v260_harness_lowrisk", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen production harness unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE_MIN_DATE = SOURCE_MIN
    module.END_DATE = VALIDATION_END
    module.FIRST_BUY_DATE = FIRST_BUY
    return module


def blended_score(arrays, weight_1d: float, window: int, alpha: float):
    raw = (
        float(weight_1d) * arrays["rank_1d"]
        + (1.0 - float(weight_1d)) * arrays["rank_10d"]
    ).astype(np.float32)
    finite = np.isfinite(raw)
    sums = np.vstack(
        [np.zeros((1, raw.shape[1]), dtype=np.float64), np.cumsum(np.where(finite, raw, 0.0), axis=0)]
    )
    counts = np.vstack(
        [np.zeros((1, raw.shape[1]), dtype=np.int32), np.cumsum(finite, axis=0, dtype=np.int32)]
    )
    end = np.arange(1, len(raw) + 1)
    start = np.maximum(end - int(window), 0)
    smooth = np.divide(
        sums[end] - sums[start],
        counts[end] - counts[start],
        out=np.full_like(sums[end], np.nan),
        where=(counts[end] - counts[start]) > 0,
    ).astype(np.float32)
    score = (float(alpha) * raw + (1.0 - float(alpha)) * smooth).astype(np.float32)
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")
    return score, order


def interval(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    values = frame["date"].astype(str)
    return frame[(values >= start) & (values <= end)].copy().reset_index(drop=True)


def metrics(daily: pd.DataFrame) -> dict:
    if daily.empty:
        raise RuntimeError("empty interval")
    returns = daily["return"].astype(float).to_numpy()
    curve = np.cumprod(1.0 + returns)
    dates = pd.to_datetime(daily["date"].astype(str), format="%Y%m%d")
    years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, len(daily) / 252.0)
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    running_max = np.maximum.accumulate(curve)
    annual = {}
    work = daily.assign(year=dates.dt.year.astype(str))
    for year, group in work.groupby("year"):
        annual[str(year)] = float(np.prod(1.0 + group["return"].astype(float)) - 1.0)
    return {
        "start": str(daily.iloc[0]["date"]),
        "end": str(daily.iloc[-1]["date"]),
        "days": int(len(daily)),
        "cumulative_return": float(curve[-1] - 1.0),
        "cagr": float(curve[-1] ** (1.0 / years) - 1.0),
        "sharpe": float(np.mean(returns) / std * math.sqrt(252.0)) if std > 0 else 0.0,
        "max_drawdown": float(np.max(1.0 - curve / running_max)),
        "annual_returns": annual,
        "turnover_sum": float(daily["turnover"].astype(float).sum()),
        "turnover_annualized": float(daily["turnover"].astype(float).sum() / years),
        "average_invested_ratio": float(daily["invested_ratio"].astype(float).mean()),
        "average_positions": float(daily["positions"].astype(float).mean()),
        "trade_events": int(daily["trades"].astype(int).sum()),
    }


def utility(item: dict) -> float:
    annual = [float(v) for y, v in item["annual_returns"].items() if y in {"2023", "2024", "2025"}]
    dispersion = float(np.std(annual, ddof=0)) if annual else 0.0
    return float(
        0.50 * item["cagr"]
        + 0.20 * item["sharpe"]
        - 0.30 * item["max_drawdown"]
        - 0.20 * dispersion
        - 0.001 * item["turnover_annualized"]
    )


def score_digest(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def run_shell(harness, arrays, protocol, score, order, definition, mode: str, end: str, actions=False):
    if mode == "production_shell":
        return harness.v260.run_case(
            arrays, score, order, definition, protocol, end, FIRST_BUY, record_actions=actions
        )

    selection = harness.v174.selection_mask(arrays, definition["max_rank_deterioration"])
    multiplier = harness.v260.v258.v252.warmup_multiplier(
        arrays, score, definition, protocol, FIRST_BUY
    )
    positions = harness.v260.position_schedule(arrays, definition, FIRST_BUY)
    min_hold = None
    if mode in {"no_deterioration", "no_both"}:
        selection = None
    if mode in {"no_trend_multiplier", "no_both"}:
        multiplier = None
    if mode == "fixed15_positions":
        positions = np.full(len(arrays["dates"]), 15, dtype=np.int16)
    if mode == "fixed4_min_hold":
        min_hold = np.full(len(arrays["dates"]), 4, dtype=np.int16)
    return harness.v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        FIRST_BUY,
        record_actions=actions,
        selection_mask_override=selection,
        candidate_target_multiplier_override=multiplier,
        min_hold_days_schedule_override=min_hold,
        max_positions_schedule_override=positions,
    )


def evaluate(daily: pd.DataFrame, params: dict) -> dict:
    train = metrics(interval(daily, FIRST_BUY, TRAIN_END))
    holdout_2025 = metrics(interval(daily, "20250102", PRE2026_END))
    full = metrics(interval(daily, FIRST_BUY, PRE2026_END))
    combined_utility = 0.60 * utility(train) + 0.40 * utility(holdout_2025)
    return {
        **params,
        "selection_utility": float(combined_utility),
        "train_2022_2024": train,
        "holdout_2025": holdout_2025,
        "pre2026": full,
    }


def select_best(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            row["selection_utility"],
            row["pre2026"]["sharpe"],
            row["pre2026"]["cagr"],
            -row["pre2026"]["max_drawdown"],
            json.dumps(row, sort_keys=True, default=str),
        ),
    )


def flatten(row: dict) -> dict:
    output = {k: v for k, v in row.items() if not isinstance(v, dict)}
    for block in ("train_2022_2024", "holdout_2025", "pre2026"):
        for key, value in row[block].items():
            if not isinstance(value, dict):
                output[f"{block}_{key}"] = value
    return output


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    harness = load_harness()
    protocol, rules, manifests = harness.validate_control_plane()
    arrays, access = harness.build_development_arrays()
    if str(arrays["dates"][0]) != SOURCE_MIN or str(arrays["dates"][-1]) != VALIDATION_END:
        raise PermissionError("date coverage mismatch")

    definition = harness.production_definition(protocol)
    baseline_score, baseline_order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    baseline_daily, baseline_actions = run_shell(
        harness,
        arrays,
        protocol,
        baseline_score,
        baseline_order,
        definition,
        "production_shell",
        VALIDATION_END,
        actions=True,
    )
    baseline_eval = evaluate(baseline_daily, {"case": "production_baseline"})

    score_rows = []
    score_cache = {}
    for weight_1d in (0.0, 0.10, 0.25, 0.50, 0.70):
        for window in (3, 5, 7, 10):
            for alpha in (0.10, 0.30, 0.50):
                params = {
                    "weight_1d": weight_1d,
                    "weight_10d": 1.0 - weight_1d,
                    "smooth_window": window,
                    "current_weight": alpha,
                }
                score, order = blended_score(arrays, weight_1d, window, alpha)
                key = score_digest(params)
                score_cache[key] = (score, order)
                daily = run_shell(
                    harness,
                    arrays,
                    protocol,
                    score,
                    order,
                    definition,
                    "production_shell",
                    PRE2026_END,
                )
                score_rows.append(evaluate(daily, {"score_id": key, **params, "mode": "production_shell"}))
    score_winner = select_best(score_rows)
    winner_score, winner_order = score_cache[score_winner["score_id"]]

    ablation_rows = []
    for mode in (
        "production_shell",
        "no_deterioration",
        "no_trend_multiplier",
        "no_both",
        "fixed15_positions",
        "fixed4_min_hold",
    ):
        daily = run_shell(
            harness,
            arrays,
            protocol,
            winner_score,
            winner_order,
            definition,
            mode,
            PRE2026_END,
        )
        ablation_rows.append(
            evaluate(
                daily,
                {
                    "mode": mode,
                    "score_id": score_winner["score_id"],
                    "weight_1d": score_winner["weight_1d"],
                    "weight_10d": score_winner["weight_10d"],
                    "smooth_window": score_winner["smooth_window"],
                    "current_weight": score_winner["current_weight"],
                },
            )
        )
    ablation_winner = select_best(ablation_rows)

    exit_rows = []
    for sell_below in (0.80, 0.85, 0.90):
        for advantage in (0.03, 0.05, 0.08):
            for max_hold in (15, 20, 30):
                case_definition = {
                    **definition,
                    "sell_score_below": sell_below,
                    "replacement_advantage": advantage,
                    "max_hold_days": max_hold,
                }
                daily = run_shell(
                    harness,
                    arrays,
                    protocol,
                    winner_score,
                    winner_order,
                    case_definition,
                    ablation_winner["mode"],
                    PRE2026_END,
                )
                exit_rows.append(
                    evaluate(
                        daily,
                        {
                            "mode": ablation_winner["mode"],
                            "score_id": score_winner["score_id"],
                            "weight_1d": score_winner["weight_1d"],
                            "weight_10d": score_winner["weight_10d"],
                            "smooth_window": score_winner["smooth_window"],
                            "current_weight": score_winner["current_weight"],
                            "sell_score_below": sell_below,
                            "replacement_advantage": advantage,
                            "max_hold_days": max_hold,
                        },
                    )
                )
    final_winner = select_best(exit_rows)

    baseline_pre = baseline_eval["pre2026"]
    winner_pre = final_winner["pre2026"]
    development_pass = bool(
        final_winner["selection_utility"] > baseline_eval["selection_utility"]
        and winner_pre["cagr"] > baseline_pre["cagr"]
        and winner_pre["sharpe"] >= baseline_pre["sharpe"]
        and winner_pre["max_drawdown"] <= baseline_pre["max_drawdown"] * 1.10
    )

    final_definition = {
        **definition,
        "sell_score_below": final_winner["sell_score_below"],
        "replacement_advantage": final_winner["replacement_advantage"],
        "max_hold_days": final_winner["max_hold_days"],
    }
    candidate_daily, candidate_actions = run_shell(
        harness,
        arrays,
        protocol,
        winner_score,
        winner_order,
        final_definition,
        final_winner["mode"],
        VALIDATION_END,
        actions=True,
    )
    repeat_daily, repeat_actions = run_shell(
        harness,
        arrays,
        protocol,
        winner_score,
        winner_order,
        final_definition,
        final_winner["mode"],
        VALIDATION_END,
        actions=True,
    )
    deterministic = {
        "daily": frame_hash(candidate_daily) == frame_hash(repeat_daily),
        "actions": frame_hash(candidate_actions) == frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("determinism failure")

    baseline_validation = metrics(interval(baseline_daily, VALIDATION_START, VALIDATION_END))
    candidate_validation = metrics(interval(candidate_daily, VALIDATION_START, VALIDATION_END))
    validation_pass = bool(
        development_pass
        and candidate_validation["cumulative_return"] > baseline_validation["cumulative_return"]
        and candidate_validation["sharpe"] >= baseline_validation["sharpe"]
        and candidate_validation["max_drawdown"] <= baseline_validation["max_drawdown"] * 1.10
    )

    pd.DataFrame([flatten(row) for row in score_rows]).sort_values(
        "selection_utility", ascending=False
    ).to_csv(OUTPUT_ROOT / "score_search_pre2026.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([flatten(row) for row in ablation_rows]).sort_values(
        "selection_utility", ascending=False
    ).to_csv(OUTPUT_ROOT / "ablation_pre2026.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([flatten(row) for row in exit_rows]).sort_values(
        "selection_utility", ascending=False
    ).to_csv(OUTPUT_ROOT / "exit_search_pre2026.csv", index=False, encoding="utf-8-sig")
    baseline_daily.to_csv(OUTPUT_ROOT / "baseline_daily.csv", index=False, encoding="utf-8-sig")
    candidate_daily.to_csv(OUTPUT_ROOT / "candidate_daily.csv", index=False, encoding="utf-8-sig")
    baseline_actions.to_csv(OUTPUT_ROOT / "baseline_actions.csv", index=False, encoding="utf-8-sig")
    candidate_actions.to_csv(OUTPUT_ROOT / "candidate_actions.csv", index=False, encoding="utf-8-sig")

    result = {
        "schema_version": 1,
        "status": "passed_independent_validation" if validation_pass else "rejected_no_production_change",
        "candidate": "v260_lowrisk_score_tuned_v1",
        "source_strategy": rules["strategy_id"],
        "boundaries": {
            "score_search_and_selection": [FIRST_BUY, PRE2026_END],
            "inner_train": [FIRST_BUY, TRAIN_END],
            "inner_holdout": ["20250102", PRE2026_END],
            "independent_validation": [VALIDATION_START, VALIDATION_END],
            "validation_used_for_parameter_selection": False,
        },
        "search_budget": {
            "score": len(score_rows),
            "ablation": len(ablation_rows),
            "exit": len(exit_rows),
            "total": len(score_rows) + len(ablation_rows) + len(exit_rows),
        },
        "frozen_candidate_before_validation": {
            key: final_winner[key]
            for key in (
                "mode",
                "score_id",
                "weight_1d",
                "weight_10d",
                "smooth_window",
                "current_weight",
                "sell_score_below",
                "replacement_advantage",
                "max_hold_days",
            )
        },
        "development_gate_passed": development_pass,
        "development": {"baseline": baseline_eval, "candidate": final_winner},
        "validation_2026": {
            "baseline": baseline_validation,
            "candidate": candidate_validation,
            "delta": {
                key: float(candidate_validation[key] - baseline_validation[key])
                for key in (
                    "cumulative_return",
                    "cagr",
                    "sharpe",
                    "max_drawdown",
                    "turnover_annualized",
                    "average_invested_ratio",
                )
            },
            "passed": validation_pass,
        },
        "deterministic_replay": deterministic,
        "access": access,
        "formal_manifests": manifests,
        "production_modified": False,
    }
    (OUTPUT_ROOT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["frozen_candidate_before_validation"], ensure_ascii=False))
    print(json.dumps({"development_gate_passed": development_pass, "pre2026": winner_pre}, ensure_ascii=False))
    print(json.dumps(result["validation_2026"], ensure_ascii=False))


if __name__ == "__main__":
    main()
