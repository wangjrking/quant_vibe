from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_v260_lowrisk_score_tuning_20260821 as research_base
import research_v260_production_rule_simplification_ablation_20260823 as ablation


REPO = Path(r"D:/work/quant/quant_mcp")
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports"
    / "strategy_agent_v260_simple_ema_score_optimization_20260823"
)
RESULT_PATH = OUTPUT_ROOT / "result.json"
EMA_SPAN = 7
EMA_ALPHA = 2.0 / (EMA_SPAN + 1.0)


def causal_ema_score(raw: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("raw score must be a date-by-stock matrix")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    state = np.full(values.shape[1], np.nan, dtype=np.float32)
    for row in range(values.shape[0]):
        current = values[row]
        valid = np.isfinite(current)
        initialized = valid & np.isfinite(state)
        first = valid & ~np.isfinite(state)
        state[first] = current[first]
        state[initialized] = (
            float(alpha) * current[initialized]
            + (1.0 - float(alpha)) * state[initialized]
        )
        result[row, valid] = state[valid]
    return result


def stable_order(score: np.ndarray) -> np.ndarray:
    return np.argsort(
        -np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable"
    )


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def selection_decision(baseline: dict, candidate: dict) -> dict:
    base = baseline["0.003"]
    cand = candidate["0.003"]
    stress_base = baseline["0.0065"]
    stress_cand = candidate["0.0065"]
    annual_delta = {
        year: float(cand["annual_returns"][year] - base["annual_returns"][year])
        for year in sorted(base["annual_returns"])
    }
    gates = {
        "baseline_cumulative_improves": (
            cand["cumulative_return"] > base["cumulative_return"]
        ),
        "baseline_sharpe_not_worse": cand["sharpe"] >= base["sharpe"],
        "drawdown_not_over_10pct_worse": (
            cand["max_drawdown"] <= base["max_drawdown"] * 1.10
        ),
        "stress_cumulative_not_worse": (
            stress_cand["cumulative_return"] >= stress_base["cumulative_return"]
        ),
        "no_annual_drop_over_10pct_points": min(annual_delta.values()) >= -0.10,
    }
    return {
        "selected_for_future_validation": all(gates.values()),
        "gates": gates,
        "annual_return_delta": annual_delta,
        "baseline_cost_delta": {
            key: float(cand[key] - base[key])
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        },
        "stress_cost_delta": {
            key: float(stress_cand[key] - stress_base[key])
            for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
        },
    }


def run_arm(harness, arrays, protocol, definition, score, order) -> dict:
    output = {}
    for cost in (ablation.BASELINE_COST, ablation.STRESS_COST):
        daily, actions = ablation.run_case(
            harness,
            arrays,
            protocol,
            definition,
            score,
            order,
            {},
            cost,
        )
        output[str(cost)] = ablation.summarize(daily, actions)
    return output


def build_report() -> dict:
    harness = research_base.load_harness()
    harness.END_DATE = ablation.DEVELOPMENT_END
    protocol, rules, manifests = harness.validate_control_plane()
    arrays, access = harness.build_development_arrays()
    if str(access["logical_max_date"]) != ablation.DEVELOPMENT_END:
        raise RuntimeError("development boundary drifted")
    if any(str(date) >= "20260101" for date in arrays["dates"]):
        raise PermissionError("2026 entered simple score selection")

    definition = harness.production_definition(protocol)
    baseline_score, baseline_order = harness.v95.score_pair(
        arrays, 0.0, 7, 0.1
    )
    candidate_score = causal_ema_score(arrays["rank_10d"])
    candidate_order = stable_order(candidate_score)
    baseline = run_arm(
        harness, arrays, protocol, definition, baseline_score, baseline_order
    )
    candidate = run_arm(
        harness, arrays, protocol, definition, candidate_score, candidate_order
    )
    repeat = run_arm(
        harness, arrays, protocol, definition, candidate_score, candidate_order
    )
    deterministic = candidate == repeat
    if not deterministic:
        raise RuntimeError("candidate replay is not deterministic")

    decision = selection_decision(baseline, candidate)
    return {
        "status": (
            "simple_candidate_ready_for_future_validation"
            if decision["selected_for_future_validation"]
            else "simple_candidate_rejected_on_pre2026"
        ),
        "candidate": "pure10d_causal_ema_span7_v1",
        "change": {
            "only_changed_component": "score_smoothing",
            "raw_score": "cross_sectional_rank_10d",
            "formula": "ema_t = 0.25 * rank10d_t + 0.75 * ema_t_minus_1",
            "ema_span": EMA_SPAN,
            "alpha": EMA_ALPHA,
            "missing_current_score": "output_nan_without_carrying_stale_score",
        },
        "unchanged": [
            "universe",
            "entry",
            "exit",
            "position sizing",
            "market regimes",
            "costs",
            "raw open and T+1 execution",
        ],
        "selection_boundary": [
            ablation.DEVELOPMENT_START,
            ablation.DEVELOPMENT_END,
        ],
        "2026_used_for_selection": False,
        "baseline": baseline,
        "candidate": candidate,
        "decision": decision,
        "deterministic_replay": deterministic,
        "source_strategy": rules["strategy_id"],
        "source_manifest_count": len(manifests),
        "production_modified": False,
        "registry_modified": False,
        "trading_triggered": False,
    }


def main() -> None:
    report = build_report()
    atomic_json(RESULT_PATH, report)
    print(json.dumps({"status": report["status"], **report["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
