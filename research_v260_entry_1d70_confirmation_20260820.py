# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant/main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


STRATEGY_ID = "prod_v260_10d_regime_warmup_all4key_v20260724"
PRODUCTION_ROOT = MAIN / "strategy_library/production" / STRATEGY_ID
PROTOCOL_PATH = PRODUCTION_ROOT / "inputs/preregistered_protocol.json"
OUT = ROOT / "quant/data_file/reports/strategy_agent_v260_entry_1d70_confirmation_20260820"
DEVELOPMENT_START = "20220607"
DEVELOPMENT_END = "20251231"
VALIDATION_START = "20260105"
ENTRY_CONFIRMATION_FLOOR = 0.70
BASELINE_COST = 0.003
STRESS_COST = 0.0065


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype.kind == "f":
            normalized[column] = normalized[column].round(12)
    raw = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run_case(
    arrays: dict[str, np.ndarray],
    score: np.ndarray,
    order: np.ndarray,
    definition: dict,
    protocol: dict,
    end: str,
    start: str,
    entry_confirmation: bool,
    entry_floor: float = ENTRY_CONFIRMATION_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_mask = v260.v258.v252.v174.selection_mask(
        arrays, definition["max_rank_deterioration"]
    )
    if entry_confirmation:
        selection_mask = selection_mask & (
            np.isfinite(arrays["rank_1d"])
            & (arrays["rank_1d"] >= float(entry_floor))
        )

    return v260.v258.v252.v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start,
        record_actions=True,
        selection_mask_override=selection_mask,
        candidate_target_multiplier_override=v260.v258.v252.warmup_multiplier(
            arrays, score, definition, protocol, start
        ),
        max_positions_override=int(definition["max_positions"]),
        max_positions_schedule_override=v260.position_schedule(
            arrays, definition, start
        ),
    )


def metrics_for(daily: pd.DataFrame, start: str, end: str) -> dict:
    result = core.metrics(daily, start, end)
    return {
        key: (int(value) if isinstance(value, (np.integer,)) else float(value))
        for key, value in result.items()
    }


def annual_metrics(daily: pd.DataFrame, start: str, end: str) -> dict[str, dict]:
    output = {}
    for year in range(int(start[:4]), int(end[:4]) + 1):
        year_start = max(start, f"{year}0101")
        year_end = min(end, f"{year}1231")
        selected = daily[
            daily["date"].astype(str).between(year_start, year_end)
        ]
        if not selected.empty:
            output[str(year)] = metrics_for(daily, year_start, year_end)
    return output


def evaluate_pair(
    arrays: dict[str, np.ndarray],
    score: np.ndarray,
    order: np.ndarray,
    definition: dict,
    protocol: dict,
    start: str,
    end: str,
    cost: float,
    scope: str,
    candidate_id: str = "entry_1d70",
    entry_floor: float = ENTRY_CONFIRMATION_FLOOR,
) -> tuple[dict, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    case_protocol = copy.deepcopy(protocol)
    case_protocol["execution"]["fixed_slippage_ratio"] = cost
    frames = {}
    rows = {}
    for case_id, confirmed in (("production", False), (candidate_id, True)):
        daily, actions = run_case(
            arrays,
            score,
            order,
            definition,
            case_protocol,
            end,
            start,
            confirmed,
            entry_floor,
        )
        frames[case_id] = (daily, actions)
        rows[case_id] = {
            "metrics": metrics_for(daily, start, end),
            "annual": annual_metrics(daily, start, end),
            "daily_digest": frame_digest(daily),
            "actions_digest": frame_digest(actions),
        }
        daily.to_csv(
            OUT / f"{scope}_{case_id}_cost_{cost:.4f}_daily.csv",
            index=False,
            encoding="utf-8-sig",
        )
        actions.to_csv(
            OUT / f"{scope}_{case_id}_cost_{cost:.4f}_actions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    baseline = rows["production"]["metrics"]
    candidate = rows[candidate_id]["metrics"]
    rows["candidate_minus_production"] = {
        key: candidate[key] - baseline[key]
        for key in baseline
        if key in candidate and isinstance(candidate[key], (int, float))
    }
    return rows, frames


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    arrays = v260.v258.v252.fresh_arrays()
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definition = v260.definition_for(protocol, 50)
    latest_executable_date = str(arrays["dates"].astype(str)[-1])
    latest_signal_date = str(arrays["dates"].astype(str)[-2])

    contract = {
        "strategy_id": STRATEGY_ID,
        "research_candidate": "production_entry_1d70_confirmation_v1",
        "single_change": (
            "Only new entries require finite rank_1d >= 0.70; production 10D "
            "score, exits, sizing, position limits, execution and costs are unchanged."
        ),
        "development": [DEVELOPMENT_START, DEVELOPMENT_END],
        "validation": [VALIDATION_START, latest_signal_date],
        "costs": [BASELINE_COST, STRESS_COST],
        "development_gate": {
            "baseline_cost_cumulative_return_delta_gt": 0.0,
            "baseline_cost_sharpe_delta_gt": 0.0,
            "baseline_cost_max_drawdown_delta_lte": 0.02,
            "stress_cost_cumulative_return_delta_gt": 0.0,
            "deterministic_replay": True,
        },
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "production_modified": False,
    }
    (OUT / "research_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    development = {}
    replay_digests = {}
    for cost in (BASELINE_COST, STRESS_COST):
        result, frames = evaluate_pair(
            arrays,
            score,
            order,
            definition,
            protocol,
            DEVELOPMENT_START,
            DEVELOPMENT_END,
            cost,
            "development",
        )
        development[f"{cost:.4f}"] = result
        replay_digests[f"{cost:.4f}"] = {
            case: {
                "daily": frame_digest(items[0]),
                "actions": frame_digest(items[1]),
            }
            for case, items in frames.items()
        }

    replay_result, _ = evaluate_pair(
        arrays,
        score,
        order,
        definition,
        protocol,
        DEVELOPMENT_START,
        DEVELOPMENT_END,
        BASELINE_COST,
        "development_replay",
    )
    deterministic = all(
        replay_result[case][key] == development[f"{BASELINE_COST:.4f}"][case][key]
        for case in ("production", "entry_1d70")
        for key in ("daily_digest", "actions_digest")
    )
    delta_base = development[f"{BASELINE_COST:.4f}"]["candidate_minus_production"]
    delta_stress = development[f"{STRESS_COST:.4f}"]["candidate_minus_production"]
    development_passed = bool(
        delta_base["cumulative_return"] > 0
        and delta_base["sharpe"] > 0
        and delta_base["max_drawdown"] <= 0.02
        and delta_stress["cumulative_return"] > 0
        and deterministic
    )

    validation = None
    if development_passed:
        validation = {}
        for cost in (BASELINE_COST, STRESS_COST):
            result, _ = evaluate_pair(
                arrays,
                score,
                order,
                definition,
                protocol,
                VALIDATION_START,
                latest_signal_date,
                cost,
                "validation",
            )
            validation[f"{cost:.4f}"] = result

    result = {
        "status": "validation_completed" if development_passed else "rejected_in_development",
        "development_passed": development_passed,
        "deterministic_replay": deterministic,
        "latest_market_date": latest_executable_date,
        "latest_signal_date": latest_signal_date,
        "development": development,
        "validation": validation,
        "production_modified": False,
    }
    (OUT / "ab_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
