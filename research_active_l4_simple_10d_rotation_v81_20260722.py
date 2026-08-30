# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_diversified_equalweight_v77_20260722 as v77
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_simple_10d_rotation_v81_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "v81_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-validation", action="store_true")
    return parser.parse_args()


def make_state(arrays, weights, fixed):
    score = v77.core.blend_scores(arrays, weights)
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")
    signal_clean = arrays["signal_clean"].copy()
    signal_clean &= np.isfinite(arrays["amount"]) & (arrays["amount"] >= fixed["amount_min"])
    signal_clean &= np.isfinite(arrays["total_mv"]) & (arrays["total_mv"] >= fixed["mv_min"])
    signal_clean &= np.isfinite(arrays["turnover_rate"])
    signal_clean &= (arrays["turnover_rate"] >= 0) & (arrays["turnover_rate"] <= fixed["turnover_max"])
    masked = {"signal_clean": signal_clean, "buy_clean": arrays["buy_clean"].copy()}
    exposure = np.ones(len(arrays["dates"]), dtype=np.float32)
    return score, order, masked, exposure


def slice_time(value, start, total_dates):
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == total_dates:
        return value[start:]
    return value


def main():
    args = parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("代码哈希不一致")
    cache = ROOT / protocol["input_cache"]["path"]
    if metrics_lib.digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("输入缓存哈希不一致")
    with np.load(cache, allow_pickle=False) as saved:
        full_arrays = {key: saved[key] for key in saved.files}

    if args.open_validation:
        frozen_path = OUT / "frozen_before_2026.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if metrics_lib.digest(OUT / "observation_results.csv") != frozen["observation_results_sha256"]:
            raise RuntimeError("观察期结果发生漂移")
        rows = []
        action_dir = OUT / "validation_actions"
        action_dir.mkdir(parents=True, exist_ok=True)
        dates = full_arrays["dates"].astype(str)
        total_dates = len(dates)
        for candidate in frozen["candidates"]:
            weights = protocol["blends"][candidate["blend_id"]]
            state = make_state(full_arrays, weights, protocol["fixed"])
            portfolio = v77.PortfolioProfile(
                int(candidate["top_n"]),
                float(candidate["gross_exposure"]),
                int(protocol["fixed"]["amount_min"]),
                int(protocol["fixed"]["mv_min"]),
                float(protocol["fixed"]["turnover_max"]),
            )
            exit_profile = sell_lib.ExitProfile(
                "rank_10d",
                int(candidate["min_hold"]),
                int(candidate["max_hold"]),
                float(candidate["sell_rank_below"]),
                float(candidate["replacement_advantage"]),
            )
            for buy_start in protocol["validation"]["empty_start_buy_dates"]:
                match = np.where(dates[1:] == buy_start)[0]
                if len(match) != 1:
                    raise RuntimeError("无法定位启动日 {}".format(buy_start))
                start = int(match[0])
                arrays = {key: slice_time(value, start, total_dates) for key, value in full_arrays.items()}
                score, order, masked, exposure = state
                sliced_state = (
                    score[start:],
                    order[start:],
                    {key: value[start:] for key, value in masked.items()},
                    exposure[start:],
                )
                daily, actions = v77.simulate(
                    arrays,
                    sliced_state,
                    portfolio,
                    exit_profile,
                    protocol["validation"]["end"],
                    record_actions=True,
                )
                action_path = action_dir / "{}_{}_actions.csv".format(candidate["case_id"], buy_start)
                actions.to_csv(action_path, index=False, encoding="utf-8-sig")
                result = v77.core.metrics(daily, buy_start, protocol["validation"]["end"])
                rows.append(
                    {
                        "case_id": candidate["case_id"],
                        "buy_start": buy_start,
                        "action_path": str(action_path.relative_to(ROOT)).replace("\\", "/"),
                        "action_sha256": metrics_lib.digest(action_path),
                        "buy_rows": int((actions["action"] == "BUY").sum()) if len(actions) else 0,
                        **result,
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "validation_results.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return

    arrays = robust.truncate_observation(full_arrays, protocol["observation_end"])
    rows = []
    definitions = {}
    for blend_id, top_n, gross, min_hold, max_hold, sell_rank, advantage in product(
        protocol["grid"]["blend_id"],
        protocol["grid"]["top_n"],
        protocol["grid"]["gross_exposure"],
        protocol["grid"]["min_hold"],
        protocol["grid"]["max_hold"],
        protocol["grid"]["sell_rank_below"],
        protocol["grid"]["replacement_advantage"],
    ):
        if max_hold < min_hold:
            continue
        definition = {
            "blend_id": blend_id,
            "top_n": top_n,
            "gross_exposure": gross,
            "min_hold": min_hold,
            "max_hold": max_hold,
            "sell_rank_below": sell_rank,
            "replacement_advantage": advantage,
        }
        case_id = stable_id(definition)
        definitions[case_id] = definition
        state = make_state(arrays, protocol["blends"][blend_id], protocol["fixed"])
        portfolio = v77.PortfolioProfile(
            top_n,
            gross,
            int(protocol["fixed"]["amount_min"]),
            int(protocol["fixed"]["mv_min"]),
            float(protocol["fixed"]["turnover_max"]),
        )
        exit_profile = sell_lib.ExitProfile("rank_10d", min_hold, max_hold, sell_rank, advantage)
        daily = v77.simulate(arrays, state, portfolio, exit_profile, protocol["observation_end"])
        rows.append({"case_id": case_id, **definition, **robust.evaluate_robust(daily, protocol)})

    frame = pd.DataFrame(rows)
    frame["eligible"] = (
        frame["robust_positive"]
        & (frame["min_year_sharpe"] >= protocol["gate"]["min_year_sharpe"])
        & (frame["full_sharpe"] >= protocol["gate"]["full_sharpe"])
        & (frame["full_max_drawdown"] <= protocol["gate"]["max_drawdown"])
        & (frame["full_trades"] >= protocol["gate"]["trades_min"])
    )
    frame = frame.sort_values(
        ["eligible", "min_year_sharpe", "min_year_return", "full_sharpe", "full_linear_annual_proxy", "case_id"],
        ascending=[False, False, False, False, False, True],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "observation_results.csv"
    frame.to_csv(result_path, index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(protocol["gate"]["freeze_count"])
    candidates = [{"case_id": row.case_id, **definitions[row.case_id]} for row in selected.itertuples()]
    frozen = {
        "status": "frozen_before_2026",
        "observation_results_sha256": metrics_lib.digest(result_path),
        "candidates": candidates,
        "known_2026_used_for_selection": false,
    }
    (OUT / "frozen_before_2026.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates), "top": selected.to_dict("records")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
