# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_active_l4_agreement_v8_20260721 as agreement_lib
from . import production_v260_active_l4_buy_coverage_v39_20260721 as simulator
from . import production_v260_active_l4_candidate_strength_v30_20260721 as metrics_lib
from . import production_v260_active_l4_continuous_market_v53_20260721 as exposure_lib
from . import production_v260_active_l4_independent_sell_v36_20260721 as sell_lib
from . import production_v260_active_l4_market_state_v26_20260721 as market_lib
from . import production_v260_active_l4_weak_open_refine_v49_20260721 as weak_open
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_robust_objective_v56_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


def digest_payload(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def truncate_observation(arrays, end_date):
    dates = arrays["dates"].astype(str)
    keep = int(np.searchsorted(dates, end_date, side="right"))
    # Keep no date after observation end. The simulator uses dates[:-1], so
    # the last executable buy date also remains inside the observation period.
    result = {}
    for key, value in arrays.items():
        result[key] = value[:keep] if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(dates) else value
    return result


def score_array(arrays, weights):
    return core.blend_scores(arrays, weights)


def apply_agreement(arrays, profile):
    return agreement_lib.masked_arrays(arrays, profile)


def period_metrics(daily, start_date, end_date):
    subset = daily[(daily["date"].astype(str) >= start_date) & (daily["date"].astype(str) <= end_date)].copy()
    if subset.empty:
        return {"cumulative_return": -1.0, "sharpe": -99.0, "max_drawdown": 1.0, "trades": 0, "invested_ratio_mean": 0.0}
    return core.metrics(subset, start_date, end_date)


def evaluate_robust(daily, protocol):
    result = metrics_lib.evaluate(daily, protocol["year_folds"])
    dates = daily["date"].astype(str).to_numpy()
    for window in protocol["recent_windows"]:
        count = int(window["sessions"])
        start = str(dates[-count]) if len(dates) >= count else str(dates[0])
        values = period_metrics(daily, start, protocol["observation_end"])
        for key, value in values.items():
            result[f"recent{count}_{key}"] = value
    year_sharpes = [result[f"{fold['id']}_sharpe"] for fold in protocol["year_folds"]]
    year_returns = [result[f"{fold['id']}_cumulative_return"] for fold in protocol["year_folds"]]
    result["min_year_sharpe"] = float(min(year_sharpes))
    result["min_year_return"] = float(min(year_returns))
    result["robust_sharpe_floor"] = float(min(result["min_year_sharpe"], result["recent60_sharpe"], result["recent120_sharpe"]))
    result["robust_positive"] = bool(result["min_year_return"] > 0 and result["recent60_cumulative_return"] > 0 and result["recent120_cumulative_return"] > 0)
    return result


def build_state(arrays, weights, agreement, entry_off, entry_on, fixed):
    buy_score = score_array(arrays, weights)
    order = np.argsort(-np.nan_to_num(buy_score, nan=-np.inf), axis=1).astype(np.int32)
    masked = apply_agreement(arrays, agreement)
    masked = weak_open.gated_mask(masked, arrays, fixed["min_open_gap"])
    market_mean = exposure_lib.rolling_mean(market_lib.market_open_return(arrays), fixed["market_lookback"])
    market_on = np.isfinite(market_mean) & (market_mean > fixed["market_entry_threshold"])
    entry = np.where(market_on, entry_on, entry_off).astype(np.float32)
    masked["signal_clean"] &= buy_score >= entry[:, None]
    exposure = exposure_lib.continuous_exposure(market_mean, fixed["exposure_floor"], fixed["linear_low_mean_return"], fixed["linear_high_mean_return"])
    return buy_score, order, masked, exposure


def run_case(arrays, protocol, state, top_n, weight_scheme, exit_profile):
    buy_score, order, masked, exposure = state
    fixed = protocol["fixed"]
    profile = simulator.BuyProfile(
        int(top_n), -1.0, int(fixed["amount_min"]), int(fixed["mv_min"]), 1.0,
        False, 0.99, 0.005, 1.0, weight_scheme,
    )
    daily = simulator.simulate(arrays, masked, buy_score, order, exposure, profile, exit_profile, protocol["observation_end"])
    return profile, daily


def select_stage(df, count):
    pool = df[df["eligible"]].copy()
    return pool.sort_values(
        ["robust_sharpe_floor", "min_year_sharpe", "recent60_sharpe", "recent120_sharpe", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, False, False, True],
    ).head(count)


def main():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if p["status"] != "frozen_research_only":
        raise RuntimeError("protocol is not frozen")
    cache = ROOT / p["input_cache"]["path"]
    if metrics_lib.digest(cache) != p["input_cache"]["sha256"]:
        raise RuntimeError("input cache hash mismatch")
    if metrics_lib.digest(Path(__file__)) != p["code_sha256"]:
        raise RuntimeError("code hash mismatch")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = truncate_observation({key: saved[key] for key in saved.files}, p["observation_end"])

    base_exit = sell_lib.ExitProfile(**p["stage1_fixed_exit"])
    stage1_rows, states = [], {}
    for weights in p["stage1_grid"]["score_weights"]:
        for agreement in p["stage1_grid"]["agreement_profiles"]:
            for entry in p["stage1_grid"]["entry_rank_pairs"]:
                state_key = digest_payload({"weights": weights, "agreement": agreement, "entry": entry})
                state = build_state(arrays, weights, agreement, entry["risk_off"], entry["risk_on"], p["fixed"])
                states[state_key] = state
                for top_n in p["stage1_grid"]["top_n"]:
                    profile, daily = run_case(arrays, p, state, top_n, "equal", base_exit)
                    values = evaluate_robust(daily, p)
                    definition = {"state_key": state_key, "weights": weights, "agreement": agreement, "entry": entry, "top_n": top_n, "weight_scheme": "equal", "exit": asdict(base_exit)}
                    cid = "r1_" + digest_payload(definition)
                    stage1_rows.append({"case_id": cid, "state_key": state_key, "weights_json": json.dumps(weights, sort_keys=True), "agreement_json": json.dumps(agreement, sort_keys=True), "entry_json": json.dumps(entry, sort_keys=True), "top_n": top_n, "weight_scheme": "equal", **asdict(base_exit), **values})

    stage1 = pd.DataFrame(stage1_rows)
    stage1["eligible"] = stage1["robust_positive"] & (stage1["full_max_drawdown"] <= 0.40) & (stage1["full_trades"] >= 80)
    stage1 = stage1.sort_values(["eligible", "robust_sharpe_floor", "full_cumulative_return", "case_id"], ascending=[False, False, False, True])
    OUT.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(OUT / "stage1_grid_results.csv", index=False, encoding="utf-8-sig")
    promoted = select_stage(stage1, int(p["stage1_grid"]["promote_count"]))

    stage2_rows, stage2_defs = [], {}
    for _, seed in promoted.iterrows():
        state = states[str(seed.state_key)]
        for exposure_floor in p["stage2_grid"]["exposure_floor"]:
            # Rebuild only exposure while preserving the promoted score state.
            market_mean = exposure_lib.rolling_mean(market_lib.market_open_return(arrays), p["fixed"]["market_lookback"])
            exposure = exposure_lib.continuous_exposure(market_mean, exposure_floor, p["fixed"]["linear_low_mean_return"], p["fixed"]["linear_high_mean_return"])
            state2 = (state[0], state[1], state[2], exposure)
            for weight_scheme in p["stage2_grid"]["weight_scheme"]:
                for exit_values in p["stage2_grid"]["exit_profiles"]:
                    exit_profile = sell_lib.ExitProfile("rank_5d", **exit_values)
                    _, daily = run_case(arrays, p, state2, int(seed.top_n), weight_scheme, exit_profile)
                    values = evaluate_robust(daily, p)
                    definition = {"seed": seed.case_id, "exposure_floor": exposure_floor, "weight_scheme": weight_scheme, "exit": asdict(exit_profile)}
                    cid = "r2_" + digest_payload(definition)
                    stage2_defs[cid] = {"seed": seed.to_dict(), "exposure_floor": exposure_floor, "weight_scheme": weight_scheme, "exit": asdict(exit_profile), "state_key": str(seed.state_key)}
                    stage2_rows.append({"case_id": cid, "seed_case_id": seed.case_id, "state_key": seed.state_key, "top_n": int(seed.top_n), "exposure_floor": exposure_floor, "weight_scheme": weight_scheme, **asdict(exit_profile), **values})

    stage2 = pd.DataFrame(stage2_rows)
    stage2["eligible"] = stage2["robust_positive"] & (stage2["full_max_drawdown"] <= 0.40) & (stage2["full_trades"] >= 80)
    stage2 = stage2.sort_values(["eligible", "robust_sharpe_floor", "full_cumulative_return", "case_id"], ascending=[False, False, False, True])
    stage2.to_csv(OUT / "stage2_grid_results.csv", index=False, encoding="utf-8-sig")
    frozen = select_stage(stage2, int(p["stage2_grid"]["promote_count"]))
    (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": metrics_lib.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        definition = stage2_defs[item.case_id]
        state = states[definition["state_key"]]
        market_mean = exposure_lib.rolling_mean(market_lib.market_open_return(arrays), p["fixed"]["market_lookback"])
        exposure = exposure_lib.continuous_exposure(market_mean, definition["exposure_floor"], p["fixed"]["linear_low_mean_return"], p["fixed"]["linear_high_mean_return"])
        state2 = (state[0], state[1], state[2], exposure)
        exit_profile = sell_lib.ExitProfile(**definition["exit"])
        buy_profile = simulator.BuyProfile(int(item.top_n), -1.0, p["fixed"]["amount_min"], p["fixed"]["mv_min"], 1.0, False, 0.99, 0.005, 1.0, definition["weight_scheme"])
        _, actions = simulator.simulate(arrays, state2[2], state2[0], state2[1], state2[3], buy_profile, exit_profile, p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": metrics_lib.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": int(item.top_n)})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_eligible": int(stage1.eligible.sum()), "stage1_promoted": len(promoted), "stage2_cases": len(stage2), "stage2_eligible": int(stage2.eligible.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
