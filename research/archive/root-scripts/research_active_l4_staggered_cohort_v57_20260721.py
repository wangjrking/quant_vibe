# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_best_corrected_v6_20260721 as corrected
import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_continuous_market_v53_20260721 as exposure_lib
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_market_state_v26_20260721 as market_lib
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_staggered_cohort_v57_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


@dataclass(frozen=True)
class CohortProfile:
    max_positions: int
    daily_buy_limit: int
    exposure_floor: float
    min_hold: int
    max_hold: int
    sell_rank_below: float
    replacement_advantage: float


def case_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def simulate(arrays, state, profile, fixed, end_date, record_actions=False):
    buy_score, order, masked, exposure = state
    dates, stocks = arrays["dates"].astype(str), arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    sell_score = sell_lib.exit_score_array(arrays, buy_score, "rank_5d")
    cash, previous_equity = 700_000.0, 700_000.0
    shares, entry_index, last_price = {}, {}, {}
    rows, actions = [], []

    for t, signal_date in enumerate(dates[:-1]):
        buy_date = str(dates[t + 1])
        if signal_date > end_date or buy_date > end_date:
            break
        opens, pre_close = arrays["buy_open"][t], arrays["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        for idx in list(shares):
            if valid_open[idx]:
                last_price[idx] = float(opens[idx])
        equity_before = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)

        universe = masked["signal_clean"][t] & masked["buy_clean"][t] & valid_open & np.isfinite(buy_score[t])
        universe &= (arrays["amount"][t] >= fixed["amount_min"]) & (arrays["total_mv"][t] >= fixed["mv_min"]) & (arrays["listed_days"][t] >= 60)
        universe &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        ranked = [int(idx) for idx in order[t] if universe[int(idx)]]
        best_unheld = max((float(buy_score[t, idx]) for idx in ranked if idx not in shares), default=-np.inf)
        target_positions = max(min(int(math.floor(profile.max_positions * float(exposure[t]) + 1e-9)), profile.max_positions), 0)

        normal_sells = []
        for idx in list(shares):
            age = t - entry_index[idx]
            held_exit = float(sell_score[t, idx]) if np.isfinite(sell_score[t, idx]) else -np.inf
            held_buy = float(buy_score[t, idx]) if np.isfinite(buy_score[t, idx]) else -np.inf
            score_exit = age >= profile.min_hold and held_exit < profile.sell_rank_below and best_unheld - held_buy >= profile.replacement_advantage
            if age >= profile.max_hold or score_exit:
                normal_sells.append(idx)
        remaining = [idx for idx in shares if idx not in normal_sells]
        excess = max(len(remaining) - target_positions, 0)
        scale_sells = sorted(
            (idx for idx in remaining if t - entry_index[idx] >= profile.min_hold),
            key=lambda idx: (float(sell_score[t, idx]) if np.isfinite(sell_score[t, idx]) else -np.inf, stocks[idx]),
        )[:excess]

        turnover, trades, sell_failed = 0.0, 0, False
        for idx in normal_sells + scale_sells:
            if not valid_open[idx]:
                sell_failed = True
                continue
            rate = corrected.corrected_limit_rate(arrays, t, idx, board_rate)
            if opens[idx] <= pre_close[idx] * (1.0 - rate) * 1.005:
                sell_failed = True
                continue
            gross = float(shares[idx] * opens[idx])
            slip = core.adaptive_slippage(gross, arrays["amount"][t, idx], "sell")
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx])})
            del shares[idx]
            del entry_index[idx]
            last_price.pop(idx, None)

        slots = 0 if sell_failed else min(max(target_positions - len(shares), 0), profile.daily_buy_limit)
        candidates = [idx for idx in ranked if idx not in shares][:slots]
        per_position_target = float(exposure[t]) / max(target_positions, 1)
        for idx in candidates:
            gross_budget = min(equity_before * per_position_target, cash / 1.001)
            if gross_budget < 1000:
                continue
            slip = core.adaptive_slippage(gross_budget, arrays["amount"][t, idx], "buy")
            buy_price = float(opens[idx]) * (1.0 + slip)
            quantity = gross_budget / (buy_price * (1.0 + 0.0003))
            spend = quantity * buy_price * (1.0 + 0.0003)
            if quantity <= 0 or spend > cash + 1e-6:
                continue
            cash -= spend
            shares[idx], entry_index[idx], last_price[idx] = quantity, t, float(opens[idx])
            turnover += quantity * float(opens[idx])
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": per_position_target, "execution_open_raw": float(opens[idx])})

        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    return (daily, pd.DataFrame(actions)) if record_actions else daily


def rank_results(frame):
    return frame.sort_values(
        ["eligible", "robust_sharpe_floor", "min_year_sharpe", "recent60_sharpe", "recent120_sharpe", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, False, False, False, True],
    )


def main():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if p["status"] != "frozen_research_only":
        raise RuntimeError("protocol is not frozen")
    cache = ROOT / p["input_cache"]["path"]
    if metrics_lib.digest(cache) != p["input_cache"]["sha256"] or metrics_lib.digest(Path(__file__)) != p["code_sha256"]:
        raise RuntimeError("frozen input or code hash mismatch")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = robust.truncate_observation({key: saved[key] for key in saved.files}, p["observation_end"])

    stage1_rows, states, definitions = [], {}, {}
    fixed_profile = CohortProfile(**p["stage1_fixed_profile"])
    for weights in p["stage1_grid"]["score_weights"]:
        for agreement in p["stage1_grid"]["agreement_profiles"]:
            for entry in p["stage1_grid"]["entry_rank_pairs"]:
                state_key = case_id({"weights": weights, "agreement": agreement, "entry": entry})
                state = robust.build_state(arrays, weights, agreement, entry["risk_off"], entry["risk_on"], p["fixed"])
                states[state_key] = state
                daily = simulate(arrays, state, fixed_profile, p["fixed"], p["observation_end"])
                values = robust.evaluate_robust(daily, p)
                cid = "sc1_" + case_id({"state": state_key, "profile": asdict(fixed_profile)})
                definitions[cid] = {"state_key": state_key, "profile": asdict(fixed_profile)}
                stage1_rows.append({"case_id": cid, "state_key": state_key, "weights_json": json.dumps(weights, sort_keys=True), "agreement_json": json.dumps(agreement, sort_keys=True), "entry_json": json.dumps(entry, sort_keys=True), **asdict(fixed_profile), **values})

    stage1 = pd.DataFrame(stage1_rows)
    stage1["eligible"] = stage1["robust_positive"] & (stage1["full_max_drawdown"] <= 0.40) & (stage1["full_trades"] >= 80)
    stage1 = rank_results(stage1)
    OUT.mkdir(parents=True, exist_ok=True)
    stage1.to_csv(OUT / "stage1_grid_results.csv", index=False, encoding="utf-8-sig")
    seeds = stage1[stage1["eligible"]].head(p["stage1_grid"]["promote_count"])

    if seeds.empty:
        pd.DataFrame(columns=["case_id", "eligible"]).to_csv(OUT / "stage2_grid_results.csv", index=False, encoding="utf-8-sig")
        (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": metrics_lib.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {"status": "research_only_observation_no_stage2_seed", "stage1_cases": len(stage1), "stage1_eligible": 0, "stage1_promoted": 0, "stage2_cases": 0, "stage2_eligible": 0, "frozen_candidates": 0, "unique_juejin_paths": 0, "known_2026_used": False, "production_changed": False}
        (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return

    stage2_rows, stage2_defs = [], {}
    for _, seed in seeds.iterrows():
        state_base = states[str(seed.state_key)]
        for max_positions in p["stage2_grid"]["max_positions"]:
            for daily_buy_limit in p["stage2_grid"]["daily_buy_limit"]:
                for exposure_floor in p["stage2_grid"]["exposure_floor"]:
                    market_mean = exposure_lib.rolling_mean(market_lib.market_open_return(arrays), p["fixed"]["market_lookback"])
                    exposure = exposure_lib.continuous_exposure(market_mean, exposure_floor, p["fixed"]["linear_low_mean_return"], p["fixed"]["linear_high_mean_return"])
                    state = (state_base[0], state_base[1], state_base[2], exposure)
                    for exit_values in p["stage2_grid"]["exit_profiles"]:
                        profile = CohortProfile(max_positions, daily_buy_limit, exposure_floor, **exit_values)
                        daily = simulate(arrays, state, profile, p["fixed"], p["observation_end"])
                        values = robust.evaluate_robust(daily, p)
                        cid = "sc2_" + case_id({"seed": seed.case_id, "profile": asdict(profile)})
                        stage2_defs[cid] = {"state_key": str(seed.state_key), "profile": asdict(profile)}
                        stage2_rows.append({"case_id": cid, "seed_case_id": seed.case_id, "state_key": seed.state_key, **asdict(profile), **values})

    stage2 = pd.DataFrame(stage2_rows)
    stage2["eligible"] = stage2["robust_positive"] & (stage2["full_max_drawdown"] <= 0.40) & (stage2["full_trades"] >= 80)
    stage2 = rank_results(stage2)
    stage2.to_csv(OUT / "stage2_grid_results.csv", index=False, encoding="utf-8-sig")
    frozen = stage2[stage2["eligible"]].head(p["stage2_grid"]["promote_count"])
    (OUT / "frozen_juejin_candidates.json").write_text(json.dumps({"protocol_sha256": metrics_lib.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}, ensure_ascii=False, indent=2), encoding="utf-8")

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        definition = stage2_defs[item.case_id]
        profile = CohortProfile(**definition["profile"])
        state_base = states[definition["state_key"]]
        market_mean = exposure_lib.rolling_mean(market_lib.market_open_return(arrays), p["fixed"]["market_lookback"])
        exposure = exposure_lib.continuous_exposure(market_mean, profile.exposure_floor, p["fixed"]["linear_low_mean_return"], p["fixed"]["linear_high_mean_return"])
        state = (state_base[0], state_base[1], state_base[2], exposure)
        _, actions = simulate(arrays, state, profile, p["fixed"], p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": metrics_lib.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": profile.max_positions})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "stage1_eligible": int(stage1.eligible.sum()), "stage1_promoted": len(seeds), "stage2_cases": len(stage2), "stage2_eligible": int(stage2.eligible.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
