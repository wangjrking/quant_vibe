# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_agreement_v8_20260721 as agreement
import research_active_l4_best_corrected_v6_20260721 as corrected
import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_continuous_market_v53_20260721 as v53
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_market_state_v26_20260721 as market
import research_active_l4_weak_open_refine_v49_20260721 as weak_open
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_dynamic_exit_v54_20260721"
PROTOCOL = OUT / "preregistered_protocol.json"


@dataclass(frozen=True)
class DynamicExitProfile:
    min_hold: int
    risk_off_max_hold: int
    risk_off_sell_rank_below: float
    risk_off_replacement_advantage: float
    risk_on_max_hold: int
    risk_on_sell_rank_below: float
    risk_on_replacement_advantage: float

    @property
    def case_id(self) -> str:
        raw = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return "dx_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def simulate(arrays, masked, buy_score, order, exposure, market_on, profile, fixed, end_date, record_actions=False):
    dates, stocks = arrays["dates"].astype(str), arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    sell_score = sell_lib.exit_score_array(arrays, buy_score, fixed["exit_score_source"])
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
        target_positions = max(min(int(math.floor(2 * float(exposure[t]) + 1e-9)), 2), 0)
        best_unheld = max((float(buy_score[t, idx]) for idx in ranked if idx not in shares), default=-np.inf)

        if market_on[t]:
            max_hold = profile.risk_on_max_hold
            sell_rank = profile.risk_on_sell_rank_below
            advantage = profile.risk_on_replacement_advantage
        else:
            max_hold = profile.risk_off_max_hold
            sell_rank = profile.risk_off_sell_rank_below
            advantage = profile.risk_off_replacement_advantage

        normal_sells = []
        for idx in list(shares):
            age = t - entry_index[idx]
            held_exit = float(sell_score[t, idx]) if np.isfinite(sell_score[t, idx]) else -np.inf
            held_buy = float(buy_score[t, idx]) if np.isfinite(buy_score[t, idx]) else -np.inf
            score_exit = age >= profile.min_hold and held_exit < sell_rank and best_unheld - held_buy >= advantage
            if age >= max_hold or score_exit:
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
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx]), "market_regime": "on" if market_on[t] else "off"})
            del shares[idx]
            del entry_index[idx]
            last_price.pop(idx, None)

        slots = 0 if sell_failed else max(target_positions - len(shares), 0)
        candidates = [idx for idx in ranked if idx not in shares][:slots]
        weights = [1.0] if len(candidates) == 1 else ([fixed["top1_weight"], 1.0 - fixed["top1_weight"]] if len(candidates) == 2 else [])
        for idx, weight in zip(candidates, weights):
            target_pct = float(exposure[t]) * weight
            gross_budget = min(equity_before * target_pct, cash / 1.001)
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
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": target_pct, "execution_open_raw": float(opens[idx]), "market_regime": "on" if market_on[t] else "off"})

        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    return (daily, pd.DataFrame(actions)) if record_actions else daily


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
        arrays = {key: saved[key] for key in saved.files}
    buy_score = core.blend_scores(arrays, p["buy_score"])
    order = np.argsort(-np.nan_to_num(buy_score, nan=-np.inf), axis=1).astype(np.int32)
    masked = agreement.masked_arrays(arrays, p["agreement"])
    masked = weak_open.gated_mask(masked, arrays, p["fixed"]["min_open_gap"])

    market_mean = v53.rolling_mean(market.market_open_return(arrays), p["market_state"]["lookback"])
    market_on = np.isfinite(market_mean) & (market_mean > p["market_state"]["entry_threshold"])
    dynamic_entry = np.where(market_on, p["market_state"]["risk_on_entry_rank_min"], p["market_state"]["risk_off_entry_rank_min"]).astype(np.float32)
    masked["signal_clean"] &= buy_score >= dynamic_entry[:, None]
    exposure = v53.continuous_exposure(market_mean, p["fixed"]["exposure_floor"], p["fixed"]["linear_low_mean_return"], p["fixed"]["linear_high_mean_return"])

    rows = []
    g = p["grid"]
    for min_hold in g["min_hold"]:
        for off_max in g["risk_off_max_hold"]:
            for off_rank in g["risk_off_sell_rank_below"]:
                for off_adv in g["risk_off_replacement_advantage"]:
                    for on_max in g["risk_on_max_hold"]:
                        for on_rank in g["risk_on_sell_rank_below"]:
                            for on_adv in g["risk_on_replacement_advantage"]:
                                profile = DynamicExitProfile(min_hold, off_max, off_rank, off_adv, on_max, on_rank, on_adv)
                                daily = simulate(arrays, masked, buy_score, order, exposure, market_on, profile, p["fixed"], p["observation_end"])
                                result = metrics_lib.evaluate(daily, p["year_folds"])
                                rows.append({"case_id": profile.case_id, **profile.__dict__, **result})

    results = pd.DataFrame(rows)
    results["eligible"] = results["all_year_positive"] & (results["full_max_drawdown"] <= 0.40) & (results["full_trades"] >= 80)
    results = results.sort_values(["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, True])
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "grid_results.csv", index=False, encoding="utf-8-sig")
    pool = results[results["eligible"]]
    by_return = pool.sort_values(["full_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, True]).head(g["promote_return"])
    by_sharpe = pool.sort_values(["full_sharpe", "full_cumulative_return", "case_id"], ascending=[False, False, True]).head(g["promote_sharpe"])
    frozen = pd.concat([by_return, by_sharpe], ignore_index=True).drop_duplicates("case_id")
    frozen_payload = {"protocol_sha256": metrics_lib.digest(PROTOCOL), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    (OUT / "frozen_juejin_candidates.json").write_text(json.dumps(frozen_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        profile = DynamicExitProfile(**{key: item[key] for key in DynamicExitProfile.__dataclass_fields__})
        _, actions = simulate(arrays, masked, buy_score, order, exposure, market_on, profile, p["fixed"], p["observation_end"], record_actions=True)
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": item.case_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": metrics_lib.digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "max_positions": 2})
    (OUT / "juejin_action_manifest.json").write_text(json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "grid_cases": len(results), "all_year_positive": int(results.all_year_positive.sum()), "eligible_cases": len(pool), "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows), "known_2026_used": False, "production_changed": False}
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
