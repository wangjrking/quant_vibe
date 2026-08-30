# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_continuous_refill_v102_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"
ACTION_COLUMNS = ["signal_date", "buy_date", "action", "stock_code", "target_pct", "execution_open_raw"]


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v102_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def simulate(arrays: dict, score: np.ndarray, order: np.ndarray, definition: dict, protocol: dict, end_date: str, start: str | None = None, record_actions: bool = False):
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    cash = float(protocol["execution"]["initial_cash"])
    previous_equity = cash
    shares, entry_index, last_price = {}, {}, {}
    rows, actions = [], []
    hold_days = int(definition["hold_days"])
    max_positions = int(definition["max_positions"])
    daily_entry_cap = int(definition["daily_entry_cap"])
    target_pct = float(definition["target_gross_exposure"]) / max_positions
    slip = float(protocol["execution"]["fixed_slippage_ratio"])

    for t, signal_date in enumerate(dates[:-1]):
        if signal_date > end_date:
            break
        buy_date = str(dates[t + 1])
        if start and buy_date < start:
            continue
        opens, pre_close = arrays["buy_open"][t], arrays["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        for idx in list(shares):
            if valid_open[idx]:
                last_price[idx] = float(opens[idx])
        equity_before = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        turnover = 0.0
        trades = 0
        for idx in sorted(list(shares), key=lambda item: stocks[item]):
            if t - entry_index[idx] < hold_days or not valid_open[idx]:
                continue
            if opens[idx] <= pre_close[idx] * (1.0 - board_rate[idx]) * 1.005:
                continue
            gross = float(shares[idx] * opens[idx])
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx])})
            del shares[idx], entry_index[idx]
            last_price.pop(idx, None)

        clean = arrays["signal_clean"][t] & arrays["buy_clean"][t] & valid_open & np.isfinite(score[t])
        clean &= arrays["listed_days"][t] >= int(definition["listed_days_min"])
        clean &= np.isfinite(arrays["amount"][t]) & (arrays["amount"][t] >= int(definition["amount_min"]))
        clean &= np.isfinite(arrays["total_mv"][t]) & (arrays["total_mv"][t] >= int(definition["mv_min"]))
        clean &= np.isfinite(arrays["turnover_rate"][t]) & (arrays["turnover_rate"][t] >= 0) & (arrays["turnover_rate"][t] <= float(definition["turnover_max"]))
        clean &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        slots = min(daily_entry_cap, max(max_positions - len(shares), 0))
        candidates = [int(idx) for idx in order[t] if clean[int(idx)] and int(idx) not in shares][:slots]
        for idx in candidates:
            gross_budget = min(equity_before * target_pct, cash / 1.001)
            buy_price = float(opens[idx]) * (1.0 + slip)
            quantity = math.floor(gross_budget / (buy_price * 1.0003) / 100.0) * 100.0
            spend = quantity * buy_price * 1.0003
            if quantity < 100 or spend > cash + 1e-6:
                continue
            cash -= spend
            shares[idx], entry_index[idx], last_price[idx] = quantity, t, float(opens[idx])
            turnover += quantity * float(opens[idx])
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": target_pct, "execution_open_raw": float(opens[idx])})
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    return (daily, pd.DataFrame(actions, columns=ACTION_COLUMNS)) if record_actions else daily


def main() -> None:
    parser = argparse.ArgumentParser(description="连续低换手补仓研究")
    parser.add_argument("--open-2026", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    score_cfg = protocol["score"]
    score, order = v95.score_pair(full, score_cfg["weight_5d"], score_cfg["smooth_window"], score_cfg["current_weight"])
    if args.open_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for item in frozen["candidates"]:
            for start in protocol["validation_buy_starts"]:
                daily = simulate(full, score, order, item["definition"], protocol, protocol["validation_end"], start)
                rows.append({"case_id": item["case_id"], "buy_start": start, **core.metrics(daily, start, protocol["validation_end"])})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return
    observation = robust.truncate_observation(full, protocol["observation_end"])
    score_obs, order_obs = score[: len(observation["dates"])], order[: len(observation["dates"])]
    grid, fixed = protocol["grid"], protocol["fixed_universe"]
    rows, definitions = [], {}
    for positions, hold, cap, gross in product(grid["max_positions"], grid["hold_days"], grid["daily_entry_cap"], grid["target_gross_exposure"]):
        definition = {**fixed, "max_positions": positions, "hold_days": hold, "daily_entry_cap": cap, "target_gross_exposure": gross}
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = simulate(observation, score_obs, order_obs, definition, protocol, protocol["observation_end"])
        rows.append({"case_id": case_id, **definition, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    gate = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= gate["min_year_return"]) & (frame["full_linear_annual_proxy"] >= gate["full_linear_annual_proxy"]) & (frame["full_sharpe"] >= gate["full_sharpe"]) & (frame["full_max_drawdown"] <= gate["max_drawdown"])
    frame = frame.sort_values(["eligible", "full_linear_annual_proxy", "min_year_cumulative_return", "full_sharpe", "case_id"], ascending=[False, False, False, False, True])
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_2026", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
