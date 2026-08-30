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

from . import production_v260_active_l4_10d_cohorts_v82_20260722 as v82
from . import production_v260_active_l4_robust_objective_v56_20260721 as robust
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_relaxed_universe_v86_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def stable_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "v86_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def stagger(arrays: dict, every: int) -> dict:
    result = dict(arrays)
    result["signal_clean"] = arrays["signal_clean"].copy()
    active = np.zeros(len(arrays["dates"]), dtype=np.bool_)
    active[::every] = True
    result["signal_clean"] &= active[:, None]
    return result


def engine_params(definition: dict) -> dict:
    cohorts = int(math.ceil(definition["hold_days"] / definition["rebalance_every"]))
    internal_gross = definition["target_gross_exposure"] * definition["hold_days"] / cohorts
    return {
        "daily_top_n": definition["top_n_per_rebalance"],
        "hold_days": definition["hold_days"],
        "gross_exposure": internal_gross,
        "amount_min": definition["amount_min"],
        "mv_min": definition["mv_min"],
        "turnover_max": definition["turnover_max"],
        "listed_days_min": definition["listed_days_min"],
    }


def simulate(
    arrays: dict,
    score: np.ndarray,
    order: np.ndarray,
    params: dict,
    end_date: str,
    start_buy_date: str | None = None,
    record_actions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    # Local copy of the cohort engine with listed-days threshold made explicit.
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    cash = 700_000.0
    previous_equity = cash
    shares, entry_index, last_price = {}, {}, {}
    rows = []
    actions = []
    hold_days = int(params["hold_days"])
    daily_top_n = int(params["daily_top_n"])
    target_pct = float(params["gross_exposure"]) / float(hold_days * daily_top_n)
    max_positions = hold_days * daily_top_n
    for t, signal_date in enumerate(dates[:-1]):
        if signal_date > end_date:
            break
        buy_date = str(dates[t + 1])
        if start_buy_date and buy_date < start_buy_date:
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
            fixed_slippage = params.get("fixed_slippage_ratio")
            slip = (
                float(fixed_slippage)
                if fixed_slippage is not None
                else core.adaptive_slippage(gross, arrays["amount"][t, idx], "sell")
            )
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            if record_actions:
                actions.append(
                    {
                        "signal_date": signal_date,
                        "buy_date": buy_date,
                        "action": "SELL",
                        "stock_code": stocks[idx],
                        "target_pct": 0.0,
                        "execution_open_raw": float(opens[idx]),
                    }
                )
            del shares[idx], entry_index[idx]
            last_price.pop(idx, None)
        clean = arrays["signal_clean"][t] & arrays["buy_clean"][t] & valid_open & np.isfinite(score[t])
        clean &= arrays["listed_days"][t] >= int(params["listed_days_min"])
        clean &= np.isfinite(arrays["amount"][t]) & (arrays["amount"][t] >= int(params["amount_min"]))
        clean &= np.isfinite(arrays["total_mv"][t]) & (arrays["total_mv"][t] >= int(params["mv_min"]))
        clean &= np.isfinite(arrays["turnover_rate"][t]) & (arrays["turnover_rate"][t] >= 0) & (arrays["turnover_rate"][t] <= float(params["turnover_max"]))
        clean &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        slots = min(daily_top_n, max(max_positions - len(shares), 0))
        candidates = [int(idx) for idx in order[t] if clean[int(idx)] and int(idx) not in shares][:slots]
        for idx in candidates:
            gross_budget = min(equity_before * target_pct, cash / 1.001)
            if gross_budget < 1000:
                continue
            fixed_slippage = params.get("fixed_slippage_ratio")
            slip = (
                float(fixed_slippage)
                if fixed_slippage is not None
                else core.adaptive_slippage(gross_budget, arrays["amount"][t, idx], "buy")
            )
            buy_price = float(opens[idx]) * (1.0 + slip)
            quantity = gross_budget / (buy_price * (1.0 + 0.0003))
            if params.get("round_lot_100"):
                quantity = math.floor(quantity / 100.0) * 100.0
            spend = quantity * buy_price * (1.0 + 0.0003)
            if quantity <= 0 or spend > cash + 1e-6:
                continue
            cash -= spend
            shares[idx], entry_index[idx], last_price[idx] = quantity, t, float(opens[idx])
            turnover += quantity * float(opens[idx])
            trades += 1
            if record_actions:
                actions.append(
                    {
                        "signal_date": signal_date,
                        "buy_date": buy_date,
                        "action": "BUY",
                        "stock_code": stocks[idx],
                        "target_pct": target_pct,
                        "execution_open_raw": float(opens[idx]),
                    }
                )
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    if not record_actions:
        return daily
    columns = ["signal_date", "buy_date", "action", "stock_code", "target_pct", "execution_open_raw"]
    return daily, pd.DataFrame(actions, columns=columns)


def evaluate(arrays: dict, ranks: dict, definition: dict, protocol: dict, start: str | None = None) -> pd.DataFrame:
    score, order = ranks[definition["blend"]]
    prepared = stagger(arrays, int(definition["rebalance_every"]))
    return simulate(prepared, score[: len(prepared["dates"])], order[: len(prepared["dates"])], engine_params(definition), protocol["validation_end"] if start else protocol["observation_end"], start)


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--open-2026", action="store_true")
    opt = args.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]
    with np.load(cache, allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    ranks = {
        "10d100": v82.build_rank(full, {"w5": 0.0, "w10": 1.0}),
        "10d80_5d20": v82.build_rank(full, {"w5": 0.2, "w10": 0.8}),
    }
    if opt.open_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for candidate in frozen["candidates"]:
            for start in protocol["validation_buy_starts"]:
                daily = evaluate(full, ranks, candidate["definition"], protocol, start)
                rows.append({"case_id": candidate["case_id"], "buy_start": start, **core.metrics(daily, start, protocol["validation_end"])})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    fixed = protocol["stage1_fixed_strategy"]
    stage1_rows = []
    for listed, amount, mv, turn in product(*protocol["stage1_filter_grid"].values()):
        definition = {**fixed, "listed_days_min": listed, "amount_min": amount, "mv_min": mv, "turnover_max": turn}
        daily = evaluate(observation, ranks, definition, protocol)
        stage1_rows.append({**definition, **robust.evaluate_robust(daily, protocol)})
    stage1 = pd.DataFrame(stage1_rows).sort_values(["min_year_cumulative_return", "min_year_sharpe", "full_sharpe", "full_linear_annual_proxy"], ascending=False)
    stage1.to_csv(OUT / "stage1_filter_grid.csv", index=False, encoding="utf-8-sig")
    filters = stage1.head(int(protocol["stage1_keep"]))[["listed_days_min", "amount_min", "mv_min", "turnover_max"]].to_dict("records")
    stage2_rows, definitions = [], {}
    grid = protocol["stage2_strategy_grid"]
    for filt in filters:
        for blend, top_n, hold, every, gross in product(grid["blend"], grid["top_n_per_rebalance"], grid["hold_days"], grid["rebalance_every"], grid["target_gross_exposure"]):
            definition = {**filt, "blend": blend, "top_n_per_rebalance": top_n, "hold_days": hold, "rebalance_every": every, "target_gross_exposure": gross}
            case_id = stable_id(definition)
            definitions[case_id] = definition
            daily = evaluate(observation, ranks, definition, protocol)
            stage2_rows.append({"case_id": case_id, **definition, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(stage2_rows)
    gate = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= gate["min_year_return"]) & (frame["min_year_sharpe"] >= gate["min_year_sharpe"]) & (frame["full_sharpe"] >= gate["full_sharpe"]) & (frame["full_max_drawdown"] <= gate["max_drawdown"]) & (frame["full_trades"] >= gate["trades_min"])
    frame = frame.sort_values(["eligible", "min_year_cumulative_return", "min_year_sharpe", "full_sharpe", "full_linear_annual_proxy"], ascending=False)
    frame.to_csv(OUT / "stage2_strategy_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"stage1": len(stage1), "stage2": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
