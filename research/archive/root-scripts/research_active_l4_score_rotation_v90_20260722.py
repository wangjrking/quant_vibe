# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_cohorts_v82_20260722 as v82
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_score_rotation_v90_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v90_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def simulate(a: dict, score: np.ndarray, order: np.ndarray, p: dict, end_date: str, start_buy_date: str | None = None) -> pd.DataFrame:
    dates, stocks = a["dates"].astype(str), a["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    cash = previous_equity = 700_000.0
    shares, entry_index, last_price = {}, {}, {}
    rows = []
    target_pct = float(p["target_gross_exposure"]) / int(p["max_positions"])
    for t, signal_date in enumerate(dates[:-1]):
        if signal_date > end_date:
            break
        buy_date = str(dates[t + 1])
        if start_buy_date and buy_date < start_buy_date:
            continue
        opens, pre_close = a["buy_open"][t], a["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        for idx in list(shares):
            if valid_open[idx]:
                last_price[idx] = float(opens[idx])
        equity_before = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        clean = a["signal_clean"][t] & a["buy_clean"][t] & valid_open & np.isfinite(score[t])
        clean &= a["listed_days"][t] >= int(p["listed_days_min"])
        clean &= np.isfinite(a["amount"][t]) & (a["amount"][t] >= int(p["amount_min"]))
        clean &= np.isfinite(a["total_mv"][t]) & (a["total_mv"][t] >= int(p["mv_min"]))
        clean &= np.isfinite(a["turnover_rate"][t]) & (a["turnover_rate"][t] >= 0) & (a["turnover_rate"][t] <= float(p["turnover_max"]))
        clean &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        ranked = [int(idx) for idx in order[t] if clean[int(idx)]]
        best_unheld = max((float(score[t, idx]) for idx in ranked if idx not in shares), default=-np.inf)
        turnover = 0.0
        trades = 0
        for idx in sorted(list(shares), key=lambda item: stocks[item]):
            age = t - entry_index[idx]
            current = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
            score_exit = age >= int(p["min_hold_days"]) and current < float(p["sell_rank_below"]) and best_unheld - current >= float(p["replacement_advantage"])
            if age < int(p["max_hold_days"]) and not score_exit:
                continue
            if not valid_open[idx] or opens[idx] <= pre_close[idx] * (1.0 - board_rate[idx]) * 1.005:
                continue
            gross = float(shares[idx] * opens[idx])
            slip = core.adaptive_slippage(gross, a["amount"][t, idx], "sell")
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            del shares[idx], entry_index[idx]
            last_price.pop(idx, None)
        if t % int(p["rebalance_every"]) == 0:
            slots = min(int(p["top_n_per_rebalance"]), max(int(p["max_positions"]) - len(shares), 0))
            candidates = [idx for idx in ranked if idx not in shares][:slots]
            for idx in candidates:
                gross_budget = min(equity_before * target_pct, cash / 1.001)
                if gross_budget < 1000:
                    continue
                slip = core.adaptive_slippage(gross_budget, a["amount"][t, idx], "buy")
                buy_price = float(opens[idx]) * (1.0 + slip)
                quantity = gross_budget / (buy_price * 1.0003)
                spend = quantity * buy_price * 1.0003
                if quantity <= 0 or spend > cash + 1e-6:
                    continue
                cash -= spend
                shares[idx], entry_index[idx], last_price[idx] = quantity, t, float(opens[idx])
                turnover += quantity * float(opens[idx])
                trades += 1
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-2026", action="store_true")
    opt = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    ranks = {
        "10d100": v82.build_rank(full, {"w5": 0.0, "w10": 1.0}),
        "10d80_5d20": v82.build_rank(full, {"w5": 0.2, "w10": 0.8}),
    }
    if opt.open_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for candidate in frozen["candidates"]:
            p = candidate["definition"]
            score, order = ranks[p["blend"]]
            for start in protocol["validation_buy_starts"]:
                daily = simulate(full, score, order, p, protocol["validation_end"], start)
                for period, begin, end in (("through_20260630", start, "20260630"), ("20260701_20260720", "20260701", "20260720"), ("through_20260720", start, "20260720")):
                    rows.append({"case_id": candidate["case_id"], "buy_start": start, "period": period, **core.metrics(daily, begin, end)})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return
    observation = robust.truncate_observation(full, protocol["observation_end"])
    fixed, grid = protocol["fixed"], protocol["grid"]
    rows, definitions = [], {}
    for blend, min_hold, max_hold, sell_rank, advantage in product(grid["blend"], grid["min_hold_days"], grid["max_hold_days"], grid["sell_rank_below"], grid["replacement_advantage"]):
        p = {**fixed, "blend": blend, "min_hold_days": min_hold, "max_hold_days": max_hold, "sell_rank_below": sell_rank, "replacement_advantage": advantage}
        cid = stable_id(p)
        definitions[cid] = p
        score, order = ranks[blend]
        daily = simulate(observation, score[: len(observation["dates"])], order[: len(observation["dates"])], p, protocol["observation_end"])
        rows.append({"case_id": cid, **p, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    g = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= g["min_year_return"]) & (frame["min_year_sharpe"] >= g["min_year_sharpe"]) & (frame["full_sharpe"] >= g["full_sharpe"]) & (frame["full_max_drawdown"] <= g["max_drawdown"]) & (frame["full_trades"] >= g["trades_min"])
    frame = frame.sort_values(["eligible", "min_year_cumulative_return", "min_year_sharpe", "full_sharpe", "full_linear_annual_proxy"], ascending=False)
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
