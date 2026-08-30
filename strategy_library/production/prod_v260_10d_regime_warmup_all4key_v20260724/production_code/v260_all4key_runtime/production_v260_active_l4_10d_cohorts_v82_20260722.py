# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_active_l4_candidate_strength_v30_20260721 as metrics_lib
from . import production_v260_active_l4_robust_objective_v56_20260721 as robust
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_10d_cohorts_v82_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def parse_args():
    parser = argparse.ArgumentParser(description="10D每日分批持有研究")
    parser.add_argument("--open-2026", action="store_true")
    return parser.parse_args()


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "v82_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_rank(arrays, weights):
    score = (
        float(weights["w5"]) * arrays["rank_5d"]
        + float(weights["w10"]) * arrays["rank_10d"]
    ).astype(np.float32)
    return score, np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")


def simulate(arrays, score, order, params, end_date, start_buy_date=None, record_actions=False):
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    cash = 700_000.0
    previous_equity = 700_000.0
    shares = {}
    entry_index = {}
    last_price = {}
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
        opens = arrays["buy_open"][t]
        pre_close = arrays["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        for idx in list(shares):
            if valid_open[idx]:
                last_price[idx] = float(opens[idx])
        equity_before = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)

        turnover = 0.0
        trades = 0
        for idx in sorted(list(shares), key=lambda item: stocks[item]):
            if t - entry_index[idx] < hold_days:
                continue
            if not valid_open[idx]:
                continue
            lower = pre_close[idx] * (1.0 - board_rate[idx])
            if opens[idx] <= lower * 1.005:
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

        clean = arrays["signal_clean"][t] & arrays["buy_clean"][t]
        clean &= valid_open & np.isfinite(score[t])
        clean &= arrays["listed_days"][t] >= 60
        clean &= np.isfinite(arrays["amount"][t]) & (arrays["amount"][t] >= int(params["amount_min"]))
        clean &= np.isfinite(arrays["total_mv"][t]) & (arrays["total_mv"][t] >= int(params["mv_min"]))
        clean &= np.isfinite(arrays["turnover_rate"][t])
        clean &= (arrays["turnover_rate"][t] >= 0.0) & (arrays["turnover_rate"][t] <= float(params["turnover_max"]))
        clean &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        slots = min(daily_top_n, max(max_positions - len(shares), 0))
        candidates = [int(idx) for idx in order[t] if clean[int(idx)] and int(idx) not in shares][:slots]
        for idx in candidates:
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
            shares[idx] = quantity
            entry_index[idx] = t
            last_price[idx] = float(opens[idx])
            turnover += quantity * float(opens[idx])
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": target_pct, "execution_open_raw": float(opens[idx])})

        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    frame = pd.DataFrame(rows)
    return (frame, pd.DataFrame(actions)) if record_actions else frame


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
    ranks = {item["id"]: build_rank(full_arrays, item) for item in protocol["blends"]}

    if not args.open_2026:
        arrays = robust.truncate_observation(full_arrays, protocol["observation_end"])
        rows = []
        definitions = {}
        for blend, top_n, hold_days, gross in product(protocol["blends"], protocol["grid"]["daily_top_n"], protocol["grid"]["hold_days"], protocol["grid"]["gross_exposure"]):
            params = {"blend": blend, "daily_top_n": int(top_n), "hold_days": int(hold_days), "gross_exposure": float(gross), **protocol["fixed"]}
            case_id = stable_id(params)
            definitions[case_id] = params
            score, order = ranks[blend["id"]]
            daily = simulate(arrays, score[: len(arrays["dates"])], order[: len(arrays["dates"])], params, protocol["observation_end"])
            rows.append({"case_id": case_id, "blend_id": blend["id"], **{key: value for key, value in params.items() if key != "blend"}, **robust.evaluate_robust(daily, protocol)})
        frame = pd.DataFrame(rows)
        gate = protocol["gate"]
        frame["eligible"] = frame["robust_positive"] & (frame["min_year_cumulative_return"] >= gate["min_year_return"]) & (frame["min_year_sharpe"] >= gate["min_year_sharpe"]) & (frame["full_sharpe"] >= gate["full_sharpe"]) & (frame["full_max_drawdown"] <= gate["max_drawdown"]) & (frame["full_trades"] >= gate["trades_min"])
        frame = frame.sort_values(["eligible", "min_year_sharpe", "min_year_return", "full_sharpe", "full_linear_annual_proxy", "case_id"], ascending=[False, False, False, False, False, True])
        OUT.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
        selected = frame[frame.eligible].head(int(protocol["freeze_count"]))
        candidates = []
        action_dir = OUT / "observation_actions"
        action_dir.mkdir(parents=True, exist_ok=True)
        for _, item in selected.iterrows():
            params = definitions[str(item.case_id)]
            score, order = ranks[params["blend"]["id"]]
            _, actions = simulate(arrays, score[: len(arrays["dates"])], order[: len(arrays["dates"])], params, protocol["observation_end"], record_actions=True)
            path = action_dir / "{}.csv".format(item.case_id)
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            candidates.append({"case_id": str(item.case_id), "params": params, "observation_metrics": item.to_dict(), "action_path": str(path.relative_to(ROOT)).replace("\\", "/"), "action_sha256": metrics_lib.digest(path)})
        FROZEN.write_text(json.dumps({"status": "frozen_before_2026", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"grid": len(frame), "eligible": int(frame.eligible.sum()), "frozen": len(candidates)}, ensure_ascii=False))
        return

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    rows = []
    action_dir = OUT / "validation_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    for candidate in frozen["candidates"]:
        params = candidate["params"]
        score, order = ranks[params["blend"]["id"]]
        for buy_start in protocol["validation_buy_starts"]:
            daily, actions = simulate(full_arrays, score, order, params, protocol["validation_end"], start_buy_date=buy_start, record_actions=True)
            path = action_dir / "{}_{}.csv".format(candidate["case_id"], buy_start)
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            result = core.metrics(daily, buy_start, protocol["validation_end"])
            rows.append({"case_id": candidate["case_id"], "buy_start": buy_start, "action_path": str(path.relative_to(ROOT)).replace("\\", "/"), "action_sha256": metrics_lib.digest(path), "buy_rows": int((actions.action == "BUY").sum()) if len(actions) else 0, **result})
    pd.DataFrame(rows).to_csv(OUT / "validation_2026_local.csv", index=False, encoding="utf-8-sig")
    (OUT / "validation_action_manifest.json").write_text(json.dumps({"status": "known_2026_stress_only", "runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["case_id", "buy_start", "cumulative_return", "sharpe", "max_drawdown"]].to_json(orient="records"))


if __name__ == "__main__":
    main()
