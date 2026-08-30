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
import research_active_l4_relaxed_universe_v86_20260722 as v86
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_shadow_drawdown_gate_v91_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v91_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def causal_gate(dates: np.ndarray, shadow: pd.DataFrame, window: int, trigger: float, cooldown: int) -> np.ndarray:
    equity = shadow.set_index("date")["equity"].sort_index()
    gate = np.ones(len(dates), dtype=np.bool_)
    blocked = 0
    for t, signal_date in enumerate(dates.astype(str)):
        if blocked > 0:
            gate[t] = False
            blocked -= 1
            continue
        history = equity[equity.index <= signal_date].tail(window)
        if len(history) < window:
            continue
        drawdown = float(history.iloc[-1] / history.max() - 1.0)
        if drawdown <= -trigger:
            gate[t] = False
            blocked = cooldown - 1
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-2026", action="store_true")
    opt = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    v86_protocol = json.loads(v86.PROTOCOL.read_text(encoding="utf-8"))
    bases_all = json.loads((ROOT / protocol["base_candidate_path"]).read_text(encoding="utf-8"))["candidates"]
    bases = {item["case_id"]: item["definition"] for item in bases_all if item["case_id"] in protocol["base_case_ids"]}
    with np.load(ROOT / v86_protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    ranks = {
        "10d100": v82.build_rank(full, {"w5": 0.0, "w10": 1.0}),
        "10d80_5d20": v82.build_rank(full, {"w5": 0.2, "w10": 0.8}),
        "10d60_5d40": v82.build_rank(full, {"w5": 0.4, "w10": 0.6}),
    }
    if opt.open_2026:
        candidates = json.loads(FROZEN.read_text(encoding="utf-8"))["candidates"]
        rows = []
        for item in candidates:
            base = bases[item["base_case_id"]]
            shadow = v86.evaluate(full, ranks, base, {**v86_protocol, "validation_end": protocol["validation_end"]})
            rule = item["gate"]
            gate = causal_gate(full["dates"], shadow, int(rule["rolling_sessions"]), float(rule["drawdown_trigger"]), int(rule["cooldown_sessions"]))
            gated = dict(full)
            gated["signal_clean"] = full["signal_clean"] & gate[:, None]
            for start in protocol["validation_buy_starts"]:
                daily = v86.evaluate(gated, ranks, base, {**v86_protocol, "validation_end": protocol["validation_end"]}, start)
                for period, begin, end in (("through_20260630", start, "20260630"), ("20260701_20260720", "20260701", "20260720"), ("through_20260720", start, "20260720")):
                    rows.append({"case_id": item["case_id"], "buy_start": start, "period": period, "gate_open_ratio_2026": float(gate[full["dates"].astype(str) >= "20260101"].mean()), **core.metrics(daily, begin, end)})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return
    observation = robust.truncate_observation(full, protocol["observation_end"])
    ranks_obs = {key: (value[0][: len(observation["dates"])], value[1][: len(observation["dates"])]) for key, value in ranks.items()}
    rows, definitions = [], {}
    for base_id, base in bases.items():
        shadow = v86.evaluate(observation, ranks_obs, base, v86_protocol)
        for window, trigger, cooldown in product(protocol["grid"]["rolling_sessions"], protocol["grid"]["drawdown_trigger"], protocol["grid"]["cooldown_sessions"]):
            rule = {"rolling_sessions": window, "drawdown_trigger": trigger, "cooldown_sessions": cooldown}
            cid = stable_id({"base_case_id": base_id, **rule})
            definitions[cid] = {"base_case_id": base_id, "gate": rule}
            gate = causal_gate(observation["dates"], shadow, window, trigger, cooldown)
            gated = dict(observation)
            gated["signal_clean"] = observation["signal_clean"] & gate[:, None]
            daily = v86.evaluate(gated, ranks_obs, base, v86_protocol)
            rows.append({"case_id": cid, "base_case_id": base_id, **rule, "gate_open_ratio": float(gate.mean()), **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    g = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= g["min_year_return"]) & (frame["min_year_sharpe"] >= g["min_year_sharpe"]) & (frame["full_sharpe"] >= g["full_sharpe"]) & (frame["full_max_drawdown"] <= g["max_drawdown"]) & (frame["full_trades"] >= g["trades_min"])
    frame = frame.sort_values(["eligible", "min_year_cumulative_return", "min_year_sharpe", "full_sharpe", "full_linear_annual_proxy"], ascending=False)
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), **definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
