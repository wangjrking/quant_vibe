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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_causal_health_gate_v87_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def case_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v87_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_mature_3d_outcomes(a: dict) -> np.ndarray:
    score = a["rank_3d"]
    buy = a["buy_open"]
    sell = np.full_like(buy, np.nan, dtype=np.float32)
    sell[:-3] = buy[3:]
    valid = np.isfinite(score) & np.isfinite(buy) & (buy > 0) & np.isfinite(sell) & (sell > 0)
    valid &= a["signal_clean"] & a["buy_clean"]
    result = np.full(len(a["dates"]), np.nan, dtype=float)
    for s in range(len(a["dates"]) - 3):
        idx = np.flatnonzero(valid[s])
        if len(idx) == 0:
            continue
        selected = idx[np.argsort(-score[s, idx], kind="stable")[:3]]
        buy_cost = buy[s, selected] * 1.003 * 1.0003
        sell_value = sell[s, selected] * 0.997 * (1.0 - 0.0003 - 0.0005)
        result[s] = float(np.mean(sell_value / buy_cost - 1.0))
    return result


def causal_gate(outcomes: np.ndarray, window: int, mean_min: float, positive_min: float) -> np.ndarray:
    gate = np.ones(len(outcomes), dtype=np.bool_)
    for t in range(len(outcomes)):
        # 3D trade from signal s matures at the open of signal s+4.
        mature_end = t - 4
        if mature_end < 0:
            continue
        history = outcomes[: mature_end + 1]
        history = history[np.isfinite(history)]
        if len(history) < window:
            continue
        recent = history[-window:]
        gate[t] = bool(np.mean(recent) >= mean_min and np.mean(recent > 0) >= positive_min)
    return gate


def gated_arrays(a: dict, gate: np.ndarray) -> dict:
    result = dict(a)
    result["signal_clean"] = a["signal_clean"] & gate[:, None]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-2026", action="store_true")
    opt = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    base = json.loads((ROOT / protocol["base_candidate_path"]).read_text(encoding="utf-8"))["candidates"][int(protocol["base_candidate_index"])]["definition"]
    cache = ROOT / json.loads(v86.PROTOCOL.read_text(encoding="utf-8"))["input_cache"]
    with np.load(cache, allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    outcomes = build_mature_3d_outcomes(full)
    ranks = {
        "10d100": v82.build_rank(full, {"w5": 0.0, "w10": 1.0}),
        "10d80_5d20": v82.build_rank(full, {"w5": 0.2, "w10": 0.8}),
    }
    if opt.open_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for candidate in frozen["candidates"]:
            d = candidate["health"]
            gate = causal_gate(outcomes, int(d["window"]), float(d["mean_net_min"]), float(d["positive_ratio_min"]))
            a = gated_arrays(full, gate)
            for start in protocol["validation_buy_starts"]:
                daily = v86.evaluate(a, ranks, base, {**json.loads(v86.PROTOCOL.read_text(encoding="utf-8")), "validation_end": protocol["validation_end"]}, start)
                rows.append({"case_id": candidate["case_id"], "buy_start": start, "gate_open_ratio": float(gate[np.asarray(full["dates"], dtype=str) >= "20260101"].mean()), **core.metrics(daily, start, protocol["validation_end"])})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    ranks_obs = {key: (value[0][: len(observation["dates"])], value[1][: len(observation["dates"])]) for key, value in ranks.items()}
    rows, definitions = [], {}
    for window, mean_min, positive_min in product(protocol["health_grid"]["window"], protocol["health_grid"]["mean_net_min"], protocol["health_grid"]["positive_ratio_min"]):
        health = {"window": window, "mean_net_min": mean_min, "positive_ratio_min": positive_min}
        cid = case_id(health)
        definitions[cid] = health
        gate = causal_gate(outcomes[: len(observation["dates"])], window, mean_min, positive_min)
        daily = v86.evaluate(gated_arrays(observation, gate), ranks_obs, base, json.loads(v86.PROTOCOL.read_text(encoding="utf-8")))
        rows.append({"case_id": cid, **health, "gate_open_ratio": float(gate.mean()), **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    g = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= g["min_year_return"]) & (frame["min_year_sharpe"] >= g["min_year_sharpe"]) & (frame["full_sharpe"] >= g["full_sharpe"]) & (frame["full_max_drawdown"] <= g["max_drawdown"]) & (frame["full_trades"] >= g["trades_min"])
    frame = frame.sort_values(["eligible", "min_year_cumulative_return", "min_year_sharpe", "full_sharpe", "full_linear_annual_proxy"], ascending=False)
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "health": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_selection": False, "base_strategy": base, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
