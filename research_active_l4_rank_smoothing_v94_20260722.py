# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_relaxed_universe_v86_20260722 as v86
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_rank_smoothing_v94_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def stable_id(p: dict) -> str:
    raw = json.dumps(p, sort_keys=True, separators=(",", ":"))
    return "v94_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(values)
    sums = np.vstack([np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(np.where(finite, values, 0.0), axis=0, dtype=np.float64)])
    counts = np.vstack([np.zeros((1, values.shape[1]), dtype=np.int32), np.cumsum(finite, axis=0, dtype=np.int32)])
    end = np.arange(1, values.shape[0] + 1)
    start = np.maximum(end - window, 0)
    total = sums[end] - sums[start]
    count = counts[end] - counts[start]
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0).astype(np.float32)


def raw_blend(a: dict, blend: str) -> np.ndarray:
    weights = {"10d100": (0.0, 1.0), "10d90_5d10": (0.1, 0.9), "10d80_5d20": (0.2, 0.8)}[blend]
    return (weights[0] * a["rank_5d"] + weights[1] * a["rank_10d"]).astype(np.float32)


def build_score(a: dict, blend: str, window: int, current_weight: float) -> tuple[np.ndarray, np.ndarray]:
    raw = raw_blend(a, blend)
    smooth = rolling_nanmean(raw, int(window))
    score = (float(current_weight) * raw + (1.0 - float(current_weight)) * smooth).astype(np.float32)
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")
    return score, order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-2026", action="store_true")
    opt = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    if opt.open_2026:
        candidates = json.loads(FROZEN.read_text(encoding="utf-8"))["candidates"]
        rows = []
        for item in candidates:
            p = item["definition"]
            score_id = p["blend"]
            score_map = {score_id: build_score(full, p["base_blend"], int(p["smooth_window"]), float(p["current_weight"]))}
            for start in protocol["validation_buy_starts"]:
                daily = v86.evaluate(full, score_map, p, protocol, start)
                for period, begin, end in (("through_20260630", start, "20260630"), ("20260701_20260720", "20260701", "20260720"), ("through_20260720", start, "20260720")):
                    rows.append({"case_id": item["case_id"], "buy_start": start, "period": period, **core.metrics(daily, begin, end)})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return
    observation = robust.truncate_observation(full, protocol["observation_end"])
    fixed, sg, pg = protocol["fixed_universe"], protocol["score_grid"], protocol["portfolio_grid"]
    rows, definitions = [], {}
    for blend, window, current_weight in product(sg["blend"], sg["smooth_window"], sg["current_weight"]):
        score_id = f"{blend}_w{window}_c{current_weight}"
        score, order = build_score(full, blend, int(window), float(current_weight))
        score_obs = {score_id: (score[: len(observation["dates"])], order[: len(observation["dates"])])}
        for top_n, hold, gross in product(pg["top_n_per_rebalance"], pg["hold_days"], pg["target_gross_exposure"]):
            p = {**fixed, "blend": score_id, "base_blend": blend, "smooth_window": window, "current_weight": current_weight, "top_n_per_rebalance": top_n, "hold_days": hold, "target_gross_exposure": gross}
            cid = stable_id(p)
            definitions[cid] = p
            daily = v86.evaluate(observation, score_obs, p, protocol)
            rows.append({"case_id": cid, **p, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    g = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= g["min_year_return"]) & (frame["full_linear_annual_proxy"] >= g["full_linear_annual_proxy"]) & (frame["full_sharpe"] >= g["full_sharpe"]) & (frame["full_max_drawdown"] <= g["max_drawdown"]) & (frame["full_trades"] >= g["trades_min"])
    frame = frame.sort_values(["eligible", "min_year_cumulative_return", "full_linear_annual_proxy", "full_sharpe", "full_max_drawdown"], ascending=[False, False, False, False, True])
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
