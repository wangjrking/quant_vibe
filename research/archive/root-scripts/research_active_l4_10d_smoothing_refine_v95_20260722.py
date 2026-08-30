# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_rank_smoothing_v94_20260722 as v94
import research_active_l4_relaxed_universe_v86_20260722 as v86
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_10d_smoothing_refine_v95_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def stable_id(p: dict) -> str:
    return "v95_" + hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def score_pair(a: dict, weight_5d: float, window: int, current_weight: float) -> tuple[np.ndarray, np.ndarray]:
    raw = (weight_5d * a["rank_5d"] + (1.0 - weight_5d) * a["rank_10d"]).astype(np.float32)
    smooth = v94.rolling_nanmean(raw, window)
    score = (current_weight * raw + (1.0 - current_weight) * smooth).astype(np.float32)
    return score, np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")


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
            score_id = item["case_id"]
            rank_map = {score_id: score_pair(full, p["weight_5d"], p["smooth_window"], p["current_weight"])}
            definition = {**protocol["fixed"], "blend": score_id}
            for start in protocol["validation_buy_starts"]:
                daily = v86.evaluate(full, rank_map, definition, protocol, start)
                for period, begin, end in (("through_20260630", start, "20260630"), ("20260701_20260720", "20260701", "20260720"), ("through_20260720", start, "20260720")):
                    rows.append({"case_id": item["case_id"], "buy_start": start, "period": period, **core.metrics(daily, begin, end)})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return
    observation = robust.truncate_observation(full, protocol["observation_end"])
    rows, definitions = [], {}
    grid = protocol["score_grid"]
    for weight_5d, window, current_weight in product(grid["weight_5d"], grid["smooth_window"], grid["current_weight"]):
        p = {"weight_5d": weight_5d, "smooth_window": window, "current_weight": current_weight}
        cid = stable_id(p)
        definitions[cid] = p
        score, order = score_pair(full, weight_5d, window, current_weight)
        definition = {**protocol["fixed"], "blend": cid}
        daily = v86.evaluate(observation, {cid: (score[: len(observation["dates"])], order[: len(observation["dates"])])}, definition, protocol)
        rows.append({"case_id": cid, **p, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    g = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= g["min_year_return"]) & (frame["full_linear_annual_proxy"] >= g["full_linear_annual_proxy"]) & (frame["full_sharpe"] >= g["full_sharpe"]) & (frame["full_max_drawdown"] <= g["max_drawdown"])
    frame = frame.sort_values(["eligible", "min_year_cumulative_return", "full_linear_annual_proxy", "full_sharpe"], ascending=False)
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
