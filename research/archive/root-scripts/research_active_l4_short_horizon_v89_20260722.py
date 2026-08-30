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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_short_horizon_v89_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "v89_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_ranks(a: dict) -> dict:
    return {
        "10d100": v82.build_rank(a, {"w5": 0.0, "w10": 1.0}),
        "10d80_5d20": v82.build_rank(a, {"w5": 0.2, "w10": 0.8}),
        "10d60_5d40": v82.build_rank(a, {"w5": 0.4, "w10": 0.6}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-2026", action="store_true")
    opt = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    ranks = build_ranks(full)

    if opt.open_2026:
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        rows = []
        for candidate in frozen["candidates"]:
            definition = candidate["definition"]
            for start in protocol["validation_buy_starts"]:
                daily = v86.evaluate(full, ranks, definition, protocol, start)
                rows.append({"case_id": candidate["case_id"], "buy_start": start, "period": "through_20260630", **core.metrics(daily, start, "20260630")})
                rows.append({"case_id": candidate["case_id"], "buy_start": start, "period": "20260701_20260720", **core.metrics(daily, "20260701", "20260720")})
                rows.append({"case_id": candidate["case_id"], "buy_start": start, "period": "through_20260720", **core.metrics(daily, start, protocol["validation_end"])})
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(frame.to_json(orient="records"))
        return

    observation = robust.truncate_observation(full, protocol["observation_end"])
    ranks_obs = {key: (value[0][: len(observation["dates"])], value[1][: len(observation["dates"])]) for key, value in ranks.items()}
    fixed = protocol["fixed_universe"]
    grid = protocol["strategy_grid"]
    rows, definitions = [], {}
    for blend, top_n, hold, every, gross in product(grid["blend"], grid["top_n_per_rebalance"], grid["hold_days"], grid["rebalance_every"], grid["target_gross_exposure"]):
        definition = {**fixed, "blend": blend, "top_n_per_rebalance": top_n, "hold_days": hold, "rebalance_every": every, "target_gross_exposure": gross}
        cid = stable_id(definition)
        definitions[cid] = definition
        daily = v86.evaluate(observation, ranks_obs, definition, protocol)
        rows.append({"case_id": cid, **definition, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    gate = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= gate["min_year_return"]) & (frame["min_year_sharpe"] >= gate["min_year_sharpe"]) & (frame["full_sharpe"] >= gate["full_sharpe"]) & (frame["full_max_drawdown"] <= gate["max_drawdown"]) & (frame["full_trades"] >= gate["trades_min"])
    frame = frame.sort_values(["eligible", "min_year_cumulative_return", "min_year_sharpe", "full_sharpe", "full_linear_annual_proxy"], ascending=False)
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
