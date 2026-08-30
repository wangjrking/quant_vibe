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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_atr_risk_v93_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def stable_id(p: dict) -> str:
    return "v93_" + hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def ranks(a: dict) -> dict:
    return {
        "10d80_5d20": v82.build_rank(a, {"w5": 0.2, "w10": 0.8}),
        "10d70_5d30": v82.build_rank(a, {"w5": 0.3, "w10": 0.7}),
    }


def atr_gate(a: dict, maximum: float) -> dict:
    result = dict(a)
    ratio = a["atr_qfq"] / np.where(a["close_qfq"] > 0, a["close_qfq"], np.nan)
    result["signal_clean"] = a["signal_clean"] & np.isfinite(ratio) & (ratio <= maximum)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-2026", action="store_true")
    opt = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with np.load(ROOT / protocol["input_cache"], allow_pickle=False) as saved:
        full = {key: saved[key] for key in saved.files}
    rank_map = ranks(full)
    if opt.open_2026:
        candidates = json.loads(FROZEN.read_text(encoding="utf-8"))["candidates"]
        rows = []
        for item in candidates:
            p = item["definition"]
            a = atr_gate(full, p["atr_pct_max"])
            for start in protocol["validation_buy_starts"]:
                daily = v86.evaluate(a, rank_map, p, protocol, start)
                for period, begin, end in (("through_20260630", start, "20260630"), ("20260701_20260720", "20260701", "20260720"), ("through_20260720", start, "20260720")):
                    rows.append({"case_id": item["case_id"], "buy_start": start, "period": period, **core.metrics(daily, begin, end)})
        pd.DataFrame(rows).to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
        print(pd.DataFrame(rows).to_json(orient="records"))
        return
    observation = robust.truncate_observation(full, protocol["observation_end"])
    rank_obs = {key: (value[0][: len(observation["dates"])], value[1][: len(observation["dates"])]) for key, value in rank_map.items()}
    fixed, grid = protocol["fixed"], protocol["grid"]
    rows, definitions = [], {}
    for blend, gross, atr_max in product(grid["blend"], grid["target_gross_exposure"], grid["atr_pct_max"]):
        p = {**fixed, "blend": blend, "target_gross_exposure": gross, "atr_pct_max": atr_max}
        cid = stable_id(p)
        definitions[cid] = p
        daily = v86.evaluate(atr_gate(observation, atr_max), rank_obs, p, protocol)
        rows.append({"case_id": cid, **p, **robust.evaluate_robust(daily, protocol)})
    frame = pd.DataFrame(rows)
    g = protocol["observation_gate"]
    frame["eligible"] = (frame["min_year_cumulative_return"] >= g["min_year_return"]) & (frame["full_linear_annual_proxy"] >= g["full_linear_annual_proxy"]) & (frame["full_sharpe"] >= g["full_sharpe"]) & (frame["full_max_drawdown"] <= g["max_drawdown"])
    frame = frame.sort_values(["eligible", "full_linear_annual_proxy", "full_sharpe", "min_year_cumulative_return"], ascending=False)
    frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
    candidates = [{"case_id": str(row.case_id), "definition": definitions[str(row.case_id)], "observation_metrics": row.to_dict()} for _, row in selected.iterrows()]
    FROZEN.write_text(json.dumps({"status": "frozen_before_known_2026_stress", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"grid": len(frame), "eligible": int(frame["eligible"].sum()), "frozen": len(candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
