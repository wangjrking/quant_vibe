# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_cohorts_v82_20260722 as v82
import research_active_l4_causal_health_gate_v87_20260722 as v87
import research_active_l4_relaxed_universe_v86_20260722 as v86
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_causal_health_gate_v88_20260722"


def main() -> None:
    protocol = json.loads((OUT / "preregistered_protocol.json").read_text(encoding="utf-8"))
    v86_protocol = json.loads(v86.PROTOCOL.read_text(encoding="utf-8"))
    base_candidates = json.loads(v86.FROZEN.read_text(encoding="utf-8"))["candidates"]
    base = next(item["definition"] for item in base_candidates if item["case_id"] == protocol["base_candidate"])
    cache = ROOT / v86_protocol["input_cache"]
    with np.load(cache, allow_pickle=False) as saved:
        a = {key: saved[key] for key in saved.files}
    ranks = {
        "10d100": v82.build_rank(a, {"w5": 0.0, "w10": 1.0}),
        "10d80_5d20": v82.build_rank(a, {"w5": 0.2, "w10": 0.8}),
    }
    rule = protocol["health_rule"]
    outcomes = v87.build_mature_3d_outcomes(a)
    gate = v87.causal_gate(outcomes, int(rule["window"]), float(rule["mean_net_min"]), float(rule["positive_ratio_min"]))
    gated = v87.gated_arrays(a, gate)
    rows = []
    for start in protocol["validation_buy_starts"]:
        daily = v86.evaluate(gated, ranks, base, {**v86_protocol, "validation_end": protocol["validation_end"]}, start)
        rows.append({"buy_start": start, "period": "through_20260630", **core.metrics(daily, start, "20260630")})
        rows.append({"buy_start": start, "period": "20260701_20260720", **core.metrics(daily, "20260701", "20260720")})
        rows.append({"buy_start": start, "period": "through_20260720", **core.metrics(daily, start, protocol["validation_end"])})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "known_2026_stress.csv", index=False, encoding="utf-8-sig")
    dates = np.asarray(a["dates"], dtype=str)
    gate_frame = pd.DataFrame({"signal_date": dates, "health_gate_open": gate.astype(int)})
    gate_frame[gate_frame["signal_date"] >= "20260101"].to_csv(OUT / "health_gate_2026.csv", index=False, encoding="utf-8-sig")
    print(frame.to_json(orient="records"))


if __name__ == "__main__":
    main()
