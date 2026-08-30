from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_best_research_selection_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
RESULT_PATH = REPORT_DIR / "stage2_observation_confirmation.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    source = ROOT / protocol["source_screen"]["path"]
    cache = ROOT / protocol["input_cache"]["path"]
    if digest(source) != protocol["source_screen"]["sha256"] or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("frozen input hash drift")
    screen = pd.read_csv(source)
    base = screen[
        (screen["cumulative_return"] > 0)
        & (screen["max_drawdown"] <= 0.50)
        & (screen["trades"] >= 100)
    ].sort_values(["sharpe", "cagr", "max_drawdown", "profile_id"], ascending=[False, False, True, True]).head(12)
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    weights = {
        "w10_100": {"w1": 0.0, "w3": 0.0, "w5": 0.0, "w10": 1.0},
        "w10_80_w5_20": {"w1": 0.0, "w3": 0.0, "w5": 0.2, "w10": 0.8},
        "w10_70_w5_20_w3_10": {"w1": 0.0, "w3": 0.1, "w5": 0.2, "w10": 0.7},
    }
    scores = {key: core.blend_scores(arrays, value) for key, value in weights.items()}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    rows = []
    grid = protocol["parameter_grid"]
    for _, seed in base.iterrows():
        for min_hold in grid["min_hold"]:
            for max_hold in grid["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for sell_below in grid["sell_rank_below"]:
                    for advantage in grid["replacement_advantage"]:
                        for invested in grid["invested_ratio"]:
                            profile = core.Profile(str(seed["blend_id"]), int(seed["amount_min"]), int(seed["mv_min"]), int(seed["top_n"]), float(seed["entry_rank_min"]), min_hold, max_hold, sell_below, advantage, invested)
                            daily = core.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231")
                            selection = core.metrics(daily, "20220606", "20241231")
                            confirmation = core.metrics(daily, "20250101", "20251231")
                            full = core.metrics(daily, "20220606", "20251231")
                            rows.append({"case_id": profile.profile_id, **asdict(profile), **{f"selection_{k}": v for k, v in selection.items()}, **{f"confirmation_{k}": v for k, v in confirmation.items()}, **{f"full_{k}": v for k, v in full.items()}, "min_period_sharpe": min(selection["sharpe"], confirmation["sharpe"])})
    result = pd.DataFrame(rows).drop_duplicates("case_id")
    result["both_positive"] = (result["selection_cumulative_return"] > 0) & (result["confirmation_cumulative_return"] > 0)
    result = result.sort_values(["both_positive", "min_period_sharpe", "confirmation_sharpe", "full_max_drawdown", "full_turnover", "case_id"], ascending=[False, False, False, True, True, True])
    result.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    frozen = result[result["both_positive"]].head(int(protocol["juejin_candidate_count"]))
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "source_sha256": digest(source), "result_sha256": digest(RESULT_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    if not FROZEN_PATH.exists():
        FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        existing = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        if [x["case_id"] for x in existing["profiles"]] != [x["case_id"] for x in payload["profiles"]]:
            raise RuntimeError("frozen candidate drift")
    summary = {"status": "research_only", "base_profiles": len(base), "stage2_cases": len(result), "frozen_juejin_candidates": len(frozen), "best": frozen.head(1).to_dict("records"), "production_changed": False, "latest_signal_generated": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
