from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_best_corrected_v6_20260721 as engine
import research_active_l4_yearly_robust_v11_20260721 as yearly
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_worst_year_repair_v12_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
GRID_PATH = REPORT_DIR / "repair_grid.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    seed_path = ROOT / protocol["seed_source"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"] or digest(seed_path) != protocol["seed_source"]["sha256"]:
        raise RuntimeError("frozen protocol, cache, or seed source drift")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score_rules = {item["id"]: item for item in protocol["score_grid"]}
    scores = {key: core.blend_scores(arrays, value) for key, value in score_rules.items()}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    seeds = pd.read_csv(seed_path, encoding="utf-8-sig")
    year_return_columns = [f"{fold['id']}_cumulative_return" for fold in protocol["year_folds"]]
    seeds["positive_years"] = (seeds[year_return_columns] > 0).sum(axis=1)
    seeds = seeds.sort_values(["positive_years", "min_year_sharpe", "full_sharpe", "case_id"], ascending=[False, False, False, True]).head(int(protocol["seed_source"]["count"]))
    grid = protocol["parameter_grid"]
    rows = []
    for _, seed in seeds.iterrows():
        if str(seed.blend_id) not in scores:
            raise RuntimeError(f"missing frozen score rule: {seed.blend_id}")
        for min_hold in grid["min_hold"]:
            for max_hold in grid["max_hold"]:
                for sell_rank_below in grid["sell_rank_below"]:
                    for replacement_advantage in grid["replacement_advantage"]:
                        for invested_ratio in grid["invested_ratio"]:
                            profile = core.Profile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), min_hold, max_hold, sell_rank_below, replacement_advantage, invested_ratio)
                            daily = engine.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231")
                            result = yearly.evaluate(daily, protocol["year_folds"])
                            year_returns = [float(result[column]) for column in year_return_columns]
                            rows.append({"case_id": profile.profile_id, "seed_case_id": str(seed.case_id), **asdict(profile), **result, "min_year_cumulative_return": min(year_returns)})
    result = pd.DataFrame(rows).drop_duplicates("case_id")
    result["eligible"] = (result.full_max_drawdown <= float(protocol["full_max_drawdown_max"])) & (result.full_trades >= int(protocol["full_trades_min"]))
    result = result.sort_values(["all_year_positive", "min_year_cumulative_return", "min_year_sharpe", "median_year_sharpe", "full_sharpe", "full_max_drawdown", "case_id"], ascending=[False, False, False, False, False, True, True])
    result.to_csv(GRID_PATH, index=False, encoding="utf-8-sig")
    frozen = result[result.eligible].head(int(protocol["promote_for_juejin"]))
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "seed_source_sha256": digest(seed_path), "grid_sha256": digest(GRID_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    actions_out, seen = [], set()
    for item in payload["profiles"]:
        profile = core.Profile(**{key: item[key] for key in core.Profile.__dataclass_fields__})
        _, actions = engine.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231", record_actions=True)
        action_key = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if action_key in seen:
            continue
        seen.add(action_key)
        path = ACTION_DIR / f"{profile.profile_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        actions_out.append({"case_id": profile.profile_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": actions_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "seed_count": len(seeds), "grid_cases": len(result), "all_year_positive": int(result.all_year_positive.sum()), "frozen_candidates": len(frozen), "unique_juejin_paths": len(actions_out), "best": frozen.head(1).to_dict("records"), "known_2026_used_for_selection": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
