from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_active_l4_best_corrected_v6_20260721 as engine
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_yearly_robust_v11_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evaluate(daily: pd.DataFrame, folds: list[dict]) -> dict:
    result = {}
    year_sharpes = []
    year_positive = []
    for fold in folds:
        metrics = core.metrics(daily, fold["start"], fold["end"])
        prefix = fold["id"]
        result.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        year_sharpes.append(float(metrics["sharpe"]))
        year_positive.append(float(metrics.get("cumulative_return", 0.0)) > 0.0)
    full = core.metrics(daily, folds[0]["start"], folds[-1]["end"])
    result.update({f"full_{key}": value for key, value in full.items()})
    result["all_year_positive"] = all(year_positive)
    result["min_year_sharpe"] = min(year_sharpes)
    result["median_year_sharpe"] = float(np.median(year_sharpes))
    return result


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("frozen protocol or input drift")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {item["id"]: core.blend_scores(arrays, item) for item in protocol["score_grid"]}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    folds = protocol["year_folds"]
    fixed = protocol["stage1"]["fixed"]
    rows = []
    for blend_id in scores:
        for amount_min in protocol["stage1"]["amount_min"]:
            for mv_min in protocol["stage1"]["total_mv_min"]:
                for top_n in protocol["stage1"]["top_n"]:
                    for entry_rank_min in protocol["stage1"]["entry_rank_min"]:
                        profile = core.Profile(blend_id, amount_min, mv_min, top_n, entry_rank_min, **fixed)
                        daily = engine.simulate(arrays, scores[blend_id], orders[blend_id], profile, protocol["observation_end"])
                        rows.append({"case_id": profile.profile_id, **asdict(profile), **evaluate(daily, folds)})
    stage1 = pd.DataFrame(rows)
    gate = protocol["stage1"]
    stage1["eligible"] = stage1.all_year_positive & (stage1.full_max_drawdown <= float(gate["full_max_drawdown_max"])) & (stage1.full_trades >= int(gate["full_trades_min"]))
    stage1 = stage1.sort_values(["eligible", "min_year_sharpe", "median_year_sharpe", "full_sharpe", "full_max_drawdown", "case_id"], ascending=[False, False, False, False, True, True])
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    seeds = stage1[stage1.eligible].head(int(gate["promote"]))
    stage2_rows = []
    grid = protocol["stage2"]
    for _, seed in seeds.iterrows():
        for min_hold in grid["min_hold"]:
            for max_hold in grid["max_hold"]:
                for sell_rank_below in grid["sell_rank_below"]:
                    for replacement_advantage in grid["replacement_advantage"]:
                        for invested_ratio in grid["invested_ratio"]:
                            profile = core.Profile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), min_hold, max_hold, sell_rank_below, replacement_advantage, invested_ratio)
                            daily = engine.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, protocol["observation_end"])
                            stage2_rows.append({"case_id": profile.profile_id, **asdict(profile), **evaluate(daily, folds)})
    stage2 = pd.DataFrame(stage2_rows).drop_duplicates("case_id")
    if stage2.empty:
        stage2 = pd.DataFrame(columns=["case_id", "all_year_positive", "min_year_sharpe", "median_year_sharpe", "full_sharpe", "full_max_drawdown", "full_turnover"])
    stage2 = stage2.sort_values(["all_year_positive", "min_year_sharpe", "median_year_sharpe", "full_sharpe", "full_max_drawdown", "full_turnover", "case_id"], ascending=[False, False, False, False, True, True, True])
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = stage2[stage2.all_year_positive].head(int(grid["promote_for_juejin"]))
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH), "stage2_sha256": digest(STAGE2_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    actions_out, seen = [], set()
    for item in payload["profiles"]:
        profile = core.Profile(**{key: item[key] for key in core.Profile.__dataclass_fields__})
        _, actions = engine.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, protocol["observation_end"], record_actions=True)
        action_key = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if action_key in seen:
            continue
        seen.add(action_key)
        path = ACTION_DIR / f"{profile.profile_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        actions_out.append({"case_id": profile.profile_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": actions_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only_observation", "stage1_cases": len(stage1), "eligible_stage1": int(stage1.eligible.sum()), "stage2_cases": len(stage2), "all_year_positive_stage2": int(stage2.all_year_positive.sum()) if not stage2.empty else 0, "frozen_candidates": len(frozen), "unique_juejin_paths": len(actions_out), "best": frozen.head(1).to_dict("records"), "known_2026_used_for_selection": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
