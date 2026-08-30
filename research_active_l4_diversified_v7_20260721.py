from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_best_corrected_v6_20260721 as engine
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_diversified_v7_20260721"
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


def period_metrics(daily):
    selection = core.metrics(daily, "20220606", "20241231")
    confirmation = core.metrics(daily, "20250101", "20251231")
    full = core.metrics(daily, "20220606", "20251231")
    return selection, confirmation, full


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("frozen protocol or input drift")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {item["id"]: core.blend_scores(arrays, item) for item in protocol["score_grid"]}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    fixed = protocol["stage1"]["fixed"]
    rows = []
    for blend_id in scores:
        for amount_min in protocol["stage1"]["amount_min"]:
            for mv_min in protocol["stage1"]["total_mv_min"]:
                for top_n in protocol["stage1"]["top_n"]:
                    for entry_min in protocol["stage1"]["entry_rank_min"]:
                        profile = core.Profile(blend_id, amount_min, mv_min, top_n, entry_min, **fixed)
                        daily = engine.simulate(arrays, scores[blend_id], orders[blend_id], profile, "20251231")
                        selection, confirmation, full = period_metrics(daily)
                        rows.append({"profile_id": profile.profile_id, **asdict(profile), **{f"selection_{k}": v for k, v in selection.items()}, **{f"confirmation_{k}": v for k, v in confirmation.items()}, **{f"full_{k}": v for k, v in full.items()}, "min_period_sharpe": min(selection["sharpe"], confirmation["sharpe"])})
    stage1 = pd.DataFrame(rows)
    gates = protocol["stage1"].get("promotion_filter_values", {})
    max_drawdown = float(gates.get("full_max_drawdown_max", 0.50))
    min_trades = int(gates.get("full_trades_min", 100))
    eligible = stage1[
        (stage1.selection_cumulative_return > 0)
        & (stage1.confirmation_cumulative_return > 0)
        & (stage1.full_max_drawdown <= max_drawdown)
        & (stage1.full_trades >= min_trades)
    ]
    base = eligible.sort_values(["min_period_sharpe", "full_sharpe", "full_max_drawdown", "profile_id"], ascending=[False, False, True, True]).head(protocol["stage1"]["promote"])
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    stage2_rows = []
    grid = protocol["stage2"]
    for _, seed in base.iterrows():
        for min_hold in grid["min_hold"]:
            for max_hold in grid["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for sell_below in grid["sell_rank_below"]:
                    for advantage in grid["replacement_advantage"]:
                        for invested in grid["invested_ratio"]:
                            profile = core.Profile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), min_hold, max_hold, sell_below, advantage, invested)
                            daily = engine.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231")
                            selection, confirmation, full = period_metrics(daily)
                            stage2_rows.append({"case_id": profile.profile_id, **asdict(profile), **{f"selection_{k}": v for k, v in selection.items()}, **{f"confirmation_{k}": v for k, v in confirmation.items()}, **{f"full_{k}": v for k, v in full.items()}, "min_period_sharpe": min(selection["sharpe"], confirmation["sharpe"])})
    stage2 = pd.DataFrame(stage2_rows).drop_duplicates("case_id")
    stage2["both_positive"] = (stage2.selection_cumulative_return > 0) & (stage2.confirmation_cumulative_return > 0)
    stage2 = stage2.sort_values(["both_positive", "min_period_sharpe", "full_sharpe", "full_max_drawdown", "full_turnover", "case_id"], ascending=[False, False, False, True, True, True])
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = stage2[stage2.both_positive].head(grid["promote_for_juejin"])
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH), "stage2_sha256": digest(STAGE2_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    actions_out = []
    seen = set()
    for item in payload["profiles"]:
        profile = core.Profile(**{key: item[key] for key in core.Profile.__dataclass_fields__})
        _, actions = engine.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231", record_actions=True)
        raw = actions.to_csv(index=False).encode("utf-8")
        action_key = hashlib.sha256(raw).hexdigest()
        if action_key in seen:
            continue
        seen.add(action_key)
        path = ACTION_DIR / f"{profile.profile_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        actions_out.append({"case_id": profile.profile_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": actions_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only", "stage1_cases": len(stage1), "eligible_stage1": len(eligible), "base_profiles": len(base), "stage2_cases": len(stage2), "frozen_candidates": len(frozen), "unique_juejin_paths": len(actions_out), "best": frozen.head(1).to_dict("records"), "validation_not_opened": True, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
