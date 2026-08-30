from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_horizon_rotation_preregistration_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1_observation_screen.csv"
STAGE2_PATH = REPORT_DIR / "stage2_observation_screen.csv"
FROZEN_PATH = REPORT_DIR / "frozen_candidates_before_validation.json"
VALIDATION_PATH = REPORT_DIR / "final_validation_once.csv"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-validation", action="store_true")
    return parser.parse_args()


def load_protocol() -> dict:
    p = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if p["status"] != "frozen_research_only":
        raise RuntimeError("protocol not frozen")
    cache = ROOT / p["input_cache"]["path"]
    if digest(cache) != p["input_cache"]["sha256"]:
        raise RuntimeError("input cache hash drift")
    return p


def main() -> None:
    opt = args()
    p = load_protocol()
    cache = ROOT / p["input_cache"]["path"]
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {item["id"]: core.blend_scores(arrays, item) for item in p["score_grid"]}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    fixed = p["stage1"]["fixed"]
    rows = []
    for blend_id in scores:
        for amount_min in p["stage1"]["amount_min"]:
            for mv_min in p["stage1"]["total_mv_min"]:
                for top_n in p["stage1"]["top_n"]:
                    for entry_min in p["stage1"]["entry_rank_min"]:
                        profile = core.Profile(blend_id, amount_min, mv_min, top_n, entry_min, **fixed)
                        daily = core.simulate(arrays, scores[blend_id], orders[blend_id], profile, "20241231")
                        m = core.metrics(daily, "20220606", "20241231")
                        rows.append({"profile_id": profile.profile_id, **asdict(profile), **m, "yearly_all_positive": core.yearly_positive(daily, "20220606", "20241231")})
    stage1 = pd.DataFrame(rows).sort_values(["yearly_all_positive", "sharpe", "max_drawdown", "profile_id"], ascending=[False, False, True, True])
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    promoted = stage1[stage1["yearly_all_positive"]].head(int(p["stage1"]["promote"]))
    stage2_rows = []
    for _, base in promoted.iterrows():
        for min_hold in p["stage2"]["min_hold"]:
            for max_hold in p["stage2"]["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for sell_below in p["stage2"]["sell_rank_below"]:
                    for advantage in p["stage2"]["replacement_advantage"]:
                        for invested in p["stage2"]["invested_ratio"]:
                            profile = core.Profile(str(base["blend_id"]), int(base["amount_min"]), int(base["mv_min"]), int(base["top_n"]), float(base["entry_rank_min"]), min_hold, max_hold, sell_below, advantage, invested)
                            daily = core.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231")
                            selection = core.metrics(daily, "20220606", "20241231")
                            confirm = core.metrics(daily, "20250101", "20251231")
                            stage2_rows.append({"profile_id": profile.profile_id, **asdict(profile), **{f"select_{k}": v for k, v in selection.items()}, **{f"confirm_{k}": v for k, v in confirm.items()}, "selection_min_sharpe": min(selection["sharpe"], confirm["sharpe"])})
    stage2 = pd.DataFrame(stage2_rows)
    if stage2.empty:
        finalists = stage2
    else:
        stage2["confirm_positive"] = stage2["confirm_cumulative_return"] > 0
        stage2 = stage2.sort_values(["confirm_positive", "selection_min_sharpe", "confirm_max_drawdown", "select_turnover", "profile_id"], ascending=[False, False, True, True, True])
        finalists = stage2[stage2["confirm_positive"]].head(int(p["stage2"]["promote_for_juejin"]))
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = {"protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH), "stage2_sha256": digest(STAGE2_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "validation_opened": False, "profiles": finalists.to_dict("records")}
    if not FROZEN_PATH.exists():
        FROZEN_PATH.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        existing = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        if [x["profile_id"] for x in existing["profiles"]] != [x["profile_id"] for x in frozen["profiles"]]:
            raise RuntimeError("frozen candidate drift")
    if opt.open_validation:
        if VALIDATION_PATH.exists():
            raise RuntimeError("validation already opened")
        existing = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        validation = []
        for item in existing["profiles"]:
            profile = core.Profile(**{key: item[key] for key in core.Profile.__dataclass_fields__})
            daily = core.simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "99999999")
            full = core.metrics(daily, "20220606", "99999999")
            valid = core.metrics(daily, "20260101", "99999999")
            validation.append({"profile_id": profile.profile_id, **asdict(profile), **{f"full_{k}": v for k, v in full.items()}, **{f"validation_{k}": v for k, v in valid.items()}})
        pd.DataFrame(validation).to_csv(VALIDATION_PATH, index=False, encoding="utf-8-sig")
        existing.update({"validation_opened": True, "validation_opened_at": datetime.now().astimezone().isoformat(timespec="seconds"), "validation_sha256": digest(VALIDATION_PATH)})
        FROZEN_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only", "protocol_sha256": digest(PROTOCOL_PATH), "stage1_cases": len(stage1), "stage1_promoted": len(promoted), "stage2_cases": len(stage2), "frozen_candidates": len(finalists), "validation_opened_this_run": opt.open_validation, "juejin_run": False, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
