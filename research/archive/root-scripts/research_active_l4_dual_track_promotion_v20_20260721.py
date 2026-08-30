# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_agreement_v8_20260721 as agreement
import research_active_l4_agreement_health_v16_20260721 as candidate_health
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_dual_track_promotion_v20_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"
REPORT_PATH = REPORT_DIR / "双轨晋级研究说明.md"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    source_path = ROOT / protocol["seed_source"]["path"]
    source_protocol_path = ROOT / protocol["source_protocol"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(source_path) != protocol["seed_source"]["sha256"] or digest(source_protocol_path) != protocol["source_protocol"]["sha256"]:
        raise RuntimeError("冻结协议或V19源证据发生漂移")
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(source_path, encoding="utf-8-sig")
    pool = candidates[candidates.eligible & candidates.all_year_positive].copy()
    return_track = pool.sort_values(["full_cumulative_return", "full_sharpe", "full_max_drawdown", "case_id"], ascending=[False, False, True, True]).head(int(protocol["return_track"]["count"]))
    sharpe_track = pool.sort_values(["full_sharpe", "full_cumulative_return", "full_max_drawdown", "case_id"], ascending=[False, False, True, True]).head(int(protocol["sharpe_track"]["count"]))
    selected = pd.concat([return_track.assign(promotion_track="return"), sharpe_track.assign(promotion_track="sharpe")], ignore_index=True).drop_duplicates("case_id")
    cache = ROOT / source_protocol["input_cache"]["path"]
    if digest(cache) != source_protocol["input_cache"]["sha256"]:
        raise RuntimeError("输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score = core.blend_scores(arrays, source_protocol["score"])
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
    masked = agreement.masked_arrays(arrays, source_protocol["agreement"])
    shadow_values = sorted({int(value) for value in selected.shadow_top_n})
    qualities = {top_n: candidate_health.candidate_quality(masked, score, order, top_n) for top_n in shadow_values}
    exposure_keys = {(int(row.shadow_top_n), int(row.health_lookback), float(row.floor_fraction), float(row.scale_denominator)) for _, row in selected.iterrows()}
    exposures = {key: continuous.exposure_series(qualities[key[0]], key[1], key[2], key[3]) for key in exposure_keys}
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    selected_rows = []
    for _, row in selected.iterrows():
        profile = continuous.ExposureProfile(**{key: row[key] for key in continuous.ExposureProfile.__dataclass_fields__})
        exposure_key = (int(row.shadow_top_n), profile.health_lookback, profile.floor_fraction, profile.scale_denominator)
        _, actions = continuous.simulate(masked, score, order, exposures[exposure_key], profile, source_protocol["observation_end"], record_actions=True)
        action_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        record = row.to_dict()
        record["action_content_sha256"] = action_hash
        selected_rows.append(record)
        if action_hash in seen:
            continue
        seen.add(action_hash)
        path = ACTION_DIR / f"{row.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": row.case_id, "promotion_track": row.promotion_track, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "content_sha256": action_hash, "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    FROZEN_PATH.write_text(json.dumps({"protocol_sha256": digest(PROTOCOL_PATH), "source_stage2_sha256": digest(source_path), "profiles": selected_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(["# 双轨晋级研究说明", "", f"- V19逐年全正候选池：{len(pool)} 组。", f"- 收益轨冻结：{len(return_track)} 组。", f"- Sharpe轨冻结：{len(sharpe_track)} 组。", f"- 去重后参数候选：{len(selected)} 组。", f"- 去重后交易路径：{len(action_rows)} 条。", "- 本轮不新增参数，不读取2026验证期。", "- 本地结果只用于冻结晋级对象，正式收益以掘金为准。"]), encoding="utf-8")
    print(json.dumps({"pool": len(pool), "selected": len(selected), "unique_paths": len(action_rows), "known_2026_used_for_selection": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
