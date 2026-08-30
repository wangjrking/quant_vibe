# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_agreement_v8_20260721 as agreement
import research_active_l4_agreement_health_v16_20260721 as candidate_health
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_platform_breadth_v23_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
FROZEN_PATH = REPORT_DIR / "frozen_platform_candidates.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"
REPORT_PATH = REPORT_DIR / "平台宽评估研究说明.md"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    stage2_path = ROOT / protocol["source_stage2"]["path"]
    source_protocol_path = ROOT / protocol["source_protocol"]["path"]
    if protocol["status"] != "frozen_research_only":
        raise RuntimeError("协议未冻结")
    if digest(stage2_path) != protocol["source_stage2"]["sha256"]:
        raise RuntimeError("V22结果发生漂移")
    if digest(source_protocol_path) != protocol["source_protocol"]["sha256"]:
        raise RuntimeError("V22协议发生漂移")

    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    cache = ROOT / source_protocol["input_cache"]["path"]
    if digest(cache) != source_protocol["input_cache"]["sha256"]:
        raise RuntimeError("正式L4排名缓存发生漂移")
    stage2 = pd.read_csv(stage2_path)
    eligible = stage2[
        stage2["eligible"].astype(bool)
        & stage2["all_year_positive"].astype(bool)
        & (stage2["full_max_drawdown"] <= 0.40)
        & (stage2["full_trades"] >= 80)
    ].copy()

    selection = protocol["selection"]
    pieces = [
        eligible.sort_values(
            ["full_cumulative_return", "full_sharpe", "case_id"],
            ascending=[False, False, True]
        ).head(int(selection["top_return"])).assign(selection_track="return"),
        eligible.sort_values(
            ["full_sharpe", "full_cumulative_return", "case_id"],
            ascending=[False, False, True]
        ).head(int(selection["top_sharpe"])).assign(selection_track="sharpe"),
    ]
    per_group = int(selection["stratified_return_top_each"])
    for field in selection["stratify_fields"]:
        for value, group in eligible.groupby(field, dropna=False):
            pieces.append(group.sort_values(
                ["full_cumulative_return", "full_sharpe", "case_id"],
                ascending=[False, False, True]
            ).head(per_group).assign(selection_track=f"stratified_{field}_{value}"))
    frozen = pd.concat(pieces, ignore_index=True).drop_duplicates("case_id")

    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {
        rule["id"]: core.blend_scores(arrays, rule)
        for rule in source_protocol["score_grid"]
    }
    orders = {
        key: np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
        for key, score in scores.items()
    }
    masked = agreement.masked_arrays(arrays, source_protocol["agreement"])
    qualities = {
        key: candidate_health.candidate_quality(
            masked, score, orders[key], int(source_protocol["health_to_exposure"]["shadow_top_n"])
        ) for key, score in scores.items()
    }
    needed_exposures = {
        (str(row.blend_id), int(row.health_lookback), float(row.floor_fraction), float(row.scale_denominator))
        for _, row in frozen.iterrows()
    }
    exposures = {
        key: continuous.exposure_series(qualities[key[0]], key[1], key[2], key[3])
        for key in needed_exposures
    }

    payload = {
        "protocol_sha256": digest(PROTOCOL_PATH),
        "source_stage2_sha256": digest(stage2_path),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eligible_rows": len(eligible),
        "selected_profiles": frozen.to_dict("records"),
    }
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for _, item in frozen.iterrows():
        profile = continuous.ExposureProfile(**{
            key: item[key] for key in continuous.ExposureProfile.__dataclass_fields__
        })
        _, actions = continuous.simulate(
            masked, scores[profile.blend_id], orders[profile.blend_id],
            exposures[(profile.blend_id, profile.health_lookback,
                       profile.floor_fraction, profile.scale_denominator)],
            profile, protocol["observation_end"], record_actions=True
        )
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = ACTION_DIR / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({
            "case_id": item.case_id,
            "selection_track": item.selection_track,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(path),
            "content_sha256": content_hash,
            "rows": len(actions),
            "buy_rows": int((actions.action == "BUY").sum()),
            "sell_rows": int((actions.action == "SELL").sum()),
        })
    ACTION_MANIFEST.write_text(json.dumps({
        "frozen_sha256": digest(FROZEN_PATH),
        "actions": action_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join([
        "# 平台宽评估研究说明", "",
        f"- V22合格候选：{len(eligible)} 组。",
        f"- 分层冻结参数：{len(frozen)} 组。",
        f"- 去重后掘金路径：{len(action_rows)} 条。",
        "- 未使用2026数据，不修改生产策略。",
        "- 该轮不新增参数，仅扩大观察期平台评估覆盖。"
    ]), encoding="utf-8")
    print(json.dumps({
        "status": "research_only_observation",
        "eligible_rows": len(eligible),
        "frozen_profiles": len(frozen),
        "unique_juejin_paths": len(action_rows),
        "known_2026_used_for_selection": False,
        "production_changed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
