# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_agreement_v8_20260721 as agreement
import research_active_l4_agreement_health_v16_20260721 as candidate_health
import research_active_l4_continuous_exposure_v18_20260721 as continuous
import research_active_l4_yearly_robust_v11_20260721 as yearly


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_nonlinear_consensus_v25_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
REPORT_PATH = REPORT_DIR / "非线性共识评分研究结论.md"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deterministic_percentile(raw):
    result = np.full(raw.shape, np.nan, dtype=np.float32)
    stock_index = np.arange(raw.shape[1], dtype=np.int32)
    for t in range(raw.shape[0]):
        valid = np.isfinite(raw[t])
        indices = stock_index[valid]
        if not len(indices):
            continue
        order = np.lexsort((indices, raw[t, indices]))
        ordered = indices[order]
        if len(ordered) == 1:
            result[t, ordered] = 1.0
        else:
            result[t, ordered] = np.arange(len(ordered), dtype=np.float32) / float(len(ordered) - 1)
    return result


def build_score(arrays, formula):
    r3 = np.clip(arrays["rank_3d"], 1e-6, 1.0)
    r5 = np.clip(arrays["rank_5d"], 1e-6, 1.0)
    r10 = np.clip(arrays["rank_10d"], 1e-6, 1.0)
    kind = formula["type"]
    if kind == "arithmetic":
        raw = formula["w3"] * r3 + formula["w5"] * r5 + formula["w10"] * r10
    elif kind == "geometric":
        raw = np.exp(
            formula["w3"] * np.log(r3)
            + formula["w5"] * np.log(r5)
            + formula["w10"] * np.log(r10)
        )
    elif kind == "minimum":
        raw = np.minimum(np.minimum(r3, r5), r10)
    elif kind == "long_plus_consensus":
        consensus = np.sqrt(r3 * r5)
        raw = formula["long_weight"] * r10 + formula["consensus_weight"] * consensus
    else:
        raise ValueError(f"未知评分公式: {kind}")
    return deterministic_percentile(raw.astype(np.float32))


def case_id(profile, formula_id, agreement_id):
    payload = {"profile": asdict(profile), "formula_id": formula_id, "agreement_id": agreement_id}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "nc_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evaluate(daily, folds):
    result = yearly.evaluate(daily, folds)
    result["min_year_cumulative_return"] = min(
        float(result[f"{fold['id']}_cumulative_return"]) for fold in folds
    )
    return result


def dual_track(frame, return_count, sharpe_count):
    eligible = frame[frame.eligible & frame.all_year_positive]
    by_return = eligible.sort_values(
        ["full_cumulative_return", "full_sharpe", "min_year_cumulative_return", "case_id"],
        ascending=[False, False, False, True]
    ).head(return_count).assign(promotion_track="return")
    by_sharpe = eligible.sort_values(
        ["full_sharpe", "full_cumulative_return", "min_year_cumulative_return", "case_id"],
        ascending=[False, False, False, True]
    ).head(sharpe_count).assign(promotion_track="sharpe")
    return pd.concat([by_return, by_sharpe], ignore_index=True).drop_duplicates("case_id")


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("冻结协议或输入缓存发生漂移")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}

    scores = {formula["id"]: build_score(arrays, formula) for formula in protocol["score_formulas"]}
    orders = {
        key: np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
        for key, score in scores.items()
    }
    gates = {item["id"]: item for item in protocol["agreement_grid"]}
    masked = {key: agreement.masked_arrays(arrays, item) for key, item in gates.items()}
    health = protocol["health_to_exposure"]
    exposures = {}
    for formula_id, score in scores.items():
        for agreement_id, masked_arrays in masked.items():
            quality = candidate_health.candidate_quality(
                masked_arrays, score, orders[formula_id], int(health["shadow_top_n"])
            )
            exposures[(formula_id, agreement_id)] = continuous.exposure_series(
                quality, int(health["rolling_lookback"]),
                float(health["floor_fraction"]), float(health["scale_denominator"])
            )

    rows = []
    stage1_grid = protocol["stage1"]
    fixed = stage1_grid["fixed"]
    for formula_id in scores:
        for agreement_id in gates:
            for amount_min in stage1_grid["amount_min"]:
                for mv_min in stage1_grid["total_mv_min"]:
                    for top_n in stage1_grid["top_n"]:
                        for entry_rank_min in stage1_grid["entry_rank_min"]:
                            profile = continuous.ExposureProfile(
                                formula_id, amount_min, mv_min, top_n, entry_rank_min,
                                int(health["rolling_lookback"]), float(health["floor_fraction"]),
                                float(health["scale_denominator"]), **fixed
                            )
                            daily = continuous.simulate(
                                masked[agreement_id], scores[formula_id], orders[formula_id],
                                exposures[(formula_id, agreement_id)], profile,
                                protocol["observation_end"]
                            )
                            rows.append({
                                "case_id": case_id(profile, formula_id, agreement_id),
                                "formula_id": formula_id, "agreement_id": agreement_id,
                                **asdict(profile), **evaluate(daily, protocol["year_folds"])
                            })
    stage1 = pd.DataFrame(rows)
    stage1["eligible"] = (stage1.full_max_drawdown <= 0.40) & (stage1.full_trades >= 80)
    stage1 = stage1.sort_values(
        ["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"],
        ascending=[False, False, False, True]
    )
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    seeds = dual_track(stage1, int(stage1_grid["promote_return"]), int(stage1_grid["promote_sharpe"]))

    rows = []
    for _, seed in seeds.iterrows():
        for exit_rule in protocol["stage2_exit_profiles"]:
            profile = continuous.ExposureProfile(
                str(seed.formula_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n),
                float(seed.entry_rank_min), int(health["rolling_lookback"]),
                float(health["floor_fraction"]), float(health["scale_denominator"]),
                int(exit_rule["min_hold"]), int(exit_rule["max_hold"]),
                float(exit_rule["sell_rank_below"]), float(exit_rule["replacement_advantage"]), 1.0
            )
            daily = continuous.simulate(
                masked[str(seed.agreement_id)], scores[profile.blend_id], orders[profile.blend_id],
                exposures[(profile.blend_id, str(seed.agreement_id))], profile,
                protocol["observation_end"]
            )
            rows.append({
                "case_id": case_id(profile, profile.blend_id, str(seed.agreement_id)),
                "seed_case_id": str(seed.case_id), "formula_id": profile.blend_id,
                "agreement_id": str(seed.agreement_id), "exit_profile_id": exit_rule["id"],
                **asdict(profile), **evaluate(daily, protocol["year_folds"])
            })
    stage2 = pd.DataFrame(rows).drop_duplicates("case_id")
    stage2["eligible"] = (stage2.full_max_drawdown <= 0.40) & (stage2.full_trades >= 80)
    stage2 = stage2.sort_values(
        ["all_year_positive", "full_cumulative_return", "full_sharpe", "case_id"],
        ascending=[False, False, False, True]
    )
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = dual_track(
        stage2, int(protocol["stage2"]["promote_return"]),
        int(protocol["stage2"]["promote_sharpe"])
    )

    payload = {
        "protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH),
        "stage2_sha256": digest(STAGE2_PATH),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profiles": frozen.to_dict("records")
    }
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    action_rows, seen = [], set()
    for item in payload["profiles"]:
        profile = continuous.ExposureProfile(**{
            key: item[key] for key in continuous.ExposureProfile.__dataclass_fields__
        })
        _, actions = continuous.simulate(
            masked[item["agreement_id"]], scores[profile.blend_id], orders[profile.blend_id],
            exposures[(profile.blend_id, item["agreement_id"])], profile,
            protocol["observation_end"], record_actions=True
        )
        content_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = ACTION_DIR / f"{item['case_id']}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({
            "case_id": item["case_id"], "promotion_track": item["promotion_track"],
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(path), "content_sha256": content_hash,
            "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()),
            "sell_rows": int((actions.action == "SELL").sum())
        })
    ACTION_MANIFEST.write_text(json.dumps({
        "frozen_sha256": digest(FROZEN_PATH), "actions": action_rows
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "research_only_observation",
        "stage1_cases": len(stage1), "stage1_all_year_positive": int(stage1.all_year_positive.sum()),
        "stage2_cases": len(stage2), "stage2_all_year_positive": int(stage2.all_year_positive.sum()),
        "frozen_candidates": len(frozen), "unique_juejin_paths": len(action_rows),
        "best_return": frozen.sort_values("full_cumulative_return", ascending=False).head(1).to_dict("records"),
        "best_sharpe": frozen.sort_values("full_sharpe", ascending=False).head(1).to_dict("records"),
        "known_2026_used_for_selection": False, "production_changed": False
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join([
        "# 非线性共识评分研究结论", "",
        "- 参数选择仅使用 `20220606-20251231`。",
        "- 所有复合分数按T日日截面确定性重排。",
        "- 2026数据未参与筛选。",
        "- 本地只作预筛，正式结果以掘金为准。", "",
        f"第一阶段 {len(stage1)} 组，逐年为正 {int(stage1.all_year_positive.sum())} 组。",
        f"第二阶段 {len(stage2)} 组，逐年为正 {int(stage2.all_year_positive.sum())} 组。",
        f"冻结候选 {len(frozen)} 组，独立交易路径 {len(action_rows)} 条。", "",
        "本轮为research-only，未修改生产策略或正式信号。"
    ]), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in summary.items()
        if key not in {"best_return", "best_sharpe"}
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
