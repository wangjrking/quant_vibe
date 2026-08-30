# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_diversified_equalweight_v77_20260722 as v77
import research_active_l4_fast_health_neighborhood_v75_20260722 as health_lib
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_lagged_health_liquidity_v66_20260721 as quality_lib
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_diversified_exit_v78_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"


def stable_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def local_gate(frame: pd.DataFrame, gate: dict) -> pd.Series:
    return (
        frame["robust_positive"]
        & (frame["full_linear_annual_proxy"] >= gate["linear_annual_proxy_min"])
        & (frame["full_sharpe"] >= gate["sharpe_min"])
        & (frame["full_max_drawdown"] <= gate["max_drawdown_max"])
        & (frame["full_trades"] >= gate["trades_min"])
        & (frame["full_invested_ratio_mean"] >= gate["invested_ratio_min"])
    )


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    dependency = ROOT / protocol["dependency"]["path"]
    if protocol["status"] != "frozen_research_only":
        raise RuntimeError("研究协议未冻结")
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("研究代码哈希不一致")
    if metrics_lib.digest(dependency) != protocol["dependency"]["sha256"]:
        raise RuntimeError("V77依赖哈希不一致")
    if metrics_lib.digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("输入缓存哈希不一致")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = robust.truncate_observation(
            {key: saved[key] for key in saved.files}, protocol["observation_end"]
        )

    state_profile = protocol["fixed_state"]
    base_state = robust.build_state(
        arrays,
        protocol["score_weights"],
        state_profile["agreement"],
        state_profile["entry"]["risk_off"],
        state_profile["entry"]["risk_on"],
        protocol["fixed"],
    )
    liquidity_profile = v77.PortfolioProfile(
        3,
        1.0,
        int(protocol["fixed"]["amount_min"]),
        int(protocol["fixed"]["mv_min"]),
        float(protocol["fixed"]["turnover_max"]),
    )
    base_state = v77.apply_liquidity(base_state, arrays, liquidity_profile)
    quality = quality_lib.candidate_quality(
        arrays,
        base_state,
        int(protocol["health_contract"]["shadow_top_n"]),
        float(protocol["health_contract"]["round_trip_cost"]),
    )

    state_cache: dict[str, tuple] = {}
    rows: list[dict] = []
    definitions: dict[str, dict] = {}
    for seed in protocol["seed_profiles"]:
        health = health_lib.fast_health_exposure(
            quality,
            int(seed["lookback"]),
            float(seed["mean_threshold"]),
            float(seed["positive_fraction"]),
            float(seed["weak_floor"]),
        )
        state_cache[seed["id"]] = health_lib.state_with_health(base_state, health)
        for gross_exposure, min_hold, max_hold, sell_rank, advantage in product(
            protocol["grid"]["gross_exposure"],
            protocol["grid"]["min_hold"],
            protocol["grid"]["max_hold"],
            protocol["grid"]["sell_rank_below"],
            protocol["grid"]["replacement_advantage"],
        ):
            if int(max_hold) < int(min_hold):
                continue
            portfolio = v77.PortfolioProfile(
                3,
                float(gross_exposure),
                int(protocol["fixed"]["amount_min"]),
                int(protocol["fixed"]["mv_min"]),
                float(protocol["fixed"]["turnover_max"]),
            )
            exit_profile = sell_lib.ExitProfile(
                "rank_5d",
                int(min_hold),
                int(max_hold),
                float(sell_rank),
                float(advantage),
            )
            definition = {
                "seed": seed,
                "portfolio": asdict(portfolio),
                "exit": asdict(exit_profile),
            }
            case_id = "v78_" + stable_id(definition)
            definitions[case_id] = definition
            daily = v77.simulate(
                arrays,
                state_cache[seed["id"]],
                portfolio,
                exit_profile,
                protocol["observation_end"],
            )
            rows.append(
                {
                    "case_id": case_id,
                    "seed_id": seed["id"],
                    "lookback": int(seed["lookback"]),
                    "mean_threshold": float(seed["mean_threshold"]),
                    "positive_fraction": float(seed["positive_fraction"]),
                    "weak_floor": float(seed["weak_floor"]),
                    **asdict(portfolio),
                    **asdict(exit_profile),
                    **robust.evaluate_robust(daily, protocol),
                }
            )

    results = pd.DataFrame(rows)
    results["eligible"] = local_gate(results, protocol["gate"])
    results = results.sort_values(
        [
            "eligible",
            "robust_sharpe_floor",
            "min_year_sharpe",
            "full_sharpe",
            "full_linear_annual_proxy",
            "case_id",
        ],
        ascending=[False, False, False, False, False, True],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "grid_results.csv", index=False, encoding="utf-8-sig")
    frozen = results[results["eligible"]].head(int(protocol["grid"]["freeze_count"]))

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows: list[dict] = []
    seen: set[str] = set()
    for _, item in frozen.iterrows():
        definition = definitions[str(item.case_id)]
        portfolio = v77.PortfolioProfile(**definition["portfolio"])
        exit_profile = sell_lib.ExitProfile(**definition["exit"])
        _, actions = v77.simulate(
            arrays,
            state_cache[str(item.seed_id)],
            portfolio,
            exit_profile,
            protocol["observation_end"],
            record_actions=True,
        )
        content_hash = hashlib.sha256(
            actions.to_csv(index=False).encode("utf-8")
        ).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append(
            {
                "case_id": str(item.case_id),
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": metrics_lib.digest(path),
                "rows": len(actions),
                "buy_rows": int((actions.action == "BUY").sum()),
                "max_positions": 3,
                "per_name_target_pct": portfolio.gross_exposure / 3.0,
            }
        )

    (OUT / "frozen_candidates.json").write_text(
        json.dumps(
            {
                "protocol_sha256": metrics_lib.digest(PROTOCOL),
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "profiles": frozen.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "juejin_action_manifest.json").write_text(
        json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "status": (
            "research_only_observation_candidates_frozen"
            if action_rows
            else "research_only_stopped_at_local_gate"
        ),
        "grid_cases": len(results),
        "eligible_cases": int(results.eligible.sum()),
        "frozen_candidates": len(frozen),
        "unique_juejin_paths": len(action_rows),
        "known_2026_used": False,
        "production_changed": False,
    }
    (OUT / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
