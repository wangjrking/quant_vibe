# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_diversified_v79_2026_20260722"
PROTOCOL = OUT / "frozen_validation_protocol.json"


def slice_time(value, start, total_dates):
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == total_dates:
        return value[start:]
    return value


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_2026_validation_only":
        raise RuntimeError("2026验证协议未冻结")
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("2026验证代码哈希不一致")
    cache = ROOT / protocol["input_cache"]["path"]
    if metrics_lib.digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("输入缓存哈希不一致")

    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    dates = arrays["dates"].astype(str)
    total_dates = len(dates)

    source_protocol = json.loads((ROOT / protocol["source_protocol"]["path"]).read_text(encoding="utf-8"))
    fixed = protocol["fixed_rule"]
    source_state = source_protocol["fixed_state"]
    base_state = robust.build_state(
        arrays,
        source_protocol["score_weights"],
        source_state["agreement"],
        source_state["entry"]["risk_off"],
        source_state["entry"]["risk_on"],
        source_protocol["fixed"],
    )
    portfolio = v77.PortfolioProfile(
        3,
        0.8,
        int(fixed["amount_min"]),
        int(fixed["mv_min"]),
        float(fixed["turnover_max"]),
    )
    base_state = v77.apply_liquidity(base_state, arrays, portfolio)
    quality = quality_lib.candidate_quality(
        arrays,
        base_state,
        int(source_protocol["health_contract"]["shadow_top_n"]),
        float(source_protocol["health_contract"]["round_trip_cost"]),
    )
    health = health_lib.fast_health_exposure(
        quality,
        int(fixed["lookback"]),
        float(fixed["mean_threshold"]),
        float(fixed["positive_fraction"]),
        float(fixed["weak_floor"]),
    )
    state = health_lib.state_with_health(base_state, health)
    exit_profile = sell_lib.ExitProfile(
        "rank_5d",
        int(fixed["min_hold"]),
        int(fixed["max_hold"]),
        float(fixed["sell_rank_below"]),
        float(fixed["replacement_advantage"]),
    )

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for buy_start in protocol["empty_start_buy_dates"]:
        matches = np.where(dates[1:] == buy_start)[0]
        if len(matches) != 1:
            raise RuntimeError("无法唯一定位空仓启动日: {}".format(buy_start))
        signal_index = int(matches[0])
        sliced_arrays = {
            key: slice_time(value, signal_index, total_dates)
            for key, value in arrays.items()
        }
        score, order, masked, exposure = state
        sliced_state = (
            score[signal_index:],
            order[signal_index:],
            {key: value[signal_index:] for key, value in masked.items()},
            exposure[signal_index:],
        )
        daily, actions = v77.simulate(
            sliced_arrays,
            sliced_state,
            portfolio,
            exit_profile,
            protocol["validation_end"],
            record_actions=True,
        )
        action_path = action_dir / "{}_actions.csv".format(buy_start)
        actions.to_csv(action_path, index=False, encoding="utf-8-sig")
        local = v77.core.metrics(daily, buy_start, protocol["validation_end"])
        summaries.append(
            {
                "buy_start": buy_start,
                "signal_start": str(dates[signal_index]),
                "action_path": str(action_path.relative_to(ROOT)).replace("\\", "/"),
                "action_sha256": metrics_lib.digest(action_path),
                "action_rows": len(actions),
                "buy_rows": int((actions["action"] == "BUY").sum()) if len(actions) else 0,
                "local_cumulative_return": local["cumulative_return"],
                "local_linear_annual_proxy": local["linear_annual_proxy"],
                "local_sharpe": local["sharpe"],
                "local_max_drawdown": local["max_drawdown"],
            }
        )

    result = {
        "status": "frozen_2026_validation_actions_generated",
        "known_2026_used_for_parameter_selection": False,
        "production_changed": False,
        "runs": summaries,
    }
    (OUT / "validation_action_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(
        OUT / "local_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
