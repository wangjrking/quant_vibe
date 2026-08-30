# -*- coding: utf-8 -*-
from __future__ import annotations

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
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_diversified_v80_frozen_candidates_2026_20260722"
PROTOCOL = OUT / "frozen_validation_protocol.json"


def slice_time(value, start, total_dates):
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == total_dates:
        return value[start:]
    return value


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("代码哈希不一致")
    cache = ROOT / protocol["input_cache"]["path"]
    if metrics_lib.digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("输入缓存哈希不一致")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    dates = arrays["dates"].astype(str)
    total_dates = len(dates)
    source = json.loads((ROOT / protocol["source_protocol"]).read_text(encoding="utf-8"))
    source_state = source["fixed_state"]
    base_state = robust.build_state(
        arrays,
        source["score_weights"],
        source_state["agreement"],
        source_state["entry"]["risk_off"],
        source_state["entry"]["risk_on"],
        source["fixed"],
    )
    rows = []
    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    for candidate in protocol["candidates"]:
        portfolio = v77.PortfolioProfile(
            3,
            float(candidate["gross_exposure"]),
            int(candidate["amount_min"]),
            int(candidate["mv_min"]),
            float(candidate["turnover_max"]),
        )
        liquid_state = v77.apply_liquidity(base_state, arrays, portfolio)
        quality = quality_lib.candidate_quality(
            arrays,
            liquid_state,
            int(source["health_contract"]["shadow_top_n"]),
            float(source["health_contract"]["round_trip_cost"]),
        )
        health = health_lib.fast_health_exposure(
            quality,
            int(candidate["lookback"]),
            float(candidate["mean_threshold"]),
            float(candidate["positive_fraction"]),
            float(candidate["weak_floor"]),
        )
        state = health_lib.state_with_health(liquid_state, health)
        exit_profile = sell_lib.ExitProfile(
            "rank_5d",
            int(candidate["min_hold"]),
            int(candidate["max_hold"]),
            float(candidate["sell_rank_below"]),
            float(candidate["replacement_advantage"]),
        )
        for buy_start in protocol["empty_start_buy_dates"]:
            match = np.where(dates[1:] == buy_start)[0]
            if len(match) != 1:
                raise RuntimeError("无法定位启动日 {}".format(buy_start))
            start = int(match[0])
            sliced_arrays = {key: slice_time(value, start, total_dates) for key, value in arrays.items()}
            score, order, masked, exposure = state
            sliced_state = (
                score[start:],
                order[start:],
                {key: value[start:] for key, value in masked.items()},
                exposure[start:],
            )
            daily, actions = v77.simulate(
                sliced_arrays,
                sliced_state,
                portfolio,
                exit_profile,
                protocol["validation_end"],
                record_actions=True,
            )
            path = action_dir / "{}_{}_actions.csv".format(candidate["case_id"], buy_start)
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            result = v77.core.metrics(daily, buy_start, protocol["validation_end"])
            rows.append(
                {
                    "case_id": candidate["case_id"],
                    "buy_start": buy_start,
                    "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "action_sha256": metrics_lib.digest(path),
                    "buy_rows": int((actions["action"] == "BUY").sum()) if len(actions) else 0,
                    **result,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "local_validation_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "validation_action_manifest.json").write_text(
        json.dumps({"status": "frozen_candidates_2026_generated", "runs": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(frame[["case_id", "buy_start", "cumulative_return", "linear_annual_proxy", "sharpe", "max_drawdown"]].to_json(orient="records"))


if __name__ == "__main__":
    main()
