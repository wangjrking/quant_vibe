# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_cohorts_v82_20260722 as v82
import research_active_l4_10d_staggered_v83_20260722 as v83
import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_continuous_market_v53_20260721 as market_roll
import research_active_l4_market_state_v26_20260721 as market_state
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_10d_market_gate_v84_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def parse_args():
    parser = argparse.ArgumentParser(description="10D低频错峰加历史市场门控")
    parser.add_argument("--open-2026", action="store_true")
    return parser.parse_args()


def gated_arrays(arrays, every, lookback, threshold):
    result = v83.staggered_arrays(arrays, every)
    mean_return = market_roll.rolling_mean(market_state.market_open_return(arrays), int(lookback))
    market_on = np.isfinite(mean_return) & (mean_return > float(threshold))
    result["signal_clean"] &= market_on[:, None]
    return result


def main():
    args = parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("代码哈希不一致")
    cache = ROOT / protocol["input_cache"]["path"]
    if metrics_lib.digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("输入缓存哈希不一致")
    with np.load(cache, allow_pickle=False) as saved:
        full_arrays = {key: saved[key] for key in saved.files}
    ranks = {item["id"]: v82.build_rank(full_arrays, item) for item in protocol["blends"]}
    market = protocol["market_gate"]

    if not args.open_2026:
        base = robust.truncate_observation(full_arrays, protocol["observation_end"])
        rows, definitions = [], {}
        for blend, top_n, hold_days, every, gross in product(protocol["blends"], protocol["grid"]["top_n_per_rebalance"], protocol["grid"]["hold_days"], protocol["grid"]["rebalance_every"], protocol["grid"]["target_gross_exposure"]):
            definition = {"blend": blend, "top_n_per_rebalance": int(top_n), "hold_days": int(hold_days), "rebalance_every": int(every), "target_gross_exposure": float(gross)}
            case_id = v83.stable_id({"v84_market_gate": market, **definition})
            definitions[case_id] = definition
            arrays = gated_arrays(base, every, market["lookback"], market["threshold"])
            score, order = ranks[blend["id"]]
            params = v83.engine_params(definition, protocol["fixed"])
            daily = v82.simulate(arrays, score[: len(arrays["dates"])], order[: len(arrays["dates"])], params, protocol["observation_end"])
            rows.append({"case_id": case_id, "blend_id": blend["id"], **{key: value for key, value in definition.items() if key != "blend"}, **robust.evaluate_robust(daily, protocol)})
        frame = pd.DataFrame(rows)
        gate = protocol["gate"]
        frame["eligible"] = frame["robust_positive"] & (frame["min_year_cumulative_return"] >= gate["min_year_return"]) & (frame["min_year_sharpe"] >= gate["min_year_sharpe"]) & (frame["full_sharpe"] >= gate["full_sharpe"]) & (frame["full_max_drawdown"] <= gate["max_drawdown"]) & (frame["full_trades"] >= gate["trades_min"])
        frame = frame.sort_values(["eligible", "min_year_sharpe", "min_year_return", "full_sharpe", "full_linear_annual_proxy", "case_id"], ascending=[False, False, False, False, False, True])
        OUT.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
        selected = frame[frame.eligible].head(int(protocol["freeze_count"]))
        candidates = []
        action_dir = OUT / "observation_actions"
        action_dir.mkdir(parents=True, exist_ok=True)
        for _, item in selected.iterrows():
            definition = definitions[str(item.case_id)]
            arrays = gated_arrays(base, definition["rebalance_every"], market["lookback"], market["threshold"])
            score, order = ranks[definition["blend"]["id"]]
            params = v83.engine_params(definition, protocol["fixed"])
            _, actions = v82.simulate(arrays, score[: len(arrays["dates"])], order[: len(arrays["dates"])], params, protocol["observation_end"], record_actions=True)
            path = action_dir / "{}.csv".format(item.case_id)
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            candidates.append({"case_id": str(item.case_id), "definition": definition, "observation_metrics": item.to_dict(), "action_path": str(path.relative_to(ROOT)).replace("\\", "/"), "action_sha256": metrics_lib.digest(path)})
        FROZEN.write_text(json.dumps({"status": "frozen_before_2026", "known_2026_used_for_selection": False, "candidates": candidates}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"grid": len(frame), "eligible": int(frame.eligible.sum()), "frozen": len(candidates)}, ensure_ascii=False))
        return

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    rows = []
    action_dir = OUT / "validation_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    for candidate in frozen["candidates"]:
        definition = candidate["definition"]
        arrays = gated_arrays(full_arrays, definition["rebalance_every"], market["lookback"], market["threshold"])
        score, order = ranks[definition["blend"]["id"]]
        params = v83.engine_params(definition, protocol["fixed"])
        for buy_start in protocol["validation_buy_starts"]:
            daily, actions = v82.simulate(arrays, score, order, params, protocol["validation_end"], start_buy_date=buy_start, record_actions=True)
            path = action_dir / "{}_{}.csv".format(candidate["case_id"], buy_start)
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            result = core.metrics(daily, buy_start, protocol["validation_end"])
            rows.append({"case_id": candidate["case_id"], "buy_start": buy_start, "action_path": str(path.relative_to(ROOT)).replace("\\", "/"), "action_sha256": metrics_lib.digest(path), "buy_rows": int((actions.action == "BUY").sum()) if len(actions) else 0, **result})
    pd.DataFrame(rows).to_csv(OUT / "validation_2026_local.csv", index=False, encoding="utf-8-sig")
    (OUT / "validation_action_manifest.json").write_text(json.dumps({"status": "known_2026_stress_only", "runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["case_id", "buy_start", "cumulative_return", "sharpe", "max_drawdown"]].to_json(orient="records"))


if __name__ == "__main__":
    main()
