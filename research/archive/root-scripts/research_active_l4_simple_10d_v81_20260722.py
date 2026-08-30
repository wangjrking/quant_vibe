# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_diversified_equalweight_v77_20260722 as v77
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_simple_10d_v81_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def parse_args():
    parser = argparse.ArgumentParser(description="简化10D策略预注册研究")
    parser.add_argument("--open-2026", action="store_true")
    return parser.parse_args()


def stable_id(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "v81_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_state(arrays, weights, fixed):
    score = (
        float(weights["w5"]) * arrays["rank_5d"]
        + float(weights["w10"]) * arrays["rank_10d"]
    ).astype(np.float32)
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable")
    valid = arrays["signal_clean"].copy()
    valid &= np.isfinite(arrays["amount"]) & (arrays["amount"] >= int(fixed["amount_min"]))
    valid &= np.isfinite(arrays["total_mv"]) & (arrays["total_mv"] >= int(fixed["mv_min"]))
    valid &= np.isfinite(arrays["turnover_rate"])
    valid &= (arrays["turnover_rate"] >= 0.0) & (arrays["turnover_rate"] <= float(fixed["turnover_max"]))
    masked = {
        "signal_clean": valid,
        "buy_clean": arrays["buy_clean"].copy(),
    }
    exposure = np.ones(len(arrays["dates"]), dtype=np.float32)
    return score, order, masked, exposure


def slice_time(value, start, total_dates):
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == total_dates:
        return value[start:]
    return value


def slice_state(state, start):
    score, order, masked, exposure = state
    return (
        score[start:],
        order[start:],
        {key: value[start:] for key, value in masked.items()},
        exposure[start:],
    )


def observation_gate(frame, gate):
    return (
        frame["robust_positive"]
        & (frame["min_year_cumulative_return"] >= float(gate["min_year_return"]))
        & (frame["min_year_sharpe"] >= float(gate["min_year_sharpe"]))
        & (frame["full_sharpe"] >= float(gate["full_sharpe"]))
        & (frame["full_max_drawdown"] <= float(gate["max_drawdown"]))
        & (frame["full_trades"] >= int(gate["trades_min"]))
    )


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
    states = {
        item["id"]: make_state(full_arrays, item, protocol["fixed"])
        for item in protocol["blends"]
    }

    if not args.open_2026:
        arrays = robust.truncate_observation(full_arrays, protocol["observation_end"])
        rows = []
        definitions = {}
        for blend, top_n, gross, min_hold, max_hold, sell_rank, advantage in product(
            protocol["blends"],
            protocol["grid"]["top_n"],
            protocol["grid"]["gross_exposure"],
            protocol["grid"]["min_hold"],
            protocol["grid"]["max_hold"],
            protocol["grid"]["sell_rank_below"],
            protocol["grid"]["replacement_advantage"],
        ):
            if int(max_hold) < int(min_hold):
                continue
            portfolio = v77.PortfolioProfile(
                int(top_n),
                float(gross),
                int(protocol["fixed"]["amount_min"]),
                int(protocol["fixed"]["mv_min"]),
                float(protocol["fixed"]["turnover_max"]),
            )
            exit_profile = sell_lib.ExitProfile(
                "rank_10d", int(min_hold), int(max_hold), float(sell_rank), float(advantage)
            )
            definition = {
                "blend": blend,
                "portfolio": asdict(portfolio),
                "exit": asdict(exit_profile),
            }
            case_id = stable_id(definition)
            definitions[case_id] = definition
            state = states[blend["id"]]
            truncated_state = (
                state[0][: len(arrays["dates"])],
                state[1][: len(arrays["dates"])],
                {key: value[: len(arrays["dates"])] for key, value in state[2].items()},
                state[3][: len(arrays["dates"])],
            )
            daily = v77.simulate(
                arrays, truncated_state, portfolio, exit_profile, protocol["observation_end"]
            )
            rows.append(
                {
                    "case_id": case_id,
                    "blend_id": blend["id"],
                    **asdict(portfolio),
                    **asdict(exit_profile),
                    **robust.evaluate_robust(daily, protocol),
                }
            )
        frame = pd.DataFrame(rows)
        frame["eligible"] = observation_gate(frame, protocol["gate"])
        frame = frame.sort_values(
            ["eligible", "min_year_sharpe", "min_year_return", "full_sharpe", "full_linear_annual_proxy", "case_id"],
            ascending=[False, False, False, False, False, True],
        )
        OUT.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUT / "observation_grid.csv", index=False, encoding="utf-8-sig")
        selected = frame[frame["eligible"]].head(int(protocol["freeze_count"]))
        frozen_rows = []
        action_dir = OUT / "observation_actions"
        action_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        for _, item in selected.iterrows():
            definition = definitions[str(item.case_id)]
            portfolio = v77.PortfolioProfile(**definition["portfolio"])
            exit_profile = sell_lib.ExitProfile(**definition["exit"])
            state = states[definition["blend"]["id"]]
            truncated_state = (
                state[0][: len(arrays["dates"])],
                state[1][: len(arrays["dates"])],
                {key: value[: len(arrays["dates"])] for key, value in state[2].items()},
                state[3][: len(arrays["dates"])],
            )
            _, actions = v77.simulate(
                arrays,
                truncated_state,
                portfolio,
                exit_profile,
                protocol["observation_end"],
                record_actions=True,
            )
            action_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
            if action_hash in seen:
                continue
            seen.add(action_hash)
            path = action_dir / "{}.csv".format(item.case_id)
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            frozen_rows.append(
                {
                    "case_id": str(item.case_id),
                    "definition": definition,
                    "observation_metrics": item.to_dict(),
                    "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "action_sha256": metrics_lib.digest(path),
                }
            )
        FROZEN.write_text(
            json.dumps(
                {
                    "status": "frozen_before_2026",
                    "known_2026_used_for_selection": False,
                    "protocol_sha256": metrics_lib.digest(PROTOCOL),
                    "candidates": frozen_rows,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"grid": len(frame), "eligible": int(frame.eligible.sum()), "frozen": len(frozen_rows)}, ensure_ascii=False))
        return

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_before_2026":
        raise RuntimeError("候选未在2026前冻结")
    dates = full_arrays["dates"].astype(str)
    total_dates = len(dates)
    action_dir = OUT / "validation_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in frozen["candidates"]:
        definition = candidate["definition"]
        portfolio = v77.PortfolioProfile(**definition["portfolio"])
        exit_profile = sell_lib.ExitProfile(**definition["exit"])
        state = states[definition["blend"]["id"]]
        for buy_start in protocol["validation_buy_starts"]:
            match = np.where(dates[1:] == buy_start)[0]
            if len(match) != 1:
                raise RuntimeError("无法定位启动日 {}".format(buy_start))
            start = int(match[0])
            arrays = {key: slice_time(value, start, total_dates) for key, value in full_arrays.items()}
            daily, actions = v77.simulate(
                arrays,
                slice_state(state, start),
                portfolio,
                exit_profile,
                protocol["validation_end"],
                record_actions=True,
            )
            path = action_dir / "{}_{}.csv".format(candidate["case_id"], buy_start)
            actions.to_csv(path, index=False, encoding="utf-8-sig")
            result = v77.core.metrics(daily, buy_start, protocol["validation_end"])
            rows.append(
                {
                    "case_id": candidate["case_id"],
                    "buy_start": buy_start,
                    "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "action_sha256": metrics_lib.digest(path),
                    "buy_rows": int((actions.action == "BUY").sum()) if len(actions) else 0,
                    **result,
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "validation_2026_local.csv", index=False, encoding="utf-8-sig")
    (OUT / "validation_action_manifest.json").write_text(
        json.dumps({"status": "known_2026_stress_only", "runs": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows)[["case_id", "buy_start", "cumulative_return", "sharpe", "max_drawdown"]].to_json(orient="records"))


if __name__ == "__main__":
    main()
