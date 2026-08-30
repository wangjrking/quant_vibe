# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_cost_aware_v98_20260722 as v98
import research_active_l4_relaxed_universe_v86_20260722 as v86


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_cost_aware_v98_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"
FROZEN = OUT / "frozen_candidates_before_2026.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 V98 冻结候选的掘金动作")
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--end-date", default="20260720")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    definitions = {item["case_id"]: item["definition"] for item in frozen["candidates"]}
    missing = sorted(set(args.case_id) - set(definitions))
    if missing:
        raise SystemExit("冻结候选中不存在: {}".format(", ".join(missing)))
    cache = ROOT / protocol["input_cache"]
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score_cfg = protocol["score"]
    score, order = v95.score_pair(arrays, score_cfg["weight_5d"], score_cfg["smooth_window"], score_cfg["current_weight"])
    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for case_id in args.case_id:
        definition = definitions[case_id]
        prepared = v86.stagger(arrays, int(definition["rebalance_every"]))
        daily, actions = v86.simulate(
            prepared,
            score,
            order,
            v98.engine_params(definition, protocol["execution"]),
            args.end_date,
            record_actions=True,
        )
        path = action_dir / "{}_through_{}.csv".format(case_id, args.end_date)
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        runs.append(
            {
                "case_id": case_id,
                "definition": definition,
                "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "action_sha256": digest(path),
                "rows": int(len(actions)),
                "buy_rows": int((actions["action"] == "BUY").sum()),
                "sell_rows": int((actions["action"] == "SELL").sum()),
                "stock_count": int(actions["stock_code"].nunique()),
                "target_pct": float(actions.loc[actions["action"] == "BUY", "target_pct"].iloc[0]),
                "local_final_equity": float(daily["equity"].iloc[-1]),
            }
        )
    manifest = {
        "status": "research_only_frozen_action_export",
        "source_protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "source_protocol_sha256": digest(PROTOCOL),
        "input_cache": protocol["input_cache"],
        "input_cache_sha256": digest(cache),
        "score_contract": score_cfg,
        "execution_contract": protocol["execution"],
        "end_date": args.end_date,
        "runs": runs,
        "production_change_allowed": False,
    }
    path = action_dir / "action_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
