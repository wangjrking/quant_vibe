# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_strong_trend_concentration_v153_20260722 as v153


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_strong_trend_micro_v154_20260722"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 V154 冻结候选的掘金动作")
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--end-date", default="20260720")
    parser.add_argument("--source-report-dir")
    args = parser.parse_args()

    out = ROOT / args.source_report_dir if args.source_report_dir else DEFAULT_OUT
    protocol_path = out / "preregistered_protocol.json"
    frozen_path = out / "frozen_candidates_before_known_2026.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    definitions = {item["case_id"]: item["definition"] for item in frozen["candidates"]}
    missing = sorted(set(args.case_id) - set(definitions))
    if missing:
        raise SystemExit("冻结候选中不存在：{}".format(", ".join(missing)))

    cache = ROOT / protocol["input_cache"]
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(
        arrays,
        0.0,
        int(protocol["fixed_definition"]["smooth_window"]),
        float(protocol["fixed_definition"]["current_weight"]),
    )

    action_dir = out / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for case_id in args.case_id:
        definition = definitions[case_id]
        daily, actions = v153.run_case(
            arrays,
            score,
            order,
            definition,
            protocol,
            args.end_date,
            record_actions=True,
        )
        path = action_dir / "{}_through_{}.csv".format(case_id, args.end_date)
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        buys = actions[actions["action"] == "BUY"]
        runs.append(
            {
                "case_id": case_id,
                "definition": definition,
                "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "action_sha256": digest(path),
                "rows": int(len(actions)),
                "buy_rows": int(len(buys)),
                "sell_rows": int((actions["action"] == "SELL").sum()),
                "target_pct_values": sorted(float(value) for value in buys["target_pct"].unique()),
                "local_final_equity": float(daily["equity"].iloc[-1]),
            }
        )

    manifest = {
        "status": "research_only_frozen_action_export",
        "source_protocol": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
        "source_protocol_sha256": digest(protocol_path),
        "source_frozen_candidates": str(frozen_path.relative_to(ROOT)).replace("\\", "/"),
        "source_frozen_candidates_sha256": digest(frozen_path),
        "input_cache": protocol["input_cache"],
        "input_cache_sha256": digest(cache),
        "execution_contract": protocol["execution"],
        "end_date": args.end_date,
        "runs": runs,
        "production_change_allowed": False,
    }
    manifest_path = action_dir / "action_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest_sha256": digest(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
