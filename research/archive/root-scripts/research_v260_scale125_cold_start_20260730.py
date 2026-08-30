from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v258_position_warmup_v260_20260723 as v260


ROOT = Path(__file__).resolve().parents[2]
BASE_PROTOCOL = (
    ROOT
    / "quant/main/strategy_library/production/"
    "prod_v260_10d_regime_warmup_all4key_v20260724/"
    "inputs/preregistered_protocol.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base_protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    arrays = v260.v258.v252.fresh_arrays()
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definition = v260.definition_for(base_protocol, 50)
    end = str(protocol["end_signal_date"])
    actions_dir = protocol_path.parent / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in protocol["cold_starts"]:
        _, actions = v260.run_case(
            arrays,
            score,
            order,
            definition,
            base_protocol,
            end,
            start=str(start),
            record_actions=True,
        )
        buy = actions["action"].eq("BUY")
        actions.loc[buy, "target_pct"] = (
            actions.loc[buy, "target_pct"].astype(float)
            * float(protocol["target_scale"])
        ).clip(upper=float(protocol["single_target_cap"]))
        case_id = f"scale125_start_{start}"
        path = actions_dir / f"{case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "case_id": case_id,
                "cold_start": str(start),
                "first_buy_date": str(actions["buy_date"].min()),
                "rows": int(len(actions)),
                "buy_rows": int(buy.sum()),
                "sell_rows": int((actions["action"] == "SELL").sum()),
                "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "action_sha256": sha256(path),
            }
        )
    manifest = {
        "status": "research_only_cold_start_actions_generated",
        "protocol_sha256": sha256(protocol_path),
        "production_change_allowed": False,
        "cases": rows,
    }
    (protocol_path.parent / "action_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
