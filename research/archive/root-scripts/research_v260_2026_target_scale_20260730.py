from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_ACTIONS = (
    ROOT
    / "quant/main/strategy_library/production/"
    "prod_v260_10d_regime_warmup_all4key_v20260724/"
    "signals/full_history_actions_current.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(BASE_ACTIONS) != str(protocol["base_actions_sha256"]).upper():
        raise RuntimeError("生产动作哈希漂移")
    base = pd.read_csv(
        BASE_ACTIONS,
        dtype={"signal_date": str, "buy_date": str, "stock_code": str},
    )
    actions_dir = protocol_path.parent / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in protocol["cases"]:
        output = base.copy()
        buy = output["action"].eq("BUY")
        output.loc[buy, "target_pct"] = (
            output.loc[buy, "target_pct"].astype(float)
            * float(case["target_scale"])
        ).clip(upper=float(case["single_target_cap"]))
        path = actions_dir / f"{case['case_id']}.csv"
        output.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "case_id": case["case_id"],
                "target_scale": float(case["target_scale"]),
                "single_target_cap": float(case["single_target_cap"]),
                "rows": int(len(output)),
                "buy_rows": int(buy.sum()),
                "sell_rows": int((output["action"] == "SELL").sum()),
                "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "action_sha256": sha256(path),
            }
        )
    manifest = {
        "status": "research_only_actions_generated",
        "protocol_sha256": sha256(protocol_path),
        "base_actions_sha256": sha256(BASE_ACTIONS),
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
