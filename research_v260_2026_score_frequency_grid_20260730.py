from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v258_position_warmup_v260_20260723 as v260


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = (
    ROOT
    / "quant/main/strategy_library/production/"
    "prod_v260_10d_regime_warmup_all4key_v20260724"
)
BASE_PROTOCOL = STRATEGY_DIR / "inputs/preregistered_protocol.json"
BASE_ACTIONS = STRATEGY_DIR / "signals/full_history_actions_current.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def action_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in ("signal_date", "buy_date", "action", "stock_code"):
        if not left[column].astype(str).equals(right[column].astype(str)):
            return False
    for column in ("target_pct", "execution_open_raw"):
        if not np.allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base_protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    arrays = v260.v258.v252.fresh_arrays()
    end = str(arrays["dates"][-2])
    if end != str(protocol["action_end_signal_date"]):
        raise RuntimeError(f"动作结束日期漂移: {end}")

    output_dir = protocol_path.parent
    actions_dir = output_dir / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(
        BASE_ACTIONS,
        dtype={"signal_date": str, "buy_date": str, "stock_code": str},
    )
    rows = []
    for case in protocol["cases"]:
        definition = v260.definition_for(base_protocol, 50)
        score, order = v95.score_pair(
            arrays,
            float(case["weight_5d"]),
            int(case["smooth_window"]),
            float(case["current_weight"]),
        )
        _, actions = v260.run_case(
            arrays,
            score,
            order,
            definition,
            base_protocol,
            end,
            record_actions=True,
        )
        path = actions_dir / f"{case['case_id']}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        equals_production = action_equal(base, actions)
        if case["case_id"] == "baseline_w7_c10_10d" and not equals_production:
            raise RuntimeError("基线动作未能精确复现生产动作")
        rows.append(
            {
                "case_id": case["case_id"],
                "weight_5d": float(case["weight_5d"]),
                "smooth_window": int(case["smooth_window"]),
                "current_weight": float(case["current_weight"]),
                "rows": int(len(actions)),
                "buy_rows": int((actions["action"] == "BUY").sum()),
                "sell_rows": int((actions["action"] == "SELL").sum()),
                "equals_production_actions": bool(equals_production),
                "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "action_sha256": sha256(path),
            }
        )
    manifest = {
        "status": "research_only_actions_generated",
        "protocol_sha256": sha256(protocol_path),
        "production_actions_sha256": sha256(BASE_ACTIONS),
        "action_end_signal_date": end,
        "production_change_allowed": False,
        "cases": rows,
    }
    (output_dir / "action_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
