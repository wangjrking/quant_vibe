"""Create a 2022-2024-only readiness snapshot from strict expanding OOF baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_expanding_pit_oof_baselines_20260829 as base


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
BASELINE_ROOT = DATA_DIR / "reports" / "model_agent_expanding_pit_oof_baselines_20260829_r3"
OUTPUT_DIR = DATA_DIR / "reports" / "model_agent_pre2025_development_baseline_snapshot_20260829"
DEVELOPMENT_END = "20241231"


def subset(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[("trade_date", "<=", DEVELOPMENT_END)])
    if frame.empty or frame["trade_date"].max() > DEVELOPMENT_END:
        raise RuntimeError("pre-2025 filter did not close the development boundary")
    return frame


def summary(frame: pd.DataFrame) -> dict[str, object]:
    mature = frame.loc[frame["label_mature_within_dev"]].copy()
    if mature.duplicated(["trade_date", "stock_code"]).any() or mature["stock_code"].astype(str).str.endswith(".BJ").any():
        raise RuntimeError("pre-2025 quality gate failed")
    if mature["pred_prob"].isna().any() or not np.isfinite(mature["pred_prob"]).all():
        raise RuntimeError("pre-2025 prediction quality gate failed")
    metrics, _ = base.evaluate(mature)
    return {
        "rows": int(len(frame)),
        "mature_rows": int(len(mature)),
        "date_range": [str(frame["trade_date"].min()), str(frame["trade_date"].max())],
        "duplicate_key_groups": 0,
        "bj_rows": 0,
        "null_pred_prob": 0,
        "nonfinite_pred_prob": 0,
        "mature_oof_key_sha256": base.canonical_frame_hash(mature, ["trade_date", "stock_code", "target", "pred_prob", "fold_id"]),
        "metrics": metrics,
    }


def main() -> int:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite readiness evidence: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    horizons = {}
    for horizon in ("1d", "3d", "5d", "10d"):
        horizons[horizon] = summary(subset(BASELINE_ROOT / f"{horizon}_oof.parquet"))
    report = {
        "status": "pre2025_development_baseline_ready",
        "development_window": ["20220101", DEVELOPMENT_END],
        "confirmation_window": "2025 sealed and not included in this report",
        "validation_2026_closed": True,
        "source_role": "strict_expanding_pit_oof_baselines",
        "horizons": horizons,
        "production_unchanged": True,
        "allow_next_layer_continue": False,
        "runtime_agent_route": {"model": "gpt-5.6-terra", "thinking": "medium"},
    }
    base.dump_json(OUTPUT_DIR / "pre2025_development_baseline_snapshot.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
