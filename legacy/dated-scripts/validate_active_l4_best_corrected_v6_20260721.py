from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import research_active_l4_best_corrected_v6_20260721 as v6
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_best_corrected_v6_validation_20260721"
PROTOCOL_PATH = REPORT_DIR / "validation_protocol.json"
SUMMARY_PATH = REPORT_DIR / "local_validation_summary.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    source = ROOT / protocol["source_candidates"]
    cache = ROOT / protocol["input_cache"]
    if digest(source) != protocol["source_candidates_sha256"] or digest(cache) != protocol["input_cache_sha256"]:
        raise RuntimeError("frozen validation input drift")
    frozen = json.loads(source.read_text(encoding="utf-8"))
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    start_idx = int(np.flatnonzero(arrays["dates"].astype(str) >= protocol["validation_signal_start"])[0])
    sliced = {key: value[start_idx:] if getattr(value, "ndim", 0) >= 1 and len(value) == len(arrays["dates"]) else value for key, value in arrays.items()}
    weights = {
        "w10_100": {"w1": 0.0, "w3": 0.0, "w5": 0.0, "w10": 1.0},
        "w10_80_w5_20": {"w1": 0.0, "w3": 0.0, "w5": 0.2, "w10": 0.8},
        "w10_70_w5_20_w3_10": {"w1": 0.0, "w3": 0.1, "w5": 0.2, "w10": 0.7},
    }
    score_cache, order_cache = {}, {}
    unique = {}
    for item in frozen["profiles"]:
        profile = core.Profile(**{key: item[key] for key in core.Profile.__dataclass_fields__})
        score_cache.setdefault(profile.blend_id, core.blend_scores(sliced, weights[profile.blend_id]))
        order_cache.setdefault(profile.blend_id, np.argsort(-np.nan_to_num(score_cache[profile.blend_id], nan=-np.inf), axis=1).astype(np.int32))
        daily, actions = v6.simulate(sliced, score_cache[profile.blend_id], order_cache[profile.blend_id], profile, protocol["validation_end"], record_actions=True)
        action_hash = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if action_hash in unique:
            continue
        unique[action_hash] = profile.profile_id
        ACTION_DIR.mkdir(parents=True, exist_ok=True)
        path = ACTION_DIR / f"{profile.profile_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        unique[action_hash] = {"case_id": profile.profile_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum()), "local_metrics": core.metrics(daily, protocol["validation_execution_start"], protocol["validation_end"])}
    rows = list(unique.values())
    ACTION_MANIFEST.write_text(json.dumps({"protocol_sha256": digest(PROTOCOL_PATH), "actions": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps({"status": "validation_opened_once", "candidates": rows, "production_changed": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "validation_opened_once", "unique_paths": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
