from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_active_l4_best_corrected_v6_20260721 as engine
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_agreement_v8_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1.csv"
STAGE2_PATH = REPORT_DIR / "stage2.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def case_id(profile, agreement_id):
    raw = json.dumps({"profile": asdict(profile), "agreement_id": agreement_id}, sort_keys=True, separators=(",", ":"))
    return "agr_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def masked_arrays(arrays, gate):
    out = dict(arrays)
    agreement = (
        (arrays["rank_10d"] >= float(gate["rank10_min"]))
        & (arrays["rank_5d"] >= float(gate["rank5_min"]))
        & (arrays["rank_3d"] >= float(gate["rank3_min"]))
    )
    out["signal_clean"] = arrays["signal_clean"] & agreement
    return out


def metrics(daily):
    selection = core.metrics(daily, "20220606", "20241231")
    confirmation = core.metrics(daily, "20250101", "20251231")
    full = core.metrics(daily, "20220606", "20251231")
    return selection, confirmation, full


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("frozen protocol or input drift")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {item["id"]: core.blend_scores(arrays, item) for item in protocol["score_grid"]}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    gates = {item["id"]: item for item in protocol["agreement_grid"]}
    masked = {key: masked_arrays(arrays, gate) for key, gate in gates.items()}
    fixed = protocol["stage1"]["fixed"]
    rows = []
    for blend_id in scores:
        for agreement_id in gates:
            for amount_min in protocol["stage1"]["amount_min"]:
                for top_n in protocol["stage1"]["top_n"]:
                    for entry_min in protocol["stage1"]["entry_rank_min"]:
                        profile = core.Profile(blend_id, amount_min, 200000, top_n, entry_min, **fixed)
                        daily = engine.simulate(masked[agreement_id], scores[blend_id], orders[blend_id], profile, "20251231")
                        selection, confirmation, full = metrics(daily)
                        rows.append({"case_id": case_id(profile, agreement_id), "agreement_id": agreement_id, **asdict(profile), **{f"selection_{k}": v for k, v in selection.items()}, **{f"confirmation_{k}": v for k, v in confirmation.items()}, **{f"full_{k}": v for k, v in full.items()}, "min_period_sharpe": min(selection["sharpe"], confirmation["sharpe"])})
    stage1 = pd.DataFrame(rows)
    eligible = stage1[(stage1.selection_cumulative_return > 0) & (stage1.confirmation_cumulative_return > 0) & (stage1.full_max_drawdown <= 0.50) & (stage1.full_trades >= 80)]
    base = eligible.sort_values(["min_period_sharpe", "full_sharpe", "full_max_drawdown", "case_id"], ascending=[False, False, True, True]).head(protocol["stage1"]["promote"])
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    rows = []
    grid = protocol["stage2"]
    for _, seed in base.iterrows():
        for max_hold in grid["max_hold"]:
            for sell_below in grid["sell_rank_below"]:
                for advantage in grid["replacement_advantage"]:
                    for invested in grid["invested_ratio"]:
                        profile = core.Profile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), int(grid["fixed_min_hold"]), max_hold, sell_below, advantage, invested)
                        daily = engine.simulate(masked[str(seed.agreement_id)], scores[profile.blend_id], orders[profile.blend_id], profile, "20251231")
                        selection, confirmation, full = metrics(daily)
                        rows.append({"case_id": case_id(profile, str(seed.agreement_id)), "agreement_id": str(seed.agreement_id), **asdict(profile), **{f"selection_{k}": v for k, v in selection.items()}, **{f"confirmation_{k}": v for k, v in confirmation.items()}, **{f"full_{k}": v for k, v in full.items()}, "min_period_sharpe": min(selection["sharpe"], confirmation["sharpe"])})
    stage2 = pd.DataFrame(rows).drop_duplicates("case_id")
    stage2["both_positive"] = (stage2.selection_cumulative_return > 0) & (stage2.confirmation_cumulative_return > 0)
    stage2 = stage2.sort_values(["both_positive", "min_period_sharpe", "full_sharpe", "full_max_drawdown", "full_turnover", "case_id"], ascending=[False, False, False, True, True, True])
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = stage2[stage2.both_positive].head(grid["promote_for_juejin"])
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH), "stage2_sha256": digest(STAGE2_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    actions_out, seen = [], set()
    for item in payload["profiles"]:
        profile = core.Profile(**{key: item[key] for key in core.Profile.__dataclass_fields__})
        _, actions = engine.simulate(masked[item["agreement_id"]], scores[profile.blend_id], orders[profile.blend_id], profile, "20251231", record_actions=True)
        action_key = hashlib.sha256(actions.to_csv(index=False).encode("utf-8")).hexdigest()
        if action_key in seen:
            continue
        seen.add(action_key)
        path = ACTION_DIR / f"{item['case_id']}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        actions_out.append({"case_id": item["case_id"], "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": actions_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only", "stage1_cases": len(stage1), "eligible_stage1": len(eligible), "base_profiles": len(base), "stage2_cases": len(stage2), "frozen_candidates": len(frozen), "unique_juejin_paths": len(actions_out), "best": frozen.head(1).to_dict("records"), "validation_not_opened": True, "production_changed": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
