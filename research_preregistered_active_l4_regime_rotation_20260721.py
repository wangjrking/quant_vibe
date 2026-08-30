from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_regime_rotation_preregistration_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1_observation_screen.csv"
STAGE2_PATH = REPORT_DIR / "stage2_observation_screen.csv"
FROZEN_PATH = REPORT_DIR / "frozen_candidates_before_true_forward.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_research_only":
        raise RuntimeError("protocol is not frozen")
    for item in protocol["inputs"].values():
        path = ROOT / item["path"]
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"input hash drift: {path}")
    return protocol


def regime_masks(protocol: dict, dates: np.ndarray) -> dict[str, np.ndarray]:
    l2 = ROOT / protocol["inputs"]["l2"]["path"]
    con = duckdb.connect(str(l2), read_only=True)
    try:
        frame = con.execute(
            """
            SELECT trade_date, max(index_2000_close) AS index_close
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= '20220101' AND trade_date <= '20251231'
            GROUP BY trade_date ORDER BY trade_date
            """
        ).fetchdf()
    finally:
        con.close()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ma20"] = frame["index_close"].rolling(20, min_periods=20).mean()
    frame["ma60"] = frame["index_close"].rolling(60, min_periods=60).mean()
    frame["ma20_lag5"] = frame["ma20"].shift(5)
    frame = frame.set_index("trade_date").reindex(dates.astype(str))
    return {
        "always": np.ones(len(dates), dtype=np.bool_),
        "index_close_above_ma20": (frame["index_close"] > frame["ma20"]).fillna(False).to_numpy(dtype=np.bool_),
        "index_ma20_above_ma60": (frame["ma20"] > frame["ma60"]).fillna(False).to_numpy(dtype=np.bool_),
        "index_close_above_ma60_and_ma20_rising": ((frame["index_close"] > frame["ma60"]) & (frame["ma20"] > frame["ma20_lag5"])).fillna(False).to_numpy(dtype=np.bool_),
    }


def masked_arrays(base: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    result = dict(base)
    result["signal_clean"] = base["signal_clean"] & mask[:, None]
    return result


def main() -> None:
    protocol = load_protocol()
    cache = ROOT / protocol["inputs"]["rank_cache"]["path"]
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    regimes = regime_masks(protocol, arrays["dates"])
    scores = {item["id"]: core.blend_scores(arrays, item) for item in protocol["score_grid"]}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    fixed = protocol["stage1"]["fixed"]
    rows = []
    for regime_id, regime_mask in regimes.items():
        current = masked_arrays(arrays, regime_mask)
        for blend_id in scores:
            for amount_min in protocol["stage1"]["amount_min"]:
                for mv_min in protocol["stage1"]["total_mv_min"]:
                    for top_n in protocol["stage1"]["top_n"]:
                        for entry_min in protocol["stage1"]["entry_rank_min"]:
                            profile = core.Profile(blend_id, amount_min, mv_min, top_n, entry_min, **fixed)
                            daily = core.simulate(current, scores[blend_id], orders[blend_id], profile, "20241231")
                            stat = core.metrics(daily, "20220606", "20241231")
                            rows.append({"case_id": f"{regime_id}__{profile.profile_id}", "regime_id": regime_id, "profile_id": profile.profile_id, **asdict(profile), **stat, "yearly_all_positive": core.yearly_positive(daily, "20220606", "20241231")})
    stage1 = pd.DataFrame(rows).sort_values(["yearly_all_positive", "sharpe", "max_drawdown", "case_id"], ascending=[False, False, True, True])
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    promoted = stage1[stage1["yearly_all_positive"]].head(int(protocol["stage1"]["promote"]))
    stage2_rows = []
    for _, base in promoted.iterrows():
        current = masked_arrays(arrays, regimes[str(base["regime_id"])])
        for min_hold in protocol["stage2"]["min_hold"]:
            for max_hold in protocol["stage2"]["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for sell_below in protocol["stage2"]["sell_rank_below"]:
                    for advantage in protocol["stage2"]["replacement_advantage"]:
                        for invested in protocol["stage2"]["invested_ratio"]:
                            profile = core.Profile(str(base["blend_id"]), int(base["amount_min"]), int(base["mv_min"]), int(base["top_n"]), float(base["entry_rank_min"]), min_hold, max_hold, sell_below, advantage, invested)
                            daily = core.simulate(current, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231")
                            select = core.metrics(daily, "20220606", "20241231")
                            confirm = core.metrics(daily, "20250101", "20251231")
                            stage2_rows.append({"case_id": f"{base['regime_id']}__{profile.profile_id}", "regime_id": base["regime_id"], "profile_id": profile.profile_id, **asdict(profile), **{f"select_{k}": v for k, v in select.items()}, **{f"confirm_{k}": v for k, v in confirm.items()}, "min_sharpe": min(select["sharpe"], confirm["sharpe"])})
    stage2 = pd.DataFrame(stage2_rows)
    if stage2.empty:
        finalists = stage2
    else:
        stage2["confirm_positive"] = stage2["confirm_cumulative_return"] > 0
        stage2 = stage2.sort_values(["confirm_positive", "min_sharpe", "confirm_max_drawdown", "select_turnover", "case_id"], ascending=[False, False, True, True, True])
        finalists = stage2[stage2["confirm_positive"]].head(int(protocol["stage2"]["promote"]))
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = {"protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH), "stage2_sha256": digest(STAGE2_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "true_forward_start": "20260721", "profiles": finalists.to_dict("records")}
    if not FROZEN_PATH.exists():
        FROZEN_PATH.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        existing = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        if [x["case_id"] for x in existing["profiles"]] != [x["case_id"] for x in frozen["profiles"]]:
            raise RuntimeError("frozen finalist drift")
    summary = {"status": "research_only", "protocol_sha256": digest(PROTOCOL_PATH), "stage1_cases": len(stage1), "stage1_promoted": len(promoted), "stage2_cases": len(stage2), "frozen_candidates": len(finalists), "juejin_run": False, "production_changed": False, "true_forward_required": True}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
