from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_best_research_selection_20260721"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
CACHE_PATH = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_rank_rotation_preregistration_20260721" / "active_formal_rank_arrays_v1.npz"
ACTION_DIR = REPORT_DIR / "juejin_actions"
MANIFEST_PATH = REPORT_DIR / "juejin_action_manifest.json"


WEIGHTS = {
    "w10_100": {"w1": 0.0, "w3": 0.0, "w5": 0.0, "w10": 1.0},
    "w10_80_w5_20": {"w1": 0.0, "w3": 0.0, "w5": 0.2, "w10": 0.8},
    "w10_70_w5_20_w3_10": {"w1": 0.0, "w3": 0.1, "w5": 0.2, "w10": 0.7},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_from_row(row: dict) -> core.Profile:
    names = {field.name for field in fields(core.Profile)}
    return core.Profile(**{key: row[key] for key in names})


def build_actions(arrays: dict[str, np.ndarray], score: np.ndarray, order: np.ndarray, profile: core.Profile) -> pd.DataFrame:
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    limit_rate = core.board_limit_rate(stocks)
    held: set[int] = set()
    entry_index: dict[int, int] = {}
    rows: list[dict] = []
    for t, signal_date in enumerate(dates[:-1]):
        buy_date = str(dates[t + 1])
        if buy_date > "20251231":
            break
        opens = arrays["buy_open"][t]
        pre_close = arrays["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        universe = (
            arrays["signal_clean"][t]
            & arrays["buy_clean"][t]
            & valid_open
            & np.isfinite(score[t])
            & (score[t] >= profile.entry_rank_min)
            & (arrays["amount"][t] >= profile.amount_min)
            & (arrays["total_mv"][t] >= profile.mv_min)
            & (arrays["listed_days"][t] >= 60)
        )
        limit_up = opens >= pre_close * (1.0 + limit_rate) * 0.995
        universe &= ~limit_up
        ranked = [int(idx) for idx in order[t] if universe[int(idx)]]
        best_unheld_score = max((float(score[t, idx]) for idx in ranked if idx not in held), default=-np.inf)
        sell_list: list[int] = []
        for idx in sorted(held):
            age = t - entry_index[idx]
            current_score = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
            rank_exit = (
                age >= profile.min_hold
                and current_score < profile.sell_rank_below
                and best_unheld_score - current_score >= profile.replacement_advantage
            )
            if age >= profile.max_hold or rank_exit:
                sell_list.append(idx)
        for idx in sell_list:
            if not valid_open[idx]:
                continue
            limit_down = opens[idx] <= pre_close[idx] * (1.0 - limit_rate[idx]) * 1.005
            if limit_down:
                continue
            rows.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx])})
            held.remove(idx)
            del entry_index[idx]
        slots = max(profile.top_n - len(held), 0)
        for idx in ranked:
            if slots <= 0 or idx in held:
                continue
            rows.append(
                {
                    "signal_date": signal_date,
                    "buy_date": buy_date,
                    "action": "BUY",
                    "stock_code": stocks[idx],
                    "target_pct": profile.invested_ratio / profile.top_n,
                    "execution_open_raw": float(opens[idx]),
                }
            )
            held.add(idx)
            entry_index[idx] = t
            slots -= 1
    return pd.DataFrame(rows, columns=["signal_date", "buy_date", "action", "stock_code", "target_pct", "execution_open_raw"])


def main() -> None:
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    with np.load(CACHE_PATH, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    score_cache: dict[str, np.ndarray] = {}
    order_cache: dict[str, np.ndarray] = {}
    manifest_rows = []
    for row in frozen["profiles"]:
        profile = profile_from_row(row)
        if profile.blend_id not in score_cache:
            score_cache[profile.blend_id] = core.blend_scores(arrays, WEIGHTS[profile.blend_id])
            order_cache[profile.blend_id] = np.argsort(
                -np.nan_to_num(score_cache[profile.blend_id], nan=-np.inf), axis=1
            ).astype(np.int32)
        actions = build_actions(arrays, score_cache[profile.blend_id], order_cache[profile.blend_id], profile)
        path = ACTION_DIR / f"{profile.profile_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        manifest_rows.append(
            {
                "case_id": profile.profile_id,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "rows": int(len(actions)),
                "buy_rows": int((actions["action"] == "BUY").sum()),
                "sell_rows": int((actions["action"] == "SELL").sum()),
                "min_buy_date": str(actions["buy_date"].min()),
                "max_buy_date": str(actions["buy_date"].max()),
            }
        )
    payload = {
        "status": "frozen_research_only",
        "frozen_candidate_sha256": sha256(FROZEN_PATH),
        "input_cache_sha256": sha256(CACHE_PATH),
        "actions": manifest_rows,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
