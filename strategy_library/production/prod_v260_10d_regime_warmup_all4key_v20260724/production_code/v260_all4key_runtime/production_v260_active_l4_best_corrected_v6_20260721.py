from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_active_l4_best_corrected_v6_20260721"
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
STAGE1_PATH = REPORT_DIR / "stage1_corrected.csv"
STAGE2_PATH = REPORT_DIR / "stage2_corrected.csv"
FROZEN_PATH = REPORT_DIR / "frozen_juejin_candidates.json"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"
ACTION_DIR = REPORT_DIR / "juejin_actions"
ACTION_MANIFEST = REPORT_DIR / "juejin_action_manifest.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def corrected_limit_rate(arrays: dict[str, np.ndarray], t: int, idx: int, board_rate: np.ndarray) -> float:
    return float(board_rate[idx]) if bool(arrays["buy_clean"][t, idx]) else 0.05


def simulate(arrays, score, order, profile, end_signal_date, stress=1.0, record_actions=False):
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    cash = 700_000.0
    shares = {}
    entry_index = {}
    last_price = {}
    previous_equity = cash
    rows, actions = [], []
    for t, signal_date in enumerate(dates[:-1]):
        if signal_date > end_signal_date:
            break
        buy_date = str(dates[t + 1])
        opens = arrays["buy_open"][t]
        pre_close = arrays["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        for idx in list(shares):
            if valid_open[idx]:
                last_price[idx] = float(opens[idx])
        equity_before = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        universe = (
            arrays["signal_clean"][t] & arrays["buy_clean"][t] & valid_open & np.isfinite(score[t])
            & (score[t] >= profile.entry_rank_min) & (arrays["amount"][t] >= profile.amount_min)
            & (arrays["total_mv"][t] >= profile.mv_min) & (arrays["listed_days"][t] >= 60)
        )
        limit_up = opens >= pre_close * (1.0 + board_rate) * 0.995
        universe &= ~limit_up
        ranked = [int(idx) for idx in order[t] if universe[int(idx)]]
        best_unheld_score = max((float(score[t, idx]) for idx in ranked if idx not in shares), default=-np.inf)
        sell_list = []
        for idx in list(shares):
            age = t - entry_index[idx]
            current_score = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
            rank_exit = age >= profile.min_hold and current_score < profile.sell_rank_below and best_unheld_score - current_score >= profile.replacement_advantage
            if age >= profile.max_hold or rank_exit:
                sell_list.append(idx)
        turnover = 0.0
        trades = 0
        for idx in sell_list:
            if not valid_open[idx]:
                continue
            rate = corrected_limit_rate(arrays, t, idx, board_rate)
            if opens[idx] <= pre_close[idx] * (1.0 - rate) * 1.005:
                continue
            gross = float(shares[idx] * opens[idx])
            slip = core.adaptive_slippage(gross, arrays["amount"][t, idx], "sell", stress)
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "SELL", "stock_code": stocks[idx], "target_pct": 0.0, "execution_open_raw": float(opens[idx])})
            del shares[idx]
            del entry_index[idx]
            last_price.pop(idx, None)
        slots = max(profile.top_n - len(shares), 0)
        target_value = equity_before * profile.invested_ratio / profile.top_n
        for idx in ranked:
            if slots <= 0 or idx in shares:
                continue
            gross_budget = min(target_value, cash / 1.001)
            if gross_budget < 1000.0:
                break
            slip = core.adaptive_slippage(gross_budget, arrays["amount"][t, idx], "buy", stress)
            buy_price = float(opens[idx]) * (1.0 + slip)
            quantity = gross_budget / (buy_price * (1.0 + 0.0003))
            spend = quantity * buy_price * (1.0 + 0.0003)
            if quantity <= 0 or spend > cash + 1e-6:
                continue
            cash -= spend
            shares[idx] = quantity
            entry_index[idx] = t
            last_price[idx] = float(opens[idx])
            turnover += quantity * float(opens[idx])
            trades += 1
            slots -= 1
            if record_actions:
                actions.append({"signal_date": signal_date, "buy_date": buy_date, "action": "BUY", "stock_code": stocks[idx], "target_pct": profile.invested_ratio / profile.top_n, "execution_open_raw": float(opens[idx])})
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        rows.append({"date": buy_date, "return": equity_after / previous_equity - 1.0, "equity": equity_after, "turnover": turnover / max(equity_before, 1.0), "invested_ratio": 1.0 - cash / equity_after if equity_after > 0 else 0.0, "positions": len(shares), "trades": trades})
        previous_equity = equity_after
    daily = pd.DataFrame(rows)
    return (daily, pd.DataFrame(actions)) if record_actions else daily


def main():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only" or digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("frozen protocol or input drift")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    scores = {item["id"]: core.blend_scores(arrays, item) for item in protocol["score_grid"]}
    orders = {key: np.argsort(-np.nan_to_num(value, nan=-np.inf), axis=1).astype(np.int32) for key, value in scores.items()}
    fixed = protocol["stage1"]["fixed"]
    stage1_rows = []
    for blend_id in scores:
        for amount_min in protocol["stage1"]["amount_min"]:
            for mv_min in protocol["stage1"]["total_mv_min"]:
                for top_n in protocol["stage1"]["top_n"]:
                    for entry_min in protocol["stage1"]["entry_rank_min"]:
                        profile = core.Profile(blend_id, amount_min, mv_min, top_n, entry_min, **fixed)
                        daily = simulate(arrays, scores[blend_id], orders[blend_id], profile, "20241231")
                        stage1_rows.append({"profile_id": profile.profile_id, **asdict(profile), **core.metrics(daily, "20220606", "20241231")})
    stage1 = pd.DataFrame(stage1_rows)
    base = stage1[(stage1.cumulative_return > 0) & (stage1.max_drawdown <= 0.50) & (stage1.trades >= 100)].sort_values(["sharpe", "cagr", "max_drawdown", "profile_id"], ascending=[False, False, True, True]).head(protocol["stage1"]["base_count"])
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    rows = []
    grid = protocol["stage2"]
    for _, seed in base.iterrows():
        for min_hold in grid["min_hold"]:
            for max_hold in grid["max_hold"]:
                if max_hold <= min_hold:
                    continue
                for sell_below in grid["sell_rank_below"]:
                    for advantage in grid["replacement_advantage"]:
                        for invested in grid["invested_ratio"]:
                            profile = core.Profile(str(seed.blend_id), int(seed.amount_min), int(seed.mv_min), int(seed.top_n), float(seed.entry_rank_min), min_hold, max_hold, sell_below, advantage, invested)
                            daily = simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231")
                            selection = core.metrics(daily, "20220606", "20241231")
                            confirmation = core.metrics(daily, "20250101", "20251231")
                            full = core.metrics(daily, "20220606", "20251231")
                            rows.append({"case_id": profile.profile_id, **asdict(profile), **{f"selection_{k}": v for k, v in selection.items()}, **{f"confirmation_{k}": v for k, v in confirmation.items()}, **{f"full_{k}": v for k, v in full.items()}, "min_period_sharpe": min(selection["sharpe"], confirmation["sharpe"])})
    result = pd.DataFrame(rows).drop_duplicates("case_id")
    result["both_positive"] = (result.selection_cumulative_return > 0) & (result.confirmation_cumulative_return > 0)
    result = result.sort_values(["both_positive", "min_period_sharpe", "confirmation_sharpe", "full_max_drawdown", "full_turnover", "case_id"], ascending=[False, False, False, True, True, True])
    result.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    frozen = result[result.both_positive].head(grid["juejin_candidates"])
    payload = {"protocol_sha256": digest(PROTOCOL_PATH), "stage1_sha256": digest(STAGE1_PATH), "stage2_sha256": digest(STAGE2_PATH), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "profiles": frozen.to_dict("records")}
    FROZEN_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ACTION_DIR.mkdir(parents=True, exist_ok=True)
    action_rows = []
    for item in payload["profiles"]:
        profile = core.Profile(**{key: item[key] for key in core.Profile.__dataclass_fields__})
        _, actions = simulate(arrays, scores[profile.blend_id], orders[profile.blend_id], profile, "20251231", record_actions=True)
        path = ACTION_DIR / f"{profile.profile_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append({"case_id": profile.profile_id, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "rows": len(actions), "buy_rows": int((actions.action == "BUY").sum()), "sell_rows": int((actions.action == "SELL").sum())})
    ACTION_MANIFEST.write_text(json.dumps({"frozen_sha256": digest(FROZEN_PATH), "actions": action_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "research_only", "stage1_cases": len(stage1), "base_profiles": len(base), "stage2_cases": len(result), "frozen_candidates": len(frozen), "best": frozen.head(1).to_dict("records"), "production_changed": False, "latest_signal_generated": False}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "best"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
