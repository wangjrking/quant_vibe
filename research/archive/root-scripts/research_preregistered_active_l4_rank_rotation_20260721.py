from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_active_l4_rank_rotation_preregistration_20260721"
)
PROTOCOL_PATH = REPORT_DIR / "preregistered_protocol.json"
CACHE_PATH = REPORT_DIR / "active_formal_rank_arrays_v1.npz"
STAGE1_PATH = REPORT_DIR / "stage1_observation_screen.csv"
STAGE1_FROZEN_PATH = REPORT_DIR / "stage1_promoted_profiles.json"
STAGE2_PATH = REPORT_DIR / "stage2_observation_screen.csv"
FROZEN_CANDIDATES_PATH = REPORT_DIR / "frozen_candidates_before_validation.json"
VALIDATION_PATH = REPORT_DIR / "final_validation_once.csv"
SUMMARY_PATH = REPORT_DIR / "research_summary.json"

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


@dataclass(frozen=True)
class Profile:
    blend_id: str
    amount_min: int
    mv_min: int
    top_n: int
    entry_rank_min: float
    min_hold: int
    max_hold: int
    sell_rank_below: float
    replacement_advantage: float
    invested_ratio: float

    @property
    def profile_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "rr_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预注册 active formal L4 排名轮动研究")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--open-validation", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_research_only":
        raise RuntimeError("Protocol is not frozen_research_only")
    expected = {item["label"].lower(): item for item in protocol["input_assets"]["l4_manifests"]}
    for label, path in MANIFESTS.items():
        actual_hash = sha256(path)
        if actual_hash != expected[label]["sha256"]:
            raise RuntimeError(f"Manifest hash drift: {label}: {actual_hash}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("approval_status") != "approved_for_l5":
            raise RuntimeError(f"Manifest not approved_for_l5: {label}")
        if manifest.get("source_type") != "duckdb_table":
            raise RuntimeError(f"Manifest not DuckDB: {label}")
    l2_path = ROOT / protocol["input_assets"]["l2"]["path"]
    if sha256(l2_path) != protocol["input_assets"]["l2"]["sha256"]:
        raise RuntimeError("L2 hash drift; freeze a new protocol before continuing")
    return protocol


def resolve_manifest(path: Path) -> tuple[Path, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (path.parent / payload["db_path"]).resolve(), str(payload["table"])


def clean_sql(alias: str) -> str:
    return f"""
      coalesce(cast({alias}.ST_TYPE as varchar), '') in ('', '0', '0.0', 'None', 'NONE')
      and upper(coalesce(cast({alias}.ST_TYPE_name as varchar), '')) not like '%ST%'
      and upper(coalesce(cast({alias}.name as varchar), '')) not like 'ST%'
      and upper(coalesce(cast({alias}.name as varchar), '')) not like '*ST%'
      and coalesce(cast({alias}.name as varchar), '') not like '%退%'
    """


def build_cache(protocol: dict) -> dict[str, np.ndarray]:
    l2_path = ROOT / protocol["input_assets"]["l2"]["path"]
    l4 = {label: resolve_manifest(path) for label, path in MANIFESTS.items()}
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{l2_path.as_posix()}' AS m (READ_ONLY)")
        for label, (db_path, _) in l4.items():
            con.execute(f"ATTACH '{db_path.as_posix()}' AS p{label} (READ_ONLY)")
        common_max = min(
            str(con.execute(f'SELECT MAX(trade_date) FROM p{label}."{table}"').fetchone()[0])
            for label, (_, table) in l4.items()
        )
        dates = [
            str(row[0])
            for row in con.execute(
                "SELECT DISTINCT trade_date FROM m.STOCK_DAILY_DATA "
                "WHERE trade_date >= '20220606' AND trade_date <= ? ORDER BY trade_date",
                [common_max],
            ).fetchall()
        ]
        stocks = [
            str(row[0])
            for row in con.execute(
                f'SELECT DISTINCT stock_code FROM p10d."{l4["10d"][1]}" '
                "WHERE stock_code NOT LIKE '%.BJ' ORDER BY stock_code"
            ).fetchall()
        ]
        date_map = pd.DataFrame({"trade_date": dates, "d_idx": np.arange(len(dates), dtype=np.int32)})
        stock_map = pd.DataFrame({"stock_code": stocks, "s_idx": np.arange(len(stocks), dtype=np.int32)})
        con.register("date_map", date_map)
        con.register("stock_map", stock_map)
        n_dates, n_stocks = len(dates), len(stocks)
        shape = (n_dates, n_stocks)
        arrays: dict[str, np.ndarray] = {
            "rank_1d": np.full(shape, np.nan, dtype=np.float32),
            "rank_3d": np.full(shape, np.nan, dtype=np.float32),
            "rank_5d": np.full(shape, np.nan, dtype=np.float32),
            "rank_10d": np.full(shape, np.nan, dtype=np.float32),
            "amount": np.full(shape, np.nan, dtype=np.float32),
            "total_mv": np.full(shape, np.nan, dtype=np.float32),
            "listed_days": np.full(shape, -1, dtype=np.int16),
            "signal_clean": np.zeros(shape, dtype=np.bool_),
            "buy_clean": np.zeros(shape, dtype=np.bool_),
            "buy_open": np.full(shape, np.nan, dtype=np.float32),
            "buy_pre_close": np.full(shape, np.nan, dtype=np.float32),
        }
        query = f"""
        WITH calendar AS (
          SELECT trade_date,
                 lead(trade_date) OVER (ORDER BY trade_date) AS buy_date
          FROM (SELECT DISTINCT trade_date FROM m.STOCK_DAILY_DATA)
        ), scores AS (
          SELECT p10.trade_date, p10.stock_code,
                 percent_rank() OVER (PARTITION BY p10.trade_date ORDER BY p1.pred_prob, p10.stock_code) AS rank_1d,
                 percent_rank() OVER (PARTITION BY p10.trade_date ORDER BY p3.pred_prob, p10.stock_code) AS rank_3d,
                 percent_rank() OVER (PARTITION BY p10.trade_date ORDER BY p5.pred_prob, p10.stock_code) AS rank_5d,
                 percent_rank() OVER (PARTITION BY p10.trade_date ORDER BY p10.pred_prob, p10.stock_code) AS rank_10d
          FROM p10d."{l4['10d'][1]}" p10
          JOIN p5d."{l4['5d'][1]}" p5 USING (trade_date, stock_code)
          JOIN p3d."{l4['3d'][1]}" p3 USING (trade_date, stock_code)
          JOIN p1d."{l4['1d'][1]}" p1 USING (trade_date, stock_code)
          WHERE p10.trade_date >= '20220606' AND p10.trade_date <= '{common_max}'
            AND p10.stock_code NOT LIKE '%.BJ'
        )
        SELECT dm.d_idx, smap.s_idx,
               s.rank_1d, s.rank_3d, s.rank_5d, s.rank_10d,
               sig.amount, sig.total_mv,
               greatest(0, date_diff('day', try_strptime(sig.list_date, '%Y%m%d'), try_strptime(sig.trade_date, '%Y%m%d'))) AS listed_days,
               ({clean_sql('sig')}) AS signal_clean,
               ({clean_sql('buy')}) AS buy_clean,
               buy.open AS buy_open, buy.pre_close AS buy_pre_close
        FROM scores s
        JOIN date_map dm ON dm.trade_date=s.trade_date
        JOIN stock_map smap ON smap.stock_code=s.stock_code
        JOIN calendar c ON c.trade_date=s.trade_date AND c.buy_date IS NOT NULL
        JOIN m.STOCK_DAILY_DATA sig ON sig.trade_date=s.trade_date AND sig.stock_code=s.stock_code
        LEFT JOIN m.STOCK_DAILY_DATA buy ON buy.trade_date=c.buy_date AND buy.stock_code=s.stock_code
        ORDER BY dm.d_idx, smap.s_idx
        """
        reader = con.execute(query).fetch_record_batch(rows_per_batch=200_000)
        for batch in reader:
            frame = batch.to_pandas()
            d = frame["d_idx"].to_numpy(dtype=np.intp, copy=False)
            s = frame["s_idx"].to_numpy(dtype=np.intp, copy=False)
            for column in ("rank_1d", "rank_3d", "rank_5d", "rank_10d", "amount", "total_mv"):
                arrays[column][d, s] = frame[column].to_numpy(dtype=np.float32, copy=False)
            arrays["listed_days"][d, s] = frame["listed_days"].fillna(-1).to_numpy(dtype=np.int16)
            arrays["signal_clean"][d, s] = frame["signal_clean"].fillna(False).to_numpy(dtype=np.bool_)
            arrays["buy_clean"][d, s] = frame["buy_clean"].fillna(False).to_numpy(dtype=np.bool_)
            arrays["buy_open"][d, s] = frame["buy_open"].to_numpy(dtype=np.float32, copy=False)
            arrays["buy_pre_close"][d, s] = frame["buy_pre_close"].to_numpy(dtype=np.float32, copy=False)
    finally:
        con.close()
    arrays["dates"] = np.asarray(dates, dtype="U8")
    arrays["stocks"] = np.asarray(stocks, dtype="U9")
    np.savez_compressed(CACHE_PATH, **arrays)
    return arrays


def load_arrays(protocol: dict, rebuild: bool) -> dict[str, np.ndarray]:
    if rebuild or not CACHE_PATH.exists():
        return build_cache(protocol)
    with np.load(CACHE_PATH, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def blend_scores(arrays: dict[str, np.ndarray], weights: dict) -> np.ndarray:
    return (
        float(weights["w1"]) * arrays["rank_1d"]
        + float(weights["w3"]) * arrays["rank_3d"]
        + float(weights["w5"]) * arrays["rank_5d"]
        + float(weights["w10"]) * arrays["rank_10d"]
    ).astype(np.float32)


def board_limit_rate(stock_codes: np.ndarray) -> np.ndarray:
    codes = stock_codes.astype("U9")
    board20 = np.char.startswith(codes, "300") | np.char.startswith(codes, "301") | np.char.startswith(codes, "688")
    return np.where(board20, 0.20, 0.10).astype(np.float32)


def adaptive_slippage(order_value: float, amount_thousand: float, side: str, stress: float = 1.0) -> float:
    if not np.isfinite(amount_thousand) or amount_thousand <= 0:
        return 0.0065 if side == "buy" else 0.0030
    participation = max(order_value / (float(amount_thousand) * 1000.0), 0.0)
    if side == "buy":
        value = np.clip(0.0010 + 0.03 * math.sqrt(participation), 0.0015, 0.0065)
    else:
        value = np.clip(0.0002 + 0.015 * math.sqrt(participation), 0.0002, 0.0030)
    return float(min(value * stress, 0.02))


def metrics(daily: pd.DataFrame, start: str, end: str) -> dict:
    part = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    if part.empty:
        return {"linear_annual_proxy": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "turnover": 0.0, "days": 0}
    returns = part["return"].fillna(0.0).to_numpy(dtype=float)
    equity = np.cumprod(1.0 + returns)
    cumulative = float(equity[-1] - 1.0)
    years = max(len(part) / 252.0, 1.0 / 252.0)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    std = float(np.std(returns, ddof=0))
    sharpe = float(np.mean(returns) / std * math.sqrt(252.0)) if std > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    mdd = float(abs(np.min(equity / peak - 1.0)))
    return {
        "cumulative_return": cumulative,
        "linear_annual_proxy": cumulative / years,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "turnover": float(part["turnover"].sum()),
        "invested_ratio_mean": float(part["invested_ratio"].mean()),
        "trades": int(part["trades"].sum()),
        "days": int(len(part)),
    }


def simulate(
    arrays: dict[str, np.ndarray],
    score: np.ndarray,
    order: np.ndarray,
    profile: Profile,
    end_signal_date: str,
    stress: float = 1.0,
) -> pd.DataFrame:
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    limit_rate = board_limit_rate(stocks)
    cash = 700_000.0
    shares: dict[int, float] = {}
    entry_index: dict[int, int] = {}
    last_price: dict[int, float] = {}
    previous_equity = cash
    rows = []
    for t, signal_date in enumerate(dates):
        if signal_date > end_signal_date:
            break
        opens = arrays["buy_open"][t]
        pre_close = arrays["buy_pre_close"][t]
        valid_open = np.isfinite(opens) & (opens > 0) & np.isfinite(pre_close) & (pre_close > 0)
        for stock_idx in list(shares):
            if valid_open[stock_idx]:
                last_price[stock_idx] = float(opens[stock_idx])
        equity_before = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
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
        best_unheld_score = max((float(score[t, idx]) for idx in ranked if idx not in shares), default=-np.inf)
        sell_list = []
        for idx in list(shares):
            age = t - entry_index[idx]
            current_score = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
            rank_exit = (
                age >= profile.min_hold
                and current_score < profile.sell_rank_below
                and best_unheld_score - current_score >= profile.replacement_advantage
            )
            if age >= profile.max_hold or rank_exit:
                sell_list.append(idx)
        turnover = 0.0
        trades = 0
        for idx in sell_list:
            if not valid_open[idx]:
                continue
            limit_down = opens[idx] <= pre_close[idx] * (1.0 - limit_rate[idx]) * 1.005
            if limit_down:
                continue
            gross = float(shares[idx] * opens[idx])
            slip = adaptive_slippage(gross, arrays["amount"][t, idx], "sell", stress)
            proceeds = gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            cash += proceeds
            turnover += gross
            trades += 1
            del shares[idx]
            del entry_index[idx]
            last_price.pop(idx, None)
        slots = max(profile.top_n - len(shares), 0)
        target_value = equity_before * profile.invested_ratio / profile.top_n
        for idx in ranked:
            if slots <= 0 or idx in shares:
                continue
            affordable = cash / 1.001
            gross_budget = min(target_value, affordable)
            if gross_budget < 1000.0:
                break
            slip = adaptive_slippage(gross_budget, arrays["amount"][t, idx], "buy", stress)
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
        equity_after = cash + sum(shares[idx] * last_price.get(idx, 0.0) for idx in shares)
        day_return = equity_after / previous_equity - 1.0 if previous_equity > 0 else 0.0
        invested = 1.0 - cash / equity_after if equity_after > 0 else 0.0
        rows.append(
            {
                "date": str(dates[t + 1]) if t + 1 < len(dates) else str(signal_date),
                "return": day_return,
                "equity": equity_after,
                "turnover": turnover / max(equity_before, 1.0),
                "invested_ratio": invested,
                "positions": len(shares),
                "trades": trades,
            }
        )
        previous_equity = equity_after
    return pd.DataFrame(rows)


def yearly_positive(daily: pd.DataFrame, start: str, end: str) -> bool:
    part = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    if part.empty:
        return False
    part["year"] = part["date"].str[:4]
    return all(float((1.0 + group["return"]).prod() - 1.0) > 0 for _, group in part.groupby("year"))


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    arrays = load_arrays(protocol, args.rebuild_cache)
    blends = {item["id"]: item for item in protocol["score_grid"]}
    blend_cache = {blend_id: blend_scores(arrays, weights) for blend_id, weights in blends.items()}
    order_cache = {
        blend_id: np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
        for blend_id, score in blend_cache.items()
    }
    stage1_rows = []
    fixed = protocol["search_stages"][0]["fixed_parameters"]
    for blend_id in blends:
        for amount_min in protocol["universe"]["signal_day_amount_min_thousand_cny"]:
            for mv_min in protocol["universe"]["signal_day_total_mv_min_ten_thousand_cny"]:
                for top_n in protocol["parameter_grid"]["top_n"]:
                    for entry_min in protocol["parameter_grid"]["entry_rank_percentile_min"]:
                        profile = Profile(
                            blend_id, amount_min, mv_min, top_n, entry_min,
                            fixed["min_holding_days"], fixed["max_holding_days"],
                            fixed["sell_rank_percentile_below"], fixed["replacement_score_advantage"],
                            fixed["target_invested_ratio"],
                        )
                        daily = simulate(arrays, blend_cache[blend_id], order_cache[blend_id], profile, "20241231")
                        row = {"profile_id": profile.profile_id, **asdict(profile), **metrics(daily, "20220606", "20241231")}
                        row["yearly_all_positive"] = yearly_positive(daily, "20220606", "20241231")
                        stage1_rows.append(row)
    stage1 = pd.DataFrame(stage1_rows).sort_values(
        ["yearly_all_positive", "sharpe", "max_drawdown", "profile_id"],
        ascending=[False, False, True, True],
    )
    stage1.to_csv(STAGE1_PATH, index=False, encoding="utf-8-sig")
    promoted = stage1[stage1["yearly_all_positive"]].head(12)
    stage1_payload = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profiles": promoted.to_dict("records"),
    }
    STAGE1_FROZEN_PATH.write_text(json.dumps(stage1_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if promoted.empty:
        empty_payload = {
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "stage1_sha256": sha256(STAGE1_PATH),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "validation_opened": False,
            "profiles": [],
            "stop_reason": "no_stage1_profile_passed_yearly_all_positive_gate",
        }
        if not FROZEN_CANDIDATES_PATH.exists():
            FROZEN_CANDIDATES_PATH.write_text(json.dumps(empty_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {
            "status": "research_only_no_candidate",
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "cache": str(CACHE_PATH),
            "stage1_cases": int(len(stage1)),
            "stage1_promoted": 0,
            "stage2_cases": 0,
            "frozen_candidates": 0,
            "validation_opened_this_run": False,
            "production_changed": False,
            "juejin_run": False,
            "stop_reason": empty_payload["stop_reason"],
        }
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return
    stage2_rows = []
    for _, base in promoted.iterrows():
        for min_hold in protocol["parameter_grid"]["min_holding_days"]:
            for max_hold in protocol["parameter_grid"]["max_holding_days"]:
                if max_hold <= min_hold:
                    continue
                for sell_below in protocol["parameter_grid"]["sell_rank_percentile_below"]:
                    for advantage in protocol["parameter_grid"]["replacement_score_advantage"]:
                        for invested in protocol["parameter_grid"]["target_invested_ratio"]:
                            profile = Profile(
                                str(base["blend_id"]), int(base["amount_min"]), int(base["mv_min"]),
                                int(base["top_n"]), float(base["entry_rank_min"]), min_hold, max_hold,
                                sell_below, advantage, invested,
                            )
                            daily = simulate(
                                arrays, blend_cache[profile.blend_id], order_cache[profile.blend_id], profile, "20251231"
                            )
                            select_metrics = metrics(daily, "20220606", "20241231")
                            confirm_metrics = metrics(daily, "20250101", "20251231")
                            stage2_rows.append(
                                {
                                    "profile_id": profile.profile_id,
                                    **asdict(profile),
                                    **{f"select_{k}": v for k, v in select_metrics.items()},
                                    **{f"confirm_{k}": v for k, v in confirm_metrics.items()},
                                    "selection_min_sharpe": min(select_metrics["sharpe"], confirm_metrics["sharpe"]),
                                }
                            )
    stage2 = pd.DataFrame(stage2_rows)
    stage2["confirm_positive"] = stage2["confirm_cumulative_return"] > 0
    stage2 = stage2.sort_values(
        ["confirm_positive", "selection_min_sharpe", "confirm_max_drawdown", "select_turnover", "profile_id"],
        ascending=[False, False, True, True, True],
    )
    stage2.to_csv(STAGE2_PATH, index=False, encoding="utf-8-sig")
    finalists = stage2[stage2["confirm_positive"]].head(8).copy()
    frozen_payload = {
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "stage1_sha256": sha256(STAGE1_PATH),
        "stage2_sha256": sha256(STAGE2_PATH),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "validation_opened": False,
        "profiles": finalists.to_dict("records"),
    }
    if not FROZEN_CANDIDATES_PATH.exists():
        FROZEN_CANDIDATES_PATH.write_text(json.dumps(frozen_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        existing = json.loads(FROZEN_CANDIDATES_PATH.read_text(encoding="utf-8"))
        if [row["profile_id"] for row in existing["profiles"]] != [row["profile_id"] for row in frozen_payload["profiles"]]:
            raise RuntimeError("Frozen candidate drift; refusing to overwrite")
    validation_rows = []
    if args.open_validation:
        frozen = json.loads(FROZEN_CANDIDATES_PATH.read_text(encoding="utf-8"))
        if VALIDATION_PATH.exists():
            raise RuntimeError("Final validation was already opened; refusing a second selection cycle")
        for base in frozen["profiles"]:
            profile = Profile(**{key: base[key] for key in Profile.__dataclass_fields__})
            daily = simulate(arrays, blend_cache[profile.blend_id], order_cache[profile.blend_id], profile, "99999999")
            full = metrics(daily, "20220606", "99999999")
            validation = metrics(daily, "20260101", "99999999")
            validation_rows.append(
                {
                    "profile_id": profile.profile_id,
                    **asdict(profile),
                    **{f"full_{k}": v for k, v in full.items()},
                    **{f"validation_{k}": v for k, v in validation.items()},
                }
            )
        pd.DataFrame(validation_rows).to_csv(VALIDATION_PATH, index=False, encoding="utf-8-sig")
        frozen["validation_opened"] = True
        frozen["validation_opened_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        frozen["validation_result_sha256"] = sha256(VALIDATION_PATH)
        FROZEN_CANDIDATES_PATH.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "research_only",
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "cache": str(CACHE_PATH),
        "stage1_cases": int(len(stage1)),
        "stage1_promoted": int(len(promoted)),
        "stage2_cases": int(len(stage2)),
        "frozen_candidates": int(len(finalists)),
        "validation_opened_this_run": bool(args.open_validation),
        "production_changed": False,
        "juejin_run": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
