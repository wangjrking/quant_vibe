from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant/main"
DATA = ROOT / "quant/data_file"
STRATEGY_ID = "prod_v260_10d_regime_warmup_all4key_v20260724"
STRATEGY_DIR = MAIN / "strategy_library/production" / STRATEGY_ID
PRODUCTION_CODE_ROOT = STRATEGY_DIR / "production_code"
if str(PRODUCTION_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_CODE_ROOT))

from v260_all4key_runtime import (  # noqa: E402
    production_v260_active_l4_10d_smoothing_refine_v95_20260722 as v95,
)
from v260_all4key_runtime import (  # noqa: E402
    production_v260_active_l4_score_deterioration_v174_20260722 as v174,
)
from v260_all4key_runtime import (  # noqa: E402
    production_v260_active_l4_v258_position_warmup_v260_20260723 as v260,
)


REGISTRY = MAIN / "strategy_library/registry.json"
L2 = DATA / "production_assets/duckdb/l2_stock_daily_data.duckdb"
FORMAL_MANIFESTS = {
    "1d": MAIN
    / "config/prediction_manifests/"
    "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN
    / "config/prediction_manifests/"
    "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN
    / "config/prediction_manifests/"
    "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN
    / "config/prediction_manifests/"
    "executable_10d_open_return_l4_formal_20260617.json",
}
PROTOCOL = STRATEGY_DIR / "inputs/preregistered_protocol.json"
FROZEN_ACTIONS = STRATEGY_DIR / "signals/full_history_actions.csv"
CURRENT_ACTIONS = STRATEGY_DIR / "signals/full_history_actions_current.csv"
VERIFIER = STRATEGY_DIR / "code_snapshot/verify_candidate.py"
L5_REGISTRY_DUCKDB = (
    DATA
    / "production_assets"
    / "duckdb"
    / "production"
    / "l5"
    / "prod_l5_strategy_registry_current.duckdb"
)
L5_MANIFEST_DUCKDB = L5_REGISTRY_DUCKDB.with_name(
    "prod_l5_strategy_manifest_current.duckdb"
)
L6_VALIDATION_DUCKDB = (
    DATA
    / "production_assets"
    / "duckdb"
    / "production"
    / "l6"
    / "prod_l6_strategy_validation_current.duckdb"
)
INTENT_COLUMNS = [
    "strategy_id",
    "signal_date",
    "buy_date",
    "action",
    "stock_code",
    "name",
    "market",
    "strategy_score",
    "score_rank",
    "score_denominator",
    "target_pct",
    "reason",
    "execution_open_raw",
    "status",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": None,
            "mtime_utc": None,
            "sha256": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        ),
        "sha256": sha256(path),
    }


def rollback_sources() -> dict[str, Path]:
    return {
        "latest_csv": DATA / "production_signals" / f"{STRATEGY_ID}_latest.csv",
        "latest_status": DATA
        / "production_signals"
        / f"{STRATEGY_ID}_latest_status.json",
        "strategy_manifest": STRATEGY_DIR / "strategy_manifest.json",
        "validation_json": STRATEGY_DIR / "validation.json",
        "current_actions": CURRENT_ACTIONS,
        "l5_registry_duckdb": L5_REGISTRY_DUCKDB,
        "l5_manifest_duckdb": L5_MANIFEST_DUCKDB,
        "l6_validation_duckdb": L6_VALIDATION_DUCKDB,
    }


def capture_prewrite_snapshot(
    report_dir: Path,
    signal_date: str,
    buy_date: str,
    *,
    sources: dict[str, Path] | None = None,
) -> dict[str, Any]:
    snapshot_sources = sources or rollback_sources()
    snapshot_dir = report_dir / "rollback_before_write"
    manifest_path = report_dir / "rollback_snapshot_manifest.json"

    if manifest_path.is_file():
        payload = load_json(manifest_path)
        for label, source in snapshot_sources.items():
            recorded = payload.get("sources", {}).get(label, {}).get("source_state")
            if recorded is None:
                raise RuntimeError(
                    f"existing rollback manifest is missing source_state for {label}"
                )
            if file_state(source) != recorded:
                raise RuntimeError(
                    "existing rollback snapshot manifest no longer matches live "
                    f"source file: {label}"
                )
        return payload

    missing = [label for label, path in snapshot_sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            "prewrite rollback snapshot requires existing active sources, missing: "
            + ", ".join(sorted(missing))
        )

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for label, source in snapshot_sources.items():
        suffix = "".join(source.suffixes)
        snapshot_path = snapshot_dir / f"{label}{suffix}"
        shutil.copy2(source, snapshot_path)
        entries[label] = {
            "source_state": file_state(source),
            "snapshot_path": str(snapshot_path),
            "snapshot_state": file_state(snapshot_path),
        }

    payload = {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "signal_date": signal_date,
        "buy_date": buy_date,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_dir": str(snapshot_dir),
        "sources": entries,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def clean_sql(alias: str) -> str:
    return f"""
      coalesce(cast({alias}.ST_TYPE as varchar), '') in ('', '0', '0.0', 'None', 'NONE')
      and upper(coalesce(cast({alias}.ST_TYPE_name as varchar), '')) not like '%ST%'
      and upper(coalesce(cast({alias}.name as varchar), '')) not like 'ST%'
      and upper(coalesce(cast({alias}.name as varchar), '')) not like '*ST%'
      and coalesce(cast({alias}.name as varchar), '') not like '%退%'
    """


def validate_route(expected_signal_date: str | None = None) -> dict[str, dict]:
    registry = load_json(REGISTRY)
    production = registry.get("production", {})
    if production.get("current") != STRATEGY_ID:
        raise RuntimeError("production.current does not point to remediated V260")
    if production.get("state") != "production_active":
        raise RuntimeError(
            f"production strategy is not active: {production.get('state')}"
        )
    strategies = production.get("strategies", [])
    if len(strategies) != 1 or strategies[0].get("strategy_id") != STRATEGY_ID:
        raise RuntimeError("production strategy registry is not uniquely remediated V260")

    manifests = {label: load_json(path) for label, path in FORMAL_MANIFESTS.items()}
    max_dates = set()
    for label, manifest in manifests.items():
        if (
            manifest.get("approval_status") != "approved_for_l5"
            or manifest.get("source_type") != "duckdb_table"
        ):
            raise RuntimeError(f"{label} manifest is not approved DuckDB formal")
        serialized = json.dumps(manifest, ensure_ascii=False).lower()
        if (
            "sqlite" in serialized
            or "odb.db" in serialized
            or "research_only" in serialized
        ):
            raise RuntimeError(
                f"legacy or research-only production input detected: {label}"
            )
        db_path, table = resolve_formal_source(FORMAL_MANIFESTS[label], manifest)
        with duckdb.connect(str(db_path), read_only=True) as con:
            row = con.execute(
                f'SELECT MAX(trade_date), COUNT(*) FROM "{table}"'
            ).fetchone()
        if str(row[0]) != str(manifest.get("max_trade_date")):
            raise RuntimeError(f"{label} manifest/table max date mismatch")
        max_dates.add(str(manifest.get("max_trade_date")))
    if len(max_dates) != 1:
        raise RuntimeError(f"misaligned formal signal dates: {max_dates}")
    signal_date = next(iter(max_dates))
    if expected_signal_date and signal_date != expected_signal_date:
        raise RuntimeError(
            f"formal signal date mismatch: expected {expected_signal_date}, got {signal_date}"
        )
    subprocess.run([sys.executable, str(VERIFIER)], cwd=str(ROOT), check=True)
    return manifests


def resolve_formal_source(manifest_path: Path, manifest: dict) -> tuple[Path, str]:
    db_path = (manifest_path.parent / str(manifest["db_path"])).resolve()
    if db_path.suffix.lower() != ".duckdb" or not db_path.is_file():
        raise RuntimeError(f"formal DuckDB source is unavailable: {manifest_path}")
    return db_path, str(manifest["table"])


def build_arrays(manifests: dict[str, dict]) -> dict[str, np.ndarray]:
    arrays = v260.v258.v252.fresh_arrays()
    signal_date = str(manifests["10d"]["max_trade_date"])
    if str(arrays["dates"][-1]) != signal_date:
        raise RuntimeError("four-manifest key-domain rebuild did not reach signal date")
    return arrays


def active_holdings(actions: pd.DataFrame, dates: np.ndarray) -> dict[str, int]:
    date_index = {str(value): idx for idx, value in enumerate(dates.astype(str))}
    holdings: dict[str, int] = {}
    for row in actions.itertuples(index=False):
        stock = str(row.stock_code)
        if str(row.action).upper() == "BUY":
            holdings[stock] = date_index[str(row.signal_date)]
        else:
            holdings.pop(stock, None)
    return holdings


def compare_frozen_prefix(actions: pd.DataFrame) -> None:
    frozen = pd.read_csv(FROZEN_ACTIONS, dtype={"signal_date": str, "buy_date": str})
    if list(frozen.columns) != list(actions.columns):
        raise RuntimeError("frozen action schema mismatch")
    if len(frozen) != len(actions):
        raise RuntimeError(
            f"frozen action row mismatch: expected {len(frozen)}, got {len(actions)}"
        )
    for column in ("signal_date", "buy_date", "action", "stock_code"):
        if not frozen[column].astype(str).equals(actions[column].astype(str)):
            raise RuntimeError(f"frozen action key mismatch: {column}")
    for column in ("target_pct", "execution_open_raw"):
        if not np.allclose(
            frozen[column].to_numpy(dtype=float),
            actions[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
            equal_nan=True,
        ):
            raise RuntimeError(f"frozen action value mismatch: {column}")


def next_weekday(date_text: str) -> str:
    value = datetime.strptime(date_text, "%Y%m%d").date() + timedelta(days=1)
    while value.weekday() >= 5:
        value += timedelta(days=1)
    return value.strftime("%Y%m%d")


def names_for(signal_date: str) -> pd.DataFrame:
    with duckdb.connect(str(L2), read_only=True) as con:
        frame = con.execute(
            """
            SELECT stock_code, any_value(name) AS name, any_value(market) AS market
            FROM STOCK_DAILY_DATA
            WHERE trade_date=?
            GROUP BY stock_code
            """,
            [signal_date],
        ).fetchdf()
    return frame.drop_duplicates("stock_code").set_index("stock_code")


def pending_intents(
    arrays: dict[str, np.ndarray],
    score: np.ndarray,
    order: np.ndarray,
    actions: pd.DataFrame,
    protocol: dict,
    definition: dict,
    signal_date: str,
    buy_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    stock_index = {stock: idx for idx, stock in enumerate(stocks)}
    t = int(np.where(dates == signal_date)[0][0])
    holdings = active_holdings(actions, dates)

    selection = v174.selection_mask(
        arrays, float(definition["max_rank_deterioration"])
    )[t]
    clean = arrays["signal_clean"][t] & np.isfinite(score[t]) & selection
    clean &= arrays["listed_days"][t] >= int(protocol["fixed_universe"]["listed_days_min"])
    clean &= np.isfinite(arrays["amount"][t]) & (
        arrays["amount"][t] >= int(protocol["fixed_universe"]["amount_min"])
    )
    clean &= np.isfinite(arrays["total_mv"][t]) & (
        arrays["total_mv"][t] >= int(protocol["fixed_universe"]["mv_min"])
    )
    clean &= np.isfinite(arrays["turnover_rate"][t])
    clean &= arrays["turnover_rate"][t] >= 0
    clean &= arrays["turnover_rate"][t] <= float(
        protocol["fixed_universe"]["turnover_max"]
    )
    ranked = [int(idx) for idx in order[t] if clean[int(idx)]]
    best_unheld = max(
        (
            float(score[t, idx])
            for idx in ranked
            if stocks[idx] not in holdings
        ),
        default=-np.inf,
    )
    min_hold = v260.v258.v252.v162.min_hold_schedule(
        arrays, definition["min_hold_policy"]
    )
    sell_stocks: list[str] = []
    sell_reasons: dict[str, str] = {}
    for stock, entry_t in sorted(holdings.items()):
        idx = stock_index[stock]
        age = t - entry_t
        current = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
        score_exit = (
            age >= int(min_hold[t])
            and current < float(definition["sell_score_below"])
            and best_unheld - current >= float(definition["replacement_advantage"])
        )
        if age >= int(definition["max_hold_days"]) or score_exit:
            sell_stocks.append(stock)
            sell_reasons[stock] = (
                "max_hold"
                if age >= int(definition["max_hold_days"])
                else "score_replacement"
            )

    holdings_after_sell = set(holdings) - set(sell_stocks)
    max_positions = int(v260.position_schedule(arrays, definition, None)[t])
    slots = min(1, max(max_positions - len(holdings_after_sell), 0))
    buy_indices = [
        idx for idx in ranked if stocks[idx] not in holdings_after_sell
    ][:slots]
    target = v260.v258.v252.v162.v153.schedules(arrays, definition)[1]
    multiplier = v260.v258.v252.warmup_multiplier(
        arrays, score, definition, protocol, None
    )
    names = names_for(signal_date)
    valid_scores = score[t][np.isfinite(score[t])]

    def enrich(stock: str, idx: int) -> dict:
        value = float(score[t, idx]) if np.isfinite(score[t, idx]) else None
        return {
            "stock_code": stock,
            "name": str(names.loc[stock, "name"]) if stock in names.index else "",
            "market": str(names.loc[stock, "market"]) if stock in names.index else "",
            "strategy_score": value,
            "score_rank": (
                int(np.sum(valid_scores > value) + 1) if value is not None else None
            ),
            "score_denominator": int(len(valid_scores)),
        }

    rows: list[dict] = []
    for stock in sell_stocks:
        idx = stock_index[stock]
        rows.append(
            {
                "strategy_id": STRATEGY_ID,
                "signal_date": signal_date,
                "buy_date": buy_date,
                "action": "SELL",
                **enrich(stock, idx),
                "target_pct": 0.0,
                "reason": sell_reasons[stock],
                "execution_open_raw": None,
                "status": "pending_buy_day_hard_gate",
            }
        )
    for idx in buy_indices:
        stock = stocks[idx]
        rows.append(
            {
                "strategy_id": STRATEGY_ID,
                "signal_date": signal_date,
                "buy_date": buy_date,
                "action": "BUY",
                **enrich(stock, idx),
                "target_pct": float(target[t] * multiplier[t, idx]),
                "reason": "top_signal_candidate_after_planned_sells",
                "execution_open_raw": None,
                "status": "pending_buy_day_hard_gate",
            }
        )

    candidates = []
    for rank_position, idx in enumerate(ranked[:20], start=1):
        stock = stocks[idx]
        candidates.append(
            {
                "rank_position": rank_position,
                **enrich(stock, idx),
                "already_held": stock in holdings,
                "planned_sell": stock in sell_stocks,
                "amount_thousand_cny": float(arrays["amount"][t, idx]),
                "turnover_rate": float(arrays["turnover_rate"][t, idx]),
                "status": "pending_buy_day_hard_gate",
            }
        )
    audit = {
        "signal_day_universe_count": int(np.sum(clean)),
        "holding_count_before": len(holdings),
        "planned_sell_count": len(sell_stocks),
        "planned_buy_count": len(buy_indices),
        "max_positions": max_positions,
        "holding_count_after_planned_actions": (
            len(holdings_after_sell) + len(buy_indices)
        ),
    }
    return (
        pd.DataFrame(rows, columns=INTENT_COLUMNS),
        pd.DataFrame(candidates),
        audit,
    )


def write_pending_assets(
    manifests: dict[str, dict],
    signal_dir: Path,
    buy_date: str,
    *,
    buy_date_source: str,
    l5_only: bool = False,
) -> dict:
    protocol = load_json(PROTOCOL)
    arrays = build_arrays(manifests)
    signal_date = str(manifests["10d"]["max_trade_date"])
    dates = arrays["dates"].astype(str)
    signal_idx = int(np.where(dates == signal_date)[0][0])
    previous_signal_date = str(dates[signal_idx - 1])
    definition = v260.definition_for(protocol, 50)
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)

    _, frozen_prefix = v260.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        "20260721",
        record_actions=True,
    )
    compare_frozen_prefix(frozen_prefix)

    _, actions = v260.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        previous_signal_date,
        record_actions=True,
    )
    actions.to_csv(CURRENT_ACTIONS, index=False, encoding="utf-8-sig")
    intents, candidates, audit = pending_intents(
        arrays,
        score,
        order,
        actions,
        protocol,
        definition,
        signal_date,
        buy_date,
    )
    duplicate_keys = int(
        intents.duplicated(["signal_date", "buy_date", "action", "stock_code"]).sum()
    )
    bj_rows = int(intents["stock_code"].astype(str).str.endswith(".BJ").sum())
    if duplicate_keys or bj_rows:
        raise RuntimeError("pending signal duplicate or BJ gate failed")

    signal_dir.mkdir(parents=True, exist_ok=True)
    latest_path = signal_dir / f"{STRATEGY_ID}_latest.csv"
    archive_path = (
        STRATEGY_DIR
        / "signals"
        / f"latest_signal_{signal_date}_for_{buy_date}.csv"
    )
    candidate_path = (
        STRATEGY_DIR
        / "signals"
        / f"latest_candidates_{signal_date}_for_{buy_date}.csv"
    )
    intents.to_csv(latest_path, index=False, encoding="utf-8-sig")
    intents.to_csv(archive_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    no_signal = bool(intents.empty)
    signal_semantics = (
        "no_signal_hold_only"
        if no_signal
        else "trade_actions_pending_buy_day_hard_gate"
    )
    status = {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "status": "pending_buy_day_hard_gate",
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_date_source": buy_date_source,
        "signal_semantics": signal_semantics,
        "no_signal": no_signal,
        "hold_only": no_signal,
        "no_signal_reason": (
            "strategy_generated_no_new_buy_or_sell_intents"
            if no_signal
            else None
        ),
        "source_manifests": {
            label: str(path) for label, path in FORMAL_MANIFESTS.items()
        },
        "source_prediction_tables": {
            label: str(manifests[label]["table"]) for label in FORMAL_MANIFESTS
        },
        "key_domain_contract": "inner_intersection_1d_3d_5d_10d",
        "score_contract": "10d_rank_only_other_labels_key_domain_only",
        "source_file": str(archive_path),
        "output_file": str(latest_path),
        "candidate_file": str(candidate_path),
        "current_action_history": str(CURRENT_ACTIONS),
        "action_count": int(len(intents)),
        "row_count": int(len(intents)),
        "stock_count": int(intents["stock_code"].nunique()),
        "buy_count": int((intents["action"] == "BUY").sum()),
        "sell_count": int((intents["action"] == "SELL").sum()),
        "duplicate_key_count": duplicate_keys,
        "bj_rows": bj_rows,
        "buy_day_market_available": False,
        "buy_day_hard_gate_complete": False,
        "l7_execution_allowed": False,
        "execution_allowed": False,
        "approved_for_execution": False,
        "auto_execution_allowed": False,
        "live_execution_allowed": False,
        "approval_status": "pending_l5_audit" if l5_only else "pending_l5_l6_audit",
        "formal_signal_generated": True,
        "formal_batch_generated": True,
        "formal_signal_semantics": "l5_strategy_intent_pending_l7_buy_day_gate",
        "frozen_prefix_action_sha256": sha256(FROZEN_ACTIONS),
        "legacy_or_research_data_input_used": False,
        "audit": audit,
        "note": (
            "该资产是正式 L5 策略意图，不是可执行交易指令；"
            f"{buy_date} 真实开盘行情硬门控和 "
            f"{'L5' if l5_only else 'L5/L6'} 审计均未完成。"
        ),
    }
    status_path = STRATEGY_DIR / "signals/latest_signal_status.json"
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (signal_dir / f"{STRATEGY_ID}_latest_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    strategy_manifest_path = STRATEGY_DIR / "strategy_manifest.json"
    strategy_manifest = load_json(strategy_manifest_path)
    strategy_manifest["latest_signal"] = {
        "status": status["status"],
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_day_market_available": False,
        "buy_day_hard_gate_complete": False,
        "l7_execution_allowed": False,
        "signal_semantics": signal_semantics,
        "no_signal": no_signal,
        "hold_only": no_signal,
        "execution_allowed": False,
        "approved_for_execution": False,
        "auto_execution_allowed": False,
        "live_execution_allowed": False,
    }
    strategy_manifest["current_signal"] = {
        "latest_file": str(latest_path),
        "signal_date": signal_date,
        "buy_date": buy_date,
        "latest_signal_date": signal_date,
        "latest_buy_date": buy_date,
        "buy_day_hard_gate_complete": False,
        "status": status["status"],
        "l7_execution_allowed": False,
        "signal_semantics": signal_semantics,
        "no_signal": no_signal,
        "hold_only": no_signal,
        "execution_allowed": False,
        "approved_for_execution": False,
        "auto_execution_allowed": False,
        "live_execution_allowed": False,
    }
    strategy_manifest_path.write_text(
        json.dumps(strategy_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not l5_only:
        validation_path = STRATEGY_DIR / "validation.json"
        validation = load_json(validation_path)
        validation["latest_signal_status"] = {
            key: status[key]
            for key in (
                "status",
                "signal_date",
                "buy_date",
                "buy_date_source",
                "signal_semantics",
                "no_signal",
                "hold_only",
                "action_count",
                "row_count",
                "stock_count",
                "buy_count",
                "sell_count",
                "duplicate_key_count",
                "bj_rows",
                "buy_day_market_available",
                "buy_day_hard_gate_complete",
                "l7_execution_allowed",
                "execution_allowed",
                "approved_for_execution",
                "auto_execution_allowed",
                "live_execution_allowed",
                "approval_status",
                "formal_signal_generated",
                "formal_batch_generated",
            )
        }
        validation_path.write_text(
            json.dumps(validation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 正式 L5 信号导出入口")
    parser.add_argument(
        "--signal-dir", default=str(DATA / "production_signals")
    )
    parser.add_argument("--signal-date")
    parser.add_argument("--buy-date")
    parser.add_argument(
        "--buy-date-source",
        default="next_weekday_estimate_l2_buy_day_market_not_available",
    )
    parser.add_argument("--report-dir")
    parser.add_argument("--l5-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifests = validate_route(args.signal_date)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "strategy_id": STRATEGY_ID,
                    "manifest_max_trade_date": manifests["10d"]["max_trade_date"],
                    "formal_signal_generated": False,
                },
                ensure_ascii=False,
            )
        )
        return
    buy_date = str(
        args.buy_date or next_weekday(str(manifests["10d"]["max_trade_date"]))
    )
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else DATA
        / "reports"
        / f"strategy_agent_v260_all4key_l5_l6_latest_{manifests['10d']['max_trade_date']}"
    )
    with duckdb.connect(str(L2), read_only=True) as con:
        buy_day_rows = int(
            con.execute(
                "SELECT COUNT(*) FROM STOCK_DAILY_DATA WHERE trade_date=?",
                [buy_date],
            ).fetchone()[0]
        )
    if buy_day_rows:
        raise RuntimeError(
            "buy-day market is already available; pending-only L5 generation refuses "
            "to bypass the separate execution hard-gate stage"
        )
    rollback = rollback_sources()
    if args.l5_only:
        rollback = {
            key: value
            for key, value in rollback.items()
            if key not in {"validation_json", "l6_validation_duckdb"}
        }
    capture_prewrite_snapshot(
        Path(report_dir),
        str(manifests["10d"]["max_trade_date"]),
        buy_date,
        sources=rollback,
    )
    status = write_pending_assets(
        manifests,
        Path(args.signal_dir).resolve(),
        buy_date,
        buy_date_source=str(args.buy_date_source),
        l5_only=args.l5_only,
    )
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
