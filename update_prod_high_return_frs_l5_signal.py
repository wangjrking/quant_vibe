from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
sys.path.insert(0, str(MAIN))

from l5_duckdb_sync import sync_strategy_registry_to_duckdb  # noqa: E402
from l6_duckdb_sync import sync_strategy_backtests_to_duckdb  # noqa: E402
from research_top1_three_tier_open_quality_20260714 import quality_flags  # noqa: E402
import update_fw_soft_deepdrop_l5_signal as fw  # noqa: E402


STRATEGY_ID = "prod_high_return_frs_scale090_cap090_v20260716"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / STRATEGY_ID
SIGNAL_DIR = STRATEGY_DIR / "signals"
FULL_HISTORY = SIGNAL_DIR / "full_history_high_return_frs_scale090_cap090.csv"
PRODUCTION_SIGNAL_DIR = ROOT / "quant" / "data_file" / "production_signals"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports"
BASE_REPORT_DIR = REPORT_DIR / "strategy_agent_high_return_frs_incremental_20260717_recovery_20260718"

TOP_N = 2
MF_HIGH = 0.72
MF_MID = 0.36
MF_LOW = 0.216
MF_SHALLOW_SCALE = 0.75
MF_POS_GAP_SCALE = 0.50
MF_DEEP_SCALE = 1.00
MF_DEEP_GAP_SCALE = 1.05
SS_SCALE = 0.92
HPS_SCALE = 1.04
LBC_BASE = 1.075
LBC_DEEP_SCALE = 0.86
LBC_Q1_SCALE = 1.20
LBC_Q0_SCALE = 1.50
OGD_DEEP_THRESHOLD = -8.0
OGD_DEEP_SCALE = 1.10
CORE_SCALE = 2.0
FRS_SCALE = 0.90
FRS_CAP = 0.90
FRS_REFILL_CAP = 0.03


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def expected_columns() -> list[str]:
    return pd.read_csv(FULL_HISTORY, nrows=0, encoding="utf-8-sig").columns.tolist()


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def base_row_target(row: pd.Series) -> float:
    bucket = int(row.get("quality_bucket", 0))
    pct = float(row.get("signal_pct_chg_raw", 0.0))
    gap = row.get("exec_open_gap_pct")
    gap = float(gap) if pd.notna(gap) else 99.0
    target = MF_HIGH if bucket == 2 else MF_MID if bucket == 1 else MF_LOW
    if pct > -3.0:
        target *= MF_SHALLOW_SCALE
    elif pct <= -5.0:
        target *= MF_DEEP_SCALE
    if gap > 0.5:
        target *= MF_POS_GAP_SCALE
    elif gap <= -3.0:
        target *= MF_DEEP_GAP_SCALE
    return max(0.0, target)


def cap_daily_sum(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    out = df.copy()
    daily_sum = out.groupby("buy_date")["target_pct"].transform("sum")
    scale = (cap / daily_sum).clip(upper=1.0)
    out["target_pct"] = out["target_pct"] * scale
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    return out


def build_base_candidate(signal_date: str, buy_date: str, buy_day_market_available: bool) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    con = duckdb.connect()
    try:
        sources = fw.attach_sources(con)
        l4_summary = fw.l4_coverage(con, sources)
        latest, audit = fw.build_latest(con, sources, signal_date, buy_date, buy_day_market_available)
    finally:
        con.close()
    return latest, audit, l4_summary


def apply_chain(base_latest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = base_latest.copy()
    ensure_numeric(
        df,
        [
            "pred_prob",
            "entry_score",
            "pred_1d",
            "pred_3d",
            "pred_5d",
            "pred_10d",
            "amount",
            "turnover_rate",
            "total_mv",
            "atr_qfq",
            "signal_pct_chg_raw",
            "buy_open_gap_pct",
            "buy_open_gap_raw_pct",
            "feature_weight_scale",
            "target_pct",
        ],
    )
    df["exec_open_gap_pct"] = df["buy_open_gap_raw_pct"].fillna(df["buy_open_gap_pct"])
    high_mask, mid_mask = quality_flags(df, "mid_atr12")
    df["quality_bucket"] = 0
    df.loc[mid_mask, "quality_bucket"] = 1
    df.loc[high_mask, "quality_bucket"] = 2
    df["sort_score"] = (
        df["quality_bucket"].astype(float) * 10.0
        + df["pred_10d"].fillna(0.0)
        + df["pred_1d"].fillna(0.0) * 0.05
    )
    mf = (
        df.sort_values(["buy_date", "sort_score", "rank"], ascending=[True, False, True])
        .groupby("buy_date", group_keys=False)
        .head(TOP_N)
        .copy()
    )
    mf["target_pct"] = mf.apply(base_row_target, axis=1)
    mf = mf[mf["target_pct"] > 0].copy()
    mf["rank"] = mf.groupby("buy_date")["sort_score"].rank(method="first", ascending=False).astype(int)
    mf = cap_daily_sum(mf, 1.0)

    final = mf.copy()
    final["target_pct"] = final["target_pct"] * SS_SCALE
    final = cap_daily_sum(final, 1.0)
    final["holding_days"] = 1
    final["max_holding_days"] = 3
    final["score_exit_entry_ratio"] = 0.98
    final["score_continue_entry_ratio"] = 1.02
    final["min_holding_days_before_score_exit"] = 1

    final["target_pct"] = final["target_pct"] * HPS_SCALE
    final = cap_daily_sum(final, 1.0)

    pct = final["signal_pct_chg_raw"].fillna(0.0)
    bucket = final["quality_bucket"].fillna(0).astype(int)
    scale = pd.Series(LBC_BASE, index=final.index, dtype=float)
    deep_mask = pct < -4.5
    q1_mask = (~deep_mask) & (bucket == 1)
    q0_mask = (~deep_mask) & (bucket == 0)
    scale = scale.mask(deep_mask, LBC_BASE * LBC_DEEP_SCALE)
    scale = scale.mask(q1_mask, scale * LBC_Q1_SCALE)
    scale = scale.mask(q0_mask, scale * LBC_Q0_SCALE)
    final["target_pct"] = final["target_pct"] * scale
    final["lowbucket_q0_boost"] = LBC_Q0_SCALE
    final["lowbucket_q0_boosted"] = q0_mask
    final = cap_daily_sum(final, 1.0)

    ogd_deep_mask = final["signal_pct_chg_raw"].fillna(0.0) < OGD_DEEP_THRESHOLD
    final["target_pct"] = final["target_pct"] * pd.Series(1.0, index=final.index).mask(ogd_deep_mask, OGD_DEEP_SCALE)
    final["open_gap_deep_rebalance_case"] = "ogd_deep8_up110_refresh"
    final = cap_daily_sum(final, 1.0)

    final["target_pct"] = final["target_pct"] * CORE_SCALE
    final["layer"] = "core"
    final = cap_daily_sum(final, 1.0)

    refill_mask = final["layer"].astype(str).eq("tiny_refill")
    scaled = final["target_pct"] * FRS_SCALE
    scaled.loc[refill_mask] = final.loc[refill_mask, "target_pct"].clip(upper=FRS_REFILL_CAP)
    final["target_pct"] = scaled.clip(lower=0.0, upper=FRS_CAP)
    final = cap_daily_sum(final, 1.0)

    final["strategy_variant"] = "frs_scale090_cap090_refresh"
    final["source_strategy_variant"] = "fw_soft_rebuilt_from_current_production_full_history"
    final["filter_name"] = "frs_scale090_cap090_refresh"
    final["entry_weight_name"] = "w25_25_00_50_reconstructed"
    final["dynamic_hold_name"] = "h1m1_reconstructed"
    final["hybrid_source"] = "base_production"
    final["rank"] = final.groupby("buy_date")["sort_score"].rank(method="first", ascending=False).astype(int)
    final = final.sort_values(["signal_date", "rank", "stock_code"], kind="stable").reset_index(drop=True)
    final = final.reindex(columns=expected_columns())
    return mf, final


def append_full_history(latest: pd.DataFrame) -> dict[str, Any]:
    history = pd.read_csv(FULL_HISTORY, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    before = len(history)
    dates = set(latest["signal_date"].astype(str))
    history = history[~history["signal_date"].astype(str).isin(dates)]
    history = pd.concat([history, latest], ignore_index=True, sort=False)
    history = history.reindex(columns=expected_columns())
    history = history.sort_values(["signal_date", "rank", "stock_code"], kind="stable")
    history.to_csv(FULL_HISTORY, index=False, encoding="utf-8-sig")
    return {
        "before_rows": int(before),
        "after_rows": int(len(history)),
        "signal_days": int(history["signal_date"].astype(str).nunique()),
        "buy_days": int(history["buy_date"].astype(str).nunique()),
        "duplicate_buy_stock_keys": int(history.duplicated(["buy_date", "stock_code"]).sum()),
        "full_history": str(FULL_HISTORY),
    }


def update_strategy_contracts(
    latest: pd.DataFrame,
    signal_date: str,
    buy_date: str,
    buy_day_market_available: bool,
    buy_date_source: str,
    audit: dict[str, Any],
    l4_summary: dict[str, Any],
    outputs: dict[str, str],
) -> dict[str, Any]:
    status = {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "status": "pending_buy_day_hard_gate" if not buy_day_market_available else "latest_signal_generated",
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_date_source": buy_date_source,
        "latest_signal_date": signal_date,
        "latest_buy_date": buy_date,
        "source_file": outputs["archive_latest_signal"],
        "output_file": outputs["production_latest_signal"],
        "row_count": int(len(latest)),
        "stock_count": int(latest["stock_code"].nunique()) if not latest.empty else 0,
        "duplicate_signal_stock_keys": int(latest.duplicated(["signal_date", "stock_code"]).sum()) if not latest.empty else 0,
        "buy_day_market_available": bool(buy_day_market_available),
        "buy_day_hard_gate_complete": bool(buy_day_market_available),
        "buy_day_realtime_checks_required": not bool(buy_day_market_available),
        "l7_execution_allowed": False,
        "note": (
            "L5/L6 latest signal asset is ready, but buy-date market data for 20260720 is not available in L2; "
            "L7 buy-day hard gate remains pending and no execution is allowed."
        ),
        "freshness": {
            "latest_signal_date": signal_date,
            "latest_buy_date": buy_date,
            "formal_manifest_checks": [
                {
                    "manifest_path": str(Path(info["manifest"])),
                    "max_trade_date": info["max_trade_date"],
                    "latest_rows": info["latest_rows"],
                    "duplicate_key_groups": info["duplicate_key_groups"],
                }
                for info in l4_summary.values()
            ],
        },
        "audit": audit,
    }

    manifest_path = STRATEGY_DIR / "strategy_manifest.json"
    validation_path = STRATEGY_DIR / "validation.json"
    manifest = load_json(manifest_path)
    manifest["latest_signal_file"] = outputs["archive_latest_signal"]
    manifest["latest_signal_status"] = status["status"]
    manifest["latest_signal_date"] = signal_date
    manifest["latest_buy_date"] = buy_date
    manifest["latest_signal_status_file"] = outputs["archive_status"]
    manifest["current_signal"] = {
        "latest_file": outputs["production_latest_signal"],
        "archive_latest_file": outputs["archive_latest_signal"],
        "signal_date": signal_date,
        "buy_date": buy_date,
        "latest_signal_date": signal_date,
        "latest_buy_date": buy_date,
        "buy_day_market_available": bool(buy_day_market_available),
        "buy_day_hard_gate_complete": bool(buy_day_market_available),
        "l7_execution_allowed": False,
        "status": status["status"],
    }
    write_json(manifest_path, manifest)

    validation = load_json(validation_path)
    validation.setdefault("date_coverage", {})["signal_date_max"] = signal_date
    validation.setdefault("date_coverage", {})["buy_date_max"] = buy_date
    signal_audit = validation.setdefault("signal_audit", {})
    history = audit.get("full_history_update", {})
    signal_audit["signal_rows"] = int(history.get("after_rows", signal_audit.get("signal_rows", 0)))
    signal_audit["signal_days"] = int(history.get("signal_days", signal_audit.get("signal_days", 0)))
    signal_audit["buy_days"] = int(history.get("buy_days", signal_audit.get("buy_days", 0)))
    signal_audit["latest_signal_rows"] = int(len(latest))
    signal_audit["duplicate_key_groups"] = int(history.get("duplicate_buy_stock_keys", 0))
    signal_audit["bj_rows"] = int(audit.get("bj_rows", 0))
    signal_audit["latest_target_pct_sum"] = float(audit.get("target_sum", 0.0))
    validation["latest_signal_status"] = status
    write_json(validation_path, validation)
    return status


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {STRATEGY_ID} L5/L6 incremental signal report",
        "",
        "## Conclusion",
        f"- signal_date: `{payload['signal_date']}`",
        f"- buy_date: `{payload['buy_date']}`",
        f"- buy_date_source: `{payload['buy_date_source']}`",
        f"- row_count: `{payload['status']['row_count']}`",
        f"- stock_count: `{payload['status']['stock_count']}`",
        f"- target_sum: `{payload['audit']['target_sum']:.12f}`",
        f"- status: `{payload['status']['status']}`",
        f"- l7_execution_allowed: `{payload['status']['l7_execution_allowed']}`",
        "",
        "## Latest rows",
        "",
        "| rank | stock_code | name | target_pct | sort_score | signal_pct_chg_raw | quality_bucket |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["latest_rows"]:
        lines.append(
            f"| {row['rank']} | {row['stock_code']} | {row['name']} | "
            f"{float(row['target_pct']):.12f} | {float(row['sort_score']):.12f} | "
            f"{float(row['signal_pct_chg_raw']):.4f} | {int(row['quality_bucket'])} |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "- active formal DuckDB manifests only",
            "- no-BJ",
            "- DuckDB-only",
            "- pending_buy_day_hard_gate until buy-date market rows land in L2",
            "",
            "## Evidence",
            f"- production latest: `{payload['outputs']['production_latest_signal']}`",
            f"- production status: `{payload['outputs']['production_status']}`",
            f"- archive latest: `{payload['outputs']['archive_latest_signal']}`",
            f"- archive status: `{payload['outputs']['archive_status']}`",
            f"- full history: `{payload['outputs']['full_history']}`",
            f"- base candidate: `{payload['outputs']['base_candidate']}`",
            f"- report json: `{payload['outputs']['report_json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-date", default="20260717")
    args = parser.parse_args()

    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCTION_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BASE_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    signal_date = str(args.signal_date)
    con = duckdb.connect()
    try:
        sources = fw.attach_sources(con)
        latest_common = str(fw.latest_common_signal_date(con, sources))
        if signal_date != latest_common:
            raise RuntimeError(f"signal_date {signal_date} does not match active formal L4 max common date {latest_common}")
        buy_date, buy_day_market_available, buy_date_source = fw.resolve_buy_date(con, signal_date)
    finally:
        con.close()

    base_latest, base_audit, l4_summary = build_base_candidate(signal_date, buy_date, buy_day_market_available)
    base_candidate_path = BASE_REPORT_DIR / f"{STRATEGY_ID}_fw_base_candidate_{signal_date}.csv"
    base_latest.to_csv(base_candidate_path, index=False, encoding="utf-8-sig")

    mf_stage, latest = apply_chain(base_latest)
    archive_latest = SIGNAL_DIR / f"latest_signal_{signal_date}_for_{buy_date}.csv"
    production_latest = PRODUCTION_SIGNAL_DIR / f"{STRATEGY_ID}_latest.csv"
    archive_status = SIGNAL_DIR / f"latest_signal_{signal_date}_for_{buy_date}_status.json"
    production_status = PRODUCTION_SIGNAL_DIR / f"{STRATEGY_ID}_latest_status.json"
    auto_status = SIGNAL_DIR / "latest_signal_status_auto.json"
    report_json = REPORT_DIR / f"strategy_agent_{STRATEGY_ID}_incremental_signal_{signal_date}.json"
    report_md = REPORT_DIR / f"strategy_agent_{STRATEGY_ID}_incremental_signal_{signal_date}.md"

    latest.to_csv(archive_latest, index=False, encoding="utf-8-sig")
    latest.to_csv(production_latest, index=False, encoding="utf-8-sig")
    history_update = append_full_history(latest)

    audit = {
        "base_candidate_file": str(base_candidate_path),
        "base_candidate_rows": int(len(base_latest)),
        "mf_stage_rows": int(len(mf_stage)),
        "final_signal_rows": int(len(latest)),
        "target_sum": float(latest["target_pct"].sum()) if not latest.empty else 0.0,
        "bj_rows": int(latest["stock_code"].astype(str).str.endswith(".BJ").sum()) if not latest.empty else 0,
        "duplicate_buy_stock_keys_full_history": int(history_update["duplicate_buy_stock_keys"]),
        "full_history_rows": int(history_update["after_rows"]),
        "full_history_signal_days": int(history_update["signal_days"]),
        "full_history_buy_days": int(history_update["buy_days"]),
        "legacy_or_research_only_input_used": False,
        "input_mode": "active formal DuckDB manifests only; production strategy chain replayed without parameter changes",
        "prod_fw_base_audit": base_audit,
        "full_history_update": history_update,
    }

    outputs = {
        "archive_latest_signal": str(archive_latest),
        "production_latest_signal": str(production_latest),
        "archive_status": str(archive_status),
        "production_status": str(production_status),
        "auto_status": str(auto_status),
        "full_history": str(FULL_HISTORY),
        "base_candidate": str(base_candidate_path),
        "report_json": str(report_json),
        "report_md": str(report_md),
    }
    status = update_strategy_contracts(
        latest,
        signal_date,
        buy_date,
        buy_day_market_available,
        buy_date_source,
        audit,
        l4_summary,
        outputs,
    )
    for path in (archive_status, production_status, auto_status):
        write_json(path, status)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_id": STRATEGY_ID,
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_date_source": buy_date_source,
        "latest_rows": latest.to_dict("records"),
        "base_candidate_rows": base_latest.to_dict("records"),
        "status": status,
        "audit": audit,
        "l4_coverage": l4_summary,
        "outputs": outputs,
    }
    write_json(report_json, payload)
    report_md.write_text(render_report(payload), encoding="utf-8")

    sync_l5 = sync_strategy_registry_to_duckdb(project_dir=MAIN)
    sync_l6 = sync_strategy_backtests_to_duckdb(project_dir=MAIN)
    print(
        json.dumps(
            {
                "strategy_id": STRATEGY_ID,
                "signal_date": signal_date,
                "buy_date": buy_date,
                "rows": int(len(latest)),
                "buy_day_market_available": buy_day_market_available,
                "l7_execution_allowed": False,
                "outputs": outputs,
                "sync_l5": sync_l5,
                "sync_l6": sync_l6,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
