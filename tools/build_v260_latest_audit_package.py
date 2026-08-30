from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
STRATEGY_ID = "prod_v260_10d_regime_warmup_all4key_v20260724"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / STRATEGY_ID
LATEST_CSV = DATA / "production_signals" / f"{STRATEGY_ID}_latest.csv"
LATEST_STATUS = DATA / "production_signals" / f"{STRATEGY_ID}_latest_status.json"
REGISTRY = MAIN / "strategy_library" / "registry.json"
MANIFEST = STRATEGY_DIR / "strategy_manifest.json"
L5_REGISTRY = (
    DATA
    / "production_assets"
    / "duckdb"
    / "production"
    / "l5"
    / "prod_l5_strategy_registry_current.duckdb"
)
L5_MANIFEST = L5_REGISTRY.with_name("prod_l5_strategy_manifest_current.duckdb")
L6_VALIDATION = (
    DATA
    / "production_assets"
    / "duckdb"
    / "production"
    / "l6"
    / "prod_l6_strategy_validation_current.duckdb"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def fetch_one(path: Path, sql: str) -> dict[str, Any]:
    with duckdb.connect(str(path), read_only=True) as connection:
        frame = connection.execute(sql).fetchdf()
    if len(frame) != 1:
        raise RuntimeError(f"{path.name} expected one row, got {len(frame)}")
    return frame.iloc[0].where(pd.notna(frame.iloc[0]), None).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 V260 L5/L6 latest 只读审计证据")
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--buy-date", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rollback_manifest = output_dir / "rollback_snapshot_manifest.json"
    registry = load_json(REGISTRY)
    manifest = load_json(MANIFEST)
    status = load_json(LATEST_STATUS)
    signals = pd.read_csv(LATEST_CSV, dtype={"signal_date": str, "buy_date": str})
    l5_registry = fetch_one(
        L5_REGISTRY,
        'SELECT * FROM "prod_l5_strategy_registry_current"',
    )
    l5_manifest = fetch_one(
        L5_MANIFEST,
        'SELECT * FROM "prod_l5_strategy_manifest_current"',
    )
    l6_validation = fetch_one(
        L6_VALIDATION,
        'SELECT * FROM "prod_l6_strategy_validation_current"',
    )
    current = registry["production"]["current"]
    archive_csv = (
        STRATEGY_DIR
        / "signals"
        / f"latest_signal_{args.signal_date}_for_{args.buy_date}.csv"
    )
    no_signal = bool(status.get("no_signal"))
    no_signal_contract = (
        signals.empty
        and status.get("signal_semantics") == "no_signal_hold_only"
        and status.get("hold_only") is True
        and int(status.get("action_count", -1)) == 0
        and int(status.get("row_count", -1)) == 0
        and int(status.get("buy_count", -1)) == 0
        and int(status.get("sell_count", -1)) == 0
    )
    trade_action_contract = (
        not signals.empty
        and not no_signal
        and status.get("signal_semantics")
        == "trade_actions_pending_buy_day_hard_gate"
    )

    checks = {
        "production_current_match": current == STRATEGY_ID,
        "manifest_strategy_match": manifest.get("strategy_id") == STRATEGY_ID,
        "signal_date_match": (
            status.get("signal_date") == args.signal_date
            if signals.empty
            else set(signals["signal_date"]) == {args.signal_date}
        ),
        "buy_date_match": (
            status.get("buy_date") == args.buy_date
            if signals.empty
            else set(signals["buy_date"]) == {args.buy_date}
        ),
        "status_date_match": status.get("signal_date") == args.signal_date
        and status.get("buy_date") == args.buy_date,
        "current_date_match": str(l5_manifest.get("latest_signal_date")) == args.signal_date
        and str(l5_manifest.get("latest_buy_date")) == args.buy_date
        and str(l6_validation.get("latest_signal_date")) == args.signal_date
        and str(l6_validation.get("latest_buy_date")) == args.buy_date,
        "duplicate_zero": int(
            signals.duplicated(["signal_date", "buy_date", "action", "stock_code"]).sum()
        )
        == 0,
        "bj_zero": int(signals["stock_code"].astype(str).str.endswith(".BJ").sum()) == 0,
        "pending_hard_gate": status.get("status") == "pending_buy_day_hard_gate"
        and status.get("buy_day_hard_gate_complete") is False
        and status.get("l7_execution_allowed") is False,
        "batch_semantics_valid": no_signal_contract or trade_action_contract,
        "execution_flags_fail_closed": all(
            status.get(field) is False
            for field in (
                "execution_allowed",
                "approved_for_execution",
                "auto_execution_allowed",
                "live_execution_allowed",
            )
        )
        and status.get("approval_status") == "pending_l5_l6_audit",
        "approved_duckdb_all4_contract": status.get("key_domain_contract")
        == "inner_intersection_1d_3d_5d_10d"
        and status.get("score_contract")
        == "10d_rank_only_other_labels_key_domain_only"
        and status.get("legacy_or_research_data_input_used") is False,
        "archive_hash_match": archive_csv.is_file()
        and sha256(archive_csv) == sha256(LATEST_CSV),
        "rollback_snapshot_manifest_present": rollback_manifest.is_file(),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"fail-closed checks failed: {failed}")

    rows = signals.where(pd.notna(signals), None).to_dict(orient="records")
    report = {
        "schema_version": 1,
        "task_id": (
            f"incremental-trading-signal-{args.signal_date}-"
            "L5-L6-v260-all4key-latest"
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_id": STRATEGY_ID,
        "signal_date": args.signal_date,
        "buy_date": args.buy_date,
        "buy_date_source": status.get("buy_date_source"),
        "latest": {
            "rows": len(signals),
            "stocks": int(signals["stock_code"].nunique()),
            "buys": int((signals["action"] == "BUY").sum()),
            "sells": int((signals["action"] == "SELL").sum()),
            "duplicate_keys": 0,
            "bj_rows": 0,
            "target_pct_sum": float(signals["target_pct"].fillna(0).sum()),
            "signals": rows,
        },
        "status": {
            "value": status.get("status"),
            "signal_semantics": status.get("signal_semantics"),
            "no_signal": no_signal,
            "hold_only": status.get("hold_only"),
            "buy_day_market_available": status.get("buy_day_market_available"),
            "buy_day_hard_gate_complete": status.get("buy_day_hard_gate_complete"),
            "l7_execution_allowed": status.get("l7_execution_allowed"),
            "execution_allowed": status.get("execution_allowed"),
            "approved_for_execution": status.get("approved_for_execution"),
            "auto_execution_allowed": status.get("auto_execution_allowed"),
            "live_execution_allowed": status.get("live_execution_allowed"),
            "approval_status": status.get("approval_status"),
        },
        "current_readback": {
            "l5_registry": l5_registry,
            "l5_manifest": l5_manifest,
            "l6_validation": l6_validation,
        },
        "checks": checks,
        "artifacts": {
            "latest_csv": str(LATEST_CSV),
            "latest_csv_sha256": sha256(LATEST_CSV),
            "latest_status": str(LATEST_STATUS),
            "latest_status_sha256": sha256(LATEST_STATUS),
            "archive_csv": str(archive_csv),
            "strategy_manifest": str(MANIFEST),
            "strategy_manifest_sha256": sha256(MANIFEST),
            "l5_registry": f"{L5_REGISTRY}::prod_l5_strategy_registry_current",
            "l5_registry_sha256": sha256(L5_REGISTRY),
            "l5_manifest": f"{L5_MANIFEST}::prod_l5_strategy_manifest_current",
            "l5_manifest_sha256": sha256(L5_MANIFEST),
            "l6_validation": f"{L6_VALIDATION}::prod_l6_strategy_validation_current",
            "l6_validation_sha256": sha256(L6_VALIDATION),
            "rollback_snapshot_manifest": (
                str(rollback_manifest) if rollback_manifest.is_file() else None
            ),
            "rollback_snapshot_manifest_sha256": (
                sha256(rollback_manifest) if rollback_manifest.is_file() else None
            ),
        },
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    handoff = {
        "schema_version": 1,
        "from_agent": "strategy-agent",
        "to_agent": "audit-agent",
        "task_id": report["task_id"],
        "status": "ready_for_audit_review",
        "strategy_id": STRATEGY_ID,
        "signal_date": args.signal_date,
        "buy_date": args.buy_date,
        "audit_focus": [
            "production.current 与 V260 all4key 策略一致",
            "四表共同键域且仅 10D 评分",
            "正式 latest 与策略归档哈希一致",
            "L5/L6 current 日期与 latest 一致",
            "重复键为 0、北交所为 0、未读取 legacy/research",
            "买入日硬门未完成且 L7 不允许执行",
        ],
        "evidence": list(report["artifacts"].values()),
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
    }
    (output_dir / "latest_generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "latest_audit_handoff.json").write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = (
        f"# V260 {args.signal_date} L5/L6 最新信号自审\n\n"
        f"- 策略：`{STRATEGY_ID}`\n"
        f"- 信号日：`{args.signal_date}`\n"
        f"- 买入日：`{args.buy_date}`\n"
        f"- 正式信号：{len(signals)} 行，{signals['stock_code'].nunique()} 只股票\n"
        "- 重复键：0\n"
        "- 北交所：0\n"
        "- 状态：`pending_buy_day_hard_gate`\n"
        "- L7 执行允许：`false`\n"
        "- 自审结论：通过，已准备送审；审计前不得推进下游。\n"
    )
    (output_dir / "最新信号自审说明.md").write_text(summary, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
