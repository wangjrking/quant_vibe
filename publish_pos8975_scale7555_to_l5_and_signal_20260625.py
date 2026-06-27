from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
MODEL_DB = DATA / "model_predictions" / "MODEL_PREDICTIONS.db"
MARKET_DB = DATA / "STOCK_DAILY_DATA.db"
MANIFEST_DIR = MAIN / "config" / "prediction_manifests"
PRODUCTION_ROOT = MAIN / "strategy_library" / "production"
EXPLORATION_ROOT = MAIN / "strategy_library" / "exploration"
REGISTRY = MAIN / "strategy_library" / "registry.json"
PRODUCTION_SIGNALS_DIR = DATA / "production_signals"

SOURCE_STRATEGY_ID = "candidate_dynamic_h2m3_sl06dd0712_pos8975_c975_tp070_scale7555_v20260625"
PROD_STRATEGY_ID = "prod_dynamic_h2m3_pos8975_scale7555_v20260625"
SOURCE_DIR = EXPLORATION_ROOT / SOURCE_STRATEGY_ID
PROD_DIR = PRODUCTION_ROOT / PROD_STRATEGY_ID

SIGNAL_DATE = "20260624"
BUY_DATE = "20260625"

MANIFESTS = {
    "3d": MANIFEST_DIR / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MANIFEST_DIR / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MANIFEST_DIR / "executable_10d_open_return_l4_formal_20260617.json",
}

ENTRY_WEIGHTS = {"10d": 0.78, "5d": 0.12, "3d": 0.10}
BASE_FILTER = {"amount_min": 100000.0, "total_mv_min": 200000.0}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def to_gm_symbol(stock_code: str) -> str:
    code = str(stock_code)
    if code.endswith(".SH"):
        return f"SHSE.{code[:6]}"
    if code.endswith(".SZ"):
        return f"SZSE.{code[:6]}"
    if code.startswith(("6", "9")):
        return f"SHSE.{code[:6]}"
    return f"SZSE.{code[:6]}"


def is_bj_code(stock_code: str) -> bool:
    text = str(stock_code or "")
    return text.endswith(".BJ") or text.startswith(("8", "4"))


def is_blocked_name(name: object) -> bool:
    text = str(name or "")
    return text.startswith(("ST", "*ST")) or "退" in text or "退市" in text or "风险" in text


def is_st_like(row: dict | None) -> bool:
    if not row:
        return True
    name = str(row.get("name") or "")
    st_type = row.get("st_type", row.get("ST_TYPE"))
    st_name = str(row.get("st_type_name", row.get("ST_TYPE_name")) or "")
    if is_blocked_name(name):
        return True
    if "风险" in st_name or "退市" in st_name:
        return True
    if st_type in (None, "", "None", "NONE"):
        return False
    return str(st_type).strip().upper() not in {"0", "0.0", "FALSE", "NONE", "NAN"}


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_limit_buy(row: dict | None) -> bool:
    if not row:
        return True
    limit_times = as_float(row.get("limit_times"), 0.0)
    if limit_times and limit_times > 0:
        return True
    pre_close = as_float(row.get("pre_close"))
    open_price = as_float(row.get("open"))
    if pre_close is None or open_price is None or pre_close <= 0 or open_price <= 0:
        return True
    code = str(row.get("stock_code") or "")
    pct = 0.20 if code.startswith(("300", "301", "688")) else 0.10
    return open_price >= pre_close * (1.0 + pct) * 0.995


def quote_table(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_manifest(label: str) -> dict:
    path = MANIFESTS[label]
    payload = read_json(path)
    if payload.get("approval_status") != "approved_for_l5":
        raise RuntimeError(f"{path} is not approved_for_l5")
    db_path = (path.parent / payload["db_path"]).resolve()
    if db_path != MODEL_DB.resolve():
        raise RuntimeError(f"unexpected model db for {label}: {db_path}")
    return payload


def table_summary(table: str) -> dict:
    con = sqlite3.connect(MODEL_DB)
    try:
        row = con.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT trade_date) AS trade_days,
                COUNT(DISTINCT stock_code) AS stock_count,
                MIN(trade_date) AS min_trade_date,
                MAX(trade_date) AS max_trade_date,
                SUM(CASE WHEN pred_prob IS NULL THEN 1 ELSE 0 END) AS null_pred_prob
            FROM {quote_table(table)}
            """
        ).fetchone()
        latest_date = row[4]
        latest_rows = con.execute(
            f"SELECT COUNT(*) FROM {quote_table(table)} WHERE trade_date = ?",
            (latest_date,),
        ).fetchone()[0]
        duplicate_keys = con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT trade_date, stock_code
                FROM {quote_table(table)}
                GROUP BY trade_date, stock_code
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        return {
            "row_count": int(row[0]),
            "trade_days": int(row[1]),
            "stock_count": int(row[2]),
            "min_trade_date": str(row[3]),
            "max_trade_date": str(row[4]),
            "latest_day_rows": int(latest_rows),
            "null_pred_prob": int(row[5] or 0),
            "duplicate_key_groups": int(duplicate_keys),
        }
    finally:
        con.close()


def build_prediction_table_meta() -> dict:
    assets = {}
    for label in MANIFESTS:
        manifest_path = MANIFESTS[label]
        manifest = load_manifest(label)
        assets[label] = {
            "manifest_path": str(manifest_path),
            "approval_status": manifest.get("approval_status"),
            "source_type": manifest.get("source_type"),
            "db_path": str((manifest_path.parent / manifest["db_path"]).resolve()),
            "table": manifest["table"],
            "summary": table_summary(manifest["table"]),
        }
    return {
        "schema_version": 1,
        "strategy_id": PROD_STRATEGY_ID,
        "source_strategy_id": SOURCE_STRATEGY_ID,
        "created_at": "2026-06-25",
        "formal_l4_assets": assets,
        "legacy_odb_allowed": False,
    }


def load_signal_frame() -> pd.DataFrame:
    manifests = {label: load_manifest(label) for label in MANIFESTS}
    tables = {label: manifests[label]["table"] for label in manifests}
    con = sqlite3.connect(MODEL_DB)
    try:
        con.execute("ATTACH DATABASE ? AS market", (str(MARKET_DB),))
        query = f"""
        SELECT
            t10.trade_date,
            t10.stock_code,
            t3.pred_prob AS pred_3d,
            t5.pred_prob AS pred_5d,
            t10.pred_prob AS pred_10d,
            PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t3.pred_prob) AS rank_3d,
            PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t5.pred_prob) AS rank_5d,
            PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t10.pred_prob) AS rank_10d,
            m.name, m.pre_close, m.open, m.close, m.amount, m.turnover_rate,
            m.total_mv, m.atr_qfq, m.limit_times, m.ST_TYPE AS st_type, m.ST_TYPE_name AS st_type_name
        FROM {quote_table(tables['10d'])} t10
        JOIN {quote_table(tables['5d'])} t5
          ON t10.trade_date = t5.trade_date AND t10.stock_code = t5.stock_code
        JOIN {quote_table(tables['3d'])} t3
          ON t10.trade_date = t3.trade_date AND t10.stock_code = t3.stock_code
        LEFT JOIN market.STOCK_DAILY_DATA m
          ON t10.trade_date = m.trade_date AND t10.stock_code = m.stock_code
        WHERE t10.trade_date = ?
        """
        frame = pd.read_sql_query(query, con, params=(SIGNAL_DATE,))
    finally:
        con.close()
    if frame.empty:
        raise RuntimeError(f"no formal L4 rows for signal_date={SIGNAL_DATE}")
    for column in ["rank_3d", "rank_5d", "rank_10d", "amount", "total_mv", "turnover_rate", "atr_qfq"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def market_row(trade_date: str, stock_code: str) -> dict | None:
    con = sqlite3.connect(MARKET_DB)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            """
            SELECT trade_date, stock_code, name, pre_close, open, close,
                   limit_times, ST_TYPE AS st_type, ST_TYPE_name AS st_type_name
            FROM STOCK_DAILY_DATA
            WHERE trade_date = ? AND stock_code = ?
            """,
            (trade_date, stock_code),
        ).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def latest_market_date() -> str:
    con = sqlite3.connect(MARKET_DB)
    try:
        return str(con.execute("SELECT max(trade_date) FROM STOCK_DAILY_DATA").fetchone()[0])
    finally:
        con.close()


def build_signal_rows() -> tuple[list[dict], dict]:
    frame = load_signal_frame()
    frame = frame.loc[~frame["stock_code"].astype(str).map(is_bj_code)].copy()
    frame = frame.loc[frame["amount"] >= BASE_FILTER["amount_min"]].copy()
    frame = frame.loc[frame["total_mv"] >= BASE_FILTER["total_mv_min"]].copy()
    signal_market_missing = int(frame["name"].isna().sum())
    frame = frame.loc[frame["name"].notna()].copy()
    st_mask = frame.apply(lambda row: is_st_like(row.to_dict()), axis=1)
    frame = frame.loc[~st_mask].copy()
    frame = frame.loc[~frame["name"].map(is_blocked_name)].copy()
    for label, weight in ENTRY_WEIGHTS.items():
        frame[f"score_{label}"] = float(weight) * frame[f"rank_{label}"]
    frame["entry_score"] = frame[[f"score_{label}" for label in ENTRY_WEIGHTS]].sum(axis=1)
    frame = frame.sort_values(["entry_score", "stock_code"], ascending=[False, True]).copy()

    chosen = None
    skipped: list[dict] = []
    buy_market_available = latest_market_date() >= BUY_DATE
    for item in frame.to_dict("records"):
        code = str(item["stock_code"])
        buy_row = market_row(BUY_DATE, code) if buy_market_available else None
        if buy_market_available and (is_st_like(buy_row) or is_limit_buy(buy_row)):
            skipped.append({"stock_code": code, "name": item.get("name"), "reason": "buy_day_hard_gate_rejected"})
            continue
        chosen = item
        break
    if chosen is None:
        raise RuntimeError("no eligible signal candidate")

    code = str(chosen["stock_code"])
    buy_row = market_row(BUY_DATE, code) if buy_market_available else None
    status = {
        "signal_date": SIGNAL_DATE,
        "buy_date": BUY_DATE,
        "latest_market_date": latest_market_date(),
        "buy_day_market_available": bool(buy_market_available),
        "buy_day_hard_gate_complete": bool(buy_market_available),
        "signal_market_missing_rows_after_filter_input": signal_market_missing,
        "skipped_candidates": skipped[:20],
    }
    row = {
        "signal_date": SIGNAL_DATE,
        "buy_date": BUY_DATE,
        "symbol": to_gm_symbol(code),
        "stock_code": code,
        "name": chosen.get("name"),
        "rank": 1,
        "pred_prob": chosen.get("entry_score"),
        "entry_score": chosen.get("entry_score"),
        "pred_3d": chosen.get("pred_3d"),
        "pred_5d": chosen.get("pred_5d"),
        "pred_10d": chosen.get("pred_10d"),
        "rank_3d": chosen.get("rank_3d"),
        "rank_5d": chosen.get("rank_5d"),
        "rank_10d": chosen.get("rank_10d"),
        "amount": chosen.get("amount"),
        "turnover_rate": chosen.get("turnover_rate"),
        "total_mv": chosen.get("total_mv"),
        "atr_qfq": chosen.get("atr_qfq"),
        "target_pct": "0.89750",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit_entry_ratio": "0.97000",
        "min_holding_days_before_score_exit": 1,
        "filter_name": "liq_amt10w_mv20w_no_bj_st_delist",
        "entry_weight_name": "w78_5d12_3d10",
        "dynamic_hold_name": "h2_m3_c0975",
        "score_continue_entry_ratio": "0.97500",
        "exit_score_basis": "10d_core_score_table",
        "execution_variant": "force_sell_mkt_intraday_sl_tp_dd",
        "signal_stop_loss_pct": "0.06000",
        "strategy_variant": PROD_STRATEGY_ID,
        "signal_take_profit_pct": "0.07000",
        "buy_day_market_available": bool(buy_market_available),
        "buy_day_hard_gate_complete": bool(buy_market_available),
        "buy_day_st_rejected": bool(buy_market_available and is_st_like(buy_row)),
        "buy_day_open_limit_up_rejected": bool(buy_market_available and is_limit_buy(buy_row)),
    }
    return [row], status


def copytree_fresh(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def publish_strategy(signal_rows: list[dict], signal_status: dict) -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(SOURCE_DIR)
    copytree_fresh(SOURCE_DIR, PROD_DIR)

    signal_file = PROD_DIR / "signals" / f"latest_signal_{SIGNAL_DATE}_for_{BUY_DATE}.csv"
    status_file = PROD_DIR / "signals" / f"latest_signal_status_{SIGNAL_DATE}_for_{BUY_DATE}.json"
    latest_file = PROD_DIR / "signals" / "signals_latest.csv"
    external_latest = PRODUCTION_SIGNALS_DIR / f"{PROD_STRATEGY_ID}_latest.csv"
    write_rows(signal_file, signal_rows)
    write_rows(latest_file, signal_rows)
    write_rows(external_latest, signal_rows)
    write_json(status_file, signal_status)

    manifest = read_json(PROD_DIR / "strategy_manifest.json")
    manifest.update(
        {
            "strategy_id": PROD_STRATEGY_ID,
            "source_strategy_id": SOURCE_STRATEGY_ID,
            "status": "production_user_approved_pending_buy_day_hard_gate",
            "published_at": "2026-06-25",
            "published_by": "strategy-agent",
            "production_registry_updated": True,
            "production_published": True,
            "current_signal": {
                "signal_date": SIGNAL_DATE,
                "buy_date": BUY_DATE,
                "archive_file": str(signal_file),
                "latest_file": str(external_latest),
                "status_file": str(status_file),
                "buy_day_hard_gate_complete": signal_status["buy_day_hard_gate_complete"],
                "buy_day_realtime_checks_required": not signal_status["buy_day_hard_gate_complete"],
            },
        }
    )
    write_json(PROD_DIR / "strategy_manifest.json", manifest)

    rules = read_json(PROD_DIR / "trading_rules.json")
    rules["strategy_id"] = PROD_STRATEGY_ID
    rules["status"] = "production_user_approved_pending_buy_day_hard_gate"
    rules["execution_rule"]["current_signal_status"] = (
        "pending_buy_day_market_audit" if not signal_status["buy_day_hard_gate_complete"] else "hard_gate_complete"
    )
    write_json(PROD_DIR / "trading_rules.json", rules)

    validation = read_json(PROD_DIR / "validation.json")
    validation["status"] = "production_user_approved_pending_buy_day_hard_gate"
    validation["validation_platform"] = "juejin"
    validation["production_publication"] = {
        "published_at": "2026-06-25",
        "user_approved": True,
        "buy_day_hard_gate_complete_for_latest_signal": signal_status["buy_day_hard_gate_complete"],
    }
    write_json(PROD_DIR / "validation.json", validation)
    write_json(PROD_DIR / "prediction_table_meta.json", build_prediction_table_meta())
    write_json(
        PROD_DIR / "reproduction_v20260625.json",
        {
            "schema_version": 1,
            "strategy_id": PROD_STRATEGY_ID,
            "source_strategy_id": SOURCE_STRATEGY_ID,
            "published_at": "2026-06-25",
            "publisher": "strategy-agent",
            "script": str(MAIN / "publish_pos8975_scale7555_to_l5_and_signal_20260625.py"),
            "signal_date": SIGNAL_DATE,
            "buy_date": BUY_DATE,
            "production_archive": str(PROD_DIR),
            "latest_signal": str(PRODUCTION_SIGNALS_DIR / f"{PROD_STRATEGY_ID}_latest.csv"),
            "validation_source": str(SOURCE_DIR),
            "notes": [
                "Only formal L4 approved_for_l5 assets are used.",
                "The latest 20260625 signal is pending buy-day hard-gate completion until 20260625 market data is available locally.",
            ],
        },
    )

    report = (PROD_DIR / "report.md").read_text(encoding="utf-8")
    report = (
        "# L5 生产策略发布说明\n\n"
        f"- 当前策略 ID：`{PROD_STRATEGY_ID}`\n"
        "- 发布状态：用户批准发布到 L5；最新 20260625 信号因买入日行情尚未入库，开盘涨停硬过滤待复核。\n"
        f"- 最新信号文件：`{signal_file}`\n"
        f"- 最新状态文件：`{status_file}`\n\n"
        + report
    )
    (PROD_DIR / "report.md").write_text(report, encoding="utf-8")


def update_registry() -> None:
    registry = read_json(REGISTRY)
    old_current = registry.get("production", {}).get("current")
    old_strategies = registry.get("production", {}).get("strategies", [])
    historical = registry.setdefault("historical_production", {}).setdefault("retained_directories", [])
    for item in old_strategies:
        if (
            old_current
            and old_current != PROD_STRATEGY_ID
            and item.get("strategy_id") == old_current
            and not any(x.get("strategy_id") == old_current for x in historical)
        ):
            historical.insert(
                0,
                {
                    "strategy_id": item.get("strategy_id"),
                    "path": item.get("path"),
                    "status": "historical_archive",
                    "reason": "用户于 2026-06-25 批准发布 pos8975_scale7555 为当前 L5 策略后下架，保留用于审计复现。",
                },
            )
    validation = read_json(PROD_DIR / "validation.json")
    metrics = validation["metrics"]
    registry["updated_at"] = "2026-06-25"
    registry["production"] = {
        "current": PROD_STRATEGY_ID,
        "strategies": [
            {
                "strategy_id": PROD_STRATEGY_ID,
                "name": "dynamic Top1 pos8975 scale7555",
                "path": f"strategy_library/production/{PROD_STRATEGY_ID}",
                "status": "production_user_approved_pending_buy_day_hard_gate",
                "published_at": "2026-06-25",
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "validation_platform": "juejin",
            }
        ],
    }
    write_json(REGISTRY, registry)


def main() -> int:
    signal_rows, signal_status = build_signal_rows()
    publish_strategy(signal_rows, signal_status)
    update_registry()
    print(json.dumps({"strategy_id": PROD_STRATEGY_ID, "signal": signal_rows, "status": signal_status}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
