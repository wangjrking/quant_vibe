from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd

from gm_signal_module import to_gm_symbol


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_ROOT = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623"
OUT_DIR = REPORT_ROOT / "top1_soft_rank_latest_formal_20260623"
MANIFEST_DIR = MAIN / "config" / "prediction_manifests"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_END = "2026-06-23 15:30:00"
FILTER_VERSION = "no_delist_v2"

MANIFESTS = {
    "1d": MANIFEST_DIR / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MANIFEST_DIR / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MANIFEST_DIR / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MANIFEST_DIR / "executable_10d_open_return_l4_formal_20260617.json",
}

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00"),
    ("late_20250701", "2025-07-01 09:00:00"),
    ("late_20251009", "2025-10-09 09:00:00"),
    ("late_20260105", "2026-01-05 09:00:00"),
]

CANDIDATES = [
    {"name": "latest_w80_5d20_h5_g200_e098", "weights": {"10d": 0.80, "5d": 0.20}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "latest_w80_5d20_h10_g138_e094", "weights": {"10d": 0.80, "5d": 0.20}, "holding_days": 10, "gamma": 1.38, "exit_ratio": 0.94, "min_hold": 1},
    {"name": "latest_w70_5d30_h5_g200_e098", "weights": {"10d": 0.70, "5d": 0.30}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "latest_w75_5d25_h5_g200_e098", "weights": {"10d": 0.75, "5d": 0.25}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "latest_w85_5d15_h5_g200_e098", "weights": {"10d": 0.85, "5d": 0.15}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "latest_w90_5d10_h5_g200_e098", "weights": {"10d": 0.90, "5d": 0.10}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "latest_w60_5d20_1d20_h5_g200_e098", "weights": {"10d": 0.60, "5d": 0.20, "1d": 0.20}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "latest_w60_5d20_3d20_h5_g200_e098", "weights": {"10d": 0.60, "5d": 0.20, "3d": 0.20}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "latest_w50_5d20_1d15_3d15_h5_g200_e098", "weights": {"10d": 0.50, "5d": 0.20, "1d": 0.15, "3d": 0.15}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "latest_1d60_10d20_5d20_h5_g200_e098", "weights": {"1d": 0.60, "10d": 0.20, "5d": 0.20}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
    {"name": "latest_3d60_10d20_5d20_h5_g200_e098", "weights": {"3d": 0.60, "10d": 0.20, "5d": 0.20}, "holding_days": 5, "gamma": 2.0, "exit_ratio": 0.98, "min_hold": 1},
]

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_STOP_LOSS_PCT": "0.08",
    "GM_TAKE_PROFIT_PCT": "none",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "1.02",
    "GM_EQUITY_DD_RISK_MODE": "0",
}


def _load_manifest(label: str) -> dict:
    manifest = json.loads(MANIFESTS[label].read_text(encoding="utf-8"))
    if manifest.get("approval_status") != "approved_for_l5":
        raise RuntimeError(f"{label} manifest not approved_for_l5: {manifest.get('approval_status')}")
    return manifest


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _to_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _is_missing(value) -> bool:
    if value in (None, "", "None", "NONE", "nan", "NaN"):
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _has_blocked_name(name: object) -> bool:
    text = str(name or "")
    return (
        text.startswith(("ST", "*ST"))
        or "退" in text
        or "闁偓" in text
        or "閫€" in text
    )


def _is_st_like(row: dict | None) -> bool:
    if not row:
        return True
    name = str(row.get("name") or "")
    st_type = row.get("st_type", row.get("ST_TYPE"))
    st_name = str(row.get("st_type_name", row.get("ST_TYPE_name")) or "")
    if _has_blocked_name(name):
        return True
    if name.startswith(("ST", "*ST")) or "閫€" in name or "退" in name:
        return True
    if "椋庨櫓" in st_name or "风险" in st_name:
        return True
    if _is_missing(st_type):
        return False
    return str(st_type).strip().upper() not in {"0", "0.0", "FALSE", "NONE", "NAN"}


def _is_limit_buy(row: dict | None) -> bool:
    if not row:
        return True
    limit_times = row.get("limit_times")
    if not _is_missing(limit_times) and (_to_float(limit_times, 0.0) or 0.0) > 0.0:
        return True
    pre_close = _to_float(row.get("pre_close"))
    open_price = _to_float(row.get("open"))
    if pre_close is None or open_price is None or pre_close <= 0 or open_price <= 0:
        return True
    code = str(row.get("stock_code") or "")
    pct = 0.20 if code.startswith(("300", "301", "688")) else 0.10
    return open_price >= pre_close * (1.0 + pct) * 0.995


def _tailpow(value: float, gamma: float, upper: float) -> float:
    x = max(0.0, min(1.0, float(value) / upper))
    return upper * (1.0 - ((1.0 - x) ** float(gamma)))


def _build_base() -> pd.DataFrame:
    cache = OUT_DIR / f"latest_formal_base_{FILTER_VERSION}.parquet"
    manifest_status = {label: _load_manifest(label) for label in MANIFESTS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest_status.json").write_text(json.dumps(manifest_status, ensure_ascii=False, indent=2), encoding="utf-8")
    if cache.exists():
        return pd.read_parquet(cache)
    tables = {label: manifest_status[label]["table"] for label in MANIFESTS}
    conn = sqlite3.connect(MODEL_DB)
    try:
        conn.execute("ATTACH DATABASE ? AS market", (str(MARKET_DB),))
        query = f"""
        SELECT
            t10.trade_date,
            t10.stock_code,
            t1.pred_prob AS pred_1d,
            t3.pred_prob AS pred_3d,
            t5.pred_prob AS pred_5d,
            t10.pred_prob AS pred_10d,
            PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t1.pred_prob) AS rank_1d,
            PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t3.pred_prob) AS rank_3d,
            PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t5.pred_prob) AS rank_5d,
            PERCENT_RANK() OVER (PARTITION BY t10.trade_date ORDER BY t10.pred_prob) AS rank_10d,
            m.name, m.pre_close, m.open, m.close, m.amount, m.turnover_rate,
            m.total_mv, m.atr_qfq, m.limit_times, m.ST_TYPE AS st_type, m.ST_TYPE_name AS st_type_name
        FROM {_quote(tables['10d'])} t10
        JOIN {_quote(tables['5d'])} t5 ON t10.trade_date = t5.trade_date AND t10.stock_code = t5.stock_code
        JOIN {_quote(tables['3d'])} t3 ON t10.trade_date = t3.trade_date AND t10.stock_code = t3.stock_code
        JOIN {_quote(tables['1d'])} t1 ON t10.trade_date = t1.trade_date AND t10.stock_code = t1.stock_code
        LEFT JOIN market.STOCK_DAILY_DATA m ON t10.trade_date = m.trade_date AND t10.stock_code = m.stock_code
        """
        frame = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    name = frame["name"].fillna("").astype(str)
    st_text = frame["st_type"].fillna("").astype(str).str.strip().str.upper()
    limit_num = pd.to_numeric(frame["limit_times"], errors="coerce").fillna(0.0)
    blocked_name = name.map(_has_blocked_name)
    frame = frame[
        ~frame["stock_code"].astype(str).str.endswith(".BJ")
        & ~blocked_name
        & ~name.str.startswith("ST")
        & ~name.str.startswith("*ST")
        & ~name.str.contains("閫€", regex=False)
        & (st_text.isin(["", "0", "0.0", "FALSE", "NONE", "NAN"]))
        & (limit_num == 0.0)
        & frame["rank_1d"].notna()
        & frame["rank_3d"].notna()
        & frame["rank_5d"].notna()
        & frame["rank_10d"].notna()
        & frame["close"].notna()
    ].copy()
    frame.to_parquet(cache, index=False)
    return frame


def _market_rows() -> dict[str, dict[str, dict]]:
    conn = sqlite3.connect(MARKET_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT trade_date, stock_code, name, pre_close, open, close,
                   limit_times, ST_TYPE AS st_type, ST_TYPE_name AS st_type_name
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= '20240604'
            """
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        out.setdefault(str(row["trade_date"]), {})[str(row["stock_code"])] = dict(row)
    return out


def _date_map(frame: pd.DataFrame) -> dict[str, str]:
    dates = sorted(str(value) for value in frame["trade_date"].dropna().unique())
    return {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}


def _entry_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for label, weight in weights.items():
        score += float(weight) * frame[f"rank_{label}"].astype(float)
    return score


def _write_signal(base: pd.DataFrame, market: dict[str, dict[str, dict]], next_date: dict[str, str], cfg: dict) -> Path:
    frame = base.copy()
    frame["entry_score"] = _entry_score(frame, cfg["weights"])
    frame = frame.sort_values(["trade_date", "entry_score", "stock_code"], ascending=[True, False, True]).copy()
    upper = max(float(frame["rank_10d"].max()), 1.0)
    rows = []
    for signal_date, day in frame.groupby("trade_date", sort=True):
        buy_date = next_date.get(str(signal_date))
        if not buy_date:
            continue
        chosen = None
        for row in day.to_dict("records"):
            stock_code = str(row["stock_code"])
            buy_market = market.get(buy_date, {}).get(stock_code)
            if _is_st_like(buy_market) or _is_limit_buy(buy_market):
                continue
            chosen = row
            break
        if chosen is None:
            continue
        stock_code = str(chosen["stock_code"])
        rows.append(
            {
                "signal_date": str(signal_date),
                "buy_date": buy_date,
                "symbol": to_gm_symbol(stock_code),
                "stock_code": stock_code,
                "name": chosen.get("name"),
                "rank": 1,
                "pred_prob": _tailpow(float(chosen["rank_10d"]), float(cfg["gamma"]), upper),
                "entry_score": chosen["entry_score"],
                "pred_1d": chosen.get("pred_1d"),
                "pred_3d": chosen.get("pred_3d"),
                "pred_5d": chosen.get("pred_5d"),
                "pred_10d": chosen.get("pred_10d"),
                "rank_1d": chosen.get("rank_1d"),
                "rank_3d": chosen.get("rank_3d"),
                "rank_5d": chosen.get("rank_5d"),
                "rank_10d": chosen.get("rank_10d"),
                "amount": chosen.get("amount"),
                "turnover_rate": chosen.get("turnover_rate"),
                "total_mv": chosen.get("total_mv"),
                "atr_qfq": chosen.get("atr_qfq"),
                "target_pct": "0.99000",
                "holding_days": int(cfg["holding_days"]),
                "max_holding_days": int(cfg["holding_days"]),
                "score_exit_entry_ratio": f"{float(cfg['exit_ratio']):.5f}",
                "min_holding_days_before_score_exit": int(cfg["min_hold"]),
            }
        )
    fields = [
        "signal_date", "buy_date", "symbol", "stock_code", "name", "rank", "pred_prob", "entry_score",
        "pred_1d", "pred_3d", "pred_5d", "pred_10d", "rank_1d", "rank_3d", "rank_5d", "rank_10d",
        "amount", "turnover_rate", "total_mv", "atr_qfq", "target_pct", "holding_days", "max_holding_days",
        "score_exit_entry_ratio", "min_holding_days_before_score_exit",
    ]
    path = OUT_DIR / "signals" / f"{cfg['name']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])
    return path


def _write_score_table(base: pd.DataFrame, cfg: dict) -> tuple[Path, str]:
    score_db = OUT_DIR / "scores_latest_formal_top1.db"
    table = f"score_{cfg['name']}"
    upper = max(float(base["rank_10d"].max()), 1.0)
    payload = base[["trade_date", "stock_code", "rank_10d"]].copy()
    payload["pred_prob"] = payload["rank_10d"].map(lambda value: _tailpow(float(value), float(cfg["gamma"]), upper))
    payload = payload[["trade_date", "stock_code", "pred_prob"]]
    score_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(score_db)
    try:
        payload.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_date_code" ON "{table}" (trade_date, stock_code)')
        conn.commit()
    finally:
        conn.close()
    return score_db, table


def _extract_indicator(log_file: Path) -> dict | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _run_case(cfg: dict, signal_file: Path, score_db: Path, score_table: str, start_name: str, start: str) -> dict:
    log_file = OUT_DIR / "logs" / f"{cfg['name']}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(float(cfg["exit_ratio"]))
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(int(cfg["min_hold"]))
        command = [
            str(JUEJIN_PYTHON), str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir", str(STRATEGY_DIR),
            "--signal-file", str(signal_file),
            "--log-file", str(log_file),
            "--max-positions", "1",
            "--holding-days", str(int(cfg["holding_days"])),
            "--max-holding-days", str(int(cfg["holding_days"])),
            "--target-position-pct", "1.0",
            "--score-db", str(score_db),
            "--score-table", score_table,
            "--market-db", str(MARKET_DB),
            "--backtest-start", start,
            "--backtest-end", BACKTEST_END,
            "--backtest-adjust", "none",
            "--backtest-initial-cash", "600000",
            "--backtest-slippage-ratio", "0.0015",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        indicator = _extract_indicator(log_file)
        returncode = proc.returncode
    else:
        returncode = 0
    return {
        "candidate": cfg["name"],
        "weights_json": json.dumps(cfg["weights"], ensure_ascii=False, sort_keys=True),
        "holding_days": cfg["holding_days"],
        "gamma": cfg["gamma"],
        "exit_ratio": cfg["exit_ratio"],
        "min_hold": cfg["min_hold"],
        "start_name": start_name,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "signal_file": str(signal_file),
        "score_db": str(score_db),
        "score_table": score_table,
        "log_file": str(log_file),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for candidate in sorted({row["candidate"] for row in rows}):
        items = [row for row in rows if row["candidate"] == candidate]
        full = next(row for row in items if row["start_name"] == "full_20240605")
        if full["annual"] is None:
            continue
        late = [float(row["annual"]) for row in items if row["start_name"].startswith("late_") and row["annual"] is not None]
        if not late:
            continue
        summary.append({
            "candidate": candidate,
            "weights_json": full["weights_json"],
            "holding_days": full["holding_days"],
            "gamma": full["gamma"],
            "exit_ratio": full["exit_ratio"],
            "min_hold": full["min_hold"],
            "full_annual": full["annual"],
            "full_sharpe": full["sharpe"],
            "full_max_drawdown": full["max_drawdown"],
            "late_min_annual": min(late),
            "late_median_annual": float(pd.Series(late).median()),
            "late_mean_annual": float(pd.Series(late).mean()),
            "late_max_annual": max(late),
            "score": min(late) * 0.35 + float(pd.Series(late).median()) * 0.25 + float(full["annual"]) * 0.40,
        })
    summary.sort(key=lambda row: (row["score"], row["full_annual"]), reverse=True)
    return summary


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "signal_count": len(rows),
        "signal_days": len({row["signal_date"] for row in rows}),
        "buy_days": len({row["buy_date"] for row in rows}),
        "min_signal_date": min((row["signal_date"] for row in rows), default=""),
        "max_signal_date": max((row["signal_date"] for row in rows), default=""),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _build_base()
    market = _market_rows()
    next_date = _date_map(base)
    rows = []
    stats = []
    total = len(CANDIDATES) * len(STARTS)
    done = 0
    for cfg in CANDIDATES:
        signal_file = _write_signal(base, market, next_date, cfg)
        stats.append({"candidate": cfg["name"], **_signal_stats(signal_file)})
        score_db, score_table = _write_score_table(base, cfg)
        for start_name, start in STARTS:
            done += 1
            row = _run_case(cfg, signal_file, score_db, score_table, start_name, start)
            rows.append(row)
            print(
                f"[{done}/{total}] {cfg['name']} {start_name} annual={row['annual']} "
                f"sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )
            _write_rows(OUT_DIR / "latest_formal_cases.csv", rows)
    _write_rows(OUT_DIR / "latest_formal_signal_stats.csv", stats)
    summary = _summarize(rows)
    _write_rows(OUT_DIR / "latest_formal_summary.csv", summary)
    (OUT_DIR / "latest_formal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
