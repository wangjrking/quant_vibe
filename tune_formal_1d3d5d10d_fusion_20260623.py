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


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_ROOT = DATA_DIR / "reports" / "strategy_agent_goal_high_annual_20260623"
OUT_DIR = REPORT_ROOT / "current_formal_1d3d5d10d_fusion"
MANIFEST_DIR = MAIN / "config" / "prediction_manifests"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
BACKTEST_END = "2026-06-23 15:30:00"

MANIFESTS = {
    "1d": MANIFEST_DIR / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MANIFEST_DIR / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MANIFEST_DIR / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MANIFEST_DIR / "executable_10d_open_return_l4_formal_20260617.json",
}

STARTS = [
    ("full_20240605", "2024-06-05 09:00:00", False),
    ("late_20250701", "2025-07-01 09:00:00", True),
    ("late_20251009", "2025-10-09 09:00:00", True),
    ("late_20260105", "2026-01-05 09:00:00", True),
]

CANDIDATES = [
    {
        "name": "f_10d80_5d20_orig",
        "weights": {"10d": 0.80, "5d": 0.20},
        "top_k": 4,
        "target_each": 0.3305,
        "holding_days": 5,
        "top1_holding_days": 7,
        "top1_score_exit": 0.91,
        "score_exit": 0.942,
        "min_score_hold": 2,
        "l5_scope": "formal_l5",
    },
    {
        "name": "f_10d70_5d20_3d10",
        "weights": {"10d": 0.70, "5d": 0.20, "3d": 0.10},
        "top_k": 4,
        "target_each": 0.3305,
        "holding_days": 5,
        "top1_holding_days": 7,
        "top1_score_exit": 0.91,
        "score_exit": 0.942,
        "min_score_hold": 2,
        "l5_scope": "formal_l5",
    },
    {
        "name": "f_10d60_5d20_3d20",
        "weights": {"10d": 0.60, "5d": 0.20, "3d": 0.20},
        "top_k": 4,
        "target_each": 0.3305,
        "holding_days": 5,
        "top1_holding_days": 7,
        "top1_score_exit": 0.91,
        "score_exit": 0.942,
        "min_score_hold": 2,
        "l5_scope": "formal_l5",
    },
    {
        "name": "f_10d70_5d10_3d20",
        "weights": {"10d": 0.70, "5d": 0.10, "3d": 0.20},
        "top_k": 4,
        "target_each": 0.3305,
        "holding_days": 5,
        "top1_holding_days": 7,
        "top1_score_exit": 0.91,
        "score_exit": 0.942,
        "min_score_hold": 2,
        "l5_scope": "formal_l5",
    },
    {
        "name": "r_10d65_5d15_3d10_1d10",
        "weights": {"10d": 0.65, "5d": 0.15, "3d": 0.10, "1d": 0.10},
        "top_k": 4,
        "target_each": 0.3305,
        "holding_days": 5,
        "top1_holding_days": 7,
        "top1_score_exit": 0.91,
        "score_exit": 0.942,
        "min_score_hold": 2,
        "l5_scope": "research_includes_1d_l4_only",
    },
    {
        "name": "r_10d55_5d15_3d15_1d15",
        "weights": {"10d": 0.55, "5d": 0.15, "3d": 0.15, "1d": 0.15},
        "top_k": 4,
        "target_each": 0.3305,
        "holding_days": 5,
        "top1_holding_days": 7,
        "top1_score_exit": 0.91,
        "score_exit": 0.942,
        "min_score_hold": 2,
        "l5_scope": "research_includes_1d_l4_only",
    },
    {
        "name": "r_10d50_5d10_3d20_1d20",
        "weights": {"10d": 0.50, "5d": 0.10, "3d": 0.20, "1d": 0.20},
        "top_k": 4,
        "target_each": 0.3305,
        "holding_days": 5,
        "top1_holding_days": 7,
        "top1_score_exit": 0.91,
        "score_exit": 0.942,
        "min_score_hold": 2,
        "l5_scope": "research_includes_1d_l4_only",
    },
    {
        "name": "f_10d70_5d20_3d10_h4",
        "weights": {"10d": 0.70, "5d": 0.20, "3d": 0.10},
        "top_k": 4,
        "target_each": 0.3305,
        "holding_days": 4,
        "top1_holding_days": 4,
        "top1_score_exit": 0.942,
        "score_exit": 0.942,
        "min_score_hold": 2,
        "l5_scope": "formal_l5",
    },
]

BASE_ENV = {
    "GM_MAX_DAILY_SELLS": "0",
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
    return json.loads(MANIFESTS[label].read_text(encoding="utf-8"))


def _symbol(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".", 1)
    return f"{'SHSE' if suffix == 'SH' else 'SZSE'}.{code}"


def _is_missing(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NAN", "NULL"}:
        return True
    try:
        return math.isnan(float(text))
    except Exception:
        return False


def _is_st(row: pd.Series) -> bool:
    name = str(row.get("name") or "").strip().upper()
    if name.startswith("ST") or name.startswith("*ST"):
        return True
    st_type_name = str(row.get("ST_TYPE_name") or row.get("st_type_name") or "")
    if "风险" in st_type_name or "椋庨櫓" in st_type_name:
        return True
    st_type = row.get("ST_TYPE")
    if _is_missing(st_type):
        return False
    text = str(st_type).strip().upper()
    if text in {"0", "0.0", "FALSE", "NONE", "NAN"}:
        return False
    try:
        return float(text) != 0.0
    except Exception:
        return True


def _is_current_limit(row: pd.Series) -> bool:
    value = row.get("limit_times")
    if _is_missing(value):
        return False
    try:
        return float(value) > 0
    except Exception:
        return True


def _load_model_frame(label: str) -> pd.DataFrame:
    manifest = _load_manifest(label)
    table = manifest["table"]
    conn = sqlite3.connect(MODEL_DB)
    try:
        frame = pd.read_sql_query(
            f'SELECT trade_date, stock_code, pred_prob AS pred_{label} FROM "{table}"',
            conn,
        )
    finally:
        conn.close()
    frame[f"rank_{label}"] = frame.groupby("trade_date")[f"pred_{label}"].rank(pct=True, method="average")
    return frame


def _build_base_frame() -> pd.DataFrame:
    cache = OUT_DIR / "fusion_base.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    labels = ["10d", "5d", "3d", "1d"]
    frame = _load_model_frame(labels[0])
    for label in labels[1:]:
        frame = frame.merge(_load_model_frame(label), on=["trade_date", "stock_code"], how="inner")
    conn = sqlite3.connect(MARKET_DB)
    try:
        market = pd.read_sql_query(
            """
            SELECT trade_date, stock_code, name, open, close, pre_close, amount,
                   turnover_rate, total_mv, limit_times, ST_TYPE, ST_TYPE_name
            FROM STOCK_DAILY_DATA
            WHERE trade_date >= '20240604' AND trade_date <= '20260622'
            """,
            conn,
        )
    finally:
        conn.close()
    frame = frame.merge(market, on=["trade_date", "stock_code"], how="left")
    frame = frame[~frame["stock_code"].str.endswith(".BJ", na=False)].copy()
    frame = frame[~frame.apply(_is_st, axis=1)].copy()
    frame = frame[~frame["name"].fillna("").str.contains("退市", regex=False)].copy()
    frame = frame[~frame.apply(_is_current_limit, axis=1)].copy()
    frame = frame[frame["close"].notna()].copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def _write_signal_and_score(base: pd.DataFrame, candidate: dict) -> tuple[Path, Path, str]:
    name = candidate["name"]
    frame = base.copy()
    frame["entry_score"] = 0.0
    for label, weight in candidate["weights"].items():
        frame["entry_score"] += float(weight) * frame[f"rank_{label}"]
    frame["day_rank"] = frame.groupby("trade_date")["entry_score"].rank(method="first", ascending=False)
    selected = frame[frame["day_rank"] <= int(candidate["top_k"])].copy()
    selected.sort_values(["trade_date", "day_rank"], inplace=True)
    selected["signal_date"] = selected["trade_date"]
    all_dates = sorted(frame["trade_date"].unique().tolist())
    next_date = {date: all_dates[index + 1] for index, date in enumerate(all_dates[:-1])}
    selected["buy_date"] = selected["trade_date"].map(next_date)
    selected = selected[selected["buy_date"].notna()].copy()
    selected["symbol"] = selected["stock_code"].map(_symbol)
    selected["rank"] = selected["day_rank"].astype(int)
    selected["pred_prob"] = selected["entry_score"]
    selected["target_pct"] = f"{float(candidate['target_each']):.5f}"
    selected["holding_days"] = int(candidate["holding_days"])
    selected["max_holding_days"] = int(candidate["holding_days"])
    selected["score_exit_entry_ratio"] = float(candidate["score_exit"])
    selected["min_holding_days_before_score_exit"] = int(candidate["min_score_hold"])
    top1 = selected["rank"] == 1
    selected.loc[top1, "holding_days"] = int(candidate["top1_holding_days"])
    selected.loc[top1, "max_holding_days"] = int(candidate["top1_holding_days"])
    selected.loc[top1, "score_exit_entry_ratio"] = float(candidate["top1_score_exit"])

    signal_cols = [
        "signal_date",
        "buy_date",
        "symbol",
        "stock_code",
        "name",
        "rank",
        "pred_prob",
        "entry_score",
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "rank_1d",
        "rank_3d",
        "rank_5d",
        "rank_10d",
        "target_pct",
        "holding_days",
        "max_holding_days",
        "score_exit_entry_ratio",
        "min_holding_days_before_score_exit",
    ]
    signal_file = OUT_DIR / "signals" / f"{name}.csv"
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    selected[signal_cols].to_csv(signal_file, index=False, encoding="utf-8-sig")

    score_db = OUT_DIR / "scores_1d3d5d10d_fusion.db"
    score_table = f"score_{name}"
    score_frame = frame[["trade_date", "stock_code", "entry_score"]].rename(columns={"entry_score": "pred_prob"})
    conn = sqlite3.connect(score_db)
    try:
        score_frame.to_sql(score_table, conn, if_exists="replace", index=False)
        conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{score_table}_date_code" ON "{score_table}" (trade_date, stock_code)')
        conn.commit()
    finally:
        conn.close()
    return signal_file, score_db, score_table


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


def _run_case(candidate: dict, signal_file: Path, score_db: Path, score_table: str, start_name: str, start: str) -> dict:
    log_file = OUT_DIR / "logs" / f"{candidate['name']}_{start_name}.log"
    indicator = _extract_indicator(log_file)
    if indicator is None:
        env = os.environ.copy()
        env.update(BASE_ENV)
        env["GM_OPEN_DAILY_SCORE_EXIT"] = "1"
        env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(float(candidate["score_exit"]))
        env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = str(int(candidate["min_score_hold"]))
        command = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            str(int(candidate["top_k"])),
            "--holding-days",
            str(int(candidate["holding_days"])),
            "--max-holding-days",
            str(int(candidate["holding_days"])),
            "--target-position-pct",
            "0.5",
            "--score-db",
            str(score_db),
            "--score-table",
            score_table,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            start,
            "--backtest-end",
            BACKTEST_END,
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        indicator = _extract_indicator(log_file)
        returncode = proc.returncode
    else:
        returncode = 0
    return {
        "candidate": candidate["name"],
        "l5_scope": candidate["l5_scope"],
        "weights_json": json.dumps(candidate["weights"], ensure_ascii=False, sort_keys=True),
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


def _summarize(rows: list[dict]) -> list[dict]:
    out = []
    for candidate in sorted({row["candidate"] for row in rows}):
        items = [row for row in rows if row["candidate"] == candidate]
        late = [float(row["annual"]) for row in items if row["start_name"].startswith("late_")]
        full = next(row for row in items if row["start_name"] == "full_20240605")
        out.append(
            {
                "candidate": candidate,
                "l5_scope": full["l5_scope"],
                "weights_json": full["weights_json"],
                "full_annual": full["annual"],
                "full_sharpe": full["sharpe"],
                "full_max_drawdown": full["max_drawdown"],
                "late_min_annual": min(late),
                "late_median_annual": float(pd.Series(late).median()),
                "late_mean_annual": float(pd.Series(late).mean()),
                "late_max_annual": max(late),
                "score": min(late) * 0.35 + float(pd.Series(late).median()) * 0.25 + float(full["annual"]) * 0.40,
            }
        )
    out.sort(key=lambda row: (row["score"], row["full_annual"]), reverse=True)
    return out


def main() -> int:
    manifest_status = {label: _load_manifest(label)["approval_status"] for label in MANIFESTS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest_status.json").write_text(json.dumps(manifest_status, ensure_ascii=False, indent=2), encoding="utf-8")
    base = _build_base_frame()
    rows = []
    for candidate in CANDIDATES:
        signal_file, score_db, score_table = _write_signal_and_score(base, candidate)
        for start_name, start, _is_late in STARTS:
            row = _run_case(candidate, signal_file, score_db, score_table, start_name, start)
            rows.append(row)
            print(
                f"{candidate['name']} {start_name} annual={row['annual']} "
                f"sharpe={row['sharpe']} maxdd={row['max_drawdown']}",
                flush=True,
            )
    with (OUT_DIR / "fusion_cases.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows)
    with (OUT_DIR / "fusion_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (OUT_DIR / "fusion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
