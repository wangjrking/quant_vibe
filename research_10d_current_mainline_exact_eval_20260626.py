from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_10d_current_mainline_exact_eval_20260626"
LABEL = "executable_10d_open_return"

TABLES = {
    "base": "stock_predict_data_model_agent_10d_risk_balanced_v5_shrink_20260625_executable_10d_open_return_research",
    "candidate_raw": "stock_predict_data_model_agent_10d_v5shrink_fixed4y_top20zero_gate_20260626_executable_10d_open_return_research",
    "gate": "stock_predict_data_model_agent_10d_regime_gate_best_20260626_executable_10d_open_return_research",
    "veto": "stock_predict_data_model_agent_10d_regime_gate_veto_best_20260626_executable_10d_open_return_research",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_scores(table: str) -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
            conn,
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)].dropna(subset=[LABEL])
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else 0.0


def eval_scores(scores: pd.DataFrame, labels: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    rows = []
    for trade_date, group in eval_frame.groupby("trade_date", sort=True):
        pred_rank = group["pred_prob"].rank(method="average", pct=True)
        label_rank = group[LABEL].rank(method="average", pct=True)
        group = group.assign(_pred_rank=pred_rank)
        rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(pred_rank.corr(label_rank)),
                "top1": top_mean(group, "_pred_rank", 1),
                "top5": top_mean(group, "_pred_rank", 5),
                "top10": top_mean(group, "_pred_rank", 10),
                "top20": top_mean(group, "_pred_rank", 20),
            }
        )
    daily = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)

    def summarize(window: int | None) -> dict[str, float]:
        sub = daily if window is None else daily.tail(window)
        return {col: float(sub[col].mean()) for col in ["rank_ic", "top1", "top5", "top10", "top20"]}

    summary = {
        "days": int(len(daily)),
        "min_eval_date": str(daily["trade_date"].min()),
        "max_eval_date": str(daily["trade_date"].max()),
        "full": summarize(None),
        "recent126": summarize(126),
        "recent63": summarize(63),
        "recent20": summarize(20),
    }
    return summary, daily


def table_quality(table: str) -> dict[str, object]:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(table)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(table)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        latest = conn.execute(
            f"select trade_date, count(*), count(distinct stock_code) from {quote(table)} group by trade_date order by trade_date desc limit 5"
        ).fetchall()
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def delta(left: dict[str, object], right: dict[str, object]) -> dict[str, dict[str, float]]:
    out = {}
    for scope in ["full", "recent126", "recent63", "recent20"]:
        out[scope] = {metric: float(left[scope][metric] - right[scope][metric]) for metric in left[scope]}
    return out


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {name: load_scores(table) for name, table in TABLES.items()}
    min_date = min(frame["trade_date"].min() for frame in loaded.values())
    max_date = max(frame["trade_date"].max() for frame in loaded.values())
    labels = load_labels(str(min_date), str(max_date))

    metrics = {}
    daily_paths = {}
    for name, frame in loaded.items():
        summary, daily = eval_scores(frame, labels)
        metrics[name] = summary
        path = REPORT_DIR / f"{name}_daily_eval.csv"
        daily.to_csv(path, index=False, encoding="utf-8-sig")
        daily_paths[name] = str(path).replace("\\", "/")

    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_10d_current_mainline_exact_eval",
        "label": LABEL,
        "tables": TABLES,
        "quality": {name: table_quality(table) for name, table in TABLES.items()},
        "metrics": metrics,
        "veto_vs_base_delta": delta(metrics["veto"], metrics["base"]),
        "veto_vs_gate_delta": delta(metrics["veto"], metrics["gate"]),
        "daily_eval_paths": daily_paths,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "current_mainline_exact_eval_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
