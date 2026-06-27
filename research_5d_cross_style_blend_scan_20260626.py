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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_cross_style_blend_scan_20260626"
LABEL = "executable_5d_open_return"

BASE_TABLE = "stock_predict_data_model_agent_5d_v6shrink_fixed4y_rankcorr_gate_refined_20260626_executable_5d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_5d_refined_topnblend_20260626_executable_5d_open_return_research"
SOURCES = {
    "base": BASE_TABLE,
    "daily_gate": "stock_predict_data_model_agent_5d_daily_gate_d3d1_20260623_executable_5d_open_return_research",
    "topgate_v3": "stock_predict_data_model_agent_5d_v3_topgate_3drefined_20260623_executable_5d_open_return_research",
    "daily_gate_v2": "stock_predict_data_model_agent_5d_daily_gate_d3d1_v2_20260623_executable_5d_open_return_research",
    "fixed4y_raw": "stock_predict_data_model_agent_5d_v6shrink_fixed4y_rankcorr_gate_20260626_executable_5d_open_return_research",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        out = None
        for alias, table in SOURCES.items():
            frame = pd.read_sql_query(
                f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
                conn,
            )
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame["stock_code"] = frame["stock_code"].astype(str)
            frame = frame.rename(columns={"pred_prob": f"{alias}_score"})
            out = frame if out is None else out.merge(frame, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    assert out is not None
    for alias in SOURCES:
        out[f"{alias}_rank"] = out.groupby("trade_date")[f"{alias}_score"].rank(method="average", pct=True)
    return out


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


def daily_metrics(frame: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(group[rank_col].corr(label_rank)),
                "top1": top_mean(group, rank_col, 1),
                "top5": top_mean(group, rank_col, 5),
                "top10": top_mean(group, rank_col, 10),
            }
        )
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def summary(daily: pd.DataFrame, window: int | None) -> dict[str, float]:
    sub = daily if window is None else daily.tail(window)
    return {col: float(sub[col].mean()) for col in ["rank_ic", "top1", "top5", "top10"]}


def objective(base_full: dict[str, float], base_63: dict[str, float], base_20: dict[str, float], cur_full: dict[str, float], cur_63: dict[str, float], cur_20: dict[str, float]) -> float:
    return (
        2.4 * (cur_20["top5"] - base_20["top5"])
        + 1.8 * (cur_20["top1"] - base_20["top1"])
        + 1.2 * (cur_63["top5"] - base_63["top5"])
        + 0.8 * (cur_full["rank_ic"] - base_full["rank_ic"])
        + 0.4 * (cur_full["top5"] - base_full["top5"])
    )


def write_candidate(scores: pd.DataFrame, best: pd.Series) -> dict[str, object]:
    alt = str(best["alt_source"])
    w = float(best["weight_alt"])
    out = scores[["trade_date", "stock_code", "base_rank", f"{alt}_rank"]].copy()
    out["pred_prob"] = (1.0 - w) * out["base_rank"] + w * out[f"{alt}_rank"]
    out["score_formula"] = f"{1.0 - w:.2f}*base_rank + {w:.2f}*{alt}_rank"
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[["trade_date", "stock_code", "pred_prob", "base_rank", f"{alt}_rank", "score_formula"]].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_5d_cross_style_blend_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.commit()
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
    return {
        "table": TARGET_TABLE,
        "formula": out["score_formula"].iloc[0],
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")

    base_daily = daily_metrics(eval_frame, "base_rank")
    base_full = summary(base_daily, None)
    base_63 = summary(base_daily, 63)
    base_20 = summary(base_daily, 20)

    rows = []
    for alt in ["daily_gate", "topgate_v3", "daily_gate_v2", "fixed4y_raw"]:
        for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
            rank_col = f"blend_{alt}_{int(round(w*100)):02d}"
            eval_frame[rank_col] = (1.0 - w) * eval_frame["base_rank"] + w * eval_frame[f"{alt}_rank"]
            daily = daily_metrics(eval_frame, rank_col)
            full = summary(daily, None)
            r63 = summary(daily, 63)
            r20 = summary(daily, 20)
            row = {
                "alt_source": alt,
                "weight_alt": w,
                "full_rank_ic_delta": full["rank_ic"] - base_full["rank_ic"],
                "full_top1_delta": full["top1"] - base_full["top1"],
                "full_top5_delta": full["top5"] - base_full["top5"],
                "recent63_rank_ic_delta": r63["rank_ic"] - base_63["rank_ic"],
                "recent63_top1_delta": r63["top1"] - base_63["top1"],
                "recent63_top5_delta": r63["top5"] - base_63["top5"],
                "recent20_rank_ic_delta": r20["rank_ic"] - base_20["rank_ic"],
                "recent20_top1_delta": r20["top1"] - base_20["top1"],
                "recent20_top5_delta": r20["top5"] - base_20["top5"],
            }
            row["pass_basic"] = bool(
                row["recent20_top5_delta"] >= 0.00150
                and row["recent63_top5_delta"] >= 0.00080
                and row["full_rank_ic_delta"] >= -0.00120
                and row["full_top5_delta"] >= -0.00050
            )
            row["objective"] = objective(base_full, base_63, base_20, full, r63, r20)
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["pass_basic", "objective", "recent20_top5_delta", "recent63_top5_delta", "full_rank_ic_delta"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    result.to_csv(REPORT_DIR / "cross_style_blend_results.csv", index=False, encoding="utf-8-sig")

    best = result.iloc[0].to_dict() if not result.empty else None
    asset_stats = None
    if not result.empty and bool(result.iloc[0]["pass_basic"]) and float(result.iloc[0]["objective"]) > 0:
        asset_stats = write_candidate(scores, result.iloc[0])

    summary_obj = {
        "generated_at": now_iso(),
        "scope": "research_only_5d_cross_style_blend_scan",
        "base_table": BASE_TABLE,
        "sources": SOURCES,
        "rows": int(len(result)),
        "pass_basic_count": int(result["pass_basic"].sum()) if not result.empty else 0,
        "best": best,
        "asset_stats": asset_stats,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "research_prediction_table_write_only": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "cross_style_blend_summary.json").write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
