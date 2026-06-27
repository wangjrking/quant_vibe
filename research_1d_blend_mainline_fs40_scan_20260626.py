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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_1d_blend_mainline_fs40_scan_20260626"

LABEL = "executable_1d_open_return"
MAINLINE_TABLE = "stock_predict_data_model_agent_1d_new5d_condition_gate_v6_20260625_executable_1d_open_return_research"
FS40_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_research_1d_fixed4y_fold08_fs40_structure_scan_20260626"
    / "fs40_l16_lr005_n3600"
    / "fold08"
    / "fold_predictions"
    / "fold08.parquet"
)


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_labels() -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def load_mainline(start: str, end: str) -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from '{MAINLINE_TABLE}' where trade_date >= ? and trade_date <= ? order by trade_date, stock_code",
            conn,
            params=(start, end),
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def evaluate_frame(frame: pd.DataFrame, labels: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    merged = frame.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    rows = []
    for trade_date, group in merged.groupby("trade_date", sort=True):
        rank = group["pred_prob"].rank(method="average", pct=True)
        y_rank = group[LABEL].rank(method="average", pct=True)
        row = {
            "trade_date": trade_date,
            "rank_ic": float(rank.corr(y_rank)),
            "pearson_ic": float(group["pred_prob"].corr(group[LABEL])),
        }
        for n in [1, 3, 5, 10, 20, 50]:
            row[f"top{n}"] = float(group.nlargest(min(n, len(group)), "pred_prob")[LABEL].mean())
        row["top_bottom"] = float(
            group.nlargest(min(50, len(group)), "pred_prob")[LABEL].mean()
            - group.nsmallest(min(50, len(group)), "pred_prob")[LABEL].mean()
        )
        rows.append(row)
    daily = pd.DataFrame(rows)
    summary = {
        "trade_days": int(len(daily)),
        "rank_ic": float(daily["rank_ic"].mean()),
        "rank_ic_positive_ratio": float((daily["rank_ic"] > 0).mean()),
        "pearson_ic": float(daily["pearson_ic"].mean()),
        "top_bottom": float(daily["top_bottom"].mean()),
    }
    for n in [1, 3, 5, 10, 20, 50]:
        summary[f"top{n}"] = float(daily[f"top{n}"].mean())
    return summary, daily


def objective(summary: dict) -> float:
    return (
        2.0 * summary["top1"]
        + 1.2 * summary["top3"]
        + 0.9 * summary["top5"]
        + 0.3 * summary["top10"]
        + 0.35 * summary["rank_ic"]
        + 0.15 * summary["top_bottom"]
    )


def blend_scores(mainline: pd.DataFrame, fs40: pd.DataFrame, weight_main: float) -> pd.DataFrame:
    merged = mainline.merge(
        fs40,
        on=["trade_date", "stock_code"],
        how="inner",
        suffixes=("_main", "_fs40"),
        validate="one_to_one",
    )
    parts = []
    for _, group in merged.groupby("trade_date", sort=True):
        main_rank = group["pred_prob_main"].rank(method="average", pct=True)
        fs40_rank = group["pred_prob_fs40"].rank(method="average", pct=True)
        score = weight_main * main_rank + (1.0 - weight_main) * fs40_rank
        out = group[["trade_date", "stock_code"]].copy()
        out["pred_prob"] = score.astype(float)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fs40 = pd.read_parquet(FS40_PATH, columns=["trade_date", "stock_code", "pred_prob"])
    fs40["trade_date"] = fs40["trade_date"].astype(str)
    fs40["stock_code"] = fs40["stock_code"].astype(str)
    start = str(fs40["trade_date"].min())
    end = str(fs40["trade_date"].max())
    mainline = load_mainline(start, end)
    labels = load_labels()

    weights = [round(x / 10.0, 1) for x in range(0, 11)]
    rows = []
    daily_rows = []

    for weight_main in weights:
        weight_fs40 = round(1.0 - weight_main, 1)
        asset = f"blend_main{int(weight_main * 100):02d}_fs40{int(weight_fs40 * 100):02d}"
        if weight_main == 1.0:
            pred = mainline[["trade_date", "stock_code", "pred_prob"]].copy()
        elif weight_main == 0.0:
            pred = fs40[["trade_date", "stock_code", "pred_prob"]].copy()
        else:
            pred = blend_scores(mainline, fs40, weight_main)
        summary, daily = evaluate_frame(pred, labels)
        summary.update(
            {
                "asset": asset,
                "weight_mainline": weight_main,
                "weight_fs40": weight_fs40,
                "date_min": start,
                "date_max": end,
                "objective": objective(summary),
            }
        )
        rows.append(summary)
        daily = daily.copy()
        daily["asset"] = asset
        daily["weight_mainline"] = weight_main
        daily["weight_fs40"] = weight_fs40
        daily_rows.append(daily)

    summary_df = pd.DataFrame(rows).sort_values(["objective", "top1", "rank_ic"], ascending=False).reset_index(drop=True)
    baseline = next(row for row in rows if row["asset"] == "blend_main100_fs4000")
    fs40_base = next(row for row in rows if row["asset"] == "blend_main00_fs40100")
    delta_rows = []
    for row in rows:
        for baseline_name, base in [("mainline", baseline), ("fs40_l16_lr005_n3600", fs40_base)]:
            delta = {"asset": row["asset"], "baseline": baseline_name}
            for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "top_bottom", "objective"]:
                delta[f"delta_{metric}"] = row[metric] - base[metric]
            delta_rows.append(delta)
    delta_df = pd.DataFrame(delta_rows)
    daily_df = pd.concat(daily_rows, ignore_index=True)

    summary_path = REPORT_DIR / "blend_scan_summary.csv"
    delta_path = REPORT_DIR / "blend_scan_deltas.csv"
    daily_path = REPORT_DIR / "blend_scan_daily.csv"
    packet_path = REPORT_DIR / "blend_scan_packet.json"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta_df.to_csv(delta_path, index=False, encoding="utf-8-sig")
    daily_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    packet = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_mainline_fs40_blend_scan",
        "label": LABEL,
        "mainline_table": MAINLINE_TABLE,
        "fs40_prediction_path": str(FS40_PATH),
        "date_min": start,
        "date_max": end,
        "summary_csv": str(summary_path),
        "delta_csv": str(delta_path),
        "daily_csv": str(daily_path),
        "weights": weights,
    }
    packet_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
