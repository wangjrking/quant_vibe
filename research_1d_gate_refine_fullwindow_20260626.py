from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_1d_gate_refine_fullwindow_20260626"

LABEL = "executable_1d_open_return"
BASE_TABLE = "stock_predict_data_model_agent_1d_new5d_condition_gate_v6_20260625_executable_1d_open_return_research"
ALT_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_research_1d_fixed4y_fold08_fs40_structure_scan_20260626"
    / "fs40_l16_lr005_n3600"
    / "fold08"
    / "fold_predictions"
    / "fold08.parquet"
)
FEATURES = ["base_std", "top20_overlap", "top10_overlap", "rank_corr", "mean_abs_rank_gap", "alt_std"]
METRICS = ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "top_bottom"]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_base_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from {quote(BASE_TABLE)} order by trade_date, stock_code",
            conn,
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": "base_score"})


def read_alt_scores() -> pd.DataFrame:
    frame = pd.read_parquet(ALT_PATH, columns=["trade_date", "stock_code", "pred_prob"])
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": "alt_score"})


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nsmallest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def build_daily_tables(base: pd.DataFrame, alt: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = base.merge(alt, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    scores = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    scores["base_rank"] = scores.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    alt_mask = scores["alt_score"].notna()
    scores.loc[alt_mask, "alt_rank"] = scores.loc[alt_mask].groupby("trade_date")["alt_score"].rank(method="average", pct=True)

    daily_base = []
    daily_alt = []
    feature_rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        for name, col in [("base", "base_rank"), ("alt", "alt_rank")]:
            if name == "alt" and group["alt_rank"].isna().all():
                row = {m: np.nan for m in METRICS}
            else:
                row = {
                    "rank_ic": float(group[col].corr(label_rank)),
                    "top1": top_mean(group, col, 1),
                    "top3": top_mean(group, col, 3),
                    "top5": top_mean(group, col, 5),
                    "top10": top_mean(group, col, 10),
                    "top20": top_mean(group, col, 20),
                    "top50": top_mean(group, col, 50),
                    "top_bottom": top_mean(group, col, 50) - bottom_mean(group, col, 50),
                }
            if name == "base":
                daily_base.append({"trade_date": trade_date, **row})
            else:
                daily_alt.append({"trade_date": trade_date, **row})
        if not group["alt_rank"].isna().all():
            top10_base = set(group.nlargest(min(10, len(group)), "base_rank")["stock_code"])
            top10_alt = set(group.nlargest(min(10, len(group)), "alt_rank")["stock_code"])
            top20_base = set(group.nlargest(min(20, len(group)), "base_rank")["stock_code"])
            top20_alt = set(group.nlargest(min(20, len(group)), "alt_rank")["stock_code"])
            feature_rows.append(
                {
                    "trade_date": trade_date,
                    "base_std": float(group["base_score"].std()),
                    "top10_overlap": float(len(top10_base & top10_alt) / max(1, len(top10_base | top10_alt))),
                    "top20_overlap": float(len(top20_base & top20_alt) / max(1, len(top20_base | top20_alt))),
                    "rank_corr": float(group["base_rank"].corr(group["alt_rank"])),
                    "mean_abs_rank_gap": float((group["base_rank"] - group["alt_rank"]).abs().mean()),
                    "alt_std": float(group["alt_score"].std()),
                }
            )
    return pd.DataFrame(daily_base), pd.DataFrame(daily_alt), pd.DataFrame(feature_rows)


def objective_from_summary(summary: dict[str, float]) -> float:
    return (
        1.8 * summary["top1"]
        + 1.2 * summary["top3"]
        + 1.0 * summary["top5"]
        + 0.4 * summary["top10"]
        + 0.45 * summary["rank_ic"]
        + 0.2 * summary["top_bottom"]
    )


def summarize_daily(daily: pd.DataFrame) -> dict[str, float]:
    return {m: float(daily[m].mean()) for m in METRICS}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = read_base_scores()
    alt = read_alt_scores()
    labels = load_labels(str(base["trade_date"].min()), str(base["trade_date"].max()))
    daily_base, daily_alt, feature_daily = build_daily_tables(base, alt, labels)

    base_summary = summarize_daily(daily_base)
    base_objective = objective_from_summary(base_summary)
    overlap_days = set(feature_daily["trade_date"])
    alt_map = daily_alt.set_index("trade_date")
    base_map = daily_base.set_index("trade_date")

    rows = []
    for feature in FEATURES:
        values = sorted(feature_daily[feature].dropna().unique().tolist())
        for threshold in values:
            for direction in ["le", "ge"]:
                if direction == "le":
                    active_days = set(feature_daily.loc[feature_daily[feature] <= threshold, "trade_date"])
                else:
                    active_days = set(feature_daily.loc[feature_daily[feature] >= threshold, "trade_date"])
                mixed_rows = []
                for trade_date, row in base_map.iterrows():
                    source_row = alt_map.loc[trade_date] if trade_date in active_days else row
                    mixed_rows.append({"trade_date": trade_date, **{m: float(source_row[m]) for m in METRICS}})
                mixed = pd.DataFrame(mixed_rows)
                summary = summarize_daily(mixed)
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": float(threshold),
                        "active_days": len(active_days),
                        "overlap_days": len(overlap_days),
                        "objective": objective_from_summary(summary),
                        **summary,
                    }
                )

    result = pd.DataFrame(rows).sort_values(["objective", "top1", "top5"], ascending=False).reset_index(drop=True)
    best = result.iloc[0].to_dict()
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_1d_fullwindow_gate_refine",
        "base_table": BASE_TABLE,
        "alt_path": str(ALT_PATH),
        "base_objective": base_objective,
        "best_candidate": best,
        "improves_over_base": bool(best["objective"] > base_objective),
    }
    result.to_csv(REPORT_DIR / "fullwindow_gate_refine_results.csv", index=False)
    feature_daily.to_csv(REPORT_DIR / "fullwindow_gate_refine_features.csv", index=False)
    (REPORT_DIR / "fullwindow_gate_refine_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 1D 全窗口门控细化扫描报告（20260626）",
        "",
        f"- 基线 objective：`{base_objective:.12f}`",
        f"- 最优候选 objective：`{best['objective']:.12f}`",
        f"- 最优规则：`if {best['feature']} {best['direction']} {best['threshold']:.12f}: alt_rank else base_rank`",
        f"- 触发交易日数：`{int(best['active_days'])}` / overlap `{int(best['overlap_days'])}`",
        f"- 是否优于当前主线：`{'是' if best['objective'] > base_objective else '否'}`",
    ]
    (REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
