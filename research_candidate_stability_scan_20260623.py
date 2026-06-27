from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_research_candidate_stability_20260623"


@dataclass(frozen=True)
class Candidate:
    label: str
    candidate_id: str
    candidate_table: str
    label_table: str
    baseline_table: str | None
    score_style: str


CANDIDATES = [
    Candidate(
        label="executable_1d_open_return",
        candidate_id="model_agent_1d_proxy5d_20260623",
        candidate_table="stock_predict_data_model_agent_1d_proxy5d_20260623_executable_1d_open_return_research",
        label_table="stock_predict_data_model_agent_1d_gate092_new5d040_bj0_20260623_executable_1d_open_return_research",
        baseline_table="stock_predict_data_model_agent_1d_gate092_new5d040_bj0_20260623_executable_1d_open_return_research",
        score_style="1d",
    ),
    Candidate(
        label="executable_3d_open_return",
        candidate_id="model_agent_3d_proxy10d5d_blend_20260623",
        candidate_table="stock_predict_data_model_agent_3d_proxy10d5d_blend_20260623_executable_3d_open_return_research",
        label_table="stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal",
        baseline_table="stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal",
        score_style="3d",
    ),
    Candidate(
        label="executable_5d_open_return",
        candidate_id="model_agent_5d_daily_gate_d3d1_v2_20260623",
        candidate_table="stock_predict_data_model_agent_5d_daily_gate_d3d1_v2_20260623_executable_5d_open_return_research",
        label_table="stock_predict_data_model_agent_5d_topgate_3dformal_dategate_refine_20260623_executable_5d_open_return_research",
        baseline_table="stock_predict_data_model_agent_5d_topgate_3dformal_dategate_refine_20260623_executable_5d_open_return_research",
        score_style="5d",
    ),
    Candidate(
        label="executable_10d_open_return",
        candidate_id="model_agent_10d_proxy5d_topzone_blend_20260623",
        candidate_table="stock_predict_data_model_agent_10d_proxy5d_topzone_blend_20260623_executable_10d_open_return_research",
        label_table="stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_executable_10d_open_return_research",
        baseline_table="stock_predict_data_model_agent_10d_no_star_r3gate991_r5gate98_20260623_executable_10d_open_return_research",
        score_style="10d",
    ),
]


def score_style(summary: dict, style: str) -> float:
    if style == "1d":
        return (
            summary["recent63"]["top1"] * 6
            + summary["recent63"]["top3"] * 4
            + summary["recent63"]["top5"] * 5
            + summary["recent126"]["top1"] * 4
            + summary["recent126"]["top3"] * 3
            + summary["recent126"]["top5"] * 3
            + summary["full"]["top1"] * 0.5
            + summary["full"]["top3"] * 0.4
            + summary["full"]["top5"] * 0.3
        )
    if style == "3d":
        return (
            summary["recent63"]["top1"] * 5
            + summary["recent63"]["top3"] * 4
            + summary["recent63"]["top5"] * 5
            + summary["recent63"]["top10"] * 2
            + summary["recent126"]["top1"] * 3
            + summary["recent126"]["top3"] * 2
            + summary["recent126"]["top5"] * 1.5
            + summary["full"]["top1"] * 0.5
            + summary["full"]["top3"] * 0.3
            + summary["full"]["top5"] * 0.2
        )
    if style == "5d":
        return (
            summary["recent80"]["top1"] * 5
            + summary["recent80"]["top3"] * 4
            + summary["recent80"]["top5"] * 5
            + summary["recent80"]["top10"] * 2
            + summary["recent1y"]["top1"] * 3
            + summary["recent1y"]["top3"] * 2
            + summary["recent1y"]["top5"] * 1.5
            + summary["full"]["top1"] * 0.5
            + summary["full"]["top3"] * 0.3
            + summary["full"]["top5"] * 0.2
        )
    if style == "10d":
        return (
            summary["recent63"]["top1"] * 6
            + summary["recent63"]["top3"] * 4
            + summary["recent63"]["top5"] * 5
            + summary["recent126"]["top1"] * 4
            + summary["recent126"]["top3"] * 3
            + summary["recent126"]["top5"] * 3
            + summary["full"]["top1"] * 0.5
            + summary["full"]["top3"] * 0.4
            + summary["full"]["top5"] * 0.3
        )
    raise ValueError(f"unknown style: {style}")


def period_windows(style: str) -> dict[str, tuple[str, str]]:
    if style == "5d":
        return {
            "full": ("20240604", "20260605"),
            "recent1y": ("20250301", "20260605"),
            "recent80": ("20260201", "20260605"),
        }
    if style == "10d":
        return {
            "full": ("20240604", "20260528"),
            "recent126": ("20251118", "20260528"),
            "recent63": ("20260225", "20260528"),
        }
    if style == "3d":
        return {
            "full": ("20240604", "20260609"),
            "recent126": ("20251201", "20260609"),
            "recent63": ("20260304", "20260609"),
        }
    if style == "1d":
        return {
            "full": ("20240604", "20260611"),
            "recent126": ("20251201", "20260611"),
            "recent63": ("20260304", "20260611"),
        }
    raise ValueError(f"unknown style: {style}")


def quarter_key(trade_date: str) -> str:
    year = trade_date[:4]
    month = int(trade_date[4:6])
    q = (month - 1) // 3 + 1
    return f"{year}Q{q}"


def halfyear_key(trade_date: str) -> str:
    year = trade_date[:4]
    month = int(trade_date[4:6])
    half = "H1" if month <= 6 else "H2"
    return f"{year}{half}"


def evaluate_table(conn: sqlite3.Connection, pred_table: str, label_table: str, label_col: str) -> pd.DataFrame:
    labels = pd.read_sql_query(
        f"select trade_date, stock_code, [{label_col}] as y from '{label_table}' where [{label_col}] is not null",
        conn,
    )
    preds = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from '{pred_table}'",
        conn,
    )
    labels["trade_date"] = labels["trade_date"].astype(str)
    preds["trade_date"] = preds["trade_date"].astype(str)
    merged = labels.merge(preds, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    rows: list[dict] = []
    for trade_date, group in merged.groupby("trade_date", sort=True):
        ordered = group.sort_values("pred_prob", ascending=False, kind="mergesort").reset_index(drop=True)
        rows.append(
            {
                "trade_date": trade_date,
                "rows": int(len(group)),
                "top1": float(ordered.head(1)["y"].mean()),
                "top3": float(ordered.head(3)["y"].mean()),
                "top5": float(ordered.head(5)["y"].mean()),
                "top10": float(ordered.head(10)["y"].mean()),
                "top20": float(ordered.head(20)["y"].mean()),
            }
        )
    daily = pd.DataFrame(rows)
    daily["year"] = daily["trade_date"].str[:4]
    daily["halfyear"] = daily["trade_date"].map(halfyear_key)
    daily["quarter"] = daily["trade_date"].map(quarter_key)
    return daily


def summarize_window(daily: pd.DataFrame, start: str, end: str) -> dict:
    win = daily[(daily["trade_date"] >= start) & (daily["trade_date"] <= end)].copy()
    return {
        "date_from": start,
        "date_to": end,
        "trade_days": int(len(win)),
        "top1": float(win["top1"].mean()),
        "top3": float(win["top3"].mean()),
        "top5": float(win["top5"].mean()),
        "top10": float(win["top10"].mean()),
        "top20": float(win["top20"].mean()),
    }


def summarize_group(daily: pd.DataFrame, by_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for key, group in daily.groupby(by_col, sort=True):
        rows.append(
            {
                by_col: key,
                "trade_days": int(len(group)),
                "top1": float(group["top1"].mean()),
                "top3": float(group["top3"].mean()),
                "top5": float(group["top5"].mean()),
                "top10": float(group["top10"].mean()),
                "top20": float(group["top20"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    halfyear_frames: list[pd.DataFrame] = []
    quarter_frames: list[pd.DataFrame] = []

    with sqlite3.connect(MODEL_DB) as conn:
        for cfg in CANDIDATES:
            candidate_daily = evaluate_table(conn, cfg.candidate_table, cfg.label_table, cfg.label)
            baseline_daily = (
                evaluate_table(conn, cfg.baseline_table, cfg.label_table, cfg.label) if cfg.baseline_table else None
            )

            windows = period_windows(cfg.score_style)
            candidate_windows = {name: summarize_window(candidate_daily, start, end) for name, (start, end) in windows.items()}
            candidate_focus = score_style(candidate_windows, cfg.score_style)

            baseline_focus = None
            baseline_windows = None
            if baseline_daily is not None:
                baseline_windows = {
                    name: summarize_window(baseline_daily, start, end) for name, (start, end) in windows.items()
                }
                baseline_focus = score_style(baseline_windows, cfg.score_style)

            yearly = summarize_group(candidate_daily, "year")
            yearly.insert(0, "candidate_id", cfg.candidate_id)
            yearly.insert(1, "label", cfg.label)
            yearly_frames.append(yearly)

            halfyear = summarize_group(candidate_daily, "halfyear")
            halfyear.insert(0, "candidate_id", cfg.candidate_id)
            halfyear.insert(1, "label", cfg.label)
            halfyear_frames.append(halfyear)

            quarter = summarize_group(candidate_daily, "quarter")
            quarter.insert(0, "candidate_id", cfg.candidate_id)
            quarter.insert(1, "label", cfg.label)
            quarter_frames.append(quarter)

            strongest_recent = max(
                candidate_windows.get("recent63", {}).get("top5", float("-inf")),
                candidate_windows.get("recent80", {}).get("top5", float("-inf")),
            )
            weakest_year_top5 = float(yearly["top5"].min()) if not yearly.empty else float("nan")

            summary_rows.append(
                {
                    "label": cfg.label,
                    "candidate_id": cfg.candidate_id,
                    "candidate_table": cfg.candidate_table,
                    "label_table": cfg.label_table,
                    "baseline_table": cfg.baseline_table or "",
                    "score_style": cfg.score_style,
                    "candidate_focus": candidate_focus,
                    "baseline_focus": baseline_focus,
                    "focus_delta_vs_baseline": None if baseline_focus is None else candidate_focus - baseline_focus,
                    "full_top1": candidate_windows["full"]["top1"],
                    "full_top3": candidate_windows["full"]["top3"],
                    "full_top5": candidate_windows["full"]["top5"],
                    "recent_top1": candidate_windows.get("recent63", candidate_windows.get("recent80"))["top1"],
                    "recent_top3": candidate_windows.get("recent63", candidate_windows.get("recent80"))["top3"],
                    "recent_top5": candidate_windows.get("recent63", candidate_windows.get("recent80"))["top5"],
                    "strongest_recent_top5": strongest_recent,
                    "weakest_year_top5": weakest_year_top5,
                    "year_count": int(len(yearly)),
                    "quarter_count": int(len(quarter)),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("candidate_focus", ascending=False)
    yearly_df = pd.concat(yearly_frames, ignore_index=True)
    halfyear_df = pd.concat(halfyear_frames, ignore_index=True)
    quarter_df = pd.concat(quarter_frames, ignore_index=True)

    summary_csv = OUT_DIR / "candidate_stability_summary.csv"
    yearly_csv = OUT_DIR / "candidate_yearly_stability.csv"
    halfyear_csv = OUT_DIR / "candidate_halfyear_stability.csv"
    quarter_csv = OUT_DIR / "candidate_quarter_stability.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    yearly_df.to_csv(yearly_csv, index=False, encoding="utf-8-sig")
    halfyear_df.to_csv(halfyear_csv, index=False, encoding="utf-8-sig")
    quarter_df.to_csv(quarter_csv, index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "db_path": str(MODEL_DB),
        "candidate_count": int(len(summary_df)),
        "summary_csv": str(summary_csv),
        "yearly_csv": str(yearly_csv),
        "halfyear_csv": str(halfyear_csv),
        "quarter_csv": str(quarter_csv),
        "top_candidates_by_focus": summary_df[["label", "candidate_id", "candidate_focus", "focus_delta_vs_baseline"]]
        .to_dict(orient="records"),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (OUT_DIR / "stability_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 当前最强 research 资产稳健性筛查",
        "",
        "## 当前结论",
        "",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- `{row.label}` / `{row.candidate_id}`: focus=`{row.candidate_focus:.6f}`，"
            f"相对基线增量=`{'' if pd.isna(row.focus_delta_vs_baseline) else f'{row.focus_delta_vs_baseline:.6f}'}`，"
            f"最近窗口 top5=`{row.recent_top5:.6f}`，年度最弱 top5=`{row.weakest_year_top5:.6f}`"
        )
    lines += [
        "",
        "## 输出文件",
        "",
        f"- `{summary_csv.as_posix()}`",
        f"- `{yearly_csv.as_posix()}`",
        f"- `{halfyear_csv.as_posix()}`",
        f"- `{quarter_csv.as_posix()}`",
        "",
        "## 边界说明",
        "",
        "- 本次仅做研究资产稳健性统计。",
        "- 未训练模型。",
        "- 未修改 formal / production。",
        "- 未生成交易信号、未跑回测。",
    ]
    (OUT_DIR / "stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
