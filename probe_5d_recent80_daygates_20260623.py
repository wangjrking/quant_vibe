from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_5d_recent80_daygates_20260623"

BASE_TABLE = "stock_predict_data_model_agent_5d_gate6_balanced_bonus_20260622_executable_5d_open_return_research"
CAND_TABLE = "stock_predict_data_model_agent_5d_topgate_3dformal_dategate_refine_20260623_executable_5d_open_return_research"
AUX3_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
AUX10_TABLE = "stock_predict_data_model_agent_10d_topgate_fusion_20260622_executable_10d_open_return_research"
LABEL_COL = "executable_5d_open_return"

WINDOWS = {
    "full": ("20240604", "20260605"),
    "recent1y": ("20250301", "20260605"),
    "recent80": ("20260201", "20260605"),
}


@dataclass(frozen=True)
class ProbeResult:
    condition: str
    days: int
    focus_score: float
    full_top1: float
    full_top3: float
    full_top5: float
    recent1y_top1: float
    recent1y_top3: float
    recent1y_top5: float
    recent80_top1: float
    recent80_top3: float
    recent80_top5: float


def evaluate(frame: pd.DataFrame, score_col: str) -> dict:
    data = frame[["trade_date", "stock_code", LABEL_COL, score_col]].dropna().copy()
    daily_rows: list[dict] = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "top1": float(ordered.head(1)[LABEL_COL].mean()),
                "top3": float(ordered.head(3)[LABEL_COL].mean()),
                "top5": float(ordered.head(5)[LABEL_COL].mean()),
                "top10": float(ordered.head(10)[LABEL_COL].mean()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    out = {}
    for name, (date_from, date_to) in WINDOWS.items():
        win = daily[(daily["trade_date"] >= date_from) & (daily["trade_date"] <= date_to)].copy()
        out[name] = {
            "top1": float(win["top1"].mean()),
            "top3": float(win["top3"].mean()),
            "top5": float(win["top5"].mean()),
            "top10": float(win["top10"].mean()),
        }
    return out


def focus_score(summary: dict) -> float:
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


def flatten(condition: str, days: int, summary: dict) -> ProbeResult:
    return ProbeResult(
        condition=condition,
        days=days,
        focus_score=focus_score(summary),
        full_top1=summary["full"]["top1"],
        full_top3=summary["full"]["top3"],
        full_top5=summary["full"]["top5"],
        recent1y_top1=summary["recent1y"]["top1"],
        recent1y_top3=summary["recent1y"]["top3"],
        recent1y_top5=summary["recent1y"]["top5"],
        recent80_top1=summary["recent80"]["top1"],
        recent80_top3=summary["recent80"]["top3"],
        recent80_top5=summary["recent80"]["top5"],
    )


def build_report_lines(summary_payload: dict, result_df: pd.DataFrame) -> list[str]:
    lines = [
        "# 5D recent80 日级门控探针",
        "",
        "## 当前结论",
        "",
        f"- 当前 5D 研究候选 `topgate_3dformal_dategate_refine` 的 focus 分数为 `{summary_payload['candidate_focus_score']:.12f}`。",
        f"- 本轮所有已测日级门控中，最佳条件是 `{summary_payload['best_probe']}`，其 focus 分数为 `{summary_payload['best_probe_focus_score']:.12f}`。",
        "- 结果显示，当前 5D 研究候选已经明显优于 formal base，但简单的日级门控条件没有继续把它往上推。",
        "",
        "## 排序前十",
        "",
    ]
    for _, row in result_df.head(10).iterrows():
        lines.extend(
            [
                f"### {row['condition']}",
                f"- focus_score：`{row['focus_score']:.12f}`",
                f"- days：`{int(row['days'])}`",
                f"- recent80：`Top1={row['recent80_top1']:.8f}` `Top3={row['recent80_top3']:.8f}` `Top5={row['recent80_top5']:.8f}`",
                f"- recent1y：`Top1={row['recent1y_top1']:.8f}` `Top3={row['recent1y_top3']:.8f}` `Top5={row['recent1y_top5']:.8f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 研究判断",
            "",
            "- `5D` 当前更像是局部最优附近，继续叠加简单 day-gate 的边际收益有限。",
            "- 后续如果继续优化 5D，更可能有效的方向是重新设计局部重排公式，而不是再叠加轻量日级条件。",
            "",
            "## 治理说明",
            "",
            "- 本轮未训练模型。",
            "- 本轮未发布 formal 资产。",
            "- 本轮未生成交易信号、未制定交易规则、未跑回测。",
            "",
            "## 证据路径",
            "",
            f"- `{(OUT_DIR / 'daygate_probe_summary.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'daygate_probe_summary.json').as_posix()}`",
        ]
    )
    return lines


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MODEL_DB) as conn:
        base = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as base_pred_prob, [{LABEL_COL}] as {LABEL_COL} from '{BASE_TABLE}' where trade_date <= '20260605'",
            conn,
        )
        cand = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as cand_pred_prob from '{CAND_TABLE}' where trade_date <= '20260605'",
            conn,
        )
        aux3 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred3d from '{AUX3_TABLE}' where trade_date <= '20260605'",
            conn,
        )
        aux10 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as pred10d from '{AUX10_TABLE}' where trade_date <= '20260605'",
            conn,
        )

    for df in [base, cand, aux3, aux10]:
        df["trade_date"] = df["trade_date"].astype(str)

    frame = base.merge(cand, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame = frame.merge(aux3, on=["trade_date", "stock_code"], how="left")
    frame = frame.merge(aux10, on=["trade_date", "stock_code"], how="left")
    frame["rank3d_pct"] = frame.groupby("trade_date")["pred3d"].rank(method="average", pct=True)
    frame["rank10d_pct"] = frame.groupby("trade_date")["pred10d"].rank(method="average", pct=True)
    frame["board"] = frame["stock_code"].map(
        lambda s: "BJ" if s.startswith(("8", "9")) else ("STAR" if s.startswith("688") else ("GEM" if s.startswith("3") else "MAIN"))
    )

    daily_rows: list[dict] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        base_top10 = group.sort_values("base_pred_prob", ascending=False).head(10)
        cand_top10 = group.sort_values("cand_pred_prob", ascending=False).head(10)
        base_top5 = group.sort_values("base_pred_prob", ascending=False).head(5)
        cand_top5 = group.sort_values("cand_pred_prob", ascending=False).head(5)
        base_codes = set(base_top10["stock_code"])
        newcomers = cand_top10[~cand_top10["stock_code"].isin(base_codes)].copy()
        daily_rows.append(
            {
                "trade_date": trade_date,
                "newcomer_count": int(len(newcomers)),
                "newcomer_mean_3d_pct": float(newcomers["rank3d_pct"].mean()) if len(newcomers) else 1.0,
                "newcomer_mean_10d_pct": float(newcomers["rank10d_pct"].mean()) if len(newcomers) else 1.0,
                "cand_top10_mean_3d_pct": float(cand_top10["rank3d_pct"].mean()),
                "cand_top10_mean_10d_pct": float(cand_top10["rank10d_pct"].mean()),
                "cand_top10_star_count": int((cand_top10["board"] == "STAR").sum()),
                "newcomer_star_count": int((newcomers["board"] == "STAR").sum()),
                "top1_changed": int(base_top10.iloc[0]["stock_code"] != cand_top10.iloc[0]["stock_code"]),
                "top3_overlap": len(set(base_top5.head(3)["stock_code"]) & set(cand_top5.head(3)["stock_code"])),
                "top5_overlap": len(set(base_top5["stock_code"]) & set(cand_top5["stock_code"])),
                "cand_gap_1_2": float(cand_top10.iloc[0]["cand_pred_prob"] - cand_top10.iloc[1]["cand_pred_prob"]),
                "cand_gap_1_5": float(cand_top10.iloc[0]["cand_pred_prob"] - cand_top5.iloc[4]["cand_pred_prob"]),
                "cand_std_top10": float(cand_top10["cand_pred_prob"].std()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    frame = frame.merge(daily, on="trade_date", how="left")

    conditions: list[tuple[str, pd.Series]] = []
    for thr in [0.97, 0.975, 0.98, 0.985, 0.99]:
        conditions.append((f"cand3d>={thr}", daily["cand_top10_mean_3d_pct"] >= thr))
    for thr in [0.995, 0.996, 0.997, 0.998]:
        conditions.append((f"cand10d>={thr}", daily["cand_top10_mean_10d_pct"] >= thr))
    for thr in [0.975, 0.98, 0.985, 0.99]:
        conditions.append((f"new3d>={thr}", daily["newcomer_mean_3d_pct"] >= thr))
    for thr in [0.993, 0.994, 0.995, 0.996]:
        conditions.append((f"new10d>={thr}", daily["newcomer_mean_10d_pct"] >= thr))
    conditions.extend(
        [
            ("newcomer_count==0", daily["newcomer_count"] == 0),
            ("newcomer_count>=3", daily["newcomer_count"] >= 3),
            ("newcomer_count==5", daily["newcomer_count"] == 5),
            ("top1_same", daily["top1_changed"] == 0),
            ("top1_changed", daily["top1_changed"] == 1),
            ("top3_overlap>=2", daily["top3_overlap"] >= 2),
            ("top5_overlap>=3", daily["top5_overlap"] >= 3),
            ("star_count<=1", daily["cand_top10_star_count"] <= 1),
            ("star_new0", daily["newcomer_star_count"] == 0),
            ("std10<=0.0010", daily["cand_std_top10"] <= 0.0010),
            ("std10<=0.0008", daily["cand_std_top10"] <= 0.0008),
            ("gap12>=0.00005", daily["cand_gap_1_2"] >= 0.00005),
            ("gap12>=0.00010", daily["cand_gap_1_2"] >= 0.00010),
        ]
    )
    for thr3 in [0.975, 0.98, 0.985]:
        for thr10 in [0.995, 0.996, 0.997]:
            conditions.append(
                (
                    f"cand3d>={thr3}&cand10d>={thr10}",
                    (daily["cand_top10_mean_3d_pct"] >= thr3) & (daily["cand_top10_mean_10d_pct"] >= thr10),
                )
            )

    probe_results: list[ProbeResult] = []
    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_tables": {
            "base_table": BASE_TABLE,
            "candidate_table": CAND_TABLE,
            "aux3_table": AUX3_TABLE,
            "aux10_table": AUX10_TABLE,
        },
        "windows": WINDOWS,
        "probes": {},
    }

    for name, condition in conditions:
        good_days = set(daily.loc[condition, "trade_date"])
        variant = frame.copy()
        variant["hybrid_score"] = variant["base_pred_prob"]
        variant.loc[variant["trade_date"].isin(good_days), "hybrid_score"] = variant.loc[
            variant["trade_date"].isin(good_days), "cand_pred_prob"
        ]
        summary = evaluate(variant, "hybrid_score")
        probe_results.append(flatten(name, len(good_days), summary))
        summary_payload["probes"][name] = {
            "days": len(good_days),
            "summary": summary,
            "focus_score": focus_score(summary),
        }

    base_summary = evaluate(frame, "base_pred_prob")
    cand_summary = evaluate(frame, "cand_pred_prob")
    summary_payload["baseline"] = base_summary
    summary_payload["candidate"] = cand_summary
    summary_payload["baseline_focus_score"] = focus_score(base_summary)
    summary_payload["candidate_focus_score"] = focus_score(cand_summary)

    result_df = pd.DataFrame([row.__dict__ for row in probe_results]).sort_values("focus_score", ascending=False)
    result_df.to_csv(OUT_DIR / "daygate_probe_summary.csv", index=False, encoding="utf-8-sig")
    summary_payload["ranking"] = result_df.to_dict(orient="records")
    summary_payload["best_probe"] = result_df.iloc[0]["condition"]
    summary_payload["best_probe_focus_score"] = float(result_df.iloc[0]["focus_score"])
    (OUT_DIR / "daygate_probe_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = build_report_lines(summary_payload, result_df)
    (OUT_DIR / "daygate_probe_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
