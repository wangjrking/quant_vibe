from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
OUT_DIR = DATA_DIR / "reports" / "model_agent_current_best_research_snapshot_20260624_v8"

CHOICES = {
    "1d": {
        "label": "executable_1d_open_return",
        "name": "1D narrow balanced gate body formal anchor",
        "table": "stock_predict_data_model_agent_1d_narrow_balanced_grid_20260624_executable_1d_open_return_research",
        "tier": "model_side_audit_candidate",
        "risk_note": "Full 和近期 TopN 同向改善，适合作为模型侧审计候选。",
    },
    "3d": {
        "label": "executable_3d_open_return",
        "name": "3D narrow balanced max guarded 5d balanced 1d gate",
        "table": "stock_predict_data_model_agent_3d_narrow_balanced_grid_20260624_executable_3d_open_return_research",
        "tier": "research_candidate_with_risk",
        "risk_note": "近期 TopN 改善明显，但 Full RankIC 下降，不适合直接替代生产。",
    },
    "5d": {
        "label": "executable_5d_open_return",
        "name": "5D narrow balanced mean 5dmax 10drepair 3dguard",
        "table": "stock_predict_data_model_agent_5d_narrow_balanced_grid_20260624_executable_5d_open_return_research",
        "tier": "research_candidate_with_risk",
        "risk_note": "Full Top1 和近期 TopN 改善，但 Full Top5 略弱于生产 baseline。",
    },
    "10d": {
        "label": "executable_10d_open_return",
        "name": "10D no-star fullcoverage gridbest",
        "table": "stock_predict_data_model_agent_10d_no_star_fullcoverage_gridbest_20260624_executable_10d_open_return_research",
        "tier": "research_candidate_cleaner_chain",
        "risk_note": "相对 V7 指标略低，但全覆盖门控链条更干净，不再依赖 20260623 formal fallback。",
    },
}

FORMAL = {
    "1d": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
    "3d": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
    "5d": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
    "10d": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
}

WINDOWS = {
    "full": None,
    "recent252": 252,
    "recent126": 126,
    "recent63": 63,
    "recent20": 20,
}


def load_labels() -> pd.DataFrame:
    columns = ["trade_date", "stock_code"] + sorted({item["label"] for item in CHOICES.values()})
    table = ds.dataset(str(LABEL_DIR), format="parquet").to_table(
        columns=columns,
        filter=ds.field("trade_date") >= "20240604",
    )
    frame = table.to_pandas()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def read_prediction(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    frame = pd.read_sql_query(f"select trade_date, stock_code, pred_prob from '{table}' where trade_date >= '20240604'", conn)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def table_summary(conn: sqlite3.Connection, table: str) -> dict:
    row = conn.execute(
        f"""
        select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
               count(distinct stock_code), sum(case when trade_date = '20260623' then 1 else 0 end),
               sum(case when pred_prob is null then 1 else 0 end)
        from '{table}'
        """
    ).fetchone()
    dup = conn.execute(
        f"""
        select count(*) from (
          select trade_date, stock_code, count(*) c
          from '{table}'
          group by trade_date, stock_code
          having c > 1
        )
        """
    ).fetchone()[0]
    return {
        "table": table,
        "rows": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "rows_20260623": int(row[5] or 0),
        "null_pred_prob": int(row[6] or 0),
        "duplicate_key_groups": int(dup),
    }


def daily_eval(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    data = frame[["trade_date", "stock_code", "pred_prob", label]].dropna().copy()
    rows = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        if len(group) < 100:
            continue
        ordered = group.sort_values(["pred_prob", "stock_code"], ascending=[False, True], kind="mergesort")
        rows.append(
            {
                "trade_date": trade_date,
                "rows": int(len(group)),
                "rank_ic": float(group["pred_prob"].corr(group[label], method="spearman")),
                "top1": float(ordered.head(1)[label].mean()),
                "top3": float(ordered.head(3)[label].mean()),
                "top5": float(ordered.head(5)[label].mean()),
                "top10": float(ordered.head(10)[label].mean()),
                "top20": float(ordered.head(20)[label].mean()),
                "top50": float(ordered.head(50)[label].mean()),
                "bottom50": float(ordered.tail(50)[label].mean()),
            }
        )
    daily = pd.DataFrame(rows)
    if not daily.empty:
        daily["top_bottom50"] = daily["top50"] - daily["bottom50"]
    return daily


def summarize(daily: pd.DataFrame) -> list[dict]:
    rows = []
    for window, count in WINDOWS.items():
        win = daily.tail(count) if count else daily
        if win.empty:
            continue
        row = {
            "window": window,
            "eval_date_min": str(win["trade_date"].min()),
            "eval_date_max": str(win["trade_date"].max()),
            "eval_trade_days": int(win["trade_date"].nunique()),
            "avg_rows": float(win["rows"].mean()),
        }
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "top_bottom50"]:
            row[metric] = float(win[metric].mean())
        rows.append(row)
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    labels = load_labels()

    metrics_rows = []
    daily_parts = []
    coverage_rows = []
    readiness_rows = []

    with sqlite3.connect(MODEL_DB) as conn:
        for horizon, cfg in CHOICES.items():
            label = cfg["label"]
            pred = read_prediction(conn, cfg["table"])
            formal = read_prediction(conn, FORMAL[horizon])
            coverage = table_summary(conn, cfg["table"])
            coverage.update({"horizon": horizon, "label": label, "name": cfg["name"]})
            coverage_rows.append(coverage)

            model_eval = daily_eval(pred.merge(labels[["trade_date", "stock_code", label]], on=["trade_date", "stock_code"], how="inner"), label)
            formal_eval = daily_eval(formal.merge(labels[["trade_date", "stock_code", label]], on=["trade_date", "stock_code"], how="inner"), label)

            model_eval.insert(0, "asset", "research")
            model_eval.insert(0, "horizon", horizon)
            daily_parts.append(model_eval)

            model_summary = pd.DataFrame(summarize(model_eval))
            formal_summary = pd.DataFrame(summarize(formal_eval))
            merged_summary = model_summary.merge(
                formal_summary[["window", "rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]],
                on="window",
                how="left",
                suffixes=("", "_formal"),
            )
            for _, row in merged_summary.iterrows():
                item = {
                    "horizon": horizon,
                    "label": label,
                    "name": cfg["name"],
                    "table": cfg["table"],
                    "window": row["window"],
                    "eval_date_min": row["eval_date_min"],
                    "eval_date_max": row["eval_date_max"],
                    "eval_trade_days": int(row["eval_trade_days"]),
                    "rank_ic": float(row["rank_ic"]),
                    "top1": float(row["top1"]),
                    "top3": float(row["top3"]),
                    "top5": float(row["top5"]),
                    "top10": float(row["top10"]),
                    "top20": float(row["top20"]),
                    "top50": float(row["top50"]),
                }
                for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
                    item[f"delta_vs_formal_{metric}"] = float(row[metric] - row[f"{metric}_formal"])
                metrics_rows.append(item)

            full = next(item for item in metrics_rows if item["horizon"] == horizon and item["window"] == "full")
            recent63 = next(item for item in metrics_rows if item["horizon"] == horizon and item["window"] == "recent63")
            readiness_rows.append(
                {
                    "horizon": horizon,
                    "tier": cfg["tier"],
                    "full_delta_rank_ic": full["delta_vs_formal_rank_ic"],
                    "full_delta_top1": full["delta_vs_formal_top1"],
                    "full_delta_top5": full["delta_vs_formal_top5"],
                    "recent63_delta_rank_ic": recent63["delta_vs_formal_rank_ic"],
                    "recent63_delta_top1": recent63["delta_vs_formal_top1"],
                    "recent63_delta_top5": recent63["delta_vs_formal_top5"],
                    "risk_note": cfg["risk_note"],
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    readiness = pd.DataFrame(readiness_rows)
    daily = pd.concat(daily_parts, ignore_index=True)

    metrics_path = OUT_DIR / "current_best_research_metrics_20260624_v8.csv"
    coverage_path = OUT_DIR / "current_best_research_coverage_20260624_v8.csv"
    readiness_path = OUT_DIR / "current_best_research_readiness_20260624_v8.csv"
    daily_path = OUT_DIR / "current_best_research_daily_eval_20260624_v8.csv"
    json_path = OUT_DIR / "current_best_research_snapshot_20260624_v8.json"
    md_path = OUT_DIR / "current_best_research_snapshot_20260624_v8.md"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    coverage_df.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    readiness.to_csv(readiness_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    snapshot = {
        "generated_at": generated_at,
        "actor": "model-agent",
        "snapshot_id": "model_agent_current_best_research_snapshot_20260624_v8",
        "asset_role": "research_snapshot_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "label_source": str(LABEL_DIR),
        "db_path": str(MODEL_DB),
        "formal_baseline": FORMAL,
        "choices": CHOICES,
        "coverage": coverage_rows,
        "readiness": readiness_rows,
        "changes_vs_v7": [
            "10D 从 V7 no-star fallback 扩展候选切换为 no-star fullcoverage gridbest 候选。",
            "V8 的 10D 指标略低于 V7，但全链条覆盖到 20260623，不再依赖 latest-day formal fallback。",
            "1D/3D/5D 保持 V7 研究候选不变。",
        ],
        "outputs": {
            "metrics_csv": str(metrics_path),
            "coverage_csv": str(coverage_path),
            "readiness_csv": str(readiness_path),
            "daily_csv": str(daily_path),
            "snapshot_json": str(json_path),
            "snapshot_md": str(md_path),
        },
        "governance": {
            "no_training": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
            "promotion_requires_user_and_audit_confirmation": True,
        },
    }
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 当前最佳研究模型快照 20260624 V8",
        "",
        f"- 生成时间：{generated_at}",
        "- 资产身份：research snapshot，不是 formal L4/L5 生产资产。",
        "- 本次未训练模型、未修改生产 manifest、未生成交易信号、未跑回测。",
        "",
        "## V8 变化",
        "",
    ]
    lines.extend(f"- {item}" for item in snapshot["changes_vs_v7"])
    lines.extend(["", "## 当前选择", ""])
    for horizon in ["1d", "3d", "5d", "10d"]:
        cfg = CHOICES[horizon]
        cov = coverage_df[coverage_df["horizon"] == horizon].iloc[0]
        full = metrics[(metrics["horizon"] == horizon) & (metrics["window"] == "full")].iloc[0]
        recent63 = metrics[(metrics["horizon"] == horizon) & (metrics["window"] == "recent63")].iloc[0]
        lines.extend(
            [
                f"### {horizon.upper()}",
                "",
                f"- 表：`{cfg['table']}`",
                f"- 覆盖：`{cov['min_trade_date']}` 到 `{cov['max_trade_date']}`，最新日行数 `{int(cov['rows_20260623'])}`。",
                f"- Full：ΔRankIC=`{full['delta_vs_formal_rank_ic']:.6f}`，ΔTop1=`{full['delta_vs_formal_top1']:.6f}`，ΔTop5=`{full['delta_vs_formal_top5']:.6f}`。",
                f"- Recent63：ΔRankIC=`{recent63['delta_vs_formal_rank_ic']:.6f}`，ΔTop1=`{recent63['delta_vs_formal_top1']:.6f}`，ΔTop5=`{recent63['delta_vs_formal_top5']:.6f}`。",
                f"- 风险：{cfg['risk_note']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 证据路径",
            "",
            f"- 指标：`{metrics_path}`",
            f"- 覆盖：`{coverage_path}`",
            f"- readiness：`{readiness_path}`",
            f"- 日频评价：`{daily_path}`",
            f"- JSON：`{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT_DIR), "readiness": readiness_rows, "coverage": coverage_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
