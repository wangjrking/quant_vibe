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
OUT_DIR = DATA_DIR / "reports" / "model_agent_3d5d_stability_anchor_20260624"

FORMAL = {
    "3d": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
    "5d": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
}
RESEARCH = {
    "3d": "stock_predict_data_model_agent_3d_narrow_balanced_grid_20260624_executable_3d_open_return_research",
    "5d": "stock_predict_data_model_agent_5d_narrow_balanced_grid_20260624_executable_5d_open_return_research",
}
TARGETS = {
    "3d": "stock_predict_data_model_agent_3d_stability_anchor_gridbest_20260624_executable_3d_open_return_research",
    "5d": "stock_predict_data_model_agent_5d_stability_anchor_gridbest_20260624_executable_5d_open_return_research",
}
LABELS = {
    "3d": "executable_3d_open_return",
    "5d": "executable_5d_open_return",
}

START_DATES = ["20240604", "20250102", "20251001", "20260102", "20260302"]
ALPHAS = [0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]
TOP_ZONES = [None, 0.98, 0.99, 0.995]
WINDOWS = {
    "full": None,
    "recent252": 252,
    "recent126": 126,
    "recent63": 63,
    "recent20": 20,
}


def load_labels() -> pd.DataFrame:
    cols = ["trade_date", "stock_code"] + list(LABELS.values())
    table = ds.dataset(str(LABEL_DIR), format="parquet").to_table(
        columns=cols,
        filter=ds.field("trade_date") >= "20240604",
    )
    frame = table.to_pandas()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def read_pred(conn: sqlite3.Connection, table: str, alias: str) -> pd.DataFrame:
    frame = pd.read_sql_query(f"select trade_date, stock_code, pred_prob as {alias} from '{table}' where trade_date >= '20240604'", conn)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def evaluate(frame: pd.DataFrame, label: str, score_col: str) -> tuple[dict, pd.DataFrame]:
    data = frame[["trade_date", "stock_code", label, score_col]].dropna().copy()
    daily_rows = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        if len(group) < 100:
            continue
        ordered = group.sort_values([score_col, "stock_code"], ascending=[False, True], kind="mergesort")
        daily_rows.append(
            {
                "trade_date": trade_date,
                "rows": int(len(group)),
                "rank_ic": float(group[score_col].corr(group[label], method="spearman")),
                "top1": float(ordered.head(1)[label].mean()),
                "top3": float(ordered.head(3)[label].mean()),
                "top5": float(ordered.head(5)[label].mean()),
                "top10": float(ordered.head(10)[label].mean()),
                "top20": float(ordered.head(20)[label].mean()),
                "top50": float(ordered.head(50)[label].mean()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    summary = {}
    for window, count in WINDOWS.items():
        win = daily.tail(count) if count else daily
        summary[window] = {
            "date_from": str(win["trade_date"].min()),
            "date_to": str(win["trade_date"].max()),
            "trade_days": int(win["trade_date"].nunique()),
        }
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
            summary[window][metric] = float(win[metric].mean())
    return summary, daily


def objective(summary: dict, formal: dict, horizon: str) -> float:
    if horizon == "3d":
        return (
            0.8 * (summary["full"]["rank_ic"] - formal["full"]["rank_ic"])
            + 1.0 * (summary["full"]["top5"] - formal["full"]["top5"])
            + 1.5 * (summary["recent126"]["top5"] - formal["recent126"]["top5"])
            + 2.5 * (summary["recent63"]["top1"] - formal["recent63"]["top1"])
            + 2.0 * (summary["recent63"]["top5"] - formal["recent63"]["top5"])
        )
    return (
        0.8 * (summary["full"]["rank_ic"] - formal["full"]["rank_ic"])
        + 2.0 * (summary["full"]["top1"] - formal["full"]["top1"])
        + 2.0 * (summary["full"]["top5"] - formal["full"]["top5"])
        + 1.5 * (summary["recent126"]["top5"] - formal["recent126"]["top5"])
        + 2.0 * (summary["recent63"]["top1"] - formal["recent63"]["top1"])
        + 2.0 * (summary["recent63"]["top5"] - formal["recent63"]["top5"])
    )


def build_score(base: pd.DataFrame, start_date: str, alpha: float, top_zone: float | None) -> pd.DataFrame:
    frame = base.copy()
    frame["score"] = frame["formal_rank"]
    date_mask = frame["trade_date"].ge(start_date)
    if top_zone is None:
        mask = date_mask
    else:
        mask = date_mask & frame["formal_rank"].ge(top_zone)
    frame.loc[mask, "score"] = (
        (1.0 - alpha) * frame.loc[mask, "formal_rank"] + alpha * frame.loc[mask, "research_rank"]
    )
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report = {
        "generated_at": generated_at,
        "actor": "model-agent",
        "asset_role": "research_grid_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "results": {},
        "governance": {
            "no_training": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }

    with sqlite3.connect(MODEL_DB) as conn:
        for horizon in ["3d", "5d"]:
            label = LABELS[horizon]
            formal = read_pred(conn, FORMAL[horizon], "formal_score")
            research = read_pred(conn, RESEARCH[horizon], "research_score")
            frame = formal.merge(research, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
            frame = frame.merge(labels[["trade_date", "stock_code", label]], on=["trade_date", "stock_code"], how="inner")
            frame["formal_rank"] = frame.groupby("trade_date")["formal_score"].rank(method="average", pct=True)
            frame["research_rank"] = frame.groupby("trade_date")["research_score"].rank(method="average", pct=True)

            formal_eval, _ = evaluate(frame, label, "formal_rank")
            research_eval, _ = evaluate(frame, label, "research_rank")
            grid_rows = []
            best = None
            best_frame = None
            best_eval = None
            for start_date in START_DATES:
                for alpha in ALPHAS:
                    for top_zone in TOP_ZONES:
                        candidate = build_score(frame, start_date, alpha, top_zone)
                        summary, _ = evaluate(candidate, label, "score")
                        score = objective(summary, formal_eval, horizon)
                        row = {
                            "horizon": horizon,
                            "start_date": start_date,
                            "alpha": alpha,
                            "top_zone": "all" if top_zone is None else top_zone,
                            "objective": score,
                        }
                        for window in ["full", "recent126", "recent63"]:
                            for metric in ["rank_ic", "top1", "top5", "top10"]:
                                row[f"{window}_{metric}"] = summary[window][metric]
                                row[f"{window}_delta_{metric}"] = summary[window][metric] - formal_eval[window][metric]
                        grid_rows.append(row)
                        if best is None or row["objective"] > best["objective"]:
                            best = row
                            best_frame = candidate
                            best_eval = summary

            grid = pd.DataFrame(grid_rows).sort_values(
                ["objective", "recent63_delta_top1", "full_delta_rank_ic"],
                ascending=[False, False, False],
            )
            grid.to_csv(OUT_DIR / f"{horizon}_stability_anchor_grid.csv", index=False, encoding="utf-8-sig")
            grid.head(20).to_csv(OUT_DIR / f"{horizon}_stability_anchor_grid_top20.csv", index=False, encoding="utf-8-sig")

            out = best_frame[["trade_date", "stock_code", label, "score", "formal_score", "research_score", "formal_rank", "research_rank"]].rename(
                columns={"score": "pred_prob"}
            )
            out.to_sql(TARGETS[horizon], conn, if_exists="replace", index=False)
            conn.execute(f"create index if not exists idx_{TARGETS[horizon]}_date_code on '{TARGETS[horizon]}'(trade_date, stock_code)")
            conn.execute(f"create index if not exists idx_{TARGETS[horizon]}_date_pred on '{TARGETS[horizon]}'(trade_date, pred_prob desc)")
            summary = table_summary(conn, TARGETS[horizon])

            report["results"][horizon] = {
                "target_table": TARGETS[horizon],
                "formal_table": FORMAL[horizon],
                "research_source_table": RESEARCH[horizon],
                "best_params": best,
                "target_eval": best_eval,
                "formal_eval": formal_eval,
                "research_eval": research_eval,
                "coverage": summary,
            }

    json_path = OUT_DIR / "stability_anchor_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 3D/5D 稳定锚研究报告 20260624",
        "",
        f"- 生成时间：{generated_at}",
        "- 方法：以 formal baseline 为稳定锚，对研究候选做日期门控、头部区域门控和线性 rank blend。",
        "- 本次未训练模型、未修改生产 manifest、未生成交易信号、未跑回测。",
        "",
    ]
    for horizon in ["3d", "5d"]:
        result = report["results"][horizon]
        best = result["best_params"]
        lines.extend(
            [
                f"## {horizon.upper()}",
                "",
                f"- 新研究表：`MODEL_PREDICTIONS.db::{result['target_table']}`",
                f"- 最优参数：start_date=`{best['start_date']}`，alpha=`{best['alpha']}`，top_zone=`{best['top_zone']}`。",
                f"- Full ΔRankIC=`{best['full_delta_rank_ic']:.6f}`，Full ΔTop1=`{best['full_delta_top1']:.6f}`，Full ΔTop5=`{best['full_delta_top5']:.6f}`。",
                f"- Recent63 ΔTop1=`{best['recent63_delta_top1']:.6f}`，Recent63 ΔTop5=`{best['recent63_delta_top5']:.6f}`。",
                f"- 覆盖：`{result['coverage']['min_trade_date']}` 到 `{result['coverage']['max_trade_date']}`，最新日行数 `{result['coverage']['rows_20260623']}`。",
                "",
            ]
        )
    lines.extend(
        [
            "## 证据路径",
            "",
            f"- JSON：`{json_path.as_posix()}`",
            f"- 3D 网格：`{(OUT_DIR / '3d_stability_anchor_grid.csv').as_posix()}`",
            f"- 5D 网格：`{(OUT_DIR / '5d_stability_anchor_grid.csv').as_posix()}`",
        ]
    )
    (OUT_DIR / "stability_anchor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
