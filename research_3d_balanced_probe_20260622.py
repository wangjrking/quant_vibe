from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from evaluate_prediction_asset import evaluate_frame


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_3d_balanced_probe_20260622"

LABEL_COL = "executable_3d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260618"
VALID_FROM = "20260101"
TOP_K = [1, 3, 5, 10, 20, 50]

TABLES = {
    "current_3d_aux": "stock_predict_data_model_agent_3d_aux_horizon_fusion_20260622_executable_3d_open_return_research",
    "formal_3d": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_3d_open_return_score_20240604_20260618",
    "research_1d_dense": "stock_predict_data_model_agent_1d_dense_aux_fusion_20260622_executable_1d_open_return_research",
    "research_5d_topgate": "stock_predict_data_model_agent_5d_topgate_fusion_20260622_executable_5d_open_return_research",
    "research_10d_topgate": "stock_predict_data_model_agent_10d_topgate_fusion_20260622_executable_10d_open_return_research",
    "research_10d_aux": "stock_predict_data_model_agent_10d_aux_horizon_fusion_20260622_executable_10d_open_return_research",
}


def u(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


def read_table(conn: sqlite3.Connection, alias: str, table: str, with_label: bool = False) -> pd.DataFrame:
    cols = "trade_date, stock_code, pred_prob"
    if with_label:
        cols += f", {LABEL_COL}"
    frame = pd.read_sql_query(
        f"select {cols} from '{table}' where trade_date >= ? and trade_date <= ?",
        conn,
        params=[DATE_FROM, DATE_TO],
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.rename(columns={"pred_prob": f"{alias}_score"})


def add_rank(frame: pd.DataFrame, alias: str) -> None:
    frame[f"{alias}_rank"] = frame.groupby("trade_date")[f"{alias}_score"].rank(method="average", pct=True)


def front_score(summary: dict) -> float:
    top = summary["top_k_mean_returns"]
    return (
        float(top["1"]) * 3.0
        + float(top["3"]) * 2.5
        + float(top["5"]) * 1.5
        + float(top["10"])
        + float(summary["daily_rank_ic_mean"]) * 0.10
    )


def compact(name: str, period: str, summary: dict) -> dict:
    top = summary["top_k_mean_returns"]
    return {
        "name": name,
        "period": period,
        "trade_days": summary["trade_days"],
        "date_min": summary["date_min"],
        "date_max": summary["date_max"],
        "front_score": front_score(summary),
        "rank_ic": summary["daily_rank_ic_mean"],
        "rank_ic_pos": summary["rank_ic_positive_ratio"],
        "top_bottom": summary["top_minus_bottom_mean"],
        "top1": top["1"],
        "top3": top["3"],
        "top5": top["5"],
        "top10": top["10"],
        "top20": top["20"],
        "top50": top["50"],
    }


def eval_score(frame: pd.DataFrame, name: str, score: pd.Series) -> list[dict]:
    data = frame[["trade_date", "stock_code", LABEL_COL]].copy()
    data["pred_prob"] = score.astype(float)
    full, _ = evaluate_frame(data, score_col="pred_prob", label_col=LABEL_COL, top_k=TOP_K, quantiles=10)
    valid, _ = evaluate_frame(
        data[data["trade_date"] >= VALID_FROM].copy(),
        score_col="pred_prob",
        label_col=LABEL_COL,
        top_k=TOP_K,
        quantiles=10,
    )
    return [compact(name, "full", full), compact(name, "valid", valid)]


def candidate_scores(frame: pd.DataFrame):
    base = frame["current_3d_aux_rank"]
    yield "current_3d_aux_rank", base
    for alias in TABLES:
        yield f"{alias}_rank", frame[f"{alias}_rank"]

    focused_aux = ["formal_3d", "research_10d_topgate", "research_5d_topgate", "research_1d_dense"]
    for alias in focused_aux:
        aux = frame[f"{alias}_rank"]
        for base_w in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
            aux_w = 1.0 - base_w
            yield f"blend_currentw{base_w:.2f}_{alias}w{aux_w:.2f}".replace(".", "p"), base * base_w + aux * aux_w
        for threshold in [0.985, 0.990, 0.995, 0.9975]:
            for boost in [0.001, 0.003, 0.005, 0.010, 0.020]:
                yield (
                    f"gate_current_t{threshold:.4f}_{alias}_b{boost:.3f}".replace(".", "p"),
                    base.where(base < threshold, base + aux * boost),
                )

    triples = [
        ("formal_3d", "research_10d_topgate", 0.80, 0.10, 0.10),
        ("formal_3d", "research_10d_topgate", 0.70, 0.20, 0.10),
        ("formal_3d", "research_5d_topgate", 0.70, 0.20, 0.10),
        ("research_10d_topgate", "research_5d_topgate", 0.70, 0.15, 0.15),
        ("formal_3d", "research_1d_dense", 0.75, 0.15, 0.10),
    ]
    for aux1, aux2, base_w, w1, w2 in triples:
        yield (
            f"triple_current{base_w:.2f}_{aux1}{w1:.2f}_{aux2}{w2:.2f}".replace(".", "p"),
            base * base_w + frame[f"{aux1}_rank"] * w1 + frame[f"{aux2}_rank"] * w2,
        )


def flatten_delta(row: dict, current: dict) -> dict:
    metrics = ["front_score", "rank_ic", "rank_ic_pos", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    return {f"{metric}_delta": row[metric] - current[metric] for metric in metrics}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB) as conn:
        frame = read_table(conn, "current_3d_aux", TABLES["current_3d_aux"], with_label=True)
        for alias, table in TABLES.items():
            if alias == "current_3d_aux":
                continue
            frame = frame.merge(
                read_table(conn, alias, table),
                on=["trade_date", "stock_code"],
                how="left",
                validate="one_to_one",
            )
    for alias in TABLES:
        frame[f"{alias}_score"] = frame[f"{alias}_score"].fillna(frame["current_3d_aux_score"])
        add_rank(frame, alias)
        frame[f"{alias}_rank"] = frame[f"{alias}_rank"].fillna(frame["current_3d_aux_rank"])

    rows = []
    scores = {}
    for idx, (name, score) in enumerate(candidate_scores(frame), start=1):
        scores[name] = score
        rows.extend(eval_score(frame, name, score))
        if idx % 20 == 0:
            print(json.dumps({"evaluated": idx}, ensure_ascii=False), flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "all_candidates.csv", index=False, encoding="utf-8-sig")

    current_full = result[(result["name"] == "current_3d_aux_rank") & (result["period"] == "full")].iloc[0].to_dict()
    current_valid = result[(result["name"] == "current_3d_aux_rank") & (result["period"] == "valid")].iloc[0].to_dict()
    wide = []
    for name, group in result.groupby("name", sort=False):
        full = group[group["period"] == "full"].iloc[0].to_dict()
        valid = group[group["period"] == "valid"].iloc[0].to_dict()
        wide.append(
            {
                "name": name,
                **{f"full_{key}": value for key, value in full.items() if key not in {"name", "period"}},
                **{f"valid_{key}": value for key, value in valid.items() if key not in {"name", "period"}},
                **{f"full_{key}": value for key, value in flatten_delta(full, current_full).items()},
                **{f"valid_{key}": value for key, value in flatten_delta(valid, current_valid).items()},
            }
        )
    wide_df = pd.DataFrame(wide)
    balanced = wide_df[
        (wide_df["full_top3_delta"] > 0.0005)
        & (wide_df["full_top5_delta"] >= 0)
        & (wide_df["full_top10_delta"] >= 0)
        & (wide_df["full_top1_delta"] >= -0.003)
        & (wide_df["full_rank_ic_delta"] >= -0.003)
        & (wide_df["valid_front_score_delta"] >= -0.001)
    ].copy()
    if not balanced.empty:
        balanced["selection_score"] = (
            balanced["full_top3_delta"] * 3.0
            + balanced["full_top5_delta"] * 1.5
            + balanced["full_top10_delta"]
            + balanced["full_top1_delta"] * 1.5
            + balanced["valid_front_score_delta"]
            + balanced["full_rank_ic_delta"] * 0.10
        )
        balanced = balanced.sort_values("selection_score", ascending=False)
    balanced.to_csv(OUT_DIR / "balanced_candidates.csv", index=False, encoding="utf-8-sig")

    manifest = None
    if not balanced.empty and balanced.iloc[0]["selection_score"] > 0.001:
        best_name = str(balanced.iloc[0]["name"])
        table = "stock_predict_data_model_agent_3d_balanced_probe_20260622_executable_3d_open_return_research"
        out = frame.copy()
        out["pred_prob"] = scores[best_name].astype(float)
        keep = ["trade_date", "stock_code", "pred_prob", LABEL_COL]
        keep += [col for col in out.columns if col.endswith("_score") or col.endswith("_rank")]
        out = out[[col for col in keep if col in out.columns]]
        with sqlite3.connect(MODEL_DB) as conn:
            out.to_sql(table, conn, if_exists="replace", index=False)
            conn.execute(f"create index if not exists idx_{table}_date_code on '{table}'(trade_date, stock_code)")
            conn.execute(f"create index if not exists idx_{table}_date_pred on '{table}'(trade_date, pred_prob desc)")
            row = conn.execute(
                f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), count(distinct stock_code) from '{table}'"
            ).fetchone()
            dup = conn.execute(
                f"select count(*) from (select trade_date, stock_code, count(*) c from '{table}' "
                "group by trade_date, stock_code having c > 1)"
            ).fetchone()[0]
            null_pred = conn.execute(f"select count(*) from '{table}' where pred_prob is null").fetchone()[0]
        manifest = {
            "asset_status": "research_only_not_l5_approved",
            "approval_status": "research_only",
            "source_type": "sqlite_table",
            "asset_role": "l4_research_prediction_asset",
            "label_col": LABEL_COL,
            "db_path": str(MODEL_DB),
            "table": table,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "score_formula": best_name,
            "row_count": int(row[0]),
            "min_trade_date": str(row[1]),
            "max_trade_date": str(row[2]),
            "trade_days": int(row[3]),
            "stocks": int(row[4]),
            "duplicate_keys": int(dup),
            "null_pred_prob": int(null_pred),
            "inputs": TABLES,
            "notes": u("\\u0033\\u0044 \\u5747\\u8861\\u6539\\u5584\\u7814\\u7a76\\u8d44\\u4ea7\\uff0c\\u4e0d\\u4f5c\\u4e3a L5 \\u6b63\\u5f0f\\u5165\\u53e3\\u3002"),
        }
        (OUT_DIR / "research_prediction_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "research_only_3d_balanced_probe",
        "candidate_count": int(result["name"].nunique()),
        "balanced_count": int(len(balanced)),
        "best_balanced": balanced.head(5).to_dict("records"),
        "manifest": manifest,
        "model_side_limits": [
            "No model trained",
            "No formal manifest changed",
            "No trading signal generated",
            "No strategy rule or backtest conclusion generated",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 3D 均衡候选搜索报告（20260622）",
        "",
        "本轮只使用已存在预测表做 rank 组合搜索；未训练模型，未发布 formal，未生成信号或回测结论。",
        "",
        f"- 候选数：{summary['candidate_count']}",
        f"- 均衡通过数：{summary['balanced_count']}",
        f"- 落地研究表：`{manifest['table'] if manifest else ''}`",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'all_candidates.csv').as_posix()}`",
        f"- `{(OUT_DIR / 'balanced_candidates.csv').as_posix()}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_dir": str(OUT_DIR), "balanced_count": len(balanced), "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
