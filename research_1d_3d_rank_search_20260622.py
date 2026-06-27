from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

from evaluate_prediction_asset import evaluate_frame


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_3d_rank_search_20260622"

DATE_FROM = "20240604"
DATE_TO = "20260618"
VALID_FROM = "20260101"
TOP_K = [1, 3, 5, 10, 20, 50]

SOURCE_TABLES = {
    "formal_1d": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_1d_open_return_score_20240604_20260618",
    "research_1d_aux": "stock_predict_data_model_agent_1d_aux_horizon_fusion_20260622_executable_1d_open_return_research",
    "research_1d_dense": "stock_predict_data_model_agent_1d_dense_aux_fusion_20260622_executable_1d_open_return_research",
    "formal_3d": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_3d_open_return_score_20240604_20260618",
    "research_3d_aux": "stock_predict_data_model_agent_3d_aux_horizon_fusion_20260622_executable_3d_open_return_research",
    "formal_5d": "stock_predict_data_model_agent_5d_tune_20260620_executable_5d_open_return_d4_l4_fs160_gate1_score_20240604_20260618",
    "research_5d_aux": "stock_predict_data_model_agent_5d_aux_horizon_fusion_20260622_executable_5d_open_return_research",
    "research_5d_topgate": "stock_predict_data_model_agent_5d_topgate_fusion_20260622_executable_5d_open_return_research",
    "formal_10d": "stock_predict_data_model_agent_10d_tune_20260620_executable_10d_open_return_d4_l4_fs160_gate2_score_20240604_20260618",
    "research_10d_aux": "stock_predict_data_model_agent_10d_aux_horizon_fusion_20260622_executable_10d_open_return_research",
    "research_10d_topgate": "stock_predict_data_model_agent_10d_topgate_fusion_20260622_executable_10d_open_return_research",
}

TARGETS = {
    "executable_1d_open_return": {
        "current": "research_1d_dense",
        "formal": "formal_1d",
        "sources": [
            "research_1d_dense",
            "research_1d_aux",
            "formal_1d",
            "research_3d_aux",
            "research_5d_topgate",
            "research_10d_topgate",
        ],
    },
    "executable_3d_open_return": {
        "current": "research_3d_aux",
        "formal": "formal_3d",
        "sources": [
            "research_3d_aux",
            "formal_3d",
            "research_1d_dense",
            "research_5d_topgate",
            "research_10d_topgate",
            "research_10d_aux",
        ],
    },
}


def u(text: str) -> str:
    return text.encode("ascii").decode("unicode_escape")


def read_source(conn: sqlite3.Connection, alias: str, table: str, label_col: str | None = None) -> pd.DataFrame:
    cols = "trade_date, stock_code, pred_prob"
    if label_col:
        cols += f", {label_col}"
    frame = pd.read_sql_query(
        f"select {cols} from '{table}' where trade_date >= ? and trade_date <= ?",
        conn,
        params=[DATE_FROM, DATE_TO],
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    rename = {"pred_prob": f"{alias}_score"}
    return frame.rename(columns=rename)


def rank_col(frame: pd.DataFrame, score_col: str, out_col: str) -> None:
    frame[out_col] = frame.groupby("trade_date")[score_col].rank(method="average", pct=True)


def front_score(summary: dict) -> float:
    top = summary["top_k_mean_returns"]
    return (
        float(top["1"]) * 3.0
        + float(top["3"]) * 2.0
        + float(top["5"]) * 1.5
        + float(top["10"])
        + float(summary["daily_rank_ic_mean"]) * 0.10
        + float(summary["top_minus_bottom_mean"]) * 0.50
    )


def compact(summary: dict, name: str, period: str) -> dict:
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


def evaluate_candidate(frame: pd.DataFrame, label_col: str, name: str, score: pd.Series) -> dict:
    data = frame[["trade_date", "stock_code", label_col]].copy()
    data["pred_prob"] = score.astype(float)
    full, _ = evaluate_frame(data, score_col="pred_prob", label_col=label_col, top_k=TOP_K, quantiles=10)
    valid_data = data[data["trade_date"] >= VALID_FROM].copy()
    valid, _ = evaluate_frame(valid_data, score_col="pred_prob", label_col=label_col, top_k=TOP_K, quantiles=10)
    return {
        "name": name,
        "full": compact(full, name, "full"),
        "valid": compact(valid, name, "valid"),
    }


def iter_candidates(frame: pd.DataFrame, target: dict):
    current = target["current"]
    source_aliases = target["sources"]
    yield f"{current}_raw", frame[f"{current}_score"]
    yield f"{current}_rank", frame[f"{current}_rank"]
    for alias in source_aliases:
        yield f"{alias}_rank", frame[f"{alias}_rank"]

    base = frame[f"{current}_rank"]
    blend_weights = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
    for alias in source_aliases:
        if alias == current:
            continue
        aux = frame[f"{alias}_rank"]
        for base_w in blend_weights:
            aux_w = 1.0 - base_w
            name = f"blend_{current}w{base_w:.2f}_{alias}w{aux_w:.2f}".replace(".", "p")
            yield name, base * base_w + aux * aux_w
        for threshold in [0.990, 0.995, 0.9975, 0.999]:
            for boost in [0.001, 0.003, 0.005, 0.010]:
                name = f"gate_{current}_t{threshold:.4f}_{alias}_b{boost:.3f}".replace(".", "p")
                yield name, base.where(base < threshold, base + aux * boost)

    aux_pool = [alias for alias in source_aliases if alias != current]
    for combo_size in [2, 3]:
        for combo in combinations(aux_pool, combo_size):
            cols = [f"{alias}_rank" for alias in combo]
            name = "mean_" + "_".join(combo)
            yield name, frame[cols].mean(axis=1)
            name = "min_" + "_".join(combo)
            yield name, frame[cols].min(axis=1)
            name = "max_" + "_".join(combo)
            yield name, frame[cols].max(axis=1)


def delta(candidate: dict, baseline: dict, prefix: str) -> dict:
    keys = ["front_score", "rank_ic", "rank_ic_pos", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    return {f"{prefix}_{key}_delta": candidate[key] - baseline[key] for key in keys}


def choose_balanced(rows: pd.DataFrame, current_name: str) -> pd.DataFrame:
    current = rows[(rows["name"] == f"{current_name}_rank") & (rows["period"] == "full")].iloc[0].to_dict()
    current_valid = rows[(rows["name"] == f"{current_name}_rank") & (rows["period"] == "valid")].iloc[0].to_dict()
    wide = []
    for name, group in rows.groupby("name", sort=False):
        full = group[group["period"] == "full"].iloc[0].to_dict()
        valid = group[group["period"] == "valid"].iloc[0].to_dict()
        row = {
            "name": name,
            **{f"full_{key}": value for key, value in full.items() if key not in {"name", "period"}},
            **{f"valid_{key}": value for key, value in valid.items() if key not in {"name", "period"}},
            **delta(full, current, "vs_current_full"),
            **delta(valid, current_valid, "vs_current_valid"),
        }
        wide.append(row)
    result = pd.DataFrame(wide)
    balanced = result[
        (result["vs_current_full_top1_delta"] >= 0)
        & (result["vs_current_full_top3_delta"] >= 0)
        & (result["vs_current_full_top5_delta"] >= 0)
        & (result["vs_current_full_top10_delta"] >= 0)
        & (result["vs_current_valid_front_score_delta"] >= 0)
        & (result["vs_current_valid_rank_ic_delta"] >= -0.002)
    ].copy()
    if balanced.empty:
        return balanced
    balanced["selection_score"] = (
        balanced["vs_current_full_top1_delta"] * 3.0
        + balanced["vs_current_full_top3_delta"] * 2.0
        + balanced["vs_current_full_top5_delta"] * 1.5
        + balanced["vs_current_valid_front_score_delta"]
        + balanced["vs_current_full_rank_ic_delta"] * 0.10
    )
    return balanced.sort_values("selection_score", ascending=False)


def materialize_research(label_col: str, frame: pd.DataFrame, score: pd.Series, candidate_name: str) -> dict:
    suffix = "1d" if "1d" in label_col else "3d"
    table = f"stock_predict_data_model_agent_{suffix}_rank_search_20260622_{label_col}_research"
    out = frame.copy()
    out["pred_prob"] = score.astype(float)
    keep_cols = ["trade_date", "stock_code", "pred_prob", label_col]
    keep_cols += [col for col in out.columns if col.endswith("_score") or col.endswith("_rank")]
    out = out[[col for col in keep_cols if col in out.columns]]
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
        "label_col": label_col,
        "db_path": str(MODEL_DB),
        "table": table,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "score_formula": candidate_name,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stocks": int(row[4]),
        "duplicate_keys": int(dup),
        "null_pred_prob": int(null_pred),
        "inputs": SOURCE_TABLES,
        "notes": u("\\u7814\\u7a76\\u5e93\\u5019\\u9009\\u8d44\\u4ea7\\uff0c\\u4ec5\\u7528\\u5df2\\u6709\\u9884\\u6d4b\\u8868\\u505a rank \\u7ec4\\u5408\\u641c\\u7d22\\uff0c\\u672a\\u8bad\\u7ec3\\u65b0\\u6a21\\u578b\\uff0c\\u4e0d\\u4f5c\\u4e3a L5 \\u6b63\\u5f0f\\u5165\\u53e3\\u3002"),
    }
    return manifest


def target_run(label_col: str, target: dict) -> dict:
    with sqlite3.connect(MODEL_DB) as conn:
        frame = read_source(conn, target["current"], SOURCE_TABLES[target["current"]], label_col=label_col)
        for alias in target["sources"]:
            if alias == target["current"]:
                continue
            frame = frame.merge(
                read_source(conn, alias, SOURCE_TABLES[alias]),
                on=["trade_date", "stock_code"],
                how="left",
                validate="one_to_one",
            )

    source_aliases = target["sources"]
    for alias in source_aliases:
        score_col = f"{alias}_score"
        if score_col not in frame.columns:
            continue
        rank_col(frame, score_col, f"{alias}_rank")
    current_rank = f"{target['current']}_rank"
    for alias in source_aliases:
        rank_name = f"{alias}_rank"
        if rank_name in frame.columns:
            frame[rank_name] = frame[rank_name].fillna(frame[current_rank])
    for alias in source_aliases:
        score_name = f"{alias}_score"
        if score_name in frame.columns:
            frame[score_name] = frame[score_name].fillna(frame[f"{target['current']}_score"])

    rows = []
    candidate_count = 0
    for name, score in iter_candidates(frame, target):
        candidate_count += 1
        evaluation = evaluate_candidate(frame, label_col, name, score)
        rows.append(evaluation["full"])
        rows.append(evaluation["valid"])
        if candidate_count % 25 == 0:
            print(json.dumps({"label": label_col, "evaluated_candidates": candidate_count}, ensure_ascii=False))

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / f"{label_col}_all_candidates.csv", index=False, encoding="utf-8-sig")
    balanced = choose_balanced(result, target["current"])
    balanced.to_csv(OUT_DIR / f"{label_col}_balanced_candidates.csv", index=False, encoding="utf-8-sig")

    manifest = None
    if not balanced.empty:
        best_name = str(balanced.iloc[0]["name"])
        if best_name not in {f"{target['current']}_raw", f"{target['current']}_rank"}:
            best_score = None
            for name, score in iter_candidates(frame, target):
                if name == best_name:
                    best_score = score
                    break
            if best_score is None:
                raise KeyError(f"Best candidate {best_name} was not regenerated")
            manifest = materialize_research(label_col, frame, best_score, best_name)
            (OUT_DIR / f"{label_col}_research_prediction_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return {
        "label": label_col,
        "candidate_count": candidate_count,
        "balanced_count": int(len(balanced)),
        "best_balanced": balanced.head(1).to_dict("records"),
        "manifest": manifest,
    }


def write_report(summary: dict) -> None:
    title = u("\\u0031\\u0044\\u002f\\u0033\\u0044 rank \\u7ec4\\u5408\\u641c\\u7d22\\u62a5\\u544a\\uff0820260622\\uff09")
    lines = [
        f"# {title}",
        "",
        u("\\u672c\\u8f6e\\u53ea\\u4f7f\\u7528\\u5df2\\u6709\\u9884\\u6d4b\\u8d44\\u4ea7\\u505a rank \\u7ec4\\u5408\\u548c\\u95e8\\u63a7\\u641c\\u7d22\\uff0c\\u672a\\u8bad\\u7ec3\\u6a21\\u578b\\uff0c\\u672a\\u6539 formal manifest\\uff0c\\u672a\\u751f\\u6210\\u4ea4\\u6613\\u4fe1\\u53f7\\u6216\\u56de\\u6d4b\\u7ed3\\u8bba\\u3002"),
        "",
        "| 标签 | 候选数 | 均衡通过数 | 最优候选 | 是否落研究表 |",
        "|---|---:|---:|---|---|",
    ]
    for item in summary["targets"]:
        best = item["best_balanced"][0]["name"] if item["best_balanced"] else ""
        table = item["manifest"]["table"] if item["manifest"] else ""
        lines.append(
            f"| `{item['label']}` | {item['candidate_count']} | {item['balanced_count']} | `{best}` | `{table}` |"
        )
    lines += [
        "",
        "## 证据路径",
        "",
        f"- JSON: `{(OUT_DIR / 'rank_search_summary.json').as_posix()}`",
        f"- 1D 全候选: `{(OUT_DIR / 'executable_1d_open_return_all_candidates.csv').as_posix()}`",
        f"- 1D 均衡候选: `{(OUT_DIR / 'executable_1d_open_return_balanced_candidates.csv').as_posix()}`",
        f"- 3D 全候选: `{(OUT_DIR / 'executable_3d_open_return_all_candidates.csv').as_posix()}`",
        f"- 3D 均衡候选: `{(OUT_DIR / 'executable_3d_open_return_balanced_candidates.csv').as_posix()}`",
    ]
    (OUT_DIR / "rank_search_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "research_only_1d_3d_rank_search",
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "valid_from": VALID_FROM,
        "targets": [],
        "model_side_limits": [
            "No model trained",
            "No formal manifest changed",
            "No trading signal generated",
            "No strategy rule or backtest conclusion generated",
        ],
    }
    for label_col, target in TARGETS.items():
        summary["targets"].append(target_run(label_col, target))
    (OUT_DIR / "rank_search_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps({"report_dir": str(OUT_DIR), "targets": len(summary["targets"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
