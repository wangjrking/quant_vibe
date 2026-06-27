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
OUT_DIR = DATA_DIR / "reports" / "model_agent_5d_weakmonth_reblend_20260623"

BASE_TABLE = "stock_predict_data_model_agent_5d_gate6_balanced_bonus_20260622_executable_5d_open_return_research"
AUX_TABLE = "stock_predict_data_model_agent_5d_weakdef_20260622_executable_5d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_5d_weakmonth_reblend_20260623_executable_5d_open_return_research"

LABEL_COL = "executable_5d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260605"
TOP_K = [1, 3, 5, 10, 20, 50]
WEAK_MONTHS = {"202604", "202605"}
DATE_GATES = ["20260401"]
ZONE_PCTS = [0.005, 0.01, 0.02]
ALPHAS = [0.002, 0.005, 0.01]


def read_prediction(conn: sqlite3.Connection, table: str, alias: str, keep_payload: bool = False) -> pd.DataFrame:
    if keep_payload:
        frame = pd.read_sql_query(
            f"select * from '{table}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
    else:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob from '{table}' where trade_date >= ? and trade_date <= ?",
            conn,
            params=[DATE_FROM, DATE_TO],
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame.rename(columns={"pred_prob": alias})


def add_rank(frame: pd.DataFrame, score_col: str, rank_col: str) -> None:
    frame[rank_col] = frame.groupby("trade_date")[score_col].rank(method="average", pct=True)


def evaluate_full(frame: pd.DataFrame, score_col: str) -> tuple[dict, pd.DataFrame]:
    subset = frame.copy()
    subset["pred_prob"] = subset[score_col].astype(float)
    summary, daily = evaluate_frame(
        subset,
        score_col="pred_prob",
        label_col=LABEL_COL,
        top_k=TOP_K,
        quantiles=10,
    )
    return summary, daily


def summarize_daily_window(daily: pd.DataFrame, date_from: str, date_to: str) -> dict:
    window = daily[(daily["trade_date"] >= date_from) & (daily["trade_date"] <= date_to)].copy()
    if window.empty:
        raise ValueError(f"No daily evaluation rows for window {date_from}..{date_to}")
    return {
        "date_from": date_from,
        "date_to": date_to,
        "trade_days": int(window["trade_date"].nunique()),
        "rank_ic": float(window["rank_ic"].mean()),
        "top1": float(window["top1_mean"].mean()),
        "top3": float(window["top3_mean"].mean()),
        "top5": float(window["top5_mean"].mean()),
        "top10": float(window["top10_mean"].mean()),
        "top20": float(window["top20_mean"].mean()),
        "top50": float(window["top50_mean"].mean()),
        "top_bottom": float(window["top_minus_bottom"].mean()),
    }


def composite_score(full_metrics: dict, y1_metrics: dict, recent_metrics: dict) -> float:
    full_score = (
        full_metrics["top1"] * 2.0
        + full_metrics["top3"] * 1.5
        + full_metrics["top5"] * 1.0
        + full_metrics["top10"] * 0.5
        + full_metrics["rank_ic"] * 0.10
    )
    y1_score = (
        y1_metrics["top1"] * 2.0
        + y1_metrics["top3"] * 1.5
        + y1_metrics["top5"] * 1.25
        + y1_metrics["top10"] * 0.75
        + y1_metrics["rank_ic"] * 0.10
    )
    recent_score = (
        recent_metrics["top1"] * 2.5
        + recent_metrics["top3"] * 2.0
        + recent_metrics["top5"] * 1.5
        + recent_metrics["top10"] * 1.0
        + recent_metrics["rank_ic"] * 0.10
    )
    return full_score * 0.35 + y1_score * 0.25 + recent_score * 0.40


def materialize(
    merged: pd.DataFrame,
    score_col: str,
    selected: dict,
    base_windows: dict,
    selected_windows: dict,
) -> dict:
    out = merged.copy()
    out["pred_prob"] = out[score_col].astype(float)
    keep_cols = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "base_5d_pred_prob",
        "rank_base_5d",
        "aux_weakdef_pred_prob",
        "rank_aux_weakdef",
        "gate_active",
        "gate_reason",
        LABEL_COL,
    ]
    out = out[[col for col in keep_cols if col in out.columns]]
    with sqlite3.connect(MODEL_DB) as conn:
        out.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_code on '{RESEARCH_TABLE}'(trade_date, stock_code)"
        )
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_pred on '{RESEARCH_TABLE}'(trade_date, pred_prob desc)"
        )
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), "
            f"count(distinct stock_code) from '{RESEARCH_TABLE}'"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from '{RESEARCH_TABLE}' "
            "group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        null_pred = conn.execute(f"select count(*) from '{RESEARCH_TABLE}' where pred_prob is null").fetchone()[0]
    manifest = {
        "asset_status": "research_only_not_l5_approved",
        "approval_status": "research_only",
        "source_type": "sqlite_table",
        "asset_role": "l4_research_prediction_asset",
        "label_col": LABEL_COL,
        "db_path": str(MODEL_DB),
        "table": RESEARCH_TABLE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_table": BASE_TABLE,
        "aux_table": AUX_TABLE,
        "score_formula": (
            "5D weakmonth/dategate local rerank: if gate active and base_5d_rank >= zone_pct, "
            "score = base_5d_rank + alpha * weakdef_rank; otherwise keep base_5d_rank"
        ),
        "selected_rule": selected,
        "base_windows": base_windows,
        "selected_windows": selected_windows,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stocks": int(row[4]),
        "duplicate_keys": int(dup),
        "null_pred_prob": int(null_pred),
        "notes": "5D 弱月/近期局部重排研究资产，仅供研究验证，不进入 formal 或 L5。",
    }
    (OUT_DIR / "research_prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB) as conn:
        merged = read_prediction(conn, BASE_TABLE, "base_5d_pred_prob", keep_payload=True)
        aux = read_prediction(conn, AUX_TABLE, "aux_weakdef_pred_prob")
        merged = merged.merge(aux, on=["trade_date", "stock_code"], how="left", validate="one_to_one")

    add_rank(merged, "base_5d_pred_prob", "rank_base_5d")
    add_rank(merged, "aux_weakdef_pred_prob", "rank_aux_weakdef")
    merged["rank_aux_weakdef"] = merged["rank_aux_weakdef"].fillna(merged["rank_base_5d"])
    merged["trade_month"] = merged["trade_date"].str.slice(0, 6)

    windows = {
        "full": ("20240604", "20260605"),
        "recent1y": ("20250301", "20260605"),
        "recent80": ("20260201", "20260605"),
    }
    _, base_daily = evaluate_full(merged, "base_5d_pred_prob")
    base_window_eval = {
        name: summarize_daily_window(base_daily, date_from, date_to)
        for name, (date_from, date_to) in windows.items()
    }
    base_composite = composite_score(
        base_window_eval["full"], base_window_eval["recent1y"], base_window_eval["recent80"]
    )

    rows: list[dict] = []
    best_name: str | None = None
    best_score = float("-inf")

    for gate_mode in ("weak_months", "date_gate"):
        gate_values = sorted(WEAK_MONTHS) if gate_mode == "weak_months" else DATE_GATES
        for gate_value in gate_values:
            if gate_mode == "weak_months":
                gate_mask = merged["trade_month"].isin({gate_value})
            else:
                gate_mask = merged["trade_date"] >= gate_value
            for zone_pct in ZONE_PCTS:
                zone_mask = merged["rank_base_5d"] >= (1.0 - zone_pct)
                for alpha in ALPHAS:
                    score_col = f"score_{gate_mode}_{str(gate_value).replace('.', '')}_z{str(zone_pct).replace('.', 'p')}_a{str(alpha).replace('.', 'p')}"
                    merged[score_col] = merged["rank_base_5d"]
                    active = gate_mask & zone_mask
                    merged.loc[active, score_col] = (
                        merged.loc[active, "rank_base_5d"] + alpha * merged.loc[active, "rank_aux_weakdef"]
                    )
                    _, candidate_daily = evaluate_full(merged, score_col)
                    evals = {
                        name: summarize_daily_window(candidate_daily, date_from, date_to)
                        for name, (date_from, date_to) in windows.items()
                    }
                    score = composite_score(evals["full"], evals["recent1y"], evals["recent80"])
                    row = {
                        "gate_mode": gate_mode,
                        "gate_value": gate_value,
                        "zone_pct": zone_pct,
                        "alpha": alpha,
                        "active_rows": int(active.sum()),
                        "full_rank_ic": evals["full"]["rank_ic"],
                        "full_top1": evals["full"]["top1"],
                        "full_top3": evals["full"]["top3"],
                        "full_top5": evals["full"]["top5"],
                        "full_top10": evals["full"]["top10"],
                        "recent1y_top1": evals["recent1y"]["top1"],
                        "recent1y_top3": evals["recent1y"]["top3"],
                        "recent1y_top5": evals["recent1y"]["top5"],
                        "recent80_top1": evals["recent80"]["top1"],
                        "recent80_top3": evals["recent80"]["top3"],
                        "recent80_top5": evals["recent80"]["top5"],
                        "recent80_top10": evals["recent80"]["top10"],
                        "recent80_rank_ic": evals["recent80"]["rank_ic"],
                        "composite": score,
                    }
                    rows.append(row)
                    if score > best_score:
                        best_score = score
                        best_name = score_col
                        best_rule = row
                        best_window_eval = evals

    summary = pd.DataFrame(rows).sort_values(
        ["composite", "recent80_top1", "recent80_top3", "recent80_top5"],
        ascending=False,
    )
    summary.to_csv(OUT_DIR / "grid_summary.csv", index=False, encoding="utf-8-sig")

    selected = None
    manifest = None
    if best_name is not None:
        passes = (
            best_window_eval["full"]["top1"] >= base_window_eval["full"]["top1"] - 0.0005
            and best_window_eval["full"]["top3"] >= base_window_eval["full"]["top3"] - 0.0005
            and best_window_eval["full"]["top5"] >= base_window_eval["full"]["top5"] - 0.0005
            and best_window_eval["recent80"]["top1"] > base_window_eval["recent80"]["top1"]
            and best_window_eval["recent80"]["top3"] >= base_window_eval["recent80"]["top3"] - 0.002
            and best_window_eval["recent80"]["top5"] >= base_window_eval["recent80"]["top5"] - 0.002
        )
        selected = {
            **best_rule,
            "base_composite": base_composite,
            "delta_composite": best_score - base_composite,
            "selection_passed": passes,
        }
        if passes:
            merged["gate_active"] = False
            merged["gate_reason"] = ""
            if best_rule["gate_mode"] == "weak_months":
                gate_mask = merged["trade_month"].isin({best_rule["gate_value"]})
                merged.loc[gate_mask, "gate_reason"] = f"weak_month:{best_rule['gate_value']}"
            else:
                gate_mask = merged["trade_date"] >= str(best_rule["gate_value"])
                merged.loc[gate_mask, "gate_reason"] = f"date_gate:{best_rule['gate_value']}"
            zone_mask = merged["rank_base_5d"] >= (1.0 - float(best_rule["zone_pct"]))
            active = gate_mask & zone_mask
            merged.loc[active, "gate_active"] = True
            merged[best_name] = merged["rank_base_5d"]
            merged.loc[active, best_name] = (
                merged.loc[active, "rank_base_5d"]
                + float(best_rule["alpha"]) * merged.loc[active, "rank_aux_weakdef"]
            )
            manifest = materialize(merged, best_name, selected, base_window_eval, best_window_eval)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_table": BASE_TABLE,
        "aux_table": AUX_TABLE,
        "windows": base_window_eval,
        "base_composite": base_composite,
        "best_rule": selected,
        "best_windows": best_window_eval if best_name is not None else None,
        "materialized": manifest is not None,
        "research_table": RESEARCH_TABLE if manifest is not None else None,
        "selection_rule": (
            "优先提升 recent80 Top1/Top3/Top5，同时要求 full Top1/Top3/Top5 不明显回撤；"
            "仅生成 research-only 候选，不改 formal。"
        ),
    }
    (OUT_DIR / "grid_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 5D 弱月局部重排研究",
        "",
        "## 当前结论",
        "",
    ]
    if manifest is not None:
        lines.append("本轮找到满足门槛的 5D research-only 候选，并已落库。")
    else:
        lines.append("本轮未找到同时满足全窗口与 recent80 门槛的 5D research-only 候选。")
    lines.extend(
        [
            "",
            "## 基线",
            "",
            f"- 基线表：`{BASE_TABLE}`",
            f"- 辅助表：`{AUX_TABLE}`",
            f"- full Top1：`{base_window_eval['full']['top1']:.8f}`",
            f"- full Top3：`{base_window_eval['full']['top3']:.8f}`",
            f"- full Top5：`{base_window_eval['full']['top5']:.8f}`",
            f"- recent80 Top1：`{base_window_eval['recent80']['top1']:.8f}`",
            f"- recent80 Top3：`{base_window_eval['recent80']['top3']:.8f}`",
            f"- recent80 Top5：`{base_window_eval['recent80']['top5']:.8f}`",
            "",
            "## 最优规则",
            "",
            json.dumps(selected, ensure_ascii=False, indent=2) if selected is not None else "无",
            "",
            "## 证据路径",
            "",
            f"- 汇总 JSON：`{(OUT_DIR / 'grid_summary.json').as_posix()}`",
            f"- 汇总 CSV：`{(OUT_DIR / 'grid_summary.csv').as_posix()}`",
        ]
    )
    if manifest is not None:
        lines.append(f"- research manifest：`{(OUT_DIR / 'research_prediction_manifest.json').as_posix()}`")
    (OUT_DIR / "reblend_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
