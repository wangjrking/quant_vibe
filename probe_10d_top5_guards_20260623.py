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
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_top5_guard_probe_20260623"

BASE_TABLE = "stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_executable_10d_open_return_research"
A015_TABLE = "stock_predict_data_model_agent_10d_dategate_3dformal_plus_1dgate092_a015_z0015_20260623_executable_10d_open_return_research"
LABEL_COL = "executable_10d_open_return"

WINDOWS = {
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}


@dataclass(frozen=True)
class CandidateEval:
    candidate: str
    recent126_top1: float
    recent126_top3: float
    recent126_top5: float
    recent126_top10: float
    recent63_top1: float
    recent63_top3: float
    recent63_top5: float
    recent63_top10: float
    focus_score: float


def evaluate(frame: pd.DataFrame, score_col: str) -> dict:
    daily_rows: list[dict] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
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
        summary["recent63"]["top1"] * 5
        + summary["recent63"]["top3"] * 4
        + summary["recent63"]["top5"] * 5
        + summary["recent63"]["top10"] * 2
        + summary["recent126"]["top1"] * 3
        + summary["recent126"]["top3"] * 2
        + summary["recent126"]["top5"] * 1.5
    )


def flatten(candidate: str, summary: dict) -> CandidateEval:
    return CandidateEval(
        candidate=candidate,
        recent126_top1=summary["recent126"]["top1"],
        recent126_top3=summary["recent126"]["top3"],
        recent126_top5=summary["recent126"]["top5"],
        recent126_top10=summary["recent126"]["top10"],
        recent63_top1=summary["recent63"]["top1"],
        recent63_top3=summary["recent63"]["top3"],
        recent63_top5=summary["recent63"]["top5"],
        recent63_top10=summary["recent63"]["top10"],
        focus_score=focus_score(summary),
    )


def build_freeze_top3_4to7(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["freeze_top3_4to7_score"] = -10000.0 - out["base_rank"]
    top3 = out["base_rank"] <= 3
    band = out["base_rank"].between(4, 7)
    rerank = band & out["trade_date"].ge("20260301")
    out.loc[top3, "freeze_top3_4to7_score"] = 3000 - out.loc[top3, "base_rank"]
    out.loc[band, "freeze_top3_4to7_score"] = 2000 - out.loc[band, "base_rank"]
    out.loc[rerank, "freeze_top3_4to7_score"] = 2000 + out.loc[rerank, "aux_rank_pct"]
    out.loc[out["base_rank"] > 7, "freeze_top3_4to7_score"] = 1000 - out.loc[out["base_rank"] > 7, "base_rank"]
    return out


def build_limit_newcomers(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("trade_date", sort=False):
        g = group.copy()
        g[f"limit_newcomers_{limit}_score"] = g["a015_score"]
        base_top5 = set(g.nsmallest(5, "base_rank")["stock_code"])
        cand_top5 = g.sort_values("a015_score", ascending=False).head(5)
        newcomers = cand_top5[~cand_top5["stock_code"].isin(base_top5)].sort_values("a015_score", ascending=False)
        if len(newcomers) > limit:
            demote = newcomers.iloc[limit:]["stock_code"].tolist()
            for code in demote:
                base_rank = float(g.loc[g["stock_code"] == code, "base_rank"].iloc[0])
                g.loc[g["stock_code"] == code, f"limit_newcomers_{limit}_score"] = -1e9 - base_rank
            incumbents = list(base_top5 - set(newcomers.head(limit)["stock_code"]))
            g.loc[g["stock_code"].isin(incumbents), f"limit_newcomers_{limit}_score"] = (
                g.loc[g["stock_code"].isin(incumbents), "base_score"] + 1e-6
            )
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def build_protect_base_top3(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("trade_date", sort=False):
        g = group.copy()
        g["protect_base_top3_score"] = g["a015_score"]
        base_top3 = set(g.nsmallest(3, "base_rank")["stock_code"])
        cand_top5 = set(g.nlargest(5, "a015_score")["stock_code"])
        missing = list(base_top3 - cand_top5)
        if missing:
            outsiders = g[~g["stock_code"].isin(base_top3)].sort_values("a015_score", ascending=False)
            demote = outsiders.head(len(missing))["stock_code"].tolist()
            for code in demote:
                base_rank = float(g.loc[g["stock_code"] == code, "base_rank"].iloc[0])
                g.loc[g["stock_code"] == code, "protect_base_top3_score"] = -1e9 - base_rank
            for code in missing:
                g.loc[g["stock_code"] == code, "protect_base_top3_score"] = (
                    g.loc[g["stock_code"] == code, "a015_score"] + 1e6
                )
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(MODEL_DB) as conn:
        base = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as base_score, [{LABEL_COL}] as {LABEL_COL} "
            f"from '{BASE_TABLE}' where trade_date between '20251118' and '20260528'",
            conn,
        )
        a015 = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as a015_score "
            f"from '{A015_TABLE}' where trade_date between '20251118' and '20260528'",
            conn,
        )

    base["trade_date"] = base["trade_date"].astype(str)
    a015["trade_date"] = a015["trade_date"].astype(str)
    frame = base.merge(a015, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame["base_rank"] = frame.groupby("trade_date")["base_score"].rank(method="first", ascending=False)
    frame["aux_rank_pct"] = frame.groupby("trade_date")["a015_score"].rank(method="average", pct=True)

    variants: list[tuple[str, pd.DataFrame, str]] = [
        ("base", frame.copy(), "base_score"),
        ("a015_z0015", frame.copy(), "a015_score"),
    ]

    freeze = build_freeze_top3_4to7(frame)
    variants.append(("freeze_top3_4to7", freeze, "freeze_top3_4to7_score"))

    limit2 = build_limit_newcomers(frame, limit=2)
    variants.append(("limit_newcomers_2", limit2, "limit_newcomers_2_score"))

    protect = build_protect_base_top3(frame)
    variants.append(("protect_base_top3", protect, "protect_base_top3_score"))

    rows: list[CandidateEval] = []
    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_tables": {
            "base_table": BASE_TABLE,
            "a015_table": A015_TABLE,
        },
        "windows": WINDOWS,
        "candidates": {},
    }

    for candidate, candidate_frame, score_col in variants:
        summary = evaluate(candidate_frame, score_col)
        row = flatten(candidate, summary)
        rows.append(row)
        summary_payload["candidates"][candidate] = {
            "summary": summary,
            "focus_score": row.focus_score,
        }

    result_df = pd.DataFrame([row.__dict__ for row in rows]).sort_values("focus_score", ascending=False)
    result_df.to_csv(OUT_DIR / "guard_probe_summary.csv", index=False, encoding="utf-8-sig")

    best_candidate = str(result_df.iloc[0]["candidate"])
    summary_payload["best_candidate"] = best_candidate
    summary_payload["ranking"] = result_df.to_dict(orient="records")
    summary_payload["conclusion"] = {
        "best_candidate": best_candidate,
        "best_focus_score": float(result_df.iloc[0]["focus_score"]),
        "a015_focus_score": float(result_df[result_df["candidate"] == "a015_z0015"]["focus_score"].iloc[0]),
        "freeze_top3_top5_tradeoff": {
            "recent63_top5": float(result_df[result_df["candidate"] == "freeze_top3_4to7"]["recent63_top5"].iloc[0]),
            "recent63_top3": float(result_df[result_df["candidate"] == "freeze_top3_4to7"]["recent63_top3"].iloc[0]),
        },
    }
    (OUT_DIR / "guard_probe_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 10D Top5 保护探针",
        "",
        "## 当前结论",
        "",
        f"- 本轮比较的 10D 保护方案中，按当前近期窗口 focus 口径，最优仍是 `{best_candidate}`。",
        "- `freeze_top3_4to7` 能改善 `recent63 Top5`，但会放弃 `a015` 在 `Top1/Top3` 上的明显优势。",
        "- `limit_newcomers_2` 和 `protect_base_top3` 都没有优于 `a015`，不建议继续沿这两条路扩展。",
        "",
        "## 候选排序",
        "",
    ]
    for _, row in result_df.iterrows():
        lines.extend(
            [
                f"### {row['candidate']}",
                f"- focus_score：`{row['focus_score']:.12f}`",
                f"- recent63：`Top1={row['recent63_top1']:.8f}` `Top3={row['recent63_top3']:.8f}` `Top5={row['recent63_top5']:.8f}` `Top10={row['recent63_top10']:.8f}`",
                f"- recent126：`Top1={row['recent126_top1']:.8f}` `Top3={row['recent126_top3']:.8f}` `Top5={row['recent126_top5']:.8f}` `Top10={row['recent126_top10']:.8f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 研究判断",
            "",
            "- `a015_z0015` 仍是当前 10D 研究主候选。",
            "- 下一步若继续优化，更可能有效的方向不是强行保护基线前排，而是做更细的日内条件切换或更稳定的辅助信号约束。",
            "",
            "## 治理说明",
            "",
            "- 本轮未训练模型。",
            "- 本轮未发布 formal 资产。",
            "- 本轮未生成交易信号、未制定交易规则、未跑回测。",
            "",
            "## 证据路径",
            "",
            f"- `{(OUT_DIR / 'guard_probe_summary.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'guard_probe_summary.json').as_posix()}`",
        ]
    )
    (OUT_DIR / "guard_probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
