from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_5d_candidate_complementarity_20260630"

BASE_NAME = "current_bestset_direct_guard_v1"
BASE_DAILY = (
    DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_current_bestset_direct_guard_v1_20260630"
    / "standard_eval"
    / "executable_5d_open_return_four_year_current_bestset_direct_guard_v1_20260630_daily_eval.csv"
)

CANDIDATES = {
    "bucketed_blend_v5": DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_bucketed_blend_v5_20260629"
    / "standard_eval"
    / "executable_5d_open_return_four_year_bucketed_blend_v5_20260629_daily_eval.csv",
    "active_clear_gate_hybrid": DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_active_clear_gate_hybrid_20260628"
    / "standard_eval"
    / "executable_5d_open_return_four_year_active_clear_gate_hybrid_20260628_daily_eval.csv",
    "front_rank_blend_v3": DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_front_rank_blend_v3_20260629"
    / "standard_eval"
    / "executable_5d_open_return_four_year_front_rank_blend_v3_20260629_daily_eval.csv",
    "tri_source_microblend_v7": DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_tri_source_microblend_v7_20260629"
    / "standard_eval"
    / "executable_5d_open_return_four_year_tri_source_microblend_v7_20260629_daily_eval.csv",
    "active_clear_microblend_v6": DATA_DIR
    / "reports"
    / "model_agent_four_year_5d_active_clear_microblend_v6_20260629"
    / "standard_eval"
    / "executable_5d_open_return_four_year_active_clear_microblend_v6_20260629_daily_eval.csv",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def load_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame.sort_values("trade_date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def analyze_pair(base: pd.DataFrame, cand: pd.DataFrame, name: str) -> dict[str, object]:
    joined = base.merge(cand, on="trade_date", suffixes=("_base", "_cand"))
    recent63 = joined.tail(63).copy()
    recent20 = joined.tail(20).copy()

    def ratio(frame: pd.DataFrame, expr: pd.Series) -> float:
        return float(expr.mean()) if len(frame) else 0.0

    top5_delta = joined["top5_cand"] - joined["top5_base"]
    rank_ic_delta = joined["rank_ic_cand"] - joined["rank_ic_base"]
    top1_delta = joined["top1_cand"] - joined["top1_base"]
    return {
        "candidate": name,
        "joined_days": int(len(joined)),
        "full_top5_delta": float(top5_delta.mean()),
        "full_top1_delta": float(top1_delta.mean()),
        "full_rank_ic_delta": float(rank_ic_delta.mean()),
        "positive_top5_day_ratio": ratio(joined, top5_delta > 0.0),
        "positive_rankic_day_ratio": ratio(joined, rank_ic_delta > 0.0),
        "salvage_top5_day_ratio": ratio(joined, (joined["top5_cand"] > joined["top5_base"]) & (joined["top5_base"] < 0.0)),
        "salvage_rankic_day_ratio": ratio(joined, (joined["rank_ic_cand"] > joined["rank_ic_base"]) & (joined["rank_ic_base"] < 0.0)),
        "recent63_top5_delta": float((recent63["top5_cand"] - recent63["top5_base"]).mean()),
        "recent63_rank_ic_delta": float((recent63["rank_ic_cand"] - recent63["rank_ic_base"]).mean()),
        "recent20_top5_delta": float((recent20["top5_cand"] - recent20["top5_base"]).mean()),
        "recent20_rank_ic_delta": float((recent20["rank_ic_cand"] - recent20["rank_ic_base"]).mean()),
        "recent63_positive_top5_day_ratio": ratio(recent63, (recent63["top5_cand"] - recent63["top5_base"]) > 0.0),
        "recent20_positive_top5_day_ratio": ratio(recent20, (recent20["top5_cand"] - recent20["top5_base"]) > 0.0),
    }


def choose_route(rows: list[dict[str, object]]) -> dict[str, object]:
    ranked = sorted(
        rows,
        key=lambda item: (
            item["recent20_top5_delta"],
            item["recent63_top5_delta"],
            item["salvage_top5_day_ratio"],
            item["positive_top5_day_ratio"],
        ),
        reverse=True,
    )
    best = ranked[0]
    promising = [
        row
        for row in ranked
        if float(row["recent20_top5_delta"]) > 0.0
        and float(row["recent63_top5_delta"]) > 0.0
        and float(row["salvage_top5_day_ratio"]) >= 0.15
    ]
    if promising:
        return {
            "next_route": "cross_source_rank_blend_or_date_gate",
            "reason": "存在近期窗口为正且对 base 负日有明显补救能力的候选，优先尝试跨源融合。",
            "best_candidate": best,
            "promising_candidates": promising,
        }
    return {
        "next_route": "new_training_family",
        "reason": "现有候选虽然局部有增益，但互补补救能力不够，继续做融合的性价比偏低，应切到新的训练族。",
        "best_candidate": best,
        "promising_candidates": [],
    }


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    decision = payload["decision"]
    best = decision["best_candidate"]
    lines = [
        "# 四年 5D 候选互补性分析",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 结论",
        "",
        f"- 下一条优先路线：`{decision['next_route']}`",
        f"- 原因：{decision['reason']}",
        f"- 当前最值得继续观察的互补候选：`{best['candidate']}`",
        f"- 该候选的 `recent20_top5_delta={best['recent20_top5_delta']:.6f}`，`recent63_top5_delta={best['recent63_top5_delta']:.6f}`，`salvage_top5_day_ratio={best['salvage_top5_day_ratio']:.4f}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_daily(BASE_DAILY)
    rows = [analyze_pair(base, load_daily(path), name) for name, path in CANDIDATES.items()]
    decision = choose_route(rows)
    payload = {
        "generated_at": now_iso(),
        "scope": "analyze_four_year_5d_candidate_complementarity",
        "base_asset": BASE_NAME,
        "base_daily": str(BASE_DAILY),
        "candidates": rows,
        "decision": decision,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "candidate_complementarity_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(REPORT_DIR / "candidate_complementarity_summary.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
