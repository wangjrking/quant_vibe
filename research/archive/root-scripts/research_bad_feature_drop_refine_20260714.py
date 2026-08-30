from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "bad_feature_drop_refine"
LOG_DIR = REPORT_DIR / "logs" / "bad_feature_drop_refine"
OUT_CSV = REPORT_DIR / "bad_feature_drop_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "bad_feature_drop_refine_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = {
    "ss092": REPORT_DIR / "signals" / "top3_sell_scale_refine" / "ss_mf_t2_sc1200_cap100_h3e98c102_sc092.csv",
    "hps": REPORT_DIR / "signals" / "highsharpe_position_scale" / "hps_sc092_x1040_cap100.csv",
    "ms_h3": REPORT_DIR / "signals" / "top3_sell_frequency_refine" / "sell_ms_t3_sc110_cap91_h3_exit098_cont102.csv",
}


CASES = [
    {"name": "bfd_ss092_drop_shallow_scale120", "source": "ss092", "drop": "shallow", "scale": 1.20},
    {"name": "bfd_ss092_drop_posgap_scale120", "source": "ss092", "drop": "posgap", "scale": 1.20},
    {"name": "bfd_ss092_drop_atr10_scale120", "source": "ss092", "drop": "atr10", "scale": 1.20},
    {"name": "bfd_ss092_drop_anybad_scale122", "source": "ss092", "drop": "anybad", "scale": 1.22},
    {"name": "bfd_hps_drop_shallow_scale105", "source": "hps", "drop": "shallow", "scale": 1.05},
    {"name": "bfd_hps_drop_posgap_scale105", "source": "hps", "drop": "posgap", "scale": 1.05},
    {"name": "bfd_hps_drop_anybad_scale110", "source": "hps", "drop": "anybad", "scale": 1.10},
    {"name": "bfd_msh3_drop_shallow_scale116", "source": "ms_h3", "drop": "shallow", "scale": 1.16},
    {"name": "bfd_msh3_drop_posgap_scale116", "source": "ms_h3", "drop": "posgap", "scale": 1.16},
    {"name": "bfd_msh3_drop_anybad_scale122", "source": "ms_h3", "drop": "anybad", "scale": 1.22},
]


def n(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def drop_mask(df: pd.DataFrame, kind: str) -> pd.Series:
    pct = n(df, "signal_pct_chg_raw")
    atr = n(df, "atr_qfq", 99.0)
    gap = n(df, "buy_open_gap_raw_pct", 999.0)
    if "buy_open_gap_pct" in df.columns:
        gap = gap.where(gap != 999.0, n(df, "buy_open_gap_pct", 999.0))
    shallow = pct > -2.5
    posgap = gap > 0.75
    atr10 = atr >= 10.0
    if kind == "shallow":
        return shallow
    if kind == "posgap":
        return posgap
    if kind == "atr10":
        return atr10
    if kind == "anybad":
        return shallow | posgap | atr10
    raise ValueError(kind)


def write_variant(case: dict) -> dict:
    src = SOURCES[case["source"]]
    df = pd.read_csv(src, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    drop = drop_mask(df, case["drop"])
    kept = df.loc[~drop].copy()
    kept["target_pct"] = n(kept, "target_pct") * float(case["scale"])
    daily_sum = kept.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    kept["target_pct"] = kept["target_pct"] * cap_scale
    kept["strategy_variant"] = case["name"]
    kept["filter_name"] = case["name"]
    kept["drop_rule"] = case["drop"]
    kept["daily_target_sum_after_cap"] = kept.groupby("buy_date")["target_pct"].transform("sum")
    out = SIGNAL_DIR / f"{case['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["name"],
        "source": case["source"],
        "source_file": str(src),
        "signal_file": str(out),
        "drop_rule": case["drop"],
        "scale": float(case["scale"]),
        "source_rows": int(len(df)),
        "rows": int(len(kept)),
        "dropped_rows": int(drop.sum()),
        "buy_days": int(kept["buy_date"].nunique()) if len(kept) else 0,
        "stock_count": int(kept["stock_code"].nunique()) if len(kept) else 0,
        "mean_daily_target_sum": float(kept.groupby("buy_date")["target_pct"].sum().mean()) if len(kept) else 0.0,
        "max_daily_target_sum": float(kept.groupby("buy_date")["target_pct"].sum().max()) if len(kept) else 0.0,
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    return base_mod.run_juejin(row)


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [write_variant(case) for case in CASES]
    results = []
    for row in rows:
        result = run_or_parse(row)
        results.append(result)
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "pnl_ratio_annual": result.get("pnl_ratio_annual"),
                    "sharp_ratio": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
