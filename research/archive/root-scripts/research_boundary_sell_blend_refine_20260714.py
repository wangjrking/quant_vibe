from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "boundary_sell_blend_refine"
LOG_DIR = REPORT_DIR / "logs" / "boundary_sell_blend_refine"
OUT_CSV = REPORT_DIR / "boundary_sell_blend_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "boundary_sell_blend_refine_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


KEYS = ["signal_date", "buy_date", "stock_code"]

HIGH_ANNUAL_SOURCES = {
    "sell_h3": REPORT_DIR
    / "signals"
    / "top3_sell_frequency_refine"
    / "sell_mf_t2_sc1200_cap100_h3_exit098_cont102.csv",
    "sell_h1": REPORT_DIR / "signals" / "top3_sell_frequency_refine" / "sell_mf_t2_sc1200_cap100_h1_base.csv",
    "hps": REPORT_DIR / "signals" / "highsharpe_position_scale" / "hps_sc092_x1040_cap100.csv",
}

HIGH_SHARPE_SOURCES = {
    "ss092": REPORT_DIR
    / "signals"
    / "top3_sell_scale_refine"
    / "ss_mf_t2_sc1200_cap100_h3e98c102_sc092.csv",
    "inv120": REPORT_DIR / "signals" / "target_hit_inverse_refine" / "inv_mf092_l120_m110_h098.csv",
}


CASES = [
    {"annual": "sell_h3", "sharpe": "ss092", "alpha": 0.55, "scale": 1.020, "cap": 1.0},
    {"annual": "sell_h3", "sharpe": "ss092", "alpha": 0.60, "scale": 1.010, "cap": 1.0},
    {"annual": "sell_h3", "sharpe": "ss092", "alpha": 0.65, "scale": 1.000, "cap": 1.0},
    {"annual": "sell_h3", "sharpe": "ss092", "alpha": 0.70, "scale": 0.990, "cap": 1.0},
    {"annual": "sell_h3", "sharpe": "inv120", "alpha": 0.55, "scale": 1.020, "cap": 1.0},
    {"annual": "sell_h3", "sharpe": "inv120", "alpha": 0.65, "scale": 1.000, "cap": 1.0},
    {"annual": "sell_h1", "sharpe": "ss092", "alpha": 0.60, "scale": 1.030, "cap": 1.0},
    {"annual": "sell_h1", "sharpe": "ss092", "alpha": 0.70, "scale": 1.015, "cap": 1.0},
    {"annual": "sell_h1", "sharpe": "inv120", "alpha": 0.60, "scale": 1.030, "cap": 1.0},
    {"annual": "sell_h1", "sharpe": "inv120", "alpha": 0.70, "scale": 1.015, "cap": 1.0},
    {"annual": "hps", "sharpe": "ss092", "alpha": 0.75, "scale": 1.025, "cap": 1.0},
    {"annual": "hps", "sharpe": "inv120", "alpha": 0.75, "scale": 1.025, "cap": 1.0},
]


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})


def write_variant(case: dict, annual_df: pd.DataFrame, sharpe_df: pd.DataFrame) -> dict:
    annual_name = case["annual"]
    sharpe_name = case["sharpe"]
    alpha = float(case["alpha"])
    scale = float(case["scale"])
    cap = float(case["cap"])
    name = f"bsb_{annual_name}_{sharpe_name}_a{int(alpha * 100):02d}_sc{int(scale * 1000):04d}"

    right = sharpe_df[KEYS + ["target_pct"]].rename(columns={"target_pct": "target_sharpe"})
    df = annual_df.merge(right, on=KEYS, how="left")
    missing = int(df["target_sharpe"].isna().sum())
    if missing:
        raise RuntimeError(f"{name} missing paired target rows: {missing}")

    target_annual = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    target_sharpe = pd.to_numeric(df["target_sharpe"], errors="coerce").fillna(0.0)
    blended = (alpha * target_annual + (1.0 - alpha) * target_sharpe) * scale
    daily_sum = blended.groupby(df["buy_date"]).transform("sum")
    cap_scale = (cap / daily_sum).clip(upper=1.0)

    df["target_pct"] = blended * cap_scale
    df["strategy_variant"] = name
    df["filter_name"] = name
    df["blend_alpha_high_annual"] = alpha
    df["blend_scale"] = scale
    df["blend_high_annual_source"] = annual_name
    df["blend_high_sharpe_source"] = sharpe_name
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df = df.drop(columns=["target_sharpe"])

    out = SIGNAL_DIR / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": name,
        "annual_source": annual_name,
        "sharpe_source": sharpe_name,
        "alpha_high_annual": alpha,
        "scale": scale,
        "cap": cap,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
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

    annual_sources = {key: _load(path) for key, path in HIGH_ANNUAL_SOURCES.items()}
    sharpe_sources = {key: _load(path) for key, path in HIGH_SHARPE_SOURCES.items()}

    manifest = [
        write_variant(case, annual_sources[case["annual"]], sharpe_sources[case["sharpe"]])
        for case in CASES
    ]

    results = []
    for row in manifest:
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

    frame = pd.DataFrame(results)
    frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig"
    )
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    print(
        json.dumps(
            {"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results), "target_hits": int(len(hits))},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
