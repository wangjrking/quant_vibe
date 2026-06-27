from __future__ import annotations

import csv
import json
from pathlib import Path

import research_formal_5d10d_attribution_scale_refine_20260622 as base


REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_gap_threshold_refine_20260622"
)


def _rules(good_threshold: float, good_scale: float, bad_scale: float) -> list[dict]:
    return [
        {"field": "pred_gap", "op": "lt", "threshold": good_threshold, "scale": good_scale},
        {"field": "pred_gap", "op": "between", "lower": 0.05, "upper": 0.10, "scale": bad_scale},
    ]


def _variant(name: str, good_threshold: float, good_scale: float, bad_scale: float) -> dict:
    return {
        "name": name,
        "rules": _rules(good_threshold, good_scale, bad_scale),
        "normalize_day": True,
        "min_day_target": None,
        "max_positions": 5,
        "holding_days": 7,
        "max_holding_days": 10,
        "extra_env": {},
    }


VARIANTS = [
    _variant(
        f"gap_t{str(threshold).replace('.', 'p')}_g{str(good).replace('.', 'p')}_b{str(bad).replace('.', 'p')}",
        threshold,
        good,
        bad,
    )
    for threshold in [0.015, 0.020, 0.025, 0.030]
    for good in [1.08, 1.12, 1.16]
    for bad in [0.82, 0.88, 0.94]
]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return base._to_float(row.get(key), float("-inf"))


def _write_report(rows: list[dict]) -> None:
    hits = [
        row
        for row in rows
        if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
    ]
    best_objective = max(
        rows,
        key=lambda row: min(
            _metric(row, "annual") / 3.0,
            _metric(row, "sharpe") / 4.0,
            _metric(row, "avg_invested_pct") / 0.80,
        ),
    )
    best_annual = max(rows, key=lambda row: _metric(row, "annual"))
    best_sharpe = max(rows, key=lambda row: _metric(row, "sharpe"))
    report = f"""# 5D10D pred_gap 阈值调参结论

## 当前结论

本轮只使用标准链路 formal L4 的 5D+10D 资产，不使用 1D、不使用行业限制、不使用月份或日期过滤。候选只调整 `pred_gap` 的加减仓阈值和缩放强度，通过掘金回测验证。

- 候选数量：{len(rows)}
- 达到 `年化 >= 3.0`、`夏普 >= 4.0`、`平均持仓 >= 0.80` 的候选数量：{len(hits)}
- 是否建议替换当前 L5：{"是" if hits else "否"}

## 最优候选

- 综合最接近目标：`{best_objective["name"]}`，年化 `{_metric(best_objective, "annual"):.6f}`，夏普 `{_metric(best_objective, "sharpe"):.6f}`，平均持仓 `{_metric(best_objective, "avg_invested_pct"):.6f}`。
- 最高年化：`{best_annual["name"]}`，年化 `{_metric(best_annual, "annual"):.6f}`，夏普 `{_metric(best_annual, "sharpe"):.6f}`，平均持仓 `{_metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高夏普：`{best_sharpe["name"]}`，年化 `{_metric(best_sharpe, "annual"):.6f}`，夏普 `{_metric(best_sharpe, "sharpe"):.6f}`，平均持仓 `{_metric(best_sharpe, "avg_invested_pct"):.6f}`。

## 证据路径

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 信号目录：`{REPORT_DIR / "signals"}`
- 掘金日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "5D10D_gap阈值调参结论.md").write_text(report, encoding="utf-8")


def main() -> int:
    base.REPORT_DIR = REPORT_DIR
    (REPORT_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    base_rows = base._load_rows(base.BASE_SIGNAL)
    keys = {(str(row.get("signal_date")), str(row.get("stock_code"))) for row in base_rows}
    fusion_features = base._load_fusion_features(keys)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        signal_file = REPORT_DIR / "signals" / f"{variant['name']}.csv"
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        base._write_variant_signal(variant, signal_file, base_rows, fusion_features)
        returncode = base._run_backtest(variant, signal_file, log_file)
        indicator = base._extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "rules_json": json.dumps(variant["rules"], ensure_ascii=False, sort_keys=True),
            "returncode": returncode,
            "signal_file": str(signal_file),
            "log_file": str(log_file),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(base._signal_stats(signal_file))
        row.update(base._exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} "
            f"avg_inv={row.get('avg_invested_pct')}"
        )
    _write_rows(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual"), reverse=True))
    _write_rows(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_rows(
        REPORT_DIR / "summary_target_hits.csv",
        [
            row
            for row in results
            if _metric(row, "annual") >= 3.0 and _metric(row, "sharpe") >= 4.0 and _metric(row, "avg_invested_pct") >= 0.80
        ],
    )
    _write_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
