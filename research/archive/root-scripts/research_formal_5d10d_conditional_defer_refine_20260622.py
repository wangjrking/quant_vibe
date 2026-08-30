from __future__ import annotations

import csv
import json
from pathlib import Path

import research_formal_5d10d_risk_overlay_refine_20260622 as base


REPORT_DIR = (
    base.DATA_DIR
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_5d10d_conditional_defer_refine_20260622"
)


def _variant(source: str, max_hold: int, min_return: float | None, extra: dict[str, str] | None = None) -> dict:
    suffix = f"defer{max_hold}_minret{'none' if min_return is None else str(min_return).replace('-', 'n').replace('.', 'p')}"
    env = {
        "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "1",
        "GM_NO_SIGNAL_MAX_HOLDING_DAYS": str(max_hold),
    }
    if min_return is not None:
        env["GM_DEFER_EXIT_MIN_POSITION_RETURN"] = str(min_return)
    if extra:
        env.update(extra)
        suffix += "_x" + str(len(extra))
    return {
        "name": f"{source}_{suffix}",
        "source": source,
        "signal_file": base.SOURCES[source],
        "max_positions": 5,
        "holding_days": 7,
        "max_holding_days": 10,
        "target_position_pct": 0.28,
        "extra_env": env,
    }


VARIANTS = []
for source in ["current_gap", "goal_rank5_23", "weak_rw2422"]:
    for max_hold in [5, 8, 12, 20]:
        for min_return in [-0.05, -0.02, 0.0, 0.03, 0.06]:
            VARIANTS.append(_variant(source, max_hold, min_return))
    for min_return in [0.0, 0.03]:
        VARIANTS.append(
            _variant(
                source,
                12,
                min_return,
                {"GM_EQUITY_DD_SOFT_TRIGGER": "0.15", "GM_EQUITY_DD_HARD_TRIGGER": "0.25", "GM_EQUITY_DD_SOFT_SCALE": "0.95", "GM_EQUITY_DD_HARD_SCALE": "0.85"},
            )
        )


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
        key=lambda row: min(_metric(row, "annual") / 3.0, _metric(row, "sharpe") / 4.0, _metric(row, "avg_invested_pct") / 0.80),
    )
    best_annual = max(rows, key=lambda row: _metric(row, "annual"))
    best_sharpe = max(rows, key=lambda row: _metric(row, "sharpe"))
    report = f"""# 5D10D 条件化空信号延迟退出调参结论

## 当前结论

本轮只使用 formal L4 的 5D 与 10D 研究/生产信号源；新增执行层开关只控制空信号日是否延迟卖出，不使用行业限制、月份过滤或日期过滤。所有结论以掘金回测日志为准。

- 候选数量：{len(rows)}
- 达到 `年化 >= 3.0`、`夏普 >= 4.0`、`平均持仓 >= 0.80` 的候选数量：{len(hits)}
- 是否建议替换当前 L5：{"是" if hits else "否"}

## 最优候选

- 综合最接近目标：`{best_objective["name"]}`，年化 `{_metric(best_objective, "annual"):.6f}`，夏普 `{_metric(best_objective, "sharpe"):.6f}`，平均持仓 `{_metric(best_objective, "avg_invested_pct"):.6f}`。
- 最高年化：`{best_annual["name"]}`，年化 `{_metric(best_annual, "annual"):.6f}`，夏普 `{_metric(best_annual, "sharpe"):.6f}`，平均持仓 `{_metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高夏普：`{best_sharpe["name"]}`，年化 `{_metric(best_sharpe, "annual"):.6f}`，夏普 `{_metric(best_sharpe, "sharpe"):.6f}`，平均持仓 `{_metric(best_sharpe, "avg_invested_pct"):.6f}`。

## 证据路径

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 掘金日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "5D10D条件化空信号延迟退出调参结论.md").write_text(report, encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for index, variant in enumerate(VARIANTS, start=1):
        log_file = REPORT_DIR / "logs" / f"{variant['name']}.log"
        returncode = base._run_backtest(variant, log_file)
        indicator = base._extract_indicator(log_file) or {}
        row = {
            "name": variant["name"],
            "source": variant["source"],
            "returncode": returncode,
            "signal_file": str(variant["signal_file"]),
            "log_file": str(log_file),
            "extra_env_json": json.dumps(variant.get("extra_env", {}), ensure_ascii=False, sort_keys=True),
            "annual": indicator.get("pnl_ratio_annual", indicator.get("annual_return")),
            "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
            "max_drawdown": indicator.get("max_drawdown"),
            "win_ratio": indicator.get("win_ratio"),
            "open_count": indicator.get("open_count"),
            "close_count": indicator.get("close_count"),
        }
        row.update(base._signal_stats(variant["signal_file"]))
        row.update(base._exposure_stats(log_file))
        results.append(row)
        _write_rows(REPORT_DIR / "summary.csv", results)
        print(
            f"[{index}/{len(VARIANTS)}] {variant['name']} "
            f"annual={row.get('annual')} sharpe={row.get('sharpe')} avg={row.get('avg_invested_pct')}"
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
