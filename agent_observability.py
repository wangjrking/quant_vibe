from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from project_paths import PROJECT_ROOT, resolve_data_dir


SH_TZ = ZoneInfo("Asia/Shanghai")

AGENT_LABELS = {
    "audit-agent": "审计智能体",
    "architect-agent": "架构师智能体",
    "commander-agent": "指挥官智能体",
    "orchestrator-agent": "主管智能体",
    "data-ingestion-agent": "数据接入智能体",
    "data-integration-agent": "数据整合智能体",
    "deploy-agent": "部署智能体",
    "factor-agent": "因子智能体",
    "model-agent": "模型智能体",
    "research-agent": "投研智能体",
    "strategy-agent": "策略智能体",
    "trading-agent": "交易智能体",
}

AGENT_ORDER = [
    "data-ingestion-agent",
    "data-integration-agent",
    "factor-agent",
    "model-agent",
    "strategy-agent",
    "trading-agent",
    "audit-agent",
    "architect-agent",
    "research-agent",
    "commander-agent",
    "orchestrator-agent",
    "deploy-agent",
]

TOP_LEVEL_DATA_ASSETS = {
    "daily_data.parquet",
    "daily_index_data.parquet",
    "stk_factor.parquet",
    "moneyflow.parquet",
    "limit_list_data.parquet",
    "cyq_perf.parquet",
    "adj_factor.parquet",
    "stock_st.parquet",
    "index_daily.parquet",
    "stock_basic_data.parquet",
    "finan_data_season.parquet",
    "finan_data_year.parquet",
    "top_list.parquet",
    "ths_hot.parquet",
    "dc_hot.parquet",
    "STOCK_DAILY_DATA.db",
}

TEXT_EXTENSIONS = {".md", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml"}
OUTPUT_IGNORE_EXTENSIONS = {".log", ".png", ".jpg", ".jpeg", ".pdf", ".docx", ".sqlite", ".db-journal"}


@dataclass(frozen=True)
class OutputEvent:
    snapshot_date: str
    agent_id: str
    relative_path: str
    absolute_path: str
    output_kind: str
    is_formal: int
    byte_size: int
    source: str


def project_root_from_path(project_root: str | Path | None = None) -> Path:
    if project_root not in (None, ""):
        return Path(project_root).resolve()
    return PROJECT_ROOT.parent.parent.resolve()


def observability_runtime_dir(data_dir: Path) -> Path:
    return data_dir / "runtime" / "agent_observability"


def observability_reports_dir(data_dir: Path) -> Path:
    return data_dir / "reports" / "agent_observability"


def observability_db_path(data_dir: Path) -> Path:
    return observability_runtime_dir(data_dir) / "agent_observability.db"


def collaboration_requests_path(data_dir: Path) -> Path:
    return data_dir / "runtime" / "agent_memory" / "collaboration_requests.jsonl"


def exact_token_usage_dir(data_dir: Path) -> Path:
    return observability_runtime_dir(data_dir) / "token_usage"


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def to_snapshot_date(value: datetime) -> str:
    return value.astimezone(SH_TZ).strftime("%Y%m%d")


def parse_any_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").replace(tzinfo=SH_TZ)
    normalized = text.replace("Z", "+00:00")
    if "." in normalized:
        head, tail = normalized.split(".", 1)
        frac = tail
        zone = ""
        for sep in ("+", "-"):
            if sep in tail:
                frac, zone_tail = tail.split(sep, 1)
                zone = sep + zone_tail
                break
        frac = frac[:6]
        normalized = f"{head}.{frac}{zone}"
    return datetime.fromisoformat(normalized)


def approx_token_count(text: str) -> int:
    clean = str(text or "")
    if not clean:
        return 0
    return max(1, math.ceil(len(clean) / 4))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def resolve_evidence_path(raw: str | Path, project_root: Path) -> Path | None:
    try:
        path = Path(str(raw))
    except Exception:
        return None
    candidates = [path]
    if not path.is_absolute():
        candidates = [project_root / path, project_root / "quant" / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def build_evidence_agent_index(rows: Iterable[dict[str, Any]], project_root: Path) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        agent_id = str(row.get("from_agent") or "").strip()
        if not agent_id:
            continue
        for evidence in row.get("evidence", []) or []:
            path = resolve_evidence_path(evidence, project_root)
            if path is not None:
                index[normalize_path(path)].add(agent_id)
    return index


def infer_agent_from_path(path: Path, evidence_index: dict[str, set[str]]) -> str | None:
    normalized = normalize_path(path)
    matched = evidence_index.get(normalized)
    if matched:
        return sorted(matched)[0]

    text = normalized
    name = path.name.lower()

    if "/raw_table_dbs/" in text or name in {item.lower() for item in TOP_LEVEL_DATA_ASSETS if item.endswith(".parquet")}:
        return "data-ingestion-agent"
    if name == "stock_daily_data.db" or "stock_daily" in name or "l2_" in name:
        return "data-integration-agent"
    if "/production_factor_parts/" in text or "/prediction_label_parts/" in text or "/raw_factor_by_stock_parts/" in text:
        return "factor-agent"
    if "/model_predictions/" in text or "/prediction_manifests/" in text:
        return "model-agent"
    if "/production_signals/" in text or "/strategy_library/production/" in text:
        return "strategy-agent"
    if "audit" in name:
        return "audit-agent"
    if "architect" in name:
        return "architect-agent"
    if "research" in name:
        return "research-agent"
    if "strategy" in name or "signal" in name:
        return "strategy-agent"
    if "factor" in name or "gtja" in name or "industry_encode" in name:
        return "factor-agent"
    if "raw_data" in name or "tushare" in name or "adj_factor" in name or "moneyflow" in name:
        return "data-ingestion-agent"
    if "model" in name or "prediction" in name:
        return "model-agent"
    return None


def classify_output(path: Path, project_root: Path) -> tuple[str, str, int]:
    relative = path.resolve().relative_to(project_root.resolve())
    parts = [part.lower() for part in relative.parts]
    suffix = path.suffix.lower()
    is_formal = 0
    output_kind = "other"
    display_relative = str(relative).replace("\\", "/")

    if "production_factor_parts" in parts:
        display_relative = "quant/data_file/production_factor_parts/"
        output_kind = "factor_parts"
        is_formal = 1
    elif "prediction_label_parts" in parts:
        display_relative = "quant/data_file/prediction_label_parts/"
        output_kind = "label_parts"
        is_formal = 1
    elif "raw_factor_by_stock_parts" in parts:
        display_relative = "quant/data_file/raw_factor_by_stock_parts/"
        output_kind = "raw_factor_parts"
    elif "raw_table_dbs" in parts:
        output_kind = "raw_split_db"
        display_relative = "/".join(relative.parts[-2:])
    elif path.name in TOP_LEVEL_DATA_ASSETS:
        output_kind = "top_level_data_asset"
        is_formal = 1 if path.name == "STOCK_DAILY_DATA.db" else 0
    elif "model_predictions" in parts:
        output_kind = "model_prediction_asset"
        if path.name == "prediction_manifest.json":
            display_relative = "/".join(relative.parts[-2:])
        is_formal = 1 if "approved_for_l5" in safe_read_text(path) else 0
    elif "prediction_manifests" in parts and suffix == ".json":
        output_kind = "formal_manifest"
        text = safe_read_text(path)
        is_formal = 1 if "approved_for_l5" in text else 0
    elif "production_signals" in parts:
        output_kind = "production_signal"
        is_formal = 1
    elif "strategy_library" in parts and "production" in parts:
        output_kind = "strategy_archive"
        is_formal = 1
        try:
            production_idx = parts.index("production")
            display_relative = "/".join(relative.parts[: production_idx + 3])
        except ValueError:
            pass
    elif "reports" in parts:
        if suffix == ".md":
            output_kind = "report_md"
        elif suffix == ".json":
            output_kind = "report_json"
        elif suffix == ".csv":
            output_kind = "report_csv"
        else:
            output_kind = "report_other"

    return display_relative, output_kind, is_formal


def safe_read_text(path: Path) -> str:
    if not path.exists() or path.suffix.lower() not in TEXT_EXTENSIONS:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            return ""


def file_snapshot_date(path: Path) -> str:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=SH_TZ)
    return to_snapshot_date(mtime)


def collect_candidate_paths(project_root: Path, data_dir: Path) -> list[Path]:
    roots = [
        data_dir / "reports",
        data_dir / "model_predictions",
        data_dir / "production_signals",
        data_dir / "runtime" / "agent_memory",
        data_dir / "raw_table_dbs",
        data_dir / "production_factor_parts",
        data_dir / "prediction_label_parts",
        data_dir / "raw_factor_by_stock_parts",
        project_root / "quant" / "main" / "config" / "prediction_manifests",
        project_root / "quant" / "main" / "strategy_library" / "production",
    ]
    candidates = [data_dir / name for name in TOP_LEVEL_DATA_ASSETS]
    paths: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            paths.append(candidate)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if path.suffix.lower() in OUTPUT_IGNORE_EXTENSIONS:
                continue
            paths.append(path)
    return paths


def compress_output_events(events: Iterable[OutputEvent]) -> list[OutputEvent]:
    grouped: dict[tuple[str, str, str], OutputEvent] = {}
    for event in events:
        key = (event.agent_id, event.relative_path, event.output_kind)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = event
            continue
        grouped[key] = OutputEvent(
            snapshot_date=event.snapshot_date,
            agent_id=event.agent_id,
            relative_path=event.relative_path,
            absolute_path=event.absolute_path,
            output_kind=event.output_kind,
            is_formal=max(existing.is_formal, event.is_formal),
            byte_size=max(existing.byte_size, event.byte_size),
            source=existing.source,
        )
    return sorted(grouped.values(), key=lambda item: (item.agent_id, item.relative_path, item.output_kind))


def collect_output_events(
    snapshot_date: str,
    project_root: Path,
    data_dir: Path,
    evidence_index: dict[str, set[str]],
) -> list[OutputEvent]:
    events: list[OutputEvent] = []
    for row in read_jsonl(collaboration_requests_path(data_dir)):
        agent_id = str(row.get("from_agent") or "").strip()
        parsed_time = parse_any_datetime(row.get("time"))
        if not agent_id or parsed_time is None or to_snapshot_date(parsed_time) != snapshot_date:
            continue
        for evidence in row.get("evidence", []) or []:
            path = resolve_evidence_path(evidence, project_root)
            if path is None or not path.exists():
                continue
            try:
                relative_path, output_kind, is_formal = classify_output(path, project_root)
            except ValueError:
                continue
            if output_kind == "other":
                continue
            events.append(
                OutputEvent(
                    snapshot_date=snapshot_date,
                    agent_id=agent_id,
                    relative_path=relative_path,
                    absolute_path=str(path.resolve()),
                    output_kind=output_kind,
                    is_formal=is_formal,
                    byte_size=path.stat().st_size,
                    source="collaboration_evidence",
                )
            )
    for path in collect_candidate_paths(project_root, data_dir):
        if file_snapshot_date(path) != snapshot_date:
            continue
        agent_id = infer_agent_from_path(path, evidence_index)
        if not agent_id:
            continue
        try:
            relative_path, output_kind, is_formal = classify_output(path, project_root)
        except ValueError:
            continue
        if output_kind == "other":
            continue
        events.append(
            OutputEvent(
                snapshot_date=snapshot_date,
                agent_id=agent_id,
                relative_path=relative_path,
                absolute_path=str(path.resolve()),
                output_kind=output_kind,
                is_formal=is_formal,
                byte_size=path.stat().st_size,
                source="filesystem_scan",
            )
        )
    return compress_output_events(events)


def collect_exact_token_usage(snapshot_date: str, data_dir: Path) -> dict[str, dict[str, int]]:
    usage_dir = exact_token_usage_dir(data_dir)
    aggregated: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    if not usage_dir.exists():
        return {}
    for path in usage_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        agent_id = str(payload.get("agent_id") or "").strip()
        if not agent_id:
            continue
        payload_date = str(payload.get("snapshot_date") or "")
        if payload_date and payload_date != snapshot_date:
            continue
        prompt_tokens = int(payload.get("prompt_tokens") or 0)
        completion_tokens = int(payload.get("completion_tokens") or 0)
        total_tokens = int(payload.get("total_tokens") or (prompt_tokens + completion_tokens))
        aggregated[agent_id]["prompt_tokens"] += prompt_tokens
        aggregated[agent_id]["completion_tokens"] += completion_tokens
        aggregated[agent_id]["total_tokens"] += total_tokens
    return dict(aggregated)


def estimate_agent_tokens(
    agent_id: str,
    snapshot_date: str,
    collaboration_rows: list[dict[str, Any]],
    output_events: list[OutputEvent],
) -> tuple[int | None, str]:
    token_total = 0
    bases: list[str] = []

    relevant_rows = [
        row
        for row in collaboration_rows
        if str(row.get("from_agent") or "").strip() == agent_id
        and to_snapshot_date(parse_any_datetime(row.get("time")) or datetime.now(tz=SH_TZ)) == snapshot_date
    ]
    if relevant_rows:
        payload_parts: list[str] = []
        keep_fields = [
            "summary",
            "requested_action",
            "task_id",
            "layer",
            "type",
            "label",
        ]
        list_text_fields = [
            "model_side_notes",
            "model_side_limits",
            "current_l4_status",
        ]
        for row in relevant_rows:
            for field in keep_fields:
                value = row.get(field)
                if value not in (None, ""):
                    payload_parts.append(str(value))
            for field in list_text_fields:
                value = row.get(field)
                if isinstance(value, list):
                    payload_parts.extend(str(item) for item in value)
                elif isinstance(value, dict):
                    payload_parts.extend(f"{key}:{val}" for key, val in value.items())
        payload_text = "\n".join(payload_parts)
        token_total += approx_token_count(payload_text)
        bases.append("collaboration_requests")

    text_outputs = [
        event for event in output_events
        if event.agent_id == agent_id
        and Path(event.absolute_path).suffix.lower() in TEXT_EXTENSIONS
        and event.output_kind in {"report_md", "report_other", "formal_manifest", "strategy_archive"}
    ]
    if text_outputs:
        total_chars = 0
        for event in text_outputs:
            total_chars += len(safe_read_text(Path(event.absolute_path)))
        token_total += approx_token_count("x" * total_chars)
        bases.append("text_outputs")

    machine_outputs = [
        event
        for event in output_events
        if event.agent_id == agent_id and event not in text_outputs
    ]
    if machine_outputs:
        token_total += len(machine_outputs) * 120
        bases.append("machine_output_overhead")

    if token_total <= 0:
        return None, ""
    return token_total, " + ".join(bases)


def empty_agent_row(snapshot_date: str, agent_id: str) -> dict[str, Any]:
    return {
        "snapshot_date": snapshot_date,
        "agent_id": agent_id,
        "agent_label": AGENT_LABELS.get(agent_id, agent_id),
        "exact_prompt_tokens": None,
        "exact_completion_tokens": None,
        "exact_total_tokens": None,
        "estimated_total_tokens": None,
        "token_status": "none",
        "estimate_basis": "",
        "business_output_count": 0,
        "formal_output_count": 0,
        "report_output_count": 0,
        "activity_count": 0,
        "output_bytes": 0,
        "top_outputs": [],
    }


def build_daily_snapshot(
    *,
    snapshot_date: str,
    project_root: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = project_root_from_path(project_root)
    resolved_data_dir = resolve_data_dir(data_dir or (root / "quant" / "data_file"))

    collaboration_rows = read_jsonl(collaboration_requests_path(resolved_data_dir))
    evidence_index = build_evidence_agent_index(collaboration_rows, root)
    output_events = collect_output_events(snapshot_date, root, resolved_data_dir, evidence_index)
    exact_usage = collect_exact_token_usage(snapshot_date, resolved_data_dir)

    rows_by_agent: dict[str, dict[str, Any]] = {}
    for agent_id in AGENT_ORDER:
        rows_by_agent[agent_id] = empty_agent_row(snapshot_date, agent_id)

    for event in output_events:
        row = rows_by_agent.setdefault(event.agent_id, empty_agent_row(snapshot_date, event.agent_id))
        row["business_output_count"] += 1
        row["formal_output_count"] += int(event.is_formal)
        if event.output_kind.startswith("report_"):
            row["report_output_count"] += 1
        row["activity_count"] += 1
        row["output_bytes"] += int(event.byte_size)
        row["top_outputs"].append(
            {
                "relative_path": event.relative_path,
                "output_kind": event.output_kind,
                "is_formal": event.is_formal,
            }
        )

    for row in rows_by_agent.values():
        row["top_outputs"] = row["top_outputs"][:8]
        usage = exact_usage.get(row["agent_id"])
        if usage and usage.get("total_tokens", 0) > 0:
            row["exact_prompt_tokens"] = usage["prompt_tokens"]
            row["exact_completion_tokens"] = usage["completion_tokens"]
            row["exact_total_tokens"] = usage["total_tokens"]
            row["token_status"] = "exact"
            continue
        estimated_total, basis = estimate_agent_tokens(
            row["agent_id"],
            snapshot_date,
            collaboration_rows,
            output_events,
        )
        if estimated_total is not None:
            row["estimated_total_tokens"] = estimated_total
            row["estimate_basis"] = basis
            row["token_status"] = "estimated"

    agents = [row for row in rows_by_agent.values() if row["business_output_count"] > 0 or row["token_status"] != "none"]
    return {
        "snapshot_date": snapshot_date,
        "generated_at": datetime.now(tz=SH_TZ).isoformat(timespec="seconds"),
        "project_root": str(root),
        "data_dir": str(resolved_data_dir),
        "agents": agents,
        "outputs": [
            {
                "snapshot_date": event.snapshot_date,
                "agent_id": event.agent_id,
                "relative_path": event.relative_path,
                "absolute_path": event.absolute_path,
                "output_kind": event.output_kind,
                "is_formal": event.is_formal,
                "byte_size": event.byte_size,
                "source": event.source,
            }
            for event in output_events
        ],
    }


def ensure_db_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_daily_metrics (
            snapshot_date TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            agent_label TEXT NOT NULL,
            exact_prompt_tokens INTEGER,
            exact_completion_tokens INTEGER,
            exact_total_tokens INTEGER,
            estimated_total_tokens INTEGER,
            token_status TEXT NOT NULL,
            estimate_basis TEXT NOT NULL,
            business_output_count INTEGER NOT NULL,
            formal_output_count INTEGER NOT NULL,
            report_output_count INTEGER NOT NULL,
            activity_count INTEGER NOT NULL,
            output_bytes INTEGER NOT NULL,
            generated_at TEXT NOT NULL,
            PRIMARY KEY (snapshot_date, agent_id)
        );
        CREATE TABLE IF NOT EXISTS agent_output_events (
            snapshot_date TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            absolute_path TEXT NOT NULL,
            output_kind TEXT NOT NULL,
            is_formal INTEGER NOT NULL,
            byte_size INTEGER NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (snapshot_date, agent_id, relative_path, output_kind)
        );
        """
    )


def render_daily_markdown_report(snapshot: dict[str, Any]) -> str:
    lines = [
        "# 智能体产出与 Token 监测日报",
        "",
        f"- 日期：`{snapshot['snapshot_date']}`",
        f"- 生成时间：`{snapshot['generated_at']}`",
        "",
        "## 汇总",
        "",
        "| 智能体 | Token | 业务产出 | 正式产出 | 报告产出 | 活动次数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in snapshot["agents"]:
        if row["token_status"] == "exact":
            token_text = f"{row['exact_total_tokens']}（精确）"
        elif row["token_status"] == "estimated":
            token_text = f"{row['estimated_total_tokens']}（估算）"
        else:
            token_text = "0"
        lines.append(
            f"| {row['agent_label']} | {token_text} | {row['business_output_count']} | "
            f"{row['formal_output_count']} | {row['report_output_count']} | {row['activity_count']} |"
        )

    for row in snapshot["agents"]:
        lines.extend(
            [
                "",
                f"## {row['agent_label']}",
                "",
                f"- Token 状态：`{row['token_status']}`",
            ]
        )
        if row["token_status"] == "exact":
            lines.append(
                f"- Token：prompt `{row['exact_prompt_tokens']}` / completion `{row['exact_completion_tokens']}` / total `{row['exact_total_tokens']}`"
            )
        elif row["token_status"] == "estimated":
            lines.append(f"- Token 估算：`{row['estimated_total_tokens']}`")
            lines.append(f"- 估算依据：`{row['estimate_basis']}`")
        lines.append(f"- 业务产出：`{row['business_output_count']}`")
        lines.append(f"- 正式产出：`{row['formal_output_count']}`")
        lines.append(f"- 报告产出：`{row['report_output_count']}`")
        if row["top_outputs"]:
            lines.append("- 代表性产出：")
            for item in row["top_outputs"]:
                suffix = "（正式）" if item["is_formal"] else ""
                lines.append(f"  - `{item['relative_path']}` `{item['output_kind']}`{suffix}")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "snapshot_date",
        "agent_id",
        "agent_label",
        "exact_prompt_tokens",
        "exact_completion_tokens",
        "exact_total_tokens",
        "estimated_total_tokens",
        "token_status",
        "estimate_basis",
        "business_output_count",
        "formal_output_count",
        "report_output_count",
        "activity_count",
        "output_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


def write_daily_snapshot(
    *,
    snapshot_date: str,
    project_root: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, str]:
    snapshot = build_daily_snapshot(
        snapshot_date=snapshot_date,
        project_root=project_root,
        data_dir=data_dir,
    )
    resolved_data_dir = Path(snapshot["data_dir"])
    runtime_dir = observability_runtime_dir(resolved_data_dir)
    reports_dir = observability_reports_dir(resolved_data_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    db_path = observability_db_path(resolved_data_dir)
    with sqlite3.connect(db_path) as conn:
        ensure_db_schema(conn)
        conn.execute("DELETE FROM agent_daily_metrics WHERE snapshot_date = ?", (snapshot_date,))
        conn.execute("DELETE FROM agent_output_events WHERE snapshot_date = ?", (snapshot_date,))
        conn.executemany(
            """
            INSERT INTO agent_daily_metrics (
                snapshot_date, agent_id, agent_label,
                exact_prompt_tokens, exact_completion_tokens, exact_total_tokens,
                estimated_total_tokens, token_status, estimate_basis,
                business_output_count, formal_output_count, report_output_count,
                activity_count, output_bytes, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["snapshot_date"],
                    row["agent_id"],
                    row["agent_label"],
                    row["exact_prompt_tokens"],
                    row["exact_completion_tokens"],
                    row["exact_total_tokens"],
                    row["estimated_total_tokens"],
                    row["token_status"],
                    row["estimate_basis"],
                    row["business_output_count"],
                    row["formal_output_count"],
                    row["report_output_count"],
                    row["activity_count"],
                    row["output_bytes"],
                    snapshot["generated_at"],
                )
                for row in snapshot["agents"]
            ],
        )
        conn.executemany(
            """
            INSERT INTO agent_output_events (
                snapshot_date, agent_id, relative_path, absolute_path,
                output_kind, is_formal, byte_size, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event["snapshot_date"],
                    event["agent_id"],
                    event["relative_path"],
                    event["absolute_path"],
                    event["output_kind"],
                    event["is_formal"],
                    event["byte_size"],
                    event["source"],
                )
                for event in snapshot["outputs"]
            ],
        )
        conn.commit()

    json_path = reports_dir / f"daily_agent_observability_{snapshot_date}.json"
    csv_path = reports_dir / f"daily_agent_observability_{snapshot_date}.csv"
    md_path = reports_dir / f"daily_agent_observability_{snapshot_date}.md"
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, snapshot["agents"])
    md_path.write_text(render_daily_markdown_report(snapshot), encoding="utf-8")
    return {
        "db_path": str(db_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
    }


def discover_active_dates(
    *,
    project_root: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> list[str]:
    root = project_root_from_path(project_root)
    resolved_data_dir = resolve_data_dir(data_dir or (root / "quant" / "data_file"))
    dates: set[str] = set()
    for row in read_jsonl(collaboration_requests_path(resolved_data_dir)):
        parsed = parse_any_datetime(row.get("time"))
        if parsed is not None:
            dates.add(to_snapshot_date(parsed))
    evidence_index = build_evidence_agent_index(read_jsonl(collaboration_requests_path(resolved_data_dir)), root)
    for path in collect_candidate_paths(root, resolved_data_dir):
        try:
            if infer_agent_from_path(path, evidence_index):
                dates.add(file_snapshot_date(path))
        except FileNotFoundError:
            continue
    return sorted(dates)


def write_date_range(
    *,
    start_date: str,
    end_date: str,
    project_root: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> list[dict[str, str]]:
    start = datetime.strptime(start_date, "%Y%m%d").date()
    end = datetime.strptime(end_date, "%Y%m%d").date()
    if end < start:
        raise ValueError("end_date must be >= start_date")
    current = start
    results = []
    while current <= end:
        results.append(
            write_daily_snapshot(
                snapshot_date=current.strftime("%Y%m%d"),
                project_root=project_root,
                data_dir=data_dir,
            )
        )
        current = current.fromordinal(current.toordinal() + 1)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily agent token and business-output observability.")
    parser.add_argument("--date", help="Snapshot date in YYYYMMDD. Defaults to today (Asia/Shanghai).")
    parser.add_argument("--data-dir")
    parser.add_argument("--project-root")
    parser.add_argument("--backfill-history", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root_from_path(args.project_root)
    data_dir = resolve_data_dir(args.data_dir or (root / "quant" / "data_file"))

    if args.backfill_history:
        if args.start_date and args.end_date:
            results = write_date_range(
                start_date=args.start_date,
                end_date=args.end_date,
                project_root=root,
                data_dir=data_dir,
            )
        else:
            results = []
            for snapshot_date in discover_active_dates(project_root=root, data_dir=data_dir):
                results.append(
                    write_daily_snapshot(
                        snapshot_date=snapshot_date,
                        project_root=root,
                        data_dir=data_dir,
                    )
                )
        print(json.dumps({"status": "ok", "runs": results}, ensure_ascii=False, indent=2))
        return 0

    snapshot_date = args.date or datetime.now(tz=SH_TZ).strftime("%Y%m%d")
    result = write_daily_snapshot(
        snapshot_date=snapshot_date,
        project_root=root,
        data_dir=data_dir,
    )
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
