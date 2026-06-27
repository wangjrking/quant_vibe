from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


OUTER_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BIN_ROOT = OUTER_PROJECT_ROOT / "quant" / "data_file" / "runtime" / "recycle_bin"
DEFAULT_RETENTION_DAYS = 30


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_isoformat_z(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def ensure_bin_layout(bin_root: Path) -> None:
    (bin_root / "files").mkdir(parents=True, exist_ok=True)
    (bin_root / "logs").mkdir(parents=True, exist_ok=True)


def manifest_path(bin_root: Path) -> Path:
    return bin_root / "manifest.jsonl"


def relative_to_project(path: Path, project_root: Path) -> Path:
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {path}") from exc


def move_to_recycle_bin(
    paths: Iterable[str],
    *,
    actor: str,
    reason: str,
    bin_root: Path,
    project_root: Path,
    retention_days: int,
) -> list[dict[str, str]]:
    ensure_bin_layout(bin_root)
    manifest = manifest_path(bin_root)
    now = utc_now()
    batch_id = now.strftime("%Y%m%dT%H%M%SZ")
    batch_root = bin_root / "files" / batch_id
    batch_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for raw_path in paths:
        source = Path(raw_path).expanduser()
        if not source.is_absolute():
            source = (project_root / source).resolve()
        else:
            source = source.resolve()

        if not source.exists():
            raise FileNotFoundError(f"path does not exist: {source}")

        relative_path = relative_to_project(source, project_root)
        destination = batch_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

        expires_at = now + timedelta(days=retention_days)
        record = {
            "status": "trashed",
            "actor": actor,
            "reason": reason,
            "trashed_at": isoformat_z(now),
            "expires_at": isoformat_z(expires_at),
            "batch_id": batch_id,
            "original_path": str(source),
            "original_relative_path": str(relative_path).replace("\\", "/"),
            "trashed_path": str(destination),
        }
        records.append(record)

    with manifest.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


def latest_mtime(path: Path) -> datetime:
    if path.is_file():
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    latest = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    for item in path.rglob("*"):
        item_time = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc)
        if item_time > latest:
            latest = item_time
    return latest


def cleanup_recycle_bin(bin_root: Path, *, retention_days: int, dry_run: bool) -> list[dict[str, str]]:
    ensure_bin_layout(bin_root)
    manifest = manifest_path(bin_root)
    if not manifest.exists():
        return []

    now = utc_now()
    removed: list[dict[str, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") != "trashed":
            continue

        trashed_path = Path(record["trashed_path"])
        if not trashed_path.exists():
            continue

        reference_time = latest_mtime(trashed_path)
        delete_after = reference_time + timedelta(days=retention_days)
        if now < delete_after:
            continue

        removed_record = {
            "original_relative_path": record["original_relative_path"],
            "trashed_path": str(trashed_path),
            "delete_after": isoformat_z(delete_after),
            "deleted_at": isoformat_z(now),
            "dry_run": str(dry_run).lower(),
        }
        removed.append(removed_record)
        if not dry_run:
            if trashed_path.is_dir():
                shutil.rmtree(trashed_path)
            else:
                trashed_path.unlink()

    if removed:
        cleanup_log = bin_root / "logs" / f"cleanup_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
        cleanup_log.write_text(json.dumps(removed, ensure_ascii=False, indent=2), encoding="utf-8")

    return removed


def list_recycle_bin(bin_root: Path) -> list[dict[str, str]]:
    manifest = manifest_path(bin_root)
    if not manifest.exists():
        return []
    records = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-local recycle bin helper.")
    parser.add_argument("--bin-root", default=str(DEFAULT_BIN_ROOT))
    parser.add_argument("--project-root", default=str(OUTER_PROJECT_ROOT))
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)

    subparsers = parser.add_subparsers(dest="command", required=True)

    move_parser = subparsers.add_parser("move", help="Move files or directories into the project recycle bin.")
    move_parser.add_argument("paths", nargs="+")
    move_parser.add_argument("--actor", required=True)
    move_parser.add_argument("--reason", required=True)

    cleanup_parser = subparsers.add_parser("cleanup", help="Delete recycle-bin entries older than retention window.")
    cleanup_parser.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("list", help="List recycle-bin manifest entries.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    bin_root = Path(args.bin_root).resolve()
    project_root = Path(args.project_root).resolve()

    if args.command == "move":
        records = move_to_recycle_bin(
            args.paths,
            actor=args.actor,
            reason=args.reason,
            bin_root=bin_root,
            project_root=project_root,
            retention_days=args.retention_days,
        )
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0

    if args.command == "cleanup":
        removed = cleanup_recycle_bin(
            bin_root,
            retention_days=args.retention_days,
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(removed, ensure_ascii=False, indent=2))
        return 0

    if args.command == "list":
        print(json.dumps(list_recycle_bin(bin_root), ensure_ascii=False, indent=2))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
