"""Record a no-op JSON encoding remediation without changing business data.

Evidence JSON remains standard UTF-8 without a BOM so both Python and Node can
parse it directly.  Windows PowerShell 5 reviewers must request UTF-8
explicitly instead of relying on their ANSI default code page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--remediation", required=True)
    args = parser.parse_args()

    target = Path(args.json).resolve()
    remediation = Path(args.remediation).resolve()
    original = target.with_name(target.stem + "_utf8_nobom_original_preserved.json")
    before = json.loads(target.read_text(encoding="utf-8"))
    before_sha = file_sha256(target)
    shutil.copy2(target, original)
    shutil.copy2(original, target)
    after = json.loads(target.read_text(encoding="utf-8"))
    result = {
        "remediation_type": "evidence_encoding_only_utf8_no_bom",
        "business_assets_modified": False,
        "prediction_values_recomputed": False,
        "target": str(target),
        "original_preserved": str(original),
        "before_sha256": before_sha,
        "preserved_original_sha256": file_sha256(original),
        "after_sha256": file_sha256(target),
        "semantic_json_equality": before == after,
        "serialization": "UTF-8 without BOM; PowerShell 5 validation must use Get-Content -Encoding UTF8",
    }
    remediation.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
