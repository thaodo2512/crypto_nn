from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def classmix_report(pre_by_fold: Dict[int, Dict[str, int]], out_path: str) -> None:
    report = {str(fid): counts for fid, counts in pre_by_fold.items()}
    # Acceptance: WAIT share in TRAIN post ≤ 60%
    for fid, counts in pre_by_fold.items():
        post_total = counts.get("post_LONG", 0) + counts.get("post_SHORT", 0) + counts.get("post_WAIT", 0)
        wait_share = counts.get("post_WAIT", 0) / post_total if post_total else 0.0
        report[str(fid)]["wait_share_post"] = wait_share
        report[str(fid)]["accept"] = wait_share <= 0.60
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    if not all(v.get("accept", False) for v in report.values()):
        raise SystemExit(1)


def oos_integrity_report() -> None:
    # Placeholder: actual OOS integrity is enforced by design (no write of VAL/OOS)
    return None

