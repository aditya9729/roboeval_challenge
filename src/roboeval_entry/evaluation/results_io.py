"""Read official episode logs without importing vendor results.py."""
from __future__ import annotations

import json
from pathlib import Path


def load_episode_logs(logs_dir: str | Path) -> dict[str, list[dict]]:
    """{family: [episode record, ...]} from logs/<Family>_<Variation>/episode_*.json."""
    logs_dir = Path(logs_dir)
    out: dict[str, list[dict]] = {}
    for vdir in sorted(logs_dir.iterdir()) if logs_dir.exists() else []:
        if not vdir.is_dir():
            continue
        family, sep, _ = vdir.name.partition("_")
        family = family if sep else vdir.name
        files = sorted(vdir.glob("episode_*.json"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if files:
            out[family] = [json.loads(p.read_text()) for p in files]
    return out


def load_episode_metrics(logs_dir: str | Path) -> dict[str, list[dict[str, float]]]:
    """{family: [metrics dict, ...]} ready for scoring.rds/suite_rds."""
    return {fam: [ep["metrics"] for ep in eps]
            for fam, eps in load_episode_logs(logs_dir).items()}
