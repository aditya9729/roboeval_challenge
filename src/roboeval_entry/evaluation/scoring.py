"""Official-score access and bootstrap confidence intervals.

``roboeval.challenge.scoring`` and ``roboeval.challenge.spec`` are stdlib-only
and imported directly from the pinned vendor tree via ``ROBOEVAL_VENDOR``.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

DEFAULT_VENDOR = "/workspace/roboeval/vendor/RoboEval"


def vendor_root() -> Path:
    return Path(os.environ.get("ROBOEVAL_VENDOR", DEFAULT_VENDOR))


def import_official_scoring():
    root = str(vendor_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("roboeval.challenge.scoring")


def _mean_metrics(episodes: list[dict[str, float]]) -> dict[str, float]:
    keys = list(episodes[0])
    return {k: float(np.mean([e[k] for e in episodes])) for k in keys}


def rds_from_episode_metrics(family: str, episodes: list[dict[str, float]]) -> dict:
    """Official task_scores for one family's per-episode metric dicts."""
    scoring = import_official_scoring()
    return scoring.task_scores(family, _mean_metrics(episodes))


def suite_rds(per_family_episodes: Mapping[str, list[dict[str, float]]]) -> dict:
    """Official suite_scores from per-family episode metric lists."""
    scoring = import_official_scoring()
    per_family = {fam: _mean_metrics(eps) for fam, eps in per_family_episodes.items()}
    return scoring.suite_scores(per_family)


def _suite_score(per_family_episodes: Mapping[str, list[dict[str, float]]]) -> float:
    return float(suite_rds(per_family_episodes)["overall"]["score"])


def bootstrap_suite(per_family_episodes: Mapping[str, list[dict[str, float]]],
                    n_boot: int = 1000, seed: int = 0) -> dict:
    """Resample episodes within each family; report CI on the suite score."""
    rng = np.random.default_rng(seed)
    point = _suite_score(per_family_episodes)
    task_scores = {f: rds_from_episode_metrics(f, eps)["score"]
                   for f, eps in per_family_episodes.items()}
    boot_suite = np.empty(n_boot)
    boot_task: dict[str, np.ndarray] = {f: np.empty(n_boot) for f in per_family_episodes}
    for b in range(n_boot):
        sampled = {}
        for fam, eps in per_family_episodes.items():
            take = rng.integers(0, len(eps), size=len(eps))
            resampled = [eps[i] for i in take]
            sampled[fam] = resampled
            boot_task[fam][b] = rds_from_episode_metrics(fam, resampled)["score"]
        boot_suite[b] = _suite_score(sampled)
    return {
        "mean": point,
        "ci95": [float(np.percentile(boot_suite, 2.5)),
                 float(np.percentile(boot_suite, 97.5))],
        "per_task_ci": {
            f: {"mean": task_scores[f],
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for f, v in boot_task.items()
        },
    }


def paired_bootstrap_delta(a: Mapping[str, list[dict[str, float]]],
                           b: Mapping[str, list[dict[str, float]]],
                           n_boot: int = 1000, seed: int = 0) -> dict:
    """Suite-score delta (a - b) with CI, resampling the same episode indices."""
    if set(a) != set(b):
        raise ValueError("paired bootstrap requires the same families")
    for fam in a:
        if len(a[fam]) != len(b[fam]):
            raise ValueError(f"paired bootstrap requires equal episode counts in {fam}")
    rng = np.random.default_rng(seed)
    point = _suite_score(a) - _suite_score(b)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        sa, sb = {}, {}
        for fam in a:
            take = rng.integers(0, len(a[fam]), size=len(a[fam]))
            sa[fam] = [a[fam][i] for i in take]
            sb[fam] = [b[fam][i] for i in take]
        deltas[k] = _suite_score(sa) - _suite_score(sb)
    return {
        "delta": point,
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
    }
