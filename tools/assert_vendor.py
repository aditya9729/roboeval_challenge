"""Fail closed unless the current simulator environment imports the pinned tree."""
from __future__ import annotations
import argparse
import importlib
import json
import platform
import subprocess
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor", type=Path)
    args = parser.parse_args()
    root = args.vendor.resolve()
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/challenge.json").read_text())
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        parser.error("Official replay target is Linux x86-64; use that host")
    if "avx2" not in Path("/proc/cpuinfo").read_text().lower().split():
        parser.error("AVX2 CPU capability not found")
    if git(root, "rev-parse", "HEAD") != config["expected_commit"]:
        parser.error("Benchmark commit differs from reviewed source snapshot")
    if git(root, "status", "--porcelain", "--untracked-files=normal"):
        parser.error("Benchmark tree has edits or untracked files; use a clean external checkout")
    submodules = git(root, "submodule", "status", "--recursive")
    if any(line.startswith(("+", "-", "U")) for line in submodules.splitlines()):
        parser.error("Submodule revision is changed, missing or conflicted")
    # This imports the package in the actual interpreter used for evaluation.
    module = importlib.import_module("roboeval")
    location = Path(module.__file__).resolve()
    if not location.is_relative_to(root):
        parser.error(f"Wrong installed RoboEval: {location}")
    spec = importlib.import_module("roboeval.challenge.spec")
    if spec.CONTROL_FREQUENCY != config["control_frequency_hz"]:
        parser.error("Control frequency mismatch")
    if tuple(spec.EVAL_SEEDS) != tuple(range(config["official_seed_start"], config["official_seed_end_inclusive"] + 1)):
        parser.error("Official seed specification changed")
    if {t.key: t.budget_seconds for t in spec.TASK_FAMILIES} != config["tasks_budget_seconds"]:
        parser.error("Task manifest changed")
    print(f"Pinned clean source verified: {config['expected_commit']}")
    print(f"Imported package: {location}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
