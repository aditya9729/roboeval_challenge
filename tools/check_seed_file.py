"""Validate a local evaluator JSON seed list before launching a run."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.seed_guard import validate_seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--split", required=True, choices=["train", "dev", "holdout"])
    args = parser.parse_args()
    try:
        data = json.loads(args.path.read_text())
        if not isinstance(data, list):
            raise ValueError("Expected a JSON list of integers")
        checked = validate_seeds(data, args.split)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Seed validation failed: {error}\n")
    print(f"Validated {len(checked)} {args.split} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
