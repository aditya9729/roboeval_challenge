"""Record host hardware into reports/hardware.json (any env; run with sim python)."""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from roboeval_entry.data.io import sha256_file, write_json_atomic  # noqa: E402
from roboeval_entry.evaluation.scoring import vendor_root  # noqa: E402


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL,
                                       timeout=30).strip()
    except Exception as e:
        return f"unavailable: {e}"


def _nvidia_smi() -> dict:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"])
    gpus = []
    for line in out.splitlines():
        name, mem, driver = (p.strip() for p in line.split(","))
        gpus.append({"name": name, "memory": mem, "driver": driver})
    return {"gpus": gpus, "count": len(gpus)}


def _cpu() -> dict:
    out = _run(["lscpu"])
    info = {}
    for line in out.splitlines():
        k, _, v = line.partition(":")
        info[k.strip()] = v.strip()
    return {
        "model": info.get("Model name"),
        "sockets": info.get("Socket(s)"),
        "cores_per_socket": info.get("Core(s) per socket"),
        "threads_per_core": info.get("Thread(s) per core"),
        "cpus": info.get("CPU(s)"),
        "flags_avx2": "avx2" in info.get("Flags", "").split(),
    }


def _mem_gb() -> float:
    text = Path("/proc/meminfo").read_text()
    m = re.search(r"MemTotal:\s+(\d+) kB", text)
    return round(int(m.group(1)) / 1e6, 1) if m else -1.0


def main() -> int:
    vendor = vendor_root()
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": _cpu(),
        "gpu": _nvidia_smi(),
        "mem_gb": _mem_gb(),
        "vendor_root": str(vendor),
        "vendor_commit": _run(["git", "-C", str(vendor), "rev-parse", "HEAD"]),
        "vendor_dirty": bool(_run(["git", "-C", str(vendor), "status", "--porcelain",
                                   "--untracked-files=normal"])),
        "locks": {
            p.name: sha256_file(p)
            for p in sorted(Path("/workspace/roboeval/envs").glob("*requirements.lock.txt"))
        },
    }
    out = write_json_atomic(REPO / "reports" / "hardware.json", record)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
