"""Build decoded-image memmap caches for RoboEval variants (openpi env).

Per variant writes <out_dir>/<variant>/<cam>.u8 (uint8 N,256,256,3) in global
(episode_index, frame_index) order plus index.json recording that order, the
uuids, per-cam sha256, and the dataset commit.
"""
from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_OUT = Path("/workspace/roboeval/data/image_cache")
DATASET_COMMIT = "918b4072cd1d15f28b1f96a440ac7aa84591ce92"


def _decode_file(task):
    path, cam_cols = task
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path, columns=cam_cols)
    out = {}
    for c in cam_cols:
        imgs = []
        for entry in table.column(c).to_pylist():
            blob = entry["bytes"] if isinstance(entry, dict) else entry
            imgs.append(np.asarray(Image.open(io.BytesIO(blob)).convert("RGB")))
        out[c] = np.stack(imgs)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_image_cache(dataset_root: str | Path, variant: str,
                      cameras: Sequence[str] = ("head", "left_wrist", "right_wrist"),
                      out_dir: str | Path = DEFAULT_OUT,
                      procs: int = 32) -> Path:
    import pyarrow.parquet as pq

    vdir = Path(dataset_root) / variant
    files = sorted((vdir / "data").rglob("*.parquet"))
    cam_cols = [f"cam_{c}" for c in cameras]

    order: list[tuple[int, int]] = []
    uuids: list[str] = []
    file_row_pos: list[tuple[Path, list[int]]] = []
    for path in files:
        t = pq.read_table(path, columns=["episode_index", "frame_index", "uuid"])
        eps = t.column("episode_index").to_pylist()
        frs = t.column("frame_index").to_pylist()
        file_row_pos.append((path, list(range(len(eps)))))
        order.extend(zip(eps, frs))
        uuids.extend(str(u) for u in t.column("uuid").to_pylist())

    sort_idx = sorted(range(len(order)), key=lambda i: order[i])
    position = [0] * len(order)
    sorted_order = [order[i] for i in sort_idx]
    sorted_uuids = [uuids[i] for i in sort_idx]
    for pos, orig in enumerate(sort_idx):
        position[orig] = pos

    out = Path(out_dir) / variant
    out.mkdir(parents=True, exist_ok=True)
    n = len(order)
    maps = {}
    for cam, col in zip(cameras, cam_cols):
        mm = np.memmap(out / f"{cam}.u8", dtype=np.uint8, mode="w+",
                       shape=(n, 256, 256, 3))
        maps[col] = mm

    with mp.Pool(procs) as pool:
        for file_i, decoded in enumerate(
                pool.imap(_decode_file, [(p, cam_cols) for p, _ in file_row_pos])):
            path, rows = file_row_pos[file_i]
            base = sum(len(r) for _, r in file_row_pos[:file_i])
            for col, imgs in decoded.items():
                mm = maps[col]
                for local_i, img in enumerate(imgs):
                    mm[position[base + local_i]] = img
    for mm in maps.values():
        mm.flush()

    index = {
        "n_frames": n,
        "order": [[int(e), int(f)] for e, f in sorted_order],
        "uuids": sorted_uuids,
        "cameras": list(cameras),
        "dataset_commit": DATASET_COMMIT,
        "sha256_per_cam": {},
    }
    for cam, col in zip(cameras, cam_cols):
        index["sha256_per_cam"][cam] = _sha256(out / f"{cam}.u8")
    (out / "index.json").write_text(json.dumps(index))
    return out
