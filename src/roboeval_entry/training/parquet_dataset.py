"""LeRobot-v3 parquet dataset for RoboEval (openpi env only).

Alignment contract (contracts/action_spec.json dataset_alignment): frame t's
images/state are the observation AFTER actions[t], so the BC pairing is
obs(frame t-1) -> actions[t]. The dataset has no reset observation, so t=0 is
dropped — one sample per episode is lost.

Requires pyarrow + pillow (lazy-imported); safe to import the module on any
interpreter.
"""
from __future__ import annotations

import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

INDEX_COLS = ["episode_index", "frame_index", "uuid", "task_index", "state", "actions"]


class _FileCache:
    """Per-worker LRU of camera columns read from parquet files."""

    def __init__(self, max_files: int = 4):
        self._cache: OrderedDict[Path, dict[str, list]] = OrderedDict()
        self._max = max_files

    def frames(self, path: Path, cam_cols: list[str]) -> dict[str, list]:
        import pyarrow.parquet as pq

        hit = self._cache.get(path)
        if hit is None or any(c not in hit for c in cam_cols):
            table = pq.read_table(path, columns=cam_cols)
            hit = {c: table.column(c).to_pylist() for c in cam_cols}
            self._cache[path] = hit
            self._cache.move_to_end(path)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return hit


def _png_decode(b: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


class RoboEvalParquetDataset:
    """Random-access dataset over RoboEval LeRobot-v3 parquets.

    Returns LIBERO-like raw samples:
      observation/{head,left_wrist,right_wrist}_image  uint8 (H,W,3)
      observation/state                                float32[18] (qpos incl. fingers)
      actions                                          float32[H,14]
      actions_is_pad                                   bool[H]
      prompt                                           str
      episode_uuid, t                                  debug fields
    """

    def __init__(
        self,
        dataset_root: str | Path,
        variants: Sequence[str],
        split: str,
        registry_path: str | Path,
        action_horizon: int,
        cameras: Sequence[str] = ("head", "left_wrist", "right_wrist"),
        state_mode: Literal["qpos18"] = "qpos18",
        episode_limit: int | None = None,
    ):
        if state_mode != "qpos18":
            raise ValueError(f"unsupported state_mode {state_mode!r}")
        self.dataset_root = Path(dataset_root)
        self.cameras = tuple(cameras)
        self.action_horizon = action_horizon
        self.split = split

        registry = json.loads(Path(registry_path).read_text())
        uuid_split = {e["episode_id"]: e["split"] for e in registry["episodes"]}

        self._episodes: list[dict] = []
        self._files: list[Path] = []
        self._index: list[tuple[int, int]] = []  # (episode i, t)
        for variant in variants:
            vdir = self.dataset_root / variant
            if not vdir.is_dir():
                raise FileNotFoundError(f"variant dir missing: {vdir}")
            tasks = self._load_tasks(vdir)
            episodes = self._index_variant(vdir, uuid_split, tasks)
            if episode_limit is not None:
                episodes = episodes[:episode_limit]
            for ep in episodes:
                ei = len(self._episodes)
                self._episodes.append(ep)
                for t in range(1, ep["length"]):
                    self._index.append((ei, t))

    @staticmethod
    def _load_tasks(vdir: Path) -> dict[int, str]:
        import pyarrow.parquet as pq

        tasks_file = vdir / "meta" / "tasks.parquet"
        table = pq.read_table(tasks_file)
        df = table.to_pydict()
        str_cols = [c for c in df if c != "task_index"
                    and all(isinstance(v, str) for v in df[c])]
        if len(str_cols) != 1:
            raise ValueError(f"cannot find instruction column in {tasks_file}")
        by_index = {int(i): str(t) for i, t in
                    zip(df["task_index"], df[str_cols[0]])}
        return by_index

    def _index_variant(self, vdir: Path, uuid_split: dict[str, str],
                       tasks: list[str]) -> list[dict]:
        import pyarrow.parquet as pq

        files = sorted((vdir / "data").rglob("*.parquet"))
        # file -> ordered list of (episode_index, frame_index) for row locators
        file_rows: list[tuple[Path, list[tuple[int, int]]]] = []
        rows: dict[str, list] = {c: [] for c in INDEX_COLS}
        locators: list[tuple[int, int]] = []
        for fi, path in enumerate(files):
            key_table = pq.read_table(path, columns=["episode_index", "frame_index"])
            keys = list(zip(key_table.column("episode_index").to_pylist(),
                            key_table.column("frame_index").to_pylist()))
            file_rows.append((path, keys))
            table = pq.read_table(path, columns=INDEX_COLS)
            n = table.num_rows
            locators.extend((fi, r) for r in range(n))
            for c in INDEX_COLS:
                rows[c].extend(table.column(c).to_pylist())
        ep_idx = np.asarray(rows["episode_index"], dtype=np.int64)
        fr_idx = np.asarray(rows["frame_index"], dtype=np.int64)
        order = np.lexsort((fr_idx, ep_idx))

        episodes: dict[int, dict] = {}
        for r in order:
            uuid = str(rows["uuid"][r])
            ep = int(ep_idx[r])
            if uuid not in uuid_split:
                raise KeyError(f"episode uuid {uuid} not in registry")
            if uuid_split[uuid] != self.split:
                continue
            e = episodes.setdefault(ep, {"variant": vdir.name, "episode_index": ep,
                                         "uuid": uuid, "rows": []})
            e["rows"].append((r, locators[r]))
        out = []
        for ep in sorted(episodes):
            e = episodes[ep]
            e["rows"].sort(key=lambda rl: int(fr_idx[rl[0]]))
            e["state"] = np.stack([np.asarray(rows["state"][r], np.float32)
                                   for r, _ in e["rows"]])
            e["actions"] = np.stack([np.asarray(rows["actions"][r], np.float32)
                                     for r, _ in e["rows"]])
            e["locators"] = [loc for _, loc in e["rows"]]
            e["prompt"] = tasks[int(rows["task_index"][e["rows"][0][0]])]  # noqa: E501
            e["length"] = len(e["rows"])
            del e["rows"]
            out.append(e)
        # locators are relative to this variant's file list; offset into the
        # global file table
        base = len(self._files)
        self._files.extend(f for f, _ in file_rows)
        for e in out:
            e["locators"] = [(fi + base, row) for fi, row in e["locators"]]
        return out

    def __len__(self) -> int:
        return len(self._index)

    def _frame_image(self, fi: int, row: int, cam: str) -> np.ndarray:
        cache = getattr(self, "_file_cache", None)
        if cache is None:
            cache = self._file_cache = _FileCache()
        table = cache.frames(self._files[fi], [f"cam_{cam}"])
        entry = table[f"cam_{cam}"][row]
        blob = entry["bytes"] if isinstance(entry, dict) else entry
        return _png_decode(blob)

    def __getitem__(self, index: int) -> dict:
        ei, t = self._index[int(index)]
        ep = self._episodes[ei]
        obs_fi, obs_row = ep["locators"][t - 1]
        state = ep["state"][t - 1][:18].astype(np.float32)
        horizon = self.action_horizon
        chunk = ep["actions"][t: t + horizon]
        is_pad = np.zeros(horizon, dtype=bool)
        if len(chunk) < horizon:
            pad = np.repeat(chunk[-1:], horizon - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad])
            is_pad[len(ep["actions"][t: t + horizon]):] = True
        out = {
            f"observation/{cam}_image": self._frame_image(obs_fi, obs_row, cam)
            for cam in self.cameras
        }
        return {
            **out,
            "observation/state": state,
            "actions": chunk.astype(np.float32),
            "actions_is_pad": is_pad,
            "prompt": ep["prompt"],
            "episode_uuid": ep["uuid"],
            "t": t,
        }

    def summary(self) -> dict:
        return {
            "num_episodes": len(self._episodes),
            "num_samples": len(self._index),
            "variants": sorted({e["variant"] for e in self._episodes}),
            "split": self.split,
        }
