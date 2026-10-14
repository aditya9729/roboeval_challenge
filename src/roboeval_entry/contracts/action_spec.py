"""Typed contract for the challenge action space, loaded from a JSON file.

The JSON is the single source of truth for the 14-dim ee-delta action layout,
the measured env bounds, and the data-derived guard bounds. Fields that can
only be measured on a running simulator stay null until the probe fills them;
loading with nulls is allowed but ``ActionGuard`` refuses to construct then.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


class ActionContractError(Exception):
    """Raised when the action contract file or an action violates the contract."""


@dataclass(frozen=True)
class ActionLayout:
    LEFT_POS: slice = slice(0, 3)
    LEFT_ROT: slice = slice(3, 6)
    RIGHT_POS: slice = slice(6, 9)
    RIGHT_ROT: slice = slice(9, 12)
    LEFT_GRIP: int = 12
    RIGHT_GRIP: int = 13
    DIM: int = 14


@dataclass(frozen=True)
class StateLayout:
    LEFT_ARM_QPOS: slice = slice(0, 7)
    LEFT_FINGER_QPOS: slice = slice(7, 9)
    RIGHT_ARM_QPOS: slice = slice(9, 16)
    RIGHT_FINGER_QPOS: slice = slice(16, 18)
    QVEL_OFFSET: int = 18
    DIM: int = 36


GRIPPER_STATE_DIM = 2


@dataclass(frozen=True)
class GripperSpec:
    command_range: tuple[float, float]
    discrete: bool
    open_command: Optional[int]
    closed_command: Optional[int]
    verified: bool


@dataclass(frozen=True)
class Bounds:
    low: tuple[float, ...]
    high: tuple[float, ...]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.low, np.float64), np.asarray(self.high, np.float64)


@dataclass(frozen=True)
class ActionSpec:
    version: int
    source_commit: str
    control_frequency_hz: int
    layout: tuple[str, ...]
    rotation_representation: str
    rotation_composition: str
    position_frame: str
    delta_reference: str
    env_bounds: Optional[Bounds]
    guard_bounds: Optional[Bounds]
    guard_bounds_provenance: Optional[dict]
    gripper: GripperSpec
    joint_speed_limit_rad_per_step: float
    finite_required: bool

    @property
    def dim(self) -> int:
        return len(self.layout)


def _bounds(obj: object, name: str, dim: int) -> Optional[Bounds]:
    if obj is None:
        return None
    try:
        low, high = obj["low"], obj["high"]
    except (TypeError, KeyError) as e:
        raise ActionContractError(f"{name} must have 'low'/'high' arrays") from e
    if low is None or high is None or \
            any(v is None for v in low) or any(v is None for v in high):
        return None
    if len(low) != dim or len(high) != dim:
        raise ActionContractError(
            f"{name} length mismatch: expected {dim}, got low={len(low)} high={len(high)}"
        )
    lo, hi = np.asarray(low, np.float64), np.asarray(high, np.float64)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ActionContractError(f"{name} contains non-finite values")
    if not np.all(lo <= hi):
        raise ActionContractError(f"{name} has low > high")
    return Bounds(tuple(lo.tolist()), tuple(hi.tolist()))


def load_action_spec(path: str | Path, expected_commit: Optional[str] = None) -> ActionSpec:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ActionContractError(f"cannot read action spec {path}: {e}") from e
    if not isinstance(data, dict):
        raise ActionContractError(f"{path} must contain a JSON object")

    commit = data.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ActionContractError("action_spec.source_commit must be a 40-char git sha")
    if expected_commit is not None and commit != expected_commit:
        raise ActionContractError(
            f"action_spec.source_commit {commit} != expected {expected_commit}"
        )

    layout = data.get("layout")
    if not isinstance(layout, list) or len(layout) != ActionLayout.DIM or \
            not all(isinstance(n, str) for n in layout):
        raise ActionContractError(f"layout must be a list of {ActionLayout.DIM} names")

    g = data.get("gripper") or {}
    gripper = GripperSpec(
        command_range=tuple(g.get("command_range", (0.0, 1.0))),
        discrete=bool(g.get("discrete", True)),
        open_command=g.get("open_command"),
        closed_command=g.get("closed_command"),
        verified=bool(g.get("verified", False)),
    )

    spec = ActionSpec(
        version=int(data.get("version", 0)),
        source_commit=commit,
        control_frequency_hz=int(data.get("control_frequency_hz", 20)),
        layout=tuple(layout),
        rotation_representation=str(data.get("rotation_representation", "")),
        rotation_composition=str(data.get("rotation_composition", "")),
        position_frame=str(data.get("position_frame", "")),
        delta_reference=str(data.get("delta_reference", "")),
        env_bounds=_bounds(data.get("env_bounds"), "env_bounds", ActionLayout.DIM),
        guard_bounds=_bounds(data.get("guard_bounds"), "guard_bounds", ActionLayout.DIM),
        guard_bounds_provenance=data.get("guard_bounds_provenance"),
        gripper=gripper,
        joint_speed_limit_rad_per_step=float(data.get("joint_speed_limit_rad_per_step", 0.0)),
        finite_required=bool(data.get("finite_required", True)),
    )
    if spec.rotation_representation != "rotvec":
        raise ActionContractError(
            f"unexpected rotation_representation {spec.rotation_representation!r}")
    if spec.rotation_composition != "premultiply_pelvis_frame":
        raise ActionContractError(
            f"unexpected rotation_composition {spec.rotation_composition!r}")
    if spec.position_frame != "pelvis":
        raise ActionContractError(
            f"unexpected position_frame {spec.position_frame!r}")
    if spec.delta_reference != "actuator_ctrl_targets":
        raise ActionContractError(
            f"unexpected delta_reference {spec.delta_reference!r}")
    if spec.control_frequency_hz != 20:
        raise ActionContractError(
            f"control_frequency_hz must be 20, got {spec.control_frequency_hz}")

    align = data.get("dataset_alignment")
    if not isinstance(align, dict):
        raise ActionContractError("missing required 'dataset_alignment' block")
    expected_align = {
        "frame_t_state_is": "observation_after_actions_t",
        "bc_pairing": "obs=frame[t-1] (env reset observation for t=0) -> actions[t]",
    }
    for key, want in expected_align.items():
        if align.get(key) != want:
            raise ActionContractError(
                f"dataset_alignment.{key} must be {want!r}, got {align.get(key)!r}")
    return spec
