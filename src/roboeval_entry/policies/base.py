"""Policy-facing interfaces, decoupled from the official evaluator objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from roboeval_entry.contracts.action_spec import GRIPPER_STATE_DIM, StateLayout


class ObservationContractError(Exception):
    """Raised when an observation dict violates the declared schema."""


@dataclass(frozen=True)
class EpisodeContext:
    """The only episode metadata a policy may keep.

    Deliberately excludes seed / episode_index / policy_seed / env_name /
    action_space from the official EpisodeSpec so seeds can never become
    learned inputs or lookup keys.
    """

    family: str
    variation: str
    instruction: str
    control_frequency: int
    max_steps: int
    action_dim: int

    @classmethod
    def from_episode_spec(cls, episode) -> "EpisodeContext":
        return cls(
            family=str(episode.family),
            variation=str(episode.variation),
            instruction=str(episode.instruction),
            control_frequency=int(episode.control_frequency),
            max_steps=int(episode.max_steps),
            action_dim=int(episode.action_space.shape[0]),
        )


@runtime_checkable
class PolicyAdapter(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...
    def act(self, obs: Mapping[str, np.ndarray]) -> np.ndarray: ...


_FLOATING_BASE_KEYS = ("proprioception_floating_base", "proprioception_floating_base_actions")


class ObservationSchema:
    """Declared cameras define the exact observation contract.

    Any key outside the expected set is rejected so privileged or
    observation-leaking keys surface immediately.
    """

    def __init__(self, cameras: tuple[str, ...] = (), resolution: tuple[int, int] = (256, 256),
                 depth: bool = False):
        self.cameras = tuple(cameras)
        self.resolution = tuple(resolution)
        self.depth = depth

    def expected_keys(self) -> tuple[str, ...]:
        keys = ["proprioception", "proprioception_grippers", *_FLOATING_BASE_KEYS]
        for cam in self.cameras:
            keys.append(f"rgb_{cam}")
            if self.depth:
                keys.append(f"depth_{cam}")
        return tuple(keys)

    def validate(self, obs: Mapping[str, np.ndarray]) -> None:
        expected = set(self.expected_keys())
        missing = expected - set(obs)
        extra = set(obs) - expected
        if missing:
            raise ObservationContractError(f"missing observation keys: {sorted(missing)}")
        if extra:
            raise ObservationContractError(
                f"unexpected observation keys (privileged leakage guard): {sorted(extra)}")
        checks = {
            "proprioception": ((StateLayout.DIM,), np.float32),
            "proprioception_grippers": ((GRIPPER_STATE_DIM,), np.float32),
        }
        for k in _FLOATING_BASE_KEYS:
            checks[k] = ((0,), np.float32)
        h, w = self.resolution
        for cam in self.cameras:
            checks[f"rgb_{cam}"] = ((3, h, w), np.uint8)
            if self.depth:
                checks[f"depth_{cam}"] = ((h, w), np.float32)
        for key, (shape, dtype) in checks.items():
            v = np.asarray(obs[key])
            if tuple(v.shape) != shape:
                raise ObservationContractError(
                    f"{key} shape {tuple(v.shape)} != expected {shape}")
            if v.dtype != dtype:
                raise ObservationContractError(
                    f"{key} dtype {v.dtype} != expected {dtype}")
