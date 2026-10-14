"""SGS-inspired reset probabilities, NOT an implementation of the SGS paper.

The warm-up, finite pool, rolling window and quality mixture are proposed
engineering choices. The caller must supply permitted training outcomes only.
No training seed, quality label or sampler state belongs in policy observations.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Settings:
    tau: float = 0.5
    kappa: float = 2.0
    temperature: float = 1.0
    epsilon: float = 1e-6
    window: int = 20
    minimum_attempts: int = 2
    sgs_mass: float = 0.5
    uniform_mass: float = 0.25
    quality_mass: float = 0.25

    def validate(self) -> None:
        scalars = (self.tau, self.kappa, self.temperature, self.epsilon,
                   self.sgs_mass, self.uniform_mass, self.quality_mass)
        if not all(math.isfinite(x) for x in scalars):
            raise ValueError("Settings must be finite")
        if not 0 <= self.tau <= 1 or self.kappa < 0:
            raise ValueError("Invalid target or concentration")
        if self.temperature <= 0 or not 0 < self.epsilon < 1:
            raise ValueError("Temperature and epsilon must be positive")
        if type(self.window) is not int or type(self.minimum_attempts) is not int:
            raise ValueError("Window and minimum attempts must be integers")
        if not 1 <= self.minimum_attempts <= self.window:
            raise ValueError("Require 1 <= minimum_attempts <= window")
        masses = (self.sgs_mass, self.uniform_mass, self.quality_mass)
        if any(x < 0 for x in masses) or self.uniform_mass <= 0:
            raise ValueError("Masses must be nonnegative and uniform mass positive")
        if not math.isclose(sum(masses), 1.0, abs_tol=1e-12):
            raise ValueError("Mixture masses must sum to one")


def probabilities(
    histories: Mapping[str, Sequence[int]],
    settings: Settings = Settings(),
    quality_priority: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return probabilities keyed by configuration ID.

Histories are binary valid episode outcomes, ordered oldest to newest. Clear or
version these externally when replacing a policy. Quality weights, when used,
are nonnegative priorities from authorized diagnostics, not rewards.
"""
    settings.validate()
    if not histories:
        raise ValueError("Configuration pool cannot be empty")
    if any(not isinstance(k, str) or not k for k in histories):
        raise ValueError("Configuration IDs must be nonempty strings")
    keys = sorted(histories)
    for values in histories.values():
        if any(type(v) is not int or v not in (0, 1) for v in values):
            raise ValueError("History outcomes must be integer zero or one")
    if quality_priority is not None:
        if set(quality_priority) - set(keys):
            raise ValueError("Quality priority references an unknown configuration")
        if any(not math.isfinite(v) or v < 0 for v in quality_priority.values()):
            raise ValueError("Quality priorities must be finite and nonnegative")
    n = len(keys)
    uniform = 1.0 / n
    if any(len(histories[k]) < settings.minimum_attempts for k in keys):
        return {k: uniform for k in keys}
    logs: list[float] = []
    for key in keys:
        h = histories[key][-settings.window:]
        p = sum(h) / len(h)
        log_w = (settings.kappa * settings.tau * math.log(p + settings.epsilon)
                 + settings.kappa * (1 - settings.tau)
                 * math.log(1 - p + settings.epsilon))
        scaled_log = log_w / settings.temperature
        if not math.isfinite(scaled_log):
            raise ValueError("Concentration/temperature overflows log weights")
        logs.append(scaled_log)
    max_log = max(logs)
    weights = [math.exp(x - max_log) for x in logs]
    total = math.fsum(weights)
    sgs = [w / total for w in weights]
    quality = [float((quality_priority or {}).get(k, 0.0)) for k in keys]
    quality_total = math.fsum(quality)
    if quality_total:
        quality = [w / quality_total for w in quality]
    else:
        quality = [uniform] * n  # unavailable quality diagnostics -> uniform
    result = {
        key: settings.sgs_mass * sgs[i] + settings.uniform_mass * uniform
        + settings.quality_mass * quality[i]
        for i, key in enumerate(keys)
    }
    # Normalize floating-point drift while preserving the uniform support floor.
    norm = math.fsum(result.values())
    return {k: p / norm for k, p in result.items()}


def choose(probability: Mapping[str, float], rng: Random) -> str:
    """Sample with an explicit RNG; distribution must have strictly positive mass."""
    if not probability:
        raise ValueError("Empty distribution")
    values = list(probability.values())
    if any(not math.isfinite(p) or p <= 0 for p in values):
        raise ValueError("Probabilities must be finite and positive")
    if not math.isclose(math.fsum(values), 1.0, abs_tol=1e-9):
        raise ValueError("Probabilities must sum to one")
    return rng.choices(list(probability), weights=values, k=1)[0]
