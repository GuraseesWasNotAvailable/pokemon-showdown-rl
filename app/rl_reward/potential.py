"""
The potential function: how good a position is, as a number between -1 and 1.

Phi(s) = cap * tanh(w . features(s)), with the feature differences from `features.py` and the
weights and cap fitted on recorded games by `fit_potential.py`. Read as a chance of winning it
is P = (1 + Phi) / 2. There is no intercept and tanh is odd, so the position seen from the
other side has exactly the opposite potential. The cap is below 1 because in these games even
a large lead is lost about one time in six.

The potential is not the reward. The reward is built from differences of it (see `shaping.py`),
which is what keeps the best policy the same as under plain win/loss.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional, Sequence

import numpy as np

from .features import FEATURES, battle_features

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")


class Potential:
    def __init__(self, weights: Sequence[float], names: Optional[Sequence[str]] = None, meta: Optional[dict] = None, cap: float = 1.0):
        names = list(names) if names is not None else list(FEATURES)
        if len(weights) != len(names):
            raise ValueError("one weight per feature is required")
        by_name = dict(zip(names, weights))
        unknown = [n for n in names if n not in FEATURES]
        if unknown:
            raise ValueError(f"weights for features this code does not compute: {unknown}")
        # Features the file does not mention get weight 0, so a pruned model loads cleanly.
        self.w = np.array([float(by_name.get(n, 0.0)) for n in FEATURES], dtype=np.float64)
        self.meta = meta or {}
        if not 0.0 < cap <= 1.0:
            raise ValueError("cap must be in (0, 1]")
        self.cap = float(cap)

    def phi(self, features: np.ndarray) -> float:
        return self.cap * float(np.tanh(np.dot(self.w, features)))

    def win_prob(self, features: np.ndarray) -> float:
        return 0.5 * (1.0 + self.phi(features))

    def __call__(self, battle) -> float:
        """Potential of a live poke-env battle, from its owner's side. 0.0 if it cannot be read."""
        try:
            if getattr(battle, "finished", False):
                return 0.0  # the potential of a finished game is zero by definition
            return float(self.phi(battle_features(battle)))
        except Exception as e:  # noqa: BLE001
            # A zero here turns that step's shaping off, it does not break the reward. But if it
            # happens every step the run is silently training on win/loss alone, so say so once.
            self.failures = getattr(self, "failures", 0) + 1
            if self.failures == 1:
                logging.getLogger(__name__).warning("potential could not be read from a battle (%r); using 0 for that step", e)
            return 0.0


def load_potential(path: Optional[str] = None) -> Potential:
    path = path or DEFAULT_WEIGHTS
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Potential(d["weights"], d.get("features"), d.get("meta"), d.get("cap", 1.0))
