"""
The reward for one finished game.

    r_t = scale * (gamma * Phi(s_{t+1}) - Phi(s_t))          for every decision but the last
    r_T = outcome + scale * (0 - Phi(s_T))                    for the last decision

`outcome` is +1 for a win, -1 for a loss and 0 for a tie. Phi is the potential from
`potential.py`, and the potential of the finished game is zero.

This is potential-based shaping (Ng, Harada and Russell, 1999). Summed over a game with the
same discount the learner uses, the shaping terms cancel except for the first:

    sum_t gamma^t * r_t  =  gamma^(T-1) * outcome  -  scale * Phi(s_0)

Phi(s_0) does not depend on anything the agent does, so the policy that maximises this is the
policy that maximises the win/loss return. What the shaping changes is how early the learner
hears about a good or bad turn: the change in estimated winning chances arrives on the step
that caused it instead of twenty steps later with the result.

The potential of the finished game has to be zero for that to hold (Grzes, 2017). The older
"+0.1 per faint" reward breaks it: at the end of a won game the faint count is not zero, so it
pays more for wide wins than for narrow ones, which is a different objective from winning.

Returns lie in [-1 - scale, 1 + scale]. With the default scale of 0.5 that is the [-1.5, 1.5]
support the value head already uses.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def shaped_rewards(phi: Sequence[float], outcome: float, gamma: float, scale: float = 0.5) -> np.ndarray:
    """
    phi[t] is the potential at the agent's t-th decision of the game. Returns one reward per
    decision. With scale 0 this is the plain sparse reward.
    """
    phi = np.asarray(phi, dtype=np.float64)
    T = phi.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.float32)
    nxt = np.empty(T, dtype=np.float64)
    nxt[:-1] = phi[1:]
    nxt[-1] = 0.0  # the finished game
    r = float(scale) * (float(gamma) * nxt - phi)
    r[-1] += float(outcome)
    return r.astype(np.float32)


def discounted_return(rewards: Sequence[float], gamma: float) -> float:
    g = 0.0
    for r in reversed(list(rewards)):
        g = float(r) + gamma * g
    return g
