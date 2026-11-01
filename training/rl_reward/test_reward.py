#!/usr/bin/env python
"""
Checks on the reward. Run with:  .venv/Scripts/python.exe -m rl_reward.test_reward

Each test states a property the reward is supposed to have and fails loudly if it does not.
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys

import numpy as np
import pytest

from .features import FEATURES, replay_states
from .potential import Potential, load_potential
from .shaping import discounted_return, shaped_rewards

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_shaping_cancels():
    """Summed with the learner's discount, the shaping leaves gamma^(T-1) * outcome - scale * Phi(s0)."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        T = int(rng.integers(1, 80))
        phi = rng.uniform(-0.85, 0.85, size=T)
        outcome = float(rng.choice([-1.0, 0.0, 1.0]))
        gamma = float(rng.choice([1.0, 0.999, 0.99]))
        scale = float(rng.choice([0.0, 0.25, 0.5, 1.0]))
        g = discounted_return(shaped_rewards(phi, outcome, gamma, scale), gamma)
        want = gamma ** (T - 1) * outcome - scale * phi[0]
        assert abs(g - want) < 1e-4, (g, want)


def test_same_start_same_ranking():
    """
    Two games from the same position that run the same number of decisions are ranked by the
    shaped return exactly as by the result, whatever happened in between. A lead that was built
    and thrown away earns nothing.
    """
    rng = np.random.default_rng(1)
    for _ in range(100):
        T = int(rng.integers(2, 60))
        start = float(rng.uniform(-0.5, 0.5))
        a = np.concatenate([[start], rng.uniform(0.5, 0.85, size=T - 1)])    # ahead all game, then loses
        b = np.concatenate([[start], rng.uniform(-0.85, -0.5, size=T - 1)])  # behind all game, then wins
        ga = discounted_return(shaped_rewards(a, -1.0, 0.999, 0.5), 0.999)
        gb = discounted_return(shaped_rewards(b, +1.0, 0.999, 0.5), 0.999)
        assert gb > ga
        assert abs((gb - ga) - 2 * 0.999 ** (T - 1)) < 1e-4


def test_scale_zero_is_sparse():
    r = shaped_rewards([0.3, -0.2, 0.7], 1.0, 0.999, 0.0)
    assert np.allclose(r, [0.0, 0.0, 1.0])


def test_returns_fit_the_value_head():
    """Every return-to-go lies inside the value head's support."""
    from config import RunConfig
    cfg = RunConfig.default()
    pot = load_potential(cfg.reward.potential_file or None)
    bound = 1.0 + cfg.reward.shaping_scale * pot.cap
    assert cfg.learner.v_min <= -bound and bound <= cfg.learner.v_max, (bound, cfg.learner.v_min, cfg.learner.v_max)
    rng = np.random.default_rng(2)
    for _ in range(200):
        T = int(rng.integers(1, 80))
        phi = rng.uniform(-pot.cap, pot.cap, size=T)
        r = shaped_rewards(phi, float(rng.choice([-1.0, 1.0])), cfg.learner.gamma, cfg.reward.shaping_scale)
        g = 0.0
        for x in r[::-1]:
            g = float(x) + cfg.learner.gamma * g
            assert cfg.learner.v_min - 1e-6 <= g <= cfg.learner.v_max + 1e-6


def test_potential_is_zero_sum():
    """The same position from the other side has the opposite potential."""
    pot = load_potential()
    rng = np.random.default_rng(3)
    for _ in range(200):
        f = rng.normal(size=len(FEATURES))
        assert abs(pot.phi(f) + pot.phi(-f)) < 1e-12
        assert abs(pot.phi(f)) <= pot.cap + 1e-12
    assert pot.phi(np.zeros(len(FEATURES))) == 0.0


def test_features_mirror_in_a_real_game():
    """Run a recorded game from both seats: the feature vectors are each other's negatives."""
    files = sorted(glob.glob(os.path.join(ROOT, "heuristic", "lessons", "*.json")))
    if not files:
        pytest.skip("no recorded games under heuristic/lessons (real ladder games are not published)")
    checked = 0
    for f in files[:8]:
        rec = json.load(io.open(f, encoding="utf-8"))
        lines = rec["log"]
        a, ta, wa = replay_states(lines, "Me")
        b, tb, wb = replay_states(lines, "Foe")
        assert ta == tb and len(a) > 3
        for x, y in zip(a, b):
            assert np.allclose(x, -y, atol=1e-9), (f, x, y)
            checked += 1
    assert checked > 50


def test_weights_have_the_expected_signs():
    """More Pokemon, more HP and an unused Tera help; status, hazards on my side and lost stages hurt."""
    pot = load_potential()
    w = dict(zip(FEATURES, pot.w))
    assert w["hp_share"] > 0 and w["alive_diff"] > 0 and w["hp_diff"] > 0
    assert w["par_diff"] < 0 and w["brn_diff"] < 0 and w["psn_diff"] < 0
    assert w["rocks_diff"] < 0 and w["spikes_diff"] < 0
    assert w["off_boost_diff"] > 0 and w["tera_diff"] > 0
    # one Pokemon up at full health, everything else level, is a clear but not decisive edge
    f = np.zeros(len(FEATURES))
    f[FEATURES.index("alive_diff")] = 1 / 6
    f[FEATURES.index("hp_diff")] = 1 / 6
    f[FEATURES.index("alive_share")] = 6 / 11 - 0.5
    f[FEATURES.index("hp_share")] = 6 / 11 - 0.5
    assert 0.52 < pot.win_prob(f) < 0.75, pot.win_prob(f)


def test_faint_reward_pays_for_margin():
    """
    Why the per-faint reward was replaced. Under it a 6-0 win returns 1.6 and a 6-5 win 1.1, so
    it prefers a line that wins less often by more. Under the potential both return the same.
    """
    wide = 1.0 + 0.1 * 6 - 0.1 * 0
    narrow = 1.0 + 0.1 * 6 - 0.1 * 5
    assert wide - narrow == 0.5
    T = 30
    phi_wide = np.linspace(0.0, 0.8, T)
    phi_narrow = np.concatenate([np.linspace(0.0, -0.6, T - 1), [0.1]])
    gw = discounted_return(shaped_rewards(phi_wide, 1.0, 1.0, 0.5), 1.0)
    gn = discounted_return(shaped_rewards(phi_narrow, 1.0, 1.0, 0.5), 1.0)
    assert abs(gw - gn) < 1e-6


def test_reward_config_default():
    from config import RunConfig
    cfg = RunConfig.default()
    assert cfg.reward.shaping == "potential"
    assert cfg.reward.terminal_win == 1.0 and cfg.reward.terminal_loss == -1.0 and cfg.reward.terminal_tie == 0.0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS ", t.__name__)
        except pytest.skip.Exception as e:
            print("SKIP ", t.__name__, "->", str(e)[:300])
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL ", t.__name__, "->", repr(e)[:300])
    print(f"\n{len(tests) - failed}/{len(tests)} reward tests pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
