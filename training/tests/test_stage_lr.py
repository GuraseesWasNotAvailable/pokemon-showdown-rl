"""cloud/stage_lr.py: the flags put the rate where asked at the resume and at the end of the stage."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cloud"))

from check_restart import rate_share  # noqa: E402
from stage_lr import FLOOR, path, plan  # noqa: E402

# stage3's live run, checkpoint 016375: optimizer step and decisions (the numbers the schedule was planned on)
STEP, DEC = 1_104_534, 604_489_541


def share(p, step):
    return rate_share(step, p["warmup"], p["lr_hold_steps"], p["lr_total_steps"])


@pytest.mark.parametrize("r0", [6e-5, 3e-5, 2.5e-5, 3e-4])
def test_decline_starts_at_the_rate_and_ends_at_the_floor(r0):
    p = plan(STEP, DEC, 900_000_000, r0)
    assert p["lr"] == 3e-4
    assert p["rate_at_resume"] == pytest.approx(r0, rel=1e-4)
    assert p["lr_total_steps"] == STEP + round((900_000_000 - DEC) / 384)
    assert share(p, p["lr_total_steps"]) == pytest.approx(FLOOR)
    rates = [r for _d, _s, r in path(p)]
    assert rates[0] == pytest.approx(r0, rel=1e-4) and rates[-1] == pytest.approx(3e-4 * FLOOR)
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_hold_then_decline():
    p = plan(STEP, DEC, 900_000_000, 6e-5, hold_decisions=100_000_000)
    assert p["lr"] == 6e-5 and p["rate_at_resume"] == pytest.approx(6e-5)
    edge = p["warmup"] + p["lr_hold_steps"]
    assert edge == STEP + round(100_000_000 / 384)
    assert share(p, edge - 1) == 1.0 and share(p, edge + 10) < 1.0
    assert share(p, p["lr_total_steps"]) == pytest.approx(FLOOR)


def test_today_is_the_floor():
    """That run's own flags (hold 500,000, total 1,000,000) give 1.11e-5 at this step: the decline at the floor is a no-op."""
    assert 3e-4 * rate_share(STEP, 1000, 500_000, 1_000_000) == pytest.approx(3e-4 / 27)


@pytest.mark.parametrize("kw", [dict(start_rate=1e-5), dict(start_rate=1.2e-5), dict(start_rate=4e-4), dict(start_rate=6e-5, end_decisions=DEC),
                                dict(start_rate=6e-5, hold_decisions=400_000_000)])
def test_impossible_stages_are_refused(kw):
    """1.2e-5: the decline from so close to the floor would have to start before step 0 (use a hold)."""
    args = dict(step=STEP, decisions=DEC, end_decisions=900_000_000)
    args.update(kw)
    with pytest.raises(ValueError):
        plan(**args)


def test_hold_then_decline_to_an_end_rate():
    """The stage's shape: hold 3e-5, then decline to today's floor (1.11e-5) at the end of the stage."""
    p = plan(STEP, DEC, 900_000_000, 3e-5, hold_decisions=100_000_000, end_rate=3e-4 / 27)
    assert p["lr"] == 3e-5 and p["rate_at_resume"] == pytest.approx(3e-5)
    edge = p["warmup"] + p["lr_hold_steps"]
    assert edge == STEP + round(100_000_000 / 384)
    assert share(p, edge - 1) == 1.0
    assert p["end_step"] == STEP + round((900_000_000 - DEC) / 384)
    assert 3e-5 * share(p, p["end_step"]) == pytest.approx(3e-4 / 27, rel=1e-3)
    assert p["rate_at_end"] == pytest.approx(3e-4 / 27, rel=1e-3)
    assert p["lr_total_steps"] > p["end_step"]                     # it goes on falling, slowly, after the stage
    assert share(p, p["lr_total_steps"]) == pytest.approx(FLOOR)
    rates = [r for _d, _s, r in path(p)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    assert rates[0] == pytest.approx(3e-5) and rates[-1] == pytest.approx(3e-4 / 27, rel=1e-3)


@pytest.mark.parametrize("kw", [dict(start_rate=3e-5, hold_decisions=100_000_000, end_rate=3e-5),
                                dict(start_rate=3e-5, hold_decisions=100_000_000, end_rate=1e-6),
                                dict(start_rate=6e-5, end_rate=1.11e-5)])
def test_impossible_end_rates_are_refused(kw):
    with pytest.raises(ValueError):
        plan(step=STEP, decisions=DEC, end_decisions=900_000_000, **kw)
