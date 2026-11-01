"""
The release tree of run T (branch topstart: topstart-lr-decisions + v2-fast + topstart-topstart).

The two learner-side changes meet in one place, learner._perform_update right before ppo_update:
the decisions schedule (--lr-hold-decisions) sets every group's rate from D before the update, and
v2-fast's --fast-update runs ppo_update's minibatch loop without GPU syncs. These tests hold that
together they are what each is alone, on real GPU updates (test_lr_decisions.py's child process and
data, run in this tree):

  * fast_update with the schedule off: the same weights, fp32 masters, Adam state, rates and update
    lines as without it (v2-fast's own claim, here on the merged tree);
  * fast_update with the schedule at its floor: the same states as without fast_update, the same
    optimizer steps (so the same KL stops), and every optimizer step at base x 1/27.

GPU only (skipped on a CPU, and on a tree without v2-fast). Run on box c, never on the laptop.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "tests"), os.path.join(REPO, "cloud")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import learner  # noqa: E402
import test_lr_decisions as tld  # noqa: E402

GPU = torch.cuda.is_available()
HAS_FAST = "fast_update" in inspect.signature(learner.LearnerActor.__init__).parameters
HAS_DEC = hasattr(learner.LearnerActor, "_apply_decision_rate")

INIT_LINE = "    la._init_if_needed()\n"
FAST_LINE = INIT_LINE + "    la.fast_update = bool(run.get(\"fast_update\", False))\n"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not GPU:
        pytest.skip("a whole update needs a GPU")
    if not (HAS_FAST and HAS_DEC):
        pytest.skip("not the release tree: the learner lacks fast_update or the decisions schedule")
    assert tld.CHILD.count(INIT_LINE) == 1, "test_lr_decisions.CHILD changed: this test sets la.fast_update after its init"
    child_src = tld.CHILD.replace(INIT_LINE, FAST_LINE)
    d = tmp_path_factory.mktemp("release_real")
    obs, ep, kind = tld.micro_ref.observations(n_min=2048)
    updates = [tld.tsu._update_data(obs, ep, 1024, seed=10), tld.tsu._update_data(obs[1024:], ep[1024:], 1024, seed=11)]
    torch.save({"updates": updates}, str(d / "data.pt"))
    child = d / "child.py"
    child.write_text(child_src, encoding="utf-8")
    floor = {"mode": "ppo", "lr_hold_decisions": 0, "lr_total_decisions": 1, "lr_warmup_steps": 0}
    spec = [{"name": "off", "learner": {"mode": "ppo"}},
            {"name": "off_fast", "learner": {"mode": "ppo"}, "fast_update": True},
            {"name": "floor", "learner": dict(floor), "record": True},
            {"name": "floor_fast", "learner": dict(floor), "record": True, "fast_update": True}]
    out = d / "release.json"
    p = subprocess.run([sys.executable, str(child), REPO, str(d / "data.pt"), str(out), json.dumps(spec)],
                       capture_output=True, text=True, timeout=3600)
    assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-6000:])
    with open(out, "r", encoding="utf-8") as f:
        res = json.load(f)
    print(f"\nreal updates on {kind} observations in {REPO}")
    return res


@pytest.mark.skipif(not GPU, reason="a whole update needs a GPU")
def test_fast_update_with_the_schedule_off_is_the_same_update(runs):
    off, fast = runs["off"], runs["off_fast"]
    assert len(off["states"]) == len(fast["states"]) == 2
    for i, (a, b) in enumerate(zip(off["states"], fast["states"])):
        diff = sorted(k for k in a if a[k] != b.get(k))
        assert diff == [], (i, diff[:8])
    assert off["lines"] == fast["lines"]


@pytest.mark.skipif(not GPU, reason="a whole update needs a GPU")
def test_fast_update_with_the_schedule_at_its_floor_keeps_every_step_at_the_scheduled_rate(runs):
    slow, fast = runs["floor"], runs["floor_fast"]
    for i, (a, b) in enumerate(zip(slow["states"], fast["states"])):
        diff = sorted(k for k in a if a[k] != b.get(k))
        assert diff == [], (i, diff[:8])
    assert slow["lines"] == fast["lines"]
    # the same optimizer steps (the same KL stops), each at the same rate, and that rate base x 1/27
    assert [(s["update"], s["step"], s["dec"]) for s in fast["steps"]] == [(s["update"], s["step"], s["dec"]) for s in slow["steps"]]
    assert [s["lr"] for s in fast["steps"]] == [s["lr"] for s in slow["steps"]]
    base = [float(b) for b in fast["base_lrs"]]
    assert fast["steps"] and {s["update"] for s in fast["steps"]} == {0, 1}
    for s in fast["steps"]:
        assert s["dec"] == 1024 * (s["update"] + 1)
        want = [b * tld.check_restart.rate_share_decisions(s["step"], s["dec"], 0, 0, 1) for b in base]
        assert [float(x) for x in s["lr"]] == want, s
        assert abs(float(s["lr"][0]) / base[0] - tld.FLOOR) < 1e-12
