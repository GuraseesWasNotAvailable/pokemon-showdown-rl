"""
The learning-rate schedule counted in a phase's decisions: train.py --lr-hold-decisions and
--lr-total-decisions (LearnerConfig.lr_hold_decisions / lr_total_decisions), off by default.

    python -m pytest tests/test_lr_decisions.py -q -p no:cacheprovider

Why. The step schedule counts optimizer steps, and an update takes about 49.6 of them while it
stops early on KL (about 743 decisions a step) and 96 once it runs both epochs (384 a step). The
from-scratch run's planned hold of 800,000 steps ("about 307M decisions") would have ended near
596M, and its anneal near 903M (1.19B had the KL stops gone on); it was fixed by hand. With the
schedule in decisions the rate is a pure function of the phase's decisions D (the learner's
total_steps, the counter the supervisor's target_decisions reads), set before every update.

Held here:
  (a) off: the config is config.py's; the step lambda over optimizer steps 0 to 2M is a verbatim
      copy of 4cfd496's; one real update (two, in fact) of the learner gives the same weights,
      fp32 masters, Adam state, per-group rates and update line, bit for bit, as 4cfd496's
      learner (git archive 4cfd496, or $PSPPO_LRDEC_BASE, run in a subprocess; needs a GPU);
  (b) on, 16,300 simulated updates of 36,864 decisions (49.6 steps an update switching to 96 at
      the hold's end as the from-scratch run did, all at 96, all at 49.6): the full rate for every
      update with D < H, the first lower update has D in [H, H + 36,864), the floor of 1/27 from
      the first update with D >= T (within one update of T), every group keeps its zone
      multiplier, every optimizer step of an update has the same rate; and a real update on the
      GPU at the floor and in the hold;
  (c) the step schedule with the from-scratch run's original flags (hold 800k, total 1.6M) on the
      same simulation: where it really lands (the bug this fixes);
  (d) the first 1,000 optimizer steps of a phase ramp exactly as now;
  (e) a save mid-anneal, rebuilt and loaded: the rate at the resume and at the next update equal
      the uninterrupted run's; a phase change resets D to 0;
  (f) train.py refuses (exit 2) one flag alone, total <= hold, a negative hold, and the decision
      flags next to the step flags;
  (g) cloud/check_restart.py: the same flags pass (exit 0); a changed hold is exit 3, or 0 with
      --allow-rate-change; a step-schedule checkpoint resumed with decision flags (and the other
      way round) is exit 3 unless allowed; its rate_share_decisions is the learner's own.

No Ray, no server, no port. (b) to (g) run on the CPU. Nothing is removed: what the tests write
stays in pytest's temporary directory and $PSPPO_TEST_DIR.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import io
import json
import os
import subprocess
import sys
import tarfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, TESTS, os.path.join(REPO, "cloud")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

import torch  # noqa: E402

import config as new_config  # noqa: E402
import learner as new_learner  # noqa: E402
import ppo_core as new_core  # noqa: E402
import train as new_train  # noqa: E402
import test_speed_update as tsu  # noqa: E402  (its learner helpers)
import check_restart  # noqa: E402
from bench import micro_ref  # noqa: E402

GPU = torch.cuda.is_available()
needs_gpu = pytest.mark.skipif(not GPU, reason="a whole update needs a GPU (flex_attention has no backward on a CPU)")

# The from-scratch run's schedule as it ended up after its hand fix, now in decisions.
H, T = 357_000_000, 600_000_000
PER = 36_864                    # decisions an update (steps_per_update)
N_UPDATES = 16_300              # 600.9M decisions
W = 1_000                       # lr_warmup_steps
FLAGS = ["--lr-hold-decisions", str(H), "--lr-total-decisions", str(T)]
MULT = {"embeddings": 0.1, "subnets": 0.1, "transformer": 0.1, "readout": 1.0, "pi": 1.0, "v": 2.0}
FLOOR = 1.0 / ((8 * 1.0 + 1) ** 1.5)

BASE_COMMIT = "4cfd496"
# git's names for 4cfd496's copies (git rev-parse 4cfd496:<file>): a directory given in
# $PSPPO_LRDEC_BASE is checked against them, so it can only be that commit's code.
BASE_BLOBS = {
    "learner.py": "199c311f132535d10945d6986cc0c9503f71b00e",
    "config.py": "5b7984a264fc588f946fa53bbf17d17e13441dc6",
    "ppo_core.py": "61a98c47c656d00c3ca72e1a25ef3f8d884f53d6",
    "train.py": "2c63fee4bb9a7b9d944cfd0c9d32a0f33ca30612",
    "utils.py": "ef3181b0341dbd26eb139408f8dc3e5736501ace",
}


def _cfg(argv, mode="ppo", device="cpu", **over):
    cfg = new_train.build_config(new_train.parse_args(["--mode", mode] + list(argv)))
    lc = dict(device=device, resume=False, save_every_updates=10 ** 9, ckpt_dir=os.path.join(tsu.TEST_ROOT, "ckpt_unused"))
    lc.update(over)
    return dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **lc))


def _learner(argv, mode="ppo", **over):
    """A learner on the CPU, without Ray and its loops (test_speed_update's helper)."""
    la = tsu._make_learner(new_learner, new_core, _cfg(argv, mode, **over))
    la.opt.inner.step()             # LambdaLR wants an optimizer step before its own (no gradients: nothing moves)
    return la


def _verbatim_4cfd496_lambda(w_steps: int, h_steps: int, t_steps: int):
    """learner.py's lr_lambda at 4cfd496, copied verbatim."""
    def lr_lambda(step: int) -> float:
        if step < w_steps:
            return float(step + 1) / float(w_steps)
        if step < (w_steps + h_steps):
            return 1.0

        anneal_start = w_steps + h_steps
        # Kept between 0 and 1 whatever the config says. A total at or below the end of the
        # hold leaves no room to anneal in: the rate then drops to 1/27 in one step (warned
        # about below), as it always did.
        progress = max(0.0, min(1.0, (step - anneal_start) / max(1, t_steps - anneal_start)))
        return 1.0 / ((8 * progress + 1) ** 1.5)
    return lr_lambda


# --------------------------------------------------------------------------- (a) off


def test_off_the_config_is_config_pys_and_the_settings_line_is_unchanged():
    base = new_config.RunConfig.default()
    assert new_train.build_config(new_train.parse_args([])) == base
    lc = base.learner
    assert (lc.lr_hold_decisions, lc.lr_total_decisions) == (0, 0)
    assert (lc.lr, lc.lr_warmup_steps, lc.lr_hold_steps, lc.lr_total_steps) == (3e-4, 1_000, 500_000, 1_000_000)
    # a Namespace built by hand without the new flags still builds
    plain = argparse.Namespace(mode="ppo", ports=None, battles=None, rooms_per_pair=None, teacher=None, steps_per_update=None,
                               minibatch=None, seq_len=None, grad_accum=None, save_every=None, ckpt_dir=None, shaping=None,
                               minutes=0.0, stop_file=None)
    assert new_train.build_config(plain) == dataclasses.replace(base, learner=dataclasses.replace(base.learner, mode="ppo"))
    assert new_train.ppo_settings_line(base) == ("ppo settings: clip_coef=0.1 update_epochs=2 lr=0.0003 target_kl=0.02 ent_coef=0.02 "
                                                 "lr_warmup_steps=1000 lr_hold_steps=500000 lr_total_steps=1000000")
    on = new_train.build_config(new_train.parse_args(FLAGS))
    assert new_train.ppo_settings_line(on).endswith("lr_total_steps=1000000 lr_hold_decisions=357000000 lr_total_decisions=600000000")
    assert on.learner.ppo_kwargs() == base.learner.ppo_kwargs()
    # the only trace in a checkpoint: two keys in run_cfg.learner, 0 when off
    assert base.as_dict()["learner"]["lr_hold_decisions"] == 0 and base.as_dict()["learner"]["lr_total_decisions"] == 0


@pytest.mark.parametrize("argv,whd", [([], (1_000, 500_000, 1_000_000)),
                                      (["--lr-hold-steps", "800000", "--lr-total-steps", "1600000"], (1_000, 800_000, 1_600_000))])
def test_off_the_lambda_over_2m_steps_is_4cfd496s(argv, whd):
    la = _learner(argv)
    assert not la._lr_by_decisions()
    fns = la.sched.lr_lambdas
    assert all(f is fns[0] for f in fns) and fns[0].__name__ == "lr_lambda"
    fn, ref = fns[0], _verbatim_4cfd496_lambda(*whd)
    moved = [s for s in range(0, 2_000_001) if fn(s) != ref(s)]
    assert moved == [], moved[:5]
    assert not hasattr(la, "_lr_dec"), "with the flags off the learner never sets the decisions counter"
    del la


# The child process of the real-update tests: the same text runs 4cfd496's tree and this one.
CHILD = r'''
import asyncio, contextlib, dataclasses, hashlib, io, json, os, random, sys
tree, data_path, out_path, runs_json = sys.argv[1:5]
sys.path.insert(0, tree)
os.chdir(tree)
import numpy as np
import torch
import config, learner, ppo_core


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


class FakeStore:
    def __init__(self):
        self.update = _Remote(lambda weights, version: None)
        self.set_temp = _Remote(lambda t: None)


def digest(t):
    t = t.detach().to("cpu").contiguous().reshape(-1)
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() + f" {tuple(t.shape)} {t.dtype}"


def state(la):
    out = {}
    for k, v in la.net.state_dict().items():
        out["model/" + k] = digest(v)
    for k, v in la.target_net.state_dict().items():
        out["target/" + k] = digest(v)
    for i, (_p, m) in enumerate(la.opt.pairs):
        out[f"master/{i}"] = digest(m)
    for i, st in enumerate(la.opt.inner.state.values()):
        for k, v in st.items():
            out[f"adam/{i}/{k}"] = digest(v) if torch.is_tensor(v) else repr(v)
    out["lr"] = [repr(float(pg["lr"])) for pg in la.opt.param_groups]
    out["sched"] = int(la.sched.last_epoch)
    out["update_idx"] = int(la.update_idx)
    return out


data = torch.load(data_path, weights_only=False)
result = {}
for run in json.loads(runs_json):
    cfg = config.RunConfig.default()
    lc = dict(device="cuda", minibatch_size=256, batch_seq_len=256, grad_accum_steps=1, steps_per_update=1024, resume=False,
              save_every_updates=10 ** 9, ckpt_dir=os.path.join(os.path.dirname(out_path), "ckpt_unused"))
    lc.update(run["learner"])
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **lc))
    torch.manual_seed(1234); np.random.seed(1234); random.seed(1234)
    la = object.__new__(learner.LearnerActor)
    la.run_cfg, la.cfg, la.weight_store = cfg, cfg.learner, FakeStore()
    la.net = la.opt = la.sched = None
    la.dataset = ppo_core.AsyncEpisodeDataset(act_dim=cfg.env.act_dim, device="cpu")
    la.update_idx = la.total_episodes = la.total_steps = la.ep_counter = 0
    la._init_if_needed()
    steps = []
    if run.get("record"):
        inner = la.opt.inner
        orig = inner.step

        def rec(*a, _orig=orig, **k):
            steps.append({"lr": [repr(float(pg["lr"])) for pg in inner.param_groups], "step": int(la.sched.last_epoch),
                          "dec": int(getattr(la, "_lr_dec", -1)), "update": int(la.update_idx)})
            return _orig(*a, **k)
        rec._wrapped_by_lr_sched = True       # the scheduler's own wrapper is still called (no warning about it)
        inner.step = rec
    states, lines = [], []
    for k, cols in enumerate(data["updates"]):
        la.dataset.add_steps(*cols)
        la.total_steps += int(cols[1].shape[0])
        torch.manual_seed(50 + k); np.random.seed(50 + k); random.seed(50 + k)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            asyncio.run(la._perform_update())
        text = buf.getvalue()
        got = [ln for ln in text.splitlines() if ln.startswith("[learner] upd=")]
        if la.update_idx != k + 1 or len(got) != 1:
            raise SystemExit(f"update {k + 1} of run {run['name']} did not complete: {text[-3000:]}")
        lines.append(" ".join(t for t in got[0].split() if not t.startswith("sec=")))
        states.append(state(la))
    result[run["name"]] = {"states": states, "lines": lines, "steps": steps,
                           "base_lrs": [repr(float(b)) for b in la.sched.base_lrs],
                           "names": [pg["name"] for pg in la.opt.param_groups]}
    del la
    torch.cuda.empty_cache()
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f)
'''


def _base_tree(factory) -> str:
    """4cfd496's files: $PSPPO_LRDEC_BASE (checked against git's names for them), or git archive here."""
    d = os.environ.get("PSPPO_LRDEC_BASE", "")
    if not d:
        d = str(factory.mktemp("base_4cfd496"))
        try:
            tar = subprocess.run(["git", "-C", REPO, "-c", "core.autocrlf=false", "archive", BASE_COMMIT],
                                 capture_output=True, check=True).stdout
        except Exception as e:      # noqa: BLE001
            pytest.skip(f"no {BASE_COMMIT} tree to compare with: git archive failed ({e!r}); set PSPPO_LRDEC_BASE")
        with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
            tf.extractall(d)
    for name, blob in BASE_BLOBS.items():
        with open(os.path.join(d, name), "rb") as f:
            got = micro_ref.git_blob_id(f.read().replace(b"\r\n", b"\n"))
        assert got == blob, f"{os.path.join(d, name)} is not {BASE_COMMIT}'s"
    return d


@pytest.fixture(scope="module")
def real_updates(tmp_path_factory):
    """Two real updates of the learner, in 4cfd496's tree and in this one (three configurations), in subprocesses."""
    if not GPU:
        pytest.skip("a whole update needs a GPU")
    base = _base_tree(tmp_path_factory)
    d = tmp_path_factory.mktemp("lrdec_real")
    obs, ep, kind = micro_ref.observations(n_min=2048)
    updates = [tsu._update_data(obs, ep, 1024, seed=10), tsu._update_data(obs[1024:], ep[1024:], 1024, seed=11)]
    torch.save({"updates": updates}, str(d / "data.pt"))
    child = d / "child.py"
    child.write_text(CHILD, encoding="utf-8")

    def run(tree, runs, name):
        out = d / f"{name}.json"
        p = subprocess.run([sys.executable, str(child), tree, str(d / "data.pt"), str(out), json.dumps(runs)],
                           capture_output=True, text=True, timeout=3600)
        assert p.returncode == 0, (p.stdout[-3000:], p.stderr[-6000:])
        with open(out, "r", encoding="utf-8") as f:
            return json.load(f)

    off = {"name": "off", "learner": {"mode": "ppo"}}
    base_res = run(base, [off], "base")
    new_res = run(REPO, [off,
                         {"name": "on_hold", "learner": {"mode": "ppo", "lr_hold_decisions": 10 ** 9, "lr_total_decisions": 2 * 10 ** 9},
                          "record": True},
                         {"name": "on_floor", "learner": {"mode": "ppo", "lr_hold_decisions": 0, "lr_total_decisions": 1,
                                                          "lr_warmup_steps": 0}, "record": True}], "new")
    print(f"\nreal updates on {kind} observations; base tree {base}")
    return {"base": base_res, "new": new_res, "again": lambda: run(base, [off], "base_again")}


@needs_gpu
def test_off_a_real_update_is_4cfd496s_bit_for_bit(real_updates):
    want, got = real_updates["base"]["off"], real_updates["new"]["off"]
    assert len(want["states"]) == len(got["states"]) == 2
    if want["states"] != got["states"] or want["lines"] != got["lines"]:
        again = real_updates["again"]()["off"]
        if again["states"] != want["states"]:
            pytest.skip("4cfd496's own update does not repeat itself on this machine: nothing to hold the new code to")
        for i, (a, b) in enumerate(zip(want["states"], got["states"])):
            diff = sorted(k for k in a if a[k] != b.get(k))
            assert diff == [], (i, diff[:8])
        assert want["lines"] == got["lines"]
    assert want["base_lrs"] == got["base_lrs"] and want["names"] == got["names"]
    for st in got["states"]:
        assert st["sched"] > 0 and any(k.startswith("adam/") for k in st)
    print("\n" + "\n".join(got["lines"]))


@needs_gpu
def test_on_in_the_hold_a_real_update_is_the_same_as_off(real_updates):
    """Flags on with D < H: the hook sets the rate the step schedule has anyway, so nothing moves, bit for bit."""
    off, on = real_updates["new"]["off"], real_updates["new"]["on_hold"]
    assert on["states"] == off["states"] and on["lines"] == off["lines"]
    assert on["steps"] and all(s["dec"] in (1024, 2048) for s in on["steps"])


@needs_gpu
def test_on_at_the_floor_every_optimizer_step_of_a_real_update_has_the_scheduled_rate(real_updates):
    run = real_updates["new"]["on_floor"]
    base = [float(b) for b in run["base_lrs"]]
    mult = [MULT[n.split("_")[0]] for n in run["names"]]
    assert all(abs(b - 3e-4 * m) <= 1e-18 for b, m in zip(base, mult))
    by_update = {}
    for s in run["steps"]:
        by_update.setdefault(s["update"], []).append(s)
    assert sorted(by_update) == [0, 1] and all(len(v) >= 4 for v in by_update.values())
    for u, steps in by_update.items():
        assert all(s["dec"] == 1024 * (u + 1) for s in steps), "D is the learner's total_steps at the update"
        for s in steps:
            want = [b * check_restart.rate_share_decisions(s["step"], s["dec"], 0, 0, 1) for b in base]
            assert [float(x) for x in s["lr"]] == want, (u, s["step"])
            assert s["lr"] == steps[0]["lr"], "every optimizer step of an update has the same rate"
        assert abs(float(steps[0]["lr"][0]) / base[0] - 1 / 27) < 1e-12
    assert [float(x) for x in run["states"][-1]["lr"]] == [b * FLOOR for b in base]
    # the weights moved less than in the run at the full rate, but they moved
    off = real_updates["new"]["off"]["states"][-1]
    assert any(v != off[k] for k, v in run["states"][-1].items() if k.startswith("master/"))


# --------------------------------------------------------------------------- (b) on: the simulation


class _FakeUpdate:
    """
    ppo_update as far as the schedule sees it: k optimizer steps, the scheduler stepped after each
    one, as ppo_core.ppo_update does. Every step's rates are read before it and checked against the
    schedule worked out independently (check_restart.rate_share_decisions x the group's base rate).
    """

    def __init__(self, la, hold, total, warm):
        self.la, self.hold, self.total, self.warm = la, hold, total, warm
        self.k = 0
        self.rows = []          # one per update: (D, k, first optimizer step, policy rate at it, all steps the same, oracle misses)
        self.pi = next(i for i, pg in enumerate(la.opt.param_groups) if pg["name"].startswith("pi_"))
        self.base = list(la.sched.base_lrs)
        self.bad_mult = 0

    def __call__(self, net=None, opt=None, dataset=None, scheduler=None, mode="ppo", target_net=None, human=None, **cfg):
        groups = scheduler.optimizer.param_groups
        D, step0, k = self.la._lr_dec, scheduler.last_epoch, self.k
        first = [pg["lr"] for pg in groups]
        share0 = check_restart.rate_share_decisions(step0, D, self.warm, self.hold, self.total)
        if first != [b * share0 for b in self.base]:
            self.bad_mult += 1
        same, misses, pi, b_pi = True, 0, self.pi, self.base[self.pi]
        ref = None              # the rates of the update's first step past the ramp: every later step must have them
        for _ in range(k):
            s = scheduler.last_epoch
            lrs = [pg["lr"] for pg in groups]
            if s >= self.warm:
                if ref is None:
                    ref = lrs
                elif lrs != ref:
                    same = False
            if lrs[pi] != b_pi * check_restart.rate_share_decisions(s, D, self.warm, self.hold, self.total):
                misses += 1
            scheduler.step()
        self.rows.append((D, k, step0, first[pi], same, misses))
        return new_core.PPOUpdateStats(approx_kl=0.0, clip_frac=0.0, entropy=0.0, v_loss=0.0, pg_loss=0.0, total_loss=0.0, n_mb=k)


def _tiny_columns(n=8):
    z = torch.zeros(n)
    return (torch.zeros(n, 4), torch.zeros(n, dtype=torch.long), z.clone(), z.clone(), torch.randn(n), z.clone(), z.clone(),
            torch.arange(n, dtype=torch.long) // 4)


def _steps_rule(case):
    """Optimizer steps of update u (1, 2, ...) given whether the policy rate was still full at its start."""
    def kl_49_6(u):
        return 49 if u % 2 else 50
    if case == "kl-then-full":         # the from-scratch run: KL stops at the full rate, both epochs once it falls
        return lambda u, full: kl_49_6(u) if full else 96
    if case == "all-96":
        return lambda u, full: 96
    if case == "all-49.6":
        return lambda u, full: kl_49_6(u)
    raise ValueError(case)


def _simulate(la, case, through_update, monkeypatch, n_updates=N_UPDATES, hold=H, total=T):
    """
    n_updates updates of PER decisions each, the update sizes of `case`. through_update: every one
    through the learner's own _perform_update (ppo_update replaced by the fake); otherwise the hook
    alone (_apply_decision_rate, what _perform_update calls before ppo_update) and the fake.
    """
    fake = _FakeUpdate(la, hold, total, W)
    rule = _steps_rule(case)
    sched = la.sched

    async def go():
        for u in range(1, n_updates + 1):
            la.total_steps += PER
            fake.k = rule(u, sched.last_epoch < W or la.total_steps < hold)     # the rate is full for this update
            if through_update:
                la.dataset.add_steps(*_tiny_columns())
                before = la.update_idx
                await la._perform_update()
                assert la.update_idx == before + 1, "the update failed (the learner logs why)"
            else:
                la._apply_decision_rate()
                fake(scheduler=sched)

    lines = 0
    if through_update:
        monkeypatch.setattr(new_learner, "ppo_update", fake)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            asyncio.run(go())
        lines = buf.getvalue().count("[learner] upd=")
    else:
        asyncio.run(go())
    return fake, lines


@pytest.mark.parametrize("case,through_update", [("kl-then-full", True), ("all-96", False), ("all-49.6", False)])
def test_on_the_rate_lands_where_planned_in_decisions(case, through_update, monkeypatch):
    la = _learner(FLAGS)
    assert la._lr_by_decisions() and la.sched.lr_lambdas[0].__name__ == "lr_lambda_decisions"
    fake, lines = _simulate(la, case, through_update, monkeypatch)
    rows = fake.rows
    assert len(rows) == N_UPDATES and (not through_update or lines == N_UPDATES)
    assert fake.bad_mult == 0, "a group's rate is not its base rate (zone multiplier) x the schedule"
    assert sum(r[5] for r in rows) == 0, "an optimizer step's rate is not the schedule's"
    assert all(r[4] for r in rows), "the optimizer steps of an update do not all have the same rate"
    assert [r[0] for r in rows] == [PER * u for u in range(1, N_UPDATES + 1)], "D is the learner's total_steps"
    b_pi = fake.base[fake.pi]
    assert abs(b_pi - 3e-4) < 1e-18
    post = [r for r in rows if r[2] >= W]
    assert all(r[3] == b_pi for r in post if r[0] < H), "the full rate for every update with D < H"
    first_low = next(r for r in post if r[3] < b_pi)
    assert H <= first_low[0] < H + PER, first_low
    first_floor = next(r for r in rows if abs(r[3] / b_pi - 1 / 27) < 1e-12)
    assert first_floor[0] == min(r[0] for r in rows if r[0] >= T) and first_floor[0] - T < PER, first_floor
    assert all(r[3] == b_pi * FLOOR for r in rows if r[0] >= T)
    rates = [r[3] for r in post]
    assert all(a >= b for a, b in zip(rates, rates[1:])), "the rate never rises after the ramp"
    steps = sum(r[1] for r in rows)
    print(f"\n{case}: {steps:,} optimizer steps; the rate first falls at D={first_low[0]:,} (update {rows.index(first_low) + 1:,}, "
          f"optimizer step {first_low[2]:,}); 1/27 from D={first_floor[0]:,} (optimizer step {first_floor[2]:,})")
    del la


# --------------------------------------------------------------------------- (c) the bug


def _land_steps_schedule(fn, case, n_max=60_000):
    """The step schedule fn on the same simulation: (D where the rate first falls, D where it reaches 1/27)."""
    rule = _steps_rule(case)
    step, D, fall, floor = 0, 0, None, None
    for u in range(1, n_max + 1):
        D += PER
        share = fn(step)
        if step >= W and share < 1.0 and fall is None:
            fall = D
        if abs(share - 1 / 27) < 1e-12:
            floor = D
            break
        step += rule(u, step < W or share == 1.0)
    return fall, floor


def test_the_step_schedule_with_the_runs_original_flags_lands_far_from_the_plan():
    """
    The from-scratch run started PPO with --lr-hold-steps 800000 --lr-total-steps 1600000, planned as
    a hold to about 307M decisions (800,000 steps x 384). An update takes 49.6 optimizer steps while it
    stops early on KL (743 decisions a step), so the hold really ran to about 596M decisions: the run
    would have ended at 600M almost without an anneal. The anneal would have ended at about 903M (96
    steps an update once the rate fell), or 1.19B had the KL stops gone on.
    """
    la = _learner(["--lr-hold-steps", "800000", "--lr-total-steps", "1600000"])
    fn = la.sched.lr_lambdas[0]
    planned_hold = 800_000 * 384
    fall, floor = _land_steps_schedule(fn, "kl-then-full")
    assert 590_000_000 < fall < 600_000_000 and fall > 1.9 * planned_hold, fall
    assert 895_000_000 < floor < 910_000_000, floor
    fall49, floor49 = _land_steps_schedule(fn, "all-49.6")
    assert fall49 == fall and 1_180_000_000 < floor49 < 1_200_000_000, floor49
    fall96, floor96 = _land_steps_schedule(fn, "all-96")
    # only at 96 steps an update (384 decisions a step) does it land as planned
    assert abs(fall96 - (W + 800_000) * 384) < 2 * PER and abs(floor96 - 1_600_000 * 384) < 2 * PER, (fall96, floor96)
    print(f"\nstep schedule hold 800k / total 1.6M: planned hold end ~{planned_hold / 1e6:.0f}M; with KL stops then 96 steps: "
          f"falls at {fall / 1e6:.1f}M, 1/27 at {floor / 1e6:.1f}M; all 49.6: 1/27 at {floor49 / 1e6:.1f}M; all 96: "
          f"falls at {fall96 / 1e6:.1f}M, 1/27 at {floor96 / 1e6:.1f}M. Decisions schedule: falls in [357.0M, 357.04M), 1/27 at 600.0M+")
    del la


# --------------------------------------------------------------------------- (d) the ramp


def test_the_first_1000_optimizer_steps_ramp_exactly_as_now():
    off, on = _learner([]), _learner(FLAGS)
    f_off, f_on = off.sched.lr_lambdas[0], on.sched.lr_lambdas[0]
    for D in (0, PER, 1_000_000, H - 1):
        on._lr_dec = D
        assert [f_on(s) for s in range(0, 1_200)] == [f_off(s) for s in range(0, 1_200)], D
    assert [f_on(s) for s in (0, 499, 999, 1_000)] == [0.001, 0.5, 1.0, 1.0]
    # the rates a fresh learner starts with are the same in every group
    assert [pg["lr"] for pg in off.opt.param_groups] == [pg["lr"] for pg in on.opt.param_groups]
    # and the first update's hook leaves them where they are (D = 36,864 < H)
    on.total_steps = PER
    on._apply_decision_rate()
    assert [pg["lr"] for pg in on.opt.param_groups] == [pg["lr"] for pg in off.opt.param_groups]
    # past T during the ramp the two multiply
    on._lr_dec = T
    assert f_on(499) == 0.5 * FLOOR
    del off, on


# --------------------------------------------------------------------------- (e) save, rebuild, load


def _one_update(la, fake, k=96):
    la.total_steps += PER
    fake.k = k
    la.dataset.add_steps(*_tiny_columns())
    before = la.update_idx
    asyncio.run(la._perform_update())
    assert la.update_idx == before + 1


def test_a_save_mid_anneal_resumes_at_the_uninterrupted_rate(monkeypatch, capsys):
    STEP, DEC = 1_104_534, 480_017_408                    # in the anneal (H < D < T)
    run = _learner(FLAGS)
    run.sched.last_epoch = STEP - 1
    run.sched.step()
    run.update_idx, run.total_steps = 13_021, DEC - PER
    fake = _FakeUpdate(run, H, T, W)
    monkeypatch.setattr(new_learner, "ppo_update", lambda **kw: fake(**kw))
    _one_update(run, fake)                                  # an update at D = DEC: the learner's rate is now that of DEC
    assert run._lr_dec == DEC and run.total_steps == DEC
    uninterrupted = [pg["lr"] for pg in run.opt.param_groups]
    share = check_restart.rate_share_decisions(run.sched.last_epoch, DEC, W, H, T)
    assert 1 / 27 < share < 1 and uninterrupted == [b * share for b in run.sched.base_lrs]
    ck = tsu._checkpoint_in_memory(run)
    assert ck["total_steps"] == DEC and ck["run_cfg"]["learner"]["lr_hold_decisions"] == H
    assert ck["run_cfg"]["learner"]["lr_total_decisions"] == T

    monkeypatch.setattr(torch, "load", lambda *a, **k: ck)
    capsys.readouterr()
    back = _learner(FLAGS)
    back._load_checkpoint("held in memory")
    said = capsys.readouterr().out
    assert back._lr_dec == DEC and back.total_steps == DEC and back.sched.last_epoch == run.sched.last_epoch
    assert [pg["lr"] for pg in back.opt.param_groups] == uninterrupted, "the rate at the resume"
    assert back.sched.get_last_lr() == uninterrupted
    assert f"policy rate at the resume: {back._policy_lr():.3e}" in said and "schedule in decisions" in said
    assert f"{DEC:,} decisions of this phase" in said and "lr_hold_decisions=357000000" in said

    # the next update of each: the same rate at every optimizer step and after
    fake_back = _FakeUpdate(back, H, T, W)
    monkeypatch.setattr(new_learner, "ppo_update", lambda **kw: fake(**kw))
    _one_update(run, fake)
    monkeypatch.setattr(new_learner, "ppo_update", lambda **kw: fake_back(**kw))
    _one_update(back, fake_back)
    assert fake.rows[-1][:4] == fake_back.rows[-1][:4] and fake_back.rows[-1][4] and fake_back.rows[-1][5] == 0
    assert [pg["lr"] for pg in back.opt.param_groups] == [pg["lr"] for pg in run.opt.param_groups]
    assert fake.rows[-1][3] < uninterrupted[fake.pi], "the anneal goes on"

    # a phase change: D starts again at 0, the optimizer and the schedule afresh
    ck2 = dict(ck)
    ck2["run_cfg"] = json.loads(json.dumps(ck["run_cfg"]))
    ck2["run_cfg"]["learner"]["mode"] = "warmup"
    monkeypatch.setattr(torch, "load", lambda *a, **k: ck2)
    fresh = _learner(FLAGS)
    start = [pg["lr"] for pg in fresh.opt.param_groups]
    fresh._load_checkpoint("held in memory")
    assert fresh.total_steps == 0 and fresh._lr_dec == 0 and fresh.update_idx == 0 and fresh.sched.last_epoch == 0
    assert [pg["lr"] for pg in fresh.opt.param_groups] == start
    fake_fresh = _FakeUpdate(fresh, H, T, W)
    monkeypatch.setattr(new_learner, "ppo_update", lambda **kw: fake_fresh(**kw))
    _one_update(fresh, fake_fresh, k=49)
    assert fake_fresh.rows[-1][0] == PER and fake_fresh.rows[-1][5] == 0 and fake_fresh.bad_mult == 0
    assert fake_fresh.rows[-1][3] == fake_fresh.base[fake_fresh.pi] * (1.0 / 1000.0)    # the ramp's first step, at D = 36,864
    del run, back, fresh


def test_a_steps_checkpoint_resumed_with_the_decision_flags_takes_d_from_it(monkeypatch, capsys):
    """A checkpoint of a run without the flags (no lr_*_decisions keys in its run_cfg): D is its total_steps all the same."""
    run = _learner([])
    run.sched.last_epoch = 409_999
    run.sched.step()
    run.update_idx, run.total_steps = 8_200, 400_000_000
    ck = tsu._checkpoint_in_memory(run)
    for key in ("lr_hold_decisions", "lr_total_decisions"):
        ck["run_cfg"]["learner"].pop(key)
    monkeypatch.setattr(torch, "load", lambda *a, **k: ck)
    back = _learner(FLAGS)
    back._load_checkpoint("held in memory")
    share = check_restart.rate_share_decisions(410_000, 400_000_000, W, H, T)
    assert back._lr_dec == 400_000_000 and share < 1.0
    assert [pg["lr"] for pg in back.opt.param_groups] == [b * share for b in back.sched.base_lrs]
    capsys.readouterr()
    del run, back


# --------------------------------------------------------------------------- (f) train.py refuses


@pytest.mark.parametrize("argv,said", [
    (["--lr-hold-decisions", "357000000"], "go together"),
    (["--lr-total-decisions", "600000000"], "go together"),
    (["--lr-hold-decisions", "600000000", "--lr-total-decisions", "600000000"], "is not after"),
    (["--lr-hold-decisions", "600000000", "--lr-total-decisions", "357000000"], "is not after"),
    (["--lr-hold-decisions", "-1", "--lr-total-decisions", "600000000"], "must not be negative"),
    (FLAGS + ["--lr-hold-steps", "800000"], "give one schedule"),
    (FLAGS + ["--lr-total-steps", "1600000"], "give one schedule"),
    (FLAGS + ["--lr-hold-steps", "800000", "--lr-total-steps", "1600000"], "give one schedule"),
    (["--lr-hold-decisions", "3.5", "--lr-total-decisions", "600000000"], "not a whole number"),
    (["--lr-hold-decisions", "soon", "--lr-total-decisions", "600000000"], "not a number"),
])
def test_train_py_refuses_a_schedule_that_cannot_be(argv, said, capsys):
    with pytest.raises(SystemExit) as e:
        new_train.parse_args(argv)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert said in err, err[-400:]


def test_train_py_takes_the_flags():
    for argv, want in ((FLAGS, (H, T)), (["--lr-hold-decisions", "357e6", "--lr-total-decisions", "6e8"], (H, T)),
                       (["--lr-hold-decisions", "0", "--lr-total-decisions", "1"], (0, 1)),
                       (FLAGS + ["--lr", "1e-4"], (H, T))):
        lc = new_train.build_config(new_train.parse_args(argv)).learner
        assert (lc.lr_hold_decisions, lc.lr_total_decisions) == want, argv
        assert (lc.lr_hold_steps, lc.lr_total_steps) == (500_000, 1_000_000), "the step flags are untouched"
    base = new_config.RunConfig.default()
    cfg = new_train.build_config(new_train.parse_args(FLAGS))
    back = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, lr_hold_decisions=0, lr_total_decisions=0))
    assert back == base, "nothing else moved"
    with pytest.raises(ValueError):
        new_config.LearnerConfig(lr_hold_decisions=5, lr_total_decisions=0)
    with pytest.raises(ValueError):
        new_config.LearnerConfig(lr_hold_decisions=5, lr_total_decisions=5)


# --------------------------------------------------------------------------- (g) check_restart


def _config(tmp_path, extra):
    with open(os.path.join(REPO, "cloud", "run_big.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["repo_dir"] = REPO
    raw["phases"]["ppo"]["extra_train_args"] = list(extra)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return str(path)


def _check(tmp_path, monkeypatch, extra, ckpt, **kw):
    lines = []
    monkeypatch.setattr(check_restart, "read_checkpoint", lambda path: dict(ckpt))
    code = check_restart.check(_config(tmp_path, extra), str(tmp_path / "run"), checkpoint="held in memory", out=lines.append, **kw)
    return code, lines


def _ckpt(step, decisions, hold_d=None, total_d=None, mode="ppo", lr=3e-4):
    ck = {"update": decisions // PER, "total_steps": decisions, "optimizer_step": step, "mode": mode, "lr": lr,
          "warmup": 1_000, "hold": 500_000, "total": 1_000_000, "obs_settings": {"own_tera_type": True}}
    if hold_d is not None:
        ck.update(hold_decisions=hold_d, total_decisions=total_d)
    return ck


def test_check_restart_same_flags_pass_and_say_where_the_schedule_is(tmp_path, monkeypatch):
    for dec, where in ((100_007_424, "the hold ends in 256,992,576 decisions"), (480_017_408, "the anneal began 123,017,408 decisions ago"),
                       (610_000_000, "the anneal ended 10,000,000 decisions ago")):
        code, lines = _check(tmp_path, monkeypatch, FLAGS, _ckpt(1_104_534, dec, H, T))
        assert code == 0 and lines[-1].startswith("OK to start"), lines
        assert any(where in line for line in lines), (where, lines)
        assert any("in decisions of this phase: full rate up to 357,000,000 decisions, then down to 1/27 at 600,000,000" in line
                   for line in lines), lines
        said = [line for line in lines if line.startswith("the trainer will log:")][0]
        assert said.endswith("lr_hold_decisions=357000000 lr_total_decisions=600000000"), said
    # a checkpoint of another phase (the top-player start, mode imitation): afresh, nothing to compare
    code, lines = _check(tmp_path, monkeypatch, FLAGS, _ckpt(0, 5_148_000, mode="imitation"))
    assert code == 0 and any("of phase 'imitation'" in line and "start afresh" in line for line in lines), lines


def test_check_restart_a_changed_hold_is_refused_unless_allowed(tmp_path, monkeypatch):
    other = ["--lr-hold-decisions", "300000000", "--lr-total-decisions", str(T)]
    # in the anneal: the rate at the resume moves
    code, lines = _check(tmp_path, monkeypatch, other, _ckpt(1_104_534, 480_017_408, H, T))
    want = (check_restart.rate_share_decisions(1_104_534, 480_017_408, W, 300_000_000, T)
            / check_restart.rate_share_decisions(1_104_534, 480_017_408, W, H, T))
    assert code == 3 and any("DO NOT START" in line and f"factor of {want:.2f}" in line for line in lines), lines
    code, lines = _check(tmp_path, monkeypatch, other, _ckpt(1_104_534, 480_017_408, H, T), allow_rate_change=True)
    assert code == 0 and "was allowed" in lines[-1], lines
    # in the hold of both: the rate at the resume is the same, the schedule is not
    code, lines = _check(tmp_path, monkeypatch, other, _ckpt(300_000, 100_007_424, H, T))
    assert code == 3 and any("DO NOT START: this start changes the learning-rate schedule" in line for line in lines), lines
    assert not any("factor of" in line for line in lines), lines
    code, lines = _check(tmp_path, monkeypatch, other, _ckpt(300_000, 100_007_424, H, T), allow_rate_change=True)
    assert code == 0 and any(line.startswith("NOTE: this start changes the learning-rate schedule") for line in lines), lines


def test_check_restart_a_change_of_kind_is_refused_unless_allowed(tmp_path, monkeypatch):
    # a step-schedule checkpoint (no decision keys), in its hold, resumed with the decision flags
    code, lines = _check(tmp_path, monkeypatch, FLAGS, _ckpt(409_000, 157_056_000))
    assert code == 3 and any("written under a schedule in optimizer steps" in line for line in lines), lines
    code, lines = _check(tmp_path, monkeypatch, FLAGS, _ckpt(409_000, 157_056_000), allow_rate_change=True)
    assert code == 0 and lines[-1].startswith("OK to start (a change of the learning rate"), lines
    # past the step hold: the rate moves too (1/27 x ... against full)
    code, lines = _check(tmp_path, monkeypatch, FLAGS, _ckpt(1_104_534, 300_000_000))
    assert code == 3 and any("factor of" in line for line in lines), lines
    # a decisions checkpoint resumed without the flags (they were dropped from run.json)
    code, lines = _check(tmp_path, monkeypatch, [], _ckpt(300_000, 100_007_424, H, T))
    assert code == 3 and any("this start has one in optimizer steps" in line for line in lines), lines
    # and a plain step checkpoint with plain flags is as before
    code, lines = _check(tmp_path, monkeypatch, [], _ckpt(409_000, 157_056_000))
    assert code == 0 and lines[-1].startswith("OK to start"), lines


def test_check_restarts_share_is_the_learners_own():
    la = _learner(FLAGS)
    fn = la.sched.lr_lambdas[0]
    decs = [0, 1, PER, H - 1, H, H + 1, H + PER, 480_017_408, T - 1, T, T + 1, 2 * T] + list(range(0, 2 * T, 7_919_731))
    steps = [0, 1, 499, 998, 999, 1_000, 1_001, 50_000, 1_104_534]
    for D in decs:
        la._lr_dec = D
        for s in steps:
            assert check_restart.rate_share_decisions(s, D, W, H, T) == fn(s), (s, D)
            share = 1.0 if D < H else 1.0 / ((8 * min(1.0, (D - H) / (T - H)) + 1) ** 1.5)
            assert abs(fn(s) - (min(1.0, (s + 1) / W)) * share) < 1e-15, (s, D)
    assert check_restart.rate_share_decisions(5_000, T, W, H, T) == FLOOR
    assert new_learner.decision_rate_share(H, H, T) == 1.0 and new_learner.decision_rate_share(T, H, T) == FLOOR
    del la
