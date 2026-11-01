"""
Several inference actors and the other speed flags (infer_pool.py, train.py --infer-actors,
--infer-gpus, --infer-restarts, --infer-sync-s, --fast-sample).

    python -m pytest tests/test_infer_actors.py -q -p no:cacheprovider -rs

Held here:
  * the flags: absent, the options are the defaults and train.py takes its old path; bad values
    are refused before Ray starts; scratch_setup.py's opt-in reaches train.py through the
    supervisor's command line (extra_train_args) and trainer.env, and without it the config is
    what it was; the opt-in is refused for a release without infer_pool.py, and the config it
    writes passes check_restart only from the release it names;
  * the GPU plan: one actor is train.gpu_shares; shares never add up past a GPU, whatever order
    and whichever free GPU Ray places them on;
  * the worker plan: every worker has one actor, the pairs spread evenly, the same every time;
  * the telemetry: one actor's dict is its own; N are summed, the oldest version is model_version,
    and the line still parses for pipeline.py and bench/throughput.py;
  * the weights (the property the speed-up must not break): with the store's pull, every decision
    any actor serves at time t uses a version at least as new as the store's at t - sync_s, for 1,
    2 or 3 actors alike: exactly the bound the single actor has today. So two actors are never
    further apart than the versions published within one sync interval, and with a sync interval
    below the time between updates every actor loads every version;
  * --fast-sample: the log-probability is Categorical's to the bit for the same action, the
    actions follow the same distribution, a masked action is never drawn (not even with a uniform
    draw of exactly 0, which CUDA's torch.rand returns about once in 2^25), and infer_batch with it
    returns the same values, variances, dtypes and stored histories as without it;
  * a dead actor (Ray, with stand-in actors): the telemetry loop stops the run with exit code 1
    within seconds; with --infer-restarts Ray starts it again and the loop goes on.

The Ray tests start a small local Ray (CPU only) and are skipped where Ray cannot start. Nothing
here starts a server or a training run.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "tests"), os.path.join(REPO, "cloud"), os.path.join(REPO, "cloud", "v2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

import infer_pool  # noqa: E402
import inference  # noqa: E402
import ppo_core  # noqa: E402
from config import RunConfig  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU = DEV == "cuda"


# ----------------------------------------------------------------------------------------- flags

def test_no_flag_is_the_default_and_train_takes_its_old_path():
    import train
    args = train.parse_args([])
    assert infer_pool.from_args(args) == infer_pool.SpeedOptions() and infer_pool.from_args(args).is_default()
    assert infer_pool.SpeedOptions().actor_kwargs() == {}
    # The live run's own command line, with no speed flag: still the default.
    live = ["--obs-matchup", "--obs-movesets", "--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1",
            "--read-bonus", "0.3", "--read-anneal", "20000000:45000000", "--opponent-seats", "setupnet:0.15",
            "--boosted-row-weight", "2", "--lr-hold-steps", "800000", "--lr-total-steps", "1600000"]
    assert infer_pool.from_args(train.parse_args(live)).is_default()


def test_the_flags_and_their_checks():
    import train
    o = infer_pool.from_args(train.parse_args(["--infer-actors", "3", "--infer-gpus", "3", "--infer-restarts", "2",
                                                "--infer-sync-s", "1", "--fast-sample"]))
    assert o == infer_pool.SpeedOptions(infer_actors=3, infer_gpus=3, infer_restarts=2, infer_sync_s=1.0, fast_sample=True)
    assert not o.is_default()
    assert o.actor_kwargs() == {"fast_sample": True, "sync_every_s": 1.0}
    assert infer_pool.from_args(train.parse_args(["--no-fast-sample"])).is_default()
    # One actor with a speed flag is no longer the default path (the dead-actor check is on), and
    # its actor gets only the keyword arguments given.
    one = infer_pool.from_args(train.parse_args(["--fast-sample"]))
    assert not one.is_default() and one.infer_actors == 1 and one.actor_kwargs() == {"fast_sample": True}
    for bad in (["--infer-actors", "0"], ["--infer-gpus", "0"], ["--infer-restarts", "-1"], ["--infer-sync-s", "0"],
                ["--infer-sync-s", "nan"], ["--infer-actors", "two"]):
        with pytest.raises(SystemExit):
            train.parse_args(bad)


def test_the_actor_takes_no_new_behaviour_without_its_keywords():
    sig = inference.InferenceActor.__init__.__code__.co_varnames
    assert sig[:3] == ("self", "cfg", "weight_store")
    probe = object.__new__(inference.InferenceActor)
    assert inference.InferenceActor.SYNC_EVERY_S == 5.0
    assert not hasattr(probe, "_fast_sample")      # set in __init__ only


# ------------------------------------------------------------------------------- scratch_setup

def test_scratch_setup_opt_in(tmp_path):
    import scratch_setup
    import pipeline
    import train
    plain = scratch_setup.run_json(36)
    assert "env" not in plain["trainer"] and plain["train_defaults"]["extra_train_args"] == ["--obs-matchup", "--obs-movesets"]
    assert plain["repo_dir"] == scratch_setup.REL
    fast = scratch_setup.run_json(36, 3, "0,1,2,3", REPO)
    assert fast["trainer"]["env"] == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"} and fast["repo_dir"] == REPO
    # nothing else differs
    assert {k: v for k, v in fast.items() if k not in ("trainer", "train_defaults", "repo_dir")} == \
           {k: v for k, v in plain.items() if k not in ("trainer", "train_defaults", "repo_dir")}
    for name, raw in (("plain", plain), ("fast", fast)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        cfg = pipeline.load_config(str(path))
        cmd = pipeline.build_trainer_command(cfg, pipeline.Paths(str(tmp_path / name)), "ppo")
        i = cmd.index("train.py") if "train.py" in cmd else [j for j, t in enumerate(cmd) if t.endswith("train.py")][0]
        args = train.parse_args(cmd[i + 1:])
        o = infer_pool.from_args(args)
        if name == "plain":
            assert o.is_default() and cfg["trainer"]["env"] == {}
        else:
            assert o == infer_pool.SpeedOptions(infer_actors=3, infer_sync_s=1.0, fast_sample=True)
            assert cfg["trainer"]["env"] == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}


def test_scratch_setup_writes_the_speed_flags_only_for_a_release_that_has_them(tmp_path, monkeypatch, capsys):
    """The default release (v2-movesets) has no infer_pool.py and its train.py refuses the speed flags
    with exit code 2 at every start (five such starts and the run is marked failed). So --infer-actors
    is refused unless --rel names a release with infer_pool.py, and the config it then writes passes
    check_restart run from that release, while check_restart from any other release refuses it."""
    import check_restart
    import pipeline
    import scratch_setup
    import train
    old = tmp_path / "rel" / "v2-movesets"
    old.mkdir(parents=True)
    run = tmp_path / "run"
    monkeypatch.setattr(scratch_setup, "REL", str(old))
    monkeypatch.setattr(scratch_setup, "RUN", str(run))
    assert scratch_setup.release_problem(str(old), 0) is None              # no speed flag: any release
    assert "infer_pool.py" in scratch_setup.release_problem(str(old), 3)
    assert scratch_setup.release_problem(REPO, 3) is None

    def setup(*argv):
        monkeypatch.setattr(sys, "argv", ["scratch_setup.py", "--servers", "36", *argv])
        scratch_setup.main()

    for argv in (["--infer-actors", "3", "--trainer-gpus", "0,1,2,3", "--write"],
                 ["--infer-actors", "3", "--rel", str(old), "--write"]):
        with pytest.raises(SystemExit) as e:
            setup(*argv)
        assert e.value.code == 2 and "infer_pool.py" in capsys.readouterr().err
        assert not run.exists()                                             # refused before anything is written
    setup("--write")                                                        # no speed flag: the old release, as before
    assert json.loads((run / "run.json").read_text(encoding="utf-8"))["repo_dir"] == str(old)

    setup("--infer-actors", "3", "--trainer-gpus", "0,1,2,3", "--rel", REPO, "--write")
    raw = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert raw["repo_dir"] == os.path.abspath(REPO) and raw["trainer"]["env"] == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    assert raw["train_defaults"]["extra_train_args"][-5:] == ["--infer-actors", "3", "--fast-sample", "--infer-sync-s", "1"]
    assert f"{os.path.abspath(REPO)}/cloud/check_restart.py --config {run}/run.json" in capsys.readouterr().out
    # check_restart from this release: OK, with a checkpoint written under the same settings
    cfg = pipeline.load_config(str(run / "run.json"))
    cmd = pipeline.build_trainer_command(cfg, pipeline.Paths(str(run)), "ppo")
    rc = train.build_config(train.parse_args(cmd[2:]))
    ck = {"update": 6300, "total_steps": 200_000_000, "optimizer_step": 400_000, "mode": rc.learner.mode, "lr": rc.learner.lr,
          "warmup": rc.learner.lr_warmup_steps, "hold": rc.learner.lr_hold_steps, "total": rc.learner.lr_total_steps,
          "obs_settings": check_restart.obs_settings_of_config(rc)}
    monkeypatch.setattr(check_restart, "read_checkpoint", lambda path: dict(ck))
    lines = []
    assert check_restart.check(str(run / "run.json"), str(run), checkpoint="held in memory", out=lines.append) == 0, lines
    assert lines[-1].startswith("OK to start"), lines
    # ... and the same flags in a config naming another release: refused, whatever that release has
    (old / "infer_pool.py").write_text("", encoding="utf-8")
    raw["repo_dir"] = str(old)
    (run / "run.json").write_text(json.dumps(raw), encoding="utf-8")
    lines = []
    assert check_restart.check(str(run / "run.json"), str(run), checkpoint="held in memory", out=lines.append) == 2
    assert lines[-1].startswith("DO NOT START: the trainer would run") and str(old) in lines[-1], lines


# ------------------------------------------------------------------------------------ GPU plan

def test_one_actor_is_gpu_shares(monkeypatch):
    import train
    for split in ("0", "1"):
        monkeypatch.setenv("PSPPO_GPU_SPLIT", split)
        for g in (0, 1, 2, 4, 8):
            i, l = train.gpu_shares(g)
            assert infer_pool.plan_gpus(1, g) == ([i], l)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("g", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7])
def test_the_shares_always_fit(split, g, n):
    for k in [None] + list(range(1, 9)):
        try:
            shares, learner = infer_pool.plan_gpus(n, g, k, split=split)
        except ValueError:
            assert k is not None and (not split or g < 2 or k > g - 1)
            continue
        assert len(shares) == n and all(0 < s <= 1.0 for s in shares)
        assert all(abs(round(s * 10_000) - s * 10_000) < 1e-6 for s in shares)    # whole Ray resource units
        if split and g >= 2:
            assert learner == 1.0
            want = min(n, k or n, g - 1)
            assert sum(shares) <= want + 1e-9 and (n <= want) == (shares[0] == 1.0)
        else:
            assert learner == 0.65 and sum(shares) <= 0.35 + 1e-9
        # Place the learner first (as train.py does), then the actors, on any free GPU Ray might
        # choose: the fullest that fits, the emptiest, the first, a random one. All must fit.
        for rule in ("first", "best", "worst", "random"):
            rng = random.Random(n * 100 + g)
            free = [10_000] * max(g, 1)
            order = [learner] + shares
            for want_units in (round(x * 10_000) for x in order):
                fits = [j for j, f in enumerate(free) if f >= want_units]
                assert fits, (rule, n, g, k, split, shares, learner, free)
                j = {"first": fits[0], "best": min(fits, key=lambda j: free[j]), "worst": max(fits, key=lambda j: free[j]),
                     "random": rng.choice(fits)}[rule]
                free[j] -= want_units


def test_gpus_asked_for_without_the_split_are_refused():
    with pytest.raises(ValueError):
        infer_pool.plan_gpus(2, 4, 2, split=False)
    with pytest.raises(ValueError):
        infer_pool.plan_gpus(2, 4, 4, split=True)      # only 3 besides the learner's
    assert infer_pool.plan_gpus(3, 4, None, split=True) == ([1.0, 1.0, 1.0], 1.0)
    assert infer_pool.plan_gpus(2, 1, None, split=True) == ([0.175, 0.175], 0.65)
    assert infer_pool.plan_gpus(3, 1, None, split=False) == ([0.1166] * 3, 0.65)
    assert infer_pool.plan_gpus(4, 2, None, split=True) == ([0.25] * 4, 1.0)


# --------------------------------------------------------------------------------- worker plan

def test_workers_spread_evenly_and_the_same_every_time():
    assert infer_pool.assign_workers([4] * 36, 3) == [0, 1, 2] * 12
    assert infer_pool.assign_workers([4] * 36, 1) == [0] * 36
    rng = random.Random(3)
    for _ in range(200):
        pairs = [rng.randint(1, 9) for _ in range(rng.randint(1, 40))]
        n = rng.randint(1, 5)
        a = infer_pool.assign_workers(pairs, n)
        assert a == infer_pool.assign_workers(list(pairs), n) and all(0 <= k < n for k in a)
        load = [sum(p for p, k in zip(pairs, a) if k == j) for j in range(n)]
        if len(pairs) >= n:
            assert max(load) - min(load) <= max(pairs)


# ----------------------------------------------------------------------------------- telemetry

def test_merged_stats_parse_for_the_supervisor_and_the_throughput_report():
    import pipeline
    from bench import throughput
    one = {"total_requests": 10, "avg_batch_size": 28.5, "model_version": 4}
    assert infer_pool.merge_stats([one]) is one
    m = infer_pool.merge_stats([{"total_requests": 10, "avg_batch_size": 20.0, "model_version": 5},
                                {"total_requests": 32, "avg_batch_size": 30.0, "model_version": 4},
                                {"total_requests": 1, "avg_batch_size": 40.0, "model_version": 5}])
    assert m == {"total_requests": 43, "avg_batch_size": 30.0, "model_version": 4, "actors": 3, "versions": [5, 4, 5]}
    lstats = {"update": 5, "episodes": 9, "steps_in_dataset": 100, "total_steps": 4000}
    line = f"2026-10-17 10:00:00,000 [INFO] [Telemetry] ACTIVE: 512 | INFER: {m} | TRAIN: {lstats}"
    ev = pipeline.parse_trainer_line(line)
    assert ev["kind"] == "telemetry" and ev["infer"] == m and ev["total_steps"] == 4000
    t = throughput.parse_telemetry(line)
    assert t["total_requests"] == 43 and t["model_version"] == 4 and t["total_steps"] == 4000 and t["avg_batch_size"] == 30.0


def test_version_watch():
    w = infer_pool.VersionWatch(after_s=60)
    assert w.check([5, 5, 4], 0.0) == []
    assert w.check([6, 6, 4], 1.0) == []
    assert w.check([6, 6, 4], 60.0) == []
    said = w.check([7, 7, 4], 62.0)
    assert len(said) == 1 and "InferenceActor 2" in said[0]
    assert w.check([7, 7, 4], 70.0) == []                  # once
    assert w.check([7, 7, 7], 71.0) == []
    assert w.check([9, 9, 7], 72.0) == [] and w.check([9, 9, 7], 140.0)   # again after catching up


# ------------------------------------------------------------- weights: the staleness property

class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


class _Store:
    """WeightStore as a plain object: version v published at publish_times[v]."""

    def __init__(self, clock):
        self.clock = clock
        self.publish_times = []
        self.get_version = _Remote(lambda: self.version())
        self.get_weights = _Remote(lambda: {"v": self.version()})
        self.get_temp = _Remote(lambda: 1.0)

    def version(self, at=None):
        t = self.clock.now if at is None else at
        return sum(1 for p in self.publish_times if p <= t) - 1        # -1 before the first


class _Clock:
    now = 0.0

    def time(self):
        return self.now


def _bare_actor(store, sync_every_s):
    """An InferenceActor with only what _sync_weights reads (no model: the weights are a version tag)."""
    a = object.__new__(inference.InferenceActor)
    a.weight_store = store
    a.current_version = -1
    a.current_temp = 1.0
    a._last_sync_time = None
    a._sync_every_s = inference.InferenceActor.SYNC_EVERY_S if sync_every_s is None else sync_every_s
    loaded = []
    a.net = SimpleNamespace(load_state_dict=lambda w, strict=False: loaded.append(w["v"]))
    return a, loaded


def _simulate(monkeypatch, n_actors, sync_every_s, update_every, seconds=400.0, rate=60.0, seed=0):
    """
    n_actors actors serving decisions at random times (Poisson, `rate` a second each) while the
    store publishes a version every `update_every` seconds (with jitter). Every decision goes
    through InferenceActor._sync_weights, as infer_batch does first. Returns the decisions
    [(t, actor, version)], the store and each actor's loaded versions.
    """
    clock = _Clock()
    monkeypatch.setattr(inference, "time", clock)
    monkeypatch.setattr(inference, "ray", SimpleNamespace(get=lambda x, timeout=None: x))
    store = _Store(clock)
    rng = random.Random(seed)
    t, v = 0.3, 0
    while t < seconds:
        store.publish_times.append(t)
        t += update_every * rng.uniform(0.8, 1.2)
        v += 1
    actors = [_bare_actor(store, sync_every_s) for _ in range(n_actors)]
    events = []
    for k in range(n_actors):
        t = rng.uniform(0, 0.05)
        while t < seconds:
            events.append((t, k))
            t += rng.expovariate(rate)
    events.sort()
    decisions = []
    for t, k in events:
        clock.now = t
        a, _ = actors[k]
        a._sync_weights()
        decisions.append((t, k, a.current_version))
    return decisions, store, [loaded for _a, loaded in actors]


@pytest.mark.parametrize("n_actors", [1, 2, 3])
@pytest.mark.parametrize("sync_every_s,update_every", [(None, 7.7), (None, 4.4), (1.0, 4.4), (1.0, 7.7)])
def test_no_actor_serves_older_weights_than_one_actor_does_today(monkeypatch, n_actors, sync_every_s, update_every):
    s = inference.InferenceActor.SYNC_EVERY_S if sync_every_s is None else sync_every_s
    decisions, store, loaded = _simulate(monkeypatch, n_actors, sync_every_s, update_every, seed=n_actors)
    first_sync = {}
    for t, k, v in decisions:
        first_sync.setdefault(k, t)
        # the bound of the single actor today: the store's version of s seconds ago, or newer
        assert v >= store.version(t - s - 1e-9), (t, k, v, store.version(t - s))
        assert v <= store.version(t)
    # two actors' decisions one after the other (t1 <= t2): never further apart than the versions
    # published from t1 - s to t2, the spread one actor alone has between two of its own decisions
    for (t1, k1, v1), (t2, k2, v2) in zip(decisions, decisions[1:]):
        if k1 != k2:
            assert abs(v1 - v2) <= store.version(t2) - store.version(t1 - s - 1e-9), (t1, k1, v1, t2, k2, v2)
    # with a sync interval below the time between updates, every actor loads every version
    if s < 0.8 * update_every:
        for seq in loaded:
            assert seq == list(range(seq[0], seq[0] + len(seq))), seq[:20]
            assert seq[-1] == store.version(decisions[-1][0]) or seq[-1] >= store.version(decisions[-1][0] - s)


def test_the_bound_is_the_same_for_one_and_three_actors(monkeypatch):
    """The worst lag behind the store any decision has: the same with 3 actors as with 1."""
    def worst(n):
        d, store, _ = _simulate(monkeypatch, n, None, 4.4, seconds=600, seed=11)
        return max(store.version(t) - v for t, _k, v in d)
    w1, w3 = worst(1), worst(3)
    assert w1 <= 2 and w3 <= 2 and abs(w1 - w3) <= 1


# ----------------------------------------------------------------------------- --fast-sample

def _old_sample(logits, mask, temp):
    """infer_batch's old ending: the bad-row fix, then ppo_core.masked_sample."""
    mask_sub = mask.float().clone()
    bad = mask_sub.sum(dim=1) <= 0.49
    if bad.any():
        mask_sub[bad, 0] = 1.0
    return ppo_core.masked_sample(logits, mask_sub, temp=temp), mask_sub


def _valid(mask):
    v = mask > 0.5
    v[:, 0] |= mask.sum(dim=1) <= 0.49
    return v


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if GPU else []))
@pytest.mark.parametrize("temp", [1.0, 0.7, 1.5])
def test_fast_logprob_is_categoricals_to_the_bit(dev, temp):
    g = torch.Generator().manual_seed(5)
    B, A = 4096, 14
    logits = (torch.randn(B, A, generator=g) * 3).to(dev)
    mask = (torch.rand(B, A, generator=g) < 0.6).float().to(dev)
    mask[::97] = 0.0                                                # rows with no legal action
    valid = _valid(mask)
    act, logp = inference.fast_masked_sample(logits, valid, temp)
    assert act.dtype == torch.int64 and logp.dtype == torch.float32
    assert bool(valid.gather(1, act[:, None]).all())               # never a masked action
    assert bool((act[::97] == 0).all())                             # no legal action: action 0, as before
    (_a, _lp, _ent), mask_sub = _old_sample(logits, mask, temp)
    old_lp = torch.distributions.Categorical(logits=ppo_core.masked_logits(logits, mask_sub)).log_prob(act)
    assert torch.equal(logp, old_lp)


@pytest.mark.parametrize("temp", [1.0, 0.6])
def test_fast_actions_follow_the_same_distribution(temp):
    torch.manual_seed(0)
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0, 3.0, 0.0, -2.0, 1.5, 0.2, 0.1, -0.5, 2.5, -3.0, 0.9]])
    mask = torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.float32)
    n = 400_000
    act, _ = inference.fast_masked_sample(logits.expand(n, -1), _valid(mask.expand(n, -1)), temp)
    counts = torch.bincount(act, minlength=14).double()
    p = torch.softmax(ppo_core.masked_logits(logits / temp, mask), dim=-1)[0].double()
    assert counts[mask[0] == 0].sum() == 0
    keep = p > 1e-6
    expected = p[keep] * n
    chi2 = float((((counts[keep] - expected) ** 2) / expected).sum())
    dof = int(keep.sum()) - 1
    # a chi-square of dof degrees is beyond dof + 6 sqrt(2 dof) about once in 10^5 tries
    assert chi2 < dof + 6 * (2 * dof) ** 0.5, (chi2, dof)
    # and the old sampler, for comparison, on the same footing
    old = torch.distributions.Categorical(logits=ppo_core.masked_logits(logits / temp, mask).expand(n, -1)).sample()
    oc = torch.bincount(old, minlength=14).double()
    assert float((((oc[keep] - expected) ** 2) / expected).sum()) < dof + 6 * (2 * dof) ** 0.5


@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if GPU else []))
@pytest.mark.parametrize("temp", [1.0, 0.7])
@pytest.mark.parametrize("zeros", ["everywhere", "on the legal entries"])
def test_a_uniform_draw_of_zero_never_draws_a_masked_action(dev, temp, zeros, monkeypatch):
    """torch.rand on CUDA returns exactly 0.0 about once in 2^25 draws. Before the clamp, u = 0 on the
    one legal entry of a forced row (recharge, Struggle, a last forced switch) gave it a Gumbel of
    -inf and argmax drew a masked action (logp about -1e4). Here every such draw is 0."""
    g = torch.Generator().manual_seed(11)
    B, A = 2048, 14
    logits = (torch.randn(B, A, generator=g) * 3).to(dev)
    mask = (torch.rand(B, A, generator=g) < 0.6).float()
    forced = torch.arange(B) % 2 == 0                               # half the rows: one legal action
    only = torch.randint(0, A, (B,), generator=g)
    mask[forced] = 0.0
    mask[forced, only[forced]] = 1.0
    mask[1::97] = 0.0                                               # and rows with none (action 0)
    mask = mask.to(dev)
    valid = _valid(mask)
    real = torch.rand_like

    def rand_like(x, *a, **k):
        u = real(x, *a, **k)
        return torch.zeros_like(u) if zeros == "everywhere" else u.masked_fill(valid, 0.0)

    monkeypatch.setattr(torch, "rand_like", rand_like)
    act, logp = inference.fast_masked_sample(logits, valid, temp)
    monkeypatch.undo()
    assert bool(valid.gather(1, act[:, None]).all())                # never a masked action
    rows = forced.to(dev) & (mask.sum(dim=1) > 0.5)
    assert torch.equal(act[rows], only.to(dev)[rows])               # a forced row plays its one action
    assert bool(torch.isfinite(logp).all()) and float(logp.min()) > -1e3
    (_a, _lp, _ent), mask_sub = _old_sample(logits, mask, temp)
    assert torch.equal(logp, torch.distributions.Categorical(logits=ppo_core.masked_logits(logits, mask_sub)).log_prob(act))


def _small(cfg):
    return dataclasses.replace(
        cfg,
        learner=dataclasses.replace(cfg.learner, resume=False),
        infer=dataclasses.replace(cfg.infer, device=DEV),
        rollout=dataclasses.replace(cfg.rollout, target_concurrent_battles=8, rooms_per_pair=2),
    )


@pytest.fixture(scope="module")
def twin_actors():
    from bench import micro_ref
    mp = pytest.MonkeyPatch()
    mp.setattr(inference, "ray", SimpleNamespace(get=lambda x, timeout=None: x))
    saved = ppo_core.flex_attention
    torch.manual_seed(3)
    try:
        plain = inference.InferenceActor(_small(RunConfig.default()), None)
        ppo_core.flex_attention = saved
        fast = inference.InferenceActor(_small(RunConfig.default()), None, fast_sample=True)
    finally:
        ppo_core.flex_attention = saved
    if GPU:
        micro_ref.shared_compiled_flex_attention(ppo_core)     # both run the one compiled function
    fast.net.load_state_dict(plain.net.state_dict(), strict=True)
    plain._last_sync_time = fast._last_sync_time = 1e18
    obs, _ep, _kind = micro_ref.observations(n_min=4096)
    yield plain, fast, obs
    ppo_core.flex_attention = saved
    mp.undo()


def test_infer_batch_with_fast_sample_returns_the_same_values_and_histories(twin_actors):
    plain, fast, obs = twin_actors
    m0, m1 = plain.assembler.offsets["action_mask"]
    rng = np.random.default_rng(1)
    seen = []
    inner = fast.net.forward

    def spy(o, **kw):
        out = inner(o, **kw)
        seen.append(out[0])
        return out

    fast.net.forward = spy
    tags = [f"side{i}" for i in range(24)]
    steps = 40 if GPU else 6
    try:
        for step in range(steps):
            B = int(rng.integers(1, len(tags) + 1))
            pick = sorted(rng.choice(len(tags), size=B, replace=False))
            batch_tags = [tags[i] for i in pick]
            rows = obs[rng.integers(0, obs.shape[0], size=B)].copy()
            if step % 5 == 0:
                rows[0, m0:m1] = 0                                   # a row with no legal action
            a_plain = plain.infer_batch(batch_tags, rows)
            a_fast = fast.infer_batch(batch_tags, rows)
            for x, y in zip(a_plain, a_fast):
                assert x.dtype == y.dtype and x.shape == y.shape
            assert np.array_equal(a_plain[2], a_fast[2]) and np.array_equal(a_plain[3], a_fast[3])   # value, variance
            logits = seen[-1]
            mask = torch.from_numpy(rows[:, m0:m1]).to(DEV).float()
            mask_sub = mask.clone()
            mask_sub[mask_sub.sum(dim=1) <= 0.49, 0] = 1.0
            act = torch.from_numpy(a_fast[0]).to(DEV)
            assert bool((mask_sub.gather(1, act[:, None]) > 0.5).all())
            want = torch.distributions.Categorical(logits=ppo_core.masked_logits(logits, mask_sub)).log_prob(act)
            assert np.array_equal(want.cpu().numpy(), a_fast[1])
            if step % 5 == 0:
                assert a_fast[0][0] == 0 and a_plain[0][0] == 0
            if rng.random() < 0.2:
                gone = tags.pop(int(rng.integers(0, len(tags))))
                plain.clear_cache(gone), fast.clear_cache(gone)
                tags.append(f"side{100 + step}")
        for tag in tags:
            rp, rf = plain._kv_slot.get(tag), fast._kv_slot.get(tag)
            assert (rp is None) == (rf is None)
            if rp is not None:
                assert int(plain._kv_len[rp]) == int(fast._kv_len[rf])
                assert torch.equal(plain._kv_pool[:, rp], fast._kv_pool[:, rf])
    finally:
        fast.net.forward = inner


def _head_inference():
    """
    inference.py of the commit this work started from (e429f6f), loaded as a module of its own:
    from git, or from the file $PSPPO_SPEED_HEAD names (a copy, for a box without the repository).
    """
    import importlib.util
    import subprocess
    import tempfile
    path = os.environ.get("PSPPO_SPEED_HEAD")
    if not path:
        try:
            text = subprocess.run(["git", "-C", REPO, "show", "e429f6f:inference.py"], capture_output=True, text=True,
                                  check=True).stdout
        except Exception as e:      # noqa: BLE001
            pytest.skip(f"no copy of the old inference.py: {e!r} (set PSPPO_SPEED_HEAD)")
        path = os.path.join(tempfile.mkdtemp(), "inference_head.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    spec = importlib.util.spec_from_file_location("inference_head", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inference_head"] = mod             # its dataclass looks its module up there
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "fast_masked_sample")
    return mod


def test_without_its_keywords_the_actor_is_the_old_one_bit_for_bit(monkeypatch):
    from bench import micro_ref
    old_inf = _head_inference()
    fake_ray = SimpleNamespace(get=lambda x, timeout=None: x)
    monkeypatch.setattr(old_inf, "ray", fake_ray)
    monkeypatch.setattr(inference, "ray", fake_ray)
    saved = ppo_core.flex_attention
    torch.manual_seed(9)
    try:
        old = old_inf.InferenceActor(_small(RunConfig.default()), None)
        ppo_core.flex_attention = saved
        new = inference.InferenceActor(_small(RunConfig.default()), None)
    finally:
        ppo_core.flex_attention = saved
    if GPU:
        micro_ref.shared_compiled_flex_attention(ppo_core)
    try:
        new.net.load_state_dict(old.net.state_dict(), strict=True)
        old._last_sync_time = new._last_sync_time = 1e18
        obs, _ep, _kind = micro_ref.observations(n_min=4096)
        m0, m1 = new.assembler.offsets["action_mask"]
        rng = np.random.default_rng(4)
        tags = [f"side{i}" for i in range(16)]
        for step in range(30 if GPU else 5):
            B = int(rng.integers(1, len(tags) + 1))
            batch_tags = [tags[i] for i in sorted(rng.choice(len(tags), size=B, replace=False))]
            rows = obs[rng.integers(0, obs.shape[0], size=B)].copy()
            if step % 4 == 0:
                rows[0, m0:m1] = 0
            torch.manual_seed(1000 + step)
            a = old.infer_batch(batch_tags, rows)
            torch.manual_seed(1000 + step)
            b = new.infer_batch(batch_tags, rows)
            for x, y in zip(a, b):
                assert x.dtype == y.dtype and np.array_equal(x, y)
        assert torch.equal(old._kv_pool, new._kv_pool) and torch.equal(old._kv_len, new._kv_len)
    finally:
        ppo_core.flex_attention = saved


# ------------------------------------------------------------------------- Ray: a dead actor

def _ray_or_skip():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        # The stand-in actors live in this file: Ray's worker processes must be able to import it.
        os.environ["PYTHONPATH"] = os.pathsep.join([REPO, os.path.join(REPO, "tests")] +
                                                   ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
        try:
            ray.init(address="local", num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False,
                     ignore_reinit_error=True)
        except Exception as e:      # noqa: BLE001
            pytest.skip(f"Ray cannot start here: {e!r}")
    return ray


class _StubInfer:
    def __init__(self, version=3):
        self.version = version
        self.calls = 0

    def get_stats(self):
        self.calls += 1
        return {"total_requests": 10 * self.calls, "avg_batch_size": 20.0, "model_version": self.version}

    def pid(self):
        return os.getpid()


class _StubLearner:
    def get_stats(self):
        return {"update": 3, "episodes": 1, "steps_in_dataset": 0, "total_steps": 100}


class _StubWorker:
    def heartbeat(self):
        return {"active_battles_worker": 4}

    async def run(self):                      # async, as RolloutWorker.run: heartbeats are answered meanwhile
        await asyncio.sleep(3600)


@pytest.fixture(scope="module")
def ray_mod():
    ray = _ray_or_skip()
    yield ray
    ray.shutdown()


def test_gather_stats_alive_and_dead(ray_mod):
    ray = ray_mod
    A = ray.remote(num_cpus=0)(_StubInfer)
    actors = [A.remote(5), A.remote(6)]
    answers, dead, silent = infer_pool.gather_stats([a.get_stats.remote() for a in actors], 30)
    assert [s["model_version"] for s in answers] == [5, 6] and dead == [] and silent == []
    ray.kill(actors[1])
    time.sleep(1.0)
    answers, dead, silent = infer_pool.gather_stats([a.get_stats.remote() for a in actors], 30)
    assert answers[0]["model_version"] == 5 and answers[1] is None and [k for k, _ in dead] == [1] and silent == []


def _loop_args():
    return SimpleNamespace(minutes=0.0, stop_file=None)


def _run_loop_until_exit(train, infers, learner, workers, run_refs, pool, script):
    """
    train._telemetry_loop in an event loop of its own, as main() runs it, with `script` (killing
    actors) in a thread. -> (exit code, seconds from the script's last kill to the exit).
    SystemExit leaves asyncio's loop directly, as it leaves main() in train.py.
    """
    import threading
    marks = {}
    th = threading.Thread(target=script, args=(marks,), daemon=True)
    th.start()
    with pytest.raises(SystemExit) as e:
        asyncio.run(asyncio.wait_for(train._telemetry_loop(_loop_args(), time.time(), infers, learner, workers, run_refs, pool=pool), 90))
    return e.value.code, time.time() - marks.get("last_kill", time.time()), marks


def test_a_dead_inference_actor_stops_the_run_at_once(ray_mod, caplog):
    import train
    ray = ray_mod
    A = ray.remote(num_cpus=0)(_StubInfer)
    infers = [A.remote(), A.remote()]
    learner = ray.remote(num_cpus=0)(_StubLearner).remote()
    W = ray.remote(num_cpus=0)(_StubWorker)
    workers = [W.remote() for _ in range(4)]
    run_refs = [w.run.remote() for w in workers]
    pool = train.PoolWatch([0, 1, 0, 1], [11200, 11201, 11202, 11203])

    def script(marks):
        time.sleep(6.0)
        marks["last_kill"] = time.time()
        ray.kill(infers[1])

    with caplog.at_level("INFO", logger="train"):
        code, took, _ = _run_loop_until_exit(train, infers, learner, workers, run_refs, pool, script)
    assert code == 1 and took < 15, took
    text = caplog.text
    assert "InferenceActor 1 is dead" in text and "[11201, 11203]" in text
    assert "'actors': 2" in text and "'versions': [3, 3]" in text          # the merged telemetry line came first
    for w in workers + infers[:1] + [learner]:
        ray.kill(w)


def test_with_restarts_ray_starts_it_again_and_the_run_goes_on(ray_mod, caplog):
    import train
    ray = ray_mod
    opts = {k: v for k, v in infer_pool.actor_options(0.0, 1).items() if k != "num_gpus"}
    assert opts == {"max_restarts": 1, "max_task_retries": -1}
    A = ray.remote(_StubInfer).options(num_cpus=0, **opts)
    infers = [A.remote(), A.remote()]
    learner = ray.remote(num_cpus=0)(_StubLearner).remote()
    W = ray.remote(num_cpus=0)(_StubWorker)
    workers = [W.remote()]
    run_refs = [w.run.remote() for w in workers]
    pool = train.PoolWatch([0], [11200])
    pid0 = ray.get(infers[1].pid.remote())

    def script(marks):
        time.sleep(3.0)
        ray.kill(infers[1], no_restart=False)            # dies once: Ray starts it again
        time.sleep(2.0)                                  # (the kill is asynchronous)
        marks["pid1"] = ray.get(infers[1].pid.remote(), timeout=60)   # this call waits for the new process
        marks["lines_before"] = caplog.text.count("[Telemetry]")
        time.sleep(9.0)
        marks["lines_after"] = caplog.text.count("[Telemetry]")
        marks["last_kill"] = time.time()
        ray.kill(infers[1], no_restart=False)            # dies again: no restart left

    with caplog.at_level("INFO", logger="train"):
        code, took, marks = _run_loop_until_exit(train, infers, learner, workers, run_refs, pool, script)
    assert marks["pid1"] != pid0 and code == 1 and took < 15, (marks, pid0, took)
    assert marks["lines_after"] >= marks["lines_before"] + 1          # the run went on after the first death
    assert caplog.text.count("is dead") == 1 and "InferenceActor 1 is dead" in caplog.text
    for w in workers + infers[:1] + [learner]:
        ray.kill(w)
