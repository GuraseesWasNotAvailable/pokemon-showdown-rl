"""
The learner's side of the speed work, each change held to the code before it:

  * advantages (ppo_core.gae_from_episode) on NumPy float32 scalars: the same bits as the
    torch loop, on thousands of episodes
  * minibatches cut on the GPU: the same rows in the same order as on the CPU, and from the
    CPU again when the GPU has no room for the data after all (made up, and for real)
  * a whole update (ppo_core.ppo_update) in imitation, warmup and PPO: after it every weight,
    every optimizer moment and every loss figure equals what the old code produces from the
    same start, the same data and the same seed. This covers the attention mask, the data on
    the GPU and the accuracy figure together
  * the imitation accuracy figure against a plain recomputation
  * the learner: the "[learner] upd=" line with its two new keys as pipeline.py and
    bench/throughput.py read it, the checkpoint made durable before its rename, the learning
    rate a resumed run takes its first step with, and the schedule itself, which must be the
    one the old code builds at every optimizer step (its length is not a speed change's to move)
  * a checkpoint written by the old code ($PSPPO_OLD_CKPT): loads with nothing missing, gives
    the old code's outputs, and resumes into each of the three phases

    python -m pytest tests/test_speed_update.py -q -p no:cacheprovider

"Old code" is the reference commit of bench/micro_ref.py. Whole updates need a GPU (the
attention has no backward pass on a CPU) and are skipped without one. Nothing here starts
Ray, a server or a training run, and nothing is removed: the one small checkpoint a test
writes stays in $PSPPO_TEST_DIR (default: a new directory in the system temp folder).
"""
from __future__ import annotations

import asyncio
import dataclasses
import io
import os
import random
import sys
import tempfile
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import config as new_config  # noqa: E402
import learner as new_learner  # noqa: E402
import ppo_core as new_core  # noqa: E402
import worker as new_worker  # noqa: E402
from bench import micro_ref  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU = DEV == "cuda"
needs_gpu = pytest.mark.skipif(not GPU, reason="flex_attention has no backward on a CPU")
OLD_CKPT = os.environ.get("PSPPO_OLD_CKPT", "")
TEST_ROOT = os.environ.get("PSPPO_TEST_DIR") or tempfile.mkdtemp(prefix="psppo_speed_test_")
os.makedirs(TEST_ROOT, exist_ok=True)


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


class FakeStore:
    """The weight store actor, as far as the learner uses it."""

    def __init__(self):
        self.updates, self.temps = [], []
        self.update = _Remote(lambda weights, version: self.updates.append((version, len(weights))))
        self.set_temp = _Remote(self.temps.append)


@pytest.fixture(scope="module")
def ref():
    try:
        return micro_ref.Reference.export()
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(f"no reference code to compare with: {e}")


@pytest.fixture(scope="module")
def old(ref):
    """The old modules, sharing one compiled flex_attention with the new ones."""
    core, lrn, cfg = ref.load("ppo_core"), ref.load("learner"), ref.load("config")
    assert lrn.ppo_core is core and core is not new_core and not hasattr(core, "update_data_fits")
    if GPU:
        micro_ref.shared_compiled_flex_attention(core, new_core)
    # Both learners wrap flex_attention in torch.compile once per process. It is compiled above.
    core._learner_flex_compiled = True
    new_core._learner_flex_compiled = True
    return type("Old", (), {"core": core, "learner": lrn, "config": cfg, "ref": ref})


@pytest.fixture(scope="module")
def data():
    obs, ep, kind = micro_ref.observations(n_min=4096)
    return obs, ep, kind


def _learner_cfg(config_mod, mode: str, **over):
    cfg = config_mod.RunConfig.default()
    lc = dict(mode=mode, device=DEV, minibatch_size=256, batch_seq_len=256, grad_accum_steps=3, steps_per_update=1024,
              resume=False, ckpt_dir=os.path.join(TEST_ROOT, "ckpt_unused"))
    lc.update(over)
    return dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **lc))


def _make_learner(learner_mod, core_mod, cfg, store=None):
    """A LearnerActor without Ray and without its asyncio loops: the state __init__ sets, then the lazy build."""
    la = object.__new__(learner_mod.LearnerActor)
    la.run_cfg, la.cfg, la.weight_store = cfg, cfg.learner, store or FakeStore()
    la.net = la.opt = la.sched = None
    la.dataset = core_mod.AsyncEpisodeDataset(act_dim=cfg.env.act_dim, device="cpu")
    la.update_idx = la.total_episodes = la.total_steps = la.ep_counter = 0
    la._init_if_needed()
    return la


def _old_learner(old, mode: str, **over):
    with old.ref.active():
        return _make_learner(old.learner, old.core, _learner_cfg(old.config, mode, **over))


def _new_learner(mode: str, **over):
    return _make_learner(new_learner, new_core, _learner_cfg(new_config, mode, **over))


def _update_data(obs_np, ep_np, n: int, seed: int):
    """n real steps with plausible targets: a legal action, its old log-probability, advantage, return."""
    rng = np.random.default_rng(seed)
    from obs_assembler import ObservationAssembler
    m0, m1 = ObservationAssembler().offsets["action_mask"]
    obs = torch.from_numpy(np.ascontiguousarray(obs_np[:n]))
    mask = obs_np[:n, m0:m1].astype(np.float32) > 0.5
    act = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in mask], dtype=np.int64)
    logp = (-np.log(np.maximum(mask.sum(axis=1), 1.0))).astype(np.float32)
    adv = rng.standard_normal(n).astype(np.float32)
    ret = np.clip(rng.standard_normal(n) * 0.5, -1.4, 1.4).astype(np.float32)
    ep = torch.from_numpy(np.ascontiguousarray(ep_np[:n])).long()
    z = torch.zeros(n)
    return obs, torch.from_numpy(act), torch.from_numpy(logp), z, torch.from_numpy(adv), torch.from_numpy(ret), z, ep


# --------------------------------------------------------------------------- advantages


def _episodes(n: int, seed: int):
    rng = np.random.default_rng(seed)
    for i in range(n):
        T = int(rng.choice([1, 1, 2, 3, 5, 17, 40, 41, 64, 65, 130, 300]))
        scale = float(rng.choice([1e-6, 0.01, 0.3, 1.0, 3.0, 1e4]))
        rewards = (rng.standard_normal(T) * scale * 0.1).astype(np.float32)
        if rng.random() < 0.3:
            rewards[:] = 0.0
        rewards[-1] += np.float32(rng.choice([-1.0, 0.0, 1.0]))
        values = (rng.standard_normal(T) * scale).astype(np.float32)
        variances = (rng.random(T) * scale * scale).astype(np.float32)
        if rng.random() < 0.2:
            variances[:] = 0.0
        dones = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0
        if rng.random() < 0.1:
            dones[rng.integers(0, T)] = 1.0          # an episode cut in the middle
        if rng.random() < 0.1:
            dones[-1] = 0.0                          # not ended: the value after the last step counts
        yield (torch.from_numpy(rewards), torch.from_numpy(values), torch.from_numpy(dones), torch.from_numpy(variances),
               float(rng.choice([0.999, 0.99, 1.0, 0.9])), float(rng.choice([0.75, 0.95, 0.0, 1.0])),
               float(rng.choice([0.0, 0.0, 0.37, -1.2])))


def test_advantages_are_bit_identical(old):
    n = 0
    for rewards, values, dones, variances, gamma, lam, last in _episodes(4000, 1):
        for var in (variances, None):                # the learner's default uses the variances
            kw = dict(gamma=gamma, lam=lam, last_value=last, variances=var, lam_min=0.55, lam_max=0.95, lam_eps=1e-8)
            a_adv, a_ret = old.core.gae_from_episode(rewards, values, dones, **kw)
            b_adv, b_ret = new_core.gae_from_episode(rewards, values, dones, **kw)
            assert a_adv.dtype == b_adv.dtype == torch.float32 and a_ret.dtype == b_ret.dtype
            assert micro_ref.same_bits(a_adv, b_adv) and micro_ref.same_bits(a_ret, b_ret), (n, var is None, gamma, lam, last)
            n += 1
    assert n == 8000


def test_advantages_on_other_inputs_take_the_torch_loop(old):
    """float64 inputs (and anything not plain float32 on the CPU) must give what the old loop gives."""
    for rewards, values, dones, variances, gamma, lam, last in _episodes(60, 2):
        for cast in (lambda t: t.double(), lambda t: t.to(DEV)):
            r, v, d, var = cast(rewards), cast(values), cast(dones), cast(variances)
            for vv in (var, None):
                kw = dict(gamma=gamma, lam=lam, last_value=last, variances=vv)
                a = old.core.gae_from_episode(r, v, d, **kw)
                b = new_core.gae_from_episode(r, v, d, **kw)
                assert a[0].dtype == b[0].dtype and a[0].device == b[0].device
                assert micro_ref.same_bits(a[0], b[0]) and micro_ref.same_bits(a[1], b[1])


def test_advantage_loop_is_faster():
    T = 40
    r, v, d, var = (torch.randn(T), torch.randn(T), torch.zeros(T), torch.rand(T))
    t0 = time.perf_counter()
    for _ in range(200):
        new_core.gae_from_episode(r, v, d, gamma=0.999, lam=0.75, variances=var)
    fast = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        new_core.gae_from_episode(r.double(), v.double(), d.double(), gamma=0.999, lam=0.75, variances=var.double())   # the torch loop
    slow = (time.perf_counter() - t0) / 200
    print(f"\ngae, 40 steps: {fast * 1e6:.0f} us with NumPy scalars, {slow * 1e6:.0f} us with torch scalars")
    assert fast < slow


# --------------------------------------------------------------------------- minibatches


def test_minibatches_from_the_gpu_are_the_same_rows(old, data):
    obs_np, ep_np, _kind = data
    n = 2048
    cols = _update_data(obs_np, ep_np, n, seed=4)
    a = old.core.AsyncEpisodeDataset(14, "cpu")
    b = new_core.AsyncEpisodeDataset(14, "cpu")
    c = new_core.AsyncEpisodeDataset(14, "cpu")
    for ds in (a, b, c):
        ds.add_steps(*cols)
    assert b.tensorize()[0].data_ptr() == cols[0].data_ptr(), "one chunk should be handed on as it is, not copied"
    for epoch in range(2):
        batches = []
        for ds, dev in ((a, None), (b, None), (c, DEV)):
            torch.manual_seed(100 + epoch)
            kw = {} if ds is a else {"device": dev}
            batches.append(list(ds.iter_minibatches(256, shuffle_episodes=True, **kw)))
        assert len(batches[0]) == len(batches[1]) == len(batches[2]) == n // 256
        for x, y, z in zip(*batches):
            for i in range(8):
                assert x[i].dtype == y[i].dtype == z[i].dtype
                assert torch.equal(x[i], y[i]) and torch.equal(x[i], z[i].cpu()), (epoch, i)
                assert z[i].device.type == DEV
    assert c._device_cache is not None
    c.clear()
    assert c._device_cache is None


def test_data_goes_to_the_gpu_only_when_it_is_small_next_to_free_memory(monkeypatch, data):
    obs_np, ep_np, _kind = data
    ds = new_core.AsyncEpisodeDataset(14, "cpu")
    ds.add_steps(*_update_data(obs_np, ep_np, 512, seed=5))
    assert new_core.update_data_fits(ds, "cpu") is False
    if not GPU:
        return
    need = sum(d.numel() * d.element_size() for d in ds.tensorize())
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev=None: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (need * 10, need * 100))
    assert new_core.update_data_fits(ds, DEV) is True
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (need * 10 - 1, need * 100))
    assert new_core.update_data_fits(ds, DEV) is False

    def broken(dev=None):
        raise RuntimeError("no such device")
    monkeypatch.setattr(torch.cuda, "mem_get_info", broken)
    assert new_core.update_data_fits(ds, DEV) is False


def _gpu_is_full(*_a, **_k):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory (made up by the test)")


@pytest.mark.parametrize("fails", ["before anything is moved", "after the data is over"])
def test_minibatches_are_cut_on_the_cpu_when_the_gpu_refuses_the_data(monkeypatch, data, fails):
    """
    The rule said the data fits, and then the move runs out of memory (the GPU is shared).
    The pass must go on from the CPU with the same rows in the same order, as it did before
    the data was ever held on the GPU. Before the fix the error left iter_minibatches, the
    update was dropped and its 36,864 decisions with it.
    """
    obs_np, ep_np, _kind = data
    n = 1024
    cols = _update_data(obs_np, ep_np, n, seed=6)
    plain = new_core.AsyncEpisodeDataset(14, "cpu")
    ds = new_core.AsyncEpisodeDataset(14, "cpu")
    for d in (plain, ds):
        d.add_steps(*cols)
    real, calls = ds.on_device, []

    def refuse(device):
        calls.append(device)
        if fails == "after the data is over":
            real(device)                                   # the data is over and kept, and the next allocation fails
            assert ds._device_cache is not None
        _gpu_is_full()

    monkeypatch.setattr(ds, "on_device", refuse)
    for epoch in range(2):
        torch.manual_seed(300 + epoch)
        want = list(plain.iter_minibatches(256, shuffle_episodes=True))
        torch.manual_seed(300 + epoch)
        got = list(ds.iter_minibatches(256, shuffle_episodes=True, device=DEV))
        assert len(want) == len(got) == n // 256
        for x, y in zip(want, got):
            for i in range(8):
                assert y[i].device.type == "cpu" and x[i].dtype == y[i].dtype and torch.equal(x[i], y[i]), (epoch, i)
        assert ds._device_cache is None, "a copy the pass does not use was left on the device"
    assert len(calls) == 2, "each pass asks again: the memory may have come back"


@needs_gpu
def test_a_real_out_of_memory_on_the_move_is_survived(data, caplog):
    """
    The same with torch's own error. This process is capped at the GPU memory it already
    holds and what is spare inside that is filled, so the move of 140 MB of observations
    fails for real, while update_data_fits, which looks at the whole card, still says yes.
    """
    import gc
    import logging
    obs_np, ep_np, _kind = data
    reps = -(-8192 // obs_np.shape[0])
    obs = torch.from_numpy(np.ascontiguousarray(np.concatenate([obs_np] * reps)[:8192]))
    n = obs.shape[0]
    z = torch.zeros(n)
    cols = (obs, torch.zeros(n, dtype=torch.long), z, z, z, z, z, torch.arange(n) // 37)
    plain = new_core.AsyncEpisodeDataset(14, "cpu")
    ds = new_core.AsyncEpisodeDataset(14, "cpu")
    for d in (plain, ds):
        d.add_steps(*cols)
    assert obs.numel() * obs.element_size() > 130 * 2**20

    dev = torch.device("cuda", torch.cuda.current_device())
    gc.collect()
    torch.cuda.empty_cache()
    torch.zeros(1, device=dev)                             # the context is up and torch holds something
    total = torch.cuda.get_device_properties(dev).total_memory
    ballast = []
    torch.cuda.set_per_process_memory_fraction(torch.cuda.memory_reserved(dev) / total, dev)
    try:
        for mib in (64, 16, 4, 1):                         # use up what is spare inside the memory torch already holds
            while True:
                try:
                    ballast.append(torch.empty(mib * 2**20, dtype=torch.uint8, device=dev))
                except torch.cuda.OutOfMemoryError:
                    break
        fits = new_core.update_data_fits(ds, dev)
        before = torch.cuda.memory_allocated(dev)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            ds.on_device(dev)
        assert ds._device_cache is None
        with caplog.at_level(logging.WARNING, logger="ppo_core"):
            torch.manual_seed(310)
            got = list(ds.iter_minibatches(256, shuffle_episodes=True, device=dev))
        torch.manual_seed(310)
        want = list(plain.iter_minibatches(256, shuffle_episodes=True))
        assert len(got) == len(want) == n // 256
        for x, y in zip(want, got):
            for i in range(8):
                assert y[i].device.type == "cpu" and torch.equal(x[i], y[i])
        assert ds._device_cache is None
        assert torch.cuda.memory_allocated(dev) == before, "part of the data was left on the GPU"
        assert any("cutting the minibatches on the CPU" in r.getMessage() for r in caplog.records)
    finally:
        del ballast
        torch.cuda.set_per_process_memory_fraction(1.0, dev)
        torch.cuda.empty_cache()
    print(f"\nreal out-of-memory on the move: survived (the rule had said the data fits: {fits})")
    moved = ds.on_device(dev)                              # with the cap lifted the same call goes through
    assert moved[0].device.type == "cuda" and torch.equal(moved[0].cpu(), obs)
    ds.clear()
    del moved
    torch.cuda.empty_cache()


@needs_gpu
def test_an_update_goes_through_when_the_gpu_refuses_its_data(monkeypatch, data):
    """The same failure inside a whole update: it finishes, with the weights of an update whose data stayed on the CPU."""
    obs_np, ep_np, _kind = data
    cols = _update_data(obs_np, ep_np, 1536, seed=12)
    a, b = _new_learner("imitation"), _new_learner("imitation")
    _copy_state(a, b)
    sa = _run_update(new_core, a, cols, seed=60, data_on_device=False)
    want = _state(a)
    calls = []

    def refuse(self, device):
        calls.append(device)
        _gpu_is_full()

    monkeypatch.setattr(new_core.AsyncEpisodeDataset, "on_device", refuse)
    sb = _run_update(new_core, b, cols, seed=60, data_on_device=True)
    got = _state(b)
    assert len(calls) == 2, "one attempt per epoch"
    assert [k for k in want if not micro_ref.same_bits(want[k], got[k])] == []
    for f in ("approx_kl", "clip_frac", "entropy", "v_loss", "pg_loss", "total_loss", "n_mb", "acc"):
        assert getattr(sa, f) == getattr(sb, f), f
    assert sb.n_mb == 12
    del a, b
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- a whole update


def _state(la):
    """Everything an update changes: weights, fp32 masters, Adam's moments and step counts, the schedule."""
    out = {f"model/{k}": v.detach().clone() for k, v in la.net.state_dict().items()}
    for i, (_p, master) in enumerate(la.opt.pairs):
        out[f"master/{i}"] = master.detach().clone()
    for i, st in enumerate(la.opt.inner.state.values()):
        for k, v in st.items():
            out[f"adam/{i}/{k}"] = v.detach().clone() if torch.is_tensor(v) else torch.tensor(v)
    out["lr"] = torch.tensor([pg["lr"] for pg in la.opt.param_groups], dtype=torch.float64)
    out["sched"] = torch.tensor([la.sched.last_epoch])
    return out


def _copy_state(src, dst):
    dst.net.load_state_dict(src.net.state_dict(), strict=True)
    dst.target_net.load_state_dict(src.target_net.state_dict(), strict=True)
    dst.opt.sync_masters_from_model()


def _dataset_of(cols):
    ds = new_core.AsyncEpisodeDataset(14, "cpu")
    ds.add_steps(*cols)
    return ds


def _run_update(core_mod, la, cols, seed: int, **extra):
    ds = core_mod.AsyncEpisodeDataset(14, "cpu")
    ds.add_steps(*cols)
    torch.manual_seed(seed)
    return core_mod.ppo_update(net=la.net, opt=la.opt, dataset=ds, scheduler=la.sched, mode=la.cfg.mode, target_net=None,
                               **la.cfg.ppo_kwargs(), v_min=la.cfg.v_min, v_max=la.cfg.v_max, v_bins=la.cfg.v_bins, **extra)


@needs_gpu
@pytest.mark.parametrize("mode", ["imitation", "warmup", "ppo"])
def test_a_whole_update_leaves_the_same_weights(old, data, mode):
    """
    Old ppo_update and new ppo_update from the same weights, data and seed. Two updates each,
    so the second starts from Adam moments the first one left. The new code runs three times:
    with the data on the GPU, with it on the CPU, and choosing by itself.
    """
    obs_np, ep_np, _kind = data
    n = 1536                                         # 6 minibatches an epoch: two full accumulation groups
    updates = [_update_data(obs_np, ep_np, n, seed=10), _update_data(obs_np[n:], ep_np[n:], n, seed=11)]
    a = _old_learner(old, mode)
    a2 = _old_learner(old, mode)
    variants = {"on the GPU": True, "on the CPU": False, "auto": "auto"}
    news = {name: _new_learner(mode) for name in variants}
    for other in [a2] + list(news.values()):
        _copy_state(a, other)
    start = _state(a)
    print()
    print(f"{mode}: data on the GPU by the automatic rule: "
          f"{new_core.update_data_fits(_dataset_of(updates[0]), next(a.net.parameters()).device)}")

    for step, cols in enumerate(updates):
        sa = _run_update(old.core, a, cols, seed=50 + step)
        want = _state(a)
        _run_update(old.core, a2, cols, seed=50 + step)
        again = _state(a2)
        moved = [k for k in want if not micro_ref.same_bits(want[k], again[k])]
        assert moved == [], "the old code does not repeat its own update on this machine, so there is nothing to hold the new code to"
        for name, flag in variants.items():
            sb = _run_update(new_core, news[name], cols, seed=50 + step, data_on_device=flag)
            got = _state(news[name])
            assert want.keys() == got.keys()
            diff = [k for k in want if not micro_ref.same_bits(want[k], got[k])]
            assert diff == [], (mode, step, name, diff[:5])
            for f in ("approx_kl", "clip_frac", "entropy", "v_loss", "pg_loss", "total_loss", "n_mb"):
                assert getattr(sa, f) == getattr(sb, f), (mode, step, name, f)
            assert (sb.acc is not None) == (mode == "imitation")
        assert sa.n_mb == 12 or mode == "ppo"        # PPO may stop early on KL, in both alike
    assert int(want["sched"]) > 0 and any(k.startswith("adam/") and k.endswith("exp_avg") for k in want)
    trained = [k for k in want if k.startswith("master/")]
    assert len(trained) == (2 if mode == "warmup" else 92)
    assert all(not micro_ref.same_bits(start[k], want[k]) for k in trained), "every trained tensor should have moved"
    del a, a2, news
    torch.cuda.empty_cache()


@needs_gpu
def test_imitation_accuracy_is_the_share_of_best_actions_that_match(data):
    """One epoch of one minibatch: the figure must be what the weights before the step say."""
    obs_np, ep_np, _kind = data
    la = _new_learner("imitation", grad_accum_steps=1, update_epochs=1)
    n = 256
    cols = list(_update_data(obs_np, ep_np, n, seed=20))
    cols[3] = torch.arange(n, dtype=torch.float32)       # the value column is not used by the update: carry the row number in it
    ds = new_core.AsyncEpisodeDataset(14, "cpu")
    ds.add_steps(*cols)
    torch.manual_seed(7)
    (mb,) = list(ds.iter_minibatches(256, shuffle_episodes=True))
    m0, m1 = la.net.unpacker.offsets["action_mask"]
    with torch.no_grad():
        logits = la.net(mb[0].to(DEV), mb[7].to(DEV), batch_seq_len=256)[0].float().cpu().numpy()
    legal = mb[0][:, m0:m1].float().numpy() > 0.5
    best = np.where(legal, logits, -1e4).argmax(axis=1)

    # teacher = the policy's own best action for half the steps, another legal action for the rest
    teacher = best.copy()
    other = np.array([next((j for j in np.flatnonzero(l) if j != b), b) for l, b in zip(legal, best)])
    teacher[::2] = other[::2]
    want = float((teacher == best).mean())
    assert 0.3 < want < 0.9
    # put the teacher's actions into the data at the rows the minibatch drew
    act = cols[1].clone()
    act[mb[3].long()] = torch.from_numpy(teacher)
    cols[1] = act
    stats = _run_update(new_core, la, cols, seed=7)
    assert stats.n_mb == 1 and abs(stats.acc - want) < 1e-6, (stats.acc, want)
    del la
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- the learner's line, checkpoint and resume


def _fill(la, core_mod, obs_np, ep_np, n: int, seed: int):
    cols = _update_data(obs_np, ep_np, n, seed)
    la.dataset.add_steps(*cols)
    la.total_steps += n


@needs_gpu
@pytest.mark.parametrize("mode", ["imitation", "ppo"])
def test_update_line_has_the_new_keys_at_the_end_and_the_parsers_read_it(data, capsys, mode):
    import pipeline
    from bench import throughput
    obs_np, ep_np, _kind = data
    la = _new_learner(mode, save_every_updates=10**9)
    _fill(la, new_core, obs_np, ep_np, 1024, seed=30)
    t0 = time.perf_counter()
    asyncio.run(la._perform_update())
    took = time.perf_counter() - t0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[learner] upd="))
    keys = [kv.split("=")[0] for kv in line.split()[1:]]
    want = ["upd", "kl", "clip", "ent", "vloss", "ploss", "loss", "n_mb", "sec"] + (["acc"] if mode == "imitation" else [])
    assert keys == want, line
    assert la.update_idx == 1 and la.weight_store.updates == [(1, len(la.net.state_dict()))] and len(la.dataset) == 0

    ray_line = f"\x1b[36m(LearnerActor pid=5996)\x1b[0m {line}"          # as a parent process sees it on Linux
    for text in (line, ray_line, ray_line + "\r\n"):
        ev = pipeline.parse_trainer_line(text)
        assert ev["kind"] == "learner_update" and ev["update"] == 1 and ev["nonfinite"] == []
        assert set(ev["values"]) == set(want[1:]) and 0.0 <= ev["values"]["sec"] <= took + 0.1
        tp = throughput.parse_learner(text)
        assert tp["upd"] == 1 and tp["n_mb"] == ev["values"]["n_mb"] and tp["sec"] == ev["values"]["sec"]
        if mode == "imitation":
            assert 0.0 <= ev["values"]["acc"] <= 1.0 and tp["acc"] == ev["values"]["acc"]
    # the logger's next line glued on behind it, as two writers to one stdout sometimes leave it
    glued = ray_line + "2026-10-05 09:48:48,489 [INFO] [Telemetry] ACTIVE: 256 | INFER: {'total_requests': 9, 'avg_batch_size': 2.0, 'model_version': 1} | TRAIN: {'update': 1, 'episodes': 3, 'steps_in_dataset': 0, 'total_steps': 1024}"
    evs = pipeline.parse_trainer_events(glued)
    assert [e["kind"] for e in evs] == ["learner_update", "telemetry"] and set(evs[0]["values"]) == set(want[1:])
    assert evs[0]["values"]["sec"] == ev["values"]["sec"]
    tp = throughput.parse_learner(glued)
    assert tp["sec"] == ev["values"]["sec"] and tp["n_mb"] == ev["values"]["n_mb"]
    if mode == "imitation":
        assert tp["acc"] == ev["values"]["acc"]
    del la
    torch.cuda.empty_cache()


def test_old_update_lines_still_parse():
    """The lines the box is printing today, from tests/fixtures: the parsers must keep reading those too."""
    from bench import throughput
    path = os.path.join(REPO, "tests", "fixtures", "linux_trainer_stdout_smoke.log")
    if not os.path.isfile(path):
        pytest.skip("fixture not in this checkout")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln for ln in f if "[learner] upd=" in ln]
    assert lines
    for ln in lines:
        tp = throughput.parse_learner(ln)
        assert tp is not None and "n_mb" in tp and "sec" not in tp and tp.get("actor") == "LearnerActor"


def test_checkpoint_is_synced_before_it_takes_its_name(monkeypatch):
    la = object.__new__(new_learner.LearnerActor)
    la.run_cfg = new_config.RunConfig.default()
    la.net = nn.Linear(3, 2)
    la.target_net = nn.Linear(3, 2)
    la.opt = torch.optim.AdamW(la.net.parameters())
    la.sched = None
    la.update_idx, la.total_steps, la.total_episodes = 7, 1234, 56
    d = os.path.join(TEST_ROOT, f"ckpt_sync_{os.getpid()}_{int(time.time())}")
    os.makedirs(d)
    path = os.path.join(d, "learner_update_000007.pt")
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", os.path.exists(path), os.path.getsize(path + ".tmp") if os.path.exists(path + ".tmp") else -1))
        return real_fsync(fd)

    def replace(src, dst):
        events.append(("replace", os.path.basename(src), os.path.basename(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(new_learner.os, "fsync", fsync)
    monkeypatch.setattr(new_learner.os, "replace", replace)
    la._save_checkpoint(path)
    kinds = [e[0] for e in events]
    assert kinds[:2] == ["fsync", "replace"], events
    assert events[0][1] is False and events[0][2] > 0, "the temporary file must be complete and synced while the final name does not exist yet"
    assert events[1] == ("replace", "learner_update_000007.pt.tmp", "learner_update_000007.pt")
    assert kinds[2:] == (["fsync"] if os.name == "posix" else []), "on POSIX the directory is synced after the rename"
    assert not os.path.exists(path + ".tmp")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert set(ck) == {"model", "target_model", "optimizer", "scheduler", "update_idx", "total_steps", "run_cfg",
                       "total_episodes", "torch_rng", "numpy_rng"}
    assert ck["update_idx"] == 7 and ck["total_steps"] == 1234 and torch.equal(ck["model"]["weight"], la.net.weight)


def test_schedule_constants_are_the_ones_the_run_was_planned_on():
    """
    The length of the schedule decides what the model learns from optimizer step 501,001 on
    (192 million decisions into PPO). It is the owner's to change, not a speed change's: the
    speed work had set lr_total_steps to 1,000,000 and refused the old value, and that was
    taken out again. The default is the committed one, and no value is refused.

    When the owner decides on another length, this test and the next are where that decision
    is written down: change the number here with the one in config.py, and hold the next test
    to the old code only through step 501,000, where every length gives the same numbers.
    """
    lc = new_config.LearnerConfig()
    assert (lc.lr, lc.lr_warmup_steps, lc.lr_hold_steps, lc.lr_total_steps) == (3e-4, 1_000, 500_000, 100_000)
    for total in (100_000, 501_001, 644_000, 1_000_000):
        assert new_config.LearnerConfig(lr_total_steps=total).lr_total_steps == total
    # every mode train.py can be given still builds
    for mode in ("imitation", "warmup", "ppo", "ppo_with_jepa", "jepa_pretraining"):
        assert dataclasses.replace(lc, mode=mode).mode == mode


def _multiplier(step: int, w=1_000, h=500_000, t=100_000) -> float:
    if step < w:
        return (step + 1) / w
    if step < w + h:
        return 1.0
    return 1.0 / ((8 * min(1.0, (step - w - h) / max(1, t - w - h)) + 1) ** 1.5)


@pytest.mark.parametrize("mode", ["imitation", "warmup", "ppo"])
def test_schedule_is_the_one_the_old_code_builds(old, mode):
    """
    The rate of every parameter group at every optimizer step, as the old learner built from
    its own config.py has it and as the new one has it: through the warmup, the hold, the
    fall at step 501,001 and far beyond. This is what fails if the schedule's length moves.
    """
    before, after = _old_learner(old, mode), _new_learner(mode)
    assert [pg["name"] for pg in before.opt.param_groups] == [pg["name"] for pg in after.opt.param_groups]
    assert before.sched.base_lrs == after.sched.base_lrs and len(before.sched.lr_lambdas) == len(after.sched.lr_lambdas)
    steps = (list(range(0, 1_200)) + list(range(500_900, 501_100)) + list(range(999_900, 1_000_100))
             + list(range(0, 1_100_000, 997)) + [600_000, 651_041, 5_000_000])
    for f_old, f_new in zip(before.sched.lr_lambdas, after.sched.lr_lambdas):
        moved = [s for s in steps if f_old(s) != f_new(s)]
        assert moved == [], (mode, moved[:5], f_old(moved[0]), f_new(moved[0]))
    fn = after.sched.lr_lambdas[0]
    assert all(fn(s) == _multiplier(s) for s in steps)
    assert fn(501_000) == 1.0 and abs(fn(501_001) - 1 / 27) < 1e-12, "the committed schedule: one fall of 27 times after the hold"
    del before, after


def test_the_worked_out_alternative_glides_when_it_is_asked_for():
    """
    lr_total_steps=1_000_000, the value bench/SPEED_AUDIT.md section 5 works out. It is not
    the default (see above). Asked for, it gives what the audit says: no fall larger than
    0.003% from one step to the next, 1/27 at step 1,000,000 and after.
    """
    la = _new_learner("ppo", lr_total_steps=1_000_000)
    fn = la.sched.lr_lambdas[0]
    assert [fn(s) for s in (0, 999, 1_000, 501_000)] == [0.001, 1.0, 1.0, 1.0]
    assert abs(fn(1_000_000) - 1 / 27) < 1e-12 and abs(fn(5_000_000) - 1 / 27) < 1e-12
    steps = list(range(500_990, 501_010)) + list(range(999_990, 1_000_010)) + list(range(501_000, 1_000_000, 997))
    assert all(abs(fn(s) - _multiplier(s, t=1_000_000)) < 1e-15 for s in steps)
    worst = max(1 - fn(s + 1) / fn(s) for s in range(501_000, 1_000_000, 13))
    assert worst < 3e-5, "the largest fall from one optimizer step to the next"
    del la


def _checkpoint_in_memory(la):
    """What LearnerActor._save_checkpoint writes, through a real save and load, without a 219 MB file."""
    payload = {
        "model": la.net.state_dict(), "target_model": la.target_net.state_dict(),
        "optimizer": la.opt.state_dict(), "scheduler": la.sched.state_dict(),
        "update_idx": la.update_idx, "total_steps": la.total_steps, "run_cfg": la.run_cfg.as_dict(),
        "total_episodes": la.total_episodes, "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("saved_at,total", [(88, 100_000), (250_000, 100_000), (750_000, 100_000), (750_000, 1_000_000)])
def test_resumed_run_takes_its_first_step_at_the_scheduled_rate(old, monkeypatch, saved_at, total):
    """
    A checkpoint saved after `saved_at` optimizer steps (in the warmup, in the hold, after the
    hold), resumed in the same phase. The rate every group holds for the next step must be
    the one the run would have had without the restart. The old learner is shown doing what
    the audit described: the full rate for that one step. `total` is lr_total_steps: the
    committed 100,000, and for the step after the hold also the 1,000,000 the audit works out.
    """
    over = {} if total == 100_000 else {"lr_total_steps": total}
    run = _new_learner("ppo", **over)
    run.opt.inner.step()                                   # LambdaLR wants an optimizer step before its own
    run.sched.last_epoch = saved_at - 1
    run.sched.step()                                       # now at saved_at, as after that many real steps
    assert run.sched.last_epoch == saved_at
    uninterrupted = [pg["lr"] for pg in run.opt.param_groups]
    names = [pg["name"] for pg in run.opt.param_groups]
    mult = {"embeddings": 0.1, "subnets": 0.1, "transformer": 0.1, "readout": 1.0, "pi": 1.0, "v": 2.0}
    full = [3e-4 * mult[n.split("_")[0]] for n in names]
    assert all(abs(u - f * _multiplier(saved_at, t=total)) <= 1e-18 + 1e-12 * f for u, f in zip(uninterrupted, full))
    run.update_idx = 5
    ck = _checkpoint_in_memory(run)
    assert ck["scheduler"]["last_epoch"] == saved_at
    path = "held in memory"
    monkeypatch.setattr(torch, "load", lambda *a, **k: ck)

    back = _new_learner("ppo", **over)
    back._load_checkpoint(path)
    assert back.sched.last_epoch == saved_at and back.update_idx == 5
    assert [pg["lr"] for pg in back.opt.param_groups] == uninterrupted
    assert back.sched.get_last_lr() == uninterrupted
    # and the step after that is the schedule's next value, as it would have been
    back.opt.inner.step(), back.sched.step()
    run.opt.inner.step(), run.sched.step()
    assert [pg["lr"] for pg in back.opt.param_groups] == [pg["lr"] for pg in run.opt.param_groups]

    before = _old_learner(old, "ppo", **over)
    with old.ref.active():
        before._load_checkpoint(path)
    was = [pg["lr"] for pg in before.opt.param_groups]
    assert was == full, "the old learner put the full rate into every group"
    if saved_at == 250_000:
        assert was == uninterrupted                        # in the hold the two are the same number
    else:
        # 11.2 times too high 89 steps into the warmup; after the hold 27 times with the
        # committed schedule and 11.1 times half-way down the worked-out alternative
        least = {(88, 100_000): 11, (750_000, 100_000): 26.9, (750_000, 1_000_000): 11}[(saved_at, total)]
        assert max(w / u for w, u in zip(was, uninterrupted)) > least
    del run, back, before


def test_schedule_length_can_still_be_changed_at_a_restart_during_the_hold(monkeypatch):
    """
    What leaving the decision to the owner relies on. A checkpoint saved in the hold by a run
    with the committed lr_total_steps, resumed by a trainer whose config.py says 1,000,000:
    from there on every group has the rates of a run that had the longer schedule from its
    first step. (The schedule is rebuilt from the config at every start. Only the step count
    comes out of the checkpoint.)
    """
    run = _new_learner("ppo")
    assert run.cfg.lr_total_steps == 100_000
    run.opt.inner.step()                                   # LambdaLR wants an optimizer step before its own
    run.sched.last_epoch = 250_000 - 1
    run.sched.step()
    run.update_idx = 5
    ck = _checkpoint_in_memory(run)
    assert ck["scheduler"]["last_epoch"] == 250_000 and ck["run_cfg"]["learner"]["lr_total_steps"] == 100_000
    monkeypatch.setattr(torch, "load", lambda *a, **k: ck)

    back = _new_learner("ppo", lr_total_steps=1_000_000)
    back._load_checkpoint("held in memory")
    assert back.sched.last_epoch == 250_000 and back.update_idx == 5
    fresh = _new_learner("ppo", lr_total_steps=1_000_000)  # the longer schedule from the start, never interrupted
    mult = {"embeddings": 0.1, "subnets": 0.1, "transformer": 0.1, "readout": 1.0, "pi": 1.0, "v": 2.0}
    full = [3e-4 * mult[pg["name"].split("_")[0]] for pg in back.opt.param_groups]
    for la in (run, back, fresh):
        la.opt.inner.step()
    for step in (250_001, 501_000, 501_001, 600_000, 651_041, 1_000_000, 1_200_000):
        for la in (run, back, fresh):
            la.sched.last_epoch = step - 1
            la.sched.step()
        got = [pg["lr"] for pg in back.opt.param_groups]
        assert got == [pg["lr"] for pg in fresh.opt.param_groups], step
        assert all(abs(g - f * _multiplier(step, t=1_000_000)) <= 1e-18 + 1e-12 * f for g, f in zip(got, full)), step
        # and the run that was not restarted keeps the committed schedule
        kept = [pg["lr"] for pg in run.opt.param_groups]
        assert all(abs(k - f * _multiplier(step, t=100_000)) <= 1e-18 + 1e-12 * f for k, f in zip(kept, full)), step
    assert abs(got[0] / full[0] - 1 / 27) < 1e-12
    del run, back, fresh


# --------------------------------------------------------------------------- what the worker sends


def test_worker_sends_float16_observations(monkeypatch):
    """One pass of the rollout loop with stand-ins: what reaches infer_batch is the assembler's float16."""
    sent = {}

    class Infer:
        infer_batch = _Remote(lambda tags, obs: (tags, obs))
        clear_cache = _Remote(lambda tag: None)

    def fake_get(call):
        tags, obs = call
        sent["tags"], sent["obs"] = list(tags), obs
        n = len(tags)
        return np.arange(n, dtype=np.int64), np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)

    class Stop(Exception):
        pass

    rows = [np.full(8567, i + 0.5, dtype=np.float16) for i in range(3)]

    class Env:
        calls = 0

        def step(self, actions):
            Env.calls += 1
            if Env.calls == 1:
                return [("STEP", f"tag{i}", rows[i], 0.25 * i) for i in range(3)]
            sent["actions"] = dict(actions)
            raise Stop()

    monkeypatch.setattr(new_worker.ray, "get", fake_get)
    w = object.__new__(new_worker.RolloutWorker)
    w.inference_actor, w.vec_env, w._traj = Infer(), Env(), {}
    with pytest.raises(Stop):
        w._run_rl_loop()
    obs = sent["obs"]
    assert obs.dtype == np.float16 and obs.flags["C_CONTIGUOUS"] and obs.shape == (3, 8567)
    assert obs.tobytes() == np.stack(rows).tobytes() and sent["tags"] == ["tag0", "tag1", "tag2"]
    assert sent["actions"] == {"tag0": 0, "tag1": 1, "tag2": 2}
    for i in range(3):
        assert w._traj[f"tag{i}"]["obs"][0] is rows[i]       # and the trajectory keeps the same float16 rows
    # what the inference actor makes of it is what it made of the old float32 batch
    old_way = torch.from_numpy(np.ascontiguousarray(np.stack(rows), dtype=np.float32))
    assert torch.equal(torch.from_numpy(obs).float(), old_way)


# --------------------------------------------------------------------------- a checkpoint written by the old code

needs_ckpt = pytest.mark.skipif(not os.path.isfile(OLD_CKPT), reason="set PSPPO_OLD_CKPT to a checkpoint written by the old code")


@needs_ckpt
def test_old_checkpoint_loads_with_nothing_missing_or_unexpected(old):
    ck = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    assert set(ck) == {"model", "target_model", "optimizer", "scheduler", "update_idx", "total_steps", "run_cfg",
                       "total_episodes", "torch_rng", "numpy_rng"}
    net = new_config.RunConfig.default().make_model()
    net.enable_bf16_recurrent_path()                      # the dtypes the training actors hold, as the checkpoint has them
    res = net.load_state_dict(ck["model"], strict=True)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []
    assert list(ck["model"].keys()) == list(net.state_dict().keys())
    for k, v in net.state_dict().items():
        assert v.dtype == ck["model"][k].dtype and torch.equal(v, ck["model"][k]), k
    target = new_config.RunConfig.default().make_model()
    target.enable_bf16_recurrent_path()
    res = target.load_state_dict(ck["target_model"], strict=True)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []
    # the helper every player and the inference actor load a checkpoint through
    import inference
    assert inference._extract_state_dict(ck) is ck["model"]


@needs_ckpt
def test_old_checkpoint_gives_the_old_codes_outputs(old, data):
    obs_np, ep_np, _kind = data
    ck = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    with old.ref.active():
        a = old.config.RunConfig.default().make_model()
    b = new_config.RunConfig.default().make_model()
    for net in (a, b):
        net.to(DEV)
        if GPU:
            net.enable_bf16_recurrent_path()
        net.load_state_dict(ck["model"], strict=True)
    # the learner's path
    a.train(), b.train()
    for mb, T in ((256, 256), (128, 64)) if GPU else ((64, 32),):
        obs = torch.from_numpy(np.ascontiguousarray(obs_np[:mb])).to(DEV)
        ep = torch.from_numpy(np.ascontiguousarray(ep_np[:mb])).long().to(DEV)
        with torch.no_grad():
            x, y = a(obs, ep, batch_seq_len=T), b(obs, ep, batch_seq_len=T)
        assert all(torch.equal(p, q) for p, q in zip(x[:3], y[:3])), (mb, T)
    # the inference path, a few battles over several turns with their histories
    a.eval(), b.eval()
    kv_a = kv_b = len_a = len_b = None
    for t in range(70 if GPU else 6):
        rows = [(7 * i + t) % obs_np.shape[0] for i in range(5)]
        obs = torch.from_numpy(np.ascontiguousarray(obs_np[rows], dtype=np.float32)).to(DEV)
        with torch.no_grad():
            x = a(obs, kv_cache=kv_a, kv_cache_len=len_a)
            y = b(obs, kv_cache=kv_b, kv_cache_len=len_b)
        assert all(torch.equal(p, q) for p, q in zip(x[:3], y[:3])), t
        assert all(torch.equal(p, q) for p, q in zip(x[3], y[3])) and torch.equal(x[4], y[4])
        kv_a, len_a, kv_b, len_b = x[3], x[4], y[3], y[4]
    assert float(x[0].abs().max()) > 0


@needs_gpu
@needs_ckpt
@pytest.mark.parametrize("mode", ["imitation", "warmup", "ppo"])
def test_old_checkpoint_resumes_into_each_phase(old, data, capsys, mode):
    """
    The learner's own resume, then one update. In the checkpoint's own phase the optimizer and
    the schedule come back; in another phase the weights do and everything else starts fresh.
    That decision is the learner's old logic, unchanged: this checks the new code still runs it.
    """
    obs_np, ep_np, _kind = data
    ck = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    ck_mode = ck["run_cfg"]["learner"]["mode"]
    la = _new_learner(mode, save_every_updates=10**9)
    la._load_checkpoint(OLD_CKPT)
    out = capsys.readouterr().out
    same_phase = mode == ck_mode
    for k, v in la.net.state_dict().items():
        assert torch.equal(v.cpu(), ck["model"][k]), k
    if same_phase:
        assert "Optimizer state loaded successfully" in out and "Scheduler state loaded successfully" in out
        assert "PHASE CHANGE" not in out and "mismatch" not in out and "failed" not in out
        assert la.update_idx == ck["update_idx"] and la.total_steps == ck["total_steps"] and la.total_episodes == ck["total_episodes"]
        assert la.sched.last_epoch == ck["scheduler"]["last_epoch"]
        # the rate the old run was at when it saved, not the full rate
        assert [pg["lr"] for pg in la.opt.param_groups] == ck["scheduler"]["_last_lr"]
        for (_p, master), saved in zip(la.opt.pairs, ck["optimizer"]["masters"]):
            assert torch.equal(master.cpu(), saved)
    else:
        assert f"PHASE CHANGE DETECTED ({ck_mode} -> {mode})" in out
        assert la.update_idx == 0 and la.total_steps == 0 and la.sched.last_epoch == 0
        lrs = {pg["name"]: pg["lr"] for pg in la.opt.param_groups}
        assert all(lr <= 3e-4 * 2.0 / 1000 + 1e-15 for lr in lrs.values()), "a new phase starts at the bottom of the warmup"

    start = la.update_idx
    _fill(la, new_core, obs_np, ep_np, 1024, seed=40)
    asyncio.run(la._perform_update())
    out = capsys.readouterr().out
    assert "PPO Update failed" not in out and la.update_idx == start + 1
    line = next(ln for ln in out.splitlines() if ln.startswith(f"[learner] upd={start + 1} "))
    assert " sec=" in line and ((" acc=" in line) == (mode == "imitation"))
    vals = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in line.split()[1:]}
    assert all(np.isfinite(v) for v in vals.values()) and vals["n_mb"] >= 1
    assert la.weight_store.updates[-1][0] == start + 1
    del la
    torch.cuda.empty_cache()
