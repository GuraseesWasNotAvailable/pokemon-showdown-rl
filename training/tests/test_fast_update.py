"""
train.py --fast-update (ppo_core.ppo_update fast_update), held to v2-humanaux 4cfd496 bit for bit.

    python -m pytest tests/test_fast_update.py -q -p no:cacheprovider -rs
    PSPPO_BASE_DIR=<a directory with 4cfd496's files> python -m pytest tests/test_fast_update.py ...

Held here:
  * the switch: train.py --fast-update is off by default; only with it does the learner get
    fast_update=True and pass it to ppo_update; without it ppo_update is called exactly as before;
    nothing of it reaches RunConfig (so the checkpoints' run_cfg is the same);
  * masked_logits_nosync gives masked_logits' logits on random masks with all-zero rows, counts
    those rows, and never waits for the GPU (CUDA sync debug mode "error");
  * the unbind path of FlexEncoderLayer: the same outputs and the same gradients as the selects;
  * MasterWeightOptimizer.foreach_copy: the same fp32 grads, masters, Adam state and bf16 params
    as the per-parameter loop;
  * whole updates (imitation, warmup, PPO, PPO with the opponent-action head and boosted rows,
    rows with an all-zero action mask): 4cfd496's ppo_update, this branch's with the flag off and
    with it on, from the same weights, data and seed, two updates in a row. Every weight, fp32
    master, Adam moment, the schedule and every returned figure must be the same bits;
  * the KL early stop fires at the same minibatch in all three, at every minibatch where it can
    fire, including stops in the middle of an epoch. The targets come from the update's own
    per-minibatch KL (read back in a pass that never stops), so where they land does not depend
    on the network's random start or on the GPU's rounding in this process;
  * the reads: with the flag, no minibatch of the first epoch reads anything back from the GPU
    (only the start of each epoch and the end of the update do) and each later minibatch waits
    on its KL event once (what the docstring says about how far the CPU runs ahead and when an
    error surfaces);
  * a checkpoint written after a flag-on update has the same bytes in every tensor as 4cfd496's.

The reference: PSPPO_BASE_DIR, or 4cfd496's files written by `git show` (bench/micro_ref.py).
Whole updates need a GPU (the attention has no backward pass on a CPU) and are skipped without one.
The live-sized check (the live checkpoint, recorded batches, deterministic kernels) is
bench/fast_update_digest.py. Nothing here starts Ray, a server or a training run; nothing is removed.
"""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import os
import sys
import warnings
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402
import torch  # noqa: E402

import learner as new_learner  # noqa: E402
import ppo_core as new_core  # noqa: E402
import test_speed_update as tsu  # noqa: E402  (its learner and update helpers)
from bench import micro_ref  # noqa: E402

DEV = tsu.DEV
GPU = tsu.GPU
needs_gpu = pytest.mark.skipif(not GPU, reason="flex_attention has no backward on a CPU")
BASE_COMMIT = "4cfd496"
BASE_DIR = os.path.expanduser(os.environ.get("PSPPO_BASE_DIR", ""))
DEVICES = ["cpu"] + (["cuda"] if GPU else [])


@pytest.fixture(scope="module")
def base():
    """v2-humanaux 4cfd496's modules, beside the working tree's."""
    try:
        if BASE_DIR:
            ref = micro_ref.Reference(BASE_DIR, source="PSPPO_BASE_DIR")
        else:
            ref = micro_ref.Reference.export(commit=BASE_COMMIT, dest=os.path.join(tsu.TEST_ROOT, f"ref_{BASE_COMMIT}"))
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(f"no {BASE_COMMIT} code to compare with: {e}")
    core, lrn, cfg = ref.load("ppo_core"), ref.load("learner"), ref.load("config")
    assert lrn.ppo_core is core and core is not new_core
    assert not hasattr(core, "masked_logits_nosync"), "the reference holds this branch's code, not 4cfd496's"
    assert "fast_update" not in inspect.signature(core.ppo_update).parameters
    if GPU:
        micro_ref.shared_compiled_flex_attention(core, new_core)
    core._learner_flex_compiled = True
    new_core._learner_flex_compiled = True
    return SimpleNamespace(core=core, learner=lrn, config=cfg, ref=ref)


@pytest.fixture(scope="module")
def data():
    obs, ep, kind = micro_ref.observations(n_min=4096)
    return obs, ep, kind


# --------------------------------------------------------------------------- the switch


def test_the_flag_is_off_by_default_and_never_reaches_run_config():
    import train
    off = train.parse_args(["--mode", "ppo"])
    on = train.parse_args(["--mode", "ppo", "--fast-update"])
    assert off.fast_update is False and on.fast_update is True
    assert train.learner_speed_kwargs(off) == {}
    assert train.learner_speed_kwargs(on) == {"fast_update": True}
    assert train.build_config(off).as_dict() == train.build_config(on).as_dict()
    assert inspect.signature(new_learner.LearnerActor.__init__).parameters["fast_update"].default is False
    assert "learner speed (off by default)" in inspect.getsource(train.add_learner_speed_arguments)


def test_ppo_update_without_the_argument_is_the_old_default():
    assert inspect.signature(new_core.ppo_update).parameters["fast_update"].default is False
    assert new_core.FlexEncoderLayer.fast_unbind is False
    assert new_core.MasterWeightOptimizer.foreach_copy is False


@pytest.mark.parametrize("setting", ["absent", False, True])
def test_the_learner_passes_fast_update_only_when_it_is_on(monkeypatch, data, setting):
    obs_np, ep_np, _kind = data
    la = tsu._new_learner("ppo", save_every_updates=10**9)
    if setting != "absent":
        la.fast_update = setting
    seen = []

    def fake_ppo_update(**kw):
        seen.append(kw)
        return new_core.PPOUpdateStats(approx_kl=0.0, clip_frac=0.0, entropy=0.0, v_loss=0.0, pg_loss=0.0,
                                       total_loss=0.0, n_mb=1)

    monkeypatch.setattr(new_learner, "ppo_update", fake_ppo_update)
    la.dataset.add_steps(*tsu._update_data(obs_np, ep_np, 512, seed=3))
    asyncio.run(la._perform_update())
    assert len(seen) == 1 and la.update_idx == 1
    if setting is True:
        assert seen[0]["fast_update"] is True
    else:
        assert "fast_update" not in seen[0]


# --------------------------------------------------------------------------- the action mask


def _masks(n: int, a: int, seed: int, dtype):
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(n, a, generator=g) < 0.4)
    zero = torch.rand(n, generator=g) < 0.1
    m[zero] = False
    m[0] = False                            # at least one all-zero row
    m[1] = True
    return m.to(dtype), int((m.sum(-1) == 0).sum())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_masked_logits_nosync_equals_masked_logits(device, dtype):
    for seed in range(5):
        mask, n_zero = _masks(257, 14, seed, dtype)
        mask = mask.to(device)
        logits = torch.randn(257, 14, generator=torch.Generator().manual_seed(100 + seed)).to(device)
        want = new_core.masked_logits(logits, mask)
        count = torch.zeros((), dtype=torch.int64, device=device)
        got = new_core.masked_logits_nosync(logits, mask, count)
        assert torch.equal(want, got) and want.dtype == got.dtype
        assert int(count) == int((mask.sum(-1) == 0).sum()) == n_zero > 0
        assert torch.equal(new_core.masked_logits_nosync(logits, mask), want)       # no counter
        # the caller's mask is not touched
        assert int((mask.sum(-1) == 0).sum()) == n_zero


@pytest.mark.parametrize("device", DEVICES)
def test_masked_logprob_entropy_nosync_equals_the_old_path(device):
    mask, _ = _masks(300, 14, 7, torch.float16)
    mask = mask.to(device)
    logits = torch.randn(300, 14, generator=torch.Generator().manual_seed(8)).to(device)
    act = torch.randint(0, 14, (300,), generator=torch.Generator().manual_seed(9)).to(device)
    lp0, en0 = new_core.masked_logprob_entropy(logits, mask, act)
    count = torch.zeros((), dtype=torch.int64, device=device)
    lp1, en1 = new_core.masked_logprob_entropy(logits, mask, act, nosync=True, zero_rows=count)
    assert torch.equal(lp0, lp1) and torch.equal(en0, en1)
    assert int(count) == int((mask.sum(-1) == 0).sum())


@needs_gpu
def test_masked_logits_nosync_never_waits_for_the_gpu():
    mask, _ = _masks(512, 14, 11, torch.float16)
    mask = mask.cuda()
    logits = torch.randn(512, 14, device="cuda")
    count = torch.zeros((), dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        new_core.masked_logits_nosync(logits, mask, count)
        with pytest.raises(RuntimeError):
            new_core.masked_logits(logits, mask)        # the old path reads the GPU (.any())
    finally:
        torch.cuda.set_sync_debug_mode(0)


# --------------------------------------------------------------------------- q, k, v by unbind


@needs_gpu
def test_unbind_path_gives_the_same_outputs_and_gradients():
    from torch.nn.attention.flex_attention import create_block_mask
    micro_ref.shared_compiled_flex_attention(new_core)
    new_core._learner_flex_compiled = True
    torch.manual_seed(5)
    layer = new_core.FlexEncoderLayer(128, 4).cuda().to(torch.bfloat16)
    B, S = 2, 256

    def causal(b, h, q, kv):
        return q >= kv
    bm = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device="cuda")
    x0 = torch.randn(B, S, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, S, 128, device="cuda", dtype=torch.bfloat16)
    res = {}
    for flag in (False, True):
        layer.fast_unbind = flag
        layer.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)
        out = layer(x, bm)
        out.backward(g)
        res[flag] = (out.detach().clone(), x.grad.clone(), {k: p.grad.clone() for k, p in layer.named_parameters()})
    layer.fast_unbind = False
    assert torch.equal(res[False][0], res[True][0])
    assert torch.equal(res[False][1], res[True][1])
    for k in res[False][2]:
        assert torch.equal(res[False][2][k], res[True][2][k]), k


# --------------------------------------------------------------------------- foreach master copies


def _master_opt(device, seed):
    torch.manual_seed(seed)
    shapes = [(64, 32), (32,), (7, 3, 5), (1,), (128,)]
    params = [torch.nn.Parameter((torch.randn(s) * 0.1).to(device=device, dtype=torch.bfloat16)) for s in shapes]
    pairs = [(p, p.detach().to(torch.float32).clone()) for p in params]
    inner = torch.optim.AdamW([{"params": [m for _, m in pairs[:3]], "weight_decay": 0.01},
                               {"params": [m for _, m in pairs[3:]], "weight_decay": 0.0}], lr=3e-3, eps=1e-5)
    return new_core.MasterWeightOptimizer(inner, pairs)


@pytest.mark.parametrize("device", DEVICES)
def test_foreach_grad_flush_and_copy_back_equal_the_loop(device):
    loop, fe = _master_opt(device, 1), _master_opt(device, 1)
    fe.foreach_copy = True
    for step in range(4):
        for k in range(3):                      # three minibatches accumulated, the third param left out once
            gs = [(torch.randn(p.shape, generator=torch.Generator().manual_seed(1000 * step + 10 * k + i)) * 0.01)
                  .to(device=device, dtype=torch.bfloat16) for i, (p, _) in enumerate(loop.pairs)]
            for opt in (loop, fe):
                for i, ((p, _), g) in enumerate(zip(opt.pairs, gs)):
                    p.grad = None if (k == 1 and i == 2) else g.clone()
                opt.accumulate_grads()
            for (p0, m0), (p1, m1) in zip(loop.pairs, fe.pairs):
                assert p0.grad is None and p1.grad is None
                assert torch.equal(m0.grad, m1.grad) and m0.grad.dtype == m1.grad.dtype == torch.float32
        loop.clip_and_step(0.5)
        fe.clip_and_step(0.5)
        for (p0, m0), (p1, m1) in zip(loop.pairs, fe.pairs):
            assert torch.equal(m0, m1) and torch.equal(p0, p1) and p1.dtype == torch.bfloat16
        s0, s1 = loop.inner.state_dict()["state"], fe.inner.state_dict()["state"]
        for k in s0:
            for name in ("exp_avg", "exp_avg_sq", "step"):
                assert torch.equal(s0[k][name], s1[k][name]), (k, name)
        loop.zero_grad(set_to_none=True)
        fe.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------- whole updates


def _cols(obs_np, ep_np, n: int, seed: int, zero_mask_share: float = 0.0):
    cols = list(tsu._update_data(obs_np, ep_np, n, seed=seed))
    if zero_mask_share > 0.0:
        from obs_assembler import ObservationAssembler
        m0, m1 = ObservationAssembler().offsets["action_mask"]
        rng = np.random.default_rng(seed + 1)
        rows = torch.from_numpy(np.flatnonzero(rng.random(n) < zero_mask_share))
        obs = cols[0].clone()
        obs[rows, m0:m1] = 0
        cols[0] = obs
    return cols


def _extras(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return {"opp_action": torch.from_numpy(rng.integers(-1, 8, n).astype(np.int8)),
            "boosted": torch.from_numpy((rng.random(n) < 0.3).astype(np.int8))}


def _update(core_mod, la, cols, extras, seed: int, **extra):
    ds = core_mod.AsyncEpisodeDataset(14, "cpu")
    if extras is None:
        ds.add_steps(*cols)
    else:
        ds.add_steps(*cols, extras=extras)
    torch.manual_seed(seed)
    return core_mod.ppo_update(net=la.net, opt=la.opt, dataset=ds, scheduler=la.sched, mode=la.cfg.mode, target_net=None,
                               **la.cfg.ppo_kwargs(), v_min=la.cfg.v_min, v_max=la.cfg.v_max, v_bins=la.cfg.v_bins,
                               **la._aux_update_kwargs(), **extra)


FIGURES = ("approx_kl", "clip_frac", "entropy", "v_loss", "pg_loss", "total_loss", "n_mb", "acc", "aux")

# The stop fires when an epoch's mean KL is above 1.5 x the target. A minibatch's approx_kl is a
# mean of log-ratios of an update this young: far above -1.5, so -1.0 makes it fire at once.
STOP_AT_ONCE = -1.0

# variant: (mode, learner overrides, extras?, share of rows with an all-zero action mask)
VARIANTS = {
    "imitation": ("imitation", {}, False, 0.0),
    "warmup": ("warmup", {}, False, 0.0),
    "ppo": ("ppo", {"update_epochs": 3}, False, 0.0),
    "ppo_zero_masks": ("ppo", {"update_epochs": 2}, False, 0.03),
    "ppo_aux": ("ppo", {"update_epochs": 2, "opp_action_aux_coef": 0.1, "boosted_row_weight": 2.0}, True, 0.0),
    "ppo_no_target": ("ppo", {"update_epochs": 2, "target_kl": None}, False, 0.0),
    # A target below any KL these updates can have, so the stop fires at the first test: the
    # first minibatch of the second epoch (7). 1e-12 was not: the policy barely moves in the
    # warmup, and a minibatch's approx_kl (a mean log-ratio) can be 0 or below.
    "ppo_stop_at_once": ("ppo", {"update_epochs": 2, "target_kl": STOP_AT_ONCE}, False, 0.0),
    "ppo_cpu_minibatches": ("ppo", {"update_epochs": 2}, False, 0.0),
}


@needs_gpu
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_a_whole_update_is_the_same_bits_as_4cfd496_with_the_flag_off_and_on(base, data, variant, caplog):
    mode, over, with_extras, zero_share = VARIANTS[variant]
    obs_np, ep_np, _kind = data
    n = 1536                                            # 6 minibatches of 256 an epoch, grad_accum 3
    updates = [_cols(obs_np, ep_np, n, 10, zero_share), _cols(obs_np[n:], ep_np[n:], n, 11, zero_share)]
    xs = [_extras(n, 20), _extras(n, 21)] if with_extras else [None, None]
    flag_kw = {"data_on_device": False} if variant == "ppo_cpu_minibatches" else {}
    old = tsu._old_learner(base, mode, **over)
    off = tsu._new_learner(mode, **over)
    on = tsu._new_learner(mode, **over)
    for other in (off, on):
        tsu._copy_state(old, other)
    for step, (cols, x) in enumerate(zip(updates, xs)):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=base.core.__name__):
            sa = _update(base.core, old, cols, x, 50 + step, **flag_kw)
        old_warnings = [r.getMessage() for r in caplog.records if "Zero mask detected" in r.getMessage()]
        want = tsu._state(old)
        sb = _update(new_core, off, cols, x, 50 + step, **flag_kw)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=new_core.__name__):
            sc = _update(new_core, on, cols, x, 50 + step, fast_update=True, **flag_kw)
        new_warnings = [r.getMessage() for r in caplog.records if "Zero mask detected" in r.getMessage()]
        for name, la, st in (("flag off", off, sb), ("flag on", on, sc)):
            got = tsu._state(la)
            assert want.keys() == got.keys()
            diff = [k for k in want if not micro_ref.same_bits(want[k], got[k])]
            assert diff == [], (variant, step, name, diff[:5])
            for f in FIGURES:
                assert getattr(sa, f) == getattr(st, f), (variant, step, name, f, getattr(sa, f), getattr(st, f))
        if zero_share > 0.0:
            assert len(old_warnings) >= 2 and len(new_warnings) == 1, (old_warnings[:3], new_warnings)
        else:
            assert new_warnings == []
        if with_extras:
            assert sc.aux and "opp_n" in sc.aux and "boosted_share" in sc.aux
        if over.get("target_kl", 0.02) == STOP_AT_ONCE and mode == "ppo":
            assert sa.n_mb == 7                         # the first minibatch of the second epoch
    assert any(m.fast_unbind for m in on.net.modules() if isinstance(m, new_core.FlexEncoderLayer))
    assert on.opt.foreach_copy and not off.opt.foreach_copy
    del old, off, on
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- the KL early stop


def _traced_update(core_mod, la, cols, seed: int, monkeypatch):
    """
    One update of core_mod with an early-stop target nothing reaches (1e9), and the approx_kl of
    each of its minibatches as the update computed it. Two spies keep copies of what the update
    used, without changing it: the log-probabilities masked_logprob_entropy returned and the
    minibatch's old log-probabilities. (old - new).mean() of those, on the same device, is the
    update's own expression on the same tensors, so the same float32.
    """
    news, olds = [], []
    real = core_mod.masked_logprob_entropy

    def spy(logits, mask, actions, *a, **k):
        lp, ent = real(logits, mask, actions, *a, **k)
        news.append(lp.detach().clone())
        return lp, ent

    ds = core_mod.AsyncEpisodeDataset(14, "cpu")
    ds.add_steps(*cols)
    real_iter = ds.iter_minibatches

    def iter_spy(*a, **k):
        for b in real_iter(*a, **k):
            olds.append(b[2].detach().clone())
            yield b

    ds.iter_minibatches = iter_spy
    la.cfg = dataclasses.replace(la.cfg, target_kl=1e9)
    with monkeypatch.context() as m:
        m.setattr(core_mod, "masked_logprob_entropy", spy)
        torch.manual_seed(seed)
        st = core_mod.ppo_update(net=la.net, opt=la.opt, dataset=ds, scheduler=la.sched, mode=la.cfg.mode, target_net=None,
                                 **la.cfg.ppo_kwargs(), v_min=la.cfg.v_min, v_max=la.cfg.v_max, v_bins=la.cfg.v_bins,
                                 **la._aux_update_kwargs())
    assert len(news) == len(olds) == st.n_mb, (len(news), len(olds), st.n_mb)
    return st, [(o.to(n.device) - n).mean().item() for o, n in zip(olds, news)]


def _stop_at(kls, per_epoch: int, target: float) -> int:
    """The minibatch at which ppo_update's KL test stops for this target (len(kls): it never does), in its own Python floats."""
    n = 0
    for epoch in range(len(kls) // per_epoch):
        epoch_kl, epoch_mb = 0.0, 0
        for _ in range(per_epoch):
            epoch_kl += kls[n]
            n += 1
            epoch_mb += 1
            if epoch > 0 and target and (epoch_kl / epoch_mb) > target * 1.5:
                return n
    return n


def _stop_targets(kls, per_epoch: int):
    """
    {minibatch: a target that stops there} for every minibatch where some target can stop the
    update: those whose epoch mean so far is above every earlier one of the epochs that test.
    The target is half way between the two, so float rounding cannot move the stop.
    """
    out, best, n = {}, None, 0
    for epoch in range(len(kls) // per_epoch):
        epoch_kl, epoch_mb = 0.0, 0
        for _ in range(per_epoch):
            epoch_kl += kls[n]
            n += 1
            epoch_mb += 1
            if epoch == 0:
                continue
            mean = epoch_kl / epoch_mb
            if best is None or mean > best:
                lo = best if best is not None else mean - max(abs(mean), 1e-6)
                target = (lo + mean) / 3.0          # (lo + mean) / 2, over the test's 1.5
                if target and _stop_at(kls, per_epoch, target) == n:
                    out[n] = target
                best = mean
    return out


@needs_gpu
@pytest.mark.parametrize("grad_accum", [1, 3])
def test_the_kl_early_stop_fires_at_the_same_minibatch_in_the_middle_of_an_epoch(base, data, monkeypatch, grad_accum):
    """
    Each target: 4cfd496, flag off and flag on stop at the same minibatch and leave the same bits.

    The targets are not a fixed grid. A pass that never stops reads back the update's own KL at
    every minibatch, and from it comes one target for every minibatch where a stop can land.
    Each run must stop exactly there.

    With a step every minibatch (the live --grad-accum 1) at least one of them must lie strictly
    inside an epoch (after its first minibatch, where one KL read decides, and before its last),
    so that stop is decided on the sum of several KL reads. The learning rate is the live one
    with no warmup, so the policy moves at every step and the epoch's mean KL climbs inside the
    epoch: with 20 network seeds out of 20 there were such stops (bench run of 2026-10-17). With
    a step every third minibatch the KL moves only at every third, and for 6 of those 20 seeds no
    stop could land inside an epoch, so there every stop that can land is checked and none is
    required inside. (The old fixed grid, on a policy still in its warmup whose KL is noise
    around 0, put the stop in a different place in every process and failed about half the runs.)
    """
    obs_np, ep_np, _kind = data
    n, per_epoch, epochs = 1536, 6, 3                   # 6 minibatches of 256 an epoch
    cols = _cols(obs_np, ep_np, n, 30)
    over = dict(update_epochs=epochs, grad_accum_steps=grad_accum, lr_warmup_steps=0)
    torch.manual_seed(1234)
    old = tsu._old_learner(base, "ppo", **over)
    off = tsu._new_learner("ppo", **over)
    on = tsu._new_learner("ppo", **over)
    for other in (off, on):
        tsu._copy_state(old, other)
    start = {name: (la, _snapshot(la)) for name, la in (("old", old), ("off", off), ("on", on))}

    # The KL of every minibatch, from 4cfd496's update; the spies must change nothing in it.
    _restore(old, start["old"][1])
    st_trace, kls = _traced_update(base.core, old, cols, 77, monkeypatch)
    want_trace = tsu._state(old)
    _restore(old, start["old"][1])
    old.cfg = dataclasses.replace(old.cfg, target_kl=1e9)
    st_plain = _update(base.core, old, cols, None, 77)
    assert all(micro_ref.same_bits(want_trace[k], v) for k, v in tsu._state(old).items())
    for f in FIGURES:
        assert getattr(st_trace, f) == getattr(st_plain, f), f
    assert st_trace.n_mb == per_epoch * epochs == len(kls)

    targets = _stop_targets(kls, per_epoch)
    means = []
    for e in range(1, epochs):
        s = 0.0
        for j in range(per_epoch):
            s += kls[e * per_epoch + j]
            means.append(round(s / (j + 1), 6))
    print(f"\ngrad_accum {grad_accum}: epoch means of the KL test (minibatches {per_epoch + 1}-{per_epoch * epochs}): {means}")
    print("stops and their targets:", {k: f"{v:.6g}" for k, v in targets.items()})
    assert per_epoch + 1 in targets, "the first test must be able to stop the update"
    mid = [k for k in targets if k % per_epoch not in (0, 1)]
    if grad_accum == 1:
        assert mid, f"no stop can land inside an epoch: the epoch means {means}"

    seen = {}
    for want_mb, tk in list(targets.items()) + [(per_epoch * epochs, 1e9)]:
        res = {}
        for name, (la, snap) in start.items():
            _restore(la, snap)
            la.cfg = dataclasses.replace(la.cfg, target_kl=tk)
            core = base.core if name == "old" else new_core
            st = _update(core, la, cols, None, 77, **({"fast_update": True} if name == "on" else {}))
            res[name] = (st, tsu._state(la))
        (sa, want) = res["old"]
        assert sa.n_mb == want_mb, (tk, sa.n_mb, want_mb)
        for name in ("off", "on"):
            st, got = res[name]
            assert st.n_mb == sa.n_mb, (tk, name, sa.n_mb, st.n_mb)
            diff = [k for k in want if not micro_ref.same_bits(want[k], got[k])]
            assert diff == [], (tk, name, diff[:5])
            for f in FIGURES:
                assert getattr(sa, f) == getattr(st, f), (tk, name, f)
        seen[want_mb] = tk
    print("stops checked:", sorted(seen))
    del old, off, on
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- the reads


class _CountingEvent(torch.cuda.Event):
    waits = 0

    def synchronize(self):
        type(self).waits += 1
        return super().synchronize()


def _reads(la, cols, snap, monkeypatch, **extra):
    """(host reads CUDA's sync debug mode reports, Event.synchronize calls, n_mb) of one update from snap."""
    _restore(la, snap)
    _CountingEvent.waits = 0
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "Event", _CountingEvent)
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                st = _update(new_core, la, cols, None, 5, **extra)
            finally:
                torch.cuda.set_sync_debug_mode(0)
    # Only the reports of a sync (not the one-time note that the debug mode is a prototype).
    syncs = [w for w in ws if "called a synchronizing CUDA operation" in str(w.message)]
    return len(syncs), _CountingEvent.waits, st.n_mb


@needs_gpu
def test_with_the_flag_the_first_epoch_reads_nothing_and_later_minibatches_wait_on_their_kl_once(data, monkeypatch):
    """
    Counted, what the fast_update docstring says. A first epoch of 6 or 12 minibatches makes the
    same number of host reads: those before the loop, the one at the start of each epoch (the
    minibatch order copied to the GPU waits for it) and the one after the loop. None inside the
    epoch, which is why the CPU can run a whole epoch ahead and an error inside it shows only
    at a later read. A second epoch adds its own start read, one KL event wait per minibatch,
    and nothing else. Without the flag the reads grow with the minibatches, which shows the
    counter sees them.
    """
    obs_np, ep_np, _kind = data
    small, big = _cols(obs_np, ep_np, 1536, 40), _cols(obs_np, ep_np, 3072, 41)     # 6 and 12 minibatches
    la = tsu._new_learner("ppo", update_epochs=1, target_kl=1e9, grad_accum_steps=1)
    snap = _snapshot(la)
    _update(new_core, la, big, None, 5, fast_update=True)          # compile and warm the kernels
    one_small = _reads(la, small, snap, monkeypatch, fast_update=True)
    one_big = _reads(la, big, snap, monkeypatch, fast_update=True)
    snap2 = dict(snap, cfg=dataclasses.replace(snap["cfg"], update_epochs=2))
    two_small = _reads(la, small, snap2, monkeypatch, fast_update=True)
    off_small = _reads(la, small, snap, monkeypatch)
    off_big = _reads(la, big, snap, monkeypatch)
    print(f"\n(reads, KL waits, n_mb): flag on, 1 epoch {one_small} {one_big}; 2 epochs {two_small}; "
          f"flag off, 1 epoch {off_small} {off_big}")
    assert one_small[2] == 6 and one_big[2] == 12 and two_small[2] == 12
    assert one_small[0] == one_big[0], "a minibatch of the first epoch made a host read"
    assert two_small[0] == one_small[0] + 1, "the second epoch read more than its start"
    assert one_small[1] == one_big[1] == 0, "the first epoch waited on a KL event"
    assert two_small[1] == 6, "each minibatch of the second epoch waits on its KL once"
    assert off_big[0] > off_small[0] > one_small[0], "the counter does not see the reads without the flag"


# --------------------------------------------------------------------------- the checkpoint


def _ckpt_diffs(a, b, path=""):
    """Every place two loaded checkpoints differ: a tensor's dtype, shape or bytes (0-d included), or any other field."""
    out = []
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                and a.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                == b.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()):
            out.append(path)
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{path} keys {sorted(set(a) ^ set(b), key=str)}")
        for k in a:
            if k in b:
                out += _ckpt_diffs(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path} len")
        for i, (u, v) in enumerate(zip(a, b)):
            out += _ckpt_diffs(u, v, f"{path}[{i}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (np.asarray(a).dtype == np.asarray(b).dtype and np.array_equal(a, b)):
            out.append(path)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def _tensors(x):
    if torch.is_tensor(x):
        return [x]
    if isinstance(x, dict):
        return [t for v in x.values() for t in _tensors(v)]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _tensors(v)]
    return []


@needs_gpu
def test_a_checkpoint_after_a_flag_on_update_has_4cfd496s_bytes(base, data):
    """
    4cfd496 and the flag on, one update each from the same state, then each learner's own
    _save_checkpoint (the RNGs set alike first, since the file records them): the same bytes in
    every tensor, the 0-d ones (Adam's step counts) included, and every other field equal.
    """
    obs_np, ep_np, _kind = data
    cols, x = _cols(obs_np, ep_np, 1536, 50), _extras(1536, 51)
    over = dict(update_epochs=2, opp_action_aux_coef=0.1, boosted_row_weight=2.0, lr_warmup_steps=0)
    old = tsu._old_learner(base, "ppo", **over)
    on = tsu._new_learner("ppo", **over)
    tsu._copy_state(old, on)
    d = os.path.join(tsu.TEST_ROOT, f"fast_update_ckpt_{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    paths = {}
    for name, core, la, kw in (("old", base.core, old, {}), ("on", new_core, on, {"fast_update": True})):
        _update(core, la, cols, x, 60, **kw)
        la.update_idx += 1
        torch.manual_seed(61)
        np.random.seed(62)
        paths[name] = os.path.join(d, f"{name}_learner_update_000001.pt")
        la._save_checkpoint(paths[name])
    a = torch.load(paths["old"], map_location="cpu", weights_only=False)
    b = torch.load(paths["on"], map_location="cpu", weights_only=False)
    diffs = _ckpt_diffs(a, b)
    # The release branch (topstart) also has item lr-decisions, whose only trace in a checkpoint is two
    # zero-valued keys in run_cfg.learner (its flags off): those two and nothing else are allowed.
    trace = "/run_cfg/learner keys ['lr_hold_decisions', 'lr_total_decisions']"
    if trace in diffs:
        lb = b["run_cfg"]["learner"]
        assert lb["lr_hold_decisions"] == 0 and lb["lr_total_decisions"] == 0, lb
        diffs = [x for x in diffs if x != trace]
    ts = _tensors(a)
    print(f"\n{len(ts)} tensors compared ({sum(t.dim() == 0 for t in ts)} of them 0-d); differences: {diffs or 'none'}")
    assert len(ts) > 100 and any(t.dim() == 0 for t in ts)
    assert diffs == []
    del old, on
    torch.cuda.empty_cache()


def _snapshot(la):
    import copy
    return {"net": {k: v.detach().clone() for k, v in la.net.state_dict().items()},
            "opt": copy.deepcopy(la.opt.state_dict()), "sched": copy.deepcopy(la.sched.state_dict()), "cfg": la.cfg}


def _restore(la, s):
    import copy
    with torch.no_grad():
        la.net.load_state_dict(s["net"])
    la.opt.load_state_dict(copy.deepcopy(s["opt"]))
    la.sched.load_state_dict(copy.deepcopy(s["sched"]))
    la.cfg = s["cfg"]
