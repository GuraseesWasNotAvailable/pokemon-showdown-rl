"""
train.py --fast-ingest (fast_ingest.py), held to the code it stands in for:

  * gae_batch: a whole message's advantages and returns are, bit for bit, torch.cat of
    ppo_core.gae_from_episode over its episodes, on 10,000 random episodes (lengths 1 to 400,
    single steps, dones in the middle, episodes that did not end), with the dynamic lambda and
    without it; anything it does not take (another dtype or device, lengths that do not add up)
    it hands back to the per-episode loop
  * the episode ids by np.repeat are the learner's torch.full chunks, joined
  * GpuObsBuffer: rows in order, a view at the update, growth past its size, a refusal on a
    message of another layout or on no room (then the update joins its observations as before)
  * DeviceEpisodeDataset: the same minibatches, row for row, as the CPU data set
  * the learner end to end without a network: the same packed messages through the loop with the
    flag off and on give the update the same columns (every one torch.equal and the same bits),
    over three updates in a row (the buffer reused, grown once, and one update joined on the CPU
    after a failed copy), the same counters, and the same weights and temperatures at the weight
    store, in update order
  * WeightPusher: the dict the store gets (through Ray's pickling, ray_wire) is the one the learner
    used to build (keys in order, dtypes, shapes, strides, values, plain CPU memory), taken at the
    time of the push even when the weights change right after; with a slow store nothing is lost or
    reordered; a failing store or snapshot does not stop the pushes after it
  * the weights on the wire: OutOfBandTensor comes back as the tensor sent (bf16, bool, 0-dim,
    empty, NaN and -0.0 bits), anything not wire_ready goes as itself, and a push pickles a few
    kilobytes in band instead of every weight (Ray holds the Python lock while it pickles); the
    first out-of-band update is looked at without waiting until it has run, and a store that could
    not unpickle it gets the weights as the tensors themselves from then on
  * the bench's verdict on the fixed cost per update (compare_paths): the measured fixed parts and
    the intercept with one slope for both paths hold against noise and slow updates, where each
    path's own least-squares intercept did not
  * the flag: off by default, train.py gives the learner no keyword then, and the learner never
    imports fast_ingest

    python -m pytest tests/test_fast_ingest.py -q -p no:cacheprovider

The GPU parts are skipped without CUDA. Nothing here starts Ray, a server or a training run, and
nothing is removed.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import threading
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import config as cfg_mod  # noqa: E402
import fast_ingest  # noqa: E402
import learner as learner_mod  # noqa: E402
import ppo_core  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU = DEV == "cuda"
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CUDA")


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal dtype, shape and bytes (so also -0.0 against 0.0 and NaN payloads)."""
    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    a, b = a.detach().contiguous().cpu(), b.detach().contiguous().cpu()
    if a.dtype == torch.bool:
        return torch.equal(a, b)
    return a.reshape(-1).view(torch.uint8).numpy().tobytes() == b.reshape(-1).view(torch.uint8).numpy().tobytes()


def equal_column(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a.cpu(), b.cpu()) and same_bits(a, b)


# --------------------------------------------------------------------------- advantages


def _random_episode(rng, T: int):
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
        dones[-1] = 0.0                          # not ended: the value after the last step counts (0.0)
    return rewards, values, dones, variances


def _random_lengths(rng, n: int):
    """Episode lengths 1 to 400: many single steps and short games, some long ones."""
    out = []
    for _ in range(n):
        u = rng.random()
        if u < 0.15:
            out.append(1)
        elif u < 0.75:
            out.append(int(rng.integers(2, 60)))
        elif u < 0.97:
            out.append(int(rng.integers(60, 200)))
        else:
            out.append(int(rng.integers(200, 401)))
    return out


def _messages(n_episodes: int, seed: int):
    """Packed messages of 1 to 32 episodes, as a worker's SyncLearnerClient sends them."""
    rng = np.random.default_rng(seed)
    lengths = _random_lengths(rng, n_episodes)
    lengths[:3] = [1, 400, 1]
    i = 0
    while i < len(lengths):
        k = int(rng.integers(1, 33))
        L = lengths[i:i + k]
        i += k
        eps = [_random_episode(rng, T) for T in L]
        yield tuple(np.concatenate([e[j] for e in eps]) for j in range(4)) + (np.asarray(L, dtype=np.int32),)


GAE_SETTINGS = [
    # the learner's: config.LearnerConfig's gamma, gae_lambda and dynamic-lambda bounds
    dict(gamma=0.999, lam=0.75, lam_min=0.55, lam_max=0.95, lam_eps=1e-8),
    dict(gamma=0.99, lam=0.95, lam_min=0.3, lam_max=0.99, lam_eps=1e-6),
    dict(gamma=1.0, lam=0.0, lam_min=0.55, lam_max=0.95, lam_eps=1e-8),
    dict(gamma=0.9, lam=1.0, lam_min=0.0, lam_max=1.0, lam_eps=1e-8),
]


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic_lambda", "static_lambda"])
def test_gae_batch_is_the_per_episode_loop_bit_for_bit(dynamic):
    n_eps = n_msgs = 0
    for k, (r, v, d, var, L) in enumerate(_messages(10_000, seed=11 if dynamic else 12)):
        s = GAE_SETTINGS[k % len(GAE_SETTINGS)]
        last = (0.0, 0.0, 0.37, -1.2)[k % 4]
        rt, vt, dt, vart = (torch.from_numpy(x) for x in (r, v, d, var))
        want_a, want_r, curr = [], [], 0
        for T in L.tolist():
            a, b = ppo_core.gae_from_episode(rt[curr:curr + T], vt[curr:curr + T], dt[curr:curr + T], gamma=s["gamma"],
                                             lam=s["lam"], last_value=last, variances=vart[curr:curr + T] if dynamic else None,
                                             lam_min=s["lam_min"], lam_max=s["lam_max"], lam_eps=s["lam_eps"])
            want_a.append(a)
            want_r.append(b)
            curr += T
        got = fast_ingest.gae_batch(rt, vt, dt, L, variances=vart if dynamic else None, last_value=last, **s)
        assert got is not None
        assert same_bits(got[0], torch.cat(want_a)) and same_bits(got[1], torch.cat(want_r)), (k, s, last)
        n_eps += len(L)
        n_msgs += 1
    assert n_eps == 10_000 and n_msgs > 300


def test_gae_batch_on_the_learners_settings_with_a_cpu_tensor_per_column():
    """What the learner hands it: float32 views from torch.from_numpy, lengths int32, last value 0."""
    lc = cfg_mod.RunConfig.default().learner
    for r, v, d, var, L in _messages(500, seed=5):
        rt, vt, dt, vart = (torch.from_numpy(x).float() for x in (r, v, d, var))
        got = fast_ingest.gae_batch(rt, vt, dt, L, gamma=lc.gamma, lam=lc.gae_lambda, variances=vart,
                                    lam_min=lc.dynamic_lambda_min, lam_max=lc.dynamic_lambda_max, lam_eps=lc.dynamic_lambda_eps)
        want = fast_ingest.gae_per_episode(rt, vt, dt, vart, L, lc)
        assert same_bits(got[0], want[0]) and same_bits(got[1], want[1])


def test_gae_batch_hands_anything_else_back_to_the_loop():
    r = torch.randn(10)
    v = torch.randn(10)
    d = torch.zeros(10)
    L = np.array([4, 6], dtype=np.int32)
    kw = dict(gamma=0.999, lam=0.75)
    assert fast_ingest.gae_batch(r, v, d, L, **kw) is not None
    assert fast_ingest.gae_batch(r.double(), v.double(), d.double(), L, **kw) is None          # another dtype
    assert fast_ingest.gae_batch(r, v, d, np.array([4, 5]), **kw) is None                       # lengths do not add up
    assert fast_ingest.gae_batch(r, v, d, np.array([4.0, 6.0]), **kw) is None                   # not integers
    assert fast_ingest.gae_batch(r, v, d, np.array([], dtype=np.int32), **kw) is None           # no episodes
    assert fast_ingest.gae_batch(r, v, d, np.array([11, -1]), **kw) is None                     # negative
    assert fast_ingest.gae_batch(r, v[:9], d, L, **kw) is None                                  # columns of other lengths
    assert fast_ingest.gae_batch(r, v, d, L, variances=torch.rand(10).double(), **kw) is None
    if GPU:
        assert fast_ingest.gae_batch(r.cuda(), v.cuda(), d.cuda(), L, **kw) is None
    # zero-length episodes among the others: what the loop gives (it skips them)
    L0 = np.array([0, 4, 0, 6, 0], dtype=np.int64)
    got = fast_ingest.gae_batch(r, v, d, L0, variances=torch.rand(10), **kw)
    lc = dataclasses.replace(cfg_mod.RunConfig.default().learner, gamma=0.999, gae_lambda=0.75)
    var = torch.rand(10)
    got = fast_ingest.gae_batch(r, v, d, L0, variances=var, gamma=lc.gamma, lam=lc.gae_lambda)
    want = fast_ingest.gae_per_episode(r, v, d, var, L0, lc)
    assert same_bits(got[0], want[0]) and same_bits(got[1], want[1])


def test_episode_ids_are_the_learners_chunks():
    L = np.array([3, 1, 0, 5, 2], dtype=np.int32)
    want = torch.cat([torch.full((int(n),), 17 + i, dtype=torch.long) for i, n in enumerate(L.tolist())])
    got = fast_ingest.episode_ids(L, 17)
    assert got.dtype == torch.int64 and torch.equal(got, want)


def _pending_inputs(seed: int, dynamic: bool, n_eps: int = 3000):
    """Messages as the learner holds them: float32 tensors (read-only numpy underneath), int32 lengths."""
    out = []
    for r, v, d, var, L in _messages(n_eps, seed=seed):
        for x in (r, v, d, var):
            x.flags.writeable = False
        rt, vt, dt, vart = (torch.from_numpy(x) for x in (r, v, d, var))
        out.append((rt, vt, dt, vart if dynamic else None, L))
    return out


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic_lambda", "static_lambda"])
def test_pending_advantages_fill_every_message_with_the_per_episode_loops_bits(dynamic):
    """An update's worth of messages, filled at once, against gae_from_episode episode by episode (the learner's loop)."""
    lc = cfg_mod.RunConfig.default().learner
    settings = fast_ingest.gae_settings(lc, dynamic)
    msgs = _pending_inputs(31 if dynamic else 32, dynamic)
    pend = fast_ingest.PendingAdvantages()
    outs = []
    for r, v, d, var, L in msgs:
        plain = fast_ingest.plain_message(r, v, d, L, var)
        assert plain is not None
        a, b = torch.empty_like(v), torch.empty_like(v)
        pend.add(settings, r, v, d, var, plain, a, b)
        outs.append((a, b))
    assert len(pend) == len(msgs) and pend.rows == sum(int(m[0].shape[0]) for m in msgs)
    pend.flush()
    assert len(pend) == 0 and pend.flushes == 1 and pend.fallbacks == 0
    for (r, v, d, var, L), (a, b) in zip(msgs, outs):
        want = fast_ingest.gae_per_episode(r, v, d, var, L, lc)
        assert same_bits(a, want[0]) and same_bits(b, want[1])
    pend.flush()                                        # nothing pending: nothing done
    assert pend.flushes == 1


def test_pending_advantages_with_changed_settings_and_a_failing_batch(monkeypatch, capsys):
    """Runs of messages with other settings each get their own; a batch that fails falls back to the loop, same bits."""
    base = cfg_mod.RunConfig.default().learner
    other = dataclasses.replace(base, gamma=0.99, gae_lambda=0.9, dynamic_lambda_min=0.3)
    msgs = _pending_inputs(41, True, n_eps=600)
    plan = []
    for k, (r, v, d, var, L) in enumerate(msgs):
        cfg = other if (k // 5) % 2 else base
        dyn = k % 7 != 3                                   # some without the dynamic lambda
        plan.append((cfg, dyn))

    def fill(pend):
        outs = []
        for (r, v, d, var, L), (cfg, dyn) in zip(msgs, plan):
            a, b = torch.empty_like(v), torch.empty_like(v)
            pend.add(fast_ingest.gae_settings(cfg, dyn), r, v, d, var if dyn else None,
                     fast_ingest.plain_message(r, v, d, L, var if dyn else None), a, b)
            outs.append((a, b))
        pend.flush()
        return outs

    def check(outs):
        for (r, v, d, var, L), (cfg, dyn), (a, b) in zip(msgs, plan, outs):
            want = fast_ingest.gae_per_episode(r, v, d, var if dyn else None, L, cfg)
            assert same_bits(a, want[0]) and same_bits(b, want[1])

    p1 = fast_ingest.PendingAdvantages()
    check(fill(p1))
    assert p1.fallbacks == 0
    calls = {"n": 0}
    real = fast_ingest.gae_batch

    def broken(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("made up")
        if calls["n"] == 3:
            return None
        return real(*a, **k)
    monkeypatch.setattr(fast_ingest, "gae_batch", broken)
    p2 = fast_ingest.PendingAdvantages()
    check(fill(p2))
    assert p2.fallbacks == 2 and "computed one episode at a time" in capsys.readouterr().out
    # clear() drops what is pending without computing it
    p3 = fast_ingest.PendingAdvantages()
    r, v, d, var, L = msgs[0]
    a = torch.full_like(v, 7.0)
    p3.add(fast_ingest.gae_settings(base, True), r, v, d, var, L.astype(np.int64), a, a.clone())
    p3.clear()
    p3.flush()
    assert len(p3) == 0 and p3.flushes == 0 and bool((a == 7.0).all())


def test_plain_message_is_what_gae_batch_takes():
    r, v, d = torch.randn(10), torch.randn(10), torch.zeros(10)
    L = np.array([4, 6], dtype=np.int32)
    got = fast_ingest.plain_message(r, v, d, L)
    assert got.dtype == np.int64 and got.tolist() == [4, 6]
    assert fast_ingest.plain_message(r, v, d, L, torch.rand(10)) is not None
    for bad in [(r.double(), v, d, L, None), (r, v, d, np.array([4.0, 6.0]), None), (r, v, d, np.array([5, 6]), None),
                (r, v[:9], d, L, None), (r, v, d, L, torch.rand(10).double()), (r, v, d, np.array([], dtype=np.int32), None)]:
        assert fast_ingest.plain_message(*bad) is None


def test_gae_batch_is_faster_than_the_loop_on_a_typical_message():
    lc = cfg_mod.RunConfig.default().learner
    msgs = list(_messages(2000, seed=9))
    tens = [tuple(torch.from_numpy(x) for x in m[:4]) + (m[4],) for m in msgs]
    t0 = time.perf_counter()
    for r, v, d, var, L in tens:
        fast_ingest.gae_per_episode(r, v, d, var, L, lc)
    slow = time.perf_counter() - t0
    t0 = time.perf_counter()
    for r, v, d, var, L in tens:
        fast_ingest.gae_batch(r, v, d, L, gamma=lc.gamma, lam=lc.gae_lambda, variances=var)
    fast = time.perf_counter() - t0
    print(f"\n{len(msgs)} messages: per-episode loop {1000 * slow:.0f} ms, gae_batch {1000 * fast:.0f} ms")
    assert fast < slow


# --------------------------------------------------------------------------- the GPU buffer and the data set


@needs_gpu
def test_gpu_obs_buffer_rows_in_order_growth_and_refusals():
    buf = fast_ingest.GpuObsBuffer(DEV, rows_hint=10)
    parts = [torch.randn(n, 6).half() for n in (4, 5, 7)]       # 16 rows: one growth past 10
    for p in parts:
        assert buf.append(p)
    assert buf.grows == 1 and buf.capacity >= 16
    view = buf.take(16)
    assert view is not None and view.device.type == "cuda" and equal_column(view, torch.cat(parts))
    # the next update starts at row 0 of the same storage
    ptr = buf.buf.data_ptr()
    p2 = torch.randn(3, 6).half()
    assert buf.append(p2) and buf.buf.data_ptr() == ptr
    assert equal_column(buf.take(3), p2)
    # a count that does not match: None, and the reason
    buf.append(torch.randn(2, 6).half())
    assert buf.take(3) is None and "holds 2 rows" in buf.reason
    # another layout in the middle of an update: refused until the update is over
    assert buf.append(torch.randn(2, 6).half())
    assert not buf.append(torch.randn(2, 7).half()) and "after" in buf.reason
    assert not buf.append(torch.randn(2, 6).half())
    assert buf.take(4) is None
    # at the start of an update a new layout is taken
    assert buf.append(torch.randn(2, 7).float()) and equal_column(buf.take(2), buf.buf[:2])


@needs_gpu
def test_gpu_obs_buffer_survives_no_room(monkeypatch):
    buf = fast_ingest.GpuObsBuffer(DEV, rows_hint=4)
    real_empty = torch.empty

    def no_room(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")
    monkeypatch.setattr(torch, "empty", no_room)
    assert not buf.append(torch.randn(3, 5).half())
    assert "no room" in buf.reason and buf.failures == 1
    monkeypatch.setattr(torch, "empty", real_empty)
    assert buf.take(3) is None
    assert buf.append(torch.randn(3, 5).half()) and buf.take(3) is not None


def _slow_copy(delay_s: float, fail_at=None, seen=None):
    """A GpuObsBuffer._copy that sleeps first (and raises at copy number fail_at), recording the thread it ran on."""
    real = fast_ingest.GpuObsBuffer._copy
    count = {"n": 0}

    def copy(dst, src, stream):
        count["n"] += 1
        if seen is not None:
            seen.add(threading.current_thread().name)
        time.sleep(delay_s)
        if fail_at is not None and count["n"] == fail_at:
            raise RuntimeError("made up copy failure")
        real(dst, src, stream)
    return staticmethod(copy)


@needs_gpu
def test_gpu_obs_buffer_copies_on_its_thread_and_take_waits_for_them(monkeypatch):
    seen = set()
    monkeypatch.setattr(fast_ingest.GpuObsBuffer, "_copy", _slow_copy(0.01, seen=seen))
    rng = np.random.default_rng(1)
    buf = fast_ingest.GpuObsBuffer(DEV, rows_hint=2000)            # room for every update: no growth, which waits
    for upd in range(3):
        parts = [torch.from_numpy(rng.standard_normal((int(rng.integers(1, 40)), 9)).astype(np.float16)) for _ in range(25)]
        t0 = time.perf_counter()
        for p in parts:
            assert buf.append(p)
        handed_s = time.perf_counter() - t0
        n = sum(int(p.shape[0]) for p in parts)
        view = buf.take(n)                          # waits for the 25 slow copies
        assert view is not None and equal_column(view, torch.cat(parts)), upd
        assert handed_s < 0.25 * 25 * 0.01          # handing them over did not wait for them
    assert seen == {"learner-obs-copy"} and buf.handed == 75 and buf.inline == 0
    assert buf.wait_s > 0.0 and buf.thread_s >= 75 * 0.01 * 0.9 and buf.grows == 0
    buf.close()
    assert buf.append(torch.ones(3, 9).half()) and buf.inline == 1      # after close: on the caller's thread
    assert equal_column(buf.take(3), torch.ones(3, 9).half())


@needs_gpu
def test_gpu_obs_buffer_growth_waits_for_copies_into_the_old_storage(monkeypatch):
    monkeypatch.setattr(fast_ingest.GpuObsBuffer, "_copy", _slow_copy(0.02))
    buf = fast_ingest.GpuObsBuffer(DEV, rows_hint=10)
    parts = [torch.full((4, 5), float(k), dtype=torch.float16) for k in range(9)]     # 36 rows: grows twice past 10
    for p in parts:
        assert buf.append(p)
    assert buf.grows >= 2
    assert equal_column(buf.take(36), torch.cat(parts))


@needs_gpu
def test_gpu_obs_buffer_a_copy_failing_on_its_thread_fails_only_that_update(monkeypatch):
    monkeypatch.setattr(fast_ingest.GpuObsBuffer, "_copy", _slow_copy(0.0, fail_at=3))
    buf = fast_ingest.GpuObsBuffer(DEV, rows_hint=50)
    for k in range(5):
        assert buf.append(torch.full((2, 4), float(k)).half())     # the failure is not known at the hand-over
    assert buf.take(10) is None and "failed on its thread" in buf.reason and buf.failures == 1
    parts = [torch.full((3, 4), float(k)).half() for k in range(4)]
    for p in parts:
        assert buf.append(p)
    assert equal_column(buf.take(12), torch.cat(parts))
    # reset() (the learner's dataset was cleared) also waits and forgets
    assert buf.append(parts[0])
    buf.reset()
    assert buf.n == 0 and buf.ok


@needs_gpu
def test_gpu_obs_buffer_with_a_dead_thread_fails_that_update_and_copies_inline_after(monkeypatch):
    buf = fast_ingest.GpuObsBuffer(DEV, rows_hint=50)

    def dies(self):
        self._jobs.get()                            # takes one copy and never makes it
    monkeypatch.setattr(fast_ingest.GpuObsBuffer, "_run", dies)
    assert buf.append(torch.ones(2, 4).half())
    buf._thread.join(5)
    assert buf.take(2) is None and "thread is gone" in buf.reason
    assert buf.append(torch.ones(3, 4).half()) and buf.inline == 1
    assert equal_column(buf.take(3), torch.ones(3, 4).half())


@needs_gpu
def test_the_next_updates_copies_wait_for_the_update_still_reading_the_buffer():
    """
    The update reads the buffer on the stream; the next message's rows go into the same storage. They
    must land after the update's kernels: here the GPU is kept busy (torch.cuda._sleep) before a
    checksum of the update's rows, and the next copy is handed over right after it is queued.
    """
    buf = fast_ingest.GpuObsBuffer(DEV, rows_hint=4096)
    rng = np.random.default_rng(2)
    for k in range(5):
        first = torch.from_numpy(rng.standard_normal((4096, 64)).astype(np.float16))
        assert buf.append(first)
        view = buf.take(4096)
        want = first.double().sum()
        torch.cuda._sleep(200_000_000)              # the update's kernels, still running
        got = view.double().sum()                   # queued after them, reads rows 0..4095
        nxt = torch.from_numpy(rng.standard_normal((4096, 64)).astype(np.float16))
        assert buf.append(nxt)                      # rows 0..4095 again, by the copy thread
        torch.cuda.synchronize()
        assert abs(float(got) - float(want)) < 1e-6 * (1.0 + abs(float(want))), k
        assert equal_column(buf.take(4096), nxt)
    buf.close()


@needs_gpu
def test_device_data_set_cuts_the_same_minibatches():
    n, W = 2048, 33
    g = torch.Generator().manual_seed(3)
    obs = torch.randn(n, W, generator=g).half()
    lens = []
    while sum(lens) < n:
        lens.append(int(torch.randint(1, 90, (1,), generator=g)))
    lens[-1] -= sum(lens) - n
    ep = fast_ingest.episode_ids(np.array(lens), 0)
    cols = (obs, torch.randint(0, 14, (n,), generator=g), torch.randn(n, generator=g), torch.randn(n, generator=g),
            torch.randn(n, generator=g), torch.randn(n, generator=g), torch.zeros(n), ep)
    extras = {"read": torch.randint(-1, 2, (n,), generator=g).to(torch.int8),
              "set_labels": torch.randint(0, 9, (n, 48), generator=g).to(torch.int16)}
    a = ppo_core.AsyncEpisodeDataset(14, "cpu")
    a.add_steps(*cols, extras=extras)
    b = fast_ingest.DeviceEpisodeDataset(14, "cpu")
    b.add_steps(obs.to(DEV), *cols[1:], extras=extras)
    ta, tb = a.tensorize(), b.tensorize()
    assert a.extra_names == b.extra_names == ("read", "set_labels")
    assert tb[0].device.type == "cuda" and all(t.device.type == "cpu" for t in tb[1:])
    assert all(equal_column(x, y) for x, y in zip(ta, tb))
    for dev in (None, DEV):
        for epoch in range(2):
            torch.manual_seed(100 + epoch)
            ma = list(a.iter_minibatches(256, shuffle_episodes=True, device=dev))
            torch.manual_seed(100 + epoch)
            mb = list(b.iter_minibatches(256, shuffle_episodes=True, device=dev))
            assert len(ma) == len(mb) == n // 256
            for x, y in zip(ma, mb):
                assert all(equal_column(p, q) for p, q in zip(x, y))


def test_swap_out_without_obs_is_swap_out_for_every_other_column():
    a = ppo_core.AsyncEpisodeDataset(14, "cpu")
    b = ppo_core.AsyncEpisodeDataset(14, "cpu")
    for k in range(3):
        n = 5 + k
        cols = (torch.randn(n, 4).half(), torch.arange(n), torch.randn(n), torch.randn(n), torch.randn(n), torch.randn(n),
                torch.zeros(n), torch.full((n,), k, dtype=torch.long))
        ex = {"read": torch.ones(n, dtype=torch.int8)} if k != 1 else None
        for ds in (a, b):
            ds.add_steps(*cols, extras=ex)
    da, na = a.swap_out_columns()
    db, nb = fast_ingest.swap_out_columns_without_obs(b)
    assert na == nb == ("read",) and len(da) == len(db) == 9
    assert db[0].shape[0] == 0 and db[0].dtype == da[0].dtype
    assert all(equal_column(x, y) for x, y in zip(da[1:], db[1:]))
    assert len(a) == len(b) == 0 and b.obs == []


# --------------------------------------------------------------------------- the learner, end to end without a network


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


def ray_wire(obj):
    """
    obj as a Ray actor gets it: pickled as Ray pickles a task's arguments (protocol 5, out-of-band
    buffers to the object store; ray.cloudpickle when Ray is there) and unpickled from those bytes.
    Returns (the object received, bytes pickled in band, bytes handed over out of band).
    """
    try:
        import ray.cloudpickle as pk
    except Exception:       # noqa: BLE001
        import pickle as pk
    bufs = []
    blob = pk.dumps(obj, protocol=5, buffer_callback=bufs.append)
    raws = [bytearray(b.raw()) for b in bufs]         # the object store's copies
    return pk.loads(blob, buffers=raws), len(blob), sum(len(r) for r in raws)


class RecordingStore:
    """
    The weight store as the learner uses it: every call recorded in order, with what the real store
    would hold (the arguments through ray_wire, so later changes on the sender's side do not leak
    in) and how many bytes Ray would pickle in band (where it holds the Python lock).
    """

    def __init__(self, delay_s: float = 0.0, fail_on=()):
        self.calls = []
        self.delay_s = delay_s
        self.fail_on = set(fail_on)
        self.threads = set()
        self.inband = []                # per update: bytes pickled in band, bytes out of band
        self.update = _Remote(self._update)
        self.set_temp = _Remote(self._set_temp)

    def _update(self, weights, version):
        self.threads.add(threading.current_thread().name)
        (weights, version), inb, oob = ray_wire((weights, version))
        self.inband.append((inb, oob))
        if self.delay_s:
            time.sleep(self.delay_s)
        if version in self.fail_on:
            raise RuntimeError(f"store refused version {version}")
        self.calls.append(("update", version, {k: v.clone() for k, v in weights.items()},
                           {k: (v.dtype, tuple(v.shape), tuple(v.stride()), v.device.type, v.is_pinned()) for k, v in weights.items()},
                           list(weights.keys())))

    def _set_temp(self, temp):
        self.calls.append(("temp", float(temp)))


class TinyNet(nn.Module):
    """What the learner pushes and the dataset's device are read from: bf16 weights, a buffer, a transposed weight."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 8)
        self.b = nn.Linear(8, 3, bias=False)
        self.register_buffer("steps", torch.zeros(2, dtype=torch.long))
        self.c = nn.Parameter(torch.randn(5, 7).t())        # a non-contiguous parameter


class FakeOpt:
    param_groups = [{"name": "pi_wd", "lr": 3e-5}]


def _learner_cfg(steps: int, dynamic: bool = True, read_bonus: float = 0.3):
    cfg = cfg_mod.RunConfig.default()
    lc = dataclasses.replace(cfg.learner, mode="ppo", device=DEV, steps_per_update=steps, resume=False, save_every_updates=10**9,
                             use_dynamic_lambda=dynamic, read_bonus=read_bonus, read_anneal_start=0, read_anneal_end=10**7)
    return dataclasses.replace(cfg, learner=lc)


def _bare_learner(cfg, fast: bool, store):
    la = object.__new__(learner_mod.LearnerActor)
    la.run_cfg, la.cfg, la.weight_store = cfg, cfg.learner, store
    la.dataset = ppo_core.AsyncEpisodeDataset(act_dim=cfg.env.act_dim, device="cpu")
    la.update_idx = la.total_episodes = la.total_steps = la.ep_counter = 0
    la.read_origin = 0
    la.human = None
    torch.manual_seed(0)                    # the same starting weights for both learners
    la.net = TinyNet().to(DEV).to(torch.bfloat16)
    la.opt = FakeOpt()
    la.sched = None
    la.fast_ingest = bool(fast)
    la._obs_buf = None
    la._pusher = fast_ingest.WeightPusher(store) if fast else None
    la._gae_pending = fast_ingest.PendingAdvantages() if fast else None
    la._fast_fallbacks = 0
    return la


def _packed_messages(n_msgs: int, seed: int, width: int = 64, with_extras: bool = True):
    """Worker-like packed messages: float16 observations, int64 actions, float32 columns, int32 lengths, extras on some."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_msgs):
        L = np.asarray(_random_lengths(rng, int(rng.integers(1, 33))), dtype=np.int32)
        if i == 7:
            L = np.asarray([400, 400, 400], dtype=np.int32)          # a big one: the buffer has to grow
        S = int(L.sum())
        eps = [_random_episode(rng, int(T)) for T in L]
        r, v, d, var = (np.concatenate([e[j] for e in eps]) for j in range(4))
        obs = rng.standard_normal((S, width)).astype(np.float16)
        obs.flags.writeable = False                                  # as Ray hands them over
        act = rng.integers(0, 14, S).astype(np.int64)
        logp = (-rng.random(S)).astype(np.float32)
        if i == 11:
            L = L.astype(np.float64)                                 # lengths gae_batch does not take: the loop, on arrival
        msg = ("packed", obs, act, logp, v, var, r, d, L)
        if with_extras and i % 3 != 1:
            ex = {"read": rng.integers(-1, 2, S).astype(np.int8), "boosted": rng.integers(0, 2, S).astype(np.int8),
                  "opp_action": rng.integers(-1, 14, S).astype(np.int64),
                  "set_labels": rng.integers(0, 9, (S, 48)).astype(np.int16)}
            msg = msg + (ex,)
        out.append(msg)
    return out


def _drive(la, msgs, groups, capture, fail_after_updates=None):
    """
    The learner's own _loop over the messages, `groups[k]` of them queued at once, until every one is
    taken in. fail_after_updates=k: the first GPU copy after k updates fails (no room, made up).
    """
    real_append = fast_ingest.GpuObsBuffer.append
    failed = {"done": False}

    def flaky_append(self, obs):
        if fail_after_updates is not None and not failed["done"] and la.update_idx == fail_after_updates:
            failed["done"] = True
            return self._fail("no room on cuda: made up by the test")
        return real_append(self, obs)

    def fake_ppo_update(net, opt, dataset, **kw):
        data = dataset.tensorize()
        torch.manual_seed(1000 + len(capture))
        mbs = [tuple(t.cpu() for t in mb) for mb in dataset.iter_minibatches(256, shuffle_episodes=True, device=DEV)]
        capture.append({"cols": tuple(t.cpu().clone() for t in data), "names": tuple(dataset.extra_names), "mbs": mbs,
                        "obs_device": data[0].device.type, "type": type(dataset).__name__})
        with torch.no_grad():                       # the weights change at every update, as a real one would
            for p in net.parameters():
                p.add_(0.5)
            net.steps.add_(1)
        return ppo_core.PPOUpdateStats(approx_kl=0.01, clip_frac=0.1, entropy=0.5, v_loss=1.0, pg_loss=0.0,
                                       total_loss=1.0, n_mb=len(mbs))

    async def go():
        la._q = asyncio.Queue(maxsize=5)
        task = asyncio.ensure_future(la._loop())
        i = 0
        for g in groups:
            want = la.total_steps
            for msg in msgs[i:i + g]:
                want += len(msg[2])
                await la._q.put(msg)
            i += g
            while la.total_steps < want:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
        assert i == len(msgs)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    old = learner_mod.ppo_update
    learner_mod.ppo_update = fake_ppo_update
    fast_ingest.GpuObsBuffer.append = flaky_append
    try:
        asyncio.run(go())
    finally:
        learner_mod.ppo_update = old
        fast_ingest.GpuObsBuffer.append = real_append
    if la._pusher is not None:
        assert la._pusher.flush(30)


@pytest.mark.parametrize("dynamic", [True, False], ids=["dynamic_lambda", "static_lambda"])
def test_the_update_gets_the_same_columns_with_the_flag(dynamic, capsys):
    msgs = _packed_messages(40, seed=21 if dynamic else 22)
    total = sum(len(m[2]) for m in msgs)
    steps = total // 4 + 1                     # three updates and some rows left over
    rng = np.random.default_rng(5)
    groups = []
    left = len(msgs)
    while left:
        g = min(left, int(rng.integers(1, 5)))  # 1 to 4 queued at once, as the loop drains them
        groups.append(g)
        left -= g
    runs = {}
    for fast in (False, True):
        store = RecordingStore()
        la = _bare_learner(_learner_cfg(steps, dynamic=dynamic), fast, store)
        if fast and GPU:
            # a buffer a quarter of an update's size (the learner makes it 1.5 times): it has to grow
            la._obs_buf = fast_ingest.GpuObsBuffer(DEV, rows_hint=steps // 4)
        cap = []
        # with the flag: the third update's buffer fails at its first copy, so that update joins on the CPU
        _drive(la, msgs, groups, cap, fail_after_updates=2 if fast else None)
        runs[fast] = (la, store, cap)
    out = capsys.readouterr().out
    (a, sa, ca), (b, sb, cb) = runs[False], runs[True]
    assert len(ca) == len(cb) == 3
    for k, (x, y) in enumerate(zip(ca, cb)):
        assert x["names"] == y["names"] and len(x["cols"]) == len(y["cols"]) >= 8
        for i, (p, q) in enumerate(zip(x["cols"], y["cols"])):
            assert equal_column(p, q), (k, i)
        assert len(x["mbs"]) == len(y["mbs"])
        for mx, my in zip(x["mbs"], y["mbs"]):
            assert all(equal_column(p, q) for p, q in zip(mx, my))
    if GPU:
        assert [c["obs_device"] for c in cb] == ["cuda", "cuda", "cpu"]
        assert [c["type"] for c in cb] == ["DeviceEpisodeDataset", "DeviceEpisodeDataset", "AsyncEpisodeDataset"]
        assert b._obs_buf.grows >= 1 and b._fast_fallbacks == 1
        assert "joins its observations on the CPU (no room on cuda: made up by the test" in out
    assert [c["obs_device"] for c in ca] == ["cpu"] * 3
    assert (a.total_steps, a.total_episodes, a.ep_counter, a.update_idx) == (b.total_steps, b.total_episodes, b.ep_counter, b.update_idx)
    # what is left for the next update is the same too (its advantages filled as the next update would)
    assert len(a.dataset) == len(b.dataset) > 0
    assert b._gae_pending.flushes == 3 and b._gae_pending.fallbacks == 0 and len(b._gae_pending) > 0
    b._gae_pending.flush()
    assert all(equal_column(p, q) for p, q in zip(a.dataset.tensorize(), b.dataset.tensorize()))
    # the weight store: the same calls in the same order, the same dicts
    assert [c[:2] if c[0] == "update" else c for c in sa.calls] == [c[:2] if c[0] == "update" else c for c in sb.calls]
    for x, y in zip(sa.calls, sb.calls):
        if x[0] == "update":
            assert x[4] == y[4]                                  # keys in the same order
            assert x[3] == y[3]                                  # dtype, shape, stride, device, not pinned
            assert all(equal_column(x[2][k], y[2][k]) for k in x[4])
    assert sb.threads == {"learner-weight-push"} and sa.threads == {"MainThread"}
    assert b._pusher.pushed == 3 and b._pusher.errors == 0 and b._pusher.sync_pushes == 0
    assert a._gae_pending is None


def test_without_gpu_the_flag_still_gives_the_same_columns(monkeypatch):
    """A learner on the CPU has no buffer: the observations are joined as before, the rest is the flag's."""
    msgs = _packed_messages(12, seed=3)
    steps = sum(len(m[2]) for m in msgs) // 2 + 1
    runs = []
    for fast in (False, True):
        store = RecordingStore()
        la = _bare_learner(_learner_cfg(steps), fast, store)
        la.net = TinyNet().to(torch.bfloat16)                    # on the CPU
        cap = []
        _drive(la, msgs, [1] * len(msgs), cap)
        runs.append((la, cap, store))
    (a, ca, sa), (b, cb, sb) = runs
    assert b._obs_buf is None and len(ca) == len(cb) == 1
    assert all(equal_column(p, q) for p, q in zip(ca[0]["cols"], cb[0]["cols"]))
    assert len(sa.calls) == len(sb.calls) == 2


# --------------------------------------------------------------------------- the weight push


def _today(state_dict):
    return {k: v.cpu().detach() for k, v in state_dict.items()}


def test_pusher_hands_over_the_dict_the_learner_built_at_the_time_of_the_push():
    net = TinyNet().to(DEV).to(torch.bfloat16)
    store = RecordingStore(delay_s=0.02)
    p = fast_ingest.WeightPusher(store)
    want = []
    for version in range(1, 21):
        sd = net.state_dict()
        # cloned: on a machine without CUDA .cpu() is the parameter itself, which changes below
        want.append({k: v.clone() for k, v in _today(sd).items()})
        p.push(sd, version, 1.0 / version)
        with torch.no_grad():                    # changed right after the push, on the same stream
            for t in net.parameters():
                t.mul_(1.01).add_(0.25)
            net.steps.add_(3)
    assert p.flush(30)
    ups = [c for c in store.calls if c[0] == "update"]
    temps = [c[1] for c in store.calls if c[0] == "temp"]
    assert [c[1] for c in ups] == list(range(1, 21))                         # none lost, none reordered
    assert temps == [1.0 / v for v in range(1, 21)]
    kinds = [c[0] for c in store.calls]
    assert kinds == ["update", "temp"] * 20                                  # update then set_temp, as before
    for c, w in zip(ups, want):
        assert c[4] == list(w.keys())
        assert c[3] == {k: (v.dtype, tuple(v.shape), tuple(v.stride()), "cpu", False) for k, v in w.items()}
        assert all(equal_column(c[2][k], w[k]) for k in w)
    assert p.pushed == 20 and p.errors == 0 and p.sync_pushes == 0
    p.close()


def test_pusher_goes_on_after_a_failing_store_and_a_failing_snapshot(monkeypatch):
    net = TinyNet().to(DEV).to(torch.bfloat16)
    store = RecordingStore(fail_on={2})
    p = fast_ingest.WeightPusher(store)
    for v in (1, 2, 3):
        p.push(net.state_dict(), v, 0.5)
    assert p.flush(10)
    assert [c[1] for c in store.calls if c[0] == "update"] == [1, 3] and p.errors == 1
    # a snapshot that cannot be taken: pushed on the caller's thread as the learner always did
    real = torch._foreach_copy_

    def broken(*a, **k):
        raise RuntimeError("made up")
    monkeypatch.setattr(torch, "_foreach_copy_", broken)
    p.push(net.state_dict(), 4, 0.5)
    monkeypatch.setattr(torch, "_foreach_copy_", real)
    assert p.sync_pushes == 1 and [c[1] for c in store.calls if c[0] == "update"][-1] == 4
    p.push(net.state_dict(), 5, 0.5)
    assert p.flush(10) and [c[1] for c in store.calls if c[0] == "update"][-1] == 5
    # a dead thread: every push on the caller's thread
    p.close()
    p.push(net.state_dict(), 6, 0.5)
    assert p.sync_pushes == 2 and [c[1] for c in store.calls if c[0] == "update"][-1] == 6


def _wire_corner_cases():
    """What a state dict can hold, and some it cannot (for the tensors that must go as themselves)."""
    g = torch.Generator().manual_seed(7)
    base = torch.randn(40, generator=g)
    return {"bf16": torch.randn(33, 17, generator=g).to(torch.bfloat16), "fp32": torch.randn(5, 3, 2, generator=g),
            "fp16": torch.randn(9, generator=g).half(), "fp64": torch.randn(3, generator=g).double(),
            "bool": torch.rand(4, 6, generator=g) > 0.5, "int64": torch.randint(-5, 5, (7,), generator=g),
            "int8": torch.randint(-5, 5, (3, 3), generator=g).to(torch.int8), "uint8": torch.randint(0, 255, (5,), generator=g).to(torch.uint8),
            "scalar": torch.tensor(3.25), "scalar_long": torch.tensor(11), "empty": torch.empty(0, 4),
            "bits": torch.tensor([float("nan"), -0.0, float("inf"), -float("inf")]),
            # not wire_ready: they go as the tensor itself, as before
            "noncontig": torch.randn(5, 7, generator=g).t(), "view_at_offset": base[10:20], "part_of_storage": base[:5],
            "grad": torch.randn(3, generator=g).requires_grad_(), "param": nn.Parameter(torch.randn(2, generator=g))}


def test_out_of_band_tensors_come_back_as_the_tensors_sent():
    """for_the_wire through Ray's pickling: the store gets every tensor back, same dtype, shape, strides and bits."""
    import warnings
    sent = _wire_corner_cases()
    ready = {k: fast_ingest.wire_ready(v) for k, v in sent.items()}
    assert [k for k, r in ready.items() if not r] == ["noncontig", "view_at_offset", "part_of_storage", "grad", "param"]
    ref, _, _ = ray_wire(sent)                          # the tensors themselves, as before
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # no "not writable" warnings on either side
        got, _inb, oob = ray_wire(fast_ingest.for_the_wire(sent))
    assert list(got.keys()) == list(sent.keys())
    assert oob == sum(v.numel() * v.element_size() for k, v in sent.items() if ready[k])
    for k, v in sent.items():
        g, r = got[k], ref[k]
        assert type(g) is type(r), k
        assert (g.dtype, tuple(g.shape), tuple(g.stride()), g.requires_grad, g.is_pinned()) == \
               (r.dtype, tuple(r.shape), tuple(r.stride()), r.requires_grad, r.is_pinned()), k
        assert same_bits(g.detach(), v.detach()), k
        if ready[k]:
            # memory of its own, all of it (as unpickling a tensor gives), that can be written to
            assert g.storage_offset() == 0 and g.untyped_storage().nbytes() == g.numel() * g.element_size()
            if g.numel():
                g.view(-1)[0] = g.view(-1)[-1]


@needs_gpu
def test_a_pinned_snapshot_goes_out_of_band_and_comes_back_plain():
    net = TinyNet().to(DEV).to(torch.bfloat16)
    store = RecordingStore()
    p = fast_ingest.WeightPusher(store)
    p.push(net.state_dict(), 1, 1.0)
    assert p.flush(30)
    p.close()
    (_, version, got, meta, keys) = store.calls[0]
    want = _today(net.state_dict())
    assert version == 1 and keys == list(want.keys())
    assert meta == {k: (v.dtype, tuple(v.shape), tuple(v.stride()), "cpu", False) for k, v in want.items()}
    assert all(equal_column(got[k], want[k]) for k in want)


def test_the_push_pickles_a_few_kilobytes_in_band_not_the_weights():
    """
    Ray holds the Python lock while it pickles; out-of-band bytes it copies without it. The push used
    to hand Ray the tensors, which pickle in band in full (torch.save into bytes, about 60 ms for the
    policy's 32 MB on box c), on the push thread while the learner's thread took in the next update's
    messages: the review of v2-fast-ingest measured the learner's time outside the update at -35%
    instead of the -50% asked. Now only names, dtypes and shapes go in band.
    """
    torch.manual_seed(0)
    sd = {f"layer{i}.weight": torch.randn(256, 512).to(torch.bfloat16 if i % 2 else torch.float32) for i in range(24)}
    sd["c"] = torch.randn(64, 32).t()                          # non-contiguous: as the tensor itself
    sd = {k: v.to(DEV) for k, v in sd.items()}
    total = sum(v.numel() * v.element_size() for v in sd.values())
    nc = sd["c"].numel() * sd["c"].element_size()
    for oob, check in ((True, lambda inb, out: inb < 16 * 1024 + 2 * nc and out == total - nc),
                       (False, lambda inb, out: inb > total and out == 0)):
        store = RecordingStore()
        p = fast_ingest.WeightPusher(store, out_of_band=oob)
        for v in range(1, 4):
            p.push(sd, v, 0.5)
        assert p.flush(30)
        p.close()
        assert [c[1] for c in store.calls if c[0] == "update"] == [1, 2, 3]
        assert len(store.inband) == 3 and all(check(inb, out) for inb, out in store.inband), (oob, store.inband, total)
        for c in store.calls:
            if c[0] == "update":
                assert all(equal_column(c[2][k], sd[k].cpu()) for k in sd)


def test_the_learner_pushes_out_of_band_with_the_flag():
    """LearnerActor(fast_ingest=True) makes its WeightPusher with the out-of-band push on."""
    assert fast_ingest.WeightPusher(RecordingStore()).out_of_band is True
    import inspect
    src = inspect.getsource(learner_mod.LearnerActor.__init__)
    assert "fast_ingest.WeightPusher(weight_store)" in src


class ObjectRef:
    """A stand-in for ray.ObjectRef (the pusher knows a Ray store's reply by this class name)."""

    def __init__(self, error: str = ""):
        self.error = error


class RefusingStore(RecordingStore):
    """A Ray store whose process cannot unpickle out-of-band weights: such an update fails there and changes nothing."""

    def __init__(self, refuse: bool):
        super().__init__()
        self.refuse = refuse
        self.update = _Remote(self._update_ref)

    def _update_ref(self, weights, version):
        if self.refuse and any(isinstance(v, fast_ingest.OutOfBandTensor) for v in weights.values()):
            return ObjectRef("ModuleNotFoundError: No module named 'fast_ingest'")
        self._update(weights, version)
        return ObjectRef()


class GetTimeoutError(Exception):
    """A stand-in for ray.exceptions.GetTimeoutError (the pusher knows it by this class name)."""


def _fake_get(asked, not_yet: int = 0):
    """ray.get for ObjectRef stand-ins: not_yet times 'not run yet' first, then the store's verdict."""
    def get(ref, timeout=None):
        asked.append(timeout)
        if len(asked) <= not_yet:
            raise GetTimeoutError("not yet")
        if ref.error:
            raise RuntimeError(ref.error)
        return None
    return get


@pytest.mark.parametrize("not_yet", [0, 2])
def test_the_first_out_of_band_push_is_confirmed_without_waiting(not_yet):
    net = TinyNet().to(DEV).to(torch.bfloat16)
    store = RefusingStore(refuse=False)
    p = fast_ingest.WeightPusher(store)
    asked = []
    p._get = _fake_get(asked, not_yet)
    for v in range(1, 7):
        p.push(net.state_dict(), v, 0.5)
        assert p.flush(30)
        assert p.oob_confirmed == (v >= 2 + not_yet)
    p.close()
    # looked at from the second push on, never waiting, until it had run; never again after
    assert asked == [0] * (1 + not_yet) and p.out_of_band and p.errors == 0
    assert [c[1] for c in store.calls if c[0] == "update"] == list(range(1, 7))


def test_a_store_that_cannot_take_them_out_of_band_gets_the_tensors(capsys):
    net = TinyNet().to(DEV).to(torch.bfloat16)
    store = RefusingStore(refuse=True)
    p = fast_ingest.WeightPusher(store)
    asked = []
    p._get = _fake_get(asked, not_yet=1)
    want = []
    for v in (1, 2, 3, 4):
        want.append({k: t.clone() for k, t in _today(net.state_dict()).items()})
        p.push(net.state_dict(), v, 0.5)
        assert p.flush(30)
        with torch.no_grad():
            for t in net.parameters():
                t.add_(1.0)
    p.close()
    out = capsys.readouterr().out
    assert not p.out_of_band and "ModuleNotFoundError" in p.oob_refused and asked == [0, 0]
    assert "could not take version 1 out of band" in out and "from version 3 on the weights go as the tensors" in out
    # versions 1 and 2 went out of band and failed in the store (it kept what it had); from 3 on the tensors, in order
    assert [c[0] if c[0] == "temp" else (c[0], c[1]) for c in store.calls] == ["temp", "temp", ("update", 3), "temp",
                                                                                ("update", 4), "temp"]
    for c, w in zip([c for c in store.calls if c[0] == "update"], want[2:]):
        assert all(equal_column(c[2][k], w[k]) for k in w)


# --------------------------------------------------------------------------- the bench's verdict on the fixed cost


def _bench():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fast_ingest_bench_for_tests", os.path.join(REPO, "bench", "fast_ingest_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_rows(rng, fixed: float, slope: float, noise: float, spikes=()):
    """`time` rows of one run: 6 updates at 49 minibatches and 6 at 96, sec = fixed parts + slope x n_mb + noise."""
    rows = []
    for i, mb in enumerate([49] * 6 + [96] * 6):
        prep = fixed - 0.01 + float(rng.normal(0, 0.005)) + (1.6 if i in spikes else 0.0)
        upd = slope * mb + float(rng.normal(0, noise))
        rows.append({"n_mb": mb, "sec": prep + 0.01 + upd, "prep_s": prep, "copy_s": 0.0, "tail_s": 0.01, "ingest_s": 0.3})
    return rows


def test_the_fixed_cost_verdict_holds_against_noise_and_a_slow_update():
    """
    The first report gave the fixed cost per update as the least-squares intercept of each path on its
    own, and its numbers did not reproduce (the review: -0.38 s, -0.71 s, -0.91 s for the same change,
    a flag-on intercept below zero, and a 2.4 s flag-off figure from a few slow joins). compare_paths
    takes the measured fixed parts and the intercept with one slope for both paths, by medians.
    """
    B = _bench()
    rng = np.random.default_rng(11)
    err_sep, err_shared, err_fixed = [], [], []
    for _ in range(300):
        off = _synthetic_rows(rng, 0.80, 0.093, 0.35, spikes=(4, 5, 10, 11) if rng.random() < 0.3 else ())
        on = _synthetic_rows(rng, 0.14, 0.093, 0.35)
        c_off, _ = B.fit([r["n_mb"] for r in off], [r["sec"] for r in off])
        c_on, _ = B.fit([r["n_mb"] for r in on], [r["sec"] for r in on])
        err_sep.append(abs((c_on - c_off) - (-0.66)))
        out = B.compare_paths(off, on, boot=0)
        err_shared.append(abs(out["intercept_s"]["change"] - (-0.66)))
        err_fixed.append(abs(out["fixed_parts_s"]["change"] - (-0.66)))
    assert max(err_fixed) < 0.03                          # the direct measure: a slow update or noise does not move it
    assert np.median(err_shared) < 0.6 * np.median(err_sep)
    assert np.median(err_shared) < 0.2
    # the 90% interval of the change covers the truth on most runs, and the slow updates are counted
    covered = 0
    for _ in range(40):
        out = B.compare_paths(_synthetic_rows(rng, 0.80, 0.093, 0.35, spikes=(4, 5)), _synthetic_rows(rng, 0.14, 0.093, 0.35), boot=300)
        lo, hi = out["intercept_s"]["change_90"]
        covered += lo < -0.66 < hi
        assert out["slow_updates"] == {"off": 2, "on": 0}
    assert covered >= 28, covered


# --------------------------------------------------------------------------- the flag


def test_the_flag_is_off_by_default_and_reaches_the_learner_only_when_given():
    import inspect
    import train
    assert inspect.signature(learner_mod.LearnerActor.__init__).parameters["fast_ingest"].default is False
    base = ["--mode", "ppo", "--ports", "11448-11451"]
    off = train.parse_args(base)
    on = train.parse_args(base + ["--fast-ingest"])
    assert off.fast_ingest is False and on.fast_ingest is True
    assert train.fast_ingest_kwargs(off) == {} and train.fast_ingest_kwargs(on) == {"fast_ingest": True}
    # nothing of it goes into the run's config (so none into a checkpoint)
    assert train.build_config(off).as_dict() == train.build_config(on).as_dict()


def test_without_the_flag_the_learner_never_imports_fast_ingest(monkeypatch):
    import builtins
    hits = []
    real = builtins.__import__

    def spy(name, *a, **k):
        if name == "fast_ingest":
            hits.append(name)
        return real(name, *a, **k)
    msgs = _packed_messages(8, seed=4)
    steps = sum(len(m[2]) for m in msgs) // 2 + 1
    store = RecordingStore()
    la = _bare_learner(_learner_cfg(steps), False, store)
    cap = []
    monkeypatch.setattr(builtins, "__import__", spy)
    _drive(la, msgs, [1] * len(msgs), cap)
    monkeypatch.setattr(builtins, "__import__", real)
    assert len(cap) == 1 and hits == []
    assert la._obs_buf is None and la._pusher is None
