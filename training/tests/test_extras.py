"""
Named per-step columns beside the eight fixed ones (step_columns.py; ported from stage3's
AsyncEpisodeDataset.add_steps(extras=) and the learner's _label_columns, generalised to named
integer and float columns a rollout worker emits).

    python -m pytest tests/test_extras.py -q -p no:cacheprovider -rs
    PSPPO_BASE_DIR=~/dev/v2-base python -m pytest tests/test_extras.py ...   (also: against today's code)

Held here:
  * step_columns: what is a usable column, and episodes joined with a fill value for the steps of
    an episode that came without a column;
  * the worker: an episode without columns goes to the learner exactly as before (the same call,
    the same queue item, the same packed batch); with them, they are checked (a bad column is left
    out and counted, never raised) and packed in the episodes' order;
  * the learner's loop: the columns of a packed batch land in the data set row for row with the
    eight, also over several batches of which only some have them;
  * the data set: the columns stay with their rows in tensorize(), in the minibatches cut by whole
    shuffled episodes, from the CPU and from the GPU, and inside a real ppo_update;
  * with no columns an update is bit-identical: the same learner with and without unused columns
    leaves the same weights, optimizer moments and loss figures; and with PSPPO_BASE_DIR naming a
    copy of today's code (the commit this branch starts from), the loop's data and a whole
    _perform_update are bit for bit what that code produces;
  * the learner's update line ends with adv_pre_std (the advantages' spread before they are
    normalised) and lr (the policy head's rate), and the supervisor's parser reads both.

Whole updates need a GPU (the attention has no backward pass on a CPU) and are skipped without one.
Nothing here starts Ray, a server or a training run. Nothing is removed.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import queue
import sys
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

import config as new_config  # noqa: E402
import learner as new_learner  # noqa: E402
import ppo_core as new_core  # noqa: E402
import step_columns  # noqa: E402
import worker as new_worker  # noqa: E402
import test_speed_update as tsu  # noqa: E402  (its learner and update helpers)
from bench import micro_ref  # noqa: E402

DEV = tsu.DEV
GPU = tsu.GPU
needs_gpu = pytest.mark.skipif(not GPU, reason="flex_attention has no backward on a CPU")
BASE_DIR = os.path.expanduser(os.environ.get("PSPPO_BASE_DIR", ""))


@pytest.fixture(scope="module")
def data():
    obs, ep, kind = micro_ref.observations(n_min=4096)
    return obs, ep, kind


@pytest.fixture(scope="module")
def base():
    """Today's code (PSPPO_BASE_DIR: the files of the commit this branch starts from), beside the working tree's."""
    if not BASE_DIR or not os.path.isdir(BASE_DIR):
        pytest.skip("PSPPO_BASE_DIR names no copy of today's code")
    ref = micro_ref.Reference(BASE_DIR, source="PSPPO_BASE_DIR")
    core, lrn, cfg = ref.load("ppo_core"), ref.load("learner"), ref.load("config")
    assert lrn.ppo_core is core and core is not new_core
    assert not hasattr(core.AsyncEpisodeDataset, "swap_out_columns"), "PSPPO_BASE_DIR holds this branch's code, not today's"
    if GPU:
        micro_ref.shared_compiled_flex_attention(core, new_core)
    core._learner_flex_compiled = True
    new_core._learner_flex_compiled = True
    return SimpleNamespace(core=core, learner=lrn, config=cfg, ref=ref)


# --------------------------------------------------------------------------- step_columns


def test_what_is_a_usable_column():
    cols, problems = step_columns.check_columns({
        "read_tier": np.array([0.0, 1.0, 0.6], dtype=np.float32), "opp_action": [3, -1, 7], "boosted": [True, False, True],
        "shares": np.zeros((3, 14), dtype=np.float16),
        "short": [1, 2], "text": ["a", "b", "c"], "nan": [0.0, float("nan"), 1.0], 5: [1, 2, 3], "": [1, 2, 3], "scalar": 4}, 3)
    assert sorted(cols) == ["boosted", "opp_action", "read_tier", "shares"]
    assert cols["boosted"].dtype == np.int8 and cols["boosted"].tolist() == [1, 0, 1]
    assert cols["opp_action"].tolist() == [3, -1, 7] and cols["shares"].shape == (3, 14)
    assert len(problems) == 6 and all("extra column" in p for p in problems), problems
    assert step_columns.check_columns(None, 3) == (None, [])
    assert step_columns.check_columns({}, 3) == (None, [])
    assert step_columns.check_columns([1, 2, 3], 3)[0] is None


def test_episodes_are_joined_with_a_fill_value_where_a_column_is_missing(monkeypatch):
    monkeypatch.setitem(step_columns.FILL, "opp_action", -1)
    a = {"read_tier": np.array([0.5, 1.0], dtype=np.float32), "opp_action": np.array([2, 3], dtype=np.int16)}
    b = None
    c = {"read_tier": np.array([0.6], dtype=np.float64)}
    out = step_columns.pack_columns([a, b, c], [2, 3, 1])
    assert sorted(out) == ["opp_action", "read_tier"]
    assert out["read_tier"].dtype == np.float64 and out["read_tier"].tolist() == [0.5, 1.0, 0.0, 0.0, 0.0, 0.6]
    assert out["opp_action"].dtype == np.int16 and out["opp_action"].tolist() == [2, 3, -1, -1, -1, -1]
    assert step_columns.pack_columns([None, None], [2, 3]) is None
    assert step_columns.pack_columns([{}, None], [2, 3]) is None


# --------------------------------------------------------------------------- the worker


def _buf(T: int, seed: int, extras=None):
    rng = np.random.default_rng(seed)
    buf = {"obs": [rng.random(6).astype(np.float16) for _ in range(T)], "act": [int(x) for x in rng.integers(0, 14, T)],
           "logp": [float(x) for x in -rng.random(T)], "val": [float(x) for x in rng.random(T)],
           "var": [float(x) for x in rng.random(T)], "phi": [float(x) for x in rng.random(T) - 0.5]}
    if extras is not None:
        buf["extras"] = extras
    return buf


def _worker(cls=None):
    w = object.__new__(cls or new_worker.RolloutWorker)
    w.cfg = new_config.RunConfig.default()
    w.calls = []
    w.learner_client = SimpleNamespace(submit_episode=lambda *a, **k: w.calls.append((a, k)))
    w.inference_actor = SimpleNamespace(clear_cache=SimpleNamespace(remote=lambda tag: None))
    w.released = []
    w.vec_env = SimpleNamespace(release_slot=lambda: w.released.append(1))
    w._traj = {}
    return w


def test_an_episode_without_columns_goes_out_as_before():
    w = _worker()
    w._traj["t"] = _buf(5, 1)
    w._finalize_trajectory("t", SimpleNamespace(won=True, lost=False))
    (args, kwargs), = w.calls
    assert len(args) == 7 and kwargs == {} and w.released == [1]
    assert args[5][-1] > 0.5 and args[6].tolist() == [0, 0, 0, 0, 1]            # the reward and done columns of before


def test_an_episode_with_columns_sends_them_checked():
    w = _worker()
    w._traj["t"] = _buf(4, 2, {"read_tier": [0.0, 1.0, 0.0, 0.6], "opp_action": [1, 9, -1, 3], "bad": [1, 2]})
    w._finalize_trajectory("t", SimpleNamespace(won=False, lost=True))
    (args, kwargs), = w.calls
    assert len(args) == 7 and sorted(kwargs) == ["extras"]
    assert sorted(kwargs["extras"]) == ["opp_action", "read_tier"]
    assert kwargs["extras"]["read_tier"].tolist() == [0.0, 1.0, 0.0, 0.6] and kwargs["extras"]["opp_action"].tolist() == [1, 9, -1, 3]
    assert sum(w.extra_problems.values()) == 1

    class Raises(new_worker.RolloutWorker):
        def episode_extras(self, tag, battle, buf, T):
            raise KeyError("boom")

    w = _worker(Raises)
    w._traj["t"] = _buf(3, 3)
    w._finalize_trajectory("t", SimpleNamespace(won=True, lost=False))
    (args, kwargs), = w.calls
    assert len(args) == 7 and kwargs == {} and sum(w.extra_problems.values()) == 1      # sent without them, counted, not raised

    class Override(new_worker.RolloutWorker):
        def episode_extras(self, tag, battle, buf, T):
            return {"boosted": np.arange(T) % 2 == 1}

    w = _worker(Override)
    w._traj["t"] = _buf(3, 4)
    w._finalize_trajectory("t", SimpleNamespace(won=True, lost=False))
    (_args, kwargs), = w.calls
    assert kwargs["extras"]["boosted"].tolist() == [0, 1, 0] and kwargs["extras"]["boosted"].dtype == np.int8


def _client():
    c = object.__new__(new_worker.SyncLearnerClient)
    c.q = queue.Queue(maxsize=100)
    return c


def _episode(T: int, seed: int):
    rng = np.random.default_rng(seed)
    return (rng.random((T, 6)).astype(np.float16), rng.integers(0, 14, T), -rng.random(T).astype(np.float32),
            rng.random(T).astype(np.float32), rng.random(T).astype(np.float32), rng.random(T).astype(np.float32),
            np.eye(1, T, T - 1, dtype=np.float32)[0])


def _drain(c):
    items = []
    while not c.q.empty():
        items.append(c.q.get_nowait())
    return items


def test_the_packed_batch_is_the_eight_as_before_without_columns_and_carries_them_with():
    c = _client()
    eps = [_episode(T, s) for s, T in enumerate((3, 1, 5))]
    for ep in eps:
        c.submit_episode(*ep)
    items = _drain(c)
    assert [len(it) for it in items] == [7, 7, 7]
    packed = new_worker.SyncLearnerClient._prepare_batch(items)
    assert len(packed) == 8
    want = [np.concatenate([e[i] for e in eps]).astype(t) for i, t in
            ((0, np.float16), (1, np.int64), (2, np.float32), (3, np.float32), (4, np.float32), (5, np.float32), (6, np.float32))]
    for got, exp in zip(packed[:7], want):
        assert got.dtype == exp.dtype and np.array_equal(got, exp)
    assert packed[7].tolist() == [3, 1, 5] and packed[7].dtype == np.int32

    c.submit_episode(*eps[0], extras={"row": np.array([0, 1, 2], dtype=np.int32)})
    c.submit_episode(*eps[1])
    c.submit_episode(*eps[2], extras={"row": np.array([4, 5, 6, 7, 8], dtype=np.int32), "tier": np.ones(5, dtype=np.float32)})
    items = _drain(c)
    assert [len(it) for it in items] == [8, 7, 8]
    packed2 = new_worker.SyncLearnerClient._prepare_batch(items)
    assert len(packed2) == 9 and all(np.array_equal(a, b) for a, b in zip(packed2[:8], packed))
    assert packed2[8]["row"].tolist() == [0, 1, 2, 0, 4, 5, 6, 7, 8]
    assert packed2[8]["tier"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]


# --------------------------------------------------------------------------- the learner's loop


def _bare_learner(learner_mod, core_mod, config_mod):
    """A LearnerActor that only takes data in: no model, no update (steps_per_update out of reach)."""
    cfg = config_mod.RunConfig.default()
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, steps_per_update=10**9, device="cpu", resume=False))
    la = object.__new__(learner_mod.LearnerActor)
    la.run_cfg, la.cfg = cfg, cfg.learner
    la.dataset = core_mod.AsyncEpisodeDataset(act_dim=cfg.env.act_dim, device="cpu")
    la.update_idx = la.total_episodes = la.total_steps = la.ep_counter = 0
    la._init_if_needed = lambda: None
    return la


def _feed(la, batches):
    async def go():
        la._q = asyncio.Queue()
        task = asyncio.ensure_future(la._loop())
        for packed in batches:
            await la.submit_packed_batch(*packed)
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    asyncio.run(go())
    return la


def _packed(seed, lengths, extras=None):
    c = _client()
    for i, T in enumerate(lengths):
        c.submit_episode(*_episode(T, 100 * seed + i), **({"extras": extras[i]} if extras and extras[i] is not None else {}))
    return new_worker.SyncLearnerClient._prepare_batch(_drain(c))


def test_the_learners_loop_keeps_the_columns_row_for_row():
    plain = [_packed(1, (4, 2)), _packed(2, (3,))]
    with_cols = [_packed(1, (4, 2), [{"row": np.arange(0, 4)}, {"row": np.arange(4, 6)}]), _packed(2, (3,)),
                 _packed(3, (2, 3), [None, {"tier": np.full(3, 0.6, dtype=np.float32)}])]
    a = _feed(_bare_learner(new_learner, new_core, new_config), plain)
    b = _feed(_bare_learner(new_learner, new_core, new_config), with_cols)
    da, db = a.dataset.tensorize(), b.dataset.tensorize()
    assert len(da) == 8 and len(db) == 10 and b.dataset.extra_names == ("row", "tier")
    n = int(da[1].shape[0])
    assert n == 9 and all(micro_ref.same_bits(x[:n], y[:n]) for x, y in zip(da, db[:8]))
    assert db[8].tolist() == [0, 1, 2, 3, 4, 5, 0, 0, 0] + [0] * 5
    assert db[9].tolist() == pytest.approx([0.0] * 9 + [0.0, 0.0, 0.6, 0.6, 0.6])
    assert db[7].tolist() == [0] * 4 + [1] * 2 + [2] * 3 + [3] * 2 + [4] * 3          # episode ids, one per episode
    # a batch whose column has the wrong number of rows: left out, counted, the loop goes on
    bad = _packed(4, (2, 2))
    bad = bad + ({"row": np.arange(3)},)
    c = _feed(_bare_learner(new_learner, new_core, new_config), [bad, _packed(5, (1,))])
    assert len(c.dataset) == 5 and len(c.dataset.tensorize()) == 8 and sum(c.extra_problems.values()) == 1


def test_the_learners_loop_without_columns_holds_what_todays_code_holds(base):
    batches = [_packed(1, (4, 2, 7)), _packed(2, (3,)), _packed(3, (1, 1, 30))]
    old = _feed(_bare_learner(base.learner, base.core, base.config), batches)
    new = _feed(_bare_learner(new_learner, new_core, new_config), batches)
    a, b = old.dataset.tensorize(), new.dataset.tensorize()
    assert len(a) == len(b) == 8
    assert all(micro_ref.same_bits(x, y) for x, y in zip(a, b))
    assert (old.total_steps, old.total_episodes, old.ep_counter) == (new.total_steps, new.total_episodes, new.ep_counter)


# --------------------------------------------------------------------------- the data set and the minibatches


def _cols_with_row_ids(obs_np, ep_np, n, seed):
    cols = list(tsu._update_data(obs_np, ep_np, n, seed))
    cols[3] = torch.arange(n, dtype=torch.float32)          # the value column is not used by the update: carry the row number in it
    return cols


def test_columns_stay_with_their_rows_across_chunks_and_minibatches(data):
    obs_np, ep_np, _kind = data
    n = 1024
    cols = _cols_with_row_ids(obs_np, ep_np, n, seed=5)
    ds = new_core.AsyncEpisodeDataset(14, "cpu")
    cut = [0, 300, 301, 700, n]
    for k, (a, b) in enumerate(zip(cut[:-1], cut[1:])):
        chunk = [c[a:b] for c in cols]
        extras = None if k == 1 else {"row": torch.arange(a, b, dtype=torch.int32), "tier": torch.full((b - a,), 0.25 * k, dtype=torch.float16)}
        ds.add_steps(*chunk, extras=extras)
    data_t = ds.tensorize()
    assert ds.extra_names == ("row", "tier") and len(data_t) == 10
    row, tier = data_t[8], data_t[9]
    assert row.dtype == torch.int32 and tier.dtype == torch.float16
    assert row.tolist() == list(range(300)) + [0] + list(range(301, n))                # the chunk without them: fill value 0
    assert ds.extra_columns()["row"] is row
    torch.manual_seed(11)
    seen = 0
    for mb in ds.iter_minibatches(256, shuffle_episodes=True):
        rows = mb[3].long()
        assert torch.equal(mb[8].long()[rows != 300], rows[rows != 300])
        assert torch.equal(mb[7], torch.from_numpy(ep_np[:n])[rows])                    # whole episodes, ids with their rows
        assert torch.equal(mb[9], tier[rows])
        seen += int(rows.numel())
    assert seen == n
    with pytest.raises(ValueError):
        ds.add_steps(*[c[:5] for c in cols], extras={"row": torch.arange(4)})
    data2, names = ds.swap_out_columns()
    assert names == ("row", "tier") and all(x is y for x, y in zip(data2, data_t)) and len(ds) == 0 and ds.extra_names == ()


@needs_gpu
def test_columns_cut_from_the_gpu_are_the_same_rows(data):
    obs_np, ep_np, _kind = data
    n = 1024
    cols = _cols_with_row_ids(obs_np, ep_np, n, seed=6)
    out = {}
    for where in (None, DEV):
        ds = new_core.AsyncEpisodeDataset(14, "cpu")
        ds.add_steps(*cols, extras={"row": torch.arange(n, dtype=torch.int64)})
        torch.manual_seed(12)
        out[where] = [tuple(t.cpu() for t in mb) for mb in ds.iter_minibatches(256, shuffle_episodes=True, device=where)]
    for a, b in zip(out[None], out[DEV]):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert torch.equal(a[8], a[3].long())


class _Recording(new_core.AsyncEpisodeDataset):
    """The minibatches ppo_update draws, kept."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.seen = []

    def iter_minibatches(self, *a, **k):
        for mb in super().iter_minibatches(*a, **k):
            self.seen.append(mb)
            yield mb


def _update(la, cols, extras, seed, **kw):
    ds = _Recording(14, "cpu")
    ds.add_steps(*cols, extras=extras)
    torch.manual_seed(seed)
    stats = new_core.ppo_update(net=la.net, opt=la.opt, dataset=ds, scheduler=la.sched, mode=la.cfg.mode, target_net=None,
                                **la.cfg.ppo_kwargs(), v_min=la.cfg.v_min, v_max=la.cfg.v_max, v_bins=la.cfg.v_bins, **kw)
    return stats, ds


@needs_gpu
@pytest.mark.parametrize("mode", ["ppo", "imitation"])
def test_unused_columns_leave_the_update_bit_identical_and_ride_in_every_minibatch(data, mode):
    obs_np, ep_np, _kind = data
    n = 1536
    cols = _cols_with_row_ids(obs_np, ep_np, n, seed=10)
    extras = {"row": torch.arange(n, dtype=torch.int32), "tier": (torch.arange(n) % 3).to(torch.float32) * 0.5,
              "shares": torch.zeros((n, 14), dtype=torch.float16)}
    plain, with_cols = tsu._new_learner(mode), tsu._new_learner(mode)
    tsu._copy_state(plain, with_cols)
    for step in range(2):
        for flag in (True, False):
            sa, _ = _update(plain, cols, None, 50 + step, data_on_device=flag)
            sb, ds = _update(with_cols, cols, extras, 50 + step, data_on_device=flag)
            want, got = tsu._state(plain), tsu._state(with_cols)
            diff = [k for k in want if not micro_ref.same_bits(want[k], got[k])]
            assert diff == [], (mode, step, flag, diff[:5])
            for f in ("approx_kl", "clip_frac", "entropy", "v_loss", "pg_loss", "total_loss", "n_mb", "acc"):
                assert getattr(sa, f) == getattr(sb, f), (mode, f)
            assert ds.seen and all(len(mb) == 11 for mb in ds.seen)
            at = {name: 8 + i for i, name in enumerate(ds.extra_names)}
            assert ds.extra_names == ("row", "shares", "tier")
            for mb in ds.seen:
                rows = mb[3].long().cpu()
                assert torch.equal(mb[at["row"]].long().cpu(), rows) and torch.equal(mb[at["tier"]].cpu(), extras["tier"][rows])
    del plain, with_cols
    torch.cuda.empty_cache()


@needs_gpu
@pytest.mark.parametrize("mode", ["ppo", "imitation"])
def test_a_whole_learner_update_is_bit_identical_to_todays_code(base, data, mode):
    """
    _perform_update of today's learner and of this branch's, from the same weights, data and seed:
    without columns, and (this branch) with unused columns. Every weight, fp32 master, Adam moment
    and the schedule must be equal bit for bit.
    """
    obs_np, ep_np, _kind = data
    n = 1536
    cols = tsu._update_data(obs_np, ep_np, n, seed=21)
    extras = {"row": torch.arange(n, dtype=torch.int32), "tier": torch.full((n,), 0.6)}
    old = tsu._old_learner(base, mode, save_every_updates=10**9)
    new, new_x = tsu._new_learner(mode, save_every_updates=10**9), tsu._new_learner(mode, save_every_updates=10**9)
    for other in (new, new_x):
        tsu._copy_state(old, other)
    for la, x in ((old, None), (new, None), (new_x, extras)):
        if x is None:
            la.dataset.add_steps(*cols)
        else:
            la.dataset.add_steps(*cols, extras=x)
        la.total_steps += n
        torch.manual_seed(77)
        asyncio.run(la._perform_update())
        assert la.update_idx == 1 and len(la.dataset) == 0, "the update did not run"
    want = tsu._state(old)
    for la in (new, new_x):
        got = tsu._state(la)
        assert want.keys() == got.keys()
        diff = [k for k in want if not micro_ref.same_bits(want[k], got[k])]
        assert diff == [], (mode, diff[:5])
    del old, new, new_x
    torch.cuda.empty_cache()


@needs_gpu
def test_the_update_line_ends_with_adv_pre_std_and_lr(data, capsys):
    import pipeline
    from bench import throughput
    obs_np, ep_np, _kind = data
    la = tsu._new_learner("ppo", save_every_updates=10**9)
    cols = tsu._update_data(obs_np, ep_np, 1024, seed=31)
    want_std = float(cols[4].float().std(unbiased=False))
    la.dataset.add_steps(*cols)
    la.total_steps += 1024
    asyncio.run(la._perform_update())
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[learner] upd="))
    keys = [kv.split("=")[0] for kv in line.split()[1:]]
    assert keys[-2:] == ["adv_pre_std", "lr"], line
    ev = pipeline.parse_trainer_line(line)
    assert ev["kind"] == "learner_update" and ev["nonfinite"] == []
    assert ev["values"]["adv_pre_std"] == pytest.approx(want_std, rel=1e-3)
    assert ev["values"]["lr"] == pytest.approx(la._policy_lr(), rel=1e-2) and la._policy_lr() > 0
    assert throughput.parse_learner(line)["adv_pre_std"] == ev["values"]["adv_pre_std"]
    del la
    torch.cuda.empty_cache()
