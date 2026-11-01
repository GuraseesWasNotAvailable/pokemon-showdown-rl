"""
The inference actor's per-battle history kept in one pooled tensor (inference.py) against the
per-battle lists of clones it replaced: the old InferenceActor and the new one are driven
with the same long random sequence of calls and must answer every one of them identically.

    python -m pytest tests/test_speed_kvpool.py -q -p no:cacheprovider

What the sequence holds: battles starting, playing on, ending (clear_cache), coming back under
the same tag, far more battles alive than the pool starts with (it has to grow several
times), the same tag twice in one call, tags cleared that were never seen, and new weights
with a new temperature arriving in the middle. The old actor wiped every history on new
weights and the new one keeps them, so the old one's wipe is switched off for the
comparison (a test further down holds the new behaviour on its own). Every call
is made under the same random seed in both actors, and compared are: the observation tensor
the model is handed, its policy logits, value logits and value, the sampled actions, their
log-probabilities, the value variance, and from time to time every stored history.

The old actor is the reference commit of bench/micro_ref.py with its own old model code and
is fed float32 observations, as the old worker sent them. The new one is fed the float16 the
new worker sends. No Ray: the weight store is a stand-in object and ray.get is replaced by a
function that hands its argument back.

One test needs no old code: a call that fails while new sides are being given their rows
(the pool has to grow, or a row is being wiped, and the GPU is out of memory) must store
nothing, as a failed call never did. It is held to a twin actor that never had the failure.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import random
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402
import torch  # noqa: E402

import inference as new_inf  # noqa: E402
import ppo_core as new_core  # noqa: E402
from bench import micro_ref  # noqa: E402
from config import RunConfig  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU = DEV == "cuda"
NEVER = 1e18            # a "last sync" this far ahead keeps _sync_weights from asking the store


class _Remote:
    """handle.method.remote(...) of a Ray actor, answered on the spot."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


class FakeStore:
    def __init__(self):
        self.version, self.weights, self.temp = 0, None, 1.0
        self.get_version = _Remote(lambda: self.version)
        self.get_weights = _Remote(lambda: self.weights)
        self.get_temp = _Remote(lambda: self.temp)


class FakeRay:
    @staticmethod
    def get(x, timeout=None):
        return x


@pytest.fixture(scope="module")
def ref():
    try:
        return micro_ref.Reference.export()
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(f"no reference code to compare with: {e}")


@pytest.fixture(scope="module")
def data():
    obs, _ep, kind = micro_ref.observations(n_min=2048)
    return obs, kind


def _small(cfg, cls=dataclasses):
    """No checkpoint from disk, this device, and so few battles that the pool starts tiny."""
    return cls.replace(
        cfg,
        learner=cls.replace(cfg.learner, resume=False),
        infer=cls.replace(cfg.infer, device=DEV),
        rollout=cls.replace(cfg.rollout, target_concurrent_battles=2, rooms_per_pair=2),
    )


@pytest.fixture()
def actors(ref, monkeypatch):
    old_inf = ref.load("inference")
    old_core = ref.load("ppo_core")
    assert old_inf.ppo_core is old_core and not hasattr(old_core, "episodic_block_mask")
    assert hasattr(old_inf.InferenceActor, "_zero_cache_row") and not hasattr(new_inf.InferenceActor, "_zero_cache_row")
    if GPU:
        micro_ref.shared_compiled_flex_attention(old_core, new_core)
    monkeypatch.setattr(old_inf, "ray", FakeRay)
    monkeypatch.setattr(new_inf, "ray", FakeRay)
    monkeypatch.setattr(new_inf.InferenceActor, "KV_POOL_MIN_ROWS", 8)
    store = FakeStore()

    # Each constructor wraps its module's flex_attention in torch.compile. Put back what was
    # there, so both models run the one function compiled above (or the plain one on a CPU).
    saved_old, saved_new = old_core.flex_attention, new_core.flex_attention
    torch.manual_seed(7)
    try:
        with ref.active():
            from config import RunConfig as OldRunConfig
            old = old_inf.InferenceActor(_small(OldRunConfig.default()), store)
        new = new_inf.InferenceActor(_small(RunConfig.default()), store)
    finally:
        old_core.flex_attention, new_core.flex_attention = saved_old, saved_new
    assert type(old.net) is old_core.PokeTransformer and type(new.net) is new_core.PokeTransformer
    new.net.load_state_dict(old.net.state_dict(), strict=True)
    old._last_sync_time = new._last_sync_time = NEVER
    # The old actor wiped every history when new weights came in. The new one keeps them.
    monkeypatch.setattr(old, "clear_all_caches", lambda: None)
    assert int(new._kv_pool.shape[1]) == 8
    return old, new, store


def _spy_on_forward(actor, log):
    inner = actor.net.forward

    def forward(obs, **kw):
        out = inner(obs, **kw)
        log.append((obs, out[0], out[1], out[2]))
        return out

    actor.net.forward = forward


def _same_histories(old, new, tags) -> None:
    for tag in tags:
        row = new._kv_slot.get(tag)
        if tag not in old.kv_cache_by_tag:
            # Never answered for since it was last cleared: nothing stored on either side
            # (the new actor may hold a row for it only after an answer, so it holds none).
            assert row is None, tag
            continue
        assert row is not None, tag
        assert old.kv_cache_len_by_tag[tag] == int(new._kv_len[row]), tag
        for layer, cached in enumerate(old.kv_cache_by_tag[tag]):
            assert micro_ref.same_bits(cached[0], new._kv_pool[layer, row]), (tag, layer)


def test_same_answers_over_a_long_random_run(actors, data, caplog):
    old, new, store = actors
    obs_np, _kind = data
    seen_old, seen_new = [], []
    _spy_on_forward(old, seen_old)
    _spy_on_forward(new, seen_new)
    rng = random.Random(20261005)
    nrng = np.random.default_rng(20261005)
    steps = 1500 if GPU else 160
    # New weights used to wipe every history, so they come rarely enough for a battle to outlive
    # the 64-turn window in between. The marathon sides are in nearly every call and never end.
    weight_steps = {300, 700, 1100, 1300} if GPU else {40, 90, 130}
    marathon = ["pmar000a_battle-gen9randombattle-1", "pmar000b_battle-gen9randombattle-1"]
    alive, ended = [], []
    made = 0
    counts = {"calls": 0, "rows": 0, "new_tags": 0, "returned_tags": 0, "ended": 0, "weights": 0, "duplicates": 0,
              "cleared_unknown": 0, "max_alive": 0, "max_history": 0}

    with caplog.at_level(logging.WARNING, logger="inference"):
        for step in range(steps):
            r = rng.random()
            if (r < 0.10 or len(alive) > 150) and alive and step not in weight_steps:
                # some battles end (many at once when a lot are alive, so the number stays bounded)
                for _ in range(rng.randint(1, 4) if len(alive) <= 150 else 40):
                    if not alive:
                        break
                    tag = alive.pop(rng.randrange(len(alive)))
                    old.clear_cache(tag), new.clear_cache(tag)
                    ended.append(tag)
                    counts["ended"] += 1
                if rng.random() < 0.3:
                    ghost = f"never_seen_{step}"
                    old.clear_cache(ghost), new.clear_cache(ghost)
                    if ended:
                        again = rng.choice(ended)            # cleared twice
                        old.clear_cache(again), new.clear_cache(again)
                    counts["cleared_unknown"] += 1
                continue
            if step in weight_steps:
                # the learner publishes new weights and a new temperature
                sd = old.net.state_dict()
                store.weights = {k: (v.float() + 0.02 * torch.randn_like(v.float())).to(v.dtype) if v.is_floating_point() else v.clone()
                                 for k, v in sd.items()}
                store.version += 1
                store.temp = rng.choice([1.0, 0.8, 1.25])
                old._last_sync_time = new._last_sync_time = None       # the next call asks the store
                counts["weights"] += 1

            B = rng.choice([1, 1, 2, 3, 5, 8, 13, 21, 32, 40])
            tags = [m for m in marathon if rng.random() < 0.9]
            for _ in range(B - len(tags)):
                k = rng.random()
                if k < 0.12 or not alive:
                    made += 1
                    tag = f"pabc{made % 7:03d}a_battle-gen9randombattle-{made}"
                    counts["new_tags"] += 1
                elif k < 0.16 and ended:
                    tag = ended.pop(rng.randrange(len(ended)))        # a tag that ended comes back
                    counts["returned_tags"] += 1
                else:
                    tag = rng.choice(alive)
                if tag in tags:
                    if rng.random() < 0.7:
                        continue                                       # mostly one row per side, as the worker sends
                    counts["duplicates"] += 1
                tags.append(tag)
                if tag not in alive:
                    alive.append(tag)
            if not tags:
                tags.append(marathon[0])
            rows = nrng.integers(0, obs_np.shape[0], len(tags))
            obs16 = np.ascontiguousarray(obs_np[rows])
            assert obs16.dtype == np.float16
            obs32 = np.ascontiguousarray(obs16, dtype=np.float32)      # what the old worker sent

            syncing = old._last_sync_time is None
            torch.manual_seed(1_000_000 + step)
            a = old.infer_batch(list(tags), obs32)
            torch.manual_seed(1_000_000 + step)
            b = new.infer_batch(list(tags), obs16)
            if syncing:
                assert old.current_version == new.current_version == store.version
                assert old.current_temp == new.current_temp == store.temp
                old._last_sync_time = new._last_sync_time = NEVER

            (o_obs, o_pi, o_v, o_ve), (n_obs, n_pi, n_v, n_ve) = seen_old.pop(), seen_new.pop()
            assert not seen_old and not seen_new
            assert o_obs.dtype == n_obs.dtype == torch.float32 and torch.equal(o_obs, n_obs), ("model input", step)
            assert torch.equal(o_pi, n_pi), ("policy logits", step, tags)
            assert torch.equal(o_v, n_v), ("value logits", step)
            assert torch.equal(o_ve, n_ve), ("expected value", step)
            for name, x, y in zip(("actions", "log-probabilities", "values", "value variance"), a, b):
                assert micro_ref.same_bits(x, y), (name, step, tags)
            counts["calls"] += 1
            counts["rows"] += len(tags)
            counts["max_alive"] = max(counts["max_alive"], len(alive))
            counts["max_history"] = max(counts["max_history"], int(new._kv_len.max()))
            if step % 100 == 0 or step == steps - 1:
                _same_histories(old, new, alive + ended + marathon)
                assert set(new._kv_slot) == set(old.kv_cache_by_tag)

    grown = [r for r in caplog.records if "history pool grown" in r.getMessage()]
    print("\n" + str(counts) + f" pool rows {int(new._kv_pool.shape[1])} after {len(grown)} growths")
    assert counts["max_alive"] > 8 * 4 and len(grown) >= 3 and int(new._kv_pool.shape[1]) >= 64, "the pool never had to grow"
    assert counts["weights"] >= 3 and counts["returned_tags"] >= 5 and counts["ended"] >= 20 and counts["duplicates"] >= 1
    if GPU:
        assert counts["max_history"] == new.net.kv_cache_len, "no battle lived past the history window"
    rows_in_use = list(new._kv_slot.values())
    assert len(rows_in_use) == len(set(rows_in_use)), "two sides share a row"
    assert sorted(rows_in_use + new._kv_free) == list(range(int(new._kv_pool.shape[1]))), "a row is neither in use nor free"


def test_a_reused_row_does_not_leak_what_its_last_owner_left(actors, data):
    """Rows are handed out again without being cleared at the time; the wipe happens on first use."""
    old, new, _store = actors
    obs_np, _kind = data
    obs = np.ascontiguousarray(obs_np[:3])
    for turn in range(5):
        torch.manual_seed(turn)
        new.infer_batch(["x1", "x2", "x3"], obs)
    row = new._kv_slot["x2"]
    new._kv_pool[:, row] = float("nan")          # the worst a previous owner could leave behind
    new._kv_pool[:, row, :, ::2] = float("-inf")
    new.clear_cache("x2")
    torch.manual_seed(99)
    b = new.infer_batch(["fresh"], obs[:1])
    assert new._kv_slot["fresh"] == row, "the freed row was expected to be the next one handed out"
    torch.manual_seed(99)
    a = old.infer_batch(["fresh"], np.ascontiguousarray(obs[:1], dtype=np.float32))
    assert all(np.isfinite(x).all() for x in b[1:])
    for x, y in zip(a, b):
        assert micro_ref.same_bits(x, y)
    assert int(new._kv_len[row]) == 1


def test_the_same_side_twice_in_one_call_keeps_the_last_entry(actors, data):
    old, new, _store = actors
    obs_np, _kind = data
    rows = [5, 6, 7, 8, 9]
    tags = ["d1", "d2", "d1", "d3", "d1"]
    for turn in range(3):
        obs16 = np.ascontiguousarray(obs_np[[r + 11 * turn for r in rows]])
        torch.manual_seed(turn)
        a = old.infer_batch(list(tags), np.ascontiguousarray(obs16, dtype=np.float32))
        torch.manual_seed(turn)
        b = new.infer_batch(list(tags), obs16)
        for x, y in zip(a, b):
            assert micro_ref.same_bits(x, y)
        _same_histories(old, new, ["d1", "d2", "d3"])
    assert int(new._kv_len[new._kv_slot["d1"]]) == 3


@pytest.fixture()
def twins(monkeypatch):
    """Two new actors with the same weights and a pool of two rows each. Needs no reference code."""
    if GPU:
        micro_ref.shared_compiled_flex_attention(new_core)
    monkeypatch.setattr(new_inf, "ray", FakeRay)
    monkeypatch.setattr(new_inf.InferenceActor, "_initial_kv_rows", lambda self: 2)
    store = FakeStore()
    saved = new_core.flex_attention
    torch.manual_seed(11)
    try:
        hit = new_inf.InferenceActor(_small(RunConfig.default()), store)
        twin = new_inf.InferenceActor(_small(RunConfig.default()), store)
    finally:
        new_core.flex_attention = saved
    twin.net.load_state_dict(hit.net.state_dict(), strict=True)
    hit._last_sync_time = twin._last_sync_time = NEVER
    assert int(hit._kv_pool.shape[1]) == int(twin._kv_pool.shape[1]) == 2
    return hit, twin


class _FirstWriteFails:
    """Stands in for the pool's length vector: its first indexed write fails the way an allocation on a full GPU does."""

    def __init__(self, inner):
        self.inner, self.failed = inner, False

    def __setitem__(self, key, value):
        if not self.failed:
            self.failed = True
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (made up by the test)")
        self.inner[key] = value

    def __getitem__(self, key):
        return self.inner[key]


@pytest.mark.parametrize("fails_in", ["growing the pool", "wiping the row"])
def test_a_failed_call_leaves_no_side_holding_another_sides_history(twins, data, monkeypatch, fails_in):
    """
    A call that fails while new sides are being given rows (the GPU ran out of memory) must
    store nothing. Before the fix the side that had already been handed a freed row stayed
    registered on it, unwiped, and at its next decision the network was given the dead
    battle's history and length. `twin` is the same actor without the failed call.
    """
    hit, twin = twins
    obs_np, _kind = data

    def call(actor, tags, first_row, seed):
        obs = np.ascontiguousarray(obs_np[first_row:first_row + len(tags)])
        torch.manual_seed(seed)
        return actor.infer_batch(list(tags), obs)

    for turn in range(3):                                  # A and B play three turns: both rows are in use
        for actor in (hit, twin):
            call(actor, ["A", "B"], 10 * turn, seed=turn)
    row_a = hit._kv_slot["A"]
    assert int(hit._kv_len[row_a]) == 3 and float(hit._kv_pool[:, row_a].float().abs().max()) > 0
    hit.clear_cache("A"), twin.clear_cache("A")            # A's battle ends: its row is free and keeps what it held
    free_before, slots_before = list(hit._kv_free), dict(hit._kv_slot)
    assert free_before == [row_a] and set(slots_before) == {"B"}

    if fails_in == "growing the pool":
        # C and D arrive together. C is handed A's old row, D needs a third row, and growing fails.
        real_grow, grown = hit._grow_kv_pool, []

        def grow(rows):
            grown.append(rows)
            if len(grown) == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (made up by the test)")
            return real_grow(rows)

        monkeypatch.setattr(hit, "_grow_kv_pool", grow)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            call(hit, ["C", "D"], 40, seed=100)
        assert grown == [4]
    else:
        # C arrives alone and is handed A's old row. Its contents are wiped, then writing the length fails.
        hit._kv_len = _FirstWriteFails(hit._kv_len)
        with pytest.raises(torch.cuda.OutOfMemoryError):
            call(hit, ["C"], 40, seed=100)
        assert hit._kv_len.failed
        hit._kv_len = hit._kv_len.inner
        assert int(hit._kv_len[row_a]) == 3, "the made-up failure was meant to leave the old length in place"

    assert hit._kv_slot == slots_before, "a side of the failed call is still registered"
    assert hit._kv_free == free_before, "the rows handed out in the failed call did not go back as they were"

    # The worker answers a failed call with action 0 and carries on: C's next decision comes in.
    a = call(hit, ["C"], 50, seed=101)
    b = call(twin, ["C"], 50, seed=101)
    for name, x, y in zip(("actions", "log-probabilities", "values", "value variance"), a, b):
        assert micro_ref.same_bits(x, y), name
    assert hit._kv_slot["C"] == row_a
    assert int(hit._kv_len[row_a]) == 1 == int(twin._kv_len[twin._kv_slot["C"]]), "C was treated as a side with a past"
    assert micro_ref.same_bits(hit._kv_pool[:, row_a], twin._kv_pool[:, twin._kv_slot["C"]])
    # and the side that was alive throughout plays on with its own history
    a = call(hit, ["B", "C", "D"], 60, seed=102)
    b = call(twin, ["B", "C", "D"], 60, seed=102)
    for name, x, y in zip(("actions", "log-probabilities", "values", "value variance"), a, b):
        assert micro_ref.same_bits(x, y), name
    for tag, turns in (("B", 4), ("C", 2), ("D", 1)):
        assert int(hit._kv_len[hit._kv_slot[tag]]) == turns == int(twin._kv_len[twin._kv_slot[tag]])
        assert micro_ref.same_bits(hit._kv_pool[:, hit._kv_slot[tag]], twin._kv_pool[:, twin._kv_slot[tag]]), tag
    rows_in_use = list(hit._kv_slot.values())
    assert len(rows_in_use) == len(set(rows_in_use))
    assert sorted(rows_in_use + hit._kv_free) == list(range(int(hit._kv_pool.shape[1])))


def test_new_weights_keep_every_battles_history(twins, data):
    """
    New weights from the store change the weights and nothing else: battles in progress keep
    their history. They used to be wiped, so every few seconds each battle went on with no
    memory. `twin` is given the same weights by hand, without a sync, and must answer the
    same. clear_all_caches still wipes everything when asked.
    """
    hit, twin = twins
    obs_np, _kind = data
    store = hit.weight_store

    def call(actor, tags, first_row, seed):
        obs = np.ascontiguousarray(obs_np[first_row:first_row + len(tags)])
        torch.manual_seed(seed)
        return actor.infer_batch(list(tags), obs)

    for turn in range(3):
        for actor in (hit, twin):
            call(actor, ["A", "B"], 10 * turn, seed=turn)
    store.weights = {k: (v.float() + 0.02 * torch.randn_like(v.float())).to(v.dtype) if v.is_floating_point() else v.clone()
                     for k, v in hit.net.state_dict().items()}
    store.version, store.temp = hit.current_version + 1, 0.8
    hit._last_sync_time = None                             # the next call asks the store
    twin.net.load_state_dict(store.weights, strict=True)
    twin.current_temp = 0.8

    a = call(hit, ["A", "B"], 30, seed=3)
    assert hit.current_version == store.version and hit.current_temp == 0.8, "the weights were never synced"
    b = call(twin, ["A", "B"], 30, seed=3)
    for name, x, y in zip(("actions", "log-probabilities", "values", "value variance"), a, b):
        assert micro_ref.same_bits(x, y), name
    for tag in ("A", "B"):
        assert int(hit._kv_len[hit._kv_slot[tag]]) == 4, f"{tag} lost its history on new weights"
        assert micro_ref.same_bits(hit._kv_pool[:, hit._kv_slot[tag]], twin._kv_pool[:, twin._kv_slot[tag]]), tag

    hit.clear_all_caches()
    assert hit._kv_slot == {} and sorted(hit._kv_free) == list(range(int(hit._kv_pool.shape[1])))
    call(hit, ["A"], 40, seed=4)
    assert int(hit._kv_len[hit._kv_slot["A"]]) == 1


def test_pool_size_follows_the_configured_battles():
    """Sides alive at once: battles rounded up to whole pairs, two sides each, a quarter spare."""
    probe = object.__new__(new_inf.InferenceActor)
    base = RunConfig.default()
    for battles, rooms, want in ((128, 16, 320), (100, 32, 320), (800, 32, 2000), (512, 16, 1280), (4, 16, 64)):
        probe.cfg = dataclasses.replace(base, rollout=dataclasses.replace(base.rollout, target_concurrent_battles=battles, rooms_per_pair=rooms))
        assert probe._initial_kv_rows() == want, (battles, rooms)
