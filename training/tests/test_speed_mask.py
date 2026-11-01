"""
The attention's block mask built from episode ids (ppo_core.episodic_block_mask and
inference_block_mask) against torch's create_block_mask, which the code used before, and the
model's outputs with the new masks against the old code's, bit for bit.

    python -m pytest tests/test_speed_mask.py -q -p no:cacheprovider

"Old code" is the reference commit of bench/micro_ref.py, imported next to the working tree.
Observations are real ones when $PSPPO_OBS_SAMPLES names an obs_samples.npz written by
bench/micro_obs.py, made-up ones (inside every embedding table) otherwise. On a machine with
no GPU only the short sequences run: without the compiled kernel the old mask and the
attention itself need memory for every token pair.

Uses the GPU for a few seconds and about 3 GB of it for moments (that is the old mask).
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.attention.flex_attention import create_block_mask  # noqa: E402

import ppo_core as new_core  # noqa: E402
from bench import micro_ref  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU = DEV == "cuda"
TOK = new_core.TOKENS_PER_TURN


@pytest.fixture(scope="module")
def ref():
    try:
        return micro_ref.Reference.export()
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(f"no reference code to compare with: {e}")


@pytest.fixture(scope="module")
def old_core(ref):
    mod = ref.load("ppo_core")
    assert mod is not new_core and not hasattr(mod, "episodic_block_mask"), "the reference must be the code before the change"
    if GPU:
        micro_ref.shared_compiled_flex_attention(mod, new_core)
    return mod


@pytest.fixture(scope="module")
def data():
    obs, ep, kind = micro_ref.observations(n_min=4096)
    return obs, ep, kind


def _net_pair(ref, old_core, bf16: bool):
    """The old class and the new class holding the same weights."""
    torch.manual_seed(1234)
    with ref.active():
        from config import RunConfig as OldRunConfig
        old_net = OldRunConfig.default().make_model()
    from config import RunConfig
    new_net = RunConfig.default().make_model()
    assert type(old_net) is old_core.PokeTransformer and type(new_net) is new_core.PokeTransformer
    assert list(old_net.state_dict().keys()) == list(new_net.state_dict().keys())
    new_net.load_state_dict(old_net.state_dict(), strict=True)
    old_net, new_net = old_net.to(DEV), new_net.to(DEV)
    if bf16:
        old_net.enable_bf16_recurrent_path()
        new_net.enable_bf16_recurrent_path()
    return old_net, new_net


@pytest.fixture(scope="module")
def nets(ref, old_core):
    # bfloat16 weights as both training actors run them; float32 where there is no GPU.
    return _net_pair(ref, old_core, bf16=GPU)


# --------------------------------------------------------------------------- the constants


def test_block_size_is_the_one_create_block_mask_uses():
    import inspect
    assert inspect.signature(create_block_mask).parameters["BLOCK_SIZE"].default == new_core.MASK_BLOCK_SIZE == 128
    assert new_core._FAST_BLOCK_MASKS and new_core._TURNS_PER_BLOCK == 2
    assert new_core.TOKENS_PER_TURN == 64


def test_the_working_tree_still_falls_back_to_torch_for_a_long_history(old_core):
    """More than 64 history fields no longer fit one block: the dense builder must take over."""
    src = open(os.path.join(REPO, "ppo_core.py"), encoding="utf-8").read()
    assert "if _FAST_BLOCK_MASKS and S <= MASK_BLOCK_SIZE:" in src and src.count("create_block_mask(mask_mod") == 2


# --------------------------------------------------------------------------- training mask, field by field


def _layout(name: str, n_seq: int, T: int, rng: np.random.Generator) -> np.ndarray:
    n = n_seq * T
    if name == "one_episode":                     # a history far longer than the window
        return np.zeros(n, dtype=np.int64)
    if name == "every_turn_its_own":
        return np.arange(n, dtype=np.int64)
    if name == "short_episodes":                  # several episodes of 1 to 3 turns share a sequence
        out, e = [], 0
        while len(out) < n:
            out += [e] * int(rng.integers(1, 4))
            e += 1
        return np.asarray(out[:n], dtype=np.int64)
    if name == "mixed_lengths":                   # 1 to 150 turns: shorter and longer than 64
        out, e = [], 0
        while len(out) < n:
            out += [e] * int(rng.choice([1, 2, 5, 17, 40, 63, 64, 65, 66, 100, 150]))
            e += 1
        return np.asarray(out[:n], dtype=np.int64)
    if name == "cut_by_the_boundary":             # one episode runs across every sequence boundary
        out = np.zeros(n, dtype=np.int64)
        for s in range(n_seq):
            lo = s * T + T // 2
            out[lo:lo + T] = s + 1
        return out
    if name == "id_comes_back":                   # never produced by the learner, still the rule's job
        return (np.arange(n) // max(1, T // 5) % 2).astype(np.int64)
    raise KeyError(name)


LAYOUTS = ("one_episode", "every_turn_its_own", "short_episodes", "mixed_lengths", "cut_by_the_boundary", "id_comes_back")
# (sequence length, sequences, history window). 31 and 255 do not fill whole blocks.
SHAPES_SMALL = [(1, 1, 64), (2, 3, 64), (3, 2, 64), (30, 3, 64), (31, 3, 64), (64, 2, 64), (65, 2, 64), (64, 2, 8), (40, 2, 1), (40, 2, 0), (24, 2, 300)]
SHAPES_GPU = [(127, 2, 64), (128, 2, 64), (129, 2, 64), (255, 1, 64), (256, 1, 64), (256, 3, 64), (256, 1, 8), (256, 1, 100)]


@pytest.mark.parametrize("T,n_seq,K", SHAPES_SMALL + (SHAPES_GPU if GPU else []))
def test_training_mask_equals_create_block_mask(old_core, T, n_seq, K):
    rng = np.random.default_rng(T * 1000 + n_seq * 10 + K)
    for name in LAYOUTS:
        ep = torch.from_numpy(_layout(name, n_seq, T, rng)).to(DEV)
        s_tok = T * TOK
        expanded = ep.view(n_seq, T, 1).expand(-1, -1, TOK).reshape(n_seq, s_tok)
        rule = lambda b, h, q, k: old_core.pokemon_batched_episodic_mask(b, h, q, k, expanded, K)   # noqa: E731
        want = create_block_mask(rule, B=n_seq, H=1, Q_LEN=s_tok, KV_LEN=s_tok, device=DEV)
        got = new_core.episodic_block_mask(ep.view(n_seq, T), K, rule)
        assert micro_ref.block_mask_differences(want, got) == [], (name, T, n_seq, K)
        assert got.mask_mod is rule
        del want, got
    if GPU:
        torch.cuda.empty_cache()


def test_the_comparison_notices_a_wrong_mask(old_core):
    """A window two turns too wide, or a block of two turns given to another episode, must show up."""
    T, K = 64, 8
    ep = torch.zeros(1, T, dtype=torch.long, device=DEV)
    expanded = ep.view(1, T, 1).expand(-1, -1, TOK).reshape(1, T * TOK)
    rule = lambda b, h, q, k: old_core.pokemon_batched_episodic_mask(b, h, q, k, expanded, K)       # noqa: E731
    want = create_block_mask(rule, B=1, H=1, Q_LEN=T * TOK, KV_LEN=T * TOK, device=DEV)
    assert micro_ref.block_mask_differences(want, new_core.episodic_block_mask(ep, K, rule)) == []
    assert micro_ref.block_mask_differences(want, new_core.episodic_block_mask(ep, K + 2, rule)) != []
    other = ep.clone()
    other[0, 10:12] = 5
    assert micro_ref.block_mask_differences(want, new_core.episodic_block_mask(other, K, rule)) != []


@pytest.mark.parametrize("B,K", [(1, 64), (7, 64), (32, 64), (5, 4), (5, 1), (3, 0)])
def test_inference_mask_equals_create_block_mask(old_core, B, K):
    for lens in ([0] * B, [K] * B, [(i * 7) % (K + 1) for i in range(B)]):
        hist_len = torch.tensor(lens, dtype=torch.long, device=DEV)
        S = TOK + K
        rule = lambda b, h, q, k: old_core.pokemon_inference_mask(b, h, q, k, hist_len, K)          # noqa: E731
        want = create_block_mask(rule, B=B, H=1, Q_LEN=S, KV_LEN=S, device=DEV)
        got = new_core.inference_block_mask(B, S, rule, DEV)
        assert micro_ref.block_mask_differences(want, got) == [], (B, K, lens)
        assert got.mask_mod is rule


# --------------------------------------------------------------------------- the model's outputs


def _real_minibatches(obs_np, ep_np, mb: int, count: int, seed: int):
    """Minibatches cut the way the learner cuts them: whole episodes in shuffled order, every mb steps."""
    n = (obs_np.shape[0] // mb) * mb
    ds = new_core.AsyncEpisodeDataset(act_dim=14, device="cpu")
    z = torch.zeros(n)
    ds.add_steps(torch.from_numpy(obs_np[:n]), torch.zeros(n, dtype=torch.long), z, z, z, z, z, torch.from_numpy(ep_np[:n]))
    torch.manual_seed(seed)
    out = []
    for batch in ds.iter_minibatches(mb, shuffle_episodes=True):
        out.append((batch[0], batch[7]))
        if len(out) >= count:
            break
    return out


TRAIN_SHAPES = [(64, 32), (93, 31), (128, 64), (120, 30)] + ([(256, 256), (512, 256), (256, 64), (384, 128)] if GPU else [])


@pytest.mark.parametrize("mb,T", TRAIN_SHAPES)
def test_training_forward_is_bit_identical(old_core, nets, data, mb, T):
    obs_np, ep_np, kind = data
    old_net, new_net = nets
    old_net.train(), new_net.train()
    seen_cut, seen_shared, seen_long = False, False, False
    calls = []
    real_builder = new_core.episodic_block_mask

    def spy(turn_ids, K, rule):
        got = real_builder(turn_ids, K, rule)
        n_seq, t = turn_ids.shape
        want = create_block_mask(rule, B=n_seq, H=1, Q_LEN=t * TOK, KV_LEN=t * TOK, device=turn_ids.device)
        calls.append(micro_ref.block_mask_differences(want, got))
        return got

    for i, (obs, ep) in enumerate(_real_minibatches(obs_np, ep_np, mb, count=3, seed=mb + T)):
        obs, ep = obs.to(DEV), ep.to(DEV)
        seq = ep.view(mb // T, T)
        seen_shared |= any(int(torch.unique(row).numel()) > 1 for row in seq)
        seen_cut |= bool((seq[:-1, -1] == seq[1:, 0]).any()) if seq.shape[0] > 1 else False
        seen_long |= any(int(torch.bincount(row - row.min()).max()) > 64 for row in seq)
        with torch.no_grad():
            a = old_net(obs, ep, batch_seq_len=T)
            new_core.episodic_block_mask = spy if i == 0 else real_builder
            try:
                b = new_net(obs, ep, batch_seq_len=T)
            finally:
                new_core.episodic_block_mask = real_builder
        for name, x, y in zip(("policy logits", "value logits", "expected value"), a[:3], b[:3]):
            assert x.dtype == y.dtype and torch.equal(x, y), (name, mb, T, i, float((x - y).abs().max()))
        assert a[3] is None and b[3] is None
    assert calls and all(c == [] for c in calls), calls
    assert seen_shared, "no minibatch had several episodes in one sequence"
    if kind == "real" and mb // T > 1:
        assert seen_cut, "no episode was cut by a sequence boundary"
    if T > 64:
        assert seen_long, "no episode ran past the 64-turn window inside a sequence"
    if GPU:
        torch.cuda.empty_cache()


@pytest.mark.skipif(not GPU, reason="flex_attention has no backward on a CPU")
def test_training_gradients_are_bit_identical(old_core, nets, data):
    obs_np, ep_np, _kind = data
    old_net, new_net = nets
    old_net.train(), new_net.train()
    mb, T = 256, 256
    (obs, ep), = _real_minibatches(obs_np, ep_np, mb, count=1, seed=5)
    obs, ep = obs.to(DEV), ep.to(DEV)

    def grads(net):
        net.zero_grad(set_to_none=True)
        logits, v_logits, _, _, _ = net(obs, ep, batch_seq_len=T)
        (logits.square().mean() + v_logits.square().mean()).backward()
        return {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}

    a1, a2, b = grads(old_net), grads(old_net), grads(new_net)
    assert a1.keys() == b.keys() and len(a1) > 80
    repeatable = all(torch.equal(a1[k], a2[k]) for k in a1)
    assert repeatable, "the old code does not repeat its own gradients on this machine, so there is nothing to hold the new code to"
    worst = [k for k in a1 if not torch.equal(a1[k], b[k])]
    assert worst == []
    old_net.zero_grad(set_to_none=True), new_net.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()


@pytest.mark.parametrize("K_over", [None, 4])
def test_inference_forward_is_bit_identical_over_a_long_battle(old_core, nets, data, K_over):
    """
    Batches of battles stepped turn by turn through both models, each fed its own cache back.
    With the default window the battles run to 80 turns, so the history fills and rolls over;
    one battle restarts in the middle with an empty history. K_over shortens the window.
    """
    obs_np, _ep_np, _kind = data
    old_net, new_net = nets
    old_net.eval(), new_net.eval()
    k_default = new_net.kv_cache_len
    old_net.kv_cache_len = new_net.kv_cache_len = K_over or k_default
    try:
        B = 6
        turns = 80 if GPU else 12
        rng = np.random.default_rng(3)
        kv_a = kv_b = len_a = len_b = None
        for t in range(turns):
            rows = rng.integers(0, obs_np.shape[0], B)
            obs = torch.from_numpy(np.ascontiguousarray(obs_np[rows], dtype=np.float32)).to(DEV)
            if t == turns // 2:                 # battle 2 ends and a new one takes its row
                for kv, ln in ((kv_a, len_a), (kv_b, len_b)):
                    for layer in kv:
                        layer[2] = 0
                    ln[2] = 0
            with torch.no_grad():
                a = old_net(obs, kv_cache=kv_a, kv_cache_len=len_a)
                b = new_net(obs, kv_cache=kv_b, kv_cache_len=len_b)
            for name, x, y in zip(("policy logits", "value logits", "expected value"), a[:3], b[:3]):
                assert torch.equal(x, y), (name, t)
            assert len(a[3]) == len(b[3]) == new_net.n_layers
            for i, (x, y) in enumerate(zip(a[3], b[3])):
                assert x.dtype == y.dtype and torch.equal(x, y), ("history of layer", i, t)
            assert torch.equal(a[4], b[4])
            kv_a, len_a = [x.clone() for x in a[3]], a[4].clone()
            kv_b, len_b = [x.clone() for x in b[3]], b[4].clone()
        assert int(len_a.max()) == min(turns, new_net.kv_cache_len)
        assert int(len_a[2]) < int(len_a[0]) or new_net.kv_cache_len < turns // 2
    finally:
        old_net.kv_cache_len = new_net.kv_cache_len = k_default


def test_inference_forward_first_turn_and_batch_of_one(old_core, nets, data):
    obs_np, _ep, _kind = data
    old_net, new_net = nets
    old_net.eval(), new_net.eval()
    for B in (1, 2, 33):
        obs = torch.from_numpy(np.ascontiguousarray(obs_np[:B], dtype=np.float32)).to(DEV)
        with torch.no_grad():
            a, b = old_net(obs), new_net(obs)
        assert all(torch.equal(x, y) for x, y in zip(a[:3], b[:3]))
        assert all(torch.equal(x, y) for x, y in zip(a[3], b[3])) and torch.equal(a[4], b[4])
