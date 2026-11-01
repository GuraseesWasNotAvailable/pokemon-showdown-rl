"""
net_viz: the window app's view of the network (what it looks at, which neurons fire).

    python -m pytest -q tests/test_net_viz.py

The attention weights are computed again outside flex_attention, so the main check is that they
rebuild each layer's own attention output for the decision token, with histories of every length.
The rest: the network's outputs are the same with the view on, nothing is kept hooked, the
warm-up is never shown, and a failure switches the view off without costing the move.
Needs a CUDA GPU (flex_attention); run it on the box.
"""

from __future__ import annotations

import base64
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)          # vocab.json is opened by a relative path

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("flex_attention needs a CUDA GPU", allow_module_level=True)

import config  # noqa: E402
import net_viz  # noqa: E402
from ppo_core import ACTOR_IDX, TOKENS_PER_TURN  # noqa: E402

DEV = "cuda"


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(3)
    n = config.RunConfig.default().make_model().float().to(DEV).eval()
    # Untrained weights give near-uniform attention, which would hide a wrong mask or scale:
    # sharpen the queries and keys so the weights differ a lot between keys.
    with torch.no_grad():
        for layer in n.transformer:
            d = layer.qkv_proj.weight.shape[1]
            layer.qkv_proj.weight[:2 * d].mul_(6.0)
        n.readout.q_proj.weight.mul_(4.0)
        n.readout.k_proj.weight.mul_(4.0)
    return n


def random_pass(net, hist_lens):
    """condition_tokens + readout_policy on random tokens and random histories of the given lengths."""
    B, D, K = len(hist_lens), net.d_model, net.kv_cache_len
    g = torch.Generator(device="cpu").manual_seed(7)
    field = torch.randn(B, 1, D, generator=g).to(DEV)
    pok = torch.randn(B, 12, D, generator=g).to(DEV)
    mov = torch.randn(B, 48, D, generator=g).to(DEV)
    kv = [torch.randn(B, K, D, generator=g).to(DEV) for _ in range(net.n_layers)]
    lens = torch.tensor(hist_lens, dtype=torch.long, device=DEV)
    Z, _, _ = net.condition_tokens(field, pok, mov, kv_cache=kv, kv_cache_len=lens)
    net.readout_policy(Z)
    return lens


def test_encoder_weights_rebuild_each_layers_own_attention_output(net):
    seen = {}
    hooks = [layer.out_proj.register_forward_hook(lambda m, i, o, n=n: seen.__setitem__(n, i[0]))
             for n, layer in enumerate(net.transformer)]
    cap = net_viz._Capture(net)
    try:
        with torch.no_grad():
            lens = random_pass(net, [0, 1, 5, 63, 64])
            for n, layer in enumerate(net.transformer):
                x = cap.layer_in[n]
                B, S, D = x.shape
                H = layer.n_heads
                p = net_viz.encoder_attention_heads(layer, x, lens, S - TOKENS_PER_TURN)        # [B, H, S]
                v = layer.qkv_proj(layer.norm1(x)).reshape(B, S, 3, H, D // H)[:, :, 2].float()
                rebuilt = torch.einsum("bhs,bshd->bhd", p, v).reshape(B, D)
                actual = seen[n][:, ACTOR_IDX].float()
                assert torch.allclose(rebuilt, actual, atol=2e-4, rtol=1e-3), \
                    f"layer {n}: largest difference {(rebuilt - actual).abs().max().item():.2e}"
                # No weight on a history slot the row does not have.
                for b, h in enumerate(lens.tolist()):
                    if S - TOKENS_PER_TURN - h > 0:
                        empty = p[b, :, TOKENS_PER_TURN:TOKENS_PER_TURN + (S - TOKENS_PER_TURN - h)]
                        assert float(empty.abs().max()) == 0.0
                assert float(p.std()) > 1e-3, "the sharpened weights must not be uniform, or this check proves nothing"
    finally:
        cap.close()
        for h in hooks:
            h.remove()


def test_readout_weights_rebuild_the_readouts_own_attention_output(net):
    seen = {}
    hook = net.readout.out_proj.register_forward_hook(lambda m, i, o: seen.__setitem__("x", i[0]))
    cap = net_viz._Capture(net)
    try:
        with torch.no_grad():
            random_pass(net, [0, 9, 64])
            q_x, kv_x = cap.readout_in
            B, K, D = kv_x.shape
            H = net.readout.n_heads
            p = net_viz.readout_attention_heads(net.readout, q_x, kv_x)
            v = net.readout.v_proj(net.readout.norm_kv(kv_x)).view(B, K, H, D // H).float()
            rebuilt = torch.einsum("bhk,bkhd->bhd", p, v).reshape(B, D)
            actual = seen["x"][:, 0].float()
            assert torch.allclose(rebuilt, actual, atol=2e-4, rtol=1e-3), \
                f"largest difference {(rebuilt - actual).abs().max().item():.2e}"
    finally:
        cap.close()
        hook.remove()


def test_columns_put_each_pokemons_moves_with_it_and_the_history_in_memory():
    col = net_viz._column_index(TOKENS_PER_TURN + 64)
    assert col[0] == 0
    for p in range(12):
        assert col[1 + p] == 1 + p
        assert all(int(col[13 + 4 * p + m]) == 1 + p for m in range(4))
    assert [int(c) for c in col[61:64]] == [-1, -1, -1], "register, actor and critic are left out"
    assert set(int(c) for c in col[64:]) == {13}
    assert int(net_viz._column_index(TOKENS_PER_TURN).max()) == 12, "the readout has no memory column"


def zero_batch(net, tags):
    """The network's real input path, as the warm-up feeds it: zero observations, empty histories."""
    dim = max(end for _start, end in net.unpacker.offsets.values())
    x = torch.zeros(len(tags), dim, device=DEV)
    kv = [torch.zeros(len(tags), net.kv_cache_len, net.d_model, device=DEV) for _ in range(net.n_layers)]
    lens = torch.zeros(len(tags), dtype=torch.long, device=DEV)
    return x, kv, lens


def test_the_view_changes_nothing_the_network_returns_and_leaves_no_hook(net):
    tags = ["battle-a", "warm-up-0", "battle-b"]
    got = []
    viz = net_viz.NetViz(lambda tag, s: got.append((tag, s)))
    x, kv, lens = zero_batch(net, tags)
    with torch.no_grad():
        plain = net(x, kv_cache=kv, kv_cache_len=lens)
        seen = viz.forward(net, tags, x, kv, lens)
    for a, b in zip(plain[:3], seen[:3]):
        assert torch.equal(a, b)
    for a, b in zip(plain[3], seen[3]):
        assert torch.equal(a, b)
    assert torch.equal(plain[4], seen[4])
    for m in net.modules():
        assert not m._forward_hooks, f"a hook was left on {type(m).__name__}"
    assert [t for t, _s in got] == ["battle-a", "battle-b"], "the warm-up is never shown"
    rows = net.n_layers + 1
    for _t, s in got:
        assert s["layers"] == net.n_layers and len(s["att"]) == rows and all(len(r) == 14 for r in s["att"])
        for r in s["att"]:
            assert all(v >= 0 for v in r) and 0.0 < sum(r) <= 1.0001
        assert s["att"][-1][13] == 0.0, "the readout sees the current turn only"
        neu = base64.b64decode(s["neu"])
        d_ff = net.transformer[0].ff[0].out_features
        assert len(neu) == rows * d_ff and len(s["neu_max"]) == rows
        assert max(neu) == 255 or max(s["neu_max"]) == 0.0
    assert viz.off is None


def test_a_failing_view_switches_itself_off_and_the_move_is_still_made(net):
    said = []

    def bad_sink(tag, s):
        raise RuntimeError("the app went away")

    viz = net_viz.NetViz(bad_sink, say=said.append)
    tags = ["battle-a"]
    x, kv, lens = zero_batch(net, tags)
    with torch.no_grad():
        plain = net(x, kv_cache=kv, kv_cache_len=lens)
        out = viz.forward(net, tags, x, kv, lens)
        assert torch.equal(plain[0], out[0])
        assert viz.off and len(said) == 1
        assert not viz.wanted(tags), "once off it stays off"
        viz.forward(net, tags, x, kv, lens)
    assert len(said) == 1, "told once"


def test_the_diagram_has_the_strongest_neurons_their_direct_weights_and_the_legal_moves(net):
    x, kv, lens = zero_batch(net, ["battle-a"])
    s, e = net.unpacker.offsets["action_mask"]
    x[0, s:e] = 0.0
    x[0, s + 0] = x[0, s + 2] = x[0, s + 5] = 1.0          # three legal actions
    got = []
    viz = net_viz.NetViz(lambda tag, summary: got.append(summary))
    cap = net_viz._Capture(net)
    try:
        with torch.no_grad():
            logits = net(x, kv_cache=kv, kv_cache_len=lens)[0]
            ff = [cap.layer_ff[i] for i in range(net.n_layers)] + [cap.readout_ff]
    finally:
        cap.close()
    with torch.no_grad():
        viz.forward(net, ["battle-a"], x, kv, lens)
    d = got[0]
    rows = net.n_layers + 1
    assert len(d["top"]) == rows and all(len(t) == net_viz.TOP_NEURONS for t in d["top"])
    for r in range(rows):
        act = ff[r][0].clamp_min(0)
        want = set(torch.topk(act, net_viz.TOP_NEURONS).indices.tolist())
        assert set(d["top"][r]) == want and d["top"][r] == sorted(d["top"][r])
    # The direct-path weight from neuron i of layer 1 to neuron j of layer 2.
    i, j = d["top"][0][0], d["top"][1][0]
    l1, l2 = net.transformer[0], net.transformer[1]
    w = float((l2.ff[0].weight[j] * l2.norm2.weight) @ l1.ff[2].weight[:, i])
    m = (l2.ff[0].weight[d["top"][1]] * l2.norm2.weight) @ l1.ff[2].weight[:, d["top"][0]]
    assert abs(d["edges"][0][0][0] - w / float(m.abs().max())) < 0.01
    assert len(d["edges"]) == rows - 1 and all(len(E) == net_viz.TOP_NEURONS for E in d["edges"])
    p = d["probs"]
    assert abs(sum(p) - 1.0) < 1e-3 and all(p[k] == 0.0 for k in range(14) if k not in (0, 2, 5))
    assert sorted(d["acts"]) == [0, 2, 5] and p[d["acts"][0]] == max(p)
    assert len(d["out_edges"]) == 3 and all(len(r) == net_viz.TOP_NEURONS for r in d["out_edges"])
    assert torch.allclose(logits, net(x, kv_cache=kv, kv_cache_len=lens)[0])


def test_the_blocks_parameters_add_up_to_the_network_without_the_training_only_parts(net):
    blocks = net_viz.param_blocks(net)
    assert len(blocks) == net.n_layers + 3
    total = sum(p.numel() for p in net.parameters())
    unused = sum(p.numel() for m in (net.v_head, net.jepa_predictor, net.action_emb) for p in m.parameters())
    assert sum(blocks) == total - unused
    assert blocks[0] > blocks[1] > blocks[-1], "the encoder is the biggest block, the policy head the smallest"
