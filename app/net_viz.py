"""
net_viz: what the transformer looks at, and which of its neurons fire, when it picks a move.

For the window app (app/PSBotLookahead.cs, ladder.py --app-net-view). It hooks the network's
layers during the forward pass that picks the move (players.BatchedPolicyPlayer._forward_batch),
so there is no second pass, and turns what they saw into one small summary per battle:

  att   one row per transformer layer and one for the readout: the share of the decision
        token's attention (ppo_core.ACTOR_IDX, 8 heads averaged) on each of 14 columns:
        the field, the 12 Pokemon (each with its 4 move tokens), and the memory of earlier
        turns (the readout sees the current turn only: 0 there).
  neu   the same rows: the feed-forward neurons of the decision token after the GELU
        (1024 a row), negative values as 0, as bytes 0..255 of the row's largest value
        (neu_max), base64.

The attention is computed again from the layer's input, because flex_attention does not return
its weights: the same projections, the 1/sqrt(head size) scale, the soft cap of
ppo_core.logit_soft_cap and the history mask of ppo_core.pokemon_inference_mask.
tests/test_net_viz.py checks that these weights rebuild the layer's own attention output.

Nothing here may cost a move: a failure is told once and the view switches itself off, and the
network's outputs are the same with or without it.
"""

from __future__ import annotations

import base64
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ppo_core import ACTOR_IDX, FIELD_IDX, TOKENS_PER_TURN, FlexEncoderLayer

N_POKEMON = 12
MOVE0 = FIELD_IDX + 1 + N_POKEMON   # move tokens: MOVE0 + 4 * Pokemon slot + move slot
MOVES_EACH = 4
SOFT_CAP = 50.0                     # ppo_core.logit_soft_cap
N_COLUMNS = 1 + N_POKEMON + 1       # field, 12 Pokemon, memory
WARM_UP = "warm-up-"                # search_player's warm-up battles: never shown


def _column_index(n_keys: int) -> torch.Tensor:
    """Key position -> column (0 field, 1..12 Pokemon, 13 memory, -1 left out: register, actor, critic)."""
    col = torch.full((n_keys,), -1, dtype=torch.long)
    col[FIELD_IDX] = 0
    for p in range(N_POKEMON):
        col[FIELD_IDX + 1 + p] = 1 + p
        col[MOVE0 + MOVES_EACH * p: MOVE0 + MOVES_EACH * (p + 1)] = 1 + p
    if n_keys > TOKENS_PER_TURN:
        col[TOKENS_PER_TURN:] = N_COLUMNS - 1
    return col


def _to_columns(p: torch.Tensor) -> torch.Tensor:
    """[B, keys] attention shares -> [B, 14] shares per column."""
    col = _column_index(p.shape[1]).to(p.device)
    keep = col >= 0
    out = torch.zeros(p.shape[0], N_COLUMNS, dtype=p.dtype, device=p.device)
    out.index_add_(1, col[keep], p[:, keep])
    return out


def encoder_attention_heads(layer: FlexEncoderLayer, x: torch.Tensor, hist_len: torch.Tensor, n_prev: int) -> torch.Tensor:
    """
    The decision token's attention in one encoder layer, per head: [B, H, keys].
    x is the layer's input in inference ([B, TOKENS_PER_TURN + n_prev, D]); hist_len the history
    length each row had before this pass (the kv_cache_len given to the network).
    """
    B, S, D = x.shape
    H = layer.n_heads
    qkv = layer.qkv_proj(layer.norm1(x)).reshape(B, S, 3, H, D // H)
    q = qkv[:, ACTOR_IDX, 0].float()                       # [B, H, dh]
    k = qkv[:, :, 1].float()                               # [B, S, H, dh]
    scores = torch.einsum("bhd,bshd->bhs", q, k) / math.sqrt(D // H)
    scores = torch.tanh(scores / SOFT_CAP) * SOFT_CAP
    pos = torch.arange(S, device=x.device).unsqueeze(0)    # [1, S]
    slot = pos - TOKENS_PER_TURN
    hist = hist_len.to(x.device).reshape(-1, 1).long()
    valid = (pos < TOKENS_PER_TURN) | ((slot >= 0) & (slot < n_prev) & (slot >= n_prev - hist))
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    return torch.softmax(scores, dim=-1)                   # [B, H, S]


def readout_attention_heads(readout: Any, q_x: torch.Tensor, kv_x: torch.Tensor) -> torch.Tensor:
    """The actor query's attention in the readout (ppo_core.FlexReadout), per head: [B, H, keys]."""
    B, K, D = kv_x.shape
    H = readout.n_heads
    q = readout.q_proj(readout.norm_q(q_x[:, 0])).view(B, H, D // H).float()
    k = readout.k_proj(readout.norm_kv(kv_x)).view(B, K, H, D // H).float()
    scores = torch.einsum("bhd,bkhd->bhk", q, k) / math.sqrt(D // H)
    return torch.softmax(scores, dim=-1)


class _Capture:
    """Forward hooks that keep what the summary needs, removed by close()."""

    def __init__(self, net: Any):
        self.layer_in: Dict[int, torch.Tensor] = {}
        self.layer_ff: Dict[int, torch.Tensor] = {}
        self.readout_in: Optional[tuple] = None
        self.readout_ff: Optional[torch.Tensor] = None
        self.handles: List[Any] = []
        try:
            for i, layer in enumerate(net.transformer):
                self.handles.append(layer.register_forward_hook(self._keep_layer_in(i)))
                self.handles.append(layer.ff[1].register_forward_hook(self._keep_layer_ff(i)))
            self.handles.append(net.readout.register_forward_hook(self._keep_readout_in))
            self.handles.append(net.readout.ff[1].register_forward_hook(self._keep_readout_ff))
        except BaseException:
            self.close()
            raise

    def _keep_layer_in(self, i: int) -> Callable:
        def hook(_m: Any, inputs: tuple, _out: Any) -> None:
            self.layer_in[i] = inputs[0]
        return hook

    def _keep_layer_ff(self, i: int) -> Callable:
        def hook(_m: Any, _inputs: tuple, out: torch.Tensor) -> None:
            self.layer_ff[i] = out[:, ACTOR_IDX]
        return hook

    def _keep_readout_in(self, _m: Any, inputs: tuple, _out: Any) -> None:
        self.readout_in = (inputs[0], inputs[1])

    def _keep_readout_ff(self, _m: Any, _inputs: tuple, out: torch.Tensor) -> None:
        self.readout_ff = out[:, 0]

    def close(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []


def _neuron_bytes(rows: List[torch.Tensor]) -> tuple:
    """[B, d_ff] per row -> (base64 of [rows, d_ff] uint8 per battle, [rows] largest value per battle)."""
    act = torch.stack([r.float().clamp_min(0.0) for r in rows], dim=1)    # [B, rows, d_ff]
    top = act.amax(dim=2)                                                  # [B, rows]
    q = (act * (255.0 / top.clamp_min(1e-6)).unsqueeze(-1)).round().clamp(0, 255).to(torch.uint8)
    q, top = q.cpu().numpy(), top.cpu().numpy()
    return [base64.b64encode(q[b].tobytes()).decode("ascii") for b in range(q.shape[0])], top


TOP_NEURONS = 10    # per row in the diagram: the strongest firing neurons of the decision token
TOP_ACTIONS = 4     # the network's likeliest legal actions, at the end of the diagram


_PARAMS: Dict[int, List[int]] = {}


def param_blocks(net: Any) -> List[int]:
    """
    Parameters per block of the picture, in play order: the input encoder (embeddings, the Pokemon,
    move, ability and field subnets, the decision tokens), each transformer layer, the readout and
    the policy head. The value head and the JEPA predictor (training only) are not in it.
    """
    key = id(net)
    if key not in _PARAMS:
        def count(m: Any) -> int:
            return sum(p.numel() for p in m.parameters())
        layers = [count(l) for l in net.transformer]
        skip = count(net.readout) + count(net.pi_head) + count(net.v_head) + sum(layers)
        skip += count(net.jepa_predictor) + count(net.action_emb) if hasattr(net, "jepa_predictor") else 0
        total = sum(p.numel() for p in net.parameters())
        _PARAMS[key] = [total - skip] + layers + [count(net.readout), count(net.pi_head)]
    return _PARAMS[key]


def _norm(m: torch.Tensor) -> List[List[float]]:
    """A weight matrix scaled to -1..1 by its largest magnitude, rounded for the JSON line."""
    top = float(m.abs().max())
    m = m / top if top > 0 else m
    return [[round(float(v), 2) for v in row] for row in m.cpu()]


def diagram(net: Any, ff_rows: List[torch.Tensor], logits: torch.Tensor, mask: torch.Tensor) -> List[Dict[str, Any]]:
    """
    The 3Blue1Brown-style picture, per battle: in each row (layers and readout) the TOP_NEURONS
    neurons that fire hardest ("top", by index); the weights between them from one row to the next
    ("edges"[r][j][i]: from neuron i of row r to neuron j of row r + 1); the network's probabilities
    over the 14 actions, legal ones only ("probs"); its TOP_ACTIONS likeliest ("acts"); and the
    weights from the readout's top neurons to those actions ("out_edges"[a][i]).
    A weight here is the direct path: the neuron's output weights into the residual stream (ff[2]),
    then the next row's input weights (ff[0]) with its LayerNorm gain, or the policy head. It
    leaves out what attention and the normalisation's scaling do on the way, so it says how
    neuron i pushes neuron j directly, not everything that connects them.
    """
    layers = list(net.transformer)
    w_out = [l.ff[2].weight.float() for l in layers] + [net.readout.ff[2].weight.float()]                  # [D, d_ff]
    w_in = [l.ff[0].weight.float() * l.norm2.weight.float().unsqueeze(0) for l in layers] + \
           [net.readout.ff[0].weight.float() * net.readout.norm_ff.weight.float().unsqueeze(0)]            # [d_ff, D]
    w_pi = net.pi_head.weight.float()                                                                       # [A, D]
    out = []
    for b in range(logits.shape[0]):
        top = []
        for r in ff_rows:
            act = r[b].float().clamp_min(0.0)
            k = min(TOP_NEURONS, act.shape[0])
            top.append(torch.sort(torch.topk(act, k).indices).values)
        edges = [_norm(w_in[r + 1][top[r + 1]] @ w_out[r][:, top[r]]) for r in range(len(top) - 1)]
        legal = mask[b] > 0.5
        z = logits[b].float().masked_fill(~legal, float("-inf"))
        probs = torch.softmax(z, dim=-1) if bool(legal.any()) else torch.zeros_like(z)
        probs = torch.nan_to_num(probs, nan=0.0)
        n_act = min(TOP_ACTIONS, int(legal.sum()))
        acts = torch.topk(probs, n_act).indices if n_act > 0 else torch.zeros(0, dtype=torch.long, device=probs.device)
        out.append({
            "top": [[int(i) for i in t] for t in top],
            "edges": edges,
            "probs": [round(float(p), 4) for p in probs],
            "acts": [int(a) for a in acts],
            "out_edges": _norm(w_pi[acts] @ w_out[-1][:, top[-1]]) if n_act > 0 else [],
        })
    return out


def summarize(net: Any, cap: _Capture, hist_len: torch.Tensor, logits: Optional[torch.Tensor] = None,
              x: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
    """One summary per row of the pass (see the module's text); with logits and x also the diagram."""
    n_layers = len(net.transformer)
    att_rows, ff_rows = [], []
    for i, layer in enumerate(net.transformer):
        x_in = cap.layer_in[i]
        heads = encoder_attention_heads(layer, x_in, hist_len, x_in.shape[1] - TOKENS_PER_TURN)
        att_rows.append(_to_columns(heads.mean(dim=1)))
        ff_rows.append(cap.layer_ff[i])
    q_x, kv_x = cap.readout_in
    att_rows.append(_to_columns(readout_attention_heads(net.readout, q_x, kv_x).mean(dim=1)))
    ff_rows.append(cap.readout_ff)
    att = torch.stack(att_rows, dim=1).cpu().numpy()                       # [B, rows, 14]
    neu, top = _neuron_bytes(ff_rows)
    pics: List[Dict[str, Any]] = [{} for _ in range(att.shape[0])]
    if logits is not None and x is not None:
        s, e = net.unpacker.offsets["action_mask"]
        pics = diagram(net, ff_rows, logits, x[:, s:e])
        blocks = param_blocks(net)
        for p in pics:
            p["params"] = blocks
    out = []
    for b in range(att.shape[0]):
        row = {
            "layers": n_layers,
            "att": [[round(float(v), 4) for v in row] for row in att[b]],
            "neu": neu[b],
            "neu_max": [round(float(v), 4) for v in top[b]],
        }
        row.update(pics[b])
        out.append(row)
    return out


class NetViz:
    """
    Attach as player.net_viz; players.BatchedPolicyPlayer._forward_batch then runs its pass
    through forward() and sink(tag, summary) is called for every battle in it but the warm-up.
    """

    def __init__(self, sink: Callable[[str, Dict[str, Any]], None], say: Callable[[str], None] = print):
        self.sink = sink
        self.say = say
        self.off: Optional[str] = None      # why the view switched itself off

    def wanted(self, tags: Sequence[str]) -> bool:
        return self.off is None and any(not str(t).startswith(WARM_UP) for t in tags)

    def forward(self, net: Any, tags: Sequence[str], x: torch.Tensor, kv_cache: Any, kv_cache_len: torch.Tensor) -> Any:
        """net(x, kv_cache=..., kv_cache_len=...), and the summaries handed to the sink."""
        cap = None
        try:
            cap = _Capture(net)
        except Exception as exc:
            self._fail(exc)
        try:
            result = net(x, kv_cache=kv_cache, kv_cache_len=kv_cache_len)
        finally:
            if cap is not None:
                cap.close()
        if cap is not None:
            try:
                with torch.no_grad():
                    rows = summarize(net, cap, kv_cache_len, logits=result[0], x=x)
                for tag, row in zip(tags, rows):
                    if not str(tag).startswith(WARM_UP):
                        self.sink(tag, row)
            except Exception as exc:
                self._fail(exc)
        return result

    def _fail(self, exc: BaseException) -> None:
        if self.off is None:
            self.off = f"{type(exc).__name__}: {exc}"
            try:
                self.say(f"note: the network view is off for this run ({self.off}); the games are not affected")
            except Exception:
                pass
