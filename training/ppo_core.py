"""
Core Neural Network and Proximal Policy Optimization (PPO) Utilities.

This module provides the PokeTransformer architecture, customized for the complex 
structured observation spaces of Pokemon Showdown, along with high-performance 
sampling, GAE calculation, and PPO update logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Tuple, Dict, Optional, Iterator, Final, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
# Setup logger for core model events
logger = logging.getLogger(__name__)

# ----------------------------
# Numerical & Distributional Utils
# ----------------------------

TOKENS_PER_TURN = 64
FIELD_IDX = 0
ACTOR_IDX = 62
CRITIC_IDX = 63

def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a boolean mask to logits, setting invalid actions to a large negative value.
    
    Includes a safety valve to prevent NaN outputs if an entirely zero mask is provided
    by forcing at least one valid index.
    """
    mask_sum = mask.sum(dim=-1)
    if (mask_sum == 0).any():
        bad_indices = (mask_sum == 0).nonzero(as_tuple=True)[0]
        mask = mask.clone()
        mask[bad_indices, 0] = 1.0
        logger.warning(f"Zero mask detected at batch indices {bad_indices.tolist()}. Forced index 0.")

    m = (mask > 0.5).to(torch.bool)
    return logits.masked_fill(~m, -1e4)


def masked_logits_nosync(logits: torch.Tensor, mask: torch.Tensor,
                         zero_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    masked_logits without waiting for the GPU (ppo_update's fast_update path). The same output,
    bit for bit: a row whose mask sums to 0 gets index 0 allowed, as there, but by torch.where
    on every row instead of .any() and nonzero(), which read the GPU.

    zero_rows: a 0-d int64 tensor on the mask's device, or None. The number of rows fixed up is
    added to it on the GPU (the caller reads it once and logs one warning for its update).
    masked_logits stays as it was for inference and every other caller.
    """
    bad = mask.sum(dim=-1) == 0
    if zero_rows is not None:
        zero_rows += bad.sum()
    col0 = torch.arange(mask.shape[-1], device=mask.device) == 0
    mask = torch.where(bad.unsqueeze(-1) & col0, torch.ones((), dtype=mask.dtype, device=mask.device), mask)
    m = (mask > 0.5).to(torch.bool)
    return logits.masked_fill(~m, -1e4)


def twohot_targets(x: torch.Tensor, *, v_min: float, v_max: float, v_bins: int) -> torch.Tensor:
    """
    Encodes scalar values into a two-hot distribution over uniform bins.
    
    Used for distributional value prediction to reduce variance and improve 
    learning stability in reinforcement learning.
    """
    x = x.clamp(v_min, v_max)
    scale = (v_bins - 1) / (v_max - v_min)
    f = (x - v_min) * scale
    i0 = torch.floor(f).long()
    i1 = torch.clamp(i0 + 1, max=v_bins - 1)

    w1 = (f - i0.float())
    w0 = 1.0 - w1

    # Edge case: exactly at the maximum bin
    w0 = torch.where(i0 == i1, torch.ones_like(w0), w0)
    w1 = torch.where(i0 == i1, torch.zeros_like(w1), w1)

    t = torch.zeros((x.shape[0], v_bins), device=x.device, dtype=torch.float32)
    t.scatter_add_(1, i0.view(-1, 1), w0.view(-1, 1))
    t.scatter_add_(1, i1.view(-1, 1), w1.view(-1, 1))
    return t


def dist_value_loss(v_logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """Computes cross-entropy loss between predicted value logits and target distributions."""
    logp = torch.log_softmax(v_logits, dim=-1)
    return -(target_dist * logp).sum(dim=-1).mean()

import math

def get_sinusoidal_encoding(num_positions: int, embedding_dim: int) -> torch.Tensor:
    """
    Generates a sinusoidal positional encoding matrix.
    Args:
        num_positions: The vocabulary size (e.g., 101 for val_100).
        embedding_dim: The dimension of the embedding (e.g., 128).
    Returns:
        A tensor of shape (num_positions, embedding_dim)
    """
    pe = torch.zeros(num_positions, embedding_dim)
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    
    # Calculate the frequency divisor
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
    
    # Apply sine to even indices, cosine to odd indices
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

@torch.no_grad()
def masked_sample(
    logits: torch.Tensor, 
    mask: torch.Tensor, 
    greedy: bool = False, 
    temp: float = 1.0,
    top_p: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples actions using temperature scaling and optional Nucleus (Top-P) filtering.
    
    Args:
        logits: Raw action scores.
        mask: Valid action mask.
        greedy: If True, returns argmax.
        temp: Temperature (>1.0 increases entropy, <1.0 decreases it).
        top_p: Nucleus sampling threshold.
    """
    ml_pure = masked_logits(logits, mask)
    dist_pure = Categorical(logits=ml_pure)
    
    # Scale logits for exploration
    ml_explore = masked_logits(logits / max(temp, 1e-4), mask)
     
    if greedy:
        a = torch.argmax(ml_pure, dim=-1)
        return a, dist_pure.log_prob(a), torch.zeros_like(a, dtype=torch.float32)
    
    if 0.0 < top_p < 1.0:
        probs = torch.softmax(ml_explore, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens outside the nucleus
        sorted_to_remove = cumulative_probs > top_p
        sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
        sorted_to_remove[..., 0] = False
        
        indices_to_remove = sorted_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_to_remove)
        ml_explore = ml_explore.masked_fill(indices_to_remove, float('-inf'))
    
    dist_explore = Categorical(logits=ml_explore)
    a = dist_explore.sample()
    return a, dist_pure.log_prob(a), dist_pure.entropy()


def masked_logprob_entropy(
    logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor,
    nosync: bool = False, zero_rows: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates log probabilities and entropy for a specific set of actions under a mask.

    nosync=True (ppo_update's fast_update path): the mask through masked_logits_nosync, which gives
    the same logits without reading the GPU and counts the rows it fixed up into zero_rows.
    """
    ml = masked_logits_nosync(logits, mask, zero_rows) if nosync else masked_logits(logits, mask)
    dist = Categorical(logits=ml)
    return dist.log_prob(actions), dist.entropy()

def logit_soft_cap(score, b, h, q_idx, kv_idx):
    """
    Applies soft-clipping to attention logits: 
    C * tanh(score / C)
    """
    cap = 50.0
    return torch.tanh(score / cap) * cap

# ----------------------------------------
# THE MASKS
# ----------------------------------------

def pokemon_episodic_mask(b, h, q_idx, kv_idx, episode_ids, n_prev_fields: int):
    same_ep = episode_ids[q_idx] == episode_ids[kv_idx]

    q_t = q_idx // TOKENS_PER_TURN
    kv_t = kv_idx // TOKENS_PER_TURN
    kv_i = kv_idx % TOKENS_PER_TURN

    same_turn = (q_t == kv_t)
    prev_field = (
        (kv_i == FIELD_IDX)
        & (kv_t < q_t)
        & ((q_t - kv_t) <= n_prev_fields)
    )

    return same_ep & (same_turn | prev_field)

def pokemon_batched_episodic_mask(b, h, q_idx, kv_idx, expanded_ids, n_prev_fields: int):
    same_ep = expanded_ids[b, q_idx] == expanded_ids[b, kv_idx]

    q_t = q_idx // TOKENS_PER_TURN
    kv_t = kv_idx // TOKENS_PER_TURN
    kv_i = kv_idx % TOKENS_PER_TURN

    same_turn = (q_t == kv_t)
    prev_field = (
        (kv_i == FIELD_IDX)
        & (kv_t < q_t)
        & ((q_t - kv_t) <= n_prev_fields)
    )

    return same_ep & (same_turn | prev_field)

def pokemon_inference_mask(b, h, q_idx, kv_idx, hist_len: torch.Tensor, n_prev_fields: int):
    q_is_current = q_idx < TOKENS_PER_TURN
    kv_is_current = kv_idx < TOKENS_PER_TURN

    kv_is_hist = (kv_idx >= TOKENS_PER_TURN) & (kv_idx < TOKENS_PER_TURN + n_prev_fields)
    hist_slot = kv_idx - TOKENS_PER_TURN

    # Right-aligned valid history:
    # valid slots are [K - hist_len[b], ..., K-1]
    kv_hist_valid = kv_is_hist & (hist_slot >= (n_prev_fields - hist_len[b]))

    # Current queries can read current tokens + valid history fields.
    current_rule = q_is_current & (kv_is_current | kv_hist_valid)

    # History queries are inert/self-only. This also keeps padded queries safe.
    history_rule = (q_idx >= TOKENS_PER_TURN) & (q_idx == kv_idx)

    return current_rule | history_rule

# ----------------------------------------
# THE BLOCK MASKS
# ----------------------------------------
#
# flex_attention takes the mask twice: as the rule itself (one of the functions above, applied
# to every token pair inside a block) and as a coarse map of which blocks of 128 x 128 token
# pairs hold any allowed pair at all. torch's create_block_mask finds that map by evaluating
# the rule on every token pair: 16,384 x 16,384 booleans per packed sequence, built again for
# every minibatch, and 38% of the learner's minibatch went there. A block is two whole turns,
# so the same map follows from the episode ids of the turns alone. The two builders below
# return a BlockMask whose every field equals the one create_block_mask returns for these
# rules (tests/test_speed_mask.py compares them, and the model's outputs, bit for bit), so
# the attention computes exactly what it did.

# Tokens per mask block. This is torch's own default, written out because the builders are
# only correct when a block is a whole number of turns, and at least two of them: with one
# turn per block a block could be entirely allowed, which flex_attention stores separately.
MASK_BLOCK_SIZE: Final[int] = 128
_TURNS_PER_BLOCK: Final[int] = MASK_BLOCK_SIZE // TOKENS_PER_TURN
_FAST_BLOCK_MASKS: Final[bool] = (MASK_BLOCK_SIZE % TOKENS_PER_TURN == 0) and _TURNS_PER_BLOCK >= 2


def episodic_block_mask(turn_episode_ids: torch.Tensor, n_prev_fields: int, mask_mod) -> BlockMask:
    """
    The BlockMask of pokemon_batched_episodic_mask for packed training sequences.

    turn_episode_ids: [n_seq, T], the episode each turn belongs to. mask_mod is the rule
    itself, still applied per token pair by the attention kernel.

    A query turn reaches a key turn when both are in the same episode and the key turn is the
    query's own or one of the n_prev_fields before it. A block holds an allowed pair exactly
    when one of its query turns reaches one of its key turns. No block is ever entirely
    allowed (a token never reaches a non-field token of another turn), so the "full" lists
    are empty, as they are from create_block_mask.
    """
    n_seq, T = turn_episode_ids.shape
    dev = turn_episode_ids.device
    bt = _TURNS_PER_BLOCK
    nb = -(-T // bt)
    ep = turn_episode_ids
    if nb * bt != T:
        # create_block_mask pads the last block with pairs that are not allowed.
        ep = F.pad(ep, (0, nb * bt - T), value=-1)
    t = torch.arange(nb * bt, device=dev)
    d = t[:, None] - t[None, :]                       # query turn minus key turn
    real = t < T
    reach = (d >= 0) & (d <= n_prev_fields) & real[:, None] & real[None, :]
    turns = (ep[:, :, None] == ep[:, None, :]) & reach[None]
    partial = turns.view(n_seq, nb, bt, nb, bt).permute(0, 1, 3, 2, 4).reshape(n_seq, nb, nb, bt * bt).any(dim=-1)
    partial = partial[:, None].to(torch.int32)        # [n_seq, 1 head, nb, nb]
    # The same ordering create_block_mask applies: per row the count of set blocks and their
    # indices first, in ascending order (a stable sort).
    kv_num = partial.sum(dim=-1).to(torch.int32, memory_format=torch.contiguous_format)
    kv_idx = torch.argsort(partial, dim=-1, descending=True, stable=True).to(torch.int32, memory_format=torch.contiguous_format)
    full = torch.zeros_like(partial)
    full_num = full.sum(dim=-1).to(torch.int32, memory_format=torch.contiguous_format)
    full_idx = torch.argsort(full, dim=-1, descending=True, stable=True).to(torch.int32, memory_format=torch.contiguous_format)
    s_tok = T * TOKENS_PER_TURN
    return BlockMask.from_kv_blocks(
        kv_num, kv_idx, full_num, full_idx,
        BLOCK_SIZE=MASK_BLOCK_SIZE, mask_mod=mask_mod, seq_lengths=(s_tok, s_tok),
    )


def inference_block_mask(batch: int, seq_len: int, mask_mod, device) -> BlockMask:
    """
    The BlockMask of pokemon_inference_mask: the turn's 64 tokens plus the history fit in one
    block, and that block always holds allowed pairs (a token reaches itself) and is never
    entirely allowed (a history token reaches only itself). So the map is one constant per
    battle. Built field by field: BlockMask.from_kv_blocks derives the query-side lists with
    a transpose that costs over a millisecond a call, and for one block the transpose is the
    same single entry.
    """
    one = torch.ones(batch, 1, 1, dtype=torch.int32, device=device)
    none = torch.zeros(batch, 1, 1, dtype=torch.int32, device=device)
    first = torch.zeros(batch, 1, 1, 1, dtype=torch.int32, device=device)
    return BlockMask(
        seq_lengths=(seq_len, seq_len),
        kv_num_blocks=one, kv_indices=first,
        full_kv_num_blocks=none, full_kv_indices=first,
        q_num_blocks=one, q_indices=first,
        full_q_num_blocks=none, full_q_indices=first,
        BLOCK_SIZE=(MASK_BLOCK_SIZE, MASK_BLOCK_SIZE), mask_mod=mask_mod,
    )


# ----------------------------------------
# 2. FLEX MODULES (Hardware Optimized)
# ----------------------------------------

class FlexEncoderLayer(nn.Module):
    # True: q, k and v come from one qkv.unbind(0) instead of three qkv[i] selects. The same
    # views and the same outputs and gradients, with one backward node (a stack) instead of three
    # zero-filled select backwards added together. ppo_update(fast_update=True) sets it on the
    # learner's own network; every other network keeps the select path.
    fast_unbind: bool = False

    def __init__(self, d_model, n_heads, ff_expansion=2.0, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * ff_expansion)),
            nn.GELU(),
            nn.Linear(int(d_model * ff_expansion), d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, block_mask):
        res = x
        x = self.norm1(x)
        B, S, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        if self.fast_unbind:
            q, k, v = qkv.unbind(0)
            attn_out = flex_attention(q, k, v, score_mod=logit_soft_cap, block_mask=block_mask)
        else:
            # Triton-optimized kernel call
            attn_out = flex_attention(
                qkv[0], qkv[1], qkv[2],
                score_mod=logit_soft_cap,
                block_mask=block_mask,
            )
        x = res + self.dropout(self.out_proj(attn_out.transpose(1, 2).reshape(B, S, D)))
        return x + self.ff(self.norm2(x))

class FlexReadout(nn.Module):
    """
    Per-turn readout attention.

    Query tokens:
      0 = actor
      1 = critic

    Key/value tokens:
      full token current turn
    """
    def __init__(self, d_model, n_heads, ff_expansion=2.0, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * ff_expansion)),
            nn.GELU(),
            nn.Linear(int(d_model * ff_expansion), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> torch.Tensor:
        """
        q_x:  [B, 2, D]   actor/critic query tokens
        kv_x: [B, :, D]  full current-turn tokens
        """
        B, Q, D = q_x.shape
        K = kv_x.shape[1]

        q = self.q_proj(self.norm_q(q_x)).view(B, Q, self.n_heads, -1).transpose(1, 2)
        k = self.k_proj(self.norm_kv(kv_x)).view(B, K, self.n_heads, -1).transpose(1, 2)
        v = self.v_proj(self.norm_kv(kv_x)).view(B, K, self.n_heads, -1).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        q_out = q_x + self.out_proj(attn_out.transpose(1, 2).reshape(B, Q, D))
        return q_out + self.ff(self.norm_ff(q_out))

# ----------------------------------------
# 3. POKE TRANSFORMER
# ----------------------------------------

class PokeTransformer(nn.Module):
    def __init__(self, act_dim, meta, emb_dims, out_dims, bank_dims, bank_ranges, n_heads=8, n_layers=4, **kwargs):
        super().__init__()
        self.meta, self.f_map = meta, meta["feature_map"]
        self.d_model = out_dims["pokemon_vec"]
        self.n_heads, self.n_layers = n_heads, n_layers
        self.bank_ranges = bank_ranges
        self.bank_dims = bank_dims
        self.kv_cache_len = kwargs.get("kv_cache_len", 64)
        
        # Identity Embeddings
        self.pokemon_id_emb = nn.Embedding(meta["vocab_pokemon"], emb_dims["pokemon"])
        self.item_emb = nn.Embedding(meta["vocab_item"], emb_dims["item"])
        self.ability_emb = nn.Embedding(meta["vocab_ability"], emb_dims["ability"])
        self.move_emb = nn.Embedding(meta["vocab_move"], emb_dims["move"])
        self.val_100_emb = nn.Embedding(bank_ranges["val_100"], bank_dims["val_100"])
        self.stat_emb = nn.Embedding(bank_ranges["stat"], bank_dims["stat"])
        self.power_emb = nn.Embedding(bank_ranges["power"], bank_dims["power"])
        
        # The matchup block (obs_matchup.py) is in the observation only when the assembler that made
        # `meta` was built with matchup=True. Then the first Linear of move_net and pokemon_net has
        # extra input columns at the END for it (see encode_features); without it nothing changes.
        self.mu_meta = meta.get("matchup")
        self.matchup = self.mu_meta is not None
        self.mu_move_in = int(self.mu_meta["move_feats"]) if self.matchup else 0
        self.mu_pok_in = (emb_dims["ability"] + emb_dims["item"] + 3) if self.matchup else 0
        # The moveset block (obs_movesets.py), only with meta["movesets"] (which needs the matchup block):
        # pokemon_net's first Linear gets more columns after the matchup ones, sum_k p_k * move_emb(id_k)
        # of the candidate moves and the block's dense numbers (Tera, threat, roles).
        self.ms_meta = meta.get("movesets")
        self.movesets = self.ms_meta is not None
        if self.movesets and not self.matchup:
            raise ValueError("the moveset block needs the matchup block in the observation")
        self.ms_dense = (int(self.ms_meta["dense"][1]) - int(self.ms_meta["dense"][0])) if self.movesets else 0
        self.ms_pok_in = (emb_dims["move"] + self.ms_dense) if self.movesets else 0
        # The SetHead (attach_set_head) is never built here: a network has one only when asked.
        self.set_head = None
        self.set_head_coef = 0.0

        # Subnets
        self.move_net = self._build_subnet(self._calc_move_in(emb_dims, bank_dims) + self.mu_move_in, self.d_model)
        self.ability_net = self._build_subnet(emb_dims["ability"] * meta["n_ability_slots"], out_dims["ability_vec"])
        self.pokemon_net = self._build_subnet(self._calc_pok_in(emb_dims, bank_dims, out_dims) + self.mu_pok_in + self.ms_pok_in,
                                              self.d_model)
        self.field_net = self._build_subnet(
                bank_dims["power"]
                + (meta["dim_global_scalars"] - 1)
                + (emb_dims["move"] * 2)
                + (emb_dims["pokemon"] * 4)
                + (emb_dims["ability"] * 2)
                + (emb_dims["item"] * 2)
                + meta["dim_transition_scalars"]
                + meta["action_dim"],
                self.d_model,
            )
        # Decision Tokens
        self.actor_tok = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.critic_tok = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.register_toks = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        self.action_emb = nn.Embedding(meta["action_dim"], self.d_model)
        # The single shared MLP informational bottleneck
        self.jepa_predictor = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        
        # The feed-forward width and dropout come from ModelConfig. They used to be dropped here,
        # so every layer was built at the defaults (2.0, 0.0) whatever the config said.
        ff_expansion, dropout = kwargs.get("ff_expansion", 2.0), kwargs.get("dropout", 0.0)
        self.transformer = nn.ModuleList([FlexEncoderLayer(self.d_model, n_heads, ff_expansion, dropout) for _ in range(n_layers)])
        self.readout = FlexReadout(self.d_model, n_heads, ff_expansion, dropout)

        self.pi_head = nn.Linear(self.d_model, act_dim)
        self.v_head = nn.Linear(self.d_model, kwargs.get("v_bins", 51))
        self.unpacker = ObservationUnpacker(meta)
        # The values the 51 bins stand for. This must be the same range the training targets
        # are encoded on (LearnerConfig.v_min / v_max). It used to be fixed at +/-1.6 while the
        # targets used +/-1.5, so every value the critic reported was read back 6.7% too large.
        self.register_buffer("v_support", torch.linspace(float(kwargs.get("v_min", -1.6)), float(kwargs.get("v_max", 1.6)), kwargs.get("v_bins", 51)))
        self._reset_parameters()

    def _build_subnet(self, in_d, out_d):
        return nn.Sequential(nn.Linear(in_d, out_d * 2), nn.GELU(), nn.Linear(out_d * 2, out_d), nn.LayerNorm(out_d))

    @staticmethod
    def _wide_subnet(net: nn.Sequential, x_old: torch.Tensor, x_new: torch.Tensor, *x_more: torch.Tensor) -> torch.Tensor:
        """
        net(cat([x_old, x_new, *x_more], -1)) for a subnet whose first Linear has the matchup
        block's columns (and the moveset block's after them) at the end, worked out as
        W_old @ x_old + b + W_new @ x_new (+ W_more @ x_more, one product per part, added in that
        order). The same function, but each product is the one the narrower network computes, with
        the same shapes, so a widened checkpoint (tools/widen_checkpoint.py, the new W = 0) gives
        exactly the old network's numbers in fp32 and in bf16, not numbers that differ in the last
        bits because a longer sum was cut up differently.
        """
        lin = net[0]
        k0 = x_old.shape[-1]
        w = lin.weight
        widths = [x_new.shape[-1]] + [x.shape[-1] for x in x_more]
        if w.shape[1] != k0 + sum(widths):
            raise RuntimeError(f"first Linear takes {w.shape[1]} inputs, given {k0} + {' + '.join(str(k) for k in widths)}")
        if not x_more:
            h = F.linear(x_old, w[:, :k0].contiguous(), lin.bias) + F.linear(x_new, w[:, k0:].contiguous())
        else:
            h = F.linear(x_old, w[:, :k0].contiguous(), lin.bias)
            k = k0
            for x in (x_new,) + tuple(x_more):
                h = h + F.linear(x, w[:, k:k + x.shape[-1]].contiguous())
                k += x.shape[-1]
        for layer in list(net)[1:]:
            h = layer(h)
        return h

    def attach_set_head(self, coef: float = 0.0) -> "SetHead":
        """
        Adds the SetHead (opposing abilities and items from tokens 7-12) after construction, so
        its parameters come after every other one (named_parameters order). get_parameter_zones
        puts them in the "aux" zone (the learner's last groups). `coef` is the loss weight
        compute_set_head_loss uses by default; 0 trains nothing.
        """
        from obs_matchup import load_vocab_lists, set_candidate_tables
        vocab = load_vocab_lists()
        ab, it = set_candidate_tables(vocab)
        if ab.shape[1] != self.meta["vocab_ability"] or it.shape[1] != self.meta["vocab_item"] or ab.shape[0] != self.meta["vocab_pokemon"]:
            raise RuntimeError("vocab.json does not match the network's vocabulary sizes")
        if getattr(self, "movesets", False):
            # With the moveset block the head also predicts the moves not shown and the Tera type.
            from obs_movesets import candidate_tables
            mv, te = candidate_tables(vocab)
            if mv.shape[1] != self.meta["vocab_move"]:
                raise RuntimeError("vocab.json does not match the network's move vocabulary size")
            head = SetHead(self.d_model, ab, it, mv, te)
        else:
            head = SetHead(self.d_model, ab, it)
        self.set_head = head.to(self.actor_tok.device)
        self.set_head_coef = float(coef)
        return self.set_head

    def attach_opp_action_head(self, coef: float = 0.0) -> "OppActionHead":
        """
        Adds the OppActionHead (what the opponent does next, from the field token and tokens 7-12)
        after construction, like attach_set_head: its parameters come last, in the "aux" zone.
        Call it after make_model and before the learner builds its optimizer.
        """
        head = OppActionHead(self.d_model)
        self.opp_action_head = head.to(self.actor_tok.device)
        self.opp_action_coef = float(coef)
        return self.opp_action_head

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): 
                nn.init.normal_(m.weight, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
                
        self.val_100_emb.weight.data.copy_(
            get_sinusoidal_encoding(self.bank_ranges["val_100"], self.bank_dims["val_100"])
        )
        self.stat_emb.weight.data.copy_(
            get_sinusoidal_encoding(self.bank_ranges["stat"], self.bank_dims["stat"])
        )
        self.power_emb.weight.data.copy_(
            get_sinusoidal_encoding(self.bank_ranges["power"], self.bank_dims["power"])
        )
        
        # Depth-scaled initialization for residual projections
        # Scale by 1 / sqrt(2 * n_layers) because there are 2 residual additions per layer (Attn + FFN)
        std_proj = 0.02
        if self.n_layers > 7:
            std_proj = 0.02 / (2 * self.n_layers) ** 0.5
        
        # Override the 0.02 std with Xavier Uniform to ensure Q and K dot products
        # have enough variance to create sharp attention distributions.
        d = self.d_model
        for layer in self.transformer:
            # Slice the fused QKV matrix to initialize Q, K, and V correctly
            with torch.no_grad():
                # QK gets Xavier
                nn.init.xavier_uniform_(layer.qkv_proj.weight[:2*d, :])
                # V gets standard low-variance pipeline init
                nn.init.normal_(layer.qkv_proj.weight[2*d:, :], std=0.02)
                
            nn.init.normal_(layer.out_proj.weight, std=std_proj)
            nn.init.normal_(layer.ff[2].weight, std=std_proj)
            
        # 3. Apply Spiky Init to the Readout Cross-Attention
        nn.init.xavier_uniform_(self.readout.q_proj.weight, gain=1.5)
        nn.init.xavier_uniform_(self.readout.k_proj.weight, gain=1.5)
        # Keep V projection and residuals standard
        nn.init.normal_(self.readout.v_proj.weight, std=0.02)
        nn.init.normal_(self.readout.out_proj.weight, std=std_proj)
        nn.init.normal_(self.readout.ff[2].weight, std=std_proj)

        # 4. Ensure decision tokens start small so they don't drown out the board
        nn.init.normal_(self.actor_tok, std=0.02)
        nn.init.normal_(self.critic_tok, std=0.02)
        nn.init.normal_(self.register_toks, std=0.02)

    def _calc_move_in(self, emb, bank):
        m = self.f_map["move"]
        # Use flags_end to include the new STAB, Status, and Identity dimensions
        return emb["move"] + (bank["val_100"] * 2) + bank["power"] + (m["flags_end"] - m["onehots_raw"][0])

    def _calc_pok_in(self, emb, bank, out):
        """Dynamically calculates the Pokemon Net input dimension from metadata."""
        b = self.f_map["body"]
        
        # 1. Categorical ID Dimensions (Species + Item)
        id_dims = emb["pokemon"] + emb["item"]
        
        # 2. Bank Embedding Dimensions
        # HP (1) and Level (1) use val_100 bank
        # Combat Stats (15) and Weight/Height (2) use stat bank
        stat_slice_len = b["stats_int"][1] - b["stats_int"][0]
        phys_slice_len = (b["height_int"] + 1) - b["weight_int"] # Weight and Height indices
        
        bank_dims = (bank["val_100"] * 2) + (bank["stat"] * (stat_slice_len + phys_slice_len))
        
        # 3. Subnet Output Dimensions (Ability Vec)
        subnet_dims = out["ability_vec"]
        
        # 4. Raw Float Slices (Non-Embedded)
        # Slice 1: The Boosts block (91 floats)
        # Slice 2: Flags, Mechanics, Types, Effects, Status, Gender, and Slot ID (Rest of the buffer)
        raw_boost_len = b["boosts_raw"][1] - b["boosts_raw"][0]
        raw_tail_len = self.meta["dim_pokemon_body"] - b["flags_raw"][0]
        
        return id_dims + bank_dims + subnet_dims + raw_boost_len + raw_tail_len
    
    def get_parameter_zones(self) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
        """
        Categorizes model parameters into logical zones for the Learner to govern 
        gradients and learning rates depending on the training phase.
        """
        zones = {
            "embeddings": [], "subnets": [], "transformer": [],
            "jepa": [], "readout": [], "pi": [], "v": [],
            # The auxiliary heads attached after construction (attach_set_head, attach_opp_action_head).
            # Last, so their optimizer groups come after every other one; empty without heads.
            "aux": [],
        }
        for name, p in self.named_parameters():
            if name.startswith(AUX_HEAD_PREFIXES):
                zones["aux"].append((name, p))
            elif name.startswith("pi_head"):
                zones["pi"].append((name, p))
            elif name.startswith("v_head"):
                zones["v"].append((name, p))
            elif name.startswith("readout"):
                zones["readout"].append((name, p))
            elif name.startswith("action_emb") or name.startswith("jepa_predictor"):
                zones["jepa"].append((name, p))
            elif "transformer" in name or "actor_tok" in name or "critic_tok" in name or "register_toks" in name:
                zones["transformer"].append((name, p))
            elif name.endswith("_emb.weight"):
                zones["embeddings"].append((name, p))
            elif "net" in name: # move_net, ability_net, pokemon_net, field_net
                zones["subnets"].append((name, p))
            else:
                # Safe Catch-all
                zones["transformer"].append((name, p))
        return zones
    
    def enable_bf16_recurrent_path(self):
        bf16_modules = [
            self.pokemon_id_emb,
            self.item_emb,
            self.ability_emb,
            self.move_emb,
            self.val_100_emb,
            self.stat_emb,
            self.power_emb,
            self.move_net,
            self.ability_net,
            self.pokemon_net,
            self.field_net,
            self.transformer,
            self.readout,
            self.pi_head,
            self.v_head,
        ]
    
        for mod in bf16_modules:
            mod.bfloat16()
    
        self.actor_tok.data = self.actor_tok.data.bfloat16()
        self.critic_tok.data = self.critic_tok.data.bfloat16()
        self.register_toks.data = self.register_toks.data.bfloat16()
    
    def recurrent_dtype(self) -> torch.dtype:
        return self.actor_tok.dtype
        
    def encode_features(self, obs_flat):
        obs = self.unpacker(obs_flat)
        B = obs_flat.shape[0]
        rec_dtype = self.recurrent_dtype()
        b_map = self.f_map["body"]
        m_map = self.f_map["move"]

        # --- MOVE ENCODING ---
        m_sc = obs["move_scalars"]
        m_combined = torch.cat([
            self.move_emb(obs["move_ids"]),
            self.val_100_emb(m_sc[..., m_map["acc_int"]].long()),
            self.power_emb(m_sc[..., m_map["pwr_int"]].long()),
            self.val_100_emb(m_sc[..., m_map["pp_int"]].long()),
            m_sc[..., m_map["onehots_raw"][0] : m_map["flags_end"]].to(rec_dtype)
        ], dim=-1)
        if self.matchup:
            m_vecs = self._wide_subnet(
                self.move_net, m_combined.view(-1, m_combined.shape[-1]),
                obs["matchup_moves"].reshape(-1, self.mu_move_in).to(rec_dtype),
            ).view(B, 48, -1)
        else:
            m_vecs = self.move_net(m_combined.view(-1, m_combined.shape[-1])).view(B, 48, -1)

        # --- ABILITY ENCODING ---
        a_vecs = self.ability_net(self.ability_emb(obs["ability_ids"]).view(B, 12, -1))

        # --- POKEMON ENCODING ---
        p_body = obs["pokemon_body"]
        p_in = torch.cat([
            self.pokemon_id_emb(obs["pokemon_ids"][:, :, 0]), # Species
            self.item_emb(obs["pokemon_ids"][:, :, 1]),       # Item
            self.val_100_emb(p_body[:, :, b_map["hp_int"]].long()), # HP %
            self.stat_emb(p_body[:, :, b_map["stats_int"][0] : b_map["stats_int"][1]].long()).flatten(2), # Stats
            self.val_100_emb(p_body[:, :, b_map["level_int"]].long()), # Level
            self.stat_emb(p_body[:, :, b_map["weight_int"] : b_map["height_int"] + 1].long()).flatten(2), # W/H
            a_vecs, # Ability Subnet Result
            p_body[:, :, b_map["boosts_raw"][0] : b_map["boosts_raw"][1]].to(rec_dtype), # Raw Boosts
            p_body[:, :, b_map["flags_raw"][0] : ].to(rec_dtype) # Flags, Types, Status, Identity
        ], dim=-1)
        if self.matchup:
            # The candidates' embeddings weighted by their probabilities, and the three item flags.
            mu_ab = (self.ability_emb(obs["matchup_ability_ids"]) * obs["matchup_ability_p"].to(rec_dtype).unsqueeze(-1)).sum(dim=2)
            mu_it = (self.item_emb(obs["matchup_item_ids"]) * obs["matchup_item_p"].to(rec_dtype).unsqueeze(-1)).sum(dim=2)
            p_new = torch.cat([mu_ab, mu_it, obs["matchup_flags"].to(rec_dtype)], dim=-1)
            if self.movesets:
                # The candidate moves' embeddings weighted by their probabilities, then the dense numbers.
                ms_mv = (self.move_emb(obs["movesets_move_ids"]) * obs["movesets_move_p"].to(rec_dtype).unsqueeze(-1)).sum(dim=2)
                ms_new = torch.cat([ms_mv, obs["movesets_dense"].to(rec_dtype)], dim=-1)
                p_tokens = self._wide_subnet(
                    self.pokemon_net, p_in.view(-1, p_in.shape[-1]), p_new.view(-1, self.mu_pok_in),
                    ms_new.view(-1, self.ms_pok_in),
                ).view(B, 12, -1)
            else:
                p_tokens = self._wide_subnet(
                    self.pokemon_net, p_in.view(-1, p_in.shape[-1]), p_new.view(-1, self.mu_pok_in),
                ).view(B, 12, -1)
        else:
            p_tokens = self.pokemon_net(p_in.view(-1, p_in.shape[-1])).view(B, 12, -1)

        # --- FIELD ENCODING ---
        field_in = torch.cat([
            self.power_emb(obs["global_scalars"][:, self.f_map["global"]["turn_int"]].long()),
            obs["global_scalars"][:, self.f_map["global"]["remainder_raw"][0]:].to(rec_dtype),
            self.move_emb(obs["transition_move_ids"]).view(B, -1),
            self.pokemon_id_emb(obs["transition_pokemon_ids"]).view(B, -1),
            self.ability_emb(obs["transition_ability_ids"]).view(B, -1),
            self.item_emb(obs["transition_item_ids"]).view(B, -1),
            obs["transition_scalars"].to(rec_dtype),
            obs["action_mask"].to(rec_dtype),
        ], dim=-1)
        field_token = self.field_net(field_in).unsqueeze(1)

        return field_token, p_tokens, m_vecs
    
    def condition_tokens(self, field_tok, p_toks, m_toks, episode_ids=None, kv_cache=None, kv_cache_len=None, batch_seq_len=0):
        """Core Transformer Backbone. Returns fully contextualized tokens (Z_t)."""
        B = field_tok.shape[0]
        device = field_tok.device
        K = self.kv_cache_len
        
        current_turn = torch.cat([
                field_tok,                          # 0
                p_toks,                             # 1-12
                m_toks,                             # 13-60 (48 Moves)
                self.register_toks.expand(B, 1, -1),
                self.actor_tok.expand(B, 1, -1),    # 61
                self.critic_tok.expand(B, 1, -1),   # 62
            ], dim=1)  # [B, 63, D]

        # --- PACKED TRAINING MODE ---
        if episode_ids is not None:
            n_seq = B // batch_seq_len
            s_tok = batch_seq_len * TOKENS_PER_TURN
            x = current_turn.view(n_seq, batch_seq_len, TOKENS_PER_TURN, self.d_model).reshape(n_seq, s_tok, self.d_model)
            expanded_ids = episode_ids.view(n_seq, batch_seq_len, 1).expand(-1, -1, TOKENS_PER_TURN).reshape(n_seq, s_tok)

            mask_mod = lambda b, h, q, k: pokemon_batched_episodic_mask(b, h, q, k, expanded_ids, K)
            if _FAST_BLOCK_MASKS:
                b_mask = episodic_block_mask(episode_ids.view(n_seq, batch_seq_len), K, mask_mod)
            else:
                b_mask = create_block_mask(mask_mod, B=n_seq, H=1, Q_LEN=s_tok, KV_LEN=s_tok, device=device)

            for layer in self.transformer: x = layer(x, b_mask)
            turn_x = x.view(n_seq * batch_seq_len, TOKENS_PER_TURN, self.d_model)
            
            return turn_x, None, None

        # --- BATCHED INFERENCE MODE ---
        if kv_cache_len is None:
            kv_cache_len = torch.zeros(B, dtype=torch.long, device=device)
        else:
            kv_cache_len = kv_cache_len.to(device=device, dtype=torch.long)
            
        S = TOKENS_PER_TURN + K
        mask_mod = lambda b, h, q, k: pokemon_inference_mask(b, h, q, k, kv_cache_len, K)
        if _FAST_BLOCK_MASKS and S <= MASK_BLOCK_SIZE:
            b_mask = inference_block_mask(B, S, mask_mod, device)
        else:
            # A history longer than one block: let torch work the map out from the rule.
            b_mask = create_block_mask(mask_mod, B=B, H=1, Q_LEN=S, KV_LEN=S, device=device)
        
        hist_pos = torch.arange(K, device=device).unsqueeze(0)                 
        hist_valid = hist_pos >= (K - kv_cache_len.unsqueeze(1)) 
        
        x = current_turn
        new_kv_cache = []
        new_kv_cache_len = torch.clamp(kv_cache_len + 1, max=K)

        for i, layer in enumerate(self.transformer):
            cur_field_in = x[:, FIELD_IDX:FIELD_IDX + 1, :]   
            
            if kv_cache is not None:
                hist_fields_L = kv_cache[i].to(device=device, dtype=x.dtype, non_blocking=True)
            else:
                hist_fields_L = torch.zeros(B, K, self.d_model, device=device, dtype=x.dtype)
                
            hist_fields_L = hist_fields_L * hist_valid.unsqueeze(-1).to(x.dtype)
            x_full  = torch.cat([x, hist_fields_L], dim=1)
            x_full  = layer(x_full , b_mask)
            x = x_full[:, :TOKENS_PER_TURN, :]
            
            updated_hist = torch.cat([hist_fields_L, cur_field_in], dim=1)[:, -K:, :]
            new_kv_cache.append(updated_hist)
        
        return x, new_kv_cache, new_kv_cache_len
    
    def readout_policy(self, Z_t):
        """Cross-attention readout for PPO outputs."""
        read_q = Z_t[:, ACTOR_IDX:CRITIC_IDX + 1, :]      # [B, 2, D]
        readout = self.readout(read_q, Z_t)
        
        pi = self.pi_head(readout[:, 0, :])  # actor
        v  = self.v_head(readout[:, 1, :])   # critic
        v_exp = (torch.softmax(v, dim=-1) * self.v_support).sum(dim=-1)
        
        return pi.float(), v.float(), v_exp.float()

    def forward_jepa(self, obs_flat: torch.Tensor, act_t: torch.Tensor, episode_ids: torch.Tensor = None, batch_seq_len: int = 0):
        """
        JEPA Pre-training Path. 
        Returns the predicted latents for t+1 and the true latents at t.
        """
        # 1. Standard encoding & backbone conditioning
        field_tok, p_toks, m_toks = self.encode_features(obs_flat)
        Z_t, _, _ = self.condition_tokens(field_tok, p_toks, m_toks, episode_ids=episode_ids, batch_seq_len=batch_seq_len)
        
        # 2. Action Embedding & Broadcasting
        a_emb = self.action_emb(act_t)             # [B, D]
        a_broadcast = a_emb.unsqueeze(1).expand(-1, 13, -1) # Broadcast to Field + 12 Pokemon
        
        # 3. Slice the state facets we want to predict (0 through 12)
        state_facets_t = Z_t[:, 0:13, :]           # [B, 13, D]
        
        # 4. Concatenate and pass through the bottleneck MLP
        pred_input = torch.cat([state_facets_t, a_broadcast], dim=-1) # [B, 13, 2D]
        Z_pred = self.jepa_predictor(pred_input)   # [B, 13, D]
        
        return Z_pred, Z_t

    def forward(self, obs_flat: torch.Tensor, episode_ids: torch.Tensor = None, kv_cache: list[torch.Tensor] = None, kv_cache_len: torch.Tensor = None, batch_seq_len: int = 0,
                return_z: bool = False):
        """Standard Inference / PPO Path. return_z=True adds Z_t (the conditioned tokens [B, 64, D]) as a sixth output, for the auxiliary heads."""
        field_tok, p_toks, m_toks = self.encode_features(obs_flat)
        Z_t, new_kv_cache, new_kv_cache_len = self.condition_tokens(
            field_tok, p_toks, m_toks, episode_ids, kv_cache, kv_cache_len, batch_seq_len
        )

        pi, v, v_exp = self.readout_policy(Z_t)

        if episode_ids is not None:
            if return_z:
                return pi, v, v_exp, None, None, Z_t
            return pi, v, v_exp, None, None

        if return_z:
            return pi, v, v_exp, new_kv_cache, new_kv_cache_len, Z_t
        return pi, v, v_exp, new_kv_cache, new_kv_cache_len
    
class ObservationUnpacker(nn.Module):
    """
    Slices and reshapes flat observation tensors into structured components.
    
    Expects metadata containing 'offsets' and dynamic dimensions from the 
    ObservationAssembler.
    """
    def __init__(self, meta: dict):
        super().__init__()
        self.meta = meta
        self.offsets = meta["offsets"]
        self.dim_body = meta["dim_pokemon_body"]
        self.dim_move_sc = meta["dim_move_scalars"] // 4
        self.dim_transition_sc = meta["dim_transition_scalars"]
        self.transition_map = meta["feature_map"]["transition"]
        self.mu_meta = meta.get("matchup")
        self.ms_meta = meta.get("movesets")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {name: x[:, start:end] for name, (start, end) in self.offsets.items()}
        
        # Reshape structured fields
        out["pokemon_body"] = out["pokemon_body"].reshape(-1, 12, self.dim_body)
        out["pokemon_ids"]  = out["pokemon_ids"].reshape(-1, 12, 2).to(torch.long)
        out["ability_ids"]  = out["ability_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_ids"]     = out["move_ids"].reshape(-1, 12, 4).to(torch.long)
        out["move_scalars"] = out["move_scalars"].reshape(-1, 12, 4, self.dim_move_sc)
        out["transition_ids"] = out["transition_ids"].reshape(-1, self.meta["dim_transition_ids"]).to(torch.long)
        out["transition_scalars"] = out["transition_scalars"].reshape(-1, self.dim_transition_sc)
        
        tmap = self.transition_map
        out["transition_move_ids"] = out["transition_ids"][:, tmap["move_ids"][0]:tmap["move_ids"][1]]
        out["transition_pokemon_ids"] = out["transition_ids"][:, tmap["pokemon_ids"][0]:tmap["pokemon_ids"][1]]
        out["transition_ability_ids"] = out["transition_ids"][:, tmap["ability_ids"][0]:tmap["ability_ids"][1]]
        out["transition_item_ids"] = out["transition_ids"][:, tmap["item_ids"][0]:tmap["item_ids"][1]]

        if self.mu_meta is not None:
            # The matchup block (obs_matchup.py), present only when the assembler that made `meta` had matchup=True.
            mu, mm = out["matchup"], self.mu_meta["mon"]
            out["matchup_moves"] = mu[:, :self.mu_meta["move_block"]].reshape(-1, 12, 4, self.mu_meta["move_feats"])
            mon = mu[:, self.mu_meta["move_block"]:].reshape(-1, 12, self.mu_meta["mon_feats"])
            out["matchup_ability_ids"] = mon[..., mm["ability_ids"][0]:mm["ability_ids"][1]].to(torch.long)
            out["matchup_ability_p"] = mon[..., mm["ability_p"][0]:mm["ability_p"][1]]
            out["matchup_item_ids"] = mon[..., mm["item_ids"][0]:mm["item_ids"][1]].to(torch.long)
            out["matchup_item_p"] = mon[..., mm["item_p"][0]:mm["item_p"][1]]
            out["matchup_flags"] = mon[..., mm["flags"][0]:mm["flags"][1]]

        if self.ms_meta is not None:
            # The moveset block (obs_movesets.py), present only when the assembler had movesets=True.
            ms = self.ms_meta
            mon = out["movesets"].reshape(-1, 12, ms["mon_feats"])
            out["movesets_move_ids"] = mon[..., ms["move_ids"][0]:ms["move_ids"][1]].to(torch.long)
            out["movesets_move_p"] = mon[..., ms["move_p"][0]:ms["move_p"][1]]
            out["movesets_dense"] = mon[..., ms["dense"][0]:ms["dense"][1]]

        return out


# ----------------------------
# SetHead: the opposing Pokemon's abilities and items
# ----------------------------

# Rows of the labels obs_matchup.build_set_labels writes: [species id, true ability id, true item id].
SET_LABEL_SPECIES, SET_LABEL_ABILITY, SET_LABEL_ITEM = 0, 1, 2
# With the moveset block the rows are 8 wide (obs_matchup.build_set_labels(movesets=True)): then the
# true moves (+id not shown, -id shown, 0 none) and the true Tera type (obs_movesets.TERA_TYPES index + 1).
SET_LABEL_MOVES, SET_LABEL_TERA, SET_LABEL_WIDTH_MOVESETS = (3, 7), 7, 8
OPP_TOKENS = slice(7, 13)   # the opponent's six Pokemon tokens in Z_t (0 field, 1-6 ours, 7-12 theirs)


class SetHead(nn.Module):
    """
    Predicts each opposing Pokemon's ability and item from its conditioned token (Z_t 7-12),
    as logits over the vocabulary, limited at the loss to what that species' random battle sets
    allow (obs_matchup.set_candidate_tables, by vocabulary id). Runs in float32.
    """

    def __init__(self, d_model: int, ability_allowed, item_allowed, move_allowed=None, tera_allowed=None):
        super().__init__()
        ab = torch.as_tensor(ability_allowed, dtype=torch.bool)
        it = torch.as_tensor(item_allowed, dtype=torch.bool)
        self.ability = nn.Linear(d_model, ab.shape[1])
        self.item = nn.Linear(d_model, it.shape[1])
        for lin in (self.ability, self.item):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        # Rebuilt from the set data, never saved: not part of the checkpoint.
        self.register_buffer("ability_allowed", ab, persistent=False)
        self.register_buffer("item_allowed", it, persistent=False)
        # With the moveset block (obs_movesets.py): the moves not shown (one logit per move of the
        # vocabulary, multi-label) and the Tera type (obs_movesets.TERA_TYPES), after the two above,
        # so their parameters come after them. Absent otherwise: the head is what it was.
        self.has_sets = move_allowed is not None
        if self.has_sets:
            mv = torch.as_tensor(move_allowed, dtype=torch.bool)
            te = torch.as_tensor(tera_allowed, dtype=torch.bool)
            self.moves = nn.Linear(d_model, mv.shape[1])
            self.tera = nn.Linear(d_model, te.shape[1])
            for lin in (self.moves, self.tera):
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
            self.register_buffer("move_allowed", mv, persistent=False)
            self.register_buffer("tera_allowed", te, persistent=False)

    def forward(self, opp_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = opp_tokens.float()
        return self.ability(h), self.item(h)

    def forward_sets(self, opp_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(move logits, Tera logits) of the moveset part (has_sets only)."""
        h = opp_tokens.float()
        return self.moves(h), self.tera(h)


def _restricted_nll(logits: torch.Tensor, allowed: torch.Tensor, target: torch.Tensor):
    """
    Cross-entropy over the allowed ids only, summed over the kept entries, with no boolean
    indexing (nothing waits for the GPU). Left out: no label (0), a label the species' sets do
    not allow, fewer than 2 allowed ids. Returns (nll sum, kept count, top-1 count).
    """
    n_allowed = allowed.sum(dim=-1)
    t = target.clamp(min=0, max=logits.shape[-1] - 1)
    hit = allowed.gather(-1, t.unsqueeze(-1)).squeeze(-1)
    keep = ((target > 0) & hit & (n_allowed >= 2)).to(logits.dtype)
    masked = logits.masked_fill(~allowed, -1e9)
    nll = -torch.log_softmax(masked, dim=-1).gather(-1, t.unsqueeze(-1)).squeeze(-1)
    correct = (masked.argmax(dim=-1) == t).to(logits.dtype)
    return (nll * keep).sum(), keep.sum(), (correct * keep).sum()


def compute_set_head_loss(net: nn.Module, z_tokens: torch.Tensor, labels: torch.Tensor,
                          coef: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    The SetHead's term for the loss: coef * mean cross-entropy of the true ability and item of
    each opposing Pokemon the observing side has not seen, over that species' set candidates.

        z_tokens  Z_t, the conditioned tokens [B, 64, D] (condition_tokens' first output)
        labels    [B, 6, 3] integers from obs_matchup.build_set_labels, rows aligned with z_tokens;
                  [B, 6, 8] with the moveset block (then a head with has_sets also learns the moves not
                  shown, multi-label, and the Tera type: set_move_n / set_move_hits, set_tera_n / set_tera_top1)
        coef      default: net.set_head_coef (attach_set_head's argument; 0 unless asked)

    Returns (term, stats). With coef 0 the term is a zero that still depends on the head (no
    gradient reaches anything); the caller can skip the call then. stats are detached 0-d
    tensors, so nothing waits for the GPU here: the raw loss, and per kind the labelled entries
    (set_ability_n, set_item_n) and how many the head ranks first (set_ability_top1, set_item_top1).
    """
    head = getattr(net, "set_head", None)
    if head is None:
        raise RuntimeError("the network has no SetHead: call net.attach_set_head() first")
    coef = float(net.set_head_coef if coef is None else coef)
    ab_logits, it_logits = head(z_tokens[:, OPP_TOKENS, :])
    lab = labels.to(device=ab_logits.device, dtype=torch.long)
    species = lab[..., SET_LABEL_SPECIES].clamp(min=0, max=head.ability_allowed.shape[0] - 1)
    a_sum, a_n, a_top = _restricted_nll(ab_logits, head.ability_allowed[species], lab[..., SET_LABEL_ABILITY])
    i_sum, i_n, i_top = _restricted_nll(it_logits, head.item_allowed[species], lab[..., SET_LABEL_ITEM])
    if getattr(head, "has_sets", False) and lab.shape[-1] >= SET_LABEL_WIDTH_MOVESETS:
        # The moveset part: every entry of the sum and of the count is one opposing Pokemon.
        mv_logits, te_logits = head.forward_sets(z_tokens[:, OPP_TOKENS, :])
        m_sum, m_n, m_hit = _move_bce(mv_logits, head.move_allowed[species], lab[..., SET_LABEL_MOVES[0]:SET_LABEL_MOVES[1]])
        t_sum, t_n, t_top = _restricted_nll(te_logits, head.tera_allowed[species], lab[..., SET_LABEL_TERA])
        loss = (a_sum + i_sum + m_sum + t_sum) / (a_n + i_n + m_n + t_n).clamp(min=1.0)
        stats = {
            "set_loss": loss.detach(),
            "set_ability_n": a_n.detach(), "set_item_n": i_n.detach(),
            "set_ability_top1": a_top.detach(), "set_item_top1": i_top.detach(),
            "set_move_n": m_n.detach(), "set_move_hits": m_hit.detach(),
            "set_tera_n": t_n.detach(), "set_tera_top1": t_top.detach(),
        }
        return coef * loss, stats
    loss = (a_sum + i_sum) / (a_n + i_n).clamp(min=1.0)
    stats = {
        "set_loss": loss.detach(),
        "set_ability_n": a_n.detach(), "set_item_n": i_n.detach(),
        "set_ability_top1": a_top.detach(), "set_item_top1": i_top.detach(),
    }
    return coef * loss, stats


def _move_bce(logits: torch.Tensor, allowed: torch.Tensor, ids: torch.Tensor):
    """
    The move part of the SetHead's loss, with no boolean indexing (nothing waits for the GPU).

        logits   [..., V] float32, one per move vocabulary id (+1; 0 is never a move)
        allowed  [..., V] bool, the moves the species' sets hold
        ids      [..., 4] the true moves: +id not shown yet (a label), -id shown (left out), 0 none

    Candidates: the allowed moves not shown. Per Pokemon with at least one true move among its
    candidates, the binary cross-entropy averaged over its candidates (the true ones are 1).
    Returns (sum over those Pokemon, their count, sum of their precision at k: the share of the k
    highest candidates that are true, k = the number of true candidates).
    """
    V = logits.shape[-1]
    pos_ids = ids.clamp(min=0, max=V - 1)
    neg_ids = (-ids).clamp(min=0, max=V - 1)
    pos = torch.zeros_like(allowed).scatter_(-1, pos_ids, True)
    shown = torch.zeros_like(allowed).scatter_(-1, neg_ids, True)
    pos[..., 0] = False
    shown[..., 0] = False
    cand = allowed & ~shown
    pos = pos & cand
    n_pos = pos.sum(dim=-1)
    n_cand = cand.sum(dim=-1)
    keep = (n_pos > 0).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, pos.to(logits.dtype), reduction="none")
    per_mon = (bce * cand.to(logits.dtype)).sum(dim=-1) / n_cand.clamp(min=1).to(logits.dtype)
    masked = logits.detach().masked_fill(~cand, -1e9)
    rank = masked.argsort(dim=-1, descending=True).argsort(dim=-1)
    top = rank < n_pos.unsqueeze(-1)
    prec = (top & pos).sum(dim=-1).to(logits.dtype) / n_pos.clamp(min=1).to(logits.dtype)
    return (per_mon * keep).sum(), keep.sum(), (prec * keep).sum()


# ----------------------------
# OppActionHead: what the opponent does next (read_reward.py's opp_action)
# ----------------------------

# The auxiliary heads' parameter names: get_parameter_zones puts them in the "aux" zone.
AUX_HEAD_PREFIXES: Final = ("set_head.", "opp_action_head.")
FIELD_TOKEN = 0
OPP_ACTION_CLASSES = 8      # 0 moved or stayed in, 1-6 switched to the opponent's slot 0-5, 7 to one not seen yet


class OppActionHead(nn.Module):
    """
    Predicts the opponent's next action from Z_t: "switched to slot k" (classes 1-6) from that
    opponent token (7-12) alone, a pointer over the six; "moved" (0) and "switched to one not seen
    yet" (7) from the field token and the mean of the six. Runs in float32.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.slot = nn.Linear(d_model, 1)
        self.other = nn.Linear(2 * d_model, 2)
        for lin in (self.slot, self.other):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)

    def forward(self, z_tokens: torch.Tensor) -> torch.Tensor:
        opp = z_tokens[:, OPP_TOKENS, :].float()                         # [B, 6, D]
        pooled = torch.cat([z_tokens[:, FIELD_TOKEN, :].float(), opp.mean(dim=1)], dim=-1)
        o = self.other(pooled)                                           # [B, 2]
        return torch.cat([o[:, :1], self.slot(opp).squeeze(-1), o[:, 1:]], dim=-1)   # [B, 8]


def compute_opp_action_loss(net: nn.Module, z_tokens: torch.Tensor, labels: torch.Tensor,
                            coef: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    coef * mean cross-entropy of the opponent's next action over the rows that have one (labels
    -1 are left out). labels: [B] integers in -1..7, rows aligned with z_tokens. Returns (term,
    stats); stats are detached 0-d tensors: opp_loss, opp_n (labelled rows), opp_top1 (rows the
    head ranks first), opp_moved (labelled rows whose label is "moved", the majority class).
    """
    head = getattr(net, "opp_action_head", None)
    if head is None:
        raise RuntimeError("the network has no OppActionHead: call net.attach_opp_action_head() first")
    coef = float(getattr(net, "opp_action_coef", 0.0) if coef is None else coef)
    logits = head(z_tokens)
    lab = labels.to(device=logits.device, dtype=torch.long).view(-1)
    keep = ((lab >= 0) & (lab < OPP_ACTION_CLASSES)).to(logits.dtype)
    t = lab.clamp(min=0, max=OPP_ACTION_CLASSES - 1)
    nll = -torch.log_softmax(logits, dim=-1).gather(-1, t.unsqueeze(-1)).squeeze(-1)
    n = keep.sum()
    loss = (nll * keep).sum() / n.clamp(min=1.0)
    top1 = ((logits.argmax(dim=-1) == t).to(logits.dtype) * keep).sum()
    moved = ((t == 0).to(logits.dtype) * keep).sum()          # the majority class: the accuracy's baseline
    return coef * loss, {"opp_loss": loss.detach(), "opp_n": n.detach(), "opp_top1": top1.detach(), "opp_moved": moved.detach()}


def boosted_row_weights(boosted: torch.Tensor, weight: float) -> torch.Tensor:
    """
    The per-row weight of the policy loss: `weight` where the row is boosted (the opponent's active
    Pokemon has a stat at +1 or more), 1 elsewhere, divided by their mean so that the minibatch's
    mean weight is 1. Float32, one per row.
    """
    b = boosted.reshape(-1)
    w = torch.where(b > 0, torch.full(b.shape, float(weight), dtype=torch.float32, device=b.device),
                    torch.ones(b.shape, dtype=torch.float32, device=b.device))
    return w / w.mean().clamp(min=1e-8)

# ----------------------------
# Training Components
# ----------------------------

@dataclass
class PPOUpdateStats:
    """Statistics container for a single PPO update batch."""
    approx_kl: float
    clip_frac: float
    entropy: float
    v_loss: float
    pg_loss: float
    total_loss: float
    n_mb: int
    # Imitation modes only: the share of steps where the policy's best legal action is the
    # teacher's. None in every other mode.
    acc: Optional[float] = None
    # The stage 2-4 terms that were on in this update (opp_acc, opp_n, opp_loss, set_loss,
    # set_ability_top1, set_item_top1, boosted_share). None when none was.
    aux: Optional[Dict[str, float]] = None
    # ppo_update(record_first_kl=True) only (train.py --lag-stats): the approx_kl of the update's first
    # minibatch, computed before any optimizer step of the update (the data's own policy against the
    # one the update starts from). None otherwise.
    kl_first: Optional[float] = None

# The modes whose policy loss is cross-entropy against the teacher's action.
IMITATION_LOSS_MODES: Final = ("imitation", "imitation_frozen_backbone", "imitation_with_jepa")

class AsyncEpisodeDataset:
    def __init__(self, act_dim: int, device: str):
        self.act_dim, self.device = act_dim, device
        self.clear()

    def __len__(self): return int(self.n_steps)

    def clear(self):
        # Flattened storage for performance
        self.obs, self.act, self.logp, self.val, self.adv, self.ret, self.next_hp = [], [], [], [], [], [], []
        self.ep_ids = []  # New: tracks boundaries via flat IDs
        # Named per-step columns beyond the eight (step_columns.py): one dict per add_steps call
        # that brought any, keyed by the index of that call. Empty unless a caller passes extras.
        self.extras = {}
        # The names of the columns tensorize() puts after the eight, in that order.
        self.extra_names = ()
        self.n_steps = 0
        self.ep_counter = 0
        self._tensor_cache = None
        self._device_cache = None

    def add_steps(self, obs, act, logp, val, adv, ret, next_hp, ep_ids, extras=None):
        """
        Maintains the flat plumbing. 'ep_ids' should match the length of the steps.
        extras: None, or a dict of name -> tensor with one row per step (as many rows as act),
        kept in its own dtype. A column with another number of rows is refused (ValueError).
        """
        def dev(x): return x.detach().to(self.device)
        n = int(act.shape[0])
        if extras:
            bad = [name for name, x in extras.items() if not torch.is_tensor(x) or x.ndim < 1 or int(x.shape[0]) != n]
            if bad:
                raise ValueError(f"extra columns {bad} do not have one row for each of the {n} steps")
        self.obs.append(obs.detach().to("cpu", non_blocking=True))
        self.act.append(dev(act)); self.logp.append(dev(logp))
        self.val.append(dev(val)); self.adv.append(dev(adv))
        self.ret.append(dev(ret)); self.next_hp.append(dev(next_hp))
        self.ep_ids.append(dev(ep_ids))
        if extras:
            self.extras[len(self.act) - 1] = {name: dev(x) for name, x in extras.items()}
        self.n_steps += n

    def _extra_column(self, name: str) -> torch.Tensor:
        """One named column over every chunk; the chunks that did not bring it hold its fill value."""
        from step_columns import fill_value

        have = [cols[name] for cols in self.extras.values() if name in cols]
        dtype = have[0].dtype
        for x in have[1:]:
            dtype = torch.promote_types(dtype, x.dtype)
        tail = tuple(have[0].shape[1:])
        parts = []
        for i, act in enumerate(self.act):
            x = self.extras.get(i, {}).get(name)
            if x is not None and tuple(x.shape[1:]) == tail:
                parts.append(x.to(dtype))
            else:
                parts.append(torch.full((int(act.shape[0]),) + tail, fill_value(name), dtype=dtype, device=act.device))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def tensorize(self) -> Tuple:
        if self._tensor_cache: return self._tensor_cache
        # One chunk needs no joining. The learner hands a whole update over as one chunk, and
        # torch.cat of a single tensor would copy it (0.6 GB of observations) for nothing.
        def join(chunks): return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        self._tensor_cache = (
            join(self.obs), join(self.act),
            join(self.logp).float(), join(self.val).float(),
            join(self.adv).float(), join(self.ret).float(),
            join(self.next_hp).float(),
            join(self.ep_ids) # Index 7: Episode Boundaries
        )
        if self.extras:
            # Indices 8 and on: the named columns, sorted by name (extra_names), a row for every step.
            self.extra_names = tuple(sorted({name for cols in self.extras.values() for name in cols}))
            self._tensor_cache = self._tensor_cache + tuple(self._extra_column(name) for name in self.extra_names)
        return self._tensor_cache

    def extra_columns(self) -> Dict[str, torch.Tensor]:
        """The named columns as tensorize() lays them out ({} when there are none)."""
        data = self.tensorize()
        return dict(zip(self.extra_names, data[8:]))

    def on_device(self, device) -> Tuple:
        """The same tensors on `device`, moved once and kept until the dataset is cleared."""
        device = torch.device(device)
        if self._device_cache is None or self._device_cache[0] != device:
            self._device_cache = (device, tuple(d.to(device) for d in self.tensorize()))
        return self._device_cache[1]
    
    def swap_out_tensor_cache(self):
        data = self.tensorize(); self.clear(); return data

    def swap_out_columns(self) -> Tuple[Tuple, Tuple[str, ...]]:
        """swap_out_tensor_cache, with the names of the columns after the eight (extra_names)."""
        data = self.tensorize(); names = self.extra_names; self.clear(); return data, names
        
    def _episode_bounds(self, ep_ids: torch.Tensor) -> list[tuple[int, int]]:
        n = int(ep_ids.numel())
        if n == 0:
            return []

        changes = torch.nonzero(ep_ids[1:] != ep_ids[:-1], as_tuple=False).flatten() + 1
        bounds = torch.cat([
            torch.tensor([0], dtype=torch.long, device=ep_ids.device),
            changes,
            torch.tensor([n], dtype=torch.long, device=ep_ids.device),
        ])
        return [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.numel() - 1)]

    def iter_minibatches(self, mb_size: int, shuffle_episodes: bool = False, device=None) -> Iterator[Tuple]:
        """
        Minibatches of whole episodes in (optionally shuffled) order, cut every mb_size steps.

        device: where to take the rows from. None reads them from where the data lies (the
        CPU, in the learner) and leaves moving each minibatch to the caller. A device moves
        the whole data set there once and indexes it there: the same rows in the same order,
        without a 13 MB copy and transfer per minibatch. The order itself is drawn on the
        CPU either way, from the same random numbers.
        """
        # Sequence-Aware Slicing
        data = self.tensorize()
        ep_ids = data[7]
        n = int(ep_ids.shape[0])
        
        if shuffle_episodes:
            spans = self._episode_bounds(ep_ids)
            perm = torch.randperm(len(spans)).tolist()
            order = torch.cat([
                torch.arange(spans[i][0], spans[i][1], dtype=torch.long)
                for i in perm
            ], dim=0)
        else:
            order = torch.arange(n, dtype=torch.long)
            
        usable = (int(order.numel()) // mb_size) * mb_size
        if usable == 0:
            return
        
        order = order[:usable]
        if device is not None:
            try:
                moved = self.on_device(device)
                moved_order = order.to(moved[0].device)
            except torch.cuda.OutOfMemoryError:
                # The device had room when it was asked and has none now (it is shared with
                # the inference actor). Cut this pass from where the data lies, as was always
                # done before: the caller moves each minibatch over. Same rows, same order.
                self._device_cache = None
                logger.warning("no room on %s for the update's data: cutting the minibatches on the CPU", device)
            else:
                data, order = moved, moved_order
        for start in range(0, usable, mb_size):
            idx = order[start:start + mb_size]
            yield tuple(d[idx] for d in data)


def update_data_fits(dataset: AsyncEpisodeDataset, device, share: float = 0.10) -> bool:
    """
    Whether an update's data may be held on the training GPU: only when it is a small part
    (a tenth) of what this process could still allocate there. The GPU is shared with the
    inference actor and the update's own forward and backward need room that is not known in
    advance, so anything tighter stays on the CPU, where it has always been.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return False
    try:
        need = sum(d.numel() * d.element_size() for d in dataset.tensorize())
        free, _total = torch.cuda.mem_get_info(device)
        # Memory torch holds for this process but has not handed out is as good as free.
        spare = free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    except Exception:
        return False
    return need <= share * spare

@torch.no_grad()
def gae_from_episode(
    rewards: torch.Tensor,     # [T]
    values: torch.Tensor,      # [T]
    dones: torch.Tensor,       # [T] (1 at terminal step)
    gamma: float,
    lam: float,
    last_value: float = 0.0,   # terminal -> 0
    variances: Optional[torch.Tensor] = None, # [T] State-conditional variance
    lam_min: float = 0.55,
    lam_max: float = 0.95,
    lam_eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes GAE for one episode. If variances are provided, computes state-aware 
    dynamic lambda per timestep to balance bias and variance dynamically.
    """
    T = rewards.shape[0]
    adv = torch.zeros((T,), device=rewards.device)
    gae = torch.tensor(0.0, device=rewards.device)

    # ---------------------------------------------------------
    # Branch A: Dynamic Lambda calculation using per-state Variance
    # ---------------------------------------------------------
    if variances is not None:
        # Precompute 1-step TD Errors for the entire episode
        next_values = torch.cat([values[1:], torch.tensor([last_value], device=values.device)])
        next_nonterminal = 1.0 - dones
        
        deltas = rewards + gamma * next_values * next_nonterminal - values
        
        # Vectorized dynamic lambda calculation
        delta_sq = deltas ** 2
        dynamic_lambda = delta_sq / (delta_sq + variances + lam_eps)
        dynamic_lambda = torch.clamp(dynamic_lambda, min=lam_min, max=lam_max)

        if _float32_on_cpu(deltas, dynamic_lambda, next_nonterminal, adv):
            # The same recurrence on NumPy float32 scalars: 1 microsecond a step where the
            # loop below, on torch scalars, takes 30. Every product and sum is the same
            # float32 operation in the same order, so the advantages are the same bits:
            #   gamma * lambda[t] * nonterminal[t]   computed for all t at once, then
            #   gae = delta[t] + that * gae          one step at a time.
            d = deltas.numpy()
            coef = (np.float32(gamma) * dynamic_lambda.numpy()) * next_nonterminal.numpy()
            out = adv.numpy()
            acc = np.float32(0.0)
            for t in range(T - 1, -1, -1):
                acc = d[t] + coef[t] * acc
                out[t] = acc
        else:
            for t in reversed(range(T)):
                gae = deltas[t] + gamma * dynamic_lambda[t] * next_nonterminal[t] * gae
                adv[t] = gae

    # ---------------------------------------------------------
    # Branch B: Standard Static GAE
    # ---------------------------------------------------------
    elif _float32_on_cpu(rewards, values, dones, adv):
        # As above, for the plain recurrence. The loop below multiplies gamma by the value
        # as float32 and gamma by lambda as Python floats before either meets a tensor, and
        # so does this.
        r, v, dn = rewards.numpy(), values.numpy(), dones.numpy()
        g = np.float32(gamma)
        gl = np.float32(gamma * lam)
        one = np.float32(1.0)
        out = adv.numpy()
        acc = np.float32(0.0)
        for t in range(T - 1, -1, -1):
            nonterminal = one - dn[t]
            next_value = np.float32(last_value) if (t == T - 1) else v[t + 1]
            delta = r[t] + g * next_value * nonterminal - v[t]
            acc = delta + gl * nonterminal * acc
            out[t] = acc
    else:
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_value = torch.tensor(last_value, device=rewards.device) if (t == T - 1) else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            adv[t] = gae

    ret = adv + values
    return adv, ret


def _float32_on_cpu(*tensors: torch.Tensor) -> bool:
    """What gae_from_episode's NumPy loops need. Anything else takes the torch loops."""
    return all(t.dtype == torch.float32 and t.device.type == "cpu" and not t.requires_grad for t in tensors)

def masked_smoothed_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Computes cross-entropy loss with label smoothing applied ONLY to legal actions.
    Illegal actions are forced to 0.0 probability in the target distribution.
    """
    # 0. SAFETY VALVE: Guarantee the target action is always unmasked.
    # If the heuristic bot took an action that our obs mask thinks is illegal,
    # we must forcefully unmask it so the loss doesn't explode.
    target_idx = targets.unsqueeze(1).to(torch.int64)
    mask = mask.clone()  # Clone so we don't mutate the raw batch data
    mask.scatter_(1, target_idx, 1.0)
    
    # 1. Force the mask to match the logits dtype to prevent promotion mismatches
    mask = mask.to(logits.dtype)
    
    ml = masked_logits(logits, mask)
    log_probs = F.log_softmax(ml, dim=-1)
    
    with torch.no_grad():
        legal_counts = mask.sum(dim=-1, keepdim=True)
        smooth_prob = smoothing / legal_counts
        target_dist = mask * smooth_prob
        
        # 2. Force confidence_mass to dynamically match the target_dist dtype
        confidence_mass = torch.full_like(targets.unsqueeze(1), 1.0 - smoothing, dtype=logits.dtype)
        
        # 3. Force targets to int64, as scatter_add_ requires index tensors to be LongTensors
        target_idx = targets.unsqueeze(1).to(torch.int64)
        
        target_dist.scatter_add_(1, target_idx, confidence_mass)

    loss = -(target_dist * log_probs).sum(dim=-1).mean()
    return loss

class MasterWeightOptimizer:
    """
    Optimizer wrapper implementing fp32 master weights for a bf16 model.

    The model keeps bf16 parameters for fast forward/backward passes. The
    inner optimizer (AdamW) steps on persistent fp32 master copies, so small
    updates are not rounded away by bf16 (~8 mantissa bits) and Adam moments
    stay in fp32. After each step the masters are copied back into the bf16
    model params. Gradients are also accumulated in fp32 (call
    accumulate_grads() after each backward) so grad accumulation across
    minibatches doesn't lose precision either.

    pairs: list of (model_param, fp32_master) — trainable params only.
    The inner optimizer's param groups must reference the fp32 masters.

    foreach_copy (False unless ppo_update(fast_update=True) sets it): the grad flush and the
    copy-back as a few torch._foreach_* calls instead of one small copy per parameter. The same
    casts and the same single additions, so the same bits; only the number of launches differs.
    """

    foreach_copy: bool = False

    def __init__(self, inner: optim.Optimizer, pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.inner = inner
        self.pairs = pairs

    @property
    def param_groups(self):
        return self.inner.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.inner.zero_grad(set_to_none=set_to_none)
        for model_p, _ in self.pairs:
            model_p.grad = None

    def _accumulate_grads_foreach(self):
        """accumulate_grads with foreach copies and adds (foreach_copy): bf16 -> fp32 is exact, then one add."""
        fresh_m, fresh_g, acc_m, acc_g = [], [], [], []
        for model_p, master_p in self.pairs:
            if model_p.grad is None:
                continue
            if master_p.grad is None:
                fresh_m.append(master_p)
                fresh_g.append(model_p.grad.detach())
            else:
                acc_m.append(master_p.grad)
                acc_g.append(model_p.grad.detach())
            model_p.grad = None
        if fresh_g:
            outs = [torch.empty_like(g, dtype=torch.float32) for g in fresh_g]
            torch._foreach_copy_(outs, fresh_g)
            for m, o in zip(fresh_m, outs):
                m.grad = o
        if acc_g:
            torch._foreach_add_(acc_m, [g.to(torch.float32) for g in acc_g])

    def accumulate_grads(self):
        """Flush bf16 model grads into fp32 master grads, then clear them."""
        if self.foreach_copy:
            self._accumulate_grads_foreach()
            return
        for model_p, master_p in self.pairs:
            if model_p.grad is None:
                continue
            g = model_p.grad.detach().to(torch.float32)
            if master_p.grad is None:
                master_p.grad = g
            else:
                master_p.grad.add_(g)
            model_p.grad = None

    def clip_and_step(self, max_grad_norm: Optional[float] = None):
        """Clip fp32 master grads, step, and round masters back into bf16."""
        masters = [m for _, m in self.pairs if m.grad is not None]
        if max_grad_norm is not None and masters:
            nn.utils.clip_grad_norm_(masters, max_grad_norm)
        self.inner.step()
        with torch.no_grad():
            if self.foreach_copy:
                if self.pairs:
                    torch._foreach_copy_([mp.data for mp, _ in self.pairs], [ms.data for _, ms in self.pairs])
                return
            for model_p, master_p in self.pairs:
                model_p.data.copy_(master_p.data)

    def step(self):
        self.accumulate_grads()
        self.clip_and_step(None)

    def sync_masters_from_model(self):
        """Re-seed masters from model weights (after loading a checkpoint or
        externally mutating model params, e.g. an actor-head reset)."""
        with torch.no_grad():
            for model_p, master_p in self.pairs:
                master_p.data.copy_(model_p.data.to(torch.float32))

    def state_dict(self):
        return {
            "inner": self.inner.state_dict(),
            "masters": [m.detach().cpu() for _, m in self.pairs],
        }

    def load_state_dict(self, sd):
        if "inner" in sd:
            # New-style checkpoint: restore exact fp32 masters.
            self.inner.load_state_dict(sd["inner"])
            with torch.no_grad():
                for (model_p, master_p), saved in zip(self.pairs, sd["masters"]):
                    master_p.data.copy_(saved.to(master_p.device))
                    model_p.data.copy_(master_p.data)
        else:
            # Legacy checkpoint (optimizer stepped bf16 weights directly).
            # torch casts the fp state tensors to the fp32 master dtype on load;
            # masters themselves were already seeded from the model weights.
            self.inner.load_state_dict(sd)


def _clip_and_step(net: nn.Module, opt, max_grad_norm: float):
    if isinstance(opt, MasterWeightOptimizer):
        opt.clip_and_step(max_grad_norm)
    else:
        nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        opt.step()


def set_fast_update_paths(net: nn.Module, opt, on: bool = True) -> None:
    """
    The two module-level switches of fast_update: FlexEncoderLayer.fast_unbind on every encoder
    layer of `net` and MasterWeightOptimizer.foreach_copy on `opt`. Both give the same numbers as
    the paths they replace. ppo_update(fast_update=True) turns them on and leaves them on; a test
    that runs the same objects both ways turns them off with on=False.
    """
    for m in net.modules():
        if isinstance(m, FlexEncoderLayer):
            m.fast_unbind = bool(on)
    if isinstance(opt, MasterWeightOptimizer):
        opt.foreach_copy = bool(on)


def ppo_update(
    net: nn.Module,
    opt: optim.Optimizer,
    dataset: AsyncEpisodeDataset,
    scheduler=None,
    mode: str = "ppo",
    target_net: Optional[nn.Module] = None,
    human: Optional[Any] = None,
    fast_update: bool = False,
    **cfg
) -> PPOUpdateStats:
    """
    Executes a PPO update across multiple epochs and minibatches.

    Includes early stopping based on Target KL to prevent policy collapse.

    human: the top players' imitation term (human_imitation.HumanImitation), or None. With None
    (every run without train.py --human-imitation-coef) nothing below differs, bit for bit. With
    it, the modes trained on the PPO loss add human.term() (coef x the weighted cross-entropy on a
    batch of the top players' decisions) to the loss of every human.every-th minibatch; a failure
    leaves that minibatch's loss as it was and is counted in human.track["err"]. A term that is
    not finite is added as it is (counted as hum_nonfinite): its gradient would be NaN whatever
    its value, so the loss shows it at this update (human_imitation.py keeps such data out at load).

    fast_update (train.py --fast-update; False unless given): the same update without the reads
    that make the CPU wait for the GPU at every minibatch. The losses, gradients, optimizer steps,
    KL early stops and returned figures are the same bits as with False:
      * the figures (kl, clip, ent, v, pg, total) are summed on the GPU as float64 0-d tensors,
        each term x.detach().double() added in the order the Python floats were, and read once at
        the end (a float32 .item() is exact in a Python float, and a double add is a double add);
      * the zero jepa_loss is one tensor made before the loop;
      * the action mask goes through masked_logits_nosync (same logits, no .any()); the rows it
        fixes up are counted on the GPU and logged once per update when there are any;
      * from the second epoch on, each minibatch's approx_kl is copied (non_blocking) into a pinned
        host buffer right after it is computed and a CUDA event is recorded; the backward, the grad
        flush and the step are queued before the code waits on that event alone and adds the
        value to epoch_kl. The same Python double sum, the same test, the same stopping minibatch.
        The first epoch, which has no early stop, reads no KL;
      * set_fast_update_paths: q, k, v by unbind in the encoder layers, foreach grad flush and
        copy-back in the master-weight optimizer.
    What it changes:
      * the "Zero mask detected" warning becomes one count per update;
      * how far the CPU runs ahead of the GPU. Inside the first epoch nothing is read back, so the
        CPU queues work as far ahead as CUDA's launch queue lets it, up to the epoch's end (the
        start of every epoch waits for the GPU: the minibatch order is copied to it). With the
        minibatches cut on the CPU (data_on_device False) each one's copy waits too. From the
        second epoch on it waits for each minibatch's KL, so there it is at most about one
        minibatch ahead;
      * when an error raised on the GPU surfaces. An error inside a minibatch (an index out of
        range in a gather, say) shows at the next read, not in its own minibatch. In the first
        epoch that is the start of the second epoch, or the read after the loop: up to a whole
        epoch of minibatches and their queued optimizer steps later. ppo_update still raises
        before it returns, and after a CUDA error every later GPU call fails, so no checkpoint can
        be written from that state;
      * peak GPU memory, by a few MB (+5 MB measured at the live size).
    The human imitation term keeps its own reads (with it on, those bound the run-ahead too).
    """
    dev = next(net.parameters()).device
    stats = {k: 0.0 for k in ["kl", "clip", "ent", "v", "pg", "total", "acc"]}
    n_mb = 0
    stop_training = False
    # record_first_kl (the learner's --lag-stats): the first minibatch's approx_kl, kept as a tensor and
    # read once after the loop. Not given: nothing of it runs.
    record_first_kl = bool(cfg.pop("record_first_kl", False))
    first_kl: Optional[torch.Tensor] = None

    # The stage 2-4 terms, each on only with its setting and its column (the defaults leave every
    # line below as it was): the boosted-row weight of the policy loss, the opponent-action head,
    # the SetHead. Their sums are 0-d tensors, read once at the end.
    boosted_w = float(cfg.get("boosted_row_weight", 1.0) or 1.0)
    opp_coef = float(cfg.get("opp_action_aux_coef", 0.0) or 0.0)
    set_coef = float(cfg.get("set_head_coef", 0.0) or 0.0)
    use_opp = opp_coef > 0.0 and getattr(net, "opp_action_head", None) is not None
    use_set = set_coef > 0.0 and getattr(net, "set_head", None) is not None
    aux_sums: Dict[str, torch.Tensor] = {}

    def _aux_add(name: str, value: torch.Tensor) -> None:
        v = value.detach().float()
        aux_sums[name] = aux_sums[name] + v if name in aux_sums else v

    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    # The modes that take the PPO branch of the loss below: ppo, ppo_with_jepa,
    # ppo_frozen_backbone and warmup_with_actor_reset.
    uses_ppo_loss = mode not in IMITATION_LOSS_MODES and mode not in ("jepa_pretraining", "warmup")

    # Where the minibatches are cut from: the training device when the update's data is small
    # next to the memory free there ("auto"), or as the caller says (True / False).
    data_on_device = cfg.get("data_on_device", "auto")
    if data_on_device == "auto":
        data_on_device = update_data_fits(dataset, dev)
    data_dev = dev if data_on_device else None

    # fast_update: everything it needs, made once per update (nothing of it exists without it).
    if fast_update:
        set_fast_update_paths(net, opt, True)
        fu_zero = torch.tensor(0.0, device=dev)
        fu_sums = {k: torch.zeros((), dtype=torch.float64, device=dev) for k in ("kl", "clip", "ent", "v", "pg", "total")}
        fu_zero_rows = torch.zeros((), dtype=torch.int64, device=dev)
        # The early stop's KL behind a CUDA event (on a GPU, with a KL target, in the PPO modes).
        fu_kl_event = uses_ppo_loss and bool(cfg.get("target_kl")) and dev.type == "cuda"
        fu_kl_host: Optional[torch.Tensor] = None
        fu_kl_done = torch.cuda.Event() if fu_kl_event else None

    for epoch in range(cfg.get("update_epochs", 1)):
        epoch_kl, epoch_mb = 0.0, 0

        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(dataset.iter_minibatches(cfg["minibatch_size"], shuffle_episodes=True, device=data_dev)):
            mb_obs, mb_act, mb_logp_old, _, mb_adv, mb_ret, _, mb_ep_ids = batch[:8]
            # The named per-step columns of these rows (step_columns.py), by name; {} when the data
            # has none. Nothing reads them yet: they are here for the stages that add their own.
            mb_extras = dict(zip(getattr(dataset, "extra_names", ()), batch[8:]))
            mb_act, mb_logp_old, mb_adv, mb_ret, mb_ep_ids = (t.to(dev) for t in [mb_act, mb_logp_old, mb_adv, mb_ret, mb_ep_ids])

            mb_obs = mb_obs.to(dev, non_blocking=True)
            m_start, m_end = net.unpacker.offsets["action_mask"]
            mb_mask = mb_obs[:, m_start:m_end]

            mb_opp = mb_extras.get("opp_action") if use_opp else None
            mb_set = mb_extras.get("set_labels") if use_set else None
            if mb_opp is not None or mb_set is not None:
                logits, v_logits, _, _, _, z_tokens = net(mb_obs, mb_ep_ids, batch_seq_len=cfg["batch_seq_len"], return_z=True)
            else:
                z_tokens = None
                logits, v_logits, _, _, _ = net(mb_obs,
                                             mb_ep_ids,
                                             batch_seq_len=cfg["batch_seq_len"],
                                             )

            # --- 1. SHARED AUXILIARY JEPA CALCULATION ---
            jepa_loss = fu_zero if fast_update else torch.tensor(0.0, device=dev)
            if "jepa" in mode and target_net is not None:
                Z_pred, _ = net.forward_jepa(mb_obs, mb_act, episode_ids=mb_ep_ids, batch_seq_len=cfg["batch_seq_len"])
                
                with torch.no_grad():
                    f_targ, p_targ, m_targ = target_net.encode_features(mb_obs)
                    Z_target, _, _ = target_net.condition_tokens(
                        f_targ, p_targ, m_targ, episode_ids=mb_ep_ids, batch_seq_len=cfg["batch_seq_len"]
                    )
                
                B = mb_obs.shape[0]
                seq_len = cfg["batch_seq_len"]
                n_seq = B // seq_len
                
                Z_pred_seq = Z_pred.view(n_seq, seq_len, 13, -1)
                Z_target_seq = Z_target.view(n_seq, seq_len, TOKENS_PER_TURN, -1)
                
                active_idx = [0, 1, 7]
                pred_t = Z_pred_seq[:, :-1, active_idx, :]
                targ_t1 = Z_target_seq[:, 1:, active_idx, :]
                
                ep_ids_seq = mb_ep_ids.view(n_seq, seq_len)
                valid_trans = (ep_ids_seq[:, :-1] == ep_ids_seq[:, 1:])
                
                if valid_trans.any():
                    jepa_loss = F.smooth_l1_loss(pred_t[valid_trans], targ_t1[valid_trans])
                else:
                    jepa_loss = torch.tensor(0.0, device=dev, requires_grad=True)

            # --- 2. PRIMARY LOSS ROUTING ---
            if mode in IMITATION_LOSS_MODES:
                pg_loss = masked_smoothed_cross_entropy(logits.float(), mb_act, mb_mask, smoothing=0.1)
                ent_loss = jepa_loss
                approx_kl = clip_frac = torch.tensor(0.0, device=dev)
                # How often the action the policy would play (its best legal one) is the
                # teacher's. Read off the logits this step already has; nothing is trained on it.
                with torch.no_grad():
                    best = logits.detach().float().masked_fill(mb_mask <= 0.5, -1e4).argmax(dim=-1)
                    stats["acc"] += (best == mb_act).float().mean().item()
                
                target_dist = twohot_targets(mb_ret, v_min=cfg["v_min"], v_max=cfg["v_max"], v_bins=cfg["v_bins"])
                v_loss = dist_value_loss(v_logits, target_dist)
                
                loss = pg_loss + (cfg.get("vf_coef", 0.5) * v_loss) + (cfg.get("jepa_coef", 1.0) * jepa_loss)

            elif mode == "jepa_pretraining":
                v_loss = jepa_loss # Route for logging
                pg_loss = ent_loss = approx_kl = clip_frac = torch.tensor(0.0, device=dev)
                loss = jepa_loss
                
            elif mode == "warmup":
                target_dist = twohot_targets(mb_ret, v_min=cfg["v_min"], v_max=cfg["v_max"], v_bins=cfg["v_bins"])
                v_loss = dist_value_loss(v_logits, target_dist)
                loss = v_loss
                pg_loss = ent_loss = approx_kl = clip_frac = torch.tensor(0.0, device=dev)

            else: # "ppo" or "ppo_with_jepa"
                if fast_update:
                    logp_game, ent_game = masked_logprob_entropy(logits.float(), mb_mask, mb_act, nosync=True, zero_rows=fu_zero_rows)
                else:
                    logp_game, ent_game = masked_logprob_entropy(logits.float(), mb_mask, mb_act)
                ratio_game = (logp_game - mb_logp_old).exp()
                mb_boosted = mb_extras.get("boosted") if boosted_w != 1.0 else None
                if mb_boosted is not None:
                    # Rows where the opponent's active Pokemon is boosted weigh boosted_row_weight
                    # against 1, the weights normalised to a mean of 1 in this minibatch.
                    row_w = boosted_row_weights(mb_boosted.to(dev), boosted_w)
                    pg_loss = (torch.max(-mb_adv * ratio_game, -mb_adv * ratio_game.clamp(1-cfg["clip_coef"], 1+cfg["clip_coef"])) * row_w).mean()
                    _aux_add("boosted_share", (mb_boosted.to(dev).reshape(-1) > 0).float().mean())
                else:
                    pg_loss = torch.max(-mb_adv * ratio_game, -mb_adv * ratio_game.clamp(1-cfg["clip_coef"], 1+cfg["clip_coef"])).mean()

                target_dist = twohot_targets(mb_ret, v_min=cfg["v_min"], v_max=cfg["v_max"], v_bins=cfg["v_bins"])
                v_loss = dist_value_loss(v_logits, target_dist)
                ent_loss = ent_game.mean()

                loss = pg_loss + cfg["vf_coef"] * v_loss - (cfg["ent_coef"] * ent_loss) + (cfg.get("jepa_coef", 1.0) * jepa_loss)

                approx_kl = (mb_logp_old - logp_game).mean() 
                clip_frac = ((ratio_game - 1.0).abs() > cfg["clip_coef"]).float().mean()
                if record_first_kl and first_kl is None:
                    first_kl = approx_kl.detach().clone()
                if fast_update and fu_kl_event and epoch > 0:
                    # The early stop's value on its way to the host while the backward is queued.
                    if fu_kl_host is None or fu_kl_host.dtype != approx_kl.dtype:
                        fu_kl_host = torch.empty((), dtype=approx_kl.dtype, pin_memory=True)
                    fu_kl_host.copy_(approx_kl.detach(), non_blocking=True)
                    fu_kl_done.record()
                
            if z_tokens is not None:
                if mb_opp is not None:
                    term, st = compute_opp_action_loss(net, z_tokens, mb_opp, coef=opp_coef)
                    loss = loss + term
                    for k, v in st.items():
                        _aux_add(k, v)
                if mb_set is not None:
                    # 6 x 3 per row, or 6 x 8 with the moveset block's labels
                    term, st = compute_set_head_loss(net, z_tokens, mb_set.reshape(mb_set.shape[0], 6, -1), coef=set_coef)
                    loss = loss + term
                    for k, v in st.items():
                        _aux_add(k, v)

            if human is not None and uses_ppo_loss and human.wants(i):
                # The top players' imitation term (human_imitation.py). Never allowed to stop an
                # update: a failure leaves this minibatch's loss as it was, and is counted.
                try:
                    h_term = human.term(net, dev, mb_obs.dtype, cfg["batch_seq_len"], epoch == 0)
                    if h_term is not None:
                        loss = loss + h_term
                except Exception as e:
                    human.track["err"] += 1
                    logger.warning(f"human imitation term left out of a minibatch: {e!r}")

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            # Accumulate grads in fp32 masters (bf16 accumulation is lossy)
            if isinstance(opt, MasterWeightOptimizer):
                opt.accumulate_grads()

            if (i + 1) % grad_accum_steps == 0:
                _clip_and_step(net, opt, cfg["max_grad_norm"])
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            # Stats update
            n_mb += 1; epoch_mb += 1
            if fast_update:
                # The same float64 sums, on the GPU, in the same order; read once after the loop.
                for k, t in (("kl", approx_kl), ("clip", clip_frac), ("ent", ent_loss),
                             ("v", v_loss), ("pg", pg_loss), ("total", loss)):
                    fu_sums[k] += t.detach().double()
                # Only the early stop below reads epoch_kl, and only from the second epoch on.
                if epoch > 0 and uses_ppo_loss and cfg.get("target_kl"):
                    if fu_kl_event:
                        fu_kl_done.synchronize()
                        epoch_kl += float(fu_kl_host)
                    else:
                        epoch_kl += approx_kl.item()
            else:
                stats["kl"] += approx_kl.item(); epoch_kl += approx_kl.item()
                stats["clip"] += clip_frac.item(); stats["ent"] += ent_loss.item()
                stats["v"] += v_loss.item(); stats["pg"] += pg_loss.item(); stats["total"] += loss.item()

            # Every mode trained on the PPO loss stops early on KL, not only "ppo" (the others
            # have no KL to measure). ppo_with_jepa, the default, used to never stop.
            if epoch > 0 and uses_ppo_loss and cfg.get("target_kl") and (epoch_kl / epoch_mb) > cfg["target_kl"] * 1.5:
                logger.info(f"Early stop at Epoch {epoch} due to KL {epoch_kl/epoch_mb:.4f}")
                stop_training = True
                break
        
        if not stop_training and (i + 1) % grad_accum_steps != 0:
            _clip_and_step(net, opt, cfg["max_grad_norm"])
            opt.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        
        if stop_training:
            break

    if fast_update:
        # One read for the six figures and the zero-mask count.
        keys = list(fu_sums)
        read = torch.cat([torch.stack([fu_sums[k] for k in keys]), fu_zero_rows.double().reshape(1)]).tolist()
        for k, v in zip(keys, read):
            stats[k] = v
        if read[-1] > 0:
            logger.warning(f"Zero mask detected in {int(read[-1])} rows of this update's minibatches. Forced index 0.")

    out = PPOUpdateStats(
        approx_kl=stats["kl"]/max(n_mb, 1), 
        clip_frac=stats["clip"]/max(n_mb, 1), 
        entropy=stats["ent"]/max(n_mb, 1),
        v_loss=stats["v"]/max(n_mb, 1),
        pg_loss=stats["pg"]/max(n_mb, 1),
        total_loss=stats["total"]/max(n_mb, 1),
        n_mb=n_mb,
        acc=(stats["acc"]/max(n_mb, 1)) if mode in IMITATION_LOSS_MODES else None,
        aux=_aux_summary(aux_sums, n_mb) if aux_sums else None,
    )
    if record_first_kl and first_kl is not None:
        out.kl_first = float(first_kl)
    return out


def _aux_summary(sums: Dict[str, torch.Tensor], n_mb: int) -> Dict[str, float]:
    """The stage 2-4 terms of one update, from their sums over its minibatches (one GPU read each)."""
    v = {k: float(t) for k, t in sums.items()}
    out: Dict[str, float] = {}
    if "opp_n" in v:
        out["opp_n"] = v["opp_n"]
        out["opp_acc"] = v["opp_top1"] / max(v["opp_n"], 1.0)
        out["opp_base"] = v.get("opp_moved", 0.0) / max(v["opp_n"], 1.0)     # always "moved"
        out["opp_loss"] = v["opp_loss"] / max(n_mb, 1)
    if "set_loss" in v:
        out["set_loss"] = v["set_loss"] / max(n_mb, 1)
        out["set_ab_n"] = v["set_ability_n"]
        out["set_ab_acc"] = v["set_ability_top1"] / max(v["set_ability_n"], 1.0)
        out["set_it_n"] = v["set_item_n"]
        out["set_it_acc"] = v["set_item_top1"] / max(v["set_item_n"], 1.0)
        if "set_move_n" in v:
            # moves: mean precision at k over the labelled Pokemon (k = its true moves not shown)
            out["set_mv_n"] = v["set_move_n"]
            out["set_mv_acc"] = v["set_move_hits"] / max(v["set_move_n"], 1.0)
            out["set_tera_n"] = v["set_tera_n"]
            out["set_tera_acc"] = v["set_tera_top1"] / max(v["set_tera_n"], 1.0)
    if "boosted_share" in v:
        out["boosted_share"] = v["boosted_share"] / max(n_mb, 1)
    return out