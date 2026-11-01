"""
Players for evaluation and teaching: the PSBot heuristic and the upstream baselines.

Importing this module does not pull in torch or ray.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from poke_env.player import Player
from poke_env.player.baselines import SimpleHeuristicsPlayer

from psbot_bridge import PSBotMixin


def corrected_stat_estimation(mon: Any, stat: str) -> float:
    """
    A mathematically accurate stat estimator for Gen 9 Random Battles.
    Replaces the Level 100 / 0 EV logic in poke-env's SimpleHeuristicsPlayer.
    """
    # 1. FIX THE BOOST MULTIPLIER
    # Original bug checked `> 1` which made +1 act like +0, and +2 act like +2.
    boost_lvl = mon.boosts.get(stat, 0)
    if boost_lvl > 0:
        boost_mult = (2 + boost_lvl) / 2  # e.g., +1 = 1.5x
    else:
        boost_mult = 2 / (2 - boost_lvl)  # e.g., -1 = 0.66x

    # 2. APPLY TRUE GEN 9 RANDBATS MATH
    base = mon.base_stats[stat]
    level = mon.level

    # Gen 9 RandBats assigns a flat 85 EVs to every stat
    ev = 85

    # We assume 31 IVs (ignoring the 0 Speed Trick Room edge case for simplicity)
    iv = 31

    # Standard Pokémon non-HP stat formula
    raw_stat = math.floor(((2 * base + iv + math.floor(ev / 4)) * level) / 100) + 5

    # Apply boosts
    return raw_stat * boost_mult


def patch_simple_heuristics() -> None:
    """Applies the stat fix to SimpleHeuristicsPlayer. Safe to call more than once."""
    SimpleHeuristicsPlayer._stat_estimation = staticmethod(corrected_stat_estimation)


class PSBotPlayer(PSBotMixin, Player):
    """Plays whatever PSBot (psbot/psbot.js) would play."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.psbot_decisions = 0
        self.psbot_fallbacks = 0

    async def choose_move(self, battle: Any):
        self.psbot_decisions += 1
        order = await self.psbot_order(battle)
        if order is None:
            # The engine had no legal answer (or is down). Keep the game moving.
            self.psbot_fallbacks += 1
            return self.choose_random_move(battle)
        return order


# ---------------------------------------------------------------------------
# Auditing: what a win rate alone does not show
# ---------------------------------------------------------------------------

AUDIT_KEYS = (
    "requests",             # times the player was asked for an order
    "invalid_choices",      # orders the server rejected with [Invalid choice]: the player's mistake
    "unavailable_choices",  # orders rejected with [Unavailable choice]: the server revealing something the
                            # player could not know (trapped by an ability, a move disabled). Not a mistake.
    "other_errors",         # any other |error| line in a battle room
    "exceptions",           # choose_move raised; a random legal order was sent instead
    "forced_random",        # the same request was rejected repeatedly; a random order was forced
)


class ChoiceAuditMixin:
    """
    Counts the things that make a win rate lie, and keeps a game moving when its player breaks.

    Put it before the player class:  class Audited(ChoiceAuditMixin, RandomPlayer): ...
    or use audited(RandomPlayer).

    Counted in self.audit (see AUDIT_KEYS):
      * orders the server rejected as invalid. poke-env answers that by asking choose_move again,
        and a player that always gives the same answer would loop on it. After AUDIT_MAX_RETRIES
        rejections of one request a random legal order is sent instead, and counted.
      * orders the server rejected as unavailable. Those are normal play: any player learns that
        its Pokemon is trapped only by trying to switch. Counted apart, and not as a fault.
      * exceptions inside choose_move. poke-env would leave the battle waiting forever. Here a
        random legal order is sent instead, and counted.
    Recorded in self.audit_endings (battle tag to "forfeit" or "inactivity"): battles the server
    ended without one side running out of Pokemon. Such a game says nothing about strength.
    """

    AUDIT_MAX_RETRIES = 2

    def __init__(self, *args: Any, **kwargs: Any):
        # Set up before the base class: Player.__init__ starts listening, and the first battle
        # message can arrive before it returns.
        self.audit: Dict[str, int] = {k: 0 for k in AUDIT_KEYS}
        self.audit_endings: Dict[str, str] = {}
        self.audit_messages: List[str] = []      # the last few error texts, for the log
        self._audit_retries: Dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def _audit_note(self, text: str) -> None:
        self.audit_messages = (self.audit_messages + [str(text)[:200]])[-8:]

    async def _handle_battle_message(self, split_messages: List[List[str]]):
        if split_messages and split_messages[0] and split_messages[0][0].startswith(">"):
            room = split_messages[0][0][1:].strip()
            for parts in split_messages[1:]:
                if len(parts) < 3:
                    continue
                if parts[1] == "error":
                    text = "|".join(parts[2:])
                    if text.startswith("[Unavailable choice]"):
                        self.audit["unavailable_choices"] += 1
                        continue
                    if text.startswith("[Invalid choice]"):
                        self.audit["invalid_choices"] += 1
                    else:
                        self.audit["other_errors"] += 1
                    self._audit_note(f"{room}: {text}")
                elif parts[1] in ("-message", "message"):
                    text = parts[2]
                    if text.endswith(" forfeited."):
                        self.audit_endings[room] = "forfeit"
                    elif "lost due to inactivity" in text:
                        self.audit_endings[room] = "inactivity"
        await super()._handle_battle_message(split_messages)

    async def _handle_battle_request(self, battle: Any, maybe_default_order: bool = False):
        tag = battle.battle_tag
        if not maybe_default_order:
            self._audit_retries.pop(tag, None)
        elif not battle.teampreview and not getattr(battle, "_wait", False):
            n = self._audit_retries[tag] = self._audit_retries.get(tag, 0) + 1
            if n > self.AUDIT_MAX_RETRIES:
                # Still rejected: stop asking the chooser. Random legal orders first, and if even
                # those bounce, the server's own default.
                self.audit["forced_random"] += 1
                order = Player.choose_random_move(battle) if n <= self.AUDIT_MAX_RETRIES + 3 else Player.choose_default_move()
                await self.ps_client.send_message(order.message, tag)
                return
        await super()._handle_battle_request(battle, maybe_default_order)

    def choose_move(self, battle: Any):
        self.audit["requests"] += 1
        try:
            res = super().choose_move(battle)
        except Exception as e:
            return self._audit_failed(battle, e)
        if inspect.isawaitable(res):
            return self._audit_await(res, battle)
        return res

    async def _audit_await(self, awaitable: Any, battle: Any):
        try:
            return await awaitable
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return self._audit_failed(battle, e)

    def _audit_failed(self, battle: Any, exc: BaseException):
        self.audit["exceptions"] += 1
        self._audit_note(f"{battle.battle_tag}: choose_move raised {exc!r}")
        return Player.choose_random_move(battle)


_AUDITED: Dict[type, type] = {}


def audited(cls: type) -> type:
    """The given player class with ChoiceAuditMixin in front. The same class back every time."""
    if issubclass(cls, ChoiceAuditMixin):
        return cls
    if cls not in _AUDITED:
        _AUDITED[cls] = type("Audited" + cls.__name__, (ChoiceAuditMixin, cls), {})
    return _AUDITED[cls]


# ---------------------------------------------------------------------------
# The policy alone, with batched inference
# ---------------------------------------------------------------------------

_BATCHED_POLICY_PLAYER = None


def batched_policy_player() -> type:
    """
    The class of the transformer-alone player that answers all waiting battles in one forward pass.

    It is hybrid.HybridPlayer with beta 0 and no veto, which is what play.py calls "policy", and it
    picks the same action from the same numbers: same observation, same network, same per-battle
    history, same rule (hybrid.combine). The only difference is that HybridPlayer runs the network
    once per decision, 25 to 35 ms each on an RTX 4060, and this one runs it once for every battle
    that is waiting. The network already takes a batch with a different history length per row:
    inference.py feeds it that way during training.

    Built on first use so that importing players.py still does not import torch.
    """
    global _BATCHED_POLICY_PLAYER
    if _BATCHED_POLICY_PLAYER is not None:
        return _BATCHED_POLICY_PLAYER

    import numpy as np
    import torch

    from hybrid import HybridPlayer, choice_mask, combine

    class BatchedPolicyPlayer(HybridPlayer):
        # One forward pass costs about the same for 1 battle as for 30, so it pays to wait a moment
        # for the other battles' requests: run when none has arrived for batch_gap seconds, and
        # at the latest batch_wait seconds after the first.
        batch_gap = 0.005
        batch_wait = 0.030
        max_batch = 256

        def __init__(self, *args: Any, **kwargs: Any):
            self._pending: List[Any] = []
            self._flush_handle: Any = None
            self._first_at = 0.0
            self.batch_stats = {"forwards": 0, "rows": 0, "largest": 0}
            # battle tag -> (v_exp, v_std) of the value head in the forward pass that made its last
            # logits: the network's own score of the position, read by the search player's records
            # (search_player.value_fields). Taken from the same pass, never a second one.
            self.net_values: "OrderedDict[str, Tuple[float, Optional[float]]]" = OrderedDict()
            super().__init__(*args, **kwargs)
            if self.policy is None:
                # HybridPlayer plays as PSBot when it finds no network. This player must not.
                raise FileNotFoundError("no checkpoint to load for the policy player")
            if self.psbot_enabled:
                raise ValueError("BatchedPolicyPlayer is the policy alone: beta 0 and no veto")

        # -- batching ------------------------------------------------------

        def _queue_inference(self, tag: str, obs: Any) -> "asyncio.Future":
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            now = loop.time()
            if not self._pending:
                self._first_at = now
            self._pending.append((tag, obs, fut))
            if len(self._pending) >= self.max_batch:
                self._flush()
            else:
                if self._flush_handle is not None:
                    self._flush_handle.cancel()
                delay = max(0.0, min(self.batch_gap, self._first_at + self.batch_wait - now))
                self._flush_handle = loop.call_later(delay, self._flush)
            return fut

        def _flush(self) -> None:
            if self._flush_handle is not None:
                self._flush_handle.cancel()
                self._flush_handle = None
            pending, self._pending = self._pending, []
            if not pending:
                return
            # A battle's history advances by one step per forward pass, so one row per battle.
            batch, later, seen = [], [], set()
            for item in pending:
                (later if item[0] in seen else batch).append(item)
                seen.add(item[0])
            try:
                rows = self._forward_batch([t for t, _o, _f in batch], np.stack([o for _t, o, _f in batch]))
                for (_tag, _obs, fut), row in zip(batch, rows):
                    if not fut.done():
                        fut.set_result(row)
            except Exception as e:  # the waiting choose_move calls get it, and the audit counts it
                for _tag, _obs, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
            if later:
                self._pending = later + self._pending
                if self._flush_handle is None:
                    loop = asyncio.get_running_loop()
                    self._first_at = loop.time()
                    self._flush_handle = loop.call_soon(self._flush)

        @torch.no_grad()
        def _forward_batch(self, tags: List[str], obs: Any) -> Any:
            """Logits for one observation per battle. Histories live in PolicyNet's own cache."""
            pol = self.policy
            net = pol.net
            x = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).to(pol.device)
            layers: List[List[Any]] = [[] for _ in range(net.n_layers)]
            lens = []
            for tag in tags:
                kv, kv_len = pol._cache.get(tag, (None, None))
                if kv is None:
                    kv = [torch.zeros(1, net.kv_cache_len, net.d_model, device=pol.device) for _ in range(net.n_layers)]
                    kv_len = torch.zeros(1, dtype=torch.long, device=pol.device)
                for i in range(net.n_layers):
                    layers[i].append(kv[i])
                lens.append(kv_len.reshape(1).to(pol.device))
            kv_in, len_in = [torch.cat(rows, dim=0) for rows in layers], torch.cat(lens, dim=0)
            viz = getattr(self, "net_viz", None)    # net_viz.NetViz for the window app's view, else None
            if viz is not None and viz.wanted(tags):
                out, v_logits, v_exp, new_kv, new_len = viz.forward(net, tags, x, kv_in, len_in)
            else:
                out, v_logits, v_exp, new_kv, new_len = net(x, kv_cache=kv_in, kv_cache_len=len_in)
            for b, tag in enumerate(tags):
                # The same layout PolicyNet.logits stores, so PolicyNet.forget() clears it too.
                pol._cache[tag] = ([layer[b:b + 1].clone() for layer in new_kv], new_len[b:b + 1].clone())
            self._keep_values(tags, net, v_logits, v_exp)
            self.batch_stats["forwards"] += 1
            self.batch_stats["rows"] += len(tags)
            self.batch_stats["largest"] = max(self.batch_stats["largest"], len(tags))
            return out.float().cpu().numpy()

        def _keep_values(self, tags: List[str], net: Any, v_logits: Any, v_exp: Any) -> None:
            """
            net_values[tag] = (v_exp, v_std) for every row: v_exp is the network's own expected value
            (ppo_core.readout_policy: softmax(v_logits) . v_support, what the learner reads), v_std its
            spread over the same support (as inference.py computes v_var). Never raises.
            """
            try:
                exp = v_exp.float().reshape(-1)
                support = getattr(net, "v_support", None)
                if support is not None and v_logits is not None:
                    probs = torch.softmax(v_logits.float(), dim=-1)
                    var = (probs * (support.float().to(probs.device) - exp.unsqueeze(-1)) ** 2).sum(dim=-1)
                    both = torch.stack([exp, var.clamp_min(0.0).sqrt()], dim=-1).cpu().numpy()
                else:
                    both = torch.stack([exp, torch.full_like(exp, float("nan"))], dim=-1).cpu().numpy()
                for b, tag in enumerate(tags):
                    std = float(both[b, 1])
                    self.net_values[tag] = (float(both[b, 0]), None if std != std else std)
                    self.net_values.move_to_end(tag)
                while len(self.net_values) > 4000:
                    self.net_values.popitem(last=False)
            except Exception:       # a score for the record must never cost a move
                pass

        def _battle_finished_callback(self, battle: Any):
            self.net_values.pop(getattr(battle, "battle_tag", None), None)
            return super()._battle_finished_callback(battle)

        # -- play ----------------------------------------------------------

        async def choose_move(self, battle: Any):
            self.counts["decisions"] += 1
            self.counts["heuristic_missing"] += 1
            tag = battle.battle_tag
            obs = self.assembler.assemble(battle, events=self._events.pop(tag, []))
            logits = await self._queue_inference(tag, obs)
            # The observation's action mask (obs_v2's leaves out a switch the server refuses here).
            mask = choice_mask(self.assembler, battle, obs)
            action = combine(logits, None, mask, self.hybrid, None if self.hybrid.greedy else self.rng)
            target, kwargs = self.assembler.map_index_to_order(action, battle)
            if target == "DEFAULT":
                # A forced state the 14 actions do not cover.
                self.counts["fallbacks"] += 1
                return self.choose_random_move(battle)
            return self.create_order(target, **kwargs)

        def hybrid_stats(self) -> str:
            s = self.batch_stats
            return super().hybrid_stats() + f", {s['forwards']} forward passes of {s['rows'] / max(s['forwards'], 1):.1f} battles on average (largest {s['largest']})"

    _BATCHED_POLICY_PLAYER = BatchedPolicyPlayer
    return _BATCHED_POLICY_PLAYER
