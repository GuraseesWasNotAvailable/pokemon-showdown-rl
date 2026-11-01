"""
The search player: the transformer policy and a look-ahead search deciding together.

    from search_player import search_player, MixSettings, ArmPicker
    import foulplay_bridge as fb

    bridge = fb.FoulPlayBridge(helpers=4, search_ms=500, timeout=4.0).start()
    Player = search_player()                       # built on players.batched_policy_player
    p = Player(hybrid=hcfg, mode="mix", bridge=bridge, search_ms=500, deadline=2.5, ...)

Every decision the network answers as it does today (same observation, same forward pass, same
per-battle history). For a battle whose mode is not "net" the same request is also put to the
search: poke-engine's Monte Carlo tree search over sampled opponent teams, reached through
foulplay_bridge (bench/FOULPLAY_BRIDGE.md). The search looks a few turns ahead over both
sides' options, switches and Terastallization included, and knows our Tera type, which the
network does not see (bench/notes/tera.md).

Modes, one per battle (mode=, or mode_picker= for a different one per game):

    net            the network's best legal action: exactly what players.batched_policy_player
                   plays. Nothing is sent to the search for such a battle.
    search         the action with the largest share of the search.
    mix            the two combined by mix() with MixSettings.rule ("veto" unless told otherwise).
    mix-veto, mix-pool, mix-shortlist     the same with that rule, whatever the settings say.
    predict        mix, and besides a prediction of the opponent's next action (opponent_model.py:
                   the search's view of its side combined with what people did in recorded
                   games). The mix choice is left only when the prediction is confident, another
                   action is clearly worth more against it in the payoff table of the search, and
                   that action does not lose much if the read is wrong (opponent_model.choose).
                   What the opponent then did is read from the battle and counted per game.

The three rules (mix() says exactly how):

    veto       the network chooses among the actions the search did not clearly turn away from
               (those it visited at least a quarter as often as its favourite). The search is
               good at tactics and bad at telling apart positions with the same material, so it
               only removes what it found to lose something; the network's long view picks among
               the rest.
    pool       a weighted sum of the two log-probabilities, with a floor under each.
    shortlist  the reverse of veto: the search chooses among the network's few likeliest moves.

Terastallization in every rule: the network's probability for a move is the sum over its plain
and its Tera form, and the search decides between the two forms.

It never costs a game. The answer of the search is waited for until `deadline` seconds after
the request and no longer. No answer, an answer that names no legal action, a search too small
to mean anything, a bridge that is down: the network's move is played and the reason counted.
Whatever is sent, the search is told (foulplay_note), so what it infers about the opponent
stays right.

Every decision is kept as a plain dict (decisions_of(tag)): what each side wanted, what was
played, who was overruled, the search's iterations and the time. game_summary(tag) adds them up
for one battle, which is what ladder.py writes next to the game's result.

Importing this module does not pull in torch: the player class is built on first use.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import foulplay_bridge as fb

__all__ = [
    "ACTIONS", "MODES", "RULES", "FALLBACK_REASONS", "MixSettings", "SearchView", "ArmPicker",
    "split_mode", "check_modes", "legal_probabilities", "merged_prior", "read_advice", "mix",
    "base_action", "minimum_deadline", "search_player",
]

ACTIONS = 14
RULES = ("veto", "pool", "shortlist")
MODES = ("net", "search", "mix") + tuple("mix-" + rule for rule in RULES) + ("predict",)

# Why a decision of a searched battle was the network's alone.
FALLBACK_REASONS = (
    "single",      # one legal action: nothing to decide, the search is not asked
    "revive",      # the request asks whom Revival Blessing revives: the search has no answer to it
    "timeout",     # no answer by the deadline
    "expired",     # the helper did not start a search that could not have ended by the deadline
    "error",       # the bridge or its helper failed, or the battle is one the search could not follow
    "down",        # the battle's helper is not up (starting again, frozen, or its slot left down): not asked
    "rejoined",    # the battle was joined again after a lost connection: the search cannot pick it up midway
    "illegal",     # the answer names no legal action of ours, or its favourite is not one
    "narrow",      # the search saw one action only where the request offers several: it has no opinion
    "weak",        # fewer iterations than MixSettings.min_iterations
    "off",         # the player was given no bridge
)
RESERVE_S = 0.05   # of the deadline, kept for turning the answer into an order and sending it
KEPT_BATTLES = 4000


@dataclass(frozen=True)
class MixSettings:
    """How the network and the search are combined. Every number here is a first guess, not a tuned value."""
    rule: str = "veto"
    # veto: an action is dropped when its share of the search is less than `keep` times the
    # largest share. 0 is the network alone, 1 leaves only the search's favourite (and whatever
    # ties with it). The search spends its visits where the results are best, so a quarter of the
    # favourite's visits means "looked at properly and found worse", not "about as good".
    # Shares and not scores: a score is pooled over the sampled worlds, each on its own scale,
    # so the scores of two actions that different worlds preferred cannot be compared.
    keep: float = 0.25
    # pool: net_weight * log p(network) + search_weight * log share(search), each under a floor.
    net_weight: float = 1.0
    search_weight: float = 1.0
    net_floor: float = 1e-3
    search_floor: float = 1e-2
    # shortlist: moves with at least `ratio` of the network's largest probability, `top` at most.
    ratio: float = 0.10
    top: int = 3
    # every rule: Tera with the chosen move only when the search puts more on its Tera form than
    # on its plain form, and at least this share (a main line of the search, not noise).
    tera_share: float = 0.20
    # A search with fewer iterations than this is not used at all (the network plays).
    min_iterations: int = 10_000

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError("rule must be one of %s, not %r" % (", ".join(RULES), self.rule))


@dataclass
class SearchView:
    """What the search said, on our 14 actions."""
    share: List[float]               # mean over the sampled worlds of visits / iterations
    visits: List[int]
    score: List[Optional[float]]     # mean playout result from our side, 0 to 1; None without a visit
    best: int                        # the legal action with the largest share
    iterations: int


def split_mode(mode: str) -> Tuple[str, Optional[str]]:
    """("net" | "search" | "mix" | "predict", rule or None). "mix" alone and "predict" leave the rule to the settings."""
    if mode in ("net", "search", "predict"):
        return mode, None
    if mode == "mix":
        return "mix", None
    if mode.startswith("mix-") and mode[4:] in RULES:
        return "mix", mode[4:]
    raise ValueError("mode must be one of %s, not %r" % (", ".join(MODES), mode))


def check_modes(modes: Sequence[str]) -> List[str]:
    out = [str(m).strip() for m in modes if str(m).strip()]
    if not out:
        raise ValueError("no mode given")
    for mode in out:
        split_mode(mode)
    return out


def minimum_deadline(search_ms: int) -> float:
    """
    The shortest deadline with which a search of search_ms is ever started: the helper refuses a
    question it could not answer in time, and it reckons 2.5 x search_ms + 20 ms for an answer
    (fp_bridge_server.search_seconds). A third of a second on top for the wait in front of it.
    """
    return 2.5 * max(2, int(search_ms)) / 1000.0 + 0.02 + RESERVE_S + 0.3


def base_action(action: int) -> int:
    """The move or the switch, whether or not it comes with Terastallization."""
    return action - 10 if action >= 10 else action


def legal_probabilities(logits: Sequence[float], mask: Sequence[float]) -> List[float]:
    """The policy's probabilities over the legal actions (0 elsewhere): what hybrid.combine ranks."""
    legal = [i for i in range(ACTIONS) if mask[i] > 0.5]
    out = [0.0] * ACTIONS
    if not legal:
        return out
    top = max(float(logits[i]) for i in legal)
    weights = {i: math.exp(float(logits[i]) - top) for i in legal}
    total = sum(weights.values())
    for i, w in weights.items():
        out[i] = w / total
    return out


def merged_prior(probs: Sequence[float], mask: Sequence[float]) -> List[float]:
    """
    The network's opinion on each MOVE, whatever it thinks of Terastallizing with it: a move and
    its Tera form both get the sum of the two. The network's logits for the Tera actions are
    formed without seeing the Tera type, so how it splits a move between the two forms is a
    habit, not a judgement (bench/notes/tera.md).
    """
    q = [float(p) for p in probs]
    for k in range(4):
        if mask[k] > 0.5 and mask[10 + k] > 0.5:
            q[k] = q[10 + k] = float(probs[k]) + float(probs[10 + k])
    return q


def read_advice(advice: Any, mask: Sequence[float], min_iterations: int = 0) -> Tuple[Optional[SearchView], str]:
    """
    (view, "") when the search's answer can be used, else (None, reason) with a reason from
    FALLBACK_REASONS. `advice` is a foulplay_bridge.Advice after fill_actions().
    """
    if advice is None:
        return None, "error"
    share, visits, score = list(advice.policy or []), list(advice.visits or []), list(advice.score or [])
    if not (len(share) == len(visits) == len(score) == ACTIONS):
        return None, "illegal"
    seen = [a for a in range(ACTIONS) if mask[a] > 0.5 and visits[a] > 0]
    if not seen:
        return None, "illegal"          # nothing the search looked at is a legal action of ours
    best = max(seen, key=lambda a: (share[a], visits[a], -a))
    # The search's own favourite is something the request does not offer (a switch while we are
    # trapped, a move the server has disabled): it searched a position that is not the one we
    # are in, so none of its numbers are used.
    elsewhere = max((float(getattr(o, "policy", 0.0) or 0.0) for o in (getattr(advice, "unmapped", None) or [])), default=0.0)
    if elsewhere > share[best]:
        return None, "illegal"
    # One option only on our side, where the request offers several: the search believes we are
    # locked into that move (seen with Solar Beam under sun: four moves offered, one searched,
    # 1,000 iterations per world in 12 ms). Its "choice" is no opinion on what to play.
    if len(seen) == 1 and sum(1 for a in range(ACTIONS) if mask[a] > 0.5) > 1:
        return None, "narrow"
    iterations = int(advice.iterations or 0)
    if iterations < min_iterations:
        return None, "weak"
    return SearchView([float(x) for x in share], [int(x) for x in visits], score, best, iterations), ""


def _candidates(q: Sequence[float], legal: List[int], view: SearchView, s: MixSettings) -> Tuple[List[int], Callable[[int], Tuple]]:
    """The actions a rule chooses among, and what it ranks them by (largest wins)."""
    if s.rule == "pool":
        def pooled(a: int) -> Tuple:
            value = (s.net_weight * math.log(max(q[a], s.net_floor))
                     + s.search_weight * math.log(max(view.share[a], s.search_floor)))
            return (value, view.share[a], -a)
        return list(legal), pooled

    if s.rule == "shortlist":
        top_q = max(q[a] for a in legal)
        order = sorted(legal, key=lambda a: (-q[a], a))
        kept: List[int] = []
        bases: List[int] = []
        for a in order:
            if q[a] < s.ratio * top_q:
                break
            if base_action(a) not in bases:
                if len(bases) >= max(1, s.top):
                    continue
                bases.append(base_action(a))
            kept.append(a)
        if all(view.visits[a] <= 0 for a in kept):
            # The search looked at none of them: it has nothing to say here.
            return kept, lambda a: (q[a], -a)
        return kept, lambda a: (view.share[a], q[a], -a)

    # veto
    least = s.keep * view.share[view.best]

    def sound(a: int) -> bool:
        if view.visits[a] <= 0:
            return True                 # the search did not look at it (maybeTrapped, for one): no opinion, no veto
        return view.share[a] >= least

    kept = [a for a in legal if sound(a)]
    if view.best not in kept:
        kept.append(view.best)
    return kept, lambda a: (q[a], view.share[a], -a)


def mix(probs: Sequence[float], mask: Sequence[float], view: SearchView, settings: Optional[MixSettings] = None) -> int:
    """
    The action played from the network's probabilities and the search's statistics.

    probs   the network's probabilities over the legal actions (legal_probabilities).
    mask    the 14 legal-action flags.
    view    read_advice() of the search's answer.

    veto       Every legal action with at least `keep` times the largest share of the search
               is "sound". Among the sound ones the network's likeliest is played. So the
               network's own move stands unless the search clearly turned away from it.
    pool       The largest net_weight * log max(p, net_floor) + search_weight * log max(share,
               search_floor).
    shortlist  Of the network's likeliest moves (at least `ratio` of its largest probability,
               `top` moves at most) the one with the largest share of the search.

    In all three the network's probability of a move is that of its plain and Tera forms
    together (merged_prior), and when both forms are among the candidates the search decides:
    with Tera when the Tera form has the larger share and at least `tera_share`, else without.
    """
    s = settings or MixSettings()
    legal = [a for a in range(ACTIONS) if mask[a] > 0.5]
    if not legal:
        return 0
    q = merged_prior(probs, mask)
    candidates, rank = _candidates(q, legal, view, s)
    choice = max(candidates, key=rank)
    if choice <= 3 or choice >= 10:
        plain, tera = base_action(choice), base_action(choice) + 10
        if plain in candidates and tera in candidates:
            wants_tera = view.share[tera] > view.share[plain] and view.share[tera] >= s.tera_share
            choice = tera if wants_tera else plain
    return choice


class ArmPicker:
    """
    Gives every new game one of the listed modes, balanced: always one of those played least so
    far, drawn at random among them. A game keeps the mode it was given.

    counts   games already played per mode (from the results file), so that a run that is started
             again keeps the arms level over the whole experiment.
    """

    def __init__(self, arms: Sequence[str], counts: Optional[Dict[str, int]] = None, rng: Optional[random.Random] = None):
        self.arms = check_modes(arms)
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("an arm is listed twice: %s" % ",".join(self.arms))
        self.counts: Dict[str, int] = {arm: int((counts or {}).get(arm, 0)) for arm in self.arms}
        self.rng = rng or random.Random()
        self.given: "OrderedDict[str, str]" = OrderedDict()

    def __call__(self, tag: str) -> str:
        arm = self.given.get(tag)
        if arm is None:
            fewest = min(self.counts.values())
            arm = self.rng.choice([a for a in self.arms if self.counts[a] == fewest])
            self.counts[arm] += 1
            self.given[tag] = arm
            while len(self.given) > KEPT_BATTLES:
                self.given.popitem(last=False)
        return arm


def _new_game(mode: str) -> Dict[str, Any]:
    return {
        "mode": mode, "decisions": 0, "asked": 0, "searched": 0, "fallbacks": {},
        "disagreed": 0, "net_overruled": 0, "search_overruled": 0, "tera_added": 0, "tera_dropped": 0,
        "iterations": 0, "search_ms": 0.0, "decision_ms": 0.0, "slowest_ms": 0.0, "late": 0,
        "tera": False, "tera_turn": None, "tera_type": None, "tera_by": None, "tera_net": None, "tera_search": None,
        "tera_order": None,
        # "predict" only: predictions of the opponent's action, and how they turned out
        "pred_made": 0, "pred_scored": 0, "pred_top1_right": 0, "pred_p_true": 0.0,
        "pred_reads": 0, "pred_reads_scored": 0, "pred_reads_right": 0,
        "pred_changed": 0, "pred_changed_scored": 0, "pred_changed_right": 0, "pred_kept": {},
    }


PRED_COUNTS = ("pred_made", "pred_scored", "pred_top1_right", "pred_p_true", "pred_reads", "pred_reads_scored", "pred_reads_right",
               "pred_changed", "pred_changed_scored", "pred_changed_right")


# ---------------------------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------------------------

_SEARCH_PLAYER = None


def search_player() -> type:
    """The class of the search player. Built on first use, so that importing this module does not import torch."""
    global _SEARCH_PLAYER
    if _SEARCH_PLAYER is not None:
        return _SEARCH_PLAYER

    import numpy as np
    from poke_env.player.battle_order import _EmptyBattleOrder

    from hybrid import combine
    from players import batched_policy_player

    Base = batched_policy_player()

    class SearchPlayer(fb.FoulPlayMixin, Base):
        """
        players.batched_policy_player with a search beside it.

        mode         "net", "search", "mix", "mix-veto", "mix-pool" or "mix-shortlist": for every battle.
        mode_picker  a function battle tag -> mode, asked once per battle (ArmPicker). Overrides mode.
        bridge       a started foulplay_bridge.FoulPlayBridge. Without one every battle is played
                     by the network alone, and the decisions of searched battles count as "off".
        search_ms    the search time per decision given to the bridge (None: the bridge's own).
        deadline     seconds from the request to the order. The search's answer is waited for
                     until then and no longer.
        mix          MixSettings.
        search_more  further arguments for FoulPlayBridge.advice_future (threads, iterations, payoff_ms).
        on_decision  called with every decision's record (a dict), after the order is decided.
        predict      opponent_model.PredictSettings for the "predict" mode (its thresholds, the payoff budget).
        opponent_model   an opponent_model.OpponentModel (default: the fitted file next to the module).
        on_outcome   "predict" only: called after a turn with what was predicted and what the opponent did.
        warm_up      one forward pass of the network at construction (see warm_up()). On by default.
        """

        def __init__(self, *args: Any, mode: str = "net", mode_picker: Optional[Callable[[str], str]] = None,
                     bridge: Any = None, search_ms: Optional[int] = None, deadline: float = 2.5,
                     mix: Optional[MixSettings] = None, search_more: Optional[Dict[str, Any]] = None,
                     on_decision: Optional[Callable[[Dict[str, Any]], None]] = None, warm_up: bool = True,
                     predict: Any = None, opponent_model: Any = None,
                     on_outcome: Optional[Callable[[Dict[str, Any]], None]] = None,
                     **kwargs: Any):
            split_mode(mode)
            # Before the base class: Player.__init__ starts listening, and a battle message can
            # arrive before it returns.
            self.search_mode = mode
            self.mode_picker = mode_picker
            self.search_ms = int(search_ms) if search_ms else None
            self.deadline = float(deadline)
            self.mix_settings = mix or MixSettings()
            self.search_more = dict(search_more or {})
            self.on_decision = on_decision
            self.on_outcome = on_outcome
            self.predict_settings = predict
            self.opponent_model = opponent_model
            self._trackers: "OrderedDict[str, Any]" = OrderedDict()          # battle tag -> opponent_model.OpponentTracker
            self._predictions: "OrderedDict[str, Dict[int, Dict[str, Any]]]" = OrderedDict()   # battle tag -> turn -> what was predicted
            self._failed_reads: "OrderedDict[str, set]" = OrderedDict()      # battle tag -> {(its Pokemon, the read that was wrong)}
            self.foulplay_enabled = bridge is not None
            if bridge is not None:
                self.foulplay = bridge
                ms = self.search_ms or int(getattr(bridge, "search_ms", 0) or 0)
                if ms and self.deadline < minimum_deadline(ms):
                    raise ValueError("a search of %d ms is never started with a deadline of %.2f s: the deadline must be at least %.2f s"
                                     % (ms, self.deadline, minimum_deadline(ms)))
                if mode == "predict" or mode_picker is not None:
                    try:
                        self._predict_ready()       # now, not in the first game: it reads the game data
                    except Exception:
                        if mode == "predict":
                            raise                   # with arms, a battle given "predict" says so itself (its predictions fail, mix plays)
                if mode == "predict":
                    least = minimum_deadline(ms) + float(self.predict_settings.payoff_ms) / 1000.0
                    if ms and self.deadline < least:
                        raise ValueError("a search of %d ms with a payoff table of %.0f ms is never started with a deadline of %.2f s: "
                                         "the deadline must be at least %.2f s" % (ms, self.predict_settings.payoff_ms, self.deadline, least))
            self._modes: "OrderedDict[str, str]" = OrderedDict()
            self._games: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
            self._decisions: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
            self._noted: "OrderedDict[str, Any]" = OrderedDict()      # battle tag -> the order message choose_move noted last
            self._rejoined: set = set()                               # battles joined again after a lost connection
            # How late the event loop itself runs (see _watch_loop): a deadline is kept by a timer
            # of the loop, so a loop that is not run for seconds makes every waiting decision late.
            self.loop_lag: Dict[str, Any] = {"checks": 0, "max_ms": 0.0, "over_500_ms": 0, "late_ms_total": 0.0}
            self._lag_task: Any = None
            self._last_decision = 0.0
            self.search_totals: Dict[str, Any] = {
                "decisions": 0, "asked": 0, "searched": 0, "fallbacks": {}, "disagreed": 0, "net_overruled": 0,
                "search_overruled": 0, "iterations": 0, "search_ms": 0.0, "decision_ms": 0.0, "slowest_ms": 0.0,
                "late": 0, "after_end": 0, "exceptions": 0, "noted_at_send": 0, "tera_added": 0, "tera_dropped": 0,
                "by_mode": {}, "predict_errors": 0, "trackers_broken": 0, "pred_kept": {},
            }
            for name in PRED_COUNTS:
                self.search_totals[name] = 0
            super().__init__(*args, **kwargs)
            self._watch_orders()
            self.warm_up_ms: Optional[float] = None
            if warm_up:
                self.warm_up()

        def warm_up(self) -> None:
            """
            One forward pass before the first game. The first one of a process takes seconds (4.4 s
            measured on the RTX 4060: the GPU library starts up), and it runs in the event loop, so
            without this the first decision of the first game is that much over the deadline.
            Nothing of it is kept: its history entry is dropped and the counters are put back.
            """
            import time as _time

            began = _time.perf_counter()
            try:
                stats = dict(self.batch_stats)
                # One battle, then as many as play at once: the first pass of each batch size is slow too.
                for rows in (1, max(2, int(getattr(self, "_max_concurrent_battles", 1) or 1))):
                    tags = ["warm-up-%d" % i for i in range(rows)]
                    self._forward_batch(tags, np.zeros((rows, self.assembler.total_dim), dtype=np.float32))
                    for tag in tags:
                        self.policy.forget(tag)
                self.batch_stats.update(stats)
            except Exception:       # the first real decision will say what is wrong
                pass
            self.warm_up_ms = round((_time.perf_counter() - began) * 1000.0, 1)

        # -- which battle is played how --------------------------------------------------------

        def battle_mode(self, tag: str) -> str:
            """The mode of a battle, decided the first time it is asked for."""
            mode = self._modes.get(tag)
            if mode is None:
                mode = self.search_mode
                if self.mode_picker is not None:
                    try:
                        mode = str(self.mode_picker(tag))
                        split_mode(mode)
                    except Exception:
                        mode = "net"
                self._modes[tag] = mode
                self._games[tag] = _new_game(mode)
                for store in (self._modes, self._games, self._decisions):
                    while len(store) > KEPT_BATTLES:
                        store.popitem(last=False)
            return mode

        def _searched_battle(self, tag: str) -> bool:
            return self.foulplay_enabled and split_mode(self.battle_mode(tag))[0] != "net" and tag not in self._rejoined

        def battle_rejoined(self, tag: str) -> None:
            """
            The connection was lost and this battle's room joined again: the server is about to send
            the whole battle once more, from |init|. Foul Play cannot pick a battle up midway (see
            bench/FOULPLAY_BRIDGE.md, "A room that is joined again"), so the search lets it go: its
            lines are no longer sent to the bridge, and the rest of its decisions are the network's,
            at once, counted as "rejoined" (not as a failure of the search).
            """
            if tag in self._rejoined:
                return
            searched = self._searched_battle(tag)
            self._rejoined.add(tag)
            for store in (self._trackers, self._predictions):
                store.pop(tag, None)        # rebuilt from the replay by _follow_opponent, not fed it twice
            if searched:
                try:
                    self.foulplay.forget(self._foulplay_key(tag))
                except Exception:       # bookkeeping must never disturb a game
                    pass

        # -- the battle's lines ----------------------------------------------------------------

        async def _handle_battle_message(self, split_messages: List[List[str]]):
            room = None
            if split_messages and split_messages[0] and split_messages[0][0].startswith(">"):
                room = split_messages[0][0][1:].strip()
            if room is None or not room.startswith("battle-"):
                return await super(fb.FoulPlayMixin, self)._handle_battle_message(split_messages)
            try:
                self._note_tera(room, split_messages)
            except Exception:       # bookkeeping must never disturb a game
                pass
            try:
                self._follow_opponent(room, split_messages)
            except Exception:
                pass
            if self._searched_battle(room):
                return await super()._handle_battle_message(split_messages)       # the mixin forwards the lines first
            # A battle the network plays alone: nothing goes to the search.
            return await super(fb.FoulPlayMixin, self)._handle_battle_message(split_messages)

        def _note_tera(self, room: str, split_messages: List[List[str]]) -> None:
            for parts in split_messages[1:]:
                if len(parts) > 3 and parts[1] == "-terastallize":
                    battle = self.battles.get(room)
                    role = getattr(battle, "player_role", None)
                    if role and parts[2].startswith(role):
                        self.battle_mode(room)
                        game = self._games.get(room)
                        if game is not None and not game["tera"]:
                            game["tera"] = True
                            game["tera_turn"] = int(getattr(battle, "turn", 0) or 0)
                            game["tera_type"] = parts[3]
                            order = game.get("tera_order")
                            game["tera_by"] = order["source"] if order else None
                            game["tera_net"] = order["net"] if order else None
                            game["tera_search"] = order["search"] if order else None

        # -- "predict": the opponent's next action ----------------------------------------------

        def _predict_ready(self) -> None:
            """The model, the settings and the game data the tracker reads, loaded before the first game and not in one."""
            import opponent_model as om

            if self.predict_settings is None:
                self.predict_settings = om.PredictSettings()
            if self.opponent_model is None:
                self.opponent_model = om.OpponentModel.load()
            om.dex()

        def _is_predict(self, room: str) -> bool:
            return self.foulplay_enabled and self.battle_mode(room) == "predict"

        def _follow_opponent(self, room: str, split_messages: List[List[str]]) -> None:
            """Every line of a "predict" battle goes to its tracker; the turns that ended are scored."""
            if not self._is_predict(room):
                return
            import opponent_model as om

            tracker = self._trackers.get(room)
            if tracker is None:
                tracker = self._trackers[room] = om.OpponentTracker(room)
                while len(self._trackers) > KEPT_BATTLES:
                    self._trackers.popitem(last=False)
            role = getattr(self.battles.get(room), "player_role", None)
            was_broken = tracker.broken
            for parts in split_messages[1:]:
                if role is None and len(parts) > 3 and parts[1] == "player" and parts[3] == self.username:
                    role = parts[2]
                tracker.feed("|".join(parts), role)
            if tracker.broken and not was_broken:
                self.search_totals["trackers_broken"] += 1
            for turn, label, name in tracker.closed():
                self._opponent_did(room, turn, label, name)

        def _predict(self, battle: Any, advice: Any, view: SearchView, mask: Any, mix_action: int) -> Tuple[int, str, Dict[str, Any]]:
            """(action, "predict" when the mix choice was left else "mix", what was predicted and why)."""
            import opponent_model as om

            tag = battle.battle_tag
            info: Dict[str, Any] = {"mix": int(mix_action), "changed": False, "why": ""}
            if getattr(battle, "force_switch", False):
                info["why"] = "forced"          # the opponent chooses nothing at this request
                return mix_action, "mix", info
            if self.opponent_model is None or self.predict_settings is None:
                self._predict_ready()
            tracker = self._trackers.get(tag)
            turn = int(getattr(battle, "turn", 0) or 0)
            ctx = tracker.context(turn) if tracker is not None else None
            if ctx is None:
                info["why"] = "no context"
                return mix_action, "mix", info
            prediction = self.opponent_model.predict(ctx, advice)
            action, chosen = om.choose(prediction, getattr(advice, "payoff_table", None), getattr(advice, "payoff_theirs", None),
                                       om.search_view(advice), mix_action, mask, view.share, self.predict_settings)
            info.update(chosen)
            theirs_now = str((ctx["opp"]["active"] or {}).get("id") or "")
            info["against"] = theirs_now
            if chosen.get("changed") and self.predict_settings.once_bitten and (theirs_now, chosen.get("read")) in self._failed_reads.get(tag, ()):
                # This very read was wrong against this Pokemon earlier in the battle: not again.
                action = mix_action
                chosen = dict(chosen, changed=False, why="bitten")
                info.update({"changed": False, "why": "bitten"})
            top = prediction.top(3)
            payoff = getattr(advice, "payoff", None) or {}
            cells = sum(int(w.get("cells") or 0) for w in payoff.get("worlds") or [])
            info.update({
                "top": [[name, round(p, 4)] for name, p in top], "p_switch": round(prediction.p_switch, 4),
                "p_tera": round(prediction.p_tera, 4), "confident": bool((chosen.get("confidence") or 0.0) >= self.predict_settings.min_confidence),
                "payoff_ms": payoff.get("ms"), "payoff_cells": cells,
                "payoff_known": round(sum(int(w.get("known") or 0) for w in payoff.get("worlds") or []) / cells, 3) if cells else None,
            })
            if chosen.get("changed"):
                # The position in words, for whoever reads the record: who faces whom, and the two actions by name.
                ours, theirs = ctx["us"]["active"], ctx["opp"]["active"]
                info["position"] = {
                    "ours": "%s %d%%" % (ours.get("species"), round(100 * float(ours.get("hp") or 0))),
                    "theirs": "%s %d%%" % (theirs.get("species"), round(100 * float(theirs.get("hp") or 0))),
                    "their_shown_moves": sorted((theirs.get("moves") or {}).keys()), "their_last_move": theirs.get("last_move"),
                    "their_shown_bench": [m["id"] for m in ctx["opp"]["bench"] if not m.get("fainted")],
                    "mix_would_play": self._action_name(battle, mix_action), "played": self._action_name(battle, action),
                }
            turns = self._predictions.setdefault(tag, {})
            turns[turn] = {"dist": prediction.dist, "top": top[0][0], "info": info}
            while len(self._predictions) > KEPT_BATTLES:
                self._predictions.popitem(last=False)
            return int(action), ("predict" if chosen.get("changed") else "mix"), info

        def _action_name(self, battle: Any, action: int) -> str:
            """An action of ours in words: "move earthquake", "move earthquake terastallize", "switch garchomp"."""
            try:
                target, kwargs = self.assembler.map_index_to_order(int(action), battle)
                if hasattr(target, "current_hp"):
                    return "switch " + str(getattr(target, "species", target))
                return "move " + str(getattr(target, "id", target)) + (" terastallize" if kwargs.get("terastallize") else "")
            except Exception:
                return "action %d" % int(action)

        def _opponent_did(self, room: str, turn: int, label: Dict[str, Any], name: Optional[str]) -> None:
            """A turn has ended: what the opponent did in it against what was predicted before it."""
            predicted = (self._predictions.get(room) or {}).pop(turn, None)
            if predicted is None:
                return
            info = predicted["info"]
            outcome: Dict[str, Any] = {"did": label.get("action"), "name": name, "tera": bool(label.get("tera")),
                                       "why_unknown": label.get("why_unknown"), "locked": bool(label.get("locked"))}
            game = self._games.get(room)
            per_mode = self.search_totals["by_mode"].get("predict")
            groups = [c for c in (game, self.search_totals, per_mode) if c is not None]
            if name is not None:
                top_right = predicted["top"] == name
                read = info.get("read")
                read_right = (name.startswith("switch ") if read == "switch" else read == name) if info.get("confident") else None
                outcome.update({"top1_right": top_right, "p_true": round(float(predicted["dist"].get(name, 0.0)), 4), "read_right": read_right})
                if read_right is False:
                    self._failed_reads.setdefault(room, set()).add((info.get("against"), read))
                    while len(self._failed_reads) > KEPT_BATTLES:
                        self._failed_reads.popitem(last=False)
                for counts in groups:
                    counts["pred_scored"] = counts.get("pred_scored", 0) + 1
                    counts["pred_top1_right"] = counts.get("pred_top1_right", 0) + int(top_right)
                    counts["pred_p_true"] = counts.get("pred_p_true", 0.0) + outcome["p_true"]
                    if read_right is not None:
                        counts["pred_reads_scored"] = counts.get("pred_reads_scored", 0) + 1
                        counts["pred_reads_right"] = counts.get("pred_reads_right", 0) + int(read_right)
                    if info.get("changed"):
                        counts["pred_changed_scored"] = counts.get("pred_changed_scored", 0) + 1
                        counts["pred_changed_right"] = counts.get("pred_changed_right", 0) + int(bool(read_right))
            info["outcome"] = outcome          # the decision's record holds this same dict
            if self.on_outcome is not None:
                try:
                    self.on_outcome({"battle": room, "turn": turn, "predict": info})
                except Exception:
                    pass

        def _battle_finished_callback(self, battle: Any):
            if self._searched_battle(battle.battle_tag):
                return super()._battle_finished_callback(battle)                  # the mixin forgets the battle
            return super(fb.FoulPlayMixin, self)._battle_finished_callback(battle)

        # -- deciding --------------------------------------------------------------------------

        def _why_not_ask(self, battle: Any, mask: Any) -> str:
            if not self.foulplay_enabled:
                return "off"
            if battle.battle_tag in self._rejoined:
                return "rejoined"
            if int((mask > 0.5).sum()) <= 1:
                return "single"
            if getattr(battle, "reviving", False):
                return "revive"
            return ""

        def _ask(self, battle: Any, ttl: float) -> Any:
            request = getattr(battle, "last_request", None) or {}
            more = self.search_more
            if self.battle_mode(battle.battle_tag) == "predict" and not getattr(battle, "force_switch", False):
                # The table of values per (our action, the opponent's action) that opponent_model.choose reads.
                try:
                    if self.predict_settings is None:
                        self._predict_ready()
                    if self.predict_settings.payoff_ms > 0:
                        more = dict({"payoff_ms": self.predict_settings.payoff_ms, "payoff": self.predict_settings.payoff_mode}, **more)
                except Exception:       # no model, no table: the search is asked as for "mix"
                    self.search_totals["predict_errors"] += 1
            return self.foulplay.advice_future(self._foulplay_key(battle.battle_tag), self.search_ms,
                                               request.get("rqid"), ttl=ttl, **more)

        async def choose_move(self, battle: Any):
            tag = battle.battle_tag
            mode = self.battle_mode(tag)
            kind, rule = split_mode(mode)
            searched = self.foulplay_enabled and kind != "net"
            if kind != "net" and getattr(battle, "finished", False):
                # The battle ended while the last decision was being searched. The server answers
                # that late order with an error and poke-env asks again: there is nothing to
                # decide, and nothing is sent (an order with an empty message is not sent).
                self.search_totals["after_end"] += 1
                return _EmptyBattleOrder()
            order = None
            try:
                order = await self._decide(battle, mode, kind, rule)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Nothing in here may leave the battle waiting. A random legal order, counted, and
                # where the audit of players.py is in front it is told as well.
                self.search_totals["exceptions"] += 1
                audit = getattr(self, "audit", None)
                if isinstance(audit, dict) and "exceptions" in audit:
                    audit["exceptions"] += 1
                    self._audit_note("%s: choose_move raised %r" % (tag, exc))
                order = self.choose_random_move(battle)
            if searched:
                self.foulplay_note(battle, order)      # what is about to be sent
                self._noted[tag] = getattr(order, "message", None)
                while len(self._noted) > KEPT_BATTLES:
                    self._noted.popitem(last=False)
            return order

        # -- what was really sent --------------------------------------------------------------

        def _watch_orders(self) -> None:
            """
            Every order goes out through the client's send_message, also the few that do not come
            from choose_move: the random order players.ChoiceAuditMixin forces after repeated
            rejections, and poke-env's own "/choose default". Those are noted here, so that the
            search is told what was really sent whoever chose it.
            """
            client = self.ps_client
            self._order_send = client.send_message      # the client's own; the tests put theirs here

            async def send_message(message: str, room: str = "", message_2: Optional[str] = None):
                try:
                    self._order_sent(room, message)
                except Exception:       # bookkeeping must never disturb a game
                    pass
                return await self._order_send(message, room, message_2)

            client.send_message = send_message

        def _order_sent(self, room: Any, message: Any) -> None:
            if not (isinstance(room, str) and isinstance(message, str) and message.startswith("/choose ")):
                return
            if room not in self._modes or not self._searched_battle(room):
                return
            if self._noted.pop(room, None) == message:
                return                      # choose_move's own order, already noted
            battle = self.battles.get(room)
            if battle is None or getattr(battle, "finished", False):
                return
            self.search_totals["noted_at_send"] += 1
            self.foulplay_note(battle, self._order_of_message(battle, message))

        def _order_of_message(self, battle: Any, message: str) -> Any:
            """The poke-env order that reads as `message` on the wire ("/choose default" for anything else)."""
            words = message.split()
            if len(words) >= 3 and words[1] == "move":
                for move in battle.available_moves:
                    if move.id == words[2] or (words[2] == "1" and move.id == "recharge"):
                        return self.create_order(move, terastallize="terastallize" in words[3:])
            elif len(words) >= 3 and words[1] == "switch":
                name = " ".join(words[2:])
                for mon in battle.available_switches:
                    if mon.name == name:
                        return self.create_order(mon)
            return self.choose_default_move()

        def _watch_loop(self, loop: Any) -> None:
            """
            Measures how late the event loop runs while games are played: a task that sleeps a
            quarter of a second and notes by how much it overslept. It ends by itself a minute
            after the last decision and is started again by the next one.
            """
            self._last_decision = loop.time()
            if self._lag_task is not None and not self._lag_task.done():
                return

            async def watch(step: float = 0.25) -> None:
                stats = self.loop_lag
                while loop.time() - self._last_decision < 60.0:
                    due = loop.time() + step
                    await asyncio.sleep(step)
                    late = (loop.time() - due) * 1000.0
                    stats["checks"] += 1
                    if late > stats["max_ms"]:
                        stats["max_ms"] = round(late, 1)
                    if late > 500.0:
                        stats["over_500_ms"] += 1
                        stats["late_ms_total"] = round(stats["late_ms_total"] + late, 1)

            self._lag_task = loop.create_task(watch())

        async def _decide(self, battle: Any, mode: str, kind: str, rule: Optional[str]):
            loop = asyncio.get_running_loop()
            started = loop.time()
            tag = battle.battle_tag
            if kind != "net":
                try:
                    self._watch_loop(loop)
                except Exception:       # a measurement must never disturb a game
                    pass
            self.counts["decisions"] += 1
            self.counts["heuristic_missing"] += 1
            obs = self.assembler.assemble(battle, events=self._events.pop(tag, []))
            m0, m1 = self.assembler.offsets["action_mask"]
            mask = obs[m0:m1].astype(np.float32)

            # The question goes to the search first: it thinks while the network answers.
            reason, future = "", None
            if kind != "net":
                reason = self._why_not_ask(battle, mask)
                if not reason:
                    try:
                        future = self._ask(battle, self.deadline - RESERVE_S - (loop.time() - started))
                    except Exception:
                        reason = "error"

            logits = await self._queue_inference(tag, obs)
            net_action = int(combine(logits, None, mask, self.hybrid, None if self.hybrid.greedy else self.rng))
            net_ms = (loop.time() - started) * 1000.0

            advice = None
            wait_over_ms = 0.0
            if future is not None:
                left = self.deadline - RESERVE_S - (loop.time() - started)
                wait_until = loop.time() + max(0.0, left)
                try:
                    timed_out = False
                    try:
                        raw = await asyncio.wait_for(asyncio.wrap_future(future), timeout=max(0.0, left))
                    except asyncio.TimeoutError:
                        # The wait was to end at wait_until. Anything beyond that is the event loop
                        # itself running late (the process not getting the processor, or something
                        # holding the loop), which no timeout in here can shorten.
                        wait_over_ms = max(0.0, (loop.time() - wait_until) * 1000.0)
                        if future.done() and not future.cancelled():
                            # The helper had answered: its answer was set by the bridge's thread and
                            # only this loop was late to read it. When the loop is held up, the
                            # timer and the answer come due together and the timer wins. The answer
                            # is in hand and costs no more time, so it is used.
                            raw = future.result()
                        else:
                            timed_out = True
                    if timed_out:
                        reason = "timeout"
                    else:
                        advice = self._foulplay_fill(battle, raw)
                except fb.FoulPlayBridgeError as exc:
                    reason = "down" if exc.down else ("expired" if exc.expired else ("revive" if exc.revive else "error"))
                    if reason == "error":
                        self._foulplay_problem("%s turn %s: %s" % (tag, getattr(battle, "turn", "?"), exc))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    reason = "error"
                    self._foulplay_problem("%s turn %s: %r" % (tag, getattr(battle, "turn", "?"), exc))

            view = None
            if advice is not None:
                view, reason = read_advice(advice, mask, self.mix_settings.min_iterations)

            probs = legal_probabilities(logits, mask)
            action, source = net_action, "net"
            predicted: Optional[Dict[str, Any]] = None
            if view is not None:
                if kind == "search":
                    action, source = view.best, "search"
                elif kind in ("mix", "predict"):
                    settings = self.mix_settings if rule is None else replace(self.mix_settings, rule=rule)
                    action, source = mix(probs, mask, view, settings), "mix"
                    if kind == "predict":
                        mixed = action
                        try:
                            action, source, predicted = self._predict(battle, advice, view, mask, mixed)
                        except Exception as exc:        # a prediction must never cost a move: the mix choice stands
                            self.search_totals["predict_errors"] += 1
                            action, source, predicted = mixed, "mix", {"mix": int(mixed), "changed": False, "why": "error", "error": repr(exc)[:200]}

            target, kwargs = self.assembler.map_index_to_order(action, battle)
            if target == "DEFAULT":
                # A forced state the 14 actions do not cover.
                self.counts["fallbacks"] += 1
                order = self.choose_random_move(battle)
            else:
                order = self.create_order(target, **kwargs)

            elapsed_ms = (loop.time() - started) * 1000.0
            try:
                self._record(battle, mode, mask, probs, net_action, view, advice, action, source, reason,
                             future is not None, elapsed_ms, net_ms, wait_over_ms, predicted)
            except Exception:       # bookkeeping must never disturb a game
                pass
            return order

        # -- the record ------------------------------------------------------------------------

        def _record(self, battle: Any, mode: str, mask: Any, probs: List[float], net_action: int, view: Optional[SearchView],
                    advice: Any, action: int, source: str, reason: str, asked: bool, elapsed_ms: float,
                    net_ms: float = 0.0, wait_over_ms: float = 0.0, predicted: Optional[Dict[str, Any]] = None) -> None:
            tag = battle.battle_tag
            request = getattr(battle, "last_request", None) or {}
            rec: Dict[str, Any] = {
                "battle": tag, "turn": int(getattr(battle, "turn", 0) or 0), "rqid": request.get("rqid"), "mode": mode,
                "legal": int((mask > 0.5).sum()), "mask": [int(m > 0.5) for m in mask],
                "forced": bool(getattr(battle, "force_switch", False)),
                "net": int(net_action), "search": None, "played": int(action), "source": source,
                "asked": bool(asked), "reason": reason, "ms": round(elapsed_ms, 1),
                "net_ms": round(net_ms, 1),      # request to the network's answer (its forward pass runs in the event loop)
                "wait_over_ms": round(wait_over_ms, 1),   # a timeout that fired this much after it was due: the loop was late
                "p_net": round(probs[net_action], 4), "p_played": round(probs[action], 4),
                "probs": [round(p, 4) for p in probs],
            }
            if view is not None:
                best = view.best
                rec.update({
                    "search": int(best), "iterations": view.iterations, "worlds": int(getattr(advice, "worlds", 0) or 0),
                    "search_ms": round(float(getattr(advice, "wall_ms", 0.0) or 0.0), 1),
                    "share_net": round(view.share[net_action], 4), "share_search": round(view.share[best], 4),
                    "share_played": round(view.share[action], 4),
                    "shares": [round(x, 4) for x in view.share],
                    "score_net": view.score[net_action], "score_search": view.score[best], "score_played": view.score[action],
                    "disagree": base_action(best) != base_action(net_action),
                    "net_overruled": base_action(action) != base_action(net_action),
                    "search_overruled": base_action(action) != base_action(best),
                    # Terastallization apart: the network wanted the plain move and it was played
                    # with Tera, or the other way round (whatever the move).
                    "tera_added": action >= 10 and net_action < 10,
                    "tera_dropped": action < 10 and net_action >= 10,
                })
                theirs = getattr(advice, "opponent", None) or []
                rec["opponent_active"] = getattr(advice, "opponent_active", None)
                rec["opponent"] = [[o.name, round(float(o.share), 4), o.seen] for o in theirs[:6]]
            if predicted is not None:
                # What was predicted of the opponent and what choose() made of it. After the turn
                # "outcome" is added to this same dict: what the opponent did.
                rec["predict"] = predicted
            self._decisions.setdefault(tag, []).append(rec)

            game = self._games.get(tag)
            totals = self.search_totals
            per_mode = totals["by_mode"].setdefault(mode, {"decisions": 0, "asked": 0, "searched": 0, "fallbacks": {},
                                                           "disagreed": 0, "net_overruled": 0, "search_overruled": 0,
                                                           "tera_added": 0, "tera_dropped": 0,
                                                           "iterations": 0, "search_ms": 0.0, "decision_ms": 0.0,
                                                           "slowest_ms": 0.0, "late": 0})
            late = elapsed_ms > self.deadline * 1000.0
            for counts in (game, totals, per_mode):
                if counts is None:
                    continue
                counts["decisions"] += 1
                counts["decision_ms"] += elapsed_ms
                counts["slowest_ms"] = max(counts["slowest_ms"], elapsed_ms)
                counts["late"] += late
                if asked:
                    counts["asked"] += 1
                if view is not None:
                    counts["searched"] += 1
                    counts["iterations"] += view.iterations
                    counts["search_ms"] += rec["search_ms"]
                    counts["disagreed"] += rec["disagree"]
                    counts["net_overruled"] += rec["net_overruled"]
                    counts["search_overruled"] += rec["search_overruled"]
                    counts["tera_added"] += rec["tera_added"]
                    counts["tera_dropped"] += rec["tera_dropped"]
                elif reason:
                    counts["fallbacks"][reason] = counts["fallbacks"].get(reason, 0) + 1
                if predicted is not None and "top" in predicted:
                    counts["pred_made"] = counts.get("pred_made", 0) + 1
                    counts["pred_reads"] = counts.get("pred_reads", 0) + int(bool(predicted.get("confident")))
                    counts["pred_changed"] = counts.get("pred_changed", 0) + int(bool(predicted.get("changed")))
                    why = predicted.get("why") or "changed"
                    kept = counts.setdefault("pred_kept", {})
                    kept[why] = kept.get(why, 0) + 1
            if game is not None and action >= 10:
                game["tera_order"] = {"turn": rec["turn"], "source": source, "net": net_action >= 10,
                                      "search": bool(view is not None and view.best >= 10)}
            if self.on_decision is not None:
                try:
                    self.on_decision(rec)
                except Exception:
                    pass

        def decisions_of(self, tag: str) -> List[Dict[str, Any]]:
            """Every decision of one battle so far, oldest first."""
            return list(self._decisions.get(tag, []))

        def forget_decisions(self, tag: str) -> None:
            """Drops the decision records of a battle that has been reported (the counts stay)."""
            self._decisions.pop(tag, None)

        def game_summary(self, tag: str) -> Optional[Dict[str, Any]]:
            """
            One battle added up, for the results file: the arm, the decisions, how many the
            search answered, why not for the others, who was overruled, the search's effort and
            the time, and whether and when it Terastallized.
            """
            game = self._games.get(tag)
            if game is None:
                return None
            searched = game["searched"]
            decisions = game["decisions"]
            fallbacks = dict(game["fallbacks"])
            # "single" and "revive" are not failures: nothing was asked.
            failed = sum(n for why, n in fallbacks.items() if why not in ("single", "revive"))
            out = {
                "arm": game["mode"], "decisions": decisions, "asked": game["asked"], "searched": searched,
                "fallbacks": failed, "fallback_reasons": fallbacks,
                "disagreed": game["disagreed"], "overruled": game["net_overruled"],
                "search_overruled": game["search_overruled"],
                "tera_added": game["tera_added"], "tera_dropped": game["tera_dropped"],
                "iterations_per_decision": int(game["iterations"] / searched) if searched else 0,
                "search_ms_mean": round(game["search_ms"] / searched, 1) if searched else None,
                "decision_ms_mean": round(game["decision_ms"] / decisions, 1) if decisions else None,
                "decision_ms_max": round(game["slowest_ms"], 1), "late": game["late"],
                "tera": bool(game["tera"]), "tera_turn": game["tera_turn"], "tera_type": game["tera_type"],
                "tera_by": game["tera_by"], "tera_net": game["tera_net"], "tera_search": game["tera_search"],
            }
            if game["mode"] == "predict":
                # Predictions of the opponent's action: made, scored (the log says what it did),
                # the likeliest action was the one it took, confident reads and how many were
                # right, turns on which the mix choice was left and on how many of those the read
                # was right, and why the mix choice was kept on the others.
                for name in PRED_COUNTS:
                    out[name] = round(game[name], 4) if isinstance(game[name], float) else game[name]
                out["pred_kept"] = dict(game["pred_kept"])
            return out

        def search_summary(self) -> str:
            """One line per mode played: what the search was asked, what it answered, who was overruled."""
            lines = []
            for mode, c in sorted(self.search_totals["by_mode"].items()):
                n, searched = c["decisions"], c["searched"]
                failed = {why: k for why, k in c["fallbacks"].items() if why not in ("single", "revive")}
                text = "%s: %d decisions" % (mode, n)
                if c["asked"] or c["fallbacks"]:
                    text += ", search asked %d, answered %d" % (c["asked"], searched)
                    text += ", fell back to the network %d" % sum(failed.values())
                    if failed:
                        text += " (" + ", ".join("%s %d" % kv for kv in sorted(failed.items())) + ")"
                    skipped = {why: k for why, k in c["fallbacks"].items() if why in ("single", "revive")}
                    if skipped:
                        text += ", not asked " + ", ".join("%s %d" % kv for kv in sorted(skipped.items()))
                    if searched:
                        text += ("; the two disagreed on %d (%.0f%%), network overruled %d, search overruled %d"
                                 ", Tera added to the network's move %d and taken off it %d"
                                 "; %d iterations and %.0f ms of search per decision"
                                 % (c["disagreed"], 100.0 * c["disagreed"] / searched, c["net_overruled"], c["search_overruled"],
                                    c["tera_added"], c["tera_dropped"],
                                    c["iterations"] / searched, c["search_ms"] / searched))
                if n:
                    text += "; request to order %.0f ms on average, %.0f ms at the slowest, %d over the deadline" % (
                        c["decision_ms"] / n, c["slowest_ms"], c["late"])
                lines.append(text)
            extra = []
            t = self.search_totals
            if t["pred_made"]:
                extra.append("opponent predicted on %d turns, scored %d: its likeliest action was right on %d (%.0f%%); "
                             "%d confident reads, %d of %d scored right; the mix choice was left on %d turns, the read right on %d of %d scored (kept: %s)"
                             % (t["pred_made"], t["pred_scored"], t["pred_top1_right"], 100.0 * t["pred_top1_right"] / max(1, t["pred_scored"]),
                                t["pred_reads"], t["pred_reads_right"], t["pred_reads_scored"], t["pred_changed"], t["pred_changed_right"],
                                t["pred_changed_scored"], ", ".join("%s %d" % kv for kv in sorted(t["pred_kept"].items()))))
            if t["predict_errors"] or t["trackers_broken"]:
                extra.append("%d predictions raised, %d battles could not be followed" % (t["predict_errors"], t["trackers_broken"]))
            if self.search_totals["after_end"]:
                extra.append("%d requests after a battle had ended" % self.search_totals["after_end"])
            if self.search_totals["exceptions"]:
                extra.append("%d decisions raised and were played at random" % self.search_totals["exceptions"])
            lag = self.loop_lag
            if lag["checks"]:
                extra.append("the event loop ran more than 0.5 s late %d times in %d checks (%.0f ms at the worst, %.1f s in all)"
                             % (lag["over_500_ms"], lag["checks"], lag["max_ms"], lag["late_ms_total"] / 1000.0))
            if self.search_totals["noted_at_send"]:
                extra.append("%d orders that did not come from the player's own choice were noted as sent" % self.search_totals["noted_at_send"])
            if extra:
                lines.append(", ".join(extra))
            return "\n".join(lines)

    _SEARCH_PLAYER = SearchPlayer
    return _SEARCH_PLAYER

