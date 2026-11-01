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
    mix-guard      mix with the veto rule and the guard on (MixSettings.guard): an answer to set-up
                   moves and sleep, OFF in every other mode unless the settings turn it on.
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

Live predictions (live_predictions=True, ladder.py's default): in every searched battle, at no
extra search cost, the opponent's next action as opponent_model.py reads it from the search's own
view of the opponent's side (every answer of the bridge carries it). The record's "predict" holds
the top 3 with their probabilities and a line to print (prediction_line); after the turn what the
opponent did is read from the battle and the read is marked right or wrong (outcome_text), and
every game counts its predictions, how often the likeliest was right and how often what happened
was in the top 3. It changes no move: only "predict" acts on a prediction, and the guard reads it.

The guard (MixSettings.guard, mode "mix-guard"; bench/notes/predict_live.md): when the opponent's
Pokemon is boosted, or the read says it sets up or puts us to sleep, the veto rule is changed in
three ways: our passive moves (set-up, recovery, Protect, hazards) are not played, Sucker Punch is
not played into a likely status move, and when the search turned away from the network's choice
the search's own best is played, never a third action. See mix() and Threat.

The counter rules (MixSettings.setup_counter, setup_other_side; arm "mix-counter"; round 3 of the
set-up work, branch counterfix), OFF unless asked for. "Use the counter": in a set-up position, a
move of our active Pokemon that answers it (phazing, Haze, Encore, Taunt, a sleep move, Perish
Song) is played when it can work, unless an attack on the board is estimated to knock the
set-upper out (at full HP only when Sturdy, Focus Sash, Multiscale and the like are ruled out, or
at twice its HP). After round 3's review it no longer switches to a benched Pokemon that holds one
(MixSettings.counter_switch, off: it cost most of the puzzle loss), and a counter counts only when
ours can use it before being knocked out, is not asleep, and (Encore) moves first or reads another
set-up move. "Don't hit the boosted defence": an attack into a defence the opponent has boosted
gives way to one of the other kind estimated to do at least as much, accuracy counted; never a
working counter, an attack in KO range or a priority move. setup_counter.py has the board.

The Luna arm (mode "luna", luna_player.py), OFF unless a battle is given that mode: mix decides as
it does today, and then OpenAI's Decisions API is asked ONE choice question whose choices are the
legal actions, each with the facts and both sides' numbers; its choice is played. Without an
answer in time, with an error, a refusal or a choice that is not legal, mix's pick is played, and
a hard guard keeps mix's pick when Luna's has under 1% of the network and of the search while
mix's has over 60% of either. Every decision's record says what Luna answered, how long it took
and why mix played when it did ("luna"), and game_summary() counts it.

Importing this module does not pull in torch: the player class is built on first use.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import foulplay_bridge as fb

__all__ = [
    "ACTIONS", "MODES", "RULES", "ARM_FIXES", "FALLBACK_REASONS", "MixSettings", "SearchView", "ArmPicker",
    "split_mode", "check_modes", "legal_probabilities", "merged_prior", "read_advice", "mix",
    "base_action", "minimum_deadline", "search_player", "Threat", "guard_reason", "guarded", "short_tag", "option_text",
    "prediction_text", "prediction_line", "did_text", "outcome_text", "setup_reason", "ThinkLonger", "ko_estimate", "ko_doubt",
    "parse_time_left", "counter_reason", "COUNTER_COUNTS", "counter_text",
    "DEEP_ARM", "SearchBudget", "DEEP_BUDGET", "full_width_turns", "value_fields", "LUNA_ARM",
    "SOL_ARM", "LINES_ARM", "LINES_ARMS", "DECIDERS", "ARM_BUDGETS", "LINES_BUDGET", "LinesRule", "lines_pick", "SOLO_ARMS",
    "GameSetting", "SettingPicker",
]

ACTIONS = 14
RULES = ("veto", "pool", "shortlist")
MODES = ("net", "search", "mix") + tuple("mix-" + rule for rule in RULES) + ("predict", "mix-guard")

# Arms of a measurement that carry set-up fixes (bench/gauntlet.py --arms mix,mix-fixdefer): the
# veto rule with these MixSettings, so one player plays the plain veto and a fix side by side.
# The flag says whether the player's ThinkLonger applies to the arm. With such an arm among a
# player's arms, a ThinkLonger applies only to the arms marked so (the "mix" control keeps the
# ordinary search). Outside a measurement nothing changes: the ladder's flags set MixSettings.
_FIX = {"no_third": True, "tera_own": True, "sure_attack": 0.70}      # --veto-fix
ARM_FIXES: Dict[str, Tuple[Dict[str, Any], bool]] = {
    "mix-fix": (dict(_FIX), False),                                    # a: --veto-fix
    "mix-defer": ({"setup_defer": True}, False),                       # b: --setup-defer
    "mix-fixdefer": (dict(_FIX, setup_defer=True), False),             # a + b
    "mix-fixdefer-long": (dict(_FIX, setup_defer=True), True),         # a + b + c: --think-longer
    "mix-fix-long": (dict(_FIX), True),                                # a + c
    # Round 2's blend (--setup-blend, bench/setup_puzzles/ROUND2.md on setupfix): no winner on the
    # puzzles, so these are for a ladder A/B against the plain "mix" only.
    "mix-blend4": ({"setup_blend": 4.0}, False),                       # best CHOOSE primary +0.29 (-0.80, +1.36), but -3.6 on the oracle guard
    "mix-blend8": ({"setup_blend": 8.0}, False),                       # +0.22 (-0.75, +1.22), the only rule past the guard (both judges)
    # Round 3 (branch counterfix): use the counter on the team, and don't hit the boosted defence.
    # From this morning's 11 set-up sweep losses (7 with an unused counter, 12 of 45 decisions into
    # the boosted stat); for a ladder A/B against the plain "mix" only. Since round 3's review
    # (bench/notes/counter_fixes.md): the counter in place only (no switch to a benched holder),
    # with the review's fixes; d573f7e's version with the switches lost 3 points on the puzzles.
    "mix-counter": ({"setup_counter": True, "setup_other_side": True}, False),
    # The guarded search (2026-10-15, reports/2026-10-15/LOSS_STUDY.md): the veto may only pick
    # among the moves the network gives 3% or more.
    "mix-trust": ({"trust": 0.03}, False),
}
MODES = MODES + tuple(ARM_FIXES)

# The deep arm (ladder.py --arms net,mix,mix-deep): today's mix with more search per decision, its
# budget a SearchBudget of its own (DEEP_BUDGET unless the player is given another). The rule and
# every other setting are mix's.
DEEP_ARM = "mix-deep"
MODES = MODES + (DEEP_ARM,)

# The Luna arm (ladder.py --player luna, --arms mix,luna): today's mix computed as ever, then
# OpenAI's Decisions API picks among the legal actions with mix's numbers in front of it
# (luna_player.py). The search, its budget and every setting are mix's.
LUNA_ARM = "luna"
MODES = MODES + (LUNA_ARM,)

# The Sol arm (--player sol, --arms luna,sol): the same with OpenAI's gpt-6-sol asked through the
# Responses API (luna_player.MODELS["sol"]): a larger and slower model, so the decision gets seconds
# of its own for the answer on top of the search's deadline (SearchBudget.decide_s).
SOL_ARM = "sol"
# The lines arms: today's search, then the search's favourite actions each played forward a few
# turns (SearchBudget.lines_ms; bench/fp_bridge_server.py "lines"). "mix-lines": mix's pick gives
# way to an action whose line ends clearly better (LinesRule). "luna-lines", "sol-lines": the model
# decides with the lines in front of it.
LINES_ARM = "mix-lines"
LINES_ARMS = (LINES_ARM, "luna-lines", "sol-lines")
# arm -> the model that decides it (a key of the player's `deciders`, a luna_player.MODELS name)
DECIDERS: Dict[str, str] = {LUNA_ARM: "luna", SOL_ARM: "sol", "luna-lines": "luna", "sol-lines": "sol"}
MODES = MODES + (SOL_ARM,) + LINES_ARMS
# The model alone (--player luna-solo, sol-solo): no search is asked and the network's pick is not
# played. The model is given the position and the legal actions with their facts (no adviser's
# number, no pick of mix's) and its choice is the order. When it gives none, a rule plays
# (luna_player.solo_rule): the strongest attack by power and type, the healthiest switch-in. The
# network still runs for every decision, because the player reads the game through it, and its
# pick is written in the record ("net") for whoever wants to compare; it decides nothing.
SOLO_ARMS = ("luna-solo", "sol-solo")
DECIDERS.update({"luna-solo": "luna", "sol-solo": "sol"})
# Claude alone (claude_player.MODELS: Fable 5.1, Opus 5.5, Sonnet 5, Haiku 4.5), the same way.
CLAUDE_SOLO_ARMS = ("fable-solo", "opus-solo", "sonnet-solo", "haiku-solo")
SOLO_ARMS = SOLO_ARMS + CLAUDE_SOLO_ARMS
DECIDERS.update({arm: arm[:-len("-solo")] for arm in CLAUDE_SOLO_ARMS})
MODES = MODES + SOLO_ARMS


@dataclass(frozen=True)
class SearchBudget:
    """
    How much search one decision of an arm gets. The player's own search_ms and deadline are
    today's budget; an arm with a SearchBudget (the deep arm) gets this one instead.

    search_ms       Foul Play's search time per sampled world (it halves it early in a game, when
                    it samples twice as many worlds)
    worlds_x        Foul Play's parallelism: worlds_x times as many sampled opponent teams (4 x
                    worlds_x early in a game, 2 x worlds_x later), searched one after the other in
                    the battle's helper, so the decision takes about worlds_x times as long
    threads         poke-engine threads per world (1: one core, as today)
    workers         the worlds searched this many at a time, each in a process of Foul Play's own
                    pool forked from the helper, one core each (1: one after the other in the
                    helper, as today). The same search of every world, so the same compute and
                    iterations, in about 1/workers of the time: a machine with many slower cores
                    gives the deep arm its iterations well inside the battle timer.
    deadline        seconds from the request to the order
    min_time_left   the battle timer: once the server says fewer seconds than this are left (this
                    turn or in all), the arm plays today's budget for the rest of the game, noted
    max_decision_s  a decision of the arm that took longer than this (a busy machine): today's
                    budget for the rest of the game, noted
    decide_s        seconds on top of the search for a model to pick (the arms of DECIDERS): once
                    the search has answered (or its deadline has passed) the model gets this long.
                    0: the model gets what the deadline leaves (the Luna arm: about 1.2 s of 2.5).
    lines_ms        > 0: after the search, about this many ms more for LINES: our lines_k options
                    the search favours most, each played forward lines_turns whole turns in the
                    lines_worlds likeliest sampled worlds (bench/fp_bridge_server.py "lines"), each
                    of their searches on lines_threads threads of the engine's own (a line's step
                    gets about 36 ms: with 4 threads it sees a good 2 times as much). The deadline
                    has to hold them too.

    The clock (pokemon-showdown server/room-battle.ts, ladder battles): 150 s plus 60 s of grace at
    the start, 10 s added at every new turn up to 150, 5 s at a request in the middle of a turn
    (a forced switch, U-turn), and 5 s taken at every 5 s tick while our order is not in. A
    decision under 10 s costs at most one tick (5 s): +10 -5 on a turn, +5 -5 on a forced switch,
    so the bank never falls before turn 200 (from turn 100 the turn adds 5: +5 -5). The deep
    budget's deadline is 7.5 s; its search answered in 5.8 s on average (6.1 s at the slowest) on the box.
    """
    search_ms: int = 500
    worlds_x: int = 1
    threads: int = 1
    deadline: float = 2.5
    min_time_left: float = 90.0
    max_decision_s: float = 9.0
    workers: int = 1
    decide_s: float = 0.0
    lines_ms: int = 0
    lines_k: int = 3
    lines_turns: int = 5
    lines_worlds: int = 2
    lines_threads: int = 4

    def __post_init__(self) -> None:
        if (int(self.search_ms) < 2 or int(self.worlds_x) < 1 or int(self.threads) < 1 or int(self.workers) < 1
                or float(self.deadline) <= 0):
            raise ValueError("a search budget needs search_ms >= 2, worlds_x >= 1, threads >= 1, workers >= 1 and a deadline")
        if float(self.decide_s) < 0 or int(self.lines_ms) < 0 or (int(self.lines_ms) > 0 and (
                not 1 <= int(self.lines_k) <= 6 or not 1 <= int(self.lines_turns) <= 8 or not 1 <= int(self.lines_worlds) <= 4)):
            raise ValueError("a search budget needs decide_s >= 0, and with lines_ms: lines_k 1 to 6, lines_turns 1 to 8, lines_worlds 1 to 4")
        least = minimum_deadline(self.search_ms, self.worlds_x, self.workers) + self.lines_s()
        if float(self.deadline) < least:
            raise ValueError("a search of %d ms on %dx the worlds%s%s is never started with a deadline of %.2f s: it must be at least %.2f s"
                             % (self.search_ms, self.worlds_x, (" on %d workers" % self.workers) if self.workers > 1 else "",
                                (" with %d ms of lines" % self.lines_ms) if int(self.lines_ms) > 0 else "",
                                self.deadline, least))

    def lines_s(self) -> float:
        """What the helper reckons for the lines (a quarter over their budget), in seconds."""
        return 1.25 * max(0, int(self.lines_ms)) / 1000.0

    def lines_spec(self) -> Optional[Dict[str, Any]]:
        """The "lines" of FoulPlayBridge.advice_future, or None without lines."""
        if int(self.lines_ms) <= 0:
            return None
        return {"k": int(self.lines_k), "turns": int(self.lines_turns), "ms": float(self.lines_ms), "worlds": int(self.lines_worlds),
                "threads": max(1, int(self.lines_threads))}

    def compute(self) -> float:
        """Search time per decision relative to search_ms=1 on one world: search_ms x worlds_x x threads (threads buy about 1.5x per doubling)."""
        return float(self.search_ms) * int(self.worlds_x) * int(self.threads)

    def as_dict(self) -> Dict[str, Any]:
        out = {"ms": int(self.search_ms), "worlds_x": int(self.worlds_x), "threads": int(self.threads),
               "deadline": float(self.deadline)}
        if int(self.workers) > 1:
            out["workers"] = int(self.workers)
        if float(self.decide_s) > 0:
            out["decide_s"] = float(self.decide_s)
        if int(self.lines_ms) > 0:
            out["lines"] = self.lines_spec()
        return out

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
    # veto only: the moves the search may choose among are those the network gives at least this
    # (and its likeliest). 0: every legal move, as before. Mode "mix-trust" sets 0.03; ladder.py --trust
    # sets it for the player's every mode (the app's guarded mode with a wider search: mix-deep).
    trust: float = 0.0
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
    # The guard (veto rule only, OFF unless asked for: mode "mix-guard" or guard=True). It fires
    # when the opponent's active Pokemon is at +guard_boost or more in attack, defence, special
    # attack, special defence or speed, or boosted at all with a set-up move as its last action,
    # or when the read of its action (opponent_model.py, else the search's own shares) puts at
    # least guard_read on a set-up move or on a sleep move; and when the search's favourite is a
    # passive move of ours (our own Swords Dance, say) while the network gives at least
    # guard_sure to an action that is not. bench/notes/predict_live.md says what it does and
    # which lost games it comes from.
    guard: bool = False
    guard_boost: int = 2
    guard_read: float = 0.40
    guard_sure: float = 0.70
    # The set-up fixes of the veto rule (bench/notes/setup_fix.md, from the 14 ladder games lost
    # to a set-up sweep and the set-up puzzles). Each is OFF unless asked for; with all of them
    # off mix() plays exactly what it played before. They apply when the guard does not fire.
    #
    # no_third: never an action that neither side picked. When the search turned away from the
    # network's likeliest move, the search's best is played, not the network's next-likeliest
    # among the actions left (the plain veto did that on 108 of 535 overrides in the puzzles).
    no_third: bool = False
    # sure_attack (> 0): an attack the network gives at least this, estimated to knock the
    # opponent's active Pokemon out (Threat.ko; not counted when ko_doubt has a doubt: full HP, a
    # known Assault Vest or Eviolite, an ability that could stop it), is played when the search's favourite is a
    # passive move of ours (a set-up move favoured by the search does not veto it: the confident
    # Shadow Sneak vetoed for a third Swords Dance). 0: off.
    sure_attack: float = 0.0
    # tera_own: when both forms of a move are left, Terastallize only when the Tera form is one
    # side's own pick: the search's best, or the network's own likeliest action (not merely the
    # larger share of a move neither picked). When the search vetoed the plain form and not the
    # Tera form, the Tera form is the only one left and is played (the plain form was vetoed).
    tera_own: bool = False
    # setup_defer: in a set-up position (setup_reason: the opponent's active at +guard_boost, or
    # boosted with a set-up move as its last action, or a set-up read of guard_read or more) the
    # search's best is played. setup_switch_sure (> 0): except a switch the network gives at least
    # that much, which is played instead.
    setup_defer: bool = False
    setup_switch_sure: float = 0.0
    # Round 2's set-up rules (bench/setup_puzzles/ROUND2_PLAN.md and ROUND2.md on branch setupfix).
    # Each is OFF unless asked for, and each acts only in a set-up position (setup_reason, the live
    # trigger: the opponent's active at +guard_boost, or boosted with a set-up move as its last
    # action, or a set-up read of guard_read or more). Elsewhere, and with all of them off, mix()
    # plays exactly what it played before. They come before the guard; with the guard on they
    # choose among the actions the guard leaves (no passive move, no Sucker Punch into a status read).
    #
    # setup_switch_trust (> 0): the network's likeliest action, when it is a switch the network gives
    # at least this, is played: the search does not veto it.
    setup_switch_trust: float = 0.0
    # setup_net_trust (> 0): the network's likeliest action of any kind (a move with its Tera form
    # counted together) is played when the network gives it at least this.
    setup_net_trust: float = 0.0
    # setup_blend (> 0): the action with the largest log p(network) + setup_blend * share / largest
    # share (the search's visit share, normalised so that its favourite has 1).
    setup_blend: float = 0.0
    # setup_keep (> 0): the veto's keep in a set-up position, in place of keep (lower: vetoes less).
    setup_keep: float = 0.0
    # guard_status (> 0): with the guard, Sucker Punch is left out when the read puts at least this
    # on a status move (0: guard_read, as before).
    guard_status: float = 0.0
    # Round 3's counter rules (branch counterfix), OFF unless asked for. They need the Board of the
    # decision (setup_counter.build_board); without one they do nothing. Any rule; never in a forced switch.
    #
    # setup_counter: in a set-up position (counter_reason: the opponent's active at +guard_boost, or
    # boosted with a set-up move as its last action, or a set-up read of counter_read or more, or any
    # stage of +2 in attack, defences, speed or evasion), unless an attack of ours is estimated to
    # knock it out (Threat.counter_ko), play our active's working counter (the network's likeliest of
    # them). counter_switch: else switch to the healthiest benched Pokemon with at least counter_hp
    # of its HP, not two-hit KO'd, that holds a working counter or Unaware (setup_counter.Board.holders);
    # off since round 3's review (the switches were 2.5 of the 3.0 points the puzzles lost). Mix's
    # own choice stands when it is already one of those.
    setup_counter: bool = False
    counter_read: float = 0.50
    counter_hp: float = 0.50
    counter_switch: bool = False
    # setup_other_side: an attack into a defence the opponent's active has boosted (+1 or more, and
    # more than the other) gives way to the legal attack, in the same Tera form, that hits the other
    # defence (or neither) and is estimated to do at least other_side_ratio times as much, each
    # damage times its accuracy; never one that knocks itself out, recharges, locks in, charges,
    # switches out, works only at times or has a category that depends on the position
    # (setup_counter.Board.no_swap). Mix's attack stays when it is a working counter (Dragon Tail),
    # in KO range (Threat.counter_ko) where the other is not, or a priority move.
    setup_other_side: bool = False
    other_side_ratio: float = 1.00

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError("rule must be one of %s, not %r" % (", ".join(RULES), self.rule))

    @property
    def fixes(self) -> bool:
        """Whether any of the set-up fixes is on (the veto rule only)."""
        return self.rule == "veto" and (self.no_third or self.sure_attack > 0 or self.tera_own or self.setup_defer)

    @property
    def setup_rules(self) -> bool:
        """Whether one of round 2's set-up rules is on (the veto rule only)."""
        return self.rule == "veto" and (self.setup_switch_trust > 0 or self.setup_net_trust > 0 or self.setup_blend > 0
                                        or self.setup_keep > 0)

    @property
    def board_rules(self) -> bool:
        """Whether one of round 3's counter rules is on (they read a setup_counter.Board)."""
        return bool(self.setup_counter or self.setup_other_side)

    def without_fixes(self) -> "MixSettings":
        return replace(self, no_third=False, sure_attack=0.0, tera_own=False, setup_defer=False, setup_switch_sure=0.0,
                       setup_switch_trust=0.0, setup_net_trust=0.0, setup_blend=0.0, setup_keep=0.0,
                       setup_counter=False, setup_other_side=False)


@dataclass(frozen=True)
class Threat:
    """
    What the guard knows at one decision (the player fills it from the battle and the read).

    boost        the opponent's active Pokemon's largest positive stat stage (atk, def, spa, spd, spe), 0 when none
    last_setup   its last action (this turn's so far, else last turn's) was a set-up move
    setup_read   the probability the read gives a set-up move of it this turn (Swords Dance, Calm Mind, Curse...)
    sleep_read   the same for a sleep move (0 when ours is already statused and cannot fall asleep)
    status_read  the same for any status move: Sucker Punch fails against those
    passive      our actions (0-3, 10-13) that do nothing to its Pokemon: our own set-up, recovery,
                 Protect, Substitute, hazards, screens (setup_moves.is_passive)
    fails        our actions that fail against a status move (Sucker Punch, Thunderclap)
    ko           our attacks (0-3, 10-13) estimated to knock its Pokemon out at the lowest damage
                 roll (ko_estimate; filled only when MixSettings.sure_attack is on)
    counter_ko   the same for the counter rules (filled only with MixSettings.setup_counter): at full
                 HP counted when the estimate is twice its HP, or Sturdy, Focus Sash, Multiscale,
                 Shadow Shield, Disguise and Ice Face are ruled out (setup_counter.full_hp_guard)
    """
    boost: int = 0
    last_setup: bool = False
    setup_read: float = 0.0
    sleep_read: float = 0.0
    status_read: float = 0.0
    passive: Tuple[int, ...] = ()
    fails: Tuple[int, ...] = ()
    ko: Tuple[int, ...] = ()
    counter_ko: Tuple[int, ...] = ()


def setup_reason(threat: Optional[Threat], settings: MixSettings) -> str:
    """Why this is a set-up position ("boosted +2", "set up last turn", "set-up read 55%"), or "". guard_reason without sleep."""
    if threat is None:
        return ""
    if threat.boost >= max(1, settings.guard_boost):
        return "boosted +%d" % threat.boost
    if threat.boost >= 1 and threat.last_setup:
        return "set up last turn"
    if threat.setup_read >= settings.guard_read:
        return "set-up read %d%%" % round(100 * threat.setup_read)
    return ""


def counter_reason(threat: Optional[Threat], board: Any, settings: MixSettings) -> str:
    """
    Why "use the counter" looks at this decision, or "": the live set-up trigger (setup_reason with
    the read at settings.counter_read) or, from the board, a stage of +2 or more in attack,
    defences, special attack, speed or evasion.
    """
    why = setup_reason(threat, replace(settings, guard_read=settings.counter_read))
    if why:
        return why
    top = int(getattr(board, "top_boost", 0) or 0)
    if top >= 2:
        return "boosted +%d" % top
    return ""


@dataclass(frozen=True)
class ThinkLonger:
    """
    A longer search on set-up turns (ladder.py --think-longer MS; OFF unless given).

    It fires BEFORE the search is asked when the opponent's active Pokemon is at +boost or more
    in attack, defences or speed, or its last action was a set-up move (then the one search of
    the decision is the long one), and AFTER the ordinary answer when the search's own read of
    the opponent puts `read` or more on a set-up move (then the search is asked again, at `ms`,
    and its answer replaces the ordinary one when it comes in time).

    ms         the search time of the long search
    deadline   seconds from the long question to its answer at the latest (0: minimum_deadline(ms) + 1.0;
               6.4 s at 2,000 ms, where a real helper answered in 4.7 s, 5.3 s at the most with 2 threads)
    threads    poke-engine threads for the long search when a helper is free (more helpers than
               battles in progress and long searches using a second thread): the free helper's
               core searches the same battle. 1: never.
    min_time_left  the battle's own timer: not when the server says fewer seconds than this
               plus the long deadline are left (this turn, or in all).
    """
    ms: int = 2000
    boost: int = 1
    read: float = 0.40
    deadline: float = 0.0
    threads: int = 2
    min_time_left: float = 30.0

    def wait_s(self) -> float:
        return float(self.deadline) if self.deadline > 0 else minimum_deadline(self.ms) + 1.0


_STAGE = {s: (2 + s) / 2.0 if s >= 0 else 2.0 / (2 - s) for s in range(-6, 7)}


def _stat(base: float, level: int, hp: bool = False) -> int:
    """A random battle stat: 84 EVs, 31 IVs, a neutral nature."""
    core = int((2 * float(base) + 31 + 21) * int(level) / 100)
    return core + int(level) + 10 if hp else core + 5


def ko_estimate(move: Any, attacker: Any, defender: Any, roll: float = 0.85) -> Optional[float]:
    """
    The damage of `move` (poke-env Move) from `attacker` (our poke-env Pokemon, its stats known
    from the request) to `defender` (theirs: its stats estimated as a random battle set) as a
    share of the defender's whole HP, at the given damage roll; None when it cannot be told.

    Rough on purpose: base power, the two stats with their stages, STAB, type effectiveness,
    burn. No items, abilities, weather, screens, critical hits or multi-hit. Fixed damage
    (Seismic Toss, Night Shade) is the attacker's level.
    """
    try:
        category = getattr(getattr(move, "category", None), "name", "")
        if category not in ("PHYSICAL", "SPECIAL"):
            return None
        level = int(getattr(attacker, "level", 0) or 0)
        d_level = int(getattr(defender, "level", 0) or 0)
        d_base = dict(getattr(defender, "base_stats", None) or {})
        if not level or not d_level or "hp" not in d_base:
            return None
        d_hp = _stat(d_base["hp"], d_level, hp=True)
        mult = float(defender.damage_multiplier(move))
        if mult <= 0:
            return 0.0
        if getattr(move, "damage", None) == "level":
            return level * mult / d_hp if mult > 0 else 0.0
        power = float(getattr(move, "base_power", 0) or 0)
        if power <= 0:
            return None
        a_key, d_key = ("atk", "def") if category == "PHYSICAL" else ("spa", "spd")
        a = float((getattr(attacker, "stats", None) or {}).get(a_key) or 0)
        if a <= 0:
            return None
        d = float(_stat(d_base[d_key], d_level))
        a *= _STAGE.get(int((getattr(attacker, "boosts", None) or {}).get(a_key, 0)), 1.0)
        d *= _STAGE.get(int((getattr(defender, "boosts", None) or {}).get(d_key, 0)), 1.0)
        damage = int(int(int(2 * level / 5 + 2) * power * a / d) / 50) + 2
        types = {getattr(t, "name", str(t)) for t in (getattr(attacker, "types", None) or ()) if t is not None}
        if getattr(attacker, "is_terastallized", False) and getattr(attacker, "tera_type", None) is not None:
            types.add(getattr(attacker.tera_type, "name", str(attacker.tera_type)))
        move_type = getattr(getattr(move, "type", None), "name", "")
        stab = 1.5 if move_type and move_type in types else 1.0
        burned = getattr(getattr(attacker, "status", None), "name", "") == "BRN" and category == "PHYSICAL"
        return damage * stab * mult * (0.5 if burned else 1.0) * float(roll) / d_hp
    except Exception:
        return None


# Abilities that make a move of this type do nothing or far less (a KO estimate without abilities
# cannot be trusted against them). Wonder Guard: everything but a super-effective hit.
_KO_ABILITIES: Dict[str, Tuple[str, ...]] = {
    "levitate": ("GROUND",), "eartheater": ("GROUND",), "flashfire": ("FIRE",), "wellbakedbody": ("FIRE",),
    "heatproof": ("FIRE",), "waterbubble": ("FIRE",), "thickfat": ("FIRE", "ICE"), "waterabsorb": ("WATER",),
    "stormdrain": ("WATER",), "dryskin": ("WATER",), "voltabsorb": ("ELECTRIC",), "lightningrod": ("ELECTRIC",),
    "motordrive": ("ELECTRIC",), "sapsipper": ("GRASS",), "purifyingsalt": ("GHOST",), "wonderguard": ("*",),
}
_KO_CATEGORY_ABILITIES: Dict[str, str] = {"furcoat": "PHYSICAL", "fluffy": "PHYSICAL", "icescales": "SPECIAL"}


def ko_doubt(move: Any, defender: Any, full_hp: bool = True) -> str:
    """
    Why a KO that ko_estimate calls may not happen (then none is counted), or "":
    "full HP" (Sturdy, Focus Sash, Multiscale, Shadow Shield, Disguise are common in random battles
    and only matter there), "Assault Vest" (known, against a special move), "Eviolite" (known, or
    an unknown item on a Pokemon that still evolves), or the ability that could stop it.
    full_hp=False leaves out the "full HP" doubt (the counter rules judge it themselves).
    """
    try:
        if full_hp and float(getattr(defender, "current_hp_fraction", 0.0) or 0.0) >= 1.0:
            return "full HP"
        category = getattr(getattr(move, "category", None), "name", "")
        item = str(getattr(defender, "item", "") or "")
        if item == "assaultvest" and category == "SPECIAL":
            return "Assault Vest"
        if item == "eviolite":
            return "Eviolite"
        if item in ("", "unknown_item"):
            try:
                entry = getattr(getattr(defender, "_data", None), "pokedex", {}).get(getattr(defender, "species", ""), {})
                if entry.get("evos"):
                    return "Eviolite"
            except Exception:
                pass
        move_type = getattr(getattr(move, "type", None), "name", "")
        known = str(getattr(defender, "ability", "") or "")
        abilities = [known] if known else [str(a) for a in (getattr(defender, "possible_abilities", None) or ())]
        for ab in abilities:
            types = _KO_ABILITIES.get(ab)
            if types and ("*" in types or move_type in types):
                return ab
            if _KO_CATEGORY_ABILITIES.get(ab) == category:
                return ab
        return ""
    except Exception:
        return "unknown"


def guard_reason(threat: Optional[Threat], settings: MixSettings) -> str:
    """Why the guard fires on the opponent's side ("boosted +2", "set up last turn", "set-up read 55%", "sleep read 61%"), or ""."""
    if threat is None:
        return ""
    if threat.boost >= max(1, settings.guard_boost):
        return "boosted +%d" % threat.boost
    if threat.boost >= 1 and threat.last_setup:
        return "set up last turn"
    if threat.setup_read >= settings.guard_read:
        return "set-up read %d%%" % round(100 * threat.setup_read)
    if threat.sleep_read >= settings.guard_read:
        return "sleep read %d%%" % round(100 * threat.sleep_read)
    return ""


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
    if mode in SOLO_ARMS:
        return "net", None              # no search: the model alone decides (and the network's pick is not played)
    if mode == "mix" or mode == DEEP_ARM or mode in DECIDERS or mode in LINES_ARMS:
        # The deep arm is mix with a larger SearchBudget, a model's arm mix with the model deciding
        # after it, a lines arm mix with the lines read after the search.
        return "mix", None
    if mode == "mix-guard":
        return "mix", "veto"            # the guard is a change of the veto rule
    if mode in ARM_FIXES:
        return "mix", "veto"            # the set-up fixes are changes of the veto rule
    if mode.startswith("mix-") and mode[4:] in RULES:
        return "mix", mode[4:]
    raise ValueError("mode must be one of %s, not %r" % (", ".join(MODES), mode))


def guarded(mode: str, settings: MixSettings) -> bool:
    """Whether a battle in this mode is played with the guard: "mix-guard", or a veto "mix" with settings.guard."""
    if mode == "mix-guard":
        return True
    kind, rule = split_mode(mode)
    return kind == "mix" and bool(settings.guard) and (rule or settings.rule) == "veto"


def check_modes(modes: Sequence[str]) -> List[str]:
    out = [str(m).strip() for m in modes if str(m).strip()]
    if not out:
        raise ValueError("no mode given")
    for mode in out:
        split_mode(mode)
    return out


def minimum_deadline(search_ms: int, worlds_x: int = 1, workers: int = 1) -> float:
    """
    The shortest deadline with which a search of search_ms is ever started: the helper refuses a
    question it could not answer in time, and it reckons 2.5 x search_ms (x worlds_x, Foul Play's
    parallelism) + 20 ms for an answer (fp_bridge_server.search_seconds; less with workers, the
    worlds searched that many at a time: fb.search_seconds). A third of a second on top for the
    wait in front of it.
    """
    return fb.search_seconds(search_ms, worlds_x, workers) + RESERVE_S + 0.3


# The deep arm's budget: 2.5 times the search per world and twice the worlds, about 5 times
# today's (500 ms on 1x the worlds). The search answers in about 5.8 s (box, 12 decisions; 4.5 to 4.7
# times mix's iterations, 6 worlds against 2.7); the deadline is 7.5 s
# (minimum_deadline: 6.62 s), so with the order spacing a decision stays under the second 5 s
# tick of the battle timer (10 s): see SearchBudget.
DEEP_BUDGET = SearchBudget(search_ms=1250, worlds_x=2, threads=1, deadline=7.5)

# The lines arms' budget: today's search (500 ms per world, about 1.3 s), then 1.5 s of lines: the
# search's 3 favourite actions, 5 turns each, in the 2 likeliest worlds (some 40 searches of about
# 36 ms). The helper reckons 1.27 + 1.88 s; the deadline is 4.6 s.
LINES_BUDGET = SearchBudget(search_ms=500, worlds_x=1, threads=1, deadline=4.6, lines_ms=1500, lines_k=3, lines_turns=5,
                            lines_worlds=2)
# A model's seconds on top of the search (SearchBudget.decide_s). Luna answers in about 0.15 s; Sol
# is asked for up to 5 s, and a decision of a Sol arm still stays under the second 5 s tick of the
# battle timer (4.6 + 5.0 s at the very worst, about 5 s as a rule): see SearchBudget.
LUNA_DECIDE_S = 1.3
SOL_DECIDE_S = 5.0
ARM_BUDGETS: Dict[str, SearchBudget] = {
    DEEP_ARM: DEEP_BUDGET,
    SOL_ARM: SearchBudget(search_ms=500, worlds_x=1, threads=1, deadline=2.5, decide_s=SOL_DECIDE_S, max_decision_s=12.0),
    LINES_ARM: LINES_BUDGET,
    "luna-lines": replace(LINES_BUDGET, decide_s=LUNA_DECIDE_S),
    "sol-lines": replace(LINES_BUDGET, decide_s=SOL_DECIDE_S, max_decision_s=12.0),
}


@dataclass(frozen=True)
class GameSetting:
    """
    One named setting of a run with settings per game (ladder.py --arm-settings, SettingPicker):
    everything a game of it searches with that a player otherwise has for all its games, as the
    player would have it if it had been made for this setting alone.

    name        what the game's records call it ("setting")
    mode        the game's mode (mix-trust, mix-deep, ...)
    search_ms   the player's search_ms: what a mode without a budget of its own searches, and what
                a mode with one falls back to
    deadline    the player's deadline, the same way
    trust       MixSettings.trust for the game (an ARM_FIXES arm such as mix-trust still sets its own)
    budget      the mode's own SearchBudget (the deep arm, the lines arms); None: the mode has
                none, or ARM_BUDGETS' when the mode has one there
    """
    name: str
    mode: str
    search_ms: int = 500
    deadline: float = 2.5
    trust: float = 0.0
    budget: Optional[SearchBudget] = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("a setting needs a name")
        kind = split_mode(self.mode)[0]
        if int(self.search_ms) < 2 or float(self.deadline) <= 0:
            raise ValueError("setting %s: search_ms is at least 2 and the deadline above 0" % self.name)
        if not 0.0 <= float(self.trust) < 1.0:
            raise ValueError("setting %s: trust is from 0 up to 1" % self.name)
        if kind != "net" and float(self.deadline) < minimum_deadline(self.search_ms):
            raise ValueError("setting %s: a search of %d ms is never started with a deadline of %.2f s: it must be at least %.2f s"
                             % (self.name, self.search_ms, self.deadline, minimum_deadline(self.search_ms)))

    def own_budget(self) -> Optional[SearchBudget]:
        """The mode's own SearchBudget for a game of this setting (None: search_ms within deadline)."""
        return self.budget if self.budget is not None else ARM_BUDGETS.get(self.mode)

    def today(self) -> Dict[str, Any]:
        """search_ms within deadline, as SearchBudget.as_dict() writes a budget."""
        return {"ms": int(self.search_ms), "worlds_x": 1, "threads": 1, "deadline": float(self.deadline)}

    def asks(self) -> Dict[str, Any]:
        """The budget a decision of this setting is asked with (until a fall back): the mode's own, else today()."""
        own = self.own_budget()
        return own.as_dict() if own is not None else self.today()


@dataclass(frozen=True)
class LinesRule:
    """
    The arm "mix-lines": when mix's pick gives way to another action because of the lines.

    margin     the other action's line has to be worth this much more than the line of mix's pick
               (the search's mean result at the line's end, 0 lost to 1 won)
    keep       and the other action needs this share of the search's largest share: one the search
               itself rates (the veto rule's own bar)
    min_turns  lines that did not reach this many whole turns (the time ran out) are not compared
    agree      and in no sampled world that has both lines may the other action's line end worse
               than mix's pick's: one line is one likely future, and on the real engine the same
               action's two lines differ by 0.1 and more between two worlds (bench/notes/lines.md)
    First guesses, not tuned values: the arm is there to be measured against mix.
    """
    margin: float = 0.10
    keep: float = 0.25
    min_turns: int = 2
    agree: bool = True


def lines_pick(mix_action: int, share: Sequence[float], lines: Dict[int, Dict[str, Any]],
               rule: Optional[LinesRule] = None) -> Tuple[int, Dict[str, Any]]:
    """
    (the action the arm "mix-lines" plays, what the lines said). lines: Advice.lines_by_action,
    action -> {"value", "turns", ...}. Mix's pick stands unless it has a line and another action
    with a line is rated by the search (rule.keep) and ends at least rule.margin better.
    """
    rule = rule or LinesRule()
    mix_action = int(mix_action)
    info: Dict[str, Any] = {"mix": mix_action, "played": mix_action, "changed": False, "why": "",
                            "values": {str(a): row.get("value") for a, row in sorted(lines.items())},
                            "turns": {str(a): row.get("turns") for a, row in sorted(lines.items())}}
    if not lines:
        info["why"] = "no_lines"
        return mix_action, info
    mine = lines.get(mix_action)
    if mine is None or mine.get("value") is None:
        info["why"] = "no_line_for_mix"
        return mix_action, info
    if int(mine.get("turns") or 0) < int(rule.min_turns):
        info["why"] = "short"
        return mix_action, info
    top = max((float(x) for x in share), default=0.0)
    best, best_value = mix_action, float(mine["value"])
    mine_by_world = {w.get("world"): w.get("value") for w in mine.get("worlds") or [] if w.get("value") is not None}
    split = False
    for a, row in lines.items():
        if int(a) == mix_action or row.get("value") is None or int(row.get("turns") or 0) < int(rule.min_turns):
            continue
        if not (0 <= int(a) < len(share)) or float(share[int(a)]) < float(rule.keep) * top:
            continue
        if rule.agree and any(w.get("world") in mine_by_world and w.get("value") is not None
                              and float(w["value"]) < float(mine_by_world[w.get("world")]) for w in row.get("worlds") or []):
            split = split or float(row["value"]) - float(mine["value"]) >= float(rule.margin)
            continue                    # a world in which this line ends worse than mix's pick's: not clearly better
        if float(row["value"]) > best_value:
            best, best_value = int(a), float(row["value"])
    gain = best_value - float(mine["value"])
    if best != mix_action and gain >= float(rule.margin):
        info.update(played=best, changed=True, gain=round(gain, 4), why="line")
        return best, info
    info["why"] = ("worlds_disagree" if split else "mix_best") if best == mix_action else "close"
    return mix_action, info


def full_width_turns(iterations: Any, worlds: Any, ours: Any, theirs: Any) -> Optional[float]:
    """
    How many turns ahead a search of this size would see if it looked at every pair of actions
    equally: log(iterations per world) / log(our options x the opponent's options). A Monte Carlo
    search goes deeper than this on the lines it favours and less deep elsewhere, so it is a
    yardstick for comparing budgets (the deep arm against mix), not the depth of any one line.
    None when it cannot be told.
    """
    try:
        per_world = float(iterations) / max(1, int(worlds or 1))
        branching = max(1, int(ours or 0)) * max(1, int(theirs or 0))
        if per_world < 2 or branching < 2:
            return None
        return round(math.log(per_world) / math.log(branching), 3)
    except Exception:
        return None


def value_fields(v_exp: Optional[float], v_std: Optional[float], phi: Optional[float], scale: float) -> Dict[str, Any]:
    """
    The network's own position score for a decision record. The value head was trained on the
    shaped return (rl_reward/shaping.py): V(s_t) = E[gamma^(T-t) outcome] - scale x Phi(s_t), so
    the network's estimate of the outcome (+1 win, -1 loss) is v_exp + scale x Phi, and of the
    chance to win (1 + v_exp + scale x Phi) / 2 (gamma 0.999 shrinks it by about 5% fifty decisions
    from the end). Phi is the potential the reward was shaped with (rl_reward.load_potential, read
    from the live battle as worker.py reads it). {} when the network gave no value.
    """
    if v_exp is None:
        return {}
    out: Dict[str, Any] = {"v_exp": round(float(v_exp), 4)}
    if v_std is not None:
        out["v_std"] = round(float(v_std), 4)
    out["phi"] = None if phi is None else round(float(phi), 4)
    out["phi_scale"] = float(scale)
    outcome = float(v_exp) + float(scale) * float(phi or 0.0)
    out["win_est"] = round(min(1.0, max(0.0, 0.5 * (1.0 + outcome))), 4) if phi is not None or not scale else None
    return out


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
    if s.trust > 0:
        # The guarded search (mode mix-trust): the veto works among the moves the network gives at
        # least `trust` (and always its likeliest), measured against the search's best of those. The
        # search can still turn the network from its first choice, never to a move the network all
        # but rules out. Over 518 ladder games those overrules (to a move under 2%) were followed by
        # a 2.8-point fall in the network's own win estimate against its own move's
        # (reports/2026-10-15/LOSS_STUDY.md).
        top = max(legal, key=lambda a: (q[a], -a))
        pool = [a for a in legal if q[a] >= s.trust or a == top]
        seen = [a for a in pool if view.visits[a] > 0]
        best = max(seen, key=lambda a: (view.share[a], view.visits[a], -a)) if seen else None
        floor = s.keep * view.share[best] if best is not None else 0.0
        kept = [a for a in pool if view.visits[a] <= 0 or view.share[a] >= floor]
        if best is not None and best not in kept:
            kept.append(best)
        return kept, lambda a: (q[a], view.share[a], -a)
    least = s.keep * view.share[view.best]

    def sound(a: int) -> bool:
        if view.visits[a] <= 0:
            return True                 # the search did not look at it (maybeTrapped, for one): no opinion, no veto
        return view.share[a] >= least

    kept = [a for a in legal if sound(a)]
    if view.best not in kept:
        kept.append(view.best)
    return kept, lambda a: (q[a], view.share[a], -a)


def _guarded_choice(q: Sequence[float], legal: List[int], view: SearchView, s: MixSettings, threat: Threat, why: str,
                    explain: Optional[Dict[str, Any]]) -> Tuple[int, List[int]]:
    """The veto rule with the guard fired: (the action, the candidates the Tera form is chosen among)."""
    drop = set()
    if not why.startswith("sleep"):
        drop |= set(threat.passive)          # no set-up, Roost, Protect or hazard of ours against a boosted or boosting Pokemon
    if threat.status_read >= (s.guard_status if s.guard_status > 0 else s.guard_read):
        drop |= set(threat.fails)            # no Sucker Punch into a likely status move
    usable = [a for a in legal if a not in drop] or list(legal)
    seen = [a for a in usable if view.visits[a] > 0]
    best = max(seen, key=lambda a: (view.share[a], view.visits[a], -a)) if seen else None
    # The veto is measured against the search's best USABLE action: a set-up move the search
    # favours does not veto the attack the network wants.
    least = s.keep * view.share[best] if best is not None else 0.0
    kept = [a for a in usable if view.visits[a] <= 0 or view.share[a] >= least]
    net_pick = max(usable, key=lambda a: (q[a], view.share[a], -a))
    # The network's own choice when the search did not turn away from it, else the search's best:
    # never a third action that neither of the two wanted.
    choice = net_pick if (net_pick in kept or best is None) else best
    if best is not None and best not in kept:
        kept.append(best)
    if explain is not None:
        explain.update({"guard": why, "dropped": sorted(a for a in drop if a in legal), "net_pick": int(net_pick),
                        "search_pick": None if best is None else int(best), "by": "net" if choice == net_pick else "search"})
    return choice, kept


def mix(probs: Sequence[float], mask: Sequence[float], view: SearchView, settings: Optional[MixSettings] = None,
        threat: Optional[Threat] = None, explain: Optional[Dict[str, Any]] = None, board: Any = None) -> int:
    """
    The action played from the network's probabilities and the search's statistics.

    probs   the network's probabilities over the legal actions (legal_probabilities).
    mask    the 14 legal-action flags.
    view    read_advice() of the search's answer.
    threat  what the guard knows (Threat), used only with settings.guard and the veto rule.
    explain a dict that is filled with what the guard did ("guard": why it fired or ""), and
            "board" with what the counter rules did when they looked.
    board   setup_counter.Board, used only with settings.setup_counter or setup_other_side: those
            two rules come last, on the action the rest of mix() chose (_board_rules).

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

    The guard (settings.guard, veto only), when guard_reason() fires, or when the search's
    favourite is one of our passive actions while the network gives guard_sure or more to an
    action that is not ("network sure"):
      * our passive actions (Threat.passive: set-up, recovery, Protect, hazards) are left out,
        unless it fired on a sleep read alone, and Sucker Punch is left out when the read puts
        guard_read or more on a status move. When that leaves nothing, nothing is left out;
      * the veto is measured against the search's largest share among the actions left;
      * the network's likeliest action left is played when the veto keeps it, else the search's
        largest share: never an action that neither of the two picked.
    """
    s = settings or MixSettings()
    legal = [a for a in range(ACTIONS) if mask[a] > 0.5]
    if not legal:
        return 0
    q = merged_prior(probs, mask)
    in_setup = bool(s.setup_rules and setup_reason(threat, s))
    if in_setup and s.setup_keep > 0:
        s = replace(s, keep=s.setup_keep)      # the veto (and the guard's veto) vetoes less in a set-up position
    why = guard_reason(threat, s) if (s.guard and s.rule == "veto") else ""
    if not why and s.guard and s.rule == "veto" and threat is not None and threat.passive:
        # Our own set-up (or Roost, Protect...) favoured by the search does not veto what the network is sure of.
        top = max(legal, key=lambda a: (q[a], -a))
        best = max((a for a in legal if view.visits[a] > 0), key=lambda a: (view.share[a], view.visits[a], -a), default=None)
        if q[top] >= s.guard_sure and top not in threat.passive and best in threat.passive:
            why = "network sure %d%%" % round(100 * q[top])
    if explain is not None:
        explain["guard"] = why
    picked = _setup_rule(q, _guard_usable(legal, s, threat, why) if why else legal, view, s) if in_setup else None
    if picked is not None:
        choice, candidates, rule_name = picked
        if explain is not None:
            explain["setup_rule"] = rule_name
    elif why:
        choice, candidates = _guarded_choice(q, legal, view, s, threat, why, explain)
    else:
        candidates, rank = _candidates(q, legal, view, s)
        choice = max(candidates, key=rank)
        if s.fixes:
            choice, candidates, fix = _setup_fixes(q, legal, view, s, threat, choice, candidates)
            if explain is not None:
                explain["fix"] = fix
    if choice <= 3 or choice >= 10:
        plain, tera = base_action(choice), base_action(choice) + 10
        if plain in candidates and tera in candidates:
            wants_tera = view.share[tera] > view.share[plain] and view.share[tera] >= s.tera_share
            if wants_tera and s.tera_own and s.rule == "veto":
                # The Tera form only when one side picked it itself: the search's best, or the
                # network's likeliest action (its own split of a move, before merged_prior).
                net_top = max(legal, key=lambda a: (float(probs[a]), -a))
                wants_tera = view.best == tera or net_top == tera
            choice = tera if wants_tera else plain
    if board is not None and s.board_rules:
        choice = _board_rules(q, legal, view, s, threat, board, choice, explain)
    return choice


def _board_rules(q: Sequence[float], legal: List[int], view: SearchView, s: MixSettings, threat: Optional[Threat], board: Any,
                 choice: int, explain: Optional[Dict[str, Any]]) -> int:
    """
    Round 3's counter rules on the action mix() chose (MixSettings.setup_counter, setup_other_side).
    Never in a forced switch; never an action that is not legal. explain["board"] gets
    {"counter": what "use the counter" saw and did, "other_side": the swap, or "other_side_kept": an
    attack into the boosted defence that stays, and why} when they looked.
    """
    if getattr(board, "forced", False):
        return choice
    before = choice
    note: Dict[str, Any] = {}
    if s.setup_counter:
        why = counter_reason(threat, board, s)
        if why:
            counter: Dict[str, Any] = {"trigger": why, "mix": int(before)}
            ko = [a for a in (threat.counter_ko if threat is not None else ()) if a in legal]
            working = sorted(a for a in (getattr(board, "counters", None) or {}) if a in legal and 0 <= a <= 3)
            holders = {a: v for a, v in (getattr(board, "holders", None) or {}).items() if a in legal and 4 <= a <= 9}
            if ko:
                counter.update(decided="kept: an attack is estimated to knock it out", ko=ko)
            elif getattr(board, "unaware", False):
                counter["decided"] = "kept: our active Pokemon has Unaware"
            elif working:
                if base_action(choice) in working:
                    counter["decided"] = "kept: mix plays a counter"
                else:
                    pick = max(working, key=lambda a: (q[a], view.share[a], -a))
                    choice = pick
                    counter.update(decided="played the counter", counter=board.counters[pick])
            elif holders and s.counter_switch:
                if choice in holders:
                    counter["decided"] = "kept: mix switches to a counter"
                else:
                    pick = max(holders, key=lambda a: (holders[a][0], q[a], view.share[a], -a))
                    choice = pick
                    counter.update(decided="switched to a counter", holder=holders[pick][1], hp=holders[pick][0])
            elif holders:
                counter["decided"] = "kept: no counter on the active (switches are off)"
            else:
                counter["decided"] = "kept: no counter can work"
            counter["pick"] = int(choice)
            note["counter"] = counter
    working_now = getattr(board, "counters", None) or {}
    if s.setup_other_side and choice == before and base_action(choice) not in working_now:
        # Never undo a working counter mix chose (Dragon Tail into +2 Def: the review's game 278391).
        keep_ko = [a for a in (threat.counter_ko if threat is not None else ()) if a in legal]
        looked = _other_side(legal, board, choice, s.other_side_ratio, keep_ko)
        if looked is not None and looked["pick"] != choice:
            note["other_side"] = looked
            choice = looked["pick"]
        elif looked is not None:
            note["other_side_kept"] = looked          # an attack into the boosted defence, and nothing to swap it for
    if choice not in legal:          # never: every pick above is a legal action
        choice = before
    if explain is not None and note:
        explain["board"] = note
    return choice


def _other_side(legal: List[int], board: Any, choice: int, ratio: float, ko: Sequence[int] = ()) -> Optional[Dict[str, Any]]:
    """
    "Don't hit the boosted defence" on mix's choice: None when it is not an attack into a defence
    the opponent has boosted (+1 or more, more than the other); else {"boosted", "mix", "pick",
    "damage": [mix's, the pick's], "why"}, where "pick" is mix's own choice when nothing can take
    its place (no attack of the other kind, none at ratio or more, no damage estimate, mix's
    attack in KO range (ko) where the other is not, or a priority move). Damage is compared
    times accuracy (Board.accuracy); "damage" records the plain estimates.
    """
    hits = getattr(board, "hits", None) or {}
    damage = getattr(board, "damage", None) or {}
    side = hits.get(choice, "")
    if side not in ("def", "spd"):
        return None
    boosts = getattr(board, "boosts", None) or {}
    other = "spd" if side == "def" else "def"
    mine, theirs = int(boosts.get(side, 0) or 0), int(boosts.get(other, 0) or 0)
    if mine < 1 or mine <= theirs:
        return None
    out: Dict[str, Any] = {"boosted": "%s +%d" % (side, mine), "mix": int(choice), "pick": int(choice),
                           "damage": [round(float(damage[choice]), 4) if choice in damage else None, None]}
    if choice not in damage:
        out["why"] = "no damage estimate for mix's attack"
        return out
    if int((getattr(board, "priority", None) or {}).get(choice, 0) or 0) > 0:
        out["why"] = "mix's attack has priority"
        return out
    tera = choice >= 10
    never = getattr(board, "no_swap", None) or {}          # Explosion, Hyper Beam, U-turn, Sucker Punch... (setup_counter.swap_doubt)
    other_kind = [a for a in legal if a != choice and (a >= 10) == tera and (a <= 3 or a >= 10) and a in hits and hits[a] != side]
    alts = [a for a in other_kind if a in damage and damage[a] > 0 and a not in never]
    if not alts:
        if not other_kind:
            out["why"] = "no attack of the other kind"
        elif all(a in never for a in other_kind):
            out["why"] = "the other kind is never swapped in (%s)" % ", ".join(sorted({str(never[a]) for a in other_kind}))
        else:
            out["why"] = "no damage estimate for the other kind, or it does nothing"
        return out
    accuracy = getattr(board, "accuracy", None) or {}

    def expected(a: int) -> float:
        return float(damage[a]) * float(accuracy.get(a, 1.0))

    if choice in ko:
        alts = [a for a in alts if a in ko]
        if not alts:
            out["why"] = "mix's attack is in KO range and the other kind is not"
            return out
    best = max(alts, key=lambda a: (expected(a), -a))
    out["damage"][1] = round(float(damage[best]), 4)
    if expected(best) < ratio * expected(choice):
        out.update(why="the best of the other kind does %d%% of it" % round(100 * expected(best) / max(1e-9, expected(choice))),
                   best=int(best))
        return out
    out.update(pick=int(best), why="")
    return out


def _guard_usable(legal: List[int], s: MixSettings, threat: Optional[Threat], why: str) -> List[int]:
    """The actions the guard leaves when it fires (as _guarded_choice): no passive move, no Sucker Punch into a status read."""
    if threat is None:
        return list(legal)
    drop = set()
    if not why.startswith("sleep"):
        drop |= set(threat.passive)
    if threat.status_read >= (s.guard_status if s.guard_status > 0 else s.guard_read):
        drop |= set(threat.fails)
    return [a for a in legal if a not in drop] or list(legal)


def _setup_rule(q: Sequence[float], usable: List[int], view: SearchView, s: MixSettings) -> Optional[Tuple[int, List[int], str]]:
    """
    Round 2's set-up rules in a set-up position: (the action, the candidates its Tera form is
    chosen among, the rule's name), or None when none of them decides (the veto, or the guard, does).
    """
    if not usable:
        return None
    net_pick = max(usable, key=lambda a: (q[a], view.share[a], -a))
    if s.setup_net_trust > 0 and q[net_pick] >= s.setup_net_trust:
        return net_pick, _forms(net_pick, usable), "set-up: the network's sure action"
    if s.setup_switch_trust > 0 and 4 <= net_pick <= 9 and q[net_pick] >= s.setup_switch_trust:
        return net_pick, [net_pick], "set-up: the network's sure switch"
    if s.setup_blend > 0:
        top = max((view.share[a] for a in usable), default=0.0)
        if top > 0:
            def value(a: int) -> Tuple:
                return (math.log(max(q[a], 1e-6)) + s.setup_blend * view.share[a] / top, view.share[a], -a)
            choice = max(usable, key=value)
            return choice, _forms(choice, usable), "set-up: blend"
    return None


def _forms(action: int, legal: Sequence[int]) -> List[int]:
    """The legal forms of a move (plain and Tera), or the switch itself."""
    if 4 <= action <= 9:
        return [action]
    return [a for a in (base_action(action), base_action(action) + 10) if a in legal]


def _setup_fixes(q: Sequence[float], legal: List[int], view: SearchView, s: MixSettings, threat: Optional[Threat],
                 choice: int, candidates: List[int]) -> Tuple[int, List[int], str]:
    """
    The veto rule's choice with the set-up fixes of MixSettings (no_third, sure_attack,
    setup_defer): (the action, the candidates the Tera form is chosen among, which fix decided or "").
    """
    net_pick = max(legal, key=lambda a: (q[a], view.share[a], -a))
    best = view.best
    if s.setup_defer and setup_reason(threat, s):
        if s.setup_switch_sure > 0 and 4 <= net_pick <= 9 and q[net_pick] >= s.setup_switch_sure:
            return net_pick, list(candidates) + [net_pick], "set-up: the network's sure switch"
        return best, list(candidates) + [best], "set-up: the search's best"
    if (s.sure_attack > 0 and threat is not None and best in threat.passive and q[net_pick] >= s.sure_attack
            and net_pick not in threat.passive and any(a in threat.ko for a in _forms(net_pick, legal))):
        return net_pick, list(candidates) + _forms(net_pick, legal), "sure attack in KO range"
    if s.no_third and base_action(choice) != base_action(net_pick) and base_action(choice) != base_action(best):
        return best, list(candidates) + [best], "no third action"
    return choice, candidates, ""


# ---------------------------------------------------------------------------------------------
# Predictions in words: what ladder.py prints and the battle file keeps
# ---------------------------------------------------------------------------------------------

def short_tag(tag: str) -> str:
    """"battle-gen9randombattle-2694021408" (and a private room's "-...pw" suffix) -> "1408": a game's id on the console."""
    import re

    numbers = re.findall(r"-(\d+)", str(tag or ""))
    return numbers[0][-4:] if numbers else str(tag or "?")[-4:]


def parse_time_left(text: str) -> Optional[float]:
    """
    The seconds the timer leaves us, from the server's "|inactive|" line to the player: "Time
    left: 150 sec this turn | 150 sec total" (and "| 30 sec grace") -> 150.0, the smaller of the
    turn's and the total; None for any other line.
    """
    import re

    text = str(text or "")
    if not text.startswith("Time left:"):
        return None
    numbers = [float(n) for n in re.findall(r"(\d+) sec (?:this turn|total)", text)]
    return min(numbers) if numbers else None


def option_text(name: str, user_types: Optional[Sequence[str]] = None) -> str:
    """One of the opponent's options in words: "move sleeppowder" -> "Sleep Powder (sleep)", "switch lurantis" -> "switch Lurantis"."""
    import setup_moves as sm

    name = str(name or "")
    if name.startswith("move "):
        move = name[5:]
        tera = move.endswith(" terastallize")
        if tera:
            move = move[: -len(" terastallize")]
        text = sm.move_name(move) + (" + Tera" if tera else "")
        if sm.is_setup(move, user_types):
            text += " (set-up)"
        elif sm.is_sleep(move):
            text += " (sleep)"
        return text
    if name == "switch ?":
        return "switch to one not seen yet"
    if name.startswith("switch "):
        return "switch " + sm.species_name(name[7:])
    return name


def prediction_text(top: Sequence[Sequence[Any]], user_types: Optional[Sequence[str]] = None) -> str:
    """ "predict Sleep Powder (sleep) 58%, switch Lurantis 22%, Sludge Bomb 12%" from [[name, p], ...]."""
    return "predict " + ", ".join("%s %d%%" % (option_text(name, user_types), round(100.0 * float(p))) for name, p in top)


def prediction_line(turn: int, theirs: str, top: Sequence[Sequence[Any]], played: str, guard: Optional[Dict[str, Any]] = None,
                    user_types: Optional[Sequence[str]] = None) -> str:
    """
    One decision on one line: "T7 Venusaur: predict Sleep Powder (sleep) 58%, switch Lurantis 22%,
    Sludge Bomb 12% | we play Meteor Mash", and when the guard fired " [guard: boosted +2]", or
    " [guard: boosted +2, instead of Swords Dance]" when it changed the move.
    """
    text = "T%d %s: %s | we play %s" % (int(turn), theirs or "?", prediction_text(top, user_types), played)
    if guard and guard.get("guard"):
        text += " [guard: %s%s]" % (guard["guard"], (", instead of " + guard["veto_text"]) if guard.get("changed") and guard.get("veto_text") else "")
    return text


def did_text(label: Optional[Dict[str, Any]], name: Optional[str]) -> str:
    """What the opponent did, in words, from the tracker's label of the turn."""
    label = label or {}
    action = str(label.get("action") or name or "")
    if action.startswith("switch "):
        return "switched to " + option_text("switch " + str(label.get("name") or action[7:]))[7:]
    text = option_text(action) if action else "?"
    return text + (" + Tera" if label.get("tera") and "+ Tera" not in text else "")


def outcome_text(top: Sequence[str], name: Optional[str], label: Optional[Dict[str, Any]] = None) -> str:
    """
    After the turn: "read: right (Sleep Powder)", "read: wrong, it played Sludge Bomb (our 2nd
    guess)", "read: wrong, it switched to Lurantis", or "read: not scored, ..." when the log
    cannot say what it chose (locked into a move, the game ended in that turn).
    top is the names predicted, likeliest first; name is what it did in the same names (or None).
    """
    label = label or {}
    if name is None:
        if label.get("locked"):
            return "read: not scored, it was locked into %s" % did_text(label, None) if label.get("action") else "read: not scored, it had no choice"
        return "read: not scored (%s)" % (label.get("why_unknown") or "the log does not say what it chose")
    if top and name == top[0]:
        return "read: right (%s)" % did_text(label, name)
    rank = list(top).index(name) + 1 if name in top else 0
    return "read: wrong, it %s%s" % ("" if did_text(label, name).startswith("switched") else "played ", did_text(label, name)) + (
        " (our %s guess)" % {2: "2nd", 3: "3rd"}.get(rank, "%dth" % rank) if rank else "")


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


class SettingPicker:
    """
    Gives every new game one of the named settings (GameSetting), balanced as ArmPicker gives its
    arms: always one of those played least so far. Among those it takes one that has no game in
    progress when there is one (in_play), else one at random, so that with as many games at once
    as settings the games in progress are one of each setting wherever the counts allow it. A
    game keeps the setting it was given.

    counts   games already played per setting name (from the results file), so that a run that is
             started again keeps the settings level over the whole experiment.
    in_play  a function giving the battle tags of the games in progress (None: no preference);
             set after the player is made. A failure of it is no preference, never an error.
    """

    def __init__(self, settings: Sequence[GameSetting], counts: Optional[Dict[str, int]] = None,
                 rng: Optional[random.Random] = None, in_play: Optional[Callable[[], Any]] = None):
        self.settings: List[GameSetting] = list(settings)
        if not self.settings:
            raise ValueError("no setting given")
        names = [s.name for s in self.settings]
        if len(set(names)) != len(names):
            raise ValueError("a setting is named twice: %s" % ",".join(names))
        for s in self.settings:
            split_mode(s.mode)
        self.counts: Dict[str, int] = {name: int((counts or {}).get(name, 0)) for name in names}
        self.rng = rng or random.Random()
        self.in_play = in_play
        self.given: "OrderedDict[str, GameSetting]" = OrderedDict()

    def _busy(self, tag: str) -> set:
        """The names of the settings of the other games in progress."""
        if self.in_play is None:
            return set()
        try:
            return {self.given[t].name for t in list(self.in_play()) if t != tag and t in self.given}
        except Exception:
            return set()

    def __call__(self, tag: str) -> GameSetting:
        setting = self.given.get(tag)
        if setting is None:
            fewest = min(self.counts.values())
            least = [s for s in self.settings if self.counts[s.name] == fewest]
            busy = self._busy(tag)
            setting = self.rng.choice([s for s in least if s.name not in busy] or least)
            self.counts[setting.name] += 1
            self.given[tag] = setting
            while len(self.given) > KEPT_BATTLES:
                self.given.popitem(last=False)
        return setting


def _new_game(mode: str) -> Dict[str, Any]:
    return {
        "mode": mode, "decisions": 0, "asked": 0, "searched": 0, "fallbacks": {},
        "disagreed": 0, "net_overruled": 0, "search_overruled": 0, "tera_added": 0, "tera_dropped": 0,
        "iterations": 0, "search_ms": 0.0, "decision_ms": 0.0, "slowest_ms": 0.0, "late": 0,
        "tera": False, "tera_turn": None, "tera_type": None, "tera_by": None, "tera_net": None, "tera_search": None,
        "tera_order": None,
        # predictions of the opponent's action ("predict", and every searched battle with live
        # predictions), and how they turned out; reads, changed and kept are "predict" only
        "pred_made": 0, "pred_scored": 0, "pred_top1_right": 0, "pred_top3_right": 0, "pred_p_true": 0.0,
        "pred_reads": 0, "pred_reads_scored": 0, "pred_reads_right": 0,
        "pred_changed": 0, "pred_changed_scored": 0, "pred_changed_right": 0, "pred_kept": {},
        # the guard (MixSettings.guard): decisions it fired on, and on how many it played another action than the veto rule
        "guard_fired": 0, "guard_changed": 0,
        # the set-up fixes (MixSettings.fixes): decisions on which one played another action than the plain veto
        "fix_changed": 0,
        # the longer search on set-up turns (ThinkLonger): asked, answered in time, and the move it changed
        "long_asked": 0, "long_used": 0, "long_changed": 0,
        # round 3's counter rules (MixSettings.setup_counter, setup_other_side): see COUNTER_COUNTS
        "counter_looked": 0, "counter_played": 0, "counter_switched": 0, "counter_ko_kept": 0, "counter_had": 0,
        "other_side_looked": 0, "other_side": 0,
        # the search's size per decision: worlds searched, and the full-width turns ahead (full_width_turns)
        "worlds": 0, "fw_turns": 0.0, "fw_n": 0,
        # the deep arm (SearchBudget): the budget asked, and when and why it fell back to today's
        "budget": None, "deep_fallback": None,
    }


PRED_COUNTS = ("pred_made", "pred_scored", "pred_top1_right", "pred_top3_right", "pred_p_true", "pred_reads", "pred_reads_scored",
               "pred_reads_right", "pred_changed", "pred_changed_scored", "pred_changed_right")
LIVE_COUNTS = ("pred_made", "pred_scored", "pred_top1_right", "pred_top3_right", "pred_p_true")
# Round 3's counter rules, per game, per arm and in all: the set-up positions "use the counter" looked
# at, the counter moves it played and the switches to a counter it made (the two changes of the move),
# the times an attack in KO range kept mix's choice, the times mix already played or switched to a
# counter; mix's attacks into a boosted defence, and those "don't hit the boosted defence" swapped.
COUNTER_COUNTS = ("counter_looked", "counter_played", "counter_switched", "counter_ko_kept", "counter_had", "other_side_looked",
                  "other_side")


def _counter_counts(info: Dict[str, Any]) -> Dict[str, int]:
    """What one decision's "counter" record adds to COUNTER_COUNTS."""
    out = {name: 0 for name in COUNTER_COUNTS}
    counter = info.get("counter")
    if counter:
        out["counter_looked"] = 1
        decided = str(counter.get("decided") or "")
        out["counter_played"] = int(decided == "played the counter")
        out["counter_switched"] = int(decided == "switched to a counter")
        out["counter_ko_kept"] = int(decided.startswith("kept: an attack"))
        out["counter_had"] = int(decided in ("kept: mix plays a counter", "kept: mix switches to a counter"))
    out["other_side"] = int(bool(info.get("other_side")))
    out["other_side_looked"] = int(bool(info.get("other_side") or info.get("other_side_kept")))
    return out


def counter_text(counts: Dict[str, Any]) -> str:
    """ "use the counter looked at 12 set-up positions: played a counter 3, switched to one 1, ..." from COUNTER_COUNTS."""
    return ("use the counter looked at %d set-up positions: played a counter %d, switched to one %d, kept for an attack in KO range %d, "
            "mix had one already %d; don't hit the boosted defence: %d attacks into a boosted defence, %d swapped"
            % tuple(int(counts.get(name) or 0) for name in COUNTER_COUNTS))


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

    from hybrid import choice_mask, combine
    from players import batched_policy_player

    Base = batched_policy_player()

    class SearchPlayer(fb.FoulPlayMixin, Base):
        """
        players.batched_policy_player with a search beside it.

        mode         "net", "search", "mix", "mix-veto", "mix-pool" or "mix-shortlist": for every battle.
        mode_picker  a function battle tag -> mode, asked once per battle (ArmPicker). Overrides mode.
        setting_picker  a function battle tag -> GameSetting, asked once per battle (SettingPicker;
                     not with mode_picker): the game's mode, and in place of the player's own
                     search_ms, deadline, budgets and MixSettings.trust the setting's, for that game
                     only. Without one (the default) every game has the player's.
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
        on_outcome   called after a turn with what was predicted and what the opponent did ("predict",
                     and every searched battle with live_predictions).
        live_predictions  in every searched battle, the opponent's next action read from the search's
                     own view of its side (no extra search): record["predict"] with the top 3 and a
                     line to print, scored after the turn. Changes no move.
        warm_up      one forward pass of the network at construction (see warm_up()). On by default.
        think_longer a ThinkLonger: a longer search on set-up turns of the searched battles (None: off).
        budgets      {arm: SearchBudget}: the search of a battle of that arm (DEEP_ARM: DEEP_BUDGET
                     unless given). Every other arm searches search_ms within deadline.
        luna         a luna_player.Luna (the client with the key, the settings): asked at every decision
                     of a battle in mode "luna" that has more than one legal action. None: such a
                     battle is played by mix, every decision counted as "off".
        deciders     {model: luna_player.Luna} for the arms of DECIDERS ("luna", "sol"; `luna` is
                     deciders["luna"]). An arm whose model is not there is played by mix ("off").
        lines_rule   the LinesRule of the arm "mix-lines".

        Every decision record, of every arm, also carries the network's own score of the position
        (value_fields: v_exp, v_std, phi, phi_scale, win_est) from the forward pass that made the
        logits, and for a searched decision the search's size (asked_ms, worlds_x, worlds,
        iterations, iterations_per_world, full_width_turns).
        """

        def __init__(self, *args: Any, mode: str = "net", mode_picker: Optional[Callable[[str], str]] = None,
                     bridge: Any = None, search_ms: Optional[int] = None, deadline: float = 2.5,
                     mix: Optional[MixSettings] = None, search_more: Optional[Dict[str, Any]] = None,
                     on_decision: Optional[Callable[[Dict[str, Any]], None]] = None, warm_up: bool = True,
                     predict: Any = None, opponent_model: Any = None,
                     on_outcome: Optional[Callable[[Dict[str, Any]], None]] = None,
                     live_predictions: bool = False, think_longer: Optional[ThinkLonger] = None,
                     budgets: Optional[Dict[str, SearchBudget]] = None, luna: Any = None,
                     deciders: Optional[Dict[str, Any]] = None, lines_rule: Optional[LinesRule] = None,
                     setting_picker: Optional[Callable[[str], GameSetting]] = None,
                     **kwargs: Any):
            split_mode(mode)
            if setting_picker is not None and mode_picker is not None:
                raise ValueError("a mode picker and a setting picker cannot go together: the setting gives the game its mode")
            self.setting_picker = setting_picker
            self._game_settings: "OrderedDict[str, GameSetting]" = OrderedDict()   # battle tag -> its setting (setting_picker)
            self.deciders: Dict[str, Any] = {k: v for k, v in dict(deciders or {}).items() if v is not None}
            if luna is not None:
                self.deciders.setdefault("luna", luna)
            self.luna = self.deciders.get("luna")
            self.lines_rule = lines_rule or LinesRule()
            # A model alone (SOLO_ARMS) has no memory of its own: battle tag -> its last moves there, for its next brief.
            self._solo_memory: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
            self._luna_events: "OrderedDict[str, Any]" = OrderedDict()   # battle tag -> luna_player.TurnLog: the last turn's lines
            # Per-arm search budgets: the deep arm gets DEEP_BUDGET unless told otherwise; every other
            # arm search_ms and deadline (today's budget).
            self.budgets: Dict[str, SearchBudget] = dict(ARM_BUDGETS)
            self.budgets.update(dict(budgets or {}))
            for arm in self.budgets:
                split_mode(arm)
            self._potential: Any = None          # rl_reward's potential, loaded on first use (False: it could not be)
            # Before the base class: Player.__init__ starts listening, and a battle message can
            # arrive before it returns.
            self.search_mode = mode
            self.mode_picker = mode_picker
            self.search_ms = int(search_ms) if search_ms else None
            self.deadline = float(deadline)
            self.mix_settings = mix or MixSettings()
            self.think_longer = think_longer if (think_longer is not None and int(think_longer.ms) > 0) else None
            self._time_left: "OrderedDict[str, float]" = OrderedDict()   # battle tag -> seconds the timer leaves us, as last said
            self._long_threads_used = 0                                   # long searches running now with a second thread
            self.live_predictions = bool(live_predictions)
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
                if self.think_longer is not None and self.think_longer.wait_s() < minimum_deadline(self.think_longer.ms):
                    raise ValueError("a long search of %d ms is never started with a deadline of %.2f s: it must be at least %.2f s"
                                     % (self.think_longer.ms, self.think_longer.wait_s(), minimum_deadline(self.think_longer.ms)))
                if mode in ("predict", "mix-guard") or mode in DECIDERS or mode_picker is not None or self.live_predictions \
                        or self.mix_settings.guard or self._needs_threat() or setting_picker is not None:
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
            # The searches running now, and the last ones done: which ran beside which (_search_done),
            # so that the A/B can check that mix keeps its compute with deep searches beside it.
            self._inflight: Dict[str, Tuple[str, float]] = {}         # battle tag -> ("deep" or "today", start on the loop's clock)
            self._searches_done: Any = deque(maxlen=64)               # (tag, size, start, end)
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
                "guard_fired": 0, "guard_changed": 0, "guard_errors": 0,
                "fix_changed": 0, "long_asked": 0, "long_used": 0, "long_changed": 0, "long_errors": 0,
                "counter_errors": 0, "luna_errors": 0,
            }
            for name in COUNTER_COUNTS:
                self.search_totals[name] = 0
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
                        getattr(self, "net_values", {}).pop(tag, None)
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
                setting: Optional[GameSetting] = None
                if self.mode_picker is not None:
                    try:
                        mode = str(self.mode_picker(tag))
                        split_mode(mode)
                    except Exception:
                        mode = "net"
                elif self.setting_picker is not None:
                    try:
                        setting = self.setting_picker(tag)
                        split_mode(setting.mode)
                        mode = setting.mode
                    except Exception:
                        setting, mode = None, "net"
                self._modes[tag] = mode
                self._games[tag] = _new_game(mode)
                if setting is not None:
                    self._game_settings[tag] = setting
                    self._games[tag]["setting"] = setting.name
                    # searched decisions asked with the setting's own budget, and with its fall back (search_ms within deadline)
                    self._games[tag]["budget_asked"] = {"setting": 0, "fallback": 0}
                for store in (self._modes, self._games, self._decisions, self._game_settings):
                    while len(store) > KEPT_BATTLES:
                        store.popitem(last=False)
            return mode

        # -- a game's own setting (setting_picker): what it has in place of the player's --------------

        def battle_setting(self, tag: str) -> Optional[GameSetting]:
            """The GameSetting of a battle (None without a setting picker)."""
            return self._game_settings.get(tag)

        def _search_ms_of(self, tag: str) -> Optional[int]:
            setting = self._game_settings.get(tag)
            return self.search_ms if setting is None else int(setting.search_ms)

        def _deadline_of(self, tag: str) -> float:
            setting = self._game_settings.get(tag)
            return self.deadline if setting is None else float(setting.deadline)

        def _mix_of(self, tag: str) -> MixSettings:
            setting = self._game_settings.get(tag)
            return self.mix_settings if setting is None else replace(self.mix_settings, trust=float(setting.trust))

        def _budget_of(self, tag: str, mode: str) -> Optional[SearchBudget]:
            """The SearchBudget of mode in this battle: the player's, or a setting's own (never another game's)."""
            setting = self._game_settings.get(tag)
            if setting is None:
                return self.budgets.get(mode)
            return setting.own_budget() if mode == setting.mode else None

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
            for store in (self._trackers, self._predictions, self._luna_events):
                store.pop(tag, None)        # rebuilt from the replay by _follow_opponent, not fed it twice
            # A battle the bridge still follows is resumed there (FoulPlayBridge._missed_lines: only
            # the turns it missed are fed from the replay): the search goes on in it.
            try:
                knows = searched and bool(getattr(self.foulplay, "knows", None)) and self.foulplay.knows(self._foulplay_key(tag))
            except Exception:
                knows = False
            if knows:
                self.search_totals["resumed"] = self.search_totals.get("resumed", 0) + 1
                return
            self._rejoined.add(tag)
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
                # The timer's word on our time left: the longer search and the deep arm's budget read it.
                for parts in split_messages[1:]:
                    if len(parts) > 2 and parts[1] == "inactive":
                        left = parse_time_left("|".join(parts[2:]))
                        if left is not None:
                            self._time_left[room] = left
                            while len(self._time_left) > KEPT_BATTLES:
                                self._time_left.popitem(last=False)
            except Exception:       # bookkeeping must never disturb a game
                pass
            try:
                self._follow_opponent(room, split_messages)
            except Exception:
                pass
            try:
                self._luna_follow(room, split_messages)
            except Exception:
                pass
            if self._searched_battle(room):
                return await super()._handle_battle_message(split_messages)       # the mixin forwards the lines first
            # A battle the network plays alone: nothing goes to the search.
            return await super(fb.FoulPlayMixin, self)._handle_battle_message(split_messages)

        def _luna_follow(self, room: str, split_messages: List[List[str]]) -> None:
            """
            A Luna battle keeps what the Pokemon did in the turn being played and in the one before
            it (luna_player.TurnLog: no chat, no names of users), for the brief's "last turn".
            """
            if self._modes.get(room) not in DECIDERS and self.battle_mode(room) not in DECIDERS:
                return
            log = self._luna_events.get(room)
            if log is None:
                import luna_player as lp

                log = self._luna_events[room] = lp.TurnLog()
                while len(self._luna_events) > KEPT_BATTLES:
                    self._luna_events.popitem(last=False)
            log.feed(split_messages[1:])

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

        def _guard_on(self, mode: str) -> bool:
            return guarded(mode, self.mix_settings)

        def _needs_threat(self, mode: Optional[str] = None) -> bool:
            """Whether the set-up fixes or the longer search read the position (Threat) besides the guard (in a battle of mode, or in any)."""
            s = self._arm_settings(mode) if mode else self.mix_settings
            any_arm = mode is None and any(a in ARM_FIXES for a in (getattr(self.mode_picker, "arms", None) or ()))
            return bool(s.setup_defer or s.sure_attack > 0 or s.setup_rules or s.board_rules) or self.think_longer is not None or any_arm

        def _arm_settings(self, mode: Optional[str], base: Optional[MixSettings] = None) -> MixSettings:
            """The MixSettings of a battle of mode: the player's (or base: a game's own, _mix_of), with an ARM_FIXES arm's fixes on the veto rule."""
            base = self.mix_settings if base is None else base
            if mode in ARM_FIXES:
                return replace(base, rule="veto", **ARM_FIXES[mode][0])
            return base

        def _long_mode(self, mode: str) -> bool:
            """Whether the ThinkLonger applies to a battle of mode (see ARM_FIXES)."""
            if mode in ARM_FIXES:
                return bool(ARM_FIXES[mode][1])
            return not any(a in ARM_FIXES and ARM_FIXES[a][1] for a in (getattr(self.mode_picker, "arms", None) or ()))

        def _tracks(self, room: str) -> bool:
            """Whether the opponent is followed in this battle: "predict", and any searched battle with live predictions, the guard or a set-up fix."""
            if not self.foulplay_enabled:
                return False
            mode = self.battle_mode(room)
            if mode == "predict" or mode in DECIDERS:
                return True                 # (a model's brief carries the read of the opponent)
            return split_mode(mode)[0] != "net" and (self.live_predictions or self._guard_on(mode) or self._needs_threat(mode))

        def _follow_opponent(self, room: str, split_messages: List[List[str]]) -> None:
            """Every line of a followed battle goes to its tracker; the turns that ended are scored."""
            if not self._tracks(room) or getattr(self.battles.get(room), "finished", False):
                return                      # (lines after the end, the rating say, start no new tracker)
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
            info["theirs"] = str((ctx["opp"]["active"] or {}).get("species") or "")
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
            turns[turn] = {"dist": prediction.dist, "top": top[0][0], "top3": [n for n, _p in top], "info": info}
            while len(self._predictions) > KEPT_BATTLES:
                self._predictions.popitem(last=False)
            return int(action), ("predict" if chosen.get("changed") else "mix"), info

        def _live_prediction(self, battle: Any, advice: Any) -> Optional[Tuple[Any, Dict[str, Any]]]:
            """
            (opponent_model.Prediction, the record's "predict") for a move request of a followed
            battle, from the tracker's context and the search's answer already in hand; None when
            there is no context. A second request of the same turn (an order the server refused)
            gets the first one's prediction again, marked "repeat", and is not counted twice.
            """
            if self.opponent_model is None:
                self._predict_ready()
            tag = battle.battle_tag
            turn = int(getattr(battle, "turn", 0) or 0)
            known = (self._predictions.get(tag) or {}).get(turn)
            if known is not None and "prediction" in known:
                return known["prediction"], dict(known["info"], repeat=True)
            tracker = self._trackers.get(tag)
            ctx = tracker.context(turn) if tracker is not None else None
            if ctx is None:
                return None
            prediction = self.opponent_model.predict(ctx, advice)
            top = prediction.top(3)
            theirs = ctx["opp"]["active"] or {}
            info: Dict[str, Any] = {
                "top": [[name, round(p, 4)] for name, p in top], "p_switch": round(prediction.p_switch, 4),
                "p_tera": round(prediction.p_tera, 4), "against": str(theirs.get("id") or ""),
                "theirs": str(theirs.get("species") or ""), "theirs_types": list(theirs.get("types") or []),
            }
            self._predictions.setdefault(tag, {})[turn] = {"dist": prediction.dist, "top": top[0][0] if top else None,
                                                           "top3": [n for n, _p in top], "info": info, "prediction": prediction}
            while len(self._predictions) > KEPT_BATTLES:
                self._predictions.popitem(last=False)
            return prediction, info

        def _threat(self, battle: Any, mask: Any, dist: Optional[Dict[str, float]], advice: Any) -> Threat:
            """What the guard knows: the opponent's boosts (poke-env's view of the battle), the read, and which of our moves are passive."""
            import setup_moves as sm

            theirs = getattr(battle, "opponent_active_pokemon", None)
            boosts = dict(getattr(theirs, "boosts", None) or {})
            boost = max([int(v) for k, v in boosts.items() if k in sm.STATS and int(v) > 0] or [0])
            their_types = [getattr(t, "name", str(t)) for t in (getattr(theirs, "types", None) or ()) if t is not None]
            if dist is None and not getattr(battle, "force_switch", False):
                # No prediction (no context yet, the tracker broken): the search's own shares of its options.
                import opponent_model as om

                view = om.search_view(advice) or {"options": []}
                total = sum(max(0.0, float(s)) for _n, s, _seen in view["options"]) or 1.0
                dist = {}
                for name, share, _seen in view["options"]:
                    dist[name] = dist.get(name, 0.0) + max(0.0, float(share)) / total
            setup = sleep = status = 0.0
            for name, p in (dist or {}).items():
                if not name.startswith("move "):
                    continue
                move = name[5:].replace(" terastallize", "")
                if sm.is_setup(move, their_types):
                    setup += p
                if sm.is_sleep(move):
                    sleep += p
                if sm.is_status_move(move):
                    status += p
            ours = getattr(battle, "active_pokemon", None)
            if ours is None or getattr(ours, "status", None) is not None:
                sleep = 0.0                 # already statused: it cannot be put to sleep
            # Its action this turn so far (a forced switch in the middle of a turn), else last turn's.
            last_setup = self._last_setup(battle, their_types)
            our_types = [getattr(t, "name", str(t)) for t in (getattr(ours, "types", None) or ()) if t is not None]
            passive, fails, ko, counter_ko = [], [], [], []
            arm = self._arm_settings(self.battle_mode(battle.battle_tag))
            want_ko = (arm.sure_attack > 0 or arm.setup_counter) and theirs is not None and ours is not None
            hp_left = float(getattr(theirs, "current_hp_fraction", 1.0) or 0.0) if want_ko else 1.0
            full_guard = ""
            if arm.setup_counter and want_ko and hp_left >= 1.0:
                import setup_counter as sc

                worlds = [w for w in (getattr(advice, "opponent_worlds", None) or []) if isinstance(w, dict)]
                full_guard = sc.full_hp_guard(theirs, worlds)
            for a in list(range(4)) + list(range(10, 14)):
                if mask[a] <= 0.5:
                    continue
                target, _kw = self.assembler.map_index_to_order(a, battle)
                move = getattr(target, "id", None)
                if not move or hasattr(target, "current_hp"):
                    continue
                if sm.is_passive(move) or sm.is_setup(move, our_types):
                    passive.append(a)
                if sm.fails_on_status(move):
                    fails.append(a)
                if want_ko and hp_left > 0:
                    share = ko_estimate(target, ours, theirs)
                    if share is not None and share >= hp_left and not ko_doubt(target, theirs):
                        ko.append(a)
                    # The counter rules: a fresh sweeper too, when nothing lets it live at full HP, or at twice its HP.
                    if (arm.setup_counter and share is not None and share >= hp_left and not ko_doubt(target, theirs, full_hp=False)
                            and (hp_left < 1.0 or share >= 2.0 or not full_guard)):
                        counter_ko.append(a)
            return Threat(boost=boost, last_setup=last_setup, setup_read=round(setup, 4), sleep_read=round(sleep, 4),
                          status_read=round(status, 4), passive=tuple(passive), fails=tuple(fails), ko=tuple(ko),
                          counter_ko=tuple(counter_ko))

        def _action_label(self, battle: Any, action: int) -> str:
            """An action of ours for the console: "Meteor Mash", "Meteor Mash + Tera", "switch Lurantis"."""
            import setup_moves as sm

            try:
                target, kwargs = self.assembler.map_index_to_order(int(action), battle)
                if hasattr(target, "current_hp"):
                    return "switch " + sm.species_name(str(getattr(target, "species", target)))
                return sm.move_name(str(getattr(target, "id", target))) + (" + Tera" if kwargs.get("terastallize") else "")
            except Exception:
                return "action %d" % int(action)

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
            top3 = list(predicted.get("top3") or [predicted["top"]])
            try:
                outcome["text"] = outcome_text(top3, name, label)
            except Exception:       # words for the console must never cost the count
                outcome["text"] = "read: ?"
            game = self._games.get(room)
            per_mode = self.search_totals["by_mode"].get(game["mode"] if game is not None else "predict")
            groups = [c for c in (game, self.search_totals, per_mode) if c is not None]
            if name is not None:
                top_right = predicted["top"] == name
                read = info.get("read")
                read_right = (name.startswith("switch ") if read == "switch" else read == name) if info.get("confident") else None
                outcome.update({"top1_right": top_right, "top3_right": name in top3,
                                "p_true": round(float(predicted["dist"].get(name, 0.0)), 4), "read_right": read_right})
                if read_right is False:
                    self._failed_reads.setdefault(room, set()).add((info.get("against"), read))
                    while len(self._failed_reads) > KEPT_BATTLES:
                        self._failed_reads.popitem(last=False)
                for counts in groups:
                    counts["pred_scored"] = counts.get("pred_scored", 0) + 1
                    counts["pred_top1_right"] = counts.get("pred_top1_right", 0) + int(top_right)
                    counts["pred_top3_right"] = counts.get("pred_top3_right", 0) + int(name in top3)
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
                    self.on_outcome({"battle": room, "turn": turn, "mode": game["mode"] if game is not None else None, "predict": info})
                except Exception:
                    pass

        def _battle_finished_callback(self, battle: Any):
            # The battle's last lines have been read by now (_follow_opponent runs first): what
            # follows the opponent in it is not needed any more (a tracker keeps every turn).
            for store in (self._trackers, self._predictions, self._failed_reads, self._time_left, self.net_values,
                          self._luna_events):
                store.pop(battle.battle_tag, None)
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

        def _ask(self, battle: Any, ttl: float, search_ms: Optional[int] = None, threads: int = 0,
                 budget: Optional[SearchBudget] = None) -> Any:
            request = getattr(battle, "last_request", None) or {}
            more = self.search_more
            if search_ms:
                # The longer search of a set-up turn (ThinkLonger): no payoff table, the engine's
                # second thread when a helper is free.
                more = {k: v for k, v in more.items() if k not in ("threads", "payoff_ms", "payoff")}
                if threads > 1:
                    more["threads"] = int(threads)
                return self.foulplay.advice_future(self._foulplay_key(battle.battle_tag), int(search_ms),
                                                   request.get("rqid"), ttl=ttl, **more)
            if self.battle_mode(battle.battle_tag) == "predict" and not getattr(battle, "force_switch", False):
                # The table of values per (our action, the opponent's action) that opponent_model.choose reads.
                try:
                    if self.predict_settings is None:
                        self._predict_ready()
                    if self.predict_settings.payoff_ms > 0:
                        more = dict({"payoff_ms": self.predict_settings.payoff_ms, "payoff": self.predict_settings.payoff_mode}, **more)
                except Exception:       # no model, no table: the search is asked as for "mix"
                    self.search_totals["predict_errors"] += 1
            if budget is not None:
                # An arm with a budget of its own (the deep arm): its time per world, its worlds, its threads.
                more = dict(more)
                if int(budget.worlds_x) > 1:
                    more["parallelism"] = int(budget.worlds_x)
                if int(budget.threads) > 1:
                    more["threads"] = int(budget.threads)
                if int(budget.workers) > 1:
                    more["workers"] = int(budget.workers)
                if budget.lines_spec() is not None:
                    more["lines"] = budget.lines_spec()
                return self.foulplay.advice_future(self._foulplay_key(battle.battle_tag), int(budget.search_ms),
                                                   request.get("rqid"), ttl=ttl, **more)
            return self.foulplay.advice_future(self._foulplay_key(battle.battle_tag), self._search_ms_of(battle.battle_tag),
                                               request.get("rqid"), ttl=ttl, **more)

        # -- the deep arm's budget ---------------------------------------------------------------

        def _budget(self, battle: Any, mode: str) -> Optional[SearchBudget]:
            """
            The SearchBudget of this decision, or None for today's (search_ms within deadline). An
            arm with a budget falls back to today's for the rest of a game when the battle timer
            says fewer than budget.min_time_left seconds are left, or once one of its decisions
            took longer than budget.max_decision_s (_deep_fall_back: noted in the game and the record).
            """
            budget = self._budget_of(battle.battle_tag, mode)
            if budget is None:
                return None
            tag = battle.battle_tag
            game = self._games.get(tag)
            if game is not None and game.get("deep_fallback"):
                return None
            left = self._time_left.get(tag)
            if left is not None and left < float(budget.min_time_left):
                self._deep_fall_back(battle, "the battle timer said %d s left (under %d)" % (left, budget.min_time_left))
                return None
            return budget

        def _deep_fall_back(self, battle: Any, why: str) -> None:
            game = self._games.get(battle.battle_tag)
            if game is not None and not game.get("deep_fallback"):
                game["deep_fallback"] = {"why": why, "turn": int(getattr(battle, "turn", 0) or 0)}
                self.search_totals["deep_fallbacks"] = self.search_totals.get("deep_fallbacks", 0) + 1

        # -- the network's own score of the position ---------------------------------------------

        def _phi(self, battle: Any) -> Optional[float]:
            """The potential the reward was shaped with, from our side of the live battle (as worker.py reads it); None when it cannot be read."""
            if self._potential is False:
                return None
            try:
                if self._potential is None:
                    from rl_reward import load_potential

                    import os as _os

                    settings = getattr(self.policy, "value_settings", None) or {}
                    path = str(settings.get("potential_file") or "")
                    # A path of the training box that is not here: the weights shipped in rl_reward/
                    # (the same file on every branch since the potential was fitted).
                    self._potential = load_potential(path if path and _os.path.isfile(path) else None)
                from rl_reward import battle_features

                return float(self._potential.phi(battle_features(battle)))
            except Exception:       # a score for the record must never cost a move
                if self._potential is None:
                    self._potential = False
                return None

        def _value_of(self, tag: str, phi: Optional[float]) -> Dict[str, Any]:
            """value_fields of this decision's forward pass (players.BatchedPolicyPlayer.net_values), {} without one."""
            try:
                got = self.net_values.pop(tag, None)
                if got is None:
                    return {}
                settings = getattr(self.policy, "value_settings", None) or {}
                scale = float(settings.get("scale", 0.5)) if settings.get("shaping", "potential") == "potential" else 0.0
                return value_fields(got[0], got[1], phi, scale)
            except Exception:
                return {}

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
            # The observation's action mask (obs_v2's leaves out a switch the server refuses here).
            mask = choice_mask(self.assembler, battle, obs)
            # The position as the reward's potential sees it (as worker.py reads it at a decision):
            # with the value head's v_exp it makes the network's own win estimate (value_fields).
            phi = self._phi(battle)

            # This decision's search budget: the arm's own (the deep arm) unless it fell back.
            budget: Optional[SearchBudget] = None
            fell_back: Optional[Dict[str, Any]] = None
            if kind != "net" and self.budgets:
                try:
                    game = self._games.get(tag) or {}
                    had = bool(game.get("deep_fallback"))
                    budget = self._budget(battle, mode)
                    if not had and game.get("deep_fallback"):
                        fell_back = game["deep_fallback"]
                except Exception:       # the budget must never cost a move: today's is asked
                    budget = None
            deadline = float(budget.deadline) if budget is not None else self._deadline_of(tag)

            # The question goes to the search first: it thinks while the network answers.
            reason, future = "", None
            deadline_at = started + deadline                # on the loop's clock; later for a long search
            long_info: Optional[Dict[str, Any]] = None
            longer = self.think_longer is not None and kind in ("mix", "search") and self._long_mode(mode)
            if kind != "net":
                reason = self._why_not_ask(battle, mask)
                if not reason:
                    pre = ""
                    if longer:
                        try:
                            pre = self._long_before(battle) if self._long_allowed(battle) else ""
                        except Exception:       # the longer search must never cost a move: the ordinary one is asked
                            self.search_totals["long_errors"] += 1
                    try:
                        if pre:
                            future, long_info, deadline_at = self._long_ask(battle, pre, "before", loop)
                        else:
                            future = self._ask(battle, deadline - RESERVE_S - (loop.time() - started), budget=budget)
                    except Exception:
                        reason = "error"
            search_started = loop.time()
            beside: Optional[Dict[str, float]] = None
            if future is not None:
                self._inflight[tag] = ("deep" if self._more_search(budget) else "today", search_started)

            try:
                logits = await self._queue_inference(tag, obs)
            except BaseException:
                # A long search asked before the move holds a second thread: give it back, or no
                # long search would get one again.
                if long_info is not None and long_info.pop("_thread", False):
                    self._long_threads_used = max(0, self._long_threads_used - 1)
                self._inflight.pop(tag, None)
                raise
            net_action = int(combine(logits, None, mask, self.hybrid, None if self.hybrid.greedy else self.rng))
            net_ms = (loop.time() - started) * 1000.0
            value = self._value_of(tag, phi)        # the same forward pass that made the logits: no extra one

            advice = None
            wait_over_ms = 0.0
            short_view: Optional[SearchView] = None
            try:
                if future is not None:
                    advice, why_not, wait_over_ms = await self._await_advice(battle, future, deadline_at, loop)
                    reason = why_not or reason
                view = None
                if advice is not None:
                    view, reason = read_advice(advice, mask, self.mix_settings.min_iterations)
                if long_info is not None:
                    long_info.update(answered=advice is not None, used=view is not None, why_not=reason or None)
                elif longer and view is not None:
                    # The ordinary answer is in: when its read of the opponent puts enough on a
                    # set-up move, the search is asked again for longer, and its answer is used
                    # when it comes in time and can be used.
                    read = 0.0
                    try:
                        read = self._read_of_setup(battle, advice)
                        if read >= self.think_longer.read and self._long_allowed(battle):
                            future2, long_info, until = self._long_ask(battle, "set-up read %d%%" % round(100 * read), "after", loop)
                            advice2, why2, over2 = await self._await_advice(battle, future2, until, loop)
                            wait_over_ms = max(wait_over_ms, over2)
                            view2, why2 = read_advice(advice2, mask, self.mix_settings.min_iterations) if advice2 is not None else (None, why2)
                            long_info.update(answered=advice2 is not None, used=view2 is not None, why_not=why2 or None)
                            deadline_at = until
                            if view2 is not None:
                                short_view, view, advice = view, view2, advice2
                    except asyncio.CancelledError:
                        raise
                    except Exception:       # the longer search must never cost a move: the ordinary answer stands
                        self.search_totals["long_errors"] += 1
            finally:
                if long_info is not None and long_info.pop("_thread", False):
                    self._long_threads_used = max(0, self._long_threads_used - 1)
                if future is not None:
                    try:
                        beside = self._search_done(tag, search_started, loop.time())
                    except Exception:       # bookkeeping must never cost a move
                        self._inflight.pop(tag, None)

            probs = legal_probabilities(logits, mask)
            action, source = net_action, "net"
            predicted: Optional[Dict[str, Any]] = None
            guard_info: Optional[Dict[str, Any]] = None
            fix_info: Optional[Dict[str, Any]] = None
            counter_info: Optional[Dict[str, Any]] = None
            live: Any = None
            if view is not None:
                if kind in ("mix", "search") and not getattr(battle, "force_switch", False) and self._tracks(tag):
                    # The opponent's next action, from the answer in hand: no extra search.
                    try:
                        live = self._live_prediction(battle, advice)
                    except Exception:   # a prediction must never cost a move
                        self.search_totals["predict_errors"] += 1
                if kind == "search":
                    action, source = view.best, "search"
                    if short_view is not None and long_info is not None:
                        long_info["short_action"] = int(short_view.best)
                elif kind in ("mix", "predict"):
                    base = self._mix_of(tag)
                    settings = base if rule is None else replace(base, rule=rule)
                    threat = None
                    if kind == "mix" and self._guard_on(mode):
                        settings = replace(settings, guard=True, rule="veto")
                    if kind == "mix" and mode in ARM_FIXES:
                        settings = self._arm_settings(mode, base)
                    if kind == "mix" and (settings.guard or settings.fixes or settings.setup_rules or settings.board_rules):
                        try:
                            threat = self._threat(battle, mask, live[0].dist if live is not None else None, advice)
                        except Exception:       # the guard must never cost a move: the plain veto plays
                            self.search_totals["guard_errors"] += 1
                    board = None
                    if kind == "mix" and settings.board_rules:
                        try:
                            board = self._board(battle, mask, threat, advice, settings)
                        except Exception:       # the counter rules must never cost a move: mix plays as without them
                            self.search_totals["counter_errors"] += 1
                    explain: Optional[Dict[str, Any]] = {} if (threat is not None or settings.fixes or settings.setup_rules
                                                               or settings.board_rules) else None
                    action, source = mix(probs, mask, view, settings, threat=threat, explain=explain, board=board), "mix"
                    if short_view is not None and long_info is not None:
                        try:
                            long_info["short_action"] = int(mix(probs, mask, short_view, settings, threat=threat, board=board))
                        except Exception:
                            pass
                    if explain is not None and explain.get("board"):
                        try:
                            counter_info = self._counter_note(battle, explain["board"], net_action, view, board)
                        except Exception:       # words for the record must never cost a move
                            counter_info = dict(explain["board"])
                    if explain is not None and (settings.fixes or explain.get("setup_rule")) and (explain.get("setup_rule") or not explain.get("guard")):
                        # What the plain veto would have played (without the fixes, the set-up rules and the guard).
                        plain = mix(probs, mask, view, replace(settings.without_fixes(), guard=False))
                        if plain != action:
                            fix_info = {"fix": explain.get("setup_rule") or explain.get("fix") or "tera", "veto": int(plain), "veto_text": self._action_label(battle, plain),
                                        "net_pick": int(net_action), "search_pick": int(view.best)}
                            if threat is not None:
                                fix_info["threat"] = {"boost": threat.boost, "setup_read": threat.setup_read, "ko": list(threat.ko)}
                    if explain is not None and explain.get("guard"):
                        plain = mix(probs, mask, view, replace(settings, guard=False), board=board)
                        guard_info = dict(explain, changed=base_action(plain) != base_action(action) or plain != action, veto=int(plain),
                                          threat={"boost": threat.boost, "setup_read": threat.setup_read, "sleep_read": threat.sleep_read,
                                                  "status_read": threat.status_read})
                        if guard_info["changed"]:
                            guard_info["veto_text"] = self._action_label(battle, plain)
                    if kind == "predict":
                        mixed = action
                        try:
                            action, source, predicted = self._predict(battle, advice, view, mask, mixed)
                        except Exception as exc:        # a prediction must never cost a move: the mix choice stands
                            self.search_totals["predict_errors"] += 1
                            action, source, predicted = mixed, "mix", {"mix": int(mixed), "changed": False, "why": "error", "error": repr(exc)[:200]}
                if live is not None:
                    predicted = live[1]

            # The lines (an arm of LINES_ARMS): the search's favourite actions played forward a few
            # turns, on our 14 actions. "mix-lines" lets mix's pick give way to an action whose line
            # ends clearly better (lines_pick); a model's arm puts them into its brief.
            lines: Dict[int, Dict[str, Any]] = {}
            lines_info: Optional[Dict[str, Any]] = None
            if mode in LINES_ARMS and view is not None and advice is not None:
                try:
                    lines = dict(getattr(advice, "lines_by_action", None) or {})
                    if mode == LINES_ARM and source == "mix" and int((mask > 0.5).sum()) > 1:
                        picked, lines_info = lines_pick(int(action), view.share, lines, self.lines_rule)
                        if lines_info.get("changed") and mask[int(picked)] > 0.5:
                            action, source = int(picked), "lines"
                    else:
                        lines_info = {"mix": int(action), "played": int(action), "changed": False, "why": "shown" if lines else "no_lines",
                                      "values": {str(a): row.get("value") for a, row in sorted(lines.items())},
                                      "turns": {str(a): row.get("turns") for a, row in sorted(lines.items())}}
                    raw = getattr(advice, "lines", None) or {}
                    lines_info["ms"] = raw.get("ms")
                    if raw.get("error"):
                        lines_info["error"] = str(raw.get("error"))[:200]
                except Exception:       # the lines must never cost a move: mix's pick is played
                    self.search_totals["lines_errors"] = self.search_totals.get("lines_errors", 0) + 1
                    lines, lines_info = {}, None

            # A model's arm (DECIDERS): mix has decided as it does today (`action`); now the model is
            # asked, with mix's numbers in front of it, and its choice is played. Mix's pick stands
            # whenever the model gives no legal answer in time (luna_player.Luna.choose: the rails).
            # With a budget that has decide_s the model gets that long from here; without one what
            # the decision's deadline leaves.
            luna_info: Optional[Dict[str, Any]] = None
            if mode in DECIDERS and int((mask > 0.5).sum()) > 1:
                mix_action = int(action)
                solo = mode in SOLO_ARMS
                if solo:
                    # The model alone: what plays when it gives no answer is the rule's pick, never the
                    # network's, and the model gets its own timeout from here (there is no search to wait for).
                    mix_action = self._solo_action(battle, mask, mix_action)
                    action, source = mix_action, "rule"
                    alone = self.deciders.get(DECIDERS[mode])
                    deadline_at = loop.time() + (float(alone.settings.timeout) if alone is not None else 0.0) + RESERVE_S + 0.1
                elif budget is not None and float(budget.decide_s) > 0:
                    deadline_at = min(loop.time(), deadline_at) + float(budget.decide_s) + RESERVE_S
                try:
                    action, luna_info = await self._luna_decide(battle, mask, probs, net_action, mix_action, view, advice, reason,
                                                                live[0] if live is not None else None, value, deadline_at, loop,
                                                                decider=self.deciders.get(DECIDERS[mode]), lines=lines, solo=solo)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:       # Luna must never cost a move: mix's pick is played
                    self.search_totals["luna_errors"] += 1
                    action = mix_action
                    luna_info = {"asked": False, "mix": mix_action, "choice": None, "confidence": None, "probabilities": None,
                                 "latency_ms": None, "input_tokens": None, "tokens_from": "estimate", "fell_back": True,
                                 "why": "error", "overruled": False, "error": self._luna_redact(repr(exc))[:200]}
                if int(action) != mix_action and mask[int(action)] <= 0.5:
                    action = mix_action        # never: every value Luna can answer with is a legal action
                try:                        # whose record it is (the model's name and id), and what an unanswered question cost
                    who = self.deciders.get(DECIDERS[mode])
                    if who is not None:
                        who.stamp(luna_info)
                    else:
                        luna_info.setdefault("who", DECIDERS[mode])
                except Exception:
                    pass
                if solo:
                    luna_info["solo"] = True
                if luna_info.get("choice") is not None and not luna_info.get("overruled"):
                    source = DECIDERS[mode]
                if solo:
                    try:                    # what it just did, for its next brief in this battle
                        self._solo_remember(battle, int(action), luna_info, source)
                    except Exception:
                        pass
            if lines_info is not None:
                try:
                    lines_info["played"] = int(action)
                    lines_info["text"] = self._lines_text(battle, lines, int(action))
                    if lines_info.get("changed"):
                        lines_info["mix_text"] = self._action_label(battle, int(lines_info["mix"]))
                except Exception:       # words for the window must never cost a move
                    pass

            target, kwargs = self.assembler.map_index_to_order(action, battle)
            if target == "DEFAULT":
                # A forced state the 14 actions do not cover.
                self.counts["fallbacks"] += 1
                order = self.choose_random_move(battle)
            else:
                order = self.create_order(target, **kwargs)

            if predicted is not None and predicted.get("top"):
                try:
                    theirs = predicted.get("theirs") or str(getattr(getattr(battle, "opponent_active_pokemon", None), "species", "") or "")
                    predicted["line"] = prediction_line(int(getattr(battle, "turn", 0) or 0), theirs, predicted["top"],
                                                        self._action_label(battle, action), guard_info, predicted.get("theirs_types"))
                except Exception:       # words for the console must never cost a move
                    pass
            elapsed_ms = (loop.time() - started) * 1000.0
            if long_info is not None and "short_action" in long_info:
                long_info["changed"] = base_action(long_info["short_action"]) != base_action(action) or long_info["short_action"] != action
            if budget is not None and elapsed_ms > float(budget.max_decision_s) * 1000.0:
                # A busy machine: the next decisions of this game get today's budget (the clock).
                try:
                    self._deep_fall_back(battle, "a decision took %.1f s (over %.1f)" % (elapsed_ms / 1000.0, budget.max_decision_s))
                    fell_back = (self._games.get(tag) or {}).get("deep_fallback")
                except Exception:
                    pass
            try:
                self._record(battle, mode, mask, probs, net_action, view, advice, action, source, reason,
                             future is not None, elapsed_ms, net_ms, wait_over_ms, predicted, guard_info,
                             fix_info=fix_info, long_info=long_info, deadline_s=deadline_at - started, counter_info=counter_info,
                             budget=budget, value=value, fell_back=fell_back, beside=beside, luna_info=luna_info,
                             lines_info=lines_info)
            except Exception:       # bookkeeping must never disturb a game
                pass
            return order

        def _solo_remember(self, battle: Any, action: int, info: Dict[str, Any], source: str) -> None:
            """One more of a model-alone battle's own moves (the last 6 are kept): the turn, the move in words, the model's reason."""
            rows = self._solo_memory.setdefault(battle.battle_tag, [])
            rows.append({"turn": int(getattr(battle, "turn", 0) or 0), "text": self._action_label(battle, action),
                         "reason": info.get("reason"), "switch": 4 <= int(action) <= 9,
                         "forced": bool(getattr(battle, "force_switch", False)), "by": "rule" if source == "rule" else "model"})
            del rows[:-6]
            while len(self._solo_memory) > KEPT_BATTLES:
                self._solo_memory.popitem(last=False)

        def _solo_action(self, battle: Any, mask: Any, default: int) -> int:
            """The fallback rule's pick for a model-alone battle (luna_player.solo_rule); never raises (then the first legal action, else `default`)."""
            try:
                import luna_player as lp

                actions: Dict[int, Tuple[Any, Dict[str, Any]]] = {}
                for a in range(ACTIONS):
                    if mask[a] > 0.5:
                        target, kwargs = self.assembler.map_index_to_order(a, battle)
                        if target != "DEFAULT":
                            actions[a] = (target, dict(kwargs or {}))
                return int(lp.solo_rule(battle, [float(m) for m in mask], actions))
            except Exception:
                legal = [a for a in range(ACTIONS) if mask[a] > 0.5]
                return int(legal[0]) if legal else int(default)

        def _more_search(self, budget: Optional[SearchBudget]) -> bool:
            """Whether a budget asks the helper for more than today's search does (the deep arm, the lines): for the compute check."""
            if budget is None:
                return False
            return budget.compute() > float(self.search_ms or 500) + 1e-9 or int(budget.lines_ms) > 0

        def _lines_text(self, battle: Any, lines: Dict[int, Dict[str, Any]], played: int) -> str:
            """One line for the window: "*Earthquake 0.64 | switch Corviknight 0.51 | ..." (the played action marked)."""
            parts = []
            for a, row in sorted(lines.items(), key=lambda kv: -(kv[1].get("value") or 0.0)):
                if row.get("value") is None:
                    continue
                parts.append("%s%s %.2f" % ("*" if int(a) == int(played) else "", self._action_label(battle, int(a)), float(row["value"])))
            if not parts:
                return ""
            turns = min(int(row.get("turns") or 0) for row in lines.values())
            return " | ".join(parts) + "  (the search's result after %d turn%s, 0 lost to 1 won; * played)" % (turns, "" if turns == 1 else "s")

        # -- the Luna arm --------------------------------------------------------------------------

        @staticmethod
        def _luna_redact(text: str) -> str:
            try:
                import luna_player as lp

                return lp.redact(text)
            except Exception:
                return "an error"

        async def _luna_decide(self, battle: Any, mask: Any, probs: List[float], net_action: int, mix_action: int,
                               view: Optional[SearchView], advice: Any, reason: str, prediction: Any, value: Optional[Dict[str, Any]],
                               deadline_at: float, loop: Any, decider: Any = None,
                               lines: Optional[Dict[int, Dict[str, Any]]] = None, solo: bool = False) -> Tuple[int, Dict[str, Any]]:
            """
            (the action to play, the record's "luna") for one decision of a Luna battle: the facts
            mix played from, the brief and the question written from them, Luna asked within what
            the decision's deadline leaves, and the rails. See luna_player.Luna.choose.
            """
            import luna_player as lp

            tag = battle.battle_tag
            actions: Dict[int, Tuple[Any, Dict[str, Any]]] = {}
            for a in range(ACTIONS):
                if mask[a] > 0.5:
                    target, kwargs = self.assembler.map_index_to_order(a, battle)
                    if target != "DEFAULT":
                        actions[a] = (target, dict(kwargs or {}))
            events = self._luna_events.get(tag)
            theirs = list(getattr(advice, "opponent", None) or []) if advice is not None else []
            facts = lp.Facts(
                mask=[float(m) for m in mask], actions=actions, probs=[float(p) for p in probs], net_action=int(net_action),
                mix_action=int(mix_action), shares=list(view.share) if view is not None else None,
                scores=list(view.score) if view is not None else None, search_reason=str(reason or ""),
                iterations=int(view.iterations) if view is not None else 0,
                prediction=[(str(n), float(p)) for n, p in prediction.ranked()] if prediction is not None else None,
                p_switch=getattr(prediction, "p_switch", None), p_tera=getattr(prediction, "p_tera", None),
                opponent_options=[(str(o.name), float(o.share), o.seen) for o in theirs] if view is not None else None,
                worlds=list(getattr(advice, "opponent_worlds", None) or []) if view is not None else None,
                win_est=(value or {}).get("win_est"), last_turn=list(getattr(events, "last", None) or []),
                this_turn=list(getattr(events, "turn", None) or []), lines=dict(lines or {}),
                revealed=([str(getattr(advice, "opponent_active", "") or "")] + [str(n) for n in getattr(advice, "opponent_revealed", None) or []])
                if advice is not None else [], solo=bool(solo),
                memory=list(self._solo_memory.get(tag) or []) if solo else [],
                earlier=list(events.earlier()) if (solo and events is not None and hasattr(events, "earlier")) else [])
            luna = decider if decider is not None else self.luna
            if luna is None:
                return lp.Luna.fall_back(facts, "off")
            if getattr(battle, "reviving", False):
                return luna.fall_back(facts, "revive")          # whom Revival Blessing revives: not a move or a switch
            if int(mix_action) not in actions or len(actions) < 2:
                return luna.fall_back(facts, "unmapped")
            ok, why = luna.client.available()
            if not ok:
                return luna.fall_back(facts, why)
            if deadline_at - RESERVE_S - loop.time() < float(luna.settings.min_wait):
                return luna.fall_back(facts, "no_time")         # before the brief is written: there is no time to ask anyway
            try:
                built = luna.build(battle, facts)
            except Exception:
                luna.brief_errors += 1
                return luna.fall_back(facts, "brief_error")
            action, info = await luna.choose(built, facts, deadline_at - RESERVE_S - loop.time())
            info["brief_ms"] = built.ms
            try:
                info["mix_text"] = self._action_label(battle, mix_action)
                if info.get("choice") is not None:
                    info["choice_text"] = self._action_label(battle, int(info["choice"]))
            except Exception:
                pass
            if luna.settings.save_briefs and info.get("asked"):
                info["brief"] = built.brief
                info["choices"] = [dict(c) for c in built.question.get("choices") or []]
            return int(action), info

        def _search_done(self, tag: str, start: float, end: float) -> Dict[str, float]:
            """
            This battle's search (asked at `start`, answered or given up at `end`, the loop's clock) is
            over: how many other searches ran beside it, on average over its time, by size ("deep":
            an arm's own budget, "today": today's). The A/B reads it to check that mix keeps its
            compute with deep searches beside it (search_player.game_summary "it_deep_beside").
            """
            mine = self._inflight.pop(tag, None)
            size = mine[0] if mine else "today"
            span = max(end - start, 1e-6)
            beside: Dict[str, float] = {}
            for other, (s_size, s_start) in list(self._inflight.items()):
                if end - s_start > 30.0:
                    self._inflight.pop(other, None)     # no search runs this long: a decision that raised left it
                    continue
                overlap = end - max(start, s_start)
                if other != tag and overlap > 0:
                    beside[s_size] = beside.get(s_size, 0.0) + overlap
            for other, s_size, s_start, s_end in self._searches_done:
                overlap = min(end, s_end) - max(start, s_start)
                if other != tag and overlap > 0:
                    beside[s_size] = beside.get(s_size, 0.0) + overlap
            self._searches_done.append((tag, size, start, end))
            return {k: round(v / span, 3) for k, v in sorted(beside.items())}

        # -- round 3's counter rules ---------------------------------------------------------------

        def _board(self, battle: Any, mask: Any, threat: Optional[Threat], advice: Any, settings: MixSettings) -> Any:
            """The setup_counter.Board of this decision: our legal actions as the assembler orders them, the battle, the search's sets."""
            import setup_counter as sc

            actions = {}
            for a in range(ACTIONS):
                if mask[a] > 0.5:
                    target, kwargs = self.assembler.map_index_to_order(a, battle)
                    if target != "DEFAULT":
                        actions[a] = (target, dict(kwargs or {}))
            return sc.build_board(battle, actions, threat=threat, advice=advice, read_floor=settings.counter_read,
                                  holder_hp=settings.counter_hp)

        def _counter_note(self, battle: Any, note: Dict[str, Any], net_action: int, view: SearchView, board: Any) -> Dict[str, Any]:
            """
            The decision record's "counter": the network's pick, the search's, mix's, the rule's and
            why, by number and by name; what the board held (the active's counters that work and
            those that cannot, the benched Pokemon that could and could not come in).
            """
            names = dict(getattr(board, "names", None) or {})

            def name(a: Any) -> str:
                return names.get(a) or self._action_label(battle, int(a))

            out: Dict[str, Any] = {"net": int(net_action), "search": int(view.best)}
            counter, swap, kept = note.get("counter"), note.get("other_side"), note.get("other_side_kept")
            first = counter or swap or kept
            out["mix"] = int(first["mix"])
            out["pick"] = int((swap or counter or kept)["pick"])
            out["changed"] = out["pick"] != out["mix"]
            out["text"] = {"net": name(net_action), "search": name(view.best), "mix": name(out["mix"]), "pick": name(out["pick"])}
            if counter is not None:
                out["counter"] = dict(counter)
                why = "%s: %s" % (counter["trigger"], counter["decided"])
                if counter.get("counter"):
                    why += " (%s)" % counter["counter"]
                if counter.get("holder"):
                    why += " (%s holds %s, %d%% HP)" % (name(counter["pick"]), counter["holder"], round(100 * float(counter.get("hp") or 0)))
                out["why"] = why
                out["board"] = {"counters": {str(a): v for a, v in (board.counters or {}).items()},
                                "failed": {str(a): v for a, v in (board.failed or {}).items()},
                                "holders": {str(a): [v[0], v[1]] for a, v in (board.holders or {}).items()},
                                "passed": {str(a): v for a, v in (board.passed or {}).items()},
                                "outspeeds": getattr(board, "outspeeds", None), "threat_hit": getattr(board, "threat_hit", None)}
            if kept is not None:
                # An attack into the boosted defence that stays: why nothing took its place.
                out["other_side_kept"] = dict(kept)
                text = "%s boosted: %s stays, %s" % (kept["boosted"], name(kept["mix"]), kept.get("why") or "?")
                out["why"] = (out["why"] + "; " + text) if out.get("why") else text
            if swap is not None:
                out["other_side"] = dict(swap)
                text = "%s boosted: %s does %d%% where %s does %d%%" % (
                    swap["boosted"], name(swap["pick"]), round(100 * swap["damage"][1]), name(swap["mix"]), round(100 * swap["damage"][0]))
                out["why"] = (out["why"] + "; " + text) if out.get("why") else text
            return out

        # -- the longer search on set-up turns (ThinkLonger) ------------------------------------

        def _last_setup(self, battle: Any, their_types: Sequence[str]) -> bool:
            """Whether the opponent's last action (this turn's so far, else last turn's) was a set-up move, as the tracker read it."""
            import setup_moves as sm

            tracker = self._trackers.get(battle.battle_tag)
            rows = getattr(tracker, "rows", None)
            if rows is None or getattr(tracker, "broken", None):
                return False
            side = rows.sides[rows.opp]
            act = side.action if side.action is not None else None
            last = ("move " + str(act["name"])) if act and act.get("type") == "move" else (side.last_action if act is None else "")
            return bool(last and last.startswith("move ") and sm.is_setup(last[5:], their_types))

        def _long_before(self, battle: Any) -> str:
            """Why the search is asked for longer before it is asked at all ("boosted +2", "set up last turn"), or ""."""
            import setup_moves as sm

            theirs = getattr(battle, "opponent_active_pokemon", None)
            if theirs is None:
                return ""
            boost = sm.largest_boost(theirs)
            if boost >= max(1, int(self.think_longer.boost)):
                return "boosted +%d" % boost
            their_types = [getattr(t, "name", str(t)) for t in (getattr(theirs, "types", None) or ()) if t is not None]
            if self._last_setup(battle, their_types):
                return "set up last turn"
            return ""

        def _read_of_setup(self, battle: Any, advice: Any) -> float:
            """What the search's own read of the opponent (its options and shares) puts on a set-up move."""
            import setup_moves as sm

            if getattr(battle, "force_switch", False):
                return 0.0                  # the opponent chooses nothing at this request
            theirs = getattr(battle, "opponent_active_pokemon", None)
            their_types = [getattr(t, "name", str(t)) for t in (getattr(theirs, "types", None) or ()) if t is not None]
            options = [(o.name, o.share) for o in (getattr(advice, "opponent", None) or [])]
            return sm.setup_share(options, their_types)

        def _long_allowed(self, battle: Any) -> bool:
            """The battle's timer leaves room for a long search (or the server has not said what it leaves)."""
            left = self._time_left.get(battle.battle_tag)
            return left is None or left >= self.think_longer.wait_s() + float(self.think_longer.min_time_left)

        def _spare_helpers(self) -> int:
            """Helpers no battle in progress needs: more helpers than battles being played and long searches with a second thread."""
            helpers = int(getattr(self.foulplay, "helpers", 1) or 1)
            playing = sum(1 for b in list(self.battles.values()) if not getattr(b, "finished", False))
            return helpers - playing - self._long_threads_used

        def _long_ask(self, battle: Any, why: str, when: str, loop: Any) -> Tuple[Any, Dict[str, Any], float]:
            """(the long search's future, its record, the loop time by which it is waited for at the latest)."""
            tl = self.think_longer
            wait = tl.wait_s()
            threads = 1
            if int(tl.threads) > 1:
                try:
                    if self._spare_helpers() > 0:
                        threads = int(tl.threads)
                except Exception:
                    threads = 1
            future = self._ask(battle, wait - RESERVE_S, search_ms=int(tl.ms), threads=threads)
            info: Dict[str, Any] = {"why": why, "when": when, "ms": int(tl.ms), "threads": threads}
            if threads > 1:
                self._long_threads_used += 1
                info["_thread"] = True          # given back when the decision is made
            return future, info, loop.time() + wait

        async def _await_advice(self, battle: Any, future: Any, until: float, loop: Any) -> Tuple[Any, str, float]:
            """
            (the answer filled onto our actions or None, why not or "", by how many ms the wait
            overran): the search's answer waited for until `until` - RESERVE_S on the loop's clock.
            """
            tag = battle.battle_tag
            left = until - RESERVE_S - loop.time()
            wait_until = loop.time() + max(0.0, left)
            wait_over_ms = 0.0
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
                    return None, "timeout", wait_over_ms
                return self._foulplay_fill(battle, raw), "", wait_over_ms
            except fb.FoulPlayBridgeError as exc:
                reason = "down" if exc.down else ("expired" if exc.expired else ("revive" if exc.revive else "error"))
                if reason == "error":
                    self._foulplay_problem("%s turn %s: %s" % (tag, getattr(battle, "turn", "?"), exc))
                return None, reason, wait_over_ms
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._foulplay_problem("%s turn %s: %r" % (tag, getattr(battle, "turn", "?"), exc))
                return None, "error", wait_over_ms

        # -- the record ------------------------------------------------------------------------

        def _record(self, battle: Any, mode: str, mask: Any, probs: List[float], net_action: int, view: Optional[SearchView],
                    advice: Any, action: int, source: str, reason: str, asked: bool, elapsed_ms: float,
                    net_ms: float = 0.0, wait_over_ms: float = 0.0, predicted: Optional[Dict[str, Any]] = None,
                    guard_info: Optional[Dict[str, Any]] = None, fix_info: Optional[Dict[str, Any]] = None,
                    long_info: Optional[Dict[str, Any]] = None, deadline_s: Optional[float] = None,
                    counter_info: Optional[Dict[str, Any]] = None, budget: Optional[SearchBudget] = None,
                    value: Optional[Dict[str, Any]] = None, fell_back: Optional[Dict[str, Any]] = None,
                    beside: Optional[Dict[str, float]] = None, luna_info: Optional[Dict[str, Any]] = None,
                    lines_info: Optional[Dict[str, Any]] = None) -> None:
            tag = battle.battle_tag
            request = getattr(battle, "last_request", None) or {}
            rec: Dict[str, Any] = {
                "battle": tag, "turn": int(getattr(battle, "turn", 0) or 0), "rqid": request.get("rqid"), "mode": mode,
                "arm": mode,                     # the battle's arm (the same as mode; the A/B's readers look for "arm")
                "legal": int((mask > 0.5).sum()), "mask": [int(m > 0.5) for m in mask],
                "forced": bool(getattr(battle, "force_switch", False)),
                "net": int(net_action), "search": None, "played": int(action), "source": source,
                "asked": bool(asked), "reason": reason, "ms": round(elapsed_ms, 1),
                "net_ms": round(net_ms, 1),      # request to the network's answer (its forward pass runs in the event loop)
                "wait_over_ms": round(wait_over_ms, 1),   # a timeout that fired this much after it was due: the loop was late
                "p_net": round(probs[net_action], 4), "p_played": round(probs[action], 4),
                "probs": [round(p, 4) for p in probs],
            }
            setting = self._game_settings.get(tag)
            if setting is not None:
                rec["setting"] = setting.name    # the game's setting (setting_picker): its budget and trust are the setting's
            # The network's own score of the position (v_exp, v_std, phi, phi_scale, win_est), every arm.
            rec.update(value or {})
            if asked:
                # The search asked for: its time per world and its worlds multiplier (the arm's budget).
                rec["asked_ms"] = int(budget.search_ms) if budget is not None else self._search_ms_of(tag)
                rec["worlds_x"] = int(budget.worlds_x) if budget is not None else 1
                if budget is not None and int(budget.threads) > 1:
                    rec["threads"] = int(budget.threads)
                if budget is not None and int(budget.workers) > 1:
                    rec["workers"] = int(budget.workers)
            if fell_back is not None:
                rec["deep_fallback"] = dict(fell_back)
            if beside is not None:
                # Other searches beside this one, on average over its time, by size (_search_done).
                rec["beside"] = dict(beside)
            if view is not None:
                best = view.best
                worlds = int(getattr(advice, "worlds", 0) or 0)
                rec.update({
                    "search": int(best), "iterations": view.iterations, "worlds": worlds,
                    "iterations_per_world": int(view.iterations / worlds) if worlds else None,
                    "full_width_turns": full_width_turns(view.iterations, worlds, rec["legal"], len(getattr(advice, "opponent", None) or [])),
                    "search_ms": round(float(getattr(advice, "wall_ms", 0.0) or 0.0), 1),
                    # Of that, the time inside the helper: the rest is its queue and, for a helper on
                    # another machine, the network there and back.
                    "helper_ms": round(float(getattr(advice, "ms", 0.0) or 0.0), 1),
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
            if guard_info is not None:
                # The guard fired: why, what it left out, whose pick was played, and what the plain veto would have played.
                rec["guard"] = guard_info
            if fix_info is not None:
                # A set-up fix (MixSettings.fixes) played another action than the plain veto: which, and what the veto would have played.
                rec["fix"] = fix_info
            if counter_info is not None:
                # Round 3's counter rules looked at this decision: the network's, the search's, mix's and the
                # rule's picks, why, and what the board held (the battle file's decision note has it all).
                rec["counter"] = counter_info
            if long_info is not None:
                # The longer search of a set-up turn: why, before or after the ordinary answer, its time and threads,
                # whether it answered in time and was used, and (after) what the ordinary answer would have played.
                rec["long"] = long_info
            if deadline_s is not None and abs(deadline_s - self._deadline_of(tag)) > 1e-6:
                rec["deadline_s"] = round(deadline_s, 3)
            if luna_info is not None:
                # The Luna arm: what Luna chose (our action id), its confidence and probabilities, the
                # latency, the input tokens, whether mix played instead and why, whether the blunder
                # guard overruled it; "mix" is what mix would have played. Never the key.
                rec["luna"] = luna_info
            if lines_info is not None:
                # A lines arm: the value of each line read (our action id -> the search's mean result at
                # its end), the whole turns it reached, what mix would have played and whether the lines
                # changed it ("mix-lines"), and the line for the window ("text").
                rec["lines"] = lines_info
            self._decisions.setdefault(tag, []).append(rec)

            game = self._games.get(tag)
            totals = self.search_totals
            per_mode = totals["by_mode"].setdefault(mode, {"decisions": 0, "asked": 0, "searched": 0, "fallbacks": {},
                                                           "disagreed": 0, "net_overruled": 0, "search_overruled": 0,
                                                           "tera_added": 0, "tera_dropped": 0,
                                                           "iterations": 0, "search_ms": 0.0, "decision_ms": 0.0,
                                                           "slowest_ms": 0.0, "late": 0})
            base_deadline = self._deadline_of(tag)
            late = elapsed_ms > (base_deadline if deadline_s is None else max(base_deadline, deadline_s)) * 1000.0
            if setting is not None and asked and game is not None and "budget_asked" in game:
                # With the setting's own budget, or (a mode with one, after its fall back) search_ms within deadline.
                own = budget is not None or self._budget_of(tag, mode) is None
                game["budget_asked"]["setting" if own else "fallback"] += 1
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
                    counts["worlds"] = counts.get("worlds", 0) + int(rec.get("worlds") or 0)
                    if rec.get("full_width_turns") is not None:
                        counts["fw_turns"] = counts.get("fw_turns", 0.0) + float(rec["full_width_turns"])
                        counts["fw_n"] = counts.get("fw_n", 0) + 1
                    if beside is not None:
                        # The paired compute check: this search with a deep search beside it most of its
                        # time, or with none at all (in between: neither).
                        deep = float(beside.get("deep", 0.0))
                        group = "deep_beside" if deep >= 0.5 else ("no_deep" if deep <= 0.0 else None)
                        if group is not None:
                            counts["it_" + group] = counts.get("it_" + group, 0) + int(view.iterations)
                            counts["n_" + group] = counts.get("n_" + group, 0) + 1
                    counts["disagreed"] += rec["disagree"]
                    counts["net_overruled"] += rec["net_overruled"]
                    counts["search_overruled"] += rec["search_overruled"]
                    counts["tera_added"] += rec["tera_added"]
                    counts["tera_dropped"] += rec["tera_dropped"]
                elif reason:
                    counts["fallbacks"][reason] = counts["fallbacks"].get(reason, 0) + 1
                if predicted is not None and "top" in predicted and not predicted.get("repeat"):
                    counts["pred_made"] = counts.get("pred_made", 0) + 1
                    if mode == "predict":
                        counts["pred_reads"] = counts.get("pred_reads", 0) + int(bool(predicted.get("confident")))
                        counts["pred_changed"] = counts.get("pred_changed", 0) + int(bool(predicted.get("changed")))
                        why = predicted.get("why") or "changed"
                        kept = counts.setdefault("pred_kept", {})
                        kept[why] = kept.get(why, 0) + 1
                if guard_info is not None:
                    counts["guard_fired"] = counts.get("guard_fired", 0) + 1
                    counts["guard_changed"] = counts.get("guard_changed", 0) + int(bool(guard_info.get("changed")))
                if fix_info is not None:
                    counts["fix_changed"] = counts.get("fix_changed", 0) + 1
                if counter_info is not None:
                    for key, n in _counter_counts(counter_info).items():
                        counts[key] = counts.get(key, 0) + n
                if long_info is not None:
                    counts["long_asked"] = counts.get("long_asked", 0) + 1
                    counts["long_used"] = counts.get("long_used", 0) + int(bool(long_info.get("used")))
                    counts["long_changed"] = counts.get("long_changed", 0) + int(bool(long_info.get("changed")))
                if luna_info is not None:
                    try:
                        import luna_player as lp

                        lp.add_counts(counts, luna_info)
                    except Exception:       # a count must never disturb a game
                        pass
                if lines_info is not None:
                    counts["lines_read"] = counts.get("lines_read", 0) + int(bool(lines_info.get("values")))
                    counts["lines_changed"] = counts.get("lines_changed", 0) + int(bool(lines_info.get("changed")))
            if game is not None and budget is not None and game.get("budget") is None:
                game["budget"] = budget.as_dict()
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
            if searched:
                # The search's size per decision: worlds searched, and the full-width turns ahead (full_width_turns).
                out["worlds_per_decision"] = round(game.get("worlds", 0) / searched, 2)
                out["full_width_turns"] = round(game["fw_turns"] / game["fw_n"], 3) if game.get("fw_n") else None
                # The paired compute check (_search_done): iterations summed over the searches that had a
                # deep search beside them, and over those that had none, with their counts.
                for name in ("it_deep_beside", "n_deep_beside", "it_no_deep", "n_no_deep"):
                    out[name] = int(game.get(name, 0))
            setting = self._game_settings.get(tag)
            own = self.budgets.get(game["mode"]) if setting is None else self._budget_of(tag, game["mode"])
            if own is not None:
                # The arm's own budget (the deep arm), and whether it fell back to today's in this game.
                out["budget"] = dict(game.get("budget") or own.as_dict())
                out["deep_fallback"] = dict(game["deep_fallback"]) if game.get("deep_fallback") else None
            if setting is not None:
                # The game's setting (setting_picker): its name, the budget it asks a decision with, what a
                # mode with a budget of its own falls back to, and how many searched decisions had each.
                out["setting"] = setting.name
                out["setting_budget"] = setting.asks()
                if own is not None:
                    out["setting_fallback"] = setting.today()
                out["budget_asked"] = dict(game.get("budget_asked") or {})
            if game["mode"] == "predict":
                # Predictions of the opponent's action: made, scored (the log says what it did),
                # the likeliest action was the one it took, confident reads and how many were
                # right, turns on which the mix choice was left and on how many of those the read
                # was right, and why the mix choice was kept on the others.
                for name in PRED_COUNTS:
                    out[name] = round(game[name], 4) if isinstance(game[name], float) else game[name]
                out["pred_kept"] = dict(game["pred_kept"])
            elif game["pred_made"] or (self.live_predictions and split_mode(game["mode"])[0] != "net" and self.foulplay_enabled):
                # Live predictions: made, scored, the likeliest right, what it did among the top 3.
                for name in LIVE_COUNTS:
                    out[name] = round(game[name], 4) if isinstance(game[name], float) else game[name]
            if self._guard_on(game["mode"]):
                out["guard_fired"], out["guard_changed"] = game["guard_fired"], game["guard_changed"]
            kind, rule = split_mode(game["mode"])
            played = self._arm_settings(game["mode"], self._mix_of(tag))
            if kind == "mix" and played.trust > 0 and (game["mode"] in ARM_FIXES or (rule or played.rule) == "veto"):
                # The guarded search (mix-trust, or ladder.py --trust on another veto mode such as mix-deep).
                out["trust"] = float(played.trust)
            if self._arm_settings(game["mode"]).fixes and split_mode(game["mode"])[0] == "mix":
                out["fix_changed"] = game["fix_changed"]
            if self._arm_settings(game["mode"]).board_rules and split_mode(game["mode"])[0] == "mix":
                for name in COUNTER_COUNTS:
                    out[name] = game[name]
            if self.think_longer is not None and split_mode(game["mode"])[0] in ("mix", "search") and self._long_mode(game["mode"]):
                out["long_asked"], out["long_used"], out["long_changed"] = game["long_asked"], game["long_used"], game["long_changed"]
            if game["mode"] in LINES_ARMS:
                # The lines: decisions that had some, and on how many they changed mix's pick ("mix-lines").
                out["lines_read"], out["lines_changed"] = int(game.get("lines_read", 0)), int(game.get("lines_changed", 0))
            if game["mode"] in DECIDERS:
                # The model in this game: asked, answered, played, overruled, agreed with mix, the fallbacks by
                # reason, the latency, the tokens and what they cost (luna_player.result_fields).
                try:
                    import luna_player as lp

                    out.update(lp.result_fields(game))
                except Exception:
                    pass
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
                text = ("opponent predicted on %d turns, scored %d: its likeliest action was right on %d (%.0f%%), what it did was in the top 3 on %d (%.0f%%)"
                        % (t["pred_made"], t["pred_scored"], t["pred_top1_right"], 100.0 * t["pred_top1_right"] / max(1, t["pred_scored"]),
                           t["pred_top3_right"], 100.0 * t["pred_top3_right"] / max(1, t["pred_scored"])))
                if "predict" in t["by_mode"]:
                    text += ("; %d confident reads, %d of %d scored right; the mix choice was left on %d turns, the read right on %d of %d scored (kept: %s)"
                             % (t["pred_reads"], t["pred_reads_right"], t["pred_reads_scored"], t["pred_changed"], t["pred_changed_right"],
                                t["pred_changed_scored"], ", ".join("%s %d" % kv for kv in sorted(t["pred_kept"].items()))))
                extra.append(text)
            if t["guard_fired"] or t["guard_errors"]:
                extra.append("the guard fired on %d decisions and played another action than the veto rule on %d%s"
                             % (t["guard_fired"], t["guard_changed"], (" (%d raised: the veto played)" % t["guard_errors"]) if t["guard_errors"] else ""))
            if t["fix_changed"]:
                extra.append("the set-up fixes played another action than the plain veto on %d decisions" % t["fix_changed"])
            if t["counter_looked"] or t["other_side_looked"] or t["counter_errors"]:
                extra.append(counter_text(t) + ((" (%d raised: mix played)" % t["counter_errors"]) if t["counter_errors"] else ""))
            if t["long_asked"] or t["long_errors"]:
                extra.append("the longer search on set-up turns was asked %d times, used %d, and changed the move %d times%s"
                             % (t["long_asked"], t["long_used"], t["long_changed"],
                                (" (%d raised: the ordinary search played)" % t["long_errors"]) if t["long_errors"] else ""))
            if t.get("deep_fallbacks"):
                extra.append("the deep arm played today's search budget for the rest of %d games (the clock or a slow decision)"
                             % t["deep_fallbacks"])
            if t["predict_errors"] or t["trackers_broken"]:
                extra.append("%d predictions raised, %d battles could not be followed" % (t["predict_errors"], t["trackers_broken"]))
            if self.search_totals["after_end"]:
                extra.append("%d requests after a battle had ended" % self.search_totals["after_end"])
            if self.search_totals["exceptions"]:
                extra.append("%d decisions raised and were played at random" % self.search_totals["exceptions"])
            if self.search_totals.get("luna_errors"):
                extra.append("%d Luna decisions raised and were played by mix" % self.search_totals["luna_errors"])
            lag = self.loop_lag
            if lag["checks"]:
                extra.append("the event loop ran more than 0.5 s late %d times in %d checks (%.0f ms at the worst, %.1f s in all)"
                             % (lag["over_500_ms"], lag["checks"], lag["max_ms"], lag["late_ms_total"] / 1000.0))
            if self.search_totals["noted_at_send"]:
                extra.append("%d orders that did not come from the player's own choice were noted as sent" % self.search_totals["noted_at_send"])
            if extra:
                lines.append(", ".join(extra))
            if self.search_totals.get("luna_decisions"):
                try:
                    import luna_player as lp

                    # One model in this run: its name. Two: the line is the sum of both, and says so.
                    names = sorted({str(getattr(who, "label", "Luna")) for who in self.deciders.values()}) or ["Luna"]
                    lines.append(lp.counts_text(self.search_totals, names[0] if len(names) == 1 else "The models together"))
                except Exception:
                    pass
            return "\n".join(lines)

    _SEARCH_PLAYER = SearchPlayer
    return _SEARCH_PLAYER

