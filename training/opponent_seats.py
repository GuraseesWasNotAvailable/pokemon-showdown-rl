"""
Opponent seats: in a share of the PPO pairs one seat is an opponent of a fixed kind instead of a
second trained seat.

    python train.py --mode ppo ... --opponent-seats setupnet:0.15,psbot:0.05,foulplay:0.05 \
        --setupnet-p 0.5 --seat-foulplay-iterations N

Without --opponent-seats nothing here is used: every rollout worker is worker.RolloutWorker, as
before. bench/notes/OPPONENT_SEATS.md has the notes and the measurements.

The kinds:

  setupnet   The network itself, asked through the worker's ordinary STEP event, so its move comes
             from the same batched inference call as every other. When one of the set-up rules
             applies (setup_rules.py: Sleep Talk when asleep, recover when boosted and hurt, a
             sleep move when the target can sleep, a set-up move when not in danger) the rule's
             move replaces the network's with probability p (--setupnet-p). A switch the network
             chose is kept (it stands in for decide's rule 1). The trained seat of the pair so
             faces a network that sets up.
  psbot      PSBot (psbot_bridge.PSBotMixin, heuristic/engine.js).
  foulplay   Foul Play at a fixed number of search iterations per sampled world
             (--seat-foulplay-iterations; the bridge's `iterations` option), so its strength does
             not depend on how loaded the box is.

What reaches the learner: nothing of an opponent seat. Its battles never put a DONE (or an
IMITATION) event on the worker's queue. Where the network acts for the seat (always for
setupnet; as the fallback for psbot and foulplay when they have no answer) the seat puts the
ordinary STEP event, and one CLEANUP at the end of the battle, which makes the worker drop those
steps, the side's history in the inference actor and its episode slot. The other seat of the pair
is worker.SyncBridgePlayer, unchanged, trained on like any self-play seat.

What is here, all of it built ON worker.py and none of it IN it:

  OpponentSeatsConfig      the settings (not a part of RunConfig: who sits in the other seat is the
                           rollout workers' business alone; the learner and the inference actor
                           are built from the same RunConfig with or without seats)
  add_arguments, from_args the flags, for train.py
  plan_seats               which worker gets which seat kinds, in its LAST pairs, and which seat
                           ("a" or "b") the opponent takes in each pair
  SetupNetSeatPlayer, PSBotSeatPlayer, FoulPlaySeatPlayer   the seats
  SeatsVectorWrapper       worker.PokeEnvVectorWrapper whose last pairs are opponent pairs
  OpponentSeatsRolloutWorker   worker.RolloutWorker on that wrapper, with the counts in its heartbeat
  worker_class, worker_kwargs  what train.py hands Ray for each worker
  report_line              the "[Seats]" line for the trainer's log

Wiring into train.py (not done here):

    seats = opponent_seats.from_args(args)                    # None without --opponent-seats
    plan = opponent_seats.plan_seats(pairs_per_worker, rooms_per_pair, seats) if seats else None
    logger.info(opponent_seats.describe_plan(plan)) if plan else None
    for i, port in enumerate(server_ports):
        entry = plan["workers"][i] if plan else None
        cls = opponent_seats.worker_class(entry)              # worker.RolloutWorker when entry has no seats
        Remote = ray.remote(num_cpus=1)(cls).options(max_restarts=-1, max_task_retries=0)
        Remote.remote(cfg=cfg_ref, inference_actor=infer, learner_actor=learner, pairs=count,
                      server_port=port, **opponent_seats.worker_kwargs(entry))
    ... in the telemetry loop: logger.info(opponent_seats.report_line([b.get("opponent_seats") for b in wstats]))

It never stops or stalls training: a seat whose kind cannot be used in a worker (PSBot's engine
or Foul Play cannot start) is made as an ordinary self-play pair instead, and the heartbeat says
so; a decision PSBot or Foul Play cannot answer is the network's; a seat that crashes in a
decision plays a random legal move.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from poke_env import AccountConfiguration
from poke_env.battle import Move
from poke_env.player import DefaultBattleOrder

import setup_moves as sm
import setup_rules
import worker
from config import RunConfig
from foulplay_bridge import FoulPlayBridge, FoulPlayMixin   # stdlib only; a helper starts only for a worker with Foul Play seats
from worker import IMITATION_MODES, PokeEnvVectorWrapper, RolloutWorker, SyncBridgePlayer, join_lobby, mk_name, safe_send, wait_for_login

logger = logging.getLogger(__name__)

SEAT_KINDS: Tuple[str, ...] = ("setupnet", "psbot", "foulplay")
# "frozen" (frozen_pool.py): a fixed checkpoint plays the seat on its own observation, through the
# inference actors' frozen pool. It is a kind only for a run that asks for it: every plan, heartbeat and
# line of a run without it is built over SEAT_KINDS alone, as before.
ALL_KINDS: Tuple[str, ...] = SEAT_KINDS + ("frozen",)


def kinds_of(settings: Optional["OpponentSeatsConfig"]) -> Tuple[str, ...]:
    """The kinds a run's plan is built over: SEAT_KINDS, and "frozen" after them when it has a share."""
    if settings is not None and settings.share("frozen") > 0:
        return ALL_KINDS
    return SEAT_KINDS

# Foul Play: why the network moved for it. The first four hand the REST of the battle to the network.
HANDED_OVER = ("timeout", "expired", "error", "age")
SINGLE = ("revive", "illegal")          # one decision only: Foul Play is asked again at the next

# Seconds a round of a helper's Foul Play sides may take when the helpers are sized by the rule.
ROUND_TARGET_S = 2.2


# ---------------------------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SetupNetSettings:
    # Per decision on which a set-up rule applies: the probability that the rule's move is played
    # instead of the network's. 0 is the plain network (a mirror match), 1 always sets up when a
    # rule says so. Pick it with the gate (bench/notes/OPPONENT_SEATS.md): the network should win
    # 55 to 80% against the seat.
    p: float = 0.5
    # Set-up strength: rule 5 keeps raising a stat until it is at this stage (1 to 6). decide's
    # set-up player uses 4.
    max_stage: int = setup_rules.MAX_STAGE
    # Rule 3 (recover when boosted and below half HP).
    recover: bool = True
    # A switch the network chose is kept (decide's rule 1: a bad matchup goes out). False: the
    # rules may replace a switch too.
    keep_switches: bool = True

    def __post_init__(self):
        if not 0.0 <= float(self.p) <= 1.0:
            raise ValueError("setupnet p must be between 0 and 1")
        if not 1 <= int(self.max_stage) <= 6:
            raise ValueError("setupnet max_stage must be between 1 and 6")


@dataclass(frozen=True)
class FoulPlaySeatSettings:
    # Search iterations per sampled world for every decision (the bridge's `iterations`). Stage 0
    # picks it: the median Foul Play reaches at 200 ms on an idle box. 0 is refused when a
    # foulplay share is asked for.
    iterations: int = 0
    # The search time those iterations stand for. It sizes the helpers (decision_seconds) and is
    # the "ms" the helper is sent next to the iterations.
    sizing_ms: int = 200
    # Helper processes IN TOTAL over the run. 0: as many as it takes for every Foul Play side to
    # be answered about every ROUND_TARGET_S.
    helpers: int = 0
    # Never more helper processes than this over the run (the benchmark's Foul Play keeps its cores).
    max_helpers: int = 36
    # How long a decision waits for Foul Play. After that the network plays the rest of the battle.
    timeout_s: float = 20.0
    # A battle older than this is finished by the network (rlspawn.ts closes rooms at 5 minutes).
    battle_budget_s: float = 150.0

    def __post_init__(self):
        if int(self.iterations) < 0 or int(self.sizing_ms) < 1 or int(self.helpers) < 0 or int(self.max_helpers) < 1:
            raise ValueError("Foul Play seat: iterations and helpers must not be negative, sizing_ms and max_helpers at least 1")
        if not float(self.timeout_s) > 0 or not float(self.battle_budget_s) > 0:
            raise ValueError("Foul Play seat: timeout_s and battle_budget_s must be above zero")


@dataclass(frozen=True)
class OpponentSeatsConfig:
    # (kind, share of ALL the run's pairs), e.g. (("setupnet", 0.15), ("psbot", 0.05)).
    shares: Tuple[Tuple[str, float], ...] = ()
    setupnet: SetupNetSettings = field(default_factory=SetupNetSettings)
    foulplay: FoulPlaySeatSettings = field(default_factory=FoulPlaySeatSettings)
    psbot_timeout_s: float = 20.0
    # Seeds the setupnet seats' draws (with the worker's port and the pair's index).
    seed: int = 0
    # The frozen seats' nets (frozen_pool.FrozenSeatSettings); None without a frozen share.
    frozen: Any = None

    def __post_init__(self):
        seen = set()
        for kind, share in self.shares:
            if kind not in ALL_KINDS:
                raise ValueError("unknown opponent seat kind %r (known: %s)" % (kind, ", ".join(ALL_KINDS)))
            if kind in seen:
                raise ValueError("opponent seat kind %r given twice" % kind)
            seen.add(kind)
            if not 0.0 <= float(share) <= 1.0:
                raise ValueError("the share of %s must be between 0 and 1" % kind)
        if sum(float(s) for _k, s in self.shares) > 1.0 + 1e-9:
            raise ValueError("the opponent seat shares add up to more than 1")
        if self.share("foulplay") > 0 and int(self.foulplay.iterations) <= 0:
            raise ValueError("a foulplay share needs --seat-foulplay-iterations N (Stage 0 picks N)")
        if self.share("frozen") > 0:
            if self.frozen is None or not self.frozen.slots:
                raise ValueError("a frozen share needs --frozen-nets NAME=PATH:W and/or --frozen-past K:W")
            self.frozen.slot_weights()               # raises when every weight is 0

    def share(self, kind: str) -> float:
        return float(dict(self.shares).get(kind, 0.0))

    @property
    def any(self) -> bool:
        return any(float(s) > 0 for _k, s in self.shares)


def parse_shares(spec: str) -> Tuple[Tuple[str, float], ...]:
    """ "setupnet:0.15,psbot:0.05" -> (("setupnet", 0.15), ("psbot", 0.05)). Raises ValueError on anything else."""
    out: List[Tuple[str, float]] = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        kind, sep, value = part.partition(":")
        kind = kind.strip().lower()
        if not sep:
            raise ValueError("opponent seat %r: give it as kind:share, e.g. setupnet:0.15" % part)
        try:
            share = float(value)
        except ValueError:
            raise ValueError("opponent seat %r: the share is not a number" % part) from None
        if not math.isfinite(share):
            raise ValueError("opponent seat %r: the share is not a number" % part)
        out.append((kind, share))
    return tuple(out)


def add_arguments(parser: Any) -> None:
    """The flags of the opponent seats, on train.py's parser (or any argparse parser)."""
    g = parser.add_argument_group("opponent seats (opponent_seats.py, bench/notes/OPPONENT_SEATS.md)")
    g.add_argument("--opponent-seats", default="",
                   help="kind:share,... e.g. setupnet:0.15,psbot:0.05,foulplay:0.05: that share of ALL the run's pairs has "
                        "that opponent in one seat, never trained on. Kinds: " + ", ".join(ALL_KINDS))
    g.add_argument("--setupnet-p", type=float, default=SetupNetSettings.p,
                   help="setupnet: probability that a set-up rule's move replaces the network's when a rule applies")
    g.add_argument("--setupnet-max-stage", type=int, default=SetupNetSettings.max_stage,
                   help="setupnet: set-up strength, keep raising a stat up to this stage (1 to 6)")
    g.add_argument("--seat-foulplay-iterations", type=int, default=0,
                   help="foulplay seats: search iterations per sampled world for every decision (needed with a foulplay share)")
    g.add_argument("--seat-foulplay-ms", type=int, default=FoulPlaySeatSettings.sizing_ms,
                   help="foulplay seats: the search time the iterations stand for (sizes the helpers)")
    g.add_argument("--seat-foulplay-helpers", type=int, default=0, help="foulplay seats: helper processes over the run (0: by the rule)")
    g.add_argument("--seat-foulplay-max-helpers", type=int, default=FoulPlaySeatSettings.max_helpers,
                   help="foulplay seats: never more helpers than this")
    g.add_argument("--seat-foulplay-timeout", type=float, default=FoulPlaySeatSettings.timeout_s,
                   help="foulplay seats: seconds a decision waits for Foul Play before the network plays the rest of that battle")
    g.add_argument("--opponent-seats-seed", type=int, default=0, help="seeds the setupnet seats' draws")
    import frozen_pool
    frozen_pool.add_arguments(g)


def from_args(args: Any) -> Optional[OpponentSeatsConfig]:
    """The settings from add_arguments()'s flags; None when no opponent seat is asked for. Raises ValueError on a bad value."""
    shares = parse_shares(getattr(args, "opponent_seats", "") or "")
    if not any(s > 0 for _k, s in shares):
        # Still checked, so that a typo in a kind is not silently a run without seats.
        OpponentSeatsConfig(shares=shares)
        return None
    frozen = None
    if float(dict(shares).get("frozen", 0.0)) > 0:
        import frozen_pool
        frozen = frozen_pool.settings_from_args(args)
    return OpponentSeatsConfig(
        shares=shares,
        setupnet=SetupNetSettings(p=float(getattr(args, "setupnet_p", SetupNetSettings.p)),
                                  max_stage=int(getattr(args, "setupnet_max_stage", SetupNetSettings.max_stage))),
        foulplay=FoulPlaySeatSettings(iterations=int(getattr(args, "seat_foulplay_iterations", 0) or 0),
                                      sizing_ms=int(getattr(args, "seat_foulplay_ms", FoulPlaySeatSettings.sizing_ms)),
                                      helpers=int(getattr(args, "seat_foulplay_helpers", 0) or 0),
                                      max_helpers=int(getattr(args, "seat_foulplay_max_helpers", FoulPlaySeatSettings.max_helpers)),
                                      timeout_s=float(getattr(args, "seat_foulplay_timeout", FoulPlaySeatSettings.timeout_s))),
        seed=int(getattr(args, "opponent_seats_seed", 0) or 0),
        frozen=frozen,
    )


# ---------------------------------------------------------------------------------------------
# the plan: which worker, which kinds, which seat, how many Foul Play helpers
# ---------------------------------------------------------------------------------------------

def decision_seconds(search_ms: int) -> float:
    """What one Foul Play decision takes in a helper (bench/FOULPLAY_BRIDGE.md, "Cost": 126 ms at 50, 233 ms at 100)."""
    return (2.2 * float(search_ms) + 15.0) / 1000.0


def default_helpers(fp_sides: int, search_ms: int) -> int:
    """Helpers for `fp_sides` battles against Foul Play so that each is answered about every ROUND_TARGET_S."""
    if fp_sides <= 0:
        return 0
    return max(1, int(math.ceil(fp_sides * decision_seconds(search_ms) / ROUND_TARGET_S - 1e-9)))


def plan_seats(pairs_per_worker: Sequence[int], rooms_per_pair: int, settings: Optional[OpponentSeatsConfig]) -> Dict[str, Any]:
    """
    {"workers": [{"seats": [{"kind", "seat"}, ...], "helpers": n, "settings": settings}, ...], and
    the totals}. One entry per rollout worker, in train.py's order. A worker's seats are its LAST
    len(seats) pairs; "seat" is the opponent's seat in that pair ("a": the opponent is p1, "b": the
    network is p1, as /rlautospawn makes the first-named player p1).

    The shares are of ALL the run's pairs, not of each worker's (a twentieth of 4 pairs is
    nothing): each kind wants round(share x pairs), at least one when its share is above 0, and
    when that rounding asks for more pairs than the run has, the kind with the most gives one
    back (of equals the later one in SEAT_KINDS). The pairs are dealt one at a time round the workers, every kind going on from where the
    previous one stopped, so the kinds spread over different workers. Foul Play pairs go to at
    most max_helpers workers (each needs a helper of its own). The opponent's seat alternates
    within each kind, so the network is p1 in half of its pairs and p2 in the other half.

    Nothing in it is random: the same arguments give the same plan, so a worker that Ray starts
    again gets the seats it had.
    """
    pairs = [max(0, int(p)) for p in pairs_per_worker]
    total = sum(pairs)
    n = len(pairs)
    rooms = max(1, int(rooms_per_pair))
    kinds = kinds_of(settings)
    workers: List[Dict[str, Any]] = [{"seats": [], "helpers": 0, "settings": settings} for _ in pairs]
    out: Dict[str, Any] = {"workers": workers, "pairs_total": total, "rooms_per_pair": rooms,
                           "wanted": {k: 0 for k in kinds}, "placed": {k: 0 for k in kinds},
                           "net_p1": {k: 0 for k in kinds}, "net_p2": {k: 0 for k in kinds},
                           "helpers": 0, "helpers_capped": 0, "fp_hosts": 0}
    if settings is None or total <= 0 or not settings.any:
        return out
    wanted = {}
    for kind in kinds:
        share = settings.share(kind)
        wanted[kind] = min(total, max(1, int(round(share * total)))) if share > 0 else 0
    while sum(wanted.values()) > total:
        # the kind that wants most gives one back; of equals the later one (Foul Play costs most)
        k = max(kinds, key=lambda kind: (wanted[kind], kinds.index(kind)))
        wanted[k] -= 1
    out["wanted"] = dict(wanted)

    free = list(pairs)
    count = [{k: 0 for k in kinds} for _ in pairs]
    fp_cap = max(1, int(settings.foulplay.max_helpers))
    fp_hosts = 0
    pointer = 0
    for kind in kinds:
        left = wanted[kind]
        misses = 0
        while left > 0 and misses < n:
            i = pointer % n
            pointer += 1
            new_fp_host = kind == "foulplay" and count[i][kind] == 0
            if free[i] > 0 and not (new_fp_host and fp_hosts >= fp_cap):
                count[i][kind] += 1
                free[i] -= 1
                left -= 1
                misses = 0
                fp_hosts += int(new_fp_host)
            else:
                misses += 1

    for kind in kinds:
        hosts = [i for i in range(n) if count[i][kind]]
        for h, i in enumerate(hosts):
            for j in range(count[i][kind]):
                seat = "b" if (h + j) % 2 == 0 else "a"
                workers[i]["seats"].append({"kind": kind, "seat": seat})
                out["net_p1" if seat == "b" else "net_p2"][kind] += 1
        out["placed"][kind] = sum(count[i][kind] for i in range(n))

    # Foul Play helpers, per worker that hosts Foul Play pairs.
    fs = settings.foulplay
    hosts = [i for i in range(n) if count[i]["foulplay"]]
    if hosts:
        by_rule = not int(fs.helpers)
        helpers = {i: (default_helpers(count[i]["foulplay"] * rooms, fs.sizing_ms) if by_rule else 1) for i in hosts}
        spare = 0 if by_rule else int(fs.helpers) - len(hosts)
        while spare > 0:
            open_ = [i for i in hosts if helpers[i] < count[i]["foulplay"] * rooms]
            if not open_:
                break
            i = max(open_, key=lambda j: (count[j]["foulplay"] * rooms / helpers[j], -j))
            helpers[i] += 1
            spare -= 1
        asked = sum(helpers.values()) if by_rule else max(int(fs.helpers), len(hosts))
        while sum(helpers.values()) > int(fs.max_helpers):
            i = max(hosts, key=lambda j: (helpers[j], -j))
            if helpers[i] <= 1:
                break
            helpers[i] -= 1
        if asked > int(fs.max_helpers):
            out["helpers_capped"] = int(fs.max_helpers)
        for i in hosts:
            workers[i]["helpers"] = helpers[i]
        out["helpers"] = sum(helpers.values())
        out["fp_hosts"] = len(hosts)
    return out


def describe_plan(plan: Dict[str, Any]) -> str:
    """One line for the trainer's log at start."""
    settings: Optional[OpponentSeatsConfig] = next((w.get("settings") for w in plan["workers"] if w.get("settings")), None)
    if settings is None or not any(plan["placed"].values()):
        return "Opponent seats: none (%d pairs)" % plan["pairs_total"]
    parts = []
    for kind in ALL_KINDS:
        placed = plan["placed"].get(kind, 0)
        if not plan["wanted"].get(kind):
            continue
        hosts = sum(1 for w in plan["workers"] if any(s["kind"] == kind for s in w["seats"]))
        text = "%s %d of %d pairs (asked %.1f%%, %d wanted) on %d workers, the network p1 in %d and p2 in %d" % (
            kind, placed, plan["pairs_total"], 100.0 * settings.share(kind), plan["wanted"][kind], hosts,
            plan["net_p1"][kind], plan["net_p2"][kind])
        if kind == "setupnet":
            text += " (p %.2f, max stage %d)" % (settings.setupnet.p, settings.setupnet.max_stage)
        if kind == "foulplay":
            text += " (%d iterations per world, %d helpers%s)" % (
                settings.foulplay.iterations, plan["helpers"],
                ", CAPPED at %d" % plan["helpers_capped"] if plan["helpers_capped"] else "")
        if kind == "frozen" and settings.frozen is not None:
            import frozen_pool
            text += " (%s)" % frozen_pool.describe(settings.frozen)
        if placed < plan["wanted"][kind]:
            text += ", FEWER THAN WANTED"
        parts.append(text)
    return "Opponent seats: " + "; ".join(parts)


# ---------------------------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------------------------

class SeatStats:
    """
    What one worker's seats of one kind did. Written on the players' event-loop thread, read by
    the heartbeat on another: plain numbers, each read a decision out of date at worst.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        # Finished games counted for the kind, by the NETWORK's seat in the pair. A psbot or
        # foulplay game in which the network moved for the seat is a mixed game instead.
        self.games = {"p1": 0, "p2": 0}
        self.wins = {"p1": 0, "p2": 0}             # the trained network's
        self.ties = {"p1": 0, "p2": 0}
        self.mixed_games = 0
        self.decisions = 0                         # every decision the seat played
        self.bot_decisions = 0                     # played by the kind: PSBot's, Foul Play's, setupnet's rule moves
        self.net_decisions = 0                     # the network's own move played
        self.crashes = 0                           # choose_move raised: a random legal move was played
        self.extra: Dict[str, int] = {}

    def bump(self, key: str, n: int = 1) -> None:
        self.extra[key] = self.extra.get(key, 0) + n

    def as_dict(self) -> Dict[str, Any]:
        return {"games_p1": self.games["p1"], "wins_p1": self.wins["p1"], "ties_p1": self.ties["p1"],
                "games_p2": self.games["p2"], "wins_p2": self.wins["p2"], "ties_p2": self.ties["p2"],
                "mixed_games": self.mixed_games, "decisions": self.decisions, "bot_decisions": self.bot_decisions,
                "net_decisions": self.net_decisions, "crashes": self.crashes, **dict(self.extra)}


# ---------------------------------------------------------------------------------------------
# the seats
# ---------------------------------------------------------------------------------------------

class _NoEvents:
    """Stands in for the worker's event queue while a base-class method would announce something this seat must not."""

    def put_nowait(self, _event: Any) -> None:
        pass


_NO_EVENTS = _NoEvents()


class OpponentSeat(SyncBridgePlayer):
    """
    The common part of every opponent seat: worker.SyncBridgePlayer whose battles never reach the
    learner.

    Events it puts on the worker's queue: STEP, only for a decision the network makes for it, and
    one CLEANUP at the end of a battle in which it holds an episode slot (it takes one at its
    first such decision). Never DONE and never IMITATION.
    """

    seat_kind = "opponent"

    def __init__(self, *args: Any, seat_stats: SeatStats, seat: str, **kwargs: Any):
        self._seat_stats = seat_stats
        self._network_role = "p2" if seat == "a" else "p1"     # /rlautospawn makes seat "a" p1
        self._seat_net_moves: Dict[str, int] = {}              # tag -> decisions the network made here
        self._seat_decisions_seen = 0
        super().__init__(*args, **kwargs)

    # ---- one decision ------------------------------------------------------------------------

    def _decision_start(self, battle: Any) -> Tuple[str, list]:
        tag = self._get_unique_tag(battle)
        now = time.time()
        self._last_act_time[tag] = now                          # read by worker.py's stall and zombie repair
        self._battle_starts.setdefault(tag, now)
        events = self._battle_events.pop(battle.battle_tag, [])   # what happened since this side's last decision
        self._seat_decisions_seen += 1
        if self._seat_decisions_seen % 512 == 0:
            self._prune_books()
        return tag, events

    async def _network_action(self, battle: Any, tag: str, events: list) -> Optional[Tuple[int, Any, Dict[str, Any]]]:
        """
        (action index, action object, order kwargs): the network's choice, asked exactly as
        worker.SyncBridgePlayer.choose_move asks (an episode slot at the side's first such
        decision, the observation, a STEP event, the action from the worker). None when the battle
        went while a slot was awaited.
        """
        if tag not in self._episode_slot_state:
            await self.ep_sem.acquire()
            if tag not in self._battle_starts:
                self._safe_release()
                return None
            self._episode_slot_state[tag] = "acquired"
        obs_flat = self.assembler.assemble(battle, events=events)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.action_futures[tag] = (fut, loop)
        self.event_queue.put_nowait(("STEP", tag, obs_flat, 0.0))
        action_idx = await fut
        action_obj, kwargs = self.assembler.map_index_to_order(action_idx, battle)
        return int(action_idx), action_obj, kwargs

    def _order_of(self, action_obj: Any, kwargs: Dict[str, Any]) -> Any:
        return DefaultBattleOrder() if isinstance(action_obj, str) else self.create_order(action_obj, **kwargs)

    async def _network_order(self, battle: Any, tag: str, events: list) -> Any:
        """The network's move as an order, counted as a move the network made for this seat."""
        answer = await self._network_action(battle, tag, events)
        if answer is None:
            return DefaultBattleOrder()
        _idx, action_obj, kwargs = answer
        self._seat_net_moves[tag] = self._seat_net_moves.get(tag, 0) + 1
        self._seat_stats.net_decisions += 1
        return self._order_of(action_obj, kwargs)

    def _crashed(self, battle: Any, exc: BaseException) -> Any:
        self._seat_stats.crashes += 1
        logger.error(f"{type(self).__name__} crashed for {battle.battle_tag}: {exc!r}. Using fallback.")
        return self.choose_random_move(battle)

    # ---- the end of a battle -----------------------------------------------------------------

    def _is_mixed(self, tag: str) -> bool:
        """A game not to be counted for the kind: the network moved for the seat."""
        return bool(self._seat_net_moves.get(tag))

    def _drop_books(self, tag: str) -> None:
        self._seat_net_moves.pop(tag, None)

    def _prune_books(self) -> None:
        """Books of battles that left by a way that does not pass here."""
        for tag in [t for t in self._seat_net_moves if t not in self._battle_starts]:
            self._drop_books(tag)

    def _count_result(self, battle: Any, tag: str) -> None:
        stats, role = self._seat_stats, self._network_role
        if self._is_mixed(tag):
            stats.mixed_games += 1
            return
        stats.games[role] += 1
        if getattr(battle, "lost", False):                      # the seat lost: the network won
            stats.wins[role] += 1
        elif not getattr(battle, "won", False):
            stats.ties[role] += 1

    def _quietly(self, method: Any, *args: Any) -> None:
        """
        A base-class method with its events held back. worker.SyncBridgePlayer announces the end of
        a battle with DONE (the worker then hands the side's steps to the learner) and a given-up
        battle with CLEANUP. This seat's steps must not reach the learner, and it puts a CLEANUP
        only when it holds a slot. The swap is on the players' own thread, inside one call: the
        worker reads the queue object, never this attribute.
        """
        mine, self.event_queue = self.event_queue, _NO_EVENTS
        try:
            method(*args)
        finally:
            self.event_queue = mine

    def _battle_finished_callback(self, battle: Any) -> None:
        tag = self._get_unique_tag(battle)
        held_slot = tag in self._episode_slot_state
        try:
            self._count_result(battle, tag)
        except Exception:
            pass
        self._drop_books(tag)
        self._quietly(super()._battle_finished_callback, battle)
        if held_slot:
            self.event_queue.put_nowait(("CLEANUP", tag, None, None))

    def _cleanup_local_battle(self, tag: str) -> None:
        """A battle given up without a result (a stall, a zombie): as above, without a game to count."""
        held_slot = tag in self._episode_slot_state
        self._drop_books(tag)
        self._quietly(super()._cleanup_local_battle, tag)
        if held_slot:
            self.event_queue.put_nowait(("CLEANUP", tag, None, None))

    def _room_closed(self, room_id: str) -> None:
        """The server closed the room without a result. The base class puts a CLEANUP when a slot is held; never a DONE."""
        try:
            battle = self._battles.get(room_id)
            tag = self._get_unique_tag(battle) if battle is not None else f"{self.username}_{room_id}"
            self._drop_books(tag)
        except Exception:
            pass
        super()._room_closed(room_id)


class SetupNetSeatPlayer(OpponentSeat):
    """
    The network, with probability p replaced by a set-up rule's move where one applies.

    Every decision is the network's STEP (so the inference actor's history of the side is whole),
    then setupnet_order() picks what is played. Counts in its SeatStats.extra:
      eligible       decisions the rules were asked about (move requests; switches the network
                     chose are not, unless keep_switches is off)
      rule_ready     of those, decisions on which a rule applied
      drawn          of those, decisions on which the draw (probability p) came up
      agreed         drawn, and the network had chosen that very move itself
      override_<kind> drawn and played instead of the network's move, by rule
                     (sleeptalk, recover, sleep, setup)
      net_setup      the network's own choice was a set-up move (setup_moves.is_setup)
      setup_played   a set-up move was played, by the network or by a rule
    """

    seat_kind = "setupnet"

    def __init__(self, *args: Any, setupnet: SetupNetSettings, rng_seed: str = "", **kwargs: Any):
        self._setup = setupnet
        self._rng = random.Random(rng_seed) if rng_seed else random.Random()
        super().__init__(*args, **kwargs)

    async def choose_move(self, battle: Any):
        try:
            if getattr(battle, "finished", False):
                return DefaultBattleOrder()
            tag, events = self._decision_start(battle)
            answer = await self._network_action(battle, tag, events)
            if answer is None:
                return DefaultBattleOrder()
            _idx, action_obj, kwargs = answer
            order, _kind = self.setupnet_order(battle, action_obj, kwargs)
            return order
        except Exception as e:
            return self._crashed(battle, e)

    def setupnet_order(self, battle: Any, action_obj: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """
        (order, kind): the order played for the network's choice `action_obj` (as the assembler's
        map_index_to_order gives it), and the rule's kind when a rule's move replaced it (None when
        the network's own move is played).
        """
        stats, cfg = self._seat_stats, self._setup
        stats.decisions += 1
        net_move = isinstance(action_obj, Move)
        me = battle.active_pokemon
        my_types = setup_rules._types(me) if me is not None else []
        if net_move and sm.is_setup(action_obj.id, my_types):
            stats.bump("net_setup")
        played_kind: Optional[str] = None
        order = None
        if net_move or not cfg.keep_switches:
            stats.bump("eligible")
            rule = setup_rules.setup_rule(battle, cfg.max_stage, cfg.recover) if cfg.p > 0 else None
            if rule is not None:
                stats.bump("rule_ready")
                if self._rng.random() < cfg.p:
                    stats.bump("drawn")
                    kind, move = rule
                    if net_move and move.id == action_obj.id:
                        stats.bump("agreed")                    # the network chose it itself: its own order stands
                    else:
                        played_kind = kind
                        order = self.create_order(move)
                        stats.bump("override_" + kind)
                        stats.bot_decisions += 1
                        if kind == "setup" or sm.is_setup(move.id, my_types):
                            stats.bump("setup_played")
        if order is None:
            stats.net_decisions += 1
            order = self._order_of(action_obj, kwargs)
            if net_move and sm.is_setup(action_obj.id, my_types):
                stats.bump("setup_played")
        return order, played_kind


class PSBotSeatPlayer(OpponentSeat):
    """
    PSBot in the seat. When the engine has no legal answer the network moves (STEP), the game is
    a mixed one, and the battle's steps are dropped at its end (CLEANUP).
    """

    seat_kind = "psbot"

    def __init__(self, *args: Any, psbot_timeout_s: float = 20.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # After SyncBridgePlayer.__init__, which turns PSBot off outside the imitation modes. No
        # battle message has been handled yet: the player only logs in once the loop runs on.
        self.psbot_enabled = True
        self.psbot_timeout = float(psbot_timeout_s)

    async def choose_move(self, battle: Any):
        try:
            if getattr(battle, "finished", False):
                return DefaultBattleOrder()
            tag, events = self._decision_start(battle)
            self._seat_stats.decisions += 1
            order = await self.psbot_order(battle)              # already noted to the engine as played
            if tag not in self._battle_starts:
                return DefaultBattleOrder()                     # the battle went while PSBot thought
            if order is not None:
                self._seat_stats.bot_decisions += 1
                return order
            self._seat_stats.bump("no_answer")
            order = await self._network_order(battle, tag, events)
            self.psbot_note(battle, order)
            return order
        except Exception as e:
            return self._crashed(battle, e)


class FoulPlaySeatPlayer(FoulPlayMixin, OpponentSeat):
    """
    Foul Play in the seat, at a fixed number of iterations per world. Ported from branch stage3's
    fp_opponent.FoulPlayOpponentPlayer: when Foul Play gives no answer (a helper down, behind or
    hung, a battle its code cannot follow, an answer that is not legal) the network moves instead;
    after a failure of the helper or past battle_budget_s the network plays the rest of that
    battle and Foul Play's copy is dropped. Counts in SeatStats.extra: handed_<why>, single_<why>,
    iterations_not_fixed (answers whose helper did not search the fixed iterations).
    """

    seat_kind = "foulplay"
    foulplay_enabled = True          # in the class body: the player starts listening inside Player.__init__

    def __init__(self, *args: Any, fp_bridge: Any, fp_settings: FoulPlaySeatSettings, **kwargs: Any):
        self.foulplay = fp_bridge                                # this worker's own pool of helpers
        self.foulplay_search_ms = int(fp_settings.sizing_ms)
        self.foulplay_timeout = float(fp_settings.timeout_s)
        self.foulplay_pick = "choice"                            # the move Foul Play draws, as it plays on the ladder
        self._fp_settings = fp_settings
        self._fp_given_up: Dict[str, str] = {}                   # tag -> why the network plays the rest of this battle
        self._fp_last_error: Dict[str, BaseException] = {}       # battle tag -> why the last question had no answer
        super().__init__(*args, **kwargs)

    async def foulplay_advice(self, battle: Any, search_ms: Optional[int] = None, **more: Any):
        if int(self._fp_settings.iterations) > 0:
            more.setdefault("iterations", int(self._fp_settings.iterations))
        try:
            return await super().foulplay_advice(battle, search_ms, **more)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fp_last_error[battle.battle_tag] = exc         # foulplay_decision() counts it; the reason is wanted here
            raise

    async def choose_move(self, battle: Any):
        try:
            if getattr(battle, "finished", False):
                return DefaultBattleOrder()
            tag, events = self._decision_start(battle)
            self._seat_stats.decisions += 1
            started = self._battle_starts.get(tag, time.time())
            why = self._fp_given_up.get(tag)
            if why is None and time.time() - started > float(self._fp_settings.battle_budget_s):
                why = self._fp_hand_over(tag, battle, "age")
            if why is None:
                order, why = await self._fp_ask(battle)
                if tag not in self._battle_starts:
                    return DefaultBattleOrder()                  # the battle went while Foul Play searched
                if order is not None:
                    self._seat_stats.bot_decisions += 1
                    return order
                if why in HANDED_OVER:
                    self._fp_hand_over(tag, battle, why)
                else:
                    self._seat_stats.bump("single_" + str(why))
            order = await self._network_order(battle, tag, events)
            if tag not in self._fp_given_up:
                self.foulplay_note(battle, order)                # Foul Play still follows this battle
            return order
        except Exception as e:
            return self._crashed(battle, e)

    async def _fp_ask(self, battle: Any) -> Tuple[Any, Optional[str]]:
        """(order, None) with Foul Play's move, already told to it as played; or (None, why)."""
        self._fp_last_error.pop(battle.battle_tag, None)
        order, advice = await self.foulplay_decision(battle)
        if advice is not None and int(self._fp_settings.iterations) > 0 \
                and int(getattr(advice, "iterations_asked", 0) or 0) != int(self._fp_settings.iterations):
            self._seat_stats.bump("iterations_not_fixed")
        if order is not None:
            return order, None
        exc = self._fp_last_error.pop(battle.battle_tag, None)
        if exc is None:
            return None, ("illegal" if advice is not None else "error")
        if getattr(exc, "revive", False):
            return None, "revive"
        if getattr(exc, "expired", False):
            return None, "expired"
        if "did not answer" in str(exc):
            return None, "timeout"
        return None, "error"

    def _fp_hand_over(self, tag: str, battle: Any, why: str) -> str:
        """From here to its end the network plays this battle on Foul Play's side too. Foul Play's copy is dropped."""
        self._fp_given_up[tag] = why
        self._seat_stats.bump("handed_" + why)
        try:
            self.foulplay.forget(self._foulplay_key(battle.battle_tag))
        except Exception:
            pass
        return why

    def _is_mixed(self, tag: str) -> bool:
        return super()._is_mixed(tag) or tag in self._fp_given_up

    def _drop_books(self, tag: str) -> None:
        super()._drop_books(tag)
        self._fp_given_up.pop(tag, None)
        base_tag = tag.split("_", 1)[1] if "_" in tag else tag
        self._fp_last_error.pop(base_tag, None)

    def _prune_books(self) -> None:
        super()._prune_books()
        for tag in [t for t in self._fp_given_up if t not in self._battle_starts]:
            self._fp_given_up.pop(tag, None)

    def _cleanup_local_battle(self, tag: str) -> None:
        try:
            self.foulplay.forget(self._foulplay_key(tag.split("_", 1)[1] if "_" in tag else tag))
        except Exception:
            pass
        super()._cleanup_local_battle(tag)

    def _room_closed(self, room_id: str) -> None:
        try:
            self.foulplay.forget(self._foulplay_key(room_id))
        except Exception:
            pass
        super()._room_closed(room_id)


class FrozenSeatPlayer(OpponentSeat):
    """
    A frozen network in the seat (frozen_pool.py: fixed checkpoints and the run's own past saves, held by
    the inference actors and never trained). At its first decision of a battle the seat draws one of the
    pool's slots by the slots' weights (its own random stream, seeded by the run's seed, the port and the
    pair) and plays that net for the whole battle, on the observation that net was trained on: the
    player's own assembler for a net of the run's observation, an assembler of its own for another (SFIN's).
    The answers come from the worker's FrozenClient; nothing of these decisions goes on the worker's queue.
    When the pool gives no answer (the call failed or timed out) the network plays the rest of that
    battle, as for Foul Play's hand-over, and the game is a mixed one (its steps are dropped at its end).

    The draw is kept for STICKY_BATTLES battle starts in a row, then made again: a pair's 16 rooms then
    mostly play one net at a time, so a worker's call to the pool carries one or two nets instead of all of
    them (one forward each), while over the run every net still gets its weight's share of the battles.

    Counts in SeatStats.extra: battles_<slot>, games_<slot>, wins_<slot> (the trained network's),
    ties_<slot>, no_answer.
    """

    seat_kind = "frozen"
    STICKY_BATTLES = 16

    def __init__(self, *args: Any, frozen_client: Any, frozen_settings: Any, rng_seed: str = "", **kwargs: Any):
        self._frozen = frozen_client
        self._fslots = list(frozen_settings.slots)
        self._fweights = list(frozen_settings.slot_weights())
        self._fobs = list(frozen_settings.slot_obs())
        self._frng = random.Random(rng_seed) if rng_seed else random.Random()
        self._frozen_pick: Dict[str, int] = {}               # tag -> slot this battle plays
        self._sticky_slot: Optional[int] = None              # the slot new battles get, for _sticky_left more starts
        self._sticky_left = 0
        self._frozen_given_up: set = set()                    # tags the network plays to the end
        self._frozen_asm: Dict[Any, Any] = {}                 # obs key -> assembler, for nets of another observation
        super().__init__(*args, **kwargs)
        import dataclasses
        import frozen_pool
        for key in set(self._fobs):
            if key != frozen_pool.RUN_OBS:
                # Made here, beside the player's own (the working directory has vocab.json now).
                cfg = dataclasses.replace(self.cfg, obs=frozen_pool.obs_config_for(key, self.cfg.obs))
                self._frozen_asm[key] = cfg.make_assembler()

    def _frozen_assembler(self, slot: int) -> Any:
        return self._frozen_asm.get(self._fobs[slot], self.assembler)

    def _draw_slot(self) -> int:
        """The slot of a new battle: the current sticky draw, or a new one by the weights."""
        if self._sticky_slot is None or self._sticky_left <= 0:
            self._sticky_slot = self._draw_weighted()
            self._sticky_left = int(self.STICKY_BATTLES)
        self._sticky_left -= 1
        return self._sticky_slot

    def _draw_weighted(self) -> int:
        x = self._frng.random()
        acc = 0.0
        for i, w in enumerate(self._fweights):
            acc += w
            if x < acc:
                return i
        return max(i for i, w in enumerate(self._fweights) if w > 0)      # rounding left x past the sum

    async def choose_move(self, battle: Any):
        try:
            if getattr(battle, "finished", False):
                return DefaultBattleOrder()
            tag, events = self._decision_start(battle)
            stats = self._seat_stats
            stats.decisions += 1
            slot = self._frozen_pick.get(tag)
            if slot is None:
                slot = self._draw_slot()
                self._frozen_pick[tag] = slot
                stats.bump("battles_" + self._fslots[slot])
            if tag in self._frozen_given_up:
                return await self._network_order(battle, tag, events)
            asm = self._frozen_assembler(slot)
            obs = asm.assemble(battle, events=events)
            try:
                idx = await self._frozen.ask(tag, obs, slot)
            except asyncio.CancelledError:
                raise
            except Exception:
                stats.bump("no_answer")
                self._frozen_given_up.add(tag)
                return await self._network_order(battle, tag, events)
            if tag not in self._battle_starts:
                return DefaultBattleOrder()                     # the battle went while the pool answered
            action_obj, kwargs = asm.map_index_to_order(int(idx), battle)
            stats.bot_decisions += 1
            return self._order_of(action_obj, kwargs)
        except Exception as e:
            return self._crashed(battle, e)

    def _is_mixed(self, tag: str) -> bool:
        return super()._is_mixed(tag) or tag in self._frozen_given_up

    def _count_result(self, battle: Any, tag: str) -> None:
        super()._count_result(battle, tag)
        slot = self._frozen_pick.get(tag)
        if slot is None or self._is_mixed(tag):
            return
        name, st = self._fslots[slot], self._seat_stats
        st.bump("games_" + name)
        if getattr(battle, "lost", False):                      # the seat lost: the trained network won
            st.bump("wins_" + name)
        elif not getattr(battle, "won", False):
            st.bump("ties_" + name)

    def _drop_books(self, tag: str) -> None:
        super()._drop_books(tag)
        self._frozen_given_up.discard(tag)
        if self._frozen_pick.pop(tag, None) is not None:
            try:
                self._frozen.forget(tag)                        # its history in the pool
            except Exception:
                pass

    def _prune_books(self) -> None:
        super()._prune_books()
        for tag in [t for t in self._frozen_pick if t not in self._battle_starts]:
            self._drop_books(tag)


# ---------------------------------------------------------------------------------------------
# the wrapper and the worker
# ---------------------------------------------------------------------------------------------

def _make_bridge(settings: FoulPlaySeatSettings, helpers: int) -> Any:
    """This worker's pool of Foul Play helpers (a hung helper is replaced after three waits of timeout_s)."""
    timeout = float(settings.timeout_s)
    return FoulPlayBridge(helpers=max(1, int(helpers)), search_ms=int(settings.sizing_ms), timeout=timeout,
                          hang_timeout=max(20.0, 3.0 * timeout))


def _psbot_usable() -> str:
    """'' when PSBot's engine runs in this process (it is started here, once), else why not."""
    try:
        import psbot_bridge
        psbot_bridge.shared_engine()
        return ""
    except Exception as exc:
        return "PSBot's engine could not be started: %r" % (exc,)


def _foulplay_usable() -> str:
    try:
        from foulplay_bridge import available
        usable, why = available()
        return "" if usable else str(why)
    except Exception as exc:
        return "could not check for Foul Play: %r" % (exc,)


class SeatsVectorWrapper(PokeEnvVectorWrapper):
    """
    worker.PokeEnvVectorWrapper whose LAST len(seats) pairs have an opponent in one seat, one
    {"kind", "seat"} per pair (plan_seats). The other pairs are self-play pairs made by the base
    class's own method, untouched. A kind that cannot be used here (PSBot's engine or Foul Play
    does not start, or the mode learns from a teacher) is made as self-play pairs instead.
    """

    def __init__(self, cfg: RunConfig, pairs_count: int, server_port: int, seats: Sequence[Dict[str, Any]],
                 settings: Optional[OpponentSeatsConfig] = None, helpers: int = 1, frozen_actor: Any = None):
        # Everything the background thread reads has to be there before the base class starts it.
        self.seat_settings = settings or OpponentSeatsConfig()
        self.seat_port = int(server_port)
        specs = [dict(s) for s in seats if s.get("kind") in ALL_KINDS and s.get("seat") in ("a", "b")][:max(0, int(pairs_count))]
        # The kinds of this worker's numbers: "frozen" only when it has frozen pairs (so a worker without
        # them reports what it always did).
        self.seat_kinds: Tuple[str, ...] = ALL_KINDS if any(s["kind"] == "frozen" for s in specs) else SEAT_KINDS
        self.seat_planned: Dict[str, int] = {k: sum(1 for s in specs if s["kind"] == k) for k in self.seat_kinds}
        self.seat_stats: Dict[str, SeatStats] = {k: SeatStats(k) for k in self.seat_kinds}
        self.seat_unavailable: Dict[str, str] = {}
        self.seat_failed_start = ""
        self.fp_bridge: Any = None
        self.frozen_client: Any = None
        self.seat_players: List[OpponentSeat] = []
        self.seat_network_players: List[SyncBridgePlayer] = []
        kinds = {s["kind"] for s in specs}
        if kinds and getattr(cfg.learner, "mode", "ppo") in IMITATION_MODES:
            for kind in kinds:
                self.seat_unavailable[kind] = "mode %s learns from a teacher: no opponent seats" % cfg.learner.mode
        else:
            if "psbot" in kinds:
                why = _psbot_usable()
                if why:
                    self.seat_unavailable["psbot"] = why
            if "foulplay" in kinds:
                why = _foulplay_usable()
                if not why:
                    bridge = None
                    try:
                        bridge = _make_bridge(self.seat_settings.foulplay, helpers)
                        bridge.start(wait=False)                 # the helpers load while the players log in
                        self.fp_bridge = bridge
                    except Exception as exc:
                        why = "the Foul Play helpers could not be started: %r" % (exc,)
                        if bridge is not None:
                            try:
                                bridge.close()
                            except Exception:
                                pass
                if why:
                    self.seat_unavailable["foulplay"] = why
            if "frozen" in kinds:
                if frozen_actor is None or self.seat_settings.frozen is None:
                    self.seat_unavailable["frozen"] = "no frozen pool (the inference actor or the settings are missing)"
                else:
                    import frozen_pool
                    self.frozen_client = frozen_pool.FrozenClient(frozen_actor)
        for kind, why in self.seat_unavailable.items():
            logger.error(f"Opponent seat {kind} NOT used on port {server_port}: {why}. "
                         f"The {self.seat_planned[kind]} pairs meant for it play self-play.")
        self.seat_specs: List[Dict[str, Any]] = [s for s in specs if s["kind"] not in self.seat_unavailable]
        super().__init__(cfg, pairs_count, server_port)

    def _seat_player(self, index: int, side: str, spec: Dict[str, Any]) -> SyncBridgePlayer:
        common = dict(
            cfg=self.cfg,
            event_queue=self.event_queue, action_futures=self.action_futures, ep_sem=self.ep_sem,
            account_configuration=AccountConfiguration(mk_name(self.run_tag, index, side), None),
            server_configuration=self.server_conf, max_concurrent_battles=self.cfg.rollout.rooms_per_pair,
            battle_format=self.cfg.env.battle_format, start_listening=True, log_level=40,
        )
        if side != spec["seat"]:
            player = SyncBridgePlayer(**common)                  # the trained seat: exactly a self-play seat
            self.seat_network_players.append(player)
            return player
        kind = spec["kind"]
        seat = dict(seat_stats=self.seat_stats[kind], seat=spec["seat"])
        if kind == "setupnet":
            player = SetupNetSeatPlayer(setupnet=self.seat_settings.setupnet,
                                        rng_seed="%d|%d|%d" % (self.seat_settings.seed, self.seat_port, index), **seat, **common)
        elif kind == "psbot":
            player = PSBotSeatPlayer(psbot_timeout_s=self.seat_settings.psbot_timeout_s, **seat, **common)
        elif kind == "frozen":
            player = FrozenSeatPlayer(frozen_client=self.frozen_client, frozen_settings=self.seat_settings.frozen,
                                      rng_seed="%d|%d|%d|frozen" % (self.seat_settings.seed, self.seat_port, index), **seat, **common)
        else:
            player = FoulPlaySeatPlayer(fp_bridge=self.fp_bridge, fp_settings=self.seat_settings.foulplay, **seat, **common)
        self.seat_players.append(player)
        return player

    async def _init_players_and_spawn(self):
        specs = list(self.seat_specs)
        if not specs:
            await super()._init_players_and_spawn()
            return
        total = self.pairs_count
        self.pairs_count = total - len(specs)
        try:
            await super()._init_players_and_spawn()              # the self-play pairs, and the maintenance loop for all
        finally:
            self.pairs_count = total
        spawned = 0
        try:
            pairs = []
            for k, spec in enumerate(specs):
                index = total - len(specs) + k
                pairs.append((self._seat_player(index, "a", spec), self._seat_player(index, "b", spec)))
            for player in [p for pair in pairs for p in pair]:
                last_err: Optional[BaseException] = None
                for _ in range(5):
                    try:
                        await wait_for_login(player)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        await asyncio.sleep(1.0)
                if last_err is not None:
                    raise last_err
                await asyncio.sleep(0.05)
            await asyncio.gather(*[join_lobby(p) for pair in pairs for p in pair])
            rooms = self.cfg.rollout.rooms_per_pair
            for pA, pB in pairs:
                pA.partner, pB.partner = pB, pA                  # the SetHead's labels come from the other seat's team
                self.active_pairs.append((pA, pB))# the maintenance loop of the base class takes them from here
                await safe_send(pA, f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {rooms}", room="lobby")
                spawned += 1
        except Exception as e:
            # The self-play pairs of this worker are up and stay up. The opponent pairs asked for
            # before the error play on; the others do not play at all.
            self.seat_failed_start = "%d of %d opponent pairs could not be started: %r" % (len(specs) - spawned, len(specs), e)
            logger.critical(f"Opponent seats on port {self.seat_port}: {self.seat_failed_start}. "
                            f"Those pairs are not playing; the other pairs of this worker are.")

    def seats_heartbeat(self) -> Dict[str, Any]:
        """The numbers of report_line(). Cheap: nothing is asked of a helper or an engine (the bridge's lock is waited for half a second at most)."""
        beat: Dict[str, Any] = {
            "port": self.seat_port,
            "pairs": {k: sum(1 for s in self.seat_specs if s["kind"] == k) for k in self.seat_kinds},
            "planned": dict(self.seat_planned),
            "unavailable": dict(self.seat_unavailable), "failed_start": self.seat_failed_start,
            "stats": {k: self.seat_stats[k].as_dict() for k in self.seat_kinds},
            "rooms_closed": sum(getattr(p, "rooms_closed", 0) for p in self.seat_players + self.seat_network_players),
        }
        bridge = self.fp_bridge
        if bridge is not None:
            st = None
            lock = getattr(bridge, "_lock", None)
            if lock is None:
                st = bridge.stats()
            elif lock.acquire(timeout=0.5):
                try:
                    st = bridge.stats()
                finally:
                    lock.release()
            if st is not None:
                beat["foulplay_bridge"] = {
                    "helpers": len(st["helpers"]), "helpers_alive": sum(1 for h in st["helpers"] if h["alive"]),
                    "helper_exits": st["helper_exits"], "hangs": st["hangs"], "decisions": st["decisions"],
                    "recent_iterations_per_decision": st["recent_iterations_per_decision"],
                    "expired": st["expired"], "timeouts": st["timeouts"],
                }
        if self.frozen_client is not None:
            beat["frozen_client"] = dict(self.frozen_client.counts)
        return beat

    def close_seats(self) -> None:
        """Stops this worker's Foul Play helpers (PSBot's engine is the process's and closes with it)."""
        if self.fp_bridge is not None:
            try:
                self.fp_bridge.close()
            except Exception:
                pass


class OpponentSeatsRolloutWorker(RolloutWorker):
    """
    worker.RolloutWorker for a worker with opponent seats. The rollout loop, the reward, the
    episode bookkeeping and what is sent to the learner are the base class's, every line.

    seats: this worker's entry of plan_seats(): {"seats": [{"kind", "seat"}, ...], "helpers": n,
    "settings": OpponentSeatsConfig}.
    """

    def __init__(self, cfg: RunConfig, inference_actor, learner_actor, pairs: int, server_port: int,
                 seats: Optional[Dict[str, Any]] = None, frozen_actor: Any = None, **worker_kw: Any):
        """
        frozen_actor: the frozen_pool.FrozenPoolActor this worker's frozen seats ask (train.py passes it only to a
        worker with frozen seats). worker_kw: RolloutWorker's own keywords (learn_queue, lag_stats), passed on.
        """
        plan = dict(seats or {})
        specs = list(plan.get("seats") or [])
        settings = plan.get("settings") or OpponentSeatsConfig()
        helpers = int(plan.get("helpers") or 1)
        if not any(s.get("kind") == "frozen" for s in specs):
            frozen_actor = None

        def make_wrapper(cfg_, pairs_, server_port_):
            if frozen_actor is None:
                return SeatsVectorWrapper(cfg_, pairs_, server_port_, specs, settings, helpers)
            return SeatsVectorWrapper(cfg_, pairs_, server_port_, specs, settings, helpers, frozen_actor=frozen_actor)

        # The base class makes its wrapper by name. For the length of its own __init__ that name
        # gives the seats' wrapper, so whatever else that __init__ does, now or later, is done.
        original = worker.PokeEnvVectorWrapper
        worker.PokeEnvVectorWrapper = make_wrapper
        try:
            super().__init__(cfg, inference_actor, learner_actor, pairs, server_port, **worker_kw)
        finally:
            worker.PokeEnvVectorWrapper = original

    def heartbeat(self):
        beat = super().heartbeat()
        try:
            beat["opponent_seats"] = self.vec_env.seats_heartbeat()
        except Exception as e:                                   # a heartbeat always answers
            beat["opponent_seats"] = {"port": getattr(self.vec_env, "seat_port", -1), "report_error": repr(e)}
        return beat


def worker_class(entry: Optional[Dict[str, Any]]) -> type:
    """The rollout worker class for this worker's plan entry: worker.RolloutWorker when it has no seat."""
    return OpponentSeatsRolloutWorker if entry and entry.get("seats") else RolloutWorker


def has_frozen(entry: Optional[Dict[str, Any]]) -> bool:
    """Whether this worker's plan entry has frozen seats (its worker then gets a frozen actor)."""
    return bool(entry) and any(s.get("kind") == "frozen" for s in (entry.get("seats") or []))


def worker_kwargs(entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The extra keyword arguments for worker_class(entry): none for a plain worker."""
    return {"seats": entry} if entry and entry.get("seats") else {}


# ---------------------------------------------------------------------------------------------
# the line in the trainer's log
# ---------------------------------------------------------------------------------------------

def report_line(beats: Sequence[Any]) -> str:
    """
    "[Seats] setupnet: pairs 6, games 1200, network won 63.1% (p1 64.0% of 600, p2 62.2% of 600),
    rules 410 of 9000 decisions (setup 300, sleep 60, recover 40, sleeptalk 10) | psbot: ..."

    The counts are summed over the workers' last heartbeats; a worker that Ray started again
    counts from 0 again.
    """
    beats = [b for b in beats if isinstance(b, dict) and not b.get("report_error")]
    parts = []
    for kind in ALL_KINDS:
        pairs = sum(int((b.get("pairs") or {}).get(kind, 0)) for b in beats)
        planned = sum(int((b.get("planned") or {}).get(kind, 0)) for b in beats)
        if not planned:
            continue
        t: Dict[str, float] = {}
        for b in beats:
            for key, value in ((b.get("stats") or {}).get(kind) or {}).items():
                try:
                    t[key] = t.get(key, 0.0) + float(value or 0)
                except (TypeError, ValueError):
                    pass
        games = t.get("games_p1", 0.0) + t.get("games_p2", 0.0)
        text = "%s: pairs %d" % (kind, pairs)
        if pairs < planned:
            text += " (of %d planned: %s)" % (planned, "; ".join(sorted({str((b.get("unavailable") or {}).get(kind, "not started"))[:120]
                                                                        for b in beats if (b.get("unavailable") or {}).get(kind)})) or "not started")
        if games:
            score = t.get("wins_p1", 0.0) + t.get("wins_p2", 0.0) + 0.5 * (t.get("ties_p1", 0.0) + t.get("ties_p2", 0.0))
            sides = []
            for seat in ("p1", "p2"):
                g = t.get("games_" + seat, 0.0)
                if g:
                    sides.append("%s %.1f%% of %d" % (seat, 100.0 * (t.get("wins_" + seat, 0.0) + 0.5 * t.get("ties_" + seat, 0.0)) / g, g))
            text += ", games %d, network won %.1f%% (%s)" % (games, 100.0 * score / games, ", ".join(sides))
        else:
            text += ", games 0"
        decisions = t.get("decisions", 0.0)
        if kind == "setupnet":
            rules = ", ".join("%s %d" % (k, t.get("override_" + k, 0.0)) for k in setup_rules.KINDS if t.get("override_" + k))
            text += ", rule moves %d of %d decisions%s, set-up moves played %d" % (
                t.get("bot_decisions", 0.0), decisions, " (%s)" % rules if rules else "", t.get("setup_played", 0.0))
        else:
            text += ", its own moves %d of %d decisions, the network's %d, mixed games %d" % (
                t.get("bot_decisions", 0.0), decisions, t.get("net_decisions", 0.0), t.get("mixed_games", 0.0))
        if kind == "foulplay":
            handed = ", ".join("%s %d" % (w, t.get("handed_" + w, 0.0)) for w in HANDED_OVER if t.get("handed_" + w))
            if handed:
                text += ", battles taken over (%s)" % handed
            if t.get("iterations_not_fixed"):
                text += ", %d ANSWERS NOT AT THE FIXED ITERATIONS" % t["iterations_not_fixed"]
        if kind == "frozen":
            names = sorted({k[len("battles_"):] for k in t if k.startswith("battles_")})
            by = []
            for name in names:
                g = t.get("games_" + name, 0.0)
                won = t.get("wins_" + name, 0.0) + 0.5 * t.get("ties_" + name, 0.0)
                by.append("%s %d battles%s" % (name, t.get("battles_" + name, 0.0),
                                               ", network won %.1f%% of %d" % (100.0 * won / g, g) if g else ""))
            if by:
                text += ", by net: " + "; ".join(by)
            if t.get("no_answer"):
                text += ", %d battles the pool did not answer (the network took over)" % t["no_answer"]
        if t.get("crashes"):
            text += ", %d crashed decisions" % t["crashes"]
        parts.append(text)
    failed = [b for b in beats if b.get("failed_start")]
    if failed:
        parts.append("PAIRS NOT STARTED on %d workers: %s" % (len(failed), str(failed[0]["failed_start"])[:160]))
    return "[Seats] " + (" | ".join(parts) if parts else "none")
