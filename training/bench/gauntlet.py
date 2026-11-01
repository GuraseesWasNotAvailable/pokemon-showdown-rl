"""
The measurement: is the network with a search beside it stronger than the network alone?

    python bench/gauntlet.py opp  --port P --kind net|foulplay --pairs ACC:CHA:OURS[,ACC:CHA:OURS] --out DIR/opp.jsonl --stop-file DIR/opp.stop
    python bench/gauntlet.py ours --port P --user OURS --arms mix,predict --per-arm 200 --parallel 6 --helpers 6
                                  --opp-accept ACC --opp-challenge CHA --out DIR/results.jsonl --stop-file DIR/ours.stop
    python bench/gauntlet.py report DIR [DIR ...] [--json FILE]

Nothing in the players is changed. "ours" builds the player with ladder.make_player, so it is the
player ladder.py puts on a ladder (players.audited, search_player.search_player, the battle
recorder, the timer request), and only the way games are found differs: on a local server the
ladder pairs two accounts at once only while their ratings are close and lets the server choose
who is p1, which came out uneven in the earlier runs (bench/notes/search_player.md). Here every
game is a challenge, and the side is chosen:

  * the opponent process holds two accounts per player of ours. ACC accepts every challenge (we
    challenge it, so we are p1). CHA keeps one challenge to us standing (we accept it, so we are p2);
  * our process starts a game only when it has a free seat (--parallel), so no battle waits on the
    server for a seat, and it takes the side it has played less;
  * the arm of a game is drawn when its first message arrives, balanced within the side
    (RolePicker), so every arm plays as p1 and as p2 equally often.

The opponent is the network alone (--kind net: players.batched_policy_player, the same checkpoint)
or Foul Play (--kind foulplay: foulplay_player.FoulPlayBridgePlayer on helpers of its own, with
the iterations of every one of its decisions written down).

Local servers only. Apart from its orders it sends the challenge or its acceptance, the timer
request and the room leave. No chat.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FORMAT = "gen9randombattle"
FULL_STRENGTH = 60_000      # iterations per decision Foul Play needs (bench/FOULPLAY_BRIDGE.md, F2)
ROLES = ("p1", "p2")


def to_id(name: Optional[str]) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


def now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def read_lines(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def server_configuration(port: int) -> Any:
    from poke_env import ServerConfiguration

    return ServerConfiguration(f"ws://127.0.0.1:{port}/showdown/websocket", "https://play.pokemonshowdown.com/action.php?")


# ---------------------------------------------------------------------------------------------
# Which game is played in which mode
# ---------------------------------------------------------------------------------------------

class RolePicker:
    """
    Gives every new game one of the arms, balanced within the side it is played on: always one of
    the arms with the fewest games on that side, then with the fewest games in all, drawn at
    random among those. An arm that has its --per-arm games gets no more.

    The driver says which side the next game is on (next_role) before it challenges or accepts;
    the player asks for the arm when the battle's first message arrives.
    """

    def __init__(self, arms: List[str], per_arm: int, rng: Optional[random.Random] = None):
        self.arms = list(arms)
        self.per_arm = int(per_arm)
        self.counts: Dict[Tuple[str, Optional[str]], int] = {(a, r): 0 for a in self.arms for r in ROLES + (None,)}
        self.rng = rng or random.Random()
        self.next_role: Optional[str] = None
        self.given: Dict[str, str] = {}
        self.roles: Dict[str, Optional[str]] = {}

    def games(self, arm: str) -> int:
        return sum(self.counts[(arm, r)] for r in ROLES + (None,))

    def remaining(self) -> int:
        return sum(max(0, self.per_arm - self.games(a)) for a in self.arms)

    def preload(self, lines: List[Dict[str, Any]]) -> None:
        """Games already in the results file (a run that is started again)."""
        for d in lines:
            arm, role = d.get("arm"), d.get("planned_role")
            if arm in self.arms:
                self.counts[(arm, role if role in ROLES else None)] += 1

    def __call__(self, tag: str) -> str:
        arm = self.given.get(tag)
        if arm is None:
            role = self.next_role if self.next_role in ROLES else None
            open_arms = [a for a in self.arms if self.games(a) < self.per_arm] or self.arms
            key = lambda a: (self.counts[(a, role)], self.games(a))     # noqa: E731
            fewest = min(key(a) for a in open_arms)
            arm = self.rng.choice([a for a in open_arms if key(a) == fewest])
            self.counts[(arm, role)] += 1
            self.given[tag] = arm
            self.roles[tag] = role
            self.next_role = None
        return arm


# ---------------------------------------------------------------------------------------------
# Our side
# ---------------------------------------------------------------------------------------------

DECISION_KEYS = ("turn", "rqid", "mode", "legal", "forced", "net", "search", "played", "source", "asked", "reason", "ms",
                 "net_ms", "wait_over_ms", "p_net", "p_played", "probs", "iterations", "worlds", "search_ms", "shares",
                 "share_net", "share_search", "share_played", "disagree", "net_overruled", "search_overruled",
                 "tera_added", "tera_dropped", "mask")
PREDICT_KEYS = ("read", "confidence", "second", "p_switch", "p_tera", "top", "against", "candidate", "gain", "risk", "coverage",
                "why", "changed", "mix", "payoff_ms", "payoff_known", "confident", "position")


def ladder_args(args: argparse.Namespace) -> argparse.Namespace:
    """The arguments ladder.make_player reads, with ladder.py's own defaults for everything not asked here."""
    return argparse.Namespace(
        server=f"local:{args.port}", checkpoint=args.checkpoint, device=None, format=FORMAT,
        parallel=args.parallel, timer=args.timer, think_ms=0,
        player=args.arms[0], arms=list(args.arms), search_ms=args.search_ms, helpers=args.helpers, deadline=args.deadline,
        mix_rule=args.mix_rule, predict_confidence=0.60, predict_gain=0.08, predict_risk=0.08, predict_once_bitten=True,
        payoff_ms=400.0, save_battles=args.save_battles, battles_dir=os.path.join(os.path.dirname(args.out), "battles"),
        # ladder.py's defaults for what make_player and its helpers read since the live-safety review
        health_window=50, max_fallback_share=0.10, ping_interval=5.0, ping_timeout=10.0, close_timeout=2.0, reconnect_for=300.0,
    )


async def drive_ours(player: Any, args: argparse.Namespace, picker: RolePicker) -> int:
    import ladder

    client = player.ps_client
    me = to_id(args.user)
    out_dir = os.path.dirname(args.out)
    decisions_path = os.path.join(out_dir, "decisions.jsonl")
    status_path = os.path.join(out_dir, "status.jsonl")
    ckpt = os.path.basename(args.checkpoint)
    bridge = getattr(player, "_ladder_bridge", None)
    recorder = getattr(player, "_ladder_recorder", None)
    accept_id, challenge_id = to_id(args.opp_accept), to_id(args.opp_challenge)
    try:
        await asyncio.wait_for(client.logged_in.wait(), timeout=60)
    except asyncio.TimeoutError:
        print("could not log in", flush=True)
        return 2
    print(f"{now_text()} {args.user}: {ckpt} on {player.policy.device}, arms {','.join(args.arms)} x {args.per_arm}, "
          f"{args.parallel} games at a time" + (f", search {args.search_ms} ms on {args.helpers} helpers, deadline {args.deadline} s, "
                                                f"rule {args.mix_rule}" if bridge is not None else "")
          + f", p1 against {args.opp_accept}, p2 against {args.opp_challenge}", flush=True)

    known: set = set(player.battles)
    active: Dict[str, float] = {}
    planned: Dict[str, Optional[str]] = {}
    by_role = {"p1": 0, "p2": 0}
    earlier = [d for d in read_lines(args.out) if d.get("arm")]
    for d in earlier:
        if d.get("planned_role") in by_role:
            by_role[d["planned_role"]] += 1
    target = picker.per_arm * len(picker.arms)      # games in all, the ones of an earlier start of this run included
    started = len(earlier)
    pending: Optional[Tuple[str, float]] = None
    incoming = False                 # a challenge of the challenging opponent is waiting for us
    incoming_since = time.time()
    retries = 0
    stalled = 0
    finished = wins = 0      # of this start
    stopping = False
    began = time.time()
    last_start = 0.0
    last_status = 0.0

    def report(tag: str, b: Any, now: float) -> None:
        nonlocal finished, wins
        won = bool(b.won)
        finished += 1
        wins += won
        line: Dict[str, Any] = {
            "time": now_text(), "battle": tag, "won": won, "tie": b.won is None, "user": me, "turns": b.turn,
            "opponent": getattr(b, "opponent_username", None), "checkpoint": ckpt, "seconds": round(now - active[tag], 1),
            "server": f"local:{args.port}", "parallel": args.parallel, "phase": args.phase,
            "planned_role": planned.get(tag), "ending": (getattr(player, "audit_endings", None) or {}).get(tag),
        }
        try:
            summary = player.game_summary(tag) or {}
            line.update(ladder.arm_fields(summary, getattr(b, "player_role", None),
                                          args.search_ms if bridge is not None else None, args.mix_rule))
            for rec in player.decisions_of(tag):
                small = {k: rec[k] for k in DECISION_KEYS if k in rec}
                small["battle"], small["arm"] = tag, summary.get("arm")
                pred = rec.get("predict")
                if isinstance(pred, dict):
                    small["predict"] = {k: pred[k] for k in PREDICT_KEYS if k in pred}
                    if isinstance(pred.get("outcome"), dict):
                        small["predict"]["outcome"] = pred["outcome"]
                append(decisions_path, small)
            player.forget_decisions(tag)
        except Exception as exc:        # the result must be written whatever happens to the rest
            line["report_error"] = repr(exc)[:200]
        append(args.out, line)
        if recorder is not None:
            recorder.write(tag, "note", {"result": line})
            recorder.close(tag)
        print(f"{time.strftime('%H:%M:%S')} game {finished:4d}: {'WIN ' if won else 'LOSS'} {str(line.get('arm')):<14} "
              f"{str(line.get('role')):<3} {b.turn:3d} turns {line['seconds']:6.1f} s   search {line.get('searched')}/{line.get('decisions')}"
              f" fb {line.get('fallbacks')} it {line.get('iterations_per_decision')} late {line.get('late')} tera {line.get('tera_turn')}"
              f"   record {wins}-{finished - wins}", flush=True)

    while True:
        now = time.time()
        listening = getattr(client, "_listening_coroutine", None)
        if listening is not None and listening.done():
            print("the connection to the server was lost: stopping", flush=True)
            break

        queue = player._challenge_queue
        while not queue.empty():
            if to_id(queue.get_nowait()) == challenge_id and not incoming:
                incoming, incoming_since = True, now

        for tag in list(player.battles):
            if tag not in known:
                known.add(tag)
                active[tag] = now
                started += 1
                role = picker.roles.get(tag) or (pending[0] if pending else None)
                planned[tag] = role
                if role in by_role:
                    by_role[role] += 1
                if pending and pending[0] == "p2":
                    incoming = False
                pending = None

        for tag in list(active):
            b = player.battles[tag]
            if b.finished:
                report(tag, b, now)
                del active[tag]
            elif now - active[tag] > args.game_timeout:
                stalled += 1
                print(f"STALL: {tag} has no result after {args.game_timeout} s, giving it up", flush=True)
                append(args.out, {"time": now_text(), "battle": tag, "stalled": True, "user": me, "phase": args.phase,
                                  "arm": picker.given.get(tag), "planned_role": planned.get(tag)})
                del active[tag]

        if not stopping and (os.path.exists(args.stop_file) or (args.minutes and now - began > args.minutes * 60.0)):
            stopping = True
            print(f"{time.strftime('%H:%M:%S')} stopping: no new games, finishing the {len(active)} in progress", flush=True)

        if pending is not None and now - pending[1] > 10.0:
            pending = None          # the challenge or its acceptance went nowhere: ask again
            retries += 1
        if (not stopping and pending is None and len(active) < args.parallel and started < target
                and now - last_start >= args.gap_s):
            role = "p1" if by_role["p1"] <= by_role["p2"] else "p2"
            if role == "p2" and not incoming and now - incoming_since < 10.0:
                role = ""           # its challenge is on its way (it challenges again when a battle of its own starts)
            if role:
                picker.next_role = role
                if role == "p1":
                    await client.challenge(args.opp_accept, FORMAT, None)
                else:
                    await client.accept_challenge(args.opp_challenge, None)
                    incoming_since = now
                pending = (role, now)
                last_start = now

        if now - last_status > 30.0:
            last_status = now
            status: Dict[str, Any] = {"time": now_text(), "finished": finished, "wins": wins, "active": len(active),
                                      "by_role": dict(by_role), "retries": retries, "stalled": stalled,
                                      "audit": dict(getattr(player, "audit", {}) or {}),
                                      "loop_lag": dict(getattr(player, "loop_lag", {}) or {})}
            if bridge is not None:
                s = bridge.stats()
                status["bridge"] = {k: s.get(k) for k in ("decisions", "decision_errors", "timeouts", "expired", "helper_exits",
                                                            "helper_starts", "hangs", "recent_iterations_per_decision",
                                                            "iterations_per_decision", "battles")}
            append(status_path, status)

        if not active and pending is None and (stopping or started >= target):
            break
        await asyncio.sleep(0.05)

    audit = dict(getattr(player, "audit", {}) or {})
    endings = list((getattr(player, "audit_endings", None) or {}).values())
    final: Dict[str, Any] = {
        "time": now_text(), "user": me, "phase": args.phase, "arms": args.arms, "finished": finished, "wins": wins,
        "minutes": round((time.time() - began) / 60.0, 2), "by_role": by_role, "retries": retries, "stalled": stalled,
        "audit": audit, "audit_messages": list(getattr(player, "audit_messages", []) or []),
        "inactivity": endings.count("inactivity"), "forfeit": endings.count("forfeit"),
        "search_totals": getattr(player, "search_totals", None), "loop_lag": getattr(player, "loop_lag", None),
        "foulplay_counts": dict(getattr(player, "foulplay_counts", {}) or {}),
        "foulplay_problems": list(player.__dict__.get("foulplay_problems", []) or []),
        "recorder_failed": getattr(recorder, "failed", None), "warm_up_ms": getattr(player, "warm_up_ms", None),
        "parallel": args.parallel, "helpers": args.helpers if bridge is not None else 0, "search_ms": args.search_ms,
        "deadline": args.deadline, "mix_rule": args.mix_rule, "checkpoint": ckpt,
    }
    if bridge is not None:
        s = bridge.stats()
        final["bridge"] = {k: v for k, v in s.items() if k != "helpers"}
    try:
        final["search_summary"] = player.search_summary()
    except Exception:
        pass
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=1)
    print(f"\n{now_text()} finished: {wins}-{finished - wins} in {final['minutes']} minutes; invalid orders {audit.get('invalid_choices', 0)}, "
          f"exceptions {audit.get('exceptions', 0)}, random orders forced {audit.get('forced_random', 0)}, other errors "
          f"{audit.get('other_errors', 0)}, inactivity {final['inactivity']}, forfeits {final['forfeit']}, stalled {stalled}, "
          f"start retries {retries}", flush=True)
    if final.get("search_summary"):
        print(final["search_summary"], flush=True)
    if bridge is not None:
        print("helpers: " + json.dumps(final["bridge"]), flush=True)
    if recorder is not None:
        recorder.close_all()
    return 0


async def run_ours(args: argparse.Namespace) -> int:
    import ladder
    from poke_env.concurrency import handle_threaded_coroutines

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ladder.OUT = args.out                       # what make_player reads for the games already played
    ladder.GRAPH = os.path.join(os.path.dirname(args.out), "elo.html")
    largs = ladder_args(args)
    player = ladder.make_player(largs, args.user, None)
    picker = RolePicker(args.arms, args.per_arm, random.Random(args.seed) if args.seed is not None else None)
    picker.preload(read_lines(args.out))
    player.mode_picker = picker                 # the arm of a battle: ours, balanced within the side
    try:
        return await handle_threaded_coroutines(drive_ours(player, args, picker), player.ps_client.loop)
    finally:
        bridge = getattr(player, "_ladder_bridge", None)
        if bridge is not None:
            bridge.close()


# ---------------------------------------------------------------------------------------------
# The opponent
# ---------------------------------------------------------------------------------------------

def make_opponents(args: argparse.Namespace) -> Tuple[List[Tuple[Any, Any, str]], Any]:
    from poke_env import AccountConfiguration

    server = server_configuration(args.port)
    common = dict(server_configuration=server, battle_format=FORMAT, max_concurrent_battles=32, log_level=40)
    bridge = None
    if args.kind == "net":
        from hybrid import HybridConfig
        from players import audited, batched_policy_player

        cls = audited(batched_policy_player())

        def build(name: str) -> Any:
            return cls(hybrid=HybridConfig(checkpoint=args.checkpoint, beta=0.0, veto_gap=None, policy_weight=1.0,
                                           greedy=True, device=None, random_init=False),
                       account_configuration=AccountConfiguration(name, None), **common)
    else:
        import foulplay_bridge as fb
        from foulplay_player import FoulPlayBridgePlayer
        from players import audited

        ok, why = fb.available()
        if not ok:
            raise SystemExit("Foul Play cannot be used: " + why)
        print(f"Foul Play: {why}; starting {args.fp_helpers} helpers at {args.fp_ms} ms", flush=True)
        bridge = fb.FoulPlayBridge(helpers=args.fp_helpers, search_ms=args.fp_ms, timeout=60.0)
        bridge.start(timeout=180.0)

        class RecordingFoulPlay(FoulPlayBridgePlayer):
            """FoulPlayBridgePlayer, unchanged, with the iterations and the time of every decision written down."""

            def __init__(self, *a: Any, **k: Any):
                self.fp_log: Dict[str, List[Tuple[Optional[int], int, bool]]] = {}
                super().__init__(*a, **k)

            async def foulplay_decision(self, battle: Any, search_ms: Optional[int] = None, pick: Optional[str] = None):
                began = time.monotonic()
                order, advice = await super().foulplay_decision(battle, search_ms, pick)
                its = int(getattr(advice, "iterations", 0) or 0) if advice is not None else None
                self.fp_log.setdefault(battle.battle_tag, []).append((its, int((time.monotonic() - began) * 1000.0), order is not None))
                return order, advice

        cls = audited(RecordingFoulPlay)

        def build(name: str) -> Any:
            return cls(search_ms=args.fp_ms, bridge=bridge, pick="choice",
                       account_configuration=AccountConfiguration(name, None), **common)

    pairs = []
    for spec in args.pairs.split(","):
        acc, cha, ours = [s.strip() for s in spec.split(":")]
        pairs.append((build(acc), build(cha), ours))
    return pairs, bridge


async def challenge_loop(player: Any, target: str, stopped: Any) -> None:
    """One challenge to `target` standing at all times: again as soon as a battle of ours has started."""
    client = player.ps_client
    await client.logged_in.wait()
    seen = -1
    sent = 0.0
    while not stopped():
        n = len(player.battles)
        if n != seen or time.time() - sent > 20.0:       # accepted, or nothing for 20 s (the first one may have found nobody)
            seen = n
            sent = time.time()
            try:
                await client.challenge(target, FORMAT, None)
            except Exception:
                pass
        await asyncio.sleep(0.02)


async def drive_opp(pairs: List[Tuple[Any, Any, str]], bridge: Any, args: argparse.Namespace) -> int:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    status_path = os.path.join(os.path.dirname(args.out), "opp_status.jsonl")
    players = [p for acc, cha, _ in pairs for p in (acc, cha)]
    for p in players:
        await asyncio.wait_for(p.ps_client.logged_in.wait(), timeout=60)
    stop = {"now": False}
    tasks = []
    for acc, cha, ours in pairs:
        tasks.append(asyncio.ensure_future(acc._accept_challenges(None, 10 ** 7, None)))
        tasks.append(asyncio.ensure_future(challenge_loop(cha, ours, lambda: stop["now"])))
    print(f"{now_text()} opponent ({args.kind}) ready: " + ", ".join(f"{a.username} accepts, {c.username} challenges {o}" for a, c, o in pairs), flush=True)
    reported: set = set()
    finished = 0
    last_status = 0.0
    while True:
        now = time.time()
        for p in players:
            for tag, b in list(p.battles.items()):
                if b.finished and (p.username, tag) not in reported:
                    reported.add((p.username, tag))
                    finished += 1
                    line: Dict[str, Any] = {"time": now_text(), "battle": tag, "user": to_id(p.username), "kind": args.kind,
                                            "opponent": getattr(b, "opponent_username", None), "won": bool(b.won),
                                            "turns": b.turn, "role": getattr(b, "player_role", None)}
                    log = getattr(p, "fp_log", None)
                    if log is not None:
                        rows = log.pop(tag, [])
                        its = [r[0] for r in rows if r[0]]
                        line.update({"decisions": len(rows), "answered": len(its), "no_order": sum(1 for r in rows if not r[2]),
                                     "its": its, "ms": [r[1] for r in rows]})
                    append(args.out, line)
        if now - last_status > 30.0:
            last_status = now
            status: Dict[str, Any] = {"time": now_text(), "finished": finished,
                                      "audit": {to_id(p.username): dict(p.audit) for p in players}}
            if bridge is not None:
                s = bridge.stats()
                status["bridge"] = {k: s.get(k) for k in ("decisions", "decision_errors", "timeouts", "expired", "helper_exits",
                                                            "helper_starts", "hangs", "recent_iterations_per_decision",
                                                            "iterations_per_decision", "battles")}
                status["foulplay"] = {to_id(p.username): {"decisions": p.foulplay_decisions, "fallbacks": p.foulplay_fallbacks,
                                                           "after_end": p.foulplay_after_end, **p.foulplay_counts} for p in players}
            append(status_path, status)
        if os.path.exists(args.stop_file):
            break
        await asyncio.sleep(0.5)
    stop["now"] = True
    for t in tasks:
        t.cancel()
    final: Dict[str, Any] = {"time": now_text(), "kind": args.kind, "finished": finished,
                             "audit": {to_id(p.username): dict(p.audit) for p in players},
                             "endings": {to_id(p.username): list((p.audit_endings or {}).values()) for p in players}}
    if bridge is not None:
        s = bridge.stats()
        final["bridge"] = {k: v for k, v in s.items() if k != "helpers"}
        final["fp_ms"], final["fp_helpers"] = args.fp_ms, args.fp_helpers
        final["foulplay"] = {to_id(p.username): {"decisions": p.foulplay_decisions, "fallbacks": p.foulplay_fallbacks,
                                                  "after_end": p.foulplay_after_end, **p.foulplay_counts,
                                                  "problems": list(p.__dict__.get("foulplay_problems", []) or [])} for p in players}
    with open(os.path.join(os.path.dirname(args.out), "opp_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=1)
    print(f"{now_text()} opponent stopped after {finished} games: " + json.dumps({k: v for k, v in final.items() if k != "endings"})[:1500], flush=True)
    return 0


async def run_opp(args: argparse.Namespace) -> int:
    from poke_env.concurrency import POKE_LOOP, handle_threaded_coroutines

    pairs, bridge = make_opponents(args)
    try:
        return await handle_threaded_coroutines(drive_opp(pairs, bridge, args), POKE_LOOP)
    finally:
        if bridge is not None:
            bridge.close()


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------

def wilson(k: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(k: float, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100.0 * k / n:.1f}% ({100 * lo:.1f} to {100 * hi:.1f})"


def quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return s[i]


def difference(k1: int, n1: int, k2: int, n2: int) -> Dict[str, Any]:
    """Arm 1 minus arm 2: the difference of the two shares with its 95% interval (Newcombe, from the two Wilson intervals)."""
    if not n1 or not n2:
        return {}
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pbar = (p1 + p2) / 2.0
    # games per arm for a difference of this size to be 1.96 standard errors (5% two-sided, 50% power)
    # and for 80% power, when the two shares are what was seen
    need50 = math.ceil(2 * pbar * (1 - pbar) * (1.96 / d) ** 2) if abs(d) > 1e-9 else None
    need80 = math.ceil(2 * pbar * (1 - pbar) * ((1.96 + 0.84) / d) ** 2) if abs(d) > 1e-9 else None
    return {"difference": d, "low": lo, "high": hi, "z": d / se if se > 0 else 0.0, "games_per_arm_for_2se": need50,
            "games_per_arm_for_80_power": need80}


def arm_report(games: List[Dict[str, Any]], decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(games)
    wins = sum(1 for g in games if g.get("won"))
    out: Dict[str, Any] = {"games": n, "wins": wins, "losses": n - wins, "ties": sum(1 for g in games if g.get("tie")),
                           "win_share": wins / n if n else None, "wilson": wilson(wins, n)}
    for role in ROLES:
        rows = [g for g in games if g.get("role") == role]
        out[role] = {"games": len(rows), "wins": sum(1 for g in rows if g.get("won"))}
    tera = [g for g in games if g.get("tera")]
    turns = [g["tera_turn"] for g in tera if g.get("tera_turn") is not None]
    out["tera"] = {"games": len(tera), "share": len(tera) / n if n else None, "wilson": wilson(len(tera), n),
                   "turn_median": statistics.median(turns) if turns else None,
                   "turn_mean": round(statistics.fmean(turns), 1) if turns else None,
                   "turn_q1": quantile(turns, 0.25), "turn_q3": quantile(turns, 0.75),
                   "by": {k: sum(1 for g in tera if g.get("tera_by") == k) for k in sorted({str(g.get("tera_by")) for g in tera})}
                   if tera else {},
                   "net_wanted_it_too": sum(1 for g in tera if g.get("tera_net")),
                   "search_wanted_it_too": sum(1 for g in tera if g.get("tera_search")),
                   "won_with": sum(1 for g in tera if g.get("won")), "won_without": sum(1 for g in games if not g.get("tera") and g.get("won")),
                   "games_without": n - len(tera),
                   "turns_mean_game": round(statistics.fmean([g.get("turns") or 0 for g in games]), 1) if games else None}
    asked = sum(g.get("asked") or 0 for g in games)
    searched = sum(g.get("searched") or 0 for g in games)
    reasons: Dict[str, int] = {}
    for g in games:
        for why, k in (g.get("fallback_reasons") or {}).items():
            reasons[why] = reasons.get(why, 0) + k
    out["decisions"] = sum(g.get("decisions") or 0 for g in games)
    out["asked"], out["answered"] = asked, searched
    out["fallbacks"] = sum(g.get("fallbacks") or 0 for g in games)
    out["fallback_reasons"] = reasons
    out["games_with_a_fallback"] = sum(1 for g in games if g.get("fallbacks"))
    out["disagreed"] = sum(g.get("disagreed") or 0 for g in games)
    out["net_overruled"] = sum(g.get("overruled") or 0 for g in games)
    out["search_overruled"] = sum(g.get("search_overruled") or 0 for g in games)
    out["tera_added"] = sum(g.get("tera_added") or 0 for g in games)
    out["tera_dropped"] = sum(g.get("tera_dropped") or 0 for g in games)
    out["late"] = sum(g.get("late") or 0 for g in games)
    out["endings"] = {k: sum(1 for g in games if g.get("ending") == k) for k in ("inactivity", "forfeit")}
    out["seconds_per_game"] = round(statistics.fmean([g.get("seconds") or 0 for g in games]), 1) if games else None
    ms = [d["ms"] for d in decisions if d.get("ms") is not None]
    out["decision_ms"] = {"n": len(ms), "mean": round(statistics.fmean(ms), 1) if ms else None, "median": quantile(ms, 0.5),
                          "p95": quantile(ms, 0.95), "p99": quantile(ms, 0.99), "max": max(ms) if ms else None}
    asked_ms = [d["ms"] for d in decisions if d.get("asked") and d.get("ms") is not None]
    out["decision_ms_asked"] = {"n": len(asked_ms), "mean": round(statistics.fmean(asked_ms), 1) if asked_ms else None,
                                "median": quantile(asked_ms, 0.5), "p95": quantile(asked_ms, 0.95), "max": max(asked_ms) if asked_ms else None}
    its = [d["iterations"] for d in decisions if d.get("iterations")]
    out["iterations"] = {"n": len(its), "mean": int(statistics.fmean(its)) if its else None, "median": quantile(its, 0.5),
                         "p10": quantile(its, 0.10), "below_60000": sum(1 for x in its if x < FULL_STRENGTH)}
    # Terastallization, per decision: turns on which a Tera action was legal, what each side put on it, what was played
    offered = [d for d in decisions if d.get("mask") and any(d["mask"][10:14])]
    out["tera_decisions"] = {
        "offered": len(offered),
        "played": sum(1 for d in offered if (d.get("played") or 0) >= 10),
        "net_best": sum(1 for d in offered if (d.get("net") or 0) >= 10),
        "search_best": sum(1 for d in offered if d.get("search") is not None and d["search"] >= 10),
        "answered": sum(1 for d in offered if d.get("search") is not None),
        "net_p_mean": round(statistics.fmean([sum(d["probs"][10:14]) for d in offered if d.get("probs")]), 4) if offered else None,
        "search_share_mean": round(statistics.fmean([sum(d["shares"][10:14]) for d in offered if d.get("shares")]), 4)
        if any(d.get("shares") for d in offered) else None,
    }
    pred = [d["predict"] for d in decisions if isinstance(d.get("predict"), dict)]
    if pred:
        changed = [p for p in pred if p.get("changed")]
        why: Dict[str, int] = {}
        for p in pred:
            key = p.get("why") or "changed"
            why[key] = why.get(key, 0) + 1
        out["predict"] = {"made": sum(g.get("pred_made") or 0 for g in games), "scored": sum(g.get("pred_scored") or 0 for g in games),
                          "top1_right": sum(g.get("pred_top1_right") or 0 for g in games),
                          "reads": sum(g.get("pred_reads") or 0 for g in games),
                          "reads_scored": sum(g.get("pred_reads_scored") or 0 for g in games),
                          "reads_right": sum(g.get("pred_reads_right") or 0 for g in games),
                          "changed": sum(g.get("pred_changed") or 0 for g in games),
                          "changed_scored": sum(g.get("pred_changed_scored") or 0 for g in games),
                          "changed_right": sum(g.get("pred_changed_right") or 0 for g in games),
                          "games_with_a_change": sum(1 for g in games if g.get("pred_changed")),
                          "won_with_a_change": sum(1 for g in games if g.get("pred_changed") and g.get("won")),
                          "won_without_a_change": sum(1 for g in games if not g.get("pred_changed") and g.get("won")),
                          "kept": why, "changed_records": len(changed)}
    return out


def load_run(folder: str) -> Dict[str, Any]:
    games = [g for g in read_lines(os.path.join(folder, "results.jsonl"))]
    stalled = [g for g in games if g.get("stalled")]
    games = [g for g in games if not g.get("stalled") and g.get("arm")]
    decisions = read_lines(os.path.join(folder, "decisions.jsonl"))
    summary = None
    path = os.path.join(folder, "summary.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    return {"folder": folder, "games": games, "stalled": stalled, "decisions": decisions, "summary": summary}


def opp_report(folder: str) -> Optional[Dict[str, Any]]:
    rows = read_lines(os.path.join(folder, "opp.jsonl"))
    if not rows:
        return None
    out: Dict[str, Any] = {"games": len(rows), "kind": rows[0].get("kind")}
    by_user: Dict[str, Dict[str, Any]] = {}
    for user in sorted({r["user"] for r in rows}):
        mine = [r for r in rows if r["user"] == user]
        its = [x for r in mine for x in (r.get("its") or [])]
        ms = [x for r in mine for x in (r.get("ms") or [])]
        d: Dict[str, Any] = {"games": len(mine), "wins": sum(1 for r in mine if r.get("won"))}
        if its:
            d.update({"decisions": sum(r.get("decisions") or 0 for r in mine), "answered": len(its),
                      "no_order": sum(r.get("no_order") or 0 for r in mine),
                      "iterations_mean": int(statistics.fmean(its)), "iterations_median": quantile(its, 0.5),
                      "iterations_p10": quantile(its, 0.10), "below_60000": sum(1 for x in its if x < FULL_STRENGTH),
                      "ms_mean": round(statistics.fmean(ms), 1), "ms_p95": quantile(ms, 0.95), "ms_max": max(ms)})
        by_user[user] = d
    out["by_user"] = by_user
    path = os.path.join(folder, "opp_summary.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            out["summary"] = json.load(f)
    return out


def report(args: argparse.Namespace) -> int:
    result: Dict[str, Any] = {"runs": {}}
    for folder in args.folders:
        for sub in sorted(glob.glob(os.path.join(folder, "*", "results.jsonl"))) or [os.path.join(folder, "results.jsonl")]:
            run = load_run(os.path.dirname(sub))
            if not run["games"]:
                continue
            name = os.path.relpath(os.path.dirname(sub), os.path.dirname(os.path.abspath(folder))).replace("\\", "/")
            arms: Dict[str, Any] = {}
            for arm in sorted({g["arm"] for g in run["games"]}):
                arms[arm] = arm_report([g for g in run["games"] if g["arm"] == arm], [d for d in run["decisions"] if d.get("arm") == arm])
            s = run["summary"] or {}
            result["runs"][name] = {"arms": arms, "stalled": len(run["stalled"]), "audit": s.get("audit"),
                                    "inactivity": s.get("inactivity"), "forfeit": s.get("forfeit"), "retries": s.get("retries"),
                                    "bridge": s.get("bridge"), "loop_lag": s.get("loop_lag"), "minutes": s.get("minutes"),
                                    "exceptions": (s.get("search_totals") or {}).get("exceptions"),
                                    "predict_errors": (s.get("search_totals") or {}).get("predict_errors"),
                                    "trackers_broken": (s.get("search_totals") or {}).get("trackers_broken"),
                                    "first": run["games"][0].get("time"), "last": run["games"][-1].get("time")}
        opp = opp_report(folder)
        if opp:
            result.setdefault("opponents", {})[os.path.basename(os.path.normpath(folder))] = opp
    for name, run in result["runs"].items():
        print(f"\n== {name}   ({run['first']} to {run['last']}, {run.get('minutes')} min)")
        print(f"   invalid orders, exceptions, forced random, other errors: {run.get('audit')}; inactivity {run.get('inactivity')}, "
              f"forfeit {run.get('forfeit')}, stalled {run['stalled']}, start retries {run.get('retries')}, loop lag {run.get('loop_lag')}")
        if run.get("bridge"):
            b = run["bridge"]
            print(f"   helpers: {b.get('decisions')} decisions, {b.get('expired')} expired, {b.get('timeouts')} timeouts, "
                  f"{b.get('helper_exits')} exits, {b.get('hangs')} hangs, {b.get('iterations_per_decision')} iterations per decision")
        for arm, a in run["arms"].items():
            t = a["tera"]
            print(f"   {arm:<14} {a['games']:4d} games  {a['wins']}-{a['losses']}  {pct(a['wins'], a['games'])}"
                  f"   p1 {a['p1']['wins']}/{a['p1']['games']}  p2 {a['p2']['wins']}/{a['p2']['games']}")
            print(f"       Tera in {t['games']} ({pct(t['games'], a['games'])}), turn median {t['turn_median']} mean {t['turn_mean']}, by {t['by']}, "
                  f"won with {t['won_with']}/{t['games']} without {t['won_without']}/{t['games_without']}; game {t['turns_mean_game']} turns, {a['seconds_per_game']} s")
            print(f"       decisions {a['decisions']}, asked {a['asked']}, answered {a['answered']}, fallbacks {a['fallbacks']} {a['fallback_reasons']}, late {a['late']}")
            if a["answered"]:
                print(f"       disagreed {a['disagreed']} ({100.0 * a['disagreed'] / a['answered']:.1f}%), network overruled {a['net_overruled']} "
                      f"({100.0 * a['net_overruled'] / a['answered']:.1f}%), search overruled {a['search_overruled']} "
                      f"({100.0 * a['search_overruled'] / a['answered']:.1f}%), Tera added {a['tera_added']} dropped {a['tera_dropped']}")
                print(f"       iterations {a['iterations']}")
            print(f"       ms per decision {a['decision_ms']}")
            print(f"       Tera decisions {a['tera_decisions']}")
            if a.get("predict"):
                print(f"       predict {a['predict']}")
    # Every pair of arms that met the same opponent (the runs of one folder): the difference of their shares.
    groups: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}
    for name, run in result["runs"].items():
        for arm, a in run["arms"].items():
            groups.setdefault(name.split("/")[0], []).append((name, arm, a))
    result["differences"] = {}
    for group, members in groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (n1, a1, r1), (n2, a2, r2) = members[i], members[j]
                d = difference(r1["wins"], r1["games"], r2["wins"], r2["games"])
                if not d:
                    continue
                key = f"{n1}:{a1} minus {n2}:{a2}"
                result["differences"][key] = d
                print(f"\n   {key}: {100 * d['difference']:+.1f} points (95% {100 * d['low']:+.1f} to {100 * d['high']:+.1f}), z {d['z']:.2f}; "
                      f"games per arm for this difference to be 2 standard errors: {d['games_per_arm_for_2se']}, for 80% power: {d['games_per_arm_for_80_power']}")
    for name, opp in (result.get("opponents") or {}).items():
        print(f"\n== opponent of {name}: {opp['kind']}, {opp['games']} games")
        for user, d in opp["by_user"].items():
            print(f"   {user}: {d}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return 0


# ---------------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Head to head games between the search player's arms and an opponent, on a local server.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("ours")
    o.add_argument("--port", type=int, required=True)
    o.add_argument("--user", required=True)
    o.add_argument("--arms", required=True, help="e.g. mix,predict or net")
    o.add_argument("--per-arm", type=int, required=True, help="games per arm")
    o.add_argument("--parallel", type=int, default=6)
    o.add_argument("--helpers", type=int, default=6)
    o.add_argument("--search-ms", type=int, default=500)
    o.add_argument("--deadline", type=float, default=2.5)
    o.add_argument("--mix-rule", default="veto")
    o.add_argument("--checkpoint", required=True)
    o.add_argument("--opp-accept", required=True, help="the opponent account that accepts our challenges (we are p1)")
    o.add_argument("--opp-challenge", required=True, help="the opponent account that challenges us (we are p2)")
    o.add_argument("--out", required=True)
    o.add_argument("--stop-file", required=True)
    o.add_argument("--phase", default="")
    o.add_argument("--gap-s", type=float, default=0.0, help="at least this long between two game starts")
    o.add_argument("--minutes", type=float, default=0.0, help="start no new game after this many minutes (0: no limit)")
    o.add_argument("--game-timeout", type=float, default=1200.0)
    o.add_argument("--seed", type=int, default=None)
    o.add_argument("--timer", action=argparse.BooleanOptionalAction, default=True)
    o.add_argument("--save-battles", action=argparse.BooleanOptionalAction, default=True)

    p = sub.add_parser("opp")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--kind", choices=("net", "foulplay"), required=True)
    p.add_argument("--pairs", required=True, help="ACCEPTS:CHALLENGES:OURS, comma separated")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--fp-ms", type=int, default=150)
    p.add_argument("--fp-helpers", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--stop-file", required=True)

    r = sub.add_parser("report")
    r.add_argument("folders", nargs="+")
    r.add_argument("--json", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "report":
        return report(args)
    os.chdir(REPO)
    args.out = os.path.abspath(args.out)
    args.stop_file = os.path.abspath(args.stop_file)
    if args.cmd == "ours":
        args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        args.parallel = max(1, min(16, args.parallel))
        args.helpers = max(1, min(6, args.helpers))
        return asyncio.run(run_ours(args))
    if args.kind == "net" and not args.checkpoint:
        raise SystemExit("--kind net needs --checkpoint")
    args.fp_helpers = max(1, min(6, args.fp_helpers))
    return asyncio.run(run_opp(args))


if __name__ == "__main__":
    sys.exit(main())
