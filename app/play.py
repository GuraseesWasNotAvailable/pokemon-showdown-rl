"""
Play matches on a local Pokémon Showdown server and report the result.

    python play.py --agent psbot --opponent simple --games 100
    python play.py --agent hybrid --checkpoint checkpoints/learner_update_200.pt --opponent psbot
    python play.py --agent policy --checkpoint checkpoints/learner_update_200.pt --opponent simple

Agents:
    psbot     the PSBot heuristic (psbot/psbot.js, run headless through heuristic/engine.js)
    hybrid    the transformer policy and PSBot together (see hybrid.py)
    policy    the transformer policy alone
    simple    poke-env's SimpleHeuristicsPlayer with the stat fix, the upstream imitation teacher
    maxpower  poke-env's MaxBasePowerPlayer
    random    poke-env's RandomPlayer

The server is the one start_servers.py launches:  node pokemon-showdown start --no-security

For the regular measurement of a training run (several opponents, results on disk, a report over
time) use evaluate.py, which is built on the players made here.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import secrets
import time
from typing import Any, Optional, Tuple

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from players import PSBotPlayer, audited, batched_policy_player, patch_simple_heuristics

AGENTS = ("psbot", "hybrid", "policy", "simple", "maxpower", "random")

AUTH_URL = "https://play.pokemonshowdown.com/action.php?"   # unused on a --no-security server


def wilson(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson interval for a win rate."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def server_configuration(host: str, port: int) -> ServerConfiguration:
    return ServerConfiguration(f"ws://{host}:{port}/showdown/websocket", AUTH_URL)


def build_player(
    kind: str,
    *,
    server: ServerConfiguration,
    tag: str,
    battle_format: str = "gen9randombattle",
    concurrency: int = 8,
    checkpoint: Optional[str] = None,
    beta: float = 4.0,
    veto_gap: Optional[float] = 0.75,
    policy_weight: float = 1.0,
    sample: bool = False,
    device: Optional[str] = None,
    random_init: bool = False,
    audit: bool = False,
    batch: bool = True,
    log_level: int = 40,
) -> Any:
    """
    One player of the given kind, logged in to `server` under a name ending in `tag`.

    "policy" is the transformer alone: the action with the highest probability among the legal
    ones, or a sample from them with sample=True. "hybrid" adds PSBot's score and veto.
    With batch=True the policy answers every waiting battle in one forward pass
    (players.batched_policy_player); with batch=False it runs the network once per decision, as
    hybrid.HybridPlayer does. The moves are the same, the first is several times faster.
    With audit=True the player also counts rejected orders and its own failures
    (players.ChoiceAuditMixin).
    """
    if kind not in AGENTS:
        raise ValueError(f"unknown agent {kind!r}")
    wrap = audited if audit else (lambda cls: cls)
    common = dict(
        account_configuration=AccountConfiguration(f"{kind[:8]}{tag}", None),
        server_configuration=server,
        battle_format=battle_format,
        max_concurrent_battles=concurrency,
        log_level=log_level,
    )
    if kind == "psbot":
        return wrap(PSBotPlayer)(**common)
    if kind in ("hybrid", "policy"):
        from hybrid import HybridConfig, HybridPlayer  # torch is only needed for these two

        # HybridPlayer answers a checkpoint it cannot find by playing as PSBot alone. For a
        # named file that is never what was asked for, so stop here instead.
        if checkpoint and not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        hcfg = HybridConfig(
            checkpoint=checkpoint,
            beta=0.0 if kind == "policy" else beta,
            veto_gap=None if kind == "policy" else veto_gap,
            policy_weight=policy_weight,
            greedy=not sample,
            device=device,
            random_init=random_init,
        )
        cls = batched_policy_player() if (kind == "policy" and batch) else HybridPlayer
        player = wrap(cls)(hybrid=hcfg, **common)
        if kind == "policy" and player.policy is None:
            # No network was loaded, so this "policy" would be PSBot under another name.
            raise FileNotFoundError("no checkpoint to load for the policy agent (give --checkpoint, or --random-init for a plumbing test)")
        return player
    if kind == "simple":
        patch_simple_heuristics()
        return wrap(SimpleHeuristicsPlayer)(**common)
    if kind == "maxpower":
        return wrap(MaxBasePowerPlayer)(**common)
    return wrap(RandomPlayer)(**common)


def make_player(kind: str, args: argparse.Namespace, server: ServerConfiguration, tag: str) -> Any:
    # audit=True: without it a choose_move that raises leaves its battle waiting forever and this
    # command never returns, and a game the server ended on a forfeit or on its timer is counted
    # as a win or a loss (see count_results).
    return build_player(
        kind, server=server, tag=tag, battle_format=args.format, concurrency=args.concurrency,
        checkpoint=args.checkpoint, beta=args.beta, veto_gap=args.veto_gap, policy_weight=args.policy_weight,
        sample=args.sample, device=args.device, random_init=args.random_init, audit=True,
        batch=not getattr(args, "single_inference", False),
    )


def count_results(player: Any) -> Tuple[int, int, int, int, list]:
    """
    (wins, losses, ties, games not counted, turns of the games that are) from the player's own
    battles. A game the server ended because a side forfeited or ran out its inactivity timer says
    nothing about strength: the audit (players.ChoiceAuditMixin) knows which those were, and they
    are left out here as evaluate.py leaves them out. poke-env's n_won_battles counts them.
    """
    endings = getattr(player, "audit_endings", {})
    wins = losses = ties = cut = 0
    turns = []
    for tag, battle in list(player.battles.items()):
        if not battle.finished:
            continue
        if endings.get(tag) in ("forfeit", "inactivity"):
            cut += 1
            continue
        turns.append(int(battle.turn or 0))
        if battle.won:
            wins += 1
        elif battle.lost:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties, cut, turns


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", choices=AGENTS, default="psbot")
    ap.add_argument("--opponent", choices=AGENTS, default="simple")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=8, help="battles played at the same time")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint", default=None, help="model checkpoint for hybrid/policy (default: newest in checkpoints/)")
    ap.add_argument("--beta", type=float, default=4.0, help="hybrid: weight of the heuristic score, in nats per Pokemon of material")
    ap.add_argument("--veto-gap", type=float, default=0.75, help="hybrid: drop options this much worse than PSBot's best")
    ap.add_argument("--policy-weight", type=float, default=1.0, help="hybrid: weight of the policy's log-probabilities")
    ap.add_argument("--sample", action="store_true", help="hybrid/policy: sample the action instead of taking the best one")
    ap.add_argument("--device", default=None, help="hybrid/policy: cpu or cuda (default: cuda when available)")
    ap.add_argument("--random-init", action="store_true", help="hybrid/policy: use an untrained network when no checkpoint exists (plumbing test)")
    ap.add_argument("--single-inference", action="store_true", help="policy: run the network once per decision instead of once for all waiting battles (same moves, slower)")
    args = ap.parse_args()

    server = server_configuration(args.host, args.port)
    tag = secrets.token_hex(3)
    try:
        agent = make_player(args.agent, args, server, tag + "a")
        opponent = make_player(args.opponent, args, server, tag + "b")
    except FileNotFoundError as e:
        raise SystemExit(f"play.py: {e}")

    # Which weights, before the games and not only after them: with no --checkpoint it is the
    # newest file under this repository's own checkpoints/, which is not where a run keeps its own.
    for who, p in ((args.agent, agent), (args.opponent, opponent)):
        policy = getattr(p, "policy", None)
        if policy is not None:
            print(f"{who}: the network is {policy.path or 'UNTRAINED (--random-init)'}")
    print(f"{args.agent} vs {args.opponent}: {args.games} games of {args.format} on {args.host}:{args.port}")
    t0 = time.time()
    await agent.battle_against(opponent, n_battles=args.games)
    dt = time.time() - t0

    wins, losses, ties, cut, turns = count_results(agent)
    n = wins + losses + ties
    lo, hi = wilson(wins, n)
    print(f"\n{args.agent} won {wins} of {n} against {args.opponent} "
          f"({100 * wins / max(n, 1):.1f}%, 95% interval {100 * lo:.1f} to {100 * hi:.1f}), "
          f"{losses} lost, {ties} tied")
    if cut:
        print(f"{cut} more game{'' if cut == 1 else 's'} ended on a forfeit or on the inactivity timer and "
              f"{'is' if cut == 1 else 'are'} not counted")
    print(f"{dt:.0f}s total, {dt / max(n, 1):.1f}s per game, {sum(turns) / max(len(turns), 1):.1f} turns per game")
    for who, p in ((args.agent, agent), (args.opponent, opponent)):
        broken = getattr(p, "audit", {})
        if broken.get("exceptions") or broken.get("forced_random") or broken.get("invalid_choices"):
            print(f"{who}: {broken.get('exceptions', 0)} decisions raised and {broken.get('forced_random', 0)} were forced, "
                  f"a random move was played instead; {broken.get('invalid_choices', 0)} orders were rejected as invalid")
        if hasattr(p, "psbot_decisions"):
            print(f"{who}: {p.psbot_decisions} PSBot decisions, {p.psbot_fallbacks} fell back to a random move")
            try:
                pr = p.psbot.call_sync({"op": "ping"}).get("predictions") or {}
            except Exception:
                pr = {}
            if pr.get("made"):
                made, sw = pr["made"], pr["switched"]
                print(f"{who}: predicted a switch {made} times at {100 * pr['weight'] / made:.0f}% on average; "
                      f"the opponent switched {100 * sw / made:.0f}% of those times, "
                      f"to the Pokemon named in {100 * pr['rightMon'] / max(sw, 1):.0f}% of the switches")
        if hasattr(p, "hybrid_stats"):
            print(f"{who}: {p.hybrid_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
