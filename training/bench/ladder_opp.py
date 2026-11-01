"""
SimpleHeuristics opponents that search the ladder of a LOCAL Showdown server, for testing ladder.py
(cloud/ladder_box.sh ... test starts it):

    python bench/ladder_opp.py --port 9300 --games 30 --parallel 3 --prefix heurt1

One account playing --parallel games at once: several accounts on one local ladder mostly find
each other instead of the bot. No password, no timer requests; a --no-security server only.
"""

from __future__ import annotations

import argparse
import asyncio


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--accounts", type=int, default=1)
    ap.add_argument("--prefix", default="heurt")
    ap.add_argument("--format", default="gen9randombattle")
    args = ap.parse_args()

    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.player import SimpleHeuristicsPlayer

    server = ServerConfiguration(f"ws://127.0.0.1:{args.port}/showdown/websocket",
                                 "https://play.pokemonshowdown.com/action.php?")
    players = [SimpleHeuristicsPlayer(account_configuration=AccountConfiguration(f"{args.prefix}{i}", None),
                                      server_configuration=server, battle_format=args.format,
                                      max_concurrent_battles=args.parallel, log_level=40,
                                      start_timer_on_battle_start=False)
               for i in range(args.accounts)]
    print(f"{len(players)} SimpleHeuristics account(s) laddering on port {args.port}, {args.games} games each", flush=True)
    await asyncio.gather(*(p.ladder(args.games) for p in players))
    for p in players:
        print(f"{p.username}: {p.n_won_battles} won of {p.n_finished_battles}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
