"""
A poke-env player that plays Foul Play's move, asked through foulplay_bridge.

    from foulplay_player import FoulPlayBridgePlayer

    player = FoulPlayBridgePlayer(search_ms=100, helpers=4, battle_format="gen9randombattle", ...)

It is an ordinary Player: it can be the opponent in battle_against(), accept challenges, or sit
in a training pool, anywhere players.PSBotPlayer can. Unlike bench/foulplay.py, which starts real
Foul Play processes that log in and play their own battles, this player is OUR process on OUR
websocket, and only the thinking is Foul Play's (bench/fp_bridge_server.py says how).

Importing this module does not pull in torch or ray.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from poke_env.player import Player

from foulplay_bridge import FoulPlayBridge, FoulPlayMixin, available

__all__ = ["FoulPlayBridgePlayer", "available"]


class FoulPlayBridgePlayer(FoulPlayMixin, Player):
    """
    Plays whatever Foul Play would play.

    search_ms   Foul Play's --search-time-ms for this player's decisions (default: the bridge's, 100).
    helpers     helper processes of the shared bridge, when this player is the first to use it.
    bridge      a FoulPlayBridge of your own instead of the process-wide one.
    pick        "choice": the move Foul Play draws, as it plays on the ladder. "best": always the
                option with the largest share of the search.
    keep_advice keep the Advice of the last decision of every running battle in last_advice
                (battle tag -> Advice), for a caller that wants the search statistics.

    When the bridge fails or answers with something poke-env does not hold legal, a random legal
    order is played instead and counted: foulplay_fallbacks, and the reasons in foulplay_problems.
    The same happens when the request asks whom Revival Blessing revives: Foul Play has no answer
    to that request, so a random fainted Pokemon is revived (foulplay_counts["revive"]).
    """

    foulplay_enabled = True

    def __init__(self, *args: Any, search_ms: Optional[int] = None, helpers: Optional[int] = None,
                 bridge: Optional[FoulPlayBridge] = None, pick: str = "choice", keep_advice: bool = False,
                 **kwargs: Any):
        if pick not in ("choice", "best"):
            raise ValueError("pick must be 'choice' or 'best'")
        # Before the base class: Player.__init__ starts listening, and a battle message can
        # arrive before it returns.
        self.foulplay_search_ms = search_ms
        self.foulplay_helpers = helpers
        self.foulplay_pick = pick
        if bridge is not None:
            self.foulplay = bridge
        self.keep_advice = keep_advice
        self.last_advice: Dict[str, Any] = {}
        self.foulplay_decisions = 0
        self.foulplay_fallbacks = 0
        self.foulplay_after_end = 0
        super().__init__(*args, **kwargs)

    async def choose_move(self, battle: Any):
        if getattr(battle, "finished", False):
            # The battle ended while the last decision was being searched. The server answers that
            # late order with an error and poke-env asks again: there is nothing left to decide.
            self.foulplay_after_end += 1
            return self.choose_default_move()
        self.foulplay_decisions += 1
        order, advice = await self.foulplay_decision(battle)
        if self.keep_advice and advice is not None:
            self.last_advice[battle.battle_tag] = advice
        if order is None:
            # No answer, or not a legal one. Keep the game moving, and tell Foul Play what was played.
            self.foulplay_fallbacks += 1
            order = self.choose_random_move(battle)
            self.foulplay_note(battle, order)
        return order

    def _battle_finished_callback(self, battle: Any):
        self.last_advice.pop(battle.battle_tag, None)
        super()._battle_finished_callback(battle)

    def foulplay_summary(self) -> str:
        counts = self.foulplay_counts
        return ("%d decisions, %d fell back to a random move (%d bridge errors, %d illegal answers, %d Revival Blessing "
                "requests Foul Play cannot answer), %d Tera requests dropped on locked moves" % (
                    self.foulplay_decisions, self.foulplay_fallbacks, counts["errors"], counts["illegal"], counts["revive"],
                    counts["tera_dropped"]))
