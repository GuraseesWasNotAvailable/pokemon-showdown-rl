"""
How the network alone uses Terastallization, measured in real games on a local server.

    python bench/tera_usage.py --checkpoint CKPT --opponent simple --games 150 --port 8610
    python bench/tera_usage.py --checkpoint CKPT --opponent self   --games 150 --port 8610
    python bench/tera_usage.py --summarize runs/tera/*.json

The player is the one ladder.py puts on the ladder: players.batched_policy_player (the best
legal action of the policy, no search, no heuristic), unchanged. This script only listens:

  * to the network's answer for every decision (the logits, before the action is picked),
  * to what the player sends to the server ("/choose move ... terastallize"),
  * to what the server sends back ("|-terastallize|p1a: ...|Type", the next request).

It reports, per match:

  share of games in which it Terastallized, and on which turn;
  the probability the policy puts on the four Tera actions together on turns where Tera is
  offered (mean, median, share of turns above 0.1 and above 0.5), and how often a Tera action
  is the best action;
  whether the games with Tera were won more often (NOT causal: a game that lasts longer has more
  chances to Terastallize, and a player who is winning gets to choose when);
  what the Tera was used for (the move's type against the Tera type, the Tera type against the
  Pokemon's own types);
  and three checks of the plumbing: every Tera order sent was answered by a "-terastallize"
  line, the mask offers Tera exactly when the request does, and what the observation shows of
  the player's own Tera types (--own-tera-type switches ObservationAssembler(own_tera_type=True)
  on, an input the checkpoints so far were not trained on: a probe, not a fix).

--wire N writes everything sent and received in the first N battles to <out>.wire.txt.

Local servers only. It sends nothing but the challenge, its orders and (poke-env's) room leave.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import secrets
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TERA = slice(10, 14)


def wilson(k: float, n: int, z: float = 1.96) -> tuple:
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
    return f"{100 * k / n:.1f}% ({k}/{n}, 95% {100 * lo:.1f} to {100 * hi:.1f})"


def legal_probabilities(logits: Any, mask: Any) -> Any:
    """The policy's probabilities over the legal actions: what hybrid.combine ranks."""
    import numpy as np

    legal = mask > 0.5
    z = np.where(legal, logits.astype(np.float64), -np.inf)
    z = z - z[legal].max()
    p = np.exp(z)
    return p / p.sum()


# ---------------------------------------------------------------------------
# The listening player
# ---------------------------------------------------------------------------

def probe_class() -> type:
    import numpy as np
    from poke_env.battle import PokemonType
    from poke_env.data import GenData

    from hybrid import combine
    from obs_assembler import own_tera_type_name
    from players import audited, batched_policy_player

    base = audited(batched_policy_player())

    class TeraProbe(base):
        """players.batched_policy_player with taps. choose_move is the parent's, untouched."""

        def __init__(self, *args: Any, **kwargs: Any):
            self.decisions: List[Dict[str, Any]] = []       # one per move request that offers Tera
            self.tera_events: Dict[str, Dict[str, Any]] = {}  # room -> our "-terastallize"
            self.opp_tera_events: Dict[str, Dict[str, Any]] = {}
            self.sent_tera: Dict[str, List[str]] = {}       # room -> orders sent with "terastallize"
            self.check = {
                "decisions": 0,
                "request_offers_tera": 0, "mask_offers_tera": 0, "mask_request_disagree": 0,
                "own_slots_with_request_tera_type": 0, "own_slots_tera_type_in_obs": 0,
                "own_slots_pokeenv_knows_tera_type": 0,
                "tera_flag_obs_set": 0, "tera_flag_obs_wrong": 0,
                "tera_orders_sent": 0, "tera_lines_received": 0,
            }
            self.wire_rooms: List[str] = []
            self.wire_limit = 0
            self.wire: List[str] = []
            super().__init__(*args, **kwargs)
            body = self.assembler.meta["feature_map"]["body"]
            t0, t1 = body["types_raw"]
            self._tera_lo = t0 + (t1 - t0) // 2
            self._tera_hi = t1
            self._type_chart = GenData.from_gen(9).type_chart
            client = self.ps_client
            send, handle = client.send_message, client._handle_message

            async def tapped_send(message: str, room: str = "", message_2: Optional[str] = None):
                self._on_send(room, message)
                await send(message, room, message_2)

            async def tapped_handle(message: str):
                self._on_receive(message)
                await handle(message)

            client.send_message = tapped_send
            client._handle_message = tapped_handle

        # -- the wire ------------------------------------------------------

        def _logged(self, room: str) -> bool:
            if not room.startswith("battle-"):
                return False
            if room not in self.wire_rooms and len(self.wire_rooms) < self.wire_limit:
                self.wire_rooms.append(room)
            return room in self.wire_rooms

        def _on_send(self, room: str, message: str) -> None:
            if message.startswith("/choose") and " terastallize" in message:
                self.check["tera_orders_sent"] += 1
                self.sent_tera.setdefault(room, []).append(message)
            if self._logged(room):
                self.wire.append(f">>> {room}|{message}")

        def _on_receive(self, message: str) -> None:
            if not message.startswith(">battle-"):
                return
            room = message.split("\n", 1)[0][1:].strip()
            if self._logged(room):
                self.wire.append("<<< " + message.replace("\n", "\n    "))

        async def _handle_battle_message(self, split_messages: List[List[str]]):
            if split_messages and split_messages[0] and split_messages[0][0].startswith(">"):
                room = split_messages[0][0][1:].strip()
                for parts in split_messages[1:]:
                    if len(parts) > 3 and parts[1] == "-terastallize":
                        battle = self.battles.get(room)
                        role = getattr(battle, "player_role", None)
                        event = {"turn": int(getattr(battle, "turn", 0) or 0), "pokemon": parts[2], "type": parts[3]}
                        if role and parts[2].startswith(role):
                            self.check["tera_lines_received"] += 1
                            self.tera_events[room] = event
                        else:
                            self.opp_tera_events[room] = event
            await super()._handle_battle_message(split_messages)

        # -- the network's answer -----------------------------------------

        def _queue_inference(self, tag: str, obs: Any):
            fut = super()._queue_inference(tag, obs)
            battle = self.battles.get(tag)
            snapshot = self._snapshot(battle, obs) if battle is not None else None
            fut.add_done_callback(lambda f: self._on_logits(tag, obs, snapshot, f))
            return fut

        def _snapshot(self, battle: Any, obs: Any) -> Dict[str, Any]:
            """What the request said and what the observation shows, at this decision."""
            c = self.check
            c["decisions"] += 1
            off = self.assembler.offsets
            m0, m1 = off["action_mask"]
            mask = obs[m0:m1].astype(np.float32)
            request = getattr(battle, "last_request", None) or {}
            active_req = (request.get("active") or [{}])[0]
            offers = bool(active_req.get("canTerastallize"))
            mask_offers = bool(mask[TERA].any())
            c["request_offers_tera"] += offers
            c["mask_offers_tera"] += mask_offers
            c["mask_request_disagree"] += offers != mask_offers
            body0 = off["pokemon_body"][0]
            dim = self.assembler.pokemon_scalar_dim
            for i, mon in enumerate(list(battle.team.values())[:6]):
                req_type = (getattr(mon, "_last_request", None) or {}).get("teraType")
                if not req_type or mon.is_terastallized:
                    continue
                c["own_slots_with_request_tera_type"] += 1
                c["own_slots_pokeenv_knows_tera_type"] += mon.tera_type is not None
                block = obs[body0 + i * dim + self._tera_lo: body0 + i * dim + self._tera_hi]
                c["own_slots_tera_type_in_obs"] += bool(np.any(block != 0))
            g0 = off["global_scalars"][0]
            flag = bool(obs[g0 + 1] > 0.5)
            c["tera_flag_obs_set"] += flag
            c["tera_flag_obs_wrong"] += flag != bool(battle.used_tera)
            active = battle.active_pokemon
            snap: Dict[str, Any] = {"turn": int(battle.turn or 0), "offers": offers, "mask": mask}
            if active is not None and mask_offers:
                move_ids = [m.id for m in active.moves.values()]
                moves = list(active.moves.values())
                snap.update({
                    "species": active.species,
                    "tera_type": own_tera_type_name(active) or str(active_req.get("canTerastallize") or "").lower(),
                    "own_types": [t.name.lower() for t in (active.type_1, active.type_2) if t],
                    "move_types": [m.type.name.lower() if m.type else "" for m in moves[:4]],
                    "move_cats": [m.category.name.lower() if m.category else "" for m in moves[:4]],
                    "move_ids": move_ids[:4],
                    "hp": round(float(active.current_hp_fraction or 0.0), 3),
                    "own_left": sum(1 for m in battle.team.values() if not m.fainted),
                    "opp_left": 6 - sum(1 for m in battle.opponent_team.values() if m.fainted),
                })
                snap.update(self._defence(active, battle.opponent_active_pokemon, snap["tera_type"]))
            return snap

        def _defence(self, active: Any, opp: Any, tera_name: str) -> Dict[str, Any]:
            """
            The worst multiplier the opposing active Pokemon's own types have against ours, now
            and after Terastallizing. A stand-in for the threat: its real moves are mostly unknown.
            """
            if opp is None or not tera_name:
                return {}
            try:
                tera = PokemonType.from_name(tera_name)
                threats = [t for t in (opp.type_1, opp.type_2) if t]
                chart = self._type_chart
                before = max(t.damage_multiplier(active.type_1, active.type_2, type_chart=chart) for t in threats)
                after = before if tera == PokemonType.STELLAR else max(t.damage_multiplier(tera, None, type_chart=chart) for t in threats)
            except Exception:
                return {}
            return {"def_before": float(before), "def_after": float(after), "opp_types": [t.name.lower() for t in threats]}

        def _on_logits(self, tag: str, obs: Any, snap: Optional[Dict[str, Any]], fut: Any) -> None:
            if snap is None or fut.cancelled() or fut.exception() is not None:
                return
            mask = snap.pop("mask")
            if not mask[TERA].any():
                return
            logits = fut.result()
            p = legal_probabilities(logits, mask)
            action = combine(logits, None, mask, self.hybrid, None)   # greedy: the action choose_move plays
            best_tera = int(np.argmax(np.where(mask[TERA] > 0.5, p[TERA], -1.0)))
            rec = dict(snap)
            rec.update({
                "battle": tag, "p_tera": round(float(p[TERA].sum()), 6), "p_best": round(float(p.max()), 6),
                "p_switch": round(float(p[4:10].sum()), 6), "action": int(action),
                "best_tera_slot": best_tera, "p_best_tera": round(float(p[10 + best_tera]), 6),
                "p_same_move_plain": round(float(p[best_tera]), 6),
            })
            self.decisions.append(rec)

    return TeraProbe


def warm_up(player: Any) -> None:
    import numpy as np

    size = max(end for _s, end in player.policy.assembler.offsets.values())
    try:
        player._forward_batch(["__warm_up__"], np.zeros((1, size), dtype=np.float32))
        player.batch_stats.update({"forwards": 0, "rows": 0, "largest": 0})
    finally:
        player.policy.forget("__warm_up__")


# ---------------------------------------------------------------------------
# One match
# ---------------------------------------------------------------------------

def side_report(player: Any) -> Dict[str, Any]:
    """Everything one probe player saw, as plain data."""
    endings = getattr(player, "audit_endings", {})
    by_battle: Dict[str, List[Dict[str, Any]]] = {}
    for d in player.decisions:
        by_battle.setdefault(d["battle"], []).append(d)
    games = []
    for tag, battle in player.battles.items():
        if not battle.finished:
            continue
        ev = player.tera_events.get(tag)
        opp = player.opp_tera_events.get(tag)
        ds = by_battle.get(tag, [])
        games.append({
            "battle": tag, "won": bool(battle.won), "turns": int(battle.turn or 0), "ending": endings.get(tag, "normal"),
            "tera": ev is not None, "tera_turn": ev["turn"] if ev else None, "tera_type": ev["type"] if ev else None,
            "tera_pokemon": ev["pokemon"] if ev else None,
            "opp_tera": opp is not None, "opp_tera_turn": opp["turn"] if opp else None,
            "tera_offered_turns": len(ds), "max_p_tera": max((d["p_tera"] for d in ds), default=None),
        })
    return {"games": games, "decisions": player.decisions, "check": dict(player.check), "audit": dict(player.audit),
            "audit_messages": list(player.audit_messages), "batch": dict(player.batch_stats)}


def summarize(sides: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The numbers of one match (one side, or both sides of self-play pooled)."""
    games = [g for s in sides for g in s["games"] if g["ending"] == "normal"]
    cut = sum(1 for s in sides for g in s["games"] if g["ending"] != "normal")
    ok = {g["battle"] + str(i) for i, s in enumerate(sides) for g in s["games"] if g["ending"] == "normal"}
    decisions = [d for i, s in enumerate(sides) for d in s["decisions"] if d["battle"] + str(i) in ok]
    n = len(games)
    tera = [g for g in games if g["tera"]]
    no_tera = [g for g in games if not g["tera"]]
    turns = sorted(g["tera_turn"] for g in tera)
    p = sorted(d["p_tera"] for d in decisions)
    chosen = [d for d in decisions if d["action"] >= 10]
    out: Dict[str, Any] = {
        "games": n, "games_not_counted": cut, "wins": sum(g["won"] for g in games),
        "mean_turns": round(statistics.fmean(g["turns"] for g in games), 1) if games else None,
        "tera_games": len(tera),
        "tera_share": round(len(tera) / n, 4) if n else None,
        "tera_share_ci": [round(x, 4) for x in wilson(len(tera), n)],
        "tera_turn_mean": round(statistics.fmean(turns), 2) if turns else None,
        "tera_turn_median": statistics.median(turns) if turns else None,
        "tera_turn_quartiles": [turns[len(turns) // 4], turns[(3 * len(turns)) // 4]] if turns else None,
        "tera_on_turn_1_to_3": sum(1 for t in turns if t <= 3),
        "tera_relative_time_mean": round(statistics.fmean(g["tera_turn"] / max(g["turns"], 1) for g in tera), 3) if tera else None,
        "opp_tera_games": sum(g["opp_tera"] for g in games),
        "tera_offered_turns": len(decisions),
        "tera_offered_turns_per_game": round(len(decisions) / n, 2) if n else None,
        "p_tera_mean": round(statistics.fmean(p), 4) if p else None,
        "p_tera_median": round(statistics.median(p), 5) if p else None,
        "p_tera_above_0.1": round(sum(x > 0.1 for x in p) / len(p), 4) if p else None,
        "p_tera_above_0.5": round(sum(x > 0.5 for x in p) / len(p), 4) if p else None,
        "p_tera_below_0.01": round(sum(x < 0.01 for x in p) / len(p), 4) if p else None,
        "tera_action_best_turns": len(chosen),
        "tera_action_best_share": round(len(chosen) / len(decisions), 4) if decisions else None,
        "win_rate_with_tera": round(sum(g["won"] for g in tera) / len(tera), 4) if tera else None,
        "win_rate_without_tera": round(sum(g["won"] for g in no_tera) / len(no_tera), 4) if no_tera else None,
        "wins_with_tera": sum(g["won"] for g in tera), "wins_without_tera": sum(g["won"] for g in no_tera),
        "mean_turns_with_tera": round(statistics.fmean(g["turns"] for g in tera), 1) if tera else None,
        "mean_turns_without_tera": round(statistics.fmean(g["turns"] for g in no_tera), 1) if no_tera else None,
    }
    # Games that ended without Tera: how close did the policy ever get to it.
    never = [g["max_p_tera"] for g in no_tera if g["max_p_tera"] is not None]
    out["no_tera_games_max_p_median"] = round(statistics.median(never), 4) if never else None
    # What the Tera was used for.
    if chosen:
        def move_type(d: Dict[str, Any]) -> str:
            slot = d["action"] - 10
            return d["move_types"][slot] if slot < len(d["move_types"]) else ""

        def move_cat(d: Dict[str, Any]) -> str:
            slot = d["action"] - 10
            return d["move_cats"][slot] if slot < len(d["move_cats"]) else ""

        k = len(chosen)
        out["use"] = {
            "tera_decisions": k,
            "tera_type_is_an_own_type": sum(d["tera_type"] in d["own_types"] for d in chosen),
            "move_type_is_tera_type": sum(bool(d["tera_type"]) and move_type(d) == d["tera_type"] for d in chosen),
            "move_is_terablast": sum((d["move_ids"][d["action"] - 10] if d["action"] - 10 < len(d["move_ids"]) else "") == "terablast" for d in chosen),
            "move_is_status": sum(move_cat(d) == "status" for d in chosen),
            "attack_of_another_type": sum(move_cat(d) != "status" and move_type(d) != d["tera_type"] for d in chosen),
            "hp_mean": round(statistics.fmean(d["hp"] for d in chosen), 3),
            "own_left_mean": round(statistics.fmean(d["own_left"] for d in chosen), 2),
            "opp_left_mean": round(statistics.fmean(d["opp_left"] for d in chosen), 2),
            "p_tera_when_chosen_mean": round(statistics.fmean(d["p_tera"] for d in chosen), 3),
        }
    # Does the policy Terastallize more when the Tera type it cannot see would help? "better" and
    # "worse" are the opposing active Pokemon's own types against ours, before and after.
    graded = [d for d in decisions if "def_before" in d]
    if graded:
        def kind(d: Dict[str, Any]) -> str:
            return "better" if d["def_after"] < d["def_before"] else ("worse" if d["def_after"] > d["def_before"] else "same")

        def has_tera_attack(d: Dict[str, Any]) -> bool:
            return any(t == d["tera_type"] and c != "status" for t, c in zip(d["move_types"], d["move_cats"]))

        def boosted(d: Dict[str, Any]) -> bool:
            slot = d["action"] - 10
            return 0 <= slot < len(d["move_types"]) and d["move_types"][slot] == d["tera_type"] and d["move_cats"][slot] != "status"

        quality: Dict[str, Any] = {"turns": len(graded)}
        for name in ("better", "same", "worse"):
            rows = [d for d in graded if kind(d) == name]
            took = [d for d in rows if d["action"] >= 10]
            quality[name] = {
                "turns": len(rows), "share_of_turns": round(len(rows) / len(graded), 4),
                "p_tera_mean": round(statistics.fmean(d["p_tera"] for d in rows), 4) if rows else None,
                "tera_chosen": len(took), "chosen_rate": round(len(took) / len(rows), 4) if rows else None,
            }
        took = [d for d in graded if d["action"] >= 10]
        quality["chosen"] = {
            "tera_decisions": len(took),
            "defence_better": sum(kind(d) == "better" for d in took),
            "defence_same": sum(kind(d) == "same" for d in took),
            "defence_worse": sum(kind(d) == "worse" for d in took),
            "move_boosted": sum(boosted(d) for d in took),
            "boosted_or_defence_better": sum(boosted(d) or kind(d) == "better" for d in took),
            "neither": sum((not boosted(d)) and kind(d) == "same" for d in took),
            "defence_worse_and_move_not_boosted": sum((not boosted(d)) and kind(d) == "worse" for d in took),
        }
        with_attack = [d for d in graded if has_tera_attack(d)]
        without = [d for d in graded if not has_tera_attack(d)]
        quality["has_attack_of_tera_type"] = {
            "turns": len(with_attack),
            "chosen_rate": round(sum(d["action"] >= 10 for d in with_attack) / len(with_attack), 4) if with_attack else None,
            "turns_without": len(without),
            "chosen_rate_without": round(sum(d["action"] >= 10 for d in without) / len(without), 4) if without else None,
        }
        out["quality"] = quality
    # Self-play: the same network on both sides, so any difference between them is the Tera and
    # whatever led to it (still not causal: the side under pressure may be the one that uses it).
    if len(sides) == 2:
        a = {g["battle"]: g for g in sides[0]["games"] if g["ending"] == "normal"}
        b = {g["battle"]: g for g in sides[1]["games"] if g["ending"] == "normal"}
        one = one_won = both = first_won = first_n = neither = 0
        for tag in a.keys() & b.keys():
            ga, gb = a[tag], b[tag]
            if ga["tera"] != gb["tera"]:
                one += 1
                one_won += (ga if ga["tera"] else gb)["won"]
            elif ga["tera"]:
                both += 1
                if ga["tera_turn"] != gb["tera_turn"]:
                    first_n += 1
                    first_won += (ga if ga["tera_turn"] < gb["tera_turn"] else gb)["won"]
            else:
                neither += 1
        out["pairs"] = {"games": len(a.keys() & b.keys()), "only_one_side_tera": one, "that_side_won": one_won,
                        "both": both, "both_on_different_turns": first_n, "first_to_tera_won": first_won, "neither": neither}
    check: Dict[str, int] = {}
    for s in sides:
        for key, v in s["check"].items():
            check[key] = check.get(key, 0) + int(v)
    out["check"] = check
    audit: Dict[str, int] = {}
    for s in sides:
        for key, v in s["audit"].items():
            audit[key] = audit.get(key, 0) + int(v)
    out["audit"] = audit
    return out


def print_summary(label: str, s: Dict[str, Any]) -> None:
    n = s["games"]
    print(f"\n== {label}")
    print(f"games {n} ({s['games_not_counted']} more ended on a forfeit or the timer and are not counted), "
          f"won {pct(s['wins'], n)}, {s['mean_turns']} turns a game")
    print(f"Terastallized in {pct(s['tera_games'], n)} of its games")
    if s["tera_games"]:
        q = s["tera_turn_quartiles"]
        print(f"  turn of the Tera: mean {s['tera_turn_mean']}, median {s['tera_turn_median']}, quartiles {q[0]} and {q[1]}, "
              f"{s['tera_on_turn_1_to_3']} of {s['tera_games']} on turns 1 to 3; on average {100 * s['tera_relative_time_mean']:.0f}% of the way through the game")
    print(f"turns on which Tera was offered: {s['tera_offered_turns']} ({s['tera_offered_turns_per_game']} a game)")
    if s["tera_offered_turns"]:
        print(f"  probability on the four Tera actions together: mean {s['p_tera_mean']}, median {s['p_tera_median']}, "
              f"above 0.1 on {100 * s['p_tera_above_0.1']:.1f}% of those turns, above 0.5 on {100 * s['p_tera_above_0.5']:.1f}%, "
              f"below 0.01 on {100 * s['p_tera_below_0.01']:.1f}%")
        print(f"  a Tera action was the best action on {s['tera_action_best_turns']} turns ({100 * s['tera_action_best_share']:.2f}%)")
    if s["no_tera_games_max_p_median"] is not None:
        print(f"  in the games without Tera, the highest it ever put on Tera: median {s['no_tera_games_max_p_median']}")
    if s["win_rate_with_tera"] is not None and s["win_rate_without_tera"] is not None:
        a, b = s["tera_games"], n - s["tera_games"]
        print(f"won {pct(s['wins_with_tera'], a)} of the games with Tera ({s['mean_turns_with_tera']} turns) and "
              f"{pct(s['wins_without_tera'], b)} of the games without ({s['mean_turns_without_tera']} turns). Not causal.")
    u = s.get("use")
    if u:
        k = u["tera_decisions"]
        print(f"the {k} Tera decisions: Tera type is one of the Pokemon's own types in {u['tera_type_is_an_own_type']}, "
              f"the move has the Tera type in {u['move_type_is_tera_type']} (Tera Blast in {u['move_is_terablast']}), "
              f"a status move in {u['move_is_status']}, an attack of another type in {u['attack_of_another_type']}; "
              f"own HP {100 * u['hp_mean']:.0f}%, {u['own_left_mean']} own and {u['opp_left_mean']} opposing Pokemon left")
    q = s.get("quality")
    if q:
        ch = q["chosen"]
        k = max(ch["tera_decisions"], 1)
        print(f"what the Tera did, {ch['tera_decisions']} decisions (defence = the opposing active Pokemon's own types against ours): "
              f"defence better {ch['defence_better']} ({100 * ch['defence_better'] / k:.0f}%), same {ch['defence_same']}, "
              f"worse {ch['defence_worse']} ({100 * ch['defence_worse'] / k:.0f}%); the move played got the Tera boost in {ch['move_boosted']} "
              f"({100 * ch['move_boosted'] / k:.0f}%); boost or better defence {ch['boosted_or_defence_better']} "
              f"({100 * ch['boosted_or_defence_better'] / k:.0f}%), neither {ch['neither']}, worse defence and no boost "
              f"{ch['defence_worse_and_move_not_boosted']} ({100 * ch['defence_worse_and_move_not_boosted'] / k:.0f}%)")
        for name in ("better", "same", "worse"):
            r = q[name]
            print(f"  turns where Tera would make the defence {name}: {r['turns']} ({100 * r['share_of_turns']:.0f}% of the offered turns), "
                  f"mean p(Tera) {r['p_tera_mean']}, Tera chosen on {r['tera_chosen']} ({100 * (r['chosen_rate'] or 0):.2f}%)")
        h = q["has_attack_of_tera_type"]
        print(f"  turns with an attack of the Tera type: {h['turns']}, Tera chosen on {100 * (h['chosen_rate'] or 0):.2f}%; "
              f"without one: {h['turns_without']}, chosen on {100 * (h['chosen_rate_without'] or 0):.2f}%")
    pr = s.get("pairs")
    if pr:
        print(f"self-play pairs: in {pr['only_one_side_tera']} of {pr['games']} games only one side Terastallized, and that side won "
              f"{pct(pr['that_side_won'], pr['only_one_side_tera'])}; in {pr['both']} both did, and the first to do it won "
              f"{pct(pr['first_to_tera_won'], pr['both_on_different_turns'])}; in {pr['neither']} neither did")
    c = s["check"]
    print(f"plumbing: {c['tera_orders_sent']} Tera orders sent, {c['tera_lines_received']} '-terastallize' lines for our side received; "
          f"request offered Tera on {c['request_offers_tera']} decisions, the mask on {c['mask_offers_tera']} "
          f"({c['mask_request_disagree']} disagreements)")
    print(f"  own Tera type: the request named it for {c['own_slots_with_request_tera_type']} (decision, own Pokemon not yet Terastallized) pairs; "
          f"poke-env knew it in {c['own_slots_pokeenv_knows_tera_type']}, the observation showed it in {c['own_slots_tera_type_in_obs']}")
    print(f"  'we used Tera' flag in the observation: set on {c['tera_flag_obs_set']} of {c['decisions']} decisions, wrong on {c['tera_flag_obs_wrong']}")
    a = s["audit"]
    print(f"  orders rejected as invalid {a.get('invalid_choices', 0)}, decisions that raised {a.get('exceptions', 0)}, forced random {a.get('forced_random', 0)}")


async def run_match(args: argparse.Namespace) -> Dict[str, Any]:
    from poke_env import AccountConfiguration, ServerConfiguration

    from hybrid import HybridConfig
    from play import AUTH_URL, build_player

    server = ServerConfiguration(f"ws://127.0.0.1:{args.port}/showdown/websocket", AUTH_URL)
    Probe = probe_class()
    tag = secrets.token_hex(3)

    def probe(name: str, checkpoint: str, own_tera_type: bool) -> Any:
        hcfg = HybridConfig(checkpoint=checkpoint, beta=0.0, veto_gap=None, policy_weight=1.0, greedy=True,
                            device=args.device, random_init=False)   # ladder.py's settings
        p = Probe(hybrid=hcfg, account_configuration=AccountConfiguration(name, None), server_configuration=server,
                  battle_format=args.format, max_concurrent_battles=args.concurrency, log_level=40)
        if not p.policy.path or not os.path.samefile(p.policy.path, checkpoint):
            raise RuntimeError(f"the player did not load {checkpoint}")
        p.assembler.own_tera_type = bool(own_tera_type)
        p.wire_limit = args.wire
        warm_up(p)
        return p

    subject = probe(f"tera{tag}a", args.checkpoint, args.own_tera_type)
    if args.opponent == "self":
        other = probe(f"tera{tag}b", args.opponent_checkpoint or args.checkpoint, args.own_tera_type)
        other.wire_limit = 0
    else:
        other = build_player(args.opponent, server=server, tag=tag + "b", battle_format=args.format,
                             concurrency=args.concurrency, audit=True)
    deadline = time.time() + 30
    while not all(p.ps_client.logged_in.is_set() for p in (subject, other)):
        if time.time() > deadline:
            raise RuntimeError("could not log in to the local server")
        await asyncio.sleep(0.05)
    t0 = time.time()
    stop = "finished"
    try:
        await asyncio.wait_for(subject.battle_against(other, n_battles=args.games), timeout=600 + 60 * args.games)
    except Exception as e:  # the games played until then still count
        stop = f"{type(e).__name__}: {str(e)[:160]}"
    seconds = time.time() - t0
    sides = [side_report(subject)]
    if args.opponent == "self":
        sides.append(side_report(other))
    result = {
        "checkpoint": os.path.basename(args.checkpoint), "opponent": args.opponent,
        "opponent_checkpoint": os.path.basename(args.opponent_checkpoint or args.checkpoint) if args.opponent == "self" else None,
        "own_tera_type": bool(args.own_tera_type), "format": args.format, "requested": args.games,
        "concurrency": args.concurrency, "seconds": round(seconds, 1), "stop": stop,
        "device": str(subject.policy.device), "summary": summarize(sides[:1]),
        "sides": sides,
    }
    if args.opponent == "self":
        result["summary_both_sides"] = summarize(sides)
    other_audit = getattr(other, "audit", {})
    result["opponent_audit"] = dict(other_audit)
    if subject.wire:
        result["wire_rooms"] = list(subject.wire_rooms)
        result["_wire"] = list(subject.wire)
    for p in (subject, other):
        try:
            await asyncio.wait_for(p.ps_client.stop_listening(), timeout=5.0)
        except Exception:
            pass
    return result


def label_of(r: Dict[str, Any]) -> str:
    who = r["checkpoint"].replace("learner_update_", "u").replace(".pt", "")
    return f"{who} vs {r['opponent']}" + (" [own Tera type shown]" if r.get("own_tera_type") else "")


def summarize_files(paths: List[str]) -> int:
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            rows.append((label_of(r), r))
    print("| match | games | won | Tera in | turn (median) | Tera offered turns | p(Tera) mean | median | >0.1 | >0.5 | Tera best | won with | won without |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, r in rows:
        for which, s in (("", r["summary"]),) + ((("  both sides", r["summary_both_sides"]),) if "summary_both_sides" in r else ()):
            n = s["games"]
            lo, hi = s["tera_share_ci"]

            def f(x: Any, scale: float = 1.0, digits: int = 1) -> str:
                return "n/a" if x is None else f"{scale * x:.{digits}f}"

            print(f"| {label}{which} | {n} | {f(s['wins'] / n if n else None, 100)}% | "
                  f"{f(s['tera_share'], 100)}% ({s['tera_games']}/{n}, {100 * lo:.0f}-{100 * hi:.0f}) | "
                  f"{f(s['tera_turn_median'])} | {s['tera_offered_turns']} | {f(s['p_tera_mean'], 1, 4)} | {f(s['p_tera_median'], 1, 5)} | "
                  f"{f(s['p_tera_above_0.1'], 100)}% | {f(s['p_tera_above_0.5'], 100)}% | {f(s['tera_action_best_share'], 100, 2)}% | "
                  f"{f(s['win_rate_with_tera'], 100)}% ({s['wins_with_tera']}/{s['tera_games']}) | "
                  f"{f(s['win_rate_without_tera'], 100)}% ({s['wins_without_tera']}/{n - s['tera_games']}) |")
    for label, r in rows:
        print_summary(label, r["summary"])
        if "summary_both_sides" in r:
            print_summary(label + ", both sides", r["summary_both_sides"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="the learner checkpoint to play")
    ap.add_argument("--opponent", default="simple", help="simple, maxpower, random, psbot, or self (the same network on the other side)")
    ap.add_argument("--opponent-checkpoint", default=None, help="with --opponent self: another checkpoint for the other side")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--concurrency", type=int, default=16, help="battles at the same time")
    ap.add_argument("--port", type=int, default=8610, help="a LOCAL Showdown server (start_servers.py)")
    ap.add_argument("--format", default="gen9randombattle")
    ap.add_argument("--device", default=None)
    ap.add_argument("--own-tera-type", action="store_true",
                    help="probe: show the network its own Pokemon's Tera types (an input it was not trained on)")
    ap.add_argument("--wire", type=int, default=0, help="write all traffic of the first N battles to <out>.wire.txt")
    ap.add_argument("--out", default=None, help="result file (default runs/tera/<checkpoint>_<opponent>.json)")
    ap.add_argument("--summarize", nargs="+", default=None, help="print the table for result files and stop")
    args = ap.parse_args()
    if args.summarize:
        return summarize_files(args.summarize)
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise SystemExit("--checkpoint must name a learner checkpoint file")
    args.checkpoint = os.path.abspath(args.checkpoint)
    if args.opponent_checkpoint:
        args.opponent_checkpoint = os.path.abspath(args.opponent_checkpoint)
    os.chdir(REPO)   # vocab.json is opened by a relative path
    out = args.out or os.path.join(REPO, "runs", "tera", "{}_{}{}.json".format(
        os.path.basename(args.checkpoint).replace(".pt", ""), args.opponent, "_owntera" if args.own_tera_type else ""))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    result = asyncio.run(run_match(args))
    wire = result.pop("_wire", None)
    if wire:
        with open(out + ".wire.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(wire) + "\n")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"{label_of(result)}: {result['seconds']:.0f}s, stopped: {result['stop']}; written to {out}")
    print_summary(label_of(result), result["summary"])
    if "summary_both_sides" in result:
        print_summary(label_of(result) + ", both sides", result["summary_both_sides"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
