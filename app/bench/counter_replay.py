"""
Round 3's counter rules replayed on recorded ladder battles THROUGH THE REAL PLAYER: on which
decisions "use the counter" and "don't hit the boosted defence" (search_player.MixSettings
.setup_counter, setup_other_side; the arm "mix-counter") would have played something else than
the mix that played the game, and why.

    python bench/counter_replay.py --scan ~/dev/ladder-1008/scan.json ~/dev/ladder-1008/data/battles/*.jsonl

Every battle file's own lines are fed to an offline search player (no server, no network, no
search): poke-env rebuilds the battle as it was, and at each request the player decides as on the
ladder, with the network's probabilities and the search's shares of that decision's note in the
file in place of a forward pass and a search. So mix() sees exactly the numbers it saw live, and
the counter rules see the battle poke-env saw (build_board). Each replayed mix choice is checked
against what was played live (the "same as live" count); a decision whose note is missing is
played by the stub network and not counted.

What it cannot know: the search's sampled sets (the notes keep its six likeliest opponent
options, not the worlds), so the opponent's ability or item counts as known only when poke-env
knows it. What happened after a changed move is not known either: this says where the rules act,
not whether that wins (the ladder A/B does).

--scan (bench/setup_ladder_scan.py's scan.json) gives each game its category ("loss: set-up
sweep", ...), so the changes are counted per category.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

import search_player as sp  # noqa: E402
import setup_moves as sm  # noqa: E402


def read_battle(path: str) -> Dict[str, Any]:
    """The battle's room, our account id, its incoming messages in order and its decision notes by rqid."""
    room, user, msgs, notes, won = None, None, [], {}, None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            data = row.get("data")
            if row.get("dir") == "note" and isinstance(data, dict):
                room = room or data.get("battle")
                user = user or data.get("user")
                rec = data.get("decision")
                if rec and rec.get("rqid") is not None:
                    notes.setdefault(int(rec["rqid"]), rec)          # the first decision of a request (a refused order is decided again)
                if "result" in data:
                    won = data["result"].get("won")
            elif row.get("dir") == "in" and isinstance(data, str):
                msgs.append(data)
    return {"room": room, "user": user, "msgs": msgs, "notes": notes, "won": won, "path": path}


def player_name(msgs: List[str], user: str) -> str:
    for data in msgs:
        for line in data.split("\n"):
            parts = line.split("|")
            if len(parts) > 3 and parts[1] == "player" and sm.to_id(parts[3]) == user:
                return parts[3]
    return user


class ReplayBridge:
    """Stands in for foulplay_bridge.FoulPlayBridge: every question is answered at once with the decision note's own numbers."""

    def __init__(self) -> None:
        self.notes: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.search_ms, self._closed, self.helpers = 20, False, 4
        self.missing = 0

    def feed(self, key, user, room, data):
        pass

    def note(self, key, choice):
        pass

    def forget(self, key):
        pass

    def stats(self):
        return {}

    def close(self):
        self._closed = True

    def advice_future(self, key, search_ms=None, rqid=None, ttl=None, **more):
        future: concurrent.futures.Future = concurrent.futures.Future()
        room = str(key).split("|", 1)[-1]
        rec = self.notes.get((room, int(rqid))) if rqid is not None else None
        if rec is None or rec.get("shares") is None:
            self.missing += 1
            future.set_result(None)
            return future
        shares = [float(x) for x in rec["shares"]]
        iterations = int(rec.get("iterations") or 100_000)
        visits = [max(1, int(round(s * iterations))) if s > 0 else 0 for s in shares]
        opponent = [SimpleNamespace(name=o[0], share=float(o[1]), seen=bool(o[2]) if len(o) > 2 else True)
                    for o in (rec.get("opponent") or [])]
        future.set_result(SimpleNamespace(policy=shares, visits=visits, score=[0.5 if v else None for v in visits], unmapped=[],
                                          iterations=iterations, opponent=opponent, opponent_worlds=[],
                                          opponent_active=rec.get("opponent_active"), wall_ms=float(rec.get("search_ms") or 0.0),
                                          worlds=int(rec.get("worlds") or 0), payoff=None, payoff_table=None, payoff_theirs=None))
        return future


def snapshot(player: Any, battle: Any, mask: Any, threat: Any) -> Dict[str, Any]:
    """The board at a decision the counter rules looked at: both active Pokemon, the opponent's
    boosts, and every legal action as poke-env orders it ("move encore", "switch Milotic"), for the rerun."""
    theirs = getattr(battle, "opponent_active_pokemon", None)
    ours = getattr(battle, "active_pokemon", None)
    orders: Dict[str, str] = {}
    for a in range(len(mask)):
        if mask[a] > 0.5:
            target, kwargs = player.assembler.map_index_to_order(a, battle)
            if target != "DEFAULT":
                orders[str(a)] = player.create_order(target, **(kwargs or {})).message.replace("/choose ", "", 1)
    return {"opp": sm.species_name(str(getattr(theirs, "species", "") or "")) if theirs is not None else "",
            "opp_hp": round(float(getattr(theirs, "current_hp_fraction", 0.0) or 0.0), 3) if theirs is not None else None,
            "opp_boosts": {k: int(v) for k, v in (getattr(theirs, "boosts", None) or {}).items() if int(v)} if theirs is not None else {},
            "ours": sm.species_name(str(getattr(ours, "species", "") or "")) if ours is not None else "",
            "ours_hp": round(float(getattr(ours, "current_hp_fraction", 0.0) or 0.0), 3) if ours is not None else None,
            "threat": ({"boost": int(threat.boost), "last_setup": bool(threat.last_setup), "setup_read": round(float(threat.setup_read), 3)}
                       if threat is not None else None),
            "orders": orders}


def same_mon(a: str, b: str) -> bool:
    """The same Pokemon by species, a forme matching its base name ("Necrozma" / "Necrozma-Dusk-Mane")."""
    x, y = sm.to_id(a), sm.to_id(b)
    return bool(x and y) and (x == y or x.startswith(y) or y.startswith(x))


def judge(change: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Whether a changed decision plausibly answers the game's set-up: what the new action does, at
    whom, at which point of the set-up (scan.json's events: the set-up turn, the KOs after it).
    "answers": "yes" for a counter aimed at a Pokemon of the opponent's that set up in this game
    (a phaze, Haze or Perish Song on it boosted, an Encore after its set-up move, a Taunt or a sleep
    move on it, a switch to a standing Pokemon of ours that holds such a move), and the sweeper
    itself named; "other Pokemon" when the opponent's active is not one that set up in the game;
    for "don't hit the boosted defence", "yes" when the new attack hits the unboosted defence for
    at least 20% (estimated), "little" when it does less.
    """
    import setup_counter as sc

    snap = change.get("snap") or {}
    opp = snap.get("opp") or ""
    sweepers = list(info.get("sweepers") or [])
    events = [e for e in (info.get("events") or []) if same_mon(e.get("mon", ""), opp)]
    before = [e for e in events if int(e["turn"]) <= int(change["turn"])]
    event = (before or events or [None])[-1]
    boosts = snap.get("opp_boosts") or {}
    boosted = ", ".join("%s %+d" % kv for kv in sorted(boosts.items()) if kv[1] > 0) or "no boost"
    out: Dict[str, Any] = {"at": opp, "sweeper": any(same_mon(opp, s) for s in sweepers), "set_up_here": event is not None,
                           "boosts": boosted}
    if event is not None:
        kos = [t for t in (event.get("ko_turns") or []) if int(t) < int(change["turn"])]
        out.update(setup_turn=int(event["turn"]), via=event.get("via"), kos_before=len(kos), kos_total=int(event.get("kos") or 0))
    c = change.get("record") or {}
    if change["kind"] == "other side":
        swap = c.get("other_side") or {}
        dmg = (swap.get("damage") or [None, None])
        out["what"] = "%s instead of %s (%s boosted; estimated %s%% vs %s%%)" % (
            change["pick"], change["mix"], swap.get("boosted"), "?" if dmg[1] is None else round(100 * dmg[1]),
            "?" if dmg[0] is None else round(100 * dmg[0]))
        out["answers"] = "yes" if (dmg[1] or 0) >= 0.20 else "little"
        return out
    counter = c.get("counter") or {}
    if counter.get("decided") == "played the counter":
        kind, move = (str(counter.get("counter") or " ").split(" ", 1) + [""])[:2]
        out["what"] = "%s (%s) instead of %s" % (change["pick"], kind, change["mix"])
    else:
        held = str(counter.get("holder") or "")
        kinds = sorted({sc.counter_kind(m) or ("unaware" if sm.to_id(m) == "unaware" else "") for m in held.split(", ")} - {""})
        kind = "/".join(kinds) or "?"
        out["what"] = "%s (holds %s, %d%% HP) instead of %s" % (change["pick"], held, round(100 * float(counter.get("hp") or 0)),
                                                                change["mix"])
    out["counter_kind"] = kind
    if event is None:
        out["answers"] = "other Pokemon"
    else:
        out["answers"] = "yes"
    return out


def make_player_class():
    """The search player with the network replaced by the decision notes' probabilities (and the stub policy of the tests)."""
    import numpy as np

    import hybrid
    from obs_assembler import ObservationAssembler

    class NotesPolicy:
        def __init__(self, cfg):
            with hybrid._in_repo_root():
                self.assembler = ObservationAssembler()
            self.device, self.path, self.net, self._cache = "cpu", None, None, {}

        def forget(self, tag):
            self._cache.pop(tag, None)

    hybrid.PolicyNet = NotesPolicy
    Base = sp.search_player()

    class Replayer(Base):
        def _board(self, battle, mask, threat, advice, settings):
            board = super()._board(battle, mask, threat, advice, settings)
            try:
                self.snapshots[(battle.battle_tag, int((battle.last_request or {}).get("rqid")))] = snapshot(self, battle, mask, threat)
            except Exception:           # the record of the replay must never change what is replayed
                pass
            return board

        def _forward_batch(self, tags, obs):
            rows = []
            for tag, row in zip(tags, obs):
                battle = self.battles.get(tag)
                rqid = (getattr(battle, "last_request", None) or {}).get("rqid")
                rec = self.replay_bridge.notes.get((tag, int(rqid))) if rqid is not None else None
                if rec is not None and rec.get("probs"):
                    rows.append(np.array([math.log(max(float(p), 1e-9)) for p in rec["probs"]], dtype=np.float32))
                else:
                    rows.append(np.zeros(14, dtype=np.float32))
                self.policy._cache[tag] = True
            return np.stack(rows)

        def _foulplay_fill(self, battle, raw):
            return raw                  # the bridge answers on our 14 actions already

    return Replayer


def replay(paths: List[str], scan: Dict[str, Dict[str, Any]], mode: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from poke_env import AccountConfiguration, ServerConfiguration

    from hybrid import HybridConfig

    Replayer = make_player_class()
    hcfg = HybridConfig(checkpoint=None, beta=0.0, veto_gap=None, policy_weight=1.0, greedy=True, device="cpu", random_init=True)
    games: List[Dict[str, Any]] = []
    for path in paths:
        b = read_battle(path)
        if not b["room"] or not b["user"]:
            continue
        bridge = ReplayBridge()
        bridge.notes = {(b["room"], rqid): rec for rqid, rec in b["notes"].items()}
        name = player_name(b["msgs"], b["user"])
        player = Replayer(hybrid=hcfg, account_configuration=AccountConfiguration(name, None),
                          server_configuration=ServerConfiguration("ws://127.0.0.1:9/showdown/websocket", "http://127.0.0.1:9/action.php?"),
                          battle_format="gen9randombattle", start_listening=False, log_level=50, max_concurrent_battles=3,
                          mode=mode, bridge=bridge, search_ms=20, deadline=0.5, warm_up=False,
                          mix=sp.MixSettings(min_iterations=0, **(extra or {})), live_predictions=True)
        player.replay_bridge = bridge
        player.snapshots = {}
        player.DEFAULT_CHOICE_CHANCE = 0.0

        async def send_message(message, room="", message_2=None):
            return None

        player._order_send = send_message

        async def feed():
            for data in b["msgs"]:
                await player._handle_battle_message([[">" + b["room"]]] + [line.split("|") for line in data.split("\n")])

        asyncio.run_coroutine_threadsafe(feed(), player.ps_client.loop).result(timeout=600)
        decisions = player.decisions_of(b["room"])
        same = compared = 0
        changes = []
        kept_why: Dict[str, int] = {}
        decided: Dict[str, int] = {}
        failed: Dict[str, int] = {}
        info = scan.get(b["room"], {})
        for d in decisions:
            rec = b["notes"].get(int(d["rqid"])) if d.get("rqid") is not None else None
            if rec is None or d.get("search") is None:
                continue
            mixed = d["counter"]["mix"] if "counter" in d else d["played"]
            compared += 1
            same += int(mixed == rec.get("played"))
            kept = (d.get("counter") or {}).get("other_side_kept")
            if kept:
                why = str(kept.get("why") or "?")
                why = "the best of the other kind does less than 80% of it" if why.startswith("the best of the other kind") else why
                kept_why[why] = kept_why.get(why, 0) + 1
            counter = (d.get("counter") or {}).get("counter")
            if counter:
                decided[counter.get("decided") or "?"] = decided.get(counter.get("decided") or "?", 0) + 1
                board = (d.get("counter") or {}).get("board") or {}
                for why in list((board.get("failed") or {}).values()) + list((board.get("passed") or {}).values()):
                    why = str(why).split(": ", 1)[-1] if str(why).startswith("switch ") else str(why)
                    why = "two hits would knock it out" if why.startswith("two hits") else why
                    why = "under 50% HP" if why.endswith("% HP") and " at " in why else why
                    failed[why] = failed.get(why, 0) + 1
            if "counter" in d and d["counter"].get("changed"):
                c = d["counter"]
                rqid = int(d["rqid"])
                snap = player.snapshots.get((b["room"], rqid)) or {}
                change = {"turn": d["turn"], "rqid": rqid, "why": c.get("why"), "net": c["text"]["net"], "search": c["text"]["search"],
                          "mix": c["text"]["mix"], "pick": c["text"]["pick"], "live": rec.get("played") == mixed,
                          "mix_action": int(c["mix"]), "pick_action": int(c["pick"]), "first": not changes,
                          "share_mix": round(float(rec["shares"][int(c["mix"])]), 4), "share_pick": round(float(rec["shares"][int(c["pick"])]), 4),
                          "p_mix": round(float(rec["probs"][int(c["mix"])]), 4), "p_pick": round(float(rec["probs"][int(c["pick"])]), 4),
                          "order_mix": (snap.get("orders") or {}).get(str(int(c["mix"]))),
                          "order_pick": (snap.get("orders") or {}).get(str(int(c["pick"]))),
                          "kind": "counter" if (c.get("counter") or {}).get("decided") in ("played the counter", "switched to a counter")
                          else "other side", "snap": {k: v for k, v in snap.items() if k != "orders"}, "record": c}
                change["judge"] = judge(change, info)
                changes.append(change)
        summary = player.game_summary(b["room"]) or {}
        games.append({"battle": b["room"], "won": b["won"] if b["won"] is not None else info.get("won"), "category": info.get("category", "?"),
                      "opponent": info.get("opponent"), "decisions": len(decisions), "compared": compared, "same_as_live": same,
                      "path": path, "sweepers": info.get("sweepers") or [], "events": [
                          {k: e.get(k) for k in ("turn", "mon", "via", "kos", "ko_turns", "end", "end_turn", "our_answers")}
                          for e in (info.get("events") or [])],
                      "decided": decided, "failed": failed,
                      "missing_notes": bridge.missing, "counts": {k: summary.get(k, 0) for k in sp.COUNTER_COUNTS}, "changes": changes, "kept_why": kept_why,
                      "errors": player.search_totals.get("counter_errors", 0), "exceptions": player.search_totals.get("exceptions", 0)})
        try:
            player.ps_client.loop.call_soon_threadsafe(lambda: None)
        except Exception:
            pass
    return {"games": games}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """The 95% Wilson interval of k out of n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def sweep_table(games: List[Dict[str, Any]]) -> str:
    """One row per set-up sweep loss: who swept, whether a rule changed a decision, the first change and whether it answers."""
    lines = ["| game | sweeper (set-up turn, KOs) | answers on our team (scan: phaze, Haze, Encore, Taunt, status, priority, Unaware...) | changes / set-up positions looked | first change | answers the sweep? | what use-the-counter decided otherwise |",
             "|---|---|---|---|---|---|---|"]
    for g in games:
        if g["category"] != "loss: set-up sweep":
            continue
        sweep = [e for e in g.get("events") or [] if any(same_mon(e["mon"], s) for s in g.get("sweepers") or [])]
        who = "; ".join("%s (T%d, %d KO)" % (e["mon"], e["turn"], e.get("kos") or 0) for e in sweep) or ", ".join(g.get("sweepers") or [])
        answers = sorted({a.split("(")[0].rstrip("*") + ":" + a.split(":", 1)[1] for e in sweep for a in (e.get("our_answers") or []) if ":" in a})
        ch = g["changes"]
        first = ch[0] if ch else None
        if first is not None:
            j = first["judge"]
            text = "T%d %s" % (first["turn"], j["what"])
            verdict = "%s, at %s (%s)%s" % (j["answers"], j["at"], j["boosts"],
                                            ", %d of %d KOs already" % (j["kos_before"], j["kos_total"]) if "kos_before" in j else "")
            if not j["sweeper"] and j["answers"] == "yes":
                verdict += ", not the sweeper"
        else:
            text, verdict = "none", "-"
        held = ", ".join("%s %d" % (k.replace("kept: ", ""), v) for k, v in sorted(g.get("decided", {}).items(), key=lambda kv: -kv[1])
                         if not k.startswith("played") and not k.startswith("switched"))
        lines.append("| %s | %s | %s | %d / %d | %s | %s | %s |" % (
            sp.short_tag(g["battle"]), who, ", ".join(answers) or "none", len(ch), sum(g.get("decided", {}).values()), text, verdict, held or "-"))
    return "\n".join(lines)


def report(out: Dict[str, Any]) -> str:
    games = out["games"]
    lines = []
    compared = sum(g["compared"] for g in games)
    same = sum(g["same_as_live"] for g in games)
    lines.append("%d games, %d searched decisions replayed; the replayed mix played what was played live on %d (%.1f%%); "
                 "counter errors %d, decisions that raised %d"
                 % (len(games), compared, same, 100.0 * same / max(1, compared), sum(g["errors"] for g in games),
                    sum(g["exceptions"] for g in games)))
    cats: Dict[str, List[Dict[str, Any]]] = {}
    for g in games:
        cats.setdefault(g["category"], []).append(g)
    lines.append("")
    lines.append("| category | games | decisions | use the counter looked | played a counter | switched to one | kept for KO | mix had one "
                 "| attacks into a boosted defence | swapped | games with a change |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cat in sorted(cats):
        rows = cats[cat]
        n = {k: sum(g["counts"].get(k, 0) for g in rows) for k in sp.COUNTER_COUNTS}
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d |" % (
            cat, len(rows), sum(g["compared"] for g in rows), n["counter_looked"], n["counter_played"], n["counter_switched"],
            n["counter_ko_kept"], n["counter_had"], n["other_side_looked"], n["other_side"], sum(1 for g in rows if g["changes"])))
    lines.append("")
    for cat in sorted(cats):
        why: Dict[str, int] = {}
        for g in cats[cat]:
            for k, v in g.get("kept_why", {}).items():
                why[k] = why.get(k, 0) + v
        if why:
            lines.append("%s: attacks into a boosted defence that stayed: %s" % (cat, ", ".join("%s %d" % kv for kv in sorted(why.items(), key=lambda kv: -kv[1]))))
    lines.append("")
    lines.append("Changed decisions per kind of game (95% Wilson intervals):")
    lines.append("")
    lines.append("| games | n | decisions | changed decisions | per 100 decisions | games with a change | share of games |")
    lines.append("|---|---|---|---|---|---|---|")
    groups = (("set-up sweep losses", lambda g: g["category"] == "loss: set-up sweep"),
              ("other losses with a set-up", lambda g: g["category"] == "loss: other (opp set up, took <2)"),
              ("losses without a set-up", lambda g: g["category"] == "loss: other (no set-up)"),
              ("wins", lambda g: str(g["category"]).startswith("win")))
    for label, test in groups:
        rows = [g for g in games if test(g)]
        n = sum(g["compared"] for g in rows)
        k = sum(len(g["changes"]) for g in rows)
        kg = sum(1 for g in rows if g["changes"])
        lo, hi = wilson(k, n)
        glo, ghi = wilson(kg, len(rows))
        lines.append("| %s | %d | %d | %d | %.1f (%.1f to %.1f) | %d | %d%% (%d to %d) |" % (
            label, len(rows), n, k, 100.0 * k / max(1, n), 100 * lo, 100 * hi, kg, round(100.0 * kg / max(1, len(rows))),
            round(100 * glo), round(100 * ghi)))
    lines.append("")
    lines.append(sweep_table(games))
    lines.append("")
    for g in games:
        if not g["changes"]:
            continue
        lines.append("%s (%s, %s, vs %s): %d changes" % (sp.short_tag(g["battle"]), "won" if g["won"] else "lost", g["category"],
                                                        g.get("opponent") or "?", len(g["changes"])))
        for c in g["changes"]:
            j = c.get("judge") or {}
            lines.append("  T%d %s: mix %s -> %s (net %s, search %s): %s" % (c["turn"], c["kind"], c["mix"], c["pick"], c["net"],
                                                                         c["search"], c["why"]))
            lines.append("      vs %s (%s)%s; answers: %s%s; live search share mix %.2f, rule %.2f; network mix %.2f, rule %.2f" % (
                j.get("at") or "?", j.get("boosts") or "?",
                (", set up T%d via %s, %d of %d KOs before" % (j["setup_turn"], j.get("via"), j["kos_before"], j["kos_total"])) if "setup_turn" in j else "",
                j.get("answers") or "?", "" if j.get("sweeper") or not j.get("set_up_here") else " (not the sweeper)",
                c["share_mix"], c["share_pick"], c["p_mix"], c["p_pick"]))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--scan", default="", help="scan.json of bench/setup_ladder_scan.py: each game's category")
    ap.add_argument("--mode", default="mix-counter", help="the arm replayed (default mix-counter: both rules)")
    ap.add_argument("--mix", default="", help="more MixSettings for the player as JSON, under the arm's own flags: "
                    "{\"counter_hp\": 1.01} replays use-the-counter without its switches (no benched Pokemon qualifies)")
    ap.add_argument("--json", default="", help="also write everything here")
    a = ap.parse_args(argv)
    scan: Dict[str, Dict[str, Any]] = {}
    if a.scan:
        with open(a.scan, encoding="utf-8") as fh:
            scan = {g["battle"]: g for g in json.load(fh)}
    out = replay(sorted(a.files), scan, a.mode, json.loads(a.mix) if a.mix else None)
    out["mode"], out["mix"] = a.mode, (json.loads(a.mix) if a.mix else {})
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    print(report(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
