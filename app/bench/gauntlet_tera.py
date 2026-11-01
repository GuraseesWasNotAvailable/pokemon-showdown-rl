"""
What each arm of a gauntlet run did with Terastallization, read from its saved battles.

    python bench/gauntlet_tera.py runs/gauntlet/vsnet/final runs/gauntlet/vsnet/control ... [--json FILE]

The same questions bench/tera_usage.py asks of the network alone (bench/notes/tera.md), asked of
every arm, from the battle files ladder.py's recorder keeps (battles/<tag>.jsonl: what came in,
what went out). Nothing is replayed and no player is involved.

For every request that offers Terastallization (request.active[0].canTerastallize, which is the
Tera type) and the order that answered it:

  defence   the worst multiplier the opposing active Pokemon's own types have against ours
            (its Tera type alone once it has Terastallized), before and after our Tera. The
            scout's stand-in for the threat: its real moves are mostly unknown.
  boosted   the move played with the Tera is an attack of the Tera type (bench/tera_usage.py's
            rule; Tera Blast is counted apart, because its type in the data is Normal).

Our own types are the species' types (a Pokemon whose type was changed in the battle is judged
by its species), as in the scout's numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def to_id(name: Optional[str]) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum())


class Dex:
    def __init__(self) -> None:
        from poke_env.battle import PokemonType
        from poke_env.data import GenData

        data = GenData.from_gen(9)
        self.pokedex, self.moves, self.chart, self.Type = data.pokedex, data.moves, data.type_chart, PokemonType

    def types(self, species: str) -> List[Any]:
        entry = self.pokedex.get(to_id(species)) or {}
        return [self.Type.from_name(t) for t in entry.get("types") or []]

    def worst(self, attackers: List[Any], defenders: List[Any]) -> Optional[float]:
        if not attackers or not defenders:
            return None
        d1, d2 = defenders[0], (defenders[1] if len(defenders) > 1 else None)
        return max(float(t.damage_multiplier(d1, d2, type_chart=self.chart)) for t in attackers)

    def move(self, move_id: str) -> Dict[str, Any]:
        return self.moves.get(to_id(move_id)) or {}


def read_battle(path: str, dex: Dex) -> Optional[Dict[str, Any]]:
    arm = user = role = None
    result: Optional[Dict[str, Any]] = None
    active: Dict[str, Dict[str, Any]] = {}          # "p1" / "p2" -> the active Pokemon
    tera_of: Dict[str, str] = {}                    # "p1: Name" -> its Tera type once it has Terastallized
    request: Optional[Dict[str, Any]] = None
    turn = 0
    rows: List[Dict[str, Any]] = []
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return None
    for raw in lines:
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        kind, data = d.get("dir"), d.get("data")
        if kind == "note" and isinstance(data, dict):
            if "arm" in data and "battle" in data:
                arm, user = data.get("arm"), data.get("user")
            if isinstance(data.get("result"), dict):
                result = data["result"]
        elif kind == "in" and isinstance(data, str):
            for line in data.split("\n"):
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                what = parts[1]
                if what == "player" and len(parts) > 3 and to_id(parts[3]) == user:
                    role = parts[2]
                elif what in ("switch", "drag", "replace") and len(parts) > 3:
                    ident = parts[2].replace("a:", ":", 1)
                    active[parts[2][:2]] = {"ident": ident, "species": parts[3].split(",")[0], "tera": tera_of.get(ident)}
                elif what in ("detailschange", "-formechange") and len(parts) > 3 and parts[2][:2] in active:
                    active[parts[2][:2]]["species"] = parts[3].split(",")[0]
                elif what == "-terastallize" and len(parts) > 3:
                    ident = parts[2].replace("a:", ":", 1)
                    tera_of[ident] = parts[3]
                    if parts[2][:2] in active:
                        active[parts[2][:2]]["tera"] = parts[3]
                elif what == "turn" and len(parts) > 2 and parts[2].isdigit():
                    turn = int(parts[2])
                elif what == "request" and len(parts) > 2 and parts[2]:
                    try:
                        request = json.loads("|".join(parts[2:]))
                    except ValueError:
                        request = None
        elif kind == "out" and isinstance(data, str) and data.startswith("/choose") and request is not None:
            req, request = request, None            # one order per request (a refused order is asked again with a new one)
            act = (req.get("active") or [{}])[0] or {}
            tera_type = act.get("canTerastallize")
            if not tera_type or req.get("forceSwitch") or role is None:
                continue
            own = next((p for p in (req.get("side") or {}).get("pokemon") or [] if p.get("active")), None)
            other = active.get("p2" if role == "p1" else "p1")
            if own is None or other is None:
                continue
            species = str(own.get("details") or "").split(",")[0]
            own_types = dex.types(species)
            their_types = [dex.Type.from_name(other["tera"])] if other.get("tera") and other["tera"] != "Stellar" else dex.types(other["species"])
            before = dex.worst(their_types, own_types)
            after = before if tera_type == "Stellar" else dex.worst(their_types, [dex.Type.from_name(tera_type)])
            if before is None or after is None:
                continue
            words = data.split("|")[0].split()
            chosen = "terastallize" in words
            move_id = words[2] if len(words) > 2 and words[1] == "move" else ""
            move = dex.move(move_id)
            cond = str(own.get("condition") or "")
            try:
                hp = float(cond.split()[0].split("/")[0]) / float(cond.split()[0].split("/")[1])
            except (ValueError, IndexError, ZeroDivisionError):
                hp = None
            move_ids = [m.get("id") for m in act.get("moves") or []]
            has_attack = any((dex.move(m).get("type") or "").lower() == tera_type.lower() and dex.move(m).get("category") != "Status"
                             for m in move_ids if m)
            rows.append({
                "turn": turn, "species": species, "tera_type": tera_type, "opp": other["species"], "before": before, "after": after,
                "chosen": chosen, "move": move_id, "is_move": bool(move_id),
                "boosted": bool(chosen and move and (move.get("type") or "").lower() == tera_type.lower() and move.get("category") != "Status"),
                "terablast": bool(chosen and to_id(move_id) == "terablast"),
                "status": bool(chosen and move and move.get("category") == "Status"),
                "has_attack_of_tera_type": has_attack, "hp": hp,
            })
    if arm is None:
        return None
    return {"arm": arm, "role": role, "won": (result or {}).get("won"), "rows": rows, "finished": result is not None}


def kind_of(r: Dict[str, Any]) -> str:
    return "better" if r["after"] < r["before"] else ("worse" if r["after"] > r["before"] else "same")


def summarise(rows: List[Dict[str, Any]], games: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"games": games, "offered_turns": len(rows)}
    took = [r for r in rows if r["chosen"]]
    out["tera_orders"] = len(took)
    out["rate_per_offered_turn"] = round(len(took) / len(rows), 4) if rows else None
    for name in ("better", "same", "worse"):
        sub = [r for r in rows if kind_of(r) == name]
        k = sum(1 for r in sub if r["chosen"])
        out["defence_" + name] = {"turns": len(sub), "share_of_turns": round(len(sub) / len(rows), 4) if rows else None,
                                  "tera": k, "rate": round(k / len(sub), 4) if sub else None}
    weak = [r for r in rows if r["before"] > 1.0]
    removes = [r for r in weak if r["after"] <= 1.0]
    leaves = [r for r in weak if r["after"] > 1.0]
    out["weak_to_its_types"] = {
        "turns": len(weak),
        "tera_removes_the_weakness": {"turns": len(removes), "tera": sum(1 for r in removes if r["chosen"]),
                                      "rate": round(sum(1 for r in removes if r["chosen"]) / len(removes), 4) if removes else None},
        "tera_leaves_it": {"turns": len(leaves), "tera": sum(1 for r in leaves if r["chosen"]),
                           "rate": round(sum(1 for r in leaves if r["chosen"]) / len(leaves), 4) if leaves else None},
    }
    with_attack = [r for r in rows if r["has_attack_of_tera_type"]]
    without = [r for r in rows if not r["has_attack_of_tera_type"]]
    out["has_attack_of_tera_type"] = {"turns": len(with_attack), "tera": sum(1 for r in with_attack if r["chosen"]),
                                      "rate": round(sum(1 for r in with_attack if r["chosen"]) / len(with_attack), 4) if with_attack else None}
    out["no_attack_of_tera_type"] = {"turns": len(without), "tera": sum(1 for r in without if r["chosen"]),
                                     "rate": round(sum(1 for r in without if r["chosen"]) / len(without), 4) if without else None}
    if took:
        n = len(took)
        better = sum(1 for r in took if kind_of(r) == "better")
        worse = sum(1 for r in took if kind_of(r) == "worse")
        boosted = sum(1 for r in took if r["boosted"])
        hps = [r["hp"] for r in took if r["hp"] is not None]
        out["what_the_tera_did"] = {
            "n": n, "defence_better": better, "defence_same": n - better - worse, "defence_worse": worse,
            "move_boosted": boosted, "tera_blast": sum(1 for r in took if r["terablast"]),
            "status_move": sum(1 for r in took if r["status"]),
            "boosted_or_defence_better": sum(1 for r in took if r["boosted"] or kind_of(r) == "better"),
            "neither": sum(1 for r in took if not r["boosted"] and kind_of(r) == "same"),
            "defence_worse_and_move_not_boosted": sum(1 for r in took if not r["boosted"] and kind_of(r) == "worse"),
            "worse_not_boosted_not_tera_blast": sum(1 for r in took if not r["boosted"] and not r["terablast"] and kind_of(r) == "worse"),
            "hp_mean": round(statistics.fmean(hps), 3) if hps else None,
            "hp_over_two_thirds": sum(1 for h in hps if h > 2 / 3),
            "turn_median": statistics.median([r["turn"] for r in took]),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="run folders (with battles/ inside)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    dex = Dex()
    result: Dict[str, Any] = {}
    for folder in args.folders:
        by_arm: Dict[str, Dict[str, Any]] = {}
        unread = 0
        for path in sorted(glob.glob(os.path.join(folder, "battles", "*.jsonl"))):
            battle = read_battle(path, dex)
            if battle is None or not battle["finished"]:
                unread += 1
                continue
            slot = by_arm.setdefault(battle["arm"], {"games": 0, "rows": []})
            slot["games"] += 1
            slot["rows"].extend(battle["rows"])
        name = folder.replace("\\", "/").rstrip("/")
        result[name] = {arm: summarise(v["rows"], v["games"]) for arm, v in sorted(by_arm.items())}
        result[name]["_files_without_a_result"] = unread
        print(f"\n== {name}" + (f"   ({unread} battle files without a result line, left out)" if unread else ""))
        for arm, s in result[name].items():
            if arm.startswith("_"):
                continue
            print(f"  {arm}: {s['games']} games, Tera offered on {s['offered_turns']} turns, played on {s['tera_orders']} ({s['rate_per_offered_turn']})")
            for k in ("defence_better", "defence_same", "defence_worse"):
                print(f"     turns where Tera makes the {k.replace('_', ' ')}: {s[k]}")
            print(f"     weak to its types: {s['weak_to_its_types']}")
            print(f"     with an attack of the Tera type: {s['has_attack_of_tera_type']}; without: {s['no_attack_of_tera_type']}")
            print(f"     what the Tera did: {s.get('what_the_tera_did')}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
