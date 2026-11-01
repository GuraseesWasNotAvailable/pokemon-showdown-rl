"""
Makes tests/fixtures/luna_ladder_games.jsonl.gz: real decisions of logged ladder games, for the
tests of the Luna arm's brief (tests/test_luna.py: under the token budget, nothing hidden in it).

    python bench/luna_fixture.py --battles C:/Users/you/ps-ppo-predict/runs/ladder/battles --games 10

From ladder.py's battle files (runs/ladder/battles/<battle tag>.jsonl) of games played by "mix" it
keeps, per game, what the battle room sent our player ("in": exactly what our side saw, the
requests included) and the decision notes (the network's probabilities, the search's shares, the
read of the opponent, the win estimate), in the order they came. Nothing a person typed is kept:
chat, joins, the title, the rating box and the timer's lines are dropped, and both user names are
replaced ("Player" for ours, "Opponent" for theirs), also inside the requests.

replaced ("Player" for ours, "Opponent" for theirs), also inside the requests; a battle's tag is
kept without the key of its room.

One JSON object per line: {"t": "game", "battle", "arm", "role"}, then {"t": "in", "data"} and
{"t": "decision", ...} in order.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "fixtures", "luna_ladder_games.jsonl.gz")
DROP = frozenset(("c", "c:", "chat", "j", "J", "join", "l", "L", "leave", "n", "N", "name", "raw", "html", "uhtml", "uhtmlchange",
                  "badge", "title", "inactive", "inactiveoff", "unlink", "notify", "tempnotify", "tempnotifyoff", "hidelines",
                  "bigerror", "debug", "error", "deinit", "noinit", "sentchoice"))
KEEP = ("turn", "rqid", "mode", "legal", "mask", "forced", "net", "search", "played", "source", "asked", "reason", "probs",
        "win_est", "iterations", "worlds", "shares", "score_net", "score_search", "score_played", "opponent_active", "opponent")


def read_game(path: str) -> Optional[Dict[str, Any]]:
    head: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    names: Dict[str, str] = {}
    me = ""
    decisions = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            data = e.get("data")
            if e.get("dir") == "note" and isinstance(data, dict):
                if "battle" in data and "arm" in data and not head:
                    head = data
                    me = str(data.get("user") or "")
                elif isinstance(data.get("decision"), dict):
                    d = data["decision"]
                    row = {"t": "decision"}
                    row.update({k: d[k] for k in KEEP if k in d})
                    pred = d.get("predict") or {}
                    if pred.get("top"):
                        row["predict"] = {"top": pred.get("top"), "p_switch": pred.get("p_switch"), "p_tera": pred.get("p_tera")}
                    rows.append(row)
                    decisions += int((d.get("legal") or 0) > 1)
            elif e.get("dir") == "in" and isinstance(data, str):
                for part in data.split("\n"):
                    bits = part.split("|")
                    if len(bits) > 3 and bits[1] == "player" and bits[3]:
                        names[bits[2]] = bits[3]
                rows.append({"t": "in", "data": data})
    if not head or not decisions or len(names) < 2:
        return None
    to_id = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # noqa: E731
    role = next((r for r, n in names.items() if to_id(n) == to_id(me)), None)
    if role is None:
        return None
    swap = {names[role]: "Player", names["p2" if role == "p1" else "p1"]: "Opponent"}

    def clean(data: str) -> str:
        kept = []
        for part in data.split("\n"):
            bits = part.split("|")
            if len(bits) > 1 and bits[1] in DROP:
                continue
            if len(bits) > 1 and bits[1] == "player" and len(bits) > 3:
                bits = bits[:3] + [swap.get(bits[3], bits[3]), "1", ""]
                part = "|".join(bits)
            for name, new in swap.items():
                if name and name in part:
                    part = part.replace(name, new)
            kept.append(part)
        return "\n".join(kept)

    out_rows = []
    for row in rows:
        if row["t"] == "in":
            data = clean(row["data"])
            if data.strip("\n|") or "|request|" in data:
                out_rows.append({"t": "in", "data": data})
        else:
            out_rows.append(row)
    text = json.dumps(out_rows)
    for name in swap:
        if name and name in text:
            return None                     # a name survived somewhere: the game is left out
    # A battle's tag without the key of its room (a hidden room's name ends in one).
    tag = re.sub(r"^(battle-[a-z0-9]+-\d+).*$", r"\1", str(head["battle"]))
    return {"head": {"t": "game", "battle": tag, "arm": head.get("arm"), "role": role}, "rows": out_rows,
            "decisions": decisions}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Make the fixture of real ladder decisions for the Luna arm's tests.")
    ap.add_argument("--battles", required=True, help="a folder of ladder.py's battle files")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--arm", default="mix")
    ap.add_argument("--checkpoint", default="learner_update_024381.pt")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    games = []
    for path in sorted(glob.glob(os.path.join(args.battles, "battle-*.jsonl"))):
        if os.path.getsize(path) > 400_000:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                first = json.loads(fh.readline())
        except (OSError, ValueError):
            continue
        note = first.get("data") if isinstance(first.get("data"), dict) else {}
        if note.get("arm") != args.arm or note.get("checkpoint") != args.checkpoint or note.get("server") != "public":
            continue
        game = read_game(path)
        if game is not None and game["decisions"] >= 15:
            games.append(game)
    if not games:
        print("no games of arm %s on %s in %s" % (args.arm, args.checkpoint, args.battles))
        return 1
    step = max(1, len(games) // args.games)
    chosen = games[::step][:args.games]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for game in chosen:
            fh.write(json.dumps(game["head"]) + "\n")
            for row in game["rows"]:
                fh.write(json.dumps(row) + "\n")
    print("%d of %d games, %d decisions with a choice -> %s (%d bytes)" % (
        len(chosen), len(games), sum(g["decisions"] for g in chosen), args.out, os.path.getsize(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
