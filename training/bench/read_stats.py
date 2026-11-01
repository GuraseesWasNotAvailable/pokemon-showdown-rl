"""
bench/read_stats.py: run the stage 2 read tracker (read_reward.py) over battle records and print what it
finds, so a person can check the reads by eye.

    python bench/read_stats.py rec/                     a folder of ladder records (*.jsonl / *.jsonl.gz)
    python bench/read_stats.py selfplay.jsonl           self-play logs from tests/selfplay_logs.js
    python bench/read_stats.py rec/ --examples 40 --seed 3

Ladder records (the app's battle-*.jsonl: "in" protocol chunks, "note" decisions) are one seat's view:
that seat is run with its real decisions, the other seat with turn-start decisions made up from the
protocol, and the two seats' nets must cancel. Self-play logs hold both seats' own views and choices.

Prints: games, decisions, voluntary switches, reads per game and per voluntary switch, the tier mix,
what was left out and why, the opponent-action labels, the zero-sum check, and random examples with the
turn's lines.
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from read_reward import (SIDES, other, protocol_decisions, read_ladder_record, track_steps)  # noqa: E402
from matchup import load_randbats_sets  # noqa: E402


def selfplay_steps(entries, drop_turn_lines=False):
    """Steps for one seat of a self-play game (tests/selfplay_logs.js)."""
    steps = []
    for e in entries:
        if e[0] == "in":
            lines = e[1].split("\n")
            if drop_turn_lines:
                lines = [ln for ln in lines if not ln.startswith(("|turn|", "|request|", "|t:|"))]
            steps.append(("lines", lines))
        elif e[0] == "choice":
            steps.append(("decide", e[1], e[2]))
    return steps


def iter_games(paths):
    """Yields (name, {seat: steps}, our_seat) for every game in the given files and folders."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.jsonl")) + glob.glob(os.path.join(p, "*.jsonl.gz")))
        else:
            files.append(p)
    for f in files:
        opener = (lambda x: __import__("gzip").open(x, "rt", encoding="utf-8")) if f.endswith(".gz") else (
            lambda x: open(x, encoding="utf-8"))
        with opener(f) as fh:
            first = fh.readline()
        try:
            head = json.loads(first)
        except ValueError:
            continue
        if isinstance(head, dict) and "p1" in head and "p2" in head:
            with opener(f) as fh:
                for raw in fh:
                    if raw.strip():
                        g = json.loads(raw)
                        yield (f"{os.path.basename(f)}#{g.get('game')}",
                               {s: selfplay_steps(g[s]) for s in SIDES}, None)
        else:
            seat, steps = read_ladder_record(f)
            if seat not in SIDES:
                continue
            yield os.path.basename(f), {seat: steps, other(seat): protocol_decisions(steps, other(seat))}, seat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--examples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write every judged event to this jsonl file")
    args = ap.parse_args(argv)

    sets = load_randbats_sets()
    if not sets:
        print("warning: heuristic/cache/gen9randombattle.json not found; abilities from sets are not used")
    c = Counter()
    tiers = Counter()
    excluded = Counter()
    labels = Counter()
    events = []
    per_game_reads = []
    zero_sum_bad = []
    for name, seats, ours in iter_games(args.paths):
        trackers = {s: track_steps(st, s, keep_lines=True, sets=sets) for s, st in seats.items()}
        c["games"] += 1
        nets = {s: t.net() for s, t in trackers.items()}
        if abs(sum(nets.values())) > 1e-9:
            zero_sum_bad.append((name, nets))
        t = trackers[ours or "p1"]
        for r in t.regions:
            if not r.closed:
                continue
            for s in SIDES:
                if r.vswitch[s] is not None and not r.moved[s]:
                    c["voluntary switches"] += 1
                    if ours and s == other(ours):
                        c["opponent voluntary switches"] += 1
        game_reads = 0
        for ev in t.events:
            if ev.excluded:
                excluded[ev.excluded] += 1
                continue
            tiers[ev.tier] += 1
            if ev.tier > 0:
                game_reads += 1
                if ours and ev.reader == ours:
                    c["our reads"] += 1
                if ours and ev.reader == other(ours):
                    c["reads against us"] += 1
            events.append((name, ev))
        per_game_reads.append(game_reads)
        for s, tr in trackers.items():
            if ours and s != ours:
                continue
            for d in tr.labels():
                c["decisions"] += 1
                c["forced decisions"] += d.forced
                labels[d.opp_action] += 1
            c["action mismatches"] += tr.action_mismatch

    n = max(1, c["games"])
    reads = sum(v for k, v in tiers.items() if k > 0)
    vs = max(1, c["voluntary switches"])
    print(f"games {c['games']}, decisions {c['decisions']} ({c['forced decisions']} forced)")
    print(f"voluntary switches {c['voluntary switches']} ({c['voluntary switches'] / n:.2f} per game)"
          + (f", by the opponent {c['opponent voluntary switches']}" if c["opponent voluntary switches"] else ""))
    print(f"hits super effective on a switch-in, judged: {sum(tiers.values())}; reads (tier > 0): {reads} "
          f"= {reads / n:.2f} per game, {reads / vs:.3f} per voluntary switch")
    if c["opponent voluntary switches"]:
        print(f"  ours: {c['our reads']} reads = {c['our reads'] / max(1, c['opponent voluntary switches']):.3f} "
              f"per opponent voluntary switch; against us: {c['reads against us']}")
    print("tier mix:", {f"{k:.1f}": v for k, v in sorted(tiers.items(), reverse=True)})
    print("left out:", dict(excluded) or "none")
    print("reads per game: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(per_game_reads).items())))
    names = {0: "moved/stayed", 7: "unseen", -1: "n/a"}
    print("opp_action labels:", {names.get(k, f"slot {k}"): v for k, v in sorted(labels.items())})
    print(f"zero-sum: {c['games'] - len(zero_sum_bad)}/{c['games']} games cancel exactly"
          + (f"; NOT: {zero_sum_bad[:5]}" if zero_sum_bad else ""))
    if c["action mismatches"]:
        print(f"decisions whose action was not a move though the protocol shows an attack: {c['action mismatches']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            for name, ev in events:
                d = dict(ev.__dict__)
                d["game"] = name
                f.write(json.dumps(d) + "\n")

    rng = random.Random(args.seed)
    picks = [e for e in events if e[1].tier > 0]
    rng.shuffle(picks)
    print(f"\n{min(args.examples, len(picks))} random reads:")
    for name, ev in picks[:args.examples]:
        print(f"\n--- {name} turn {ev.turn}: {ev.reader} read {ev.switcher}'s switch "
              f"{ev.out_species} -> {ev.in_species} with {ev.move} ({ev.move_type}); on the outgoing "
              f"Pokemon it was {ev.label}: tier {ev.tier}")
        for ln in (ev.lines or []):
            print("   " + ln)


if __name__ == "__main__":
    main()
