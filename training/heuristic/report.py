"""
Summarises the games the browser bot has recorded.

    python heuristic/report.py <battles folder> [--since ISO_TIME] [--last N] [--losses N]

Prints the record and rating, how each bot build has done, and where the games are being lost,
so a review can start from numbers instead of from reading every log.
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import os
import re

SETUP = {
    "swordsdance", "nastyplot", "calmmind", "dragondance", "quiverdance", "shellsmash", "bulkup", "coil", "curse",
    "bellydrum", "agility", "rockpolish", "shiftgear", "tailglow", "irondefense", "cosmicpower", "victorydance",
    "workup", "growth", "tidyup", "autotomize", "acidarmor", "amnesia", "honeclaws", "noretreat", "clangoroussoul",
    "filletaway",
}
RATING = re.compile(r"\|raw\|(.*?)'s rating: (\d+) (?:&rarr;|→) <strong>(\d+)</strong>")


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load(folder: str, since: str | None):
    games = []
    for f in glob.glob(os.path.join(folder, "**", "*.json"), recursive=True):
        try:
            r = json.load(io.open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(r, dict) or "decisions" not in r or "log" not in r:
            continue
        if since and r.get("endedAt", "") < since:
            continue
        r["_file"] = f
        games.append(r)
    games.sort(key=lambda r: r.get("endedAt", ""))
    return games


def opp_rating(r):
    """The opponent's rating before the game: recorded directly by newer builds, otherwise from the log."""
    if isinstance(r.get("ratings"), dict) and r["ratings"].get("theirs"):
        return r["ratings"]["theirs"]
    if isinstance(r.get("opponent"), dict) and r["opponent"].get("elo"):
        return r["opponent"]["elo"].get("before")
    me = to_id(r["me"]["name"])
    for ln in r["log"]:
        m = RATING.search(ln)
        if m and to_id(re.sub(r"&[#a-z0-9]+;", "", m.group(1))) != me:
            return int(m.group(2))
    return None


def perf(games):
    """Performance rating: the rating at which this record against these opponents is par."""
    rated = [(opp_rating(g), g["result"]) for g in games]
    rated = [(o, res) for o, res in rated if o is not None and res in ("win", "loss")]
    if not rated:
        return None
    n = len(rated)
    w = sum(1 for _, res in rated if res == "win")
    avg = sum(o for o, _ in rated) / n
    p = min(max(w / n, 0.01), 0.99)
    import math
    return round(avg + 400 * math.log10(p / (1 - p)))


def side_of(r):
    me = r["me"]["side"]
    return me, ("p1" if me == "p2" else "p2")


def first_actions(events, me, opp):
    mine = next((e for e in events if e.startswith(me + " ") and (" used " in e or " sent in " in e)), "")
    theirs = next((e for e in events if e.startswith(opp + " ") and (" used " in e or " sent in " in e)), "")
    return mine, theirs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--since", default=None, help="only games that ended at or after this ISO time")
    ap.add_argument("--last", type=int, default=0, help="only the last N games")
    ap.add_argument("--losses", type=int, default=8, help="how many recent losses to list")
    args = ap.parse_args()

    games = load(args.folder, args.since)
    if args.last:
        games = games[-args.last:]
    if not games:
        print("no recorded games")
        return

    n = len(games)
    wins = sum(1 for g in games if g["result"] == "win")
    losses = sum(1 for g in games if g["result"] == "loss")
    elos = [g["elo"] for g in games if g.get("elo")]
    print(f"GAMES {n}: {wins} won, {losses} lost ({100 * wins / max(wins + losses, 1):.1f}%)  "
          f"{games[0]['endedAt'][:16]} to {games[-1]['endedAt'][:16]}")
    if elos:
        afters = [e["after"] for e in elos]
        print(f"RATING {elos[0]['before']} -> {afters[-1]}  (low {min(afters)}, high {max(afters)}, last 20 avg {sum(afters[-20:]) / len(afters[-20:]):.0f})"
              f"  performance {perf(games)}")

    # ---- by build
    print("\nBY BUILD (version/build: games, win rate, rating at end, performance)")
    by = collections.OrderedDict()
    for g in games:
        by.setdefault(f"{g.get('bot', '?')}/{g.get('build', '-')}", []).append(g)
    for k, gs in by.items():
        w = sum(1 for g in gs if g["result"] == "win")
        l = sum(1 for g in gs if g["result"] == "loss")
        last = next((g["elo"]["after"] for g in reversed(gs) if g.get("elo")), None)
        print(f"  {k:<28} {len(gs):>4} games  {100 * w / max(w + l, 1):5.1f}%  rating {last}  performance {perf(gs)}")

    # ---- by arm: live games split at random between the new scoring and the old
    arms = collections.OrderedDict()
    for g in games:
        if g.get("arm"):
            arms.setdefault(f"{g.get('arms', '')} {g['arm']}", []).append(g)
    if arms:
        print("\nBY ARM (each game assigned by its id: games, win rate with 95% interval, performance)")
        for k in sorted(arms):
            gs = arms[k]
            w = sum(1 for g in gs if g["result"] == "win")
            l = sum(1 for g in gs if g["result"] == "loss")
            n2 = max(w + l, 1)
            ph = w / n2
            z = 1.96
            mid = (ph + z * z / (2 * n2)) / (1 + z * z / n2)
            half = z * ((ph * (1 - ph) / n2 + z * z / (4 * n2 * n2)) ** 0.5) / (1 + z * z / n2)
            print(f"  {k:<28} {len(gs):>4} games  {100 * ph:5.1f}%  [{100 * (mid - half):.0f}, {100 * (mid + half):.0f}]  performance {perf(gs)}")

    # ---- by hour
    print("\nBY HOUR (games, win rate, rating at end)")
    hours = collections.OrderedDict()
    for g in games:
        hours.setdefault(g["endedAt"][:13], []).append(g)
    for k, gs in list(hours.items())[-12:]:
        w = sum(1 for g in gs if g["result"] == "win")
        l = sum(1 for g in gs if g["result"] == "loss")
        last = next((g["elo"]["after"] for g in reversed(gs) if g.get("elo")), None)
        print(f"  {k}  {len(gs):>3}  {100 * w / max(w + l, 1):5.1f}%  {last}")

    # ---- what the bot did
    tera = collections.Counter()
    setup_used = collections.Counter()
    setup_turns = 0
    switches = collections.Counter()
    predict = collections.Counter()
    fallback = 0
    decisions = 0
    lost_lead = 0
    for g in games:
        me, opp = side_of(g)
        hist = {h["turn"]: h["events"] for h in g["history"]}
        team = {p["details"].split(",")[0]: p for p in (g.get("myTeam") or [])}
        res = g["result"]
        for d in g["decisions"]:
            decisions += 1
            if d["choice"] == "default":
                fallback += 1
            if d["forced"]:
                continue
            label = to_id(d["why"].split(" vs ")[0].split(" +")[0])
            if " terastallize" in d["choice"]:
                note = "secures the KO" if "secures the KO" in d["why"] else "survives the hit" if "survives the hit" in d["why"] else "wins the trade"
                tera[(note, res)] += 1
            active = next((m for m in d["mine"] if m["active"]), None)
            if active and set(team.get(active["species"], {}).get("moves", [])) & SETUP:
                setup_turns += 1
            if label in SETUP:
                setup_used[res] += 1
            if d["choice"].startswith("switch"):
                switches[res] += 1
            if d.get("predict"):
                _, theirs = first_actions(hist.get(d["turn"], []), me, opp)
                predict["made"] += 1
                if " sent in " in theirs:
                    predict["switched"] += 1
                    if d["predict"]["name"].split("-")[0] in theirs:
                        predict["right"] += 1
        # a game lost after being at least two Pokemon ahead
        if res == "loss":
            ahead = 0
            mine_f = opp_f = 0
            for h in g["history"]:
                for e in h["events"]:
                    if e.endswith(" fainted"):
                        if e.startswith(me + " "):
                            mine_f += 1
                        elif e.startswith(opp + " "):
                            opp_f += 1
                ahead = max(ahead, opp_f - mine_f)
            if ahead >= 2:
                lost_lead += 1

    gw, gl = max(wins, 1), max(losses, 1)
    ms = [d["ms"] for g in games for d in g["decisions"] if d.get("ms") is not None]
    rej = sum(g.get("rejections", 0) or 0 for g in games)
    evs = collections.Counter(e.get("type", "?") for g in games for e in (g.get("events") or []))
    print(f"\nDECISIONS {decisions}  (fallback 'default' choices: {fallback}, rejected by the server: {rej})")
    if ms:
        ms.sort()
        print(f"  time per decision: median {ms[len(ms) // 2]} ms, 99th percentile {ms[int(len(ms) * 0.99)]} ms, slowest {ms[-1]} ms")
    if evs:
        print("  in-game events: " + ", ".join(f"{k} x{v}" for k, v in evs.most_common()))
    print(f"  switches per game: {switches['win'] / gw:.1f} in wins, {switches['loss'] / gl:.1f} in losses")
    print(f"  setup moves used: {setup_used['win']} in wins, {setup_used['loss']} in losses, on {setup_turns} turns one was available")
    print("  Tera: " + ", ".join(f"{k[0]} ({k[1]}) x{v}" for k, v in sorted(tera.items())))
    if predict["made"]:
        print(f"  switch predictions: {predict['made']} made, opponent switched {100 * predict['switched'] / predict['made']:.0f}%, "
              f"named the right Pokemon in {predict['right']} of {predict['switched']}")
    print(f"  losses after being two or more Pokemon ahead: {lost_lead} of {losses}")

    # ---- who beats us
    killers = collections.Counter()
    seen = collections.Counter()
    for g in games:
        me, opp = side_of(g)
        last_opp_move_mon = ""
        for h in g["history"]:
            for e in h["events"]:
                if e.startswith(opp + " ") and " used " in e:
                    last_opp_move_mon = e.split(" used ")[0][3:]
                elif e.startswith(me + " ") and e.endswith(" fainted") and last_opp_move_mon:
                    killers[last_opp_move_mon] += 1
        for p in g["oppTeam"]:
            seen[p["species"]] += 1
    print("\nOPPOSING POKEMON THAT TOOK THE MOST KOs (KOs / games seen)")
    print("  " + ", ".join(f"{k} {v}/{seen.get(k, 0) or '?'}" for k, v in killers.most_common(14)))

    # ---- recent losses
    print(f"\nLAST {args.losses} LOSSES")
    for g in [g for g in games if g["result"] == "loss"][-args.losses:]:
        me, opp = side_of(g)
        mf = sum(1 for h in g["history"] for e in h["events"] if e.startswith(opp + " ") and e.endswith(" fainted"))
        print(f"  {g['endedAt'][5:16]}  {g['turns']:>2} turns  took {mf} of theirs  vs {opp_rating(g)}  {os.path.relpath(g['_file'], args.folder)}")


def health(root: str) -> None:
    """What the launcher wrote next to the exe: the live status and the event log."""
    st = os.path.join(root, "status.json")
    if os.path.exists(st):
        try:
            d = json.load(io.open(st, encoding="utf-8"))
            import datetime
            age = datetime.datetime.now(datetime.timezone.utc).timestamp() - d.get("ts", 0) / 1000
            print(f"\nSTATUS ({age:.0f}s old): {d.get('version')} {d.get('build')}  auto-queue {'on' if d.get('auto') else 'off'}  live {d.get('live')}  "
                  f"searching {d.get('searching')}  logged in {d.get('named')}  disconnected {d.get('disconnected')}")
            rec = d.get("record") or {}
            print(f"  this run: {rec.get('wins')}W {rec.get('losses')}L  rating {d.get('eloStart')} -> {d.get('elo')}  up {d.get('uptimeMin')} min  reloads {d.get('reloads')}  "
                  f"decisions {d.get('decisions')}  errors {d.get('errors')}  heap {d.get('heapMB')} MB  slowest decision {d.get('slowestMs')} ms")
        except Exception as e:
            print("\nSTATUS unreadable:", e)
    else:
        print("\nSTATUS: no status.json yet (the bot has not reported since PSBot.exe was updated)")
    ev = os.path.join(root, "logs", "events.jsonl")
    if os.path.exists(ev):
        lines = io.open(ev, encoding="utf-8").read().splitlines()[-400:]
        kinds = collections.Counter()
        rows = []
        for ln in lines:
            try:
                e = json.loads(ln)
            except Exception:
                continue
            kinds[e.get("type", "?")] += 1
            rows.append(e)
        print("EVENTS (last 400): " + ", ".join(f"{k} x{v}" for k, v in kinds.most_common()))
        for e in rows[-12:]:
            det = e.get("detail")
            print(f"  {str(e.get('t', ''))[5:19]}  {e.get('type')}: {det if isinstance(det, str) else json.dumps(det)}"[:220])


if __name__ == "__main__":
    main()
    import sys as _sys
    _folders = [a for a in _sys.argv[1:] if not a.startswith("--") and os.path.isdir(a)]
    if _folders:
        health(os.path.dirname(os.path.abspath(_folders[0])))
