"""
The search asked again on every decision the counter rules changed (round 3 of the set-up work):
is the search's value for the rule's action clearly worse than for mix's? bench/ladder_rerun.py's
way (branch setupfix, round 1's rerun.py): each recorded battle is fed to its own Foul Play helper
(bench/fp_bridge_server.py) up to the decision, and the search is asked there, as many times as
--repeat, at --ms. Read only: nothing is sent anywhere.

    python bench/counter_replay.py --scan scan.json --json replay.json data/battles/*.jsonl
    python bench/counter_rerun.py --replay replay.json --ms 500 --repeat 2 --parallel 6 --json rerun.json

--replay is counter_replay.py's --json (each change has its rqid and both actions as poke-env
orders them). The search's answer has, for every action of ours, its share of the visits and
its score, the mean playout result from our side (0 to 1, a win chance): the rule's action and
mix's are compared on that score, within each repeat (the repeats sample other sets of the
opponent, so their levels differ). Verdict per decision, set before the runs: "clearly worse"
when the rule's action scores 5 points or more below mix's in every repeat, "worse" 2 points or
more in every repeat, "better" 2 points or more above in every repeat, else "about even".
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tid(s: Any) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def slot_choice(text: str, req: Optional[Dict[str, Any]]) -> str:
    """poke-env's "move nastyplot [terastallize]" / "switch Golduck" as Showdown slots ("move 2", "switch 3"), as ladder_rerun.py."""
    parts = str(text or "").split()
    if not parts or req is None:
        return text
    if parts[0] == "move" and len(parts) > 1:
        moves = [m["id"] for m in ((req.get("active") or [{}])[0].get("moves") or [])]
        if parts[1] in moves:
            return " ".join(["move", str(moves.index(parts[1]) + 1)] + parts[2:])
    if parts[0] == "switch":
        name = " ".join(parts[1:])
        for i, p in enumerate(req["side"]["pokemon"]):
            species = str(p.get("details") or "").split(",")[0]
            if p["ident"][4:] == name or tid(p["ident"][4:]) == tid(name) or tid(species) == tid(name):
                return "switch %d" % (i + 1)
    return text


def rerun_battle(path: str, asks: Dict[int, Dict[str, Any]], ms: int, repeat: int, code: str, fp: str, log: str,
                 live_effort: bool = False) -> Dict[int, Any]:
    """Feeds one battle to a fresh helper and asks the search `repeat` times at each rqid of asks: rqid -> answers.
    live_effort: each world is searched for as many iterations as the live search made per world at that decision."""
    env = dict(os.environ, PYTHONHASHSEED="0")
    out: Dict[int, Any] = {}
    with open(log, "w") as err:
        proc = subprocess.Popen([os.path.join(fp, ".venv/bin/python"), "-u", os.path.join(code, "bench/fp_bridge_server.py"),
                                 "--foulplay-dir", fp], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, env=env)
        nid = [0]

        def send(m):
            proc.stdin.write(json.dumps(m) + "\n")
            proc.stdin.flush()

        def ask(m):
            nid[0] += 1
            m = dict(m, id=nid[0])
            send(m)
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("helper gone (see %s)" % log)
                r = json.loads(line)
                if r.get("id") == m["id"]:
                    return r

        try:
            ask({"op": "ping"})
            rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
            uid = next(d["data"]["user"] for d in rows if d["dir"] == "note" and isinstance(d["data"], dict) and "user" in d["data"])
            room = next(d["data"]["battle"] for d in rows if d["dir"] == "note" and isinstance(d["data"], dict) and "battle" in d["data"])
            user = uid
            for d in rows:
                if d["dir"] == "in" and isinstance(d["data"], str):
                    for x in d["data"].split("\n"):
                        p = x.split("|")
                        if len(p) > 3 and p[1] == "player" and tid(p[3]) == tid(uid):
                            user = p[3]
            key = user + "|" + room
            last_req: Optional[Dict[str, Any]] = None
            done = set()
            for d in rows:
                if d["dir"] == "out" and str(d["data"]).startswith("/choose "):
                    send({"op": "chose", "key": key, "choice": slot_choice(d["data"][8:], last_req)})
                    continue
                if d["dir"] == "in" and isinstance(d["data"], str):
                    send({"op": "feed", "key": key, "user": user, "room": room, "data": d["data"]})
                    for x in d["data"].split("\n"):
                        if x.startswith("|request|") and len(x) > 10:
                            r = json.loads(x[9:])
                            if not r.get("wait"):
                                last_req = r
                    continue
                # Ask where the live bot asked: at its decision note (after the request and the turn's log were in).
                if d["dir"] != "note" or not isinstance(d["data"], dict) or "decision" not in d["data"]:
                    continue
                rqid = d["data"]["decision"].get("rqid")
                if rqid is None or int(rqid) not in asks or int(rqid) in done:
                    continue
                done.add(int(rqid))
                want = asks[int(rqid)]
                slots = {"mix": slot_choice(want["order_mix"], last_req), "pick": slot_choice(want["order_pick"], last_req)}
                answers = []
                question = {"op": "decide", "key": key, "ms": ms, "rqid": rqid}
                live = d["data"]["decision"]
                if live_effort and live.get("iterations") and live.get("worlds"):
                    question["iterations"] = max(1, int(round(float(live["iterations"]) / max(1, int(live["worlds"])))))
                for _ in range(repeat):
                    r = ask(question)
                    if not r.get("ok"):
                        answers.append({"error": r.get("error")})
                        continue
                    opts = {o.get("choice"): o for o in (r.get("options") or [])}
                    total = sum(int(o.get("visits") or 0) for o in opts.values()) or 1
                    row = {"best": r.get("best"), "iterations": r.get("iterations"), "worlds": r.get("worlds"),
                           "live_iterations": live.get("iterations"), "live_worlds": live.get("worlds")}
                    for side, slot in slots.items():
                        o = opts.get(slot)
                        row[side] = None if o is None else {"fp": o.get("fp"), "share": round(float(o.get("policy") or 0.0), 4),
                                                            "visits": int(o.get("visits") or 0), "score": o.get("score")}
                    ranked = sorted(opts.values(), key=lambda o: -float(o.get("policy") or 0.0))
                    row["top"] = [[o.get("fp"), round(float(o.get("policy") or 0.0), 3), o.get("score")] for o in ranked[:5]]
                    row["visits_total"] = total
                    opp = (r.get("opponent") or {}).get("options") or []
                    row["opp"] = [[o.get("name"), round(float(o.get("share") or 0.0), 3)] for o in opp[:4]]
                    answers.append(row)
                out[int(rqid)] = {"slots": slots, "answers": answers}
            send({"op": "end", "key": key})
            send({"op": "exit"})
            proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
    return out


def verdict(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The score gaps (rule minus mix, in points of win chance) per repeat, and the verdict of the module's docstring."""
    gaps: List[Optional[float]] = []
    for a in answers:
        if "error" in a or a.get("mix") is None or a.get("pick") is None:
            gaps.append(None)
            continue
        s_mix, s_pick = a["mix"].get("score"), a["pick"].get("score")
        gaps.append(None if s_mix is None or s_pick is None else round(100.0 * (float(s_pick) - float(s_mix)), 1))
    known = [g for g in gaps if g is not None]
    if not known or len(known) < len(gaps):
        word = "no answer"
    elif all(g <= -5 for g in known):
        word = "clearly worse"
    elif all(g <= -2 for g in known):
        word = "worse"
    elif all(g >= 2 for g in known):
        word = "better"
    else:
        word = "about even"
    return {"gaps": gaps, "verdict": word}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True, help="counter_replay.py --json")
    ap.add_argument("--category", default="", help="only the games of this scan category (default: every change)")
    ap.add_argument("--ms", type=int, default=500)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=6, help="helpers at once (one battle each, one core each)")
    ap.add_argument("--live-effort", action="store_true",
                    help="search each world for the live search's iterations per world at that decision instead of --ms (the box's"
                         " cores make about a third of the laptop's iterations in 500 ms)")
    ap.add_argument("--code", default=REPO, help="a checkout with bench/fp_bridge_server.py (the ladder's is d7ace06's)")
    ap.add_argument("--fp", default=os.path.expanduser("~/foul-play"))
    ap.add_argument("--logs", default="rerun_logs")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    games = json.load(open(a.replay, encoding="utf-8"))["games"]
    os.makedirs(a.logs, exist_ok=True)
    jobs = {}
    for g in games:
        if a.category and g["category"] != a.category:
            continue
        if g["changes"]:
            jobs[g["path"]] = (g, {int(c["rqid"]): c for c in g["changes"]})
    results: Dict[str, Dict[int, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.parallel)) as pool:
        futures = {pool.submit(rerun_battle, path, asks, a.ms, a.repeat, a.code, a.fp,
                               os.path.join(a.logs, os.path.basename(path) + ".log"), a.live_effort): path
                   for path, (_g, asks) in jobs.items()}
        for f in concurrent.futures.as_completed(futures):
            path = futures[f]
            try:
                results[path] = f.result()
            except Exception as exc:
                results[path] = {}
                print("%s: %r" % (os.path.basename(path), exc), file=sys.stderr)
    rows = []
    for path, (g, asks) in jobs.items():
        got = results.get(path) or {}
        for rqid, c in sorted(asks.items(), key=lambda kv: kv[1]["turn"]):
            r = got.get(rqid) or {"answers": []}
            v = verdict(r["answers"]) if r["answers"] else {"gaps": [], "verdict": "not asked"}
            rows.append({"battle": g["battle"], "won": g["won"], "category": g["category"], "turn": c["turn"], "rqid": rqid,
                         "first": c.get("first"), "kind": c["kind"], "mix": c["mix"], "pick": c["pick"], "net": c["net"],
                         "search_live": c["search"], "share_live": [c.get("share_mix"), c.get("share_pick")], "slots": r.get("slots"),
                         "answers": r["answers"], **v})
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"ms": a.ms, "repeat": a.repeat, "live_effort": a.live_effort, "rows": rows}, fh, indent=1)
    import re
    for r in rows:
        num = (re.findall(r"-(\d+)", r["battle"]) or ["?"])[0][-4:]
        per = []
        for ans in r["answers"]:
            if "error" in ans:
                per.append("error: %s" % str(ans["error"])[:80])
                continue
            m, p = ans.get("mix") or {}, ans.get("pick") or {}
            per.append("mix %s %.2f/%s, rule %s %.2f/%s, best %s" % (m.get("fp"), m.get("share") or 0, m.get("score"), p.get("fp"),
                                                                    p.get("share") or 0, p.get("score"), ans.get("best")))
        print("%s %-4s %-26s T%-3d %s mix %-22s -> rule %-22s gaps %-14s %-13s | %s" % (
            num, "WON" if r["won"] else "LOST", r["category"][:26], r["turn"], "1st" if r["first"] else "   ", r["mix"], r["pick"],
            ",".join("?" if x is None else "%+.1f" % x for x in r["gaps"]), r["verdict"], " || ".join(per)))
    counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        grp = "sweep loss" if r["category"] == "loss: set-up sweep" else ("win" if str(r["category"]).startswith("win") else "other loss")
        counts.setdefault(grp, {})
        counts[grp][r["verdict"]] = counts[grp].get(r["verdict"], 0) + 1
    print()
    for grp, c in counts.items():
        print("%s: %s" % (grp, ", ".join("%s %d" % kv for kv in sorted(c.items(), key=lambda kv: -kv[1]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
