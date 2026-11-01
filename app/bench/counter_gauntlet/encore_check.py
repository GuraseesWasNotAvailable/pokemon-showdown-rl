"""
    python bench/counter_gauntlet/encore_check.py ~/dev/counter-gauntlet/g/su ~/dev/counter-gauntlet/g/fp ~/dev/counter-gauntlet/g/nt

For every Encore that mix-counter's rule played (decided "played the counter", counter "encore ..."): what the battle log
shows on that turn: did Encore start, fail, or not happen (we were asleep, fainted, or switched), and which move it locked in
(the opponent's last move before our Encore); and whether that move is a set-up move (setup_moves.py)."""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repo, for setup_moves
import setup_moves as sm
def to_id(s): return "".join(c for c in str(s).lower() if c.isalnum())
out = collections.Counter(); rows = []
for ph in sys.argv[1:]:
    res = {json.loads(l)["battle"]: json.loads(l) for l in open(os.path.join(ph, "results.jsonl")) if '"battle"' in l}
    for l in open(os.path.join(ph, "decisions.jsonl")):
        d = json.loads(l); r = d.get("counter")
        if not isinstance(r, dict): continue
        cc = r.get("counter") or {}
        if cc.get("decided") != "played the counter" or "encore" not in str(cc.get("counter", "")): continue
        b, turn = d["battle"], d["turn"]; g = res.get(b) or {}
        me = g.get("role"); them = "p2" if me == "p1" else "p1"
        path = os.path.join(ph, "battles", b + ".jsonl")
        lines = []
        for x in open(path):
            data = json.loads(x).get("data")
            if isinstance(data, str): lines.extend(data.split("\n"))
        cur = None; last_theirs = None; result = "no Encore in the log"; locked = None
        for line in lines:
            p = line.split("|")
            if len(p) > 2 and p[1] == "turn": cur = int(p[2]); continue
            if cur is None: continue
            if len(p) > 3 and p[1] == "move" and p[2].startswith(them): last_theirs = p[3]
            if cur == turn:
                if len(p) > 3 and p[1] == "move" and p[2].startswith(me) and to_id(p[3]) == "encore":
                    result = "Encore used"
                    if "[still]" in line or "[miss]" in line: result = "Encore missed"
                if len(p) > 3 and p[1] == "-start" and p[2].startswith(them) and "Encore" in p[3]: result = "Encore started"; locked = last_theirs
                if len(p) > 2 and p[1] == "-fail" and p[2].startswith(me) and result == "Encore used": result = "Encore failed"
                if len(p) > 2 and p[1] == "cant" and p[2].startswith(me): result = "we could not move (" + (p[3] if len(p) > 3 else "") + ")"
            if cur is not None and cur > turn: break
        key = result + ((": locked " + ("a set-up move" if sm.is_setup(to_id(locked)) else "an attack or other move")) if locked else "")
        out[key] += 1
        rows.append((os.path.basename(ph), b, turn, g.get("won"), result, locked))
print(dict(out.most_common()))
for r in rows: print(r)
