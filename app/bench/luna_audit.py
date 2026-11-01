"""
Checks a run of the Luna arm from its files: nothing hidden in the briefs, the token budget, no
late or illegal orders, the rails' counts, and that a secret (the API key) is in none of them.

    python bench/luna_audit.py --run runs/luna-smoke                (the folder with results.jsonl and battles/)
    FAKE_LUNA_KEY=sk-fake-... python bench/luna_audit.py --run runs/luna-smoke --secret-env FAKE_LUNA_KEY

What it reads: every battle file's lines in order (ladder.BattleRecorder: "in" what the room sent
our player, "out" what we sent, "note" the decision records). For every decision note with a
"luna" record:

  * hidden information (hidden_problems): the brief and the choices may name only what our side
    had been shown when the question was asked. Checked: no opponent Pokemon that had not come
    out yet (by name; our own team's species apart), every move listed as "seen" was used by the
    opponent before, a stated item or ability of the opponent's active Pokemon was shown in the
    battle (or is the only ability its species can have), and no user name of the battle is in
    the text. Guesses the brief marks as guesses ("likely", "possible", "may also have") come from
    the public random battle sets and from Foul Play's samples of them, and are not facts.
  * the estimate of the request's tokens against the budget (luna_player.TOKEN_BUDGET);
  * the order: the action played is a legal one (the note's mask), the decision took no longer
    than its deadline (the note's "ms"), and the server refused no order as invalid (|error| lines);
  * the record: Luna's choice when it answered, else why mix played.

With --secret-env NAME the text of every file under the run folder is searched for the value of
that environment variable (the fake key of a test, never printed here); a file that has it is a
failure.

Exit code 0 when nothing is wrong, 1 otherwise. --json prints the numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def to_id(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


class Seen:
    """What a battle's lines had shown our side so far: fed line by line, in order."""

    def __init__(self) -> None:
        self.role: Optional[str] = None          # ours (known from the first request; the leads come out before it)
        self.users: Dict[str, str] = {}          # role -> the user's name
        self.species: Dict[str, Set[str]] = {"p1": set(), "p2": set()}     # side -> names as the details give them
        self.moves: Dict[str, Dict[str, Set[str]]] = {"p1": {}, "p2": {}}  # side -> its Pokemon (the name in the lines) -> move ids used
        self.our_species: Set[str] = set()       # our team, from the requests
        self.text: List[str] = []                # every line so far (for items and abilities shown)

    @property
    def foe(self) -> Optional[str]:
        return None if self.role is None else ("p2" if self.role == "p1" else "p1")

    @property
    def foe_species(self) -> Set[str]:
        """The opponent's Pokemon that have come out ("Polteageist-Antique")."""
        return set(self.species.get(self.foe or "", ()))

    @property
    def foe_moves(self) -> Dict[str, Set[str]]:
        return {k: set(v) for k, v in (self.moves.get(self.foe or "") or {}).items()}

    def feed(self, data: str) -> None:
        for line in str(data).split("\n"):
            parts = line.split("|")
            if len(parts) < 2:
                continue
            kind = parts[1]
            if kind == "player" and len(parts) > 3 and parts[3]:
                self.users[parts[2]] = parts[3]
            elif kind == "request" and len(parts) > 2 and parts[2].strip():
                try:
                    request = json.loads("|".join(parts[2:]))
                except ValueError:
                    continue
                side = request.get("side") or {}
                self.role = side.get("id") or self.role
                for mon in side.get("pokemon") or []:
                    self.our_species.add(str(mon.get("details") or "").split(",")[0].strip())
                continue                     # a request is ours: nothing of the opponent's is in it
            elif kind in ("switch", "drag", "replace", "detailschange", "-formechange") and len(parts) > 3:
                side = parts[2].strip()[:2]
                if side in self.species:
                    self.species[side].add(parts[3].split(",")[0].strip())
            elif kind == "move" and len(parts) > 3:
                side = parts[2].strip()[:2]
                if side in self.moves:
                    who = parts[2].split(":", 1)[-1].strip()
                    self.moves[side].setdefault(who, set()).add(to_id(parts[3]))
            if kind not in ("c", "c:", "chat", "j", "J", "l", "L", "n", "N", "raw", "html", "uhtml", "request"):
                self.text.append(line)

    def copy(self) -> "Seen":
        other = Seen()
        other.role, other.users = self.role, dict(self.users)
        other.species = {k: set(v) for k, v in self.species.items()}
        other.moves = {side: {k: set(v) for k, v in mons.items()} for side, mons in self.moves.items()}
        other.our_species = set(self.our_species)
        other.text = list(self.text)
        return other


def _mentions(name: str, text: str) -> bool:
    return bool(name) and re.search(r"(?<![A-Za-z0-9\-])" + re.escape(name) + r"(?![A-Za-z0-9\-])", text) is not None


def _species_names(name: str) -> Set[str]:
    """The ways a species can be written: as the lines give it and as the dex names it."""
    out = {name}
    try:
        import setup_moves as sm

        out.add(sm.species_name(to_id(name)))
    except Exception:
        pass
    return {n for n in out if n}


def hidden_problems(brief: str, choices: Iterable[Dict[str, Any]], seen: Seen, whole: Seen) -> List[str]:
    """
    What a request says that our side had not been shown when it was asked ([] when nothing).
    seen: the battle up to the question; whole: the battle to its end (what came out later).
    """
    problems: List[str] = []
    text = brief + "\n" + "\n".join(str(c.get("description") or "") + " " + str(c.get("value") or "") for c in choices)
    ours = set()
    for name in whole.our_species:
        ours |= _species_names(name)
    shown = set()
    for name in seen.foe_species:
        shown |= _species_names(name)
    for name in sorted(whole.foe_species - seen.foe_species):
        for written in _species_names(name):
            if written in ours or written in shown or any(written != s and written in s for s in ours | shown):
                continue
            if _mentions(written, text):
                problems.append("names %s, which the opponent had not sent out yet" % written)
    for user in whole.users.values():
        if len(user) >= 4 and user in text:
            problems.append("has a user's name in it")
    line = next((ln for ln in brief.split("\n") if ln.startswith("FOE'S ACTIVE:")), "")
    if line:
        try:
            import setup_moves as sm

            listed = re.search(r"moves seen: ([^;.]*)", line)
            names = [m.strip() for m in (listed.group(1).split(",") if listed else []) if m.strip() and m.strip() != "none yet"]
            used = set()
            for moves in seen.foe_moves.values():
                used |= moves
            for name in names:
                if to_id(name) not in used and to_id(name) not in {to_id(sm.move_name(m)) for m in used}:
                    problems.append("lists %s as a move seen, which the opponent had not used" % name)
        except Exception as exc:
            problems.append("the moves seen could not be read (%s)" % type(exc).__name__)
        earlier = to_id(" ".join(seen.text))
        item = re.search(r"; item ([^;.(]+?)(?:;|\.|$)", line)
        if item and "not shown" not in item.group(1) and to_id(item.group(1)) not in earlier:
            problems.append("states the opponent's item %s, which the battle had not shown" % item.group(1).strip())
        ability = re.search(r"; ability ([^;.(]+?)(?:;|\.|$)", line)
        if ability and "not shown" not in ability.group(1) and to_id(ability.group(1)) not in earlier:
            names = set(seen.foe_species) | {line[len("FOE'S ACTIVE:"):].strip().split(" L", 1)[0]}
            if not any(_only_ability(name, ability.group(1)) for name in names):
                problems.append("states the opponent's ability %s, which the battle had not shown" % ability.group(1).strip())
    return problems


def _only_ability(species: str, ability: str) -> bool:
    """
    Whether this species or forme can have this ability only (then it is no secret once the forme
    has been shown: a Terapagos that changed to its Terastal forme has Tera Shell).
    """
    try:
        import setup_moves as sm

        sm._data()
        entry = (sm._DEX or {}).get(to_id(species)) or {}
        abilities = {to_id(a) for a in (entry.get("abilities") or {}).values()}
        if len(abilities) == 1 and to_id(ability) in abilities:
            return True
        import luna_player as lp

        public = [to_id(a) for a in lp._set_of_species(species).get("abilities") or []]
        return len(set(public)) == 1 and to_id(ability) in public
    except Exception:
        return False


def read_battle(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def audit_battle(rows: List[Dict[str, Any]], budget: int, deadline_s: float = 2.5) -> Dict[str, Any]:
    """One battle file's checks: see the module's text. deadline_s: the run's --deadline."""
    whole = Seen()
    for e in rows:
        if e.get("dir") == "in" and isinstance(e.get("data"), str):
            whole.feed(e["data"])
    seen = Seen()
    out: Dict[str, Any] = {"decisions": 0, "luna": 0, "asked": 0, "answered": 0, "played": 0, "overruled": 0, "agreed": 0,
                           "fallbacks": {}, "tokens_max": 0, "tokens": [], "latency": [], "late": 0, "illegal": 0, "invalid_orders": 0,
                           "hidden": [], "over_budget": 0, "briefs": 0, "detail": {}, "problems": [], "arm": None, "brief_problems": 0}
    deadline_default = float(deadline_s)
    for e in rows:
        data = e.get("data")
        if e.get("dir") == "in" and isinstance(data, str):
            seen.feed(data)
            out["invalid_orders"] += sum(1 for line in data.split("\n") if line.startswith("|error|[Invalid choice]"))
            continue
        if e.get("dir") != "note" or not isinstance(data, dict):
            continue
        if "arm" in data and "battle" in data and out["arm"] is None:
            out["arm"] = data.get("arm")
        d = data.get("decision")
        if not isinstance(d, dict):
            continue
        out["decisions"] += 1
        mask = d.get("mask") or []
        played = d.get("played")
        if isinstance(played, int) and (played >= len(mask) or not mask[played]) and int(d.get("legal") or 0) > 0:
            out["illegal"] += 1
            out["problems"].append("turn %s: action %s was played and is not legal" % (d.get("turn"), played))
        limit = float(d.get("deadline_s") or deadline_default) * 1000.0
        if float(d.get("ms") or 0.0) > limit:
            out["late"] += 1
            out["problems"].append("turn %s: the decision took %.0f ms (deadline %.0f)" % (d.get("turn"), float(d.get("ms") or 0), limit))
        luna = d.get("luna")
        if not isinstance(luna, dict):
            continue
        out["luna"] += 1
        out["asked"] += int(bool(luna.get("asked")))
        if luna.get("choice") is not None:
            out["answered"] += 1
            out["agreed"] += int(luna.get("choice") == luna.get("mix"))
            out["overruled" if luna.get("overruled") else "played"] += 1
            expected = luna.get("mix") if luna.get("overruled") else luna.get("choice")
            if played != expected:
                out["problems"].append("turn %s: Luna's record says %s is played, the order was %s" % (d.get("turn"), expected, played))
        elif played != luna.get("mix"):
            out["problems"].append("turn %s: Luna gave no answer and %s was played, not mix's %s" % (d.get("turn"), played, luna.get("mix")))
        if luna.get("fell_back"):
            why = str(luna.get("why"))
            out["fallbacks"][why] = out["fallbacks"].get(why, 0) + 1
        if luna.get("latency_ms") is not None:
            out["latency"].append(float(luna["latency_ms"]))
        if luna.get("brief_problems"):
            out["brief_problems"] += 1
            out["problems"].append("turn %s: parts of the brief could not be written: %s" % (d.get("turn"), ", ".join(luna["brief_problems"])))
        if luna.get("asked") and luna.get("input_tokens") is not None:
            tokens = int(luna["input_tokens"])
            out["tokens"].append(tokens)
            out["tokens_max"] = max(out["tokens_max"], tokens)
            if tokens > budget:
                out["over_budget"] += 1
                out["problems"].append("turn %s: %d tokens (budget %d)" % (d.get("turn"), tokens, budget))
        level = str(luna.get("detail") or 0)
        if luna.get("asked"):
            out["detail"][level] = out["detail"].get(level, 0) + 1
        if isinstance(luna.get("brief"), str):
            out["briefs"] += 1
            for problem in hidden_problems(luna["brief"], luna.get("choices") or [], seen, whole):
                out["hidden"].append("turn %s: the brief %s" % (d.get("turn"), problem))
    return out


def secret_files(folder: str, secret: str) -> List[str]:
    """The files under `folder` whose bytes hold the secret (never printed)."""
    found = []
    needle = secret.encode("utf-8")
    for root, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            try:
                with open(path, "rb") as fh:
                    if needle in fh.read():
                        found.append(path)
            except OSError:
                continue
    return found


def audit_run(folder: str, budget: int, secret: Optional[str] = None, deadline_s: float = 2.5) -> Dict[str, Any]:
    total: Dict[str, Any] = {"battles": 0, "luna_battles": 0, "decisions": 0, "luna": 0, "asked": 0, "answered": 0, "played": 0,
                             "overruled": 0, "agreed": 0, "fallbacks": {}, "tokens_max": 0, "late": 0, "illegal": 0,
                             "invalid_orders": 0, "hidden": [], "over_budget": 0, "briefs": 0, "detail": {}, "problems": [],
                             "latency": [], "tokens": [], "brief_problems": 0}
    for path in sorted(glob.glob(os.path.join(folder, "battles", "battle-*.jsonl"))):
        one = audit_battle(read_battle(path), budget, deadline_s)
        total["battles"] += 1
        total["luna_battles"] += int(one["luna"] > 0)
        name = os.path.basename(path)
        for key in ("decisions", "luna", "asked", "answered", "played", "overruled", "agreed", "late", "illegal", "invalid_orders",
                    "over_budget", "briefs", "brief_problems"):
            total[key] += one[key]
        total["tokens_max"] = max(total["tokens_max"], one["tokens_max"])
        total["latency"] += one["latency"]
        total["tokens"] += one["tokens"]
        for why, k in one["fallbacks"].items():
            total["fallbacks"][why] = total["fallbacks"].get(why, 0) + k
        for level, k in one["detail"].items():
            total["detail"][level] = total["detail"].get(level, 0) + k
        total["hidden"] += ["%s %s" % (name, p) for p in one["hidden"]]
        total["problems"] += ["%s %s" % (name, p) for p in one["problems"]]
    results = os.path.join(folder, "results.jsonl")
    lines = []
    if os.path.isfile(results):
        with open(results, encoding="utf-8") as fh:
            for line in fh:
                try:
                    lines.append(json.loads(line))
                except ValueError:
                    continue
    total["games"] = len(lines)
    total["games_by_arm"] = {}
    for d in lines:
        arm = str(d.get("arm"))
        row = total["games_by_arm"].setdefault(arm, {"games": 0, "wins": 0, "timer_losses": 0})
        row["games"] += 1
        row["wins"] += int(bool(d.get("won")))
        row["timer_losses"] += int(d.get("ending") == "inactivity" and d.get("ending_by") == "us")
    # The results lines against the notes: the same counts, game by game summed.
    total["results_asked"] = sum(int(d.get("luna_asked") or 0) for d in lines)
    total["results_answered"] = sum(int(d.get("luna_answered") or 0) for d in lines)
    total["results_overruled"] = sum(int(d.get("luna_overruled") or 0) for d in lines)
    total["results_fallbacks"] = {}
    for d in lines:
        for why, k in (d.get("luna_fallbacks") or {}).items():
            total["results_fallbacks"][why] = total["results_fallbacks"].get(why, 0) + int(k)
    total["secret_files"] = None if secret is None else secret_files(folder, secret)
    return total


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Check a run of the Luna arm from its files.")
    ap.add_argument("--run", required=True, help="the folder with results.jsonl and battles/")
    ap.add_argument("--budget", type=int, default=None, help="the token budget (default: luna_player.TOKEN_BUDGET)")
    ap.add_argument("--secret-env", default=None, help="the environment variable that holds a secret no file may contain")
    ap.add_argument("--deadline", type=float, default=2.5, help="the run's --deadline: a decision slower than this is late")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    import luna_player as lp

    budget = int(args.budget or lp.TOKEN_BUDGET)
    secret = os.environ.get(args.secret_env) if args.secret_env else None
    if args.secret_env and not secret:
        print("the environment variable %s is not set" % args.secret_env)
        return 2
    t = audit_run(os.path.abspath(args.run), budget, secret, args.deadline)
    bad = bool(t["hidden"] or t["over_budget"] or t["late"] or t["illegal"] or t["invalid_orders"] or t["problems"] or t["secret_files"])
    if args.json:
        t = dict(t, latency_n=len(t["latency"]), ok=not bad)
        t["latency_mean"] = round(sum(t["latency"]) / len(t["latency"]), 1) if t["latency"] else None
        t["tokens_mean"] = round(sum(t["tokens"]) / len(t["tokens"]), 1) if t["tokens"] else None
        t.pop("latency")
        t.pop("tokens")
        print(json.dumps(t, indent=1))
        return 1 if bad else 0
    print("run %s: %d battle files (%d with Luna decisions), %d games in results.jsonl %s" % (
        args.run, t["battles"], t["luna_battles"], t["games"], json.dumps(t["games_by_arm"], sort_keys=True)))
    print("decisions %d, of them Luna's %d: asked %d, answered %d (played %d, overruled %d), its pick was mix's on %d; mix instead %d %s"
          % (t["decisions"], t["luna"], t["asked"], t["answered"], t["played"], t["overruled"], t["agreed"],
             sum(t["fallbacks"].values()), json.dumps(t["fallbacks"], sort_keys=True)))
    same = (t["results_asked"] == t["asked"] and t["results_answered"] == t["answered"] and t["results_overruled"] == t["overruled"]
            and t["results_fallbacks"] == t["fallbacks"])
    print("results.jsonl says: asked %d, answered %d, overruled %d, fallbacks %s: %s" % (
        t["results_asked"], t["results_answered"], t["results_overruled"], json.dumps(t["results_fallbacks"], sort_keys=True),
        "the same as the notes" if same else "NOT the same as the notes (a game without a results line, or a count is off)"))
    if t["tokens"]:
        ordered = sorted(t["tokens"])
        print("tokens per question (estimate): mean %.0f, median %d, largest %d (budget %d); briefs shortened to fit: %s"
              % (sum(ordered) / len(ordered), ordered[len(ordered) // 2], ordered[-1], budget,
                 json.dumps({k: v for k, v in sorted(t["detail"].items()) if k != "0"}) or "none"))
    if t["latency"]:
        ordered = sorted(t["latency"])
        print("latency: mean %.0f ms, median %.0f, 95th %.0f, slowest %.0f" % (
            sum(ordered) / len(ordered), ordered[len(ordered) // 2], ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], ordered[-1]))
    print("late decisions %d, illegal actions %d, orders the server refused as invalid %d, briefs over the budget %d, "
          "briefs with a part missing %d" % (t["late"], t["illegal"], t["invalid_orders"], t["over_budget"], t["brief_problems"]))
    print("hidden information: %d briefs checked, %d problems" % (t["briefs"], len(t["hidden"])))
    for line in (t["hidden"] + t["problems"])[:20]:
        print("  " + line)
    if t["secret_files"] is not None:
        print("the secret (%s) is in %d files%s" % (args.secret_env, len(t["secret_files"]),
                                                   ": " + ", ".join(t["secret_files"][:5]) if t["secret_files"] else ""))
    print("AUDIT " + ("FAILED" if bad else "OK"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
