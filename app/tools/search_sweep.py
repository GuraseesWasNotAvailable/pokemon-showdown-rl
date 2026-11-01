"""
Search sweep: real ladder games on the owner's account, the same number for every thinking setting of the app,
played so that every setting meets the same hours, opponents, rating and load on the laptop.

    python tools/search_sweep.py --dry-run                       # the settings, the plan and the command; plays nothing
    python tools/search_sweep.py                                  # 2, 4, 6 s x Deeper, Wider: 5 real games each (30), 6 at once
    python tools/search_sweep.py --games-per 10
    python tools/search_sweep.py --sequential                     # one game at a time in shuffled rounds instead
    python tools/search_sweep.py --report runs/sweep/<stamp>      # the table, any time (also while it runs)
    python tools/search_sweep.py --resume runs/sweep/<stamp>      # go on after a stop or a crash

Run it from this folder (next to "PSBot Lookahead.exe" and ladder.py) with the venv's Python, with the window app's
ladder run stopped (one ladder.py per account; ladder.py refuses a second one itself).

The settings. Each setting is (seconds, style) with the mode the app uses (default: the "mode" of PSBot Lookahead.settings,
guarded or lookahead). Its ladder.py arguments are read from the app itself (PSBot Lookahead.exe --test-think: Thinking.Plan),
so a game here plays exactly as the slider and the Deeper/Wider switch would. Below 2 s the app's Wider is the same search as
Deeper, so those pairs are refused as duplicates.

Together (the default). ONE ladder.py plays all the settings at the same time: --parallel is the number of settings (6),
--games is settings x --games-per (30), and --arm-settings gives it the settings, so that every new game gets a setting of
its own and plays it exactly as ladder.py started with that setting's arguments alone would (its mode, its search budget,
its guard). Which setting a new game gets: always one of those with the fewest games so far, and among those one that has
no game in progress (ladder.py's SettingPicker). So the first 6 games are one of each setting; as a game ends, the next
one goes to a setting that is behind; no setting is ever more than one game ahead of another, and the 6 games in progress
are one of each setting wherever the counts allow it. Every setting meets the same hours, the same opponents, the same
rating and the same load.

Why 6 at most: Showdown lets one account play 6 battles at once (ladder.py MAX_HELPERS: 6 games at once need 7 search
helpers), and only one ladder.py may play an account (it takes a lock: two clients would both get every battle and both
answer it), so all settings have to be one ladder.py with one game each. More than 6 settings do not fit: give fewer, or
--sequential.

The cores. The 6 games search at the same time and share the laptop's cores (7 helpers, one core each while it searches),
so a setting does fewer search iterations a decision here than it does played alone, and the 6 s settings take more of the
machine than the 2 s ones. At any moment it is the same for every setting, which is what makes the comparison fair; the
absolute iterations are not those of a game played alone (the report shows them, and how many games were in progress
beside each setting's games on average).

--sequential, the other order: one game at a time (--parallel 1), in rounds; every round plays each setting once, in a
fresh random order (seeded, written to plan.json). A game then has the laptop to itself, but the settings are hours apart
within a round. A game that does not finish (no result line) is played again for the same setting, at most --retries times.

The password. Asked once at the start (not shown, not written anywhere), kept in memory and given to ladder.py on its
standard input (--password-stdin), as the window app does. The account is the "user" of PSBot Lookahead.settings unless
--user. A local test server needs none and none is asked.

The output, in runs/sweep/<stamp>/: plan.json (settings, their arguments, the plan), settings.json (the --arm-settings
file), results.jsonl (ladder.py's own record of every game, --out; each game's "setting", its "setting_budget" and how
many decisions searched with it or with a fall back, "budget_asked"), logs/ (ladder.py's output), REPORT.md. Write a file
named STOP there to start no more games (together: the games in progress are finished; sequential: after the game in
progress). Together, a setting whose game ended without a result is played again (another ladder.py, for the games still
missing), at most --retries more times.

The check (together). Every finished game is printed with what it recorded (its time per guessed team, the times as many
teams, the cap, the guard, its mode) next to what its setting plans in the app's table. When one does not match, the
sweep writes the STOP file itself: ladder.py finishes the games in progress and starts no more, and the sweep plays no
more, so a sweep whose settings do not reach the games costs a round, not 30 games. A Wider game that fell back to the
1 s search (the battle timer) is no mismatch; the line says so.

The report. Per setting: games, W-L, win rate with its 95% interval, the opponents' mean rating, and the score against the
score the ratings predicted (Elo expected score from our rating before the game and the opponent's), which takes the luck of
the opponents' strength out; plus the measured seconds a decision and search iterations, how often the search fell back to
the network, how often a game fell back to the 1 s search (Wider's guard against the battle timer), and whether every game
recorded the budget its setting plans. How sure: 5 games a setting give a win rate an interval of about +-44 points, and two
settings 10 points apart need about 190 games each to tell apart. 30 games find a broken setting (one that falls back,
times out or plays far worse); they cannot rank close ones.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(HERE, "PSBot Lookahead.exe")
SETTINGS = os.path.join(HERE, "PSBot Lookahead.settings")
DEFAULT_CKPT = r"C:\Users\you\ps-ppo\runs\vm\checkpoints\learner_update_024381.pt"   # old main 24381, the best ladder network
MAX_TOGETHER = 6           # Showdown's battles at once per account (ladder.py MAX_HELPERS)
GUARD_TRUST = 0.03         # search_player.ARM_FIXES["mix-trust"], the app's Thinking.GuardTrust


def app_settings() -> dict:
    out = {}
    try:
        for line in open(SETTINGS, encoding="utf-8"):
            if "=" in line and not line.startswith("#"):
                k, v = line.rstrip("\n").split("=", 1)
                if "pass" not in k.lower() and "key" not in k.lower():
                    out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def read_think_table(path: str) -> dict:
    """{(seconds, style, mode): row} from the text of PSBot Lookahead.exe --test-think (row: player, teams, ms, cap, args)."""
    table = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(None, 9)
        if len(parts) < 10:
            continue
        try:
            secs = float(parts[0])
            row = {"player": parts[3], "teams": int(parts[4]), "ms": int(parts[5]), "cap": float(parts[7]), "args": parts[9].split()}
        except ValueError:
            continue
        table[(secs, parts[1], parts[2])] = row
    return table


def think_table(path: str = None) -> dict:
    """read_think_table of PSBot Lookahead.exe --test-think, or of the table's text in path (as the app printed it)."""
    if path:
        table = read_think_table(path)
    else:
        if not os.path.exists(EXE):
            sys.exit(f"no {EXE}: run this from the app's folder")
        fd, out = tempfile.mkstemp(suffix=".txt", prefix="think_")
        os.close(fd)
        subprocess.run([EXE, "--test-think", out], check=True, timeout=120)
        table = read_think_table(out)
    if not table:
        sys.exit(f"could not read the thinking table from {path or EXE + ' --test-think'}")
    return table


def build_settings(table: dict, seconds: list, styles: list, mode: str) -> list:
    settings, seen = [], {}
    for s in seconds:
        for st in styles:
            key = (s, st, mode)
            if key not in table:
                have = sorted({k[0] for k in table})
                sys.exit(f"the app has no setting {s:g} s {st} {mode} (seconds it has: {', '.join(f'{x:g}' for x in have)})")
            row = table[key]
            args = row["args"]
            sig = " ".join(args)
            if sig in seen:
                sys.exit(f"{s:g} s {st} is the same search as {seen[sig]} in the app (below 2 s Wider is Deeper): drop one")
            name = f"{s:g}s-{st}"
            seen[sig] = name
            # What a game of it must record (from the app's own columns, not from ladder.py): the time per guessed team,
            # the times as many teams, the cap, the guard.
            expect = {"ms": row["ms"], "worlds_x": max(1, row["teams"] // 2), "deadline": row["cap"],
                      "trust": GUARD_TRUST if mode == "guarded" else 0.0}
            settings.append({"name": name, "seconds": s, "style": st, "mode": mode, "args": args, "expect": expect})
    return settings


def schedule(settings: list, games_per: int, seed: int) -> list:
    rng = random.Random(seed)
    order = []
    for r in range(games_per):
        names = [s["name"] for s in settings]
        rng.shuffle(names)
        order += [{"round": r + 1, "setting": n} for n in names]
    return order


def results_lines(path: str) -> list:
    try:
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except OSError:
        return []


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def expected_score(rb, ro):
    """Elo's expected score from our rating before the game and the opponent's (None without both)."""
    if isinstance(rb, (int, float)) and isinstance(ro, (int, float)):
        return 1 / (1 + 10 ** ((ro - rb) / 400))
    return None


def score_text(games: list) -> str:
    exp = [x for x in games if x.get("expected") is not None]
    if not exp:
        return "-"
    d = [(1.0 if x["won"] else 0.0) - x["expected"] for x in exp]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / max(1, len(d) - 1))
    half = 1.96 * sd / math.sqrt(len(d))
    return f"{100 * m:+.1f} ({100 * (m - half):+.1f} to {100 * (m + half):+.1f})"


def sure_note(games_per: int) -> str:
    return (f"How sure: with {games_per} games a setting the 95% interval of a win rate is about "
            f"+-{100 * 0.98 / math.sqrt(max(1, games_per)):.0f} points; two settings 5 points apart need about 750 "
            "games each to tell apart (10 points apart: about 190). Read a small sweep for broken settings, not for a winner.")


def report(run_dir: str) -> str:
    plan = json.load(open(os.path.join(run_dir, "plan.json"), encoding="utf-8"))
    if plan.get("how") == "together":
        return report_together(run_dir, plan)
    games = [json.loads(l) for l in open(os.path.join(run_dir, "sweep.jsonl"), encoding="utf-8") if l.strip()] \
        if os.path.exists(os.path.join(run_dir, "sweep.jsonl")) else []
    lines = [f"# Search sweep {os.path.basename(run_dir)}", "",
             f"Network {plan['checkpoint']}; mode {plan['mode']}; {plan['games_per']} games a setting planned, "
             f"one game at a time in shuffled rounds (seed {plan['seed']}). {len(games)} games played.", "",
             "| Setting | Games | W-L | Win rate (95%) | Score minus expected (95%) | Opponents' rating | Seconds a decision | Iterations a decision |",
             "|---|---:|---|---|---|---:|---:|---:|"]
    for s in plan["settings"]:
        g = [x for x in games if x["setting"] == s["name"]]
        n, w = len(g), sum(1 for x in g if x["won"])
        lo, hi = wilson(w, n)
        opp = [x["opponent_rating"] for x in g if isinstance(x.get("opponent_rating"), (int, float))]
        dec = [x["decision_ms_mean"] / 1000 for x in g if isinstance(x.get("decision_ms_mean"), (int, float))]
        it = [x["iterations_per_decision"] for x in g if isinstance(x.get("iterations_per_decision"), (int, float))]
        rate = f"{100 * w / n:.0f}% ({100 * lo:.0f} to {100 * hi:.0f})" if n else "-"
        opp_s = f"{sum(opp) / len(opp):.0f}" if opp else "-"
        dec_s = f"{sum(dec) / len(dec):.2f}" if dec else "-"
        it_s = f"{sum(it) / len(it) / 1e6:.2f}M" if it else "-"
        lines.append(f"| {s['name']} | {n} | {w}-{n - w} | {rate} | {score_text(g)} | {opp_s} | {dec_s} | {it_s} |")
    lines += ["", "Score minus expected: the share of games won minus the share the two ratings predicted (Elo, from our rating "
              "before each game and the opponent's), so a setting that drew strong opponents is not penalised for it.",
              sure_note(plan["games_per"])]
    text = "\n".join(lines) + "\n"
    open(os.path.join(run_dir, "REPORT.md"), "w", encoding="utf-8").write(text)
    return text


# ---------------------------------------------------------------------------------------------
# Together: one ladder.py, one game of every setting at once (--arm-settings)
# ---------------------------------------------------------------------------------------------

def together_results(run_dir: str, plan: dict) -> list:
    """ladder.py's result lines of this sweep: its experiment and network, a setting of the plan, a result seen."""
    names = {s["name"] for s in plan["settings"]}
    ckpt = os.path.basename(plan["checkpoint"])
    out = []
    for r in results_lines(os.path.join(run_dir, "results.jsonl")):
        if (r.get("experiment") == plan["experiment"] and r.get("checkpoint") == ckpt and r.get("setting") in names
                and r.get("won") in (True, False)):
            out.append(dict(r, expected=expected_score(r.get("rating_before"), r.get("opponent_rating"))))
    return out


def finished_counts(run_dir: str, plan: dict) -> dict:
    """Games with a result per setting, counted as ladder.py counts them for its balance (ladder.setting_counts)."""
    counts = {s["name"]: 0 for s in plan["settings"]}
    for r in together_results(run_dir, plan):
        counts[r["setting"]] += 1
    return counts


def as_planned(r: dict, expect: dict) -> bool:
    """Whether a game recorded the budget and the guard its setting plans (from the app's own table)."""
    b = r.get("setting_budget") or {}
    trust = float(r.get("trust") or 0.0)
    return (b.get("ms") == expect["ms"] and b.get("worlds_x") == expect["worlds_x"]
            and abs(float(b.get("deadline") or 0.0) - float(expect["deadline"])) < 1e-6
            and abs(trust - float(expect["trust"])) < 1e-9)


def at_once(games: list, every: list) -> list:
    """For each game of `games`: how many games of `every` were in progress beside it, on average over its time."""
    def span(r):
        try:
            end = dt.datetime.strptime(r["time"], "%Y-%m-%dT%H:%M:%SZ").timestamp()
            return end - float(r["seconds"]), end
        except (KeyError, TypeError, ValueError):
            return None
    out = []
    for g in games:
        sg = span(g)
        if sg is None or sg[1] <= sg[0]:
            continue
        busy = 0.0
        for h in every:
            sh = span(h)
            if h is g or sh is None:
                continue
            busy += max(0.0, min(sg[1], sh[1]) - max(sg[0], sh[0]))
        out.append(busy / (sg[1] - sg[0]))
    return out


def report_together(run_dir: str, plan: dict) -> str:
    games = together_results(run_dir, plan)
    lines = [f"# Search sweep {os.path.basename(run_dir)}", "",
             f"Network {plan['checkpoint']}; mode {plan['mode']}; {plan['games_per']} games a setting planned, all the settings "
             f"together: one ladder.py, {plan['parallel']} games at once, each new game a setting with the fewest games "
             f"(experiment {plan['experiment']}). {len(games)} games with a result.", "",
             "| Setting | Games | W-L | Win rate (95%) | Score minus expected (95%) | Opponents' rating | Seconds a decision "
             "| Iterations a decision | Search fell back | Budget fell back | Games beside | As planned |",
             "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in plan["settings"]:
        g = [x for x in games if x["setting"] == s["name"]]
        n, w = len(g), sum(1 for x in g if x["won"])
        lo, hi = wilson(w, n)
        opp = [x["opponent_rating"] for x in g if isinstance(x.get("opponent_rating"), (int, float))]
        dec = [x["decision_ms_mean"] / 1000 for x in g if isinstance(x.get("decision_ms_mean"), (int, float))]
        it = [x["iterations_per_decision"] for x in g if isinstance(x.get("iterations_per_decision"), (int, float))]
        asked = sum(int(x.get("asked") or 0) for x in g)
        fell = sum(int(x.get("fallbacks") or 0) for x in g)
        budget_fell = sum(1 for x in g if x.get("deep_fallback"))
        beside = at_once(g, games)
        rate = f"{100 * w / n:.0f}% ({100 * lo:.0f} to {100 * hi:.0f})" if n else "-"
        lines.append(
            f"| {s['name']} | {n} | {w}-{n - w} | {rate} | {score_text(g)} | {f'{sum(opp) / len(opp):.0f}' if opp else '-'} "
            f"| {f'{sum(dec) / len(dec):.2f}' if dec else '-'} | {f'{sum(it) / len(it) / 1e6:.2f}M' if it else '-'} "
            f"| {f'{fell} of {asked} ({100 * fell / asked:.1f}%)' if asked else '-'} "
            f"| {f'{budget_fell} of {n} games' if any('setting_fallback' in x for x in g) else '-'} "
            f"| {f'{sum(beside) / len(beside):.1f}' if beside else '-'} "
            f"| {sum(1 for x in g if as_planned(x, s['expect']))} of {n} |")
    lines += ["", "Score minus expected: the share of games won minus the share the two ratings predicted (Elo, from our rating "
              "before each game and the opponent's), so a setting that drew strong opponents is not penalised for it.",
              "Search fell back: searched decisions that played the network's move instead (no answer in time, an error). "
              "Budget fell back: games of a Wider setting that went back to the 1 s search for the rest of the game (the battle "
              "timer was low, or a decision took over 9 s). Games beside: the other games in progress during a game, on average "
              "(5 when one of each setting is always playing). As planned: games whose recorded budget (time per guessed team, "
              "teams, cap) and guard are the app's for that setting.",
              "The iterations are those of 6 games searching at once on the laptop's cores, the same for every setting at the "
              "same moment: fewer than a game played alone would do.",
              sure_note(plan["games_per"]), "", "Settings (ladder.py arguments from the app; the run's --helpers is "
              f"{plan['parallel'] + 1}, not the app's):"]
    lines += [f"- {s['name']}: {' '.join(s['args'])}" for s in plan["settings"]]
    text = "\n".join(lines) + "\n"
    open(os.path.join(run_dir, "REPORT.md"), "w", encoding="utf-8").write(text)
    return text


def ladder_check(settings: list, parallel: int, server: str, checkpoint: str, user: str) -> list:
    """ladder.py's own check of the settings (load_arm_settings, in this process: nothing is started): its settings, or exit."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import ladder

    base = ["--games", "1", "--parallel", str(parallel), "--helpers", str(parallel + 1), "--server", server,
            "--checkpoint", checkpoint, "--user", user]
    args = ladder.parse_args(base)
    args.arm_settings_file = "(the sweep's settings)"
    return ladder.load_arm_settings(args, base + ["--arm-settings", "x"], data=[{"name": s["name"], "args": s["args"]}
                                                                              for s in settings])


def together_command(plan: dict, run_dir: str, python: str, games: int) -> list:
    return [python, "-u", "ladder.py", "--password-stdin", "--games", str(games), "--parallel", str(plan["parallel"]),
            "--helpers", str(plan["parallel"] + 1), "--arm-settings", os.path.join(run_dir, "settings.json"),
            "--experiment", plan["experiment"], "--checkpoint", plan["checkpoint"], "--user", plan["user"],
            "--server", plan["server"], "--out", os.path.join(run_dir, "results.jsonl"),
            "--stop-file", os.path.join(run_dir, "STOP")]


def budget_text(ms, worlds, deadline, trust) -> str:
    return f"{ms} ms x{worlds} within {deadline} s, trust {trust:g}"


def check_game(r: dict, setting: dict) -> tuple:
    """(one line: the game's recorded budget next to its setting's, whether they match). A fall back of a Wider game is
    no mismatch (it is the setting's own guard against the battle timer); it is said in the line."""
    e = setting["expect"]
    b = r.get("setting_budget") or {}
    trust = float(r.get("trust") or 0.0)
    player = setting["args"][setting["args"].index("--player") + 1] if "--player" in setting["args"] else None
    ok = (as_planned(r, e) and r.get("search_ms") == e["ms"] and r.get("arm") == player
          and (e["worlds_x"] == 1 or r.get("budget") == b))      # a Wider game: the budget its decisions were asked with
    line = (f"recorded {budget_text(r.get('search_ms'), b.get('worlds_x'), b.get('deadline'), trust)} ({r.get('arm')}); "
            f"the setting's {budget_text(e['ms'], e['worlds_x'], e['deadline'], e['trust'])} ({player}): "
            + ("as planned" if ok else "MISMATCH"))
    fell = r.get("deep_fallback")
    if isinstance(fell, dict):
        line += f"; fell back to the 1 s search at turn {fell.get('turn')} ({fell.get('why')})"
    return line, ok


def note_game(run_dir: str, plan: dict, r: dict, number: int) -> bool:
    """A finished game, in two lines; on a budget that is not its setting's the STOP file is written (ladder.py finishes the
    games in progress and starts no more; the sweep runs no more), so a broken sweep costs a few games, not 30."""
    setting = {s["name"]: s for s in plan["settings"]}[r["setting"]]
    total = len(plan["settings"]) * plan["games_per"]
    print(f"[{dt.datetime.now():%H:%M:%S}] game {number}/{total} {r['setting']}: {'WON' if r['won'] else 'lost'} in "
          f"{r.get('turns')} turns, rating {r.get('rating_before')} -> {r.get('rating_after')}", flush=True)
    line, ok = check_game(r, setting)
    print("    " + line, flush=True)
    if not ok:
        stop = os.path.join(run_dir, "STOP")
        with open(stop, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} game {r.get('battle')} ({r['setting']}): {line}\n")
        print(f"    STOPPING: this game's budget is not its setting's; {stop} written, so ladder.py finishes the games in "
              f"progress and starts no more. Look at {os.path.join(run_dir, 'results.jsonl')} before going on.", flush=True)
    return ok


def run_ladder(cmd: list, password: str, log: str, run_dir: str, plan: dict, shown: set) -> int:
    """One ladder.py: its output into log (the lines that are not one game's also on the console), a line here per game."""
    env = dict(os.environ)
    # As the window app starts it (PSBotLookahead.cs PythonStart): unbuffered, UTF-8, no PS_PASSWORD (the password goes on stdin).
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                "PYTHONWARNINGS": "ignore:flex_attention called without torch.compile"})
    env.pop("PS_PASSWORD", None)
    p = subprocess.Popen(cmd, cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
    try:
        p.stdin.write(password + "\n")
        p.stdin.flush()
        p.stdin.close()
    except OSError:     # it ended before reading it: its output says why
        pass

    def copy_out():
        with open(log, "a", encoding="utf-8") as out:
            for line in p.stdout:
                out.write(line)
                out.flush()
                # the games' own lines ([tag] ...: decisions, predictions, results with the opponent) and the rating trail stay in the log
                if line.strip() and not line.startswith("[") and not line.startswith("          "):
                    print("    | " + line.rstrip(), flush=True)

    reader = threading.Thread(target=copy_out, daemon=True)
    reader.start()
    while True:
        alive = p.poll() is None
        for r in together_results(run_dir, plan):
            if r.get("battle") not in shown:
                shown.add(r.get("battle"))
                note_game(run_dir, plan, r, len(shown))
                report(run_dir)
        if not alive:
            break
        time.sleep(3)
    reader.join(timeout=30)
    return p.wait()


def play_together(run_dir: str, password: str, retries: int, python: str) -> None:
    plan = json.load(open(os.path.join(run_dir, "plan.json"), encoding="utf-8"))
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    stop = os.path.join(run_dir, "STOP")
    shown = {r.get("battle") for r in together_results(run_dir, plan)}
    runs = 0
    while True:
        counts = finished_counts(run_dir, plan)
        missing = sum(max(0, plan["games_per"] - n) for n in counts.values())
        if missing == 0:
            print("every setting has its games")
            return
        if os.path.exists(stop):
            print(f"STOP file found: no more games ({missing} still missing; take {stop} away and --resume to go on)")
            return
        if runs > retries:
            sys.exit(f"{missing} games are still missing after {runs} runs of ladder.py: see {os.path.join(run_dir, 'logs')}, "
                     f"then --resume {run_dir}")
        runs += 1
        log = os.path.join(run_dir, "logs", f"ladder_{dt.datetime.now():%Y%m%d-%H%M%S}.log")
        print(f"[{dt.datetime.now():%H:%M:%S}] ladder.py: {missing} games, {plan['parallel']} at once"
              + (f" (run {runs}; games with a result so far: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ")"
                 if runs > 1 or any(counts.values()) else "") + f"; its output in {log}", flush=True)
        code = run_ladder(together_command(plan, run_dir, python, missing), password, log, run_dir, plan, shown)
        if code == 4:   # ladder.py: the login server refused the login (a wrong password); never tried again
            sys.exit(f"the login was refused (exit 4, see {log}): check the account and the password, then --resume {run_dir}")
        if code == 3:   # ladder.py: another bot plays this account (the window app?)
            sys.exit(f"another ladder.py plays this account (exit 3, see {log}): stop it, then --resume {run_dir}")
        if code != 0:
            print(f"    ladder.py ended with exit {code}; see {log}", flush=True)
            if not os.path.exists(stop):
                time.sleep(20)


def plan_together(a, settings: list, user: str, mode: str, stamp: str) -> dict:
    if len(settings) > MAX_TOGETHER:
        sys.exit(f"{len(settings)} settings: together plays one game of each at once, and Showdown lets an account play "
                 f"{MAX_TOGETHER} at once. Give at most {MAX_TOGETHER} (--seconds, --styles), or --sequential")
    return {"created": stamp, "how": "together", "user": user, "mode": mode, "checkpoint": a.checkpoint, "games_per": a.games_per,
            "server": a.server, "parallel": len(settings), "experiment": f"sweep-{stamp}", "settings": settings}


# ---------------------------------------------------------------------------------------------
# Sequential: one game at a time, in shuffled rounds
# ---------------------------------------------------------------------------------------------

def play(run_dir: str, password: str, retries: int, python: str) -> None:
    plan = json.load(open(os.path.join(run_dir, "plan.json"), encoding="utf-8"))
    if plan.get("how") == "together":
        return play_together(run_dir, password, retries, python)
    by_name = {s["name"]: s for s in plan["settings"]}
    sweep_path = os.path.join(run_dir, "sweep.jsonl")
    res_path = os.path.join(run_dir, "results.jsonl")
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    done = results_lines(sweep_path)
    for i, slot in enumerate(plan["order"]):
        if i < len(done):
            continue   # resume: this slot was played
        if os.path.exists(os.path.join(run_dir, "STOP")):
            print("STOP file found: stopping before game", i + 1)
            return
        s = by_name[slot["setting"]]
        for attempt in range(1, retries + 2):
            before = len(results_lines(res_path))
            log = os.path.join(run_dir, "logs", f"{i + 1:03d}_{s['name']}_try{attempt}.log")
            cmd = [python, "-u", "ladder.py", "--password-stdin", "--games", "1", "--parallel", "1",
                   "--checkpoint", plan["checkpoint"], "--user", plan["user"], "--server", plan.get("server", "public"),
                   "--out", res_path, "--stop-file", os.path.join(run_dir, "ladder.stop")] + s["args"]
            print(f"[{dt.datetime.now():%H:%M:%S}] game {i + 1}/{len(plan['order'])} round {slot['round']} {s['name']}"
                  + (f" (try {attempt})" if attempt > 1 else ""), flush=True)
            with open(log, "w", encoding="utf-8") as out:
                p = subprocess.run(cmd, cwd=HERE, input=password + "\n", text=True, stdout=out, stderr=subprocess.STDOUT)
            new = results_lines(res_path)[before:]
            if new:
                r = new[-1]
                rb, ro = r.get("rating_before"), r.get("opponent_rating")
                row = {"slot": i + 1, "round": slot["round"], "setting": s["name"], "seconds": s["seconds"], "style": s["style"],
                       "battle": r.get("battle"), "won": bool(r.get("won")), "rating_before": rb, "rating_after": r.get("rating_after"),
                       "opponent_rating": ro, "expected": expected_score(rb, ro), "turns": r.get("turns"), "arm": r.get("arm"),
                       "decision_ms_mean": r.get("decision_ms_mean"), "iterations_per_decision": r.get("iterations_per_decision"),
                       "time": r.get("time"), "attempt": attempt, "exit": p.returncode}
                with open(sweep_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"    {'WON' if row['won'] else 'lost'} in {row['turns']} turns, rating {rb} -> {row['rating_after']}", flush=True)
                break
            print(f"    no result (ladder.py exit {p.returncode}); see {log}", flush=True)
            if p.returncode == 4:   # ladder.py: the login server refused the login (a wrong password); never retried
                sys.exit(f"the login was refused (exit 4, see {log}): check the account and the password, then --resume {run_dir}")
            time.sleep(20)
        else:
            sys.exit(f"game {i + 1} ({s['name']}) did not finish after {retries + 1} tries: stopping (resume with --resume)")
        done.append(row)
        print(report(run_dir).splitlines()[0], "- report updated", flush=True)


def ask_password(plan: dict) -> str:
    """The account's password, once (not shown, not saved); a local test server needs none."""
    if plan.get("server", "public") != "public":
        print(f"server {plan['server']}: a local server needs no password, none is asked")
        return ""
    return getpass.getpass(f"Showdown password for {plan['user']} (not shown, not saved): ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", default="2,4,6", help="thinking seconds of the app's slider (default 2,4,6)")
    ap.add_argument("--styles", default="deeper,wider")
    ap.add_argument("--games-per", type=int, default=5, help="real games for every setting (default 5)")
    ap.add_argument("--sequential", action="store_true",
                    help="one game at a time in shuffled rounds, instead of one game of every setting at once (the default)")
    ap.add_argument("--mode", default=None, help="guarded or lookahead (default: the app's saved mode)")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--user", default=None, help="account name (default: the app's saved user)")
    ap.add_argument("--seed", type=int, default=None, help="--sequential: the rounds' order")
    ap.add_argument("--retries", type=int, default=2,
                    help="plays again a game that did not finish (sequential: per game; together: more runs of ladder.py "
                         "for the games still missing), at most this many times")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="print the settings, the plan and the commands; play nothing")
    ap.add_argument("--report", metavar="RUN_DIR")
    ap.add_argument("--resume", metavar="RUN_DIR")
    # Tests only: a local Showdown server, and the table as the app printed it (no app there).
    ap.add_argument("--server", default="public", help=argparse.SUPPRESS)
    ap.add_argument("--think-table", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not re.fullmatch(r"public|local:\d+", a.server):
        sys.exit("--server is public or local:PORT")

    if a.report:
        print(report(a.report))
        return 0
    if a.resume:
        plan = json.load(open(os.path.join(a.resume, "plan.json"), encoding="utf-8"))
        print(f"resuming {a.resume}: {plan['user']}, " + (f"{len(plan['settings'])} settings together, {plan['games_per']} games each"
                                                          if plan.get("how") == "together" else f"{len(plan['order'])} games planned"))
        play(a.resume, ask_password(plan), a.retries, a.python)
        print(report(a.resume))
        return 0

    saved = app_settings()
    mode = a.mode or saved.get("mode", "guarded")
    user = a.user or saved.get("user")
    if not user:
        sys.exit("no account: give --user (or save one in the app)")
    if not os.path.exists(a.checkpoint):
        sys.exit(f"no network at {a.checkpoint}: give --checkpoint")
    if a.games_per < 1:
        sys.exit("--games-per is at least 1")
    seconds = [float(x) for x in a.seconds.split(",") if x.strip()]
    styles = [x.strip().lower() for x in a.styles.split(",") if x.strip()]
    settings = build_settings(think_table(a.think_table), seconds, styles, mode)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(HERE, "runs", "sweep", stamp)

    if a.sequential:
        seed = a.seed if a.seed is not None else int(time.time())
        order = schedule(settings, a.games_per, seed)
        plan = {"created": stamp, "user": user, "mode": mode, "checkpoint": a.checkpoint, "games_per": a.games_per,
                "seed": seed, "server": a.server, "settings": settings, "order": order}
        print(f"account {user}, mode {mode}, network {a.checkpoint}")
        for s in settings:
            print(f"  {s['name']:12s} {' '.join(s['args'])}")
        print(f"{len(order)} games, one at a time, in {a.games_per} shuffled rounds of {len(settings)}; first round: "
              + ", ".join(o["setting"] for o in order[:len(settings)]))
        if a.dry_run:
            s = settings[0]
            print("first command:", " ".join([a.python, "-u", "ladder.py", "--password-stdin", "--games", "1", "--parallel", "1",
                                              "--checkpoint", a.checkpoint, "--user", user, "--server", a.server,
                                              "--out", os.path.join(run_dir, "results.jsonl")] + s["args"]))
            print("dry run: nothing played, nothing written")
            return 0
    else:
        plan = plan_together(a, settings, user, mode, stamp)
        checked = ladder_check(settings, plan["parallel"], a.server, a.checkpoint, user)   # ladder.py's own check, before anything
        print(f"account {user}, mode {mode}, network {a.checkpoint}")
        for s, c in zip(settings, checked):
            print(f"  {s['name']:12s} {c.text()}\n  {'':12s} {' '.join(s['args'])}")
        print(f"{len(settings) * a.games_per} games, all {len(settings)} settings at once in one ladder.py ({plan['parallel']} games "
              f"at once, {plan['parallel'] + 1} search helpers), every new game a setting with the fewest games so far")
        if a.dry_run:
            print("command:", " ".join(together_command(plan, run_dir, a.python, len(settings) * a.games_per)))
            print("dry run: nothing played, nothing written")
            return 0
    os.makedirs(run_dir, exist_ok=True)
    json.dump(plan, open(os.path.join(run_dir, "plan.json"), "w", encoding="utf-8"), indent=1)
    if not a.sequential:
        json.dump([{"name": s["name"], "args": s["args"]} for s in settings],
                  open(os.path.join(run_dir, "settings.json"), "w", encoding="utf-8"), indent=1)
    print(f"run folder {run_dir} (write a file named STOP there to start no more games)")
    password = ask_password(plan)
    play(run_dir, password, a.retries, a.python)
    print(report(run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
