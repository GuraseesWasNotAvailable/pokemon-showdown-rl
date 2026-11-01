"""Every game of games.json (games.py) as one row: a CSV for programs and a Markdown table to read.
Opponents' names are left out on purpose (their rating is kept).

    python bench/ladder_log/log_table.py games.json <out.csv> <out.md>
"""
import collections
import csv
import re
import sys

games = __import__("json").load(open(sys.argv[1], encoding="utf-8"))
MODE = {"mix": "network + search", "mix-lines": "network + search + read-ahead lines", "sol-lines": "GPT-6 Sol decides, with lines",
        "luna": "GPT-6 Luna decides on top of mix", "sol-solo": "GPT-6 Sol alone", "fable-solo": "Claude Fable 5.1 alone"}


def sweeper(g):
    c = collections.Counter()
    boosted = collections.defaultdict(int)
    for t, side, killer, move, victim, kb in g["kos"]:
        if side == "foe":
            c[killer] += 1
            boosted[killer] = max(boosted[killer], kb)
    if not c:
        return ""
    k, n = c.most_common(1)[0]
    return "%s %d KO%s%s" % (k, n, "" if n == 1 else "s", " (boosted +%d)" % boosted[k] if boosted[k] else "")


rows = []
for i, g in enumerate(games, 1):
    us_t, foe_t = g["tera"]["us"], g["tera"]["foe"]
    rows.append({
        "n": i, "time_utc": (g["time"] or "")[:16].replace("T", " "), "mode": g["arm"], "network": (g["checkpoint"] or "").replace(".pt", ""),
        "result": "W" if g["won"] else "L", "turns": g["turns"], "rating_before": g["rating_before"], "rating_after": g["rating_after"],
        "foe_rating": g["opp_rating"], "ours_left": g["left"]["us"], "foes_left": g["left"]["foe"],
        "first_faint": {"us": "us", "foe": "foe"}.get(g["first_faint"], ""),
        "our_tera": "T%d %s" % (us_t[0], us_t[2]) if us_t else "", "foe_tera": "T%d %s" % (foe_t[0], foe_t[2]) if foe_t else "",
        "foe_setups": g["setups"]["foe"], "our_setups": g["setups"]["us"], "top_foe_ko": sweeper(g),
        "our_team": " / ".join(g["team"]["us"]), "foe_team": " / ".join(g["team"]["foe"]),
        # a hidden room's id ends in its password: the number alone names the battle
        "battle": re.sub(r"^(battle-[a-z0-9]+-\d+)-.*$", r"\1", g["battle"] or "")})

with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

out = ["# Every recorded ladder game, 2026-10-08 to 2026-10-14", "",
       "One row per game on play.pokemonshowdown.com, gen9randombattle, account YourShowdownName, oldest first. "
       "Times are UTC. Opponents' names are left out (their rating is kept). The same rows with both teams are in `games.csv`.",
       "",
       "Columns: mode (the `--player` arm), network (checkpoint), W/L, turns, our rating before and after, the foe's rating, "
       "Pokemon left on each side at the end, who lost a Pokemon first, each side's Tera (turn and type), set-up moves used by "
       "the foe and by us, and the foe Pokemon that knocked out most of ours (with its highest attack, special attack or speed "
       "boost when it did).", ""]
for arm in dict.fromkeys(g["arm"] for g in games):
    out.append("- `%s`: %s" % (arm, MODE.get(arm, arm)))
out += ["", "| # | time (UTC) | mode | network | W/L | turns | rating | foe | left us-foe | lost a Pokemon first | our Tera | foe Tera | foe set-ups | top foe KOs | battle |",
        "|---:|---|---|---|:-:|---:|---|---:|:-:|---|---|---|---:|---|---|"]
for r in rows:
    out.append("| %d | %s | %s | %s | %s | %s | %s->%s | %s | %s-%s | %s | %s | %s | %s | %s | %s |" % (
        r["n"], r["time_utc"], r["mode"], r["network"].replace("_update_", " "), r["result"], r["turns"], r["rating_before"],
        r["rating_after"], r["foe_rating"] or "", r["ours_left"], r["foes_left"], r["first_faint"], r["our_tera"], r["foe_tera"],
        r["foe_setups"], r["top_foe_ko"], r["battle"].replace("battle-gen9randombattle-", "")[:10]))
open(sys.argv[3], "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print(len(rows), "rows written")
