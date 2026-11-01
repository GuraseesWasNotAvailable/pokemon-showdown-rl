"""
The Elo graph for ladder.py: one self-contained HTML page (inline SVG, no scripts, nothing
loaded from the network) written from the games in runs/ladder/results.jsonl.

It shows the rating after every game, a ROLL-game average of it, and the level the results
point to. The level is the rating at which the wins scored against these opponents' ratings
are exactly the expected number, with one standard error either way. It is worked out from
the games of the newest network only, and it does not lag the way the rating itself does: a
rating below its level is still climbing, a rating inside the band has reached its plateau.

A settled Showdown rating still wanders about 60 points either way (one game moves it about
20), and the ROLL-game average of it about WANDER. The verdict allows for both that and the
error of the level, says "too early" until there are MIN_GAMES games, and does not say
"plateau" until the level is known to within TIGHT points.

Games that carry an "arm" (ladder.py --arms, or --player search / mix: the mode the game was
played in) are also shown by arm: a table with each arm's games, record and the level its own
results point to, and the dots of the graph in the arm's colour (filled for a win, hollow for a
loss) with a legend. A page whose games have no arm reads exactly as it did before.
"""

from __future__ import annotations

import calendar
import html
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

ROLL = 10          # games in the moving average
MIN_GAMES = 20     # games with one network before a level is called
BAND_GAMES = 10    # games with one network before the level is drawn
WANDER = 54.0      # how far the ROLL-game average of a settled rating strays (one standard deviation, K = 40)
SLACK = 1.5        # the rating and the level disagree when they are further apart than SLACK margins
TIGHT = 60.0       # a plateau is only called when the level's standard error is below this
TABLE_ROWS = 300


def expected(own: float, opp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opp - own) / 400.0))


def performance(games: List[Dict[str, Any]]) -> Optional[Tuple[float, float, int]]:
    """(level, standard error, games used) from the results against rated opponents.

    One win and one loss against the average opponent are added, so that a run of only wins
    or only losses still gives a finite level; with 20 real games that moves it very little."""
    rows = [(1.0 if d.get("won") else 0.0, float(d["opponent_rating"])) for d in games if d.get("opponent_rating")]
    if len(rows) < 5:
        return None
    mean_opp = sum(o for _, o in rows) / len(rows)
    padded = rows + [(1.0, mean_opp), (0.0, mean_opp)]
    wins = sum(s for s, _ in padded)
    lo, hi = mean_opp - 1500.0, mean_opp + 1500.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if sum(expected(mid, o) for _, o in padded) < wins:
            lo = mid
        else:
            hi = mid
    level = (lo + hi) / 2.0
    info = sum(e * (1.0 - e) for e in (expected(level, o) for _, o in padded)) * (math.log(10.0) / 400.0) ** 2
    if info <= 0:
        return None
    return level, 1.0 / math.sqrt(info), len(rows)


def newest_network_games(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The games at the end of the list played with the same checkpoint as the last one."""
    if not games:
        return []
    last = games[-1].get("checkpoint")
    i = len(games)
    while i > 0 and games[i - 1].get("checkpoint") == last:
        i -= 1
    return games[i:]


ARM_COLOURS = 6    # colours in the page's style (a0 to a5); a seventh arm starts again at the first


# Arms whose own name says that the guarded search played them (search_player.ARM_FIXES with a "trust").
GUARDED_ARMS = ("mix-trust",)


def arm_of(d: Dict[str, Any]) -> str:
    """
    The arm a game is shown under: its "arm", with "+trust" after it when the guarded search played
    it under another arm's name (ladder.py --trust: the window's guarded mode searched wider records
    "mix-deep" and "trust" 0.03), so guarded and unguarded games of one arm are never pooled. "" for
    a game without an arm.
    """
    arm = str(d.get("arm") or "")
    if arm and d.get("trust") and arm not in GUARDED_ARMS:
        arm += "+trust"
    return arm


def arm_names(games: List[Dict[str, Any]]) -> List[str]:
    """The arms among the games (arm_of), in the order they first appear. Empty when no game has one."""
    names: List[str] = []
    for d in games:
        arm = arm_of(d)
        if arm and arm not in names:
            names.append(arm)
    return names


def arm_rows(games: List[Dict[str, Any]], names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    One row per arm, over the given games: its games and record, the level its results point to
    (performance() of its games alone; None with fewer than 5 rated opponents), what the search
    did in them (decisions, answered by the search, played by the network because the search
    failed, the network overruled) and how often and when it Terastallized.

    names: the arms in the order that gives them their colours (default: as they appear in
    these games). The page passes the arms of all its games, so that an arm has one colour in
    the graph and in the table.
    """
    rows: List[Dict[str, Any]] = []
    names = list(names) if names is not None else arm_names(games)
    for index, arm in enumerate(names):
        own = [d for d in games if arm_of(d) == arm]
        if not own:
            continue
        wins = sum(1 for d in own if d.get("won"))
        tera_turns = sorted(int(d["tera_turn"]) for d in own if d.get("tera") and d.get("tera_turn") is not None)
        searched = sum(int(d.get("searched") or 0) for d in own)
        iterations = sum(int(d.get("iterations_per_decision") or 0) * int(d.get("searched") or 0) for d in own)
        rows.append({
            "arm": arm, "colour": index % ARM_COLOURS, "games": len(own), "wins": wins, "losses": len(own) - wins,
            "level": performance(own),
            "decisions": sum(int(d.get("decisions") or 0) for d in own), "searched": searched,
            "fallbacks": sum(int(d.get("fallbacks") or 0) for d in own),
            "overruled": sum(int(d.get("overruled") or 0) for d in own),
            "iterations_per_decision": iterations // searched if searched else None,
            "tera_games": sum(1 for d in own if d.get("tera")),
            "tera_turn": tera_turns[len(tera_turns) // 2] if tera_turns else None,
        })
    return rows


def verdict(games: List[Dict[str, Any]]) -> Tuple[str, str]:
    """(one word, one sentence) on whether the rating has reached its level."""
    current = newest_network_games(games)
    rated = [int(d["rating_after"]) for d in current if d.get("rating_after") is not None]
    perf = performance(current)
    if len(current) < MIN_GAMES or perf is None or not rated:
        return "Too early", (f"{len(current)} games with this network so far. A level shows after about {MIN_GAMES}, "
                             f"and it takes 50 or more to be sure of it.")
    level, se, used = perf
    recent = rated[-ROLL:]
    avg = sum(recent) / len(recent)
    band = f"about {level:.0f}, give or take {se:.0f}"
    margin = SLACK * math.sqrt(se * se + WANDER * WANDER)
    if avg < level - margin:
        return "Still climbing", (f"The results point to a level of {band}. The rating is averaging {avg:.0f} "
                                  f"over the last {len(recent)} games, so it has further to go.")
    if avg > level + margin:
        return "Above its level", (f"The results point to a level of {band}. The rating is averaging {avg:.0f} "
                                   f"over the last {len(recent)} games, so expect it to come back down.")
    if se > TIGHT:
        more = max(5, int(math.ceil(used * ((se / TIGHT) ** 2 - 1.0) / 5.0)) * 5)
        return "Near its level", (f"The rating is averaging {avg:.0f} over the last {len(recent)} games and the results "
                                  f"point to {band}. That is the same within the noise, but the margin is still wide: "
                                  f"about {more} more games will settle it.")
    return "Plateau", (f"The rating is averaging {avg:.0f} over the last {len(recent)} games, and the results point to "
                       f"a level of {band}. It has reached its level: more games will narrow the figure, not move it much.")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _step(span: float, most: int) -> int:
    for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if span / s <= most:
            return s
    return 10000


def _local(stamp: Optional[str]) -> str:
    try:
        return time.strftime("%d %b %H:%M", time.localtime(calendar.timegm(time.strptime(stamp or "", "%Y-%m-%dT%H:%M:%SZ"))))
    except ValueError:
        return "?"


def _network(name: Optional[str]) -> str:
    digits = "".join(ch for ch in os.path.splitext(str(name or ""))[0].rsplit("_", 1)[-1] if ch.isdigit())
    return f"network {int(digits)}" if digits else str(name or "?")


def _svg(games: List[Dict[str, Any]]) -> str:
    rated = [d for d in games if d.get("rating_after") is not None]
    if not rated:
        return '<p class="note">No rated games yet.</p>'
    series = [int(d["rating_after"]) for d in rated]
    start = rated[0].get("rating_before")
    start = int(start) if start is not None else None

    current = newest_network_games(games)
    perf = performance(current) if len(current) >= BAND_GAMES else None
    first_current = len([d for d in games[:len(games) - len(current)] if d.get("rating_after") is not None])

    width, height, left, right, top, bottom = 1100, 430, 58, 26, 16, 38
    n = len(series)
    values = series + ([start] if start is not None else [])
    if perf:
        values += [perf[0] - perf[1], perf[0] + perf[1]]
    step_y = _step(max(values) - min(values) + 30, 8)
    lo = math.floor((min(values) - 12) / step_y) * step_y
    hi = math.ceil((max(values) + 12) / step_y) * step_y

    def x(k: float) -> float:
        return left + (width - left - right) * k / max(n, 1)

    def y(v: float) -> float:
        return top + (height - top - bottom) * (hi - v) / (hi - lo)

    out: List[str] = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Rating after each game">']
    v = lo
    while v <= hi:
        out.append(f'<line class="grid" x1="{left}" y1="{y(v):.1f}" x2="{width - right}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
        v += step_y
    step_x = _step(n, 12)
    k = 0
    while k <= n:
        out.append(f'<text class="tick" x="{x(k):.1f}" y="{height - bottom + 18}" text-anchor="middle">{k}</text>')
        k += step_x
    out.append(f'<text class="tick" x="{(left + width - right) / 2:.0f}" y="{height - 4}" text-anchor="middle">games played</text>')

    if perf:
        level, se, _ = perf
        x0, x1 = x(max(first_current, 0)), x(n)
        out.append(f'<rect class="band" x="{x0:.1f}" y="{y(level + se):.1f}" width="{max(x1 - x0, 1):.1f}" height="{y(level - se) - y(level + se):.1f}"/>'
                   f'<line class="level" x1="{x0:.1f}" y1="{y(level):.1f}" x2="{x1:.1f}" y2="{y(level):.1f}"/>'
                   f'<text class="leveltext" x="{x0 + 6:.1f}" y="{y(level + se) + 14:.1f}">level {level:.0f} &#177; {se:.0f}</text>')

    # A dashed line where the network changes.
    seen_rated = 0
    previous = None
    for d in games:
        if previous is not None and d.get("checkpoint") != previous and seen_rated > 0:
            out.append(f'<line class="change" x1="{x(seen_rated):.1f}" y1="{top}" x2="{x(seen_rated):.1f}" y2="{height - bottom}"/>'
                       f'<text class="tick" x="{x(seen_rated) + 5:.1f}" y="{top + 12}">{html.escape(_network(d.get("checkpoint")))}</text>')
        previous = d.get("checkpoint")
        if d.get("rating_after") is not None:
            seen_rated += 1

    points = ([(0, start)] if start is not None else []) + [(i + 1, r) for i, r in enumerate(series)]
    out.append('<polyline class="rating" points="' + " ".join(f"{x(k):.1f},{y(r):.1f}" for k, r in points) + '"/>')
    rolled = [sum(series[max(0, i - ROLL + 1):i + 1]) / len(series[max(0, i - ROLL + 1):i + 1]) for i in range(n)]
    if n >= 2:
        out.append('<polyline class="roll" points="' + " ".join(f"{x(i + 1):.1f},{y(r):.1f}" for i, r in enumerate(rolled)) + '"/>')
    radius = 4 if n <= 60 else 3 if n <= 200 else 2
    arms = arm_names(games)
    for i, d in enumerate(rated):
        before = d.get("rating_before")
        tip = (f"game {i + 1}: {'won' if d.get('won') else 'lost'} against {d.get('opponent') or '?'}"
               + (f" ({d['opponent_rating']})" if d.get("opponent_rating") else "")
               + (f", {before} to {d['rating_after']}" if before is not None else f", {d['rating_after']}"))
        kind = "win" if d.get("won") else "loss"
        if d.get("arm"):
            # The arm's colour, filled for a win and hollow for a loss.
            kind = f"arm a{arms.index(arm_of(d)) % ARM_COLOURS}" + ("" if d.get("won") else " lost")
            tip += f", played as {arm_of(d)}"
        out.append(f'<circle class="{kind}" cx="{x(i + 1):.1f}" cy="{y(series[i]):.1f}" r="{radius}">'
                   f'<title>{html.escape(tip)}</title></circle>')
    out.append("</svg>")
    return "".join(out)


CSS = """
:root { --bg:#f6f7f9; --card:#ffffff; --ink:#1b1f24; --dim:#667080; --grid:#e3e6eb; --line:#8aa0b8;
        --roll:#e8590c; --win:#2f9e44; --loss:#e03131; --band:rgba(28,126,214,.13); --level:#1c7ed6; }
@media (prefers-color-scheme: dark) {
 :root { --bg:#12151a; --card:#1a1f26; --ink:#e7eaee; --dim:#8b96a5; --grid:#2a313b; --line:#5f738a;
         --roll:#ff922b; --win:#51cf66; --loss:#ff6b6b; --band:rgba(77,171,247,.16); --level:#4dabf7; } }
* { box-sizing:border-box; }
body { margin:0; padding:20px 16px 40px; background:var(--bg); color:var(--ink);
       font:15px/1.45 "Segoe UI", system-ui, sans-serif; }
main { max-width:1140px; margin:0 auto; }
h1 { font-size:20px; margin:0 0 2px; }
.sub { color:var(--dim); margin:0 0 14px; font-size:13px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin-bottom:12px; }
.card { background:var(--card); border-radius:10px; padding:10px 14px; }
.card b { display:block; font-size:22px; font-variant-numeric:tabular-nums; }
.card span { color:var(--dim); font-size:12px; }
.verdict { background:var(--card); border-radius:10px; padding:12px 14px; margin-bottom:12px; }
.verdict b { font-size:17px; margin-right:8px; }
.plot { background:var(--card); border-radius:10px; padding:10px 8px 4px; }
svg { width:100%; height:auto; display:block; }
.grid { stroke:var(--grid); stroke-width:1; }
.tick { fill:var(--dim); font-size:12px; }
.rating { fill:none; stroke:var(--line); stroke-width:1.4; }
.roll { fill:none; stroke:var(--roll); stroke-width:3; stroke-linejoin:round; stroke-linecap:round; }
.win { fill:var(--win); } .loss { fill:var(--loss); }
.band { fill:var(--band); } .level { stroke:var(--level); stroke-width:1.5; stroke-dasharray:6 5; }
.leveltext { fill:var(--level); font-size:12px; }
.change { stroke:var(--dim); stroke-width:1; stroke-dasharray:3 4; }
.key { color:var(--dim); font-size:12.5px; margin:8px 6px 6px; }
.key i { display:inline-block; width:22px; height:0; border-top:3px solid var(--roll); vertical-align:middle; margin:0 5px 0 12px; }
.key i.thin { border-top:1.4px solid var(--line); } .key i.dash { border-top:1.5px dashed var(--level); }
.key u { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--win); margin:0 5px 0 12px; }
.key u.l { background:var(--loss); }
.note { color:var(--dim); font-size:12.5px; margin:10px 4px; }
.tablewrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; margin-top:14px; background:var(--card); border-radius:10px; overflow:hidden; font-variant-numeric:tabular-nums; }
th, td { padding:6px 10px; text-align:right; font-size:13px; white-space:nowrap; }
th { color:var(--dim); font-weight:600; border-bottom:1px solid var(--grid); }
td.t, th.t { text-align:left; }
td.w { color:var(--win); font-weight:600; } td.x { color:var(--loss); font-weight:600; }
tr + tr td { border-top:1px solid var(--grid); }
"""

# Only in a page whose games have arms. The colours keep clear of the win and loss colours.
ARM_CSS = """
:root { --a0:#1c7ed6; --a1:#9c36b5; --a2:#0c8599; --a3:#e67700; --a4:#5c7cfa; --a5:#862e9c; }
@media (prefers-color-scheme: dark) {
 :root { --a0:#4dabf7; --a1:#da77f2; --a2:#3bc9db; --a3:#ffa94d; --a4:#91a7ff; --a5:#e599f7; } }
.arm { stroke-width:1.6; }
.a0 { fill:var(--a0); stroke:var(--a0); } .a1 { fill:var(--a1); stroke:var(--a1); } .a2 { fill:var(--a2); stroke:var(--a2); }
.a3 { fill:var(--a3); stroke:var(--a3); } .a4 { fill:var(--a4); stroke:var(--a4); } .a5 { fill:var(--a5); stroke:var(--a5); }
.arm.lost { fill:var(--card); }
.key u.a0, td u.a0 { background:var(--a0); } .key u.a1, td u.a1 { background:var(--a1); } .key u.a2, td u.a2 { background:var(--a2); }
.key u.a3, td u.a3 { background:var(--a3); } .key u.a4, td u.a4 { background:var(--a4); } .key u.a5, td u.a5 { background:var(--a5); }
.key u.hollow { background:transparent; border:1.6px solid var(--dim); }
td u { display:inline-block; width:9px; height:9px; border-radius:50%; margin:0 7px 0 0; }
h2 { font-size:15px; margin:16px 4px 0; }
table.arms { margin-top:8px; }
"""


def _arms_block(games: List[Dict[str, Any]], mixed: bool = False, names: Optional[List[str]] = None) -> Tuple[str, str]:
    """(the legend line under the graph, the table of arms). Two empty strings when no game has an arm."""
    rows = arm_rows(games, names)
    if not rows:
        return "", ""
    legend = ('<p class="key">dots by arm:' + "".join(f'<u class="a{r["colour"]}"></u>{html.escape(r["arm"])}' for r in rows)
              + '<u class="hollow"></u>hollow: a loss'
              + ('<u></u><u class="l"></u>green and red: games without an arm' if mixed else "")
              + "</p>")
    body: List[str] = []
    for r in rows:
        n = r["games"]
        level = f'{r["level"][0]:.0f} &#177; {r["level"][1]:.0f}' if r["level"] else "?"
        search = "none"
        if r["searched"] or r["fallbacks"]:
            search = f'{r["searched"]} of {r["decisions"]} decisions'
        per = f'{r["iterations_per_decision"]:,}' if r["iterations_per_decision"] else ""
        tera = f'{r["tera_games"]} of {n}' + (f', turn {r["tera_turn"]}' if r["tera_turn"] is not None else "")
        body.append(f'<tr><td class="t"><u class="a{r["colour"]}"></u>{html.escape(r["arm"])}</td><td>{n}</td>'
                    f'<td>{r["wins"]} - {r["losses"]}</td><td>{100.0 * r["wins"] / n:.0f}%</td><td>{level}</td>'
                    f'<td>{search}</td><td>{per}</td><td>{r["fallbacks"]}</td><td>{r["overruled"]}</td><td>{tera}</td></tr>')
    table = ('<h2>By arm</h2><div class="tablewrap"><table class="arms"><tr><th class="t">arm</th><th>games</th>'
             '<th>wins - losses</th><th>won</th><th>level its results point to</th><th>answered by the search</th>'
             '<th>iterations per decision</th><th>search failed, network played</th><th>network overruled</th>'
             '<th>Terastallized (games, median turn)</th></tr>' + "".join(body) + "</table></div>"
             '<p class="note">An arm is the way a game was played: net is the network alone, search is the choice of the '
             'look-ahead search, mix is the two combined. Every new game is given one of the arms in turn, so the arms meet '
             'the same opponents at the same ratings. The table counts the games of the newest network only, and the level '
             'of an arm is worked out from its own games, so with few games its margin is wide: two arms differ when their '
             'margins do not overlap.</p>')
    return legend, table


def render(games: List[Dict[str, Any]], server: str, account: Optional[str], refresh: int = 15) -> str:
    wins = sum(1 for d in games if d.get("won"))
    rated = [int(d["rating_after"]) for d in games if d.get("rating_after") is not None]
    start = next((int(d["rating_before"]) for d in games if d.get("rating_before") is not None), None)
    last = games[-20:]
    last_wins = sum(1 for d in last if d.get("won"))
    current = newest_network_games(games)
    perf = performance(current) if len(current) >= BAND_GAMES else None
    word, sentence = verdict(games)
    where = "the public ladder" if server == "public" else server
    # Arms: the dots are coloured over all the games shown, the table is of the newest network's games.
    arm_legend, _ = _arms_block(games, mixed=any(not d.get("arm") for d in games))
    _, arm_table = _arms_block(current, names=arm_names(games))
    css = CSS + (ARM_CSS if arm_legend else "")

    def card(value: str, label: str) -> str:
        return f'<div class="card"><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>'

    cards = [
        card(str(rated[-1]) if rated else "?", "rating now" + (f", started at {start}" if start is not None else "")),
        card(str(max(rated + ([start] if start is not None else []))) if rated else "?", "highest so far"),
        card(f"{wins} - {len(games) - wins}", f"wins - losses ({100.0 * wins / len(games):.0f}% of {len(games)})" if games else "wins - losses"),
        card(f"{last_wins} - {len(last) - last_wins}", f"last {len(last)} games"),
        card(f"{perf[0]:.0f} ± {perf[1]:.0f}" if perf else "?", "level the results point to"),
    ]
    rows: List[str] = []
    numbered = list(enumerate(games, 1))
    for i, d in reversed(numbered[-TABLE_ROWS:]):
        before, after = d.get("rating_before"), d.get("rating_after")
        opp = str(d.get("opponent") or "?") + (f" ({d['opponent_rating']})" if d.get("opponent_rating") else "")
        change = f"{int(after) - int(before):+d}" if after is not None and before is not None else ""
        rows.append(f'<tr><td>{i}</td><td class="t">{html.escape(_local(d.get("time")))}</td>'
                    f'<td class="t {"w" if d.get("won") else "x"}">{"WIN" if d.get("won") else "LOSS"}</td>'
                    f'<td>{html.escape(str(d.get("turns", "?")))}</td><td class="t">{html.escape(opp)}</td>'
                    f'<td>{after if after is not None else "?"}</td><td>{change}</td>'
                    f'<td class="t">{html.escape(_network(d.get("checkpoint")))}</td></tr>')
    table = ('<div class="tablewrap"><table><tr><th>#</th><th class="t">when</th><th class="t">result</th><th>turns</th>'
             '<th class="t">opponent (rating)</th><th>rating</th><th>change</th><th class="t">played by</th></tr>'
             + "".join(rows) + "</table></div>") if rows else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{int(refresh)}">
<title>PSBot network Elo</title><style>{css}</style></head>
<body><main>
<h1>PSBot network on {html.escape(where)}</h1>
<p class="sub">{html.escape("account " + account + " · " if account else "")}updated {time.strftime("%d %b %H:%M:%S")} · this page reloads itself every {int(refresh)} seconds</p>
<div class="cards">{"".join(cards)}</div>
<div class="verdict"><b>{html.escape(word)}</b>{html.escape(sentence)}</div>
<div class="plot">{_svg(games)}
<p class="key"><i></i>average of the last {ROLL} games<i class="thin"></i>rating after each game<u></u>win<u class="l"></u>loss<i class="dash"></i>level the results point to, with its margin shaded</p>{arm_legend}</div>{arm_table}
<p class="note">How to read it. One game moves the rating about 20 points, and a rating that has settled still wanders about 60 either way, so follow the thick line and not single games. The level is the rating at which this record against these opponents would be exactly the expected one. It is worked out from the games of the newest network only and does not lag the way the rating does. While the thick line is well below the shaded band the rating is still climbing, and once it moves around the band it has plateaued.</p>
{table}
</main></body></html>
"""


def write_page(games: List[Dict[str, Any]], path: str, server: str, account: Optional[str] = None) -> str:
    """Write the page next to the results, whole or not at all."""
    page = render(games, server, account)
    # A name of its own: a running bot and the app's "Elo graph" button may write the page at once.
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(page)
    try:
        os.replace(tmp, path)
    except OSError:   # Windows: the other writer's replace, or a browser reading it, held the file a moment
        time.sleep(0.5)
        os.replace(tmp, path)
    return path
