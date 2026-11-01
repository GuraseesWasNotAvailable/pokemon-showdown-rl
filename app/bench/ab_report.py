"""
The ladder A/B of one experiment, read from runs/ladder/results.jsonl, with its pre-registered verdict.

    python bench/ab_report.py --experiment ab3-net-mix-deep-024381
    python bench/ab_report.py --experiment ab3-net-mix-deep-024381 --results runs/ladder/results.jsonl --json

Per arm: games, wins, the raw win rate with its 95% interval (Wilson), the mean opponent rating,
the p1/p2 split, forfeits (the opponent's, ours), losses on time, the typing limit's notices and
the orders sent again, the order spacing's longest wait, the search's iterations, worlds and
full-width turns ahead per decision (search_player.full_width_turns), and how often the deep arm
fell back to today's budget. Then every arm against the control (mix): the difference in raw win
rate, its standard error and z, and the verdict the plan fixed before the first game
(scratchpad better/PLAN.md, "Measurement plan", and the owner's third arm):

  * 500 games per arm, fixed. One interim look at 250 per arm: an arm stops early only at
    |z| >= 2.8 against mix. The looks use each arm's FIRST 250 and FIRST 500 games with a result,
    in the order they were played, so reading the report more often is not more looks.
  * The yardstick is the raw win rate, arm minus mix, with its standard error. Not the rating.
  * The arm with the highest raw win rate is kept; of the arms within 1 point of that highest
    rate, the cheapest is kept (net over mix over mix-deep). A difference is significant only at
    |z| >= 2.
  * Valid only when every arm's mean opponent rating is within 15 points of mix's (checked on the
    same games as the look: the first 250 or 500 per arm; printed with the gap's standard error
    and z, so that a gap of 16 at z 1.5 reads as chance), and when the search kept its compute:
    mix's iterations per decision within 10% of today's (--mix-iterations-ref; today's is 538,000,
    the 90 games of mix on 024381 at 3 games at a time on 2026-10-08, weighted by searched
    decisions as here), mix-deep's about 5 times mix's (flagged under 4 times).
  * The compute check that answers "does mix lose compute because the arms share the helpers"
    directly: mix's iterations per decision when a deep search ran beside it (half its time or
    more) against when none did (search_player._search_done, results "it_deep_beside" and
    "it_no_deep"). Flagged when the first is more than 10% under the second, with 200 or more
    decisions on each side. Same positions on average (which game happens to search beside which
    is chance), so unlike the comparison with today's figure it does not depend on the positions.

The "turns" column and mix-deep's ratio to mix compare different positions across arms
(diagnostics, unpaired): the depth evidence is bench/ab_deep_measure.py's paired measurement.

The Luna arm (ladder.py --arms mix,luna --experiment luna-vs-mix-024381; luna_player.py) is read
by the same rules, with mix as the control, and counted as the most expensive arm (the tie rule
keeps mix within 1 point). One more line says what Luna did in its games: the decisions it was
asked, answered and changed, the ones the blunder guard overruled, and the ones mix played
instead with the reason. A Luna arm whose answers were mostly missing is today's bot under
Luna's name, and is flagged.

Pure JSON and the standard library: nothing is started, nothing is written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "runs", "ladder", "results.jsonl")

# What an arm costs, cheapest first: the tie rule keeps the cheaper one (net over mix, mix over mix-deep).
COST = {"net": 0, "mix": 1, "mix-deep": 2, "mix-lines": 2.5, "luna": 3, "luna-lines": 4, "sol": 5, "sol-lines": 6,
        "luna-solo": 0.5, "sol-solo": 4.5}
LUNA_SUMS = ("luna_decisions", "luna_asked", "luna_answered", "luna_played", "luna_overruled", "luna_agreed_with_mix",
             "luna_changed", "luna_tokens")
LUNA_MIN_PLAYED = 0.80      # of the Luna arm's decisions with a choice: under this, its games are mostly today's bot

DEFAULTS = {"control": "mix", "interim": 250, "final": 500, "stop_z": 2.8, "sig_z": 2.0, "tie_points": 1.0,
            "max_rating_gap": 15.0, "mix_iterations_ref": 538_000, "iterations_tolerance": 0.10, "deep_ratio_min": 4.0,
            "paired_min_decisions": 200}


def load(path: str, experiment: str, checkpoint: Optional[str] = None) -> List[Dict[str, Any]]:
    """The result lines of one experiment, in the order they were written."""
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or d.get("experiment") != experiment or not d.get("arm"):
                continue
            if checkpoint and d.get("checkpoint") != checkpoint:
                continue
            rows.append(d)
    return rows


def wilson(wins: int, n: int, z: float = 1.959964) -> List[Optional[float]]:
    if n <= 0:
        return [None, None]
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def _weighted(rows: Iterable[Dict[str, Any]], key: str, weight: str = "searched") -> Optional[float]:
    total = n = 0.0
    for d in rows:
        value, w = d.get(key), d.get(weight) or 0
        if isinstance(value, (int, float)) and w:
            total += float(value) * float(w)
            n += float(w)
    return total / n if n else None


def arm_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One arm's numbers over its result lines (games with no result seen are counted apart)."""
    decided = [d for d in rows if d.get("won") in (True, False)]
    n = len(decided)
    wins = sum(1 for d in decided if d.get("won"))
    ratings = [float(d["opponent_rating"]) for d in decided if isinstance(d.get("opponent_rating"), (int, float))]
    roles: Dict[str, Dict[str, int]] = {}
    for d in decided:
        role = str(d.get("role") or "?")
        r = roles.setdefault(role, {"games": 0, "wins": 0})
        r["games"] += 1
        r["wins"] += int(bool(d.get("won")))
    ending = lambda how, by: sum(1 for d in decided if d.get("ending") == how and d.get("ending_by") == by)  # noqa: E731
    searched = sum(int(d.get("searched") or 0) for d in decided)
    mean_rating = sum(ratings) / len(ratings) if ratings else None
    sd_rating = (math.sqrt(sum((r - mean_rating) ** 2 for r in ratings) / (len(ratings) - 1))
                 if mean_rating is not None and len(ratings) > 1 else None)
    paired = {name: sum(int(d.get(name) or 0) for d in decided) for name in ("it_deep_beside", "n_deep_beside", "it_no_deep", "n_no_deep")}
    return {
        "games": n, "wins": wins, "win_rate": round(wins / n, 4) if n else None, "ci95": wilson(wins, n),
        "no_result": len(rows) - n,
        # picked up again after a lost connection (the search lets such a game go: the network plays the rest)
        "rejoined": sum(1 for d in decided if d.get("rejoined")),
        "opp_rating_mean": round(mean_rating, 1) if mean_rating is not None else None, "opp_rated": len(ratings),
        "opp_rating_sd": round(sd_rating, 1) if sd_rating is not None else None,
        # the paired compute check: mean iterations with a deep search beside, and with none
        "iterations_deep_beside": round(paired["it_deep_beside"] / paired["n_deep_beside"]) if paired["n_deep_beside"] else None,
        "decisions_deep_beside": paired["n_deep_beside"],
        "iterations_no_deep": round(paired["it_no_deep"] / paired["n_no_deep"]) if paired["n_no_deep"] else None,
        "decisions_no_deep": paired["n_no_deep"],
        "roles": roles,
        "their_forfeits": ending("forfeit", "them"), "our_forfeits": ending("forfeit", "us"),
        "timer_losses": ending("inactivity", "us"), "their_timer_losses": ending("inactivity", "them"),
        "throttle_notices": sum(int(d.get("throttle_notices") or 0) for d in decided),
        "throttle_resends": sum(int(d.get("throttle_resends") or 0) for d in decided),
        "order_wait_ms_max": max([float(d.get("order_wait_ms_max") or 0.0) for d in decided] or [0.0]),
        "decisions": sum(int(d.get("decisions") or 0) for d in decided), "searched": searched,
        "search_failures": sum(int(d.get("fallbacks") or 0) for d in decided),
        "late": sum(int(d.get("late") or 0) for d in decided),
        "iterations_per_decision": round(_weighted(decided, "iterations_per_decision")) if searched else None,
        "worlds_per_decision": (round(_weighted(decided, "worlds_per_decision"), 2)
                                if _weighted(decided, "worlds_per_decision") is not None else None),
        "full_width_turns": (round(_weighted(decided, "full_width_turns"), 2)
                             if _weighted(decided, "full_width_turns") is not None else None),
        "decision_ms_mean": (round(_weighted(decided, "decision_ms_mean", "decisions"))
                             if _weighted(decided, "decision_ms_mean", "decisions") is not None else None),
        "deep_fallbacks": sum(1 for d in decided if d.get("deep_fallback")),
        "luna": luna_stats(decided),
    }


def luna_stats(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What Luna did in an arm's games (None for an arm without Luna): the sums of the results lines' luna_* counts."""
    rows = [d for d in rows if "luna_decisions" in d]
    if not rows:
        return None
    out: Dict[str, Any] = {key[len("luna_"):]: sum(int(d.get(key) or 0) for d in rows) for key in LUNA_SUMS}
    fallbacks: Dict[str, int] = {}
    for d in rows:
        for why, k in (d.get("luna_fallbacks") or {}).items():
            fallbacks[why] = fallbacks.get(why, 0) + int(k or 0)
    asked = sum(int(d.get("luna_asked") or 0) for d in rows if d.get("luna_latency_ms_mean") is not None)
    out.update(games=len(rows), fallbacks=fallbacks, cost_usd=round(out["tokens"] * 0.10 / 1e6, 4),
               latency_ms_mean=(round(sum(float(d["luna_latency_ms_mean"]) * int(d.get("luna_asked") or 0) for d in rows
                                          if d.get("luna_latency_ms_mean") is not None) / asked) if asked else None),
               played_share=round(out["played"] / out["decisions"], 4) if out["decisions"] else None)
    return out


def where_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Where the games' search ran (ladder.py's "search_at": "remote" on the box over ssh, "wsl",
    "local"; games written before the field have none) and, for the remote ones, the median of
    the round trip measured when their run's helpers started ("rtt_ms").
    """
    out: Dict[str, Dict[str, Any]] = {}
    for d in rows:
        place = str(d.get("search_at") or "not recorded")
        w = out.setdefault(place, {"games": 0, "rtt": []})
        w["games"] += 1
        if isinstance(d.get("rtt_ms"), (int, float)):
            w["rtt"].append(float(d["rtt_ms"]))
    for w in out.values():
        trips = sorted(w.pop("rtt"))
        w["rtt_ms_median"] = round(trips[len(trips) // 2], 1) if trips else None
    return out


def compare(a: Dict[str, Any], c: Dict[str, Any]) -> Dict[str, Any]:
    """Arm minus control, raw win rate: difference, standard error (unpooled), z."""
    if not a["games"] or not c["games"]:
        return {"diff": None, "se": None, "z": None}
    pa, pc = a["wins"] / a["games"], c["wins"] / c["games"]
    se = math.sqrt(pa * (1 - pa) / a["games"] + pc * (1 - pc) / c["games"])
    diff = pa - pc
    z = diff / se if se > 0 else (0.0 if diff == 0 else math.copysign(float("inf"), diff))
    return {"diff": round(diff, 4), "se": round(se, 4), "z": round(z, 3) if math.isfinite(z) else z}


def first_n(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """The first n lines with a result, in the order they were played."""
    out = [d for d in rows if d.get("won") in (True, False)]
    return out[:n]


def _pts(x: Optional[float]) -> str:
    return "?" if x is None else f"{100.0 * x:+.1f}"


def report(rows: List[Dict[str, Any]], settings: Optional[Dict[str, Any]] = None, arms: Optional[List[str]] = None) -> Dict[str, Any]:
    """Everything the report prints, as a dict: per arm, the comparisons, the looks, the verdict lines and the warnings."""
    s = dict(DEFAULTS, **(settings or {}))
    control = s["control"]
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for d in rows:
        by_arm.setdefault(str(d["arm"]), []).append(d)
    names = list(arms) if arms else sorted(by_arm, key=lambda a: (COST.get(a, 1.5), a))
    for name in names:
        by_arm.setdefault(name, [])
    stats = {a: arm_stats(by_arm[a]) for a in names}
    others = [a for a in names if a != control]
    out: Dict[str, Any] = {"settings": s, "arms": names, "stats": stats, "now": {}, "looks": {}, "verdict": [], "warnings": [],
                           "search_at": where_stats(rows)}
    if control not in stats:
        out["verdict"].append(f"no arm {control!r} in this experiment: nothing to compare against")
        return out
    for a in others:
        out["now"][a] = compare(stats[a], stats[control])

    # Validity: the search's compute where it was (the opponents' strength: after the looks, on their games).
    c = stats[control]
    it = c["iterations_per_decision"]
    ref = float(s["mix_iterations_ref"] or 0)
    if it is not None and ref > 0 and abs(it - ref) > s["iterations_tolerance"] * ref:
        out["warnings"].append(f"COMPUTE: {control} searched {it:,} iterations per decision, {100.0 * (it - ref) / ref:+.0f}% against "
                               f"today's {ref:,.0f} (more than {100 * s['iterations_tolerance']:.0f}% off): {control} is not today's mix")
    if "mix-deep" in stats and it:
        deep = stats["mix-deep"]["iterations_per_decision"]
        if deep is not None:
            ratio = deep / it
            out["deep_ratio"] = round(ratio, 2)
            if ratio < s["deep_ratio_min"]:
                out["warnings"].append(f"COMPUTE: mix-deep searched {ratio:.1f} times mix's iterations per decision (meant: about 5, "
                                       f"flagged under {s['deep_ratio_min']:.0f}; unpaired: the arms met different positions)")
        if stats["mix-deep"]["games"] and stats["mix-deep"]["deep_fallbacks"]:
            share = stats["mix-deep"]["deep_fallbacks"] / stats["mix-deep"]["games"]
            if share > 0.05:
                out["warnings"].append(f"mix-deep fell back to today's budget in {stats['mix-deep']['deep_fallbacks']} games "
                                       f"({100 * share:.0f}%): read the console lines for why")
    # The paired check: mix with a deep search beside it against mix with none (same positions on average).
    beside, alone = c.get("iterations_deep_beside"), c.get("iterations_no_deep")
    if beside is not None and alone:
        out["paired_compute"] = {"with_deep": beside, "without": alone, "ratio": round(beside / alone, 3),
                                 "decisions_with_deep": c["decisions_deep_beside"], "decisions_without": c["decisions_no_deep"]}
        enough = min(c["decisions_deep_beside"], c["decisions_no_deep"]) >= s["paired_min_decisions"]
        if enough and beside < (1.0 - s["iterations_tolerance"]) * alone:
            out["warnings"].append(f"COMPUTE: {control} searched {beside:,} iterations per decision with a deep search beside it and "
                                   f"{alone:,} with none ({100.0 * (beside - alone) / alone:+.0f}%, more than "
                                   f"{100 * s['iterations_tolerance']:.0f}% under): the deep arm takes compute from {control}")
    for a in names:
        if stats[a]["timer_losses"]:
            out["warnings"].append(f"{a} lost {stats[a]['timer_losses']} game(s) on time")
        lu = stats[a].get("luna")
        if lu and lu["decisions"] >= 100 and lu["played_share"] is not None and lu["played_share"] < LUNA_MIN_PLAYED:
            out["warnings"].append(f"{a}: Luna's choice was the order on only {100 * lu['played_share']:.0f}% of its decisions "
                                   f"(under {100 * LUNA_MIN_PLAYED:.0f}%): the rest were {control}'s, so this arm is mostly {control}")

    def finish(basis: Dict[str, Dict[str, Any]], label: str) -> Dict[str, Any]:
        """The opponents' strength, on the games the verdict reads (label), with the gap's SE and z."""
        out["rating_gaps"] = {"games": label}
        for a in others:
            g = rating_gap(basis[a], basis[control])
            out["rating_gaps"][a] = g
            if g is not None and abs(g["gap"]) > s["max_rating_gap"]:
                chance = (f"SE {g['se']:.1f}, z {g['z']:+.2f}" + (": a gap this size comes by chance" if abs(g["z"]) < 2.0 else "")
                          if g.get("z") is not None else "no SE")
                out["warnings"].append(f"INVALID: {a}'s opponents averaged {basis[a]['opp_rating_mean']:.0f}, {control}'s "
                                       f"{basis[control]['opp_rating_mean']:.0f} ({label}): a gap of {abs(g['gap']):.0f} (over "
                                       f"{s['max_rating_gap']:.0f}; {chance}); the arms did not meet the same ladder")
        if "kept" in out and any(w.startswith("INVALID") for w in out["warnings"]):
            out["verdict"].append("but the run is INVALID (see the warnings): do not act on it")
        return out

    # The looks, on each arm's first games in the order they were played.
    least = min(stats[a]["games"] for a in names)
    interim, final = int(s["interim"]), int(s["final"])
    stopped: Dict[str, str] = {}
    if least >= interim:
        look = {a: arm_stats(first_n(by_arm[a], interim)) for a in names}
        out["looks"]["interim"] = {a: compare(look[a], look[control]) for a in others}
        for a in others:
            z = out["looks"]["interim"][a]["z"]
            if z is not None and abs(z) >= s["stop_z"]:
                stopped[a] = "better" if z > 0 else "worse"
                out["verdict"].append(f"interim look at {interim} per arm: STOP {a}: {a} - {control} = "
                                      f"{_pts(out['looks']['interim'][a]['diff'])} points (SE {100 * out['looks']['interim'][a]['se']:.1f}, "
                                      f"z {z:+.2f}, |z| >= {s['stop_z']}): {a} is {stopped[a]} than {control}")
            else:
                out["verdict"].append(f"interim look at {interim} per arm: {a} - {control} = {_pts(out['looks']['interim'][a]['diff'])} "
                                      f"points (z {z:+.2f}): under {s['stop_z']}, {a} goes on to {final} per arm")
        out["stopped"] = stopped
    else:
        out["verdict"].append(f"before the interim look: {least} of {interim} games in the arm with the fewest; "
                              f"the numbers above are for information only, no verdict")
        return finish(stats, "all games so far")

    going = [a for a in others if a not in stopped]
    if going and min(stats[a]["games"] for a in going + [control]) < final:
        n_least = min(stats[a]['games'] for a in going + [control])
        out["verdict"].append(f"before the final look: {n_least} of {final} games in the arm with the fewest still running")
        return finish(look, f"the interim look's first {interim} per arm")

    # The final look: each arm's first `final` games (an arm stopped at the interim keeps its interim numbers).
    fin = {a: arm_stats(first_n(by_arm[a], interim if a in stopped else final)) for a in names}
    out["looks"]["final"] = {a: compare(fin[a], fin[control]) for a in others}
    rates = {a: fin[a]["win_rate"] for a in names if fin[a]["win_rate"] is not None and stopped.get(a) != "worse"}
    top = max(rates, key=lambda a: (rates[a], -COST.get(a, 1.5)))
    tie = s["tie_points"] / 100.0
    # The cheapest arm within 1 point of the highest raw win rate (net over mix over mix-deep).
    # Fixed by the owner before game 1: a cheaper arm wins a tie with the BEST arm only, so net at
    # 49.2%, mix 50.0%, mix-deep 50.8% keeps mix (net is 1.6 points below the best).
    close = [a for a in rates if rates[top] - rates[a] <= tie + 1e-12]
    best = min(close, key=lambda a: (COST.get(a, 1.5), -rates[a]))
    if best == top:
        why = f"the highest raw win rate ({100 * rates[best]:.1f}%)"
    else:
        why = (f"{best} is within {s['tie_points']:.0f} point of the highest, {top} "
               f"({_pts(rates[best] - rates[top])}): the cheaper {best}")
    out["kept"] = best
    for a in others:
        cmp_ = out["looks"]["final"][a]
        sig = cmp_["z"] is not None and abs(cmp_["z"]) >= s["sig_z"]
        out["verdict"].append(f"final ({'interim, stopped' if a in stopped else str(final) + ' per arm'}): {a} - {control} = "
                              f"{_pts(cmp_['diff'])} points, SE {100 * (cmp_['se'] or 0):.1f}, z {cmp_['z']:+.2f}: "
                              + ("significant" if sig else f"not significant (|z| under {s['sig_z']})"))
    out["verdict"].append(f"KEEP {best}: {why}")
    return finish(fin, f"the final look's first {final} per arm")


def rating_gap(a: Dict[str, Any], c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """An arm's mean opponent rating minus the control's, with its standard error (unpooled) and z."""
    ra, rc = a.get("opp_rating_mean"), c.get("opp_rating_mean")
    if ra is None or rc is None:
        return None
    gap = ra - rc
    sa, sc, na, nc = a.get("opp_rating_sd"), c.get("opp_rating_sd"), a.get("opp_rated") or 0, c.get("opp_rated") or 0
    if sa is None or sc is None or na < 2 or nc < 2:
        return {"gap": round(gap, 1), "se": None, "z": None}
    se = math.sqrt(sa * sa / na + sc * sc / nc)
    return {"gap": round(gap, 1), "se": round(se, 1), "z": round(gap / se, 2) if se > 0 else None}


def text(out: Dict[str, Any], experiment: str) -> str:
    s, stats = out["settings"], out["stats"]
    lines = [f"A/B {experiment}: {sum(st['games'] for st in stats.values())} games with a result"]
    head = (f"  {'arm':<10} {'games':>5} {'wins':>5} {'win rate':>9} {'95% interval':>15} {'opp rating':>10} {'p1 (won)':>10} "
            f"{'p2 (won)':>10} {'forfeits them/us':>16} {'timer losses':>12} {'resends':>7} {'iter/dec':>10} {'worlds':>6} {'turns':>5} {'fell back':>9}")
    lines.append(head)
    for a in out["arms"]:
        st = stats[a]
        ci = st["ci95"]
        p1, p2 = st["roles"].get("p1", {}), st["roles"].get("p2", {})
        lines.append(
            f"  {a:<10} {st['games']:>5} {st['wins']:>5} {('%.1f%%' % (100 * st['win_rate'])) if st['win_rate'] is not None else '-':>9} "
            f"{('%.1f to %.1f' % (100 * ci[0], 100 * ci[1])) if ci[0] is not None else '-':>15} "
            f"{('%.0f' % st['opp_rating_mean']) if st['opp_rating_mean'] is not None else '-':>10} "
            f"{('%d (%d)' % (p1.get('games', 0), p1.get('wins', 0))):>10} {('%d (%d)' % (p2.get('games', 0), p2.get('wins', 0))):>10} "
            f"{('%d/%d' % (st['their_forfeits'], st['our_forfeits'])):>16} {st['timer_losses']:>12} {st['throttle_resends']:>7} "
            f"{('{:,}'.format(st['iterations_per_decision'])) if st['iterations_per_decision'] is not None else '-':>10} "
            f"{st['worlds_per_decision'] if st['worlds_per_decision'] is not None else '-':>6} "
            f"{st['full_width_turns'] if st['full_width_turns'] is not None else '-':>5} {st['deep_fallbacks']:>9}")
    extra = [f"{a}: {stats[a]['no_result']} without a result" for a in out["arms"] if stats[a]["no_result"]]
    if extra:
        lines.append("  (" + "; ".join(extra) + ", not counted)")
    rejoined = [f"{a} {stats[a]['rejoined']}" for a in out["arms"] if stats[a].get("rejoined")]
    if rejoined:
        lines.append("  games picked up again after a lost connection (the network played the rest, counted in their arm): "
                     + ", ".join(rejoined))
    places = out.get("search_at") or {}
    if places:
        names = {"remote": "on the box over ssh", "wsl": "inside WSL on the laptop", "local": "on the machine that played",
                 "not recorded": "not recorded (written before ladder.py kept it)"}
        lines.append("  where the search ran: " + "; ".join(
            f"{names.get(p, p)} {w['games']} games" + (f" (round trip {w['rtt_ms_median']:.0f} ms, median of the runs' starts)"
                                                       if w.get("rtt_ms_median") is not None else "")
            for p, w in sorted(places.items(), key=lambda kv: -kv[1]["games"])))
    lines.append(f"  typing-limit notices per arm: " + ", ".join(f"{a} {stats[a]['throttle_notices']}" for a in out["arms"])
                 + "; the order spacing's longest wait: " + ", ".join(f"{a} {stats[a]['order_wait_ms_max']:.0f} ms" for a in out["arms"]))
    for a in out["arms"]:
        lu = stats[a].get("luna")
        if lu:
            fell = sum(lu["fallbacks"].values())
            lines.append(
                f"  Luna in {a}'s {lu['games']} games: asked on {lu['asked']} of {lu['decisions']} decisions with a choice, answered "
                f"{lu['answered']} (played {lu['played']}, overruled by the blunder guard {lu['overruled']}); its pick was "
                f"{s['control']}'s own on {lu['agreed_with_mix']}, another action on {lu['answered'] - lu['agreed_with_mix']}; "
                f"{s['control']} played instead on {fell}"
                + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(lu["fallbacks"].items())) + ")" if fell else "")
                + (f"; {lu['latency_ms_mean']} ms per answer" if lu["latency_ms_mean"] is not None else "")
                + f"; {lu['tokens']:,} input tokens, about ${lu['cost_usd']:.2f}")
    pc = out.get("paired_compute")
    if pc:
        lines.append(f"  compute, paired: {s['control']} searched {pc['with_deep']:,} iterations per decision with a deep search beside it "
                     f"({pc['decisions_with_deep']} decisions) and {pc['without']:,} with none ({pc['decisions_without']}): "
                     f"{100.0 * (pc['ratio'] - 1.0):+.1f}% (flagged under -{100 * s['iterations_tolerance']:.0f}% with "
                     f"{s['paired_min_decisions']} or more on each side)")
    c_it = stats.get(s["control"], {}).get("iterations_per_decision")
    if c_it:
        lines.append(f"  compute against today's: {s['control']} {c_it:,} iterations per decision, today's {s['mix_iterations_ref']:,.0f} "
                     f"({100.0 * (c_it - s['mix_iterations_ref']) / s['mix_iterations_ref']:+.0f}%; single games range about 200,000 "
                     f"to 800,000, so read it after 70 or more {s['control']} games)")
    if "deep_ratio" in out:
        lines.append(f"  mix-deep searched {out['deep_ratio']:.1f} times mix's iterations per decision (a diagnostic: unpaired, the arms "
                     f"met different positions; the paired measurement on the box gave 4.9 to 5.0 times)")
    lines.append("  turns (a diagnostic, unpaired): full-width turns ahead per decision, log(iterations per world) / log(our options"
                 " x theirs), on the positions each arm met. The depth evidence is bench/ab_deep_measure.py's paired measurement"
                 " (fixed depth 2.92 to 2.98 for mix, 3.23 for mix-deep on the same positions)")
    gaps = out.get("rating_gaps") or {}
    if gaps:
        parts = []
        for a in out["arms"]:
            g = gaps.get(a)
            if isinstance(g, dict):
                parts.append(f"{a} {g['gap']:+.0f}" + (f" (SE {g['se']:.1f}, z {g['z']:+.2f})" if g.get("z") is not None else ""))
        if parts:
            lines.append(f"  opponents' rating against {s['control']}'s, on {gaps.get('games')}: " + ", ".join(parts)
                         + f" (valid within {s['max_rating_gap']:.0f})")
    lines.append("")
    lines.append(f"now (all games so far, for information): " + "; ".join(
        f"{a} - {s['control']} = {_pts(c['diff'])} points (SE {100 * c['se']:.1f}, z {c['z']:+.2f})" if c["diff"] is not None else f"{a}: -"
        for a, c in out["now"].items()))
    for w in out["warnings"]:
        lines.append("WARNING " + w)
    lines.append("")
    lines.append(f"pre-registered: {s['final']} games per arm, one interim look at {s['interim']} (stop an arm only at |z| >= "
                 f"{s['stop_z']}), raw win rate against {s['control']}, significant at |z| >= {s['sig_z']}, within "
                 f"{s['tie_points']:.0f} point of the highest the cheapest arm, opponent ratings within {s['max_rating_gap']:.0f}")
    for v in out["verdict"]:
        lines.append("VERDICT " + v)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="The ladder A/B of one experiment, with its pre-registered verdict.")
    ap.add_argument("--experiment", required=True, help="the tag in results.jsonl (ladder.py --experiment)")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--checkpoint", default=None, help="only games of this checkpoint file name")
    ap.add_argument("--arms", default=None, help="the arms in this order (default: cheapest first)")
    ap.add_argument("--control", default=DEFAULTS["control"])
    ap.add_argument("--interim", type=int, default=DEFAULTS["interim"])
    ap.add_argument("--final", type=int, default=DEFAULTS["final"])
    ap.add_argument("--mix-iterations-ref", type=float, default=DEFAULTS["mix_iterations_ref"],
                    help="today's iterations per decision of mix (the compute it must keep, within 10%%)")
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON")
    args = ap.parse_args(argv)
    rows = load(args.results, args.experiment, args.checkpoint)
    if not rows:
        print(f"no games of experiment {args.experiment!r} in {args.results}")
        return 0
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else None
    out = report(rows, {"control": args.control, "interim": args.interim, "final": args.final,
                        "mix_iterations_ref": args.mix_iterations_ref}, arms)
    print(json.dumps(out, indent=1) if args.json else text(out, args.experiment))
    return 0


if __name__ == "__main__":
    sys.exit(main())
