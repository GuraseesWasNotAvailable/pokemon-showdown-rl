"""
Scores a predictor of the opponent's next action on the recorded ladder games, and the baselines.

    python bench/opp_predict_eval.py                 # baselines on the test split, and the base rates
    python bench/opp_predict_eval.py --pred runs/opp_pred/replay/decisions_<tag>.jsonl   # a file of predictions

A PREDICTOR is a function  ctx -> {action name: probability}  where ctx is row["ctx"] of
runs/opp_pred/dataset.jsonl (what was known before the turn) and an action name is

    "move <move id>"          "move earthquake"
    "switch <species id>"     "switch garchomp"
    "switch ?"                a switch to a Pokemon the opponent has not shown yet

Probabilities are normalised to sum to 1. Terastallization is not part of the name: a Tera move
counts as the move (the base rates below say how often it comes with one).

SCORED ROWS: kind "turn", the opponent's choice known, and not locked into a move. Per predictor:

    top1, top3        the true action is the most likely / among the three most likely. Ties are
                      scored by their expected value (a uniform predictor gets 1/n, not luck).
    logloss           mean of -ln(max(p(true), 1e-4)).
    switch_or_stay    accuracy of "P(any switch) > 0.5" against "it switched".
    switch_auc        how well P(any switch) ranks the turns it switched above the ones it stayed.
    named_on_switch   on the turns it switched: was that switch our top-1 action?
    target_given_sw   on the turns it switched: the most likely SWITCH TARGET was the right one
                      (who comes in, given that it switches). "revealed" = the Pokemon had been shown.
    top1_move_turns   on the turns it used a move: top-1.

Anything fitted (the base rates, the mixing weights of the baselines) is fitted on the TRAIN
split only. The numbers printed are from the TEST split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.opp_dataset import OUT_DIR, dex, load_rows  # noqa: E402

EPS = 1e-4
Predictor = Callable[[Dict[str, Any]], Dict[str, float]]


# =============================================================================================
# Rows and the truth
# =============================================================================================

def scored_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r["kind"] == "turn" and r["label"]["known"] and not r["label"]["locked"]]


def truth(row: Dict[str, Any]) -> str:
    label = row["label"]
    if label["type"] == "switch" and not label["target_revealed"]:
        return "switch ?"
    return label["action"]


def options(ctx: Dict[str, Any], pool: bool = True) -> Tuple[List[str], List[str]]:
    """(moves, switches) the opponent may have: revealed moves (plus its species' pool), revealed living bench, "switch ?"."""
    opp = ctx["opp"]
    active = opp["active"] or {}
    moves = list((active.get("moves") or {}).keys())
    if pool and len(moves) < 4:
        moves += [m for m in active.get("pool") or [] if m not in moves]
    switches = [m["id"] for m in opp["bench"] if not m["fainted"]]
    hidden_alive = opp["alive"] - 1 - len(switches) if not active.get("fainted") else opp["alive"] - len(switches)
    if opp["unrevealed"] > 0 and hidden_alive > 0:
        switches.append("?")
    return ["move " + m for m in moves], ["switch " + s for s in switches]


def _norm(pred: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in pred.values() if v > 0)
    return {k: v / total for k, v in pred.items() if v > 0} if total > 0 else {}


# =============================================================================================
# Scoring
# =============================================================================================

def _topk(pred: Dict[str, float], true: str, k: int) -> float:
    p = pred.get(true, 0.0)
    if p <= 0:
        return 0.0
    greater = sum(1 for v in pred.values() if v > p + 1e-12)
    equal = sum(1 for v in pred.values() if abs(v - p) <= 1e-12)
    if greater >= k:
        return 0.0
    return min(1.0, (k - greater) / equal)


def _auc(scores: List[float], labels: List[int]) -> Optional[float]:
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum, i = 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        rank_sum += mean_rank * sum(labels[order[t]] for t in range(i, j + 1))
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def score(predict: Predictor, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = scored_rows(rows)
    n = len(rows)
    acc = Counter()
    sw_scores: List[float] = []
    sw_labels: List[int] = []
    for row in rows:
        pred = _norm(predict(row["ctx"]) or {})
        true = truth(row)
        is_switch = row["label"]["type"] == "switch"
        acc["top1"] += _topk(pred, true, 1)
        acc["top3"] += _topk(pred, true, 3)
        acc["logloss"] += -math.log(max(pred.get(true, 0.0), EPS))
        p_switch = sum(v for k, v in pred.items() if k.startswith("switch "))
        sw_scores.append(p_switch)
        sw_labels.append(1 if is_switch else 0)
        acc["switch_or_stay"] += 1.0 if (p_switch > 0.5) == is_switch else 0.0
        if is_switch:
            acc["n_switch"] += 1
            acc["named_on_switch"] += _topk(pred, true, 1)
            targets = _norm({k: v for k, v in pred.items() if k.startswith("switch ")})
            hit = _topk(targets, true, 1)
            acc["target_given_sw"] += hit
            if row["label"]["target_revealed"]:
                acc["n_switch_revealed"] += 1
                acc["target_given_sw_revealed"] += hit
                acc["named_on_switch_revealed"] += _topk(pred, true, 1)
        else:
            acc["n_move"] += 1
            acc["top1_move_turns"] += _topk(pred, true, 1)
            if row["label"]["move_revealed"]:
                acc["n_move_revealed"] += 1
                acc["top1_move_revealed"] += _topk(pred, true, 1)

    def frac(key: str, denom: str = "") -> Optional[float]:
        d = acc[denom] if denom else n
        return round(acc[key] / d, 4) if d else None

    return {"n": n, "top1": frac("top1"), "top3": frac("top3"), "logloss": frac("logloss"),
            "switch_or_stay": frac("switch_or_stay"), "switch_auc": None if _auc(sw_scores, sw_labels) is None else round(_auc(sw_scores, sw_labels), 4),
            "n_switch": acc["n_switch"], "named_on_switch": frac("named_on_switch", "n_switch"),
            "target_given_sw": frac("target_given_sw", "n_switch"), "n_switch_revealed": acc["n_switch_revealed"],
            "target_given_sw_revealed": frac("target_given_sw_revealed", "n_switch_revealed"),
            "named_on_switch_revealed": frac("named_on_switch_revealed", "n_switch_revealed"),
            "n_move": acc["n_move"], "top1_move_turns": frac("top1_move_turns", "n_move"),
            "n_move_revealed": acc["n_move_revealed"], "top1_move_revealed": frac("top1_move_revealed", "n_move_revealed")}


# =============================================================================================
# Baselines
# =============================================================================================

def uniform_known(ctx: Dict[str, Any]) -> Dict[str, float]:
    """Uniform over what the opponent has shown: its revealed moves and its revealed living bench."""
    moves, switches = options(ctx, pool=False)
    return {name: 1.0 for name in moves + switches}


def uniform_pool(ctx: Dict[str, Any]) -> Dict[str, float]:
    """Uniform over revealed moves, the unrevealed moves of its species' random battle pool, and the switches."""
    moves, switches = options(ctx, pool=True)
    return {name: 1.0 for name in moves + switches}


def _mix(point: Optional[str], ctx: Dict[str, Any], weight: float) -> Dict[str, float]:
    base = _norm(uniform_pool(ctx))
    if point is None:
        return base
    out = {k: (1.0 - weight) * v for k, v in base.items()}
    out[point] = out.get(point, 0.0) + weight
    return out


def _last_move(ctx: Dict[str, Any]) -> Optional[str]:
    last = (ctx["opp"]["active"] or {}).get("last_move")
    return "move " + last if last else None


def _most_used(ctx: Dict[str, Any]) -> Optional[str]:
    active = ctx["opp"]["active"] or {}
    moves = active.get("moves") or {}
    if not moves:
        return None
    best = max(moves.values())
    tied = [m for m, c in moves.items() if c == best]
    return "move " + (active.get("last_move") if active.get("last_move") in tied else tied[-1])


def damage_score(move_id: str, ctx: Dict[str, Any]) -> float:
    """Base power x STAB x type effectiveness against our active Pokemon. Status moves are 0."""
    d = dex()
    move = d.move(move_id)
    power = float(move.get("basePower") or 0)
    if not power or str(move.get("category")) == "Status":
        return 0.0
    mine = ctx["us"]["active"] or {}
    theirs = ctx["opp"]["active"] or {}
    move_type = str(move.get("type") or "").upper()
    stab = 1.5 if move_type in (theirs.get("base_types") or []) or move_type in (theirs.get("types") or []) else 1.0
    return power * stab * d.effectiveness(move_type, mine.get("types") or [])


def _max_damage(ctx: Dict[str, Any], pool: bool) -> Optional[str]:
    active = ctx["opp"]["active"] or {}
    names = list((active.get("moves") or {}).keys())
    if pool or not names:
        names += [m for m in active.get("pool") or [] if m not in names]
    scored = [(damage_score(m, ctx), m) for m in names]
    scored = [s for s in scored if s[0] > 0]
    return "move " + max(scored)[1] if scored else None


def switch_bucket(ctx: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    active = ctx["opp"]["active"] or {}
    return (bool((ctx.get("matchup") or {}).get("opp_disadvantage")), float(active.get("hp") or 0) < 0.34,
            int(active.get("turns_on_field") or 0) == 0)


class Fitted:
    """Everything the baselines learn, from TRAIN rows only."""

    def __init__(self, train: List[Dict[str, Any]]) -> None:
        rows = scored_rows(train)
        self.switch_rate = sum(1 for r in rows if r["label"]["type"] == "switch") / max(1, len(rows))
        by: Dict[Any, List[int]] = defaultdict(lambda: [0, 0])
        for r in rows:
            cell = by[switch_bucket(r["ctx"])]
            cell[0] += 1 if r["label"]["type"] == "switch" else 0
            cell[1] += 1
        self.bucket_rate = {k: (s + 2.0 * self.switch_rate) / (t + 2.0) for k, (s, t) in by.items()}
        self.weights: Dict[str, float] = {}
        grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for name, point in (("repeat_last", _last_move), ("most_used", _most_used),
                            ("max_damage_revealed", lambda c: _max_damage(c, False)), ("max_damage_pool", lambda c: _max_damage(c, True))):
            best = min(grid, key=lambda w: sum(-math.log(max(_mix(point(r["ctx"]), r["ctx"], w).get(truth(r), 0.0), EPS)) for r in rows))
            self.weights[name] = best
        # the combined prior: P(switch | bucket), then last move, strongest move and the rest
        self.combo = (0.3, 0.3)
        best_loss = None
        for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            for b in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                if a + b > 0.9:
                    continue
                self.combo = (a, b)
                loss = sum(-math.log(max(_norm(self.prior(r["ctx"])).get(truth(r), 0.0), EPS)) for r in rows)
                if best_loss is None or loss < best_loss:
                    best_loss, best = loss, (a, b)
        self.combo = best

    def base_rate(self, ctx: Dict[str, Any]) -> Dict[str, float]:
        """The overall switch rate spread over the switches, the rest spread over the moves."""
        return self._split(ctx, self.switch_rate)

    def base_rate_bucket(self, ctx: Dict[str, Any]) -> Dict[str, float]:
        """The same with the switch rate of the turn's bucket: (type disadvantage, HP under a third, just came in)."""
        return self._split(ctx, self.bucket_rate.get(switch_bucket(ctx), self.switch_rate))

    @staticmethod
    def _split(ctx: Dict[str, Any], p_switch: float, move_dist: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        moves, switches = options(ctx, pool=True)
        if not switches:
            p_switch = 0.0
        if not moves:
            p_switch = 1.0
        out = {s: p_switch / len(switches) for s in switches}
        if move_dist is None:
            move_dist = {m: 1.0 / len(moves) for m in moves}
        for m, v in move_dist.items():
            out[m] = out.get(m, 0.0) + (1.0 - p_switch) * v
        return out

    def prior(self, ctx: Dict[str, Any]) -> Dict[str, float]:
        """P(switch | bucket) x uniform targets + P(stay) x (a: its last move, b: its strongest move, rest: uniform)."""
        a, b = self.combo
        moves, _ = options(ctx, pool=True)
        if not moves:
            return self._split(ctx, 1.0)
        dist = {m: (1.0 - a - b) / len(moves) for m in moves}
        last, strong = _last_move(ctx), _max_damage(ctx, True)
        for point, w in ((last, a), (strong, b)):
            if point is not None:
                dist[point] = dist.get(point, 0.0) + w
            else:
                for m in moves:
                    dist[m] += w / len(moves)
        return self._split(ctx, self.bucket_rate.get(switch_bucket(ctx), self.switch_rate), dist)

    def predictors(self) -> Dict[str, Predictor]:
        w = self.weights
        return {
            "uniform_known": uniform_known,
            "uniform_pool": uniform_pool,
            "repeat_last": lambda c: _mix(_last_move(c), c, w["repeat_last"]),
            "most_used": lambda c: _mix(_most_used(c), c, w["most_used"]),
            "max_damage_revealed": lambda c: _mix(_max_damage(c, False), c, w["max_damage_revealed"]),
            "max_damage_pool": lambda c: _mix(_max_damage(c, True), c, w["max_damage_pool"]),
            "base_rate": self.base_rate,
            "base_rate_bucket": self.base_rate_bucket,
            "prior_combo": self.prior,
        }


# =============================================================================================
# Base rates
# =============================================================================================

def _rate(k: int, n: int) -> str:
    if not n:
        return "- (0)"
    p = k / n
    z = 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return "%.1f%% (%.1f to %.1f), %d of %d" % (100 * p, 100 * (centre - half), 100 * (centre + half), k, n)


def base_rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    turns = [r for r in rows if r["kind"] == "turn"]
    free = scored_rows(rows)
    out: Dict[str, Any] = {"turn_rows": len(turns), "scored_rows": len(free)}

    def sw(sel: List[Dict[str, Any]]) -> str:
        return _rate(sum(1 for r in sel if r["label"]["type"] == "switch"), len(sel))

    def act(r: Dict[str, Any]) -> Dict[str, Any]:
        return r["ctx"]["opp"]["active"] or {}

    def mu(r: Dict[str, Any]) -> Dict[str, Any]:
        return r["ctx"].get("matchup") or {}

    has_bench = [r for r in free if r["ctx"]["opp"]["alive"] > 1]
    s: Dict[str, str] = {}
    s["all scored turns"] = sw(free)
    s["it has somebody to switch to"] = sw(has_bench)
    s["type disadvantage (our STAB hits it harder than its STAB hits us)"] = sw([r for r in has_bench if mu(r).get("opp_disadvantage")])
    s["  strong: we hit it super effectively, it does not hit us so"] = sw([r for r in has_bench if mu(r).get("we_threaten", 1) >= 2 and mu(r).get("opp_threatens", 1) <= 1])
    s["even types"] = sw([r for r in has_bench if not mu(r).get("opp_disadvantage") and not mu(r).get("opp_advantage")])
    s["type advantage"] = sw([r for r in has_bench if mu(r).get("opp_advantage")])
    s["HP under 1/3"] = sw([r for r in has_bench if act(r).get("hp", 1) < 0.34])
    s["HP 1/3 to 2/3"] = sw([r for r in has_bench if 0.34 <= act(r).get("hp", 1) < 0.67])
    s["HP over 2/3"] = sw([r for r in has_bench if act(r).get("hp", 1) >= 0.67])
    s["just came in (any way)"] = sw([r for r in has_bench if act(r).get("turns_on_field") == 0])
    s["  came in by its own switch last turn"] = sw([r for r in has_bench if act(r).get("turns_on_field") == 0 and act(r).get("entered_by") == "switch"])
    s["  came in after a faint"] = sw([r for r in has_bench if act(r).get("turns_on_field") == 0 and act(r).get("entered_by") == "forced"])
    s["  came in by a pivot move"] = sw([r for r in has_bench if act(r).get("turns_on_field") == 0 and act(r).get("entered_by") == "pivot"])
    s["on the field for 1 turn"] = sw([r for r in has_bench if act(r).get("turns_on_field") == 1])
    s["on the field for 2 turns or more"] = sw([r for r in has_bench if (act(r).get("turns_on_field") or 0) >= 2])
    s["it is statused"] = sw([r for r in has_bench if act(r).get("status")])
    s["it has a boost"] = sw([r for r in has_bench if any(v > 0 for v in (act(r).get("boosts") or {}).values())])
    s["it has a drop"] = sw([r for r in has_bench if any(v < 0 for v in (act(r).get("boosts") or {}).values())])
    s["we switched last turn"] = sw([r for r in has_bench if str(r["ctx"]["us"].get("last_action") or "").startswith("switch")])
    out["switch_rate"] = s
    switches = [r for r in free if r["label"]["type"] == "switch"]
    out["switch_target_was_already_revealed"] = _rate(sum(1 for r in switches if r["label"]["target_revealed"]), len(switches))
    out["we_also_switched_when_it_switched"] = _rate(sum(1 for r in switches if (r.get("our_action") or {}).get("type") == "switch"), len(switches))

    # the same move twice
    stay = [r for r in free if r["label"]["type"] == "move" and act(r).get("last_move") and (act(r).get("turns_on_field") or 0) >= 1]
    rep: Dict[str, str] = {}
    rep["it moved, and had moved before with this Pokemon still on the field"] = _rate(sum(1 for r in stay if r["label"]["name"] == act(r)["last_move"]), len(stay))
    all_prev = [r for r in free if act(r).get("last_move") and (act(r).get("turns_on_field") or 0) >= 1]
    rep["same, counting a switch as not repeating"] = _rate(sum(1 for r in all_prev if r["label"]["type"] == "move" and r["label"]["name"] == act(r)["last_move"]), len(all_prev))
    for k in (1, 2, 3):
        sel = [r for r in stay if len(act(r).get("moves") or {}) == k]
        rep["  with %d move%s revealed" % (k, "" if k == 1 else "s")] = _rate(sum(1 for r in sel if r["label"]["name"] == act(r)["last_move"]), len(sel))
    sel = [r for r in stay if len(act(r).get("moves") or {}) >= 4]
    rep["  with 4 moves revealed"] = _rate(sum(1 for r in sel if r["label"]["name"] == act(r)["last_move"]), len(sel))
    moves = [r for r in free if r["label"]["type"] == "move"]
    rep["the move it used had been revealed before"] = _rate(sum(1 for r in moves if r["label"]["move_revealed"]), len(moves))
    rep["the move it used is in its species' random battle pool"] = _rate(sum(1 for r in moves if r["label"]["name"] in (act(r).get("pool") or [])), len(moves))
    out["same_move"] = rep

    # Terastallization
    games: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in turns:
        games[r["game"]].append(r)
    tera_turn = [min(r["turn"] for r in g if r["label"]["tera"]) for g in games.values() if any(r["label"]["tera"] for r in g)]
    length = [max(r["turn"] for r in g) for g in games.values()]
    t: Dict[str, Any] = {}
    t["games in which the opponent Terastallized"] = _rate(len(tera_turn), len(games))
    can = [r for r in turns if not r["ctx"]["opp"]["tera_used"] and r["label"]["known"] and r["label"]["type"] == "move"]
    t["per turn in which it could and used a move"] = _rate(sum(1 for r in can if r["label"]["tera"]), len(can))
    if tera_turn:
        ordered = sorted(tera_turn)
        t["turn of the Tera: mean / median / quartiles"] = "%.1f / %d / %d to %d" % (sum(ordered) / len(ordered), ordered[len(ordered) // 2], ordered[len(ordered) // 4], ordered[(3 * len(ordered)) // 4])
        t["game length in turns: mean / median"] = "%.1f / %d" % (sum(length) / len(length), sorted(length)[len(length) // 2])
        hist = Counter(min(x, 21) // 3 for x in tera_turn)
        t["turn of the Tera, share by turn"] = {("%d-%d" % (3 * b, 3 * b + 2) if b < 7 else "21+"): "%.1f%%" % (100 * hist[b] / len(tera_turn)) for b in sorted(hist)}
        frac = [x / max(1, l) for x, l in zip(tera_turn, [max(r["turn"] for r in g) for g in games.values() if any(r["label"]["tera"] for r in g)])]
        t["Tera in the first / second / last third of the game"] = "%.1f%% / %.1f%% / %.1f%%" % tuple(100 * sum(1 for f in frac if lo <= f < hi) / len(frac) for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 9)))
    tera_rows = [r for r in turns if r["label"]["tera"] and r["label"]["known"]]
    t["on the Tera turn its Pokemon had just come in"] = _rate(sum(1 for r in tera_rows if act(r).get("turns_on_field") == 0), len(tera_rows))
    t["on the Tera turn it was at a type disadvantage"] = _rate(sum(1 for r in tera_rows if mu(r).get("opp_disadvantage")), len(tera_rows))
    t["on the Tera turn its HP was over 2/3"] = _rate(sum(1 for r in tera_rows if act(r).get("hp", 1) >= 0.67), len(tera_rows))
    t["on the Tera turn it had a boost"] = _rate(sum(1 for r in tera_rows if any(v > 0 for v in (act(r).get("boosts") or {}).values())), len(tera_rows))
    t["the Tera move was a damaging move"] = _rate(sum(1 for r in tera_rows if float(dex().move(r["label"]["name"]).get("basePower") or 0) > 0), len(tera_rows))
    t["the Tera move was Tera Blast"] = _rate(sum(1 for r in tera_rows if r["label"]["name"] == "terablast"), len(tera_rows))
    t["Tera type was one of its own types"] = _rate(sum(1 for r in tera_rows if str(r["label"].get("tera_type") or "").upper() in (act(r).get("base_types") or [])), len(tera_rows))
    alive = Counter(r["ctx"]["opp"]["alive"] for r in tera_rows)
    t["its Pokemon left when it Terastallized"] = {str(k): alive[k] for k in sorted(alive, reverse=True)}
    out["tera"] = t

    # after a faint, and after a pivot
    forced = [r for r in rows if r["kind"] == "forced"]
    pivot = [r for r in rows if r["kind"] == "pivot"]
    out["replacement_was_already_revealed"] = {"after a faint": _rate(sum(1 for r in forced if r["label"]["target_revealed"]), len(forced)),
                                               "after its pivot move": _rate(sum(1 for r in pivot if r["label"]["target_revealed"]), len(pivot))}
    out["unknown"] = _rate(sum(1 for r in turns if not r["label"]["known"]), len(turns))
    return out


# =============================================================================================
# A file of predictions (the search's, written by bench/opp_replay.py)
# =============================================================================================

def file_predictor(path: str) -> Tuple[Predictor, set]:
    table: Dict[Tuple[str, int], Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            pred = d.get("opp_pred") or d.get("pred")
            if d.get("request", "move") == "move" and pred:
                table[(d["game"], int(d["turn"]))] = pred
    current: Dict[str, Any] = {}

    def predict(ctx: Dict[str, Any]) -> Dict[str, float]:
        return table.get((current.get("game"), ctx["turn"]), {})

    predict.current = current  # type: ignore[attr-defined]
    return predict, set(table)


def _print_table(results: Dict[str, Dict[str, Any]]) -> None:
    cols = ["n", "top1", "top3", "logloss", "switch_or_stay", "switch_auc", "named_on_switch", "target_given_sw",
            "target_given_sw_revealed", "top1_move_turns", "top1_move_revealed"]
    print("| predictor | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, r in results.items():
        print("| %s | " % name + " | ".join("-" if r.get(c) is None else str(r[c]) for c in cols) + " |")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scores predictors of the opponent's next action on the test split.")
    parser.add_argument("--data", default=OUT_DIR)
    parser.add_argument("--pred", default="", help="a jsonl of {game, turn, opp_pred: {action: p}} to score on the rows it covers")
    parser.add_argument("--rates-split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    rows = list(load_rows(args.data))
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    fitted = Fitted(train)
    report: Dict[str, Any] = {"fitted_on_train": {"switch_rate": round(fitted.switch_rate, 4), "mix_weights": fitted.weights,
                                                    "prior_combo_a_last_b_strongest": fitted.combo,
                                                    "bucket_rate (disadvantage, hp<1/3, just came in)": {str(k): round(v, 4) for k, v in sorted(fitted.bucket_rate.items())}}}
    if args.pred:
        predict, keys = file_predictor(args.pred)
        covered = [r for r in rows if (r["game"], r["turn"]) in keys and r["kind"] == "turn"]
        results: Dict[str, Dict[str, Any]] = {}

        def from_file(ctx: Dict[str, Any], _rows=covered) -> Dict[str, float]:
            return {}

        by_key = {id(r["ctx"]): r["game"] for r in covered}

        def search(ctx: Dict[str, Any]) -> Dict[str, float]:
            predict.current["game"] = by_key.get(id(ctx))  # type: ignore[attr-defined]
            return predict(ctx)

        results["file: " + os.path.basename(args.pred)] = score(search, covered)
        for name, fn in fitted.predictors().items():
            results[name] = score(fn, covered)
        report["rows_covered_by_the_file"] = {"games": len({r["game"] for r in covered}), "turn_rows": len(covered),
                                               "splits": dict(Counter(r["split"] for r in covered))}
        print("\nOn the %d turn rows the file covers (%s):\n" % (len(covered), report["rows_covered_by_the_file"]["splits"]))
        _print_table(results)
        report["file_results"] = results
    else:
        results = {name: score(fn, test) for name, fn in fitted.predictors().items()}
        print("\nTEST split: %d games, %d scored turn rows\n" % (len({r["game"] for r in test}), len(scored_rows(test))))
        _print_table(results)
        report["test_results"] = results
        results_train = {name: score(fn, train) for name, fn in fitted.predictors().items()}
        report["train_results"] = results_train
        which = rows if args.rates_split == "all" else [r for r in rows if r["split"] == args.rates_split]
        rates = base_rates(which)
        report["base_rates_" + args.rates_split] = rates
        report["base_rates_train"] = base_rates(train)
        report["base_rates_test"] = base_rates(test)
        print("\nBase rates (%s games):\n" % args.rates_split)
        print(json.dumps(rates, indent=1))
    print("\nfitted on train:", json.dumps(report["fitted_on_train"]))
    out = args.json or os.path.join(args.data, "eval_file.json" if args.pred else "eval_baselines.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
