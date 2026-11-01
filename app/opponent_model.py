"""
What the opponent does next turn, as probabilities, and what to play against it.

    from opponent_model import OpponentModel, OpponentTracker, PredictSettings, choose

    model = OpponentModel.load()                       # opponent_model.json, fitted on recorded ladder games
    tracker = OpponentTracker(battle_tag, "p1")        # fed every line of the battle as it arrives
    prediction = model.predict(tracker.context(), advice)      # advice: the search's answer (foulplay_bridge.Advice)
    prediction.dist                                    # {"move earthquake": 0.41, "switch garchomp": 0.22, "switch ?": 0.05, ...}
    action, info = choose(prediction, advice.payoff_table, advice.payoff_theirs, search_view(advice),
                          mix_action, mask, shares, PredictSettings())

Nobody can know a move the opponent has not chosen yet. What can be had is a probability for
each thing it can do, and the measure of such a probability is whether it is CALIBRATED: of the
turns on which the model says 70%, the opponent should do that thing on about 70%. `evaluate()`
prints exactly that table next to accuracy and log loss, and choose() acts on a prediction only
above a confidence that is a setting.

TWO SOURCES, combined:

  * the search's own view of the opponent's side: poke-engine's Monte Carlo tree search plays
    both sides, so every answer of the bridge carries the share of its iterations each option of
    the opponent got, over opponent sets sampled for what has not been shown
    (foulplay_bridge.Advice.opponent). It knows damage, speed and sets. It does not know people.
  * what real opponents did in recorded ladder games (bench/opp_dataset.py: 426 games of an older
    bot against people): how often they switch in which situation, whom they bring in, how often
    they repeat a move.

THE MODEL is three small fitted parts, each a few numbers (opponent_model.json), all fitted on
the TRAIN split of the recorded games replayed through the search (bench/opp_model_replay.py):

  P(switch)            logistic regression on the search's own switch share (as a logit) and the
                       situation: type disadvantage or advantage, just came in (after a faint,
                       or otherwise), on the field for one turn, we switched last turn, a stat
                       drop, a boost, a status.
  P(target | switch)   the search's shares over the switch targets (to a power), mixed with an
                       even spread over the Pokemon it has shown and a weight for "one it has
                       not shown yet" ("switch ?").
  P(move | stays)      a mixture: the search's shares over its moves (to a power 1/T), its last
                       move, its strongest shown move, an even spread over its shown moves, and
                       an even spread over every move it may have (its species' random battle
                       pool and whatever the search sampled). The weights by EM, T by a grid.
  P(Tera | move)       a table: how often a move came with Tera, by the search's Tera share of
                       the move shares in six bands, each drawn towards the overall rate. (A
                       straight line through it was tried and was wrong at the top: where the
                       search is nearly sure of a Tera, people Terastallized on 3 of 26 turns.)

Without a search answer the same parts are used with the weights fitted without it ("prior").

ACTING ON IT (choose): the value of each of our actions against the predicted distribution, from
the payoff table the bridge returns per sampled world (our option x the opponent's option, the
engine's evaluation two plies deep, payoff="full"). The combined network-and-search choice
("mix") stands unless ALL of these hold, each a setting of PredictSettings:

  * a read: the model gives one action of the opponent, or "it switches" as a whole, at least
    `min_confidence`;
  * a clear gain: another action of ours is worth more against the whole predicted distribution
    by at least `min_gain` (on the engine's own 0-to-1 scale, sigmoid(0.0125 * difference));
  * a wrong read does not lose the game on the spot: against the opponent's second most likely
    action, and against every other action it takes with at least `risk_floor`, the new action
    is not worse than the mix choice by more than `max_risk`;
  * the table knows enough: the cells behind at least `min_coverage` of the predicted
    probability exist for both actions (the table is computed under a time budget and is
    seldom complete).

This is where a double switch comes from: the opponent very probably switches to X, the action
that is best against X coming in is not the one that is best against the Pokemon that is there
now, and the difference is in the table.

    python opponent_model.py fit  --replay runs/opp_pred/model/replay_all_ms150.jsonl
    python opponent_model.py eval --replay runs/opp_pred/model/replay_all_ms150.jsonl [--json out.json]
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from bench.opp_dataset import GameRows, dex, to_id  # noqa: E402
from bench.opp_predict_eval import damage_score, options as ctx_options  # noqa: E402

__all__ = [
    "PARAMS_FILE", "EVAL_SCALE", "OpponentModel", "Prediction", "PredictSettings", "OpponentTracker", "search_view",
    "fold_search", "truth_name", "column_probabilities", "choose", "fit", "evaluate", "reliability",
]

PARAMS_FILE = os.path.join(HERE, "opponent_model.json")
ACTIONS = 14
EVAL_SCALE = 0.0125          # poke-engine's own: a difference d in its evaluation is worth sigmoid(0.0125 * d) - 0.5
TERA = " terastallize"
SWITCH_FEATURES = ("bias", "search_logit", "disadvantage", "advantage", "in_after_faint", "in_otherwise", "one_turn",
                   "we_switched", "drop", "boost", "statused")
MOVE_PARTS = ("search", "last", "strongest", "shown", "pool")
T_GRID = (0.5, 0.7, 1.0, 1.4, 2.0, 3.0)
TERA_EDGES = (0.05, 0.10, 0.20, 0.35, 0.50)      # bands of the search's Tera share of its move shares
EPS = 1e-4


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p: float, lo: float = 0.005) -> float:
    p = min(1.0 - lo, max(lo, float(p)))
    return math.log(p / (1.0 - p))


def _band(value: float, edges: Sequence[float]) -> int:
    return sum(1 for edge in edges if value >= edge)


def _normalise(d: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in d.values() if v > 0)
    return {k: v / total for k, v in d.items() if v > 0} if total > 0 else {}


# =============================================================================================
# The search's view of the opponent, from an Advice or from a replayed answer
# =============================================================================================

def search_view(source: Any) -> Optional[Dict[str, Any]]:
    """
    {"options": [(name, share, seen)], "sets": [[move ids] per sampled world]} from a
    foulplay_bridge.Advice, from the helper's "opponent" block, or from the "opp" block
    bench/opp_model_replay.py writes. None when there is nothing to read.
    """
    if source is None:
        return None
    if isinstance(source, dict):
        if "options" not in source:
            return None
        options: List[Tuple[str, float, Any]] = []
        for o in source.get("options") or []:
            if isinstance(o, dict):
                options.append((str(o.get("name") or ""), float(o.get("share") or 0.0), o.get("seen")))
            else:
                options.append((str(o[0]), float(o[1] or 0.0), o[2] if len(o) > 2 else None))
        sets = [list(w) if isinstance(w, (list, tuple)) else list(w.get("moves") or (w.get("active") or {}).get("moves") or [])
                for w in (source.get("sets") or source.get("worlds") or [])]
        return {"options": options, "sets": sets} if options else None
    theirs = getattr(source, "opponent", None)
    if not theirs:
        return None
    options = [(str(o.name), float(o.share or 0.0), o.seen) for o in theirs]
    sets = [list((w.get("active") or {}).get("moves") or []) for w in (getattr(source, "opponent_worlds", None) or [])]
    return {"options": options, "sets": sets}


def fold_search(view: Optional[Dict[str, Any]], bench: Sequence[str], hidden: bool) -> Optional[Dict[str, Any]]:
    """
    The search's shares in the names the model predicts: per move (Terastallization folded in,
    its part kept apart in "tera"), per shown switch target, and everything else that is a
    switch under "?" (a Pokemon the sampling invented, or one our rows do not list).
    """
    if not view:
        return None
    moves: Dict[str, float] = {}
    tera: Dict[str, float] = {}
    switches: Dict[str, float] = {}
    unseen = 0.0
    for name, share, seen in view["options"]:
        if name.startswith("move "):
            move = name[5:]
            if move.endswith(TERA):
                move = move[: -len(TERA)]
                tera[move] = tera.get(move, 0.0) + share
            move = to_id(move)
            moves[move] = moves.get(move, 0.0) + share
        elif name.startswith("switch "):
            target = name[7:]
            if seen is not False and target in bench:
                switches[target] = switches.get(target, 0.0) + share
            else:
                unseen += share
    total = sum(moves.values()) + sum(switches.values()) + unseen
    if total <= 0:
        return None
    if not hidden:
        unseen = 0.0
    return {"moves": moves, "tera": {to_id(k): v for k, v in tera.items()}, "switches": switches, "unseen": unseen,
            "switch_share": (sum(switches.values()) + unseen) / total, "total": total,
            "pool": sorted({to_id(m) for s in view.get("sets") or [] for m in s})}


def truth_name(label: Dict[str, Any]) -> Optional[str]:
    """What the opponent did, in the model's names (bench/opp_predict_eval.truth). None when the log cannot say or it had no choice."""
    if not label or not label.get("known") or label.get("locked"):
        return None
    if label.get("type") == "switch" and not label.get("target_revealed"):
        return "switch ?"
    return label.get("action")


# =============================================================================================
# The situation, read from a row's ctx (bench/opp_dataset.py). The same code at fit and at play time.
# =============================================================================================

def switch_features(ctx: Dict[str, Any], search_switch: Optional[float]) -> List[float]:
    opp = ctx.get("opp") or {}
    active = opp.get("active") or {}
    matchup = ctx.get("matchup") or {}
    boosts = active.get("boosts") or {}
    just_in = int(active.get("turns_on_field") or 0) == 0
    after_faint = just_in and active.get("entered_by") == "forced"
    return [
        1.0,
        _logit(search_switch) if search_switch is not None else 0.0,
        1.0 if matchup.get("opp_disadvantage") else 0.0,
        1.0 if matchup.get("opp_advantage") else 0.0,
        1.0 if after_faint else 0.0,
        1.0 if just_in and not after_faint else 0.0,
        1.0 if int(active.get("turns_on_field") or 0) == 1 else 0.0,
        1.0 if str((ctx.get("us") or {}).get("last_action") or "").startswith("switch") else 0.0,
        1.0 if any(v < 0 for v in boosts.values()) else 0.0,
        1.0 if any(v > 0 for v in boosts.values()) else 0.0,
        1.0 if active.get("status") else 0.0,
    ]


def _strongest(ctx: Dict[str, Any]) -> Optional[str]:
    """Its strongest shown move against our active Pokemon (its pool when it has shown none): bench/opp_predict_eval._max_damage."""
    active = (ctx.get("opp") or {}).get("active") or {}
    names = list((active.get("moves") or {}).keys()) or list(active.get("pool") or [])
    scored = [(damage_score(m, ctx), m) for m in names]
    scored = [s for s in scored if s[0] > 0]
    return max(scored)[1] if scored else None


def move_parts(ctx: Dict[str, Any], folded: Optional[Dict[str, Any]], candidates: List[str], temperature: float) -> Dict[str, Optional[Dict[str, float]]]:
    """Each part of the move mixture as a distribution over the candidate moves (None: the part has nothing to say here)."""
    active = (ctx.get("opp") or {}).get("active") or {}
    shown = [m for m in (active.get("moves") or {}) if m in candidates]
    parts: Dict[str, Optional[Dict[str, float]]] = {name: None for name in MOVE_PARTS}
    if folded and folded["moves"]:
        power = 1.0 / max(1e-6, temperature)
        parts["search"] = _normalise({m: (s ** power) for m, s in folded["moves"].items() if s > 0 and m in candidates}) or None
    last = active.get("last_move")
    if last and last in candidates:
        parts["last"] = {last: 1.0}
    strong = _strongest(ctx)
    if strong and strong in candidates:
        parts["strongest"] = {strong: 1.0}
    if shown:
        parts["shown"] = {m: 1.0 / len(shown) for m in shown}
    if candidates:
        parts["pool"] = {m: 1.0 / len(candidates) for m in candidates}
    return parts


def _candidates(ctx: Dict[str, Any], folded: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str], bool]:
    """(move ids it may use, shown living bench ids, whether it has a living Pokemon it has not shown)."""
    moves, switches = ctx_options(ctx, pool=True)
    move_ids = [m[5:] for m in moves]
    if folded:
        move_ids += [m for m in folded["moves"] if m not in move_ids]
    bench = [s[7:] for s in switches if s != "switch ?"]
    return move_ids, bench, "switch ?" in switches


def _bench_of(ctx: Dict[str, Any]) -> Tuple[List[str], bool]:
    _moves, switches = ctx_options(ctx, pool=False)
    return [s[7:] for s in switches if s != "switch ?"], "switch ?" in switches


# =============================================================================================
# The model
# =============================================================================================

@dataclass
class Prediction:
    """The opponent's next action as the model sees it."""
    dist: Dict[str, float]                       # "move <id>", "switch <species id>", "switch ?": sums to 1
    p_switch: float = 0.0
    p_tera: float = 0.0                          # that a move comes with Terastallization (0 once it has used it)
    tera: Dict[str, float] = field(default_factory=dict)     # per move id: P(Tera | that move)
    with_search: bool = False

    def ranked(self) -> List[Tuple[str, float]]:
        return sorted(self.dist.items(), key=lambda kv: (-kv[1], kv[0]))

    def top(self, n: int = 1) -> List[Tuple[str, float]]:
        return self.ranked()[:n]


def default_params() -> Dict[str, Any]:
    """Unfitted numbers: the base rates of bench/notes/opp_dataset.md by hand. Used only when no fitted file exists."""
    return {
        "version": 1, "fitted": None,
        "switch": {"features": list(SWITCH_FEATURES),
                   "with_search": [-0.6, 0.6, 0.5, -0.4, -2.0, -0.6, 0.6, 0.6, 0.4, -0.6, 0.0],
                   "prior": [-1.9, 0.0, 1.0, -0.7, -2.5, -1.0, 0.9, 1.0, 0.6, -0.8, 0.2]},
        "target": {"w": 0.5, "kappa": 2.0, "gamma": 1.0, "prior_kappa": 2.0},
        "move": {"T": 1.0, "weights": {"search": 0.6, "last": 0.1, "strongest": 0.1, "shown": 0.1, "pool": 0.1},
                 "prior_weights": {"search": 0.0, "last": 0.25, "strongest": 0.25, "shown": 0.2, "pool": 0.3}},
        "tera": {"edges": list(TERA_EDGES), "rates": [0.045] * (len(TERA_EDGES) + 1), "base_rate": 0.045},
    }


class OpponentModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or default_params()
        self.fitted = bool(self.params.get("fitted"))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "OpponentModel":
        path = path or PARAMS_FILE
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return cls(json.load(fh))
        return cls(None)

    # ---- the parts ------------------------------------------------------------------------------

    def p_switch(self, ctx: Dict[str, Any], folded: Optional[Dict[str, Any]]) -> float:
        p = self.params["switch"]
        if folded is not None:
            x = switch_features(ctx, folded["switch_share"])
            beta = p["with_search"]
        else:
            x = switch_features(ctx, None)
            beta = p["prior"]
        return _sigmoid(sum(b * v for b, v in zip(beta, x)))

    def targets(self, bench: List[str], hidden: bool, folded: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """P(who comes in | it switches) over its shown living bench and "?"."""
        p = self.params["target"]
        names = list(bench) + (["?"] if hidden else [])
        if not names:
            return {}
        kappa = float(p["kappa"] if folded is not None else p.get("prior_kappa", p["kappa"]))
        even = _normalise({n: (kappa if n == "?" else 1.0) for n in names})
        if folded is None:
            return even
        gamma = float(p.get("gamma", 1.0))
        raw = {n: (folded["unseen"] if n == "?" else folded["switches"].get(n, 0.0)) for n in names}
        search = _normalise({n: v ** gamma for n, v in raw.items() if v > 0})
        if not search:
            return even
        w = float(p["w"])
        return {n: w * search.get(n, 0.0) + (1.0 - w) * even[n] for n in names}

    def moves(self, ctx: Dict[str, Any], folded: Optional[Dict[str, Any]], candidates: List[str]) -> Dict[str, float]:
        """P(which move | it stays) over the candidate moves."""
        p = self.params["move"]
        weights = p["weights"] if folded is not None else p["prior_weights"]
        parts = move_parts(ctx, folded, candidates, float(p.get("T", 1.0)))
        pool = parts["pool"] or {}
        out = {m: 0.0 for m in candidates}
        for name in MOVE_PARTS:
            w = float(weights.get(name, 0.0))
            if w <= 0:
                continue
            for m, v in (parts[name] or pool).items():      # a part with nothing to say is the even spread
                out[m] += w * v
        return _normalise(out)

    def p_tera(self, ctx: Dict[str, Any], folded: Optional[Dict[str, Any]]) -> float:
        if (ctx.get("opp") or {}).get("tera_used"):
            return 0.0
        p = self.params["tera"]
        if folded is None or not folded["moves"]:
            return float(p["base_rate"])
        share = sum(folded["tera"].values()) / max(1e-9, sum(folded["moves"].values()))
        return float(p["rates"][_band(share, p["edges"])])

    # ---- the whole ------------------------------------------------------------------------------

    def predict(self, ctx: Dict[str, Any], search: Any = None) -> Prediction:
        """
        ctx     the situation before the turn, as bench/opp_dataset.py writes it (OpponentTracker.context()).
        search  the search's answer for the same request: an Advice, a search_view(), or None.
        """
        view = search_view(search)
        bench, hidden = _bench_of(ctx)
        folded = fold_search(view, bench, hidden)
        move_ids, bench, hidden = _candidates(ctx, folded)
        targets = self.targets(bench, hidden, folded)
        p_sw = self.p_switch(ctx, folded) if targets else 0.0
        if not move_ids:
            p_sw = 1.0 if targets else 0.0
        dist: Dict[str, float] = {}
        for name, v in targets.items():
            dist["switch " + name] = p_sw * v
        for move, v in (self.moves(ctx, folded, move_ids) if move_ids else {}).items():
            dist["move " + move] = (1.0 - p_sw) * v
        dist = _normalise(dist)
        p_tera = self.p_tera(ctx, folded)
        tera: Dict[str, float] = {}
        if folded is not None and p_tera > 0:
            # The model says how often a move comes with Tera; the search says with which move rather.
            ratio = {m: folded["tera"].get(m, 0.0) / s for m, s in folded["moves"].items() if s > 0}
            mean = sum(dist.get("move " + m, 0.0) * r for m, r in ratio.items()) / max(1e-9, sum(dist.get("move " + m, 0.0) for m in ratio))
            for m, r in ratio.items():
                tera[m] = min(0.95, p_tera * (r / mean)) if mean > 0 else p_tera
        return Prediction(dist=dist, p_switch=p_sw, p_tera=p_tera, tera=tera, with_search=folded is not None)

    def predictor(self, searches: Dict[int, Any]) -> Any:
        """A predictor for bench/opp_predict_eval.score: ctx -> {name: p}. searches: id(ctx) -> the search's block (or absent)."""
        return lambda ctx: self.predict(ctx, searches.get(id(ctx))).dist


# =============================================================================================
# The battle as it is played: the same context the model was fitted on, and what the opponent did
# =============================================================================================

class OpponentTracker:
    """
    Follows one battle from its lines (bench/opp_dataset.GameRows, the code that made the
    training rows) and gives, at the start of a turn, the context the model was fitted on, and
    after the turn what the opponent did in it.

        tracker = OpponentTracker(tag); tracker.feed(line, role) for every line of the room
        tracker.context(turn)        the ctx at the start of `turn`, None when it is not there
        tracker.closed()             the turns that ended since the last call: [(turn, label, name or None)]
    """

    def __init__(self, tag: str, role: Optional[str] = None) -> None:
        self.tag = tag
        self.role = role
        self.rows: Optional[GameRows] = None
        self.waiting: List[str] = []         # lines that came before we knew which side is ours
        self.read = 0
        self.broken: Optional[str] = None
        if role:
            self._start(role)

    def _start(self, role: str) -> None:
        self.role = role
        self.rows = GameRows({"me": {"side": role}, "id": self.tag, "myTeam": []})
        waiting, self.waiting = self.waiting, []
        for line in waiting:
            self._feed(line)

    def _feed(self, line: str) -> None:
        if self.broken or self.rows is None:
            return
        try:
            self.rows.feed(line)
        except Exception as exc:            # a line the parser does not know must not disturb a game
            self.broken = "%s: %s" % (type(exc).__name__, str(exc)[:120])

    def feed(self, line: str, role: Optional[str] = None) -> None:
        if self.rows is None:
            if role in ("p1", "p2"):
                self._start(role)
            else:
                if len(self.waiting) < 5000:
                    self.waiting.append(line)
                return
        self._feed(line)

    def context(self, turn: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if self.rows is None or self.broken:
            return None
        ctx = self.rows.start_ctx
        if ctx is None or (turn is not None and int(ctx.get("turn") or 0) != int(turn)):
            return None
        if not (ctx.get("opp") or {}).get("active") or not (ctx.get("us") or {}).get("active"):
            return None
        return ctx

    def closed(self) -> List[Tuple[int, Dict[str, Any], Optional[str]]]:
        if self.rows is None:
            return []
        out = []
        rows = self.rows.rows
        while self.read < len(rows):
            row = rows[self.read]
            self.read += 1
            if row["kind"] == "turn":
                out.append((int(row["turn"]), row["label"], truth_name(row["label"])))
        return out


# =============================================================================================
# Acting on a prediction
# =============================================================================================

@dataclass(frozen=True)
class PredictSettings:
    """When the "predict" mode leaves the mix choice. First guesses from the calibration table, not tuned on games."""
    # A read: the model gives one action of the opponent, or "it switches" as a whole, at least this.
    min_confidence: float = 0.60
    # The other action is worth at least this much more against the whole predicted distribution,
    # on the engine's 0-to-1 scale: sigmoid(0.0125 * (difference of evaluations)) - 0.5.
    # 0.08 is about 26 points of evaluation, a quarter of a Pokemon's full HP.
    min_gain: float = 0.08
    # A wrong read must not lose the game on the spot: against the second most likely action of
    # the opponent, and against every other action it takes with at least risk_floor, the new
    # action may be worse than the mix choice by this much at most (same scale).
    max_risk: float = 0.08
    risk_floor: float = 0.15
    # The payoff cells behind at least this share of the predicted probability must exist for
    # both actions, and the cell against the second most likely action must exist.
    min_coverage: float = 0.70
    # An action the search did not look at (share below this) is not a candidate.
    min_share: float = 0.01
    # The payoff table asked of the bridge: this many ms on top of the search, all worlds
    # together, "full" = every cell two plies deep, most visited pairs first.
    payoff_ms: float = 400.0
    payoff_mode: str = "full"
    # Terastallization is not this rule's business: a move keeps the form the mix rule gives it.
    tera_share: float = 0.20
    # A read that has failed against this Pokemon in this battle is not acted on again ("it
    # switches" said and it stayed, "its move is X" said and it was another). The player keeps
    # the list (search_player), choose() itself has no memory. Added after the first 84 real
    # games, in one of which the same wrong "it switches" was acted on seven turns running.
    once_bitten: bool = True


def _column_name(column: str, seen: Dict[str, Any], names: Dict[str, float]) -> Optional[str]:
    """The model's name for a column of the payoff table ("move x terastallize" -> "move x", an unshown switch -> "switch ?")."""
    if column.startswith("move "):
        move = column[5:]
        if move.endswith(TERA):
            move = move[: -len(TERA)]
        return "move " + to_id(move)
    if column.startswith("switch "):
        if column in names and seen.get(column) is not False:
            return column
        return "switch ?"
    return None


def column_probabilities(prediction: Prediction, columns: Sequence[str], view: Optional[Dict[str, Any]]) -> Tuple[List[float], List[Optional[str]]]:
    """
    The predicted probability of each column of the payoff table, and the model's name of each.
    A move's probability is split between its plain and its Tera column by P(Tera | move); the
    probability of "switch ?" is spread over the unshown switch columns as the search's shares.
    Probability on names without a column (a move no sampled world has) is on no column.
    """
    share = {name: s for name, s, _seen in (view or {}).get("options") or []}
    seen = {name: sn for name, _s, sn in (view or {}).get("options") or []}
    names = [_column_name(c, seen, prediction.dist) for c in columns]
    probs = [0.0] * len(columns)
    have = set(columns)
    unshown = [i for i, n in enumerate(names) if n == "switch ?"]
    unshown_total = sum(share.get(columns[i], 0.0) for i in unshown)
    for i, (column, name) in enumerate(zip(columns, names)):
        if name is None:
            continue
        p = prediction.dist.get(name, 0.0)
        if name.startswith("move "):
            move = name[5:]
            plain, tera = "move " + move, "move " + move + TERA
            t = prediction.tera.get(move, prediction.p_tera) if tera in have else 0.0
            if plain not in have:
                t = 1.0
            probs[i] = p * (t if column.endswith(TERA) else 1.0 - t)
        elif name == "switch ?":
            probs[i] = p * (share.get(column, 0.0) / unshown_total if unshown_total > 0 else 1.0 / len(unshown))
        else:
            probs[i] = p
    return probs, names


def _gain(difference: float) -> float:
    return _sigmoid(EVAL_SCALE * difference) - 0.5


def _form(base: int, mix_action: int, mask: Sequence[float], shares: Sequence[float], tera_share: float) -> int:
    """The action that plays move or switch `base`: the form the mix rule would give it (search_player.mix)."""
    if base >= 4:
        return base
    if (mix_action - 10 if mix_action >= 10 else mix_action) == base:
        return mix_action
    plain, tera = base, base + 10
    if mask[plain] > 0.5 and mask[tera] > 0.5:
        return tera if (shares[tera] > shares[plain] and shares[tera] >= tera_share) else plain
    return tera if mask[tera] > 0.5 and not mask[plain] > 0.5 else plain


def choose(prediction: Optional[Prediction], table: Optional[Sequence[Sequence[Optional[float]]]], columns: Optional[Sequence[str]],
           view: Optional[Dict[str, Any]], mix_action: int, mask: Sequence[float], shares: Sequence[float],
           settings: Optional[PredictSettings] = None) -> Tuple[int, Dict[str, Any]]:
    """
    (action, info). The mix choice unless the prediction is confident, another action is clearly
    better against it, and that action does not lose much if the read is wrong. info["why"] says
    which test kept the mix choice ("" when it was left), and holds the numbers.

    table[a][c]   our action a (0-13) against columns[c]: the engine's evaluation, None = unknown.
    shares        the search's share of each of our actions.
    """
    s = settings or PredictSettings()
    info: Dict[str, Any] = {"changed": False, "why": "", "read": None, "confidence": None, "second": None, "candidate": None,
                            "gain": None, "risk": None, "coverage": None}
    if prediction is None or not prediction.dist:
        info["why"] = "no prediction"
        return mix_action, info
    ranked = prediction.ranked()
    top_name, top_p = ranked[0]
    # The read: "it switches" as a whole when that is confident, else its single likeliest action.
    if prediction.p_switch >= s.min_confidence:
        read, confidence = "switch", prediction.p_switch
        agrees = lambda name: name.startswith("switch ")      # noqa: E731
    else:
        read, confidence = top_name, top_p
        agrees = lambda name: name == top_name                # noqa: E731
    second = next(((n, p) for n, p in ranked if not agrees(n)), None)
    info.update({"read": read, "confidence": round(confidence, 4), "second": second[0] if second else None,
                 "second_p": round(second[1], 4) if second else None})
    if confidence < s.min_confidence:
        info["why"] = "unsure"
        return mix_action, info
    if not table or not columns or len(table) < ACTIONS:
        info["why"] = "no table"
        return mix_action, info

    probs, names = column_probabilities(prediction, columns, view)
    by_name: Dict[str, List[int]] = {}
    for i, name in enumerate(names):
        if name is not None and probs[i] > 0:
            by_name.setdefault(name, []).append(i)

    def against(action: int, name: str) -> Optional[float]:
        """Our action against one predicted action of the opponent: the mean over that action's known columns."""
        value, weight = 0.0, 0.0
        for i in by_name.get(name, ()):
            cell = table[action][i]
            if cell is not None:
                value += probs[i] * cell
                weight += probs[i]
        return value / weight if weight > 0 else None

    mix_row = {name: against(mix_action, name) for name in by_name}
    if all(v is None for v in mix_row.values()):
        info["why"] = "no table"
        return mix_action, info

    mix_base = mix_action - 10 if mix_action >= 10 else mix_action
    best: Optional[Tuple[float, int, float, Dict[str, Optional[float]]]] = None
    for base in range(10):
        if base == mix_base:
            continue
        action = _form(base, mix_action, mask, shares, s.tera_share)
        if not mask[action] > 0.5 or shares[action] < s.min_share:
            continue
        row = {name: against(action, name) for name in by_name}
        both = [name for name in by_name if row[name] is not None and mix_row[name] is not None]
        coverage = sum(prediction.dist.get(name, 0.0) for name in both)
        if coverage < s.min_coverage or coverage <= 0:
            continue
        difference = sum(prediction.dist[name] * (row[name] - mix_row[name]) for name in both) / coverage
        gain = _gain(difference)
        if best is None or gain > best[0]:
            best = (gain, action, coverage, row)
    if best is None:
        info["why"] = "coverage"
        return mix_action, info
    gain, action, coverage, row = best
    info.update({"candidate": int(action), "gain": round(gain, 4), "coverage": round(coverage, 4)})
    if gain < s.min_gain:
        info["why"] = "no gain"
        return mix_action, info

    # A wrong read: what the new action is worth against what else the opponent may do.
    checks = [second[0]] if second else []
    checks += [n for n, p in ranked if p >= s.risk_floor and not agrees(n) and n not in checks]
    worst = 0.0
    for name in checks:
        ours, theirs = row.get(name), mix_row.get(name)
        if ours is None or theirs is None:
            if second and name == second[0]:
                info["why"] = "blind"            # nothing known about the second most likely action
                return mix_action, info
            continue
        worst = max(worst, -_gain(ours - theirs))
    info["risk"] = round(worst, 4)
    if worst > s.max_risk:
        info["why"] = "risk"
        return mix_action, info
    info["changed"] = True
    return int(action), info


# =============================================================================================
# Fitting (train split only) and scoring
# =============================================================================================

def load_replay(path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """(game, turn) -> the search's answer for the move request of that turn, from bench/opp_model_replay.py."""
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("ok") and d.get("request") == "move" and d.get("opp"):
                out[(d["game"], int(d["turn"]))] = d
    return out


def _scored(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r["kind"] == "turn" and r["label"]["known"] and not r["label"]["locked"]]


def _logistic_fit(x: Any, y: Any, ridge: float = 1.0, steps: int = 60) -> List[float]:
    """Newton's method with a ridge on everything but the first column (the bias)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    beta = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(x @ beta)))
        gradient = x.T @ (p - y) + penalty @ beta
        hessian = (x * (p * (1 - p))[:, None]).T @ x + penalty + np.eye(x.shape[1]) * 1e-8
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.abs(step).max()) < 1e-8:
            break
    return [round(float(b), 5) for b in beta]


def _em(columns: Any, steps: int = 300) -> Tuple[List[float], float]:
    """Mixture weights for per-row part likelihoods columns[n][k]; returns (weights, mean log loss)."""
    import numpy as np

    c = np.asarray(columns, dtype=np.float64)
    w = np.full(c.shape[1], 1.0 / c.shape[1])
    for _ in range(steps):
        joint = c * w
        total = joint.sum(axis=1, keepdims=True)
        total[total <= 0] = 1e-300
        new = (joint / total).mean(axis=0)
        if float(np.abs(new - w).max()) < 1e-7:
            w = new
            break
        w = new
    loss = float(-np.log(np.maximum((c * w).sum(axis=1), EPS)).mean())
    return [float(v) for v in w], loss


def fit(train_rows: List[Dict[str, Any]], replay: Dict[Tuple[str, int], Dict[str, Any]], note: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The model's numbers from TRAIN rows. The "prior" parts use every train row, the others those the replay covers."""
    rows = _scored(train_rows)
    covered = [(r, replay[(r["game"], r["turn"])]) for r in rows if (r["game"], r["turn"]) in replay]
    params = default_params()

    def folded_of(row: Dict[str, Any], answer: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        bench, hidden = _bench_of(row["ctx"])
        return fold_search(search_view(answer["opp"]), bench, hidden) if answer else None

    # P(switch): only turns on which it has somebody to switch to.
    def can_switch(row: Dict[str, Any]) -> bool:
        bench, hidden = _bench_of(row["ctx"])
        return bool(bench) or hidden

    xs, ys = [], []
    for row, answer in covered:
        f = folded_of(row, answer)
        if f is None or not can_switch(row):
            continue
        xs.append(switch_features(row["ctx"], f["switch_share"]))
        ys.append(1.0 if row["label"]["type"] == "switch" else 0.0)
    params["switch"]["with_search"] = _logistic_fit(xs, ys)
    xp = [switch_features(r["ctx"], None) for r in rows if can_switch(r)]
    yp = [1.0 if r["label"]["type"] == "switch" else 0.0 for r in rows if can_switch(r)]
    prior = _logistic_fit([[v for i, v in enumerate(x) if i != 1] for x in xp], yp)
    params["switch"]["prior"] = prior[:1] + [0.0] + prior[1:]

    # P(target | switch)
    model = OpponentModel(params)
    switches = [(r, a) for r, a in covered if r["label"]["type"] == "switch"]
    best: Optional[Tuple[float, float, float, float]] = None
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97):
        for kappa in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
            for gamma in (0.5, 0.75, 1.0, 1.5):
                params["target"].update({"w": w, "kappa": kappa, "gamma": gamma})
                loss = 0.0
                for row, answer in switches:
                    bench, hidden = _bench_of(row["ctx"])
                    t = model.targets(bench, hidden, folded_of(row, answer))
                    loss += -math.log(max(t.get(truth_name(row["label"])[7:], 0.0), EPS))
                if best is None or loss < best[0]:
                    best = (loss, w, kappa, gamma)
    if best is not None:
        params["target"].update({"w": best[1], "kappa": best[2], "gamma": best[3]})
    best_k: Optional[Tuple[float, float]] = None
    for kappa in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        params["target"]["prior_kappa"] = kappa
        loss = 0.0
        for row in rows:
            if row["label"]["type"] != "switch":
                continue
            bench, hidden = _bench_of(row["ctx"])
            loss += -math.log(max(model.targets(bench, hidden, None).get(truth_name(row["label"])[7:], 0.0), EPS))
        if best_k is None or loss < best_k[0]:
            best_k = (loss, kappa)
    params["target"]["prior_kappa"] = best_k[1] if best_k else 2.0

    # P(move | it stays): mixture weights by EM for every temperature, the best temperature kept.
    def likelihoods(pairs: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]], temperature: float, parts: Sequence[str]) -> List[List[float]]:
        out = []
        for row, answer in pairs:
            f = folded_of(row, answer)
            candidates, _bench, _hidden = _candidates(row["ctx"], f)
            p = move_parts(row["ctx"], f, candidates, temperature)
            pool = p["pool"] or {}
            true = row["label"]["name"]
            out.append([(p[name] or pool).get(true, 0.0) for name in parts])
        return out

    stays = [(r, a) for r, a in covered if r["label"]["type"] == "move"]
    best_m: Optional[Tuple[float, float, List[float]]] = None
    for temperature in T_GRID:
        weights, loss = _em(likelihoods(stays, temperature, MOVE_PARTS))
        if best_m is None or loss < best_m[0]:
            best_m = (loss, temperature, weights)
    if best_m is not None:
        params["move"]["T"] = best_m[1]
        params["move"]["weights"] = {name: round(w, 5) for name, w in zip(MOVE_PARTS, best_m[2])}
    prior_parts = [p for p in MOVE_PARTS if p != "search"]
    weights, _loss = _em(likelihoods([(r, None) for r in rows if r["label"]["type"] == "move"], 1.0, prior_parts))
    params["move"]["prior_weights"] = dict({"search": 0.0}, **{name: round(w, 5) for name, w in zip(prior_parts, weights)})

    # P(Tera | a move, and it still can)
    can = [r for r in rows if r["label"]["type"] == "move" and not r["ctx"]["opp"]["tera_used"]]
    base_rate = sum(1 for r in can if r["label"]["tera"]) / max(1, len(can))
    params["tera"]["base_rate"] = round(base_rate, 5)
    xt, yt = [], []
    bands = [[0, 0] for _ in range(len(TERA_EDGES) + 1)]
    for row, answer in stays:
        if row["ctx"]["opp"]["tera_used"]:
            continue
        f = folded_of(row, answer)
        if f is None or not f["moves"]:
            continue
        share = sum(f["tera"].values()) / max(1e-9, sum(f["moves"].values()))
        xt.append(share)
        yt.append(1.0 if row["label"]["tera"] else 0.0)
        cell = bands[_band(share, TERA_EDGES)]
        cell[0] += int(row["label"]["tera"])
        cell[1] += 1
    # Each band's rate, drawn towards the overall rate by ten turns' worth of it.
    params["tera"]["edges"] = list(TERA_EDGES)
    params["tera"]["rates"] = [round((k + 10.0 * base_rate) / (n + 10.0), 5) for k, n in bands]
    params["tera"]["band_counts"] = bands

    its = sorted(int(a.get("iterations") or 0) for _r, a in covered)
    params["fitted"] = dict({
        "train_games": len({r["game"] for r in rows}), "train_rows": len(rows),
        "replayed_games": len({r["game"] for r, _a in covered}), "replayed_rows": len(covered),
        "switch_rows": len(xs), "target_rows": len(switches), "move_rows": len(stays), "tera_rows": len(xt), "tera_turns": int(sum(yt)),
        "iterations_per_decision_mean": round(sum(its) / len(its), 1) if its else None,
        "iterations_per_decision_median": its[len(its) // 2] if its else None,
    }, **(note or {}))
    return params


def reliability(pairs: List[Tuple[float, float]], edges: Sequence[float] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)) -> Dict[str, Any]:
    """[(what the model said, 1.0 if it happened else 0.0)] in bins: how often it happened where the model said so much."""
    bins = []
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = [(p, y) for p, y in pairs if lo <= p < hi]
        if not inside:
            continue
        said = sum(p for p, _y in inside) / len(inside)
        happened = sum(y for _p, y in inside) / len(inside)
        error += abs(said - happened) * len(inside)
        bins.append({"from": lo, "to": min(1.0, hi), "n": len(inside), "said": round(said, 4), "happened": round(happened, 4)})
    return {"bins": bins, "ece": round(error / max(1, len(pairs)), 4), "n": len(pairs)}


def evaluate(model: OpponentModel, rows: List[Dict[str, Any]], replay: Dict[Tuple[str, int], Dict[str, Any]],
             baselines: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Scores on the rows the replay covers: bench/opp_predict_eval.score for the model, for the model
    without the search, for the search's own shares, and for the given baselines; then the
    reliability of the model's confidence, of P(switch) and of P(Tera).
    """
    from bench.opp_predict_eval import score
    from bench.opp_replay import _opponent_prediction

    covered = [r for r in rows if r["kind"] == "turn" and (r["game"], r["turn"]) in replay]
    searches = {id(r["ctx"]): replay[(r["game"], r["turn"])]["opp"] for r in covered}
    raw = {id(r["ctx"]): _opponent_prediction({"opponent": {"options": [{"name": o[0], "share": o[1], "seen": o[2]} for o in replay[(r["game"], r["turn"])]["opp"]["options"]]}}) or {}
           for r in covered}
    results: Dict[str, Any] = {
        "model (search + what people do)": score(model.predictor(searches), covered),
        "model without the search (what people do)": score(model.predictor({}), covered),
        "the search's own shares": score(lambda ctx: raw.get(id(ctx), {}), covered),
    }
    for name, fn in (baselines or {}).items():
        results[name] = score(fn, covered)

    scored = _scored(covered)
    top_pairs, switch_pairs, tera_pairs, true_pairs = [], [], [], []
    reads: Dict[str, List[float]] = {}
    for row in scored:
        prediction = model.predict(row["ctx"], searches.get(id(row["ctx"])))
        true = truth_name(row["label"])
        name, p = prediction.top(1)[0]
        top_pairs.append((p, 1.0 if name == true else 0.0))
        switched = row["label"]["type"] == "switch"
        switch_pairs.append((prediction.p_switch, 1.0 if switched else 0.0))
        true_pairs.append(prediction.dist.get(true, 0.0))
        if not switched and not row["ctx"]["opp"]["tera_used"]:
            tera_pairs.append((prediction.p_tera, 1.0 if row["label"]["tera"] else 0.0))
        # The reads choose() acts on, at each confidence a setting could ask for.
        for level in (0.5, 0.55, 0.6, 0.7, 0.8):
            if prediction.p_switch >= level:
                reads.setdefault("switch>=%.2f" % level, []).append(1.0 if switched else 0.0)
                target = max(((n, v) for n, v in prediction.dist.items() if n.startswith("switch ")), key=lambda kv: kv[1])[0]
                reads.setdefault("switch>=%.2f, and the named target" % level, []).append(1.0 if target == true else 0.0)
            elif p >= level:
                reads.setdefault("%s>=%.2f" % ("move" if name.startswith("move ") else "switch target", level), []).append(1.0 if name == true else 0.0)
            if max(prediction.p_switch, p) >= level:
                right = switched if prediction.p_switch >= level else name == true
                reads.setdefault("any read>=%.2f" % level, []).append(1.0 if right else 0.0)
    base = sum(y for _p, y in tera_pairs) / max(1, len(tera_pairs))

    def log_loss(pairs: List[Tuple[float, float]], constant: Optional[float] = None) -> Optional[float]:
        if not pairs:
            return None
        total = 0.0
        for p, y in pairs:
            q = constant if constant is not None else p
            q = min(1 - 1e-4, max(1e-4, q))
            total += -(y * math.log(q) + (1 - y) * math.log(1 - q))
        return round(total / len(pairs), 4)

    its = sorted(int(replay[(r["game"], r["turn"])].get("iterations") or 0) for r in scored)
    return {
        "games": len({r["game"] for r in covered}), "turn_rows": len(covered), "scored_rows": len(scored),
        "iterations_per_decision_mean": round(sum(its) / len(its), 1) if its else None,
        "iterations_per_decision_median": its[len(its) // 2] if its else None,
        "results": results,
        "confidence_in_its_likeliest_action": reliability(top_pairs),
        "p_switch": dict(reliability(switch_pairs, (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001)),
                         log_loss=log_loss(switch_pairs), log_loss_of_the_base_rate=log_loss(switch_pairs, sum(y for _p, y in switch_pairs) / max(1, len(switch_pairs)))),
        "p_tera": dict(reliability(tera_pairs, (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0001)), turns_with_tera=int(sum(y for _p, y in tera_pairs)),
                       log_loss=log_loss(tera_pairs), log_loss_of_the_base_rate=log_loss(tera_pairs, base)),
        "reads": {k: {"n": len(v), "right": round(sum(v) / len(v), 4)} for k, v in sorted(reads.items())},
        "probability_given_to_what_happened_mean": round(sum(true_pairs) / max(1, len(true_pairs)), 4),
    }


def _print_results(report: Dict[str, Any]) -> None:
    cols = ["n", "top1", "top3", "logloss", "switch_or_stay", "switch_auc", "named_on_switch", "target_given_sw",
            "target_given_sw_revealed", "top1_move_turns", "top1_move_revealed"]
    print("| predictor | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, r in report["results"].items():
        print("| %s | " % name + " | ".join("-" if r.get(c) is None else str(r[c]) for c in cols) + " |")
    for key, title in (("confidence_in_its_likeliest_action", "When the model gives its likeliest action this much, how often is it right"),
                       ("p_switch", "When the model says it switches with this much, how often does it"),
                       ("p_tera", "When the model says the move comes with Tera with this much, how often does it")):
        block = report[key]
        print("\n%s (n = %d, mean gap %.3f):" % (title, block["n"], block["ece"]))
        print("| said between | turns | said on average | happened |")
        print("|---|---|---|---|")
        for b in block["bins"]:
            print("| %.2f and %.2f | %d | %.1f%% | %.1f%% |" % (b["from"], b["to"], b["n"], 100 * b["said"], 100 * b["happened"]))
        if "log_loss" in block:
            print("log loss %s, of the base rate alone %s" % (block["log_loss"], block["log_loss_of_the_base_rate"]))
    print("\nReads at a confidence (what choose() would act on):")
    for name, r in report["reads"].items():
        print("  %-44s %4d turns, right %.1f%%" % (name, r["n"], 100 * r["right"]))


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from bench.opp_dataset import load_rows
    from bench.opp_predict_eval import Fitted

    parser = argparse.ArgumentParser(description="Fits and scores the opponent model on the recorded games.")
    parser.add_argument("cmd", choices=("fit", "eval"))
    parser.add_argument("--replay", required=True, help="runs/opp_pred/model/replay_<tag>.jsonl (bench/opp_model_replay.py)")
    parser.add_argument("--eval-replay", default="", help="eval only: score with the search answers of this file instead (same games, another search time)")
    parser.add_argument("--params", default=PARAMS_FILE)
    parser.add_argument("--exact-only", action="store_true", help="only games whose requests were recorded, not rebuilt")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    rows = list(load_rows())
    replay = load_replay(args.replay)
    if args.exact_only:
        replay = {k: v for k, v in replay.items() if v.get("exact")}
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    if args.cmd == "fit":
        params = fit(train, replay, {"replay_file": os.path.basename(args.replay), "exact_only": bool(args.exact_only)})
        with open(args.params, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(params, fh, indent=1)
        print(json.dumps(params, indent=1))
        return 0
    model = OpponentModel.load(args.params)
    if not model.fitted:
        print("no fitted model at %s: the unfitted defaults are scored" % args.params)
    baselines = Fitted(train).predictors()
    used = load_replay(args.eval_replay) if args.eval_replay else replay
    if args.exact_only:
        used = {k: v for k, v in used.items() if v.get("exact")}
    report = {"params": model.params, "replay_file": os.path.basename(args.eval_replay or args.replay)}
    for split, part in (("test", test), ("train", train)):
        report[split] = evaluate(model, part, used, baselines)
        print("\n%s split: %d games, %d scored turns, %s iterations per decision (median %s)\n" % (
            split.upper(), report[split]["games"], report[split]["scored_rows"],
            report[split]["iterations_per_decision_mean"], report[split]["iterations_per_decision_median"]))
        _print_results(report[split])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
