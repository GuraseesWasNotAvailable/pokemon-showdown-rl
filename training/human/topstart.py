#!/usr/bin/env python
"""
The start of run T ("topstart"): the from-scratch network learns the top players' own decisions
OFFLINE, from random weights, before any self-play. The output is a learner checkpoint of mode
"imitation" that pipeline.py continues with warmup and PPO (cloud/v2/topstart_setup.py), as it
continues the PSBot imitation phase of run S. Nothing here is imported by the trainer: with this
file unused, nothing changes.

    python human/topstart.py link    --manifest imitation_data.json --set 40k=DIR_A --set bulk=DIR_B --dst ALL
    python human/topstart.py index   --data ALL [--manifest imitation_data.json] [--out counts.json]
    python human/topstart.py train   --data ALL --work WORK --out START.pt [--init P0.pt (the control C0)] ...
    python human/topstart.py measure --data ALL --checkpoint X.pt --out X.measure.json

link: the manifest's own recipe. Each set's read_shards (never a skip_shard) are linked into one
directory with distinct names, shard-<set>-<rest of the name>, so human_imitation's index reads the
two sets as one (one salt, one hash per player: the 3% player cap counts both sets together).

The data (human_imitation.py's pieces, not a copy of them): HumanImitation's index of the linked
directory (ep_split roles train and test_player only, duplicates removed, the per-player cap),
its capped held-out set (_build_holdout: holdout_decisions 32,768, no player above 3% of them),
Shard (an episode with a value that is not finite is dropped at load), pieces_of, pack and gather.
The held-out shards are offered to _build_holdout in a fixed random order (--holdout-order shuffled,
the default; the seed is HOLDOUT_SEED, not --seed, so every arm and every measure has the same set):
in name order every shard-40k-* comes before shard-bulk-*, and the set was 76% 40k games and 4% 2025
games against 42% 40k sides over all of test_player. --holdout-order spread is that old order (to
resume a work directory written before). The set's digest, its decisions per set and per period
are in the counts; a resume file written on another held-out set is refused.
A shard built with another observation is refused (exit 2), never skipped.

The network: obs_matchup.build_model with own_tera_type, matchup and movesets on (10,295 values a
step) and the SetHead at coef 0.1, the learner's own build; random weights (seed 1) or --init.
S's P0 (phase_final/imitation_learner_update_000324.pt) has the matchup block but not the moveset
block (S was widened in PPO): for --init and measure use a copy widened by
tools/widen_checkpoint.py --movesets (zero columns: the same outputs).

The loss, per step of 3 packed sequences of 256 rows (the learner's 768-row minibatch, whole
episodes, padding rows carry no loss):

    smoothed CE (0.1 on the legal actions, human/imitate.py's smoothed_ce_rows) over the labelled
    rows, each weighted by its player's cap  +  vf_coef x ppo_core.dist_value_loss on the real rows
    +  ppo_core.compute_set_head_loss on the shards' set labels at 0.1

The value target is the Monte Carlo return a rollout worker would see with the run's reward:
G_t = gamma^(T-1-t) z - scale Phi_t (the potential shaping telescopes, Phi at the end is 0), with
z from the shard's ep_won. NOTE the shards' encoding (human/rebuild.py): ep_won 1 = this side won,
0 = this side lost, -1 = no winner (a tie or no result). So z = +1, -1, 0 for 1, 0, -1
(outcome_z). human/imitate.py (arm B) read it as 1 -> +1, 0 -> 0, -1 -> -1: every loss a draw.

Optimizer: AdamW (eps 1e-5, weight decay 0.01 on the learner's decay parameters) through
ppo_core.MasterWeightOptimizer (bf16 model, fp32 masters), gradient norm 0.5; every zone the
learner's "imitation" mode trains at 1.0 x lr (the critic too), the JEPA zone frozen. A step whose
loss or gradient norm is not finite is skipped and counted; three in a row stop with exit 3.

The stream: a pass reads every train shard once, in an order seeded by (seed, pass), in groups of
--group-shards; the next group loads in the background (--threads, nice 10). Packing inside a
group is seeded by (seed, pass, group), so every batch is a function of (seed, pass, group).

The schedule (pure functions below, tests/test_topstart.py holds them): warmup over --warmup steps,
hold at --lr, then a cosine to 0 over --anneal-steps. The anneal starts at the first held-out
measurement at or after --min-hold whose capped top-1 (the best so far) gained less than
--plateau-gain over the best measured --plateau-window steps before, and at --max-steps minus the
anneal at the latest. The run stops where the anneal ends. --abort-at STEP:BAR stops the run with
no start written (exit 4) when the first measurement at or after STEP is below BAR. The control C0
(--init P0 --warmup 100 --min-hold 0 --max-passes 3 --abort-at off) anneals from step 0.

Files in --work, numbered and never overwritten: resume_<step>.pt (every --resume-every steps, at a
group boundary; a rerun resumes from the newest), best_<step>.pt (model only, from --best-after on,
when the held-out measure improves), log.jsonl (appended), history.json and info.json (at the end).
--out (refused if it exists) is written from the best: a learner payload (model with set_head.*,
target_model None, no optimizer or scheduler, update_idx 0, total_steps = rows seen, run_cfg with
mode imitation, the three observation settings and set_head_coef 0.1) plus a "topstart" info dict;
also <out>.sha256 and <out>.json. Logs and outputs hold counts only, never a player hash.

measure: the capped held-out set, all of test_player (pooled and per-player mean), test_game (its
own ep_split selection), per set (link-name prefix) and per period (ep_time), SetHead accuracies
when the checkpoint has the head, value loss. Writes JSON (counts only).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import glob
import hashlib
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import human_imitation as hi  # noqa: E402

L_SEQ = 256                      # the learner's batch_seq_len
SEQS = 3                         # sequences a step: 3 x 256 = the learner's 768-row minibatch
SET_HEAD_COEF = 0.1
SMOOTHING = 0.1
ROLES_MEASURED = ("test_player", "test_game")
OBS_ALL_ON = {"own_tera_type": True, "matchup": True, "movesets": True}
REQUIRED_FIELDS = ("phi", "ep_won", "ep_time", "set_labels", "ep_split")
EXIT_REFUSED, EXIT_NONFINITE, EXIT_ABORT = 2, 3, 4
HOLDOUT_SEED = 20261018          # the held-out shards' order (--holdout-order shuffled): the same set for every arm and measure
HOLDOUT_ORDERS = ("shuffled", "spread")


def log(*a: Any) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Refused(SystemExit):
    """A refusal (exit 2) with its reason printed."""

    def __init__(self, msg: str, code: int = EXIT_REFUSED):
        print(f"REFUSED: {msg}", flush=True)
        super().__init__(code)


# =============================================================================================
# the schedule (pure functions)
# =============================================================================================

@dataclasses.dataclass(frozen=True)
class Schedule:
    lr: float = 3e-4
    warmup: int = 1000
    anneal_steps: int = 25_000
    min_hold: int = 60_000
    plateau_gain: float = 0.005
    plateau_window: int = 20_000
    max_steps: int = 150_000

    def anneal_len(self) -> int:
        """The anneal's length: --anneal-steps, or the whole run when that is shorter (C0)."""
        return max(1, min(int(self.anneal_steps), int(self.max_steps)))

    def latest_start(self) -> int:
        return max(0, int(self.max_steps) - self.anneal_len())


def lr_share(step: int, sched: Schedule, anneal_start: Optional[int]) -> float:
    """The share of --lr for the step with this index (0-based): warmup ramp x cosine anneal."""
    warm = min(1.0, float(step + 1) / float(max(1, sched.warmup)))
    if anneal_start is None or step < anneal_start:
        return warm
    p = min(1.0, float(step - anneal_start) / float(sched.anneal_len()))
    return warm * 0.5 * (1.0 + math.cos(math.pi * p))


def plateau_reached(history: Sequence[Tuple[int, float]], done: int, sched: Schedule) -> bool:
    """
    At a measurement after `done` steps: the capped top-1 (the best so far) gained less than
    plateau_gain over the best of the measurements at least plateau_window steps earlier. The best
    so far, not the last value, so one noisy measurement neither starts nor delays the anneal.
    """
    if done < sched.min_hold:
        return False
    now = [t for s, t in history if s <= done]
    before = [t for s, t in history if s <= done - sched.plateau_window]
    if not now or not before:
        return False
    return max(now) - max(before) < sched.plateau_gain


def anneal_start_before_step(step: int, anneal_start: Optional[int], sched: Schedule) -> Optional[int]:
    """Before the step with this index: the latest start forces the anneal (a measurement may have started it earlier)."""
    if anneal_start is None and step >= sched.latest_start():
        return int(step)
    return anneal_start


def anneal_start_after_measure(done: int, anneal_start: Optional[int], history: Sequence[Tuple[int, float]],
                               sched: Schedule) -> Optional[int]:
    """After the measurement at `done` steps: the plateau starts the anneal at the next step."""
    if anneal_start is None and plateau_reached(history, done, sched):
        return int(done)
    return anneal_start


def stop_step(anneal_start: Optional[int], sched: Schedule) -> Optional[int]:
    """The number of steps the run ends at (the anneal's end), once the anneal has started."""
    return None if anneal_start is None else int(anneal_start) + sched.anneal_len()


def parse_abort(spec: str) -> Optional[Tuple[int, float]]:
    if spec is None or str(spec).strip().lower() in ("", "off", "none", "0"):
        return None
    a, b = str(spec).split(":", 1)
    return int(a), float(b)


def abort_due(done: int, top1: float, abort: Optional[Tuple[int, float]], checked: bool) -> bool:
    """At the FIRST measurement at or after the abort step: below the bar stops the run."""
    return abort is not None and not checked and done >= abort[0] and top1 < abort[1]


def better(m: Dict[str, float], best: Optional[Dict[str, float]], tie: float = 0.002) -> bool:
    """Capped top-1 decides; within `tie` (0.2 points) of each other the lower cross-entropy does."""
    if best is None:
        return True
    if m["top1"] > best["top1"] + tie:
        return True
    if m["top1"] < best["top1"] - tie:
        return False
    return m["ce"] < best["ce"]


# =============================================================================================
# the value target
# =============================================================================================

def outcome_z(won: int) -> float:
    """human/rebuild.py's ep_won: 1 won, 0 lost, -1 no winner (tie or no result) -> +1, -1, 0."""
    won = int(won)
    return 1.0 if won == 1 else (-1.0 if won == 0 else 0.0)


def value_targets(phi: np.ndarray, z: float, gamma: float, scale: float, shaping: bool) -> np.ndarray:
    """G_t = gamma^(T-1-t) z - scale Phi_t for one episode (human/imitate.py's target, Phi's NaN as 0)."""
    n = len(phi)
    k = np.arange(n - 1, -1, -1, dtype=np.float64)
    ret = (gamma ** k) * float(z)
    if shaping:
        ret = ret - float(scale) * np.nan_to_num(np.asarray(phi, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return ret.astype(np.float32)


# =============================================================================================
# periods and sets
# =============================================================================================

def _utc(y: int, m: int) -> int:
    return int(datetime.datetime(y, m, 1, tzinfo=datetime.timezone.utc).timestamp())


PERIODS = (("2025", None, _utc(2026, 1)), ("2026-01..06", _utc(2026, 1), _utc(2026, 7)), ("2026-07+", _utc(2026, 7), None))


def period_of(t: int) -> str:
    for name, lo, hi_ in PERIODS:
        if (lo is None or t >= lo) and (hi_ is None or t < hi_):
            return name
    return "?"


_SET_RE = re.compile(r"^shard-([a-z0-9]+)-\d+-")


def set_of(path: str) -> str:
    """The set a linked shard came from (its link-name prefix: shard-40k-..., shard-bulk-...)."""
    m = _SET_RE.match(os.path.basename(path))
    return m.group(1) if m else "other"


# =============================================================================================
# shards with the value target, labels and the per-episode facts the measures group by
# =============================================================================================

class _Cached:
    """An open npz whose arrays are read (decompressed) once each."""

    def __init__(self, z: Any):
        self._z, self._c = z, {}
        self.files = z.files

    def __getitem__(self, k: str) -> np.ndarray:
        if k not in self._c:
            self._c[k] = self._z[k]
        return self._c[k]


def make_shard_class(gamma: float, scale: float, shaping: bool):
    """hi.Shard plus the value target (ret), each kept episode's ep_time, the row's episode and the set name."""

    class TShard(hi.Shard):
        __slots__ = ("ret", "ep_time", "row_ep", "set_name", "ep_won", "ep_src")

        def __init__(self, path, uid, z, episodes, trans0, dim, settings, player_w, rng, with_set_labels=False):
            # Set labels always (the SetHead learns and is measured on them); the rest is hi.Shard's.
            super().__init__(path, uid, z, episodes, trans0, dim, settings, player_w, rng, with_set_labels=True)
            gone = {(a, b) for a, b, _p in self.dropped}
            kept = [(a, b) for a, b, _p in episodes if (a, b) not in gone]
            if len(kept) != len(self.eps):
                raise RuntimeError(f"{os.path.basename(path)}: {len(kept)} episodes kept, the shard holds {len(self.eps)}")
            st = np.asarray(z["ep_start"], dtype=np.int64)
            phi = z["phi"]
            won, times = z["ep_won"], z["ep_time"]
            ret = np.empty(self.rows(), dtype=np.float32)
            row_ep = np.empty(self.rows(), dtype=np.int32)
            self.ep_time, self.ep_won = [], []
            for k, (a, b) in enumerate(kept):
                e = int(np.searchsorted(st, a, side="right") - 1)
                if int(st[e]) != a or int(st[e + 1]) != b:
                    raise RuntimeError(f"{os.path.basename(path)}: rows {a}:{b} are not one episode")
                la, lb = self.eps[k]
                ret[la:lb] = value_targets(phi[a:b], outcome_z(int(won[e])), gamma, scale, shaping)
                row_ep[la:lb] = k
                self.ep_time.append(int(times[e]))
                self.ep_won.append(int(won[e]))
            if len(ret) != self.rows():
                raise AssertionError("the value target's length is not the shard's rows")
            self.ret, self.row_ep, self.set_name, self.ep_src = ret, row_ep, set_of(path), kept

    return TShard


@contextlib.contextmanager
def shard_class(cls):
    """hi.Shard as `cls` while HumanImitation._build_holdout builds the held-out set (main thread, before any loader)."""
    old = hi.Shard
    hi.Shard = cls
    try:
        yield
    finally:
        hi.Shard = old


def gather_extra(seqs: Sequence[List[Tuple[Any, int, int, int]]], L: int) -> Dict[str, np.ndarray]:
    """Beside hi.gather (same row layout): the value target, real (not padding) and labelled rows."""
    S = len(seqs)
    ret = np.zeros(S * L, dtype=np.float32)
    real = np.zeros(S * L, dtype=bool)
    lab = np.zeros(S * L, dtype=bool)
    for j, seq in enumerate(seqs):
        at = j * L
        for sh, a, b, _uid in seq:
            n = b - a
            ret[at:at + n] = sh.ret[a:b]
            real[at:at + n] = True
            lab[at:at + n] = sh.lab[a:b]
            at += n
    return {"ret": ret, "real": real, "lab": lab}


def batch_digest(seqs: Sequence[List[Tuple[Any, int, int, int]]], file_no: Dict[str, int]) -> str:
    """The rows of a batch, as an id: (file, first row, end row) of every piece, in order."""
    h = hashlib.sha1()
    for seq in seqs:
        for sh, a, b, _uid in seq:
            h.update(f"{file_no.get(sh.path, -1)}:{a}:{b};".encode())
        h.update(b"|")
    return "b" + str(int(h.hexdigest()[:15], 16))          # digits only: no hex run in a log


def order_holdout_shards(files: Sequence[str], how: str = "shuffled") -> List[str]:
    """The held-out shards in the order _build_holdout is offered them. "spread": as listed (it takes every
    len/16-th first, then the rest in name order, so every shard-40k-* comes before any shard-bulk-*); "shuffled":
    the names sorted, then a fixed permutation (HOLDOUT_SEED), so the set samples both sets and every period."""
    if how not in HOLDOUT_ORDERS:
        raise Refused(f"--holdout-order {how!r}: one of {HOLDOUT_ORDERS}")
    out = list(files)
    if how == "shuffled":
        out.sort(key=os.path.basename)
        random.Random(HOLDOUT_SEED).shuffle(out)
    return out


def group_seed(seed: int, p: int, g: int) -> int:
    return int.from_bytes(hashlib.sha256(f"topstart/{seed}/{p}/{g}".encode()).digest()[:8], "little")


def pass_groups(files: Sequence[str], seed: int, p: int, G: int) -> List[List[str]]:
    """Pass p: every file once, in an order seeded by (seed, p), cut into groups of G."""
    order = np.random.default_rng([int(seed), int(p)]).permutation(len(files))
    return [[files[i] for i in order[k:k + G]] for k in range(0, len(files), G)]


# =============================================================================================
# the data: human_imitation's index, its held-out set, our loaders
# =============================================================================================

class TopData:
    """
    HumanImitation over the linked directory, with the fields start() sets (L, trans0, a0, a1, dim)
    set from the network and its thread never started; then the shard scan this module adds
    (required fields, test_game, ep_time) and the capped held-out set.
    """

    def __init__(self, data: str, offsets: Dict[str, Tuple[int, int]], seed: int, cap: float = 0.03,
                 holdout_decisions: int = 32_768, gamma: float = 0.999, scale: float = 0.5, shaping: bool = True,
                 threads: int = 6, build_holdout: bool = True, holdout_order: str = "shuffled"):
        self.data = os.path.expanduser(data)
        self.holdout_how = holdout_order
        settings = dict(data=self.data, coef=1.0, cap=float(cap), holdout_decisions=int(holdout_decisions), seed=int(seed),
                        set_coef=SET_HEAD_COEF)
        self.h = hi.HumanImitation(settings, OBS_ALL_ON)
        self.h.L = L_SEQ
        self.h.trans0 = int(offsets["transition_scalars"][0])
        self.h.a0, self.h.a1 = (int(x) for x in offsets["action_mask"])
        self.h.dim = int(max(e for _s, e in offsets.values()))
        self.h.started = True                    # never started: no loader thread
        self.h.status = "on"                     # what holdout_agreement asks of it
        self.dim = self.h.dim
        self.threads = int(threads)
        self.Shard = make_shard_class(gamma, scale, shaping)
        self.dropped_keys: set = set()
        files = hi.shard_files(self.data)
        if not files:
            raise Refused(f"no shards in {self.data}")
        self.h._build_index()
        if self.h.off_reason:
            raise Refused(self.h.off_reason)
        idx = self.h.index
        if idx["mismatch"]:
            raise Refused(f"{idx['mismatch']} of {idx['files']} shards in {self.data} were built with another observation "
                          f"than this network's ({hi.describe_obs((True, True, True, ''))}, obs_dim={self.dim}); "
                          f"every shard must be (none is skipped)")
        if idx["roles"].get("unreadable"):
            raise Refused(f"{idx['roles']['unreadable']} shard(s) in {self.data} could not be read")
        self.files = list(idx["train_shards"])
        self.file_no = {f: k for k, f in enumerate(files)}
        self._scan(files)
        self.holdout_seqs: List[Any] = []
        self.holdout_info: Dict[str, Any] = {}
        if build_holdout:
            listed = list(idx["holdout_shards"])
            idx["holdout_shards"] = order_holdout_shards(listed, holdout_order)
            try:
                with shard_class(self.Shard):
                    self.h._build_holdout()
            finally:
                idx["holdout_shards"] = listed           # the index as human_imitation built it
            if self.h.holdout is None:
                raise Refused("no held-out decisions (test_player) in the data")
            self.holdout_seqs = self.h.holdout["seqs"]
            for seq in self.holdout_seqs:
                for sh, _a, _b, _u in seq:
                    self._count_dropped(sh)
            self.holdout_info = self._holdout_info()

    def _holdout_info(self) -> Dict[str, Any]:
        """The capped held-out set's order, digest (shard names and episode rows: no player) and labelled decisions
        per set and per period."""
        shards: Dict[int, Any] = {}
        by_set: Dict[str, int] = defaultdict(int)
        by_period: Dict[str, int] = defaultdict(int)
        for seq in self.holdout_seqs:
            for sh, a, b, _u in seq:
                shards[id(sh)] = sh
                lab = np.asarray(sh.lab[a:b], dtype=bool)
                eps = np.asarray(sh.row_ep[a:b])
                by_set[sh.set_name] += int(lab.sum())
                for e in np.unique(eps[lab]):
                    by_period[period_of(sh.ep_time[int(e)])] += int((eps[lab] == e).sum())
        rows = sorted((os.path.basename(sh.path), int(a), int(b)) for sh in shards.values() for a, b in sh.ep_src)
        digest = int(hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:15], 16)    # digits only, as batch_digest
        return {"order": self.holdout_how, "digest": f"ho{digest}", "shards": len(shards),
                "by_set": dict(sorted(by_set.items())), "by_period": dict(sorted(by_period.items()))}

    def _scan(self, files: List[str]) -> None:
        """Fields every shard must have, test_game episodes (deduplicated as the index does), periods and sets."""
        seen = set()
        self.test_game: Dict[str, List[Tuple[int, int, str]]] = {}
        per_set: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        missing: Dict[str, int] = defaultdict(int)
        won = defaultdict(int)
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                for k in REQUIRED_FIELDS:
                    if k not in z.files:
                        missing[k] += 1
                if any(k not in z.files for k in REQUIRED_FIELDS):
                    continue
                st, games, sides, players = z["ep_start"], z["ep_game"], z["ep_side"], z["ep_player"]
                split, wons = z["ep_split"], z["ep_won"]
                tg = []
                for e in range(len(st) - 1):
                    g, side = str(games[e]), ("p1" if int(sides[e]) == 1 else "p2")
                    if (g, side) in seen:
                        continue
                    seen.add((g, side))
                    sp = str(split[e])
                    per_set[set_of(f)][sp] += 1
                    if sp == "train":
                        won[int(wons[e])] += 1
                    a, b = int(st[e]), int(st[e + 1])
                    if sp == "test_game" and b > a:
                        tg.append((a, b, str(players[e])))
                if tg:
                    self.test_game[f] = tg
        if missing:
            raise Refused(f"shards without fields this start needs ({dict(missing)}): the value target needs phi and ep_won, "
                          f"the SetHead set_labels, the measures ep_time and ep_split")
        self.per_set = {k: dict(v) for k, v in per_set.items()}
        self.train_won = {str(k): v for k, v in sorted(won.items())}

    def _count_dropped(self, sh: Any) -> int:
        new = [(sh.path, a) for a, _b, _p in sh.dropped if (sh.path, a) not in self.dropped_keys]
        self.dropped_keys.update(new)
        return len(new)

    @property
    def dropped(self) -> int:
        return len(self.dropped_keys)

    # -- loading -------------------------------------------------------------------------------

    def load(self, path: str, role: str, uid: int) -> Optional[Any]:
        """One shard's episodes of a role (train: with the player-cap weights) as a TShard, or None."""
        if role == "test_game":
            eps = self.test_game.get(path) or []
        else:
            eps = self.h.index["episodes"].get(path, {}).get(role) or []
        if not eps:
            return None
        with np.load(path, allow_pickle=False) as z:
            if role == "train":
                return self.Shard(path, uid, _Cached(z), eps, self.h.trans0, self.dim, self.h.s, self.h.index["player_w"], None, True)
            return self.Shard(path, uid, _Cached(z), eps, self.h.trans0, self.dim, None, None, None, True)

    def load_many(self, jobs: Sequence[Tuple[str, str, int]]) -> List[Any]:
        """Shards in parallel (--threads, each at nice 10), in the order asked."""
        with ThreadPoolExecutor(max(1, self.threads), initializer=hi.HumanImitation._lower_priority) as ex:
            out = list(ex.map(lambda j: self.load(*j), jobs))
        shards = [s for s in out if s is not None and s.rows() > 0]
        for sh in shards:
            self._count_dropped(sh)
        return shards

    def load_group(self, files: Sequence[str]) -> List[Any]:
        return self.load_many([(f, "train", 1 + j) for j, f in enumerate(files)])

    # -- counts --------------------------------------------------------------------------------

    def counts(self) -> Dict[str, Any]:
        """What the data holds, counts only (no player and no game hash)."""
        idx = self.h.index
        test_player_sides = sum(len(eps["test_player"]) for eps in idx["episodes"].values())
        out: Dict[str, Any] = {
            "files": idx["files"], "train_files": len(self.files), "roles": dict(idx["roles"]),
            "train_sides": sum(len(e["train"]) for e in idx["episodes"].values()),
            "train_labelled": idx["train_labelled"], "train_players": idx["train_players"],
            "capped_players": sum(1 for w in idx["player_w"].values() if w < 1.0),
            "test_player_sides": test_player_sides,
            "test_game_sides": sum(len(v) for v in self.test_game.values()),
            "per_set_sides": self.per_set, "train_ep_won": self.train_won,
        }
        if self.h.holdout is not None:
            out["holdout"] = {"decisions": int(self.h.ho["n"]), "players": int(self.h.holdout["players"]),
                              "limit": int(self.h.holdout["limit"]), "top_share": round(float(self.h.holdout["top_share"]), 5),
                              "sequences": len(self.holdout_seqs), **self.holdout_info}
        out["dropped_episodes"] = self.dropped
        return out

    def steps_per_group_estimate(self, seed: int, p: int, groups: List[List[str]]) -> List[int]:
        """Steps of each group of pass p from the index alone (a dropped episode makes it a little less)."""
        out = []
        for g, files in enumerate(groups):
            pieces = []
            for j, f in enumerate(files):
                for k, (a, b, _p) in enumerate(self.h.index["episodes"].get(f, {}).get("train") or []):
                    for s in range(a, b, L_SEQ):
                        pieces.append((None, s, min(b, s + L_SEQ), (1 + j) * 100_000 + k))
            seqs = hi.pack(pieces, L_SEQ, random.Random(group_seed(seed, p, g)))
            out.append(len(seqs) // SEQS)
        return out


# =============================================================================================
# the network
# =============================================================================================

@contextlib.contextmanager
def _in_repo():
    here = os.getcwd()
    os.chdir(REPO)                         # vocab.json and the set tables are opened by relative paths
    try:
        yield
    finally:
        os.chdir(here)


def run_config(set_head_coef: float = SET_HEAD_COEF, mode: str = "imitation", obs: Optional[Dict[str, bool]] = None):
    """RunConfig.default() with the three observation settings on, the learner in `mode`, the SetHead at coef 0.1, no OppActionHead."""
    from config import RunConfig
    cfg = RunConfig.default()
    o = dict(OBS_ALL_ON if obs is None else obs)
    cfg = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, own_tera_type=bool(o["own_tera_type"]),
                                                           matchup=bool(o["matchup"]), movesets=bool(o["movesets"])))
    return dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, mode=mode, set_head_coef=float(set_head_coef),
                                                                opp_action_aux_coef=0.0))


def compile_flex() -> None:
    """flex_attention compiled, as the learner does (uncompiled it falls back to dense attention)."""
    import torch
    import ppo_core
    if not getattr(ppo_core, "_learner_flex_compiled", False):
        ppo_core.flex_attention = torch.compile(ppo_core.flex_attention)
        ppo_core._learner_flex_compiled = True


def build_net(cfg: Any, seed: int, dev: Any, set_head: bool = True, opp_head: bool = False):
    """The learner's network: make_model + the SetHead (the learner's order), on the bf16 path."""
    import torch
    from obs_matchup import build_model
    torch.manual_seed(int(seed))
    with _in_repo():
        net = build_model(cfg, matchup=bool(cfg.obs.matchup), movesets=bool(cfg.obs.movesets), set_head=set_head,
                          set_head_coef=float(cfg.learner.set_head_coef), opp_action_head=opp_head)
    net = net.to(dev).train()
    net.enable_bf16_recurrent_path()
    return net


def obs_of_checkpoint(ck: Dict[str, Any]) -> Dict[str, bool]:
    from config import checkpoint_own_tera_type
    from obs_matchup import checkpoint_obs_matchup
    from obs_movesets import checkpoint_obs_movesets
    return {"own_tera_type": bool(checkpoint_own_tera_type(ck)), "matchup": bool(checkpoint_obs_matchup(ck)),
            "movesets": bool(checkpoint_obs_movesets(ck))}


def widen_hint(o: Dict[str, bool], path: str) -> str:
    """How a checkpoint with the matchup block but not the moveset one gets it (S's P0 is one: S was widened in PPO)."""
    if o.get("matchup") and not o.get("movesets"):
        return (f". Widen it first (the outputs do not change): python tools/widen_checkpoint.py --movesets --in {path} "
                f"--out <copy with movesets>")
    return ""


def load_init(net: Any, path: str) -> Dict[str, Any]:
    """--init: a checkpoint's weights into the network (strict=False). Missing learned keys other than
    set_head.* are refused, keys the network does not have other than opp_action_head.* too, and an
    observation other than the three settings on."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    o = obs_of_checkpoint(ck)
    if o != OBS_ALL_ON:
        raise Refused(f"{path} was trained with the observation {o}, this network has {OBS_ALL_ON}" + widen_hint(o, path))
    state = {k: v for k, v in ck["model"].items() if "attn_mask" not in k and not k.startswith("opp_action_head.")}
    ignored = sorted(k for k in ck["model"] if k.startswith("opp_action_head."))
    missing, unexpected = net.load_state_dict(state, strict=False)
    learned = {n for n, _ in net.named_parameters()}
    bad_missing = [k for k in missing if k in learned and not k.startswith("set_head.")]
    if bad_missing or unexpected:
        raise Refused(f"{path} does not fit the network: missing {bad_missing[:5]} unexpected {list(unexpected)[:5]}")
    return {"init": os.path.abspath(path), "update_idx": int(ck.get("update_idx") or 0),
            "mode": (ck.get("run_cfg") or {}).get("learner", {}).get("mode"),
            "set_head_from_init": not any(k.startswith("set_head.") for k in missing),
            "ignored_keys": len(ignored)}


def load_net(path: str, dev: Any):
    """A checkpoint's network as the learner holds it (agree.py's load_net): its own observation, heads and weights."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    o = obs_of_checkpoint(ck)
    lc = (ck.get("run_cfg") or {}).get("learner", {}) or {}
    sh = float(lc.get("set_head_coef") or 0.0)
    oc = float(lc.get("opp_action_aux_coef") or 0.0)
    cfg = run_config(set_head_coef=sh, obs=o)
    net = build_net(cfg, 0, dev, set_head=sh > 0 and o["matchup"], opp_head=oc > 0)
    state = {k: v for k, v in ck["model"].items() if "attn_mask" not in k}
    missing, unexpected = net.load_state_dict(state, strict=False)
    learned = {n for n, _ in net.named_parameters()}
    if [k for k in missing if k in learned] or unexpected:
        raise Refused(f"{path} does not fit the network: missing {list(missing)[:5]} unexpected {list(unexpected)[:5]}")
    net.eval()
    return net, o, ck


def zone_groups(net: Any, lr: float, wd: float):
    """
    The learner's optimizer for mode "imitation" (LearnerActor._init_optimizer with _get_zone_config):
    every zone but the JEPA one trains at 1.0 x lr, decay on all but norms, embeddings, biases and
    1-d parameters, fp32 masters through MasterWeightOptimizer.
    """
    import torch
    import ppo_core
    pairs, groups = [], []
    for zone, params in net.get_parameter_zones().items():
        train_zone = zone != "jepa"
        for _n, p in params:
            p.requires_grad = train_zone
        if not train_zone:
            continue
        decay, no_decay = [], []
        for name, p in params:
            m = p.detach().to(torch.float32).clone()
            pairs.append((p, m))
            (no_decay if (any(x in name for x in ["_emb", "norm"]) or name.endswith(".bias") or p.ndim == 1) else decay).append(m)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": wd, "name": f"{zone}_wd"})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{zone}_stable"})
    inner = torch.optim.AdamW(groups, eps=1e-5)
    return ppo_core.MasterWeightOptimizer(inner, pairs)


# =============================================================================================
# measuring
# =============================================================================================

class Tally:
    """Per key ("all", "set:40k", "period:2025", ...): top-1 hits, labelled rows, CE, value loss; per player inside (never written)."""

    def __init__(self):
        self.k: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.pp: Dict[str, Dict[int, List[int]]] = defaultdict(dict)
        self.players: Dict[str, int] = {}
        self.set: Dict[str, float] = defaultdict(float)

    def pid(self, p: str) -> int:
        return self.players.setdefault(p, len(self.players))

    def add(self, key: str, hit: np.ndarray, lab: np.ndarray, ce: np.ndarray, vrow: np.ndarray, real: np.ndarray,
            pids: np.ndarray) -> None:
        d = self.k[key]
        d["hit"] += float((hit & lab).sum())
        d["n"] += float(lab.sum())
        d["ce"] += float(ce[lab].sum())
        d["v"] += float(vrow[real].sum())
        d["vn"] += float(real.sum())
        if lab.any():
            ids = pids[lab]
            h = np.bincount(ids, weights=hit[lab].astype(np.float64), minlength=len(self.players))
            c = np.bincount(ids, minlength=len(self.players))
            pp = self.pp[key]
            for i in np.flatnonzero(c):
                cur = pp.setdefault(int(i), [0, 0])
                cur[0] += int(h[i])
                cur[1] += int(c[i])

    def result(self, key: str) -> Dict[str, Any]:
        d = self.k.get(key)
        if not d or not d["n"]:
            return {"labelled": 0}
        per = [h / c for h, c in self.pp[key].values() if c]
        return {"top1": d["hit"] / d["n"], "top1_per_player": (sum(per) / len(per)) if per else None, "ce": d["ce"] / d["n"],
                "vloss": (d["v"] / d["vn"]) if d["vn"] else None, "labelled": int(d["n"]), "players": len(per),
                "rows": int(d["vn"])}

    def results(self) -> Dict[str, Any]:
        out = {k: self.result(k) for k in sorted(self.k)}
        s = self.set
        if s.get("batches"):
            out["sethead"] = {
                "loss": s["loss"] / s["batches"],
                "ability_top1": (s["ability_top1"] / s["ability_n"]) if s["ability_n"] else None, "ability_n": int(s["ability_n"]),
                "item_top1": (s["item_top1"] / s["item_n"]) if s["item_n"] else None, "item_n": int(s["item_n"]),
                "move_hits": (s["move_hits"] / s["move_n"]) if s["move_n"] else None, "move_n": int(s["move_n"]),
                "tera_top1": (s["tera_top1"] / s["tera_n"]) if s["tera_n"] else None, "tera_n": int(s["tera_n"])}
        return out


def evaluate(net: Any, seqs: Sequence[List[Tuple[Any, int, int, int]]], dev: Any, tally: Optional[Tally] = None,
             by_set: bool = True, by_period: bool = True) -> Tally:
    """
    The network on packed held-out sequences, 3 at a time (the learner's minibatch): top-1 and CE
    on the labelled rows (pooled and per player, as HumanImitation.holdout_agreement), the value
    loss on the real rows, the SetHead's counts when the network has the head.
    """
    import torch
    import ppo_core
    tally = tally or Tally()
    lc = run_config().learner
    has_set = getattr(net, "set_head", None) is not None
    was = net.training
    net.eval()
    try:
        with torch.no_grad():
            for j in range(0, len(seqs), SEQS):
                chunk = seqs[j:j + SEQS]
                b = hi.gather(chunk, L_SEQ, with_set_labels=has_set)
                x = gather_extra(chunk, L_SEQ)
                n = len(b["act"])
                pid = np.zeros(n, dtype=np.int64)
                set_k = np.empty(n, dtype=object)
                per_k = np.empty(n, dtype=object)
                for q, seq in enumerate(chunk):
                    at = q * L_SEQ
                    for sh, a, c, _u in seq:
                        eps = sh.row_ep[a:c]
                        pid[at:at + c - a] = [tally.pid(sh.players[e]) for e in eps]
                        set_k[at:at + c - a] = sh.set_name
                        per_k[at:at + c - a] = [period_of(sh.ep_time[e]) for e in eps]
                        at += c - a
                obs = torch.from_numpy(b["obs"]).to(dev)
                ep = torch.from_numpy(b["ep"]).to(dev)
                out = net(obs, ep, batch_seq_len=L_SEQ, return_z=has_set)
                logits, v_logits = out[0].float(), out[1].float()
                mask = torch.from_numpy(b["mask"]).to(dev)
                masked = logits.masked_fill(mask <= 0.5, -1e4)
                logp = torch.log_softmax(masked, dim=-1)
                act = torch.from_numpy(b["act"]).to(dev)
                ce = -logp.gather(1, act[:, None]).squeeze(1)
                t = ppo_core.twohot_targets(torch.from_numpy(x["ret"]).to(dev), v_min=float(lc.v_min), v_max=float(lc.v_max),
                                            v_bins=int(lc.v_bins))
                vrow = -(t * torch.log_softmax(v_logits, dim=-1)).sum(-1)
                if has_set and "set_labels" in b:
                    lab_t = torch.from_numpy(b["set_labels"]).to(dev)
                    _term, st = ppo_core.compute_set_head_loss(net, out[5], lab_t.reshape(lab_t.shape[0], 6, -1), coef=SET_HEAD_COEF)
                    tally.set["batches"] += 1
                    tally.set["loss"] += float(st["set_loss"])
                    for k in ("ability", "item", "tera"):
                        if f"set_{k}_n" in st:
                            tally.set[f"{k}_n"] += float(st[f"set_{k}_n"])
                            tally.set[f"{k}_top1"] += float(st[f"set_{k}_top1"])
                    if "set_move_n" in st:
                        tally.set["move_n"] += float(st["set_move_n"])
                        tally.set["move_hits"] += float(st["set_move_hits"])
                best = masked.argmax(-1).cpu().numpy()
                ce_np = ce.cpu().numpy().astype(np.float64)
                v_np = vrow.cpu().numpy().astype(np.float64)
                hit = best == b["act"]
                lab, real = x["lab"], x["real"]
                tally.add("all", hit, lab, ce_np, v_np, real, pid)
                if by_set:
                    for name in sorted(set(s for s in set_k[real])):
                        sel = set_k == name
                        tally.add(f"set:{name}", hit, lab & sel, ce_np, v_np, real & sel, pid)
                if by_period:
                    for name in sorted(set(s for s in per_k[real])):
                        sel = per_k == name
                        tally.add(f"period:{name}", hit, lab & sel, ce_np, v_np, real & sel, pid)
    finally:
        net.train(was)
    return tally


# =============================================================================================
# files
# =============================================================================================

def unique_path(path: str) -> str:
    """path, or path with .1, .2 ... before the extension when a file of that name exists: nothing is overwritten."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(f"{stem}.{k}{ext}"):
        k += 1
    return f"{stem}.{k}{ext}"


def save_new(obj: Any, path: str, kind: str = "torch") -> str:
    """Written to a new partial file, made durable, renamed to a name nothing holds (returned)."""
    import torch
    final = unique_path(path)
    tmp = f"{final}.partial-{os.getpid()}-{time.time_ns()}"
    if kind == "torch":
        torch.save(obj, tmp)
    else:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, default=str)
    with open(tmp, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(final):                   # nothing else writes here; checked again all the same
        final = unique_path(final)
    os.rename(tmp, final)
    return final


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def newest_resume(work: str) -> Optional[str]:
    best, path = -1, None
    for p in glob.glob(os.path.join(work, "resume_*.pt")):
        m = re.match(r"resume_(\d+)(\.\d+)?\.pt$", os.path.basename(p))
        if m and int(m.group(1)) >= best:
            if int(m.group(1)) > best or os.path.getmtime(p) > os.path.getmtime(path):
                best, path = int(m.group(1)), p
    return path


class JsonLog:
    """log.jsonl in the work directory: appended, never rewritten."""

    def __init__(self, path: str):
        self.path = path

    def write(self, rec: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


# =============================================================================================
# train
# =============================================================================================

def _settings(a: argparse.Namespace, sched: Schedule) -> Dict[str, Any]:
    return {"seed": a.seed, "group_shards": a.group_shards, "cap": a.cap, "holdout_decisions": a.holdout_decisions,
            "measure_every": a.measure_every, "resume_every": a.resume_every, "best_after": a.best_after,
            "abort_at": a.abort_at, "wd": a.wd, "max_grad_norm": a.max_grad_norm, "smoothing": SMOOTHING,
            "set_head_coef": SET_HEAD_COEF, "seq": L_SEQ, "seqs_per_step": SEQS, "init": os.path.abspath(a.init) if a.init else None,
            "schedule": dataclasses.asdict(sched), "holdout_order": a.holdout_order}


def _same_run(saved: Dict[str, Any], now: Dict[str, Any]) -> List[str]:
    keys = ("seed", "group_shards", "cap", "holdout_decisions", "smoothing", "set_head_coef", "seq", "seqs_per_step", "init",
            "schedule", "wd", "max_grad_norm")
    diff = [k for k in keys if saved.get(k) != now.get(k)]
    # A resume file from before --holdout-order was made with the spread order (and has no digest to compare).
    if saved.get("holdout_order", "spread") != now.get("holdout_order", "spread"):
        diff.append("holdout_order")
    if saved.get("holdout_digest") and now.get("holdout_digest") and saved["holdout_digest"] != now["holdout_digest"]:
        diff.append("holdout_digest")
    return diff


def _batches(seqs: List[Any], file_no: Dict[str, int], stop: threading.Event, depth: int = 6) -> Iterator[Dict[str, Any]]:
    """The group's steps, gathered on a thread a few steps ahead, in order."""
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    n = len(seqs) // SEQS

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def run():
        try:
            hi.HumanImitation._lower_priority()
            for i in range(n):
                chunk = seqs[i * SEQS:(i + 1) * SEQS]
                b = hi.gather(chunk, L_SEQ, with_set_labels=True)
                b.update(gather_extra(chunk, L_SEQ))
                b["digest"] = batch_digest(chunk, file_no)
                if not put(b):
                    return
        except BaseException as e:      # noqa: BLE001  (handed to the consumer)
            put(e)
            return
        put(None)

    th = threading.Thread(target=run, name="topstart-gather", daemon=True)
    th.start()
    try:
        while True:
            b = q.get()
            if b is None:
                break
            if isinstance(b, BaseException):
                raise b
            yield b
    finally:
        stop.set()                      # a consumer that stops early (break, an error) lets the producer go
        th.join(timeout=5)


def cmd_train(a: argparse.Namespace) -> int:
    import torch
    import ppo_core
    from human.imitate import smoothed_ce_rows

    t0 = time.time()
    work = os.path.abspath(os.path.expanduser(a.work))
    os.makedirs(work, exist_ok=True)
    out_path = os.path.abspath(os.path.expanduser(a.out))
    if os.path.exists(out_path):
        raise Refused(f"--out {out_path} exists (nothing is overwritten)")
    sched = Schedule(lr=a.lr, warmup=a.warmup, anneal_steps=a.anneal_steps, min_hold=a.min_hold, plateau_gain=a.plateau_gain,
                     plateau_window=a.plateau_window, max_steps=a.max_steps)
    abort = parse_abort(a.abort_at)
    torch.set_num_threads(max(1, int(a.torch_threads)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not a.allow_cpu:
        raise Refused("no GPU (the attention has no backward pass on a CPU)")
    jlog = JsonLog(os.path.join(work, "log.jsonl"))

    cfg = run_config()
    lc, rc = cfg.learner, cfg.reward
    shaping = str(rc.shaping) == "potential"
    compile_flex()
    net = build_net(cfg, a.seed, dev, set_head=True)
    data = TopData(a.data, net.unpacker.offsets, a.seed, cap=a.cap, holdout_decisions=a.holdout_decisions,
                   gamma=float(lc.gamma), scale=float(rc.shaping_scale), shaping=shaping, threads=a.threads,
                   holdout_order=a.holdout_order)
    counts = data.counts()
    log("data", json.dumps(counts))
    groups0 = pass_groups(data.files, a.seed, 0, a.group_shards)
    est = data.steps_per_group_estimate(a.seed, 0, groups0)
    steps_per_pass = int(sum(est))
    if a.max_passes:
        sched = dataclasses.replace(sched, max_steps=int(math.ceil(a.max_passes * steps_per_pass)))
    settings = _settings(a, sched)
    settings["holdout_digest"] = counts["holdout"]["digest"]
    log(f"{len(data.files)} train shards in {len(groups0)} groups of at most {a.group_shards}; about {steps_per_pass:,} steps a pass; "
        f"schedule {dataclasses.asdict(sched)} (anneal {sched.anneal_len():,} steps, at step {sched.latest_start():,} at the latest); "
        f"abort {abort}")

    init_info = None
    if a.init:
        init_info = load_init(net, a.init)
        log(f"weights from {a.init} ({init_info['mode']} update {init_info['update_idx']}); SetHead from it: "
            f"{init_info['set_head_from_init']}")
    opt = zone_groups(net, sched.lr, a.wd)

    # -- the state, fresh or from the newest resume file
    st: Dict[str, Any] = {"step": 0, "pass": 0, "group": 0, "anneal_start": None, "history": [], "best": None,
                          "rows_seen": 0, "labelled_seen": 0, "skipped": 0, "abort_checked": False, "segment": 0,
                          "last_resume": 0, "steps_by_pass": {}}
    resume = newest_resume(work)
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        diff = _same_run(ck["settings"], settings)
        if diff:
            raise Refused(f"{resume} was written with other settings ({diff}); use another --work")
        net.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        st.update(ck["state"])
        st["segment"] = int(st.get("segment", 0)) + 1
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        log(f"resumed from {os.path.basename(resume)}: step {st['step']:,}, pass {st['pass']}, group {st['group']}")
        jlog.write({"event": "resume", "file": os.path.basename(resume), "step": st["step"], "segment": st["segment"]})
    else:
        jlog.write({"event": "start", "settings": settings, "data": counts, "steps_per_pass": steps_per_pass, "init": init_info})
    trace = open(a.trace, "a", encoding="utf-8") if a.trace else None

    v_min, v_max, v_bins, vf = float(lc.v_min), float(lc.v_max), int(lc.v_bins), float(lc.vf_coef)
    groups_this_run = 0
    nonfinite_run = 0
    stop_reason = None
    acc = None

    def new_acc():
        z = torch.zeros((), device=dev)
        return {"pi": z.clone(), "v": z.clone(), "set": z.clone(), "hit": z.clone(), "n": z.clone(), "k": 0}

    win = {"t": time.time(), "k": 0, "wait": 0.0}          # the log line's window: steps a second and the wait for data

    def measure_now(done: int) -> Dict[str, Any]:
        ta = evaluate(net, data.holdout_seqs, dev, by_set=False, by_period=False)
        r = ta.results()
        m = r["all"]
        rec = {"step": done, "pass": st["pass"], "seconds": round(time.time() - t0, 1), "lr_share": lr_share(max(0, done - 1), sched, st["anneal_start"]),
               "top1": m["top1"], "top1_per_player": m["top1_per_player"], "ce": m["ce"], "vloss": m["vloss"],
               "labelled": m["labelled"], "players": m["players"], "sethead": r.get("sethead"), "segment": st["segment"]}
        if acc is not None and acc["k"]:
            rec["train"] = {"top1": float(acc["hit"]) / max(1.0, float(acc["n"])), "pi": float(acc["pi"]) / acc["k"],
                            "v": float(acc["v"]) / acc["k"], "set": float(acc["set"]) / acc["k"], "steps": acc["k"]}
        return rec

    def write_resume(done: int) -> None:
        payload = {"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "optimizer": opt.state_dict(),
                   "state": {k: v for k, v in st.items()}, "settings": settings, "torch_rng": torch.get_rng_state(),
                   "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
        path = save_new(payload, os.path.join(work, f"resume_{done:06d}.pt"))
        st["last_resume"] = done
        jlog.write({"event": "resume_file", "file": os.path.basename(path), "step": done})
        log(f"wrote {os.path.basename(path)}")

    def save_best(done: int, rec: Dict[str, Any]) -> None:
        path = save_new({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "step": done,
                         "rows_seen": st["rows_seen"], "measure": rec}, os.path.join(work, f"best_{done:06d}.pt"))
        st["best"] = dict(rec, file=os.path.basename(path), rows_seen=st["rows_seen"])
        jlog.write({"event": "best", "file": os.path.basename(path), "step": done, "top1": rec["top1"], "ce": rec["ce"]})

    def after_measure(done: int, rec: Dict[str, Any]) -> Optional[str]:
        st["history"].append(rec)
        jlog.write(dict(rec, event="measure"))
        flag = ""
        if done >= a.best_after and better(rec, st["best"]):
            save_best(done, rec)
            flag = "  <- best"
        log(f"held-out (capped, {rec['labelled']:,} decisions of {rec['players']} players) after {done:,} steps: top1 {rec['top1']:.4f} "
            f"per player {rec['top1_per_player']:.4f} ce {rec['ce']:.4f} vloss {rec['vloss']:.4f}"
            + (f" set {rec['sethead']['loss']:.4f}" if rec.get("sethead") else "") + flag)
        if abort_due(done, rec["top1"], abort, st["abort_checked"]):
            return f"abort: capped top-1 {rec['top1']:.4f} below {abort[1]} at step {done:,} (the plateau did not break)"
        if abort is not None and done >= abort[0]:
            st["abort_checked"] = True
        hist = [(r["step"], r["top1"]) for r in st["history"]]
        was = st["anneal_start"]
        st["anneal_start"] = anneal_start_after_measure(done, st["anneal_start"], hist, sched)
        if was is None and st["anneal_start"] is not None:
            log(f"plateau: the anneal starts at step {st['anneal_start']:,} and ends at {stop_step(st['anneal_start'], sched):,}")
            jlog.write({"event": "anneal", "start": st["anneal_start"], "why": "plateau"})
        return None

    loader = ThreadPoolExecutor(1, thread_name_prefix="topstart-group")

    def schedule_from(p: int, g: int) -> Iterator[Tuple[int, int, List[str]]]:
        while True:
            gs = pass_groups(data.files, a.seed, p, a.group_shards)
            for k in range(g, len(gs)):
                yield p, k, gs[k]
            p, g = p + 1, 0

    plan = schedule_from(int(st["pass"]), int(st["group"]))
    nxt = next(plan)
    fut = loader.submit(data.load_group, nxt[2])
    acc = new_acc()
    done = int(st["step"])
    log_every = max(1, int(a.log_every))
    while stop_reason is None:
        p, g, files = nxt
        tw = time.time()
        shards = fut.result()
        win["wait"] += time.time() - tw
        nxt = next(plan)
        fut = loader.submit(data.load_group, nxt[2])
        seqs = hi.pack(hi.pieces_of(shards, L_SEQ), L_SEQ, random.Random(group_seed(a.seed, p, g)))
        st["pass"], st["group"] = p, g
        for b in _batches(seqs, data.file_no, threading.Event()):
            was = st["anneal_start"]
            st["anneal_start"] = anneal_start_before_step(done, st["anneal_start"], sched)
            if was is None and st["anneal_start"] is not None:
                log(f"the anneal starts at step {done:,} (the latest start) and ends at {stop_step(done, sched):,}")
                jlog.write({"event": "anneal", "start": done, "why": "latest start"})
            end = stop_step(st["anneal_start"], sched)
            if end is not None and done >= end:
                stop_reason = f"the anneal ended at step {done:,}"
                break
            share = lr_share(done, sched, st["anneal_start"])
            for pg in opt.param_groups:
                pg["lr"] = sched.lr * share
            if trace is not None:
                trace.write(f"{done} {b['digest']}\n")
            obs = torch.from_numpy(b["obs"]).to(dev, non_blocking=True)
            ep = torch.from_numpy(b["ep"]).to(dev, non_blocking=True)
            act = torch.from_numpy(b["act"]).to(dev, non_blocking=True)
            mask = torch.from_numpy(b["mask"]).to(dev, non_blocking=True)
            w = torch.from_numpy(b["w"]).to(dev, non_blocking=True)
            ret = torch.from_numpy(b["ret"]).to(dev, non_blocking=True)
            real_idx = torch.from_numpy(np.flatnonzero(b["real"])).to(dev, non_blocking=True)
            lab_set = torch.from_numpy(b["set_labels"]).to(dev, non_blocking=True)
            wsum = float(b["w"].sum())
            out = net(obs, ep, batch_seq_len=L_SEQ, return_z=True)
            logits, v_logits, z_tok = out[0].float(), out[1].float(), out[5]
            sce, _ = smoothed_ce_rows(logits, act, mask, SMOOTHING)
            pi_loss = (w * sce).sum() / max(wsum, 1e-6)
            t = ppo_core.twohot_targets(ret[real_idx], v_min=v_min, v_max=v_max, v_bins=v_bins)
            v_loss = ppo_core.dist_value_loss(v_logits[real_idx], t)
            s_term, s_st = ppo_core.compute_set_head_loss(net, z_tok, lab_set.reshape(lab_set.shape[0], 6, -1), coef=SET_HEAD_COEF)
            loss = pi_loss + vf * v_loss + s_term
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.accumulate_grads()
            masters = [m for _p, m in opt.pairs if m.grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(masters, a.max_grad_norm)
            ok = bool(torch.isfinite(loss).item() and torch.isfinite(norm).item())
            if ok:
                opt.inner.step()
                with torch.no_grad():
                    for model_p, master_p in opt.pairs:
                        model_p.data.copy_(master_p.data)
                nonfinite_run = 0
                with torch.no_grad():
                    lab_t = w > 0
                    best_a = logits.detach().masked_fill(mask <= 0.5, -1e4).argmax(-1)
                    acc["hit"] += ((best_a == act) & lab_t).sum()
                    acc["n"] += lab_t.sum()
                    acc["pi"] += pi_loss.detach()
                    acc["v"] += v_loss.detach()
                    acc["set"] += s_st["set_loss"]
                    acc["k"] += 1
            else:
                st["skipped"] += 1
                nonfinite_run += 1
                jlog.write({"event": "nonfinite", "step": done, "in_a_row": nonfinite_run})
                log(f"step {done:,}: the loss or the gradient is not finite; the step is skipped ({nonfinite_run} in a row)")
            opt.zero_grad(set_to_none=True)
            done += 1
            st["step"] = done
            st["rows_seen"] += int(b["real"].sum())
            st["labelled_seen"] += int((b["w"] > 0).sum())
            st["steps_by_pass"][str(p)] = int(st["steps_by_pass"].get(str(p), 0)) + 1
            win["k"] += 1
            if nonfinite_run >= 3:
                stop_reason = "nonfinite"
                break
            if done % log_every == 0 and acc["k"]:
                dt = time.time() - win["t"]
                log(f"step {done:,} pass {p} group {g} lr {sched.lr * share:.2e} pi {float(acc['pi']) / acc['k']:.4f} "
                    f"v {float(acc['v']) / acc['k']:.4f} set {float(acc['set']) / acc['k']:.4f} train top1 "
                    f"{float(acc['hit']) / max(1.0, float(acc['n'])):.4f} {win['k'] / max(dt, 1e-9):.2f} steps/s "
                    f"(waited {win['wait']:.1f} s for data)")
                win.update(t=time.time(), k=0, wait=0.0)
            if done % a.measure_every == 0:
                rec = measure_now(done)
                acc = new_acc()
                stop_reason = after_measure(done, rec)
                if stop_reason:
                    break
        if trace is not None:
            trace.flush()
        if stop_reason is not None:
            break
        # the group is done: the next one starts at a boundary a resume can reproduce
        st["pass"], st["group"] = nxt[0], nxt[1]
        groups_this_run += 1
        stop_file = os.path.join(work, "STOP")
        stopping = (a.stop_after_groups and groups_this_run >= a.stop_after_groups) or os.path.exists(stop_file)
        if stopping or done - int(st["last_resume"]) >= a.resume_every:
            write_resume(done)
        if stopping:
            loader.shutdown(wait=True, cancel_futures=True)
            log(f"stopped at a group boundary after {done:,} steps ({'--stop-after-groups' if a.stop_after_groups else 'STOP file'}); "
                f"run the same command again to resume")
            jlog.write({"event": "stopped", "step": done})
            if trace is not None:
                trace.close()
            return 0
    loader.shutdown(wait=False, cancel_futures=True)
    if trace is not None:
        trace.close()

    # -- the end
    info: Dict[str, Any] = {"code": "human/topstart.py", "code_sha256": sha256_of(os.path.abspath(__file__)),
                            "settings": settings, "data": data.counts(), "steps_per_pass_estimate": steps_per_pass,
                            "steps_by_pass": st["steps_by_pass"], "init": init_info, "anneal_start": st["anneal_start"],
                            "stop_reason": stop_reason, "steps": done, "rows_seen": st["rows_seen"],
                            "labelled_seen": st["labelled_seen"], "skipped_nonfinite": st["skipped"],
                            "seconds": round(time.time() - t0, 1)}
    if stop_reason == "nonfinite":
        info["history"] = st["history"]
        save_new(info, os.path.join(work, "info.json"), kind="json")
        log("three steps in a row were not finite: stopped, no start written")
        return EXIT_NONFINITE
    if stop_reason.startswith("abort"):
        info["history"] = st["history"]
        save_new(st["history"], os.path.join(work, "history.json"), kind="json")
        save_new(info, os.path.join(work, "info.json"), kind="json")
        log(f"{stop_reason}: no start written")
        jlog.write({"event": "abort", "step": done, "why": stop_reason})
        return EXIT_ABORT
    if not st["history"] or st["history"][-1]["step"] != done:
        rec = measure_now(done)
        after_measure(done, rec)
    if st["best"] is None:
        save_best(done, st["history"][-1])
    best = st["best"]
    bk = torch.load(os.path.join(work, best["file"]), map_location="cpu", weights_only=False)
    info["history"] = st["history"]
    info["best"] = best
    save_new(st["history"], os.path.join(work, "history.json"), kind="json")
    info_path = save_new(info, os.path.join(work, "info.json"), kind="json")
    payload = {"model": bk["model"], "target_model": None, "update_idx": 0, "total_episodes": 0,
               "total_steps": int(best["rows_seen"]), "run_cfg": cfg.as_dict(), "topstart": info}
    if os.path.exists(out_path):
        raise Refused(f"--out {out_path} exists now (nothing is overwritten); the best is {best['file']} in {work}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = save_new(payload, out_path)
    digest = sha256_of(written)
    with open(unique_path(written + ".sha256"), "w", encoding="utf-8") as f:
        f.write(f"{digest}  {os.path.basename(written)}\n")
    with open(unique_path(written + ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(info, sha256=digest, file=os.path.basename(written)), f, indent=1, default=str)
    log(f"wrote {written} (sha256 {digest[:16]}...): best capped top1 {best['top1']:.4f} ce {best['ce']:.4f} at step "
        f"{best['step']:,}; {info_path}; {stop_reason}; {time.time() - t0:.0f} s")
    jlog.write({"event": "done", "out": written, "sha256": digest, "best_step": best["step"]})
    return 0


# =============================================================================================
# measure
# =============================================================================================

def cmd_measure(a: argparse.Namespace) -> int:
    import torch
    t0 = time.time()
    out_path = os.path.abspath(os.path.expanduser(a.out))
    if os.path.exists(out_path):
        raise Refused(f"--out {out_path} exists (nothing is overwritten)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not a.allow_cpu:
        raise Refused("no GPU")
    torch.set_num_threads(max(1, int(a.torch_threads)))
    compile_flex()
    net, o, ck = load_net(os.path.expanduser(a.checkpoint), dev)
    if o != OBS_ALL_ON:
        raise Refused(f"{a.checkpoint} has the observation {o}; the shards are {OBS_ALL_ON}" + widen_hint(o, a.checkpoint))
    cfg = run_config()
    data = TopData(a.data, net.unpacker.offsets, a.seed, cap=a.cap, holdout_decisions=a.holdout_decisions,
                   gamma=float(cfg.learner.gamma), scale=float(cfg.reward.shaping_scale), shaping=str(cfg.reward.shaping) == "potential",
                   threads=a.threads, holdout_order=a.holdout_order)
    res: Dict[str, Any] = {"checkpoint": os.path.abspath(a.checkpoint), "checkpoint_sha256": sha256_of(a.checkpoint) if a.sha else None,
                           "mode": (ck.get("run_cfg") or {}).get("learner", {}).get("mode"), "update_idx": int(ck.get("update_idx") or 0),
                           "total_steps": int(ck.get("total_steps") or 0), "has_sethead": getattr(net, "set_head", None) is not None,
                           "data": data.counts()}
    log(f"capped held-out set: {len(data.holdout_seqs)} sequences")
    res["capped"] = evaluate(net, data.holdout_seqs, dev).results()
    log("capped:", json.dumps(res["capped"].get("all")))
    if not a.capped_only:
        files = hi.shard_files(data.data)
        tallies = {r: Tally() for r in ROLES_MEASURED}
        G = max(1, int(a.group_shards))
        for k in range(0, len(files), G):
            part = files[k:k + G]
            jobs = [(f, r, 1 + 2 * j + (1 if r == "test_game" else 0)) for j, f in enumerate(part) for r in ROLES_MEASURED]
            loaded = data.load_many(jobs)
            for role in ROLES_MEASURED:
                shards = [s for s in loaded if (s.uid % 2 == 0) == (role == "test_game")]
                if shards:
                    seqs = hi.pack(hi.pieces_of(shards, L_SEQ), L_SEQ, random.Random(group_seed(a.seed, 10_000, k)))
                    evaluate(net, seqs, dev, tallies[role])
            log(f"measured {min(len(files), k + G)} of {len(files)} shards")
            del loaded
        for role, ta in tallies.items():
            res[role] = ta.results()
    res["dropped_episodes"] = data.dropped
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = save_new(res, out_path, kind="json")
    log(f"wrote {written}")
    return 0


# =============================================================================================
# link and index
# =============================================================================================

def link_plan(manifest: Dict[str, Any], src: Dict[str, str], dst: str) -> List[Tuple[str, str]]:
    """(target, link) for every read shard of every set: link = dst/shard-<set>-<name without "shard-">. Skip shards never."""
    out = []
    for s in manifest["sets"]:
        name = s["name"]
        if name not in src:
            raise Refused(f"no --set {name}=DIR for the manifest's set {name!r}")
        skip = set(s.get("skip_shards") or [])
        for n in s["read_shards"]:
            if n in skip:
                raise Refused(f"{n} is in both read_shards and skip_shards of set {name}")
            if not n.startswith("shard-"):
                raise Refused(f"unexpected shard name {n!r}")
            out.append((os.path.join(os.path.abspath(os.path.expanduser(src[name])), n), os.path.join(dst, f"shard-{name}-{n[6:]}")))
    return out


def cmd_link(a: argparse.Namespace) -> int:
    with open(os.path.expanduser(a.manifest), encoding="utf-8") as f:
        manifest = json.load(f)
    src = {}
    for spec in a.set:
        k, v = spec.split("=", 1)
        src[k] = v
    dst = os.path.abspath(os.path.expanduser(a.dst))
    os.makedirs(dst, exist_ok=True)
    plan = link_plan(manifest, src, dst)
    missing = [t for t, _l in plan if not os.path.isfile(t)]
    if missing and not a.allow_missing:
        raise Refused(f"{len(missing)} read shard(s) are not downloaded (the first: {missing[0]})")
    made = same = 0
    for target, link in plan:
        if not os.path.isfile(target):
            continue
        if os.path.lexists(link):
            if os.path.realpath(link) == os.path.realpath(target):
                same += 1
                continue
            raise Refused(f"{link} exists and is not a link to {target} (nothing is replaced)")
        os.symlink(target, link)
        made += 1
    skipped = {s["name"]: len(s.get("skip_shards") or []) for s in manifest["sets"]}
    log(f"linked {made} new, {same} already there, {len(missing)} missing; read shards per set "
        f"{ {s['name']: len(s['read_shards']) for s in manifest['sets']} }; skip shards never linked {skipped}")
    return 0


def cmd_index(a: argparse.Namespace) -> int:
    cfg = run_config()
    offsets = build_net(cfg, a.seed, "cpu", set_head=False).unpacker.offsets      # the network's own (no GPU needed)
    data = TopData(a.data, offsets, a.seed, cap=a.cap, holdout_decisions=a.holdout_decisions, gamma=float(cfg.learner.gamma),
                   scale=float(cfg.reward.shaping_scale), shaping=True, threads=a.threads, build_holdout=not a.no_holdout,
                   holdout_order=a.holdout_order)
    c = data.counts()
    gs = pass_groups(data.files, a.seed, 0, a.group_shards)
    c["steps_per_pass_estimate"] = int(sum(data.steps_per_group_estimate(a.seed, 0, gs)))
    c["groups_per_pass"] = len(gs)
    code = 0
    if a.manifest:
        with open(os.path.expanduser(a.manifest), encoding="utf-8") as f:
            m = json.load(f)
        want = {"train_sides": m["totals"]["train"]["sides"], "train_labelled": m["totals"]["train"]["steps_train"],
                "test_player_sides": m["totals"]["test_player"]["sides"], "test_game_sides": m["totals"]["test_game"]["sides"]}
        got = {k: c[k] for k in want}
        c["manifest_check"] = {"want": want, "got": got, "pass": want == got}
        if want != got:
            code = EXIT_REFUSED
    print(json.dumps(c, indent=1, default=str), flush=True)
    if a.out:
        save_new(c, os.path.expanduser(a.out), kind="json")
    return code


# =============================================================================================

def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", required=True, help="the linked directory (link): shard-40k-*.npz and shard-bulk-*.npz")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cap", type=float, default=0.03, help="the per-player cap over both sets")
    p.add_argument("--holdout-decisions", type=int, default=32_768)
    p.add_argument("--holdout-order", choices=HOLDOUT_ORDERS, default="shuffled",
                   help="the held-out shards' order: shuffled (a fixed permutation, both sets and every period), or spread "
                        "(the name order of work directories written before this option: 40k first)")
    p.add_argument("--threads", type=int, default=6, help="shard loader threads (each at nice 10)")
    p.add_argument("--group-shards", type=int, default=96)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("link", help="link each set's read_shards into one directory (the manifest's recipe)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--set", action="append", default=[], help="NAME=DIR, once per set of the manifest")
    p.add_argument("--dst", required=True)
    p.add_argument("--allow-missing", action="store_true", help="link what is there (tests)")

    p = sub.add_parser("index", help="what the linked data holds (counts only), optionally against the manifest's totals")
    _common(p)
    p.add_argument("--manifest", default="")
    p.add_argument("--out", default="")
    p.add_argument("--no-holdout", action="store_true")

    p = sub.add_parser("train", help="offline imitation of the top players")
    _common(p)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True, help="the start checkpoint (refused if it exists)")
    p.add_argument("--init", default="", help="start from this checkpoint's weights (the control C0); default random weights")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--anneal-steps", type=int, default=25_000)
    p.add_argument("--min-hold", type=int, default=60_000)
    p.add_argument("--plateau-gain", type=float, default=0.005)
    p.add_argument("--plateau-window", type=int, default=20_000)
    p.add_argument("--max-steps", type=int, default=150_000)
    p.add_argument("--max-passes", type=float, default=0.0, help="max steps as passes over the train shards (C0: 3)")
    p.add_argument("--abort-at", default="40000:0.30", help="STEP:BAR, or off")
    p.add_argument("--measure-every", type=int, default=2000)
    p.add_argument("--resume-every", type=int, default=20_000)
    p.add_argument("--best-after", type=int, default=40_000, help="best_<step>.pt from this step on")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--trace", default="", help="append every step's batch id to this file (tests)")
    p.add_argument("--stop-after-groups", type=int, default=0, help="stop at a group boundary after this many groups (tests, operators)")

    p = sub.add_parser("measure", help="agreement, value loss and SetHead accuracies of a checkpoint on the held-out data")
    _common(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--capped-only", action="store_true")
    p.add_argument("--sha", action="store_true", help="add the checkpoint's sha256")

    a = ap.parse_args(argv)
    if a.cmd == "link":
        return cmd_link(a)
    if a.cmd == "index":
        return cmd_index(a)
    if a.cmd == "train":
        for name in ("measure_every", "resume_every", "group_shards", "threads"):
            if int(getattr(a, name)) < 1:
                raise Refused(f"--{name.replace('_', '-')} must be at least 1")
        if a.max_steps < 1 or a.warmup < 1 or a.anneal_steps < 1:
            raise Refused("--max-steps, --warmup and --anneal-steps must be at least 1")
        parse_abort(a.abort_at)
        return cmd_train(a)
    return cmd_measure(a)


if __name__ == "__main__":
    raise SystemExit(main())
