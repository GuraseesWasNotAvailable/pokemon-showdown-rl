"""
The top players' own decisions as an auxiliary loss of the live PPO learner.

    python train.py --mode ppo ... --human-imitation-coef 0.01 --human-imitation-data ~/dev/human_shards

Without --human-imitation-coef (or with 0) train.py never imports this module, the learner gets no
`human_imitation` argument and ppo_update gets human=None: every update is what it was, bit for bit
(tests/test_human_imitation.py holds that against the commit before this one).

Ported from branch stage3 (human_imitation.py there, bench/STAGE2.md section 2) to this code line
(train-v2: the from-scratch network with the matchup and moveset blocks, the SetHead, the
OppActionHead, several inference actors). What changed in the port:

  * the observation check is this line's: a shard is loaded only when it was built with the run's
    own_tera_type, matchup and movesets (shard fields own_tera_type, obs_matchup, obs_movesets; absent
    = off), with no obs_abilities (this line has none) and, when it says, the same obs_dim as the
    network's input. Anything else is never loaded;
  * the split is read from the shard itself when it carries one (branch v2-human's human/rebuild.py
    writes ep_split per side: "train", "test_game", "test_player", "loose"); the collector's split
    files (--human-imitation-splits / -salt) are read only for shards without it (stage3's layout);
  * the term's figures are summed on the GPU and read once per update (the update loop here waits
    for the GPU as little as it can);
  * nothing that is not finite reaches the term: a shard episode with a non-finite observation or
    mask value is dropped when the shard loads (counted, hum_drop), and settings that are not finite
    are refused. A term that is still not finite (only when the network itself already is) is
    counted (hum_nonfinite) and left IN the loss: zeroing its value would not stop its gradient
    (log_softmax's and the layers' backward multiply the zero by the NaN), it would only hide the
    NaN from the update's loss, which the supervisor's NaN guard reads;
  * optionally (--human-imitation-set-coef, 0 = off) the SetHead also learns from the shards' exact
    set labels (set_labels: the opponent's real team from the re-simulation, the layout of
    obs_matchup.build_set_labels). Only with --set-head-coef on, so the head already exists and the
    optimizer's layout never changes;
  * the loader thread runs at a lower CPU priority (nice +10, Linux) than the learner.

The data are human/rebuild.py's shards (branch v2-human): one episode is one side of one top-player
game (pre-game Elo >= 2100), every decision in order, the observation the live rollout worker
builds, the legal mask and the action the player took on our 14 actions. Train episodes are the
"train" sides; the held-out players' own sides ("test_player") measure. Nothing else is used: the
"loose" sides (the other side of a held-out player's game) and "test_game" are never trained on.

The term, in each PPO minibatch (every `every`-th one), on `seqs` packed sequences of
batch_seq_len decisions (whole episodes, as the learner packs its own; padding rows have no weight):

    coef * sum_rows w * CE(pi(.|obs), a_player) / sum_rows w

    w = player cap  x  switch_weight (the player switched: actions 4-9)
                    x  double_switch_weight (the player switched on a turn the opponent also switched)

- player cap: a player holding more than `cap` of the labelled train decisions is weighted down to
  `cap` (the human job's finding: the top 10 players hold most of the data).
- "double switch" here is the turn both sides switched by choice: the player's action is a switch
  while a move was legal (not a forced switch), and the next decision's transition block says the
  opponent switched, did not move, did not pivot and did not faint. It is read off the next
  observation, so it is a hindsight label, which imitation may use.
- shuffled (the control arm beside any run with this term): labels are permuted among the labelled
  decisions of a shard that have the same legal mask, then the same weighting rule is applied to the
  permuted labels. Whatever a run with real labels gains over its shuffled arm is what the top
  players' choices teach; the rest is the moving itself.

Rows: only labelled decisions with more than one legal action carry weight. The others are
history (the transformer sees them, nothing is learned on them). The value head never learns from
the human games.

Telemetry (the [learner] update line, only when the term is configured, at its end): hum_n
labelled human decisions trained on in the update's first epoch, hum_ce their weighted CE (the
term's loss before coef), hum_agree how often the policy's likeliest legal action was the player's
on those rows (the train accuracy), hum_sw / hum_dsw the switch and double-switch rows among them,
hum_ho / hum_ho_pp / hum_ho_ce the top-1 agreement with the HELD-OUT players (pooled, averaged per
player) and their CE, measured every holdout_every updates on a fixed set in which no player holds
more than `cap` of `holdout_decisions` (so hum_ho is not one prolific account's agreement),
hum_ho_upd the update it was measured at, hum_total the labelled human decisions used so far,
hum_pool the decisions in memory, hum_wait minibatches that had no human batch ready, hum_err
minibatches whose term raised (left out: that minibatch's loss is what it was without it),
hum_nonfinite minibatches whose term was not finite (in the loss, so the update's loss is not a
number either and the supervisor acts on that update), hum_drop episodes dropped at load for a
value that is not finite; with the SetHead part hum_set_loss and hum_set_n. Before the data are
loaded the line ends " hum=starting", and " hum=off" when the term could not be had (the reason is
printed once).

Memory: a pool of about `pool_decisions` decisions is held (21 kB each with this line's
observation: 120,000 is 2.5 GB); before each update the loader thread adds random other shards
worth an eighth of the pool and drops the oldest, so the whole train set passes through.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import logging
import math
import os
import random
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

ACT = 14
SWITCH_LO, SWITCH_HI = 4, 9
_MOVE_ACTIONS = np.array([a < SWITCH_LO or a > SWITCH_HI for a in range(ACT)])
PAD_EPISODE = 10 ** 12                    # padding rows' episode ids start here (one each)

# obs_transitions.py: per-side scalar block, side 0 = the player, side 1 = the opponent.
try:
    from obs_transitions import IDX_DID_MOVE, IDX_DID_SWITCH, IDX_FAINTED, IDX_PIVOT_SWITCH, SIDE_SCALAR_DIM
except Exception:                                    # the constants, should the module not import here
    IDX_DID_MOVE, IDX_DID_SWITCH, IDX_FAINTED, IDX_PIVOT_SWITCH, SIDE_SCALAR_DIM = 1, 2, 11, 21, 45

# The roles a shard's ep_split names (human/rebuild.py episode_split): only these two are used.
SPLIT_ROLES = {"train": "train", "test_player": "test_player"}


@dataclasses.dataclass
class HumanImitationSettings:
    data: str
    coef: float
    splits: str = "~/dev/human-data/splits"
    salt: str = "~/dev/human-rebuild/salt"
    seqs: int = 1
    every: int = 1
    switch_weight: float = 1.0
    double_switch_weight: float = 1.0
    cap: float = 0.03
    shuffled: bool = False
    pool_decisions: int = 120_000
    holdout_decisions: int = 8192
    holdout_every: int = 50
    seed: int = 0
    set_coef: float = 0.0

    def check(self) -> None:
        # inf passes the comparisons below and would make the term's gradient NaN (inf x 0)
        for name in ("coef", "switch_weight", "double_switch_weight", "cap", "set_coef"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} {getattr(self, name)} must be a finite number")
        if not (self.coef > 0):
            raise ValueError(f"coef {self.coef} must be above zero")
        if self.seqs < 1 or self.every < 1 or self.pool_decisions < 1000:
            raise ValueError("seqs and every must be at least 1, pool_decisions at least 1000")
        if self.switch_weight <= 0 or self.double_switch_weight <= 0:
            raise ValueError("the switch weights must be above zero")
        if not (0 < self.cap <= 1):
            raise ValueError(f"cap {self.cap} must lie in (0, 1]")
        if self.holdout_decisions < 0 or self.holdout_every < 1:
            raise ValueError("holdout_decisions must not be negative and holdout_every must be at least 1")
        if self.set_coef < 0:
            raise ValueError(f"set_coef {self.set_coef} must not be negative")

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------------------------
# the observation a shard was built with, against the run's
# ---------------------------------------------------------------------------------------------

def run_obs_settings(obs: Any) -> Tuple[bool, bool, bool, str]:
    """(own_tera_type, matchup, movesets, obs_abilities) of the run: a config.ObsConfig or a dict of its fields."""
    if obs is None:
        get = {}.get
    elif isinstance(obs, dict):
        get = obs.get
    else:
        def get(k, d=None):
            return getattr(obs, k, d)
    return bool(get("own_tera_type", False) is True), bool(get("matchup", False) is True), \
        bool(get("movesets", False) is True), ""


def shard_obs_settings(z: Any) -> Tuple[bool, bool, bool, str]:
    """(own_tera_type, matchup, movesets, obs_abilities) a shard records; an absent field is off."""
    def flag(name: str) -> bool:
        return bool(z[name]) if name in z.files else False
    abil = str(z["obs_abilities"]) if "obs_abilities" in z.files else ""
    return flag("own_tera_type"), flag("obs_matchup"), flag("obs_movesets"), abil


def shard_obs_dim(z: Any) -> Optional[int]:
    """The observation width a shard records (obs_dim), or None when it does not say."""
    return int(z["obs_dim"]) if "obs_dim" in z.files else None


def describe_obs(s: Tuple[bool, bool, bool, str]) -> str:
    return f"own_tera_type={s[0]} matchup={s[1]} movesets={s[2]} obs_abilities={s[3] or 'off'}"


def shard_files(data: str) -> List[str]:
    """The shard files of a data directory (a build's unfinished .tmp.npz files left out)."""
    return sorted(f for f in glob.glob(os.path.join(os.path.expanduser(data), "shard-*.npz")) if not f.endswith(".tmp.npz"))


def shard_fits(z: Any, want: Tuple[bool, bool, bool, str], dim: Optional[int]) -> bool:
    if shard_obs_settings(z) != want:
        return False
    sd = shard_obs_dim(z)
    return dim is None or sd is None or sd == dim


# ---------------------------------------------------------------------------------------------
# splits: the shard's own ep_split, or the collector's files (stage3's layout)
# ---------------------------------------------------------------------------------------------

def salted_game(salt: str, rid: str) -> str:
    """human/rebuild.py salted("g", rid)."""
    return "g" + hashlib.sha256((salt + "|" + rid.lower()).encode()).hexdigest()[:12]


def read_splits(splits_dir: str, salt_path: str) -> Dict[str, Tuple[str, str]]:
    """salted game hash -> (split name, the sides at the bar: "p1p2", "p1" or "p2")."""
    with open(os.path.expanduser(salt_path)) as f:
        salt = f.read().strip()
    out: Dict[str, Tuple[str, str]] = {}
    for name in ("train", "test_game", "test_player"):
        path = os.path.join(os.path.expanduser(splits_dir), name + ".tsv")
        with open(path) as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    out[salted_game(salt, parts[0])] = (name, parts[1])
    return out


def episode_role(ep_split: Optional[np.ndarray], e: int, side: str, game: str,
                 split_of: Optional[Dict[str, Tuple[str, str]]]) -> str:
    """'train', 'test_player', 'unused' or 'no_split' for episode e of a shard (ep_split: the shard's own, or None)."""
    if ep_split is not None:
        return SPLIT_ROLES.get(str(ep_split[e]), "unused")
    sp = (split_of or {}).get(game)
    # the sides column is "p1p2", "p1" or "p2" (human/collect.py split)
    if sp is None:
        return "no_split"
    return sp[0] if side in sp[1] and sp[0] in SPLIT_ROLES else "unused"


# ---------------------------------------------------------------------------------------------
# labels and weights of one shard's rows
# ---------------------------------------------------------------------------------------------

def double_switch_rows(obs: np.ndarray, act: np.ndarray, mask: np.ndarray, ep_start: np.ndarray,
                       trans0: int) -> np.ndarray:
    """
    Rows where the player switched by choice (a switch while a move was legal) and the next
    decision of the same episode shows the opponent switched without moving, pivoting or fainting.
    trans0: where the transition scalar block starts in the observation.
    """
    n = len(act)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out
    opp = trans0 + SIDE_SCALAR_DIM
    nxt_sw = obs[:, opp + IDX_DID_SWITCH] > 0.5
    nxt_quiet = ~((obs[:, opp + IDX_DID_MOVE] > 0.5) | (obs[:, opp + IDX_FAINTED] > 0.5) | (obs[:, opp + IDX_PIVOT_SWITCH] > 0.5))
    is_switch = (act >= SWITCH_LO) & (act <= SWITCH_HI)
    could_move = (mask[:, _MOVE_ACTIONS] > 0.5).any(axis=1)
    same_ep_next = np.ones(n, dtype=bool)
    same_ep_next[-1] = False
    ends = np.asarray(ep_start[1:], dtype=np.int64) - 1          # each episode's last row has no next row
    same_ep_next[ends[(ends >= 0) & (ends < n)]] = False
    nxt = np.zeros(n, dtype=bool)
    nxt[:-1] = nxt_sw[1:] & nxt_quiet[1:]
    return is_switch & could_move & same_ep_next & nxt


def shuffle_labels(act: np.ndarray, mask: np.ndarray, lab: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The labels of the labelled rows permuted among rows with the same legal mask (the control arm)."""
    out = act.copy()
    rows = np.flatnonzero(lab)
    if len(rows) == 0:
        return out
    keys = np.packbits(mask[rows] > 0.5, axis=1)
    keys = keys[:, 0].astype(np.int64) * 256 + (keys[:, 1].astype(np.int64) if keys.shape[1] > 1 else 0)
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    for grp in np.split(order, cuts):
        if len(grp) > 1:
            r = rows[grp]
            out[r] = act[r][rng.permutation(len(r))]
    return out


def finite_rows(x: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Per row: every value is finite (an integer array always is). In chunks, so little memory."""
    if not np.issubdtype(x.dtype, np.floating):
        return np.ones(len(x), dtype=bool)
    out = np.empty(len(x), dtype=bool)
    for s in range(0, len(x), chunk):
        part = x[s:s + chunk]
        out[s:s + len(part)] = np.isfinite(part.reshape(len(part), -1)).all(axis=1)
    return out


def labelled_rows(z: Any) -> np.ndarray:
    """A shard's labelled rows: a training decision with an action, and that action legal (as Shard's own)."""
    act = z["act"].astype(np.int64)
    lab = z["train"].astype(bool) & (act >= 0)
    return lab & (z["mask"][np.arange(len(act)), np.clip(act, 0, ACT - 1)] > 0.5)


class Shard:
    """
    The rows of one shard's train (or held-out) episodes, with their weights. An episode with an
    observation or mask value that is not finite (inf or NaN) is dropped whole when the shard loads
    and listed in `dropped` (row a, row b, player): one such row, whatever its weight (a history
    row has none), makes the term's gradient NaN for every weight it reaches, and nothing after the
    term can undo that.
    """

    __slots__ = ("path", "uid", "obs", "act", "mask", "w", "lab", "dsw", "eps", "players", "set_labels", "dropped")

    def __init__(self, path: str, uid: int, z: Any, episodes: List[Tuple[int, int, str]], trans0: int,
                 dim: int, settings: Optional[HumanImitationSettings], player_w: Optional[Dict[str, float]],
                 rng: Optional[np.random.Generator], with_set_labels: bool = False):
        rows = np.concatenate([np.arange(a, b) for a, b, _p in episodes]) if episodes else np.zeros(0, np.int64)
        self.path, self.uid = path, uid
        self.obs = z["obs"][rows]
        if self.obs.ndim != 2 or int(self.obs.shape[1]) != int(dim):
            raise ValueError(f"observation width {self.obs.shape[1:]} is not the network's {dim}")
        mask_raw = z["mask"][rows]
        ok = finite_rows(self.obs) & finite_rows(mask_raw)
        self.dropped: List[Tuple[int, int, str]] = []
        if not ok.all():                              # on the CPU, at load: the learner never waits for it
            at = np.cumsum([0] + [b - a for a, b, _p in episodes]).astype(np.int64)
            keep_ep = [bool(ok[at[k]:at[k + 1]].all()) for k in range(len(episodes))]
            self.dropped = [e for e, good in zip(episodes, keep_ep) if not good]
            keep = np.zeros(len(rows), dtype=bool)
            for k, good in enumerate(keep_ep):
                keep[at[k]:at[k + 1]] = good
            episodes = [e for e, good in zip(episodes, keep_ep) if good]
            rows, self.obs, mask_raw = rows[keep], self.obs[keep], mask_raw[keep]
        act = z["act"][rows].astype(np.int64)
        self.mask = mask_raw.astype(np.float32)
        lab = z["train"][rows].astype(bool) & (act >= 0)
        lab &= self.mask[np.arange(len(act)), np.clip(act, 0, ACT - 1)] > 0.5
        starts = np.cumsum([0] + [b - a for a, b, _p in episodes]).astype(np.int64)
        self.eps = [(int(starts[k]), int(starts[k + 1])) for k in range(len(episodes))]
        self.players = [p for _a, _b, p in episodes]
        if settings is not None and settings.shuffled and rng is not None:
            act = shuffle_labels(act, self.mask, lab, rng)
        self.act, self.lab = act, lab
        self.dsw = double_switch_rows(self.obs, act, self.mask, starts, trans0) & lab
        w = np.zeros(len(act), dtype=np.float32)
        if settings is not None:
            for k, (a, b) in enumerate(self.eps):
                w[a:b] = (player_w or {}).get(self.players[k], 1.0)
            is_switch = (act >= SWITCH_LO) & (act <= SWITCH_HI)
            w = np.where(is_switch, w * float(settings.switch_weight), w)
            w = np.where(self.dsw, w * float(settings.double_switch_weight), w)
        self.w = np.where(lab, w, 0.0).astype(np.float32)
        self.set_labels = None
        if with_set_labels and "set_labels" in z.files:
            self.set_labels = np.ascontiguousarray(z["set_labels"][rows].astype(np.int32))

    def rows(self) -> int:
        return int(len(self.act))


# ---------------------------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------------------------

def pack(pieces: List[Tuple[Any, int, int, int]], L: int, rng: random.Random) -> List[List[Tuple[Any, int, int, int]]]:
    """
    Pieces (shard, row a, row b, episode uid), each at most L rows, into sequences of at most L rows,
    first fit in a shuffled order (an episode longer than L was cut into pieces of L).
    """
    order = list(pieces)
    rng.shuffle(order)
    seqs: List[List[Tuple[Any, int, int, int]]] = []
    fill = np.zeros(0, dtype=np.int64)
    for p in order:
        n = p[2] - p[1]
        fits = np.flatnonzero(fill + n <= L)
        if len(fits):
            j = int(fits[0])
            seqs[j].append(p)
            fill[j] += n
        else:
            seqs.append([p])
            fill = np.append(fill, n)
    rng.shuffle(seqs)
    return seqs


def pieces_of(shards: Sequence[Shard], L: int) -> List[Tuple[Any, int, int, int]]:
    out = []
    for sh in shards:
        for k, (a, b) in enumerate(sh.eps):
            uid = sh.uid * 100_000 + k
            for s in range(a, b, L):
                out.append((sh, s, min(b, s + L), uid))
    return out


def gather(seqs: Sequence[List[Tuple[Any, int, int, int]]], L: int, with_set_labels: bool = False) -> Dict[str, np.ndarray]:
    """
    The rows of some packed sequences: obs [S*L, D] (float16), ep [S*L] (padding rows: an episode of
    their own each, a copy of a real row's observation and mask), act, mask, w (0 on padding), dsw,
    and with_set_labels the SetHead's labels (0, no label, on padding and where a shard has none).
    """
    S = len(seqs)
    first = seqs[0][0]
    D = first[0].obs.shape[1]
    obs = np.empty((S * L, D), dtype=first[0].obs.dtype)
    ep = PAD_EPISODE + np.arange(S * L, dtype=np.int64)
    act = np.zeros(S * L, dtype=np.int64)
    mask = np.ones((S * L, ACT), dtype=np.float32)
    w = np.zeros(S * L, dtype=np.float32)
    dsw = np.zeros(S * L, dtype=bool)
    lab_w = None
    if with_set_labels:
        widths = [sh.set_labels.shape[1] for seq in seqs for sh, _a, _b, _u in seq if sh.set_labels is not None]
        if widths:
            lab_w = np.zeros((S * L, max(widths)), dtype=np.int32)
    for j, seq in enumerate(seqs):
        at = j * L
        for sh, a, b, uid in seq:
            n = b - a
            obs[at:at + n] = sh.obs[a:b]
            ep[at:at + n] = uid
            act[at:at + n] = sh.act[a:b]
            mask[at:at + n] = sh.mask[a:b]
            w[at:at + n] = sh.w[a:b]
            dsw[at:at + n] = sh.dsw[a:b]
            if lab_w is not None and sh.set_labels is not None:
                lab_w[at:at + n, :sh.set_labels.shape[1]] = sh.set_labels[a:b]
            at += n
        if at < (j + 1) * L:
            obs[at:(j + 1) * L] = seq[0][0].obs[seq[0][1]]
            mask[at:(j + 1) * L] = seq[0][0].mask[seq[0][1]]
    act = np.clip(act, 0, ACT - 1)
    out = {"obs": obs, "ep": ep, "act": act, "mask": mask, "w": w, "dsw": dsw}
    if lab_w is not None:
        out["set_labels"] = lab_w
    return out


# ---------------------------------------------------------------------------------------------
# the provider the learner holds
# ---------------------------------------------------------------------------------------------

class HumanImitation:
    """
    Built by the learner (learner.LearnerActor, human_imitation=settings dict, obs = the run's
    config.ObsConfig). start(net) begins the loader thread once the network exists (its unpacker
    gives the observation offsets). ppo_update calls term() on its minibatches; the learner calls
    begin_update() before and report() after each update.
    """

    def __init__(self, settings: Dict[str, Any], obs: Any = None):
        self.s = HumanImitationSettings(**settings)
        self.s.check()
        self.coef = float(self.s.coef)
        self.set_coef = float(self.s.set_coef)
        self.every = int(self.s.every)
        self.run_settings = run_obs_settings(obs)
        self.off_reason: Optional[str] = None
        self.started = False
        self.L = 256
        self.dim: Optional[int] = None
        self._lock = threading.Lock()
        self._pool: List[Shard] = []
        self._seqs: List[List[Tuple[Any, int, int, int]]] = []
        self._seq_at = 0
        self._want_swap = 0
        self._wake = threading.Event()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._rng = random.Random(self.s.seed)
        self._nprng = np.random.default_rng(self.s.seed)
        self.index: Dict[str, Any] = {}
        self.holdout: Optional[Dict[str, Any]] = None
        self.ho = {"agree": None, "agree_pp": None, "ce": None, "upd": None, "n": 0}
        self.total = 0
        self.dropped = 0                      # distinct episodes dropped at load: a value that is not finite
        self._dropped_keys: set = set()
        self.track = self._new_track()
        self.status = "starting"
        self._said_status = None

    @staticmethod
    def _new_track() -> Dict[str, Any]:
        # Python numbers (wait, err, n) and GPU 0-d tensors (summed in term(), read once in report()).
        return {"n": 0, "wait": 0, "err": 0, "dev": {}}

    def _add_dev(self, name: str, value: torch.Tensor) -> None:
        v = value.detach().float()
        d = self.track["dev"]
        d[name] = d[name] + v if name in d else v

    # -- the loader ----------------------------------------------------------------------------

    def start(self, net: Any, batch_seq_len: int) -> None:
        if self.started:
            return
        self.started = True
        self.L = int(batch_seq_len)
        off = net.unpacker.offsets
        self.trans0 = int(off["transition_scalars"][0])
        self.a0, self.a1 = (int(x) for x in off["action_mask"])
        self.dim = int(max(e for _s, e in off.values()))
        if self.set_coef > 0 and getattr(net, "set_head", None) is None:
            # never attached here: a new head would change the optimizer's layout (train.py refuses
            # --human-imitation-set-coef without --set-head-coef, so this is a hand-built learner)
            logger.warning("human imitation: set_coef > 0 but the network has no SetHead; the set part is off")
            print("[learner] human imitation: the SetHead part is off (the network has no SetHead)", flush=True)
            self.set_coef = 0.0
        self._thread = threading.Thread(target=self._run, name="human-imitation-loader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        self._wake.set()

    @staticmethod
    def _lower_priority() -> None:
        """The loader thread below the learner's own (Linux: nice is per thread). Best effort."""
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except Exception:
            pass

    def _run(self) -> None:
        try:
            self._lower_priority()
            self._build_index()
            if self.off_reason:
                self.status = "off"
                logger.error(f"Human imitation: OFF ({self.off_reason}). The run goes on without the term.")
                print(f"[learner] human imitation OFF: {self.off_reason}", flush=True)
                return
            self._build_holdout()
            train = list(self.index["train_shards"])
            self._rng.shuffle(train)
            next_shard = 0
            while next_shard < len(train) and self._pool_rows() < self.s.pool_decisions and not self._stop:
                self._add(train[next_shard])
                next_shard += 1
            if not self._pool:
                self.off_reason = "no train shard could be loaded"
                self.status = "off"
                print(f"[learner] human imitation OFF: {self.off_reason}", flush=True)
                return
            self._repack()
            self.status = "on"
            print(f"[learner] human imitation ON: {len(self.index['train_shards'])} shards with train sides "
                  f"({self.index['mismatch']} of {self.index['files']} shards built otherwise, left out), "
                  f"{self.index['train_labelled']:,} labelled train decisions of {self.index['train_players']:,} players, "
                  f"pool {sum(s.rows() for s in self._pool):,} decisions; held out {self.ho['n']:,} decisions of "
                  f"{self.holdout['players'] if self.holdout else 0} players (at most "
                  f"{self.holdout['limit'] if self.holdout else 0:,} each, the largest "
                  f"{100 * (self.holdout['top_share'] if self.holdout else 0.0):.1f}%); "
                  f"{self.dropped} episode(s) dropped for a value that is not finite; roles {dict(self.index['roles'])}; "
                  f"settings {self.s.as_dict()}", flush=True)
            while not self._stop:
                self._wake.wait(timeout=60)
                self._wake.clear()
                if self._stop:
                    break
                while self._want_swap > 0 and not self._stop:
                    with self._lock:
                        self._want_swap -= 1
                    if next_shard >= len(train) and len(self._pool) >= len(train):
                        self._repack()                      # every shard is in the pool already
                        continue
                    added = tries = 0
                    while added < self.s.pool_decisions // 8 and tries < len(train) and not self._stop:
                        tries += 1
                        if next_shard >= len(train):
                            self._rng.shuffle(train)
                            next_shard = 0
                        added += self._add(train[next_shard])
                        next_shard += 1
                    self._drop_oldest()
                    self._repack()
        except Exception as e:
            self.off_reason = f"loader failed: {e!r}"
            self.status = "off"
            logger.exception("Human imitation: the loader failed; the run goes on without the term")
            print(f"[learner] human imitation OFF: {self.off_reason}", flush=True)

    def _build_index(self) -> None:
        files = shard_files(self.s.data)
        if not files:
            self.off_reason = f"no shards in {self.s.data}"
            return
        want = self.run_settings
        idx: Dict[str, Any] = {"files": len(files), "mismatch": 0, "train_shards": [], "holdout_shards": [],
                               "episodes": {}, "player_lab": defaultdict(int), "roles": defaultdict(int)}
        split_of: Optional[Dict[str, Tuple[str, str]]] = None
        split_err: Optional[str] = None
        seen = set()
        for f in files:
            try:
                with np.load(f, allow_pickle=False) as z:
                    if not shard_fits(z, want, self.dim):
                        idx["mismatch"] += 1
                        continue
                    if "ep_split" not in z.files and split_of is None and split_err is None:
                        try:                              # stage3's layout: the collector's split files
                            split_of = read_splits(self.s.splits, self.s.salt)
                        except Exception as e:
                            split_err = repr(e)
                            logger.warning(f"human imitation: the split files could not be read ({e!r}); "
                                           f"shards without ep_split are left out")
                    st = z["ep_start"]
                    games, sides, players = z["ep_game"], z["ep_side"], z["ep_player"]
                    ep_split = z["ep_split"] if "ep_split" in z.files else None
                    lab = (z["train"].astype(bool)) & (z["act"] >= 0)
                    eps: Dict[str, List[Tuple[int, int, str]]] = {"train": [], "test_player": []}
                    for e in range(len(st) - 1):
                        g, side = str(games[e]), ("p1" if int(sides[e]) == 1 else "p2")
                        if (g, side) in seen:
                            idx["roles"]["duplicate"] += 1
                            continue
                        seen.add((g, side))
                        role = episode_role(ep_split, e, side, g, split_of)
                        idx["roles"][role] += 1
                        if role not in eps:
                            continue
                        a, b = int(st[e]), int(st[e + 1])
                        if b <= a:
                            continue
                        eps[role].append((a, b, str(players[e])))
                        if role == "train":
                            idx["player_lab"][str(players[e])] += int(lab[a:b].sum())
            except Exception as e:
                idx["roles"]["unreadable"] += 1
                logger.warning(f"human imitation: shard {f} left out: {e!r}")
                continue
            idx["episodes"][f] = eps
            if eps["train"]:
                idx["train_shards"].append(f)
            if eps["test_player"]:
                idx["holdout_shards"].append(f)
        idx["train_labelled"] = int(sum(idx["player_lab"].values()))
        idx["train_players"] = len(idx["player_lab"])
        n = idx["train_labelled"]
        idx["player_w"] = {p: min(1.0, self.s.cap * n / c) for p, c in idx["player_lab"].items() if c > 0}
        self.index = idx
        if not idx["train_shards"]:
            if idx["mismatch"] == len(files):
                self.off_reason = (f"no shard was built as this run's observation ({describe_obs(want)}, obs_dim={self.dim}): "
                                   f"all {len(files)} were built otherwise; rebuild them with human/rebuild.py build "
                                   f"{'--own-tera-type on' if want[0] else '--own-tera-type off'}"
                                   f"{' --obs-matchup' if want[1] else ''}{' --obs-movesets' if want[2] else ''} ...")
            else:
                self.off_reason = (f"no train sides in the {len(files) - idx['mismatch']} shard(s) of this run's observation "
                                   f"(roles {dict(idx['roles'])})" + (f"; split files: {split_err}" if split_err else ""))

    def _load(self, path: str, role: str, uid: int, settings: Optional[HumanImitationSettings]) -> Optional[Shard]:
        eps = self.index["episodes"].get(path, {}).get(role) or []
        if not eps:
            return None
        with np.load(path, allow_pickle=False) as z:
            return Shard(path, uid, z, eps, self.trans0, int(self.dim), settings, self.index["player_w"],
                         self._nprng, with_set_labels=(self.set_coef > 0 and role == "train"))

    def _pool_rows(self) -> int:
        with self._lock:
            return sum(s.rows() for s in self._pool)

    def _add(self, path: str) -> int:
        """Loads a shard's train sides into the pool; the decisions added."""
        uid = 1 + (abs(hash((path, time.time_ns()))) % 999_999)          # x 100,000 + episode < 1e11 < padding ids
        try:
            sh = self._load(path, "train", uid, self.s)
        except Exception as e:                       # one unreadable shard is skipped, not the term
            logger.warning(f"human imitation: shard {path} left out: {e!r}")
            return 0
        if sh is None:
            return 0
        self._count_dropped(sh)
        if sh.rows() == 0:
            return 0
        with self._lock:
            self._pool.append(sh)
        return sh.rows()

    def _count_dropped(self, sh: Shard) -> None:
        """Counts the episodes a load dropped, each once (the pool reloads shards as it cycles)."""
        new = [(sh.path, a) for a, _b, _p in sh.dropped if (sh.path, a) not in self._dropped_keys]
        if new:
            self._dropped_keys.update(new)
            self.dropped = len(self._dropped_keys)
            msg = (f"human imitation: {len(new)} episode(s) of {sh.path} dropped: an observation or mask value "
                   f"that is not finite ({self.dropped} so far)")
            logger.warning(msg)
            print(f"[learner] {msg}", flush=True)

    def _drop_oldest(self) -> None:
        """Drops the oldest shards while the pool without them still holds pool_decisions."""
        with self._lock:
            total = sum(s.rows() for s in self._pool)
            while len(self._pool) > 1 and total - self._pool[0].rows() >= self.s.pool_decisions:
                total -= self._pool.pop(0).rows()

    def _repack(self) -> None:
        with self._lock:
            pool = list(self._pool)
        seqs = pack(pieces_of(pool, self.L), self.L, self._rng)
        with self._lock:
            self._seqs, self._seq_at = seqs, 0

    def _build_holdout(self) -> None:
        """
        A fixed set of held-out players' decisions: whole episodes from shards spread over the list,
        with no player above `cap` of holdout_decisions (the train side's cap). Without it one
        prolific account held half to four fifths of the set and hum_ho was mostly its agreement.
        A player's first episode is always taken; after it, an episode that would take the player
        past the limit is left out. With few held-out players the set can stay under holdout_decisions.
        """
        shards = list(self.index["holdout_shards"])
        if not shards or self.s.holdout_decisions <= 0:
            return
        target = int(self.s.holdout_decisions)
        limit = max(1, int(self.s.cap * target))
        step = max(1, len(shards) // 16)
        order = shards[::step] + [s for k, s in enumerate(shards) if k % step]
        loaded: List[Shard] = []
        per: Dict[str, int] = defaultdict(int)
        lab = 0
        for k, path in enumerate(order):
            eps = self.index["episodes"].get(path, {}).get("test_player") or []
            if not eps:
                continue
            try:
                with np.load(path, allow_pickle=False) as z:
                    lab_rows = labelled_rows(z)
                    take, want, tried = [], lab, defaultdict(int)
                    for a, b, p in eps:
                        n = int(lab_rows[a:b].sum())
                        have = per.get(p, 0) + tried[p]
                        if n == 0 or (have > 0 and have + n > limit):
                            continue
                        take.append((a, b, p))
                        tried[p] += n
                        want += n
                        if want >= target:
                            break
                    if not take:
                        continue
                    sh = Shard(path, 1_000_000 + k, z, take, self.trans0, int(self.dim), None, None, None)
            except Exception as e:
                logger.warning(f"human imitation: held-out shard {path} left out: {e!r}")
                continue
            self._count_dropped(sh)
            if sh.rows() == 0:
                continue
            loaded.append(sh)
            for e, (a, b) in enumerate(sh.eps):           # what was loaded (an episode dropped at load is not)
                n = int(sh.lab[a:b].sum())
                per[sh.players[e]] += n
                lab += n
            if lab >= target:
                break
        if not loaded:
            return
        seqs = pack(pieces_of(loaded, self.L), self.L, random.Random(12345))
        self.holdout = {"seqs": seqs, "players": sum(1 for v in per.values() if v > 0), "limit": limit,
                        "top_share": (max(per.values()) / lab) if lab else 0.0}
        self.ho["n"] = lab

    # -- what the learner calls ----------------------------------------------------------------

    def begin_update(self) -> None:
        """Before an update: asks the loader to swap an eighth of the pool."""
        self.track = self._new_track()
        if self.status == "on":
            with self._lock:
                self._want_swap += 1
            self._wake.set()

    def next_batch(self) -> Optional[Dict[str, np.ndarray]]:
        with self._lock:
            if not self._seqs:
                return None
            if self._seq_at + self.s.seqs > len(self._seqs):
                self._rng.shuffle(self._seqs)
                self._seq_at = 0
            take = self._seqs[self._seq_at:self._seq_at + self.s.seqs]
            self._seq_at += self.s.seqs
        return gather(take, self.L, with_set_labels=self.set_coef > 0)

    def wants(self, minibatch_index: int) -> bool:
        return self.status == "on" and minibatch_index % self.every == 0

    def term(self, net: Any, dev: Any, obs_dtype: Any, batch_seq_len: int, first_epoch: bool) -> Optional[torch.Tensor]:
        """coef * the weighted CE of one human batch (+ the SetHead part), or None (no batch ready, or no labelled row)."""
        b = self.next_batch()
        if b is None:
            self.track["wait"] += 1
            return None
        wsum = float(b["w"].sum())                   # on the CPU: nothing waits for the GPU here
        if not wsum > 0:
            return None
        obs = torch.from_numpy(b["obs"]).to(dev, non_blocking=True).to(obs_dtype)
        ep = torch.from_numpy(b["ep"]).to(dev)
        w = torch.from_numpy(b["w"]).to(dev)
        labels = b.get("set_labels") if self.set_coef > 0 else None
        if labels is not None:
            out = net(obs, ep, batch_seq_len=batch_seq_len, return_z=True)
            logits, z_tokens = out[0].float(), out[5]
        else:
            logits, z_tokens = net(obs, ep, batch_seq_len=batch_seq_len)[0].float(), None
        mask = torch.from_numpy(b["mask"]).to(dev)
        act = torch.from_numpy(b["act"]).to(dev)
        logp = torch.log_softmax(logits.masked_fill(mask <= 0.5, -1e4), dim=-1)
        ce = -logp.gather(1, act[:, None]).squeeze(1)
        loss = (w * ce).sum() / wsum
        term = self.coef * loss
        if z_tokens is not None:
            from ppo_core import compute_set_head_loss
            lab_t = torch.from_numpy(labels).to(dev)
            s_term, s_st = compute_set_head_loss(net, z_tokens, lab_t.reshape(lab_t.shape[0], 6, -1), coef=self.set_coef)
            term = term + s_term
            if first_epoch:
                self._add_dev("set_loss", s_st["set_loss"])
                self._add_dev("set_batches", torch.ones((), device=s_st["set_loss"].device))
                n_lab = s_st["set_ability_n"] + s_st["set_item_n"] + s_st.get("set_move_n", 0.0) + s_st.get("set_tera_n", 0.0)
                self._add_dev("set_n", n_lab)
        # A term that is not finite is counted (hum_nonfinite) and returned as it is. Zeroing it here
        # would not keep it out: the gradient through log_softmax and the layers would still be NaN
        # (0 x NaN), and the update's loss would look finite while the weights were not. What keeps
        # non-finite values out is the load (Shard drops such episodes) and HumanImitationSettings.check;
        # past those, only a network that is already not finite gives a non-finite term, and the
        # update's NaN loss then tells the supervisor at the update it happened.
        finite = torch.isfinite(term)
        self._add_dev("nonfinite", (~finite).float())
        if first_epoch:
            with torch.no_grad():
                lab = w > 0
                n = int((b["w"] > 0).sum())
                self.track["n"] += n
                self._add_dev("ce_w", torch.where(finite, (w * ce).sum(), torch.zeros((), device=w.device)))
                self._add_dev("w", w.sum())
                self._add_dev("agree", ((logp.argmax(-1) == act) & lab).sum())
                self._add_dev("sw", (lab & (act >= SWITCH_LO) & (act <= SWITCH_HI)).sum())
                self._add_dev("dsw", torch.from_numpy(b["dsw"]).to(dev).logical_and(lab).sum())
                self.total += n
        return term

    @torch.no_grad()
    def holdout_agreement(self, net: Any, dev: Any, obs_dtype: Any, update_idx: int) -> None:
        """Top-1 agreement with the held-out players on the fixed set (pooled and per player), and their CE."""
        if self.holdout is None or self.status != "on":
            return
        was = net.training
        net.eval()
        try:
            hit = defaultdict(int)
            cnt = defaultdict(int)
            ce_sum = 0.0
            seqs = self.holdout["seqs"]
            S = 3                               # the learner's own minibatch: 3 sequences of 256
            for j in range(0, len(seqs), S):
                chunk = seqs[j:j + S]
                b = gather(chunk, self.L)
                obs = torch.from_numpy(b["obs"]).to(dev).to(obs_dtype)
                ep = torch.from_numpy(b["ep"]).to(dev)
                logits = net(obs, ep, batch_seq_len=self.L)[0].float()
                mask = torch.from_numpy(b["mask"]).to(dev)
                masked = logits.masked_fill(mask <= 0.5, -1e4)
                best = masked.argmax(-1).cpu().numpy()
                logp = torch.log_softmax(masked, dim=-1)
                rows_lab = np.zeros(len(b["act"]), dtype=bool)
                at = 0
                for seq in chunk:
                    base = at
                    for sh, a, c, _uid in seq:
                        k = next(i for i, (ea, eb) in enumerate(sh.eps) if ea <= a < eb)
                        p = sh.players[k]
                        lab = sh.lab[a:c]
                        hit[p] += int(((best[base:base + (c - a)] == sh.act[a:c]) & lab).sum())
                        cnt[p] += int(lab.sum())
                        rows_lab[base:base + (c - a)] = lab
                        base += c - a
                    at += self.L
                act = torch.from_numpy(b["act"]).to(dev)
                sel = torch.from_numpy(rows_lab).to(dev)
                ce_sum += float((-logp.gather(1, act[:, None]).squeeze(1) * sel).sum())
            tot = sum(cnt.values())
            self.ho["agree"] = (sum(hit.values()) / tot) if tot else None
            self.ho["ce"] = (ce_sum / tot) if tot else None
            per = [hit[p] / cnt[p] for p in cnt if cnt[p]]
            self.ho["agree_pp"] = (sum(per) / len(per)) if per else None
            self.ho["upd"] = int(update_idx)
        finally:
            net.train(was)

    def report(self) -> str:
        """The update line's human figures (appended at the end, only when the term is configured)."""
        t = self.track
        if self.status != "on":
            return f" hum={self.status}"
        d = {k: float(v) for k, v in t["dev"].items()}         # one GPU read per figure, once per update
        n = int(t["n"])
        ce = (d.get("ce_w", 0.0) / d["w"]) if d.get("w") else 0.0
        agree = (d.get("agree", 0.0) / n) if n else 0.0
        err = int(t["err"])
        nonfinite = int(round(d.get("nonfinite", 0.0)))

        def fmt(x):
            return "na" if x is None else f"{x:.4f}"
        with self._lock:
            pool = sum(s.rows() for s in self._pool)
        line = (f" hum_n={n} hum_ce={ce:.4f} hum_agree={agree:.3f} hum_sw={int(d.get('sw', 0))} hum_dsw={int(d.get('dsw', 0))}"
                f" hum_ho={fmt(self.ho['agree'])} hum_ho_pp={fmt(self.ho['agree_pp'])} hum_ho_ce={fmt(self.ho['ce'])}"
                f" hum_ho_upd={self.ho['upd'] if self.ho['upd'] is not None else 'na'}"
                f" hum_total={self.total} hum_pool={pool} hum_wait={int(t['wait'])} hum_err={err}"
                f" hum_nonfinite={nonfinite} hum_drop={self.dropped}")
        if self.set_coef > 0:
            nb = d.get("set_batches", 0.0)
            line += f" hum_set_loss={(d.get('set_loss', 0.0) / nb) if nb else 0.0:.4f} hum_set_n={int(d.get('set_n', 0.0))}"
        return line
