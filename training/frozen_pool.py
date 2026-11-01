"""
Frozen networks for the opponent seats of kind "frozen" (opponent_seats.FrozenSeatPlayer): fixed
checkpoints that play one seat of a share of the PPO pairs and never train.

    python train.py --mode ppo ... --opponent-seats frozen:0.25,setupnet:0.15 \
        --frozen-nets m24381=/path/M24381w.pt:0.3,sfin=/path/SFIN.pt:0.2 --frozen-past 4:0.5 \
        [--frozen-past-pick uniform|recent:N] [--frozen-refresh-min 30] [--frozen-past-min-age-min 10]

What is frozen and how:
  * each net is its own nn.Module inside a frozen actor (FrozenPoolActor: train.py makes one per inference
    actor, on that actor's GPU with half its share; the workers of that inference actor ask it), built by
    the run's RunConfig.make_model and loaded from its file once. Nothing ever writes into it: the frozen
    actors never talk to the weight store, and the learner never sees these modules. Its parameters have
    requires_grad False and every call is no_grad;
  * "past" slots hold the run's own saves of THIS phase (learner_update_*.pt of the run's ckpt_dir whose
    run_cfg.learner.mode is the trainer's: not the start, not warmup's saves, which play as the start; read
    with an mmap load, about 0.03 s a file, once per file), picked again every --frozen-refresh-min minutes
    (uniform over those at least --frozen-past-min-age-min old, or the newest N of them). Until the phase
    has such a save the past slots play the newest file there is (the start, or warmup's last). A battle
    keeps the net it started with to its end (its history is that net's): a refresh only changes what NEW
    battles draw. A net no battle holds any more is dropped;
  * the seat's own observation is built in the worker with the assembler of that net's run_cfg["obs"]
    (SFIN's own Tera type and blocks, the run's own for M24381 widened and for the past saves), so every
    net plays on the observation it was trained on;
  * the actor samples at temperature 1 (ppo_core.masked_sample under the observation's own mask), as
    self-play does; the answer is the action index only. No log-probability, value or step leaves.

What reaches the learner: nothing. The seat puts no STEP, DONE or IMITATION on the worker's queue for a
decision a frozen net answers (opponent_seats.py: only the network's fallback move, when the pool does
not answer, is a STEP, and that battle's steps are dropped by its CLEANUP as for every other seat).

Without --opponent-seats frozen:... nothing here is imported: no frozen actor is made, the inference actors
keep their GPU shares, and the workers are built as before.
"""
from __future__ import annotations

import dataclasses
import glob
import logging
import math
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

AUX_PREFIXES = ("set_head.", "opp_action_head.")
RUN_OBS = "run"           # the obs key of a net that sees the run's own observation


# ---------------------------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FrozenNetSpec:
    name: str
    path: str
    weight: float
    # RUN_OBS, or the ObsConfig fields (as a sorted tuple of (key, value)) of the net's run_cfg["obs"]
    obs: Any = RUN_OBS


@dataclasses.dataclass(frozen=True)
class FrozenSeatSettings:
    nets: Tuple[FrozenNetSpec, ...] = ()
    past: int = 0                     # slots holding the run's own saves
    past_weight: float = 0.0          # their share of the frozen battles, all together
    past_pick: str = "uniform"        # "uniform" or "recent:N"
    refresh_s: float = 1800.0
    past_min_age_s: float = 600.0
    ckpt_dir: str = ""                # where the past saves are (the run's LearnerConfig.ckpt_dir), set by train.py
    seed: int = 0

    def __post_init__(self):
        names = [n.name for n in self.nets]
        if len(set(names)) != len(names):
            raise ValueError(f"frozen nets: a name given twice ({names})")
        for n in self.nets:
            if not n.name or not n.name.replace("_", "").replace("-", "").isalnum() or n.name.startswith("past"):
                raise ValueError(f"frozen net name {n.name!r}: letters, digits, - and _ only, not starting with 'past'")
            if not (float(n.weight) >= 0.0 and math.isfinite(float(n.weight))):
                raise ValueError(f"frozen net {n.name}: weight {n.weight} must be 0 or more")
        if int(self.past) < 0 or not (float(self.past_weight) >= 0.0 and math.isfinite(float(self.past_weight))):
            raise ValueError("--frozen-past K:W needs K >= 0 and W >= 0")
        if int(self.past) == 0 and float(self.past_weight) > 0:
            raise ValueError("--frozen-past: a weight for past saves needs at least one slot")
        parse_pick(self.past_pick)
        if not float(self.refresh_s) > 0 or float(self.past_min_age_s) < 0:
            raise ValueError("--frozen-refresh-min must be above 0 and --frozen-past-min-age-min not negative")

    @property
    def slots(self) -> List[str]:
        """Slot names in slot order: the fixed nets, then past0 .. past{K-1}."""
        return [n.name for n in self.nets] + [f"past{k}" for k in range(int(self.past))]

    def slot_weights(self) -> List[float]:
        """Each slot's share of the frozen battles (summing to 1)."""
        w = [float(n.weight) for n in self.nets] + [float(self.past_weight) / max(1, int(self.past))] * int(self.past)
        total = sum(w)
        if not total > 0:
            raise ValueError("frozen seats: every weight is 0 (give --frozen-nets NAME=PATH:W or --frozen-past K:W)")
        return [x / total for x in w]

    def slot_obs(self) -> List[Any]:
        return [n.obs for n in self.nets] + [RUN_OBS] * int(self.past)


def parse_pick(spec: str) -> Tuple[str, int]:
    """'uniform' -> ('uniform', 0); 'recent:8' -> ('recent', 8). ValueError otherwise."""
    s = str(spec or "").strip().lower()
    if s == "uniform":
        return "uniform", 0
    if s.startswith("recent:"):
        try:
            n = int(s.split(":", 1)[1])
        except ValueError:
            n = 0
        if n >= 1:
            return "recent", n
    raise ValueError(f"--frozen-past-pick {spec!r}: 'uniform' or 'recent:N' (N >= 1)")


def parse_nets(spec: str) -> Tuple[FrozenNetSpec, ...]:
    """'a=/x/A.pt:0.3,b=/y/B.pt:0.2' -> specs (obs unknown yet). ValueError on anything else."""
    out = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, rest = part.partition("=")
        if not sep or not rest:
            raise ValueError(f"--frozen-nets {part!r}: give NAME=PATH:WEIGHT")
        path, sep, w = rest.rpartition(":")
        if not sep or not path:
            raise ValueError(f"--frozen-nets {part!r}: give NAME=PATH:WEIGHT")
        try:
            weight = float(w)
        except ValueError:
            raise ValueError(f"--frozen-nets {part!r}: the weight is not a number") from None
        out.append(FrozenNetSpec(name=name.strip(), path=os.path.expanduser(path.strip()), weight=weight))
    return tuple(out)


def parse_past(spec: str) -> Tuple[int, float]:
    """'4:0.5' -> (4, 0.5); '' -> (0, 0.0)."""
    s = str(spec or "").strip()
    if not s:
        return 0, 0.0
    k, sep, w = s.partition(":")
    try:
        return int(k), float(w) if sep else 1.0
    except ValueError:
        raise ValueError(f"--frozen-past {spec!r}: give K:W (slots, their share)") from None


def add_arguments(g: Any) -> None:
    """The frozen seats' flags, in opponent_seats' argument group."""
    g.add_argument("--frozen-nets", default="",
                   help="frozen seats: NAME=PATH:WEIGHT,... fixed checkpoints the frozen seats play (each on the observation "
                        "of its own run_cfg), with their share of the frozen battles")
    g.add_argument("--frozen-past", default="",
                   help="frozen seats: K:W, K slots holding the run's own saves (re-picked every --frozen-refresh-min), "
                        "together W of the frozen battles")
    g.add_argument("--frozen-past-pick", default="uniform", help="frozen seats: which saves fill the past slots: uniform or recent:N")
    g.add_argument("--frozen-refresh-min", type=float, default=30.0, help="frozen seats: minutes between two picks of the past saves")
    g.add_argument("--frozen-past-min-age-min", type=float, default=10.0,
                   help="frozen seats: a save becomes a past opponent once it is this many minutes old")


def obs_key_of(saved_obs: Dict[str, Any], run_obs: Any) -> Any:
    """
    RUN_OBS when a checkpoint's run_cfg["obs"] builds the same observation as the run's ObsConfig, else
    its ObsConfig fields as a sorted tuple (the worker builds that assembler). own_tera_type absent is False
    (checkpoints from before the key), main_line absent is False.
    """
    fields = {f.name: getattr(run_obs, f.name) for f in dataclasses.fields(run_obs)}
    theirs = dict(fields)
    for k in fields:
        if k in ("own_tera_type", "matchup", "movesets", "main_line"):
            theirs[k] = bool((saved_obs or {}).get(k, False))
        elif k in (saved_obs or {}):
            theirs[k] = saved_obs[k]
    if theirs == fields:
        return RUN_OBS
    return tuple(sorted(theirs.items()))


def settings_from_args(args: Any) -> Optional[FrozenSeatSettings]:
    """The frozen seats' settings from the flags (obs keys not yet known: resolve_obs fills them). None without nets or past slots."""
    nets = parse_nets(getattr(args, "frozen_nets", "") or "")
    past, past_w = parse_past(getattr(args, "frozen_past", "") or "")
    if not nets and not past:
        return None
    return FrozenSeatSettings(nets=nets, past=past, past_weight=past_w,
                              past_pick=str(getattr(args, "frozen_past_pick", "uniform") or "uniform"),
                              refresh_s=60.0 * float(getattr(args, "frozen_refresh_min", 30.0)),
                              past_min_age_s=60.0 * float(getattr(args, "frozen_past_min_age_min", 10.0)),
                              seed=int(getattr(args, "opponent_seats_seed", 0) or 0))


def resolve(settings: FrozenSeatSettings, run_cfg: Any, template: Optional[Dict[str, Any]] = None) -> FrozenSeatSettings:
    """
    train.py (before Ray) and league_setup.py --check-run: every fixed net's file read once, its observation
    key worked out, its network checked against the run's (the attention settings in its run_cfg, and every
    tensor's name and shape against the run's make_model, the auxiliary heads aside), and the run's ckpt_dir
    for the past saves. Raises ValueError with the reason, so a wrong file stops the start, not a game.
    template: the run's network's state dict (made here from run_cfg when not given; the working directory
    must hold vocab.json then, as for every make_model).
    """
    import torch
    nets = []
    model_now = dataclasses.asdict(run_cfg.model)
    if template is None:
        template = run_cfg.make_model().state_dict()
    want = {k: tuple(v.shape) for k, v in template.items() if "attn_mask" not in k and not str(k).startswith(AUX_PREFIXES)}
    for n in settings.nets:
        if not os.path.isfile(n.path):
            raise ValueError(f"frozen net {n.name}: {n.path} is not a file")
        ck = torch.load(n.path, map_location="cpu", weights_only=False)
        rc = ck.get("run_cfg") or {}
        model = rc.get("model") or {}
        for k in ("n_layers", "n_heads", "kv_cache_len"):
            if k in model and model[k] != model_now[k]:
                raise ValueError(f"frozen net {n.name}: {k} {model[k]} but this run builds {model_now[k]}")
        have = {k: tuple(v.shape) for k, v in frozen_state_dict(ck).items()}
        if have != want:
            missing = sorted(set(want) - set(have))[:3]
            extra = sorted(set(have) - set(want))[:3]
            shaped = sorted(k for k in set(want) & set(have) if want[k] != have[k])[:3]
            raise ValueError(f"frozen net {n.name} ({n.path}) is not this run's network: missing {missing}, unexpected {extra}, "
                             f"other shapes {[(k, have[k], want[k]) for k in shaped]}")
        obs = obs_key_of(rc.get("obs") or {}, run_cfg.obs)
        nets.append(dataclasses.replace(n, obs=obs))
        del ck
    return dataclasses.replace(settings, nets=tuple(nets), ckpt_dir=str(run_cfg.learner.ckpt_dir))


def obs_config_for(key: Any, run_obs: Any) -> Any:
    """The ObsConfig a slot's observation key stands for."""
    if key == RUN_OBS:
        return run_obs
    return dataclasses.replace(run_obs, **dict(key))


def describe(settings: FrozenSeatSettings) -> str:
    w = settings.slot_weights()
    parts = []
    for i, name in enumerate(settings.slots):
        if i < len(settings.nets):
            n = settings.nets[i]
            parts.append(f"{name} {100 * w[i]:.1f}% ({os.path.basename(n.path)}, obs {'the run' if n.obs == RUN_OBS else dict(n.obs)})")
    if settings.past:
        parts.append(f"{settings.past} past saves {100 * sum(w[len(settings.nets):]):.1f}% together "
                     f"({settings.past_pick}, re-picked every {settings.refresh_s / 60:g} min from saves at least "
                     f"{settings.past_min_age_s / 60:g} min old in {settings.ckpt_dir})")
    return "frozen nets: " + "; ".join(parts)


# ---------------------------------------------------------------------------------------------
# picking the past saves
# ---------------------------------------------------------------------------------------------

def pick_past(files: Sequence[Tuple[str, float]], k: int, pick: str, rng: random.Random) -> List[str]:
    """
    k files out of (path, mtime) pairs (any order): uniform over all of them, or over the newest N
    ("recent:N"). With fewer files than k some repeat. [] when there is none.
    """
    if not files or k <= 0:
        return []
    kind, n = parse_pick(pick)
    pool = sorted(files, key=lambda f: (f[1], f[0]))
    if kind == "recent":
        pool = pool[-n:]
    paths = [p for p, _t in pool]
    if len(paths) >= k:
        return rng.sample(paths, k)
    return [paths[i % len(paths)] for i in range(k)]


_MODE_CACHE: Dict[Tuple[str, float], Optional[str]] = {}


def checkpoint_mode(path: str) -> Optional[str]:
    """A learner checkpoint's run_cfg.learner.mode, read with an mmap load (the tensors stay on disk); None when unreadable."""
    try:
        import torch
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        return ((ck.get("run_cfg") or {}).get("learner") or {}).get("mode")
    except Exception:       # noqa: BLE001  (a file being written, or not a checkpoint: not a candidate)
        return None


def list_saves(ckpt_dir: str, min_age_s: float, now: Optional[float] = None, mode: Optional[str] = None,
               mode_of=checkpoint_mode) -> List[Tuple[str, float]]:
    """
    (path, mtime) of the learner_update_*.pt in ckpt_dir at least min_age_s old; with `mode`, only those whose
    run_cfg.learner.mode is it (each file's mode read once and cached by path and mtime).
    """
    now = time.time() if now is None else now
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "learner_update_*.pt")):
        try:
            t = os.path.getmtime(p)
        except OSError:
            continue
        if now - t < min_age_s:
            continue
        if mode is not None:
            key = (p, t)
            if key not in _MODE_CACHE:
                _MODE_CACHE[key] = mode_of(p)
            if _MODE_CACHE[key] != mode:
                continue
        out.append((p, t))
    return out


# ---------------------------------------------------------------------------------------------
# the pool (inside an inference actor)
# ---------------------------------------------------------------------------------------------

def frozen_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """The network's tensors of a learner checkpoint, without the auxiliary heads and attention masks."""
    model = ckpt.get("model") if isinstance(ckpt, dict) else None
    if not isinstance(model, dict):
        raise ValueError("no model in the checkpoint")
    return {k: v for k, v in model.items() if "attn_mask" not in k and not str(k).startswith(AUX_PREFIXES)}


class FrozenPool:
    """
    The frozen nets of one inference actor and the history (kv cache) of every frozen side it answers.

    infer(tags, obs, slots) -> actions. A tag's first call pins it to its slot's net of that moment; it
    keeps that net (and its history) until forget(tag). Nets are built with make_net() (the run's
    make_model, on the actor's device, the bf16 recurrent path as the training inference uses).
    """

    def __init__(self, settings: FrozenSeatSettings, make_net, device: str, load=None, mode: Optional[str] = None,
                 mode_of=None):
        """mode: the trainer's LearnerConfig.mode (the past saves are the saves of this phase); None: every save."""
        import torch
        self.s = settings
        self.device = device
        self._make_net = make_net
        self._load = load or (lambda p: torch.load(p, map_location="cpu", weights_only=False))
        self._mode = mode
        self._mode_of = mode_of or checkpoint_mode
        self.busy_s = 0.0                               # seconds spent in infer(), for the [Frozen] line
        self.forwards = 0
        self.slot_names = settings.slots
        self._nets: Dict[int, Any] = {}                # net id -> module
        self._net_src: Dict[int, str] = {}             # net id -> file
        self._net_sum: Dict[int, float] = {}           # net id -> its parameters' checksum at load (stats() checks it)
        self._slot_net: Dict[int, int] = {}            # slot -> net id new battles get
        self._tag_net: Dict[str, int] = {}             # tag -> pinned net id
        self._net_tags: Dict[int, int] = {}            # net id -> tags pinned to it
        self._next_id = 0
        self._kv_slot: Dict[str, int] = {}
        self._kv_free: List[int] = []
        self._kv_pool = None
        self._kv_len = None
        self._refresh_idx = 0
        self._last_refresh = 0.0
        self._pending: Optional[Dict[str, Any]] = None  # a refresh loaded on a thread, swapped in at the next call
        self._loading = False
        self.counts = {"calls": 0, "rows": 0, "refreshes": 0, "refresh_errors": 0, "nets_dropped": 0, "by_slot": [0] * len(self.slot_names)}
        self.past_files: List[str] = []
        for i, n in enumerate(settings.nets):
            self._slot_net[i] = self._add_net(frozen_state_dict(self._load(n.path)), n.path)
        if settings.past:
            self._refresh_now()                         # the first pick before the first game

    # -- nets --------------------------------------------------------------------------------------

    def _add_net(self, state: Dict[str, Any], src: str) -> int:
        net = self._make_net()
        missing, unexpected = net.load_state_dict(state, strict=False)
        learned = {n for n, _p in net.named_parameters()}
        if [k for k in missing if k in learned] or unexpected:
            raise ValueError(f"frozen net {src} does not fit the run's network: missing {[k for k in missing if k in learned][:5]}, "
                             f"unexpected {list(unexpected)[:5]}")
        for p in net.parameters():
            p.requires_grad_(False)
        net.eval()
        nid = self._next_id
        self._next_id += 1
        self._nets[nid] = net
        self._net_src[nid] = src
        self._net_sum[nid] = self._checksum(net)
        self._net_tags[nid] = 0
        return nid

    @staticmethod
    def _checksum(net: Any) -> float:
        """The sum of every parameter's absolute values and of the values themselves, in float64, read once: any write changes it."""
        import torch
        with torch.no_grad():
            parts = [torch.stack((p.detach().double().abs().sum(), p.detach().double().sum())) for p in net.parameters()]
            return float(torch.stack(parts).sum()) if parts else 0.0

    def _drop_unused(self) -> None:
        live = set(self._slot_net.values())
        for nid in [n for n, c in self._net_tags.items() if c <= 0 and n not in live]:
            self._nets.pop(nid, None)
            self._net_src.pop(nid, None)
            self._net_sum.pop(nid, None)
            self._net_tags.pop(nid, None)
            self.counts["nets_dropped"] += 1

    def _picks(self) -> List[str]:
        """This phase's saves old enough; else the newest of this phase whatever its age; else the newest file there is."""
        rng = random.Random(f"{self.s.seed}|{self._refresh_idx}")
        files = list_saves(self.s.ckpt_dir, self.s.past_min_age_s, mode=self._mode, mode_of=self._mode_of)
        if files:
            return pick_past(files, int(self.s.past), self.s.past_pick, rng)
        files = list_saves(self.s.ckpt_dir, 0.0, mode=self._mode, mode_of=self._mode_of) or list_saves(self.s.ckpt_dir, 0.0)
        return pick_past(files, int(self.s.past), "recent:1", rng)

    def _install(self, loaded: List[Tuple[str, Dict[str, Any]]]) -> None:
        base = len(self.s.nets)
        for k, (path, state) in enumerate(loaded):
            self._slot_net[base + k] = self._add_net(state, path)
        self.past_files = [p for p, _s in loaded]
        self._refresh_idx += 1
        self.counts["refreshes"] += 1
        self._drop_unused()

    def _refresh_now(self) -> None:
        """At start: the past slots filled before any game (the run's start checkpoint at the least)."""
        paths = self._picks()
        if not paths:
            raise ValueError(f"frozen seats: no learner_update_*.pt in {self.s.ckpt_dir} for the past slots")
        self._install([(p, frozen_state_dict(self._load(p))) for p in paths])
        self._last_refresh = time.time()

    def _maybe_refresh(self) -> None:
        if not self.s.past:
            return
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if pending.get("error"):
                self.counts["refresh_errors"] += 1
                logger.warning(f"frozen pool: the past saves were not re-picked: {pending['error']}")
            else:
                self._install(pending["loaded"])
                logger.info(f"frozen pool: past slots now {[os.path.basename(p) for p in self.past_files]}")
            return
        if self._loading or time.time() - self._last_refresh < float(self.s.refresh_s):
            return
        self._loading = True
        self._last_refresh = time.time()
        paths = self._picks()

        def work() -> None:
            out: Dict[str, Any] = {}
            try:
                if not paths:
                    raise ValueError(f"no save old enough in {self.s.ckpt_dir}")
                out["loaded"] = [(p, frozen_state_dict(self._load(p))) for p in paths]
            except Exception as e:      # noqa: BLE001  (a refresh never costs a game: the old picks stay)
                out["error"] = repr(e)
            self._pending = out
            self._loading = False

        threading.Thread(target=work, daemon=True, name="frozen-refresh").start()

    # -- history -----------------------------------------------------------------------------------

    def _grow(self, rows: int, net: Any) -> None:
        import torch
        have = 0 if self._kv_pool is None else int(self._kv_pool.shape[1])
        if rows <= have:
            return
        pool = torch.zeros(net.n_layers, rows, net.kv_cache_len, net.d_model, device=self.device, dtype=net.recurrent_dtype())
        lens = torch.zeros(rows, dtype=torch.long, device=self.device)
        if have:
            pool[:, :have] = self._kv_pool
            lens[:have] = self._kv_len
        self._kv_pool, self._kv_len = pool, lens
        self._kv_free = list(range(rows - 1, have - 1, -1)) + self._kv_free

    def _rows(self, tags: Sequence[str], net: Any) -> List[int]:
        import torch
        rows, fresh = [], []
        for tag in tags:
            r = self._kv_slot.get(tag)
            if r is None:
                if not self._kv_free:
                    self._grow(max(64, 2 * (0 if self._kv_pool is None else int(self._kv_pool.shape[1]))), net)
                r = self._kv_free.pop()
                self._kv_slot[tag] = r
                fresh.append(r)
            rows.append(r)
        if fresh:
            f = torch.tensor(fresh, dtype=torch.long, device=self.device)
            self._kv_pool[:, f] = 0
            self._kv_len[f] = 0
        return rows

    # -- the calls ---------------------------------------------------------------------------------

    def infer(self, tags: Sequence[str], obs_np: Any, slots: Sequence[int], mask_range: Tuple[int, int]) -> Any:
        import numpy as np
        import torch
        from ppo_core import masked_sample

        started = time.perf_counter()
        self._maybe_refresh()
        n = len(tags)
        if n != int(obs_np.shape[0]) or n != len(slots):
            raise ValueError(f"frozen infer: {n} tags, {obs_np.shape[0]} observations, {len(slots)} slots")
        self.counts["calls"] += 1
        self.counts["rows"] += n
        acts = np.zeros(n, dtype=np.int64)
        groups: Dict[int, List[int]] = {}
        for i, (tag, slot) in enumerate(zip(tags, slots)):
            nid = self._tag_net.get(tag)
            if nid is None:
                s = int(slot)
                if not 0 <= s < len(self.slot_names):
                    raise ValueError(f"frozen infer: slot {s} of {len(self.slot_names)}")
                nid = self._slot_net[s]
                self._tag_net[tag] = nid
                self._net_tags[nid] = self._net_tags.get(nid, 0) + 1
                self.counts["by_slot"][s] += 1
            groups.setdefault(nid, []).append(i)
        obs_all = torch.from_numpy(obs_np).to(self.device, non_blocking=True).float()
        m0, m1 = mask_range
        with torch.no_grad():
            for nid, idx in groups.items():
                net = self._nets[nid]
                gtags = [tags[i] for i in idx]
                rows = self._rows(gtags, net)
                ridx = torch.tensor(rows, dtype=torch.long, device=self.device)
                kv = list(self._kv_pool[:, ridx].unbind(0))
                obs = obs_all[torch.tensor(idx, dtype=torch.long, device=self.device)]
                logits, _v_logits, _v_exp, new_kv, new_len = net(obs, kv_cache=kv, kv_cache_len=self._kv_len[ridx])
                mask = obs[:, m0:m1].float()
                bad = mask.sum(dim=1) <= 0.49
                if bad.any():
                    mask = mask.clone()
                    mask[bad, 0] = 1.0
                act, _logp, _ = masked_sample(logits, mask, temp=1.0)
                self._kv_pool[:, ridx] = torch.stack(new_kv, dim=0)
                self._kv_len[ridx] = new_len
                acts[np.asarray(idx)] = act.cpu().numpy()
                self.forwards += 1
        self.busy_s += time.perf_counter() - started
        return acts

    def forget(self, tags: Sequence[str]) -> None:
        for tag in tags:
            r = self._kv_slot.pop(tag, None)
            if r is not None:
                self._kv_free.append(r)
            nid = self._tag_net.pop(tag, None)
            if nid is not None:
                self._net_tags[nid] = self._net_tags.get(nid, 1) - 1
        self._drop_unused()

    def changed_nets(self) -> List[str]:
        """The files of the nets whose parameters differ from what was loaded ([] always, unless something wrote into one)."""
        return [self._net_src[n] for n, net in self._nets.items() if self._checksum(net) != self._net_sum.get(n)]

    def stats(self) -> Dict[str, Any]:
        return {"calls": self.counts["calls"], "rows": self.counts["rows"], "refreshes": self.counts["refreshes"],
                "forwards": self.forwards, "busy_s": round(self.busy_s, 3),
                "changed": [os.path.basename(p) for p in self.changed_nets()],
                "refresh_errors": self.counts["refresh_errors"], "nets": len(self._nets), "nets_dropped": self.counts["nets_dropped"],
                "sides": len(self._tag_net), "battles_by_slot": dict(zip(self.slot_names, self.counts["by_slot"])),
                "past": [os.path.basename(p) for p in self.past_files]}


# ---------------------------------------------------------------------------------------------
# the worker's side: one thread that batches the frozen seats' questions to the inference actor
# ---------------------------------------------------------------------------------------------

class FrozenClient:
    """
    In a rollout worker with frozen seats: the seats' questions (tag, observation, slot) go on a queue
    from the players' event loop; one thread of its own takes what is there (after `wait_s`), asks the
    worker's frozen actor once (infer_frozen), and answers each question's future on that loop. The
    worker's rollout loop and its training-inference calls are not touched. forget(tag) drops a side's
    history in the actor (after the questions already queued, so a finished battle never leaves one).
    """

    def __init__(self, actor: Any, wait_s: float = 0.002, max_batch: int = 512, timeout_s: float = 30.0):
        import queue
        self.actor = actor
        self.wait_s = float(wait_s)
        self.max_batch = int(max_batch)
        self.timeout_s = float(timeout_s)
        self.q: "queue.Queue" = queue.Queue()
        self.counts = {"asked": 0, "answered": 0, "failed_calls": 0, "failed_rows": 0, "forgotten": 0, "calls": 0}
        self._thread = threading.Thread(target=self._run, daemon=True, name="frozen-client")
        self._thread.start()

    async def ask(self, tag: str, obs: Any, slot: int) -> int:
        import asyncio
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.counts["asked"] += 1
        self.q.put(("ask", tag, obs, int(slot), fut, loop))
        return int(await asyncio.wait_for(fut, timeout=self.timeout_s + 5.0))

    def forget(self, tag: str) -> None:
        self.q.put(("forget", tag))

    @staticmethod
    def _answer(fut: Any, value: Any, error: Optional[BaseException]) -> None:
        if fut.done():
            return
        if error is not None:
            fut.set_exception(error)
        else:
            fut.set_result(value)

    def _run(self) -> None:
        import queue
        import numpy as np
        import ray
        while True:
            items = [self.q.get()]
            if self.wait_s > 0:
                time.sleep(self.wait_s)
            while len(items) < self.max_batch:
                try:
                    items.append(self.q.get_nowait())
                except queue.Empty:
                    break
            asks = [it for it in items if it[0] == "ask"]
            forgets = [it[1] for it in items if it[0] == "forget"]
            if asks:
                self.counts["calls"] += 1
                error: Optional[BaseException] = None
                acts = None
                try:
                    obs = np.stack([it[2] for it in asks], axis=0)
                    acts = ray.get(self.actor.infer_frozen.remote([it[1] for it in asks], obs, [it[3] for it in asks]),
                                   timeout=self.timeout_s)
                except Exception as e:      # noqa: BLE001  (the seats fall back to the network for these)
                    error = e
                    self.counts["failed_calls"] += 1
                    self.counts["failed_rows"] += len(asks)
                for i, it in enumerate(asks):
                    fut, loop = it[4], it[5]
                    try:
                        loop.call_soon_threadsafe(self._answer, fut, None if acts is None else int(acts[i]),
                                                  error if error is not None else None)
                    except RuntimeError:
                        pass                            # the loop is closed: the worker is going away
                if error is None:
                    self.counts["answered"] += len(asks)
            if forgets:
                self.counts["forgotten"] += len(forgets)
                try:
                    self.actor.frozen_forget.remote(forgets)      # fire and forget, after this batch's questions
                except Exception:       # noqa: BLE001
                    pass


class FrozenPoolActor:
    """
    The frozen nets of a run, in a Ray actor of their own (train.py makes one per inference actor, on that
    actor's GPU with half of its share, and the workers of that inference actor ask this one). Its own
    process, so its forwards never wait behind the training inference's calls, nor those behind it: in the
    inference actor itself the pool took 57% of the actor's time on box c (15.8 ms a call of 5.7 decisions)
    while the training calls served 1,374 decisions a second.
    """

    def __init__(self, cfg: Any, frozen: FrozenSeatSettings):
        import torch
        import ppo_core
        self.cfg = cfg
        self.device = str(cfg.infer.device)
        ppo_core.flex_attention = torch.compile(ppo_core.flex_attention)     # as InferenceActor does, in its own process

        def make_net():
            net = self.cfg.make_model().to(self.device).eval()
            net.enable_bf16_recurrent_path()
            return net

        self.pool = FrozenPool(frozen, make_net, self.device, mode=str(cfg.learner.mode))
        asm = cfg.make_assembler()
        self._mask_range = tuple(asm.offsets["action_mask"])
        # One forward of every net now, so the compiled attention is built before the first question (the
        # client waits 30 s for an answer; a first compile can take longer). train.py waits for this.
        self.warmup_s = self._warm_up(int(asm.total_dim))

    def _warm_up(self, dim: int) -> float:
        """Every net run on a batch of 1 and of 8 (two sizes, so the compiled attention takes the batch as dynamic)."""
        import torch
        t0 = time.perf_counter()
        for nid in list(self.pool._nets):
            net = self.pool._nets[nid]
            for b in (1, 8):
                with torch.no_grad():
                    obs = torch.zeros(b, dim, device=self.device)
                    kv = [torch.zeros(b, net.kv_cache_len, net.d_model, device=self.device, dtype=net.recurrent_dtype())
                          for _ in range(net.n_layers)]
                    net(obs, kv_cache=kv, kv_cache_len=torch.zeros(b, dtype=torch.long, device=self.device))
        return time.perf_counter() - t0

    def ready(self) -> Dict[str, Any]:
        return {"warmup_s": round(self.warmup_s, 1), "nets": len(self.pool._nets)}

    def infer_frozen(self, tags: List[str], obs_np: Any, slots: List[int]) -> Any:
        return self.pool.infer(tags, obs_np, slots, self._mask_range)

    def frozen_forget(self, tags: List[str]) -> None:
        self.pool.forget(tags)

    def frozen_stats(self) -> Dict[str, Any]:
        return self.pool.stats()


def split_share(share: float, frozen_fraction: float = 0.5) -> Tuple[float, float]:
    """(the inference actor's GPU share, its frozen actor's) out of the inference actor's share, both rounded
    DOWN to Ray's 1/10,000 so that they never add up to more than the share they are cut from."""
    unit = 10_000
    total = math.floor(float(share) * unit + 1e-9)
    frozen = math.floor(total * float(frozen_fraction))
    return (total - frozen) / unit, frozen / unit
