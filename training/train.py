"""
Cluster Driver for Distributed Pokémon Showdown Reinforcement Learning.

This script initializes the Ray cluster, spawns fractional GPU actors for 
inference and training, and distributes rollout workers across multiple 
Pokémon Showdown server instances.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Final, Optional

import ray
from config import RunConfig
from inference import InferenceActor, WeightStore
from learner import LearnerActor
from worker import RolloutWorker
import infer_pool
import league
import opponent_seats

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# List of ports where local Pokémon Showdown server instances are running
SERVER_PORTS: Final[List[int]] = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009]



def parse_ports(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _decisions(text: str) -> int:
    """A count of decisions: 357000000, or 357e6 when that is a whole number."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of decisions")
    if not value.is_integer():
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of decisions")
    return int(value)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Everything here overrides config.py for this run only. With no arguments the run is exactly
    what config.py says, on the ten default ports.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=None, help="LearnerConfig.mode: imitation, imitation_with_jepa, jepa_pretraining, warmup, ppo, ppo_with_jepa, ...")
    ap.add_argument("--ports", default=None, help="Showdown servers to play on, e.g. 8000-8003. One rollout worker per port.")
    ap.add_argument("--battles", type=int, default=None, help="RolloutConfig.target_concurrent_battles")
    ap.add_argument("--rooms-per-pair", type=int, default=None, help="RolloutConfig.rooms_per_pair")
    ap.add_argument("--teacher", choices=("psbot", "simple"), default=None, help="RolloutConfig.imitation_teacher")
    ap.add_argument("--steps-per-update", type=int, default=None, help="LearnerConfig.steps_per_update")
    ap.add_argument("--minibatch", type=int, default=None, help="LearnerConfig.minibatch_size (a multiple of the sequence length)")
    ap.add_argument("--seq-len", type=int, default=None, help="LearnerConfig.batch_seq_len")
    ap.add_argument("--grad-accum", type=int, default=None, help="LearnerConfig.grad_accum_steps")
    ap.add_argument("--save-every", type=int, default=None, help="LearnerConfig.save_every_updates")
    ap.add_argument("--ckpt-dir", default=None, help="LearnerConfig.ckpt_dir")
    ap.add_argument("--shaping", choices=("potential", "faint", "none"), default=None, help="RewardConfig.shaping")
    ap.add_argument("--own-tera-type", action=argparse.BooleanOptionalAction, default=None,
                    help="ObsConfig.own_tera_type: show each own Pokemon its Tera type before it Terastallizes "
                         "(on by default; --no-own-tera-type for the old observation, see config.ObsConfig). "
                         "Saved in the checkpoints, which the players follow.")
    ap.add_argument("--minutes", type=float, default=0.0, help="stop cleanly after this many minutes (0 = run until Ctrl+C)")
    ap.add_argument("--stop-file", default=None, help="stop cleanly, the way --minutes does, as soon as this file exists (pipeline.py creates it)")
    # The learning rate and its schedule (a learning-rate restart: cloud/stage_lr.py computes the
    # three from the checkpoint a stage resumes from, cloud/check_restart.py checks them before the
    # start). Absent: config.py's values, as always. On a resume in the same phase the learner
    # puts every rate where the schedule of THIS start has it for the checkpoint's optimizer step.
    ap.add_argument("--lr", type=float, default=None,
                    help="LearnerConfig.lr (the full rate; on a resume in the same phase it replaces the checkpoint's)")
    ap.add_argument("--lr-hold-steps", type=int, default=None,
                    help="LearnerConfig.lr_hold_steps: optimizer steps at the full rate after the warmup")
    ap.add_argument("--lr-total-steps", type=int, default=None,
                    help="LearnerConfig.lr_total_steps: the optimizer step at which the rate has come down to 1/27. "
                         "Must lie after the warmup and the hold (lr_warmup_steps + lr_hold_steps)")
    # The same schedule counted in this phase's decisions (off unless both are given; then the step
    # flags above may not be). The warmup ramp stays lr_warmup_steps optimizer steps.
    ap.add_argument("--lr-hold-decisions", type=_decisions, default=None,
                    help="LearnerConfig.lr_hold_decisions: the full rate until this many decisions of the phase "
                         "(with --lr-total-decisions; not with --lr-hold-steps / --lr-total-steps)")
    ap.add_argument("--lr-total-decisions", type=_decisions, default=None,
                    help="LearnerConfig.lr_total_decisions: the rate has come down to 1/27 at this many decisions of the "
                         "phase and stays there. Must be above --lr-hold-decisions")
    # The training features of the plan "teach the bot reads, set-up play, abilities and items".
    # Every one is off unless given; off, the run is exactly what it was without them.
    g = ap.add_argument_group("training features (stages 2-4; all off by default)")
    g.add_argument("--read-bonus", type=float, default=None,
                   help="LearnerConfig.read_bonus b: adv += b * read for the decision (read_reward.py), before normalising, "
                        "never into the value targets; 0.30 in the plan")
    g.add_argument("--read-anneal", default=None, metavar="START:END",
                   help="decisions of this arm: the full bonus before START, linear to 0 at END (e.g. 20e6:45e6, the default)")
    g.add_argument("--read-cap", type=float, default=None, help="LearnerConfig.read_cap: one game's bonuses stay within +-this (0.5)")
    g.add_argument("--opp-action-aux-coef", type=float, default=None,
                   help="LearnerConfig.opp_action_aux_coef: weight of the head predicting the opponent's next action (0.1 in the plan)")
    g.add_argument("--boosted-row-weight", type=float, default=None,
                   help="LearnerConfig.boosted_row_weight W: policy-loss weight of rows where the opponent's active Pokemon is boosted "
                        "(+1 or more), against 1, normalised to a mean of 1 per minibatch (2 in the plan; 1 = off)")
    g.add_argument("--obs-matchup", action=argparse.BooleanOptionalAction, default=None,
                   help="ObsConfig.matchup: the matchup and set-candidate block (obs_matchup.py). Needs a checkpoint that has it "
                        "(tools/widen_checkpoint.py); saved in the checkpoints")
    g.add_argument("--obs-movesets", action=argparse.BooleanOptionalAction, default=None,
                   help="ObsConfig.movesets: the moveset block (obs_movesets.py: moves, Tera types, roles and threat of every "
                        "Pokemon). Needs --obs-matchup and a checkpoint that has it (tools/widen_checkpoint.py --movesets); "
                        "with --set-head-coef the SetHead also learns the moves and Tera types")
    g.add_argument("--set-head-coef", type=float, default=None,
                   help="LearnerConfig.set_head_coef: weight of the SetHead's loss (opposing abilities and items; needs --obs-matchup)")
    g.add_argument("--obs-main-line", action="store_true",
                   help="ObsConfig.main_line: the main line's observation (obs_main.py: obs_tera's own Tera type, obs_abil's six "
                        "fixes, switch_names in the mask; the network M24381's) in place of this line's first part, the "
                        "blocks after it. Sets own_tera_type False (the main line's own Tera type is part of it). Needs a "
                        "checkpoint trained or widened with it (tools/widen_checkpoint.py --main-line); saved in the checkpoints")
    opponent_seats.add_arguments(ap)
    infer_pool.add_arguments(ap)
    add_fast_ingest_arguments(ap)
    add_human_imitation_arguments(ap)
    add_learner_speed_arguments(ap)
    add_priority_arguments(ap)
    league.add_arguments(ap)
    args = ap.parse_args(argv)
    try:
        cfg = build_config(args)    # a value that cannot be is refused here, before Ray or a server is touched
        opponent_seats.from_args(args)
        infer_pool.from_args(args)
        human_imitation_settings(args, cfg)
        rollout_nice_of(args)
        league.from_args(args)
    except ValueError as e:
        ap.error(str(e))
    return args


def add_fast_ingest_arguments(ap: argparse.ArgumentParser) -> None:
    """
    How the learner takes in its data and hands out its weights, never what it computes (fast_ingest.py).
    Off by default; off, the learner is made and runs exactly as before. Nothing here goes into
    RunConfig, so no checkpoint records it and a run can drop it at any restart.
    """
    g = ap.add_argument_group("learner ingest (off by default)")
    g.add_argument("--fast-ingest", action="store_true",
                   help="the learner copies each message's observations to its GPU as it comes in (no 0.6 GB join "
                        "and second copy at every update; a thread of its own makes the copies), computes the advantages of "
                        "an update's messages all at once, and pushes "
                        "its weights to the inference actors from a thread of its own (their bytes out of band, which "
                        "Ray copies without the Python lock). The same tensors in the same "
                        "order reach every update: the same losses, gradients, KL stops and checkpoints, bit for bit")


def fast_ingest_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """The LearnerActor keyword of --fast-ingest when it is on ({} when it is not)."""
    return {"fast_ingest": True} if getattr(args, "fast_ingest", False) else {}


def add_human_imitation_arguments(ap: argparse.ArgumentParser) -> None:
    """
    The top players' decisions as an auxiliary cross-entropy in the PPO updates (human_imitation.py,
    human/rebuild.py's shards; ported from branch stage3). Without --human-imitation-coef, or with 0,
    nothing of it is imported and the learner and every update are what they were.
    """
    g = ap.add_argument_group("the top players' imitation term (human_imitation.py; off by default)")
    g.add_argument("--human-imitation-coef", type=float, default=None,
                   help="weight of the top players' cross-entropy added to the PPO loss; absent or 0: off")
    g.add_argument("--human-imitation-data", default=None,
                   help="directory of human/rebuild.py shards built as this run's observation (own Tera type, "
                        "--obs-matchup, --obs-movesets); shards built otherwise are never loaded")
    g.add_argument("--human-imitation-splits", default="~/dev/human-data/splits",
                   help="the collector's split files (train.tsv, test_player.tsv), read only for shards without ep_split")
    g.add_argument("--human-imitation-salt", default="~/dev/human-rebuild/salt",
                   help="the rebuild's salt (the shards' game hashes), with --human-imitation-splits")
    g.add_argument("--human-imitation-seqs", type=int, default=1,
                   help="human sequences of batch_seq_len decisions per PPO minibatch")
    g.add_argument("--human-imitation-every", type=int, default=1,
                   help="add the term to every k-th PPO minibatch")
    g.add_argument("--human-imitation-switch-weight", type=float, default=1.0,
                   help="the players' switch decisions (actions 4-9) count this many times")
    g.add_argument("--human-imitation-double-switch-weight", type=float, default=1.0,
                   help="decisions where the player switched by choice on a turn the opponent also switched count "
                        "this many times (on top of the switch weight)")
    g.add_argument("--human-imitation-cap", type=float, default=0.03,
                   help="a player holding more than this share of the labelled train decisions is weighted down to it")
    g.add_argument("--human-imitation-shuffled", action="store_true",
                   help="the control arm: labels permuted among decisions with the same legal mask")
    g.add_argument("--human-imitation-pool", type=int, default=120_000,
                   help="decisions held in the learner's memory (about 21 kB each with --obs-matchup --obs-movesets); "
                        "an eighth is swapped before each update")
    g.add_argument("--human-imitation-holdout", type=int, default=8192,
                   help="held-out players' labelled decisions in the fixed agreement set")
    g.add_argument("--human-imitation-holdout-every", type=int, default=50,
                   help="measure the held-out agreement every this many updates")
    g.add_argument("--human-imitation-set-coef", type=float, default=0.0,
                   help="the SetHead also learns from the shards' exact set labels with this weight (0: not; needs "
                        "--set-head-coef, so the head exists and the optimizer's layout does not change)")


def human_imitation_settings(args: argparse.Namespace, cfg: Optional[RunConfig] = None) -> Optional[Dict[str, Any]]:
    """
    None without --human-imitation-coef (or with 0). With it: the learner's human_imitation settings
    (human_imitation.HumanImitationSettings as a dict). A value that cannot be (a negative weight, a
    coef without --human-imitation-data, the SetHead part without the SetHead) is a ValueError, which
    parse_args turns into a refusal. The data themselves are not read here: a directory that is
    missing or holds no shard of this run's observation leaves the term off at the start (the learner
    says why) and never stops a run; cloud/check_restart.py refuses it before a start.
    """
    coef = getattr(args, "human_imitation_coef", None)
    if coef is None or float(coef) == 0.0:
        return None
    if float(coef) < 0.0:
        raise ValueError(f"--human-imitation-coef {coef} must not be negative")
    if not getattr(args, "human_imitation_data", None):
        raise ValueError("--human-imitation-coef needs --human-imitation-data (a directory of human/rebuild.py shards)")
    from human_imitation import HumanImitationSettings
    s = HumanImitationSettings(
        data=args.human_imitation_data, coef=float(coef), splits=args.human_imitation_splits,
        salt=args.human_imitation_salt, seqs=int(args.human_imitation_seqs), every=int(args.human_imitation_every),
        switch_weight=float(args.human_imitation_switch_weight),
        double_switch_weight=float(args.human_imitation_double_switch_weight), cap=float(args.human_imitation_cap),
        shuffled=bool(args.human_imitation_shuffled), pool_decisions=int(args.human_imitation_pool),
        holdout_decisions=int(args.human_imitation_holdout), holdout_every=int(args.human_imitation_holdout_every),
        set_coef=float(getattr(args, "human_imitation_set_coef", 0.0) or 0.0))
    try:
        s.check()
    except ValueError as e:
        raise ValueError(f"--human-imitation-*: {e}")
    if s.set_coef > 0.0:
        cfg = cfg if cfg is not None else build_config(args)
        if not cfg.learner.set_head_coef > 0.0:
            raise ValueError("--human-imitation-set-coef needs --set-head-coef: the term never adds a head of its own "
                             "(that would change the optimizer's layout)")
    return s.as_dict()


def add_learner_speed_arguments(ap: argparse.ArgumentParser) -> None:
    """
    How the learner computes an update, never what it computes. Off by default; off, the learner is
    made and called exactly as before. Nothing here goes into RunConfig, so no checkpoint records it
    and a run can drop it at any restart.
    """
    g = ap.add_argument_group("learner speed (off by default)")
    g.add_argument("--fast-update", action="store_true",
                   help="ppo_update's sync-free minibatch loop (ppo_core.ppo_update fast_update): the figures summed on "
                        "the GPU, the early stop's KL read behind a CUDA event, the action mask fixed up without "
                        "reading the GPU, q/k/v by unbind, foreach copies of the fp32 masters. The same losses, "
                        "gradients, KL stops and checkpoints, bit for bit")


def learner_speed_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """The LearnerActor keywords of the learner speed flags that are on ({} with none)."""
    out: Dict[str, Any] = {}
    if getattr(args, "fast_update", False):
        out["fast_update"] = True
    return out


def human_imitation_line(hum: Dict[str, Any]) -> str:
    """The line a start with the term logs (and cloud/check_restart.py prints)."""
    return (f"human imitation: coef {hum['coef']:g}{' SHUFFLED labels (the control arm)' if hum['shuffled'] else ''}, "
            f"{hum['seqs']} sequence(s) every {hum['every']} minibatch(es), switch x{hum['switch_weight']:g}, "
            f"double switch x{hum['double_switch_weight']:g}, cap {hum['cap']:g}, pool {hum['pool_decisions']:,} decisions, "
            f"held out {hum['holdout_decisions']:,} every {hum['holdout_every']} updates"
            + (f", SetHead x{hum['set_coef']:g}" if hum.get("set_coef") else "") + f"; data {hum['data']}")



# --------------------------------------------------------------------------------------------
# CPU priorities (--rollout-nice; off by default, and off is every line of main() as it was)
# --------------------------------------------------------------------------------------------

# The nice Ray starts its workers at with --rollout-nice (Ray's worker_niceness; Ray 2.59's default
# is 15). The learner and the inference actors can only run at 0 if Ray starts them there: a process
# cannot lower its own nice later (ulimit -e is 0 on the boxes, so os.setpriority to 0 raises
# PermissionError). Raising is free, so the rollout workers go up to N themselves.
RAY_WORKER_NICENESS: Final[int] = 0


def add_priority_arguments(ap: argparse.ArgumentParser) -> None:
    """
    --rollout-nice: the CPU priorities of the run's processes. Scheduling only: what the learner
    computes, the updates and the checkpoints do not change. Without it Ray starts every worker at
    its own default, nice 15, as it always has.
    """
    g = ap.add_argument_group("CPU priorities (off by default)")
    g.add_argument("--rollout-nice", type=int, default=None, metavar="N",
                   help="Ray's workers start at nice 0 instead of Ray's 15 (worker_niceness), so the learner and the "
                        "inference actors run at 0, and every rollout worker puts its own threads, and those of the "
                        "processes it starts (the opponent seats' helpers), at nice N (0 to 19). "
                        "Start the Showdown servers at N as well, in run.json: servers.command = [\"nice\", \"-n\", \"N\", "
                        "\"{python}\", \"start_servers.py\", ...]. 5 on box a. Scheduling only: the updates and the "
                        "checkpoints do not change")


def rollout_nice_of(args: argparse.Namespace) -> Optional[int]:
    """--rollout-nice: None when absent (off), else the nice of the rollout workers (0 to 19; otherwise ValueError)."""
    n = getattr(args, "rollout_nice", None)
    if n is None:
        return None
    n = int(n)
    if not 0 <= n <= 19:
        raise ValueError(f"--rollout-nice {n} must be 0 to 19 (a nice value; the workers can only raise their own)")
    return n


def init_ray_with_priorities() -> bool:
    """
    ray.init for --rollout-nice: a new local Ray whose workers start at nice RAY_WORKER_NICENESS.
    True when they do. Ray refuses a _system_config when it joins a cluster that is already running
    (RAY_ADDRESS, or one started by `ray start`). Then this joins it the way train.py always has, says
    so, and returns False: that cluster's workers keep its nice, and the rollout workers still go to N.
    """
    try:
        ray.init(ignore_reinit_error=True, _system_config={"worker_niceness": RAY_WORKER_NICENESS})
        return True
    except ValueError as e:
        if "existing cluster" not in str(e) or "_system_config" not in str(e):
            raise
        logger.warning(f"--rollout-nice: Ray joins a cluster that is already running ({e}). Its workers keep that "
                       f"cluster's nice (Ray's default is 15); the rollout workers still go to their nice.")
    ray.init(ignore_reinit_error=True)
    return False


def _pid_and_nice(_actor) -> tuple:
    """Runs inside an actor (actor_nice_line): its process id and the nice of its main thread."""
    return os.getpid(), os.getpriority(os.PRIO_PROCESS, os.getpid())


def actor_nice_line(learner, infers, timeout: float = 10.0) -> str:
    """
    "learner pid 123 nice 0; inference pids [124, 125] nice [0, 0]": the nice of the learner's and the
    inference actors' processes, asked of each actor. Informational only: an actor that does not
    answer within `timeout` seconds is "?". Never raises.
    """
    actors = [learner] + list(infers)
    try:
        refs = [a.__ray_call__.remote(_pid_and_nice) for a in actors]
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=timeout)
        ready = set(ready)
        got = []
        for ref in refs:
            try:
                got.append(tuple(ray.get(ref, timeout=1)) if ref in ready else ("?", "?"))
            except Exception:       # noqa: BLE001
                got.append(("?", "?"))
    except Exception as e:          # noqa: BLE001
        return f"the actors' nice could not be read ({e!r})"
    return (f"learner pid {got[0][0]} nice {got[0][1]}; inference pids {[g[0] for g in got[1:]]} "
            f"nice {[g[1] for g in got[1:]]}")


class NiceWatch:
    """
    --rollout-nice N: every rollout worker's set_nice call (worker.RolloutWorker.set_nice), which puts
    the worker's threads and every process it has started (Foul Play's helpers, PSBot's engine) at N.

    start(): asked of every worker at once, right after they are made and before their games are
    started (Ray starts an actor's calls in the order they were sent, so the worker's rollout thread
    starts at N), then up to START_TIMEOUT_S seconds for the answers, and one line with the counts.
    A worker still starting by then is not waited for: its call runs before its games all the same,
    and its answer is logged by the telemetry loop (after_beats).

    after_beats(): a worker whose heartbeat comes back without its nice is a new process (Ray started
    the actor again, or its call failed): set_nice is asked of it again, at most once a minute.
    """

    START_TIMEOUT_S = 10.0
    ASK_AGAIN_S = 60.0

    def __init__(self, n: int, workers: List[Any], ports: List[int]):
        self.n = int(n)
        self.workers = list(workers)
        self.ports = list(ports)
        self.pending: Dict[int, Any] = {}           # worker index -> the ObjectRef of its set_nice call
        self.answers: Dict[int, Dict[str, Any]] = {}
        self.asked_at: Dict[int, float] = {}
        self.asked_again = 0

    def ask(self, indices: Optional[List[int]] = None) -> None:
        for i in (range(len(self.workers)) if indices is None else indices):
            self.pending[i] = self.workers[i].set_nice.remote(self.n)
            self.asked_at[i] = time.time()

    def collect(self, timeout: float) -> List[int]:
        """The calls that have answered within `timeout` seconds: their counts are kept. -> their worker indices."""
        if not self.pending:
            return []
        refs = list(self.pending.values())
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=timeout)
        ready = set(ready)
        done = []
        for i, ref in list(self.pending.items()):
            if ref not in ready:
                continue
            del self.pending[i]
            try:
                self.answers[i] = dict(ray.get(ref, timeout=10))
            except Exception as e:      # noqa: BLE001  (the actor died; the heartbeats will show it)
                self.answers[i] = {"error": repr(e)}
                logger.warning(f"--rollout-nice {self.n}: the worker on port {self.ports[i]} did not take its nice: {e!r}")
            done.append(i)
        return done

    def line(self, indices: Optional[List[int]] = None) -> str:
        idx = sorted(self.answers) if indices is None else sorted(indices)
        good = [self.answers[i] for i in idx if "error" not in self.answers[i]]
        text = (f"rollout nice {self.n}: {len(good)} of {len(self.workers) if indices is None else len(idx)} workers "
                f"at nice {self.n} ({sum(int(a.get('threads', 0)) for a in good)} threads, "
                f"{sum(int(a.get('raised', 0)) for a in good)} raised, {sum(int(a.get('failed', 0)) for a in good)} could not be)")
        kids = sum(int(a.get("children", 0)) for a in good)
        if kids:
            # the processes the workers started (opponent seats' helpers), at the workers' nice as well
            text += (f", and {kids} processes they started ({sum(int(a.get('child_threads', 0)) for a in good)} threads, "
                     f"{sum(int(a.get('child_raised', 0)) for a in good)} raised)")
        if indices is None and self.pending:
            text += (f"; {len(self.pending)} still starting (ports {[self.ports[i] for i in sorted(self.pending)]}): "
                     f"their nice comes before their games")
        return text

    def start(self) -> None:
        """Never raises: a nice that cannot be set costs speed, never the run."""
        try:
            self.ask()
            self.collect(self.START_TIMEOUT_S)
            logger.info(self.line())
        except Exception as e:      # noqa: BLE001
            logger.warning(f"rollout nice {self.n}: the workers' nice could not be asked for ({e!r}); "
                           f"the heartbeats ask again")

    def after_beats(self, beats: List[Any]) -> None:
        """Never raises: it runs inside the telemetry loop, whose errors the supervisor counts."""
        try:
            done = self.collect(0)
            if done:
                logger.info(self.line(done))
            now = time.time()
            again = [i for i, b in enumerate(beats[:len(self.workers)])
                     if isinstance(b, dict) and "nice" not in b and i not in self.pending
                     and now - self.asked_at.get(i, 0.0) >= self.ASK_AGAIN_S]
            if again:
                logger.warning(f"rollout nice {self.n}: the worker(s) on port(s) {[self.ports[i] for i in again]} answer without "
                               f"their nice (a new process: Ray started the actor again?); set_nice({self.n}) asked again")
                self.asked_again += len(again)
                self.ask(again)
        except Exception as e:      # noqa: BLE001
            logger.warning(f"rollout nice {self.n}: the workers' nice could not be checked this time ({e!r})")


def parse_anneal(spec: str) -> tuple:
    """'20e6:45e6' -> (20000000, 45000000). Raises ValueError unless 0 <= START < END."""
    try:
        a, b = (int(float(x)) for x in str(spec).split(":"))
    except ValueError:
        raise ValueError(f"--read-anneal {spec!r} is not START:END (decisions, e.g. 20e6:45e6)")
    if not 0 <= a < b:
        raise ValueError(f"--read-anneal {spec!r}: START must be at least 0 and below END")
    return a, b


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig.default()
    learner, rollout, reward = {}, {}, {}
    for field, value in (("mode", args.mode), ("steps_per_update", args.steps_per_update), ("minibatch_size", args.minibatch),
                         ("batch_seq_len", args.seq_len), ("grad_accum_steps", args.grad_accum),
                         ("save_every_updates", args.save_every), ("ckpt_dir", args.ckpt_dir)):
        if value is not None:
            learner[field] = value
    for field, value in (("target_concurrent_battles", args.battles), ("rooms_per_pair", args.rooms_per_pair),
                         ("imitation_teacher", args.teacher)):
        if value is not None:
            rollout[field] = value
    if args.shaping is not None:
        reward["shaping"] = args.shaping

    # The learning-rate flags. getattr: a Namespace built by hand, without them, means "not given".
    def given(name: str):
        return getattr(args, name, None)

    if given("lr") is not None:
        if not args.lr > 0.0:
            raise ValueError(f"--lr {args.lr} must be above zero")
        learner["lr"] = float(args.lr)
    if given("lr_total_steps") is not None or given("lr_hold_steps") is not None:
        # A schedule that is asked for must make sense: the anneal has to end after the hold does,
        # or the rate falls 27 times in one optimizer step at the end of the hold.
        warm = int(cfg.learner.lr_warmup_steps)
        hold = int(cfg.learner.lr_hold_steps if given("lr_hold_steps") is None else args.lr_hold_steps)
        total = int(cfg.learner.lr_total_steps if given("lr_total_steps") is None else args.lr_total_steps)
        if hold < 0:
            raise ValueError(f"--lr-hold-steps {hold} must not be negative")
        if total <= warm + hold:
            raise ValueError(
                f"the learning-rate schedule cannot be: lr_total_steps {total:,} is not after the warmup and the hold "
                f"({warm:,} + {hold:,} = {warm + hold:,}), so the whole anneal to 1/27 of the rate would happen in one optimizer step "
                f"at step {warm + hold + 1:,}. Give --lr-total-steps above {warm + hold:,}"
                + ("" if given("lr_hold_steps") is not None else f", or a shorter --lr-hold-steps (config.py has {hold:,})") + ".")
        if given("lr_hold_steps") is not None:
            learner["lr_hold_steps"] = hold
        if given("lr_total_steps") is not None:
            learner["lr_total_steps"] = total
    if given("lr_hold_decisions") is not None or given("lr_total_decisions") is not None:
        # The schedule in decisions: both flags or neither, never next to the step flags (one
        # schedule rules the rate, and a run should not read as if the other did).
        if given("lr_hold_decisions") is None or given("lr_total_decisions") is None:
            raise ValueError("--lr-hold-decisions and --lr-total-decisions go together: give both or neither")
        if given("lr_hold_steps") is not None or given("lr_total_steps") is not None:
            raise ValueError("--lr-hold-decisions / --lr-total-decisions replace --lr-hold-steps / --lr-total-steps: "
                             "give one schedule, not both")
        hold_d, total_d = int(args.lr_hold_decisions), int(args.lr_total_decisions)
        if hold_d < 0:
            raise ValueError(f"--lr-hold-decisions {hold_d} must not be negative")
        if total_d <= hold_d:
            raise ValueError(f"the learning-rate schedule cannot be: --lr-total-decisions {total_d:,} is not after "
                             f"--lr-hold-decisions {hold_d:,}")
        learner["lr_hold_decisions"] = hold_d
        learner["lr_total_decisions"] = total_d
    if learner:
        cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **learner))
    if rollout:
        cfg = dataclasses.replace(cfg, rollout=dataclasses.replace(cfg.rollout, **rollout))
    if reward:
        cfg = dataclasses.replace(cfg, reward=dataclasses.replace(cfg.reward, **reward))
    if getattr(args, "own_tera_type", None) is not None:
        cfg = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, own_tera_type=args.own_tera_type))

    # The stage 2-4 features.
    feats = {}
    if given("read_bonus") is not None:
        if args.read_bonus < 0.0:
            raise ValueError(f"--read-bonus {args.read_bonus} must not be negative")
        feats["read_bonus"] = float(args.read_bonus)
    if given("read_anneal") is not None:
        feats["read_anneal_start"], feats["read_anneal_end"] = parse_anneal(args.read_anneal)
    if given("read_cap") is not None:
        if args.read_cap < 0.0:
            raise ValueError(f"--read-cap {args.read_cap} must not be negative")
        feats["read_cap"] = float(args.read_cap)
    if given("opp_action_aux_coef") is not None:
        if args.opp_action_aux_coef < 0.0:
            raise ValueError(f"--opp-action-aux-coef {args.opp_action_aux_coef} must not be negative")
        feats["opp_action_aux_coef"] = float(args.opp_action_aux_coef)
    if given("boosted_row_weight") is not None:
        if not args.boosted_row_weight > 0.0:
            raise ValueError(f"--boosted-row-weight {args.boosted_row_weight} must be above zero")
        feats["boosted_row_weight"] = float(args.boosted_row_weight)
    if given("set_head_coef") is not None:
        if args.set_head_coef < 0.0:
            raise ValueError(f"--set-head-coef {args.set_head_coef} must not be negative")
        feats["set_head_coef"] = float(args.set_head_coef)
    if feats:
        cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **feats))
    if given("obs_matchup") is not None:
        if not args.obs_matchup and given("obs_movesets"):
            raise ValueError("--obs-movesets needs --obs-matchup (the moveset block reads the matchup block's sets)")
        cfg = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, matchup=bool(args.obs_matchup)))
    if given("obs_movesets") is not None:
        if args.obs_movesets and not cfg.obs.matchup:
            raise ValueError("--obs-movesets needs --obs-matchup (the moveset block reads the matchup block's sets)")
        cfg = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, movesets=bool(args.obs_movesets)))
    if cfg.learner.set_head_coef > 0.0 and not cfg.obs.matchup:
        raise ValueError("--set-head-coef needs --obs-matchup (the SetHead reads the matchup block's set candidates)")
    if getattr(args, "obs_main_line", False):
        # The main line's observation (obs_main.py). It brings the main line's own Tera type, so this
        # line's own_tera_type goes off; asking for both is refused.
        if getattr(args, "own_tera_type", None):
            raise ValueError("--obs-main-line carries the main line's own Tera type: do not give --own-tera-type with it")
        cfg = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, own_tera_type=False, main_line=True))
    return cfg


def features_line(cfg: RunConfig) -> str:
    """The stage 2-4 settings this start runs with (logged at every start)."""
    lc = cfg.learner
    return (f"training features: read_bonus={lc.read_bonus} read_anneal={lc.read_anneal_start}:{lc.read_anneal_end} "
            f"read_cap={lc.read_cap} opp_action_aux_coef={lc.opp_action_aux_coef} boosted_row_weight={lc.boosted_row_weight} "
            f"obs_matchup={cfg.obs.matchup} set_head_coef={lc.set_head_coef}"
            + (" obs_movesets=True" if getattr(cfg.obs, "movesets", False) else "")
            + (" obs_main_line=True" if getattr(cfg.obs, "main_line", False) else ""))


def ppo_settings_line(cfg: RunConfig) -> str:
    """The line main() logs at every start: the update's settings and the learning-rate schedule.
    cloud/check_restart.py prints the line a start WILL log from the same function."""
    lc = cfg.learner
    # The decisions schedule is named only when it is on, so every other start logs the line it did.
    by_decisions = int(getattr(lc, "lr_total_decisions", 0) or 0) > 0
    return (f"ppo settings: clip_coef={lc.clip_coef} update_epochs={lc.update_epochs} lr={lc.lr} target_kl={lc.target_kl} "
            f"ent_coef={lc.ent_coef} lr_warmup_steps={lc.lr_warmup_steps} lr_hold_steps={lc.lr_hold_steps} "
            f"lr_total_steps={lc.lr_total_steps}"
            + (f" lr_hold_decisions={lc.lr_hold_decisions} lr_total_decisions={lc.lr_total_decisions}" if by_decisions else ""))


def gpu_shares(gpus: float) -> tuple:
    """
    The GPU share Ray reserves for the inference actor and for the learner.

    By default the two share one GPU (0.35 and 0.65), as every run so far. With PSPPO_GPU_SPLIT=1
    in the environment, on a machine with two or more GPUs, each gets a GPU of its own: Ray then
    hands the two actors different devices. This is placement only. What either of them computes
    does not change.
    """
    if os.environ.get("PSPPO_GPU_SPLIT", "0") == "1" and gpus >= 2:
        return 1.0, 1.0
    return 0.35, 0.65


def spawn_inference_and_learner(cfg_ref, weight_store, speed: "infer_pool.SpeedOptions", learner_kwargs=None,
                                frozen=None):
    """
    The learner and speed.infer_actors inference actors (train.py with a speed flag; without one
    main() makes the two actors as it always has). The learner is placed first, so that it has its
    whole GPU before any share of one is handed out (infer_pool.plan_gpus). frozen: the frozen seats'
    settings (frozen_pool.py), or None. With them every inference actor's share is split in two
    (frozen_pool.split_share) and a FrozenPoolActor takes the other half, one per inference actor; the
    return is then (infers, learner, frozen_actors). Without them (infers, learner), as always.
    """
    gpus = float(ray.cluster_resources().get("GPU", 0))
    shares, learner_gpus = infer_pool.plan_gpus(speed.infer_actors, gpus, speed.infer_gpus)
    logger.info(speed.line())
    logger.info(f"GPU shares: inference {shares}, learner {learner_gpus}")
    learner = ray.remote(num_gpus=learner_gpus)(LearnerActor).remote(cfg_ref, weight_store, **(learner_kwargs or {}))
    try:
        ray.get(learner.__ray_ready__.remote(), timeout=600)
    except Exception as e:
        logger.warning(f"The learner was not ready after 10 minutes ({e!r}); placing the inference actors anyway")
    kwargs = speed.actor_kwargs()
    InferRemote = ray.remote(InferenceActor)
    if frozen is None:
        infers = [InferRemote.options(**infer_pool.actor_options(share, speed.infer_restarts)).remote(cfg_ref, weight_store, **kwargs)
                  for share in shares]
        return infers, learner
    import frozen_pool
    halves = [frozen_pool.split_share(share) for share in shares]
    logger.info(f"frozen seats: each inference actor's GPU share split in two: inference {[h[0] for h in halves]}, "
                f"frozen actors {[h[1] for h in halves]}")
    infers = [InferRemote.options(**infer_pool.actor_options(h[0], speed.infer_restarts)).remote(cfg_ref, weight_store, **kwargs)
              for h in halves]
    FrozenRemote = ray.remote(frozen_pool.FrozenPoolActor)
    frozen_actors = [FrozenRemote.options(num_gpus=h[1]).remote(cfg_ref, frozen) for h in halves]
    wait_for_frozen(frozen_actors)
    return infers, learner, frozen_actors


def wait_for_frozen(actors) -> None:
    """The frozen actors ready (their nets loaded and run once) before any worker asks them; never raises."""
    try:
        got = ray.get([a.ready.remote() for a in actors], timeout=900)
        logger.info(f"frozen actors ready: {got}")
    except Exception as e:      # noqa: BLE001  (the seats fall back to the network while a frozen actor is not there)
        logger.warning(f"the frozen actors were not ready after 15 minutes ({e!r}); the workers start anyway")


class PoolWatch:
    """What the telemetry loop needs to say which workers a dead or silent inference actor leaves without answers."""

    # Seconds the telemetry waits for every inference actor's get_stats before it names the silent ones.
    STATS_TIMEOUT_S = 180.0

    def __init__(self, actor_of: List[int], worker_ports: List[int]):
        self.actor_of = list(actor_of)
        self.worker_ports = list(worker_ports)
        self.versions = infer_pool.VersionWatch()

    def ports_of(self, k: int) -> List[int]:
        return [p for p, a in zip(self.worker_ports, self.actor_of) if a == k]


async def main(args: Optional[argparse.Namespace] = None):
    """
    Main entry point for the training cluster.
    
    Orchestrates the following:
    1. Ray initialization and configuration pinning.
    2. Fractional GPU allocation for Learner (0.65) and Inference (0.35).
    3. Distribution of self-play pairs across available local servers.
    4. Real-time telemetry and monitoring.
    """
    # --rollout-nice: Ray's workers start at nice 0 (init_ray_with_priorities). Without it, as always.
    nice = rollout_nice_of(args) if args is not None else None
    try:
        if nice is None:
            ray.init(ignore_reinit_error=True)
        else:
            ray_at_zero = init_ray_with_priorities()
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        return

    # 1. Global Configuration (Single Source of Truth)
    args = args or parse_args([])
    cfg = build_config(args)
    server_ports = parse_ports(args.ports) if args.ports else list(SERVER_PORTS)
    rooms_per_pair = int(cfg.rollout.rooms_per_pair)
    target_concurrent = int(cfg.rollout.target_concurrent_battles)
    logger.info(
        f"mode={cfg.learner.mode} teacher={cfg.rollout.imitation_teacher} reward={cfg.reward.shaping} "
        f"ports={server_ports} steps_per_update={cfg.learner.steps_per_update} "
        f"minibatch={cfg.learner.minibatch_size}x{cfg.learner.grad_accum_steps} seq={cfg.learner.batch_seq_len} ckpt={cfg.learner.ckpt_dir}"
    )
    # A line of its own: pipeline.py reads the one above to its end (ckpt= takes the rest).
    logger.info(ppo_settings_line(cfg))
    logger.info(features_line(cfg))
    if nice is not None:
        logger.info(f"CPU priorities: Ray's workers start at nice {RAY_WORKER_NICENESS if ray_at_zero else 'of the running cluster'} "
                    f"(the learner, the inference actors); every rollout worker puts its threads at nice {nice}. The Showdown "
                    f"servers' nice is servers.command's (nice -n {nice} in run.json)")
    # The top players' imitation term (human_imitation.py): only when it is asked for does the learner
    # get the argument; without it the learner is made exactly as before.
    learner_kwargs = {}
    # --fast-ingest (fast_ingest.py): only when it is on does the learner get the argument.
    ingest_kw = fast_ingest_kwargs(args)
    if ingest_kw:
        learner_kwargs.update(ingest_kw)
        logger.info("learner ingest: " + " ".join(f"{k}={v}" for k, v in ingest_kw.items()))
    hum = human_imitation_settings(args, cfg)
    if hum is not None:
        learner_kwargs["human_imitation"] = hum
        logger.info(human_imitation_line(hum) + ". The learner loads only shards built as this run's observation; "
                    "the update line ends with hum_n, hum_ce, hum_agree, hum_ho (held-out players' agreement) ...")
        if cfg.learner.mode not in ("ppo", "ppo_with_jepa"):
            logger.info(f"human imitation: mode {cfg.learner.mode} does not train on the PPO loss; the term waits for a PPO phase")
    # The learner speed flags (--fast-update): only those that are on reach the learner.
    speed_kw = learner_speed_kwargs(args)
    if speed_kw:
        learner_kwargs.update(speed_kw)
        logger.info("learner speed: " + " ".join(f"{k}={v}" for k, v in speed_kw.items()))
    # The league flags (league.py: --learn-queue-episodes, --lag-stats): only those that are on reach
    # the workers and the learner.
    league_opts = league.from_args(args)
    league_worker_kw = league.worker_kwargs(league_opts)
    if not league_opts.is_default():
        learner_kwargs.update(league.learner_kwargs(league_opts))
        logger.info(league.line(league_opts))
    # The frozen seats (frozen_pool.py, --opponent-seats frozen:...): their nets are read and checked now;
    # their frozen actors (one per inference actor, half of its GPU share) are made beside the inference
    # actors. Without them nothing of it runs.
    seats_asked = opponent_seats.from_args(args)
    frozen_settings = None
    frozen_actors = []
    if seats_asked is not None and seats_asked.share("frozen") > 0:
        import frozen_pool
        frozen_settings = frozen_pool.resolve(seats_asked.frozen, cfg)
        logger.info(frozen_pool.describe(frozen_settings))

    # Calculate distribution
    total_pairs = math.ceil(target_concurrent / rooms_per_pair)
    n_workers = len(server_ports)
    
    # Balance pairs across worker actors
    base_pairs = total_pairs // n_workers
    remainder = total_pairs % n_workers
    pairs_per_worker = [base_pairs + (1 if i < remainder else 0) for i in range(n_workers)]

    logger.info(
        f"Initializing Run: Target Battles={target_concurrent}, "
        f"Total Pairs={total_pairs}, Workers={n_workers}"
    )

    # 2. Initialize Shared Infrastructure
    # We pin the config in Ray's object store to prevent unnecessary serialization
    cfg_ref = ray.put(cfg)
    weight_store = WeightStore.remote()

    # The speed flags (infer_pool.py). At their defaults everything below is what it always was.
    speed = infer_pool.from_args(args)

    # Define Remote Actors with Resource Constraints
    # Splitting the GPU between Learner (0.65) and Inference (0.35), or one GPU each (gpu_shares)
    if speed.is_default():
        infer_gpus, learner_gpus = gpu_shares(float(ray.cluster_resources().get("GPU", 0)))
        logger.info(f"GPU shares: inference {infer_gpus}, learner {learner_gpus}")
        if frozen_settings is not None:
            import frozen_pool
            infer_gpus, frozen_gpus = frozen_pool.split_share(infer_gpus)
            logger.info(f"frozen seats: the inference actor's GPU share split in two: inference {infer_gpus}, frozen actor {frozen_gpus}")
        InferRemote = ray.remote(num_gpus=infer_gpus)(InferenceActor)
        LearnerRemote = ray.remote(num_gpus=learner_gpus)(LearnerActor)
    
    # Workers no longer need num_gpus because they just pass data!
    WorkerRemote = ray.remote(num_cpus=1)(RolloutWorker).options(
        max_restarts=-1,  
        max_task_retries=0
    )

    # Spawn Core Actors 
    if speed.is_default():
        infer = InferRemote.remote(cfg_ref, weight_store)
        learner = LearnerRemote.remote(cfg_ref, weight_store, **learner_kwargs)
        infers = [infer]
        if frozen_settings is not None:
            frozen_actors = [ray.remote(num_gpus=frozen_gpus)(frozen_pool.FrozenPoolActor).remote(cfg_ref, frozen_settings)]
            wait_for_frozen(frozen_actors)
    elif frozen_settings is None:
        infers, learner = spawn_inference_and_learner(cfg_ref, weight_store, speed, learner_kwargs)
    else:
        infers, learner, frozen_actors = spawn_inference_and_learner(cfg_ref, weight_store, speed, learner_kwargs, frozen_settings)
    # Which actor serves each worker (all of them the one actor by default).
    worker_pairs = [c for c in pairs_per_worker if c > 0]
    actor_of = infer_pool.assign_workers(worker_pairs, len(infers))

    # 3. Spawn Rollout Workers
    # Opponent seats (--opponent-seats, bench/notes/OPPONENT_SEATS.md): a share of the pairs has a
    # fixed opponent in one seat, never trained on. Without them the workers are as always.
    seats = opponent_seats.from_args(args)
    if frozen_settings is not None:
        seats = dataclasses.replace(seats, frozen=frozen_settings)      # with the nets' observations and the ckpt_dir
    plan = opponent_seats.plan_seats(pairs_per_worker, rooms_per_pair, seats) if seats else None
    if plan:
        logger.info(opponent_seats.describe_plan(plan))
    workers = []
    worker_ports = []
    for i, port in enumerate(server_ports):
        count = pairs_per_worker[i]
        if count <= 0:
            continue
        infer = infers[actor_of[len(workers)]]

        if plan is None:
            worker_actor = WorkerRemote.remote(
                cfg=cfg_ref,
                inference_actor=infer,
                learner_actor=learner,
                pairs=count,
                server_port=port,
                **league_worker_kw,
            )
        else:
            entry = plan["workers"][i]
            Remote = ray.remote(num_cpus=1)(opponent_seats.worker_class(entry)).options(max_restarts=-1, max_task_retries=0)
            # The frozen seats of this worker ask the frozen actor beside its inference actor.
            fz = {"frozen_actor": frozen_actors[actor_of[len(workers)]]} if frozen_actors and opponent_seats.has_frozen(entry) else {}
            worker_actor = Remote.remote(cfg=cfg_ref, inference_actor=infer, learner_actor=learner, pairs=count,
                                         server_port=port, **opponent_seats.worker_kwargs(entry), **league_worker_kw, **fz)
        workers.append(worker_actor)
        worker_ports.append(port)

    logger.info(f"Deployment complete. Distributed {len(workers)} rollout workers.")
    nice_watch = None
    if nice is not None:
        nice_watch = NiceWatch(nice, workers, worker_ports)
        nice_watch.start()
        logger.info("CPU priorities: " + actor_nice_line(learner, infers))
    if len(infers) > 1:
        for k in range(len(infers)):
            mine = [p for p, a in zip(worker_ports, actor_of) if a == k]
            logger.info(f"InferenceActor {k} serves {len(mine)} workers, "
                        f"{sum(c for c, a in zip(worker_pairs, actor_of) if a == k)} pairs (ports {mine})")

    # 4. Staggered Startup
    # Allow core actors to initialize and load checkpoints before starting battles
    logger.info("Staggering initialization of actors...")
    await asyncio.sleep(2)
    
    # Launch asynchronous rollout tasks
    run_refs = []
    for w in workers:
        run_refs.append(w.run.remote())
        time.sleep(4) # required to prevent overloading communication channels

    # 5. Monitoring Loop
    logger.info("Training started. Entering telemetry loop.")
    started = time.time()
    try:
        watch = {} if nice_watch is None else {"nice": nice_watch}
        if league_opts.lag_stats or frozen_settings is not None:
            # The [Lag] and [Frozen] lines (league.py), every 30 s; nothing of them without the flags.
            watch["league_actors"] = frozen_actors
            watch["league_lag"] = bool(league_opts.lag_stats)
        if speed.is_default():
            await _telemetry_loop(args, started, infer, learner, workers, run_refs, **watch)
        else:
            await _telemetry_loop(args, started, infers, learner, workers, run_refs,
                                  pool=PoolWatch(actor_of, worker_ports), **watch)
    finally:
        # Checkpoints are otherwise written only every save_every_updates updates, so a stop
        # between two of them used to lose everything since the last one.
        try:
            path = ray.get(learner.save_now.remote(), timeout=180)
            logger.info(f"Saved a final checkpoint: {path}")
        except Exception as e:
            logger.warning(f"Could not save a final checkpoint: {e!r}")
        ray.shutdown()


async def _telemetry_loop(args, started, infer, learner, workers, run_refs, pool: Optional[PoolWatch] = None,
                          nice: Optional[NiceWatch] = None, league_actors: Optional[List[Any]] = None,
                          league_lag: bool = False):
    """
    Every 5 s: a crashed worker stops the run; one telemetry line. Without `pool`, `infer` is the
    one inference actor, as always. With it (a speed flag given) `infer` is the list of inference
    actors: their stats are merged into the line (infer_pool.merge_stats), and a dead one stops the
    run at once (exit code 1) instead of leaving its workers playing zeros. With `nice`
    (--rollout-nice) the heartbeats also say which workers have their nice (NiceWatch.after_beats).
    """
    beats_seen = 0
    while True:
        if args.minutes and time.time() - started > args.minutes * 60:
            logger.info(f"Reached --minutes {args.minutes}. Shutting down.")
            return
        # Ctrl+C from a parent process is unreliable on Windows, so a supervisor asks for the
        # same clean exit (and the checkpoint main() saves on the way out) by creating a file.
        if getattr(args, "stop_file", None) and os.path.exists(args.stop_file):
            logger.info(f"Stop file {args.stop_file} found. Shutting down.")
            return
        try:
            finished, _ = ray.wait(run_refs, num_returns=len(run_refs), timeout=0)
            if finished:
                for ref in finished:
                    try:
                        ray.get(ref) # This will raise the hidden exception
                    except Exception as e:
                        logger.critical(f"FATAL: A RolloutWorker crashed in run(): {e}")
                        import traceback
                        traceback.print_exception(type(e), e, e.__traceback__)
                
                # Stop the run so you don't stare at 0s
                logger.critical("Shutting down cluster due to worker crash.")
                raise SystemExit(1)
            if pool is not None:
                istats = await _pool_stats(infer, pool)
            # Gather telemetry from all actors in parallel
            if pool is None:
                istats_ref = infer.get_stats.remote()
            lstats_ref = learner.get_stats.remote()
            wstats_refs = [w.heartbeat.remote() for w in workers]

            # Ray 2.0+ pattern: Combine refs and fetch via ray.get in a thread
            if pool is None:
                results = await asyncio.to_thread(ray.get, [istats_ref] + [lstats_ref] + wstats_refs)
            
                istats = results[0]
                lstats = results[1]
                wstats_list = results[2:]
            else:
                results = await asyncio.to_thread(ray.get, [lstats_ref] + wstats_refs)
                lstats = results[0]
                wstats_list = results[1:]

            # Aggregate worker-side metrics
            total_active = sum(w.get("active_battles_worker", 0) for w in wstats_list)
            #max_q = max((w.get("learner_q_size", 0) for w in wstats_list), default=0)
            #total_traj = sum(w.get("traj_in_memory", 0) for w in wstats_list)

            # Professional telemetry print
            stats_msg = (
                f"[Telemetry] ACTIVE: {total_active} | "
                #f"LQ: {max_q} | TRAJ_BUF: {total_traj} | "
                f"INFER: {istats} | TRAIN: {lstats}"
            )
            logger.info(stats_msg)
            if nice is not None:
                nice.after_beats(wstats_list)
            # The opponent seats' results, every 30 s (only a run with seats has them in its heartbeats).
            beats_seen += 1
            seat_beats = [w.get("opponent_seats") for w in wstats_list if isinstance(w, dict) and w.get("opponent_seats")]
            if seat_beats and beats_seen % 6 == 0:
                logger.info(opponent_seats.report_line(seat_beats))
            if (league_lag or league_actors) and beats_seen % 6 == 0:
                await _league_lines(wstats_list, league_actors or [], league_lag)

        except Exception as e:
            logger.warning(f"Telemetry loop encountered an error: {e}")
            # Do not exit; the cluster might still be healthy
        
        await asyncio.sleep(5.0)

async def _league_lines(wstats_list, actors, lag: bool) -> None:
    """The [Lag] and [Frozen] lines (league.py). Never raises: it runs inside the telemetry loop."""
    try:
        if lag:
            text = league.lag_line(wstats_list)
            if text:
                logger.info(text)
        if actors:
            refs = [a.frozen_stats.remote() for a in actors]
            stats = await asyncio.to_thread(ray.get, refs, timeout=30)
            text = league.frozen_line(stats)
            if text:
                logger.info(text)
    except Exception as e:      # noqa: BLE001
        logger.warning(f"league telemetry: not this time ({e!r})")


async def _pool_stats(infers, pool: PoolWatch) -> dict:
    """The inference actors' merged stats (infer_pool.merge_stats). A dead actor: exit code 1, at once."""
    refs = [a.get_stats.remote() for a in infers]
    answers, dead, silent = await asyncio.to_thread(infer_pool.gather_stats, refs, pool.STATS_TIMEOUT_S)
    if dead:
        for k, err in dead:
            logger.critical(f"FATAL: InferenceActor {k} is dead ({err}); its workers on ports {pool.ports_of(k)} "
                            f"have no inference.")
        logger.critical("Shutting down cluster due to a dead InferenceActor.")
        raise SystemExit(1)
    if silent:
        raise RuntimeError(f"InferenceActor(s) {silent} did not answer get_stats within {pool.STATS_TIMEOUT_S:.0f} s "
                           f"(ports {[pool.ports_of(k) for k in silent]})")
    merged = infer_pool.merge_stats(answers)
    for text in pool.versions.check(merged.get("versions") or [], time.time()):
        logger.warning(text)
    return merged


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Cluster terminated.")