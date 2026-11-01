"""
Several inference actors instead of one (train.py --infer-actors N and the other speed flags).

Why. One InferenceActor serves every rollout worker of a run. On box a (bench/notes or the speed
profile of 2026-10-17) its one Python thread is 99% busy and each call costs about 10.8 ms whether
it carries 8 rows or 96, while its GPU is about 79% idle: the run makes about 5,200 decisions a
second because of that one thread. N actors, each serving its own share of the workers on its own
GPU (or a share of one), give N times that, until the learner or the box's CPUs set the pace.

What stays the same with N actors:
  * each worker talks to one actor for the whole run (assign_workers: fixed, balanced by pairs).
    Both sides of every battle are in one worker, so a battle's history (the actor's kv cache)
    is always in the actor that answers it;
  * every actor is an InferenceActor, unchanged: it pulls the learner's weights from the one
    WeightStore with the same throttle (InferenceActor._sync_weights), so each actor is at most
    one sync interval behind the store, exactly the bound the single actor has today
    (tests/test_infer_actors.py holds it for 1, 2 and 3 actors);
  * the "[Telemetry]" line's INFER dict keeps its keys: total_requests summed over the actors,
    avg_batch_size their mean, model_version the OLDEST actor's version; two keys are added,
    "actors" and "versions" (each actor's version, in actor order).

What is new, and only with these flags:
  * an actor that dies stops the run at once (train.py exits 1, the supervisor starts it again)
    instead of its workers playing zeros until a watchdog notices; with --infer-restarts R Ray
    starts it again first (up to R times), its workers' calls waiting for it meanwhile.

Nothing here imports torch or the model.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ray's resource arithmetic works in units of 1/10,000. A share is rounded DOWN to that, so N
# shares never add up to more than the GPU they were cut from (0.35/3 rounded to 0.1167 would make
# 0.3501 + the learner's 0.65 > 1, and the learner would never be placed).
_RESOURCE_UNIT = 10_000


@dataclasses.dataclass(frozen=True)
class SpeedOptions:
    """train.py's speed flags. The defaults are the run as it always was."""
    infer_actors: int = 1
    infer_gpus: Optional[int] = None
    infer_restarts: int = 0
    infer_sync_s: Optional[float] = None
    fast_sample: bool = False

    def is_default(self) -> bool:
        return self == SpeedOptions()

    def actor_kwargs(self) -> Dict[str, Any]:
        """InferenceActor's keyword arguments for these options (none at their defaults)."""
        kw: Dict[str, Any] = {}
        if self.fast_sample:
            kw["fast_sample"] = True
        if self.infer_sync_s is not None:
            kw["sync_every_s"] = float(self.infer_sync_s)
        return kw

    def line(self) -> str:
        return (f"speed: infer_actors={self.infer_actors} infer_gpus={self.infer_gpus if self.infer_gpus is not None else 'auto'} "
                f"infer_restarts={self.infer_restarts} infer_sync_s={self.infer_sync_s if self.infer_sync_s is not None else 5.0} "
                f"fast_sample={self.fast_sample}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("speed (infer_pool.py; all off by default, and off the run is what it always was)")
    g.add_argument("--infer-actors", type=int, default=None,
                   help="inference actors (1). Each rollout worker is served by one of them for the whole run, "
                        "the workers spread evenly by pairs. With PSPPO_GPU_SPLIT=1 each gets a GPU of its own "
                        "while there are GPUs besides the learner's (see --infer-gpus)")
    g.add_argument("--infer-gpus", type=int, default=None,
                   help="how many GPUs the inference actors spread over (PSPPO_GPU_SPLIT=1 only; the learner keeps a "
                        "GPU of its own). Default: one per actor, as far as the GPUs besides the learner's go; "
                        "more actors than that share them evenly. Ray picks which physical GPUs")
    g.add_argument("--infer-restarts", type=int, default=None,
                   help="times Ray starts a dead inference actor again (0: none, the run stops at once instead, and "
                        "the supervisor starts it again). Its workers' calls wait while it starts")
    g.add_argument("--infer-sync-s", type=float, default=None,
                   help="seconds an inference actor waits at most between two looks at the weight store (5). Below "
                        "the time between two updates, every actor loads every update")
    g.add_argument("--fast-sample", action=argparse.BooleanOptionalAction, default=None,
                   help="draw the action with the Gumbel-max trick in fewer kernels and with no GPU waits "
                        "(inference.fast_masked_sample): same distribution and log-probabilities, other random draws")


def from_args(args: Any) -> SpeedOptions:
    """The options from add_arguments()'s flags. Raises ValueError on a value that cannot be."""
    def given(name: str):
        return getattr(args, name, None)

    n = 1 if given("infer_actors") is None else int(args.infer_actors)
    if n < 1:
        raise ValueError(f"--infer-actors {n} must be at least 1")
    k = given("infer_gpus")
    if k is not None and int(k) < 1:
        raise ValueError(f"--infer-gpus {k} must be at least 1")
    r = 0 if given("infer_restarts") is None else int(args.infer_restarts)
    if r < 0:
        raise ValueError(f"--infer-restarts {r} must not be negative")
    s = given("infer_sync_s")
    if s is not None and not (float(s) > 0.0 and math.isfinite(float(s))):
        raise ValueError(f"--infer-sync-s {s} must be above zero")
    return SpeedOptions(infer_actors=n, infer_gpus=None if k is None else int(k), infer_restarts=r,
                        infer_sync_s=None if s is None else float(s), fast_sample=bool(given("fast_sample") or False))


def _floor_share(x: float) -> float:
    return math.floor(x * _RESOURCE_UNIT + 1e-9) / _RESOURCE_UNIT


def plan_gpus(n_actors: int, gpus: float, infer_gpus: Optional[int] = None,
              split: Optional[bool] = None) -> Tuple[List[float], float]:
    """
    (the GPU share Ray reserves for each inference actor, the learner's share).

    One actor and no --infer-gpus: train.gpu_shares, as always (0.35 and 0.65 on one GPU, or a GPU
    each with PSPPO_GPU_SPLIT=1 and two or more GPUs).

    PSPPO_GPU_SPLIT=1 and G >= 2 GPUs: the learner has a whole GPU; the actors spread over
    K = --infer-gpus (default min(N, G - 1)) of the other G - 1, ceil(N / K) to a GPU, each asking
    1 / ceil(N / K) of one. Equal shares that divide a GPU exactly always fit, wherever Ray puts them.

    Otherwise the actors share the learner's GPU: the 0.35 one actor had, cut evenly.
    """
    if split is None:
        split = os.environ.get("PSPPO_GPU_SPLIT", "0") == "1"
    n = int(n_actors)
    if n < 1:
        raise ValueError(f"{n} inference actors")
    g = int(gpus)
    if n == 1 and infer_gpus is None:
        if split and g >= 2:
            return [1.0], 1.0
        return [0.35], 0.65
    if split and g >= 2:
        avail = g - 1
        k = min(n, avail) if infer_gpus is None else int(infer_gpus)
        if k < 1 or k > avail:
            raise ValueError(f"--infer-gpus {k}: there are {avail} GPU(s) besides the learner's ({g} in all)")
        k = min(k, n)
        per_gpu = -(-n // k)
        share = 1.0 if per_gpu == 1 else _floor_share(1.0 / per_gpu)
        return [share] * n, 1.0
    if infer_gpus is not None:
        raise ValueError(f"--infer-gpus needs PSPPO_GPU_SPLIT=1 and two or more GPUs (this machine: {g}, "
                         f"PSPPO_GPU_SPLIT={'1' if split else 'unset or 0'})")
    return [_floor_share(0.35 / n)] * n, 0.65


def assign_workers(pairs: Sequence[int], n_actors: int) -> List[int]:
    """
    The actor of each worker, in the workers' order: each goes to the actor with the fewest pairs so
    far (the lowest index of equals). Equal workers go round the actors in turn. Nothing in it is
    random, so a worker Ray starts again keeps its actor (it is handed the same handle anyway).
    """
    n = int(n_actors)
    if n < 1:
        raise ValueError(f"{n} inference actors")
    load = [0] * n
    out: List[int] = []
    for p in pairs:
        k = min(range(n), key=lambda j: (load[j], j))
        out.append(k)
        load[k] += max(0, int(p))
    return out


def actor_options(share: float, restarts: int) -> Dict[str, Any]:
    """ray.remote options of one inference actor."""
    opts: Dict[str, Any] = {"num_gpus": float(share)}
    if restarts > 0:
        # A call in flight or sent while the actor starts again waits for it (it is retried on the
        # new process) instead of failing over to the worker's zeros.
        opts.update(max_restarts=int(restarts), max_task_retries=-1)
    return opts


def merge_stats(stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The INFER dict of the telemetry line from each actor's get_stats() (one actor: its own, as it was)."""
    stats = list(stats)
    if len(stats) == 1:
        return stats[0]
    versions = [int(s.get("model_version", -1)) for s in stats]
    batches = [float(s.get("avg_batch_size", 0.0) or 0.0) for s in stats]
    return {
        "total_requests": sum(int(s.get("total_requests", 0) or 0) for s in stats),
        "avg_batch_size": round(sum(batches) / len(batches), 2),
        "model_version": min(versions),
        "actors": len(stats),
        "versions": versions,
    }


class VersionWatch:
    """
    Warns (does not stop) when an actor stays two or more versions behind the newest actor for
    longer than `after_s`: with the store's pull that means its pulls are failing (each failure
    is swallowed by InferenceActor._sync_weights, as it always was).
    """

    def __init__(self, after_s: float = 60.0):
        self.after_s = float(after_s)
        self._behind_since: Dict[int, float] = {}
        self._warned: Dict[int, bool] = {}

    def check(self, versions: Sequence[int], now: float) -> List[str]:
        out: List[str] = []
        if not versions:
            return out
        newest = max(versions)
        for k, v in enumerate(versions):
            if newest - v >= 2:
                since = self._behind_since.setdefault(k, now)
                if now - since > self.after_s and not self._warned.get(k):
                    self._warned[k] = True
                    out.append(f"InferenceActor {k} has been {newest - v} versions behind the newest actor ({v} against {newest}) "
                               f"for {now - since:.0f} s: its pulls from the weight store are failing")
            else:
                self._behind_since.pop(k, None)
                self._warned.pop(k, None)
        return out


def gather_stats(refs: Sequence[Any], timeout: Optional[float], ray_module: Any = None) -> Tuple[List[Optional[dict]], List[Tuple[int, str]], List[int]]:
    """
    Each actor's get_stats() answer, waiting at most `timeout` seconds for all of them.
    -> (answers, None where there is none; [(index, error)] of the actors that are DEAD (Ray will
    not start them again); indices of those that did not answer in time).

    An actor that Ray is starting again (--infer-restarts) is not dead: its call waits for it.
    """
    ray_ = ray_module
    if ray_ is None:
        import ray as ray_
    from ray.exceptions import ActorDiedError, ActorUnavailableError, RayActorError

    refs = list(refs)
    ready, _ = ray_.wait(refs, num_returns=len(refs), timeout=timeout)
    ready_set = set(ready)
    answers: List[Optional[dict]] = []
    dead: List[Tuple[int, str]] = []
    silent: List[int] = []
    for k, ref in enumerate(refs):
        if ref not in ready_set:
            answers.append(None)
            silent.append(k)
            continue
        try:
            answers.append(ray_.get(ref))
        except ActorUnavailableError:
            answers.append(None)
            silent.append(k)
        except (ActorDiedError, RayActorError) as e:
            answers.append(None)
            dead.append((k, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"))
        except Exception:                       # get_stats itself raised: the actor is alive
            answers.append(None)
            silent.append(k)
    return answers, dead, silent
