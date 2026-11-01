"""
The league run's train.py flags that are not opponent seats (bench/notes/LEAGUE.md). All off by default;
off, train.py builds every actor exactly as before (no keyword reaches a worker or the learner).

    --learn-queue-episodes N   each rollout worker holds at most N finished episodes for the learner and
                               drops the oldest beyond them (worker.SyncLearnerClient). Default: 160, the
                               5 x 32 of RolloutConfig, as always. The learner takes them oldest first, so
                               while it is the slower side its data is about this queue's length old; a
                               shorter queue trains on fresher data and drops more of the old.
    --lag-stats                measure that: the learner line ends with kl0 (the KL of the update's first
                               minibatch before any step: the data's own policy against the one the update
                               starts from), age_s / age_p50 / age_p90 (seconds from the decision to the
                               update) and lag_mean / lag_p50 / lag_p90 (updates the policy that chose it
                               is behind), and every 30 s a "[Lag]" line with the workers' queue counts
                               (episodes handed in and dropped). Costs one time stamp per decision and a
                               column the learner drops before the update.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class LeagueOptions:
    learn_queue_episodes: Optional[int] = None
    lag_stats: bool = False

    def is_default(self) -> bool:
        return self == LeagueOptions()


def add_arguments(ap: Any) -> None:
    g = ap.add_argument_group("league: fresher data and its measurement (league.py; off by default)")
    g.add_argument("--learn-queue-episodes", type=int, default=None, metavar="N",
                   help="episodes each rollout worker holds for the learner before it drops the oldest (default 160 = 5 x 32)")
    g.add_argument("--lag-stats", action="store_true",
                   help="the update line ends with kl0, age_s/age_p50/age_p90, lag_mean/lag_p50/lag_p90; a [Lag] line every 30 s")


def from_args(args: Any) -> LeagueOptions:
    """The options from the flags; ValueError for a queue under 1."""
    n = getattr(args, "learn_queue_episodes", None)
    if n is not None and int(n) < 1:
        raise ValueError(f"--learn-queue-episodes {n} must be at least 1")
    return LeagueOptions(learn_queue_episodes=None if n is None else int(n), lag_stats=bool(getattr(args, "lag_stats", False)))


def worker_kwargs(opts: LeagueOptions) -> Dict[str, Any]:
    """RolloutWorker's keywords for the options that are on ({} at the defaults)."""
    out: Dict[str, Any] = {}
    if opts.learn_queue_episodes is not None:
        out["learn_queue"] = int(opts.learn_queue_episodes)
    if opts.lag_stats:
        out["lag_stats"] = True
    return out


def learner_kwargs(opts: LeagueOptions) -> Dict[str, Any]:
    return {"lag_stats": True} if opts.lag_stats else {}


def line(opts: LeagueOptions) -> str:
    return (f"league: learn_queue_episodes={opts.learn_queue_episodes if opts.learn_queue_episodes is not None else 'default (160)'} "
            f"lag_stats={opts.lag_stats}")


def lag_line(beats: Sequence[Any]) -> Optional[str]:
    """'[Lag] worker queues: 1,234 of 56,789 episodes dropped (2.2%), 31 held now (max 32 each, 36 workers)', or None without lag beats."""
    lags = [b.get("lag") for b in beats if isinstance(b, dict) and isinstance(b.get("lag"), dict)]
    if not lags:
        return None
    sub = sum(int(x.get("submitted", 0)) for x in lags)
    ev = sum(int(x.get("evicted", 0)) for x in lags)
    held = sum(int(x.get("q", 0)) for x in lags)
    sizes = sorted({int(x.get("maxsize", 0)) for x in lags})
    share = 100.0 * ev / sub if sub else 0.0
    return (f"[Lag] worker queues: {ev:,} of {sub:,} episodes dropped as the oldest ({share:.1f}%), {held:,} held now "
            f"(max {'/'.join(str(s) for s in sizes)} each, {len(lags)} workers)")


def frozen_line(stats: Sequence[Any]) -> Optional[str]:
    """'[Frozen] 12,345 rows in 2,345 calls; battles m24381 40, sfin 30, past0 10 ...; 5 nets, 1 refresh; past [...]'."""
    stats = [s for s in stats if isinstance(s, dict)]
    if not stats:
        return None
    rows = sum(int(s.get("rows", 0)) for s in stats)
    calls = sum(int(s.get("calls", 0)) for s in stats)
    by: Dict[str, int] = {}
    for s in stats:
        for k, v in (s.get("battles_by_slot") or {}).items():
            by[k] = by.get(k, 0) + int(v)
    nets = sum(int(s.get("nets", 0)) for s in stats)
    forwards = sum(int(s.get("forwards", 0)) for s in stats)
    busy = sum(float(s.get("busy_s", 0.0)) for s in stats)
    refreshes = max(int(s.get("refreshes", 0)) for s in stats)
    errors = sum(int(s.get("refresh_errors", 0)) for s in stats)
    past = stats[0].get("past") or []
    changed = sorted({c for s in stats for c in (s.get("changed") or [])})
    text = (f"[Frozen] {rows:,} decisions in {calls:,} calls ({forwards:,} forwards, {busy:.0f} s of the actors' time, "
            f"{1000.0 * busy / max(1, calls):.1f} ms a call); battle sides started by net: "
            + ", ".join(f"{k} {v:,}" for k, v in by.items())
            + f"; {nets} nets held in {len(stats)} actors, {refreshes} picks of the past saves"
            + (f" ({errors} FAILED)" if errors else "") + (f"; past slots now {past}" if past else "")
            + (f"; FROZEN NETS CHANGED: {changed}" if changed else "; frozen weights unchanged"))
    return text
