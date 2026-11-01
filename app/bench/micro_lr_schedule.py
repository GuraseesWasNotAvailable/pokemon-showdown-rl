#!/usr/bin/env python
"""
What the learning-rate schedule really does over a long run.

It builds the scheduler with LearnerActor's own code (learner.py::_init_scheduler) from
config.py, reads the multiplier at chosen optimizer steps, and converts optimizer steps to
decisions for a given update shape. No GPU, no Ray, no servers.

    python bench/micro_lr_schedule.py
    python bench/micro_lr_schedule.py --steps-per-update 16384 --minibatch 256 --grad-accum 3
    python bench/micro_lr_schedule.py --total-steps 1000000        # the same with lr_total_steps changed
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def optimizer_steps_per_update(n_steps: int, minibatch: int, epochs: int, grad_accum: int) -> int:
    """
    As ppo_core.ppo_update counts them: every epoch runs floor(n / minibatch) minibatches, steps
    the optimizer after every grad_accum of them, and once more for a remainder. (A PPO update
    that stops early on KL takes fewer.)
    """
    mb = n_steps // minibatch
    per_epoch = mb // grad_accum + (1 if mb % grad_accum else 0)
    return per_epoch * epochs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps-per-update", type=int, default=None)
    ap.add_argument("--minibatch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overshoot", type=int, default=400,
                    help="decisions an update holds beyond steps_per_update on average (the last submitted batch runs over)")
    ap.add_argument("--run-decisions", type=float, default=250e6)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--hold-steps", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None, help="try another lr_total_steps")
    args = ap.parse_args()
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)

    import torch
    from config import RunConfig
    from learner import LearnerActor

    cfg = RunConfig.default()
    over = {}
    if args.warmup_steps is not None:
        over["lr_warmup_steps"] = args.warmup_steps
    if args.hold_steps is not None:
        over["lr_hold_steps"] = args.hold_steps
    if args.total_steps is not None:
        over["lr_total_steps"] = args.total_steps
    lc = dataclasses.replace(cfg.learner, mode="ppo", **over)
    spu = args.steps_per_update or lc.steps_per_update
    mb = args.minibatch or lc.minibatch_size
    ga = args.grad_accum or lc.grad_accum_steps
    ep = args.epochs or lc.update_epochs

    la = object.__new__(LearnerActor)
    la.cfg = lc
    la.opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lc.lr)
    la.sched = None
    la._init_scheduler()
    lam = la.sched.lr_lambdas[0]

    w, h, t = lc.lr_warmup_steps, lc.lr_hold_steps, lc.lr_total_steps
    n = spu + args.overshoot
    ops = optimizer_steps_per_update(n, mb, ep, ga)
    dec_per_step = n / ops
    probe = sorted({0, 1, w - 1, w, w + h - 1, w + h, w + h + 1, w + h + 2, w + h + 1000, t, 2 * (w + h),
                    int(args.run_decisions / dec_per_step)})
    rows = []
    for s in probe:
        if s < 0:
            continue
        rows.append({"optimizer_step": s, "lr_multiplier": round(lam(s), 6), "base_lr_times": round(lc.lr * lam(s), 9),
                     "decisions": round(s * dec_per_step)})
    # the largest single-step fall anywhere in the first two million steps
    worst, at = 1.0, None
    prev = lam(0)
    for s in range(1, 2_000_001):
        cur = lam(s)
        if prev > 0 and cur / prev < worst:
            worst, at = cur / prev, s
        prev = cur
    end_step = int(args.run_decisions / dec_per_step)
    res = {
        "config": {"lr": lc.lr, "lr_warmup_steps": w, "lr_hold_steps": h, "lr_total_steps": t,
                   "anneal_span_used_by_the_code": max(1, t - (w + h))},
        "update_shape": {"steps_per_update": spu, "mean_decisions_in_an_update": n, "minibatch": mb, "grad_accum": ga, "epochs": ep,
                         "optimizer_steps_per_update": ops, "decisions_per_optimizer_step": round(dec_per_step, 1)},
        "multiplier_at": rows,
        "largest_single_step_fall": {"factor": round(1.0 / worst, 3) if worst > 0 else None, "at_optimizer_step": at,
                                     "at_decisions": round(at * dec_per_step) if at else None},
        "run": {"decisions": args.run_decisions, "optimizer_steps": end_step,
                "multiplier_at_the_end": round(lam(end_step), 6),
                "share_of_the_run_after_the_hold_ends": round(max(0.0, 1.0 - (w + h) * dec_per_step / args.run_decisions), 4)},
    }
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
