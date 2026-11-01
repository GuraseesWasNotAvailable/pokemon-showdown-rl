#!/usr/bin/env python
"""
The learning-rate flags of a new stage, computed from the checkpoint the stage resumes from.

    ~/venv/bin/python ~/ps-ppo/cloud/stage_lr.py --checkpoint ~/runs/main/checkpoints/learner_update_NNNNNN.pt \
        --start-rate 6e-5 --end-decisions 900000000
    ... --start-rate 6e-5 --hold-decisions 100000000 --end-decisions 900000000      (hold, then decline)
    ... --start-rate 3e-5 --hold-decisions 100000000 --end-rate 1.11e-5             (hold, then decline to a rate)

Reads the checkpoint only (optimizer step, decisions, the run's minibatch and epochs). Prints the
three flags (--lr, --lr-hold-steps, --lr-total-steps) and the policy rate the learner will use
along the stage, computed with check_restart.rate_share (learner.py's lr_lambda to the letter).

Why. The learner puts the rate where the schedule of the NEW start has it for the checkpoint's
optimizer step (there is no warm-up at a resume), so a stage's rate is set by placing the
schedule around that step. Two shapes:

  decline (no --hold-decisions): --lr stays (default 3e-4, the run's), and the anneal is placed so
      that the checkpoint's step sits where the curve gives --start-rate, and the curve reaches
      1/27 of --lr (the floor the run has now, 1.11e-5 for 3e-4) at --end-decisions. The rate
      falls from --start-rate to the floor, fastest at first (lr_lambda's 1/(8p+1)^1.5).
  hold (--hold-decisions H): --lr becomes --start-rate, held for H decisions, then annealed to
      1/27 of it at --end-decisions. With --end-rate R the anneal is stretched instead so that the
      rate is R at --end-decisions (between 1/27 of --start-rate and --start-rate); it goes on
      falling slowly after that, to 1/27 of --start-rate at --lr-total-steps, which then lies
      past --end-decisions (a target raised later with extend_target.sh keeps the same curve).

Decisions per optimizer step: minibatch x grad_accum / update_epochs from the checkpoint's run
config (768 / 2 = 384 for the runs so far; stage3 measured 384.5 between updates 16,175 and 16,375). An
update that stops early on target_kl takes fewer steps, so the schedule then ends a little after
--end-decisions. The warm-up of config.py (1,000 steps from step 0) is long past and stays.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from check_restart import rate_share  # noqa: E402

FLOOR = 1.0 / 27.0


def plan(step: int, decisions: int, end_decisions: int, start_rate: float, lr: float = 3e-4,
         hold_decisions: int = 0, warmup: int = 1000, per_step: float = 384.0, end_rate=None) -> dict:
    """The flags for a stage that resumes at optimizer `step` (`decisions` so far). Raises ValueError
    for a stage that cannot be had."""
    if end_decisions <= decisions:
        raise ValueError(f"--end-decisions {end_decisions:,} is not after the checkpoint's {decisions:,} decisions")
    if step <= warmup:
        raise ValueError(f"the checkpoint's optimizer step {step:,} is inside the warm-up ({warmup:,})")
    end_step = step + int(round((end_decisions - decisions) / per_step))
    total = end_step
    if hold_decisions:
        if hold_decisions >= end_decisions - decisions:
            raise ValueError("the hold is longer than the stage")
        lr = float(start_rate)
        anneal_start = step + int(round(hold_decisions / per_step))
        if end_rate is not None:
            share_end = float(end_rate) / lr
            if not (FLOOR < share_end < 1.0):
                raise ValueError(f"--end-rate {end_rate:g} is not between 1/27 of --start-rate ({lr * FLOOR:.3g}) and --start-rate {lr:g}")
            p_end = (share_end ** (-2.0 / 3.0) - 1.0) / 8.0      # rate_share(progress p_end) = share_end
            total = anneal_start + int(round((end_step - anneal_start) / p_end))
    else:
        if end_rate is not None:
            raise ValueError("--end-rate goes with --hold-decisions; the decline shape ends at 1/27 of --lr")
        share = float(start_rate) / float(lr)
        if not (FLOOR < share <= 1.0):
            raise ValueError(f"--start-rate {start_rate:g} is not between the floor {lr * FLOOR:.3g} and --lr {lr:g}")
        p0 = (share ** (-2.0 / 3.0) - 1.0) / 8.0          # rate_share(progress p) = share
        anneal_start = int(round((step - p0 * total) / (1.0 - p0)))
    hold = anneal_start - warmup
    if hold < 0 or total <= anneal_start:
        raise ValueError(f"no schedule: hold {hold:,}, total {total:,}, anneal start {anneal_start:,}"
                         + ("" if hold_decisions else ": a decline that starts this close to the floor would have to "
                            "begin before step 0; give a higher --start-rate or a --hold-decisions"))
    return {"lr": lr, "lr_hold_steps": hold, "lr_total_steps": total, "warmup": warmup, "step": step,
            "decisions": decisions, "end_decisions": end_decisions, "per_step": per_step, "end_step": end_step,
            "rate_at_resume": lr * rate_share(step, warmup, hold, total),
            "rate_at_end": lr * rate_share(end_step, warmup, hold, total)}


def path(p: dict, points=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0)) -> list:
    """(decisions, optimizer step, policy rate) along the stage."""
    out = []
    for f in points:
        d = p["decisions"] + f * (p["end_decisions"] - p["decisions"])
        s = p["step"] + int(round((d - p["decisions"]) / p["per_step"]))
        out.append((d, s, p["lr"] * rate_share(s, p["warmup"], p["lr_hold_steps"], p["lr_total_steps"])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="the checkpoint the stage resumes from (opened for reading only)")
    ap.add_argument("--start-rate", type=float, required=True, help="the policy rate at the resume (the update line's lr=)")
    ap.add_argument("--end-decisions", type=float, default=900e6, help="decisions at which the rate reaches its floor")
    ap.add_argument("--hold-decisions", type=float, default=0, help="hold --start-rate this many decisions, then decline")
    ap.add_argument("--end-rate", type=float, default=None, help="with --hold-decisions: the rate at --end-decisions (default 1/27 of --start-rate)")
    ap.add_argument("--lr", type=float, default=3e-4, help="the base rate in the decline shape (the run's --lr)")
    args = ap.parse_args(argv)
    import torch
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lc = (ck.get("run_cfg") or {}).get("learner") or {}
    per_step = float(lc.get("minibatch_size", 768)) * float(lc.get("grad_accum_steps", 1) or 1) / float(lc.get("update_epochs", 2))
    p = plan(int(ck["scheduler"]["last_epoch"]), int(ck["total_steps"]), int(args.end_decisions), args.start_rate,
             args.lr, int(args.hold_decisions), int(lc.get("lr_warmup_steps", 1000)), per_step, args.end_rate)
    print(f"checkpoint {os.path.basename(args.checkpoint)}: optimizer step {p['step']:,}, {p['decisions']:,} decisions, "
          f"{per_step:g} decisions per optimizer step")
    print(f"flags: --lr {p['lr']:g} --lr-hold-steps {p['lr_hold_steps']} --lr-total-steps {p['lr_total_steps']}")
    for d, s, r in path(p):
        print(f"  {d / 1e6:7.1f}M decisions  step {s:>9,}  policy rate {r:.3e}")
    if p["lr_total_steps"] > p["end_step"]:
        after = p["decisions"] + (p["lr_total_steps"] - p["step"]) * per_step
        print(f"  after that the rate goes on falling, to {p['lr'] * FLOOR:.3e} at step {p['lr_total_steps']:,} (about {after / 1e6:,.0f}M decisions)")
    print("Check the start with cloud/check_restart.py ... --allow-rate-change (the rate moves at the resume on purpose).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
