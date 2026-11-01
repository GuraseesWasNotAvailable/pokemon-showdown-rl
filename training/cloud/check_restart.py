#!/usr/bin/env python
"""
Before a start of the supervisor with an edited run.json: what would the trainer be started with?

    ~/venv/bin/python <release>/cloud/check_restart.py --config ~/runs/<run>/run.json --run-dir ~/runs/<run>
    (<release>: the config's repo_dir, the release whose train.py the trainer will run)
    ... --expect-args "--lr 3e-05 --lr-hold-steps 1361178 --lr-total-steps 1622000"
    ... --checkpoint <the checkpoint the stage resumes from> --allow-rate-change   (a learning-rate restart)

Reads only: the config, state.json and the newest checkpoint (or --checkpoint). Starts nothing,
writes nothing (no Ray, no server, no trainer). Run it between the edit and the start.

Ported from stage3 (cloud/check_restart.py there) to this code line, which has the learning-rate
flags and the own Tera type and none of stage3's other flags.

Why it exists. The supervisor takes any string in extra_train_args; train.py refuses one it does
not know with exit code 2, and on the box that is a trainer that never starts (after five such
starts the run is marked failed and nothing runs). And the learning-rate flags move the rate at the
resume: the learner puts every rate where the schedule OF THE NEW START has it for the checkpoint's
optimizer step, so an --lr-total-steps added after the hold has ended raises the rate again, up to
27 times. A learning-rate restart (cloud/stage_lr.py) moves it on purpose: --allow-rate-change.

What it does, in this order, and it stops at the first that fails (exit code 2):
  1. the config through the supervisor's own validation (pipeline.load_config);
  2. the trainer's command line for the phase the run is in (pipeline.build_trainer_command), and
     that the train.py it runs (in the config's repo_dir) is this check's own: run the check from
     the release the config names, or it is exit code 2;
  3. that command line through train.py's own parser and config builder (a misspelt or
     impossible argument is refused here, as it would be at the start); with
     --human-imitation-coef, the data directory: at least one shard built as this run's
     observation and with train sides, or exit code 2 (the trainer would start WITHOUT the term);
  4. the settings line the trainer will log ("ppo settings: ...", train.ppo_settings_line),
     watchdog.recycle_hours and trainer.env, and with --expect-args that exactly these are the
     extra arguments;
  5. the checkpoint: its observation settings (obs_settings: here the own Tera type, which this
     code line keeps in the checkpoint's run_cfg) against this command line's. A difference is exit
     code 2 unless --allow-obs-change says it is wanted: the learner only logs it and goes on.
     Then its optimizer step, and every learning rate at the resume with this command line against
     the rate the same checkpoint has with the settings it was written under. A rate that moves is
     exit code 3 (--allow-rate-change says it is wanted).

The schedule in decisions (train.py --lr-hold-decisions / --lr-total-decisions, LearnerConfig
lr_hold_decisions / lr_total_decisions): the rate is a function of the phase's decisions (the
checkpoint's total_steps) and of the optimizer step only for the warmup ramp. Each side's rate is
computed under its own kind of schedule (the checkpoint's from its run_cfg.learner, the start's
from the command line), and where the hold and the end of the anneal lie against the checkpoint's
decisions is printed. At a resume in the same phase, a change of the kind of schedule (steps to
decisions or back) or of a decisions schedule's hold or end is exit code 3 like a rate that
moves, even when the rate at the resume happens to be the same: the rest of the run would follow
another schedule. --allow-rate-change says it is wanted.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def rate_share(step: int, warmup: int, hold: int, total: int) -> float:
    """The share of the full learning rate at an optimizer step: learner.py's lr_lambda, to the letter."""
    if step < warmup:
        return float(step + 1) / float(warmup)
    if step < (warmup + hold):
        return 1.0
    anneal_start = warmup + hold
    progress = max(0.0, min(1.0, (step - anneal_start) / max(1, total - anneal_start)))
    return 1.0 / ((8 * progress + 1) ** 1.5)


def rate_share_decisions(step: int, decisions: int, warmup: int, hold: int, total: int) -> float:
    """
    The share of the full learning rate under the schedule in decisions, at an optimizer step and
    a count of the phase's decisions: learner.py's lr_lambda_decisions with decision_rate_share,
    to the letter (the ramp over the first `warmup` optimizer steps, then 1 below `hold` decisions,
    then 1/(8p+1)^1.5 down to 1/27 at `total`).
    """
    warm = float(step + 1) / float(warmup) if step < warmup else 1.0
    if decisions < hold:
        share = 1.0
    else:
        progress = max(0.0, min(1.0, (decisions - hold) / max(1, total - hold)))
        share = 1.0 / ((8 * progress + 1) ** 1.5)
    return warm * share


def decisions_schedule(learner: dict):
    """(hold, total) of a learner config's schedule in decisions, or None when it schedules in optimizer steps."""
    total = int((learner or {}).get("lr_total_decisions") or 0)
    return (int((learner or {}).get("lr_hold_decisions") or 0), total) if total > 0 else None


def settings_line(cfg) -> str:
    """The line train.py logs at every start (train.main), from the same function."""
    import train
    return train.ppo_settings_line(cfg)


def checkpoint_obs_settings(ck: dict) -> dict:
    """The observation settings a learner checkpoint was trained with ({name: True/False})."""
    from config import checkpoint_own_tera_type
    from obs_matchup import checkpoint_obs_matchup
    from obs_movesets import checkpoint_obs_movesets
    out = {"own_tera_type": bool(checkpoint_own_tera_type(ck)), "matchup": bool(checkpoint_obs_matchup(ck))}
    if checkpoint_obs_movesets(ck):
        out["movesets"] = True          # named only when on, so the settings of every other checkpoint read as before
    from obs_main import checkpoint_main_line
    if checkpoint_main_line(ck):
        out["main_line"] = True         # the main line's observation (obs_main.py), named only when on, likewise
    return out


def read_checkpoint(path: str) -> dict:
    """The few numbers of a checkpoint this check needs. The file is opened for reading only."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    learner = (ck.get("run_cfg") or {}).get("learner") or {}
    return {"update": int(ck.get("update_idx", -1)), "total_steps": int(ck.get("total_steps", -1)),
            "optimizer_step": int((ck.get("scheduler") or {}).get("last_epoch", -1)),
            "mode": learner.get("mode"), "lr": learner.get("lr"),
            "warmup": learner.get("lr_warmup_steps"), "hold": learner.get("lr_hold_steps"), "total": learner.get("lr_total_steps"),
            "hold_decisions": learner.get("lr_hold_decisions"), "total_decisions": learner.get("lr_total_decisions"),
            "obs_settings": checkpoint_obs_settings(ck)}


def obs_settings_of_config(run_cfg) -> dict:
    """The obs_settings a learner built from this RunConfig writes into its checkpoints."""
    out = {"own_tera_type": bool(run_cfg.obs.own_tera_type), "matchup": bool(getattr(run_cfg.obs, "matchup", False))}
    if getattr(run_cfg.obs, "movesets", False):
        out["movesets"] = True
    if getattr(run_cfg.obs, "main_line", False):
        out["main_line"] = True
    return out


def obs_change(saved: dict, want: dict):
    """(added, dropped): the observation settings a resume switches on and off. A key is on when its value is True."""
    have = {k for k, v in (saved or {}).items() if v is True}
    now = {k for k, v in (want or {}).items() if v is True}
    return sorted(now - have), sorted(have - now)


def run_obs_dim(run_cfg):
    """The width of the observation this run's workers build (the network's input), or None if it cannot be had here."""
    here = os.getcwd()
    try:
        os.chdir(REPO)              # the assembler reads vocab.json from the working directory
        return int(run_cfg.make_assembler().total_dim)
    except Exception:      # noqa: BLE001
        return None
    finally:
        os.chdir(here)


def human_data_summary(data: str, run_cfg) -> dict:
    """
    What the learner would find in a human data directory for this run (human_imitation.py's own
    rules, the shards' small arrays only: no observation is read): shards, those of this run's
    observation, and in those the train and held-out sides and their labelled decisions.
    """
    import numpy as np
    import human_imitation as hi
    want = hi.run_obs_settings(run_cfg.obs)
    dim = run_obs_dim(run_cfg)
    out = {"files": 0, "fit": 0, "no_ep_split": 0, "train_sides": 0, "train_decisions": 0, "holdout_sides": 0,
           "holdout_decisions": 0, "unreadable": 0, "want": want, "dim": dim, "built": set()}
    for f in hi.shard_files(data):
        out["files"] += 1
        try:
            with np.load(f, allow_pickle=False) as z:
                out["built"].add(hi.describe_obs(hi.shard_obs_settings(z)) + f" obs_dim={hi.shard_obs_dim(z)}")
                if not hi.shard_fits(z, want, dim):
                    continue
                out["fit"] += 1
                if "ep_split" not in z.files:
                    out["no_ep_split"] += 1
                    continue
                st = z["ep_start"]
                lab = z["train"].astype(bool) & (z["act"] >= 0)
                for e, sp in enumerate(z["ep_split"]):
                    role = hi.SPLIT_ROLES.get(str(sp))
                    n = int(lab[int(st[e]):int(st[e + 1])].sum())
                    if role == "train":
                        out["train_sides"] += 1
                        out["train_decisions"] += n
                    elif role == "test_player":
                        out["holdout_sides"] += 1
                        out["holdout_decisions"] += n
        except Exception:      # noqa: BLE001  (the learner leaves such a shard out too)
            out["unreadable"] += 1
    return out


def check_human_imitation(train, args, run_cfg, out) -> int:
    """Step 3b: the top players' imitation term, when the command line asks for it. 0, or 2 (do not start)."""
    try:
        hum = train.human_imitation_settings(args, run_cfg)
    except ValueError as e:          # train.parse_args refused it already; here for a hand-built Namespace
        out(f"DO NOT START: --human-imitation-*: {e}")
        return 2
    if hum is None:
        out("human imitation: off")
        return 0
    out(train.human_imitation_line(hum))
    data = os.path.expanduser(hum["data"])
    if not os.path.isdir(data):
        out(f"DO NOT START as written: --human-imitation-data {hum['data']} is not a directory. The trainer WOULD start, "
            f"without the term.")
        return 2
    try:
        s = human_data_summary(data, run_cfg)
    except Exception as e:      # noqa: BLE001
        out(f"DO NOT START as written: the human data could not be read: {e!r}")
        return 2
    import human_imitation as hi
    if not s["fit"]:
        out(f"DO NOT START as written: none of the {s['files']} shards in {hum['data']} was built as this run's observation "
            f"({hi.describe_obs(s['want'])}, obs_dim={s['dim']}; the shards say {sorted(s['built']) or 'nothing'}); the "
            f"learner would load none. Rebuild them: human/rebuild.py build --own-tera-type {'on' if s['want'][0] else 'off'}"
            f"{' --obs-matchup' if s['want'][1] else ''}{' --obs-movesets' if s['want'][2] else ''} ...")
        return 2
    if not s["train_sides"] and s["no_ep_split"] == 0:
        out(f"DO NOT START as written: the {s['fit']} shard(s) of this run's observation in {hum['data']} hold no train "
            f"side (ep_split 'train'); the learner would have nothing to imitate.")
        return 2
    out(f"human data:      {s['fit']} of {s['files']} shards fit this run's observation; {s['train_sides']:,} train sides, "
        f"{s['train_decisions']:,} labelled train decisions; {s['holdout_sides']:,} held-out players' sides, "
        f"{s['holdout_decisions']:,} decisions"
        + (f"; {s['no_ep_split']} shard(s) without ep_split use the split files" if s["no_ep_split"] else "")
        + (f"; {s['unreadable']} unreadable" if s["unreadable"] else ""))
    if run_cfg.learner.mode not in ("ppo", "ppo_with_jepa"):
        out(f"human imitation: the start's mode is {run_cfg.learner.mode}: the term is used only in a PPO phase")
    return 0


def check(config_path: str, run_dir: str, expect_args=None, allow_rate_change: bool = False, checkpoint=None, out=print,
          allow_obs_change: bool = False, phase_override=None) -> int:
    import pipeline

    # 1. the config, as the supervisor reads it
    try:
        cfg = pipeline.load_config(config_path)
    except pipeline.ConfigError as e:
        out(f"DO NOT START: the supervisor refuses this config: {e}")
        return 2
    paths = pipeline.Paths(run_dir)
    phase = "ppo"
    try:
        with open(paths.state, "r", encoding="utf-8") as f:
            phase = json.load(f).get("phase") or "ppo"
    except (OSError, ValueError):
        pass
    if phase_override:
        # --phase: the start of this phase is checked, whatever state.json says (a run directory
        # before its first start has none, and the supervisor then starts the config's first phase).
        if phase_override == "first":
            phase_override = pipeline.first_phase(cfg)
        out(f"--phase {phase_override}: checking the start of that phase (state.json says {phase!r} or nothing)")
        phase = phase_override
    if phase not in cfg["phases"]:
        out(f"the run is in phase {phase!r}: nothing would be started")
        return 0

    # 2. and 3. the command line, through train.py's own parser
    cmd = pipeline.build_trainer_command(cfg, paths, phase)
    extra = list(cfg["train_defaults"]["extra_train_args"]) + list(cfg["phases"][phase]["extra_train_args"])
    out(f"phase            {phase}")
    out(f"trainer command  {pipeline.quote_cmd(cmd)}")
    out(f"extra arguments  {extra if extra else 'none: the settings of config.py'}")
    if "train.py" not in " ".join(cmd[:2]):
        out("the trainer command is not train.py (trainer.command is set): its arguments are not checked here")
        return 0
    # The supervisor starts train.py in repo_dir; the parse below is this release's train.py. Another
    # release's flags differ (an older train.py refuses the speed flags), so a check from the wrong
    # release could pass a command line the trainer refuses at every start.
    script = next(t for t in cmd[:2] if "train.py" in t)
    target = os.path.realpath(script if os.path.isabs(script) else os.path.join(cfg["repo_dir"], script))
    mine = os.path.realpath(os.path.join(REPO, "train.py"))
    if os.path.normcase(target) != os.path.normcase(mine):
        out(f"DO NOT START: the trainer would run {target}, and this check parses with {mine}. Run the check of that "
            f"release: {os.path.join(os.path.dirname(target), 'cloud', 'check_restart.py')} (or fix repo_dir in the config).")
        return 2
    import train
    try:
        args = train.parse_args(cmd[2:])
    except SystemExit as e:
        out(f"DO NOT START: train.py refuses this command line (exit code {e.code}; its own message is above). "
            f"On the box the trainer would never start.")
        return 2
    run_cfg = train.build_config(args)
    out("own Tera type:   " + ("ON: the workers show it and the checkpoints will say run_cfg.obs.own_tera_type = True"
                               if run_cfg.obs.own_tera_type else "off"))
    # 3b. The top players' imitation term (human_imitation.py). A data directory without one shard
    # built as this run's observation is refused here: the trainer WOULD start, without the term.
    if hasattr(train, "human_imitation_settings"):
        code = check_human_imitation(train, args, run_cfg, out)
        if code:
            return code

    # 4. what the start will say
    out(f"the trainer will log:  {settings_line(run_cfg)}")
    recycle = cfg["phases"][phase]["recycle_hours"]
    recycle = cfg["watchdog"]["recycle_hours"] if recycle is None else recycle
    out(f"planned restart  {'every ' + str(recycle) + ' hours' if recycle else 'none'}")
    out(f"trainer.env      {cfg['trainer']['env'] or 'nothing'}")
    if expect_args is not None and extra != list(expect_args):
        out(f"DO NOT START: the extra arguments are not the ones expected ({list(expect_args)}).")
        return 2

    # 5. the checkpoint and the learning rate at the resume
    lc = run_cfg.learner
    new = (int(lc.lr_warmup_steps), int(lc.lr_hold_steps), int(lc.lr_total_steps))
    new_dec = decisions_schedule({"lr_hold_decisions": getattr(lc, "lr_hold_decisions", 0),
                                  "lr_total_decisions": getattr(lc, "lr_total_decisions", 0)})
    if new_dec is not None:
        out(f"schedule         in decisions of this phase: full rate up to {new_dec[0]:,} decisions, then down to 1/27 at "
            f"{new_dec[1]:,} (after a ramp over the first {new[0]:,} optimizer steps of the phase)")
    elif new[2] <= new[0] + new[1]:
        out(f"schedule         the rate falls to 1/27 in ONE optimizer step, at step {new[0] + new[1] + 1:,} "
            f"(lr_total_steps {new[2]:,} is not after warmup + hold): config.py's own schedule")
    else:
        out(f"schedule         full rate up to optimizer step {new[0] + new[1]:,}, then down to 1/27 at step {new[2]:,}")
    path = checkpoint
    if path is None:
        newest = pipeline.newest_checkpoint(paths.checkpoints)
        path = os.path.join(paths.checkpoints, newest.name) if newest is not None else None
    if path is None:
        out("no checkpoint in the run directory: the trainer starts from nothing")
        return 0
    try:
        ck = read_checkpoint(path)
    except Exception as e:      # noqa: BLE001  (a checkpoint being written this moment, or not a checkpoint)
        out(f"could not read {path}: {e!r}. The learning rate at the resume is NOT checked.")
        return 3
    step = ck["optimizer_step"]
    out(f"checkpoint       {os.path.basename(path)}: update {ck['update']:,}, {ck['total_steps']:,} decisions, optimizer step {step:,}, mode {ck['mode']}")
    # 5a. the observation the checkpoint was trained with against the one this start builds
    saved = ck.get("obs_settings") or {}
    want = obs_settings_of_config(run_cfg)
    added, dropped = obs_change(saved, want)
    out(f"obs_settings     checkpoint {sorted(k for k, v in saved.items() if v is True) or 'none'}; "
        f"this start {sorted(k for k, v in want.items() if v is True) or 'none'}")
    if added or dropped:
        what = []
        if added:
            what.append(f"switches ON {added} (inputs the network was not trained with: a dip is expected)")
        if dropped:
            what.append(f"DROPS {dropped} (inputs the network WAS trained with: the learner only logs a warning and the "
                        f"network goes on without them)")
        out(f"{'NOTE' if allow_obs_change else 'DO NOT START'}: this start changes the observation: " + "; ".join(what) + ".")
        if "matchup" in added or "matchup" in dropped:
            # The matchup block changes the size of the network's input layers: no checkpoint loads
            # across it, whatever is allowed. A checkpoint gets it by widening.
            out("DO NOT START: the matchup block cannot be switched on or off in place (the input layers change size and the "
                "learner refuses the checkpoint). Widen it first: python tools/widen_checkpoint.py --in <ckpt> --out <ckpt_matchup> "
                "and resume from the widened one.")
            return 2
        if "movesets" in added or "movesets" in dropped:
            # Likewise the moveset block (obs_movesets.py): pokemon_net's input layer changes size.
            out("DO NOT START: the moveset block cannot be switched on or off in place (pokemon_net's input layer changes "
                "size and the learner refuses the checkpoint). Widen it first: python tools/widen_checkpoint.py --movesets "
                "--in <ckpt> --out <ckpt_movesets> and resume from the widened one.")
            return 2
        if "main_line" in added or "main_line" in dropped:
            # The main line's observation (obs_main.py): the same size, other inputs. The learner refuses
            # the checkpoint both ways, whatever is allowed.
            out("DO NOT START: the main line's observation cannot be switched on or off in place (the learner refuses "
                "the checkpoint). A main-line network is widened with python tools/widen_checkpoint.py --main-line.")
            return 2
        if not allow_obs_change:
            out("If that is wanted, run this check again with --allow-obs-change, or resume from a checkpoint trained "
                "with these settings.")
            return 2
    if ck["mode"] != lc.mode:
        out(f"the checkpoint is of phase {ck['mode']!r} and the start is {lc.mode!r}: the optimizer and the schedule start afresh")
        return 0
    if step < 0 or None in (ck["warmup"], ck["hold"], ck["total"]):
        out("the checkpoint does not say what schedule it was written under: the rate at the resume is NOT checked")
        return 3
    old = (int(ck["warmup"]), int(ck["hold"]), int(ck["total"]))
    old_dec = decisions_schedule({"lr_hold_decisions": ck.get("hold_decisions"), "lr_total_decisions": ck.get("total_decisions")})
    if old_dec is None and new_dec is None:
        share_old, share_new = rate_share(step, *old), rate_share(step, *new)
    else:
        # Either side in decisions: each side's rate under its own kind of schedule, at the
        # checkpoint's optimizer step and decisions.
        dec = int(ck["total_steps"])
        share_old = rate_share(step, *old) if old_dec is None else rate_share_decisions(step, dec, old[0], *old_dec)
        share_new = rate_share(step, *new) if new_dec is None else rate_share_decisions(step, dec, new[0], *new_dec)
    rate_old, rate_new = float(ck["lr"]) * share_old, float(lc.lr) * share_new
    written = (f"lr_hold_steps {old[1]:,}, lr_total_steps {old[2]:,}" if old_dec is None
               else f"in decisions, lr_hold_decisions {old_dec[0]:,}, lr_total_decisions {old_dec[1]:,}")
    out(f"policy rate at the resume: {rate_new:.3e} ({share_new:.4f} of {lc.lr}); the checkpoint was written at {rate_old:.3e} "
        f"({share_old:.4f} of {ck['lr']}, {written})")
    if new_dec is not None:
        # Where the hold and the end of the anneal lie against the checkpoint's decisions.
        dec, (hold_d, total_d) = int(ck["total_steps"]), new_dec
        per = max(1, int(lc.steps_per_update))
        if dec < hold_d:
            out(f"the hold ends in {hold_d - dec:,} decisions (about {(hold_d - dec) / per:,.0f} updates of {per:,}), "
                f"1/27 at {total_d:,}, in {total_d - dec:,} decisions")
        elif dec < total_d:
            out(f"the anneal began {dec - hold_d:,} decisions ago (at {hold_d:,}); 1/27 at {total_d:,}, in {total_d - dec:,} "
                f"decisions (about {(total_d - dec) / per:,.0f} updates of {per:,})")
        else:
            out(f"the anneal ended {dec - total_d:,} decisions ago (at {total_d:,}): the rate stays at 1/27")
    edge = new[0] + new[1]
    if new_dec is None and step <= edge:
        # optimizer steps an update: every epoch walks the update's steps in minibatches (96 for the
        # runs so far: 36,864 x 2 / 768); an update that stops early on target_kl takes fewer
        per_update = (float(lc.steps_per_update) * float(lc.update_epochs)
                      / (float(lc.minibatch_size) * float(max(1, int(lc.grad_accum_steps or 1)))))
        per_dec = float(lc.minibatch_size) * float(max(1, int(lc.grad_accum_steps or 1))) / float(lc.update_epochs)
        out(f"the hold ends in {edge - step:,} optimizer steps (about {max(0, edge - step) / per_update:,.0f} updates at "
            f"{per_update:g} steps an update, {max(0, edge - step) * per_dec / 1e6:,.1f}M decisions)")
    if abs(rate_new - rate_old) > 1e-12 * max(rate_new, rate_old):
        out(f"{'NOTE' if allow_rate_change else 'DO NOT START'}: this start changes the learning rate at the resume by a factor of "
            f"{rate_new / rate_old:.2f}. With --lr-total-steps given after the hold has ended that is the rate going back UP.")
        if not allow_rate_change:
            out("If that is wanted (a learning-rate restart, cloud/stage_lr.py), run this check again with --allow-rate-change.")
            return 3
    if (old_dec is not None or new_dec is not None) and old_dec != new_dec:
        # A schedule in decisions that comes, goes or moves: the rest of the run follows another
        # schedule even where the rate at the resume is the same.
        def kind(d):
            return "optimizer steps" if d is None else f"decisions (hold {d[0]:,}, 1/27 at {d[1]:,})"
        out(f"{'NOTE' if allow_rate_change else 'DO NOT START'}: this start changes the learning-rate schedule: the checkpoint "
            f"was written under a schedule in {kind(old_dec)}, this start has one in {kind(new_dec)}.")
        if not allow_rate_change:
            out("If that is wanted, run this check again with --allow-rate-change.")
            return 3
    allowed = [w for w, on in (("the learning rate", allow_rate_change), ("the observation", allow_obs_change)) if on]
    out("OK to start: the config is valid, train.py takes the command line, and no learning rate moves at the resume."
        if not allowed else f"OK to start (a change of {' and '.join(allowed)} at the resume was allowed).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the run's JSON config, the file the supervisor is started with (~/runs/<run>/run.json)")
    ap.add_argument("--run-dir", required=True, help="the run directory (~/runs/<run>)")
    ap.add_argument("--expect-args", default=None,
                    help="the extra arguments the trainer must be started with, as one string; '' for none. Anything else is refused")
    ap.add_argument("--allow-rate-change", action="store_true", help="a learning rate that moves at the resume is reported, not refused")
    ap.add_argument("--allow-obs-change", action="store_true",
                    help="observation settings that differ between the checkpoint and the start are reported, not refused")
    ap.add_argument("--checkpoint", default=None, help="read this checkpoint and not the newest of the run directory")
    ap.add_argument("--phase", default=None, choices=("imitation", "warmup", "ppo", "first"),
                    help="check the start of this phase (first: the config's first phase, where a run with no state.json "
                         "starts); default: the phase state.json names, else ppo")
    args = ap.parse_args(argv)
    expect = None if args.expect_args is None else shlex.split(args.expect_args)
    return check(args.config, args.run_dir, expect, args.allow_rate_change, args.checkpoint, allow_obs_change=args.allow_obs_change,
                 phase_override=args.phase)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # train.py imports worker.py, which starts poke-env's loop thread: leave without waiting for it.
    os._exit(code)
