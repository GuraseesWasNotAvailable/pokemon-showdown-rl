#!/usr/bin/env python
"""
A stand-in for train.py, for exercising pipeline.py without a GPU, Ray or a Showdown server.

    python tests/fake_trainer.py --mode imitation --ports 8110-8111 --steps-per-update 1000 \
        --save-every 2 --ckpt-dir RUN/checkpoints --stop-file RUN/trainer.stop --fake-scenario scenario.json

It takes train.py's command line (an unknown flag is an error, as there), prints lines in
train.py's real formats and writes small fake checkpoints named like the real ones:

  * the "[Telemetry] ACTIVE: n | INFER: {...} | TRAIN: {...}" line of train.py's _telemetry_loop
  * the "[learner] upd=N kl=... vloss=... ploss=... loss=... n_mb=..." line of learner.py, with
    the coloured "(LearnerActor pid=N)" prefix Ray puts in front of actor output
  * the resume lines of learner.py and train.py's "Saved a final checkpoint: PATH"

Its counters behave like the learner's (learner.py, _load_checkpoint): the newest checkpoint by
modification time is loaded; if it was written in the same mode, update and total_steps continue
from it; if it was written in another mode both restart at zero ("PHASE CHANGE DETECTED").

--fake-scenario FILE is a JSON object, every key optional:

  tick_seconds      seconds between telemetry lines (default 0.05)
  steps_per_tick    decisions added per tick (default 200)
  losses            {mode: {"ploss": [start, floor, tau], "vloss": [start, floor, tau]}}: the loss
                    printed is floor + (start - floor) * exp(-total_steps / tau)
  fault_dir         where the "this fault already happened" markers go
  faults            list of {"id", "kind", "mode"?, "at_steps"?, "repeat"?, "code"?}. A fault fires
                    once (its marker is written first) unless "repeat" is true. Kinds:
      hang            freeze: no more output, no reaction to the stop file
      die             exit with "code" (default 1) without saving
      die_at_start    exit with "code" (default 1) right after the first line
      worker_crash    print train.py's "RolloutWorker crashed" line, save, exit 1 (as train.py does)
      no_updates      from here on every update fails ("PPO Update failed"), decisions keep coming
      ignore_stop     never react to the stop file
      no_final_save   on a stop: "Could not save a final checkpoint", exit without saving
      resume_failed   at start, if a checkpoint exists: "resume failed from PATH", then run on with
                      untrained weights and zero counters, as the real learner would
      mute_learner    count updates but never print the "[learner] upd=" line
      nan             from here on every update prints vloss, ploss and loss as nan (and still
                      saves its periodic checkpoints, as the real learner would)
      sigterm_saves   on SIGTERM, save a final checkpoint of whatever the counters are and exit,
                      at any point after the resume lines. That is what the real train.py does on
                      Linux: Ray turns a SIGTERM to the driver into a SystemExit, and train.py's
                      "finally" then saves the learner's weights, also the untrained ones it holds
                      after a failed resume. (Windows has no SIGTERM to send: a kill there is final.)
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import signal
import sys
import time

DEFAULT_LOSSES = {
    "imitation": {"ploss": [1.90, 1.10, 3000.0], "vloss": [3.70, 3.00, 6000.0]},
    "warmup": {"ploss": [0.0, 0.0, 1.0], "vloss": [3.00, 2.30, 2000.0]},
    "ppo": {"ploss": [0.004, 0.001, 50000.0], "vloss": [2.40, 2.20, 50000.0]},
}

OUT = sys.stdout.buffer
PID = os.getpid()
ACTOR = f"\x1b[36m(LearnerActor pid={PID})\x1b[0m "


def emit(text: str) -> None:
    OUT.write((text + "\r\n").encode("utf-8"))
    OUT.flush()


def logline(text: str) -> None:
    now = datetime.datetime.now()
    emit(f"{now.strftime('%Y-%m-%d %H:%M:%S')},{now.microsecond // 1000:03d} [INFO] {text}")


def parse_ports(spec: str):
    out = []
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


def newest_checkpoint(ckpt_dir: str):
    paths = glob.glob(os.path.join(ckpt_dir, "learner_update_*.pt"))
    return max(paths, key=os.path.getmtime) if paths else None


def save(ckpt_dir: str, mode: str, update: int, total_steps: int) -> str:
    path = os.path.join(ckpt_dir, f"learner_update_{update:06d}.pt")
    payload = {"fake": True, "mode": mode, "update_idx": update, "total_steps": total_steps, "saved": time.time()}
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(path + ".tmp", path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="ppo_with_jepa")
    ap.add_argument("--ports", default="8000-8009")
    ap.add_argument("--battles", type=int, default=800)
    ap.add_argument("--rooms-per-pair", type=int, default=32)
    ap.add_argument("--teacher", choices=("psbot", "simple"), default="psbot")
    ap.add_argument("--steps-per-update", type=int, default=36864)
    ap.add_argument("--minibatch", type=int, default=768)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--shaping", choices=("potential", "faint", "none"), default="potential")
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--fake-scenario", default=None)
    ap.add_argument("--fake-note", default=None, help="written to CKPT_DIR/../fake_notes.txt, to show extra arguments arrive")
    args = ap.parse_args()

    sc = {}
    if args.fake_scenario:
        with open(args.fake_scenario, "r", encoding="utf-8") as f:
            sc = json.load(f)
    tick = float(sc.get("tick_seconds", 0.05))
    per_tick = int(sc.get("steps_per_tick", 200))
    losses = dict(DEFAULT_LOSSES)
    losses.update(sc.get("losses") or {})
    fault_dir = sc.get("fault_dir") or os.path.join(os.path.dirname(os.path.abspath(args.ckpt_dir)), "fake_faults")
    faults = [f for f in (sc.get("faults") or []) if f.get("mode") in (None, args.mode)]
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(fault_dir, exist_ok=True)
    if args.fake_note:
        with open(os.path.join(os.path.dirname(os.path.abspath(args.ckpt_dir)), "fake_notes.txt"), "a", encoding="utf-8") as f:
            f.write(f"{args.mode} {args.fake_note}\n")

    def pending(kind: str, steps: int = 0):
        """The first fault of this kind that is due and has not fired yet, marked as fired."""
        for f in faults:
            if f.get("kind") != kind or steps < int(f.get("at_steps", 0)):
                continue
            marker = os.path.join(fault_dir, f"{f.get('id', kind)}.done")
            if f.get("repeat"):
                return f
            if os.path.exists(marker):
                continue
            with open(marker, "w", encoding="utf-8") as m:
                m.write(f"{time.time()} pid {PID} steps {steps}\n")
            return f
        return None

    def active(kind: str, steps: int = 0) -> bool:
        """Like pending(), but stays true for the rest of this process once it has fired."""
        if kind in fired:
            return True
        if pending(kind, steps) is not None:
            fired.add(kind)
            return True
        return False

    fired = set()
    ports = parse_ports(args.ports)
    emit("2026-10-05 09:57:10,829\tINFO worker.py:2033 -- Started a local Ray instance.")
    logline(f"mode={args.mode} teacher={args.teacher} reward={args.shaping} ports={ports} steps_per_update={args.steps_per_update} "
            f"minibatch={args.minibatch}x{args.grad_accum} seq={args.seq_len} ckpt={args.ckpt_dir}")
    f = pending("die_at_start")
    if f is not None:
        emit("Traceback (most recent call last):")
        emit("RuntimeError: fake trainer: dying at start")
        return int(f.get("code", 1))
    logline(f"Initializing Run: Target Battles={args.battles}, Total Pairs={max(1, args.battles // max(1, args.rooms_per_pair))}, Workers={len(ports)}")
    logline(f"Deployment complete. Distributed {len(ports)} rollout workers.")

    update = 0
    total_steps = 0
    episodes = 0
    path = newest_checkpoint(args.ckpt_dir)
    if path and pending("resume_failed") is not None:
        emit(ACTOR + f"[learner] resume failed from {path}: RuntimeError('PytorchStreamReader failed reading zip archive: failed finding central directory')")
        emit(ACTOR + "[learner] pushed init policy 0 to inference")
    elif path:
        with open(path, "r", encoding="utf-8") as fh:
            ck = json.load(fh)
        emit(ACTOR + "[learner] Loaded target network EMA state from checkpoint.")
        if ck.get("mode") != args.mode:
            emit(ACTOR + f"[learner] \U0001F6D1 PHASE CHANGE DETECTED ({ck.get('mode')} -> {args.mode}).")
            emit(ACTOR + "[learner] \U0001F9F9 SKIPPING Optimizer/Scheduler load to enforce fresh LR.")
        else:
            emit(ACTOR + "[learner] Optimizer state loaded successfully.")
            emit(ACTOR + "[learner] Scheduler state loaded successfully.")
            update, total_steps = int(ck["update_idx"]), int(ck["total_steps"])
        emit(ACTOR + f"[learner] resumed from {path}")
        emit(ACTOR + "[learner] pushed resumed policy 0 to inference")
    else:
        emit(ACTOR + "[learner] pushed init policy 0 to inference")
    if any(f.get("kind") == "sigterm_saves" for f in faults) and hasattr(signal, "SIGTERM"):
        def on_sigterm(_signum, _frame):
            logline(f"Saved a final checkpoint: {save(args.ckpt_dir, args.mode, update, total_steps)}")
            os._exit(0)
        signal.signal(signal.SIGTERM, on_sigterm)
    time.sleep(float(sc.get("start_delay_seconds", 0.0)))
    logline("Training started. Entering telemetry loop.")

    def telemetry(in_dataset: int) -> None:
        logline(f"[Telemetry] ACTIVE: {2 * args.battles} | INFER: {{'total_requests': {total_steps * 2}, 'avg_batch_size': 28.21, "
                f"'model_version': {update}}} | TRAIN: {{'update': {update}, 'episodes': {episodes}, "
                f"'steps_in_dataset': {in_dataset}, 'total_steps': {total_steps}}}")

    def loss(name: str) -> float:
        start, floor, tau = losses.get(args.mode, DEFAULT_LOSSES["ppo"]).get(name, [0.0, 0.0, 1.0])
        return float(floor) + (float(start) - float(floor)) * math.exp(-total_steps / max(float(tau), 1e-9))

    started = time.time()
    in_dataset = 0
    telemetry(in_dataset)
    while True:
        if args.minutes and time.time() - started > args.minutes * 60:
            logline(f"Reached --minutes {args.minutes}. Shutting down.")
            break
        if args.stop_file and os.path.exists(args.stop_file) and not active("ignore_stop"):
            logline(f"Stop file {args.stop_file} found. Shutting down.")
            break
        if pending("hang", total_steps) is not None:
            while True:
                time.sleep(3600)
        f = pending("die", total_steps)
        if f is not None:
            emit("Windows fatal exception: access violation")
            return int(f.get("code", 1))
        if pending("worker_crash", total_steps) is not None:
            emit(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},000 [CRITICAL] FATAL: A RolloutWorker crashed in run(): fake")
            emit(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},000 [CRITICAL] Shutting down cluster due to worker crash.")
            logline(f"Saved a final checkpoint: {save(args.ckpt_dir, args.mode, update, total_steps)}")
            return 1

        total_steps += per_tick
        in_dataset += per_tick
        episodes += max(1, per_tick // 30)
        learner_line = None
        if in_dataset >= args.steps_per_update:
            in_dataset = 0
            if active("no_updates", total_steps):
                emit(ACTOR + "PPO Update failed: LoweringException: ModuleNotFoundError: No module named 'triton'")
            else:
                update += 1
                vloss, ploss = loss("vloss"), loss("ploss")
                wiggle = 0.0005 * math.sin(update)
                if active("nan", total_steps):
                    vloss = ploss = float("nan")
                if not active("mute_learner"):
                    learner_line = (ACTOR + f"[learner] upd={update} kl={0.0020 + wiggle:.4f} clip=0.120 ent=1.250 "
                                    f"vloss={vloss + wiggle:.3f} ploss={ploss + wiggle:.3f} loss={0.5 * vloss + ploss:.3f} n_mb=128")
                if update % max(1, args.save_every) == 0:
                    save(args.ckpt_dir, args.mode, update, total_steps)
        if learner_line and update % 5 == 0:
            # Seen in real output about once in 200 actor lines: train.py's logger writes its line
            # between the learner's text and its newline, so both arrive as one physical line
            # ("... n_mb=1282026-10-05 09:48:48,489 [INFO] [Telemetry] ..."), followed by an empty one.
            OUT.write(learner_line.encode("utf-8"))
            OUT.flush()
            telemetry(in_dataset)
            emit("")
        else:
            # The learner's line sometimes comes before the telemetry line that counts its update
            # and sometimes after it: do both.
            if learner_line and update % 2:
                emit(learner_line)
            telemetry(in_dataset)
            if learner_line and not update % 2:
                emit(learner_line)
        time.sleep(tick)

    if pending("no_final_save") is not None:
        emit(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},000 [WARNING] Could not save a final checkpoint: GetTimeoutError()")
        return 0
    logline(f"Saved a final checkpoint: {save(args.ckpt_dir, args.mode, update, total_steps)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
