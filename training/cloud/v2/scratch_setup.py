#!/usr/bin/env python3
"""
The from-scratch run of the v2 plan (owner's ask, 2026-10-16): random weights, every new technique on
from day one. One pipeline run in ~/runs/v2/scratch, units psppo-v2-scratch-train / -eval:

  imitation  PSBot imitation from random weights (as the imitation test's arm A), 12M decisions (arm A passed 60% agreement by ~7M), with the
             matchup block (abilities and items) in the observation from the start (--obs-matchup in every phase);
  warmup     the value head settles (plateau rule of the full run);
  ppo        600M decisions with the SetHead, the opponent-action head, the read bonus (0.30, faded
             20M -> 45M of this phase), set-up seats on 15% of pairs with boosted rows weighted 2, and a
             learning rate held at 3e-4 for 800k optimizer steps (~307M decisions), then annealed by 1.6M (~614M).

The top-player fine-tune (the imitation test's arm C) needs the replay data the deleted box held; it is
added between imitation and ppo once that data is collected again.

    python3 scratch_setup.py --gpu 0 [--write] [--units]

Opt-in speed-up (infer_pool.py; nothing changes without it): --infer-actors N gives the trainer N
inference actors with speed_args() (train.py --infer-actors N --fast-sample --infer-sync-s 1), and
--trainer-gpus 0,1,2,3 lets the trainer alone (run.json trainer.env) see more GPUs than the units'
--gpu, so that with PSPPO_GPU_SPLIT=1 each actor has a GPU of its own. The flags exist only in a
release with infer_pool.py, so --infer-actors needs --rel naming one (git archive v2-speed into
~/rel/v2-speed): with the default release it is refused, since that train.py would refuse the flags
with exit code 2 at every start and the supervisor would mark the run failed after five. Check the
written config with <that release>/cloud/check_restart.py before the start.

    python3 scratch_setup.py --rel ~/rel/v2-speed --infer-actors 3 --trainer-gpus 0,1,2,3 [--write]
"""
import argparse
import json
import os
import subprocess

HOME = os.path.expanduser("~")
REL = f"{HOME}/rel/v2-movesets"   # train-v2 with the opponent-moveset block (RECOVERY.md); before it: v2-0bf9e93
PY = f"{HOME}/venv/bin/python"
RUN = f"{HOME}/runs/v2/scratch"

PPO_ARGS = ["--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1",
            "--read-bonus", "0.3", "--read-anneal", "20000000:45000000",
            "--opponent-seats", "setupnet:0.15", "--boosted-row-weight", "2",
            "--lr-hold-steps", "800000", "--lr-total-steps", "1600000"]


def speed_args(infer_actors: int) -> list:
    """train.py's speed flags for N inference actors (none for 0: the run as it always was)."""
    if infer_actors <= 0:
        return []
    return ["--infer-actors", str(int(infer_actors)), "--fast-sample", "--infer-sync-s", "1"]


def release_problem(rel: str, infer_actors: int):
    """Why the release rel cannot take the speed flags, or None. train.py has them only next to
    infer_pool.py; an older one (v2-movesets) refuses them at every start."""
    if infer_actors > 0 and not os.path.isfile(os.path.join(rel, "infer_pool.py")):
        return (f"{rel} has no infer_pool.py: its train.py refuses --infer-actors with exit code 2 at every start, and after "
                f"five starts the run is marked failed. Copy a release that has it (git archive v2-speed) and name it with --rel.")
    return None


def run_json(servers, infer_actors=0, trainer_gpus=None, rel=REL):
    trainer = {"stop_timeout_seconds": 420, "failure_stop_timeout_seconds": 120}
    if trainer_gpus:
        trainer["env"] = {"CUDA_VISIBLE_DEVICES": str(trainer_gpus)}
    return {
        "repo_dir": rel, "python": PY, "showdown_dir": f"{HOME}/pokemon-showdown",
        "servers": {"count": servers, "base_port": 10600, "heap_mb": 2048, "ready_timeout_seconds": 420},
        "trainer": trainer,
        "train_defaults": {"battles": 64 * servers, "rooms_per_pair": 16, "minibatch": 768, "seq_len": 256, "grad_accum": 1,
                           "steps_per_update": 36864, "teacher": "psbot", "save_every": 50,
                           "extra_train_args": ["--obs-matchup", "--obs-movesets"] + speed_args(infer_actors)},
        "phases": {
            "imitation": {"skip": False, "min_decisions": 12_000_000, "max_decisions": 12_000_000},
            "warmup": {"skip": False, "min_decisions": 500_000, "max_decisions": 1_500_000,
                       "plateau": {"metric": "vloss", "window_decisions": 300_000, "min_improvement_pct": 2.0, "smooth_updates": 4},
                       "train": {"save_every": 5}},
            "ppo": {"skip": False, "target_decisions": 600_000_000, "extra_train_args": list(PPO_ARGS)},
        },
        "watchdog": {"stall_minutes": 10, "startup_grace_minutes": 20, "update_stall_minutes": 30, "max_restarts_per_hour": 6,
                     "backoff_seconds": [30, 120, 300, 600, 900], "max_fruitless_starts": 5, "nan_updates_limit": 3,
                     "telemetry_error_minutes": 5, "recycle_hours": 6},
        "disk": {"min_free_gb": 10},
        "retention": {"enabled": False, "keep_newest": 4, "milestone_every": 500, "keep_hours": 0, "check_minutes": 10},
        "on_done": "idle", "on_failed": "idle",
        "eval": {"every_hours": 4, "games": 200, "opponents": ["simple", "psbot"], "port": 10690,
                 "foulplay": {"games": 200, "every_hours": 4, "parallel": 2, "iterations": 88000}},
        "_notes": {"what this is": "v2 from-scratch run (2026-10-16): random weights, PSBot imitation with the matchup block, "
                                   "then PPO with every v2 technique on. cloud/v2/scratch_setup.py on train-v2."},
    }


UNIT = """[Unit]
Description=ps-ppo v2 from-scratch run: {kind}
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=1800
StartLimitBurst=10
[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory={rel}
{nice}Environment=PYTHONUTF8=1 PYTHONUNBUFFERED=1 PSPPO_GPU_SPLIT=1 CUDA_VISIBLE_DEVICES={gpu} FOULPLAY_DIR={home}/foul-play TORCHINDUCTOR_CACHE_DIR={run}/.inductor-{kind}
{condition}ExecStart={start}
Restart=always
RestartSec=30
RestartPreventExitStatus=2 3 4 5
KillSignal=SIGTERM
KillMode=mixed
TimeoutStopSec=600
LimitNOFILE=1048576
TasksMax=infinity
OOMPolicy=continue
[Install]
WantedBy=multi-user.target
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for the run (two GPUs: inference and learner apart)")
    ap.add_argument("--servers", type=int, default=12, help="one PSBot teacher process per server: the imitation phase is bound by them")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--infer-actors", type=int, default=0,
                    help="opt-in speed-up: this many inference actors (train.py --infer-actors, with --fast-sample and "
                         "--infer-sync-s 1); 0, the default, leaves the run as it always was")
    ap.add_argument("--trainer-gpus", default=None,
                    help="CUDA_VISIBLE_DEVICES of the trainer alone (run.json trainer.env), e.g. 0,1,2,3; default: the units' --gpu")
    ap.add_argument("--rel", default=REL,
                    help=f"the release run.json's repo_dir names (and --units' pipeline and evaluator); default {REL}. "
                         f"--infer-actors needs one with infer_pool.py")
    a = ap.parse_args()
    rel = os.path.abspath(os.path.expanduser(a.rel))
    problem = release_problem(rel, a.infer_actors)
    if problem:
        ap.error(problem)
    cfg = run_json(a.servers, a.infer_actors, a.trainer_gpus, rel)
    shown = ("servers", "phases") if not (a.infer_actors or a.trainer_gpus) else ("servers", "trainer", "train_defaults", "phases")
    print(json.dumps({k: cfg[k] for k in shown}, indent=1))
    if not a.write:
        return
    os.makedirs(f"{RUN}/checkpoints", exist_ok=True)
    with open(f"{RUN}/run.json", "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"check before the start: {PY} {rel}/cloud/check_restart.py --config {RUN}/run.json --run-dir {RUN}")
    if a.units:
        train = f"{PY} {rel}/pipeline.py --config {RUN}/run.json --run-dir {RUN}"
        units = {
            "train": UNIT.format(kind="trainer", rel=rel, nice="", gpu=a.gpu, home=HOME, run=RUN,
                                 condition=f"ExecCondition={train} --can-start\n", start=train),
            "eval": UNIT.format(kind="evaluator", rel=rel, nice="Nice=5\n", gpu=a.gpu, home=HOME, run=RUN, condition="",
                                start=f"{PY} evaluate.py --run-dir {RUN} --watch --config {RUN}/run.json"),
        }
        for kind, text in units.items():
            subprocess.run(["sudo", "tee", f"/etc/systemd/system/psppo-v2-scratch-{kind}.service"], input=text, text=True,
                           stdout=subprocess.DEVNULL, check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)


if __name__ == "__main__":
    main()
