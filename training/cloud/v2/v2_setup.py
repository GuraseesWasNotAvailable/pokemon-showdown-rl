#!/usr/bin/env python3
"""
Waves 1 and 2 of the v2 training plan on the owner's own AWS box (g6e.12xlarge, 4 L40S, 48 vCPU),
2026-10-16, after the borrowed box was deleted: 8 arms of 60M decisions from the same start network S,
in two batches of 4 (one GPU each), under their own systemd units (psppo-v2-<arm>-train / -eval).
Batch 1 answers each of the owner's asks against the control; batch 2 is the ablations.

    python3 v2_setup.py --start CKPT [--write] [--units] [--only a,b]

Without --write it only prints what it would do. Never touches psppo-full-* or anything under ~/runs/full.
"""
import argparse
import json
import os
import subprocess
import sys

HOME = os.path.expanduser("~")
REL = f"{HOME}/rel/v2-0bf9e93"
PY = f"{HOME}/venv/bin/python"
ROOT = f"{HOME}/runs/v2"
ARM_DECISIONS = 60_000_000

READ = ["--read-bonus", "0.3", "--read-anneal", "20000000:45000000"]
HEAD = ["--opp-action-aux-coef", "0.1"]
# name, GPU, extra flags of the arm (on top of the learning-rate restart every arm has)
ARMS = [
    # batch 1: the control and one arm per ask (reads, set-up, abilities and items)
    ("a0", 0, []),
    ("readhead", 1, READ + HEAD),
    ("setupw2", 2, ["--opponent-seats", "setupnet:0.15", "--boosted-row-weight", "2"]),
    ("musethead", 3, ["--obs-matchup", "--set-head-coef", "0.1"]),
    # batch 2: the ablations. Each starts on the GPU of the batch-1 arm it follows (v2_guard.sh CHAIN
    # a0:fp10 readhead:head setupw2:setup15); read runs on the second box (psppo-train-s2, GPU 0).
    ("read", 0, READ),
    ("head", 1, HEAD),
    ("setup15", 2, ["--opponent-seats", "setupnet:0.15"]),
    ("fp10", 0, ["--opponent-seats", "foulplay:0.1", "--seat-foulplay-iterations", "88000"]),
]


def run_json(start_total, lr_flags, i, extra, idx):
    return {
        "repo_dir": REL,
        "python": PY,
        "showdown_dir": f"{HOME}/pokemon-showdown",
        "servers": {"count": 4, "base_port": 10300 + 20 * idx, "heap_mb": 2048, "ready_timeout_seconds": 420},
        "trainer": {"stop_timeout_seconds": 420, "failure_stop_timeout_seconds": 120},
        "train_defaults": {"battles": 256, "rooms_per_pair": 16, "minibatch": 768, "seq_len": 256, "grad_accum": 1,
                           "steps_per_update": 36864, "teacher": "psbot", "save_every": 50, "extra_train_args": list(lr_flags)},
        "phases": {
            "imitation": {"skip": True, "min_decisions": 0, "max_decisions": 1},
            "warmup": {"skip": True, "min_decisions": 0, "max_decisions": 1},
            "ppo": {"skip": False, "target_decisions": start_total + ARM_DECISIONS, "extra_train_args": list(extra)},
        },
        "watchdog": {"stall_minutes": 10, "startup_grace_minutes": 20, "update_stall_minutes": 20, "max_restarts_per_hour": 6,
                     "backoff_seconds": [30, 120, 300, 600, 900], "max_fruitless_starts": 5, "nan_updates_limit": 3,
                     "telemetry_error_minutes": 5, "recycle_hours": 6},
        "disk": {"min_free_gb": 5},
        "retention": {"enabled": False, "keep_newest": 4, "milestone_every": 500, "keep_hours": 0, "check_minutes": 10},
        "on_done": "idle",
        "on_failed": "idle",
        "eval": {"every_hours": 4, "games": 200, "opponents": ["simple", "psbot"], "port": 10470 + 4 * idx,
                 "foulplay": {"games": 200, "every_hours": 4, "parallel": 2, "iterations": 88000}},
        "_notes": {"what this is": "v2 plan waves 1+2 on the owner's own box (2026-10-16), two batches of 4: one arm of 60M decisions from S with the "
                                   "learning-rate restart; arm flags in phases.ppo.extra_train_args. Plan: "
                                   "C:/Users/you/.claude/plans/does-it-know-about-fluffy-rocket.md"},
    }


UNIT_TRAIN = """[Unit]
Description=ps-ppo v2 arm {arm}: trainer
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=1800
StartLimitBurst=10
[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory={rel}
Environment=PYTHONUTF8=1 PYTHONUNBUFFERED=1 PSPPO_GPU_SPLIT=1 CUDA_VISIBLE_DEVICES={gpu} FOULPLAY_DIR={home}/foul-play TORCHINDUCTOR_CACHE_DIR={run}/.inductor
ExecCondition={py} {rel}/pipeline.py --config {run}/run.json --run-dir {run} --can-start
ExecStart={py} {rel}/pipeline.py --config {run}/run.json --run-dir {run}
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

UNIT_EVAL = """[Unit]
Description=ps-ppo v2 arm {arm}: evaluator
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=1800
StartLimitBurst=10
[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory={rel}
Nice=5
Environment=PYTHONUTF8=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES={gpu} FOULPLAY_DIR={home}/foul-play TORCHINDUCTOR_CACHE_DIR={run}/.inductor-eval
ExecStart={py} evaluate.py --run-dir {run} --watch --config {run}/run.json
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
    ap.add_argument("--start", required=True, help="the start checkpoint S")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--units", action="store_true", help="also write the systemd units (sudo)")
    ap.add_argument("--only", default="", help="comma separated arm names")
    a = ap.parse_args()
    import torch
    ck = torch.load(a.start, map_location="cpu", weights_only=False)
    total = int(ck["total_steps"])
    print(f"S = {a.start}: total_steps {total:,}, update {ck.get('update_idx')}, mode {ck.get('mode') or ck.get('run_cfg', {}).get('mode')}")
    out = subprocess.run([PY, f"{REL}/cloud/stage_lr.py", "--checkpoint", a.start, "--start-rate", "3e-5",
                          "--hold-decisions", "100000000", "--end-decisions", str(total + 160_000_000)],
                         capture_output=True, text=True, cwd=REL, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    line = next(l for l in out.stdout.splitlines() if l.startswith("flags:"))
    lr_flags = line.split(":", 1)[1].split()
    print("LR flags:", " ".join(lr_flags))
    only = {s for s in a.only.split(",") if s}
    for arm, gpu, extra in ARMS:
        if only and arm not in only:
            continue
        run = f"{ROOT}/{arm}"
        cfg = run_json(total, lr_flags, gpu, extra, [x[0] for x in ARMS].index(arm))
        print(f"{arm}: GPU {gpu}, servers {cfg['servers']['base_port']}+4, eval port {cfg['eval']['port']}, "
              f"target {cfg['phases']['ppo']['target_decisions']:,}, extra {' '.join(extra) or '-'}")
        if not a.write:
            continue
        os.makedirs(f"{run}/checkpoints", exist_ok=True)
        with open(f"{run}/run.json", "w") as f:
            json.dump(cfg, f, indent=1)
        dst = f"{run}/checkpoints/{os.path.basename(a.start)}"
        if not os.path.exists(dst) and not [n for n in os.listdir(f"{run}/checkpoints") if n.endswith(".pt")]:
            if "--obs-matchup" in extra:
                subprocess.run([PY, "tools/widen_checkpoint.py", "--in", a.start, "--out", dst]
                               + (["--set-head"] if "--set-head-coef" in extra else []),
                               cwd=REL, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            else:
                os.link(a.start, dst)
        if a.units:
            for kind, tmpl in (("train", UNIT_TRAIN), ("eval", UNIT_EVAL)):
                text = tmpl.format(arm=arm, rel=REL, gpu=gpu, home=HOME, run=run, py=PY)
                subprocess.run(["sudo", "tee", f"/etc/systemd/system/psppo-v2-{arm}-{kind}.service"], input=text,
                               text=True, stdout=subprocess.DEVNULL, check=True)
    if a.write and a.units:
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
