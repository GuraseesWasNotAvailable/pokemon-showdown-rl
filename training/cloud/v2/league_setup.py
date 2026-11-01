#!/usr/bin/env python3
"""
The league run (bench/notes/LEAGUE.md, owner's ask 2026-10-20): the strongest network, main update
024381 (M24381), widened to this line's inputs on the main line's own observation, trained by PPO
against a league (self-play, setupnet seats, and frozen networks: M24381 itself, SFIN and the run's own
past saves), on fresher data, with a learning-rate restart.

    python3 league_setup.py --rel ~/rel/league --start <M24381w.pt> --sfin <SFIN.pt> [--write] [--units]
    python3 league_setup.py --rel ... --start ... --sfin ... --rehearsal [--base-port N] [--write]
    cd <release> && ~/venv/bin/python cloud/v2/league_setup.py --check-run RUN     (the frozen seats, before the start)

The start is M24381 widened (tools/widen_checkpoint.py --main-line, checked bit for bit against the
window app by bench/league_exactness.py). It is copied as it is, with SFIN, to RUN/league/ (each checked by
sha256): the frozen seats play RUN/league/M24381w.pt and RUN/league/SFIN.pt, so nothing of the run reads
another run's directory. RUN/checkpoints/learner_update_000000.pt (the newest by mtime: the learner
resumes from it) is the LEAGUE START made from it (league_start): the same model and target_model tensors
(checked equal), M24381's counters at 0, no optimizer and no scheduler, and run_cfg.learner.mode
"imitation", so that warmup, and PPO too if it ever started from this file, sees a phase change and starts
its optimizer and its schedule afresh (as run T's start did). Left at M24381's mode "ppo", a PPO start from
it would be a resume at 900M decisions: the learning rate at its floor (check_restart exit 3) and the
target long past. What it was is in its "league_start" entry.

run.json is run T's (--base, default ~/runs/v2/topstart/run.json; 'builtin' for the copy below) with:
  * repo_dir the league release; servers from port 10900 (T: 10700), the evaluator on 10990;
  * --obs-main-line in every phase (train_defaults): the main line's observation, the blocks after it;
  * imitation skipped; warmup as T (500k to 1.5M decisions, plateau on vloss), --set-head-coef 0.1;
  * PPO: T's arguments without the read bonus (--read-bonus / --read-anneal: a reward change M24381
    never had, and the waves' read arms did not beat their control) and without T's schedule, plus
      --opponent-seats setupnet:0.15,frozen:0.25 --frozen-nets m24381=...:0.35,sfin=...:0.25
      --frozen-past 4:0.40 --frozen-refresh-min 30        (the league)
      --learn-queue-episodes 32 --lag-stats                (fresher data, measured)
      --lr 6e-05 --lr-hold-decisions 100000000 --lr-total-decisions 300000000   (the restart)
    target 300,000,000 decisions; --fast-update --fast-ingest and --rollout-nice 5 as T (v2-fast);
  * save_every 200 (as T, 7.4M decisions); retention off (it deletes).

Refused (exit 2): an existing RUN/state.json, a run.json that differs, checkpoints already in RUN;
a start that is not a main-line network with both blocks and a SetHead; an SFIN that is not a v2
network with both blocks; a release without the league's code (obs_main.py, frozen_pool.py, league.py,
train.py --obs-main-line) or without what T's setup checks (the decisions schedule, v2-fast, the
heuristic cache); a base whose PPO arguments are not T's.

--rehearsal: RUN ~/dev/league/rehearsal (box c), 6 servers from --base-port (11400), 384 battles, one
GPU (infer actors 1), warmup 100k to 300k, PPO to 3M decisions with the decisions schedule (hold 1.5M,
total 3M), PPO saves every 10, no evaluator; the frozen seats' past saves re-picked every 5 minutes from
saves 2 minutes old.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import topstart_setup as ts  # noqa: E402  (release and cache checks, sha256)

HOME = os.path.expanduser("~")
PY = f"{HOME}/venv/bin/python"
RUN = f"{HOME}/runs/v2/league"
REHEARSAL = f"{HOME}/dev/league/rehearsal"
BASE = f"{HOME}/runs/v2/topstart/run.json"
BASE_PORT = 10900
REHEARSAL_PORT = 11400

T_DEFAULT_ARGS = ["--obs-matchup", "--obs-movesets", "--infer-actors", "3", "--infer-sync-s", "1", "--rollout-nice", "5"]
T_PPO_ARGS = ["--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1", "--read-bonus", "0.3", "--read-anneal", "20000000:45000000",
              "--opponent-seats", "setupnet:0.15", "--boosted-row-weight", "2", "--lr-hold-decisions", "357000000",
              "--lr-total-decisions", "600000000", "--fast-update", "--fast-ingest"]

DEFAULT_ARGS = ["--obs-matchup", "--obs-movesets", "--obs-main-line", "--infer-actors", "3", "--infer-sync-s", "1",
                "--rollout-nice", "5"]
WARMUP_ARGS = ["--set-head-coef", "0.1"]
TARGET = 300_000_000
LR = ["--lr", "6e-05", "--lr-hold-decisions", "100000000", "--lr-total-decisions", "300000000"]
QUEUE = ["--learn-queue-episodes", "32", "--lag-stats"]
SEATS = "setupnet:0.15,frozen:0.25"
FROZEN_WEIGHTS = (("m24381", "M24381w.pt", 0.35), ("sfin", "SFIN.pt", 0.25))
FROZEN_PAST = "4:0.40"


def ppo_args(run, rehearsal=False):
    nets = ",".join(f"{name}={run}/league/{fname}:{w:g}" for name, fname, w in FROZEN_WEIGHTS)
    out = ["--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1", "--boosted-row-weight", "2",
           "--opponent-seats", SEATS, "--frozen-nets", nets, "--frozen-past", FROZEN_PAST,
           "--frozen-refresh-min", "5" if rehearsal else "30", "--frozen-past-min-age-min", "2" if rehearsal else "10"]
    out += list(QUEUE)
    if rehearsal:
        out += ["--lr", "6e-05", "--lr-hold-decisions", "1500000", "--lr-total-decisions", "3000000"]
    else:
        out += list(LR)
    return out + ["--fast-update", "--fast-ingest"]


# Run T's run.json as topstart_setup.py wrote it (bench/notes/TOPSTART.md section 3.5, box a ~/runs/v2/topstart/run.json),
# for --base builtin and the tests. The live file is the default base; the two must give the same run.json.
T_LIVE = {
    "repo_dir": "/home/ubuntu/rel/topstart", "python": "/home/ubuntu/venv/bin/python", "showdown_dir": "/home/ubuntu/pokemon-showdown",
    "servers": {"count": 36, "base_port": 10700, "heap_mb": 2048, "ready_timeout_seconds": 420,
                "command": list(ts.FAST_SERVER_COMMAND)},
    "trainer": {"stop_timeout_seconds": 420, "failure_stop_timeout_seconds": 120, "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}},
    "train_defaults": {"battles": 2304, "rooms_per_pair": 16, "minibatch": 768, "seq_len": 256, "grad_accum": 1, "steps_per_update": 36864,
                       "teacher": "psbot", "save_every": 200, "extra_train_args": list(T_DEFAULT_ARGS)},
    "phases": {"imitation": {"skip": True, "min_decisions": 0, "max_decisions": 1},
               "warmup": {"skip": False, "min_decisions": 500000, "max_decisions": 1500000,
                          "plateau": {"metric": "vloss", "window_decisions": 300000, "min_improvement_pct": 2.0, "smooth_updates": 4},
                          "train": {"save_every": 5}, "extra_train_args": ["--set-head-coef", "0.1"]},
               "ppo": {"skip": False, "target_decisions": 600000000, "extra_train_args": list(T_PPO_ARGS)}},
    "watchdog": {"stall_minutes": 10, "startup_grace_minutes": 20, "update_stall_minutes": 30, "max_restarts_per_hour": 6,
                 "backoff_seconds": [30, 120, 300, 600, 900], "max_fruitless_starts": 5, "nan_updates_limit": 3,
                 "telemetry_error_minutes": 5, "recycle_hours": 6},
    "disk": {"min_free_gb": 10},
    "retention": {"enabled": False, "keep_newest": 4, "milestone_every": 500, "keep_hours": 0, "check_minutes": 10},
    "on_done": "idle", "on_failed": "idle",
    "eval": {"every_hours": 4, "games": 200, "opponents": ["simple", "psbot"], "port": 10790,
             "foulplay": {"games": 200, "every_hours": 4, "parallel": 2, "iterations": 88000}},
}

Refusal = ts.Refusal


def build_run_json(base, rel, run, rehearsal=False, base_name="", base_sha="", base_port=None):
    """T's run.json -> the league's (or the rehearsal's). Refusal when the base is not T's."""
    b = copy.deepcopy(base)
    b.pop("_notes", None)
    td = b["train_defaults"]
    if list(td.get("extra_train_args") or []) != T_DEFAULT_ARGS:
        raise Refusal(f"the base's train_defaults.extra_train_args are {td.get('extra_train_args')}, run T's {T_DEFAULT_ARGS}")
    if list(b["phases"]["ppo"].get("extra_train_args") or []) != T_PPO_ARGS:
        raise Refusal(f"the base's PPO arguments are {b['phases']['ppo'].get('extra_train_args')}, run T's {T_PPO_ARGS}")
    if list((b.get("servers") or {}).get("command") or []) != ts.FAST_SERVER_COMMAND:
        raise Refusal("the base's servers.command is not v2-fast's (nice 5)")
    b["repo_dir"] = rel
    port = int(base_port) if base_port is not None else (REHEARSAL_PORT if rehearsal else BASE_PORT)
    b["servers"]["base_port"] = port
    td["extra_train_args"] = list(DEFAULT_ARGS)
    td["save_every"] = 200
    b["phases"]["imitation"] = {"skip": True, "min_decisions": 0, "max_decisions": 1}
    wu = b["phases"]["warmup"]
    wu["extra_train_args"] = list(WARMUP_ARGS)
    ppo = b["phases"]["ppo"]
    ppo["extra_train_args"] = ppo_args(run, rehearsal)
    ppo["target_decisions"] = TARGET
    b["retention"] = dict(b.get("retention") or {}, enabled=False)
    b["on_done"], b["on_failed"] = "idle", "idle"
    if rehearsal:
        b["servers"]["count"] = 6
        td["battles"] = 384
        td["extra_train_args"] = [a if a != "3" else "1" for a in DEFAULT_ARGS]       # --infer-actors 1: one GPU
        b["trainer"]["env"] = {"CUDA_VISIBLE_DEVICES": "0"}
        wu["min_decisions"], wu["max_decisions"] = 100_000, 300_000
        ppo["target_decisions"] = 3_000_000
        ppo["train"] = {"save_every": 10}
        b.pop("eval", None)
        b["_notes"] = {"what this is": "rehearsal of the league run on box c: M24381w -> warmup 100k-300k -> PPO 3M "
                                       "(lr 6e-5, hold 1.5M, total 3M) with the league seats, queue 32 and lag stats. "
                                       "Move it aside when done.", "base": f"{base_name} sha256 {base_sha}"}
    else:
        b["eval"] = dict(b.get("eval") or {}, port=port + 90)
        b["_notes"] = {"what this is": "the league run (bench/notes/LEAGUE.md): M24381 widened on the main line's observation "
                                       "(tools/widen_checkpoint.py --main-line), warmup, PPO 300M against self-play, setupnet "
                                       "seats and frozen nets (M24381, SFIN, 4 past saves), worker queue 32, lr 6e-5 held to "
                                       "100M then down to 1/27 at 300M. cloud/v2/league_setup.py.",
                       "base": f"run T's run.json ({base_name}) sha256 {base_sha}"}
    problem = config_problem(b)
    if problem:
        raise Refusal(problem)
    return b


def config_problem(cfg):
    """What would make the run go wrong, or None (topstart_setup's checks, and the league's)."""
    problem = ts.config_problem(cfg)
    if problem:
        return problem
    every = [list(cfg["train_defaults"].get("extra_train_args") or []) + list(cfg["phases"][p].get("extra_train_args") or [])
             for p in ("warmup", "ppo")]
    if not all("--obs-main-line" in e for e in every):
        return "--obs-main-line must be in every phase (the observation is the run's, not a phase's)"
    if any("--read-bonus" in e for e in every):
        return "the read bonus is not part of the league run"
    return None


def start_problem(ck):
    """Why a checkpoint cannot be the league's start, or None."""
    model = ck.get("model") if isinstance(ck, dict) else None
    if not isinstance(model, dict):
        return "it holds no model"
    if not any(str(k).startswith("set_head.") for k in model):
        return "it has no set_head.* weights (tools/widen_checkpoint.py --main-line adds a fresh SetHead)"
    obs = (ck.get("run_cfg") or {}).get("obs") or {}
    want = {"main_line": True, "matchup": True, "movesets": True, "own_tera_type": False}
    if any(bool(obs.get(k, False)) != v for k, v in want.items()):
        return f"its observation is {obs}, the run's is {want}"
    if not isinstance(ck.get("widened_main_line"), dict):
        return "it is not a widened main-line network (no widened_main_line entry)"
    if ck.get("target_model") is None:
        return "it has no target_model (the EMA network the learner loads)"
    return None


def league_start(ck):
    """The run's first checkpoint from the widened M24381: the same tensors, counters at 0, no optimizer, mode 'imitation'."""
    out = {k: v for k, v in ck.items() if k not in ("optimizer", "scheduler")}
    rc = copy.deepcopy(ck.get("run_cfg") or {})
    learner = rc.setdefault("learner", {})
    was = {"mode": learner.get("mode"), "update_idx": ck.get("update_idx"), "total_steps": ck.get("total_steps"),
           "total_episodes": ck.get("total_episodes"), "had_optimizer": ck.get("optimizer") is not None}
    learner["mode"] = "imitation"
    out["run_cfg"] = rc
    out["update_idx"], out["total_steps"], out["total_episodes"] = 0, 0, 0
    out["league_start"] = dict(was, why="the league run's start: a phase change for warmup (and PPO), the counters of the "
                                        "run from 0 (cloud/v2/league_setup.py league_start)")
    return out


def same_tensors(a, b):
    import torch
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def sfin_problem(ck):
    obs = (ck.get("run_cfg") or {}).get("obs") or {}
    if not (obs.get("matchup") and obs.get("movesets")) or obs.get("main_line"):
        return f"its observation is {obs}: SFIN is a v2 network with both blocks"
    return None


def release_problem(rel):
    problem = ts.release_problem(rel, fast=True)
    if problem:
        return problem
    for f in ("obs_main.py", "frozen_pool.py", "league.py", "obs_abil.py", "obs_tera.py", "switch_names.py",
              "randbats_abilities_gen9.json"):
        if not os.path.isfile(os.path.join(rel, f)):
            return f"{rel} has no {f}: not a league release"
    with open(os.path.join(rel, "train.py"), encoding="utf-8") as fh:
        if "--obs-main-line" not in fh.read():
            return f"{rel}/train.py does not take --obs-main-line"
    return None


def copy_in(src, dst, sidecar=None):
    digest = ts.sha256_of(src)
    side = sidecar or (src + ".sha256")
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            if f.read().split()[0] != digest:
                raise Refusal(f"{src} does not match {side}")
    if os.path.exists(dst):
        raise Refusal(f"{dst} exists (nothing is overwritten)")
    shutil.copyfile(src, dst)                   # the mtime is now
    if ts.sha256_of(dst) != digest:
        raise Refusal(f"the copy {dst} does not match {src}")
    return digest


UNIT = ts.UNIT.replace("ps-ppo v2 top-player start run", "ps-ppo v2 league run")


def unit_texts(rel, run, gpu, py=PY, home=HOME):
    name = "psppo-v2-league"
    train = f"{py} {rel}/pipeline.py --config {run}/run.json --run-dir {run}"
    return {f"{name}-train": UNIT.format(label="", kind="trainer", rel=rel, nice="", gpu=gpu, home=home, run=run,
                                         condition=f"ExecCondition={train} --can-start\n", start=train),
            f"{name}-eval": UNIT.format(label="", kind="evaluator", rel=rel, nice="Nice=5\n", gpu=gpu, home=home, run=run,
                                        condition="", start=f"{py} evaluate.py --run-dir {run} --watch --config {run}/run.json")}


def check_commands(rel, run, py=PY):
    return [f"{py} {rel}/cloud/check_restart.py --config {run}/run.json --run-dir {run} --phase warmup",
            f"{py} {rel}/cloud/check_restart.py --config {run}/run.json --run-dir {run} --phase ppo "
            f"--checkpoint {run}/checkpoints/learner_update_000000.pt"]


def check_run(run):
    """
    --check-run RUN: the PPO phase's frozen seats as the trainer will make them, before it does: run.json's
    PPO arguments through train.py's parser and config builder, the nets read and checked by
    frozen_pool.resolve (train.py does the same at every start and stops on an error), and the pool built
    on the CPU from them (every fixed net loaded strictly into the run's network, the past slots picked from
    RUN/checkpoints). Run it with the release's python from the release directory. Exit 0, or 2 with why.
    """
    with open(os.path.join(run, "run.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    rel = cfg["repo_dir"]
    if os.path.abspath(rel) != os.path.abspath(os.getcwd()):
        print(f"REFUSED: run this from the release {rel} (cd there first)")
        return 2
    sys.path.insert(0, rel)
    import train
    import frozen_pool
    import opponent_seats
    argv = (["--mode", "ppo", "--ckpt-dir", os.path.join(run, "checkpoints")] + list(cfg["train_defaults"].get("extra_train_args") or [])
            + list(cfg["phases"]["ppo"].get("extra_train_args") or []))
    try:
        args = train.parse_args(argv)
        rc = train.build_config(args)
        seats = opponent_seats.from_args(args)
        if seats is None or seats.share("frozen") <= 0:
            print("REFUSED: the PPO phase has no frozen share")
            return 2
        settings = frozen_pool.resolve(seats.frozen, rc)
        print(frozen_pool.describe(settings))
        pool = frozen_pool.FrozenPool(settings, lambda: rc.make_model().eval(), "cpu", mode=str(rc.learner.mode))
        st = pool.stats()
        print(f"pool on the CPU: {st['nets']} nets, past slots {st['past']}, changed {st['changed']}")
        if st["changed"]:
            print("REFUSED: a frozen net changed while it was loaded")
            return 2
        print(f"obs of the run: main_line={rc.obs.main_line} matchup={rc.obs.matchup} movesets={rc.obs.movesets} "
              f"own_tera_type={rc.obs.own_tera_type}")
    except (SystemExit, ValueError, RuntimeError, OSError) as e:
        print(f"REFUSED: {e!r}")
        return 2
    print("frozen seats OK")
    return 0


def main(argv=None):
    if argv is None and len(sys.argv) == 3 and sys.argv[1] == "--check-run":
        return check_run(os.path.abspath(os.path.expanduser(sys.argv[2])))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rel", required=True, help="the league release (run.json's repo_dir)")
    ap.add_argument("--start", required=True, help="M24381 widened (tools/widen_checkpoint.py --main-line)")
    ap.add_argument("--sfin", required=True, help="SFIN, the from-scratch run's final learner_update_016041.pt")
    ap.add_argument("--base", default=BASE, help=f"run T's run.json (default {BASE}); 'builtin' for the copy in this file")
    ap.add_argument("--run", default=None, help=f"the run directory (default {RUN}, with --rehearsal {REHEARSAL})")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--base-port", type=int, default=None)
    ap.add_argument("--gpu", default="0,1", help="the units' CUDA_VISIBLE_DEVICES (T's: 0,1; the trainer's own env has 0,1,2,3)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--units", action="store_true")
    a = ap.parse_args(argv)
    rel = os.path.abspath(os.path.expanduser(a.rel))
    run = os.path.abspath(os.path.expanduser(a.run or (REHEARSAL if a.rehearsal else RUN)))
    start, sfin = (os.path.abspath(os.path.expanduser(p)) for p in (a.start, a.sfin))

    def refuse(msg):
        print(f"REFUSED: {msg}", flush=True)
        return 2

    try:
        if a.base == "builtin":
            base, base_name, base_sha = T_LIVE, "builtin copy", "builtin"
        else:
            with open(os.path.expanduser(a.base), encoding="utf-8") as f:
                base = json.load(f)
            base_name, base_sha = os.path.abspath(os.path.expanduser(a.base)), ts.sha256_of(os.path.expanduser(a.base))
        cfg = build_run_json(base, rel, run, a.rehearsal, base_name, base_sha, base_port=a.base_port)
    except Refusal as e:
        return refuse(str(e))
    problem = release_problem(rel)
    if problem:
        return refuse(problem)
    for path, check in ((start, start_problem), (sfin, sfin_problem)):
        if not os.path.isfile(path):
            return refuse(f"no checkpoint at {path}")
        problem = check(ts.load_start(path))
        if problem:
            return refuse(f"{path}: {problem}")
    problem = ts.run_dir_problem(run, cfg)
    if problem:
        return refuse(problem)
    print(json.dumps({k: cfg[k] for k in ("repo_dir", "servers", "train_defaults", "phases")}, indent=1))
    if not a.write:
        print("(not written: --write)")
        return 0
    dests = [os.path.join(run, "league", n) for n in ("M24381w.pt", "SFIN.pt", "M24381w.pt.sha256", "SFIN.pt.sha256")]
    dests.append(os.path.join(run, "checkpoints", "learner_update_000000.pt"))
    there = [d for d in dests if os.path.exists(d)]
    if there:
        return refuse(f"{there} exist (nothing is overwritten; move the run directory aside to set it up again)")
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run, "league"), exist_ok=True)
    try:
        d_start = copy_in(start, os.path.join(run, "league", "M24381w.pt"))
        d_sfin = copy_in(sfin, os.path.join(run, "league", "SFIN.pt"))
    except Refusal as e:
        return refuse(str(e))
    import torch
    ck = ts.load_start(os.path.join(run, "league", "M24381w.pt"))
    first = league_start(ck)
    dst = os.path.join(run, "checkpoints", "learner_update_000000.pt")      # last: the newest file
    if os.path.exists(dst):
        return refuse(f"{dst} exists (nothing is overwritten)")
    torch.save(first, dst + ".partial")
    os.rename(dst + ".partial", dst)
    back = ts.load_start(dst)
    if not (same_tensors(back["model"], ck["model"]) and same_tensors(back["target_model"], ck["target_model"])):
        return refuse(f"{dst}: its tensors are not the widened M24381's")
    print(f"league start {dst}: model and target_model equal to M24381w's, counters 0, no optimizer, mode 'imitation' "
          f"(was {first['league_start']['mode']!r} at {first['league_start']['total_steps']:,} decisions)")
    for name, digest in (("M24381w.pt", d_start), ("SFIN.pt", d_sfin)):
        with open(os.path.join(run, "league", name + ".sha256"), "w", encoding="utf-8") as f:
            f.write(f"{digest}  {name}\n")
    rj = os.path.join(run, "run.json")
    if not os.path.exists(rj):
        with open(rj + ".partial", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        os.rename(rj + ".partial", rj)
    print(f"M24381w sha256 {d_start} -> {run}/league/M24381w.pt; SFIN {d_sfin} -> {run}/league/SFIN.pt")
    print("check before the start (each must exit 0):")
    for c in check_commands(rel, run):
        print("  " + c)
    if a.units and not a.rehearsal:
        for name, text in unit_texts(rel, run, a.gpu).items():
            subprocess.run(["sudo", "tee", f"/etc/systemd/system/{name}.service"], input=text, text=True,
                           stdout=subprocess.DEVNULL, check=True)
            print(f"wrote /etc/systemd/system/{name}.service (not enabled, not started)")
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
