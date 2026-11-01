#!/usr/bin/env python3
"""
Run T of the v2 plan ("topstart", owner's ask 2026-10-18): the from-scratch run S again with ONE
change, its start. S went random weights -> PSBot imitation (12M decisions) -> warmup -> PPO 600M.
T goes random weights -> offline imitation of 2100+ players (human/topstart.py train, no PSBot
anywhere) -> warmup -> PPO 600M, with S's PPO settings and the learning-rate schedule S actually
ended up with (3e-4 held to 357.0M decisions, then down to 1/27 at 600M), counted in decisions so
it lands where planned without a hand fix (train.py --lr-hold-decisions / --lr-total-decisions).

    python3 topstart_setup.py --rel ~/rel/topstart-<sha> --start <T0.pt> --fast [--write] [--units]
    python3 topstart_setup.py --rel ~/rel/topstart-<sha> --start <T0.pt> --fast --rehearsal [--rehearsal-full] [--base-port N] [--write] [--units]
    python3 topstart_setup.py --milestones          (what T against S measures at each milestone)

The release is branch topstart (topstart-lr-decisions + v2-fast + this branch), built over a copy of
~/rel/v2-humanaux (THE RELEASE below). On a release without v2-fast, leave out --fast.

run.json is S's live one (--base, default ~/runs/v2/scratch/run.json) with:
  * repo_dir the topstart release; servers from port 10700, the evaluator on 10790 (S: 10600, 10690);
  * imitation skipped (the start is the imitation; min 0, max 1);
  * warmup as S, plus --set-head-coef 0.1: without it the learner's strict=False load drops the
    start's set_head.* without a word (the aux zone is frozen in warmup, so the head reaches PPO as
    the start has it);
  * PPO with S's final arguments minus its hand-fixed step schedule (--lr-hold-steps 479713
    --lr-total-steps 1113566), plus --lr-hold-decisions 357000000 --lr-total-decisions 600000000;
  * save_every 200 (7.4M decisions, about 19 GB for the run); retention off (it deletes).

--start: the checkpoint is copied to RUN/checkpoints/learner_update_000000.pt (the newest by mtime:
the learner resumes from it; its mode "imitation" is not warmup's, so the learner starts the
optimizer and the schedule afresh with its counters at 0) and to RUN/topstart/ with its sha256.

Refused (exit 2): an existing RUN/state.json, run.json that differs, or checkpoint in RUN/checkpoints;
a start without set_head.* keys, with an observation other than the three settings on, or of a mode
other than imitation; a warmup phase without the PPO phase's --set-head-coef; a release whose train.py
does not take --lr-hold-decisions (the trainer would refuse the PPO phase at every start, after warmup);
a base whose arguments are not the ones planned.

--rehearsal: ~/runs/v2/topstart-rehearsal, 4 servers from port 10800, warmup 100k to 200k, PPO to 2M
with hold 1M and total 2M, no evaluator. --rehearsal --rehearsal-full: the real run's size instead (S's
36 servers and 2,304 battles, so the throughput is the real run's), warmup 100k to 200k, PPO to 745k
decisions (about 20 updates) with hold 300k and total 600k (full rate for about 8 updates, the anneal,
then 1/27 for the last 3 or 4), PPO saves every 10. --base-port N puts the servers at N (the evaluator
at N+90); the rehearsal of 2026-10-18 on box a ran at 11700. --units writes psppo-v2-topstart-train and
-eval (S's unit template; the rehearsal: psppo-v2-topstart-rehearsal-train only). What the rehearsal can show from its
files: the start's set_head.* equal to the warmup's final checkpoint's (the aux zone is frozen in
warmup; set_head_drift below). Its PPO saves come after the aux zone has trained at 1.0 x lr, so there
the head has moved by design: read the trainer log's "PHASE CHANGE DETECTED (warmup -> ppo)" and check
that update 10's head is near the start's (small drift) and far from a fresh init.

--fast (v2-fast, bench/notes/SPEED.md): the Showdown servers behind `nice -n 5` (servers.command,
pipeline's default command with that prefix), --rollout-nice 5 in train_defaults (every phase that
runs: Ray's workers start at nice 0, so the learner and the inference actors stay at 0 across every
restart, and each rollout worker goes to 5), --fast-update --fast-ingest in the PPO phase (as
SPEED.md section 5 deploys them). The same updates bit for bit, about 25-35% faster. A release whose
train.py takes these flags must be set up with --fast or --no-fast (refused otherwise); --fast with a
release that does not take them is refused. Without --fast, renice the learner and the inference
actors by hand after every trainer start (cloud/v2/RECOVERY.md): the Ray workers start at nice 15.

THE RELEASE. Build it the way every earlier release was built: copy ~/rel/v2-humanaux first, then
put the archive over it. A git archive alone has no heuristic/cache/ (.gitignore): the read tracker
behind --read-bonus (T's first 45M decisions) reads its set table from
heuristic/cache/gen9randombattle.json and quietly uses an empty one without it (matchup.py
load_randbats_sets), which changes the read tiers; PSBot's engine (evaluate.py's psbot cell) needs
heuristic/cache/battle.js. release_problem refuses a release without both.

    A=$(bash C:/Users/you/ps-ppo-v2/cloud/v2/boxssh.sh a)
    $A 'test ! -e ~/rel/topstart-<sha> && cp -a ~/rel/v2-humanaux ~/rel/topstart-<sha>'
    git -C <worktree of branch topstart> archive --format=tar topstart | $A 'tar -x -C ~/rel/topstart-<sha>'

THE COMPARISON WITH S (--milestones prints it). T against S is one variable, the start, only where S
had what T has. S ran PPO updates 0 to 1,649 (0 to 60.9M decisions) without the moveset block
(release v2-0bf9e93, --obs-matchup only; its SetHead had no move or Tera labels) and was widened at
update 1,650; T has the block from its start. S also had the PPO-time top-player imitation term from
356.9M to 374.9M. So a milestone reads "start + moveset block" until S had the block for at least half
of its PPO decisions (121.8M), and "start" from there on: B3 (25M) and B4 (100M) measure the start
and the moveset block together and cannot be credited to the start; B5 (150M, S with the block for
89M of 150M) is the first that reads the start; B6 (300M) and B7 (600M) read the start (B7 with S's
18M of the imitation term noted). An early read of the start alone needs a control from P0 widened
(tools/widen_checkpoint.py --movesets) with the block on from PPO step 0.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
PY = f"{HOME}/venv/bin/python"
RUN = f"{HOME}/runs/v2/topstart"
REHEARSAL = f"{HOME}/runs/v2/topstart-rehearsal"
BASE = f"{HOME}/runs/v2/scratch/run.json"

DEFAULT_ARGS = ["--obs-matchup", "--obs-movesets", "--infer-actors", "3", "--infer-sync-s", "1"]
WARMUP_ARGS = ["--set-head-coef", "0.1"]
PPO_ARGS = ["--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1",
            "--read-bonus", "0.3", "--read-anneal", "20000000:45000000",
            "--opponent-seats", "setupnet:0.15", "--boosted-row-weight", "2",
            "--lr-hold-decisions", "357000000", "--lr-total-decisions", "600000000"]
REHEARSAL_LR = ["--lr-hold-decisions", "1000000", "--lr-total-decisions", "2000000"]
# --rehearsal-full: the real run's size; about 20 PPO updates of about 37k decisions, the anneal inside them.
REHEARSAL_FULL_PPO_TARGET = 745_000
REHEARSAL_FULL_LR = ["--lr-hold-decisions", "300000", "--lr-total-decisions", "600000"]
STEP_SCHEDULE_FLAGS = ("--lr-hold-steps", "--lr-total-steps")
DECISION_FLAGS = ("--lr-hold-decisions", "--lr-total-decisions")

# --fast (v2-fast, bench/notes/SPEED.md section 5): the servers behind nice 5 (pipeline.build_server_command's
# default command with the prefix), every phase's rollout workers at 5 with the learner at 0, the two learner flags in PPO.
FAST_NICE = 5
FAST_SERVER_COMMAND = ["nice", "-n", str(FAST_NICE), "{python}", "start_servers.py", "--dir", "{showdown_dir}", "--ports", "{ports}",
                       "--heap-mb", "{heap_mb}", "--log-dir", "{server_log_dir}"]
FAST_DEFAULT_ARGS = ["--rollout-nice", str(FAST_NICE)]
FAST_PPO_ARGS = ["--fast-update", "--fast-ingest"]
FAST_TRAIN_FLAGS = ("--fast-update", "--fast-ingest", "--rollout-nice")

# What a release must hold beyond its git archive (.gitignore has heuristic/cache/): the read tracker's set table
# (read_reward.ReadTracker -> matchup.load_randbats_sets, {} when the file is missing) and PSBot's engine files.
RELEASE_CACHE = ("heuristic/cache/gen9randombattle.json", "heuristic/cache/battle.js")
MIN_SET_SPECIES = 100            # today's gen9 random battle table has about 500 species

# Run S's PPO history (box a ~/runs/v2/scratch: metrics.jsonl, the trainer logs, pre_movesets/run.json; read 2026-10-18).
# What S had besides its start, by PPO decisions; T has the moveset block from its start and no imitation term.
S_MOVESETS_FROM = 60_896_992    # the learner's total_steps at the resume from update 1,650 (segment 8, 17:27 10-16), where
                                #   S was widened; before it S ran --obs-matchup only (the supervisor's count there: 60,860,045)
S_HUMAN_TERM = (356_910_111, 374_867_648)       # updates 9,607 to 10,080: --human-imitation-coef 0.012
MILESTONES = (0, 25_000_000, 100_000_000, 150_000_000, 300_000_000, 600_000_000)
BARS = {0: "B2 (gate G1)", 25_000_000: "B3", 100_000_000: "B4", 150_000_000: "B5 (stop rule)", 300_000_000: "B6",
        600_000_000: "B7"}

# S's run.json as it was when S finished (2026-10-18, box a ~/runs/v2/scratch/run.json), for --base builtin
# and the tests. The live file is the default base; the two must give the same run.json.
S_LIVE = {
    "repo_dir": "/home/ubuntu/rel/v2-humanaux", "python": "/home/ubuntu/venv/bin/python",
    "showdown_dir": "/home/ubuntu/pokemon-showdown",
    "servers": {"count": 36, "base_port": 10600, "heap_mb": 2048, "ready_timeout_seconds": 420},
    "trainer": {"stop_timeout_seconds": 420, "failure_stop_timeout_seconds": 120, "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}},
    "train_defaults": {"battles": 2304, "rooms_per_pair": 16, "minibatch": 768, "seq_len": 256, "grad_accum": 1,
                       "steps_per_update": 36864, "teacher": "psbot", "save_every": 50,
                       "extra_train_args": ["--obs-matchup", "--obs-movesets", "--infer-actors", "3", "--infer-sync-s", "1"]},
    "phases": {
        "imitation": {"skip": False, "min_decisions": 12000000, "max_decisions": 12000000},
        "warmup": {"skip": False, "min_decisions": 500000, "max_decisions": 1500000,
                   "plateau": {"metric": "vloss", "window_decisions": 300000, "min_improvement_pct": 2.0, "smooth_updates": 4},
                   "train": {"save_every": 5}},
        "ppo": {"skip": False, "target_decisions": 600000000,
                "extra_train_args": ["--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1", "--read-bonus", "0.3",
                                     "--read-anneal", "20000000:45000000", "--opponent-seats", "setupnet:0.15",
                                     "--boosted-row-weight", "2", "--lr-hold-steps", "479713", "--lr-total-steps", "1113566"]}},
    "watchdog": {"stall_minutes": 10, "startup_grace_minutes": 20, "update_stall_minutes": 30, "max_restarts_per_hour": 6,
                 "backoff_seconds": [30, 120, 300, 600, 900], "max_fruitless_starts": 5, "nan_updates_limit": 3,
                 "telemetry_error_minutes": 5, "recycle_hours": 6},
    "disk": {"min_free_gb": 10},
    "retention": {"enabled": False, "keep_newest": 4, "milestone_every": 500, "keep_hours": 0, "check_minutes": 10},
    "on_done": "idle", "on_failed": "idle",
    "eval": {"every_hours": 4, "games": 200, "opponents": ["simple", "psbot"], "port": 10690,
             "foulplay": {"games": 200, "every_hours": 4, "parallel": 2, "iterations": 88000}},
    "_notes": {"what this is": "v2 from-scratch run (2026-10-16)"},
}


class Refusal(Exception):
    pass


def comparison_with_s(d):
    """What T at d PPO decisions against S at d measures: {"reads": "start" or "start + moveset block", "differs": [...]}.

    One variable (the start) only where S had the moveset block for at least half of its PPO decisions; before that
    a gap between T and S belongs to the start and the block together. At 0 (gate G1) T0 is compared with P0, which
    has no moveset block either."""
    d = int(d)
    differs = []
    block = max(0, d - S_MOVESETS_FROM)
    if d == 0:
        differs.append("moveset block: T0 has it, P0 does not (S was widened in PPO, at 60.9M)")
    elif block < d:
        differs.append(f"moveset block: T {d / 1e6:.1f}M PPO decisions with it (and its start), S {block / 1e6:.1f}M "
                       f"(S widened at {S_MOVESETS_FROM / 1e6:.1f}M)")
    lo, hi = S_HUMAN_TERM
    if d > lo:
        differs.append(f"S had the PPO-time top-player imitation term for {(min(d, hi) - lo) / 1e6:.1f}M decisions "
                       f"({lo / 1e6:.1f}M to {hi / 1e6:.1f}M); T has none")
    reads = "start" if (d > 0 and 2 * block >= d) else "start + moveset block"
    return {"decisions": d, "bar": BARS.get(d, ""), "reads": reads, "differs": differs}


def milestone_table():
    return [comparison_with_s(m) for m in MILESTONES]


def set_head_drift(a, b):
    """The largest absolute difference between two state dicts' set_head.* tensors (None when either has none)."""
    keys = sorted(k for k in a if str(k).startswith("set_head."))
    if not keys or any(k not in b for k in keys):
        return None
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in keys)


def without_step_schedule(args):
    """The arguments with every --lr-hold-steps N / --lr-total-steps N pair taken out."""
    out, skip = [], False
    for k, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in STEP_SCHEDULE_FLAGS:
            skip = True
            continue
        if any(a.startswith(f + "=") for f in STEP_SCHEDULE_FLAGS):
            continue
        out.append(a)
    return out


def without_fast(args):
    """The arguments without v2-fast's --fast-update, --fast-ingest and --rollout-nice N."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--fast-update", "--fast-ingest") or a.startswith("--rollout-nice="):
            continue
        if a == "--rollout-nice":
            skip = True
            continue
        out.append(a)
    return out


def flag_value(args, flag):
    for k, a in enumerate(args):
        if a == flag and k + 1 < len(args):
            return args[k + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def build_run_json(base, rel, rehearsal=False, base_name="", base_sha="", fast=False, base_port=None, rehearsal_full=False):
    """S's run.json -> T's (or the rehearsal's). Refusal when the base is not what the plan was made on.
    fast: v2-fast's three settings on (servers.command behind nice 5, --rollout-nice 5 in train_defaults, --fast-update
    --fast-ingest in PPO); off, the run.json has none of them.
    base_port: the servers' first port (default 10700, the rehearsal 10800); the evaluator at base_port + 90.
    rehearsal_full: the rehearsal at the real run's size (servers, battles), about 20 PPO updates with the anneal in them."""
    if rehearsal_full and not rehearsal:
        raise Refusal("--rehearsal-full is a kind of rehearsal: give --rehearsal too")
    b = copy.deepcopy(base)
    td = b["train_defaults"]
    # A base with v2-fast deployed (SPEED.md section 5 edits S's run.json): its settings out, then in again by `fast`.
    command = (b.get("servers") or {}).get("command")
    if command is not None and list(command) != FAST_SERVER_COMMAND:
        raise Refusal(f"the base's servers.command is {command}, neither the default nor v2-fast's {FAST_SERVER_COMMAND}")
    b["servers"].pop("command", None)
    td["extra_train_args"] = without_fast(list(td.get("extra_train_args") or []))
    b["phases"]["ppo"]["extra_train_args"] = without_fast(list(b["phases"]["ppo"].get("extra_train_args") or []))
    if list(td.get("extra_train_args") or []) != DEFAULT_ARGS:
        raise Refusal(f"the base's train_defaults.extra_train_args are {td.get('extra_train_args')}, the plan's {DEFAULT_ARGS}")
    ppo = b["phases"]["ppo"]
    s_args = without_step_schedule(list(ppo.get("extra_train_args") or []))
    if s_args + PPO_ARGS[-4:] != PPO_ARGS:
        raise Refusal(f"the base's PPO arguments without the step schedule are {s_args}, the plan's {PPO_ARGS[:-4]}")
    b["repo_dir"] = rel
    b["servers"]["base_port"] = int(base_port) if base_port is not None else (10800 if rehearsal else 10700)
    td["save_every"] = 200
    b["phases"]["imitation"] = {"skip": True, "min_decisions": 0, "max_decisions": 1}
    wu = b["phases"]["warmup"]
    wu["extra_train_args"] = list(WARMUP_ARGS)
    ppo["extra_train_args"] = list(PPO_ARGS)
    ppo["target_decisions"] = 600_000_000
    b["retention"] = dict(b.get("retention") or {}, enabled=False)
    b["on_done"], b["on_failed"] = "idle", "idle"
    if rehearsal and rehearsal_full:
        # the real run's servers and battles (S's), so the rehearsal's decisions a second are the real run's
        wu["min_decisions"], wu["max_decisions"] = 100_000, 200_000
        ppo["target_decisions"] = REHEARSAL_FULL_PPO_TARGET
        ppo["extra_train_args"] = list(PPO_ARGS[:-4]) + list(REHEARSAL_FULL_LR)
        ppo["train"] = {"save_every": 10}
        b.pop("eval", None)
        b["_notes"] = {"what this is": "rehearsal of run T (topstart) at the real run's size: the start -> warmup 100k-200k "
                                       "-> PPO about 20 updates (745k decisions) with the decisions schedule (hold 300k, "
                                       "total 600k). Move it aside when done.",
                       "base": f"{base_name} sha256 {base_sha}"}
    elif rehearsal:
        b["servers"]["count"] = 4
        td["battles"] = 64 * 4
        wu["min_decisions"], wu["max_decisions"] = 100_000, 200_000
        ppo["target_decisions"] = 2_000_000
        ppo["extra_train_args"] = list(PPO_ARGS[:-4]) + list(REHEARSAL_LR)
        ppo["train"] = {"save_every": 10}
        b.pop("eval", None)
        b["_notes"] = {"what this is": "rehearsal of run T (topstart): the start -> warmup 100k-200k -> PPO 2M with the "
                                       "decisions schedule (hold 1M, total 2M). Move it aside when done.",
                       "base": f"{base_name} sha256 {base_sha}"}
    else:
        b["eval"] = dict(b.get("eval") or {}, port=(int(base_port) + 90) if base_port is not None else 10790)
        b["_notes"] = {"what this is": "v2 top-player start (run T): human/topstart.py offline imitation of 2100+ players "
                                       "(sha256 in RUN/topstart), warmup, PPO 600M as S with the LR schedule in decisions "
                                       "(3e-4 to 357.0M, 1/27 at 600M). cloud/v2/topstart_setup.py.",
                       "base": f"S's run.json ({base_name}) sha256 {base_sha}",
                       "comparison with S": [f"{m['decisions'] / 1e6:.0f}M {m['bar']}: reads {m['reads']}"
                                             + (f" ({'; '.join(m['differs'])})" if m["differs"] else "")
                                             for m in milestone_table()]}
    if fast:
        b["servers"]["command"] = list(FAST_SERVER_COMMAND)
        td["extra_train_args"] = list(td["extra_train_args"]) + list(FAST_DEFAULT_ARGS)
        ppo["extra_train_args"] = list(ppo["extra_train_args"]) + list(FAST_PPO_ARGS)
        b["_notes"]["fast"] = ("v2-fast (bench/notes/SPEED.md): servers behind nice 5, --rollout-nice 5 in every phase, "
                               "--fast-update --fast-ingest in PPO; the same updates bit for bit")
    problem = config_problem(b)
    if problem:
        raise Refusal(problem)
    return b


def config_problem(cfg):
    """What would make the run go wrong, or None."""
    ph = cfg["phases"]
    ppo_sh = flag_value(ph["ppo"].get("extra_train_args") or [], "--set-head-coef")
    wu_sh = flag_value(ph["warmup"].get("extra_train_args") or [], "--set-head-coef")
    if ppo_sh is not None and float(ppo_sh) > 0 and (wu_sh is None or float(wu_sh) != float(ppo_sh)):
        return (f"the PPO phase has --set-head-coef {ppo_sh} and the warmup phase {wu_sh}: the warmup learner would drop the "
                f"start's set_head.* at its strict=False load and PPO would train a fresh head")
    if not ph["imitation"].get("skip"):
        return "the imitation phase is not skipped: the start IS the imitation"
    args = ph["ppo"].get("extra_train_args") or []
    if any(f in args for f in STEP_SCHEDULE_FLAGS) or not all(f in args for f in DECISION_FLAGS):
        return "the PPO phase must have the decisions schedule (--lr-hold-decisions, --lr-total-decisions) and no step schedule"
    if (cfg.get("retention") or {}).get("enabled"):
        return "retention is on (it deletes checkpoints)"
    command = (cfg.get("servers") or {}).get("command")
    nice_servers = bool(command) and list(command[:2]) == ["nice", "-n"]
    for name in ("warmup", "ppo"):
        every = list(cfg["train_defaults"].get("extra_train_args") or []) + list(ph[name].get("extra_train_args") or [])
        if nice_servers and "--rollout-nice" not in every:
            return (f"the servers run behind nice but the {name} phase has no --rollout-nice: its learner would start at "
                    f"Ray's nice 15, below the servers")
        if "--rollout-nice" in every and not nice_servers:
            return f"the {name} phase has --rollout-nice but servers.command does not start the servers behind nice"
    return None


def start_problem(ck):
    """Why a checkpoint cannot be T's start, or None."""
    model = ck.get("model") if isinstance(ck, dict) else None
    if not isinstance(model, dict):
        return "it holds no model"
    if not any(str(k).startswith("set_head.") for k in model):
        return "it has no set_head.* weights (the SetHead the start trained)"
    obs = (ck.get("run_cfg") or {}).get("obs") or {}
    if not all(obs.get(k) is True for k in ("own_tera_type", "matchup", "movesets")):
        return f"its observation is {obs}, the run's has own_tera_type, matchup and movesets on"
    mode = ((ck.get("run_cfg") or {}).get("learner") or {}).get("mode")
    if mode != "imitation":
        return f"its mode is {mode!r}, not 'imitation' (warmup must see a phase change to start its optimizer afresh)"
    if ck.get("optimizer") is not None or ck.get("scheduler") is not None:
        return "it carries an optimizer or a scheduler state (the start has none)"
    return None


def release_has_fast(rel):
    """True when the release's train.py takes v2-fast's three flags."""
    try:
        with open(os.path.join(rel, "train.py"), encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return False
    return all(f in src for f in FAST_TRAIN_FLAGS)


def cache_problem(rel):
    """What the release lacks of heuristic/cache (a git archive has none of it), or None."""
    sets_path, battle = (os.path.join(rel, p) for p in RELEASE_CACHE)
    how = ("build the release as the earlier ones were: cp -a ~/rel/v2-humanaux ~/rel/topstart-<sha>, then the git archive "
           "over it (topstart_setup.py's docstring)")
    try:
        with open(sets_path, encoding="utf-8") as f:
            sets = json.load(f)
    except (OSError, ValueError) as e:
        return (f"{sets_path} cannot be read ({type(e).__name__}): the read tracker behind --read-bonus would use an empty "
                f"set table without a word and change the read tiers; {how}")
    if not isinstance(sets, dict) or len(sets) < MIN_SET_SPECIES:
        n = len(sets) if isinstance(sets, dict) else type(sets).__name__
        return f"{sets_path} holds {n} species, not a random-battle set table; {how}"
    if not os.path.isfile(battle):
        return f"{battle} is missing: PSBot's engine (evaluate.py's psbot cell) would download the client files; {how}"
    return None


def release_problem(rel, fast=False):
    train = os.path.join(rel, "train.py")
    if not os.path.isfile(train):
        return f"{rel} has no train.py"
    with open(train, encoding="utf-8") as f:
        src = f.read()
    if "--lr-hold-decisions" not in src:
        return (f"{train} does not take --lr-hold-decisions: the trainer would refuse the PPO phase with exit code 2 at every "
                f"start (after warmup). Release a branch with the decisions schedule (topstart-lr) merged.")
    if not os.path.isfile(os.path.join(rel, "infer_pool.py")):
        return f"{rel} has no infer_pool.py (--infer-actors)"
    if fast and not (release_has_fast(rel) and os.path.isfile(os.path.join(rel, "fast_ingest.py"))):
        return f"--fast, but {rel} is not a v2-fast release (train.py without {', '.join(FAST_TRAIN_FLAGS)}, or no fast_ingest.py)"
    return cache_problem(rel)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir_problem(run, cfg):
    if os.path.exists(os.path.join(run, "state.json")):
        return f"{run}/state.json exists: this run directory has been started"
    rj = os.path.join(run, "run.json")
    if os.path.exists(rj):
        with open(rj, encoding="utf-8") as f:
            if json.load(f) != cfg:
                return f"{rj} exists and differs (nothing is overwritten; move it aside)"
    ck = os.path.join(run, "checkpoints")
    if os.path.isdir(ck) and any(n.startswith("learner_update_") for n in os.listdir(ck)):
        return f"{ck} already holds checkpoints"
    return None


def copy_start(start, run):
    """The start into RUN/checkpoints/learner_update_000000.pt (mtime now) and RUN/topstart/, both checked by sha256."""
    digest = sha256_of(start)
    side = start + ".sha256"
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            want = f.read().split()[0]
        if want != digest:
            raise Refusal(f"{start} does not match {side}")
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run, "topstart"), exist_ok=True)
    dst = os.path.join(run, "checkpoints", "learner_update_000000.pt")
    keep = os.path.join(run, "topstart", os.path.basename(start))
    for d in (keep, dst):
        if os.path.exists(d):
            raise Refusal(f"{d} exists (nothing is overwritten)")
    shutil.copyfile(start, keep)
    shutil.copyfile(start, dst)              # copyfile: the mtime is now, so it is the newest checkpoint
    for d in (keep, dst):
        if sha256_of(d) != digest:
            raise Refusal(f"the copy {d} does not match {start}")
    with open(keep + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{digest}  {os.path.basename(start)}\n")
    for extra in (start + ".json",):
        if os.path.exists(extra):
            shutil.copyfile(extra, os.path.join(run, "topstart", os.path.basename(extra)))
    return digest


UNIT = """[Unit]
Description=ps-ppo v2 top-player start run{label}: {kind}
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


def unit_texts(rel, run, gpu, rehearsal=False, py=PY, home=HOME):
    """{service name: text}, S's template (cloud/v2/scratch_setup.py UNIT) with T's names and paths."""
    name = "psppo-v2-topstart-rehearsal" if rehearsal else "psppo-v2-topstart"
    label = " (rehearsal)" if rehearsal else ""
    train = f"{py} {rel}/pipeline.py --config {run}/run.json --run-dir {run}"
    out = {f"{name}-train": UNIT.format(label=label, kind="trainer", rel=rel, nice="", gpu=gpu, home=home, run=run,
                                        condition=f"ExecCondition={train} --can-start\n", start=train)}
    if not rehearsal:
        out[f"{name}-eval"] = UNIT.format(label=label, kind="evaluator", rel=rel, nice="Nice=5\n", gpu=gpu, home=home, run=run,
                                          condition="", start=f"{py} evaluate.py --run-dir {run} --watch --config {run}/run.json")
    return out


def check_commands(rel, run, py=PY):
    return [f"{py} {rel}/cloud/check_restart.py --config {run}/run.json --run-dir {run} --phase warmup",
            f"{py} {rel}/cloud/check_restart.py --config {run}/run.json --run-dir {run} --phase ppo "
            f"--checkpoint {run}/checkpoints/learner_update_000000.pt"]


def load_start(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rel", help="the topstart release, run.json's repo_dir: a copy of ~/rel/v2-humanaux with the git "
                                  "archive of branch topstart over it (THE RELEASE above; a bare archive has no heuristic/cache)")
    ap.add_argument("--start", help="human/topstart.py train's --out (T0)")
    fast = ap.add_mutually_exclusive_group()
    fast.add_argument("--fast", action="store_true", help="v2-fast's settings on (servers behind nice 5, --rollout-nice 5, "
                                                          "--fast-update --fast-ingest in PPO); needs a v2-fast release")
    fast.add_argument("--no-fast", action="store_true", help="a v2-fast release without its settings (hand renice after "
                                                             "every trainer start)")
    ap.add_argument("--milestones", action="store_true", help="print what each milestone's T against S measures, and exit")
    ap.add_argument("--base", default=BASE, help=f"S's run.json (default {BASE}); 'builtin' for the copy in this file")
    ap.add_argument("--run", default=None, help=f"the run directory (default {RUN}, with --rehearsal {REHEARSAL})")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--rehearsal-full", action="store_true", help="with --rehearsal: the real run's servers and battles, "
                                                                  "about 20 PPO updates with the anneal in them")
    ap.add_argument("--base-port", type=int, default=None, help="the servers' first port (default 10700; the rehearsal "
                                                                "10800); the evaluator at this + 90")
    ap.add_argument("--gpu", default="0,1", help="the units' CUDA_VISIBLE_DEVICES (S's: 0,1; the trainer's own env has 0,1,2,3)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--no-release-check", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.milestones:
        for m in milestone_table():
            print(f"{m['decisions'] / 1e6:>5.0f}M  {m['bar']:<15} reads: {m['reads']}")
            for x in m["differs"]:
                print(f"{'':24}{x}")
        return 0
    if not a.rel or not a.start:
        ap.error("--rel and --start are required")
    rel = os.path.abspath(os.path.expanduser(a.rel))
    run = os.path.abspath(os.path.expanduser(a.run or (REHEARSAL if a.rehearsal else RUN)))
    start = os.path.abspath(os.path.expanduser(a.start))

    def refuse(msg):
        print(f"REFUSED: {msg}", flush=True)
        return 2

    try:
        if a.base == "builtin":
            base, base_name, base_sha = S_LIVE, "builtin copy", "builtin"
        else:
            with open(os.path.expanduser(a.base), encoding="utf-8") as f:
                base = json.load(f)
            base_name, base_sha = os.path.abspath(os.path.expanduser(a.base)), sha256_of(os.path.expanduser(a.base))
        cfg = build_run_json(base, rel, a.rehearsal, base_name, base_sha, fast=a.fast, base_port=a.base_port,
                             rehearsal_full=a.rehearsal_full)
    except Refusal as e:
        return refuse(str(e))
    if not a.no_release_check:
        problem = release_problem(rel, fast=a.fast)
        if problem:
            return refuse(problem)
        if release_has_fast(rel) and not (a.fast or a.no_fast):
            return refuse(f"{rel} is a v2-fast release: say --fast (its settings on, bench/notes/SPEED.md) or --no-fast")
    if not os.path.isfile(start):
        return refuse(f"no start checkpoint at {start}")
    problem = start_problem(load_start(start))
    if problem:
        return refuse(f"{start}: {problem}")
    problem = run_dir_problem(run, cfg)
    if problem:
        return refuse(problem)
    print(json.dumps({k: cfg[k] for k in ("repo_dir", "servers", "train_defaults", "phases")}, indent=1))
    if not a.write:
        print("(not written: --write)")
        return 0
    os.makedirs(run, exist_ok=True)
    try:
        digest = copy_start(start, run)
    except Refusal as e:
        return refuse(str(e))
    rj = os.path.join(run, "run.json")
    if not os.path.exists(rj):
        with open(rj + ".partial", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        os.rename(rj + ".partial", rj)
    print(f"start {os.path.basename(start)} sha256 {digest} -> {run}/checkpoints/learner_update_000000.pt and {run}/topstart/")
    print("check before the start (both must say OK / exit 0; the first: the checkpoint is of phase 'imitation' and the start "
          "is 'warmup'):")
    for c in check_commands(rel, run):
        print("  " + c)
    if not a.fast:
        print("after every trainer start (no --fast): renice the learner and the inference actors (RECOVERY.md)")
    if a.units:
        for name, text in unit_texts(rel, run, a.gpu, a.rehearsal).items():
            subprocess.run(["sudo", "tee", f"/etc/systemd/system/{name}.service"], input=text, text=True,
                           stdout=subprocess.DEVNULL, check=True)
            print(f"wrote /etc/systemd/system/{name}.service (not enabled, not started)")
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
