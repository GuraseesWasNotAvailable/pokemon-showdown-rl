#!/usr/bin/env python3
"""
The imitation test's judge and the full run's launcher (runs ON THE BOX, under systemd-run).

1. Waits for both arms: arm A (~/runs/armA, pipeline imitation phase done) and arm B
   (~/runs/armB/checkpoints/learner_update_N.pt written by human/imitate.py, its trainer gone).
2. Evaluates both final checkpoints the same way, side by side: the policy alone, best legal action,
   400 games against SimpleHeuristics and 800 against Foul Play at 200 ms (evaluate.py).
3. The rule fixed before any game (owner, 2026-10-14): arm B starts the full run only if its Foul Play
   win rate is at least 8 points above arm A's AND a two-proportion z-test gives p < 0.01 (two-sided);
   otherwise arm A does. Arm C (added 04:35 UTC, before any evaluation game, when arm B's policy was
   seen stuck at 24% held-out agreement from random weights while arm A's network reached 50% after
   0.6 passes of the same loop): arm A's final network fine-tuned on the top players (human/imitate.py
   --init), held to the SAME bar against arm A. If B and C both clear it, the higher Foul Play win
   rate starts the full run. A row with more than 5% errors is not trusted: that arm is evaluated again
   once, and if it is still not trusted the run starts from arm A and the note says why.
4. Starts the full run from the winner: ~/runs/full, imitation skipped, warmup (value head), PPO to
   600M decisions, on the baseline release, units psppo-full-train and psppo-full-eval.

Everything it decides goes to ~/runs/imtest/decision.json and DECISION.md.
"""
import glob
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time

HOME = "/home/ubuntu"
REL = f"{HOME}/rel/baseline-c988471"
PY = f"{HOME}/venv/bin/python"
A_RUN, B_RUN, C_RUN, FULL = f"{HOME}/runs/armA", f"{HOME}/runs/armB", f"{HOME}/runs/armC", f"{HOME}/runs/full"
CODE_B = f"{HOME}/dev/armb-code"
OUT = f"{HOME}/runs/imtest"
FP_GAMES, SIMPLE_GAMES, MARGIN, ALPHA = 800, 400, 0.08, 0.01
PPO_TARGET = 600_000_000
os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/orchestrator.log", "a", buffering=1)


def log(*a):
    LOG.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + " ".join(str(x) for x in a) + "\n")


def arm_a_final():
    try:
        st = json.load(open(f"{A_RUN}/state.json"))
    except (OSError, ValueError):
        return None
    ph = (st.get("phases") or {}).get("imitation") or {}
    if ph.get("status") != "done" or not ph.get("final_checkpoint"):
        return None
    path = f"{A_RUN}/checkpoints/{ph['final_checkpoint']}"
    return path if os.path.exists(path) else None


def arm_b_final():
    files = sorted(glob.glob(f"{B_RUN}/checkpoints/learner_update_*.pt"))
    if not files:
        return None
    busy = subprocess.run(["pgrep", "-f", "--", f"--out-dir {B_RUN}/checkpoints"], capture_output=True, text=True).stdout.strip()
    return None if busy else files[-1]


def train_arm_c(a_final):
    """Arm C: arm A's final network fine-tuned on the top players' games (the same loop as arm B)."""
    os.makedirs(f"{C_RUN}/checkpoints", exist_ok=True)
    os.makedirs(f"{C_RUN}/work", exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="5", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    cmd = [PY, "human/imitate.py", "train", "--init", a_final, "--data", f"{HOME}/dev/armb-human/data",
           "--out-dir", f"{C_RUN}/checkpoints", "--work", f"{C_RUN}/work", "--passes", "3", "--patience", "2", "--warmup", "100"]
    log("arm C: training from", a_final)
    return subprocess.Popen(cmd, cwd=CODE_B, env=env, stdout=open(f"{C_RUN}/train.log", "a"), stderr=subprocess.STDOUT)


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def evaluate(name, run, ckpt, port, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    cmd = [PY, "evaluate.py", "--run-dir", run, "--checkpoint", ckpt, "--opponents", "simple,foulplay",
           "--games", str(SIMPLE_GAMES), "--foulplay-games", str(FP_GAMES), "--foulplay-parallel", "24",
           "--port", str(port), "--server-dir", f"{HOME}/pokemon-showdown"]
    logf = open(f"{OUT}/eval_{name}.log", "a")
    log("evaluate", name, os.path.basename(ckpt), "port", port, "gpu", gpu)
    return subprocess.Popen(cmd, cwd=REL, env=env, stdout=logf, stderr=subprocess.STDOUT)


def result_row(run, ckpt, opponent):
    rows = []
    try:
        for line in open(f"{run}/eval/results.jsonl"):
            r = json.loads(line)
            if r.get("opponent") == opponent and r.get("checkpoint") == os.path.basename(ckpt):
                rows.append(r)
    except OSError:
        return None
    return rows[-1] if rows else None


def score(r):
    n = int(r["wins"]) + int(r["losses"]) + int(r.get("ties", 0))
    return (int(r["wins"]) + 0.5 * int(r.get("ties", 0))) / max(1, n), n


def trusted(r):
    return r is not None and r.get("valid", True) and int(r.get("errors", 0)) <= 0.05 * int(r.get("requested", r.get("games", 1)) or 1)


def main():
    log("waiting for arms A and B (arm C starts when A is done)")
    t0 = time.time()
    c_proc, c = None, None
    while True:
        a, b = arm_a_final(), arm_b_final()
        if a and c_proc is None:
            c_proc = train_arm_c(a)
        if c_proc is not None and c is None and c_proc.poll() is not None:
            files = sorted(glob.glob(f"{C_RUN}/checkpoints/learner_update_*.pt"))
            c = files[-1] if files else ""
            log("arm C: trainer exit", c_proc.returncode, "checkpoint", c or "none")
        if a and b and c is not None:
            break
        if time.time() - t0 > 8 * 3600:
            log("gave up waiting after 8 hours: arm A", a, "arm B", b, "arm C", c)
            return 3
        time.sleep(30)
    log("arm A", a, "arm B", b, "arm C", c)
    # Arm A's Showdown servers must be down before the full run takes the same ports; the evaluations use their own.
    rows = {}
    for attempt in (1, 2):
        todo = [(n, run, ck) for n, run, ck in (("A", A_RUN, a), ("B", B_RUN, b), ("C", C_RUN, c)) if ck and not trusted(rows.get(n))]
        if not todo:
            break
        procs = [(n, run, ck, evaluate(n, run, ck, 9300 + 20 * i, 3 + i)) for i, (n, run, ck) in enumerate(todo)]
        for n, run, ck, p in procs:
            code = p.wait()
            rows[n] = result_row(run, ck, "foulplay")
            rows[n + "_simple"] = result_row(run, ck, "simple")
            log("evaluated", n, "exit", code, "foulplay", None if rows[n] is None else (rows[n]["wins"], rows[n]["losses"], rows[n].get("errors")))
    ra = rows.get("A")
    decision = {"rule": f"B or C if it beats A by >= {MARGIN:.2f} on Foul Play ({FP_GAMES} games each) at two-sided p < {ALPHA}; "
                        "both: the higher win rate; neither: A", "arm_a": a, "arm_b": b, "arm_c": c}
    winner, why, starts = "A", "", {"A": a, "B": b, "C": c}
    if not trusted(ra):
        why = "arm A's Foul Play row is missing or not trusted: the full run starts from arm A (the standard recipe) anyway"
    else:
        pa, na = score(ra)
        decision["foulplay_A"], decision["games_A"] = round(pa, 4), na
        passed, notes = [], []
        for n in ("B", "C"):
            r = rows.get(n)
            if not trusted(r):
                notes.append(f"arm {n}: no trusted Foul Play row")
                continue
            pn, nn = score(r)
            pool = (pa * na + pn * nn) / (na + nn)
            se = math.sqrt(max(1e-12, pool * (1 - pool) * (1 / na + 1 / nn)))
            z = (pn - pa) / se
            p = math.erfc(abs(z) / math.sqrt(2))
            se_d = math.sqrt(pa * (1 - pa) / na + pn * (1 - pn) / nn)
            decision.update({f"foulplay_{n}": round(pn, 4), f"games_{n}": nn, f"difference_{n}": round(pn - pa, 4),
                             f"ci95_{n}": [round(pn - pa - 1.96 * se_d, 4), round(pn - pa + 1.96 * se_d, 4)],
                             f"z_{n}": round(z, 3), f"p_{n}": p})
            ok = pn - pa >= MARGIN and p < ALPHA
            notes.append(f"{n} - A = {100 * (pn - pa):+.1f} points (p = {p:.2g}): {'clears' if ok else 'does not clear'} the bar")
            if ok:
                passed.append((pn, n))
        if passed:
            winner = max(passed)[1]
        why = "; ".join(notes) + f". So arm {winner}"
    for n in ("A", "B", "C"):
        r = rows.get(n + "_simple")
        if r:
            decision[f"simple_{n}"] = round(score(r)[0], 4)
    decision.update({"winner": winner, "why": why, "start_checkpoint": starts[winner],
                     "decided": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    json.dump(decision, open(f"{OUT}/decision.json", "w"), indent=1)
    with open(f"{OUT}/DECISION.md", "w") as f:
        f.write(f"# Imitation test: arm {winner} starts the full run\n\n{why}.\n\n```\n{json.dumps(decision, indent=1)}\n```\n")
    log("decision", json.dumps(decision))
    return launch_full(decision)


def full_config(decision, start):
    src = json.load(open(f"{HOME}/runs/baseline/run.json"))
    cfg = {k: v for k, v in src.items() if not k.startswith("_")}
    cfg["repo_dir"] = REL
    cfg["phases"] = {
        "imitation": {"skip": True, "min_decisions": 0, "max_decisions": 1, "extra_train_args": []},
        "warmup": {"skip": False, "min_decisions": 500_000, "max_decisions": 1_500_000,
                   "plateau": {"metric": "vloss", "window_decisions": 300_000, "min_improvement_pct": 2.0, "smooth_updates": 4},
                   "train": {"save_every": 5}, "extra_train_args": []},
        "ppo": {"skip": False, "target_decisions": PPO_TARGET, "extra_train_args": []},
    }
    cfg["on_done"] = "shutdown"
    cfg["eval"] = {"every_hours": 3, "games": 400, "opponents": ["simple", "psbot"], "port": 9400,
                   "foulplay": {"games": 200, "every_hours": 3, "parallel": 8}}
    cfg["_notes"] = {"what this is": f"The full run after the imitation test (2026-10-14): arm {decision['winner']}'s imitation "
                                     f"network ({os.path.basename(start)}), then warmup and PPO to {PPO_TARGET:,} decisions, on release "
                                     f"baseline-c988471 (search 73f524b + GPU split + r2 worker fix). Decision: {OUT}/decision.json."}
    return cfg


def launch_full(decision):
    start = decision["start_checkpoint"]
    if os.path.exists(f"{FULL}/state.json"):
        log("the full run already exists: not touching it")
        return 4
    for port in range(8000, 8044):
        waited = 0
        while not port_free(port) and waited < 900:
            time.sleep(10)
            waited += 10
        if not port_free(port):
            log("port", port, "is still taken after 15 minutes: the full run is not started")
            return 5
    os.makedirs(f"{FULL}/checkpoints", exist_ok=True)
    shutil.copy2(start, f"{FULL}/checkpoints/{os.path.basename(start)}")
    cfg = full_config(decision, start)
    json.dump(cfg, open(f"{FULL}/run.json", "w"), indent=1)
    units = {
        "psppo-full-train": ("ps-ppo full run: training supervisor (pipeline.py)", "",
                             f"{PY} {REL}/pipeline.py --config {FULL}/run.json --run-dir {FULL}",
                             f"ExecCondition={PY} {REL}/pipeline.py --config {FULL}/run.json --run-dir {FULL} --can-start\n"),
        "psppo-full-eval": ("ps-ppo full run: evaluator", " CUDA_VISIBLE_DEVICES=7",
                            f"{PY} evaluate.py --run-dir {FULL} --watch --config {FULL}/run.json", ""),
    }
    for name, (desc, extra_env, start_cmd, cond) in units.items():
        text = (f"[Unit]\nDescription={desc}\nWants=network-online.target\nAfter=network-online.target\n"
                f"StartLimitIntervalSec=1800\nStartLimitBurst=10\n[Service]\nType=simple\nUser=ubuntu\nGroup=ubuntu\n"
                f"WorkingDirectory={REL}\nEnvironment=PYTHONUTF8=1 PYTHONUNBUFFERED=1{extra_env}\n"
                f"EnvironmentFile=-{HOME}/psppo-base.env\n{cond}ExecStart={start_cmd}\nRestart=always\nRestartSec=30\n"
                f"RestartPreventExitStatus=2 3 4 5\nKillSignal=SIGTERM\nKillMode=mixed\nTimeoutStopSec=600\n"
                f"LimitNOFILE=1048576\nTasksMax=infinity\nOOMPolicy=continue\n[Install]\nWantedBy=multi-user.target\n")
        tmp = f"{OUT}/{name}.service"
        open(tmp, "w").write(text)
        subprocess.run(["sudo", "cp", tmp, f"/etc/systemd/system/{name}.service"], check=True)
    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    ok = subprocess.run([PY, f"{REL}/pipeline.py", "--config", f"{FULL}/run.json", "--run-dir", FULL, "--can-start"],
                        cwd=REL, capture_output=True, text=True)
    log("can-start:", ok.returncode, ok.stdout.strip()[-300:], ok.stderr.strip()[-300:])
    if ok.returncode != 0:
        return 6
    subprocess.run(["sudo", "systemctl", "enable", "--now", "psppo-full-train", "psppo-full-eval"], check=True)
    log("full run started from", start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
