"""
Tests for pipeline.py (the training supervisor), status.py and train.py's --stop-file.

    python -m pytest tests/test_pipeline.py -q -p no:cacheprovider

Four layers:

  * pure functions and the Tracker, fed with train.py output captured on the laptop
    (tests/real_trainer_output.py) and on the Linux box (tests/fixtures/*.log: the bytes a parent
    process reads from train.py's stdout there, Ray's coloured prefixes and remarks included)
  * the pieces that touch files: atomic state, pruning (the removal itself is always mocked),
    the config files in cloud/
  * the programs against each other: what evaluate.py writes is what status.py and pipeline.py
    read, the systemd units and cloud/install_services.sh start both with arguments they have,
    the "eval" section of the run configs is what evaluate.py reads
  * end-to-end: pipeline.py as a real process against tests/fake_trainer.py and
    tests/fake_servers.py (no GPU, no Ray, no Showdown), through stalls, crashes, a clean stop
    and a resume, the restart limit, the disk guard and the shutdown at the end; and once with
    the real start_servers.py on a stand-in Showdown checkout (needs node)

The same things against the REAL train.py and real servers are not in here (they need the GPU and
minutes each): they were run by hand on 2026-10-05 and are written up in TRAINING.md.

Nothing here deletes a file: run directories are created under PSPPO_TEST_DIR (default: a new
directory in the system temp folder) and left there, and every child process pipeline.py is
given runs with PSPPO_NO_DELETE=1. The end-to-end tests listen on PSPPO_TEST_BASE_PORT and the
port after it (default 8110, 8111).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for p in (REPO, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline  # noqa: E402
import status  # noqa: E402
import evaluate  # noqa: E402   (standard library only at import time: torch and poke-env load when a game is played)
import real_trainer_output as real  # noqa: E402

WIN = os.name == "nt"
PIPELINE = os.path.join(REPO, "pipeline.py")
BASE_PORT = int(os.environ.get("PSPPO_TEST_BASE_PORT", "8110"))
TEST_ROOT = os.environ.get("PSPPO_TEST_DIR") or tempfile.mkdtemp(prefix="psppo_test_")
os.makedirs(TEST_ROOT, exist_ok=True)
_counter = [0]


def new_dir(name: str) -> str:
    _counter[0] += 1
    d = os.path.join(TEST_ROOT, f"{name}_{time.strftime('%H%M%S')}_{os.getpid()}_{_counter[0]}")
    os.makedirs(d)
    return d


def unit_config(**phases):
    raw = {"servers": {"command": ["x"]},
           "phases": {"imitation": {"max_decisions": 10 ** 9}, "warmup": {"max_decisions": 10 ** 9}, "ppo": {}}}
    raw["phases"].update(phases)
    return pipeline.validate_config(raw)


def make_tracker(cfg=None):
    cfg = cfg or unit_config()
    state = pipeline.new_state(cfg, pipeline.Paths(os.path.join(TEST_ROOT, "unit")), "")
    metrics, events = [], []
    tr = pipeline.Tracker(cfg, state, metrics.append, lambda name, detail: events.append((name, detail)))
    return tr, state, metrics, events


def feed(tr, lines, now, step=1.0):
    for line in lines:
        tr.feed(line, now)
        now += step
    return now


def steps_in(lines):
    """total_steps of every telemetry line, read with a plain regex, independent of the parser."""
    return [int(m.group(1)) for line in lines for m in [re.search(r"'total_steps': (\d+)", line)] if m and "[Telemetry]" in line]


# ============================================================================================
# the parser, on real captured lines
# ============================================================================================

def test_parse_real_telemetry_line_with_ray_noise_around_it():
    line = ("2026-10-05 09:59:23,609 [INFO] [Telemetry] ACTIVE: 256 | INFER: {'total_requests': 52133, 'avg_batch_size': 28.62, "
            "'model_version': 1} | TRAIN: {'update': 2, 'episodes': 949, 'steps_in_dataset': 1117, 'total_steps': 34562}\r\n")
    assert line in real.SOAK_PPO_12_MINUTES            # it is a real line, not one made up for the test
    ev = pipeline.parse_trainer_line(line)
    assert ev == {"kind": "telemetry", "active": 256, "total_steps": 34562, "update": 2, "episodes": 949, "steps_in_dataset": 1117,
                  "infer": {"total_requests": 52133, "avg_batch_size": 28.62, "model_version": 1}}


def test_parse_real_learner_line_with_colour_prefix():
    line = "\x1b[36m(LearnerActor pid=72908)\x1b[0m [learner] upd=1 kl=0.0007 clip=0.021 ent=1.261 vloss=2.404 ploss=-0.001 loss=1.176 n_mb=128\r\n"
    assert line in real.SOAK_PPO_12_MINUTES
    ev = pipeline.parse_trainer_line(line)
    assert ev["kind"] == "learner_update" and ev["update"] == 1 and ev["nonfinite"] == []
    assert ev["values"] == {"kl": 0.0007, "clip": 0.021, "ent": 1.261, "vloss": 2.404, "ploss": -0.001, "loss": 1.176, "n_mb": 128.0}


def test_parse_learner_worker_out_file_without_prefix():
    updates = [pipeline.parse_trainer_line(l) for l in real.LEARNER_WORKER_OUT if "upd=" in l]
    assert len(updates) == 9 and [u["update"] for u in updates] == list(range(1, 10))
    assert updates[0]["values"]["vloss"] == 3.719 and updates[0]["values"]["ploss"] == 1.858
    assert all(set(u["values"]) == {"kl", "clip", "ent", "vloss", "ploss", "loss", "n_mb"} for u in updates)
    kinds = [(pipeline.parse_trainer_line(l) or {}).get("kind") for l in real.LEARNER_WORKER_OUT]
    assert "init_policy" in kinds


@pytest.mark.parametrize("name", ["V1_IMITATION", "V2_PPO_AFTER_IMITATION", "V3_PPO_RESUMED", "V4_PPO_RESUMED_AGAIN",
                                  "V5_WARMUP_AFTER_PPO", "SOAK_PPO_12_MINUTES"])
def test_every_real_telemetry_and_learner_line_parses(name):
    lines = getattr(real, name)
    tele = [l for l in lines if "[Telemetry]" in l]
    upd = [l for l in lines if "[learner] upd=" in l]
    assert tele and upd
    parsed = [pipeline.parse_trainer_line(l) for l in tele]
    assert all(p and p["kind"] == "telemetry" and isinstance(p["total_steps"], int) and isinstance(p["update"], int) for p in parsed)
    assert [p["total_steps"] for p in parsed] == steps_in(lines)
    parsed = [pipeline.parse_trainer_line(l) for l in upd]
    assert all(p and p["kind"] == "learner_update" and "ploss" in p["values"] and "vloss" in p["values"] for p in parsed)


def test_parse_real_lifecycle_lines():
    kinds = {}
    for line in real.V2_PPO_AFTER_IMITATION:
        ev = pipeline.parse_trainer_line(line)
        if ev:
            kinds.setdefault(ev["kind"], ev)
    assert kinds["header"]["mode"] == "ppo" and kinds["header"]["steps_per_update"] == 4096
    assert kinds["header"]["minibatch"] == 256 and kinds["header"]["grad_accum"] == 1 and kinds["header"]["seq_len"] == 256
    assert kinds["phase_change"] == {"kind": "phase_change", "old": "imitation", "new": "ppo"}     # the line carries an emoji
    assert pipeline.path_basename(kinds["resumed"]["path"]) == "learner_update_000020.pt"          # a path with a backslash in it
    assert kinds["minutes_reached"] == {"kind": "minutes_reached"}
    assert pipeline.path_basename(kinds["final_saved"]["path"]) == "learner_update_000008.pt"


def test_parse_real_failed_update_line():
    ev = pipeline.parse_trainer_line(real.UPDATE_FAILED[0])
    assert ev == {"kind": "update_failed", "error": "LoweringException: ModuleNotFoundError: No module named 'triton'"}


def test_parse_real_line_where_two_records_ran_together():
    """Two threads write to the trainer's stdout. In one captured line the logger's telemetry record landed
    between the learner's text and its newline: "... n_mb=342026-10-05 09:48:48,489 [INFO] [Telemetry] ..."."""
    line = real.MERGED_LEARNER_AND_TELEMETRY[0]
    assert "n_mb=342026-10-05 09:48:48,489 [INFO] [Telemetry]" in line
    events = pipeline.parse_trainer_events(line)
    assert [e["kind"] for e in events] == ["learner_update", "telemetry"]
    assert events[0]["update"] == 6 and events[0]["values"]["n_mb"] == 34.0 and events[0]["values"]["loss"] == 3.016
    assert events[1]["total_steps"] == 25447 and events[1]["update"] == 6 and events[1]["active"] == 64
    tr, state, metrics, _ = make_tracker()
    tr.begin_phase("imitation", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    feed(tr, [tele(21000, 5), line], 1.0)
    assert len(metrics) == 1 and metrics[0]["update"] == 6 and metrics[0]["ploss"] == 1.499       # the update was not lost
    assert state["phases"]["imitation"]["decisions"] == 25447 and state["phases"]["imitation"]["updates"] == 6
    # the same for the other actor lines a logger line can run into
    p = pipeline.parse_trainer_events
    stamp = "2026-10-05 09:57:35,089 [INFO] Training started. Entering telemetry loop.\r\n"
    ev = p("\x1b[36m(LearnerActor pid=1)\x1b[0m [learner] resumed from C:/run/checkpoints\\learner_update_000008.pt" + stamp)
    assert [e["kind"] for e in ev] == ["resumed"] and pipeline.path_basename(ev[0]["path"]) == "learner_update_000008.pt"
    ev = p("(LearnerActor pid=1) PPO Update failed: RuntimeError: x" + stamp.replace("Training started. Entering telemetry loop.", tele(5, 1)[31:]))
    assert [e["kind"] for e in ev] == ["update_failed", "telemetry"] and ev[0]["error"] == "RuntimeError: x" and ev[1]["total_steps"] == 5
    # an ordinary line is one record, and a line that only starts with a stamp is not cut
    assert len(p(real.SOAK_PPO_12_MINUTES[20])) <= 1 and len(p(tele(7, 0))) == 1 and p("") == [] and p("\r\n") == []


def test_parse_lines_from_the_source_that_were_not_captured():
    # formats read from learner.py and train.py (no capture of these exists yet)
    p = pipeline.parse_trainer_line
    ev = p("\x1b[36m(LearnerActor pid=7)\x1b[0m [learner] resume failed from C:/run/checkpoints\\learner_update_000125.pt: RuntimeError('bad zip')\r\n")
    assert ev == {"kind": "resume_failed", "path": "C:/run/checkpoints\\learner_update_000125.pt", "error": "RuntimeError('bad zip')"}
    ev = p("(LearnerActor pid=7) [learner] resume failed from /home/ubuntu/runs/main/checkpoints/learner_update_000125.pt: EOFError()")
    assert pipeline.path_basename(ev["path"]) == "learner_update_000125.pt"
    assert p("2026-10-05 10:00:00,000 [INFO] Stop file /home/ubuntu/runs/main/trainer.stop found. Shutting down.")["kind"] == "stop_ack"
    assert p("2026-10-05 10:00:00,000 [WARNING] Could not save a final checkpoint: GetTimeoutError()")["kind"] == "final_save_failed"
    assert p("2026-10-05 10:00:00,000 [CRITICAL] FATAL: A RolloutWorker crashed in run(): boom")["kind"] == "worker_crashed"
    assert p("(LearnerActor pid=7) Learner training loop CRASHED!")["kind"] == "learner_crashed"
    assert p("2026-10-05 10:00:00,000 [WARNING] Telemetry loop encountered an error: x")["kind"] == "telemetry_error"
    assert p("some library warning\n") is None and p("") is None


def test_parse_nonfinite_losses_and_damaged_telemetry():
    ev = pipeline.parse_trainer_line("(LearnerActor pid=1) [learner] upd=7 kl=nan clip=0.100 ent=1.2 vloss=inf ploss=-0.001 loss=NaN n_mb=128")
    assert ev["values"]["kl"] is None and ev["values"]["vloss"] is None and ev["values"]["ploss"] == -0.001
    assert sorted(ev["nonfinite"]) == ["kl", "loss", "vloss"]
    json.dumps(ev, allow_nan=False)                 # nothing that is not valid JSON reaches metrics.jsonl
    # a TRAIN dict Python cannot evaluate still yields the counters
    ev = pipeline.parse_trainer_line("x [INFO] [Telemetry] ACTIVE: 12 | INFER: {} | TRAIN: {'update': 3, 'episodes': 9, 'bad': <obj>, 'total_steps': 4567}")
    assert ev["total_steps"] == 4567 and ev["update"] == 3 and ev["active"] == 12


def test_real_logs_on_disk_parse_when_present():
    """The Ray session directory and the captured driver logs of this laptop, if they are still there."""
    ray_logs = r"C:\Users\you\AppData\Local\Temp\ray\session_2026-10-05_10-48-43_423715_42188\logs"
    if not os.path.isdir(ray_logs):
        pytest.skip("the captured Ray session is not on this machine")
    n_upd = 0
    for fn in os.listdir(ray_logs):
        if fn.startswith("worker-") and fn.endswith(".out"):
            with open(os.path.join(ray_logs, fn), "r", encoding="utf-8", errors="replace", newline="") as f:
                for line in f:
                    if "[learner] upd=" in line:
                        ev = pipeline.parse_trainer_line(line)
                        assert ev and ev["kind"] == "learner_update" and ev["values"]["loss"] is not None, line
                        n_upd += 1
    assert n_upd >= 9
    print(f"parsed {n_upd} learner lines from worker-*.out in {ray_logs}")
    # the complete driver logs of the laptop runs, every line (library warnings and tracebacks included)
    scratch = r"C:\Users\you\AppData\Local\Temp\claude\C--Users-you\859d3990-c9f4-4aa8-a01c-ed63d8c4d773\scratchpad"
    checked = 0
    for fn in ("v1.log", "v2.log", "v3.log", "v4.log", "v5.log", "soak1.log", "smoke1.log", "smoke2.log", "smoke3.log",
               "smoke4.log", "smoke5.log", "doc1.log", "leak1.log"):
        path = os.path.join(scratch, fn)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            lines = f.readlines()
        tr, state, metrics, events = make_tracker()
        tr.begin_phase("ppo", 0.0)
        tr.begin_trainer(1, 0.0, fn, expect_resume=False)
        feed(tr, lines, 1.0)
        want = steps_in(lines)
        n_learner = sum(1 for l in lines if "[learner] upd=" in l)
        assert state["phases"]["ppo"]["decisions"] == want[-1] and state["trainer"]["telemetry_lines"] == len(want), fn
        assert len(metrics) == n_learner and state["collected_total"] == want[-1] - want[0], fn
        assert state["trainer"]["update_failures"] == sum(1 for l in lines if "PPO Update failed" in l), fn
        checked += 1
    print(f"ran {checked} complete driver logs through the tracker")


# ============================================================================================
# what train.py's stdout really carries on Linux (captured on the cloud box, Ray 2.59)
# ============================================================================================

FIXTURES = os.path.join(TESTS, "fixtures")
LINUX_SMOKE = os.path.join(FIXTURES, "linux_trainer_stdout_smoke.log")                 # a complete 5-minute run from nothing
LINUX_IMITATION = os.path.join(FIXTURES, "linux_trainer_stdout_imitation_excerpt.log")  # the live run, resumed from update 5
RAY_COLOUR_PREFIX = "\x1b[36m(LearnerActor pid="


def supervisor_lines(path):
    """
    A saved trainer stdout, handed on exactly as pipeline.Child._pump hands on the pipe: the bytes
    as they are, split where iterating a binary stream splits them, decoded as UTF-8 with
    replacement. Colour codes, Ray's prefixes and its "[repeated Nx across cluster]" remarks stay in.
    """
    out = []
    with open(path, "rb") as f:
        for raw in f:
            out.append(raw.decode("utf-8", "replace"))
    return out


def run_through_tracker(lines, phase="imitation", expect_resume=False, count_before=0):
    tr, state, metrics, events = make_tracker()
    tr.begin_phase(phase, 0.0)
    state["phases"][phase]["decisions"] = count_before
    tr.begin_trainer(1, 0.0, "linux.log", expect_resume=expect_resume)
    kinds = []
    now = 1.0
    for line in lines:
        kinds.append([e["kind"] for e in tr.feed(line, now)])
        now += 5.0
    return tr, state, metrics, events, kinds


def test_linux_smoke_run_every_telemetry_and_learner_line_is_recognised_and_the_counts_are_right():
    lines = supervisor_lines(LINUX_SMOKE)
    raw = open(LINUX_SMOKE, "rb").read()
    # the fixture is what a parent of train.py reads on Linux: actor lines carry Ray's coloured prefix
    assert b"\x1b[36m(LearnerActor pid=5996)\x1b[0m [learner] upd=1 " in raw and len(lines) == 85
    tr, state, metrics, events, kinds = run_through_tracker(lines)
    tele_lines = [i for i, l in enumerate(lines) if "[Telemetry]" in l]
    upd_lines = [i for i, l in enumerate(lines) if "[learner] upd=" in l]
    assert len(tele_lines) == 59 and len(upd_lines) == 10
    assert all(kinds[i] == ["telemetry"] for i in tele_lines)
    assert all(kinds[i] == ["learner_update"] and lines[i].startswith(RAY_COLOUR_PREFIX) for i in upd_lines)
    # the run did 10 updates and 40,440 decisions (the last telemetry line still says update 9:
    # the tenth update's learner line comes after it, and must count)
    ph, t = state["phases"]["imitation"], state["trainer"]
    assert ph["decisions"] == 40440 and ph["updates"] == 10 and ph["lost_decisions"] == 0
    assert state["decisions_total"] == 40440 and state["updates_total"] == 10 and state["collected_total"] == 40440
    assert t["telemetry_lines"] == 59 and t["updates_seen"] == 10 and t["total_steps"] == 40440 and t["active"] == 62
    assert [m["update"] for m in metrics] == list(range(1, 11))
    assert [(m["vloss"], m["ploss"], m["loss"], m["n_mb"]) for m in metrics][::9] == [(3.954, 2.014, 3.992, 10.0), (3.301, 1.748, 3.398, 10.0)]
    assert all(a["decisions_phase"] <= b["decisions_phase"] for a, b in zip(metrics, metrics[1:]))
    # and the lines around them
    assert (t["mode"], t["steps_per_update"], t["fresh_init"], t["resumed_from"]) == ("imitation", 4096, True, None)
    assert t["reached_minutes"] is True and t["final_checkpoint"] == "learner_update_000010.pt" and t["update_failures"] == 0
    assert events == [] and tr.fatal is None
    # independent of the parser: the numbers in the file, read with a plain regular expression
    assert steps_in(lines)[-1] == 40440 and len(steps_in(lines)) == 59


def test_linux_imitation_excerpt_resume_line_ray_remarks_and_counts():
    lines = supervisor_lines(LINUX_IMITATION)
    cut = [i for i, l in enumerate(lines) if l.startswith("...[cut]...")]
    assert len(cut) == 1                                                     # the excerpt is the head of the run and its last 300 lines
    head, tail = lines[:cut[0]], lines[cut[0] + 1:]
    assert len(tail) == 300 and tail == head[-300:]                          # the two parts overlap: the head alone is the whole stream
    # every line of both parts
    n_tele = n_upd = 0
    for line in lines:
        kinds = [e["kind"] for e in pipeline.parse_trainer_events(line)]
        if "[Telemetry]" in line:
            n_tele += 1
            assert kinds == ["telemetry"], line
        if "[learner] upd=" in line:
            n_upd += 1
            assert kinds == ["learner_update"] and line.startswith(RAY_COLOUR_PREFIX), line
    assert (n_tele, n_upd) == (568, 36)
    # Ray's remark behind a line it held back copies of is not a record of ours and breaks nothing
    remarks = [l for l in lines if "across cluster]" in l]
    assert len(remarks) == 2 and "\x1b[32m [repeated 5x across cluster] (Ray deduplicates logs by default." in remarks[0]
    assert "\x1b[32m [repeated 4x across cluster]\x1b[0m" in remarks[1]
    assert all(pipeline.parse_trainer_events(l) == [] for l in remarks)
    # the stream as the supervisor would have read it: the trainer resumed from update 5 with 184,712 decisions
    tr, state, metrics, events, kinds = run_through_tracker(head, expect_resume=True, count_before=180000)
    ph, t = state["phases"]["imitation"], state["trainer"]
    assert t["resumed_from"] == "learner_update_000005.pt" and t["fresh_init"] is False and tr.fatal is None
    assert (t["mode"], t["steps_per_update"]) == ("imitation", 36864)
    assert steps_in(head)[0] == 184712 and steps_in(head)[-1] == 862357
    assert ph["decisions"] == 862357 and ph["updates"] == 23 and ph["lost_decisions"] == 0 and events == []
    assert state["collected_total"] == 862357 - 184712 and t["updates_seen"] == 18 and t["telemetry_lines"] == len(steps_in(head))
    assert [m["update"] for m in metrics] == list(range(6, 24)) and metrics[-1]["ploss"] == 1.547 and metrics[0]["vloss"] == 2.946
    # had the supervisor counted more than the checkpoint held, the difference is recorded as lost
    tr, state, metrics, events, kinds = run_through_tracker(head, expect_resume=True, count_before=190000)
    assert events[0] == ("rollback", {"phase": "imitation", "from": 190000, "to": 184712, "lost": 5288, "resumed_from": "learner_update_000005.pt"})
    assert state["phases"]["imitation"]["lost_decisions"] == 5288 and state["phases"]["imitation"]["decisions"] == 862357


def test_rays_repeated_remark_behind_an_actor_line_is_cut_off():
    p = pipeline.parse_trainer_events
    remark = "\x1b[32m [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/x.html#log-deduplication for more options.)\x1b[0m\n"
    plain = "\x1b[36m(LearnerActor pid=5996)\x1b[0m [learner] upd=7 kl=0.0000 clip=0.000 ent=0.000 vloss=3.343 ploss=1.770 loss=3.441 n_mb=10"
    assert plain in [l.rstrip("\r\n") for l in supervisor_lines(LINUX_SMOKE)]  # a real line, with a remark put behind it
    want = p(plain + "\n")
    assert p(plain + remark) == want and p(plain + "\x1b[32m [repeated 2x across cluster]\x1b[0m\n") == want
    assert want[0]["values"] == {"kl": 0.0, "clip": 0.0, "ent": 0.0, "vloss": 3.343, "ploss": 1.77, "loss": 3.441, "n_mb": 10.0}
    ev = p("\x1b[36m(LearnerActor pid=13680)\x1b[0m [learner] resumed from /home/ubuntu/runs/main/checkpoints/learner_update_000005.pt" + remark)
    assert ev == [{"kind": "resumed", "path": "/home/ubuntu/runs/main/checkpoints/learner_update_000005.pt"}]
    ev = p("\x1b[36m(LearnerActor pid=7)\x1b[0m PPO Update failed: CUDA out of memory" + remark)
    assert ev == [{"kind": "update_failed", "error": "CUDA out of memory"}]
    # with the logger's line run into it (two writers on one stdout), both records survive
    merged = plain + "\x1b[32m [repeated 2x across cluster]\x1b[0m" + tele(29608, 7) + "\n"
    assert [e["kind"] for e in p(merged)] == ["learner_update", "telemetry"] and p(merged)[0] == want[0] and p(merged)[1]["total_steps"] == 29608


def test_a_telemetry_line_printed_late_does_not_take_an_update_back():
    """A telemetry line is put together from several actors' answers. One that was started before
    an update finished can be printed after the learner's line for that update."""
    tr, state, metrics, events = make_tracker()
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    feed(tr, [tele(0, 0), tele(4000, 0), upd(1), tele(4100, 0)], 1.0)          # the last one still says update 0
    assert state["phases"]["ppo"]["updates"] == 1 and state["trainer"]["update"] == 1 and state["updates_total"] == 1
    assert state["trainer"]["updates_seen"] == 1 and state["phases"]["ppo"]["decisions"] == 4100
    feed(tr, [tele(8200, 2), upd(2)], 10.0)                                   # the usual order the other way round
    assert state["phases"]["ppo"]["updates"] == 2 and state["trainer"]["updates_seen"] == 2 and [m["update"] for m in metrics] == [1, 2]


# ============================================================================================
# counting decisions across restarts and phase changes
# ============================================================================================

def test_decisions_across_real_restarts_and_phase_changes():
    """
    The five real runs captured on the laptop, in the order they happened: imitation; ppo on the
    imitation checkpoint (the trainer's counter restarts at 0); ppo restarted twice on its own
    checkpoints (the counter continues); warmup on the ppo checkpoint (restarts at 0).
    """
    assert steps_in(real.V1_IMITATION)[0] == 3544 and steps_in(real.V1_IMITATION)[-1] == 80998
    assert steps_in(real.V2_PPO_AFTER_IMITATION)[0] == 0 and steps_in(real.V2_PPO_AFTER_IMITATION)[-1] == 32882
    assert steps_in(real.V3_PPO_RESUMED)[0] == 35703 and steps_in(real.V3_PPO_RESUMED)[-1] == 56607
    assert steps_in(real.V4_PPO_RESUMED_AGAIN)[0] == 60203 and steps_in(real.V4_PPO_RESUMED_AGAIN)[-1] == 97607
    assert steps_in(real.V5_WARMUP_AFTER_PPO)[0] == 0 and steps_in(real.V5_WARMUP_AFTER_PPO)[-1] == 23304

    tr, state, metrics, events = make_tracker()
    now = 1000.0
    tr.begin_phase("imitation", now)
    tr.begin_trainer(11, now, "t1.log", expect_resume=False)
    now = feed(tr, real.V1_IMITATION, now)
    assert state["phases"]["imitation"]["decisions"] == 80998 and state["phases"]["imitation"]["updates"] == 20
    assert state["decisions_total"] == 80998

    tr.begin_phase("ppo", now)
    tr.begin_trainer(12, now, "t2.log", expect_resume=True)
    now = feed(tr, real.V2_PPO_AFTER_IMITATION, now)
    assert state["trainer"]["phase_change"] == "imitation -> ppo"
    assert state["phases"]["ppo"]["decisions"] == 32882                      # counted from 0, not added to 80998
    assert state["phases"]["imitation"]["decisions"] == 80998                # the finished phase keeps its count
    assert state["decisions_total"] == 80998 + 32882

    tr.begin_trainer(13, now, "t3.log", expect_resume=True)                  # same phase, restarted
    now = feed(tr, real.V3_PPO_RESUMED, now)
    assert state["trainer"]["resumed_from"] == "learner_update_000008.pt" and state["trainer"]["phase_change"] is None
    assert state["phases"]["ppo"]["decisions"] == 56607                      # not 32882 + 56607: the counter continued
    tr.begin_trainer(14, now, "t4.log", expect_resume=True)
    now = feed(tr, real.V4_PPO_RESUMED_AGAIN, now)
    assert state["phases"]["ppo"]["decisions"] == 97607 and state["phases"]["ppo"]["updates"] == 23
    assert state["phases"]["ppo"]["trainer_starts"] == 3 and state["phases"]["ppo"]["lost_decisions"] == 0

    tr.begin_phase("warmup", now)
    tr.begin_trainer(15, now, "t5.log", expect_resume=True)
    now = feed(tr, real.V5_WARMUP_AFTER_PPO, now)
    assert state["phases"]["warmup"]["decisions"] == 23304 and state["phases"]["warmup"]["updates"] == 6

    assert state["decisions_total"] == 80998 + 97607 + 23304
    assert state["updates_total"] == 20 + 23 + 6
    assert not [e for e in events if e[0] in ("rollback", "counter_reset")]
    # decisions actually collected while watched: per trainer, last report minus first report
    runs = (real.V1_IMITATION, real.V2_PPO_AFTER_IMITATION, real.V3_PPO_RESUMED, real.V4_PPO_RESUMED_AGAIN, real.V5_WARMUP_AFTER_PPO)
    assert state["collected_total"] == sum(steps_in(r)[-1] - steps_in(r)[0] for r in runs)
    # one metrics record per real learner line, each with the phase and both counts
    n_lines = sum(1 for r in runs for l in r if "[learner] upd=" in l)
    assert len(metrics) == n_lines == 49
    last = metrics[-1]
    assert last["phase"] == "warmup" and last["update"] == 6 and last["segment"] == 5
    assert last["decisions_phase"] <= 23304 and last["decisions_total"] == 80998 + 97607 + last["decisions_phase"]
    assert {"kl", "clip", "ent", "vloss", "ploss", "loss", "n_mb", "time", "wall_time", "total_steps", "updates_total"} <= set(last)
    ppo = [m for m in metrics if m["phase"] == "ppo"]
    assert [m["update"] for m in ppo] == list(range(1, 24))                  # 1..8, 9..13, 14..23 over three trainers
    assert all(a["decisions_phase"] <= b["decisions_phase"] for a, b in zip(ppo, ppo[1:]))


def tele(steps, update, active=64):
    return (f"2026-10-05 10:00:00,000 [INFO] [Telemetry] ACTIVE: {active} | INFER: {{'total_requests': 1, 'avg_batch_size': 1.0, "
            f"'model_version': {update}}} | TRAIN: {{'update': {update}, 'episodes': 1, 'steps_in_dataset': 0, 'total_steps': {steps}}}")


def upd(update, ploss=1.0, vloss=2.0):
    return f"\x1b[36m(LearnerActor pid=5)\x1b[0m [learner] upd={update} kl=0.0010 clip=0.100 ent=1.200 vloss={vloss:.3f} ploss={ploss:.3f} loss=1.000 n_mb=128"


def test_decisions_fall_back_with_the_weights_after_a_crash():
    tr, state, metrics, events = make_tracker()
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    feed(tr, [tele(0, 0), tele(20000, 0), upd(1), tele(40960, 1), tele(50000, 1)], 1.0)
    assert state["phases"]["ppo"]["decisions"] == 50000
    # the trainer dies; its newest checkpoint was written at update 1, 40,960 decisions
    tr.begin_trainer(2, 10.0, "b.log", True)
    assert state["phases"]["ppo"]["decisions"] == 50000 and not tr.synced
    tr.feed(tele(40960, 1), 11.0)
    ph = state["phases"]["ppo"]
    assert ph["decisions"] == 40960 and ph["lost_decisions"] == 9040 and ph["updates"] == 1
    assert events[-1] == ("rollback", {"phase": "ppo", "from": 50000, "to": 40960, "lost": 9040, "resumed_from": None})
    feed(tr, [tele(45000, 1), upd(2), tele(81920, 2)], 12.0)
    assert ph["decisions"] == 81920 and ph["lost_decisions"] == 9040 and state["decisions_total"] == 81920
    assert state["collected_total"] == 50000 + (81920 - 40960)               # what was collected, lost or not


def test_restart_before_the_first_checkpoint_of_a_phase_starts_the_phase_over():
    tr, state, metrics, events = make_tracker()
    tr.begin_phase("imitation", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    feed(tr, [tele(0, 0), tele(90000, 2)], 1.0)
    state["phases"]["imitation"].update(status="done")
    tr.begin_phase("warmup", 5.0)
    tr.begin_trainer(2, 5.0, "b.log", True)
    feed(tr, [tele(0, 0), tele(30000, 0)], 6.0)                              # dies before warmup wrote a checkpoint
    tr.begin_trainer(3, 10.0, "c.log", True)
    tr.feed(tele(0, 0), 11.0)                                                # newest checkpoint is still imitation's: zero again
    assert state["phases"]["warmup"]["decisions"] == 0 and state["phases"]["warmup"]["lost_decisions"] == 30000
    assert state["phases"]["imitation"]["decisions"] == 90000 and state["decisions_total"] == 90000
    tr.feed(tele(12000, 0), 12.0)
    assert state["decisions_total"] == 102000


def test_phase_rule_waits_for_the_new_trainer_to_report():
    cfg = unit_config(imitation={"max_decisions": 40000})
    tr, state, metrics, events = make_tracker(cfg)
    tr.begin_phase("imitation", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    feed(tr, [tele(0, 0), tele(39000, 1)], 1.0)
    assert tr.phase_rule() is None
    tr.feed(tele(41000, 1), 3.0)
    assert "40,000" in tr.phase_rule()
    # crash; the restart falls back to a checkpoint at 36,864. The stale 41,000 must not end the phase.
    tr.begin_trainer(2, 10.0, "b.log", True)
    assert state["phases"]["imitation"]["decisions"] == 41000 and tr.phase_rule() is None
    tr.feed(tele(36864, 1), 11.0)
    assert tr.phase_rule() is None
    tr.feed(tele(40100, 1), 12.0)
    assert tr.phase_rule() is not None


def test_metrics_for_updates_whose_learner_line_never_arrives():
    tr, state, metrics, events = make_tracker()
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    feed(tr, [tele(0, 0), tele(1000, 1), tele(2000, 2)], 1.0)                # telemetry counts two updates, no learner line
    tr.tick(10.0)
    assert metrics == []                                                     # still within the grace period
    tr.tick(3.0 + tr.GAP_GRACE_SECONDS + 1)
    assert [(m["update"], m.get("losses_missing")) for m in metrics] == [(1, True), (2, True)]
    tr.feed(upd(2), 200.0)                                                   # the line turns up after all
    assert metrics[-1]["update"] == 2 and metrics[-1]["late"] is True and metrics[-1]["ploss"] == 1.0
    tr.feed(upd(3), 201.0)
    assert metrics[-1]["update"] == 3 and "late" not in metrics[-1]


# ============================================================================================
# health: stalls and missing updates
# ============================================================================================

REAL_HEADER = [l for l in real.V1_IMITATION if " mode=imitation " in l][0]     # train.py's first line: mode, ports, steps_per_update


def watchdog_cfg():
    cfg = unit_config()
    cfg["watchdog"].update(stall_minutes=10, startup_grace_minutes=15, update_stall_minutes=30)
    return cfg


def test_stall_when_the_trainer_never_reports():
    tr, state, metrics, events = make_tracker(watchdog_cfg())
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    assert tr.health(14 * 60) is None                                        # inside the startup grace
    kind, reason, graceful = tr.health(15 * 60 + 1)
    assert kind == "stall" and "no telemetry" in reason and graceful


def test_stall_when_total_steps_stops_growing():
    tr, state, metrics, events = make_tracker(watchdog_cfg())
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    feed(tr, [tele(0, 0), tele(500, 0)], 10.0, step=5.0)                      # last growth at t=15
    for t in range(20, 1000, 5):
        tr.feed(tele(500, 0), float(t))                                      # telemetry keeps coming, the number does not move
    assert tr.health(15 + 600 - 1) is None
    kind, reason, _ = tr.health(15 + 600 + 1)
    assert kind == "stall" and "no growth in total_steps" in reason


def test_update_stall_when_decisions_flow_but_updates_do_not():
    tr, state, metrics, events = make_tracker(watchdog_cfg())
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    tr.feed(REAL_HEADER.replace("mode=imitation", "mode=ppo"), 0.0)           # the real header line: steps_per_update=4096
    assert state["trainer"]["steps_per_update"] == 4096
    feed(tr, [tele(0, 0), tele(4096, 0), upd(1), tele(4200, 1)], 1.0)          # update 1 at t=3
    for line in real.UPDATE_FAILED:
        tr.feed(line, 5.0)
    assert state["trainer"]["update_failures"] == 3 and "triton" in state["trainer"]["last_update_error"]
    assert events[0][0] == "update_failed"
    t, steps = 5.0, 4200
    while t < 1700:                                                          # decisions keep arriving, the update count stays at 1
        t += 5.0
        steps += 300
        tr.feed(tele(steps, 1), t)
    assert tr.health(1700.0) is None                                         # 28 minutes: not yet
    kind, reason, graceful = tr.health(3.0 + 1800 + 5)
    assert kind == "update_stall" and "3 failed updates" in reason and "triton" in reason and graceful


def test_no_update_stall_while_too_few_decisions_arrived():
    tr, state, metrics, events = make_tracker(watchdog_cfg())
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    tr.feed(REAL_HEADER, 0.0)                                                 # steps_per_update=4096
    feed(tr, [tele(0, 0), tele(4096, 0), upd(1)], 1.0)
    t, steps = 3.0, 4096
    while t < 2400:                                                          # a slow trainer: 2 decisions a second
        t += 5.0
        steps += 10
        tr.feed(tele(steps, 1), t)
    assert steps - 4096 < 2 * 4096
    assert tr.health(2400.0) is None                                         # 40 minutes without an update, but it is not due yet


def test_fatal_lines_ask_for_a_restart():
    tr, state, metrics, events = make_tracker()
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", True)
    tr.feed("(LearnerActor pid=7) [learner] resume failed from /r/checkpoints/learner_update_000125.pt: EOFError()", 1.0)
    tr.feed("(LearnerActor pid=7) [learner] pushed init policy 0 to inference", 1.1)
    kind, reason, graceful = tr.health(2.0)
    assert kind == "resume_failed" and graceful is False and state["trainer"]["resume_failed"] == "learner_update_000125.pt"
    # untrained weights although checkpoints exist, without the "resume failed" line
    tr.begin_trainer(2, 3.0, "b.log", True)
    tr.feed("(LearnerActor pid=7) [learner] pushed init policy 0 to inference", 4.0)
    assert tr.health(5.0)[0] == "fresh_init" and tr.health(5.0)[2] is False
    # the same line on a first start, with no checkpoint to resume, is normal
    tr.begin_trainer(3, 6.0, "c.log", False)
    tr.feed("(LearnerActor pid=7) [learner] pushed init policy 0 to inference", 7.0)
    assert tr.health(8.0) is None


def test_nan_loss_asks_for_a_hard_restart_after_three_updates_in_a_row():
    def nan(u):
        return f"\x1b[36m(LearnerActor pid=5)\x1b[0m [learner] upd={u} kl=nan clip=0.000 ent=nan vloss=nan ploss=nan loss=nan n_mb=128"
    tr, state, metrics, events = make_tracker()
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    feed(tr, [tele(0, 0), upd(1), nan(2), upd(3), nan(4), nan(5)], 1.0)       # one, then a good one, then two
    t = state["trainer"]
    assert tr.health(10.0) is None and t["nan_in_a_row"] == 2 and t["first_nan_update"] == 4 and t["last_loss_nonfinite"] is True
    assert metrics[-1]["loss"] is None and metrics[-1]["vloss"] is None and "loss" in metrics[-1]["nonfinite"]
    assert events[0][0] == "nonfinite_loss"
    tr.feed(nan(6), 11.0)
    kind, reason, graceful = tr.health(12.0)
    assert kind == "nan_loss" and graceful is False and "updates 4 to 6" in reason   # terminate, do not let it save
    # a kl that is not a number while the loss is fine is not this
    tr.begin_trainer(2, 20.0, "b.log", True)
    feed(tr, [tele(0, 6)] + [f"(LearnerActor pid=5) [learner] upd={u} kl=nan clip=0.0 ent=0.0 vloss=2.0 ploss=1.0 loss=2.0 n_mb=8" for u in range(7, 15)], 21.0)
    assert tr.health(40.0) is None and state["trainer"]["nan_in_a_row"] == 0


def test_telemetry_that_keeps_failing_is_a_restart():
    err = "2026-10-05 10:00:00,000 [WARNING] Telemetry loop encountered an error: The actor died unexpectedly before finishing this task."
    tr, state, metrics, events = make_tracker(watchdog_cfg())
    tr.begin_phase("ppo", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    feed(tr, [tele(0, 0), tele(500, 0)], 1.0, step=5.0)                       # last good report at t=6
    for t in range(10, 330, 5):
        tr.feed(err, float(t))                                               # train.py cannot reach an actor any more
    assert tr.health(300.0) is None                                          # 4 min 50 s of errors
    kind, reason, graceful = tr.health(311.0)
    assert kind == "telemetry_errors" and "actor died" in reason and graceful
    tr.feed(tele(600, 0), 312.0)                                             # it recovered: the clock starts again
    assert tr.health(313.0) is None and state["trainer"]["telemetry_error_since"] is None


# ============================================================================================
# phase rules
# ============================================================================================

def test_plateau_fires_only_when_the_loss_has_flattened():
    falling = [(i * 1000, 2.0 - 0.05 * i) for i in range(1, 21)]              # still dropping 0.05 per update
    assert pipeline.plateau_check(falling, 5000, 2.0, 3)[0] is False
    flat = falling + [(21000 + i * 1000, 1.0 + 0.001 * (i % 2)) for i in range(12)]
    fired, d = pipeline.plateau_check(flat, 5000, 2.0, 3)
    assert fired and abs(d["improvement_pct"]) < 2.0
    worse = falling + [(21000 + i * 1000, 1.0 + 0.01 * i) for i in range(12)]  # getting worse counts as no improvement
    assert pipeline.plateau_check(worse, 5000, 2.0, 3)[0] is True


def test_plateau_needs_the_window_to_be_covered_and_ignores_missing_values():
    assert pipeline.plateau_check([], 5000, 2.0)[0] is False
    assert pipeline.plateau_check([(1000, 1.0)], 5000, 2.0)[0] is False
    flat_but_short = [(i * 1000, 1.0) for i in range(1, 5)]                   # 4,000 decisions of history, window 5,000
    assert pipeline.plateau_check(flat_but_short, 5000, 2.0)[0] is False
    with_holes = [(i * 1000, None if i % 2 else 1.0) for i in range(1, 30)]
    assert pipeline.plateau_check(with_holes, 5000, 2.0)[0] is True
    assert pipeline.plateau_check([(i * 1000, float("nan")) for i in range(1, 30)], 5000, 2.0)[0] is False


def test_plateau_is_measured_exactly():
    hist = [(1000, 2.0), (2000, 2.0), (3000, 2.0), (9000, 1.9), (10000, 1.9)]
    fired, d = pipeline.plateau_check(hist, 6000, 6.0, 2)                    # then = mean(2.0, 2.0) at <= 4000, now = mean(1.9, 1.9)
    assert d["then"] == 2.0 and d["now"] == 1.9 and d["improvement_pct"] == 5.0 and d["decisions_then"] == 3000
    assert fired is True
    assert pipeline.plateau_check(hist, 6000, 5.0, 2)[0] is False            # exactly 5% is not "less than 5%"


def test_phase_end_reason_min_max_plateau_and_ppo_target():
    cfg = unit_config(imitation={"min_decisions": 20000, "max_decisions": 50000,
                                 "plateau": {"metric": "ploss", "window_decisions": 5000, "min_improvement_pct": 2.0, "smooth_updates": 2}},
                      ppo={"target_decisions": 1000})
    ph = cfg["phases"]["imitation"]
    flat = [(i * 1000, 1.0) for i in range(1, 60)]
    assert pipeline.phase_end_reason(ph, 15000, flat[:15]) is None           # flat, but below min_decisions
    assert "plateau" in pipeline.phase_end_reason(ph, 20000, flat[:20])
    falling = [(i * 1000, 5.0 - 0.05 * i) for i in range(1, 60)]
    assert pipeline.phase_end_reason(ph, 49999, falling[:49]) is None        # still improving
    assert "50,000" in pipeline.phase_end_reason(ph, 50000, falling[:50])    # max_decisions ends it regardless
    ppo = cfg["phases"]["ppo"]
    assert pipeline.phase_end_reason(ppo, 999, []) is None and pipeline.phase_end_reason(ppo, 1000, []) is not None
    never = unit_config()["phases"]["ppo"]                                    # target_decisions 0: never
    assert never["max_decisions"] == 0 and pipeline.phase_end_reason(never, 10 ** 12, flat) is None


def test_plateau_history_survives_a_supervisor_restart_and_a_rollback():
    cfg = unit_config(imitation={"min_decisions": 0, "max_decisions": 10 ** 9,
                                 "plateau": {"metric": "ploss", "window_decisions": 4000, "min_improvement_pct": 2.0, "smooth_updates": 2}})
    tr, state, metrics, events = make_tracker(cfg)
    tr.begin_phase("imitation", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", False)
    lines = [tele(0, 0)]
    for u in range(1, 11):
        lines += [tele(u * 1000, u), upd(u, ploss=1.0)]
    feed(tr, lines, 1.0)
    assert len(tr.history()) == 10 and tr.phase_rule() is not None
    # a new supervisor reads metrics.jsonl back
    tr2, state2, _, _ = make_tracker(cfg)
    state2["phases"]["imitation"].update(status="running", decisions=10000, updates=10)
    tr2.begin_phase("imitation", 100.0, history=metrics + [{"phase": "warmup", "update": 3, "decisions_phase": 5, "ploss": 9.0}])
    assert tr2.history() == tr.history()
    # the trainer it starts fell back to update 6: what was recorded after that is forgotten
    tr2.begin_trainer(2, 100.0, "b.log", True)
    tr2.feed(tele(6000, 6), 101.0)
    assert [d for d, _ in tr2.history()] == [1000, 2000, 3000, 4000, 5000, 6000]
    tr2.feed(upd(7, ploss=0.5), 102.0)
    assert tr2.history()[-1] == (6000, 0.5) and len(tr2.history()) == 7


# ============================================================================================
# pruning: the selection is pure, the removal is one guarded call (mocked here, always)
# ============================================================================================

def ck(update, mtime):
    return (f"learner_update_{update:06d}.pt", float(mtime))


def test_prune_keeps_newest_milestones_and_protected():
    files = [ck(u, 1000 + u) for u in range(25, 1025, 25)]                    # 40 checkpoints, 25 .. 1000
    keep, remove = pipeline.select_checkpoints_to_prune(files, keep_newest=3, milestone_every=250,
                                                        protected={"learner_update_000125.pt", "learner_update_000300.pt"}, now=5000)
    assert {n for n, why in keep.items() if why == "newest"} == {ck(u, 0)[0] for u in (950, 975, 1000)}
    assert {n for n, why in keep.items() if why == "milestone"} == {ck(u, 0)[0] for u in (250, 500, 750)}   # 1000 is already "newest"
    assert {n for n, why in keep.items() if why == "protected"} == {"learner_update_000125.pt", "learner_update_000300.pt"}
    assert len(keep) == 8 and len(remove) == 32 and not set(keep) & set(remove)
    assert remove == sorted(remove) and remove[0] == "learner_update_000025.pt"   # oldest first


def test_prune_orders_by_time_not_by_number():
    # after a change of phase the numbers start again: the highest number is the oldest file
    files = [ck(180, 100), ck(175, 90), ck(25, 200), ck(50, 300), ck(75, 400)]
    keep, remove = pipeline.select_checkpoints_to_prune(files, keep_newest=2, now=1000)
    assert set(keep) == {"learner_update_000050.pt", "learner_update_000075.pt"}
    assert remove == ["learner_update_000175.pt", "learner_update_000180.pt", "learner_update_000025.pt"]


def test_prune_never_offers_other_files_and_keeps_recent_ones():
    files = [ck(1, 10), ck(2, 20), ck(3, 30), ck(4, 995), ("learner_update_000005.pt.tmp", 5.0), ("notes.txt", 1.0),
             ("learner_update_12.pt", 2.0), ("phase_final", 3.0)]
    keep, remove = pipeline.select_checkpoints_to_prune(files, keep_newest=1, keep_seconds=100, now=1000)
    assert remove == ["learner_update_000001.pt", "learner_update_000002.pt", "learner_update_000003.pt"]
    assert keep["learner_update_000004.pt"] == "newest"
    assert all(keep[n] == "not a checkpoint name" for n in ("learner_update_000005.pt.tmp", "notes.txt", "learner_update_12.pt", "phase_final"))
    keep, remove = pipeline.select_checkpoints_to_prune(files[:4], keep_newest=1, keep_seconds=985, now=1000)
    assert keep["learner_update_000002.pt"] == "recent" and keep["learner_update_000003.pt"] == "recent"
    assert remove == ["learner_update_000001.pt"]
    assert pipeline.select_checkpoints_to_prune([], 5) == ({}, [])
    # keep_newest below 1 still keeps the newest: it is what a restart resumes from
    keep, remove = pipeline.select_checkpoints_to_prune(files[:4], keep_newest=0, now=1000)
    assert keep == {"learner_update_000004.pt": "newest"} and len(remove) == 3


def test_prune_milestone_zero_is_not_a_milestone():
    keep, remove = pipeline.select_checkpoints_to_prune([ck(0, 1), ck(250, 2), ck(7, 3)], keep_newest=1, milestone_every=250, now=10)
    assert keep == {"learner_update_000007.pt": "newest", "learner_update_000250.pt": "milestone"}
    assert remove == ["learner_update_000000.pt"]


def retention_supervisor(retention):
    d = new_dir("retention")
    cfg = unit_config()
    cfg["retention"].update(retention)
    sup = pipeline.Supervisor(cfg, os.path.join(d, "run"), "")
    sup.state = pipeline.new_state(cfg, sup.paths, "")
    os.makedirs(sup.paths.checkpoints)
    os.makedirs(sup.paths.eval)
    now = time.time()
    for i, u in enumerate(range(10, 110, 10)):                               # 10 checkpoints, one "minute" apart, oldest first
        path = os.path.join(sup.paths.checkpoints, f"learner_update_{u:06d}.pt")
        with open(path, "w") as f:
            f.write("x" * 100)
        os.utime(path, (now - 6000 + i * 60, now - 6000 + i * 60))
    with open(os.path.join(sup.paths.checkpoints, "learner_update_000120.pt.tmp"), "w") as f:
        f.write("half written")
    with open(os.path.join(sup.paths.eval, "results.jsonl"), "w") as f:
        f.write(json.dumps({"checkpoint": "learner_update_000020.pt", "opponent": "random", "winrate": 0.5}) + "\n")
        f.write('{"checkpoint": "learner_update_0000')                       # a torn last line
    with open(os.path.join(sup.paths.eval, "queue.json"), "w") as f:
        json.dump({"in_progress": "C:\\run\\checkpoints\\learner_update_000030.pt"}, f)
    sup.state["phases"]["imitation"]["final_checkpoint"] = "learner_update_000040.pt"
    return sup, now


def test_retention_off_by_default_never_calls_the_remover(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "remove_checkpoint_file", lambda path, d, expect=None: calls.append(path))
    assert pipeline.validate_config({"servers": {"command": ["x"]}, "phases": {"imitation": {"max_decisions": 1}, "warmup": {"max_decisions": 1}}})["retention"]["enabled"] is False
    sup, now = retention_supervisor({"enabled": False, "keep_newest": 2})
    sup._retention(now, force=True)
    assert calls == []


def test_retention_dry_run_only_reports(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "remove_checkpoint_file", lambda path, d, expect=None: calls.append(path))
    sup, now = retention_supervisor({"enabled": True, "dry_run": True, "keep_newest": 2})
    sup._retention(now, force=True)
    assert calls == []
    ev = sup.state["last_events"][-1]
    assert ev["event"] == "prune_dry_run" and len(ev["would_remove"]) == 5


def test_retention_on_removes_exactly_the_selected_files_through_the_one_guarded_call(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "remove_checkpoint_file", lambda path, d, expect=None: calls.append((os.path.basename(path), d, expect)))
    sup, now = retention_supervisor({"enabled": True, "keep_newest": 2, "milestone_every": 50})
    assert sup.protected_checkpoints() == {"learner_update_000020.pt", "learner_update_000030.pt", "learner_update_000040.pt"}
    sup._retention(now, force=True)
    # kept: 90 and 100 (newest), 20 (results.jsonl), 30 (named in eval/queue.json), 40 (last of imitation), 50 (milestone)
    assert [c[0] for c in calls] == ["learner_update_000010.pt", "learner_update_000060.pt", "learner_update_000070.pt", "learner_update_000080.pt"]
    assert all(c[1] == sup.paths.checkpoints for c in calls)
    # each call names the very file that was chosen, by its time and size, so the remover can refuse one written again since
    for name, _d, expect in calls:
        st = os.stat(os.path.join(sup.paths.checkpoints, name))
        assert expect == (st.st_mtime, st.st_size)
    ev = sup.state["last_events"][-1]
    assert ev["event"] == "prune" and ev["removed"] == [c[0] for c in calls] and ev["kept"] == 6
    assert len(os.listdir(sup.paths.checkpoints)) == 11                      # mocked: every file is still there


def test_the_remover_refuses_everything_that_is_not_a_checkpoint_in_the_checkpoint_dir(monkeypatch):
    removed = []
    monkeypatch.setattr(pipeline.os, "remove", lambda path: removed.append(path))   # nothing is deleted in this test either
    monkeypatch.delenv("PSPPO_NO_DELETE", raising=False)
    d = new_dir("remover")
    ckpt_dir = os.path.join(d, "checkpoints")
    os.makedirs(os.path.join(ckpt_dir, "quarantine"))
    good = os.path.join(ckpt_dir, "learner_update_000001.pt")
    for path in (good, os.path.join(ckpt_dir, "notes.txt"), os.path.join(d, "learner_update_000002.pt"),
                 os.path.join(ckpt_dir, "quarantine", "learner_update_000003.pt")):
        with open(path, "w") as f:
            f.write("x")
    for bad in (os.path.join(ckpt_dir, "notes.txt"), os.path.join(d, "learner_update_000002.pt"),
                os.path.join(ckpt_dir, "quarantine", "learner_update_000003.pt"), os.path.join(ckpt_dir, "learner_update_000009.pt"),
                os.path.join(ckpt_dir, "..", "learner_update_000002.pt"), ckpt_dir):
        with pytest.raises(ValueError):
            pipeline.remove_checkpoint_file(bad, ckpt_dir)
    assert removed == []
    monkeypatch.setenv("PSPPO_NO_DELETE", "1")
    with pytest.raises(PermissionError):
        pipeline.remove_checkpoint_file(good, ckpt_dir)
    assert removed == []
    monkeypatch.delenv("PSPPO_NO_DELETE")
    # a file that was written again since it was chosen (the trainer reuses the names of a finished phase) is refused
    st = os.stat(good)
    for stale in ((st.st_mtime - 5.0, st.st_size), (st.st_mtime, st.st_size + 1)):
        with pytest.raises(ValueError, match="written again"):
            pipeline.remove_checkpoint_file(good, ckpt_dir, stale)
    assert removed == []
    pipeline.remove_checkpoint_file(good, ckpt_dir, (st.st_mtime, st.st_size))
    assert removed == [os.path.realpath(good)] and os.path.exists(good)       # reached the (mocked) os.remove exactly once
    pipeline.remove_checkpoint_file(good, ckpt_dir)                           # without it, as before
    assert len(removed) == 2


def test_checkpoints_named_in_eval_tolerates_missing_and_torn_files():
    d = new_dir("evalnames")
    assert pipeline.checkpoints_named_in_eval(os.path.join(d, "nope")) == set()
    with open(os.path.join(d, "results.jsonl"), "w") as f:
        f.write('{"checkpoint": "learner_update_000100.pt", "opponent": "random"}\n')
        f.write('{"checkpoint": "/home/ubuntu/runs/main/checkpoints/learner_update_000200.pt"}\n')
        f.write('not json\n{"checkpoint": 5}\n{"checkpoint": "learner_upd')
    assert pipeline.checkpoints_named_in_eval(d) == {"learner_update_000100.pt", "learner_update_000200.pt"}


# ============================================================================================
# state and log files
# ============================================================================================

def test_atomic_write_json_never_shows_a_partial_file():
    d = new_dir("atomic")
    path = os.path.join(d, "state.json")
    pipeline.atomic_write_json(path, {"n": -1, "pad": "x" * 20000})
    bad, seen, stop = [], [0], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except OSError:
                continue                                                     # Windows: the file is being replaced right now
            try:
                obj = json.loads(text)
                assert len(obj["pad"]) == 20000
                seen[0] += 1
            except Exception as e:
                bad.append(repr(e)[:100])

    th = threading.Thread(target=reader)
    th.start()
    try:
        for i in range(300):
            pipeline.atomic_write_json(path, {"n": i, "pad": "x" * 20000})
    finally:
        stop.set()
        th.join()
    assert bad == [] and seen[0] > 0
    assert pipeline.read_json(path)["n"] == 299
    print(f"{seen[0]} reads during 300 rewrites, none partial")


def test_atomic_write_json_leaves_the_old_file_when_the_write_dies(monkeypatch):
    d = new_dir("atomic_fail")
    path = os.path.join(d, "state.json")
    pipeline.atomic_write_json(path, {"phase": "imitation", "n": 1})

    def boom(src, dst):
        raise OSError("disk went away")
    monkeypatch.setattr(pipeline.os, "replace", boom)
    with pytest.raises(OSError):
        pipeline.atomic_write_json(path, {"phase": "ppo", "n": 2})
    monkeypatch.undo()
    assert pipeline.read_json(path) == {"phase": "imitation", "n": 1}
    pipeline.atomic_write_json(path, {"phase": "ppo", "n": 3})                # and the next write goes through
    assert pipeline.read_json(path) == {"phase": "ppo", "n": 3}


def test_jsonl_writer_starts_a_new_line_after_a_torn_one_and_reader_skips_it():
    d = new_dir("jsonl")
    path = os.path.join(d, "metrics.jsonl")
    with open(path, "w") as f:
        f.write('{"update": 1}\n{"update": 2, "ploss": 1.')                  # the power went mid-line
    w = pipeline.JsonlWriter(path)
    w.write({"update": 3})
    w.close()
    assert [r["update"] for r in pipeline.read_jsonl(path)] == [1, 3]
    assert [r["update"] for r in status.read_jsonl(path)] == [1, 3]
    assert pipeline.read_jsonl(os.path.join(d, "missing.jsonl")) == []


def test_new_log_files_never_reuse_a_name_and_move_aside_never_overwrites():
    d = new_dir("names")
    a = pipeline.unique_new_file(d, "trainer_0001_ppo_x", ".log")
    b = pipeline.unique_new_file(d, "trainer_0001_ppo_x", ".log")
    c = pipeline.unique_new_file(d, "trainer_0001_ppo_x", ".log")
    assert len({a, b, c}) == 3 and all(os.path.exists(p) for p in (a, b, c))
    moved = []
    for i in range(3):
        stop = os.path.join(d, "trainer.stop")
        with open(stop, "w") as f:
            f.write(str(i))
        moved.append(pipeline.move_aside(stop, os.path.join(d, "used")))
        assert not os.path.exists(stop)
    assert len(set(moved)) == 3 and sorted(open(m).read() for m in moved) == ["0", "1", "2"]
    assert pipeline.move_aside(os.path.join(d, "trainer.stop"), os.path.join(d, "used")) is None


def test_rate_from_samples():
    samples = [[t, t * 100] for t in range(0, 4001, 50)]                      # 100 decisions a second for 4,000 s
    rate, span = pipeline.rate_from_samples(samples, 600, 4000)
    assert abs(rate - 100) < 1e-6 and span == 600
    rate, span = pipeline.rate_from_samples(samples, 3600, 4000)
    assert abs(rate - 100) < 1e-6 and span == 3600
    rate, span = pipeline.rate_from_samples(samples, 3600, 4600)             # nothing for the last 10 minutes
    assert abs(rate - 100 * 3000 / 3600) < 1e-6
    rate, span = pipeline.rate_from_samples(samples[:5], 600, 200)           # only 200 s of data
    assert abs(rate - 100) < 1e-6 and span == 200
    assert pipeline.rate_from_samples(samples[:1], 600, 10) is None and pipeline.rate_from_samples([], 600, 10) is None
    assert status.rate_from_samples(samples, 600, 4000) == pipeline.rate_from_samples(samples, 600, 4000)
    assert pipeline.rate_from_samples(samples, 600, 10 ** 6)[0] == 0.0       # long dead


# ============================================================================================
# configuration
# ============================================================================================

def test_unknown_keys_are_errors_with_a_hint():
    base = {"servers": {"command": ["x"]}, "phases": {"imitation": {"max_decisions": 1}, "warmup": {"max_decisions": 1}}}
    pipeline.validate_config(base)
    cases = [
        ({"wachdog": {}}, "unknown key 'wachdog' in config (did you mean 'watchdog'?)"),
        ({"watchdog": {"stal_minutes": 3}}, "unknown key 'stal_minutes' in watchdog (did you mean 'stall_minutes'?)"),
        ({"train_defaults": {"batles": 3}}, "unknown key 'batles' in train_defaults"),
        ({"retention": {"keep_last": 3}}, "unknown key 'keep_last' in retention"),
        ({"disk": {"min_free": 3}}, "unknown key 'min_free' in disk"),
    ]
    for extra, message in cases:
        with pytest.raises(pipeline.ConfigError) as e:
            pipeline.validate_config(dict(base, **extra))
        assert message in str(e.value)
    for phases, message in [
        ({"imitation": {"max_decisions": 1, "min_decision": 5}}, "unknown key 'min_decision' in phases.imitation"),
        ({"imitation": {"max_decisions": 1, "train": {"minbatch": 5}}}, "unknown key 'minbatch' in phases.imitation.train"),
        ({"imitation": {"max_decisions": 1, "plateau": {"metric": "ploss", "window": 5, "min_improvement_pct": 1}}}, "unknown key 'window' in phases.imitation.plateau"),
        ({"ppo": {"max_decisions": 5}}, "unknown key 'max_decisions' in phases.ppo"),
        ({"finetune": {}}, "unknown phase 'finetune'"),
        ({"imitation": {}}, "phases.imitation would never end"),
        ({"warmup": {"max_decisions": 1, "mode": "imitation"}}, "every phase needs its own mode"),
    ]:
        merged = dict(base["phases"])
        merged.update(phases)
        with pytest.raises(pipeline.ConfigError) as e:
            pipeline.validate_config(dict(base, phases=merged))
        assert message in str(e.value), str(e.value)
    with pytest.raises(pipeline.ConfigError):
        pipeline.validate_config(dict(base, on_done="poweroff"))
    with pytest.raises(pipeline.ConfigError):
        pipeline.validate_config(dict(base, retention={"keep_newest": 1}))
    with pytest.raises(pipeline.ConfigError):
        pipeline.validate_config({"phases": base["phases"]})                 # no showdown_dir and no server command


def test_comment_keys_and_the_eval_section_pass_through():
    raw = {"_notes": {"anything": "goes"}, "servers": {"_why": "x", "command": ["x"]},
           "phases": {"_note": "y", "imitation": {"_n": 1, "max_decisions": 1, "train": {"_n": 2}}, "warmup": {"max_decisions": 1}},
           "eval": {"every_hours": 3, "anything_the_evaluator_wants": {"nested": [1, 2]}}}
    cfg = pipeline.validate_config(raw)
    assert cfg["eval"] == raw["eval"] and "_notes" not in cfg and "_why" not in cfg["servers"]


def test_trainer_command_carries_every_setting_and_the_extra_arguments():
    raw = {
        "repo_dir": REPO, "python": "PY", "showdown_dir": "/sd",
        "servers": {"count": 3, "base_port": 8500, "heap_mb": 512},
        "train_defaults": {"battles": 96, "rooms_per_pair": 16, "teacher": "psbot", "steps_per_update": 8192, "minibatch": 512,
                           "seq_len": 256, "grad_accum": 2, "save_every": 25, "shaping": "potential", "extra_train_args": ["--future-flag", "1"]},
        "phases": {
            "imitation": {"max_decisions": 5, "train": {"teacher": "simple", "save_every": 10, "minutes": 240}, "extra_train_args": ["--only-imitation"]},
            "warmup": {"max_decisions": 5, "servers": {"count": 2}},
            "ppo": {"mode": "ppo_with_jepa", "servers": {"ports": "8600,8602"}, "train": {"battles": 64}},
        },
    }
    cfg = pipeline.validate_config(raw)
    paths = pipeline.Paths(os.path.join(TEST_ROOT, "cmd"))
    assert pipeline.build_trainer_command(cfg, paths, "imitation") == [
        "PY", "train.py", "--mode", "imitation", "--ports", "8500-8502", "--battles", "96", "--rooms-per-pair", "16", "--teacher", "simple",
        "--steps-per-update", "8192", "--minibatch", "512", "--seq-len", "256", "--grad-accum", "2", "--save-every", "10",
        "--shaping", "potential", "--minutes", "240", "--ckpt-dir", paths.checkpoints, "--stop-file", paths.trainer_stop,
        "--future-flag", "1", "--only-imitation"]
    # --log-dir: every Showdown server's own output stays with the run, under RUN/logs/servers
    assert paths.server_logs == os.path.join(paths.run, "logs", "servers")
    assert pipeline.build_server_command(cfg, paths, "imitation") == ["PY", "start_servers.py", "--dir", "/sd", "--ports", "8500-8502", "--heap-mb", "512",
                                                                    "--log-dir", paths.server_logs]
    warm = pipeline.build_trainer_command(cfg, paths, "warmup")
    assert warm[warm.index("--ports") + 1] == "8500-8501" and "--only-imitation" not in warm and warm[-2:] == ["--future-flag", "1"]
    ppo = pipeline.build_trainer_command(cfg, paths, "ppo")
    assert ppo[ppo.index("--mode") + 1] == "ppo_with_jepa" and ppo[ppo.index("--ports") + 1] == "8600,8602"
    assert ppo[ppo.index("--battles") + 1] == "64" and "--minutes" not in ppo
    assert pipeline.build_server_command(cfg, paths, "ppo")[5] == "8600,8602"
    # an injected command, with placeholders
    cfg = pipeline.validate_config(dict(raw, trainer={"command": ["{python}", "{repo}/tests/fake_trainer.py"]}))
    assert pipeline.build_trainer_command(cfg, paths, "ppo")[:2] == ["PY", REPO + "/tests/fake_trainer.py"]
    cfg = pipeline.validate_config(dict(raw, servers={"command": ["my_servers", "--ports", "{ports}", "--logs", "{server_log_dir}", "--run", "{run_dir}"]}))
    assert pipeline.build_server_command(cfg, paths, "imitation") == ["my_servers", "--ports", "8000-8003", "--logs", paths.server_logs, "--run", paths.run]
    assert set(pipeline.TRAIN_FLAGS) == {"battles", "rooms_per_pair", "teacher", "steps_per_update", "minibatch", "seq_len",
                                         "grad_accum", "save_every", "shaping", "minutes"}


def run_vm_shape():
    """(servers, battles) as cloud/run_vm.json has them."""
    raw = pipeline.read_json(os.path.join(REPO, "cloud", "run_vm.json"))
    return raw["servers"]["count"], raw["train_defaults"]["battles"]


def test_run_vm_json_is_what_was_asked_for():
    cfg = pipeline.load_config(os.path.join(REPO, "cloud", "run_vm.json"))
    raw = pipeline.read_json(os.path.join(REPO, "cloud", "run_vm.json"))
    assert isinstance(raw["_notes"], dict) and len(raw["_notes"]) >= 15
    paths = pipeline.Paths("/home/ubuntu/runs/main")
    assert cfg["repo_dir"].replace("\\", "/").endswith("/home/ubuntu/ps-ppo") and cfg["python"] == "/home/ubuntu/venv/bin/python"
    # How many servers and battles: one of the two shapes that have a reason. 12 and 384 is what the
    # file was written with, for a 64-vCPU box. 10 and 320 is what was measured on the 16-vCPU box
    # that the run is on (the review asked for the file to say that). Either way 32 battles a server.
    servers, battles = run_vm_shape()
    assert (servers, battles) in ((12, 384), (10, 320)) and battles == 32 * servers
    span = f"8000-{8000 + servers - 1}"
    assert pipeline.phase_ports(cfg, cfg["phases"]["ppo"]) == list(range(8000, 8000 + servers))
    assert pipeline.build_server_command(cfg, paths, "imitation") == [
        "/home/ubuntu/venv/bin/python", "start_servers.py", "--dir", "/home/ubuntu/pokemon-showdown", "--ports", span, "--heap-mb", "2048",
        "--log-dir", os.path.join(paths.run, "logs", "servers")]
    common = ["--ports", span, "--battles", str(battles), "--rooms-per-pair", "16", "--teacher", "psbot", "--steps-per-update", "36864",
              "--minibatch", "768", "--seq-len", "256", "--grad-accum", "1"]
    tail = ["--ckpt-dir", paths.checkpoints, "--stop-file", paths.trainer_stop]
    head = ["/home/ubuntu/venv/bin/python", "train.py", "--mode"]
    assert pipeline.build_trainer_command(cfg, paths, "imitation") == head + ["imitation"] + common + ["--save-every", "10"] + tail
    assert pipeline.build_trainer_command(cfg, paths, "warmup") == head + ["warmup"] + common + ["--save-every", "5"] + tail
    assert pipeline.build_trainer_command(cfg, paths, "ppo") == head + ["ppo"] + common + ["--save-every", "25"] + tail
    im, wu, ppo = (cfg["phases"][p] for p in ("imitation", "warmup", "ppo"))
    assert (im["min_decisions"], im["max_decisions"], im["plateau"]["metric"]) == (1_500_000, 4_000_000, "ploss")
    assert (wu["min_decisions"], wu["max_decisions"], wu["plateau"]["metric"]) == (500_000, 1_500_000, "vloss")
    assert ppo["target_decisions"] == 250_000_000 and ppo["max_decisions"] == 250_000_000
    assert cfg["retention"]["enabled"] is True and cfg["retention"]["dry_run"] is False
    assert cfg["on_done"] == "shutdown" and cfg["shutdown_command"] == ["sudo", "shutdown", "-h", "now"]
    assert cfg["eval"] == {"every_hours": 3, "games": 200, "opponents": ["random", "maxpower", "simple", "psbot"],
                           "foulplay": {"games": 100, "every_hours": 6}}
    # the unit's TimeoutStopSec must be longer than the supervisor's own patience
    unit = open(os.path.join(REPO, "cloud", "systemd", "psppo-train.service"), encoding="utf-8").read()
    stop_sec = int(re.search(r"^TimeoutStopSec=(\d+)", unit, re.M).group(1))
    assert stop_sec > cfg["trainer"]["stop_timeout_seconds"] + cfg["servers"]["stop_timeout_seconds"] + cfg["trainer"]["kill_timeout_seconds"]
    # ... and systemd must still be starting the supervisor when it counts one internal error too
    # many and marks the run failed itself (max_restarts_per_hour + 1 of them): after
    # StartLimitBurst starts systemd gives up, and then nothing acts on on_failed
    assert int(re.search(r"^StartLimitBurst=(\d+)", unit, re.M).group(1)) > cfg["watchdog"]["max_restarts_per_hour"] + 1


def test_run_laptop_test_json_is_tiny_and_stays_off_the_owners_ports():
    cfg = pipeline.load_config(os.path.join(REPO, "cloud", "run_laptop_test.json"))
    ports = pipeline.phase_ports(cfg, cfg["phases"]["imitation"])
    assert len(ports) == 2 and not set(ports) & {8000, 8001, 8002, 8003}
    assert all(0 < cfg["phases"][p]["max_decisions"] <= 20_000 for p in pipeline.PHASES)
    assert cfg["retention"]["enabled"] is False and cfg["on_done"] == "idle"
    assert cfg["repo_dir"] == REPO                                           # defaults to the folder pipeline.py is in


def test_dry_run_prints_commands_and_rules(capsys):
    rc = pipeline.main(["--config", os.path.join(REPO, "cloud", "run_vm.json"), "--run-dir", os.path.join(TEST_ROOT, "dry_run_never_created"), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and not os.path.exists(os.path.join(TEST_ROOT, "dry_run_never_created"))
    servers, battles = run_vm_shape()
    span = f"8000-{8000 + servers - 1}"
    for needle in ("== imitation", "== warmup", "== ppo", f"start_servers.py --dir /home/ubuntu/pokemon-showdown --ports {span}",
                   f"train.py --mode ppo --ports {span} --battles {battles}", "ends at 250,000,000 decisions",
                   "ploss (mean of 5 updates) improved by less than 1.0% over the last 600,000 decisions", "pruning ON", "sudo shutdown -h now"):
        assert needle in out, needle


# ============================================================================================
# train.py: --stop-file, and that pipeline.py only ever passes flags train.py takes
# ============================================================================================

def test_train_py_help_still_parses_and_lists_stop_file():
    env = dict(os.environ, PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(REPO, "train.py"), "--help"], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "--stop-file" in r.stdout and "--minutes" in r.stdout and "--ckpt-dir" in r.stdout


@pytest.fixture(scope="module")
def train_module():
    pytest.importorskip("ray")
    pytest.importorskip("torch")
    import train
    return train


def test_train_py_accepts_every_command_pipeline_builds(train_module):
    for name in ("run_vm.json", "run_laptop_test.json"):
        cfg = pipeline.load_config(os.path.join(REPO, "cloud", name))
        paths = pipeline.Paths(os.path.join(TEST_ROOT, "argcheck"))
        for phase in pipeline.PHASES:
            cmd = pipeline.build_trainer_command(cfg, paths, phase)
            assert cmd[1] == "train.py"
            args = train_module.parse_args(cmd[2:])                          # argparse exits on a flag train.py does not have
            assert args.mode == phase and args.stop_file == paths.trainer_stop and args.ckpt_dir == paths.checkpoints
            run_cfg = train_module.build_config(args)
            assert run_cfg.learner.mode == phase and run_cfg.learner.ckpt_dir == paths.checkpoints
            assert train_module.parse_ports(args.ports) == pipeline.phase_ports(cfg, cfg["phases"][phase])
    assert train_module.parse_args([]).stop_file is None


class _FakeRemote:
    def remote(self, *a, **k):
        return object()


class _FakeActor:
    get_stats = _FakeRemote()
    heartbeat = _FakeRemote()


class _FakeRay:
    def __init__(self):
        self.gets = 0

    def wait(self, refs, num_returns=1, timeout=None):
        return [], list(refs)

    def get(self, refs, timeout=None):
        self.gets += 1
        return [{"total_requests": 1, "avg_batch_size": 1.0, "model_version": 0},
                {"update": 0, "episodes": 0, "steps_in_dataset": 5, "total_steps": 5}, {"active_battles_worker": 2}]


def _run_telemetry_loop(train, monkeypatch, stop_file, minutes=0.0, timeout=20.0):
    fake = _FakeRay()
    monkeypatch.setattr(train, "ray", fake)
    real_sleep = asyncio.sleep

    async def quick_sleep(_seconds):
        await real_sleep(0.02)
    monkeypatch.setattr(train.asyncio, "sleep", quick_sleep)
    args = train.parse_args(["--stop-file", stop_file] if stop_file else [])
    args.minutes = minutes

    async def go():
        await asyncio.wait_for(train._telemetry_loop(args, time.time(), _FakeActor(), _FakeActor(), [_FakeActor()], [object()]), timeout)
    asyncio.run(go())
    return fake.gets


def test_train_py_telemetry_loop_leaves_when_the_stop_file_appears(train_module, monkeypatch):
    d = new_dir("stopfile")
    stop_file = os.path.join(d, "trainer.stop")
    threading.Timer(0.6, lambda: open(stop_file, "w").write("stop")).start()
    t0 = time.time()
    gets = _run_telemetry_loop(train_module, monkeypatch, stop_file)
    took = time.time() - t0
    assert gets >= 3 and 0.5 < took < 10, (gets, took)                       # it looped until the file appeared, then returned
    print(f"left after {gets} telemetry rounds, {took:.2f} s")


def test_train_py_telemetry_loop_leaves_at_once_when_the_file_is_already_there(train_module, monkeypatch):
    d = new_dir("stopfile_present")
    stop_file = os.path.join(d, "trainer.stop")
    open(stop_file, "w").write("stop")
    assert _run_telemetry_loop(train_module, monkeypatch, stop_file) == 0     # returned before talking to any actor


def test_train_py_telemetry_loop_keeps_running_without_a_stop_file(train_module, monkeypatch):
    d = new_dir("stopfile_absent")
    with pytest.raises(asyncio.TimeoutError):
        _run_telemetry_loop(train_module, monkeypatch, os.path.join(d, "never_created.stop"), timeout=0.8)
    with pytest.raises(asyncio.TimeoutError):
        _run_telemetry_loop(train_module, monkeypatch, None, timeout=0.5)       # and with no --stop-file at all


# ============================================================================================
# status.py
# ============================================================================================

def running_state(now, **trainer):
    t = {"segment": 3, "phase": "ppo", "pid": 1, "running": True, "telemetry_lines": 50, "total_steps": 500000, "update": 12,
         "progress_seen": True, "last_progress_ts": now - 5, "last_update_ts": now - 40, "steps_at_last_update": 480000,
         "update_failures": 0, "last_update_error": None, "last_loss_nonfinite": False}
    t.update(trainer)
    return {"phase": "ppo", "status": "running", "status_reason": "", "updated_ts": now - 2, "trainer": t,
            "phases": {p: {"status": "done", "decisions": 1000, "updates": 3} for p in ("imitation", "warmup")} | {"ppo": {"status": "running", "decisions": 500000, "updates": 12}},
            "limits": {"stall_minutes": 10, "startup_grace_minutes": 15, "update_stall_minutes": 30, "min_free_gb": 15, "state_seconds": 5,
                       "max_restarts_per_hour": 6, "phases": {"ppo": {"max_decisions": 250000000}}},
            "decisions_total": 502000, "updates_total": 18,
            "samples": [[now - 690 + i * 30, i * 30000] for i in range(24)]}  # 1,000 decisions a second up to now


def flags(state, now, free=100.0):
    return status.health_flags(state, now, free)[0]


def test_status_health_flags():
    now = 1_000_000.0
    assert flags(running_state(now), now) == ["OK"]
    assert flags(running_state(now, last_progress_ts=now - 601), now) == ["STALLED"]
    assert flags(running_state(now, last_update_ts=now - 1801), now) == ["NO_UPDATES"]
    assert flags(running_state(now, last_update_ts=now - 1801, steps_at_last_update=500000), now) == ["OK"]   # no decisions either: not this flag
    assert flags(running_state(now, last_loss_nonfinite=True), now) == ["NAN_LOSS"]
    assert flags(running_state(now, telemetry_lines=0, total_steps=None, progress_seen=False, last_progress_ts=now - 300), now) == ["STARTING"]
    assert flags(running_state(now, telemetry_lines=0, total_steps=None, progress_seen=False, last_progress_ts=now - 901), now) == ["STALLED"]
    assert flags(running_state(now), now, free=14.9) == ["DISK_LOW"]
    st = running_state(now)
    st["updated_ts"] = now - 400
    assert flags(st, now) == ["DEAD"]                                        # "running", but nobody has written the state for minutes
    for st_status, want in (("failed", ["FAILED"]), ("stopped", ["STOPPED"]), ("starting", ["STARTING"]), ("restarting", ["RESTARTING"]),
                            ("stopping", ["STOPPING"]), ("disk_low", ["DISK_LOW"]), ("error", ["DEAD"])):
        st = running_state(now)
        st["status"] = st_status
        assert flags(st, now) == want, st_status
    st = running_state(now)
    st.update(phase="done", status="done", updated_ts=now - 99999)
    assert flags(st, now) == ["DONE"]                                        # done and idle or shut down: not DEAD
    assert flags(None, now) == ["NO_STATE"] and flags({}, now) == ["NO_STATE"] and flags({"phase": "ppo"}, now) == ["NO_STATE"]


def test_status_on_missing_and_half_written_files(capsys):
    d = new_dir("status_empty")
    assert status.main(["--run-dir", os.path.join(d, "does_not_exist"), "--no-gpu"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "HEALTH: NO_STATE"
    run = os.path.join(d, "run")
    os.makedirs(os.path.join(run, "eval"))
    os.makedirs(os.path.join(run, "checkpoints"))
    assert status.main(["--run-dir", run, "--no-gpu"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "HEALTH: NO_STATE"
    now = time.time()
    good = json.dumps(running_state(now), indent=1)
    with open(os.path.join(run, "state.json"), "w") as f:
        f.write(good[: len(good) // 2])                                      # half a state file
    with open(os.path.join(run, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"phase": "ppo", "update": 1, "ploss": 0.004, "vloss": 2.4, "wall_time": now - 90}) + "\n")
        f.write(json.dumps({"phase": "ppo", "update": 2, "ploss": 0.002, "vloss": 2.3, "wall_time": now - 50}) + "\n")
        f.write('{"phase": "ppo", "update": 3, "ploss": 0.0')                # torn last line
    with open(os.path.join(run, "events.jsonl"), "w") as f:
        f.write('{"time": "2026-10-05T10:00:00Z", "event": "trainer_start", "name": "ppo"}\n[1, 2]\n{"time": "2026-')
    with open(os.path.join(run, "eval", "results.jsonl"), "w") as f:
        f.write(json.dumps({"time": "2026-10-05T10:00:00Z", "checkpoint": "learner_update_000025.pt", "update": 25, "phase": "ppo",
                            "opponent": "random", "games": 200, "wins": 150, "winrate": 0.75, "ci_low": 0.69, "ci_high": 0.80, "sample": False}) + "\n")
        f.write('{"time": "2026-10-05T13:00:00Z", "opponent": "random", "winra')
    assert status.main(["--run-dir", run, "--no-gpu"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "HEALTH: NO_STATE" and "75.0%" in out and "trainer_start" in out
    with open(os.path.join(run, "state.json"), "w") as f:
        f.write(json.dumps(running_state(time.time()), indent=1))
    assert status.main(["--run-dir", run, "--no-gpu", "--exit-code"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "HEALTH: OK" and out.isascii()
    assert "ploss  first 0.004   best 0.002 (update 2)   latest 0.002 (update 2)   2 updates" in out
    assert "this phase 500,000 of 250,000,000 (0.2%)" in out and re.search(r"last 10 min (9\d\d|1,0\d\d)/s", out), out
    assert status.main(["--run-dir", run, "--no-gpu", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["health"] == ["OK"] and info["loss"]["vloss"] == {"first": 2.4, "best": 2.3, "best_update": 2, "latest": 2.3, "latest_update": 2, "n": 2}
    assert info["speed"]["last_10_min"]["per_second"] == pytest.approx(1000.0, rel=0.02)
    assert info["updates"]["minutes_since_last"] == pytest.approx(0.7, abs=0.1)
    assert info["counts"] == {"metrics_records": 2, "events": 1, "evaluations": 1}
    for junk in ("", "null", "[]", "{", '{"phase": "ppo"}', "\x00\x00\x00"):  # nothing in state.json makes it crash
        with open(os.path.join(run, "state.json"), "w") as f:
            f.write(junk)
        assert status.main(["--run-dir", run, "--no-gpu", "--exit-code"]) == 1
        assert capsys.readouterr().out.splitlines()[0] == "HEALTH: NO_STATE"


def test_status_evaluation_table_last_six_with_changes():
    rows = []
    rates = [0.50, 0.55, 0.53, 0.60, 0.66, 0.70, 0.74, 0.80]
    for i, wr in enumerate(rates):
        rows.append({"time": f"2026-10-05T{10 + i:02d}:00:00Z", "checkpoint": f"learner_update_{25 * (i + 1):06d}.pt", "update": 25 * (i + 1),
                     "phase": "ppo", "total_steps": 1000 * i, "opponent": "simple", "games": 200, "wins": int(wr * 200), "losses": 200 - int(wr * 200),
                     "ties": 0, "errors": 0, "winrate": wr, "ci_low": wr - 0.07, "ci_high": wr + 0.07, "seconds": 300.0, "sample": False})
    rows.append(dict(rows[0], opponent="psbot", winrate=0.10, ci_low=0.06, ci_high=0.15))
    rows.append(dict(rows[1], opponent="simple", sample=True, winrate=0.40))
    rows.append({"opponent": "broken"})                                      # no win rate: ignored
    table = status.evaluation_table(list(reversed(rows)))                    # file order does not matter, time does
    simple = table["simple"]
    assert len(simple) == 6 and [r["update"] for r in simple] == [75, 100, 125, 150, 175, 200]
    assert simple[0]["evaluations_total"] == 8
    assert [r["delta_prev"] for r in simple] == [-2.0, 7.0, 6.0, 4.0, 4.0, 6.0]
    assert [r["delta_first"] for r in simple] == [3.0, 10.0, 16.0, 20.0, 24.0, 30.0]
    assert table["psbot"][0]["delta_prev"] is None and table["psbot"][0]["delta_first"] is None
    assert set(table) == {"simple", "psbot", "simple (sampled)"}
    info = {"health": ["OK"], "health_notes": [], "run_dir": "x", "time": "t", "state_age_seconds": 1.0, "phase": "ppo", "status": "running",
            "status_reason": "", "decisions": {"phase": 1, "total": 2, "phase_target": None, "lost": 0},
            "speed": {"last_10_min": None, "last_hour": None}, "updates": {"phase": 1, "total": 1, "minutes_since_last": None, "failed": 0, "last_error": None},
            "loss": {}, "loss_phase": "ppo", "phases": {p: {} for p in status.PHASES}, "restarts": {}, "events": [], "disk": None, "gpu": None,
            "checkpoint": None, "evaluation": table}
    text = status.render(info)
    line = [l for l in text.splitlines() if "ppo:200" in l][0]
    assert "80.0%" in line and "[73.0, 87.0]" in line and "200" in line and "+6.0" in line and "+30.0" in line
    assert text.splitlines()[0] == "HEALTH: OK"


def eval_line(opponent, wins, games, t, update=25, phase="ppo", errors=0, mtime=1000.0, **extra):
    """A line of eval/results.jsonl made by evaluate.py's own writer, not written out by hand."""
    tally = evaluate.Tally(requested=games + errors, wins=wins, losses=games - wins)
    meta = {"checkpoint": f"learner_update_{update:06d}.pt", "update": update, "phase": phase, "total_steps": 1000 * update,
            "checkpoint_mtime": mtime, "checkpoint_path": f"/run/checkpoints/learner_update_{update:06d}.pt"}
    return evaluate.result_line(meta, opponent, tally, sample=False, extras=extra, now=evaluate.parse_iso(t))


def test_status_table_leaves_out_lines_that_are_not_results():
    """evaluate.py writes a line even when no game finished (winrate 0.0, "valid": false) and marks a
    line "valid": false when too many games broke. Neither is a win rate: not in the table, and
    never what "vs prev" is measured against."""
    good1 = eval_line("simple", 100, 200, "2026-10-05T10:00:00Z", update=25)
    server_down = eval_line("simple", 0, 0, "2026-10-05T13:00:00Z", update=50, errors=200)        # no game finished
    broken = eval_line("simple", 90, 150, "2026-10-05T16:00:00Z", update=75, errors=50)            # a quarter of the games lost
    good2 = eval_line("simple", 120, 200, "2026-10-05T19:00:00Z", update=100)
    never = eval_line("psbot", 0, 0, "2026-10-05T19:05:00Z", update=100, errors=200)               # an opponent with nothing usable
    assert (server_down["games"], server_down["winrate"], server_down["valid"]) == (0, 0.0, False)
    assert (broken["games"], broken["valid"]) == (150, False) and good1["valid"] is True
    results = [good1, server_down, broken, good2, never]
    assert [status.is_result(r) for r in results] == [True, False, False, True, False]
    table = status.evaluation_table(results)
    assert set(table) == {"simple"} and [r["update"] for r in table["simple"]] == [25, 100]
    assert table["simple"][1]["delta_prev"] == 10.0 and table["simple"][1]["delta_first"] == 10.0   # 50% -> 60%, not via the 0%
    assert table["simple"][0]["evaluations_total"] == 2
    assert status.evaluation_skipped(results) == {"no_finished_game": 2, "not_valid": 1}
    # a line from before "valid" existed, or with games missing, is still shown
    assert status.is_result({"opponent": "random", "winrate": 0.5}) and status.is_result({"opponent": "random", "winrate": 0.5, "games": 10})
    d = new_dir("status_eval_skip")
    run = os.path.join(d, "run")
    os.makedirs(os.path.join(run, "eval"))
    for r in results:
        evaluate.append_jsonl(os.path.join(run, "eval", "results.jsonl"), r)
    info = status.gather(run, with_gpu=False)
    assert info["counts"]["evaluations"] == 5 and info["evaluation_skipped"] == {"no_finished_game": 2, "not_valid": 1}
    text = status.render(info)
    rows = [l for l in text.splitlines() if "ppo:" in l]
    assert len(rows) == 2 and "50.0%" in rows[0] and "60.0%" in rows[1] and "+10.0" in rows[1] and "0.0%" not in text.replace("50.0%", "").replace("60.0%", "")
    assert "not shown: 2 with no finished game, 1 marked not valid" in text
    # nothing usable at all: say so, do not print an empty table
    run2 = os.path.join(d, "run2")
    os.makedirs(os.path.join(run2, "eval"))
    evaluate.append_jsonl(os.path.join(run2, "eval", "results.jsonl"), server_down)
    text = status.render(status.gather(run2, with_gpu=False))
    assert "evaluation no usable result yet: 1 with no finished game" in text and "opponent   " not in text


def test_status_shows_the_evaluators_heartbeat_and_flags_a_stale_one(capsys):
    d = new_dir("status_evaluator")
    run = os.path.join(d, "run")
    os.makedirs(os.path.join(run, "eval"))
    with open(os.path.join(run, "state.json"), "w") as f:
        f.write(json.dumps(running_state(time.time())))
    # no evaluator has ever run here: a line that says so, and no flag
    info = status.gather(run, with_gpu=False)
    assert info["evaluator"] is None and info["health"] == ["OK"]
    assert "evaluator  no eval/watch.json" in status.render(info)
    # the heartbeat file as evaluate.py's watcher writes it (its own code, into this directory)
    args = evaluate.build_parser().parse_args(["--run-dir", run, "--watch"])
    w = evaluate.Watcher(args)
    w.poll = 60.0
    w.status = "learner_update_000025.pt is evaluated; waiting for a newer checkpoint"
    w.evaluations = 3
    w.heartbeat()
    on_disk = status.read_json(os.path.join(run, "eval", "watch.json"))
    assert {"pid", "host", "started", "heartbeat", "heartbeat_ts", "status", "evaluations", "poll_seconds"} <= set(on_disk)
    now = time.time()
    info = status.gather(run, now=now, with_gpu=False)
    ev = info["evaluator"]
    assert ev["pid"] == os.getpid() and ev["evaluations"] == 3 and ev["stale"] is False and ev["stopped"] is False
    assert 0 <= ev["heartbeat_age_seconds"] < 5 and info["health"] == ["OK"]
    text = status.render(info)
    assert f"evaluator  alive, pid {os.getpid()}, heartbeat " in text and "evaluations finished 3" in text and "waiting for a newer checkpoint" in text
    # four minutes later: still fine (one poll a minute, an evaluation beats every second)
    later = status.gather(run, now=now + 240, with_gpu=False)
    assert later["evaluator"]["stale"] is False and "EVAL_STALE" not in later["health"]
    # six minutes without a heartbeat: flagged, last in the HEALTH line, and --exit-code says so
    info = status.gather(run, now=now + 360, with_gpu=False)
    assert info["evaluator"]["stale"] is True and info["health"][-1] == "EVAL_STALE" and len(info["health"]) == 2
    assert any("evaluate.py --watch" in n and "6.0 minutes ago" in n for n in info["health_notes"])
    assert "evaluator  STALE: last heartbeat 6.0 min ago" in status.render(info)
    # a watcher with a long poll is given three polls
    on_disk["poll_seconds"] = 600.0
    with open(os.path.join(run, "eval", "watch.json"), "w") as f:
        f.write(json.dumps(on_disk))
    assert status.gather(run, now=now + 1700, with_gpu=False)["evaluator"]["stale"] is False
    assert status.gather(run, now=now + 1900, with_gpu=False)["evaluator"]["stale"] is True
    # a watcher that was stopped says so in its last heartbeat: shown, not flagged
    w.status = "stopped"
    w.heartbeat()
    info = status.gather(run, now=time.time() + 3600, with_gpu=False)
    assert info["evaluator"]["stopped"] is True and info["evaluator"]["stale"] is False and "EVAL_STALE" not in info["health"]
    assert "evaluator  STOPPED" in status.render(info)
    # the command line: a stale evaluator fails --exit-code, a half-written watch.json costs only its line
    on_disk.update(status="waiting", heartbeat_ts=time.time() - 4000, poll_seconds=60.0)
    with open(os.path.join(run, "eval", "watch.json"), "w") as f:
        f.write(json.dumps(on_disk))
    with open(os.path.join(run, "state.json"), "w") as f:
        f.write(json.dumps(running_state(time.time())))
    assert status.main(["--run-dir", run, "--no-gpu", "--exit-code"]) == 1
    assert capsys.readouterr().out.splitlines()[0] == "HEALTH: OK EVAL_STALE"
    with open(os.path.join(run, "eval", "watch.json"), "w") as f:
        f.write('{"pid": 12, "heartbeat_ts": 17')
    assert status.main(["--run-dir", run, "--no-gpu", "--exit-code"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "HEALTH: OK"


def test_status_does_not_call_a_trainers_first_report_an_update():
    """Seen on the real run: with no update yet the screen said "last one 0.2 min ago", because the
    watchdog's clock (last_update_ts) starts when a trainer first reports."""
    d = new_dir("status_updates")
    run = os.path.join(d, "run")
    os.makedirs(run)
    cfg = unit_config()
    state = pipeline.new_state(cfg, pipeline.Paths(run), "")
    tr = pipeline.Tracker(cfg, state, lambda rec: None, lambda name, detail: None)
    now = time.time()
    tr.begin_phase("imitation", now - 100)
    tr.begin_trainer(1, now - 100, "a.log", False)
    feed(tr, [tele(0, 0), tele(900, 0)], now - 30, step=5.0)
    assert state["trainer"]["updates_seen"] == 0 and state["trainer"]["last_update_ts"] is not None
    state.update(status="running", updated_ts=now, limits={"stall_minutes": 10, "startup_grace_minutes": 15, "update_stall_minutes": 30})
    pipeline.atomic_write_json(os.path.join(run, "state.json"), state)
    info = status.gather(run, now=now, with_gpu=False)
    assert info["updates"]["minutes_since_last"] is None and info["updates"]["none_since_trainer_start"] is True
    assert "last one - (none yet from the trainer running now)" in status.render(info)
    # after an update it is that update's time
    tr.feed(upd(1), now - 12)
    tr.feed(tele(4200, 1), now - 10)
    assert state["trainer"]["updates_seen"] == 1
    pipeline.atomic_write_json(os.path.join(run, "state.json"), state)
    info = status.gather(run, now=now, with_gpu=False)
    assert info["updates"]["minutes_since_last"] == pytest.approx(0.2, abs=0.05) and info["updates"]["none_since_trainer_start"] is False
    # a restarted trainer that has not updated yet: the last update is the last record of metrics.jsonl
    tr.begin_trainer(2, now - 5, "b.log", True)
    tr.feed(tele(4200, 1), now - 2)
    pipeline.atomic_write_json(os.path.join(run, "state.json"), state)
    with open(os.path.join(run, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"phase": "imitation", "update": 1, "ploss": 1.0, "vloss": 2.0, "wall_time": now - 12}) + "\n")
    info = status.gather(run, now=now, with_gpu=False)
    assert info["updates"]["minutes_since_last"] == pytest.approx(0.2, abs=0.05) and info["updates"]["none_since_trainer_start"] is True
    assert "(none yet from the trainer running now)" in status.render(info)
    # a state file written before this field existed reads as before
    del state["trainer"]["updates_seen"]
    pipeline.atomic_write_json(os.path.join(run, "state.json"), state)
    info = status.gather(run, now=now, with_gpu=False)
    assert info["updates"]["none_since_trainer_start"] is False and info["updates"]["minutes_since_last"] is not None


# ============================================================================================
# the pieces against each other: what one writes is what the other reads
# ============================================================================================

def _unit_exec_args(unit_name, key="ExecStart"):
    text = open(os.path.join(REPO, "cloud", "systemd", unit_name), encoding="utf-8").read()
    lines = [l for l in text.splitlines() if l.startswith(key + "=")]
    assert len(lines) == 1, (unit_name, key, lines)
    return lines[0][len(key) + 1:].split(), text


def test_eval_unit_starts_evaluate_py_with_arguments_it_has_on_the_run_the_trainer_unit_drives():
    argv, text = _unit_exec_args("psppo-eval.service")
    assert argv[0] == "/home/ubuntu/venv/bin/python" and argv[1] == "evaluate.py"
    assert "WorkingDirectory=/home/ubuntu/ps-ppo" in text                      # evaluate.py is named relative to it
    args = evaluate.build_parser().parse_args(argv[2:])                        # argparse exits on a flag evaluate.py does not have
    assert args.watch is True and args.once is False and args.force is False
    assert args.run_dir == "/home/ubuntu/runs/main" and args.config == "/home/ubuntu/ps-ppo/cloud/run_vm.json"
    # the same run directory and the same config as the supervisor's unit
    targv, ttext = _unit_exec_args("psppo-train.service")
    assert targv[1].endswith("/pipeline.py")
    assert targv[targv.index("--run-dir") + 1] == args.run_dir and targv[targv.index("--config") + 1] == args.config
    cargv, _ = _unit_exec_args("psppo-train.service", "ExecCondition")
    assert cargv[1:] == targv[1:] + ["--can-start"]
    # and pipeline.py takes what its unit passes (it would exit with a usage error otherwise)
    r = subprocess.run([sys.executable, PIPELINE, "--help"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and all(flag in r.stdout for flag in ("--config", "--run-dir", "--can-start", "--stop", "--clear-failed"))
    # the units keep LF line endings on every machine (.gitattributes): CRLF breaks systemd on the box
    for name in ("psppo-eval.service", "psppo-train.service"):
        assert b"\r" not in open(os.path.join(REPO, "cloud", "systemd", name), "rb").read()


def test_install_services_renders_units_whose_commands_both_programs_accept():
    """cloud/install_services.sh --print rewrites the paths of the unit files for another layout.
    What comes out must still be command lines pipeline.py and evaluate.py take."""
    import shutil
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not on PATH")
    if WIN and os.path.normcase(os.path.abspath(bash)).startswith(os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows"))):
        pytest.skip("the bash on PATH is the WSL launcher, which does not pass this test's environment on")
    script = os.path.join(REPO, "cloud", "install_services.sh")
    assert b"\r" not in open(script, "rb").read()
    layouts = [
        ({}, "/home/ubuntu/ps-ppo", "/home/ubuntu/venv/bin/python", "/home/ubuntu/runs/main", "/home/ubuntu/ps-ppo/cloud/run_vm.json", "ubuntu", "ubuntu"),
        ({"VENV_PY": "/srv/venv/bin/python", "RUN_DIR": "/data/runs/x", "CONFIG": "/srv/cfg/my.json", "RUN_USER": "trainer", "RUN_GROUP": "ml"},
         "/srv/ps-ppo", "/srv/venv/bin/python", "/data/runs/x", "/srv/cfg/my.json", "trainer", "ml"),
    ]
    for extra, repo, py, run_dir, config, user, group in layouts:
        env = dict(os.environ, REPO=repo, USER_HOME="/home/ubuntu", RUN_USER="ubuntu", RUN_GROUP="ubuntu")
        env.update(extra)
        r = subprocess.run([bash, script, "--print"], capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, r.stderr
        units = r.stdout.split("##### ")
        train = [u for u in units if u.startswith("/etc/systemd/system/psppo-train.service")][0]
        evalu = [u for u in units if u.startswith("/etc/systemd/system/psppo-eval.service")][0]
        for unit in (train, evalu):
            assert f"\nUser={user}\n" in unit and f"\nGroup={group}\n" in unit and f"\nWorkingDirectory={repo}\n" in unit
            assert "/home/ubuntu/ps-ppo" not in unit.replace(repo, "") or repo == "/home/ubuntu/ps-ppo"
        start = [l for l in train.splitlines() if l.startswith("ExecStart=")][0][len("ExecStart="):].split()
        cond = [l for l in train.splitlines() if l.startswith("ExecCondition=")][0][len("ExecCondition="):].split()
        assert start == [py, f"{repo}/pipeline.py", "--config", config, "--run-dir", run_dir] and cond == start + ["--can-start"]
        argv = [l for l in evalu.splitlines() if l.startswith("ExecStart=")][0][len("ExecStart="):].split()
        assert argv[:2] == [py, "evaluate.py"]
        args = evaluate.build_parser().parse_args(argv[2:])
        assert args.watch and args.run_dir == run_dir and args.config == config
    # with no group name to be had (as on this laptop's Git Bash) the script still renders one line per setting
    env = dict(os.environ, REPO="/home/ubuntu/ps-ppo", USER_HOME="/home/ubuntu")
    env.pop("RUN_GROUP", None)
    r = subprocess.run([bash, script, "--print"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and r.stdout.count("\nGroup=") == 2, r.stderr


def test_evaluate_check_config_says_what_watch_mode_would_do_and_finds_a_typo(capsys):
    assert evaluate.main(["--run-dir", os.path.join(TEST_ROOT, "never_created_by_check_config"), "--check-config", os.path.join(REPO, "cloud", "run_vm.json")]) == 0
    out = capsys.readouterr().out
    assert "200 games against each of random, maxpower, simple, psbot" in out and "Foul Play: 100 games every 6 h" in out
    assert "127.0.0.1:9000" in out and "/home/ubuntu/pokemon-showdown" in out and "WARNING" not in out
    assert not os.path.exists(os.path.join(TEST_ROOT, "never_created_by_check_config"))
    d = new_dir("check_config")
    bad = os.path.join(d, "bad.json")
    with open(bad, "w") as f:
        json.dump({"showdown_dir": "/sd", "eval": {"games": 50, "evry_hours": 1, "port": "9000"}}, f)
    assert evaluate.main(["--run-dir", d, "--check-config", bad]) == 2
    out = capsys.readouterr().out
    assert "WARNING: eval.evry_hours is not a known key" in out and "WARNING: eval.port='9000' is not usable: keeping 9000" in out
    assert "50 games against each of" in out
    assert evaluate.main(["--run-dir", d, "--check-config", os.path.join(d, "missing.json")]) == 2


@pytest.mark.parametrize("name", ["run_vm.json", "run_laptop_test.json"])
def test_the_eval_section_of_the_run_configs_is_what_evaluate_py_reads(name):
    raw = pipeline.read_json(os.path.join(REPO, "cloud", name))
    pipeline.validate_config(raw)                                            # the supervisor accepts the file, eval section and all
    cfg, warnings = evaluate.eval_config(raw, {})
    assert warnings == [], warnings                                          # every key is one evaluate.py knows, every value usable
    for key, value in raw["eval"].items():
        if key == "foulplay":
            for fk, fv in value.items():
                assert cfg["foulplay"][fk] == fv, fk
        else:
            assert cfg[key] == value, key
    assert set(cfg) == set(evaluate.EVAL_DEFAULTS)                           # and every key it does not give has its default
    assert cfg["server_dir"] == raw["showdown_dir"]                          # the evaluator's own server comes from the same checkout
    assert cfg["python"] == raw.get("python")
    # the evaluator's server is never on a training port, in any phase
    pcfg = pipeline.validate_config(raw)
    training = {p for ph in pipeline.PHASES for p in pipeline.phase_ports(pcfg, pcfg["phases"][ph])}
    assert cfg["port"] not in training and cfg["host"] == "127.0.0.1"
    if name == "run_vm.json":
        assert cfg["port"] == 9000 and cfg["every_hours"] == 3 and cfg["games"] == 200 and cfg["foulplay"] == {
            "games": 100, "every_hours": 6, "parallel": 4, "search_time_ms": 200}
        assert cfg["opponents"] == ["random", "maxpower", "simple", "psbot"] and cfg["concurrency"] == 32 and cfg["device"] == "auto"
    else:
        assert cfg["port"] in (8100, 8101) and not {cfg["port"]} & {8000, 8001, 8002, 8003, 9000}


def test_eval_config_reports_unknown_keys_and_takes_underscore_keys_as_comments():
    raw = {"eval": {"_note": "x", "games": 50, "evry_hours": 1, "foulplay": {"_why": "y", "games": 10, "paralel": 2}}}
    cfg, warnings = evaluate.eval_config(raw, {})
    assert cfg["games"] == 50 and cfg["foulplay"]["games"] == 10
    assert sorted(warnings) == ["eval.evry_hours is not a known key", "eval.foulplay.paralel is not a known key"]   # the typos, not the comments
    assert pipeline.validate_config({"servers": {"command": ["x"]}, "phases": {"imitation": {"max_decisions": 1}, "warmup": {"max_decisions": 1}},
                                     "eval": raw["eval"]})["eval"] == raw["eval"]


def test_what_evaluate_py_writes_protects_its_checkpoints_from_pruning():
    """pipeline.py protects by file name everything the evaluator names: the "checkpoint" field of its
    result lines, the checkpoint it is working on (in_progress.json) and the one in its heartbeat."""
    d = new_dir("eval_protects")
    run = os.path.join(d, "run")
    paths = evaluate.RunPaths(run)
    assert paths.eval == pipeline.Paths(run).eval and paths.final_done == pipeline.Paths(run).final_done
    assert paths.results == os.path.join(pipeline.Paths(run).eval, "results.jsonl")
    line = eval_line("simple", 100, 200, "2026-10-05T10:00:00Z", update=50)
    assert line["checkpoint"] == "learner_update_000050.pt"                    # a bare file name, which is what pruning compares
    evaluate.append_jsonl(paths.results, line)
    evaluate.append_jsonl(paths.hybrid_results, eval_line("simple", 100, 200, "2026-10-05T10:00:00Z", update=75, subject="hybrid"))
    evaluate.write_json_atomic(paths.in_progress, {"status": "running", "checkpoint": "learner_update_000100.pt", "started": "x", "pid": 1, "opponents": ["simple"]})
    args = evaluate.build_parser().parse_args(["--run-dir", run, "--watch"])
    w = evaluate.Watcher(args)
    w.status = "evaluating learner_update_000125.pt against random, simple (interval)"
    w.heartbeat()
    names = pipeline.checkpoints_named_in_eval(paths.eval)
    assert names == {"learner_update_000050.pt", "learner_update_000075.pt", "learner_update_000100.pt", "learner_update_000125.pt"}
    files = [pipeline.Ckpt(f"learner_update_{u:06d}.pt", u, 1000.0 + u, 10) for u in (25, 50, 75, 100, 125, 150, 175)]
    keep, remove = pipeline.select_checkpoints_to_prune(files, keep_newest=2, protected=names, now=10 ** 9)
    assert remove == ["learner_update_000025.pt"]
    assert all(keep[n] == "protected" for n in names) and keep["learner_update_000175.pt"] == "newest"


def test_a_checkpoint_name_used_again_in_a_later_phase_is_another_checkpoint_to_the_evaluator():
    """The update count restarts at every change of phase, so imitation's learner_update_000002.pt is
    later overwritten by warmup's and by ppo's (seen on the real laptop run: three versions of that
    one name were evaluated). The evaluator tells them apart by the file's time; pruning protects by
    name, which errs on the side of keeping."""
    d = new_dir("names_repeat")
    ckpt_dir = os.path.join(d, "run", "checkpoints")
    os.makedirs(ckpt_dir)
    path = os.path.join(ckpt_dir, "learner_update_000002.pt")
    results = []
    for i, phase in enumerate(("imitation", "warmup", "ppo")):
        with open(path + ".tmp", "wb") as f:                                   # the way learner.py writes: a new file renamed over the name
            f.write(phase.encode() * 10)
        os.replace(path + ".tmp", path)
        os.utime(path, (2000.0 + 100 * i, 2000.0 + 100 * i))
        ck = evaluate.newest_checkpoint(ckpt_dir, settle_s=0)
        assert ck.name == "learner_update_000002.pt"
        assert evaluate.opponents_done(results, ck) == []                      # not evaluated yet, whatever the earlier lines say
        results.append(eval_line("simple", 5, 10, f"2026-10-05T1{i}:00:00Z", update=2, phase=phase, mtime=round(ck.mtime, 3)))
        assert evaluate.opponents_done(results, ck) == ["simple"]
    assert [r["phase"] for r in results] == ["imitation", "warmup", "ppo"] and len({r["checkpoint_mtime"] for r in results}) == 3
    # status.py shows them as three rows, named by phase
    assert [f"{r['phase'][:3]}:{r['update']}" for r in status.evaluation_table(results)["simple"]] == ["imi:2", "war:2", "ppo:2"]
    # the copy pipeline.py keeps of a finished phase's last checkpoint (a hard link) survives the name being written again
    final_dir = os.path.join(d, "run", "phase_final")
    os.makedirs(final_dir)
    kept = os.path.join(final_dir, "ppo_learner_update_000002.pt")
    try:
        os.link(path, kept)
    except OSError:
        pytest.skip("hard links are not available here; pipeline.py copies the file instead")
    with open(path + ".tmp", "wb") as f:
        f.write(b"a later run")
    os.replace(path + ".tmp", path)
    assert open(kept, "rb").read() == b"ppo" * 10 and open(path, "rb").read() == b"a later run"


# ============================================================================================
# end to end: pipeline.py as a process, against the stub trainer and the fake servers
# ============================================================================================

def pid_alive(pid) -> bool:
    if not pid:
        return False
    if WIN:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return re.search(rf"\b{pid}\b", out) is not None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_tree(pid, grace=30.0) -> None:
    """Stops a process these tests started (and what it started), by its id. On POSIX every such
    process leads its own session, so its id is also its process group."""
    if not pid or not pid_alive(pid):
        return
    if WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + wait
        while time.time() < deadline and pid_alive(pid):
            time.sleep(0.2)
        if not pid_alive(pid):
            return


def deep_update(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


class Harness:
    """One run directory, one config, and the pipeline.py processes started on it."""

    def __init__(self, name, config=None, faults=None, scenario=None):
        self.dir = new_dir(name)
        self.run = os.path.join(self.dir, "run")
        self.cfg_path = os.path.join(self.dir, "config.json")
        self.scenario_path = os.path.join(self.dir, "scenario.json")
        self.ports = [BASE_PORT, BASE_PORT + 1]
        sc = {"tick_seconds": 0.05, "steps_per_tick": 200, "fault_dir": os.path.join(self.dir, "faults"), "faults": faults or []}
        sc.update(scenario or {})
        with open(self.scenario_path, "w") as f:
            json.dump(sc, f, indent=1)
        cfg = {
            "servers": {"count": 2, "base_port": BASE_PORT, "ready_timeout_seconds": 60, "settle_seconds": 0.1, "stop_timeout_seconds": 15,
                        "port_free_timeout_seconds": 30, "command": ["{python}", "tests/fake_servers.py", "--dir", "unused", "--ports", "{ports}"]},
            "trainer": {"command": ["{python}", "tests/fake_trainer.py"], "stop_timeout_seconds": 30, "failure_stop_timeout_seconds": 2,
                        "kill_timeout_seconds": 15},
            "train_defaults": {"battles": 32, "rooms_per_pair": 16, "steps_per_update": 1000, "minibatch": 256, "seq_len": 256, "save_every": 2,
                               "teacher": "psbot", "extra_train_args": ["--fake-scenario", self.scenario_path]},
            "phases": {
                "imitation": {"min_decisions": 6000, "max_decisions": 40000,
                              "plateau": {"metric": "ploss", "window_decisions": 4000, "min_improvement_pct": 2.0, "smooth_updates": 3}},
                "warmup": {"min_decisions": 4000, "max_decisions": 12000,
                           "plateau": {"metric": "vloss", "window_decisions": 3000, "min_improvement_pct": 2.0, "smooth_updates": 3},
                           "train": {"save_every": 1}},
                "ppo": {"target_decisions": 20000, "extra_train_args": ["--fake-note", "ppo-only"]},
            },
            "watchdog": {"stall_minutes": 0.08, "startup_grace_minutes": 0.5, "update_stall_minutes": 0.08, "max_restarts_per_hour": 10,
                         "backoff_seconds": [0.2, 0.5], "poll_seconds": 0.1},
            "disk": {"min_free_gb": 0.2, "check_seconds": 1},
            "retention": {"enabled": False},
            "reporting": {"state_seconds": 0.2, "sample_seconds": 0.5, "progress_log_seconds": 2},
            "on_done": "idle",
        }
        self.cfg = deep_update(cfg, config)
        with open(self.cfg_path, "w") as f:
            json.dump(self.cfg, f, indent=1)
        self.proc = None
        self.procs = []
        self.starts = 0

    # -- processes
    def env(self):
        return dict(os.environ, PYTHONUTF8="1", PSPPO_NO_DELETE="1")

    def command(self, *extra):
        return [sys.executable, PIPELINE, "--config", self.cfg_path, "--run-dir", self.run, *extra]

    def start(self):
        self.starts += 1
        self.stdout_path = os.path.join(self.dir, f"pipeline_stdout_{self.starts}.log")
        out = open(self.stdout_path, "ab")
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
        self.proc = subprocess.Popen(self.command(), cwd=REPO, env=self.env(), stdin=subprocess.DEVNULL, stdout=out,
                                     stderr=subprocess.STDOUT, **kwargs)
        out.close()
        self.procs.append(self.proc)
        return self.proc

    def call(self, *extra, timeout=120):
        return subprocess.run(self.command(*extra), cwd=REPO, env=self.env(), capture_output=True, text=True, timeout=timeout)

    def signal_stop(self):
        """What systemd does on the box (SIGTERM); on Windows the closest thing, Ctrl+Break."""
        self.proc.send_signal(signal.CTRL_BREAK_EVENT if WIN else signal.SIGTERM)

    def wait_exit(self, timeout=90):
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise AssertionError(f"pipeline.py did not exit within {timeout} s\n{self.tail()}")

    def close(self):
        for p in self.procs:
            if p.poll() is not None:
                continue
            if WIN:
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            else:
                p.send_signal(signal.SIGTERM)                                # it stops its own children on this
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
        st = self.state()
        for key in ("trainer", "servers"):                                   # whatever the supervisor could not stop itself
            stop_tree((st.get(key) or {}).get("pid") if (st.get(key) or {}).get("running") else None)

    # -- files
    def state(self):
        return status.read_json(os.path.join(self.run, "state.json")) or {}

    def events(self, name=None):
        evs = pipeline.read_jsonl(os.path.join(self.run, "events.jsonl"))
        return [e for e in evs if name is None or e["event"] == name]

    def metrics(self):
        return pipeline.read_jsonl(os.path.join(self.run, "metrics.jsonl"))

    def tail(self, n=40):
        try:
            with open(self.stdout_path, "r", encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return "(no output)"

    def wait_for(self, pred, what, timeout=120, may_exit=False):
        deadline = time.time() + timeout
        st = {}
        while time.time() < deadline:
            st = self.state()
            try:
                if pred(st):
                    return st
            except (KeyError, TypeError, IndexError):
                pass
            if not may_exit and self.proc is not None and self.proc.poll() is not None:
                raise AssertionError(f"pipeline.py exited with {self.proc.returncode} while waiting for {what}\n{self.tail()}")
            time.sleep(0.05)
        raise AssertionError(f"timed out after {timeout} s waiting for {what}; phase={st.get('phase')} status={st.get('status')}\n{self.tail()}")

    def ports_closed(self, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not any(pipeline.port_open("127.0.0.1", p, 0.2) for p in self.ports):
                return True
            time.sleep(0.2)
        return False

    def checkpoint(self, *parts):
        with open(os.path.join(self.run, *parts), "r", encoding="utf-8") as f:
            return json.load(f)


@pytest.fixture
def harness():
    made = []

    def make(*a, **k):
        assert not any(pipeline.port_open("127.0.0.1", p, 0.2) for p in (BASE_PORT, BASE_PORT + 1)), \
            f"ports {BASE_PORT} and {BASE_PORT + 1} must be free for the end-to-end tests"
        h = Harness(*a, **k)
        made.append(h)
        return h
    yield make
    for h in made:
        h.close()
    for h in made:
        assert h.ports_closed(), "a fake server was left running"


def test_e2e_all_phases_through_a_hang_a_crash_and_failing_updates(harness):
    h = harness("e2e_full",
                config={"phases": {"ppo": {"target_decisions": 60000}}},
                faults=[{"id": "hang", "kind": "hang", "mode": "imitation", "at_steps": 3000},
                        {"id": "die", "kind": "die", "mode": "warmup", "at_steps": 2500},
                        {"id": "noupd", "kind": "no_updates", "mode": "ppo", "at_steps": 6000}])
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=240)
    assert st["status"] == "done"

    # the three failures, each followed by a restart, each in its phase
    fails = {(e["phase"], e["what"]) for e in h.events("failure")}
    assert fails == {("imitation", "stall"), ("warmup", "trainer_exit"), ("ppo", "update_stall")}, h.events("failure")
    assert len(h.events("restart")) == 3 and st["restarts_total"] == 3 and len(st["failures"]) == 3
    assert h.events("update_failed")[0]["error"].startswith("LoweringException")
    exits = {e["name"] + ":" + str(e["segment"]): e for e in h.events("trainer_exit")}
    hung = [e for e in exits.values() if e["name"] == "imitation"][0]
    assert hung["killed"] is True and hung["final_checkpoint"] is None           # a hung trainer cannot save; it was terminated
    died = [e for e in exits.values() if e["name"] == "warmup"][0]
    assert died["code"] == 1 and died["killed"] is False                         # it had exited by itself

    # after the hang and after the crash the count went back to the checkpoint each resumed from
    rollbacks = {e["phase"]: e for e in h.events("rollback")}
    assert (rollbacks["imitation"]["from"], rollbacks["imitation"]["to"], rollbacks["imitation"]["lost"]) == (3000, 2000, 1000)
    assert (rollbacks["warmup"]["from"], rollbacks["warmup"]["to"], rollbacks["warmup"]["lost"]) == (2600, 2000, 600)
    assert "ppo" not in rollbacks                                                # the update stall ended in a clean stop: nothing lost
    assert st["phases"]["imitation"]["lost_decisions"] == 1000 and st["phases"]["warmup"]["lost_decisions"] == 600

    # every phase ended by its rule, and its count is the stub's own counter in its last checkpoint
    ends = {e["name"]: e for e in h.events("phase_end")}
    assert list(ends) == ["imitation", "warmup", "ppo"]
    assert "plateau: ploss" in ends["imitation"]["reason"] and "plateau: vloss" in ends["warmup"]["reason"]
    assert "60,000" in ends["ppo"]["reason"]
    total = 0
    for phase in pipeline.PHASES:
        ph = st["phases"][phase]
        assert ph["status"] == "done" and ph["final_confirmed"] is True
        kept = h.checkpoint("phase_final", f"{phase}_{ph['final_checkpoint']}")
        assert kept["mode"] == phase and kept["total_steps"] == ph["decisions"] and kept["update_idx"] == ph["updates"], (phase, kept, ph)
        total += ph["decisions"]
    assert st["phases"]["imitation"]["decisions"] >= 6000 and 4000 <= st["phases"]["warmup"]["decisions"] <= 16000
    assert 60000 <= st["phases"]["ppo"]["decisions"] <= 70000                    # the target, plus what arrived while it stopped
    assert st["decisions_total"] == total and st["updates_total"] == sum(st["phases"][p]["updates"] for p in pipeline.PHASES)

    # metrics.jsonl: one object per learner update, with phase, time, both counts and the losses
    ms = h.metrics()
    assert {m["phase"] for m in ms} == set(pipeline.PHASES)
    assert all({"time", "wall_time", "update", "total_steps", "decisions_phase", "decisions_total", "ploss", "vloss", "loss", "kl"} <= set(m) for m in ms)
    assert all(m["decisions_total"] >= m["decisions_phase"] for m in ms)
    ppo = [m for m in ms if m["phase"] == "ppo"]
    assert ppo[-1]["update"] == st["phases"]["ppo"]["updates"] and ppo[-1]["decisions_total"] <= st["decisions_total"]

    # a new log file per trainer start and per server start; the servers were restarted every time
    starts = h.events("trainer_start")
    logs = sorted(f for f in os.listdir(os.path.join(h.run, "logs")) if f.startswith("trainer_"))
    assert len(starts) == 6 and len(logs) == 6 and len({e["log"] for e in starts}) == 6 and st["segment"] == 6
    assert len([f for f in os.listdir(os.path.join(h.run, "logs")) if f.startswith("servers_")]) == 6
    # stop files: none left behind, each used one renamed aside (5 clean stop requests: 3 phase ends, the hang, the update stall)
    assert not os.path.exists(os.path.join(h.run, "trainer.stop"))
    assert len(os.listdir(os.path.join(h.run, "logs", "stop_files"))) == 5
    # the phase's own extra arguments reached the trainer
    notes = open(os.path.join(h.run, "fake_notes.txt")).read().split("\n")
    assert "ppo ppo-only" in notes and not any(n.startswith(("imitation", "warmup")) for n in notes)

    r = subprocess.run([sys.executable, os.path.join(REPO, "status.py"), "--run-dir", h.run, "--json", "--no-gpu"], capture_output=True, text=True)
    info = json.loads(r.stdout)
    assert info["health"] == ["DONE"] and info["restarts"]["total"] == 3 and info["decisions"]["total"] == total
    assert h.proc.poll() is None                                                 # on_done "idle": still there, doing nothing
    assert h.ports_closed() and not pid_alive((st.get("servers") or {}).get("pid"))
    r = h.call("--can-start")
    assert r.returncode == 1 and "done" in r.stdout
    r = subprocess.run([sys.executable, PIPELINE, "--run-dir", h.run, "--stop", "--wait", "60"], capture_output=True, text=True)
    assert r.returncode == 0 and "has stopped" in r.stdout
    assert h.wait_exit(30) == 0


def test_e2e_clean_stop_in_the_middle_then_resume_in_the_same_phase(harness):
    h = harness("e2e_resume",
                config={"phases": {"warmup": {"min_decisions": 12000, "max_decisions": 16000}, "ppo": {"target_decisions": 6000}}},
                scenario={"steps_per_tick": 100})
    h.start()
    h.wait_for(lambda s: s["phase"] == "warmup" and s["status"] == "running" and s["phases"]["warmup"]["decisions"] >= 2000,
               "warmup to be under way")
    trainer_pid = h.state()["trainer"]["pid"]
    servers_pid = h.state()["servers"]["pid"]
    h.signal_stop()
    assert h.wait_exit(60) == 0, h.tail()

    st = h.state()
    assert st["phase"] == "warmup" and st["status"] == "stopped" and "SIG" in st["status_reason"].upper(), st["status_reason"]
    stopped = h.events("stopped")[-1]
    last_exit = h.events("trainer_exit")[-1]
    assert last_exit["graceful"] is True and last_exit["killed"] is False and last_exit["code"] == 0
    assert last_exit["final_checkpoint"] and stopped["final_checkpoint"] == last_exit["final_checkpoint"]
    d1 = st["phases"]["warmup"]["decisions"]
    saved = h.checkpoint("checkpoints", last_exit["final_checkpoint"])
    assert saved["mode"] == "warmup" and saved["total_steps"] == d1 and 2000 <= d1 < 12000
    assert not os.path.exists(os.path.join(h.run, "trainer.stop"))
    assert not pid_alive(trainer_pid) and not pid_alive(servers_pid) and h.ports_closed()
    imitation_before = dict(st["phases"]["imitation"])
    n_events = len(h.events())
    r = h.call("--can-start")
    assert r.returncode == 0 and "phase warmup" in r.stdout

    # the same command again, as after a reboot
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done", "the resumed run to reach done", timeout=180)
    new = h.events()[n_events:]
    assert new[0]["event"] == "phase_resume" and new[0]["name"] == "warmup" and new[0]["decisions"] == d1
    first_start = [e for e in new if e["event"] == "trainer_start"][0]
    assert first_start["name"] == "warmup" and first_start["newest_checkpoint"] == last_exit["final_checkpoint"]
    assert [e["name"] for e in h.events("phase_start")] == ["imitation", "warmup", "ppo"]     # no phase was started twice
    assert not h.events("rollback")
    # the resumed trainer loaded the warmup checkpoint and its counter continued from d1
    with open(os.path.join(h.run, first_start["log"]), "r", encoding="utf-8", errors="replace", newline="") as f:
        parsed = [pipeline.parse_trainer_line(l) for l in f]
    kinds = [p["kind"] for p in parsed if p]
    assert "resumed" in kinds and "phase_change" not in kinds and "init_policy" not in kinds
    assert [p for p in parsed if p and p["kind"] == "telemetry"][0]["total_steps"] == d1
    warm = [m for m in h.metrics() if m["phase"] == "warmup"]
    assert all(a["decisions_phase"] <= b["decisions_phase"] for a, b in zip(warm, warm[1:]))   # never back to zero, never doubled
    assert [m["update"] for m in warm] == list(range(1, len(warm) + 1))
    assert st["phases"]["imitation"]["decisions"] == imitation_before["decisions"]
    ph = st["phases"]["warmup"]
    kept = h.checkpoint("phase_final", f"warmup_{ph['final_checkpoint']}")
    assert kept["total_steps"] == ph["decisions"] >= 12000 and ph["trainer_starts"] == 2 and ph["lost_decisions"] == 0
    assert st["decisions_total"] == sum(st["phases"][p]["decisions"] for p in pipeline.PHASES)
    # the checkpoint named like warmup's last one may since have been overwritten by ppo; the copy has not
    assert h.checkpoint("phase_final", f"imitation_{st['phases']['imitation']['final_checkpoint']}")["mode"] == "imitation"

    # stop it once more, through the stop file this time
    r = h.call("--stop", "--wait", "60")
    assert r.returncode == 0 and h.wait_exit(30) == 0


def test_e2e_restart_limit_marks_the_run_failed_and_it_stays_failed(harness):
    h = harness("e2e_limit",
                config={"watchdog": {"max_restarts_per_hour": 3, "backoff_seconds": [0.3, 0.6, 0.9]}},
                faults=[{"id": "always", "kind": "die_at_start", "repeat": True}])
    t0 = time.time()
    h.start()
    assert h.wait_exit(120) == pipeline.EXIT_FAILED, h.tail()
    took = time.time() - t0
    st = h.state()
    assert st["status"] == "failed" and st["phase"] == "imitation"
    assert "4 failures within an hour (max_restarts_per_hour is 3)" in st["status_reason"] and "exited by itself with code 1" in st["status_reason"]
    assert len(h.events("trainer_start")) == 4 and len(h.events("failure")) == 4 and len(h.events("failed")) == 1
    assert [e["wait_seconds"] for e in h.events("restart")] == [0.3, 0.6, 0.9]   # it backed off, more each time
    assert took > 0.3 + 0.6 + 0.9
    assert h.ports_closed()
    info = json.loads(subprocess.run([sys.executable, os.path.join(REPO, "status.py"), "--run-dir", h.run, "--json", "--no-gpu"],
                                     capture_output=True, text=True).stdout)
    assert info["health"] == ["FAILED"]

    # systemd asks before every start; a failed run says no, and a start by hand refuses too
    r = h.call("--can-start")
    assert r.returncode == 1 and "marked failed" in r.stdout
    h.start()
    assert h.wait_exit(60) == pipeline.EXIT_FAILED
    assert len(h.events("trainer_start")) == 4                                   # nothing was started
    r = subprocess.run([sys.executable, PIPELINE, "--run-dir", h.run, "--clear-failed"], capture_output=True, text=True)
    assert r.returncode == 0 and "resumes phase imitation" in r.stdout
    assert h.call("--can-start").returncode == 0 and h.state()["status"] == "stopped" and h.state()["failures"] == []


def test_e2e_a_trainer_that_ignores_the_stop_file_is_terminated(harness):
    h = harness("e2e_ignore_stop",
                config={"trainer": {"stop_timeout_seconds": 2}, "phases": {"imitation": {"min_decisions": 10 ** 6, "max_decisions": 10 ** 7}}},
                faults=[{"id": "deaf", "kind": "ignore_stop", "repeat": True}])
    h.start()
    st = h.wait_for(lambda s: s["status"] == "running" and s["phases"]["imitation"]["decisions"] >= 2000, "the trainer to run")
    trainer_pid = st["trainer"]["pid"]
    t0 = time.time()
    h.signal_stop()
    assert h.wait_exit(60) == 0, h.tail()
    took = time.time() - t0
    ex = h.events("trainer_exit")[-1]
    assert ex["graceful"] is True and ex["killed"] is True and ex["final_checkpoint"] is None and ex["seconds_to_stop"] >= 2.0
    assert 2.0 <= took < 30
    assert h.state()["status"] == "stopped" and not pid_alive(trainer_pid) and h.ports_closed()
    assert not os.path.exists(os.path.join(h.run, "trainer.stop"))               # the unused stop file was moved aside too


def test_e2e_disk_guard_refuses_to_start_on_a_full_disk(harness):
    h = harness("e2e_disk", config={"disk": {"min_free_gb": 10 ** 7}})
    h.start()
    assert h.wait_exit(60) == pipeline.EXIT_DISK, h.tail()
    st = h.state()
    assert st["status"] == "disk_low" and "below min_free_gb" in st["status_reason"]
    assert not h.events("trainer_start") and h.events("disk_low")
    r = h.call("--can-start")
    assert r.returncode == 1 and "min_free_gb" in r.stdout
    out = subprocess.run([sys.executable, os.path.join(REPO, "status.py"), "--run-dir", h.run, "--no-gpu"], capture_output=True, text=True).stdout
    assert out.splitlines()[0] == "HEALTH: DISK_LOW"


def test_disk_guard_stops_a_running_trainer_cleanly(harness, monkeypatch):
    """The supervisor in this process (in a thread), so the free-space reading can be faked while it runs."""
    h = harness("e2e_disk_midrun", config={"phases": {"imitation": {"min_decisions": 10 ** 6, "max_decisions": 10 ** 7}}})
    low = threading.Event()
    monkeypatch.setattr(pipeline, "free_gb", lambda path: 0.05 if low.is_set() else 500.0)
    monkeypatch.setenv("PSPPO_NO_DELETE", "1")
    sup = pipeline.Supervisor(pipeline.load_config(h.cfg_path), h.run, h.cfg_path)
    result = {}
    th = threading.Thread(target=lambda: result.update(code=sup.run()))
    th.start()
    try:
        h.wait_for(lambda s: s["status"] == "running" and s["phases"]["imitation"]["decisions"] >= 3000, "the trainer to run", may_exit=True)
        low.set()
        th.join(60)
        assert not th.is_alive()
    finally:
        sup.request_stop("test cleanup")
        th.join(60)
    assert result["code"] == pipeline.EXIT_DISK
    st = h.state()
    assert st["status"] == "disk_low" and "0.1 GB free" in st["status_reason"]
    ex = h.events("trainer_exit")[-1]
    assert ex["graceful"] is True and ex["killed"] is False and ex["final_checkpoint"]     # it saved before it stopped
    assert h.checkpoint("checkpoints", ex["final_checkpoint"])["total_steps"] == st["phases"]["imitation"]["decisions"]
    assert h.ports_closed()


def test_e2e_shutdown_when_done_waits_for_the_final_evaluation_and_runs_once(harness):
    marker_dir = new_dir("shutdown_marker")
    marker = os.path.join(marker_dir, "shutdown_was_called.txt")
    harmless = [sys.executable, "-c", "import sys; open(sys.argv[1], 'a').write('shutdown\\n')", marker]
    h = harness("e2e_shutdown",
                config={"phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 3000}},
                        "on_done": "shutdown", "on_done_wait_minutes": 2, "on_done_grace_minutes": 0, "shutdown_command": harmless})
    h.start()
    h.wait_for(lambda s: s["phase"] == "done", "ppo to finish")
    assert [e["name"] for e in h.events("phase_start")] == ["ppo"]                # the two skipped phases never ran
    time.sleep(2.0)
    assert not os.path.exists(marker) and h.proc.poll() is None                   # waiting for eval/final.done
    assert h.call("--can-start").returncode == 0                                  # after a reboot it would pick the wait up again
    with open(os.path.join(h.run, "eval", "final.done"), "w") as f:
        f.write("the evaluator writes this after the last checkpoint\n")
    h.wait_for(lambda s: s.get("shutdown_issued"), "the shutdown command to be issued", timeout=30)
    deadline = time.time() + 20
    while time.time() < deadline and not (os.path.exists(marker) and open(marker).read()):
        time.sleep(0.1)
    assert open(marker).read() == "shutdown\n"
    ev = h.events("shutdown")[0]
    assert ev["final_evaluation"] is True and ev["command"] == harmless
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0
    # the box is started again later: no second shutdown, and systemd does not even start the unit
    r = h.call("--can-start")
    assert r.returncode == 1 and "done" in r.stdout
    h.start()
    time.sleep(3.0)
    assert h.proc.poll() is None and open(marker).read() == "shutdown\n" and len(h.events("shutdown")) == 1
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


def test_e2e_shutdown_goes_ahead_when_the_final_evaluation_never_arrives(harness):
    marker_dir = new_dir("shutdown_marker_late")
    marker = os.path.join(marker_dir, "shutdown_was_called.txt")
    harmless = [sys.executable, "-c", "import sys; open(sys.argv[1], 'a').write('shutdown\\n')", marker]
    h = harness("e2e_shutdown_late",
                config={"phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 3000}},
                        "on_done": "shutdown", "on_done_wait_minutes": 0.05, "on_done_grace_minutes": 0, "shutdown_command": harmless})
    h.start()
    h.wait_for(lambda s: s.get("shutdown_issued"), "the shutdown after the wait ran out", timeout=90)
    assert h.events("shutdown")[0]["final_evaluation"] is False
    done_at, issued_at = h.events("done")[0]["ts"], h.events("shutdown")[0]["ts"]
    assert issued_at - done_at >= 3.0                                             # it did wait its 0.05 minutes
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0
    assert open(marker).read() == "shutdown\n"


def test_e2e_a_checkpoint_that_will_not_load_is_set_aside_not_trained_over(harness):
    h = harness("e2e_quarantine",
                # warmup saves only on its way out here: the stub runs thousands of times faster than the
                # real trainer, whose first periodic save is many minutes after its start
                config={"phases": {"warmup": {"min_decisions": 2000, "max_decisions": 3000, "plateau": None, "train": {"save_every": 50}},
                                   "ppo": {"target_decisions": 3000}}},
                # sigterm_saves: on Linux a SIGTERM makes train.py save a final checkpoint (Ray turns it
                # into a SystemExit). After a failed resume that is a checkpoint of untrained weights,
                # the newest file, which every later start would resume from. The supervisor's kill
                # of this trainer therefore must not begin with a SIGTERM.
                faults=[{"id": "torn", "kind": "resume_failed", "mode": "warmup"},
                        {"id": "ray", "kind": "sigterm_saves", "repeat": True}])
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=180)
    q = h.events("checkpoint_quarantined")
    assert len(q) == 1 and len(st["quarantined"]) == 1
    name = q[0]["name"]
    assert name == st["phases"]["imitation"]["final_checkpoint"]                 # the newest file when warmup first started
    moved = h.checkpoint(q[0]["moved_to"])
    assert moved["mode"] == "imitation" and os.path.dirname(q[0]["moved_to"]) == "checkpoints/quarantine"
    fail = [e for e in h.events("failure") if e["what"] == "resume_failed"]
    assert len(fail) == 1 and "could not load " + name in fail[0]["reason"]
    bad_exit = [e for e in h.events("trainer_exit") if e["name"] == "warmup"][0]
    assert bad_exit["graceful"] is False and bad_exit["killed"] is True and bad_exit["final_checkpoint"] is None   # no save of untrained weights
    assert not os.path.exists(os.path.join(h.run, "checkpoints", "learner_update_000000.pt"))
    # the restart resumed from the checkpoint before it, which was still imitation's
    second = [e for e in h.events("trainer_start") if e["name"] == "warmup"][1]
    assert second["newest_checkpoint"] != name and h.checkpoint("phase_final", f"imitation_{name}")["mode"] == "imitation"
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


def test_e2e_a_restarted_server_restarts_the_trainer(harness):
    marks = new_dir("server_kill")
    once, now_please = os.path.join(marks, "killed_once"), os.path.join(marks, "kill_now")
    h = harness("e2e_server_restart",
                config={"servers": {"command": ["{python}", "tests/fake_servers.py", "--dir", "unused", "--ports", "{ports}",
                                                "--kill-child-when-file", now_please, "--once-marker", once]},
                        "trainer": {"failure_stop_timeout_seconds": 30},
                        "phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 40000}}},
                scenario={"steps_per_tick": 100})
    h.start()
    h.wait_for(lambda s: s["status"] == "running" and s["phases"]["ppo"]["decisions"] >= 2000, "the trainer to be under way")
    with open(now_please, "w") as f:                                             # a server goes down under a running trainer
        f.write("now\n")
    st = h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=180)
    fail = h.events("failure")
    assert len(fail) == 1 and fail[0]["what"] == "server_restart" and f"server on {BASE_PORT} exited with" in fail[0]["reason"]
    ex = h.events("trainer_exit")[0]
    assert ex["graceful"] is True and ex["final_checkpoint"]                      # the trainer was stopped cleanly, nothing lost
    assert not h.events("rollback") and len(h.events("trainer_start")) == 2 and st["restarts_total"] == 1
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


FAKE_SHOWDOWN_JS = r"""
// Stand-in for the Pokemon Showdown launcher, started the way start_servers.py starts the real one:
//   node --max-old-space-size=N pokemon-showdown start --no-security --skip-build --port P
// It listens on 127.0.0.1:P, says so on stdout the way a server does, writes one line to stderr,
// and leaves its PID in pid_P next to this file so a test can stop exactly this process.
const net = require('net');
const fs = require('fs');
const path = require('path');
const port = Number(process.argv[process.argv.indexOf('--port') + 1]);
fs.writeFileSync(path.join(__dirname, 'pid_' + port), String(process.pid));
const server = net.createServer((sock) => { sock.on('error', () => {}); sock.end(); });
server.listen(port, '127.0.0.1', () => {
  console.log('Worker 1 now listening on 127.0.0.1:' + port);
  console.error('stand-in server on ' + port + ': args ' + process.argv.slice(2).join(' '));
});
"""


def fake_showdown_checkout(name):
    """A directory start_servers.py accepts as a built Showdown checkout with the plugin installed."""
    import shutil
    if not shutil.which("node"):
        pytest.skip("node is not on PATH")
    d = os.path.join(new_dir(name), "showdown")
    for rel in (("server", "chat-plugins", "rlspawn.ts"), ("dist", "server", "chat-plugins", "rlspawn.js")):
        os.makedirs(os.path.join(d, *rel[:-1]), exist_ok=True)
        with open(os.path.join(d, *rel), "w") as f:
            f.write("// stand-in\n")
    with open(os.path.join(d, "pokemon-showdown"), "w") as f:
        f.write(FAKE_SHOWDOWN_JS)
    return d


def kill_one_process(pid):
    """Stops exactly one process, by its id (no tree, no name)."""
    if WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGKILL)


def wait_until(pred, timeout=30.0, step=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def test_start_servers_py_log_dir_restart_line_and_default_location():
    """The real start_servers.py against a stand-in Showdown checkout: --log-dir is where each
    server's own output goes, a server that dies is started again, and the line it prints about
    that is the one pipeline.py looks for."""
    assert not any(pipeline.port_open("127.0.0.1", p, 0.2) for p in (BASE_PORT, BASE_PORT + 1)), "the test ports must be free"
    sd = fake_showdown_checkout("start_servers")
    log_dir = os.path.abspath(os.path.join(os.path.dirname(sd), "run", "logs", "servers"))
    out_path = os.path.join(os.path.dirname(sd), "launcher.out")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
    with open(out_path, "ab") as out:
        proc = subprocess.Popen([sys.executable, os.path.join(REPO, "start_servers.py"), "--dir", sd, "--ports", f"{BASE_PORT}-{BASE_PORT + 1}",
                                 "--heap-mb", "256", "--log-dir", log_dir], cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, **kwargs)
    try:
        assert wait_until(lambda: all(pipeline.port_open("127.0.0.1", p, 0.2) for p in (BASE_PORT, BASE_PORT + 1)), 60), read_text(out_path)
        for port in (BASE_PORT, BASE_PORT + 1):
            assert wait_until(lambda: "now listening" in read_text(os.path.join(log_dir, f"server_{port}.out")), 10)
            out_text = read_text(os.path.join(log_dir, f"server_{port}.out"))
            err_text = read_text(os.path.join(log_dir, f"server_{port}.err"))
            assert out_text.count(f"starts the server on port {port}") == 1 and f"listening on 127.0.0.1:{port}" in out_text
            assert f"--no-security --skip-build --port {port}" in err_text and "start" in err_text
        assert not os.path.exists(os.path.join(sd, "logs"))                      # nothing went to the checkout
        assert wait_until(lambda: f"logs in {log_dir}" in read_text(out_path), 15), read_text(out_path)
        # one server dies
        pid = int(read_text(os.path.join(sd, f"pid_{BASE_PORT}")))
        kill_one_process(pid)
        assert wait_until(lambda: "restarting" in read_text(out_path), 20), read_text(out_path)
        line = [l for l in read_text(out_path).splitlines() if "restarting" in l][0]
        assert re.fullmatch(rf"server on {BASE_PORT} exited with -?\d+, restarting", line.strip()), line
        assert "exited with" in line and "restarting" in line                  # the two things pipeline.Supervisor._drain looks for
        assert wait_until(lambda: read_text(os.path.join(sd, f"pid_{BASE_PORT}")) not in ("", str(pid)), 20)
        assert wait_until(lambda: pipeline.port_open("127.0.0.1", BASE_PORT, 0.2), 30)
        assert read_text(os.path.join(log_dir, f"server_{BASE_PORT}.out")).count(f"starts the server on port {BASE_PORT}") == 2   # appended to
        assert pipeline.port_open("127.0.0.1", BASE_PORT + 1, 0.2)                # the other one was left alone
    finally:
        stop_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    assert wait_until(lambda: not any(pipeline.port_open("127.0.0.1", p, 0.2) for p in (BASE_PORT, BASE_PORT + 1)), 20)
    # without --log-dir the output goes where it always went: <dir>/logs/multi
    with open(out_path, "ab") as out:
        proc = subprocess.Popen([sys.executable, os.path.join(REPO, "start_servers.py"), "--dir", sd, "--ports", str(BASE_PORT)], cwd=REPO, env=env,
                                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, **kwargs)
    try:
        assert wait_until(lambda: "now listening" in read_text(os.path.join(sd, "logs", "multi", f"server_{BASE_PORT}.out")), 60)
    finally:
        stop_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    assert wait_until(lambda: not pipeline.port_open("127.0.0.1", BASE_PORT, 0.2), 20)


def test_e2e_real_start_servers_py_logs_under_the_run_and_a_killed_server_restarts_everything(harness):
    """pipeline.py with its default server command (the real start_servers.py) and a stand-in
    Showdown checkout: the servers' own output lands in RUN/logs/servers, and killing one server
    process by its id is noticed, the trainer is stopped cleanly and everything starts again."""
    sd = fake_showdown_checkout("e2e_real_launcher")
    h = harness("e2e_real_launcher",
                config={"showdown_dir": sd, "servers": {"command": None, "heap_mb": 256},
                        "phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 10 ** 9}}},
                scenario={"tick_seconds": 0.1, "steps_per_tick": 100})
    h.start()
    st = h.wait_for(lambda s: s["status"] == "running" and s["trainer"]["progress_seen"], "the trainer to report decisions", timeout=120)
    log_dir = os.path.abspath(os.path.join(h.run, "logs", "servers"))
    cmd_line = [l for l in read_text(os.path.join(h.run, "logs", "pipeline.log")).splitlines() if "starting the servers:" in l][0]
    assert "start_servers.py" in cmd_line and "--log-dir" in cmd_line and log_dir in cmd_line
    for port in h.ports:
        assert f"listening on 127.0.0.1:{port}" in read_text(os.path.join(log_dir, f"server_{port}.out"))
        assert f"--port {port}" in read_text(os.path.join(log_dir, f"server_{port}.err"))
    assert not os.path.exists(os.path.join(sd, "logs"))
    h.wait_for(lambda s: s["phases"]["ppo"]["updates"] >= 2, "two updates", timeout=60)
    pid = int(read_text(os.path.join(sd, f"pid_{h.ports[0]}")))
    t_kill = time.time()
    kill_one_process(pid)
    st = h.wait_for(lambda s: s["segment"] >= 2 and s["status"] == "running" and s["trainer"]["progress_seen"], "the restarted trainer to report", timeout=120)
    took = time.time() - t_kill
    fail = h.events("failure")
    assert len(fail) == 1 and fail[0]["what"] == "server_restart" and f"server on {h.ports[0]} exited with" in fail[0]["reason"]
    ex = h.events("trainer_exit")[0]
    assert ex["graceful"] is True and ex["killed"] is False and ex["final_checkpoint"]   # stopped through the stop file: it saved first
    assert st["trainer"]["resumed_from"] == ex["final_checkpoint"] and not h.events("rollback")
    assert st["servers"]["running"] is True and len(h.events("trainer_start")) == 2 and st["restarts_total"] == 1
    assert read_text(os.path.join(log_dir, f"server_{h.ports[0]}.out")).count("starts the server on port") >= 2
    print(f"one killed server process: {took:.1f} s until the restarted trainer reported decisions")
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0
    assert h.ports_closed()


# A parent that starts a child of its own which holds a port, says so, and then stays for
# argv[2] seconds (0: it leaves at once and the port holder is on its own).
PORT_HOLDER_PARENT = (
    "import subprocess, sys, time\n"
    "holder = 'import socket, sys, time; s = socket.socket(); s.bind((\"127.0.0.1\", int(sys.argv[1]))); s.listen(8); time.sleep(600)'\n"
    "c = subprocess.Popen([sys.executable, '-c', holder, sys.argv[1]])\n"
    "print('holder', c.pid, flush=True)\n"
    "time.sleep(float(sys.argv[2]))\n"
)


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_port_holder_child(name, stay_seconds):
    d = new_dir(name)
    port = free_port()
    child = pipeline.Child(name, [sys.executable, "-c", PORT_HOLDER_PARENT, str(port), str(stay_seconds)], d,
                           dict(os.environ, PYTHONUNBUFFERED="1"), os.path.join(d, "out.log"))
    assert wait_until(lambda: pipeline.port_open("127.0.0.1", port, 0.2), 30), "the port holder did not come up"
    return child, port


def test_kill_tree_takes_what_a_process_started_after_the_process_itself_has_gone():
    """Seen on the laptop under memory pressure: the launcher of the servers was gone, the servers
    were not, and the supervisor exited "stopped" with both ports still open. On Windows nothing
    ties a process to its children once it has gone, so the old code had nothing left to kill."""
    child, port = start_port_holder_child("orphan", 0)
    try:
        assert wait_until(lambda: not child.alive(), 30)                        # the process we started has left by itself
        assert pipeline.port_open("127.0.0.1", port, 0.5)                       # what it started has not
        assert child.leftover() != []
        left = child.kill_tree(grace=5.0)
        assert left == []
        assert wait_until(lambda: not pipeline.port_open("127.0.0.1", port, 0.2), 15), "the orphan still holds its port"
        assert child.leftover() == [] and child.kill_tree(grace=1.0) == []      # and again is harmless
    finally:
        child.kill_tree(grace=2.0)
        child.join_output(2.0)


def test_kill_tree_does_not_depend_on_taskkill(monkeypatch):
    """The same day: taskkill itself failed (it is a new process that has to walk the tree, on a
    machine with 0.2 GB of memory free), and the fallback killed the root only."""
    if not WIN:
        pytest.skip("taskkill is the Windows path; elsewhere the kill is one system call on the process group")
    real_run = subprocess.run
    calls = []

    def run_without_taskkill(cmd, *a, **k):
        if cmd and cmd[0] == "taskkill":
            calls.append(list(cmd))
            raise OSError(1455, "The paging file is too small for this operation to complete")
        return real_run(cmd, *a, **k)

    child, port = start_port_holder_child("no_taskkill", 600)
    monkeypatch.setattr(pipeline.subprocess, "run", run_without_taskkill)
    try:
        assert child.alive() and child.job is not None and len(child.job.pids()) >= 2
        left = child.kill_tree(grace=5.0)
        assert left == [] and not child.alive()
        assert calls and calls[0][:3] == ["taskkill", "/PID", str(child.pid)]   # it was tried, it failed, and it did not matter
        assert wait_until(lambda: not pipeline.port_open("127.0.0.1", port, 0.2), 15), "the child of the killed process still holds its port"
    finally:
        monkeypatch.setattr(pipeline.subprocess, "run", real_run)
        child.kill_tree(grace=2.0)
        child.join_output(2.0)


def test_a_job_object_holds_the_whole_tree_and_ends_with_its_owner():
    if not WIN:
        pytest.skip("Windows job objects")
    child, port = start_port_holder_child("job", 600)
    try:
        pids = child.job.pids()
        assert child.pid in pids and len(pids) >= 2 and len(set(pids)) == len(pids)
        assert sorted(child.leftover()) == sorted(pids)
        # closing the last handle is what happens when the supervisor dies without stopping anything
        child.job.close()
        assert wait_until(lambda: not child.alive(), 15) and wait_until(lambda: not pipeline.port_open("127.0.0.1", port, 0.2), 15)
        child.job = None
    finally:
        child.kill_tree(grace=2.0)
        child.join_output(2.0)


def test_e2e_a_stop_that_leaves_a_server_port_answering_says_so(harness):
    """A stop must not be believed to have worked when it has not. Here one of the two server ports
    keeps answering after the servers were stopped (the test holds it, in the place of a server
    process that outlived the stop): the supervisor records it, --stop reports it and fails, and
    status.py shows it until a later start finds the ports free."""
    import socket
    one_server = "import socket, sys, time; s = socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen(8); time.sleep(3600)"
    h = harness("e2e_leftover",
                config={"servers": {"command": ["{python}", "-c", one_server, str(BASE_PORT)]},
                        "phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 10 ** 9}}},
                scenario={"tick_seconds": 0.1, "steps_per_tick": 100})
    h.start()
    h.wait_for(lambda s: (s.get("servers") or {}).get("running"), "the server launcher to be started", timeout=60)
    survivor = socket.socket()
    survivor.bind(("127.0.0.1", BASE_PORT + 1))
    survivor.listen(8)

    def answer():                                                           # like a server: take every connection
        while True:
            try:
                conn, _addr = survivor.accept()
                conn.close()
            except OSError:
                return
    threading.Thread(target=answer, daemon=True).start()
    try:
        h.wait_for(lambda s: s["status"] == "running" and s["trainer"]["progress_seen"], "the trainer to report decisions", timeout=120)
        r = h.call("--stop", "--wait", "90")
        assert r.returncode == 1, r.stdout + r.stderr
        assert "the supervisor has stopped" in r.stdout and "BUT servers could not be stopped completely" in r.stdout and str(BASE_PORT + 1) in r.stdout
        assert h.wait_exit(30) == 0
        st = h.state()
        assert st["status"] == "stopped" and st["leftover"]["servers"]["ports"] == [BASE_PORT + 1] and st["leftover"]["servers"]["pids"] == []
        left = h.events("leftover")
        assert len(left) == 1 and left[0]["what"] == "servers" and left[0]["ports"] == [BASE_PORT + 1]
        assert not pipeline.port_open("127.0.0.1", BASE_PORT, 0.3)              # the server it could stop is stopped
        info = status.gather(h.run, with_gpu=False)
        assert info["health"] == ["STOPPED"] and any("servers could not be stopped completely" in n and str(BASE_PORT + 1) in n for n in info["health_notes"])
    finally:
        survivor.close()
    # the next start finds the ports free: the note goes, and a clean stop is reported as one
    assert wait_until(lambda: not pipeline.port_open("127.0.0.1", BASE_PORT + 1, 0.2), 10)
    h.cfg["servers"]["command"] = ["{python}", "tests/fake_servers.py", "--dir", "unused", "--ports", "{ports}"]
    with open(h.cfg_path, "w") as f:
        json.dump(h.cfg, f, indent=1)
    h.start()
    st = h.wait_for(lambda s: s["status"] == "running" and s["trainer"]["progress_seen"] and s["segment"] >= 2, "the second start to run", timeout=120)
    assert st["leftover"] is None
    r = h.call("--stop", "--wait", "90")
    assert r.returncode == 0 and "BUT" not in r.stdout and h.wait_exit(30) == 0
    assert h.state()["leftover"] is None and h.ports_closed()


def test_e2e_nan_loss_restarts_from_the_last_good_checkpoint_and_gives_up_the_second_time(harness):
    ppo_only = {"phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 20000}}}
    # once: updates 6, 7 and 8 report nan; the restart is healthy
    h = harness("e2e_nan_once", config=ppo_only, faults=[{"id": "nan", "kind": "nan", "mode": "ppo", "at_steps": 6000}])
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=180)
    fail = h.events("failure")
    assert len(fail) == 1 and fail[0]["what"] == "nan_loss" and "updates 6 to 8" in fail[0]["reason"]
    bad = h.events("trainer_exit")[0]
    assert bad["graceful"] is False and bad["killed"] is True and bad["final_checkpoint"] is None    # terminated: no save of NaN weights
    set_aside = sorted(e["name"] for e in h.events("checkpoint_quarantined"))
    assert "learner_update_000006.pt" in set_aside                               # written by the first bad update
    assert all(h.checkpoint(e["moved_to"])["update_idx"] >= 6 for e in h.events("checkpoint_quarantined"))   # and nothing from before it
    second = h.events("trainer_start")[1]
    assert second["newest_checkpoint"] == "learner_update_000004.pt"             # the newest one from before the first bad update
    back = h.events("rollback")[0]
    assert back["to"] == 4000 and back["from"] >= 8000
    ppo = [m for m in h.metrics() if m["phase"] == "ppo"]
    assert [m["update"] for m in ppo if m["segment"] == 2][0] == 5 and ppo[-1]["loss"] is not None
    assert st["status"] == "done" and len(st["nan_failures"]) == 1
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0

    # every time: the second NaN ends the run instead of looping
    h2 = harness("e2e_nan_again", config=ppo_only, faults=[{"id": "nan", "kind": "nan", "mode": "ppo", "at_steps": 6000, "repeat": True}])
    h2.start()
    assert h2.wait_exit(120) == pipeline.EXIT_FAILED, h2.tail()
    st = h2.state()
    assert st["status"] == "failed" and "NaN again" in st["status_reason"] and len(st["nan_failures"]) == 2
    assert len(h2.events("trainer_start")) == 2 and [e["what"] for e in h2.events("failure")] == ["nan_loss", "nan_loss"]
    assert not [c for c in os.listdir(os.path.join(h2.run, "checkpoints")) if c.endswith(".pt") and int(c[15:21]) >= 6]


def test_e2e_starts_that_never_produce_a_decision_end_the_run(harness):
    h = harness("e2e_fruitless",
                config={"watchdog": {"stall_minutes": 0.05, "startup_grace_minutes": 0.05, "max_fruitless_starts": 2, "max_restarts_per_hour": 10}},
                faults=[{"id": "stuck", "kind": "hang", "at_steps": 0, "repeat": True}])
    h.start()
    assert h.wait_exit(120) == pipeline.EXIT_FAILED, h.tail()
    st = h.state()
    assert st["status"] == "failed" and "2 starts in a row ended without a single decision" in st["status_reason"]
    assert len(h.events("trainer_start")) == 2 and [e["what"] for e in h.events("failure")] == ["stall", "stall"]
    assert len(st["failures"]) == 1                                              # far below max_restarts_per_hour: this is the other limit


def test_e2e_planned_restarts_are_not_failures(harness):
    h = harness("e2e_recycle",
                config={"phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 30000}},
                        "watchdog": {"recycle_hours": 0.001}},                   # every 3.6 seconds
                scenario={"steps_per_tick": 100})
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=180)
    recycles = h.events("recycle")
    assert len(recycles) >= 2 and st["recycles_total"] == len(recycles) and "planned restart" in recycles[0]["reason"]
    assert not h.events("failure") and not h.events("rollback") and st["restarts_total"] == 0 and st["failures"] == []
    exits = h.events("trainer_exit")
    assert all(e["graceful"] and not e["killed"] and e["final_checkpoint"] for e in exits)   # every one saved on its way out
    assert len(h.events("trainer_start")) == len(recycles) + 1
    ppo = [m for m in h.metrics() if m["phase"] == "ppo"]
    assert [m["update"] for m in ppo] == list(range(1, len(ppo) + 1))            # no update lost or repeated across the restarts
    assert st["phases"]["ppo"]["decisions"] >= 30000 and st["phases"]["ppo"]["lost_decisions"] == 0
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


def test_e2e_trainer_leaving_after_its_minutes_is_a_planned_restart(harness):
    h = harness("e2e_minutes",
                config={"phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 20000}},
                        "train_defaults": {"minutes": 0.04}})                    # train.py --minutes: it leaves by itself after 2.4 s
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=180)
    recycles = h.events("recycle")
    assert recycles and all("--minutes" in e["reason"] for e in recycles)
    assert not h.events("failure") and st["restarts_total"] == 0 and st["phases"]["ppo"]["lost_decisions"] == 0
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


def test_e2e_phase_moves_on_from_the_last_periodic_checkpoint_when_the_final_save_fails(harness):
    h = harness("e2e_no_final_save",
                config={"phases": {"warmup": {"min_decisions": 2000, "max_decisions": 3000, "plateau": None}, "ppo": {"target_decisions": 3000}}},
                faults=[{"id": "nosave", "kind": "no_final_save", "mode": "imitation"}])
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=180)
    end = h.events("phase_end")[0]
    assert end["name"] == "imitation" and end["final_confirmed"] is False
    im = st["phases"]["imitation"]
    kept = h.checkpoint("phase_final", f"imitation_{im['final_checkpoint']}")
    assert kept["mode"] == "imitation" and kept["update_idx"] % 2 == 0           # a periodic one (save_every 2), not a final save
    assert 0 <= im["decisions"] - kept["total_steps"] <= 4000                    # about one save interval (2,000) behind the count
    first_warmup = [e for e in h.events("trainer_start") if e["name"] == "warmup"][0]
    assert first_warmup["newest_checkpoint"] == im["final_checkpoint"]
    assert not h.events("failure")
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


def test_internal_error_stops_the_trainer_cleanly_and_the_next_start_resumes(harness, monkeypatch):
    """A bug in the supervisor itself: exit code 1 (systemd restarts it), the trainer saved, nothing left running."""
    h = harness("e2e_internal_error", config={"phases": {"imitation": {"min_decisions": 10 ** 6, "max_decisions": 10 ** 7}}})
    monkeypatch.setenv("PSPPO_NO_DELETE", "1")
    boom = threading.Event()
    original = pipeline.Supervisor._progress_log

    def progress_log(self, now):
        if boom.is_set():
            raise RuntimeError("a bug in the supervisor")
        return original(self, now)
    monkeypatch.setattr(pipeline.Supervisor, "_progress_log", progress_log)
    sup = pipeline.Supervisor(pipeline.load_config(h.cfg_path), h.run, h.cfg_path)
    result = {}
    th = threading.Thread(target=lambda: result.update(code=sup.run()))
    th.start()
    try:
        h.wait_for(lambda s: s["status"] == "running" and s["phases"]["imitation"]["decisions"] >= 3000, "the trainer to run", may_exit=True)
        boom.set()
        th.join(60)
        assert not th.is_alive()
    finally:
        sup.request_stop("test cleanup")
        th.join(60)
    assert result["code"] == pipeline.EXIT_ERROR
    st = h.state()
    assert st["status"] == "error" and "a bug in the supervisor" in st["status_reason"] and st["phase"] == "imitation"
    ex = h.events("trainer_exit")[-1]
    assert ex["graceful"] is True and ex["killed"] is False and ex["final_checkpoint"]
    assert h.events("supervisor_error") and h.ports_closed() and not pid_alive(st["trainer"]["pid"])
    assert status.health_flags(st, time.time(), 100.0)[0] == ["DEAD"]
    saved = st["phases"]["imitation"]["decisions"]
    assert h.call("--can-start").returncode == 0
    # the restart (as systemd would do it): same phase, the counter continues
    monkeypatch.undo()
    h.start()
    h.wait_for(lambda s: s["status"] == "running" and s["phases"]["imitation"]["decisions"] > saved, "the run to continue past where it was")
    assert h.events("phase_resume")[-1]["decisions"] == saved and not h.events("rollback")
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


def test_state_is_rebuilt_from_events_when_state_json_is_damaged():
    d = new_dir("state_rebuild")
    cfg = unit_config()
    sup = pipeline.Supervisor(cfg, os.path.join(d, "run"), "")
    os.makedirs(sup.paths.logs)
    w = pipeline.JsonlWriter(sup.paths.events)
    for ev in [{"event": "phase_start", "name": "imitation"},
               {"event": "phase_end", "name": "imitation", "decisions": 2150000, "updates": 58, "reason": "plateau", "time": "2026-10-06T01:00:00Z",
                "final_checkpoint": "learner_update_000058.pt", "final_copy": "phase_final/imitation_learner_update_000058.pt"},
               {"event": "phase_start", "name": "warmup"}, {"event": "trainer_start", "name": "warmup"}]:
        w.write(ev)
    w.close()
    with open(sup.paths.state, "w") as f:
        f.write('{"phase": "warmup", "status": "runn')                         # torn
    sup._load_state()
    st = sup.state
    assert st["phase"] == "warmup" and st["status"] == "stopped" and st["phases"]["warmup"]["status"] == "running"
    im = st["phases"]["imitation"]
    assert (im["status"], im["decisions"], im["updates"], im["final_checkpoint"]) == ("done", 2150000, 58, "learner_update_000058.pt")
    assert not os.path.exists(sup.paths.state)                                # the damaged file was moved aside, not deleted
    moved = [f for f in os.listdir(sup.paths.logs) if f.startswith("state.json.")]
    assert len(moved) == 1 and open(os.path.join(sup.paths.logs, moved[0])).read().startswith('{"phase": "warmup", "status": "runn')
    assert sup.protected_checkpoints() == {"learner_update_000058.pt"}
    # no events either: a new run
    sup2 = pipeline.Supervisor(cfg, os.path.join(d, "run2"), "")
    os.makedirs(sup2.paths.logs)
    with open(sup2.paths.state, "w") as f:
        f.write("")
    sup2._load_state()
    assert sup2.state["phase"] == "imitation" and sup2.state["phases"]["imitation"]["status"] == "pending"


def test_e2e_only_one_supervisor_per_run_directory(harness):
    h = harness("e2e_lock", config={"phases": {"imitation": {"min_decisions": 10 ** 6, "max_decisions": 10 ** 7}}})
    r = subprocess.run([sys.executable, PIPELINE, "--run-dir", h.dir, "--stop"], capture_output=True, text=True)
    assert r.returncode == 0 and "no supervisor is running" in r.stdout
    os.makedirs(h.run)
    with open(os.path.join(h.run, "pipeline.stop"), "w") as f:                   # a stop request left over from before this start
        f.write("stale\n")
    first = h.start()
    h.wait_for(lambda s: s["status"] == "running", "the first supervisor to run")
    assert not os.path.exists(os.path.join(h.run, "pipeline.stop")) and first.poll() is None    # moved aside, not obeyed
    second = h.call(timeout=60)
    assert second.returncode == pipeline.EXIT_LOCKED and "already holds" in second.stdout
    assert first.poll() is None and len(h.events("trainer_start")) == 1           # the first one was not disturbed
    h.proc = first
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0
    assert h.state()["status"] == "stopped" and "stop file" in h.state()["status_reason"]


def test_e2e_occupied_port_is_never_taken_over(harness):
    """Something else already listens on the first port (as the owner's servers do on 8000-8003):
    the supervisor must not start a trainer against it."""
    h = harness("e2e_port_busy", config={"servers": {"port_free_timeout_seconds": 1.5}, "watchdog": {"max_restarts_per_hour": 1, "backoff_seconds": [0.2]}})
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if not WIN:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)                  # the previous test's servers closed a moment ago
    s.bind(("127.0.0.1", BASE_PORT))
    s.listen(8)

    def accept_forever():
        while True:
            try:
                conn, _ = s.accept()
                conn.close()
            except OSError:
                return
    threading.Thread(target=accept_forever, daemon=True).start()
    try:
        h.start()
        assert h.wait_exit(90) == pipeline.EXIT_FAILED, h.tail()
        st = h.state()
        assert st["status"] == "failed" and f"port {BASE_PORT} on 127.0.0.1 is already in use" in st["status_reason"]
        assert not h.events("trainer_start")
        assert {e["what"] for e in h.events("failure")} == {"start_failed"}
        assert pipeline.port_open("127.0.0.1", BASE_PORT)                         # and whoever was there is still there
    finally:
        s.close()


# ============================================================================================
# review fixes, 2026-10-05
#
# Each test below was written for one finding of the review and fails on the code as it was.
# The command a test gives as "shutdown_command" only writes to a file: nothing here can shut
# a machine down.
# ============================================================================================

def harmless_shutdown(name):
    """(marker file, a shutdown_command that appends one line to it and returns 0)."""
    marker = os.path.join(new_dir(name), "shutdown_was_called.txt")
    return marker, [sys.executable, "-c", "import sys; open(sys.argv[1], 'a').write('shutdown\\n')", marker]


# -- a run that fails must not leave the machine up (on_failed) --------------------------------

def test_on_failed_is_a_checked_setting_that_reaches_the_dry_run_and_the_limits(capsys):
    base = {"servers": {"command": ["x"]}, "phases": {"imitation": {"max_decisions": 1}, "warmup": {"max_decisions": 1}}}
    cfg = pipeline.validate_config(dict(base))
    assert (cfg["on_failed"], cfg["on_failed_grace_minutes"], cfg["shutdown_retry_minutes"]) == ("idle", 30, 10)
    with pytest.raises(pipeline.ConfigError, match="on_failed must be"):
        pipeline.validate_config(dict(base, on_failed="poweroff"))
    with pytest.raises(pipeline.ConfigError, match="on_failed is \"shutdown\" but shutdown_command is empty"):
        pipeline.validate_config(dict(base, on_failed="shutdown", shutdown_command=[]))
    with pytest.raises(pipeline.ConfigError, match="on_failed_grace_minutes"):
        pipeline.validate_config(dict(base, on_failed_grace_minutes=-1))
    with pytest.raises(pipeline.ConfigError, match="shutdown_retry_minutes"):
        pipeline.validate_config(dict(base, shutdown_retry_minutes=0))
    d = new_dir("on_failed_cfg")
    cfg = pipeline.validate_config(dict(base, on_failed="shutdown", on_failed_grace_minutes=20, shutdown_command=["true"]))
    assert pipeline.Supervisor(cfg, os.path.join(d, "run"), "")._limits()["on_failed"] == "shutdown"
    assert pipeline.dry_run(cfg, os.path.join(d, "run")) == 0
    out = capsys.readouterr().out
    assert "== when the run is marked failed, or stopped because the disk is full" in out and "waits 20 minutes" in out
    pipeline.dry_run(pipeline.validate_config(dict(base)), os.path.join(d, "run"))
    assert "leaves the machine as it is" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(d, "run"))


def test_e2e_a_run_that_is_marked_failed_shuts_the_machine_down_when_on_failed_says_so(harness):
    """The run fails at night: pipeline.py exits 3, systemd is told not to start it again, and at
    every later boot the start is skipped. Nothing was left to stop the machine costing money."""
    marker, harmless = harmless_shutdown("on_failed")
    h = harness("e2e_on_failed",
                config={"watchdog": {"max_restarts_per_hour": 1, "backoff_seconds": [0.2]},
                        "on_failed": "shutdown", "on_failed_grace_minutes": 0.03, "shutdown_command": harmless},
                faults=[{"id": "always", "kind": "die_at_start", "repeat": True}])
    h.start()
    assert h.wait_exit(120) == pipeline.EXIT_FAILED, h.tail()
    st = h.state()
    assert st["status"] == "failed" and st["shutdown_issued"] and st["shutdown_planned"] is None and st["shutdown_failed"] is None
    assert open(marker).read() == "shutdown\n"
    names = [e["event"] for e in h.events()]
    assert names.index("failed") < names.index("shutdown_planned") < names.index("shutdown")
    planned, issued = h.events("shutdown_planned")[0], h.events("shutdown")[0]
    assert planned["why"] == "failed" and planned["command"] == harmless and issued["why"] == "failed" and issued["attempt"] == 1
    assert issued["ts"] - planned["ts"] >= 1.5                                # it waited its 0.03 minutes first
    assert h.ports_closed()
    info = status.gather(h.run, with_gpu=False)
    assert info["health"] == ["FAILED"] and any("the shutdown command was run at" in n for n in info["health_notes"])
    # A person starts the machine again to look at it. The unit's start is skipped, and a start
    # by hand finds the run failed and leaves: neither shuts them out a second time.
    assert h.call("--can-start").returncode == 1
    h.start()
    assert h.wait_exit(60) == pipeline.EXIT_FAILED
    assert open(marker).read() == "shutdown\n" and len(h.events("shutdown")) == 1 and len(h.events("shutdown_planned")) == 1


def test_e2e_stopping_the_supervisor_cancels_the_shutdown_after_a_failure(harness):
    marker, harmless = harmless_shutdown("on_failed_cancel")
    h = harness("e2e_on_failed_cancel",
                config={"watchdog": {"max_restarts_per_hour": 0}, "on_failed": "shutdown", "on_failed_grace_minutes": 5,
                        "shutdown_command": harmless},
                faults=[{"id": "always", "kind": "die_at_start", "repeat": True}])
    h.start()
    st = h.wait_for(lambda s: s["status"] == "failed" and s.get("shutdown_planned"), "the shutdown to be planned", timeout=90)
    time.sleep(1.0)
    assert h.proc.poll() is None and not os.path.exists(marker)              # still there, counting down
    f, notes = status.health_flags(h.state(), time.time(), 100.0)
    assert f == ["FAILED"] and any(st["shutdown_planned"] in n and "unless the supervisor is stopped first" in n for n in notes)
    r = h.call("--stop", "--wait", "60")
    assert r.returncode == 0, r.stdout + r.stderr
    assert h.wait_exit(30) == pipeline.EXIT_FAILED
    st = h.state()
    assert st["status"] == "failed" and st["shutdown_planned"] is None and st["shutdown_issued"] is None
    assert not os.path.exists(marker) and len(h.events("shutdown_cancelled")) == 1 and not h.events("shutdown")


def test_e2e_a_full_disk_shuts_the_machine_down_too(harness):
    marker, harmless = harmless_shutdown("on_failed_disk")
    h = harness("e2e_on_failed_disk", config={"disk": {"min_free_gb": 10 ** 7}, "on_failed": "shutdown", "on_failed_grace_minutes": 0.01,
                                             "shutdown_command": harmless})
    h.start()
    assert h.wait_exit(60) == pipeline.EXIT_DISK, h.tail()
    assert open(marker).read() == "shutdown\n" and h.events("shutdown")[0]["why"] == "disk_low" and h.state()["status"] == "disk_low"
    assert h.call("--can-start").returncode == 1                              # and it stays off the next boot's start


def test_internal_errors_that_keep_coming_mark_the_run_failed():
    """systemd restarts a supervisor that hit an internal error, ten times in half an hour, and
    then gives up for good with state.json saying "error". The supervisor counts them itself and
    marks the run failed one before that, so that on_failed applies."""
    d = new_dir("supervisor_errors")
    cfg = unit_config()
    cfg["watchdog"]["max_restarts_per_hour"] = 2
    sup = pipeline.Supervisor(cfg, os.path.join(d, "run"), "")
    os.makedirs(sup.paths.run)
    sup.state = pipeline.new_state(cfg, sup.paths, "")
    assert sup._note_supervisor_error("KeyError('x')") is False and sup._note_supervisor_error("KeyError('x')") is False
    assert sup.state["status"] != "failed" and len(sup.state["supervisor_errors"]) == 2
    assert sup._note_supervisor_error("KeyError('x')") is True
    assert sup.state["status"] == "failed" and "the supervisor itself failed 3 times within an hour" in sup.state["status_reason"]
    assert "KeyError('x')" in sup.state["status_reason"]
    # errors of more than an hour ago are forgotten
    sup = pipeline.Supervisor(cfg, os.path.join(d, "run2"), "")
    os.makedirs(sup.paths.run)
    sup.state = pipeline.new_state(cfg, sup.paths, "")
    sup.state["supervisor_errors"] = [time.time() - 4000.0] * 5
    assert sup._note_supervisor_error("x") is False and len(sup.state["supervisor_errors"]) == 1


def test_an_internal_error_too_many_ends_in_failed_and_then_in_the_shutdown(harness, monkeypatch):
    marker, harmless = harmless_shutdown("internal_error_limit")
    h = harness("e2e_internal_error_limit",
                config={"phases": {"imitation": {"min_decisions": 10 ** 6, "max_decisions": 10 ** 7}},
                        "trainer": {"failure_stop_timeout_seconds": 30},
                        "watchdog": {"max_restarts_per_hour": 0}, "on_failed": "shutdown", "on_failed_grace_minutes": 0.01,
                        "shutdown_command": harmless})
    monkeypatch.setenv("PSPPO_NO_DELETE", "1")
    boom = threading.Event()
    original = pipeline.Supervisor._progress_log

    def progress_log(self, now):
        if boom.is_set():
            raise RuntimeError("a bug in the supervisor")
        return original(self, now)
    monkeypatch.setattr(pipeline.Supervisor, "_progress_log", progress_log)
    sup = pipeline.Supervisor(pipeline.load_config(h.cfg_path), h.run, h.cfg_path)
    result = {}
    th = threading.Thread(target=lambda: result.update(code=sup.run()))
    th.start()
    try:
        h.wait_for(lambda s: s["status"] == "running" and s["phases"]["imitation"]["decisions"] >= 3000, "the trainer to run", may_exit=True)
        boom.set()
        th.join(90)
        assert not th.is_alive()
    finally:
        sup.request_stop("test cleanup")
        th.join(60)
    assert result["code"] == pipeline.EXIT_FAILED
    st = h.state()
    assert st["status"] == "failed" and "the supervisor itself failed 1 times" in st["status_reason"] and "a bug in the supervisor" in st["status_reason"]
    ex = h.events("trainer_exit")[-1]
    assert ex["graceful"] is True and ex["killed"] is False and ex["final_checkpoint"]      # the trainer still saved on its way out
    assert open(marker).read() == "shutdown\n" and h.events("shutdown")[0]["why"] == "failed" and h.ports_closed()
    assert h.call("--can-start").returncode == 1


# -- the shutdown command: only a success counts, and a failure is not "DONE" ------------------

def test_e2e_a_shutdown_command_that_fails_is_tried_again_and_is_not_called_done(harness):
    d = new_dir("shutdown_retry")
    tries, allow = os.path.join(d, "tries.txt"), os.path.join(d, "allow")
    flaky = [sys.executable, "-c",
             "import os, sys; open(sys.argv[1], 'a').write('try\\n'); sys.exit(0 if os.path.exists(sys.argv[2]) else 1)", tries, allow]
    h = harness("e2e_shutdown_retry",
                config={"phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 3000}},
                        "on_done": "shutdown", "on_done_wait_minutes": 0.02, "on_done_grace_minutes": 0, "shutdown_retry_minutes": 0.02,
                        "shutdown_command": flaky})
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done" and (s.get("shutdown_failed") or {}).get("attempts", 0) >= 2, "two failed attempts", timeout=120)
    assert st["shutdown_issued"] is None and "exit code 1" in st["shutdown_failed"]["error"] and st["shutdown_failed"]["why"] == "done"
    assert h.proc.poll() is None                                              # still there, still trying
    assert len(h.events("shutdown_failed")) >= 2 and not h.events("shutdown")
    r = subprocess.run([sys.executable, os.path.join(REPO, "status.py"), "--run-dir", h.run, "--no-gpu", "--exit-code"], capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout.splitlines()[0] == "HEALTH: DONE SHUTDOWN_FAILED", r.stdout
    assert "still costs money" in r.stdout
    assert h.call("--can-start").returncode == 0                              # a restarted supervisor would go on trying
    with open(allow, "w") as f:
        f.write("now it works\n")
    st = h.wait_for(lambda s: s.get("shutdown_issued"), "the shutdown to go through", timeout=60)
    issued = h.events("shutdown")
    assert st["shutdown_failed"] is None and len(issued) == 1 and issued[0]["attempt"] >= 3 and issued[0]["why"] == "done"
    assert open(tries).read().count("try") == issued[0]["attempt"]
    assert status.gather(h.run, with_gpu=False)["health"] == ["DONE"]
    assert h.call("--can-start").returncode == 1
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


def test_status_done_is_not_healthy_while_the_machine_that_should_be_off_is_on(capsys):
    now = 1_000_000.0

    def done_state(**over):
        st = running_state(now)
        st.update(phase="done", status="done", trainer=None, updated_ts=now - 2, shutdown_issued=None, shutdown_failed=None, last_events=[])
        st["limits"]["on_done"] = "shutdown"
        st.update(over)
        return st

    assert flags(done_state(), now) == ["DONE"]                              # waiting for the last evaluation; the supervisor is there
    assert flags(done_state(updated_ts=now - 5 * 3600), now) == ["DONE", "DEAD"]   # ... and here it is not: nothing will shut it down
    assert flags(done_state(updated_ts=now - 5 * 3600, shutdown_issued="2026-10-05T12:00:00Z"), now) == ["DONE"]   # shut down, booted again later
    idle = done_state(updated_ts=now - 5 * 3600)
    idle["limits"]["on_done"] = "idle"
    assert flags(idle, now) == ["DONE"]                                      # nobody asked for a shutdown
    failed = done_state(shutdown_failed={"time": "2026-10-05T12:00:00Z", "attempts": 3, "error": "exit code 1: sudo: a password is required", "why": "done"})
    f, notes = status.health_flags(failed, now, 100.0)
    assert f == ["DONE", "SHUTDOWN_FAILED"]
    assert any("3 attempts" in n and "a password is required" in n and "still costs money" in n for n in notes)
    # a state written before the supervisor kept that field: the last word in the events decides
    legacy = done_state(shutdown_issued="2026-10-05T12:00:00Z", last_events=[{"event": "shutdown"}, {"event": "shutdown_failed", "code": 1}])
    del legacy["shutdown_failed"]
    assert flags(legacy, now) == ["DONE", "SHUTDOWN_FAILED"]
    assert flags(done_state(shutdown_issued="x", last_events=[{"event": "shutdown_failed"}, {"event": "shutdown"}]), now) == ["DONE"]
    # a failed run that is about to be shut down says when
    st = running_state(now)
    st.update(status="failed", status_reason="7 failures within an hour", shutdown_planned="2026-10-05T12:30:00Z")
    f, notes = status.health_flags(st, now, 100.0)
    assert f == ["FAILED"] and any("2026-10-05T12:30:00Z" in n and "on_failed" in n for n in notes)
    # and a program that polls status.py --exit-code is told
    run = os.path.join(new_dir("status_done"), "run")
    os.makedirs(run)
    real_now = time.time()
    st = done_state(updated_ts=real_now - 5 * 3600)
    st["limits"]["min_free_gb"] = 0                                          # this is not about the disk of the machine the test runs on
    with open(os.path.join(run, "state.json"), "w") as f:
        json.dump(st, f)
    assert status.main(["--run-dir", run, "--no-gpu", "--exit-code"]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "HEALTH: DONE DEAD" and "nothing will shut the box down" in out
    st["shutdown_issued"] = "2026-10-05T12:00:00Z"
    with open(os.path.join(run, "state.json"), "w") as f:
        json.dump(st, f)
    assert status.main(["--run-dir", run, "--no-gpu", "--exit-code"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "HEALTH: DONE"


# -- a trainer whose weights are bad is never asked to save -------------------------------------

def test_a_hard_kill_sends_no_sigterm(monkeypatch):
    """On Linux Ray turns a SIGTERM to train.py into a SystemExit, and train.py's "finally" then
    saves a final checkpoint: after a failed resume, of untrained weights, as the newest file.
    The POSIX branch is run here on any OS, with the system calls recorded instead of made."""
    sent = []

    class Proc:
        pid = 4242

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def killpg(pid, sig):
        sent.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError()                                       # "is anything left in the group?": no

    child = object.__new__(pipeline.Child)
    child.name, child.pid, child.proc, child.job = "trainer", 4242, Proc(), None
    monkeypatch.setattr(pipeline, "IS_WINDOWS", False)
    monkeypatch.setattr(pipeline.signal, "SIGTERM", 15, raising=False)
    monkeypatch.setattr(pipeline.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(pipeline.os, "killpg", killpg, raising=False)
    assert child.kill_tree(grace=30.0, hard=True) == []
    assert [s for s in sent if s[1] != 0] == [(4242, 9)]                     # SIGKILL to the group, and nothing before it
    sent.clear()
    assert child.kill_tree(grace=30.0) == []
    assert [s for s in sent if s[1] != 0] == [(4242, 15), (4242, 9)]         # the ordinary stop: a chance to leave first


def test_a_hard_kill_gives_a_process_no_chance_to_save_on_posix():
    if WIN:
        pytest.skip("POSIX signals. On Windows the kill is TerminateJobObject, which no process can answer")
    saver = ("import signal, sys, time\n"
             "def bye(*_a):\n"
             "    open(sys.argv[1], 'w').write('saved on SIGTERM')\n"
             "    sys.exit(0)\n"
             "signal.signal(signal.SIGTERM, bye)\n"
             "print('ready', flush=True)\n"
             "time.sleep(600)\n")
    for hard, saved in ((True, False), (False, True)):
        d = new_dir("hard_kill")
        marker = os.path.join(d, "saved.txt")
        child = pipeline.Child("saver", [sys.executable, "-c", saver, marker], d, dict(os.environ, PYTHONUNBUFFERED="1"), os.path.join(d, "out.log"))
        seen = []
        try:
            assert wait_until(lambda: seen.extend(child.lines()) or any("ready" in line for line in seen), 30)
            assert child.kill_tree(grace=10.0, hard=hard) == [] and not child.alive()
            assert os.path.exists(marker) is saved, f"hard={hard}"
        finally:
            child.kill_tree(grace=2.0, hard=True)
            child.join_output(2.0)


class _RecordingChild:
    """A trainer process that only records how it was stopped."""

    def __init__(self, stop_file):
        self.pid = 99999
        self.kills = []
        self.stop_file_seen = False
        self._stop_file = stop_file
        self._alive = True

    def alive(self):
        self.stop_file_seen = self.stop_file_seen or os.path.exists(self._stop_file)
        return self._alive

    def lines(self):
        return []

    def returncode(self):
        return None if self._alive else -9

    def kill_tree(self, grace=10.0, hard=False):
        self.kills.append(hard)
        self._alive = False
        return []

    def join_output(self, timeout=5.0):
        pass


def supervisor_with_a_trainer(name, lines, **watchdog):
    cfg = unit_config()
    cfg["watchdog"].update(dict({"poll_seconds": 0.05}, **watchdog))
    sup = pipeline.Supervisor(cfg, os.path.join(new_dir(name), "run"), "")
    for path in (sup.paths.logs, sup.paths.checkpoints):
        os.makedirs(path)
    sup.state = pipeline.new_state(cfg, sup.paths, "")
    sup.tracker = pipeline.Tracker(cfg, sup.state, lambda rec: None, lambda event, detail: None)
    sup.tracker.begin_phase("ppo", 0.0)
    sup.tracker.begin_trainer(99999, time.time(), "logs/x.log", True)
    feed(sup.tracker, lines, time.time())
    sup.trainer = _RecordingChild(sup.paths.trainer_stop)
    return sup, sup.trainer


def test_a_trainer_whose_weights_may_be_bad_is_never_asked_to_save_whatever_the_stop_is_for():
    """A planned restart, a spot stop, a server that went down or a full disk all stop the trainer
    through the stop file, and train.py then saves. Inside the grace of watchdog.nan_updates_limit
    that saved the NaN weights as the newest checkpoint."""
    def nan(u):
        return f"(LearnerActor pid=5) [learner] upd={u} kl=nan clip=0.0 ent=nan vloss=nan ploss=nan loss=nan n_mb=128"

    def exit_event(sup):
        return [e for e in sup.state["last_events"] if e["event"] == "trainer_exit"][-1]

    # a healthy trainer: the stop file, patience, and then a kill that still lets it save
    sup, child = supervisor_with_a_trainer("stop_healthy", [tele(1000, 27), upd(28)])
    assert sup._suspect_weights() is None
    sup._stop_trainer(True, 0.3, "a planned restart")
    assert child.stop_file_seen is True and child.kills == [False]
    assert exit_event(sup)["graceful"] is True and exit_event(sup)["no_save_because"] is None

    # one update with a NaN loss, two short of what is called a failure: the same stop must not save
    sup, child = supervisor_with_a_trainer("stop_nan", [tele(1000, 27), nan(28)])
    assert sup.tracker.health(time.time()) is None
    assert "not a number in its last 1 update(s) (from update 28)" in sup._suspect_weights()
    sup._stop_trainer(True, 30.0, "a planned restart")
    assert child.stop_file_seen is False and child.kills == [True]            # no stop file, no patience, no SIGTERM
    assert not os.path.exists(sup.paths.trainer_stop) and not os.path.isdir(sup.paths.stop_files)
    assert exit_event(sup)["graceful"] is False and "not a number" in exit_event(sup)["no_save_because"]

    # the loss came back: it is asked to save again
    sup, child = supervisor_with_a_trainer("stop_recovered", [tele(1000, 27), nan(28), upd(29)])
    assert sup._suspect_weights() is None

    # nan_updates_limit 0 switches the NaN handling off, this part of it too
    sup, child = supervisor_with_a_trainer("stop_nan_off", [tele(1000, 27), nan(28)], nan_updates_limit=0)
    assert sup._suspect_weights() is None

    # a trainer that could not load the newest checkpoint holds untrained weights
    sup, child = supervisor_with_a_trainer("stop_resume_failed", [
        "(LearnerActor pid=7) [learner] resume failed from /r/checkpoints/learner_update_000125.pt: EOFError()",
        "(LearnerActor pid=7) [learner] pushed init policy 0 to inference"])
    assert "could not load learner_update_000125.pt" in sup._suspect_weights()
    sup._stop_trainer(True, 30.0, "signal SIGTERM")
    assert child.stop_file_seen is False and child.kills == [True]
    # ... and so does the last-resort kill when the supervisor itself is leaving
    sup, child = supervisor_with_a_trainer("leftover_nan", [tele(1000, 27), nan(28)])
    sup._kill_children()
    assert child.kills == [True]
    sup, child = supervisor_with_a_trainer("leftover_healthy", [tele(1000, 27), upd(28)])
    sup._kill_children()
    assert child.kills == [False]


def test_e2e_a_phase_does_not_end_on_weights_that_may_be_ruined(harness):
    """The loss is NaN from update 6 and ppo reaches its target at update 8, before the sixth NaN
    in a row that is called a failure. Ending the phase there kept NaN weights as its result."""
    h = harness("e2e_nan_at_phase_end",
                config={"phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 8000}},
                        "watchdog": {"nan_updates_limit": 6}},
                faults=[{"id": "nan", "kind": "nan", "mode": "ppo", "at_steps": 6000}],
                scenario={"steps_per_tick": 100})
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=180)
    fail = h.events("failure")
    assert [e["what"] for e in fail] == ["nan_loss"] and "updates 6 to 11" in fail[0]["reason"], fail
    ends = h.events("phase_end")
    assert len(ends) == 1 and ends[0]["ts"] > fail[0]["ts"]                   # the phase ended after the restart, not on the NaN weights
    assert st["phases"]["ppo"]["trainer_starts"] == 2
    aside = {e["name"] for e in h.events("checkpoint_quarantined")}
    assert {"learner_update_000006.pt", "learner_update_000008.pt", "learner_update_000010.pt"} <= aside
    assert all(int(name[15:21]) >= 6 for name in aside)                       # nothing from before the first NaN
    assert h.events("trainer_start")[1]["newest_checkpoint"] == "learner_update_000004.pt"
    kept = h.checkpoint("phase_final", f"ppo_{st['phases']['ppo']['final_checkpoint']}")
    assert kept["total_steps"] >= 8000 and kept["saved"] > fail[0]["ts"]
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


def nan_supervisor(name, files, **trainer):
    cfg = unit_config()
    sup = pipeline.Supervisor(cfg, os.path.join(new_dir(name), "run"), "")
    os.makedirs(sup.paths.checkpoints)
    sup.state = pipeline.new_state(cfg, sup.paths, "")
    now = time.time()
    for update, age in files:
        path = os.path.join(sup.paths.checkpoints, f"learner_update_{update:06d}.pt")
        with open(path, "w") as f:
            f.write("x")
        os.utime(path, (now - age, now - age))
    sup.state["trainer"] = dict({"started_ts": now - 600.0}, **trainer)
    return sup


def checkpoint_updates(sup):
    return [c.update for c in pipeline.list_checkpoints(sup.paths.checkpoints)]


def test_nan_from_a_trainers_first_update_sets_the_checkpoint_it_resumed_aside_too():
    """The NaN weights were saved by the trainer before (inside its grace), so the files this
    trainer wrote are not the whole damage: the restart loaded the same file again, the NaN came
    back, and "a second NaN within six hours" ended the run with good checkpoints still on disk."""
    out = pipeline.Outcome("failure", "nan_loss", "nan", False)
    files = [(100, 7200), (125, 3600), (150, 1200), (152, 60)]               # (update, written this many seconds ago)
    resumed = {"resumed_from": "learner_update_000150.pt", "nan_in_a_row": 3}
    # it resumed 150 (written before it started 10 minutes ago), and its updates 151, 152 and 153 were all NaN
    sup = nan_supervisor("nan_resumed", files, first_nan_update=151, updates_seen=3, **resumed)
    assert sup._after_nan_loss(out) is True
    assert checkpoint_updates(sup) == [100, 125]                             # the restart resumes 125
    why = {e["name"]: e["why"] for e in sup.state["last_events"] if e["event"] == "checkpoint_quarantined"}
    assert "from its first update" in why["learner_update_000150.pt"] and "after the loss became NaN" in why["learner_update_000152.pt"]
    assert all(q["why"] == "nan_loss" and q["ts"] for q in sup.state["quarantined"]) and len(sup.state["quarantined"]) == 2
    assert len(os.listdir(sup.paths.quarantine)) == 2                        # moved, not removed
    # two good updates before the first NaN: the weights it loaded were sound
    sup = nan_supervisor("nan_later", files + [(154, 30)], first_nan_update=153, updates_seen=5, **resumed)
    assert sup._after_nan_loss(out) is True
    assert checkpoint_updates(sup) == [100, 125, 150, 152]
    # the only checkpoint there is stays: a restart never begins from nothing
    sup = nan_supervisor("nan_only", [(150, 1200)], first_nan_update=151, updates_seen=3, **resumed)
    assert sup._after_nan_loss(out) is True
    assert checkpoint_updates(sup) == [150] and sup.state["quarantined"] == []


def test_the_limit_on_unreadable_checkpoints_counts_neither_nan_set_asides_nor_old_ones():
    cfg = unit_config()
    assert cfg["watchdog"]["max_quarantined_checkpoints"] == 2
    sup = pipeline.Supervisor(cfg, os.path.join(new_dir("quarantine_budget"), "run"), "")
    os.makedirs(sup.paths.checkpoints)
    sup.state = pipeline.new_state(cfg, sup.paths, "")
    now = time.time()
    sup.state["quarantined"] = [{"name": "learner_update_000120.pt", "why": "nan_loss", "ts": now - 60},
                                {"name": "learner_update_000125.pt", "why": "nan_loss", "ts": now - 60},
                                {"name": "learner_update_000010.pt", "ts": now - 30 * 3600}]

    def torn(update):
        name = f"learner_update_{update:06d}.pt"
        with open(os.path.join(sup.paths.checkpoints, name), "w") as f:
            f.write("torn")
        return name

    # one NaN restart (two files set aside) and a torn file of last week used to be "the limit is reached"
    assert sup._quarantine(torn(175)) is True and checkpoint_updates(sup) == []
    assert sup._quarantine(torn(176)) is True and len(os.listdir(sup.paths.quarantine)) == 2
    assert sup._quarantine(torn(177)) is False and checkpoint_updates(sup) == [177]      # the third in a day: something else is wrong
    # an entry from before entries carried a time cannot be told to be old: it counts
    sup.state["quarantined"] = [{"name": "a", "time": "2026-10-01T00:00:00Z"}, {"name": "b", "time": "2026-10-01T00:00:00Z"}]
    assert sup._quarantine("learner_update_000177.pt") is False


# -- the phase must come from the run's history, and never go backwards -------------------------

def test_a_missing_state_json_is_rebuilt_from_events_too():
    """Only an unreadable state.json was rebuilt from events.jsonl. With the file missing the run
    began again at imitation, on top of whatever the newest checkpoint was."""
    cfg = unit_config()
    history = [{"event": "phase_start", "name": "imitation"},
               {"event": "phase_end", "name": "imitation", "decisions": 4000100, "updates": 108, "final_checkpoint": "learner_update_000108.pt"},
               {"event": "phase_start", "name": "warmup"},
               {"event": "phase_end", "name": "warmup", "decisions": 600000, "updates": 16, "final_checkpoint": "learner_update_000016.pt"},
               {"event": "phase_start", "name": "ppo"}, {"event": "trainer_start", "name": "ppo"}]

    def loaded(name, events):
        sup = pipeline.Supervisor(cfg, os.path.join(new_dir(name), "run"), "")
        os.makedirs(sup.paths.logs)
        os.makedirs(sup.paths.checkpoints)
        with open(os.path.join(sup.paths.checkpoints, "learner_update_002500.pt"), "w") as f:
            f.write("x")
        if events:
            w = pipeline.JsonlWriter(sup.paths.events)
            for ev in events:
                w.write(ev)
            w.close()
        assert not os.path.exists(sup.paths.state)
        sup._load_state()
        return sup

    st = loaded("state_missing", history).state
    assert st["phase"] == "ppo" and st["status"] == "stopped"
    assert (st["phases"]["imitation"]["status"], st["phases"]["warmup"]["status"], st["phases"]["ppo"]["status"]) == ("done", "done", "running")
    assert st["phases"]["imitation"]["decisions"] == 4000100 and st["phases"]["warmup"]["final_checkpoint"] == "learner_update_000016.pt"
    # a run that had finished and shut the machine down does not wait and shut it down again
    done = history + [{"event": "phase_end", "name": "ppo", "decisions": 250000000, "updates": 6800},
                      {"event": "done", "ts": 1791200000.0, "time": "2026-10-05T12:00:00Z"},
                      {"event": "shutdown", "time": "2026-10-05T12:10:00Z"}]
    sup = loaded("state_missing_done", done)
    st = sup.state
    assert (st["phase"], st["shutdown_issued"], st["done_ts"], st["done"]) == ("done", "2026-10-05T12:10:00Z", 1791200000.0, "2026-10-05T12:00:00Z")
    pipeline.atomic_write_json(sup.paths.state, st)
    shutdown_cfg = dict(cfg, on_done="shutdown")
    assert pipeline.can_start(shutdown_cfg, sup.paths.run) == 1
    # ... unless the last word about the command is that it failed
    st = loaded("state_missing_shutdown_failed", done + [{"event": "shutdown_failed", "code": 1}]).state
    assert st["phase"] == "done" and st["shutdown_issued"] is None
    # no history at all (a run handed over from a start by hand): the first phase, as before
    st = loaded("state_missing_no_events", []).state
    assert st["phase"] == "imitation" and st["status"] == "starting" and st["phases"]["imitation"]["status"] == "pending"


def test_a_checkpoint_from_a_later_phase_is_not_trained_over():
    change = "(LearnerActor pid=7) [learner] \U0001F6D1 PHASE CHANGE DETECTED ({} -> {})."
    tr, state, metrics, events = make_tracker()
    tr.begin_phase("imitation", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", True)
    tr.feed(change.format("ppo", "imitation"), 1.0)
    kind, reason, graceful = tr.health(2.0)
    assert kind == "phase_backwards" and graceful is False                   # no save: it would write ppo's weights as "imitation, update 0"
    assert "written in the ppo phase" in reason and "comes after imitation" in reason
    # forwards is the ordinary step from one phase to the next
    tr.begin_phase("warmup", 3.0)
    tr.begin_trainer(2, 3.0, "b.log", True)
    tr.feed(change.format("imitation", "warmup"), 4.0)
    assert tr.health(5.0) is None and state["trainer"]["phase_change"] == "imitation -> warmup"
    # a mode that no phase of this config uses says nothing about the order
    tr.begin_phase("imitation", 6.0)
    tr.begin_trainer(3, 6.0, "c.log", True)
    tr.feed(change.format("jepa_pretraining", "imitation"), 7.0)
    assert tr.health(8.0) is None
    # the phases are known by their modes, whatever those are called
    tr, state, metrics, events = make_tracker(unit_config(ppo={"mode": "ppo_with_jepa"}))
    tr.begin_phase("warmup", 0.0)
    tr.begin_trainer(1, 0.0, "a.log", True)
    tr.feed(change.format("ppo_with_jepa", "warmup"), 1.0)
    assert tr.health(2.0)[0] == "phase_backwards"


def test_e2e_checkpoints_that_are_ahead_of_the_state_are_left_alone(harness):
    """No state.json and no events.jsonl, and the newest checkpoint is a ppo one: the supervisor
    starts at imitation, the trainer says it is changing phase from ppo to imitation, and the run
    stops there, failed, with the checkpoint untouched."""
    h = harness("e2e_backwards", faults=[{"id": "ray", "kind": "sigterm_saves", "repeat": True}],
                scenario={"tick_seconds": 0.5, "steps_per_tick": 100})       # slow: no periodic save before it is noticed
    ckpts = os.path.join(h.run, "checkpoints")
    os.makedirs(ckpts)
    path = os.path.join(ckpts, "learner_update_002500.pt")
    with open(path, "w") as f:
        json.dump({"fake": True, "mode": "ppo", "update_idx": 2500, "total_steps": 92160000, "saved": 0}, f)
    before = open(path).read()
    h.start()
    assert h.wait_exit(120) == pipeline.EXIT_FAILED, h.tail()
    st = h.state()
    assert st["status"] == "failed" and st["phase"] == "imitation"
    assert "written in the ppo phase" in st["status_reason"] and "--clear-failed" in st["status_reason"]
    assert [e["what"] for e in h.events("failure")] == ["phase_backwards"] and len(h.events("trainer_start")) == 1
    ex = h.events("trainer_exit")[0]
    assert ex["graceful"] is False and ex["killed"] is True and ex["final_checkpoint"] is None
    assert sorted(os.listdir(ckpts)) == ["learner_update_002500.pt"] and open(path).read() == before
    assert h.ports_closed()


# -- smaller things ------------------------------------------------------------------------------

def test_e2e_a_server_that_falls_over_while_the_servers_start_does_not_restart_the_trainer(harness):
    """start_servers.py restarts a server that exits while the others are still coming up and says
    so. That line was read after the trainer had been started and took it for a server that went
    down under it: one failure and one wasted start, every time."""
    fell = os.path.join(new_dir("server_falls_at_start"), "fell_over_once")
    h = harness("e2e_server_falls_at_start",
                config={"servers": {"command": ["{python}", "tests/fake_servers.py", "--dir", "unused", "--ports", "{ports}",
                                                "--fail-first-start", fell], "settle_seconds": 1.0},
                        "phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 6000}}})
    h.start()
    st = h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=180)
    logs = os.path.join(h.run, "logs")
    said = "".join(read_text(os.path.join(logs, f)) for f in os.listdir(logs) if f.startswith("servers_"))
    assert os.path.exists(fell) and f"server on {BASE_PORT} exited with 3, restarting" in said
    assert "servers, while they were starting" in read_text(os.path.join(logs, "pipeline.log"))
    assert not h.events("failure") and len(h.events("trainer_start")) == 1 and st["restarts_total"] == 0
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


def test_the_output_reader_keeps_reading_when_the_log_cannot_be_written(monkeypatch):
    """One failed write to the trainer's log (a full disk) ended the reader. The pipe then filled,
    train.py blocked in its logger, and the watchdog took that for a stall, on every restart."""
    d = new_dir("pump")
    log_path = os.path.join(d, "out.log")
    real_open = open

    class FullDisk:
        def __init__(self, f):
            self.f = f

        def write(self, data):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def close(self):
            self.f.close()

    def open_but_full(path, *a, **k):
        f = real_open(path, *a, **k)
        return FullDisk(f) if os.path.abspath(str(path)) == os.path.abspath(log_path) else f

    talker = "for i in range(200):\n    print('line', i, 'x' * 500, flush=True)\nprint('the end', flush=True)\n"   # 100 kB: more than a pipe holds
    monkeypatch.setattr(pipeline, "open", open_but_full, raising=False)
    child = pipeline.Child("talker", [sys.executable, "-c", talker], d, dict(os.environ), log_path)
    try:
        assert wait_until(child.output_done, 60), "the child is blocked on a pipe nobody reads"
        lines = child.lines()
    finally:
        child.kill_tree(grace=2.0)
        child.join_output(2.0)
    told = [line for line in lines if line.startswith("[pipeline]")]
    assert len(told) == 1 and "could not write" in told[0] and "No space left" in told[0]       # said once, not 201 times
    assert len([line for line in lines if line.startswith("line ")]) == 200 and any(line.startswith("the end") for line in lines)
    assert child.returncode() == 0 and child.eof
    monkeypatch.undo()
    # a log that cannot even be opened (here: its path is a directory) costs the copy, not the reading
    child = pipeline.Child("talker", [sys.executable, "-c", talker], d, dict(os.environ), d)
    try:
        assert wait_until(child.output_done, 60)
        lines = child.lines()
    finally:
        child.kill_tree(grace=2.0)
        child.join_output(2.0)
    assert len([line for line in lines if line.startswith("line ")]) == 200 and "cannot open" in lines[0]


def test_e2e_the_state_is_refreshed_while_the_servers_take_their_time(harness):
    """Servers may take minutes to listen (ready_timeout_seconds is 300 on the box). state.json was
    not written in that wait, and after two minutes status.py called a starting run DEAD."""
    slow = ("import socket, sys, time\n"
            "time.sleep(4)\n"
            "socks = []\n"
            "for p in sys.argv[1:]:\n"
            "    s = socket.socket()\n"
            "    s.bind(('127.0.0.1', int(p)))\n"
            "    s.listen(8)\n"
            "    s.settimeout(0.05)\n"
            "    socks.append(s)\n"
            "while True:\n"
            "    for s in socks:\n"
            "        try:\n"
            "            c, _a = s.accept()\n"
            "            c.close()\n"
            "        except OSError:\n"
            "            pass\n")
    h = harness("e2e_slow_servers",
                config={"servers": {"command": ["{python}", "-c", slow, str(BASE_PORT), str(BASE_PORT + 1)]},
                        "phases": {"imitation": {"skip": True}, "warmup": {"skip": True}, "ppo": {"target_decisions": 3000}}})
    h.start()
    first = h.wait_for(lambda s: s["status"] == "starting" and (s.get("servers") or {}).get("running"), "the server launcher to be started")
    time.sleep(1.5)
    second = h.state()
    assert second["status"] == "starting" and not (second.get("trainer") or {}).get("running")   # still waiting for the ports
    assert second["updated_ts"] > first["updated_ts"] + 0.5, "state.json was not written while the supervisor waited for the servers"
    h.wait_for(lambda s: s["phase"] == "done", "the run to reach done", timeout=180)
    assert h.call("--stop", "--wait", "60").returncode == 0 and h.wait_exit(30) == 0


# -- cloud/teardown.sh: which box -----------------------------------------------------------------

def teardown_world(name):
    """A stand-in for cloud/.state with two targets, and an "aws" that is tests/fake_aws.py for both
    accounts: target a through plain `aws` on PATH (as the real a.env has it), target b through a
    wrapper script named in its file (as the real b.env has it). Returns what a test needs."""
    import shutil
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not on PATH")
    if WIN and os.path.normcase(os.path.abspath(bash)).startswith(os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows"))):
        pytest.skip("the bash on PATH is the WSL launcher, which does not pass this test's environment on")
    root = new_dir(name)
    python = (getattr(sys, "_base_executable", None) or sys.executable).replace("\\", "/")
    fake = os.path.join(TESTS, "fake_aws.py").replace("\\", "/")

    def p(*parts):
        return os.path.join(root, *parts).replace("\\", "/")

    os.makedirs(os.path.join(root, "state", "targets"))
    os.makedirs(os.path.join(root, "bin"))
    scenarios = {
        "a": {"sts get-caller-identity": "111111111111", "iam list-account-aliases": "own-account",
              "ec2 describe-instances": "i-0aaaaaaaaaaaaaaaa\tg6e.4xlarge\trunning\tus-east-1b\tsir-aaaa\t2026-10-05T07:00:00+00:00\t1.2.3.4",
              "ec2 describe-spot-instance-requests": "sir-aaaa\tactive\tfulfilled\tpersistent\ti-0aaaaaaaaaaaaaaaa",
              "ec2 cancel-spot-instance-requests": "sir-aaaa\tcancelled",
              "ec2 terminate-instances": "i-0aaaaaaaaaaaaaaaa\trunning\tshutting-down"},
        "b": {"sts get-caller-identity": "222222222222", "iam list-account-aliases": "temp-other",
              "ec2 describe-instances": "i-0bbbbbbbbbbbbbbbb\tg6e.16xlarge\trunning\tus-east-2c\tNone\t2026-10-05T17:02:00+00:00\t5.6.7.8",
              "ec2 describe-spot-instance-requests": "",
              "ec2 terminate-instances": "i-0bbbbbbbbbbbbbbbb\trunning\tshutting-down"},
    }
    for target, scenario in scenarios.items():
        with open(p(f"scenario_{target}.json"), "w") as f:
            json.dump(scenario, f)
    with open(p("bin", "aws"), "w", newline="\n") as f:
        f.write(f'#!/usr/bin/env bash\nexec "{python}" "{fake}" "{p("scenario_a.json")}" "{p("calls_a.jsonl")}" "$@"\n')
    with open(p("aws_b.sh"), "w", newline="\n") as f:
        f.write(f'#!/usr/bin/env bash\nexec "{python}" "{fake}" "{p("scenario_b.json")}" "{p("calls_b.jsonl")}" "$@"\n')
    os.chmod(p("bin", "aws"), 0o755)
    with open(p("state", "targets", "a.env"), "w", newline="\n") as f:
        f.write("# first box: own account, spot\nPSPPO_REGION=us-east-1\nPSPPO_NAME=psppo-train\nAWSCMD=aws\n")
    with open(p("state", "targets", "b.env"), "w", newline="\n") as f:
        f.write(f'# second box: another account, on demand\nPSPPO_REGION=us-east-2\nPSPPO_NAME=psppo-od\nAWSCMD="bash {p("aws_b.sh")}"\n')
    with open(p("state", "current"), "w", newline="\n") as f:
        f.write("b\n")
    env = dict(os.environ, PATH=os.path.join(root, "bin") + os.pathsep + os.environ.get("PATH", ""), PSPPO_STATE_DIR=p("state"))
    for key in ("PSPPO_TARGET", "PSPPO_REGION", "PSPPO_NAME", "AWSCMD", "AWS"):
        env.pop(key, None)
    # The real aws CLI must be out of reach before the script is run at all: `aws` has to be the stand-in.
    found = subprocess.run([bash, "-c", "command -v aws"], capture_output=True, text=True, env=env, timeout=120).stdout.strip()
    if os.path.basename(root) not in found or not found.endswith("/bin/aws"):
        pytest.fail(f"the stand-in aws is not first on PATH (found {found!r}): not running cloud/teardown.sh")

    def calls(target):
        return _read_calls(p(f"calls_{target}.jsonl"))

    def run(typed="", args=(), **extra):
        # Bytes in, so that what is "typed" ends in a plain newline as from a terminal (text mode on
        # Windows would send \r\n, and the script compares the line exactly).
        r = subprocess.run([bash, os.path.join(REPO, "cloud", "teardown.sh"), *args], input=typed.encode("ascii"), capture_output=True,
                           env=dict(env, **extra), timeout=600)
        r.stdout, r.stderr = r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")
        return r
    return run, calls, p


def _read_calls(path):
    out = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            out = [json.loads(line) for line in f if line.strip()]
    return out


def _changing(calls):
    return [c[:2] for c in calls if c[:1] == ["ec2"] and not c[1].startswith("describe")]


def test_teardown_looks_at_the_selected_target_and_names_the_others():
    """The script was fixed to us-east-1 and the default account. With a second, on-demand box in
    another account selected (cloud/.state/current), it still offered the first box, the live spot
    run, for termination, and never saw the one that costs list price."""
    assert b"\r" not in open(os.path.join(REPO, "cloud", "teardown.sh"), "rb").read()
    run, calls, p = teardown_world("teardown")

    # nothing typed: everything about box b is shown, nothing is changed, box a is not even asked about
    r = run()
    assert r.returncode == 1, r.stdout + r.stderr
    for needle in ("Target    b", "222222222222", "alias temp-other", "Region    us-east-2", "Name=psppo-od", "i-0bbbbbbbbbbbbbbbb",
                   "g6e.16xlarge in us-east-2c, ON DEMAND", "target    b, account 222222222222 (temp-other), region us-east-2",
                   "Not confirmed. Nothing was changed.", "PSPPO_TARGET=a bash cloud/teardown.sh"):
        assert needle in r.stdout, (needle, r.stdout)
    assert "i-0aaaaaaaaaaaaaaaa" not in r.stdout
    assert calls("a") == [] and _changing(calls("b")) == []
    ec2 = [c for c in calls("b") if c[0] == "ec2"]
    assert len(ec2) == 2 and all(c[c.index("--region") + 1] == "us-east-2" and "Name=tag:Name,Values=psppo-od" in c for c in ec2)

    # the id of the box that is shown, typed: terminated, through that target's own aws command and region
    r = run("i-0bbbbbbbbbbbbbbbb\n")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _changing(calls("b")) == [["ec2", "terminate-instances"]] and calls("a") == []
    last = calls("b")[-1]
    assert last[last.index("--instance-ids") + 1] == "i-0bbbbbbbbbbbbbbbb" and last[last.index("--region") + 1] == "us-east-2"
    assert "OTHER TARGETS ON FILE" in r.stdout

    # PSPPO_TARGET wins over cloud/.state/current, and the id of the other box confirms nothing
    r = run("i-0bbbbbbbbbbbbbbbb\n", PSPPO_TARGET="a")
    assert r.returncode == 1 and "Not confirmed. Nothing was changed." in r.stdout, r.stdout + r.stderr
    for needle in ("Target    a   (chosen by PSPPO_TARGET)", "111111111111", "alias own-account", "i-0aaaaaaaaaaaaaaaa", "g6e.4xlarge in us-east-1b, spot",
                   "sir-aaaa", "PSPPO_TARGET=b bash cloud/teardown.sh"):
        assert needle in r.stdout, (needle, r.stdout)
    assert _changing(calls("a")) == []
    # typed right: the spot request first, then the instance
    r = run("i-0aaaaaaaaaaaaaaaa\n", PSPPO_TARGET="a")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _changing(calls("a")) == [["ec2", "cancel-spot-instance-requests"], ["ec2", "terminate-instances"]]
    assert all(c[c.index("--region") + 1] == "us-east-1" for c in calls("a") if c[0] == "ec2")

    # a target that has no file is refused before AWS is asked anything
    before = (len(calls("a")), len(calls("b")))
    r = run("i-0aaaaaaaaaaaaaaaa\n", PSPPO_TARGET="c")
    assert r.returncode == 1 and "has no file" in r.stdout and "The targets on file: a b" in r.stdout
    assert (len(calls("a")), len(calls("b"))) == before

    # with no targets on file at all it is what it always was: the first box, through plain `aws`
    r = run(PSPPO_STATE_DIR=p("no_state_here"))
    assert r.returncode == 1 and "Target    a   (chosen by the default)" in r.stdout and "i-0aaaaaaaaaaaaaaaa" in r.stdout
    assert "OTHER TARGETS ON FILE" not in r.stdout
