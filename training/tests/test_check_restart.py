"""
cloud/check_restart.py (ported from stage3 to this code line): the check between an edit of a
run's config and the start of the supervisor. No Ray, no server, no trainer: the supervisor's own
config validation and command builder, train.py's own parser, and the numbers of a checkpoint.

    python -m pytest tests/test_check_restart.py -q -p no:cacheprovider

Held here:
  * the box's config (cloud/run_big.json) with no extra arguments is "OK to start", and the
    settings line it prints is the line train.py logs (train.ppo_settings_line);
  * a misspelt argument, a schedule that cannot be, a config the supervisor refuses, and extra
    arguments other than the expected ones are exit code 2;
  * a config whose repo_dir names another release than the check's own (whose train.py may take
    other flags) is exit code 2, and the message names that release's check;
  * a learning-rate restart from stage_lr.py moves the rate at the resume: exit code 3 unless
    --allow-rate-change, and then the rate it names is the planned one; --lr-total-steps given
    before the hold ends moves nothing, given after it (or taken away again once the anneal has
    begun) it is refused;
  * a start that switches the own Tera type on or off against the checkpoint is exit code 2
    unless --allow-obs-change;
  * the share of the rate the check computes is learner.py's own schedule function.

PSPPO_CHECK_RESTART_CKPT=<a real checkpoint> adds the check of a real file (read only).
"""
from __future__ import annotations

import ast
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "cloud")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)

import check_restart  # noqa: E402
import stage_lr  # noqa: E402

STEP, DEC = 1_104_534, 604_489_541          # a checkpoint past the end of config.py's anneal


def _restart_args(step=STEP, decisions=DEC):
    p = stage_lr.plan(step, decisions, decisions + 300_000_000, 3e-5, hold_decisions=100_000_000)
    return ["--lr", repr(p["lr"]), "--lr-hold-steps", str(p["lr_hold_steps"]), "--lr-total-steps", str(p["lr_total_steps"])]


def _config(tmp_path, extra=None, edit=None):
    with open(os.path.join(REPO, "cloud", "run_big.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["repo_dir"] = REPO              # the trainer runs this checkout's train.py, the one the check parses with
    if extra is not None:
        raw["phases"]["ppo"]["extra_train_args"] = list(extra)
    if edit:
        edit(raw)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return str(path)


def _run(tmp_path, monkeypatch, extra=None, ckpt=None, edit=None, **kw):
    lines = []
    if ckpt is not None:
        monkeypatch.setattr(check_restart, "read_checkpoint", lambda path: dict(ckpt))
    code = check_restart.check(_config(tmp_path, extra, edit), str(tmp_path / "run"), checkpoint="held in memory" if ckpt else None,
                               out=lines.append, **kw)
    return code, lines


def _ckpt(step, total=1_000_000, hold=500_000, lr=3e-4, own_tera=True, update=None):
    return {"update": update or step // 50, "total_steps": step * 384, "optimizer_step": step, "mode": "ppo", "lr": lr,
            "warmup": 1_000, "hold": hold, "total": total, "obs_settings": {"own_tera_type": own_tera}}


def test_the_boxs_config_is_ok_without_extra_arguments(tmp_path, monkeypatch):
    import train
    code, lines = _run(tmp_path, monkeypatch, [], _ckpt(409_000))
    assert code == 0 and lines[-1].startswith("OK to start"), lines
    said = [line for line in lines if line.startswith("the trainer will log:")][0].split("log:", 1)[1].strip()
    assert said.endswith("lr_warmup_steps=1000 lr_hold_steps=500000 lr_total_steps=1000000"), said
    assert any("full rate up to optimizer step 501,000, then down to 1/27 at step 1,000,000" in line for line in lines)
    assert any("the hold ends in 92,000 optimizer steps" in line for line in lines)
    assert any(line.startswith("own Tera type:   ON") for line in lines)
    cmd = [line for line in lines if line.startswith("trainer command")][0]
    assert " train.py " in cmd and "--mode ppo" in cmd
    # ... and it is the line train.py itself logs, from the same function
    assert said == train.ppo_settings_line(train.build_config(train.parse_args(["--mode", "ppo"])))


@pytest.mark.parametrize("extra", [["--lr-totl-steps", "5"], ["--lr-total-steps"], ["--lr-total-steps", "501000"],
                                   ["--lr-total-steps", "400000"], ["--lr", "0"], ["--lr-hold-steps", "-1"]])
def test_arguments_train_py_refuses_are_refused_before_the_start(tmp_path, monkeypatch, extra, capsys):
    code, lines = _run(tmp_path, monkeypatch, extra, _ckpt(409_000))
    assert code == 2 and lines[-1].startswith("DO NOT START: train.py refuses"), lines
    capsys.readouterr()


def test_a_config_the_supervisor_refuses_and_unexpected_arguments(tmp_path, monkeypatch):
    plan = _restart_args()
    code, lines = _run(tmp_path, monkeypatch, [], _ckpt(409_000), edit=lambda raw: raw["watchdog"].__setitem__("recycle_hours", "six"))
    assert code == 2 and "the supervisor refuses this config" in lines[-1], lines
    code, lines = _run(tmp_path, monkeypatch, ["--lr", "1e-4"], _ckpt(409_000), expect_args=plan)
    assert code == 2 and "not the ones expected" in lines[-1], lines
    code, lines = _run(tmp_path, monkeypatch, plan, _ckpt(409_000), expect_args=[])
    assert code == 2, lines
    code, lines = _run(tmp_path, monkeypatch, [], _ckpt(409_000),
                       edit=lambda raw: raw["trainer"].__setitem__("env", {"PSPPO_MEMWATCH": "/x", "PSPPO_MEMWATCH_SECONDS": "300"}))
    assert code == 0 and any("PSPPO_MEMWATCH" in line and line.startswith("trainer.env") for line in lines), lines


def test_a_config_whose_trainer_runs_another_releases_train_py_is_refused(tmp_path, monkeypatch, capsys):
    """The check parses with its own release's train.py. A config whose repo_dir names another release
    (an older one, whose train.py would refuse flags this one takes) is refused before any parse:
    otherwise the check says OK and the trainer exits 2 at every start."""
    other = tmp_path / "rel" / "v2-movesets"
    other.mkdir(parents=True)
    (other / "train.py").write_text("raise SystemExit(2)\n", encoding="utf-8")
    edit = lambda raw: raw.__setitem__("repo_dir", str(other))      # noqa: E731
    for extra in ([], ["--lr-total-steps", "1500000"]):
        code, lines = _run(tmp_path, monkeypatch, extra, _ckpt(409_000), edit=edit)
        assert code == 2 and lines[-1].startswith("DO NOT START: the trainer would run"), lines
        assert str(other / "cloud" / "check_restart.py") in lines[-1], lines
    # this checkout named through a symbolic link (a release directory linked to it) is the same train.py
    link = tmp_path / "rel" / "linked"
    try:
        os.symlink(REPO, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        link = None
    if link is not None:
        code, lines = _run(tmp_path, monkeypatch, [], _ckpt(409_000), edit=lambda raw: raw.__setitem__("repo_dir", str(link)))
        assert code == 0 and lines[-1].startswith("OK to start"), lines
    capsys.readouterr()


def test_a_learning_rate_restart_is_refused_unless_it_is_said_and_then_names_the_planned_rate(tmp_path, monkeypatch):
    plan = _restart_args()
    code, lines = _run(tmp_path, monkeypatch, plan, _ckpt(STEP), expect_args=plan)
    assert code == 3 and any("DO NOT START" in line and "factor of 2.70" in line for line in lines), lines
    assert any("policy rate at the resume: 3.000e-05" in line and "written at 1.111e-05" in line for line in lines), lines
    code, lines = _run(tmp_path, monkeypatch, plan, _ckpt(STEP), expect_args=plan, allow_rate_change=True)
    assert code == 0 and any(line.startswith("NOTE") for line in lines) and "the learning rate" in lines[-1], lines
    assert any("the hold ends in 260,417 optimizer steps" in line for line in lines), lines       # 100M decisions at 384 a step
    # the next restart of the same stage, from a checkpoint the stage wrote: nothing moves
    lr, hold, total = float(plan[1]), int(plan[3]), int(plan[5])
    for step in (STEP + 1, STEP + 200_000, STEP + 400_000):
        code, lines = _run(tmp_path, monkeypatch, plan, _ckpt(step, total=total, hold=hold, lr=lr))
        assert code == 0 and lines[-1].startswith("OK to start"), (step, lines)
    # and starting that stage's checkpoint again WITHOUT the flags drops the rate: refused too
    code, lines = _run(tmp_path, monkeypatch, [], _ckpt(STEP + 1_000, total=total, hold=hold, lr=lr))
    assert code == 3 and any("DO NOT START" in line for line in lines), lines


def test_lr_total_steps_moves_no_rate_before_the_hold_ends_and_is_refused_after(tmp_path, monkeypatch):
    longer = ["--lr-total-steps", "1500000"]
    for step in (130_592, 409_000, 500_999, 501_000):
        code, lines = _run(tmp_path, monkeypatch, longer, _ckpt(step))
        assert code == 0, (step, lines)
    code, lines = _run(tmp_path, monkeypatch, longer, _ckpt(620_000))
    want = check_restart.rate_share(620_000, 1_000, 500_000, 1_500_000) / check_restart.rate_share(620_000, 1_000, 500_000, 1_000_000)
    assert want > 1 and code == 3 and any(f"factor of {want:.2f}" in line for line in lines), lines
    code, lines = _run(tmp_path, monkeypatch, longer, _ckpt(620_000), allow_rate_change=True)
    assert code == 0 and "was allowed" in lines[-1], lines
    for step in (501_001, 620_000, 1_000_000, 1_600_000):
        code, lines = _run(tmp_path, monkeypatch, longer, _ckpt(step, total=1_500_000))
        assert code == 0, (step, lines)
    # a checkpoint of another phase: the optimizer starts afresh, nothing to compare
    other = dict(_ckpt(5_000), mode="warmup")
    code, lines = _run(tmp_path, monkeypatch, longer, other)
    assert code == 0 and any("start afresh" in line for line in lines), lines


def test_a_resume_that_changes_the_observation_is_refused_unless_that_is_said(tmp_path, monkeypatch):
    code, lines = _run(tmp_path, monkeypatch, [], _ckpt(409_000, own_tera=False))
    assert code == 2 and any("DO NOT START" in line and "switches ON ['own_tera_type']" in line for line in lines), lines
    code, lines = _run(tmp_path, monkeypatch, [], _ckpt(409_000, own_tera=False), allow_obs_change=True)
    assert code == 0 and "the observation" in lines[-1] and lines[-1].startswith("OK to start"), lines
    code, lines = _run(tmp_path, monkeypatch, ["--no-own-tera-type"], _ckpt(409_000))
    assert code == 2 and any("DROPS ['own_tera_type']" in line for line in lines), lines
    code, lines = _run(tmp_path, monkeypatch, ["--no-own-tera-type"], _ckpt(409_000, own_tera=False))
    assert code == 0, lines


def test_the_share_of_the_rate_is_the_learners_own_schedule():
    """rate_share against the function learner.py hands to LambdaLR, read from its source and run."""
    with open(os.path.join(REPO, "learner.py"), "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "lr_lambda")
    for w, h, t in ((1_000, 500_000, 100_000), (1_000, 500_000, 1_500_000), (1_000, 500_000, 1_000_000), (0, 0, 1_001),
                    (1_000, 1_364_951, 2_368_646)):
        scope = {"w_steps": w, "h_steps": h, "t_steps": t}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "learner.py", "exec"), scope)
        steps = list(range(0, 2_000, 37)) + list(range(w + h - 3, w + h + 5)) + list(range(0, 2_500_000, 9_973)) + [t - 1, t, t + 1]
        for s in steps:
            if s >= 0:
                assert check_restart.rate_share(s, w, h, t) == scope["lr_lambda"](s), (s, w, h, t)
    assert check_restart.rate_share(501_000, 1_000, 500_000, 100_000) == 1.0
    assert abs(check_restart.rate_share(501_001, 1_000, 500_000, 100_000) - 1 / 27) < 1e-12


def test_without_checkpoint_the_newest_of_the_run_directory_is_read(tmp_path, monkeypatch):
    ckdir = tmp_path / "run" / "checkpoints"
    ckdir.mkdir(parents=True)
    for i, n in enumerate((10, 30, 20)):
        f = ckdir / f"learner_update_{n:06d}.pt"
        f.write_bytes(b"x")
        os.utime(f, (1_000_000 + i, 1_000_000 + i))
    read = []
    monkeypatch.setattr(check_restart, "read_checkpoint", lambda path: (read.append(path), _ckpt(400_000))[1])
    lines = []
    code = check_restart.check(_config(tmp_path, []), str(tmp_path / "run"), expect_args=[], out=lines.append)
    assert code == 0 and lines[-1].startswith("OK to start"), lines
    assert read == [str(ckdir / "learner_update_000020.pt")], read


def test_a_real_checkpoint_is_read_and_nothing_is_written(tmp_path):
    """PSPPO_CHECK_RESTART_CKPT=<path> (skipped without): the restart's flags for that checkpoint, checked against it."""
    path = os.environ.get("PSPPO_CHECK_RESTART_CKPT", "")
    if not path or not os.path.isfile(path):
        pytest.skip("PSPPO_CHECK_RESTART_CKPT names no checkpoint here")
    before = (os.path.getmtime(path), os.path.getsize(path))
    ck = check_restart.read_checkpoint(path)
    plan = _restart_args(ck["optimizer_step"], ck["total_steps"])
    lines = []
    code = check_restart.check(_config(tmp_path, plan), str(tmp_path / "run"), expect_args=plan, checkpoint=path, out=lines.append)
    assert code == 3 and any("DO NOT START" in line for line in lines), lines
    lines = []
    code = check_restart.check(_config(tmp_path, plan), str(tmp_path / "run"), expect_args=plan, checkpoint=path, out=lines.append,
                               allow_rate_change=True)
    assert code == 0 and any("policy rate at the resume: 3.000e-05" in line for line in lines), lines
    print("\n".join(lines))
    assert (os.path.getmtime(path), os.path.getsize(path)) == before
    assert not os.path.exists(tmp_path / "run")
