"""
The learning-rate restart of Stage 1: train.py's --lr, --lr-hold-steps and --lr-total-steps with
their checks (ported from stage3), the "ppo settings:" line every start logs, the rate the learner
resumes at with them, and cloud/stage_lr.py's flags carried through all of it.

    python -m pytest tests/test_lr_restart.py -q -p no:cacheprovider

Held here:
  * without the flags the config is config.py's, field for field, and a Namespace built by hand
    without them still builds;
  * every flag reaches its LearnerConfig field and only that field;
  * a schedule that cannot be (the anneal ending at or before the end of the hold, a negative
    hold, a rate of 0 or less) is refused by parse_args (exit 2) and by build_config, but only
    when a schedule flag was given;
  * train.main logs "ppo settings: ..." (train.ppo_settings_line), which the supervisor parses as
    nothing, next to the header line it does parse;
  * the flags stage_lr.plan computes for a checkpoint past the end of the anneal (the full run's
    situation: 1/27 of 3e-4) make the learner resume at exactly the planned rate in every
    parameter group, hold it, then decline; without them the resume keeps the checkpoint's rate;
  * stage_lr.py's command line reads a checkpoint file and prints flags train.py takes.

No Ray, no server, no GPU needed (the learner is built and loaded, never stepped). Nothing is removed.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, TESTS, os.path.join(REPO, "cloud")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

import torch  # noqa: E402

import config as new_config  # noqa: E402
import learner as new_learner  # noqa: E402
import ppo_core as new_core  # noqa: E402
import train as new_train  # noqa: E402
import test_speed_update as tsu  # noqa: E402  (its learner helpers)
import stage_lr  # noqa: E402
from check_restart import rate_share  # noqa: E402

DEV = tsu.DEV
# A checkpoint past the end of config.py's anneal (step 1,001,000), as the full run's is.
STEP, DEC = 1_104_534, 604_489_541


def test_without_the_flags_the_config_is_config_pys():
    base = new_config.RunConfig.default()
    assert new_train.build_config(new_train.parse_args([])) == base
    old_style = argparse.Namespace(mode="warmup", ports=None, battles=None, rooms_per_pair=None, teacher=None, steps_per_update=None,
                                   minibatch=None, seq_len=None, grad_accum=None, save_every=None, ckpt_dir=None, shaping=None,
                                   minutes=0.0, stop_file=None)
    assert new_train.build_config(old_style) == dataclasses.replace(base, learner=dataclasses.replace(base.learner, mode="warmup"))
    lc = base.learner
    assert (lc.lr, lc.lr_warmup_steps, lc.lr_hold_steps, lc.lr_total_steps) == (3e-4, 1_000, 500_000, 1_000_000)


def test_every_flag_reaches_its_field_and_only_its_field():
    base = new_config.RunConfig.default()
    cases = [
        (["--lr", "1e-4"], "lr", 1e-4),
        (["--lr", "3e-05"], "lr", 3e-5),
        (["--lr-total-steps", "1622000"], "lr_total_steps", 1_622_000),
        (["--lr-hold-steps", "50000"], "lr_hold_steps", 50_000),
    ]
    for argv, field, value in cases:
        cfg = new_train.build_config(new_train.parse_args(argv))
        assert getattr(cfg.learner, field) == value, argv
        back = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **{field: getattr(base.learner, field)}))
        assert back == base, (argv, "something else moved")
    cfg = new_train.build_config(new_train.parse_args(["--mode", "ppo", "--battles", "64", "--lr", "2e-4", "--lr-hold-steps", "1361178",
                                                       "--lr-total-steps", "2000000"]))
    assert (cfg.learner.mode, cfg.rollout.target_concurrent_battles, cfg.learner.lr, cfg.learner.lr_hold_steps,
            cfg.learner.lr_total_steps) == ("ppo", 64, 2e-4, 1_361_178, 2_000_000)
    assert cfg.learner.ppo_kwargs() == base.learner.ppo_kwargs()          # nothing the update is handed moves


def test_a_schedule_that_cannot_be_is_refused_at_start_only_when_asked_for(capsys):
    plain = vars(new_train.parse_args([]))
    bad = [["--lr-hold-steps", "1000000"],                     # total stays 1,000,000: not after 1,000 + 1,000,000
           ["--lr-total-steps", "400000"],                     # hold stays 500,000
           ["--lr-total-steps", "501000"],                     # exactly warmup + hold: still one step
           ["--lr-hold-steps", "200000", "--lr-total-steps", "201000"],
           ["--lr-hold-steps", "-5", "--lr-total-steps", "9000"]]
    for argv in bad:
        with pytest.raises(SystemExit) as e:
            new_train.parse_args(argv)
        assert e.value.code == 2, argv
        err = capsys.readouterr().err
        assert "learning-rate schedule cannot be" in err or "must not be negative" in err, (argv, err[-300:])
        asked = {argv[i].lstrip("-").replace("-", "_"): int(argv[i + 1]) for i in range(0, len(argv), 2)}
        with pytest.raises(ValueError):
            new_train.build_config(argparse.Namespace(**{**plain, **asked}))
    good = [(["--lr-total-steps", "501001"], 500_000, 501_001), (["--lr-hold-steps", "50000"], 50_000, 1_000_000),
            (["--lr-hold-steps", "0", "--lr-total-steps", "1001"], 0, 1_001),
            (["--lr-hold-steps", "1361178", "--lr-total-steps", "2000000"], 1_361_178, 2_000_000)]
    for argv, hold, total in good:
        lc = new_train.build_config(new_train.parse_args(argv)).learner
        assert (lc.lr_hold_steps, lc.lr_total_steps) == (hold, total), argv
    for argv in (["--lr", "0"], ["--lr", "-1e-4"], ["--lr-total-steps", "soon"]):
        with pytest.raises(SystemExit) as e:
            new_train.parse_args(argv)
        assert e.value.code == 2, argv
    capsys.readouterr()


class _StopHere(Exception):
    pass


def test_the_start_log_has_the_settings_line(monkeypatch, caplog):
    import pipeline

    class Ray:
        @staticmethod
        def init(**_k):
            return None

        @staticmethod
        def put(_cfg):
            raise _StopHere()

    monkeypatch.setattr(new_train, "ray", Ray)

    def settings_of(argv):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=new_train.logger.name):
            with pytest.raises(_StopHere):
                asyncio.run(new_train.main(new_train.parse_args(argv)))
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("ppo settings:")]
        header = [r.getMessage() for r in caplog.records if r.getMessage().startswith("mode=")]
        assert len(lines) == 1 and len(header) == 1, [r.getMessage() for r in caplog.records]
        stamp = "2026-10-15 03:30:01,123 [INFO] "
        assert pipeline.parse_trainer_events(stamp + lines[0]) == [], "the supervisor read the settings line as something"
        assert [e["kind"] for e in pipeline.parse_trainer_events(stamp + header[0])] == ["header"]
        return lines[0]

    assert settings_of([]) == ("ppo settings: clip_coef=0.1 update_epochs=2 lr=0.0003 target_kl=0.02 ent_coef=0.02 "
                               "lr_warmup_steps=1000 lr_hold_steps=500000 lr_total_steps=1000000")
    argv = ["--lr", "3e-05", "--lr-hold-steps", "1361178", "--lr-total-steps", "2000000"]
    line = settings_of(argv)
    assert line.endswith("lr=3e-05 target_kl=0.02 ent_coef=0.02 lr_warmup_steps=1000 lr_hold_steps=1361178 lr_total_steps=2000000")
    assert line == new_train.ppo_settings_line(new_train.build_config(new_train.parse_args(argv)))


# --------------------------------------------------------------------------- the learner at the resume


def _learner(cfg):
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, device=DEV, resume=False,
                                                               ckpt_dir=os.path.join(tsu.TEST_ROOT, "ckpt_unused")))
    return tsu._make_learner(new_learner, new_core, cfg)


@pytest.fixture(scope="module")
def checkpoint():
    """A PPO checkpoint at optimizer step STEP under config.py's schedule (rate 1/27 of 3e-4 there)."""
    run = _learner(new_train.build_config(new_train.parse_args(["--mode", "ppo"])))
    run.opt.inner.step()                                   # LambdaLR wants an optimizer step before its own
    run.sched.last_epoch = STEP - 1
    run.sched.step()
    run.update_idx, run.total_steps = 16_375, DEC
    ck = tsu._checkpoint_in_memory(run)
    assert ck["scheduler"]["last_epoch"] == STEP and abs(run._policy_lr() - 3e-4 / 27) < 1e-15
    del run
    return ck


def _resumed(monkeypatch, ck, argv):
    monkeypatch.setattr(torch, "load", lambda *a, **k: ck)
    la = _learner(new_train.build_config(new_train.parse_args(["--mode", "ppo"] + argv)))
    la._load_checkpoint("held in memory")
    assert la.sched.last_epoch == STEP and la.update_idx == 16_375 and la.total_steps == DEC
    return la


MULT = {"embeddings": 0.1, "subnets": 0.1, "transformer": 0.1, "readout": 1.0, "pi": 1.0, "v": 2.0}


def test_without_flags_the_resume_keeps_the_checkpoints_rate(monkeypatch, checkpoint, capsys):
    la = _resumed(monkeypatch, checkpoint, [])
    for pg in la.opt.param_groups:
        assert abs(pg["lr"] - 3e-4 * MULT[pg["name"].split("_")[0]] / 27) < 1e-15, pg["name"]
    assert "policy rate at the resume: 1.111e-05" in capsys.readouterr().out
    del la


@pytest.mark.parametrize("start_rate,hold_decisions,end_decisions", [(3e-5, 100_000_000, DEC + 300_000_000),
                                                                      (3e-5, 50_000_000, DEC + 60_000_000),
                                                                      (6e-5, 0, DEC + 300_000_000)])
def test_the_planned_restart_resumes_at_the_planned_rate(monkeypatch, checkpoint, capsys, start_rate, hold_decisions, end_decisions):
    p = stage_lr.plan(STEP, DEC, end_decisions, start_rate, hold_decisions=hold_decisions)
    argv = ["--lr", repr(p["lr"]), "--lr-hold-steps", str(p["lr_hold_steps"]), "--lr-total-steps", str(p["lr_total_steps"])]
    la = _resumed(monkeypatch, checkpoint, argv)
    for pg in la.opt.param_groups:
        want = p["lr"] * MULT[pg["name"].split("_")[0]] * rate_share(STEP, 1_000, p["lr_hold_steps"], p["lr_total_steps"])
        assert abs(pg["lr"] - want) <= 1e-12 * want, (pg["name"], pg["lr"], want)
    # exact for a hold; the decline is placed by rounding its start to a whole step (stage_lr's own tolerance)
    assert la._policy_lr() == pytest.approx(start_rate, rel=1e-12 if hold_decisions else 1e-4)
    assert la._policy_lr() == pytest.approx(p["rate_at_resume"], rel=1e-12)
    assert la.sched.get_last_lr() == [pg["lr"] for pg in la.opt.param_groups]
    assert f"policy rate at the resume: {la._policy_lr():.3e}" in capsys.readouterr().out
    fn = la.sched.lr_lambdas[0]
    edge = 1_000 + p["lr_hold_steps"]
    if hold_decisions:
        assert fn(STEP) == fn(edge - 1) == 1.0 and fn(edge + 10) < 1.0     # held, then declining
        assert edge == STEP + round(hold_decisions / 384)
    else:
        assert fn(STEP + 1000) < fn(STEP)                                  # declining from the resume on
    end_step = STEP + round((end_decisions - DEC) / 384)
    assert abs(p["lr"] * fn(end_step) - p["rate_at_end"]) < 1e-15
    del la


def test_stage_lr_reads_a_checkpoint_file_and_prints_flags_train_py_takes(tmp_path, checkpoint, capsys):
    path = str(tmp_path / "learner_update_016375.pt")
    torch.save({k: checkpoint[k] for k in ("scheduler", "total_steps", "run_cfg")}, path)
    assert stage_lr.main(["--checkpoint", path, "--start-rate", "3e-5", "--hold-decisions", "100000000",
                          "--end-decisions", str(DEC + 300_000_000)]) == 0
    out = capsys.readouterr().out
    flags = next(line for line in out.splitlines() if line.startswith("flags: ")).split("flags: ", 1)[1].split()
    assert flags[0::2] == ["--lr", "--lr-hold-steps", "--lr-total-steps"] and float(flags[1]) == 3e-5
    lc = new_train.build_config(new_train.parse_args(["--mode", "ppo"] + flags)).learner
    assert 3e-5 * rate_share(STEP, lc.lr_warmup_steps, lc.lr_hold_steps, lc.lr_total_steps) == pytest.approx(3e-5)
    assert "optimizer step 1,104,534" in out and "384 decisions per optimizer step" in out
