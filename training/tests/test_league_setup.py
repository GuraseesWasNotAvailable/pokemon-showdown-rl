"""
cloud/v2/league_setup.py: the league run's run.json from run T's, the refusals, the league start.

    python -m pytest tests/test_league_setup.py -q
"""
from __future__ import annotations

import copy
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "cloud", "v2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")

import league_setup as ls  # noqa: E402

REL, RUN = "/home/ubuntu/rel/league", "/home/ubuntu/runs/v2/league"


def test_the_run_json_is_run_ts_with_the_league():
    cfg = ls.build_run_json(ls.T_LIVE, REL, RUN)
    assert cfg["repo_dir"] == REL and cfg["servers"]["base_port"] == 10900 and cfg["eval"]["port"] == 10990
    assert cfg["train_defaults"]["extra_train_args"] == ls.DEFAULT_ARGS and "--obs-main-line" in ls.DEFAULT_ARGS
    ppo = cfg["phases"]["ppo"]
    assert ppo["target_decisions"] == 300_000_000
    args = ppo["extra_train_args"]
    joined = " ".join(args)
    assert "--read-bonus" not in args and "--read-anneal" not in args
    for want in ("--opponent-seats setupnet:0.15,frozen:0.25", "--frozen-past 4:0.40", "--learn-queue-episodes 32", "--lag-stats",
                 "--lr 6e-05", "--lr-hold-decisions 100000000", "--lr-total-decisions 300000000", "--fast-update --fast-ingest",
                 f"m24381={RUN}/league/M24381w.pt:0.35", f"sfin={RUN}/league/SFIN.pt:0.25", "--frozen-refresh-min 30"):
        assert want in joined, want
    assert cfg["phases"]["warmup"]["extra_train_args"] == ["--set-head-coef", "0.1"]
    assert cfg["phases"]["imitation"]["skip"] is True and cfg["retention"]["enabled"] is False
    # everything else is run T's
    for key in ("trainer", "watchdog", "disk", "python", "showdown_dir"):
        assert cfg[key] == ls.T_LIVE[key]
    assert ls.T_LIVE["repo_dir"] == "/home/ubuntu/rel/topstart", "the base is not changed"


def test_every_argument_passes_train_pys_parser():
    import train
    cfg = ls.build_run_json(ls.T_LIVE, REL, RUN)
    for phase in ("warmup", "ppo"):
        argv = ["--mode", phase] + cfg["train_defaults"]["extra_train_args"] + cfg["phases"][phase]["extra_train_args"]
        args = train.parse_args(argv)
        rc = train.build_config(args)
        assert rc.obs.main_line and rc.obs.matchup and rc.obs.movesets and not rc.obs.own_tera_type
    import league
    import opponent_seats
    opts = league.from_args(args)
    assert opts.learn_queue_episodes == 32 and opts.lag_stats
    seats = opponent_seats.from_args(args)
    assert seats.share("frozen") == 0.25 and seats.frozen.slots == ["m24381", "sfin", "past0", "past1", "past2", "past3"]
    assert rc.learner.lr == 6e-05 and rc.learner.lr_hold_decisions == 100_000_000 and rc.learner.lr_total_decisions == 300_000_000


def test_the_rehearsal():
    cfg = ls.build_run_json(ls.T_LIVE, REL, "/x/rehearsal", rehearsal=True)
    assert cfg["servers"]["count"] == 6 and cfg["servers"]["base_port"] == 11400 and cfg["train_defaults"]["battles"] == 384
    assert "eval" not in cfg and cfg["trainer"]["env"] == {"CUDA_VISIBLE_DEVICES": "0"}
    assert cfg["train_defaults"]["extra_train_args"][cfg["train_defaults"]["extra_train_args"].index("--infer-actors") + 1] == "1"
    assert "--frozen-refresh-min 5" in " ".join(cfg["phases"]["ppo"]["extra_train_args"])


def test_a_base_that_is_not_run_ts_is_refused():
    for edit in ("ppo", "defaults", "command"):
        b = copy.deepcopy(ls.T_LIVE)
        if edit == "ppo":
            b["phases"]["ppo"]["extra_train_args"] = b["phases"]["ppo"]["extra_train_args"][:-1]
        elif edit == "defaults":
            b["train_defaults"]["extra_train_args"] = ["--obs-matchup"]
        else:
            b["servers"].pop("command")
        with pytest.raises(ls.Refusal):
            ls.build_run_json(b, REL, RUN)


def test_config_problems():
    cfg = ls.build_run_json(ls.T_LIVE, REL, RUN)
    assert ls.config_problem(cfg) is None
    bad = copy.deepcopy(cfg)
    bad["train_defaults"]["extra_train_args"] = [a for a in bad["train_defaults"]["extra_train_args"] if a != "--obs-main-line"]
    assert "--obs-main-line" in ls.config_problem(bad)
    bad = copy.deepcopy(cfg)
    bad["phases"]["ppo"]["extra_train_args"] += ["--read-bonus", "0.3"]
    assert "read bonus" in ls.config_problem(bad)
    bad = copy.deepcopy(cfg)
    bad["retention"]["enabled"] = True
    assert "retention" in ls.config_problem(bad)


def _ck(**obs):
    model = {"set_head.w": torch.zeros(1), "w": torch.ones(2)}
    return {"model": model, "target_model": {"w": torch.ones(2)}, "optimizer": {"x": 1}, "scheduler": {"last_epoch": 9},
            "update_idx": 24381, "total_steps": 900_057_460, "total_episodes": 7,
            "run_cfg": {"obs": dict(obs), "learner": {"mode": "ppo"}}, "widened_main_line": {"from": "x"}}


def test_the_start_and_sfin_checks():
    good = _ck(main_line=True, matchup=True, movesets=True, own_tera_type=False)
    assert ls.start_problem(good) is None
    assert "observation" in ls.start_problem(_ck(matchup=True, movesets=True, own_tera_type=True))
    no_head = _ck(main_line=True, matchup=True, movesets=True)
    no_head["model"] = {"w": torch.ones(2)}
    assert "set_head" in ls.start_problem(no_head)
    unwidened = _ck(main_line=True, matchup=True, movesets=True)
    unwidened.pop("widened_main_line")
    assert "widened" in ls.start_problem(unwidened)
    assert ls.sfin_problem({"run_cfg": {"obs": {"own_tera_type": True, "matchup": True, "movesets": True}}}) is None
    assert ls.sfin_problem({"run_cfg": {"obs": {"main_line": True, "matchup": True, "movesets": True}}})
    assert ls.sfin_problem({"run_cfg": {"obs": {"matchup": True}}})


def test_the_league_start_keeps_the_tensors_and_starts_the_counters_afresh():
    ck = _ck(main_line=True, matchup=True, movesets=True)
    out = ls.league_start(ck)
    assert out["model"] is ck["model"] and out["target_model"] is ck["target_model"]
    assert "optimizer" not in out and "scheduler" not in out
    assert out["update_idx"] == out["total_steps"] == out["total_episodes"] == 0
    assert out["run_cfg"]["learner"]["mode"] == "imitation" and ck["run_cfg"]["learner"]["mode"] == "ppo", "the given one unchanged"
    assert out["league_start"]["mode"] == "ppo" and out["league_start"]["total_steps"] == 900_057_460
    assert ls.same_tensors(out["model"], ck["model"]) and not ls.same_tensors({"w": torch.zeros(2)}, {"w": torch.ones(2)})


def test_the_units():
    units = ls.unit_texts(REL, RUN, "0,1", py="/home/ubuntu/venv/bin/python", home="/home/ubuntu")
    assert set(units) == {"psppo-v2-league-train", "psppo-v2-league-eval"}
    assert f"ExecStart=/home/ubuntu/venv/bin/python {REL}/pipeline.py --config {RUN}/run.json --run-dir {RUN}" in units["psppo-v2-league-train"]
    assert "--can-start" in units["psppo-v2-league-train"] and "PSPPO_GPU_SPLIT=1" in units["psppo-v2-league-train"]
    assert "evaluate.py --run-dir" in units["psppo-v2-league-eval"] and "Nice=5" in units["psppo-v2-league-eval"]
