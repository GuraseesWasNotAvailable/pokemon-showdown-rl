"""
Tests for evaluate.py and the evaluation players in players.py.

    python -m pytest tests/test_evaluate.py -q
    python tests/test_evaluate.py                 (plain unittest, no pytest needed)

No Showdown server, no GPU and no trained checkpoint is needed. The tests that touch torch or
poke-env skip themselves where those are not installed. Nothing here removes a file: every test
writes into its own new temporary directory and leaves it there.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import evaluate as ev  # noqa: E402


def new_dir() -> str:
    """A fresh directory for one test, under $TEST_EVALUATE_TMP when that is set. It is left in place."""
    base = os.environ.get("TEST_EVALUATE_TMP") or None
    if base:
        os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="test_evaluate_", dir=base)


def write_file(path: str, size: int = 64, mtime: float = None) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def line(opponent="simple", games=200, wins=100, ties=0, errors=0, t="2026-10-05T00:00:00Z", name="learner_update_000010.pt",
         mtime=1000.0, sample=False, **extra):
    """A result line the way evaluate.py writes it."""
    tally = ev.Tally(requested=games + errors, wins=wins, losses=games - wins - ties, ties=ties)
    meta = {"checkpoint": name, "update": int(name[15:21]), "phase": extra.pop("phase", "ppo"), "total_steps": 1000,
            "checkpoint_mtime": mtime}
    out = ev.result_line(meta, opponent, tally, sample=sample, extras=extra, now=ev.parse_iso(t))
    return out


class WilsonTests(unittest.TestCase):
    def test_known_values(self):
        lo, hi = ev.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        lo, hi = ev.wilson(0, 20)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.1611, places=3)
        lo, hi = ev.wilson(20, 20)
        self.assertAlmostEqual(lo, 0.8389, places=3)
        self.assertEqual(hi, 1.0)

    def test_no_games_is_the_whole_range(self):
        self.assertEqual(ev.wilson(0, 0), (0.0, 1.0))

    def test_interval_contains_the_rate_and_narrows(self):
        for wins, n in ((1, 20), (10, 20), (150, 200), (799, 800), (3, 7)):
            lo, hi = ev.wilson(wins, n)
            self.assertLessEqual(lo, wins / n)
            self.assertGreaterEqual(hi, wins / n)
        self.assertLess(ev.wilson(400, 800)[1] - ev.wilson(400, 800)[0], ev.wilson(100, 200)[1] - ev.wilson(100, 200)[0])

    def test_a_tie_is_half_a_win(self):
        self.assertEqual(ev.score(10, 2, 20), 0.55)
        self.assertEqual(ev.score(0, 0, 0), 0.0)
        self.assertEqual(ev.wilson(10.5, 21), ev.wilson(10 + 0.5 * 1, 21))
        lo, hi = ev.wilson(10.5, 21)
        self.assertTrue(lo < 0.5 < hi)

    def test_two_hundred_games_is_about_seven_points(self):
        self.assertAlmostEqual(ev.noise_points(200), 6.93, places=2)
        lo, hi = ev.wilson(100, 200)
        self.assertAlmostEqual(100 * (hi - lo) / 2, 6.86, places=2)

    def test_two_proportion_test(self):
        self.assertAlmostEqual(ev.two_proportion_p(0.5, 200, 0.6, 200), 0.0444, places=3)
        self.assertEqual(ev.two_proportion_p(1.0, 200, 1.0, 200), 1.0)
        self.assertEqual(ev.two_proportion_p(0.5, 0, 0.6, 200), 1.0)
        self.assertLess(ev.two_proportion_p(0.1, 200, 0.9, 200), 1e-20)
        self.assertAlmostEqual(ev.two_proportion_p(0.5, 200, 0.5, 200), 1.0)


class ResultLineTests(unittest.TestCase):
    META = {"checkpoint": "learner_update_000123.pt", "update": 123, "phase": "ppo", "total_steps": 2015232,
            "checkpoint_path": "/r/checkpoints/learner_update_000123.pt", "checkpoint_mtime": 1700000000.123,
            "checkpoint_size": 219392768, "total_episodes": 9, "run_decisions": None, "meta_from": "checkpoint"}

    def test_contract_keys_first_and_typed(self):
        tally = ev.Tally(requested=200, wins=120, losses=70, ties=4, turns=5820, seconds=61.0)
        tally.count("policy_decisions", 6000)
        got = ev.result_line(self.META, "simple", tally, sample=False, extras={"device": "cuda", "reason": "interval"})
        self.assertEqual(tuple(got)[:len(ev.CORE_KEYS)], ev.CORE_KEYS)
        self.assertEqual(got["checkpoint"], "learner_update_000123.pt")
        self.assertEqual((got["update"], got["phase"], got["total_steps"]), (123, "ppo", 2015232))
        self.assertEqual((got["games"], got["wins"], got["losses"], got["ties"], got["errors"]), (194, 120, 70, 4, 6))
        self.assertEqual(got["wins"] + got["losses"] + got["ties"], got["games"])
        self.assertAlmostEqual(got["winrate"], (120 + 2) / 194, places=6)
        lo, hi = ev.wilson(122, 194)
        self.assertAlmostEqual(got["ci_low"], lo, places=6)
        self.assertAlmostEqual(got["ci_high"], hi, places=6)
        self.assertIs(got["sample"], False)
        self.assertEqual(got["action"], "best")
        for key in ("winrate", "ci_low", "ci_high", "seconds"):
            self.assertIsInstance(got[key], float)
        for key in ("games", "wins", "losses", "ties", "errors"):
            self.assertIsInstance(got[key], int)
        self.assertIsNotNone(ev.parse_iso(got["time"]))
        self.assertTrue(got["time"].endswith("Z"))
        self.assertEqual(got["mean_turns"], 30.0)
        self.assertEqual(got["unfinished"], 6)
        self.assertEqual(got["device"], "cuda")
        self.assertEqual(got["policy_decisions"], 6000)
        self.assertEqual(json.loads(json.dumps(got)), got)

    def test_extras_cannot_overwrite_the_contract(self):
        tally = ev.Tally(requested=10, wins=10)
        got = ev.result_line(self.META, "random", tally, sample=True, extras={"winrate": 0.0, "opponent": "x", "games": 3})
        self.assertEqual((got["winrate"], got["opponent"], got["games"]), (1.0, "random", 10))
        self.assertIs(got["sample"], True)
        self.assertEqual(got["action"], "sample")

    def test_no_finished_game_is_marked(self):
        tally = ev.Tally(requested=50, notes=["failed: boom"])
        got = ev.result_line(self.META, "psbot", tally, sample=False)
        self.assertEqual((got["games"], got["errors"], got["winrate"], got["ci_low"], got["ci_high"]), (0, 50, 0.0, 0.0, 1.0))
        self.assertIs(got["valid"], False)
        self.assertIn("no game ran to its end", got["problems"])
        self.assertIn("failed: boom", got["notes"])

    def test_forfeits_and_timeouts_are_errors_not_results(self):
        tally = ev.Tally(requested=20, wins=10, losses=6, forfeits=3, timeouts=1)
        got = ev.result_line(self.META, "simple", tally, sample=False)
        self.assertEqual((got["games"], got["errors"], got["ended_by_forfeit"], got["ended_by_timeout"], got["unfinished"]),
                         (16, 4, 3, 1, 0))
        self.assertIs(got["valid"], False)   # a fifth of the games lost to errors

    def test_broken_play_is_visible(self):
        tally = ev.Tally(requested=100, wins=90, losses=10)
        tally.count("policy_decisions", 3000)
        tally.count("policy_invalid_orders", 2)
        tally.count("policy_exceptions", 400)
        got = ev.result_line(self.META, "random", tally, sample=False)
        self.assertIs(got["valid"], False)
        self.assertTrue(any("rejected by the server" in p for p in got["problems"]))
        self.assertTrue(any("raised" in p for p in got["problems"]))
        clean = ev.Tally(requested=100, wins=90, losses=10)
        clean.count("policy_decisions", 3000)
        clean.count("policy_invalid_orders", 2)    # reported, but two invalid orders in 3000 do not void the row
        clean.count("policy_unavailable_orders", 9)   # a trapped Pokemon that tried to switch: normal play, no problem
        clean.count("opponent_unavailable_orders", 20)
        got = ev.result_line(self.META, "random", clean, sample=False)
        self.assertIs(got["valid"], True)
        self.assertEqual(len(got["problems"]), 1)
        self.assertEqual((got["policy_unavailable_orders"], got["opponent_unavailable_orders"]), (9, 20))

    def test_an_opponent_playing_at_random_voids_the_row(self):
        tally = ev.Tally(requested=100, wins=95, losses=5)
        tally.count("opponent_decisions", 2500)
        tally.count("opponent_fallbacks", 2500)
        self.assertIs(ev.result_line(self.META, "psbot", tally, sample=False)["valid"], False)

    def test_append_and_read_back(self):
        path = os.path.join(new_dir(), "eval", "results.jsonl")
        a, b = line("random", wins=190), line("simple", wins=40)
        ev.append_jsonl(path, a)
        ev.append_jsonl(path, b)
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"time": "2026-10-05T00:0')   # a write that was cut short
        self.assertEqual(ev.read_jsonl(path), [a, b])
        with open(path, encoding="utf-8") as f:
            self.assertEqual(len(f.read().splitlines()), 3)


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.dir = os.path.join(new_dir(), "checkpoints")
        os.makedirs(self.dir)
        self.now = time.time()

    def ck(self, n: int, age: float, size: int = 64) -> str:
        return write_file(os.path.join(self.dir, f"learner_update_{n:06d}.pt"), size, self.now - age)

    def test_newest_by_time_not_by_number(self):
        self.ck(300, age=3000)          # the last checkpoint of the phase before
        self.ck(25, age=100)            # the new phase started counting again
        self.ck(50, age=10)
        write_file(os.path.join(self.dir, "learner_update_000075.pt.tmp"), 64, self.now)       # train.py is writing this
        write_file(os.path.join(self.dir, "notes.txt"), 64, self.now)
        os.makedirs(os.path.join(self.dir, "learner_update_000099.pt"))                        # a directory, not a file
        got = ev.newest_checkpoint(self.dir, settle_s=0)
        self.assertEqual(got.name, "learner_update_000050.pt")
        self.assertEqual([c.name for c in ev.list_checkpoints(self.dir)],
                         ["learner_update_000050.pt", "learner_update_000025.pt", "learner_update_000300.pt"])

    def test_missing_or_empty_directory(self):
        self.assertIsNone(ev.newest_checkpoint(os.path.join(self.dir, "nope"), settle_s=0))
        self.assertIsNone(ev.newest_checkpoint(self.dir, settle_s=0))
        self.assertEqual(ev.list_checkpoints(os.path.join(self.dir, "nope")), [])

    def test_file_still_growing_is_skipped(self):
        self.ck(10, age=500)
        growing = self.ck(20, age=0)

        def copy_goes_on(_seconds):                 # stands in for the wait: the copy writes more meanwhile
            with open(growing, "ab") as f:
                f.write(b"more")

        notes = []
        got = ev.newest_checkpoint(self.dir, settle_s=1.0, sleep=copy_goes_on, notes=notes)
        self.assertEqual(got.name, "learner_update_000010.pt")      # one evaluation by hand: the older one will do
        self.assertTrue(any("learner_update_000020.pt is still being written" in n for n in notes))
        got = ev.newest_checkpoint(self.dir, settle_s=1.0, sleep=copy_goes_on, fall_back=False)
        self.assertIsNone(got)                                       # the watch loop: come back later
        got = ev.newest_checkpoint(self.dir, settle_s=1.0, sleep=lambda s: None, fall_back=False)
        self.assertEqual(got.name, "learner_update_000020.pt")      # the copy has finished

    def test_same_size_but_touched_again_is_not_settled(self):
        path = self.ck(20, age=50)

        def rewrite(_seconds):
            os.utime(path, (self.now, self.now))

        self.assertIsNone(ev.newest_checkpoint(self.dir, settle_s=1.0, sleep=rewrite, fall_back=False))

    def test_empty_file_is_skipped(self):
        self.ck(10, age=500)
        self.ck(20, age=0, size=0)
        self.assertEqual(ev.newest_checkpoint(self.dir, settle_s=0).name, "learner_update_000010.pt")
        self.assertIsNone(ev.newest_checkpoint(self.dir, settle_s=0, fall_back=False))

    def test_file_that_does_not_load_is_skipped_and_nothing_raises(self):
        self.ck(10, age=500)
        bad = self.ck(20, age=0)
        seen = []

        def loader(path):
            seen.append(os.path.basename(path))
            if path == os.path.abspath(bad):
                raise EOFError("Ran out of input")

        notes = []
        got = ev.newest_checkpoint(self.dir, settle_s=0, loader=loader, notes=notes)
        self.assertEqual(got.name, "learner_update_000010.pt")
        self.assertEqual(seen, ["learner_update_000020.pt", "learner_update_000010.pt"])
        self.assertTrue(any("cannot be read" in n for n in notes))
        self.assertIsNone(ev.newest_checkpoint(self.dir, settle_s=0, loader=loader, fall_back=False))

    def test_skip_passes_over_a_version(self):
        self.ck(10, age=500)
        self.ck(20, age=0)
        newest = ev.newest_checkpoint(self.dir, settle_s=0)
        self.assertEqual(ev.newest_checkpoint(self.dir, settle_s=0, skip=[newest.key]).name, "learner_update_000010.pt")

    def test_a_reused_name_is_another_version(self):
        path = self.ck(25, age=5000)
        old = ev.stat_ckpt(path)
        os.utime(path, (self.now, self.now))       # a later phase wrote update 25 again
        new = ev.stat_ckpt(path)
        self.assertNotEqual(old.key, new.key)
        old_line = line("simple", name=old.name, mtime=round(old.mtime, 3))
        self.assertTrue(ev.same_version(old_line, old))
        self.assertFalse(ev.same_version(old_line, new))
        self.assertEqual(ev.opponents_done([old_line], new), [])
        self.assertEqual(ev.opponents_done([old_line], old), ["simple"])

    def test_real_torch_checkpoint_and_a_truncated_copy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        good = os.path.join(self.dir, "learner_update_000007.pt")
        payload = {"model": {"w": torch.zeros(4, 4), "b": torch.zeros(4)}, "optimizer": {}, "update_idx": 7, "total_steps": 114688,
                   "total_episodes": 321, "run_cfg": {"learner": {"mode": "warmup"}}}
        torch.save(payload, good)
        os.utime(good, (self.now - 100, self.now - 100))
        meta = ev.load_checkpoint_meta(good)
        self.assertEqual((meta["update"], meta["total_steps"], meta["phase"], meta["total_episodes"], meta["tensors"]),
                         (7, 114688, "warmup", 321, 2))
        self.assertEqual(meta["tensor_names"], ["b", "w"])                   # what the network that plays it is checked against
        with open(good, "rb") as f:
            data = f.read()
        half = os.path.join(self.dir, "learner_update_000008.pt")       # newer, and only half of it has arrived
        with open(half, "wb") as f:
            f.write(data[: len(data) // 2])
        with self.assertRaises(Exception):
            ev.load_checkpoint_meta(half)
        notes = []
        got = ev.newest_checkpoint(self.dir, settle_s=0, loader=ev.load_checkpoint_meta, notes=notes)
        self.assertEqual(got.name, "learner_update_000007.pt")
        self.assertTrue(any(n.startswith("learner_update_000008.pt cannot be read") for n in notes))
        self.assertIsNone(ev.newest_checkpoint(self.dir, settle_s=0, loader=ev.load_checkpoint_meta, fall_back=False))
        no_model = os.path.join(self.dir, "learner_update_000009.pt")
        torch.save({"update_idx": 9}, no_model)
        with self.assertRaises(ValueError):
            ev.load_checkpoint_meta(no_model)


class DescribeCheckpointTests(unittest.TestCase):
    def test_from_the_checkpoint_itself(self):
        ck = ev.Ckpt("/r/checkpoints/learner_update_000040.pt", "learner_update_000040.pt", 1234.5678, 99)
        got = ev.describe_checkpoint(ck, {"update": 41, "phase": "ppo", "total_steps": 700, "total_episodes": 5},
                                     {"phase": "warmup"}, "/nonexistent/metrics.jsonl")
        self.assertEqual((got["checkpoint"], got["update"], got["phase"], got["total_steps"]), ("learner_update_000040.pt", 41, "ppo", 700))
        self.assertEqual((got["checkpoint_mtime"], got["checkpoint_size"], got["meta_from"]), (1234.568, 99, "checkpoint"))

    def test_falls_back_to_file_name_state_and_metrics(self):
        d = new_dir()
        metrics = os.path.join(d, "metrics.jsonl")
        ev.append_jsonl(metrics, {"update": 40, "phase": "imitation", "total_steps": 111})
        ev.append_jsonl(metrics, {"update": 39, "phase": "warmup", "total_steps": 600})
        ev.append_jsonl(metrics, {"update": 40, "phase": "warmup", "total_steps": 655360})
        ck = ev.Ckpt("/r/checkpoints/learner_update_000040.pt", "learner_update_000040.pt", 1.0, 99)
        got = ev.describe_checkpoint(ck, {}, {"phase": "warmup"}, metrics)
        self.assertEqual((got["update"], got["phase"], got["total_steps"]), (40, "warmup", 655360))
        self.assertIn("file name", got["meta_from"])
        got = ev.describe_checkpoint(ck, None, None, os.path.join(d, "missing.jsonl"))
        self.assertEqual((got["update"], got["phase"], got["total_steps"]), (40, None, None))
        got = ev.describe_checkpoint(ck, None, {"phase": "done"}, os.path.join(d, "missing.jsonl"))
        self.assertIsNone(got["phase"])        # "done" is the run's state, not what the checkpoint was trained in

    def test_run_decisions_adds_the_finished_phases(self):
        state = {"phase": "ppo", "phases": {"imitation": {"status": "done", "decisions": 5000000},
                                            "warmup": {"status": "done", "decisions": 1000000}, "ppo": {"status": "running"}}}
        self.assertEqual(ev.run_decisions(state, "ppo", 250), 6000250)
        self.assertEqual(ev.run_decisions(state, "imitation", 250), 250)
        state["phases"]["warmup"] = {"status": "skipped", "decisions": 0}
        self.assertEqual(ev.run_decisions(state, "ppo", 250), 5000250)
        state["phases"]["warmup"] = {"status": "running", "decisions": 7}
        self.assertIsNone(ev.run_decisions(state, "ppo", 250))
        self.assertIsNone(ev.run_decisions(None, "ppo", 250))
        self.assertIsNone(ev.run_decisions(state, "ppo_with_jepa", 250))

    def test_state_phase(self):
        self.assertEqual(ev.state_phase({"phase": "ppo"}), "ppo")
        for bad in (None, [], {"phase": 3}, {"phase": ""}, {}):
            self.assertIsNone(ev.state_phase(bad))


HOUR = 3600.0


class DecideTests(unittest.TestCase):
    """The watch loop's "should I evaluate now" decision. Time is a number, nothing is run."""

    NOW = 1_800_000_000.0

    def setUp(self):
        self.cfg, warnings = ev.eval_config(None)
        self.assertEqual(warnings, [])
        self.ck = ev.Ckpt("/r/checkpoints/learner_update_000050.pt", "learner_update_000050.pt", self.NOW - 60, 100)
        self.older = ev.Ckpt("/r/checkpoints/learner_update_000025.pt", "learner_update_000025.pt", self.NOW - 4 * HOUR, 100)

    def snap(self, results=(), newest="ck", phase="ppo", final_done=None, failures=None, n=None):
        newest = self.ck if newest == "ck" else newest
        return ev.Snapshot(now=self.NOW, phase=phase, newest=newest, n_checkpoints=(1 if newest else 0) if n is None else n,
                           results=tuple(results), final_done=final_done, failures=failures or {})

    def lines_for(self, ck, hours_ago, opponents=("random", "maxpower", "simple", "psbot"), **extra):
        when = ev.utc_iso(self.NOW - hours_ago * HOUR)
        extra.setdefault("state_phase", "ppo")
        return [line(o, t=when, name=ck.name, mtime=round(ck.mtime, 3), started=when, **extra) for o in opponents]

    def test_nothing_without_a_checkpoint(self):
        plan, why = ev.decide(self.snap(newest=None), self.cfg)
        self.assertIsNone(plan)
        self.assertEqual(why, "no checkpoint yet")
        plan, why = ev.decide(self.snap(newest=None, phase=None), self.cfg)      # no state.json either
        self.assertIsNone(plan)

    def test_waits_for_a_checkpoint_that_is_being_written(self):
        plan, why = ev.decide(self.snap(newest=None, n=3), self.cfg)
        self.assertIsNone(plan)
        self.assertIn("still being written", why)

    def test_first_evaluation_is_at_once_and_brings_foulplay(self):
        plan, _why = ev.decide(self.snap(), self.cfg)
        self.assertEqual((plan.action, plan.reason, plan.final), ("evaluate", "first", False))
        self.assertEqual(plan.opponents, ("random", "maxpower", "simple", "psbot", "foulplay"))
        self.assertIs(plan.checkpoint, self.ck)

    def test_no_state_file_still_evaluates(self):
        plan, _why = ev.decide(self.snap(phase=None), self.cfg)
        self.assertEqual(plan.reason, "first")

    def test_waits_for_the_interval(self):
        results = self.lines_for(self.older, hours_ago=2.9)
        plan, why = ev.decide(self.snap(results), self.cfg)
        self.assertIsNone(plan)
        self.assertIn("learner_update_000050.pt is next", why)
        results = self.lines_for(self.older, hours_ago=3.1)
        plan, _why = ev.decide(self.snap(results), self.cfg)
        self.assertEqual((plan.reason, plan.checkpoint.name), ("interval", "learner_update_000050.pt"))

    def test_an_evaluated_checkpoint_is_never_played_again(self):
        results = self.lines_for(self.ck, hours_ago=50)
        plan, why = ev.decide(self.snap(results), self.cfg)
        self.assertIsNone(plan)
        self.assertIn("waiting for a newer checkpoint", why)
        plan, _why = ev.decide(self.snap(results, phase="warmup"), self.cfg)      # not even on a phase change
        self.assertIsNone(plan)

    def test_phase_change_does_not_wait_for_the_interval(self):
        results = self.lines_for(self.older, hours_ago=0.2, state_phase="imitation")
        plan, _why = ev.decide(self.snap(results, phase="imitation"), self.cfg)
        self.assertIsNone(plan)
        plan, why = ev.decide(self.snap(results, phase="warmup"), self.cfg)
        self.assertEqual(plan.reason, "phase_change")
        self.assertIn("from imitation to warmup", why)

    def test_foulplay_has_its_own_pace(self):
        results = self.lines_for(self.older, hours_ago=3.5, opponents=("random", "maxpower", "simple", "psbot", "foulplay"))
        plan, _why = ev.decide(self.snap(results), self.cfg)
        self.assertEqual(plan.opponents, ("random", "maxpower", "simple", "psbot"))        # 3.5 h < 12 h
        results = self.lines_for(self.older, hours_ago=12.5, opponents=("random", "maxpower", "simple", "psbot", "foulplay"))
        plan, _why = ev.decide(self.snap(results), self.cfg)
        self.assertEqual(plan.opponents[-1], "foulplay")
        self.cfg["foulplay"]["games"] = 0
        plan, _why = ev.decide(self.snap(results), self.cfg)
        self.assertNotIn("foulplay", plan.opponents)

    def test_interrupted_evaluation_is_finished_without_replaying(self):
        results = self.lines_for(self.older, hours_ago=5) + self.lines_for(self.ck, hours_ago=0.01, opponents=("random", "maxpower"))
        plan, _why = ev.decide(self.snap(results), self.cfg)
        self.assertEqual((plan.reason, plan.opponents[:2]), ("resume", ("simple", "psbot")))

    def test_a_line_with_no_finished_game_is_not_a_result(self):
        when = ev.utc_iso(self.NOW - 60)
        dead = line("psbot", games=0, wins=0, errors=200, t=when, name=self.ck.name, mtime=round(self.ck.mtime, 3), started=when)
        results = self.lines_for(self.ck, hours_ago=0.02, opponents=("random", "maxpower", "simple")) + [dead]
        plan, _why = ev.decide(self.snap(results), self.cfg)
        self.assertEqual((plan.reason, [o for o in plan.opponents if o != "foulplay"]), ("resume", ["psbot"]))

    def test_same_name_from_another_phase_is_not_this_checkpoint(self):
        reused = ev.Ckpt(self.ck.path, self.ck.name, self.ck.mtime - 9 * HOUR, 100)     # update 50 of the phase before
        results = self.lines_for(reused, hours_ago=9)
        plan, _why = ev.decide(self.snap(results), self.cfg)
        self.assertEqual((plan.reason, len(plan.opponents)), ("interval", 5))

    def test_sampled_lines_do_not_count_for_a_best_action_run(self):
        results = self.lines_for(self.ck, hours_ago=5, sample=True)
        plan, _why = ev.decide(self.snap(results), self.cfg)
        self.assertEqual(plan.reason, "interval")
        self.cfg["sample"] = True
        plan, _why = ev.decide(self.snap(results), self.cfg)
        self.assertIsNone(plan)

    def test_done_evaluates_the_final_checkpoint_whatever_the_clock_says(self):
        results = self.lines_for(self.older, hours_ago=0.1, opponents=("random", "maxpower", "simple", "psbot", "foulplay"))
        plan, _why = ev.decide(self.snap(results, phase="done"), self.cfg)
        self.assertEqual((plan.action, plan.reason, plan.final), ("evaluate", "final", True))
        self.assertEqual(plan.opponents, ("random", "maxpower", "simple", "psbot", "foulplay"))

    def test_done_and_already_evaluated_only_writes_the_marker(self):
        results = self.lines_for(self.ck, hours_ago=0.1, opponents=("random", "maxpower", "simple", "psbot", "foulplay"))
        plan, _why = ev.decide(self.snap(results, phase="done"), self.cfg)
        self.assertEqual((plan.action, plan.final, plan.ok, plan.opponents), ("mark_final", True, True, ()))
        results = self.lines_for(self.ck, hours_ago=0.1)                          # no Foul Play line yet
        plan, _why = ev.decide(self.snap(results, phase="done"), self.cfg)
        self.assertEqual((plan.action, plan.opponents), ("evaluate", ("foulplay",)))

    def test_done_with_the_marker_written_is_finished(self):
        final = {"checkpoint": self.ck.name, "checkpoint_mtime": round(self.ck.mtime, 3), "ok": True}
        plan, why = ev.decide(self.snap(self.lines_for(self.ck, 0.1), phase="done", final_done=final), self.cfg)
        self.assertIsNone(plan)
        self.assertIn("final evaluation is written", why)
        stale = {"checkpoint": self.older.name, "checkpoint_mtime": round(self.older.mtime, 3), "ok": True}
        plan, _why = ev.decide(self.snap((), phase="done", final_done=stale), self.cfg)      # trained on after an earlier "done"
        self.assertEqual((plan.action, plan.reason), ("evaluate", "final"))

    def test_done_without_any_checkpoint_does_not_hang_the_run(self):
        plan, _why = ev.decide(self.snap(newest=None, phase="done"), self.cfg)
        self.assertEqual((plan.action, plan.final, plan.ok, plan.checkpoint), ("mark_final", True, False, None))
        plan, _why = ev.decide(self.snap(newest=None, phase="done", final_done={"checkpoint": None}), self.cfg)
        self.assertIsNone(plan)

    def test_failed_evaluation_is_retried_later_and_given_up(self):
        failures = {self.ck.key: (1, self.NOW - 5 * 60)}
        plan, why = ev.decide(self.snap(failures=failures), self.cfg)
        self.assertIsNone(plan)
        self.assertIn("trying again in 5 min", why)
        failures = {self.ck.key: (1, self.NOW - 11 * 60)}
        plan, _why = ev.decide(self.snap(failures=failures), self.cfg)
        self.assertEqual(plan.reason, "first")
        failures = {self.ck.key: (3, self.NOW - 11 * 60)}
        plan, why = ev.decide(self.snap(failures=failures), self.cfg)
        self.assertIsNone(plan)
        self.assertIn("gave up", why)
        plan, _why = ev.decide(self.snap(failures=failures, phase="done"), self.cfg)
        self.assertEqual((plan.action, plan.ok), ("mark_final", False))            # the run must not wait forever

    def test_an_evaluation_in_which_nothing_finished_is_retried_after_the_retry_wait(self):
        """Every match of an evaluation failed (the logins timed out). Its lines, with no game in
        them, were taken for "the last evaluation", and the next attempt came after every_hours."""
        earlier = self.lines_for(self.older, hours_ago=3.2)
        when = ev.utc_iso(self.NOW - 20 * 60)
        failed = [line(o, games=0, wins=0, errors=200, t=when, name=self.ck.name, mtime=round(self.ck.mtime, 3), started=when, state_phase="ppo")
                  for o in ("random", "maxpower", "simple", "psbot")]
        failures = {self.ck.key: (1, self.NOW - 20 * 60)}
        plan, why = ev.decide(self.snap(earlier + failed, failures=failures), self.cfg)        # retry_minutes is 10
        self.assertIsNotNone(plan, why)
        self.assertEqual((plan.action, plan.reason, plan.checkpoint.name), ("evaluate", "interval", "learner_update_000050.pt"))
        plan, why = ev.decide(self.snap(earlier + failed, failures={self.ck.key: (1, self.NOW - 5 * 60)}), self.cfg)
        self.assertIsNone(plan)                                                                 # the wait after a failure still holds
        self.assertIn("trying again in 5 min", why)
        # the watcher was restarted and no longer knows of the failure: the lines alone must not hold it back either
        plan, _why = ev.decide(self.snap(earlier + failed), self.cfg)
        self.assertEqual(plan.reason, "interval")
        # a failed evaluation that the change of phase had asked for is asked for again
        recent = self.lines_for(self.older, hours_ago=0.5, state_phase="imitation")
        failed_new_phase = [dict(r, state_phase="warmup") for r in failed]
        plan, _why = ev.decide(self.snap(recent + failed_new_phase, phase="warmup", failures=failures), self.cfg)
        self.assertEqual(plan.reason, "phase_change")
        # and with nothing measured yet at all it is still the first
        plan, _why = ev.decide(self.snap(failed, failures=failures), self.cfg)
        self.assertEqual(plan.reason, "first")

    def test_final_covers(self):
        self.assertTrue(ev.final_covers({"checkpoint": self.ck.name, "checkpoint_mtime": round(self.ck.mtime, 3)}, self.ck))
        self.assertFalse(ev.final_covers({"checkpoint": self.ck.name, "checkpoint_mtime": self.ck.mtime - 5}, self.ck))
        self.assertFalse(ev.final_covers({"checkpoint": self.ck.name}, self.ck))
        self.assertFalse(ev.final_covers(None, self.ck))
        self.assertTrue(ev.final_covers({"checkpoint": None}, None))


class ConfigTests(unittest.TestCase):
    def test_every_key_has_a_default(self):
        cfg, warnings = ev.eval_config(None)
        self.assertEqual(warnings, [])
        for key in ("every_hours", "games", "opponents", "port", "concurrency", "device", "foulplay"):
            self.assertIn(key, cfg)
        for key in ("games", "every_hours", "parallel", "search_time_ms"):
            self.assertIn(key, cfg["foulplay"])
        self.assertEqual(cfg["opponents"], ["random", "maxpower", "simple", "psbot"])
        self.assertEqual(ev.eval_config({}), (cfg, []))
        self.assertEqual(ev.eval_config({"eval": {}}), (cfg, []))
        self.assertIsNot(cfg["foulplay"], ev.EVAL_DEFAULTS["foulplay"])

    def test_values_from_the_file(self):
        raw = {"showdown_dir": "/home/ubuntu/pokemon-showdown", "python": "/home/ubuntu/venv/bin/python", "phases": {},
               "eval": {"every_hours": 2, "games": 400, "opponents": ["simple", "psbot", "foulplay"], "port": 9001,
                        "concurrency": 48, "device": "cuda", "foulplay": {"games": 60, "every_hours": 6, "parallel": 8, "search_time_ms": 100}}}
        cfg, warnings = ev.eval_config(raw)
        self.assertEqual(warnings, [])
        self.assertEqual((cfg["every_hours"], cfg["games"], cfg["port"], cfg["concurrency"], cfg["device"]), (2.0, 400, 9001, 48, "cuda"))
        self.assertEqual(cfg["opponents"], ["simple", "psbot"])
        self.assertEqual(cfg["foulplay"], {"games": 60, "every_hours": 6.0, "parallel": 8, "search_time_ms": 100})
        self.assertEqual((cfg["server_dir"], cfg["python"]), ("/home/ubuntu/pokemon-showdown", "/home/ubuntu/venv/bin/python"))

    def test_bad_values_keep_the_default_and_are_reported(self):
        raw = {"eval": {"every_hours": "soon", "games": 0, "opponents": ["simpel"], "port": 99999, "concurrency": 2.5,
                        "device": "tpu", "foulplay": {"games": -1, "parallel": 0, "extra": 1}, "typo": 1, "sample": "yes"}}
        cfg, warnings = ev.eval_config(raw)
        defaults, _ = ev.eval_config(None)
        self.assertEqual(cfg, defaults)
        text = "\n".join(warnings)
        for part in ("eval.every_hours='soon'", "eval.games=0", "unknown simpel", "names no known opponent", "eval.port=99999",
                     "eval.concurrency=2.5", "eval.device='tpu'", "eval.sample='yes'", "eval.foulplay.games=-1",
                     "eval.foulplay.parallel=0", "eval.foulplay.extra is not a known key", "eval.typo is not a known key"):
            self.assertIn(part, text)
        self.assertEqual(len(warnings), 12)
        cfg, warnings = ev.eval_config({"eval": []})
        self.assertEqual(cfg, defaults)
        self.assertEqual(len(warnings), 1)
        cfg, warnings = ev.eval_config([1, 2])
        self.assertEqual((cfg, len(warnings)), (defaults, 1))

    def test_command_line_wins_and_none_is_no_override(self):
        raw = {"eval": {"games": 400, "port": 9001, "foulplay": {"games": 60}}}
        cfg, warnings = ev.eval_config(raw, {"games": 20, "port": None, "opponents": "random,simple",
                                             "foulplay": {"games": None, "parallel": 2}})
        self.assertEqual(warnings, [])
        self.assertEqual((cfg["games"], cfg["port"], cfg["opponents"]), (20, 9001, ["random", "simple"]))
        self.assertEqual((cfg["foulplay"]["games"], cfg["foulplay"]["parallel"]), (60, 2))

    def test_foulplay_switches(self):
        cfg, _ = ev.eval_config({"eval": {"foulplay": False}})
        self.assertEqual(cfg["foulplay"]["games"], 0)
        cfg, _ = ev.eval_config({"eval": {"foulplay": {"games": 0}, "opponents": "simple,foulplay"}})
        self.assertEqual((cfg["opponents"], cfg["foulplay"]["games"]), (["simple"], 100))     # naming it turns it back on

    def test_parse_opponents(self):
        self.assertEqual(ev.parse_opponents("random, Simple,psbot,simple"), (["random", "simple", "psbot"], []))
        self.assertEqual(ev.parse_opponents(["foulplay", "x"]), (["foulplay"], ["x"]))
        self.assertEqual(ev.parse_opponents(None), ([], []))


class FakeServer:
    host, port = "127.0.0.1", 1

    def ensure(self):
        return True, "fake"

    def stop(self):
        pass


def make_ctx(tmp: str, module: str = "no_such_foulplay_module", ckpt: "ev.Ckpt" = None) -> "ev.EvalContext":
    return ev.EvalContext(
        host="127.0.0.1", port=1, battle_format="gen9randombattle", concurrency=8,
        checkpoint=ckpt or ev.Ckpt(os.path.join(tmp, "learner_update_000001.pt"), "learner_update_000001.pt", 1.0, 1),
        sample=False, device="cpu", batch=True, stall_seconds=90.0, foulplay_parallel=4, foulplay_search_ms=200,
        foulplay_module=module, foulplay_log_dir=os.path.join(tmp, "fp_logs"), server=FakeServer(),
    )


def fake_module(name: str, available, run_matches) -> str:
    """A module with bench/foulplay.py's two functions, importable under `name`."""
    mod = types.ModuleType(name)
    mod.available = available
    mod.run_matches = run_matches
    sys.modules[name] = mod
    return name


class FakeFoulplayTests(unittest.TestCase):
    """evaluate.py's side of the bench.foulplay contract, against a small module that honours it."""

    def setUp(self):
        self.tmp = new_dir()
        self.calls = []

    def test_missing_module_is_a_skip(self):
        tally, why = asyncio.run(ev.play_foulplay(make_ctx(self.tmp, "no_such_foulplay_module"), 10, make_subject=object))
        self.assertIsNone(tally)
        self.assertIn("cannot be imported", why)

    def test_not_available_is_a_skip_with_its_reason(self):
        async def never(*a, **k):
            raise AssertionError("run_matches must not be called")

        name = fake_module("fake_fp_unavailable", lambda: (False, "Foul Play is not installed: ~/foul-play does not exist"), never)
        tally, why = asyncio.run(ev.play_foulplay(make_ctx(self.tmp, name), 10, make_subject=object))
        self.assertIsNone(tally)
        self.assertEqual(why, "Foul Play is not installed: ~/foul-play does not exist")

    def test_module_without_the_functions_or_with_a_broken_available(self):
        sys.modules["fake_fp_empty"] = types.ModuleType("fake_fp_empty")
        self.assertIsNone(ev.load_foulplay("fake_fp_empty")[0])

        def broken():
            raise OSError("no")

        name = fake_module("fake_fp_broken", broken, None)
        mod, why = ev.load_foulplay(name)
        self.assertIsNone(mod)

    def test_the_real_module_answers_the_contract(self):
        mod, text = ev.load_foulplay(ev.FOULPLAY_MODULE)       # installed or not, this must not raise
        self.assertIsInstance(text, str)
        self.assertTrue(text)

    def test_result_becomes_a_line(self):
        subject = types.SimpleNamespace(battles={}, audit={"requests": 900, "invalid_choices": 1, "unavailable_choices": 4,
                                                           "other_errors": 0, "exceptions": 0, "forced_random": 0},
                                        counts={"decisions": 900, "fallbacks": 0}, audit_endings={}, audit_messages=[])

        async def run_matches(player, n_games, host, port, *, parallel=4, search_time_ms=200, log_dir=None):
            self.calls.append((player, n_games, host, port, parallel, search_time_ms, log_dir))
            return {"games": 28, "wins": 5, "losses": 22, "ties": 1, "errors": 2, "seconds": 412.5,
                    "foulplay_commit": "8f541f8", "log_agrees": True, "requested": 30}

        name = fake_module("fake_fp_ok", lambda: (True, "foul-play 8f541f8 at /home/ubuntu/foul-play"), run_matches)
        ctx = make_ctx(self.tmp, name)
        tally, text = asyncio.run(ev.play_foulplay(ctx, 30, make_subject=lambda: subject))
        self.assertEqual(text, "foul-play 8f541f8 at /home/ubuntu/foul-play")
        self.assertEqual(self.calls, [(subject, 30, "127.0.0.1", 1, 4, 200, ctx.foulplay_log_dir)])
        self.assertEqual((tally.games, tally.wins, tally.losses, tally.ties, tally.errors, tally.seconds), (28, 5, 22, 1, 2, 412.5))
        got = ev.result_line(ResultLineTests.META, "foulplay", tally, sample=False)
        self.assertEqual((got["opponent"], got["games"], got["wins"], got["losses"], got["ties"], got["errors"]),
                         ("foulplay", 28, 5, 22, 1, 2))
        self.assertAlmostEqual(got["winrate"], 5.5 / 28, places=6)
        self.assertEqual(got["opponent_info"]["foulplay_commit"], "8f541f8")
        self.assertEqual(got["opponent_info"]["foulplay"], "foul-play 8f541f8 at /home/ubuntu/foul-play")
        self.assertEqual((got["policy_decisions"], got["policy_invalid_orders"], got["policy_unavailable_orders"]), (900, 1, 4))
        self.assertIs(got["valid"], True)

    def test_parallel_is_never_more_than_the_games(self):
        async def run_matches(player, n_games, host, port, *, parallel=4, search_time_ms=200, log_dir=None):
            self.calls.append(parallel)
            return {"games": 2, "wins": 1, "losses": 1, "ties": 0, "errors": 0, "seconds": 1.0}

        name = fake_module("fake_fp_small", lambda: (True, "v"), run_matches)
        asyncio.run(ev.play_foulplay(make_ctx(self.tmp, name), 2, make_subject=lambda: types.SimpleNamespace()))
        self.assertEqual(self.calls, [2])

    def test_a_careless_result_is_checked(self):
        t = ev.foulplay_tally({"games": 30, "wins": 5, "losses": 10, "ties": 0, "errors": 0, "seconds": "fast", "error": "boom"}, 30)
        self.assertEqual((t.games, t.errors, t.seconds), (15, 15, 0.0))       # 15 games are unaccounted for: errors
        self.assertTrue(any("wins+losses+ties=15" in n for n in t.notes))
        self.assertTrue(any("boom" in n for n in t.notes))
        t = ev.foulplay_tally(None, 30)
        self.assertEqual((t.games, t.errors), (0, 30))
        t = ev.foulplay_tally({"wins": -3, "losses": True, "ties": float("nan")}, 4)
        self.assertEqual((t.wins, t.losses, t.ties, t.errors), (0, 0, 0, 4))

    def test_an_opponent_that_throws_is_recorded_and_the_rest_goes_on(self):
        async def run_matches(player, n_games, host, port, *, parallel=4, search_time_ms=200, log_dir=None):
            raise RuntimeError("foul play fell over")

        name = fake_module("fake_fp_raises", lambda: (True, "v"), run_matches)
        ckpt_path = write_file(os.path.join(self.tmp, "run", "checkpoints", "learner_update_000001.pt"))
        ckpt = ev.stat_ckpt(ckpt_path)
        paths = ev.RunPaths(os.path.join(self.tmp, "run"))
        ctx = make_ctx(self.tmp, name, ckpt)
        played = []

        async def play_builtin(_ctx, subject_kind, opponent, n):
            played.append(opponent)
            if opponent == "random":
                raise ConnectionError("the opponent blew up")
            return ev.Tally(requested=n, wins=n - 1, losses=1, turns=30 * n, seconds=2.0)

        real_builtin, real_subject = ev.play_builtin, ev._make_subject
        ev.play_builtin = play_builtin
        ev._make_subject = lambda *_a, **_k: types.SimpleNamespace()
        real_login, real_disconnect = ev._wait_logged_in, ev._disconnect

        async def nothing(*_a, **_k):
            return None

        ev._wait_logged_in = ev._disconnect = nothing
        try:
            args = argparse.Namespace(also_hybrid=False, games=10, foulplay_games=4)
            meta = ev.describe_checkpoint(ckpt, {"update": 1, "phase": "imitation", "total_steps": 5}, None, paths.metrics)
            lines, code = asyncio.run(ev._run_evaluation(args, paths, ctx, meta, ["random", "foulplay", "simple"], {"reason": "manual"}))
        finally:
            ev.play_builtin, ev._make_subject, ev._wait_logged_in, ev._disconnect = real_builtin, real_subject, real_login, real_disconnect
        self.assertEqual(code, ev.EXIT_OK)
        self.assertEqual(played, ["random", "simple"])
        on_disk = ev.read_jsonl(paths.results)
        self.assertEqual(on_disk, lines)
        self.assertEqual([r["opponent"] for r in on_disk], ["random", "foulplay", "simple"])
        self.assertEqual([(r["games"], r["errors"], r["valid"]) for r in on_disk], [(0, 10, False), (0, 4, False), (10, 0, True)])
        self.assertTrue(any("the opponent blew up" in n for n in on_disk[0]["notes"]))
        self.assertTrue(any("foul play fell over" in n for n in on_disk[1]["notes"]))
        self.assertEqual(ev.opponents_done(on_disk, ckpt), ["simple"])

    def test_a_replaced_checkpoint_stops_the_evaluation(self):
        ckpt_path = write_file(os.path.join(self.tmp, "run", "checkpoints", "learner_update_000001.pt"))
        ckpt = ev.stat_ckpt(ckpt_path)
        paths = ev.RunPaths(os.path.join(self.tmp, "run"))
        ctx = make_ctx(self.tmp, "no_such_foulplay_module", ckpt)

        replace_during = ["simple"]

        async def play_builtin(_ctx, subject_kind, opponent, n):
            if opponent in replace_during:
                os.utime(ckpt_path, (ckpt.mtime + 500, ckpt.mtime + 500))  # training wrote the same name again
            return ev.Tally(requested=n, wins=n)

        real = ev.play_builtin
        ev.play_builtin = play_builtin
        try:
            args = argparse.Namespace(also_hybrid=False, games=5, foulplay_games=5)
            meta = ev.describe_checkpoint(ckpt, {}, None, paths.metrics)
            lines, code = asyncio.run(ev._run_evaluation(args, paths, ctx, meta, ["random", "simple", "maxpower"], {}))
        finally:
            ev.play_builtin = real
        self.assertEqual(code, ev.EXIT_NO_CHECKPOINT)
        # The player reads the file itself, after the check that precedes its match: the games
        # against "simple" may have been played by the new weights, so that match is not written
        # down under the old checkpoint's name. What was written before it stands.
        self.assertEqual([r["opponent"] for r in lines], ["random"])
        self.assertEqual([r["opponent"] for r in ev.read_jsonl(paths.results)], ["random"])

    def test_a_checkpoint_replaced_during_the_first_match_leaves_no_line(self):
        ckpt_path = write_file(os.path.join(self.tmp, "run", "checkpoints", "learner_update_000001.pt"))
        ckpt = ev.stat_ckpt(ckpt_path)
        paths = ev.RunPaths(os.path.join(self.tmp, "run"))
        ctx = make_ctx(self.tmp, "no_such_foulplay_module", ckpt)

        async def play_builtin(_ctx, subject_kind, opponent, n):
            with open(ckpt_path, "ab") as f:                                # the same name, another file
                f.write(b"more")
            return ev.Tally(requested=n, wins=n)

        real = ev.play_builtin
        ev.play_builtin = play_builtin
        try:
            args = argparse.Namespace(also_hybrid=False, games=5, foulplay_games=5)
            meta = ev.describe_checkpoint(ckpt, {}, None, paths.metrics)
            lines, code = asyncio.run(ev._run_evaluation(args, paths, ctx, meta, ["random", "simple"], {}))
        finally:
            ev.play_builtin = real
        self.assertEqual((code, lines, ev.read_jsonl(paths.results)), (ev.EXIT_NO_CHECKPOINT, [], []))

    def test_foulplay_that_is_not_installed_leaves_a_line_that_says_so(self):
        """Skipped used to mean silent: no line, no section in the report, nothing in status.py."""
        ckpt_path = write_file(os.path.join(self.tmp, "run", "checkpoints", "learner_update_000001.pt"))
        ckpt = ev.stat_ckpt(ckpt_path)
        paths = ev.RunPaths(os.path.join(self.tmp, "run"))
        name = fake_module("fake_fp_absent", lambda: (False, "Foul Play is not installed: /home/ubuntu/foul-play does not exist"),
                           lambda *a, **k: None)
        ctx = make_ctx(self.tmp, name, ckpt)

        async def play_builtin(_ctx, subject_kind, opponent, n):
            return ev.Tally(requested=n, wins=n, turns=20 * n, seconds=1.0)

        real = ev.play_builtin
        ev.play_builtin = play_builtin
        try:
            args = argparse.Namespace(also_hybrid=False, games=5, foulplay_games=4)
            meta = ev.describe_checkpoint(ckpt, {"update": 1, "phase": "ppo", "total_steps": 5}, None, paths.metrics)
            lines, code = asyncio.run(ev._run_evaluation(args, paths, ctx, meta, ["random", "foulplay"], {"reason": "manual"}))
        finally:
            ev.play_builtin = real
        self.assertEqual(code, ev.EXIT_OK)
        on_disk = ev.read_jsonl(paths.results)
        self.assertEqual([(r["opponent"], r["games"], r["errors"], r["valid"]) for r in on_disk], [("random", 5, 0, True), ("foulplay", 0, 0, False)])
        self.assertTrue(any("foulplay was not played: Foul Play is not installed" in n for n in on_disk[1]["notes"]))
        self.assertEqual(ev.opponents_done(on_disk, ckpt), ["random"])           # not a result: it does not count as played
        report = ev.build_report(on_disk)
        self.assertIn("## foulplay", report)
        self.assertIn("| foulplay | - | - | 0 | no usable evaluation yet |", report)
        # ... and the report says why, in the row and under "Problems seen", not only that no game ran
        said = [ln for ln in report.splitlines() if "foulplay was not played: Foul Play is not installed" in ln]
        self.assertEqual(len(said), 2, report)
        self.assertTrue(said[0].startswith("- ") and said[1].startswith("| "))
        import status
        self.assertEqual(status.evaluation_skipped(on_disk), {"no_finished_game": 1, "not_valid": 0})
        self.assertNotIn("foulplay", status.evaluation_table(on_disk))

    def test_foulplays_settings_and_its_search_are_in_the_line(self):
        subject = types.SimpleNamespace(battles={}, audit={}, counts={}, audit_endings={}, audit_messages=[])
        answers = []

        async def run_matches(player, n_games, host, port, *, parallel=4, search_time_ms=200, log_dir=None):
            return dict(answers.pop(0))

        name = fake_module("fake_fp_settings", lambda: (True, "foul-play 8f541f8 at /x"), run_matches)
        ckpt_path = write_file(os.path.join(self.tmp, "run", "checkpoints", "learner_update_000001.pt"))
        ckpt = ev.stat_ckpt(ckpt_path)
        paths = ev.RunPaths(os.path.join(self.tmp, "run"))
        ctx = make_ctx(self.tmp, name, ckpt)
        base = {"games": 20, "wins": 4, "losses": 16, "ties": 0, "errors": 0, "seconds": 60.0, "foulplay_commit": "8f541f8",
                "search_time_ms": 200, "parallel": 4, "log_agrees": True, "requested": 20}
        answers += [dict(base, foulplay_mean_iterations=100000), dict(base, foulplay_mean_iterations=96000),
                    dict(base, foulplay_mean_iterations=41000, log_agrees=False)]
        real_subject, real_login, real_disconnect = ev._make_subject, ev._wait_logged_in, ev._disconnect

        async def nothing(*_a, **_k):
            return None

        ev._make_subject = lambda *_a, **_k: subject
        ev._wait_logged_in = ev._disconnect = nothing
        try:
            args = argparse.Namespace(also_hybrid=False, games=10, foulplay_games=20)
            meta = ev.describe_checkpoint(ckpt, {"update": 1, "phase": "ppo", "total_steps": 5}, None, paths.metrics)
            for _ in range(3):
                lines, code = asyncio.run(ev._run_evaluation(args, paths, ctx, meta, ["foulplay"], {"reason": "manual"}))
                self.assertEqual(code, ev.EXIT_OK)
        finally:
            ev._make_subject, ev._wait_logged_in, ev._disconnect = real_subject, real_login, real_disconnect
        first, second, third = ev.read_jsonl(paths.results)
        for r in (first, second, third):
            self.assertEqual((r["foulplay_search_time_ms"], r["foulplay_parallel"], r["foulplay_commit"]), (200, 4, "8f541f8"))
            self.assertEqual(ev.play_settings(r), (200, 4, "8f541f8"))
        self.assertEqual((first["foulplay_mean_iterations"], third["foulplay_mean_iterations"]), (100000, 41000))
        self.assertEqual((first["problems"], second["problems"]), ([], []))
        # 41,000 against a median of 98,000: the machine was busier and the opponent searched less than half as deep
        self.assertTrue(any("41,000 iterations a decision, less than half the median of 98,000" in p for p in third["problems"]), third["problems"])
        self.assertTrue(any("Foul Play's own log counts the games" in p for p in third["problems"]))
        table = "\n".join(ev._table_rows([first, third], foulplay=True))
        self.assertIn("| Search | Iterations |", table)
        self.assertIn("| 200 ms x4 | 100,000 |", table)
        self.assertIn("| 200 ms x4 | 41,000 |", table)
        self.assertNotIn("Iterations", "\n".join(ev._table_rows([first])))

    def test_gpu_memory_of_the_match_is_in_the_line_when_the_gpu_was_used(self):
        ckpt_path = write_file(os.path.join(self.tmp, "run", "checkpoints", "learner_update_000001.pt"))
        ckpt = ev.stat_ckpt(ckpt_path)
        paths = ev.RunPaths(os.path.join(self.tmp, "run"))
        ctx = make_ctx(self.tmp, "no_such_foulplay_module", ckpt)
        resets = []

        async def play_builtin(_ctx, subject_kind, opponent, n):
            tally = ev.Tally(requested=n, wins=n, turns=20 * n, seconds=1.0)
            tally.info["device"] = "cuda" if opponent == "random" else "cpu"   # the second match fell back to the CPU
            return tally

        real = ev.play_builtin, ev._gpu_peak_mb, ev._gpu_peak_reset
        ev.play_builtin, ev._gpu_peak_mb, ev._gpu_peak_reset = play_builtin, (lambda: 147.5), (lambda: resets.append(1))
        try:
            args = argparse.Namespace(also_hybrid=False, games=5, foulplay_games=5)
            meta = ev.describe_checkpoint(ckpt, {}, None, paths.metrics)
            lines, code = asyncio.run(ev._run_evaluation(args, paths, ctx, meta, ["random", "simple"], {}))
        finally:
            ev.play_builtin, ev._gpu_peak_mb, ev._gpu_peak_reset = real
        self.assertEqual(code, ev.EXIT_OK)
        self.assertEqual([(r["opponent"], r["device"], r["gpu_peak_mb"]) for r in lines], [("random", "cuda", 147.5), ("simple", "cpu", None)])
        self.assertEqual(len(resets), 2)                                     # the peak is per match, not since the process began
        # the real probes never raise, with or without a GPU and whether or not torch has touched it
        ev._gpu_peak_reset()
        peak = ev._gpu_peak_mb()
        self.assertTrue(peak is None or peak >= 0.0)


class WatchMatchTests(unittest.TestCase):
    """The backstop around a match: it ends on its own, or it is given up for a stated reason."""

    def setUp(self):
        import socket
        self.sock = socket.socket()                  # stands in for the evaluation server's port
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.ctx = make_ctx(new_dir())
        self.ctx.port = self.sock.getsockname()[1]
        self.cancelled = []

    def tearDown(self):
        self.sock.close()

    async def forever(self):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.append(True)
            raise

    def test_a_match_that_ends_returns_none_and_one_that_raises_raises(self):
        async def fine():
            return None

        async def broken():
            raise ConnectionError("lost")

        subject = types.SimpleNamespace(battles={})
        self.assertIsNone(asyncio.run(ev._watch_match(fine(), subject, self.ctx, "t", 1, 5.0, 60.0)))
        with self.assertRaises(ConnectionError):
            asyncio.run(ev._watch_match(broken(), subject, self.ctx, "t", 1, 5.0, 60.0))

    def test_a_match_that_stops_moving_is_given_up(self):
        subject = types.SimpleNamespace(battles={})
        t0 = time.time()
        reason = asyncio.run(ev._watch_match(self.forever(), subject, self.ctx, "t", 5, 1.5, 3600.0))
        self.assertEqual(reason, "no progress for 2s")
        self.assertLess(time.time() - t0, 15)
        self.assertEqual(self.cancelled, [True])

    def test_a_match_that_keeps_moving_is_only_stopped_by_the_time_limit(self):
        battle = types.SimpleNamespace(finished=False, turn=0)
        subject = types.SimpleNamespace(battles={"b": battle})

        async def moving():
            try:
                while True:
                    battle.turn += 1
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                self.cancelled.append(True)
                raise

        reason = asyncio.run(ev._watch_match(moving(), subject, self.ctx, "t", 5, 1.5, 4.0))
        self.assertEqual(reason, "time limit of 4s reached")
        self.assertEqual(self.cancelled, [True])

    def test_a_server_that_went_away_is_noticed(self):
        self.sock.close()
        subject = types.SimpleNamespace(battles={})
        t0 = time.time()
        reason = asyncio.run(ev._watch_match(self.forever(), subject, self.ctx, "t", 5, 600.0, 3600.0))
        self.assertEqual(reason, ev.SERVER_GONE)
        self.assertLess(time.time() - t0, 30)


class SubjectTests(unittest.TestCase):
    """Building the player that is measured: it must hold the checkpoint's network, on some device."""

    def setUp(self):
        try:
            import numpy  # noqa: F401

            import play
        except ImportError as e:
            self.skipTest(f"numpy or poke-env is not installed: {e}")
        self.play = play
        self.real = play.build_player
        self.tmp = new_dir()
        self.ckpt = ev.stat_ckpt(write_file(os.path.join(self.tmp, "learner_update_000001.pt")))
        self.ctx = make_ctx(self.tmp, ckpt=self.ckpt)
        self.ctx.device = "cuda"
        self.calls = []

    def tearDown(self):
        self.play.build_player = self.real

    def player(self, device, path=None, fail_forward=False, names=None, buffers=()):
        def logits(tag, obs):
            if fail_forward:
                raise RuntimeError("CUDA error: out of memory")

        policy = types.SimpleNamespace(path=path or self.ckpt.path, device=device, logits=logits, forget=lambda tag: None,
                                       assembler=types.SimpleNamespace(offsets={"action_mask": (0, 14)}))
        if names is not None:
            policy.net = types.SimpleNamespace(state_dict=lambda: {n: None for n in list(names) + list(buffers)},
                                               named_parameters=lambda: [(n, None) for n in names])
        return types.SimpleNamespace(policy=policy)

    NETWORK = ("field_emb.weight", "layers.0.attn.qkv.weight", "pi_head.weight", "pi_head.bias", "readout.weight")

    def test_a_checkpoint_whose_tensor_names_do_not_fit_the_network_is_refused(self):
        """hybrid.PolicyNet loads with strict=False: a file whose names do not match leaves the
        network untrained, and it was played and written down under the checkpoint's name."""
        self.play.build_player = lambda kind, **kw: self.player("cpu", names=self.NETWORK)
        self.ctx.device = "cpu"
        self.ctx.tensor_names = frozenset(self.NETWORK)                       # the file has exactly the network's tensors
        self.assertIsNotNone(ev._make_subject(self.ctx, "policy", 8))
        self.ctx.tensor_names = frozenset(self.NETWORK) | {"attn_mask", "layers.0.attn_mask"}   # learner.py drops these too
        self.assertIsNotNone(ev._make_subject(self.ctx, "policy", 8))
        self.ctx.tensor_names = frozenset("_orig_mod." + n for n in self.NETWORK)   # written by a compiled or wrapped module
        with self.assertRaises(RuntimeError) as caught:
            ev._make_subject(self.ctx, "policy", 8)
        self.assertIn("does not fit the network", str(caught.exception))
        self.assertIn("5 of the network's 5 learned tensors are not in the file", str(caught.exception))
        self.ctx.tensor_names = frozenset(n for n in self.NETWORK if not n.startswith(("pi_head", "readout")))   # the model gained layers since
        with self.assertRaises(RuntimeError) as caught:
            ev._make_subject(self.ctx, "policy", 8)
        self.assertIn("3 of the network's 5 learned tensors are not in the file", str(caught.exception))
        self.ctx.tensor_names = frozenset(self.NETWORK) | {"old_head.weight"}  # the model lost one
        with self.assertRaises(RuntimeError) as caught:
            ev._make_subject(self.ctx, "policy", 8)
        self.assertIn("1 of the file's 6 tensors have no place in the network", str(caught.exception))
        # A buffer the network builds for itself (the value head's support) and the file does not
        # carry is not untrained weights: played, with a note. The same name as a parameter is refused.
        self.play.build_player = lambda kind, **kw: self.player("cpu", names=self.NETWORK, buffers=("v_support",))
        self.ctx.tensor_names = frozenset(self.NETWORK)
        self.assertIsNotNone(ev._make_subject(self.ctx, "policy", 8))
        self.assertTrue(any("1 buffer(s) of the network are not in learner_update_000001.pt" in n and "v_support" in n for n in self.ctx.notes))
        self.play.build_player = lambda kind, **kw: self.player("cpu", names=self.NETWORK + ("v_support",))
        with self.assertRaises(RuntimeError):
            ev._make_subject(self.ctx, "policy", 8)

    def test_the_loaders_own_warning_about_tensors_is_enough_to_refuse(self):
        import logging

        def build(kind, **kw):
            logging.getLogger("hybrid").warning("Checkpoint %s: %d missing and %d unexpected tensors", self.ckpt.path, 98, 98)
            return self.player("cpu")

        self.play.build_player = build
        self.ctx.device = "cpu"
        self.assertIsNone(self.ctx.tensor_names)                             # nothing else to go on
        with self.assertRaises(RuntimeError) as caught:
            ev._make_subject(self.ctx, "policy", 8)
        self.assertIn("98 missing and 98 unexpected tensors", str(caught.exception))

    def test_weights_mismatch_in_words(self):
        self.assertIsNone(ev.weights_mismatch(["a", "b"], ["b", "a"]))
        self.assertIsNone(ev.weights_mismatch(["a", "x.attn_mask"], ["a"]))
        self.assertIn("1 of the network's 2 learned tensors", ev.weights_mismatch(["a", "b"], ["a"]))
        self.assertIn("(for example c)", ev.weights_mismatch(["a"], ["a", "c"]))
        self.assertIsNone(ev.weights_mismatch(["a", "buf"], ["a"], learned_names=["a"]))            # only a buffer is missing
        self.assertIn("1 of the network's 1 learned tensors", ev.weights_mismatch(["a", "buf"], ["buf"], learned_names=["a"]))

    def test_play_py_leaves_out_games_the_server_ended_and_builds_audited_players(self):
        """play.py took poke-env's own counters, in which a forfeited or timed-out game is a win or
        a loss, and built its players without the audit, so one exception hung the command."""
        def battle(finished, won=None, lost=None, turn=20):
            return types.SimpleNamespace(finished=finished, won=won, lost=lost, turn=turn)

        player = types.SimpleNamespace(
            battles={"b1": battle(True, won=True, turn=30), "b2": battle(True, lost=True, turn=10), "b3": battle(True, turn=40),
                     "b4": battle(True, won=True), "b5": battle(True, lost=True), "b6": battle(False)},
            audit_endings={"b4": "inactivity", "b5": "forfeit"})
        self.assertEqual(self.play.count_results(player), (1, 1, 1, 2, [30, 10, 40]))
        del player.audit_endings                                             # a player without the audit: what finished counts
        self.assertEqual(self.play.count_results(player)[:4], (2, 2, 1, 0))
        seen = {}
        self.play.build_player = lambda kind, **kw: seen.update(kw) or "player"
        args = argparse.Namespace(format="gen9randombattle", concurrency=4, checkpoint=None, beta=4.0, veto_gap=0.75, policy_weight=1.0,
                                  sample=False, device=None, random_init=False)
        self.assertEqual(self.play.make_player("random", args, "server", "x"), "player")
        self.assertIs(seen["audit"], True)

    def test_cuda_that_cannot_load_falls_back_to_cpu(self):
        def build(kind, **kw):
            self.calls.append(kw["device"])
            if kw["device"] != "cpu":
                raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
            return self.player("cpu")

        self.play.build_player = build
        player = ev._make_subject(self.ctx, "policy", 8)
        self.assertEqual((self.calls, self.ctx.device, player.policy.device), (["cuda", "cpu"], "cpu", "cpu"))
        self.assertTrue(any("cuda failed" in n for n in self.ctx.notes))

    def test_cuda_that_fails_on_the_first_forward_pass_falls_back_too(self):
        def build(kind, **kw):
            self.calls.append(kw["device"])
            return self.player(kw["device"], fail_forward=kw["device"] != "cpu")

        self.play.build_player = build
        ev._make_subject(self.ctx, "policy", 8)
        self.assertEqual((self.calls, self.ctx.device), (["cuda", "cpu"], "cpu"))

    def test_another_failure_is_not_retried_on_cpu(self):
        def build(kind, **kw):
            self.calls.append(kw["device"])
            raise FileNotFoundError("checkpoint not found")

        self.play.build_player = build
        with self.assertRaises(FileNotFoundError):
            ev._make_subject(self.ctx, "policy", 8)
        self.assertEqual(self.calls, ["cuda"])

    def test_a_player_without_the_checkpoints_network_is_refused(self):
        self.play.build_player = lambda kind, **kw: types.SimpleNamespace(policy=None)       # HybridPlayer as PSBot alone
        with self.assertRaises(RuntimeError):
            ev._make_subject(self.ctx, "policy", 8)
        other = write_file(os.path.join(self.tmp, "learner_update_000002.pt"))
        self.play.build_player = lambda kind, **kw: self.player("cpu", path=other)            # some other file's weights
        with self.assertRaises(RuntimeError):
            ev._make_subject(self.ctx, "policy", 8)


class VerdictTests(unittest.TestCase):
    def test_clearly_better_and_worse(self):
        old, new = line(wins=60), line(wins=140)
        got = ev.compare(old, new)
        self.assertEqual((got["verdict"], got["overlap"], got["delta"]), ("better", False, 40.0))
        self.assertLess(got["p"], 0.001)
        self.assertEqual(ev.compare(new, old)["verdict"], "worse")

    def test_noise_is_no_clear_change(self):
        got = ev.compare(line(wins=100), line(wins=108))      # 50% to 54% on 200 games
        self.assertEqual((got["verdict"], got["overlap"], got["clear"]), ("no clear change", True, False))
        self.assertGreater(got["p"], 0.05)
        self.assertEqual(ev.compare(line(wins=100), line(wins=100))["verdict"], "no clear change")
        self.assertEqual(ev.compare(line(wins=200), line(wins=200))["verdict"], "no clear change")

    def test_overlapping_intervals_but_a_significant_test_is_clear(self):
        old, new = line(wins=100), line(wins=126)             # 50% to 63%: intervals overlap, p = 0.009
        self.assertTrue(new["ci_low"] < old["ci_high"])
        got = ev.compare(old, new)
        self.assertEqual((got["verdict"], got["overlap"]), ("better", True))
        self.assertLess(got["p"], ev.VERDICT_P)
        got = ev.compare(old, line(wins=125))                 # 62.5%: p = 0.012
        self.assertEqual(got["verdict"], "no clear change")

    def test_a_verdict_needs_p_below_one_in_a_hundred(self):
        """At p < 0.05, with two comparisons for each of four opponents, about one report in three
        of a policy that never changes carried a verdict. 50% to 60% on 200 games (p = 0.044) was
        "better"; it is not any more."""
        self.assertEqual(ev.VERDICT_P, 0.01)
        got = ev.compare(line(wins=100), line(wins=120))
        self.assertTrue(0.01 < got["p"] < 0.05)
        self.assertEqual((got["verdict"], got["clear"]), ("no clear change", False))
        self.assertEqual(ev.compare(line(wins=120), line(wins=100))["verdict"], "no clear change")
        # what a difference has to reach before it is called, as the report states it
        self.assertAlmostEqual(ev.clear_points(200), 12.9, places=1)
        self.assertAlmostEqual(ev.clear_points(600), 7.4, places=1)
        self.assertEqual(ev.clear_points(0), 100.0)

    def test_the_last_three_together_against_the_three_before(self):
        """The comparison that still sees a slow gain: 50% three times, then 58% three times."""
        rows = [line(wins=w, t=f"2026-10-05T{3 * i:02d}:00:00Z", name=f"learner_update_{25 * (i + 1):06d}.pt")
                for i, w in enumerate((100, 100, 100, 116, 116, 116))]
        v = ev.opponent_verdict(rows)
        self.assertEqual((v["vs_previous"]["verdict"], v["vs_first"]["verdict"]), ("no clear change", "no clear change"))   # 200 games a side
        self.assertEqual((v["trend"]["verdict"], v["trend"]["delta"]), ("better", 8.0))                                     # 600 a side
        self.assertLess(v["trend"]["p"], ev.VERDICT_P)
        self.assertIsNone(ev.opponent_verdict(rows[:5])["trend"])               # fewer than twice three: no trend yet
        pooled = ev.pool(rows[:3] + [line(wins=90, ties=20)])
        self.assertEqual((pooled["games"], pooled["lines"]), (800, 4))
        self.assertAlmostEqual(pooled["winrate"], (300 + 90 + 10) / 800)
        text = ev.build_report(rows)
        row = [ln for ln in text.splitlines() if ln.startswith("| simple |")][0]
        self.assertIn("| better (+8.0 points, p = 0.005) |", row)
        self.assertIn("This table makes 3 comparisons.", text)
        self.assertIn("at most about 3 reports in 100", text)
        self.assertIn("two rows of 200 games must differ by about 13 points", text)
        self.assertIn("where about 7 points are enough", text)

    def test_now_is_the_most_trained_checkpoint_not_the_last_one_measured(self):
        """The owner measures the end-of-imitation copy by hand, days into PPO, to have the baseline.
        Ordered by the time of measuring, that old checkpoint was "now" and every verdict "worse"."""
        def row(name, phase, update, wins, measured, written):
            meta = {"checkpoint": name, "update": update, "phase": phase, "total_steps": update * 36864, "checkpoint_mtime": written}
            return ev.result_line(meta, "psbot", ev.Tally(requested=200, wins=wins, losses=200 - wins), sample=False,
                                  now=ev.parse_iso(measured))

        early = row("learner_update_000500.pt", "ppo", 500, 60, "2026-10-06T00:00:00Z", 2000.0)       # 30%
        late = row("learner_update_002000.pt", "ppo", 2000, 110, "2026-10-07T00:00:00Z", 9000.0)      # 55%: the newest policy
        baseline = row("imitation_learner_update_000108.pt", "imitation", 108, 30, "2026-10-07T06:00:00Z", 1000.0)   # 15%, measured last
        rows = [early, late, baseline]
        self.assertEqual(sorted(rows, key=ev.training_order), [baseline, early, late])
        v = ev.opponent_verdict(rows)
        self.assertIs(v["latest"], late)
        self.assertIs(v["previous"], early)
        self.assertIs(v["first"], baseline)
        self.assertEqual((v["vs_previous"]["verdict"], v["vs_first"]["verdict"]), ("better", "better"))
        text = ev.build_report(rows)
        self.assertIn("Most trained checkpoint evaluated: `learner_update_002000.pt`, update 2000 (ppo)", text)
        self.assertIn("| psbot | 55.0% |", text)
        self.assertNotIn("worse", [ln for ln in text.splitlines() if ln.startswith("| psbot |")][0])
        table = [ln for ln in text.split("## psbot")[1].splitlines() if ln.startswith("| 2026-10-0")]
        self.assertEqual([ln.split("|")[3].strip() for ln in table], ["108", "500", "2000"])     # the Update column, in order of training
        # the same phase and count twice (a name written again): the file's own time decides, then the time of measuring
        again = row("learner_update_002000.pt", "ppo", 2000, 100, "2026-10-06T12:00:00Z", 9500.0)
        self.assertIs(ev.opponent_verdict([late, again])["latest"], again)
        import status
        shown = status.evaluation_table(rows)["psbot"]
        self.assertEqual([(r["phase"], r["update"]) for r in shown], [("imitation", 108), ("ppo", 500), ("ppo", 2000)])
        self.assertEqual((shown[-1]["delta_prev"], shown[-1]["delta_first"]), (25.0, 40.0))

    def test_foulplay_lines_played_with_other_settings_are_not_compared(self):
        """Foul Play's strength is its search time. A line at 1000 ms and one at 50 ms measured the
        policy against two different opponents: 10% then 24% was called "better"."""
        def fp(wins, t, ms, iterations, **extra):
            tally = ev.Tally(requested=100, wins=wins, losses=100 - wins)
            tally.info = dict({"search_time_ms": ms, "parallel": 4, "foulplay_commit": "8f541f8", "foulplay_mean_iterations": iterations}, **extra)
            meta = {"checkpoint": "learner_update_000010.pt", "update": 10, "phase": "ppo", "total_steps": 1000, "checkpoint_mtime": 1000.0}
            return ev.result_line(meta, "foulplay", tally, sample=False, now=ev.parse_iso(t))

        slow, fast = fp(10, "2026-10-05T00:00:00Z", 1000, 470000), fp(24, "2026-10-05T06:00:00Z", 50, 9000)
        self.assertLess(ev.compare(slow, fast)["p"], ev.VERDICT_P)              # the numbers alone would say "better"
        v = ev.opponent_verdict([slow, fast])
        self.assertIs(v["latest"], fast)
        self.assertEqual((v["previous"], v["vs_previous"], v["vs_first"], v["other_settings"]), (None, None, None, 1))
        text = ev.build_report([slow, fast])
        self.assertIn("1 other evaluation against it was played another way (Foul Play's search time", text)
        same = fp(24, "2026-10-05T12:00:00Z", 1000, 465000)
        v = ev.opponent_verdict([slow, fast, same])
        self.assertIs(v["previous"], slow)
        self.assertEqual((v["vs_previous"]["verdict"], v["other_settings"]), ("better", 1))
        # another commit or another number of processes is another opponent too; a line written before
        # these keys were copied to the top is read from its opponent_info
        self.assertNotEqual(ev.play_settings(fp(1, "2026-10-05T00:00:00Z", 1000, 1, foulplay_commit="abc")), ev.play_settings(slow))
        self.assertNotEqual(ev.play_settings(fp(1, "2026-10-05T00:00:00Z", 1000, 1, parallel=8)), ev.play_settings(slow))
        old_style = {k: v2 for k, v2 in slow.items() if not k.startswith("foulplay_")}
        self.assertEqual(ev.play_settings(old_style), ev.play_settings(slow))
        self.assertEqual(ev.play_settings(line("simple")), ())
        # the search: nothing to say without earlier matches, or at another search time
        self.assertIsNone(ev.foulplay_slow_search(9000, 50, [slow]))
        self.assertIsNone(ev.foulplay_slow_search(240000, 1000, [slow]))
        self.assertIn("less than half the median of 470,000", ev.foulplay_slow_search(200000, 1000, [slow, fast]))
        self.assertIsNone(ev.foulplay_slow_search(None, 1000, [slow]))

    def test_no_games_never_decides(self):
        dead = line(games=0, wins=0, errors=200)
        self.assertEqual(ev.compare(dead, line(wins=200))["verdict"], "no clear change")
        self.assertEqual(ev.compare(line(wins=0), dead)["verdict"], "no clear change")

    def test_latest_previous_and_first(self):
        rows = [line(wins=40, t="2026-10-05T00:00:00Z", name="learner_update_000010.pt"),
                line(wins=118, t="2026-10-05T03:00:00Z", name="learner_update_000020.pt"),
                line(wins=124, t="2026-10-05T06:00:00Z", name="learner_update_000030.pt")]
        v = ev.opponent_verdict(rows)
        self.assertIs(v["latest"], rows[2])
        self.assertIs(v["previous"], rows[1])
        self.assertIs(v["first"], rows[0])
        self.assertEqual((v["vs_previous"]["verdict"], v["vs_first"]["verdict"]), ("no clear change", "better"))

    def test_one_row_has_nothing_to_compare(self):
        v = ev.opponent_verdict([line(wins=40)])
        self.assertIsNone(v["vs_previous"])
        self.assertIsNone(v["vs_first"])
        self.assertIsNone(ev.opponent_verdict([])["latest"])

    def test_untrusted_rows_and_other_play_modes_are_left_out(self):
        good_old = line(wins=60, t="2026-10-05T00:00:00Z")
        good_new = line(wins=140, t="2026-10-05T03:00:00Z")
        dead = line(games=0, wins=0, errors=200, t="2026-10-05T06:00:00Z")
        sampled = line(wins=20, sample=True, t="2026-10-05T01:00:00Z")
        v = ev.opponent_verdict([good_old, sampled, good_new, dead])
        self.assertIs(v["latest"], good_new)
        self.assertIs(v["previous"], good_old)
        self.assertEqual(v["vs_previous"]["verdict"], "better")

    def test_report_text(self):
        rows = []
        for hour, (simple, psbot) in enumerate(((20, 2), (150, 60), (156, 64))):
            t = f"2026-10-05T{3 * hour:02d}:00:00Z"
            name = f"learner_update_{25 * (hour + 1):06d}.pt"
            rows += [line("random", wins=196, t=t, name=name), line("simple", wins=simple, t=t, name=name, phase="imitation"),
                     line("psbot", wins=psbot, t=t, name=name)]
        rows.append(line("psbot", games=0, wins=0, errors=200, t="2026-10-05T09:00:00Z", name="learner_update_000100.pt",
                         notes=["failed: PSBotEngineError"]))
        text = ev.build_report(rows, run_dir="/home/ubuntu/runs/main", now=ev.parse_iso("2026-10-05T09:30:00Z"))
        text.encode("ascii")                                    # plain ASCII, whatever the console is
        self.assertIn("# Evaluation report", text)
        self.assertIn("10 result lines for 4 checkpoints", text)
        self.assertIn("| simple | 78.0% |", text)
        simple_row = [ln for ln in text.splitlines() if ln.startswith("| simple |")][0]
        self.assertIn("no clear change (+3.0 points", simple_row)      # 75% to 78%
        self.assertIn("better (+68.0 points, p < 0.001)", simple_row)  # 10% to 78%
        self.assertIn("- **simple**: 78.0% at update 75 (imitation).", text)
        self.assertIn("- **random**: 98.0% at update 75 (ppo). The checkpoint before it, update 50 (ppo): 98.0%. No clear change", text)
        self.assertIn("The line of the most trained checkpoint (update 100 (ppo)) is not used: no game ran to its end", text)
        self.assertIn("p < 0.01", text)
        self.assertNotIn("p < 0.05", text)
        self.assertIn("plus or minus 7 points", text)
        self.assertIn("## Problems seen", text)
        for name in ("random", "simple", "psbot"):
            self.assertIn(f"## {name}", text)
        table = text.split("## psbot")[1]
        self.assertEqual(len([ln for ln in table.splitlines() if ln.startswith("| 2026-10-05")]), 4)
        self.assertIn("NOT TRUSTED", table)

    def test_report_with_nothing_and_with_junk(self):
        self.assertIn("No evaluation has finished yet", ev.build_report([]))
        text = ev.build_report([{"opponent": "simple"}, {"junk": 1}, line("simple", wins=10)])
        self.assertIn("1 result line for 1 checkpoint.", text)

    def test_report_file_is_rewritten(self):
        paths = ev.RunPaths(os.path.join(new_dir(), "run"))
        ev.append_jsonl(paths.results, line("simple", wins=10))
        ev.write_report(paths)
        with open(paths.report, encoding="utf-8") as f:
            first = f.read()
        ev.append_jsonl(paths.results, line("simple", wins=150, t="2026-10-05T03:00:00Z", name="learner_update_000020.pt"))
        ev.write_report(paths)
        with open(paths.report, encoding="utf-8") as f:
            second = f.read()
        self.assertNotEqual(first, second)
        self.assertIn("better (+70.0 points", second)


class SmallThingsTests(unittest.TestCase):
    def test_duration(self):
        self.assertEqual([ev.duration(s) for s in (0, 45, 89, 90, 720, 5399, 5400, 3 * 3600, -5)],
                         ["0 s", "45 s", "89 s", "2 min", "12 min", "90 min", "1.5 h", "3.0 h", "0 s"])

    def test_time_round_trip(self):
        self.assertEqual(ev.utc_iso(0), "1970-01-01T00:00:00Z")
        self.assertEqual(ev.parse_iso("1970-01-01T00:00:10Z"), 10.0)
        self.assertEqual(ev.parse_iso("1970-01-01T05:30:10+05:30"), 10.0)
        self.assertEqual(ev.parse_iso("1970-01-01T00:00:10"), 10.0)
        for bad in (None, "", "yesterday", 5):
            self.assertIsNone(ev.parse_iso(bad))

    def test_json_files_that_are_missing_or_broken(self):
        d = new_dir()
        self.assertIsNone(ev.read_json(os.path.join(d, "state.json")))
        with open(os.path.join(d, "state.json"), "w") as f:
            f.write('{"phase": "pp')
        self.assertIsNone(ev.read_json(os.path.join(d, "state.json")))
        self.assertEqual(ev.read_jsonl(os.path.join(d, "none.jsonl")), [])
        ev.write_json_atomic(os.path.join(d, "eval", "final.done"), {"ok": True})
        ev.write_json_atomic(os.path.join(d, "eval", "final.done"), {"ok": False})
        self.assertEqual(ev.read_json(os.path.join(d, "eval", "final.done")), {"ok": False})
        self.assertEqual(os.listdir(os.path.join(d, "eval")), ["final.done"])

    def test_no_server_and_no_directory_is_a_clear_no(self):
        import socket
        with socket.socket() as s:                    # a port nothing listens on
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = ev.EvalServer(None, port, new_dir())
        ok, why = server.ensure()
        self.assertFalse(ok)
        self.assertIn("no server directory", why)
        ok, why = ev.EvalServer(new_dir(), port, new_dir()).ensure()
        self.assertFalse(ok)
        self.assertIn("not a Pokemon Showdown checkout", why)
        server.stop()                                 # stops nothing: it started nothing

    def test_a_listening_port_is_used_and_never_owned(self):
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            server = ev.EvalServer(None, s.getsockname()[1], new_dir())
            ok, _why = server.ensure()
            self.assertTrue(ok)
            self.assertFalse(server.owned)
            self.assertIsNone(server.proc)

    def test_pid_alive(self):
        self.assertTrue(ev.pid_alive(os.getpid()))
        self.assertFalse(ev.pid_alive(None))
        self.assertFalse(ev.pid_alive(-1))

    def test_watch_command_line(self):
        args = ev.build_parser().parse_args(["--run-dir", "/r", "--watch", "--games", "20", "--foulplay-games", "0"])
        w = ev.Watcher(args)
        cfg = w.config()
        self.assertEqual((cfg["games"], cfg["foulplay"]["games"], cfg["port"]), (20, 0, 9000))
        ck = ev.Ckpt("/r/checkpoints/learner_update_000050.pt", "learner_update_000050.pt", 123.456789, 1)
        snap = ev.Snapshot(now=1000.0, phase="ppo", newest=ck, n_checkpoints=1, results=(), final_done=None)
        cfg["server_dir"] = os.path.join("some", "relative", "showdown")
        cmd = w.command(ev.Plan("evaluate", "interval", ck, ("simple", "psbot")), cfg, snap)
        self.assertEqual(cmd[1], os.path.join(REPO_ROOT, "evaluate.py"))
        child = ev.build_parser().parse_args(cmd[2:])
        self.assertTrue(os.path.isabs(child.server_dir))       # the evaluation runs from another folder
        self.assertTrue(os.path.isabs(cmd[0]))
        self.assertEqual((child.checkpoint, child.expect_mtime, child.opponents, child.games), (ck.path, ck.mtime, "simple,psbot", 20))
        self.assertEqual((child.reason, child.state_phase, child.started, child.port), ("interval", "ppo", "1970-01-01T00:16:40Z", 9000))
        self.assertFalse(child.watch)


FAKE_SERVER = r'''
"""A stand-in for `node pokemon-showdown start ... --port P`: listens on P and has a child of its own."""
import json, os, socket, subprocess, sys, time
if os.path.exists("die"):
    sys.exit(3)
port = [int(a) for a in sys.argv if a.isdigit()][-1]
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port))
s.listen(16)
with open("pids.json.tmp", "w") as f:
    json.dump({"server": os.getpid(), "child": child.pid, "argv": sys.argv[1:]}, f)
os.replace("pids.json.tmp", "pids.json")
while True:
    conn, _addr = s.accept()
    conn.close()
'''


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(condition, seconds: float = 10.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return bool(condition())


class ProcessTests(unittest.TestCase):
    """
    Starting and stopping the evaluation server, with a small Python program in the place of
    node and Showdown. Every process here is started by the test and stopped by its own PID.
    """

    def setUp(self):
        self.dir = new_dir()
        self.checkout = os.path.join(self.dir, "showdown")
        os.makedirs(os.path.join(self.checkout, "dist"))
        with open(os.path.join(self.checkout, "pokemon-showdown"), "w") as f:
            f.write("// stands in for the Showdown launcher\n")
        script = os.path.join(self.dir, "fake_server.py")
        with open(script, "w") as f:
            f.write(FAKE_SERVER)
        if os.name == "nt":
            self.node = os.path.join(self.dir, "fake_node.cmd")
            with open(self.node, "w") as f:
                f.write(f'@"{sys.executable}" "{script}" %*\n')
        else:
            self.node = os.path.join(self.dir, "fake_node")
            with open(self.node, "w") as f:
                f.write(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
            os.chmod(self.node, 0o755)
        self.saved = os.environ.get("PSBOT_NODE")
        os.environ["PSBOT_NODE"] = self.node
        self.servers = []

    def tearDown(self):
        for server in self.servers:
            server.stop()
        if self.saved is None:
            os.environ.pop("PSBOT_NODE", None)
        else:
            os.environ["PSBOT_NODE"] = self.saved

    def server(self, port: int) -> "ev.EvalServer":
        s = ev.EvalServer(self.checkout, port, os.path.join(self.dir, "logs"))
        self.servers.append(s)
        return s

    def pids(self):
        self.assertTrue(wait_until(lambda: os.path.exists(os.path.join(self.checkout, "pids.json"))))
        with open(os.path.join(self.checkout, "pids.json")) as f:
            return json.load(f)

    def test_starts_one_server_and_stops_it_and_what_it_started(self):
        port = free_port()
        server = self.server(port)
        ok, why = server.ensure(wait_s=30)
        self.assertTrue(ok, why)
        self.assertIn(f"started a server on 127.0.0.1:{port}", why)
        self.assertTrue(server.owned)
        self.assertTrue(ev.port_listening("127.0.0.1", port))
        pids = self.pids()
        self.assertEqual(pids["argv"][-5:], ["start", "--no-security", "--skip-build", "--port", str(port)])
        self.assertTrue(ev.pid_alive(pids["server"]) and ev.pid_alive(pids["child"]))
        ok, why = server.ensure(wait_s=30)                 # already up: nothing more is started
        self.assertTrue(ok)
        self.assertEqual(server.starts, 1)
        server.stop()
        self.assertFalse(server.owned)
        self.assertTrue(wait_until(lambda: not ev.port_listening("127.0.0.1", port, timeout=0.3)))
        self.assertTrue(wait_until(lambda: not ev.pid_alive(pids["server"]) and not ev.pid_alive(pids["child"])),
                        "the server or the process it started is still alive")
        self.assertTrue(os.path.getsize(os.path.join(self.dir, "logs", f"server_{port}.log")) > 0)

    def test_a_server_that_died_is_started_again(self):
        port = free_port()
        server = self.server(port)
        self.assertTrue(server.ensure(wait_s=30)[0])
        first = self.pids()
        ev.kill_tree(server.proc)                          # it dies under us
        self.assertTrue(wait_until(lambda: not ev.port_listening("127.0.0.1", port, timeout=0.3)))
        ok, why = server.ensure(wait_s=30)
        self.assertTrue(ok, why)
        self.assertEqual(server.starts, 2)
        self.assertTrue(wait_until(lambda: self.pids()["server"] != first["server"]))
        self.assertTrue(ev.port_listening("127.0.0.1", port))

    def test_a_server_that_exits_at_once_is_reported(self):
        with open(os.path.join(self.checkout, "die"), "w") as f:
            f.write("x")
        server = self.server(free_port())
        ok, why = server.ensure(wait_s=30)
        self.assertFalse(ok)
        self.assertIn("exited with code", why)
        self.assertFalse(server.owned)

    def test_someone_elses_server_is_left_alone(self):
        import socket
        with socket.socket() as other:                     # stands in for a server we did not start
            other.bind(("127.0.0.1", 0))
            other.listen(4)
            port = other.getsockname()[1]
            server = self.server(port)
            self.assertTrue(server.ensure(wait_s=5)[0])
            self.assertEqual((server.starts, server.owned), (0, False))
            server.stop()
            self.assertTrue(ev.port_listening("127.0.0.1", port))

    def test_kill_tree_takes_the_children_too(self):
        import subprocess
        code = ("import subprocess, sys, time; c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
                "print(c.pid, flush=True); time.sleep(600)")
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, **ev.detached_kwargs())
        try:
            child = int(proc.stdout.readline())
            self.assertTrue(ev.pid_alive(proc.pid) and ev.pid_alive(child))
        finally:
            ev.kill_tree(proc, grace=5)
            proc.stdout.close()
        self.assertIsNotNone(proc.poll())
        self.assertTrue(wait_until(lambda: not ev.pid_alive(child)), "the child of the stopped process is still alive")
        self.assertEqual(ev.kill_tree(proc), [])           # a process that is gone: nothing happens
        self.assertEqual(ev.kill_tree(None), [])

    def spawn_with_child(self, parent_stays: float):
        """A process started the way evaluate.py starts its own, which starts a child of its own,
        prints the child's id and stays for `parent_stays` seconds."""
        import subprocess
        code = ("import subprocess, sys, time; c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
                f"print(c.pid, flush=True); time.sleep({parent_stays})")
        proc = ev.start_process([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        child = int(proc.stdout.readline())
        proc.stdout.close()
        return proc, child

    def test_kill_tree_takes_what_was_started_by_a_process_that_has_already_gone(self):
        # Seen on the laptop: an evaluation ended and its Showdown server stayed up on the
        # evaluation port. Once a process has gone, Windows no longer ties its children to it.
        proc, child = self.spawn_with_child(0)
        try:
            self.assertTrue(wait_until(lambda: proc.poll() is not None), "the parent should have left by itself")
            self.assertTrue(ev.pid_alive(child))
            self.assertEqual(ev.kill_tree(proc, grace=5), [])
            self.assertTrue(wait_until(lambda: not ev.pid_alive(child)), "what the departed process had started is still alive")
        finally:
            ev.kill_tree(proc, grace=2)

    def test_kill_tree_works_when_taskkill_does_not(self):
        if os.name != "nt":
            self.skipTest("taskkill is the Windows path")
        import subprocess
        proc, child = self.spawn_with_child(600)
        real_run, calls = subprocess.run, []

        def no_taskkill(cmd, *a, **k):
            if cmd and cmd[0] == "taskkill":
                calls.append(cmd)
                raise OSError(1455, "The paging file is too small for this operation to complete")
            return real_run(cmd, *a, **k)

        ev.subprocess.run = no_taskkill
        try:
            self.assertGreaterEqual(len(proc.psppo_job.pids()), 2)
            self.assertEqual(ev.kill_tree(proc, grace=5), [])           # it did not raise either
        finally:
            ev.subprocess.run = real_run
            ev.kill_tree(proc, grace=2)
        self.assertTrue(calls)
        self.assertIsNotNone(proc.poll())
        self.assertTrue(wait_until(lambda: not ev.pid_alive(child)), "the child survived a kill that could not use taskkill")

    def test_the_server_is_stopped_even_when_taskkill_does_not_work(self):
        if os.name != "nt":
            self.skipTest("taskkill is the Windows path")
        import subprocess
        port = free_port()
        server = self.server(port)
        self.assertTrue(server.ensure(wait_s=30)[0])
        pids = self.pids()
        real_run = subprocess.run
        ev.subprocess.run = lambda cmd, *a, **k: (_ for _ in ()).throw(OSError(1455, "no memory")) if cmd and cmd[0] == "taskkill" else real_run(cmd, *a, **k)
        try:
            server.stop()
        finally:
            ev.subprocess.run = real_run
        self.assertTrue(wait_until(lambda: not ev.port_listening("127.0.0.1", port, timeout=0.3)), "the evaluation port is still open")
        self.assertTrue(wait_until(lambda: not ev.pid_alive(pids["server"]) and not ev.pid_alive(pids["child"])))


class WatcherStepTests(unittest.TestCase):
    """One turn of the watch loop, with the evaluation itself replaced by something that writes lines."""

    def setUp(self):
        self.run = os.path.join(new_dir(), "run")
        self.paths = ev.RunPaths(self.run)
        os.makedirs(self.paths.checkpoints)
        self.ck_path = write_file(os.path.join(self.paths.checkpoints, "learner_update_000025.pt"), 64, time.time() - 30)
        self.set_phase("imitation")
        args = ev.build_parser().parse_args(["--run-dir", self.run, "--watch", "--foulplay-games", "0", "--settle-seconds", "0",
                                             "--games", "20"])
        self.w = ev.Watcher(args)
        self.messages = []
        self.w.log = self.messages.append
        self.children = []            # (reason, opponents) of every evaluation that was started
        self.exit_code = 0
        self.dead = set()             # opponents the evaluation cannot finish a game against
        self.w.run_child = self.fake_child
        self.real_server = ev.EvalServer
        ev.EvalServer = lambda *a, **k: FakeServer()

    def tearDown(self):
        ev.EvalServer = self.real_server

    def set_phase(self, phase):
        with open(self.paths.state, "w") as f:
            json.dump({"phase": phase}, f)

    def fake_child(self, plan, cfg, snap):
        self.children.append((plan.reason, plan.opponents))
        if self.exit_code == 0:
            ck = plan.checkpoint
            for o in plan.opponents:
                games = 0 if o in self.dead else 20
                ev.append_jsonl(self.paths.results, line(o, games=games, wins=games // 2, errors=20 - games, t=ev.utc_iso(snap.now),
                                                         name=ck.name, mtime=round(ck.mtime, 3), started=ev.utc_iso(snap.now),
                                                         state_phase=snap.phase))
        return self.exit_code

    def test_first_evaluation_then_nothing_until_a_new_checkpoint(self):
        self.w.step()
        self.assertEqual(self.children, [("first", ("random", "maxpower", "simple", "psbot"))])
        self.assertEqual(len(ev.read_jsonl(self.paths.results)), 4)
        self.assertEqual(self.w.evaluations, 1)
        self.w.step()
        self.w.step()
        self.assertEqual(len(self.children), 1)
        self.assertIn("waiting for a newer checkpoint", self.w.status)
        # a restart: a new watcher object, the same files
        again = ev.Watcher(self.w.args)
        again.log = self.messages.append
        again.run_child = self.fake_child
        again.step()
        self.assertEqual(len(self.children), 1)
        self.assertFalse(os.path.exists(self.paths.final_done))

    def test_a_failed_evaluation_backs_off_and_is_given_up(self):
        self.exit_code = ev.EXIT_NO_CHECKPOINT
        self.w.step()
        key = ev.stat_ckpt(self.ck_path).key
        self.assertEqual(self.w.failures[key][0], 1)
        self.w.step()                                         # too soon: nothing is started
        self.assertEqual(len(self.children), 1)
        self.assertIn("trying again", self.w.status)
        for expected in (2, 3):
            self.w.failures[key] = (self.w.failures[key][0], time.time() - 3600)     # the wait is over
            self.w.step()
            self.assertEqual(self.w.failures[key][0], expected)
        self.w.failures[key] = (3, time.time() - 3600)
        self.w.step()
        self.assertEqual(len(self.children), 3)               # given up: no fourth attempt
        self.assertIn("gave up", self.w.status)
        self.assertEqual(ev.read_jsonl(self.paths.results), [])

    def test_an_opponent_with_no_finished_game_is_tried_again_alone(self):
        self.dead = {"psbot"}
        self.w.step()
        key = ev.stat_ckpt(self.ck_path).key
        self.assertEqual(self.w.failures[key][0], 1)
        self.assertTrue(any("no game finished against psbot" in m for m in self.messages))
        self.dead = set()
        self.w.failures[key] = (1, time.time() - 3600)
        self.w.step()
        self.assertEqual(self.children[-1], ("resume", ("psbot",)))
        self.assertNotIn(key, self.w.failures)
        self.assertEqual([r["games"] for r in ev.read_jsonl(self.paths.results)], [20, 20, 20, 0, 20])

    def test_done_runs_the_final_evaluation_and_writes_the_marker_once(self):
        self.w.step()
        newer = write_file(os.path.join(self.paths.checkpoints, "learner_update_000050.pt"), 64, time.time() - 5)
        self.w.step()
        self.assertEqual(len(self.children), 1)               # the interval has not passed
        self.set_phase("done")
        self.w.step()
        self.assertEqual(self.children[-1], ("final", ("random", "maxpower", "simple", "psbot")))
        final = ev.read_json(self.paths.final_done)
        self.assertEqual((final["ok"], final["checkpoint"], len(final["results"])), (True, "learner_update_000050.pt", 4))
        self.assertTrue(ev.final_covers(final, ev.stat_ckpt(newer)))
        written = os.path.getmtime(self.paths.final_done)
        self.w.step()
        self.assertEqual(len(self.children), 2)
        self.assertEqual(os.path.getmtime(self.paths.final_done), written)
        self.assertIn("final evaluation is written", self.w.status)

    def test_done_gives_up_rather_than_hold_the_run(self):
        self.set_phase("done")
        self.exit_code = 1
        key = ev.stat_ckpt(self.ck_path).key
        for _ in range(3):
            self.w.step()
            self.w.failures[key] = (self.w.failures[key][0], time.time() - 3600)
        self.assertFalse(os.path.exists(self.paths.final_done))
        self.w.step()
        final = ev.read_json(self.paths.final_done)
        self.assertEqual((final["ok"], final["checkpoint"], final["results"]), (False, "learner_update_000025.pt", []))
        self.assertIn("failed 3 times", final["note"])

    def test_an_evaluation_that_cannot_even_start_is_a_failed_attempt(self):
        def broken(plan, cfg, snap):
            raise OSError("cannot start the interpreter")

        self.w.run_child = broken
        self.w.step()                                         # does not raise
        self.assertEqual(self.w.failures[ev.stat_ckpt(self.ck_path).key][0], 1)
        self.assertTrue(any("cannot start the interpreter" in m for m in self.messages))

    def test_the_report_is_written_after_an_evaluation(self):
        self.w.step()
        with open(self.paths.report, encoding="utf-8") as f:
            self.assertIn("| simple | 50.0% |", f.read())

    def test_idle_messages_are_said_once_per_checkpoint(self):
        self.w.idle("learner_update_000020.pt is next, in 58 min")
        self.w.idle("learner_update_000020.pt is next, in 57 min")
        self.w.idle("learner_update_000030.pt is next, in 57 min")
        self.assertEqual(self.messages, ["waiting: learner_update_000020.pt is next, in 58 min",
                                         "waiting: learner_update_000030.pt is next, in 57 min"])
        self.assertEqual(self.w.status, "learner_update_000030.pt is next, in 57 min")

    def test_no_server_is_a_failed_attempt_not_a_crash(self):
        class NoServer(FakeServer):
            def ensure(self):
                return False, "nothing is listening on 127.0.0.1:9000 and no server directory was given (--server-dir)"

        ev.EvalServer = lambda *a, **k: NoServer()
        self.w.step()
        self.assertEqual(self.children, [])
        self.assertEqual(self.w.failures[ev.stat_ckpt(self.ck_path).key][0], 1)
        self.assertTrue(any("no server" in m for m in self.messages))

    def test_missing_run_directory_is_waited_for_and_not_created(self):
        gone = os.path.join(new_dir(), "not_yet")
        w = ev.Watcher(ev.build_parser().parse_args(["--run-dir", gone, "--watch"]))
        w.log = self.messages.append
        w.step()
        w.heartbeat()
        self.assertIn("does not exist yet", w.status)
        self.assertFalse(os.path.exists(gone))

    def test_a_second_watcher_is_refused_while_the_first_is_alive(self):
        self.w.status = "idle"
        self.w.heartbeat()
        state = ev.read_json(self.paths.watch_state)
        self.assertEqual(state["pid"], os.getpid())
        self.assertIsNone(self.w.other_watcher())             # our own heartbeat does not lock us out
        state["pid"] = os.getppid()                           # another live process holds the run
        ev.write_json_atomic(self.paths.watch_state, state)
        real = ev.pid_command_line
        try:
            ev.pid_command_line = lambda pid, needle: True    # ...and the system says it is an evaluate.py
            self.assertIn(str(os.getppid()), self.w.other_watcher())
            ev.pid_command_line = lambda pid, needle: None    # the system cannot say what it is (Windows): the id decides
            self.assertIn(str(os.getppid()), self.w.other_watcher())
            # The id belongs to something else now. After a reboot the units come up with nearly
            # the same ids as before, and the old heartbeat is still minutes fresh: without this
            # the evaluator would refuse to start until that heartbeat is ten minutes old.
            ev.pid_command_line = lambda pid, needle: False
            self.assertIsNone(self.w.other_watcher())
            ev.pid_command_line = lambda pid, needle: True
            state["status"] = "stopped"                       # a watcher that said goodbye is not running, whoever has its id
            ev.write_json_atomic(self.paths.watch_state, state)
            self.assertIsNone(self.w.other_watcher())
            state["status"] = "idle"
            state["heartbeat_ts"] = time.time() - 3600        # alive and an evaluate.py, but not heard for an hour
            ev.write_json_atomic(self.paths.watch_state, state)
            self.assertIsNone(self.w.other_watcher())
        finally:
            ev.pid_command_line = real

    def test_what_a_process_is_is_read_from_proc_where_there_is_one(self):
        root = os.path.join(new_dir(), "proc")
        for pid, cmdline in ((101, b"/home/ubuntu/venv/bin/python\x00evaluate.py\x00--run-dir\x00/home/ubuntu/runs/main\x00--watch\x00"),
                             (102, b"node\x00--max-old-space-size=2048\x00pokemon-showdown\x00start\x00"), (103, b"")):
            os.makedirs(os.path.join(root, str(pid)))
            with open(os.path.join(root, str(pid), "cmdline"), "wb") as f:
                f.write(cmdline)
        self.assertIs(ev.pid_command_line(101, "evaluate.py", proc_root=root), True)
        self.assertIs(ev.pid_command_line(102, "evaluate.py", proc_root=root), False)
        self.assertIsNone(ev.pid_command_line(103, "evaluate.py", proc_root=root))     # nothing to read: cannot tell
        self.assertIsNone(ev.pid_command_line(104, "evaluate.py", proc_root=root))     # no such process entry
        self.assertIsNone(ev.pid_command_line("junk", "evaluate.py", proc_root=root))
        self.assertIsNone(ev.pid_command_line(None, "evaluate.py", proc_root=root))
        here = ev.pid_command_line(os.getpid(), "definitely-not-in-any-command-line-9731")
        self.assertIn(here, (None, False))                    # this very process: False on Linux, None on Windows


class AuditMixinTests(unittest.TestCase):
    """players.ChoiceAuditMixin on a stand-in for poke-env's Player (no server)."""

    def setUp(self):
        try:
            import players
            from poke_env.battle.battle import Battle
        except ImportError as e:
            self.skipTest(f"poke-env is not installed: {e}")
        import logging

        sent = self.sent = []

        class Base:
            def __init__(self):
                async def send_message(message, room=""):
                    sent.append((message, room))
                self.ps_client = types.SimpleNamespace(send_message=send_message)
                self.asked = 0
                self.fail = False

            async def _handle_battle_message(self, split_messages):
                return None

            async def _handle_battle_request(self, battle, maybe_default_order=False):
                self.asked += 1

            def choose_move(self, battle):
                if self.fail:
                    raise ZeroDivisionError("bad policy")
                return "the order"

        class Async(Base):
            async def choose_move(self, battle):
                if self.fail:
                    raise ZeroDivisionError("bad policy")
                return "the order"

        self.players = players
        self.player = type("P", (players.ChoiceAuditMixin, Base), {})()
        self.async_player = type("A", (players.ChoiceAuditMixin, Async), {})()
        self.battle = Battle("battle-gen9randombattle-1", "me", logging.getLogger("test_evaluate"), gen=9)

    def feed(self, player, *lines):
        asyncio.run(player._handle_battle_message([[">battle-gen9randombattle-1"]] + [ln.split("|") for ln in lines]))

    def test_rejected_orders_and_endings_are_counted(self):
        p = self.player
        self.feed(p, "|error|[Invalid choice] Can't move: You can't switch to an active Pokemon",
                  "|error|[Unavailable choice] Can't move: Your Garchomp's Outrage is disabled", "|error|something else", "|turn|4")
        self.assertEqual((p.audit["invalid_choices"], p.audit["unavailable_choices"], p.audit["other_errors"]), (1, 1, 1))
        self.assertEqual(len(p.audit_messages), 2)          # the unavailable one is normal play: counted, not kept
        self.feed(p, "|-message|opp forfeited.", "|win|me")
        self.assertEqual(p.audit_endings, {"battle-gen9randombattle-1": "forfeit"})
        self.feed(p, "|-message|opp lost due to inactivity.", "|win|me")
        self.assertEqual(p.audit_endings, {"battle-gen9randombattle-1": "inactivity"})

    def test_a_request_rejected_again_and_again_is_forced(self):
        p = self.player
        for _ in range(2):
            asyncio.run(p._handle_battle_request(self.battle, maybe_default_order=True))
        self.assertEqual((p.asked, p.audit["forced_random"], self.sent), (2, 0, []))
        asyncio.run(p._handle_battle_request(self.battle, maybe_default_order=True))
        self.assertEqual((p.asked, p.audit["forced_random"]), (2, 1))
        self.assertEqual(self.sent, [("/choose default", "battle-gen9randombattle-1")])
        asyncio.run(p._handle_battle_request(self.battle))            # the next real request starts over
        asyncio.run(p._handle_battle_request(self.battle, maybe_default_order=True))
        self.assertEqual((p.asked, p.audit["forced_random"]), (4, 1))

    def test_a_chooser_that_raises_plays_a_legal_move_and_is_counted(self):
        for player, run in ((self.player, lambda x: x), (self.async_player, asyncio.run)):
            self.assertEqual(run(player.choose_move(self.battle)), "the order")
            player.fail = True
            order = run(player.choose_move(self.battle))
            self.assertEqual(order.message, "/choose default")
            self.assertEqual((player.audit["requests"], player.audit["exceptions"]), (2, 1))
            self.assertIn("bad policy", player.audit_messages[-1])

    def test_audited_is_cached_and_does_not_wrap_twice(self):
        a = self.players.audited(self.players.PSBotPlayer)
        self.assertIs(a, self.players.audited(self.players.PSBotPlayer))
        self.assertIs(a, self.players.audited(a))
        self.assertTrue(issubclass(a, self.players.ChoiceAuditMixin))


class BatchedInferenceTests(unittest.TestCase):
    """The batched policy player must compute what hybrid.PolicyNet computes one decision at a time."""

    def test_batched_equals_one_at_a_time(self):
        try:
            import numpy as np
            import torch

            import players
            from hybrid import HybridConfig, PolicyNet
        except ImportError as e:
            self.skipTest(f"torch or poke-env is not installed: {e}")
        import warnings
        warnings.simplefilter("ignore")
        torch.manual_seed(0)
        reference = PolicyNet(HybridConfig(random_init=True, device="cpu"))
        batched = PolicyNet(HybridConfig(random_init=True, device="cpu"))
        batched.net.load_state_dict(reference.net.state_dict())
        cls = players.batched_policy_player()
        player = cls.__new__(cls)                      # no __init__: no connection to a server
        player.policy = batched
        player.batch_stats = {"forwards": 0, "rows": 0, "largest": 0}

        m0, m1 = reference.assembler.offsets["action_mask"]
        size = max(end for _s, end in reference.assembler.offsets.values())
        window = reference.net.kv_cache_len
        rng = np.random.default_rng(0)

        def obs():
            o = np.zeros(size, dtype=np.float32)
            o[m0:m1] = (rng.random(m1 - m0) < 0.6).astype(np.float32)
            return o

        # Three battles of different lengths that start and end at different times. The longest
        # runs past the history window, so the rolling cache is exercised too.
        plan = {"a": (0, window + 6), "b": (3, 20), "c": (15, 12)}       # tag -> (first step, decisions)
        want, got = {t: [] for t in plan}, {t: [] for t in plan}
        seen = {t: [] for t in plan}
        for step in range(window + 6):
            tags = [t for t, (start, count) in plan.items() if start <= step < start + count]
            batch = np.stack([obs() for _ in tags])
            out = player._forward_batch(tags, batch)
            for i, t in enumerate(tags):
                got[t].append(out[i])
                seen[t].append(batch[i])
            if step == 22:
                batched.forget("b")                    # battle b ended: its history must be gone
        for t in plan:
            for o in seen[t]:
                want[t].append(reference.logits(t, o))
            reference.forget(t)
        for t in plan:
            self.assertEqual(len(got[t]), plan[t][1])
            diff = max(float(np.max(np.abs(g - w))) for g, w in zip(got[t], want[t]))
            self.assertLess(diff, 1e-4, f"battle {t}: batched and single logits differ by {diff}")
        self.assertEqual(player.batch_stats["forwards"], window + 6)
        self.assertEqual(player.batch_stats["largest"], 3)
        self.assertNotIn("b", batched._cache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
