"""
Settings per game: ladder.py --arm-settings, search_player.GameSetting and SettingPicker, and
tools/search_sweep.py's together mode.

    python -m pytest tests/test_arm_settings.py -q -p no:cacheprovider

Off (the default), ladder.py and search_player.py play byte for byte as before: the player of the
commit before the flag (BASE_COMMIT, read with git, or from PSBOT_BASE_DIR) and today's replay the
recorded battles of tests/fixtures in every mode with the same stub network and fake search, and
their orders, their questions to the search, their decision records and their game summaries are
the same (times aside); ladder.py's parse_args, search_problems and the result-line helpers give
the same. On, a game of a setting plays exactly as a player made for that setting alone: three
battles played at once with three of the app's settings ask the search, decide, send and record
what three players of one setting each do with the same battle.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import ladder  # noqa: E402
import search_player as sp  # noqa: E402
from search_player import GameSetting, MixSettings, SearchBudget, SettingPicker  # noqa: E402

_spec = importlib.util.spec_from_file_location("tsp_for_settings", os.path.join(REPO, "tests", "test_search_player.py"))
tsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tsp)

_sweep_spec = importlib.util.spec_from_file_location("search_sweep_for_test", os.path.join(REPO, "tools", "search_sweep.py"))
sweep = importlib.util.module_from_spec(_sweep_spec)
_sweep_spec.loader.exec_module(sweep)

BASE_COMMIT = "7dbbbe3"      # local-app before --arm-settings
THINK = os.path.join(REPO, "tests", "fixtures", "app_think_table.txt")

# What changes from one run to the next (the clock), left out of the comparisons.
TIMING = {"ms", "net_ms", "wait_over_ms", "search_ms", "helper_ms", "beside"}
SUMMARY_TIMING = {"search_ms_mean", "decision_ms_mean", "decision_ms_max", "late", "it_deep_beside", "n_deep_beside",
                  "it_no_deep", "n_no_deep"}
SETTING_FIELDS = {"setting", "setting_budget", "setting_fallback", "budget_asked"}


def app_settings(names, mode="guarded"):
    """[{"name", "args"}] of the app's table (tests/fixtures/app_think_table.txt) for names such as "2s-deeper"."""
    table = sweep.read_think_table(THINK)
    out = []
    for name in names:
        secs, style = name.split("s-")
        out.append({"name": name, "args": table[(float(secs), style, mode)]["args"]})
    return out


def load(rows, run=("--games", "1", "--parallel", "6")):
    """ladder.parse_args of a run with these settings (written to no file: load_arm_settings is given the list)."""
    argv = list(run)
    args = ladder.parse_args(argv)
    args.arm_settings_file = "(test)"
    args.arm_settings = ladder.load_arm_settings(args, argv + ["--arm-settings", "x"], data=rows)
    return args


def refused(rows, run=("--games", "1", "--parallel", "6")):
    with pytest.raises(SystemExit) as exc:
        load(rows, run)
    return str(exc.value)


# =============================================================================================
# The base: the commit before the flag
# =============================================================================================

def _base_path(name, tmp_path_factory):
    base_dir = os.environ.get("PSBOT_BASE_DIR")
    if base_dir:
        path = os.path.join(base_dir, name + ".py")
        if not os.path.isfile(path):
            pytest.skip(f"no {path}")
        return path
    try:
        src = subprocess.run(["git", "-C", REPO, "show", f"{BASE_COMMIT}:{name}.py"], capture_output=True, check=True,
                             timeout=60).stdout
    except Exception:
        pytest.skip(f"the commit {BASE_COMMIT} cannot be read (no git here): give PSBOT_BASE_DIR")
    path = tmp_path_factory.mktemp("base") / (name + ".py")
    path.write_bytes(src)
    return str(path)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """ladder.py and search_player.py of BASE_COMMIT, as modules of their own (their imports are today's shared modules)."""
    out = {}
    for name in ("ladder", "search_player"):
        spec = importlib.util.spec_from_file_location(name + "_base", _base_path(name, tmp_path_factory))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod             # its dataclasses look their module up while it is read
        spec.loader.exec_module(mod)
        out[name] = mod
    out["ladder"].HERE = ladder.HERE          # its defaults that name this folder (--stop-file) are today's
    assert not hasattr(out["search_player"], "GameSetting")     # it is the commit before
    return out


@pytest.fixture(scope="module")
def classes(base):
    """The search player of today and of the base, both with test_search_player's stub network."""
    import numpy as np

    import hybrid

    real = hybrid.PolicyNet
    hybrid.PolicyNet = tsp.StubPolicy

    def stubbed(cls):
        class Stubbed(cls):
            def _forward_batch(self, tags, obs):
                for tag in tags:
                    self.policy._cache[tag] = True
                self.batch_stats["forwards"] += 1
                self.batch_stats["rows"] += len(tags)
                return np.stack([tsp.stub_logits(row) for row in obs])

        return Stubbed

    yield {"new": stubbed(sp.search_player()), "base": stubbed(base["search_player"].search_player())}
    hybrid.PolicyNet = real


def made(cls, username, bridge, **kw):
    kw.setdefault("search_ms", 20)
    kw.setdefault("deadline", 0.5)
    return tsp.offline(cls, username, bridge=bridge, warm_up=False, **kw)


def strip(rec, drop):
    return {k: v for k, v in rec.items() if k not in drop}


def what_it_did(player, bridge, events):
    """Everything a player did in one battle that does not depend on the clock."""
    room = events[0]["room"]
    key = tsp.key_of(events)
    return {
        "sent": [m for r, m in player.sent if r == room],
        "asked": [(rqid, ms, sorted(more.items())) for k, rqid, _ttl, ms, more in bridge.asked if k == key],
        "decisions": [strip(d, TIMING) for d in player.decisions_of(room)],
        "summary": strip(player.game_summary(room) or {}, SUMMARY_TIMING),
    }


# =============================================================================================
# Off: byte for byte as before
# =============================================================================================

def _player_cases(mod):
    """(mode, the player's arguments) of every kind of game: the network, the search, mix and its rules, the guard, the
    guarded search (mix-trust, and --trust on mix-deep), the deep arm with its budget."""
    deep = mod.SearchBudget(search_ms=40, worlds_x=2, deadline=1.0)
    return [
        ("net", {}),
        ("search", {}),
        ("mix", {}),
        ("mix-pool", {}),
        ("mix-guard", {}),
        ("mix-trust", {}),
        ("mix-deep", {"budgets": {"mix-deep": deep}, "mix": mod.MixSettings(min_iterations=0, trust=0.03)}),
        ("mix-deep", {"budgets": {"mix-deep": mod.SearchBudget(search_ms=40, worlds_x=2, deadline=1.0, min_time_left=10_000)}}),
    ]


@tsp.needs_player
@pytest.mark.parametrize("case", range(8))
def test_off_every_mode_plays_records_and_asks_as_the_commit_before(classes, base, case):
    battles = tsp.load_fixture()
    seen = {}
    for which, mod in (("base", base["search_player"]), ("new", sp)):
        mode, kw = _player_cases(mod)[case]
        kw = dict(kw)
        kw.setdefault("mix", mod.MixSettings(min_iterations=0))
        out = []
        for events in battles:
            bridge = tsp.FakeBridge([events])
            try:
                player = made(classes[which], events[-1]["username"], bridge, mode=mode, **kw)
                tsp.play(player, events)
                out.append(what_it_did(player, bridge, events))
                if which == "new":
                    assert player.battle_setting(events[0]["room"]) is None
                    assert all("setting" not in d for d in player.decisions_of(events[0]["room"]))
            finally:
                bridge.close()
        seen[which] = out
    assert sum(len(x["decisions"]) for x in seen["new"]) > 100
    for old, new in zip(seen["base"], seen["new"]):
        assert new["sent"] == old["sent"]
        assert new["asked"] == old["asked"]
        assert new["decisions"] == old["decisions"]
        assert new["summary"] == old["summary"]
        assert not (SETTING_FIELDS & set(new["summary"]))


@tsp.needs_player
def test_off_arms_are_given_and_played_as_the_commit_before(classes, base):
    battles = [e for e in tsp.load_fixture() if e[-1]["username"] == "recA3ace"]
    seen = {}
    for which, mod in (("base", base["search_player"]), ("new", sp)):
        bridge = tsp.FakeBridge(battles)
        try:
            picker = mod.ArmPicker(["net", "mix", "mix-trust"], rng=random.Random(5))
            player = made(classes[which], "recA3ace", bridge, mode="mix", mode_picker=picker,
                          mix=mod.MixSettings(min_iterations=0))
            for events in battles:              # one after the other: the same order of first messages
                tsp.play(player, events)
            seen[which] = [what_it_did(player, bridge, events) for events in battles]
        finally:
            bridge.close()
    assert seen["new"] == seen["base"]


def _ladder_argvs():
    rows = tsp.app_think_rows()
    out = [["--games", "1", "--parallel", "2"] + r["argv"] for r in rows]
    out += [[], ["--arms", "net,mix,mix-deep", "--parallel", "6", "--helpers", "7", "--deep-workers", "4", "--deep-deadline", "4.0"],
            ["--player", "mix-deep", "--trust", "0.03"], ["--player", "mix", "--mix-rule", "pool", "--trust", "0.03"],
            ["--arms", "net,mix,search", "--trust", "0.03"], ["--player", "predict", "--deadline", "1.0"],
            ["--player", "mix-lines", "--parallel", "6", "--helpers", "7"], ["--arms", "luna,sol"],
            ["--player", "mix-deep", "--deep-deadline", "4.0"], ["--server", "local:9000", "--no-timer", "--games", "3"]]
    return out


def _or_error(f, *a):
    try:
        return f(*a)
    except ValueError as exc:
        return "ValueError: %s" % exc


def test_off_the_command_line_reads_and_checks_as_the_commit_before(base):
    old_ladder = base["ladder"]
    for argv in _ladder_argvs():
        new, old = ladder.parse_args(list(argv)), old_ladder.parse_args(list(argv))
        got = vars(new)
        assert got.pop("arm_settings_file") is None and got.pop("arm_settings") is None, argv
        assert got == vars(old), argv
        assert ladder.search_problems(new) == old_ladder.search_problems(old), argv
        assert ladder.run_problems(new) == old_ladder.search_problems(old), argv
        assert ladder.run_modes(new) == (list(old.arms) if old.arms else [old.player]), argv
        assert ladder.run_deadline(new) == float(old.deadline), argv
        assert ladder.experiment_tag(new) == old_ladder.experiment_tag(old), argv
        assert _or_error(ladder.arm_budgets, new, ladder.run_modes(new)) == _or_error(old_ladder.arm_budgets, old, ladder.run_modes(new)), argv
    games = [
        {"arm": "mix", "checkpoint": "a.pt", "won": True, "decisions": 30, "searched": 28, "fallbacks": 1, "overruled": 4, "tera": True},
        {"arm": "mix-deep", "checkpoint": "a.pt", "won": False, "decisions": 20, "searched": 20, "fallbacks": 0, "trust": 0.03},
        {"arm": "net", "checkpoint": "a.pt", "won": True, "decisions": 25, "searched": 0, "fallbacks": 0, "tera": True},
    ]
    assert ladder.arms_text(games) == old_ladder.arms_text(games)
    summary = {"arm": "mix-deep", "budget": {"ms": 40, "worlds_x": 2, "threads": 1, "deadline": 1.0}, "searched": 3}
    assert ladder.arm_fields(summary, "p1", 500, "veto") == old_ladder.arm_fields(summary, "p1", 500, "veto")


# =============================================================================================
# The settings file
# =============================================================================================

SIX = ["2s-deeper", "2s-wider", "4s-deeper", "4s-wider", "6s-deeper", "6s-wider"]


def test_the_apps_six_guarded_settings_load_with_their_own_mode_budget_and_trust():
    args = load(app_settings(SIX))
    assert args.helpers == 7 and args.parallel == 6                  # parallel + 1: not the app's 3
    got = {s.name: s.game for s in args.arm_settings}
    assert [s.name for s in args.arm_settings] == SIX
    for name, (ms, deadline) in (("2s-deeper", (900, 2.7)), ("4s-deeper", (1800, 5.0)), ("6s-deeper", (2700, 7.2))):
        g = got[name]
        assert (g.mode, g.search_ms, g.deadline, g.trust, g.budget) == ("mix-trust", ms, deadline, 0.0, None), name
        assert g.asks() == {"ms": ms, "worlds_x": 1, "threads": 1, "deadline": deadline}
    for name, (worlds, cap) in (("2s-wider", (2, 3.0)), ("4s-wider", (4, 5.5)), ("6s-wider", (5, 6.7))):
        g = got[name]
        assert (g.mode, g.search_ms, g.deadline, g.trust) == ("mix-deep", 500, 2.5, 0.03), name
        assert g.budget == SearchBudget(search_ms=500, worlds_x=worlds, threads=1, workers=1, deadline=cap,
                                        min_time_left=90.0, max_decision_s=9.0), name
        assert g.today() == {"ms": 500, "worlds_x": 1, "threads": 1, "deadline": 2.5}
    # Each setting's own arguments read exactly as ladder.py alone reads them (the app's command line).
    for s, row in zip(args.arm_settings, app_settings(SIX)):
        alone = ladder.parse_args(["--games", "1", "--parallel", "2"] + row["args"])
        for dest in ladder.SETTING_FLAGS:
            assert getattr(s.args, dest) == getattr(alone, dest), (s.name, dest)
        assert ladder.search_problems(alone) == []
        assert ladder.arm_budgets(alone, [alone.player]).get(alone.player) == s.game.budget
        assert ladder.mix_settings(alone).trust == s.game.trust
    assert ladder.run_modes(args) == ["mix-trust", "mix-deep"]
    assert ladder.run_deadline(args) == 7.2
    assert ladder.experiment_tag(args) == "settings:" + ",".join(SIX)
    assert ladder.run_problems(args) == []
    assert "2s-wider" in [s.name for s in args.arm_settings] and "mix-deep, 500 ms per world on 2x" in args.arm_settings[1].text()
    assert args.arm_settings[0].text() == "mix-trust, 900 ms within 2.7 s, trust 0.03"


def test_every_line_of_the_apps_table_is_a_setting():
    for r in tsp.app_think_rows():
        args = load([{"name": f"{r['seconds']:g}s-{r['style']}-{r['mode']}", "args": r["argv"]}], ("--games", "1", "--parallel", "2"))
        g = args.arm_settings[0].game
        own = g.own_budget()
        assert (own.search_ms if own else g.search_ms, own.deadline if own else g.deadline) == (r["ms"], r["cap"]), r
        assert (own.worlds_x if own else 1) == r["teams"] // 2, r


def test_a_file_on_disk_is_read_by_parse_args(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(app_settings(["2s-deeper", "6s-wider"])), encoding="utf-8")
    args = ladder.parse_args(["--games", "4", "--parallel", "2", "--arm-settings", str(path)])
    assert [s.name for s in args.arm_settings] == ["2s-deeper", "6s-wider"] and args.helpers == 3
    args = ladder.parse_args(["--parallel", "2", "--helpers", "5", "--arm-settings", str(path)])
    assert args.helpers == 5                                           # given: kept
    with pytest.raises(SystemExit, match="cannot be read"):
        ladder.parse_args(["--arm-settings", str(tmp_path / "nothing.json")])


def test_settings_that_cannot_be_played_are_refused_with_the_reason():
    ok = app_settings(["2s-deeper", "2s-wider"])
    assert "cannot go together" in refused(ok, ("--arms", "net,mix"))
    assert "--search-ms is given per game" in refused(ok, ("--search-ms", "700"))
    assert "--player, --trust are given per game" in refused(ok, ("--player", "mix", "--trust", "0.03"))
    pool = [ok[0], {"name": "2s-pool", "args": ["--player", "mix", "--mix-rule", "pool", "--search-ms", "900", "--deadline", "2.7"]}]
    text = refused(pool)
    assert "the settings disagree on --mix-rule (2s-deeper: veto; 2s-pool: pool)" in text and "whole run" in text
    guard = [dict(ok[0], args=ok[0]["args"] + ["--guard"]), dict(ok[1], args=ok[1]["args"] + ["--guard"])]
    assert "the settings give --guard True and this run False" in refused(guard)
    assert load(guard, ("--games", "1", "--parallel", "6", "--guard")).arm_settings[0].game.mode == "mix-trust"
    assert "disagree on --guard" in refused([guard[0], ok[1]], ("--guard",))     # one setting without it plays unguarded alone
    short = [{"name": "short", "args": ["--player", "mix-trust", "--search-ms", "900", "--deadline", "2.0"]}]
    assert "setting short: --search-ms 900 needs --deadline 2.62 or more" in refused(short)
    deep = [{"name": "deep", "args": ["--player", "mix-deep", "--deep-worlds", "5", "--deep-search-ms", "500", "--deep-deadline", "5.0"]}]
    assert "setting deep: the deep arm" in refused(deep)
    late = [{"name": "late", "args": ["--player", "mix-deep", "--deep-deadline", "10.5", "--deep-max-decision-s", "12"]}]
    assert "costs two ticks" in refused(late)
    pooled_trust = [{"name": "pt", "args": ["--player", "search", "--trust", "0.03", "--search-ms", "500"]}]
    assert "setting pt: --trust 0.03 guards the mix modes only, not search" in refused(pooled_trust)
    assert "two settings are named 2s-deeper" in refused([ok[0], ok[0]])
    assert "a name is up to 40" in refused([{"name": "2 s deeper", "args": []}])
    assert "not a" in refused([{"name": "x", "args": "--player mix"}])
    assert "not a JSON list" in refused([])
    assert "not a JSON list" in refused({"settings": ok})
    assert "unrecognized arguments" in refused([{"name": "x", "args": ["--player", "mix", "--no-such-flag"]}])
    assert "invalid choice" in refused([{"name": "x", "args": ["--player", "tree"]}])
    assert "--arms cannot be in a setting" in refused([{"name": "x", "args": ["--arms", "net,mix"]}])
    for player in ("predict", "luna", "sol", "luna-solo", "opus-solo", "sol-lines", "luna-lines"):
        assert "cannot be a setting" in refused([{"name": "x", "args": ["--player", player]}]), player
    assert "--trust is a probability" in refused([{"name": "x", "args": ["--player", "mix-deep", "--trust", "1.5"]}])
    assert "--helpers 3 with --parallel 6" in refused(ok, ("--parallel", "6", "--helpers", "3"))


def test_a_setting_may_leave_the_search_out_and_two_settings_may_differ_in_budget_only():
    args = load([{"name": "net", "args": ["--player", "net"]}] + app_settings(["2s-deeper", "6s-deeper"]))
    assert [s.game.mode for s in args.arm_settings] == ["net", "mix-trust", "mix-trust"]
    assert args.arm_settings[0].text() == "net, no search"
    assert ladder.run_modes(args) == ["net", "mix-trust"]


def test_game_setting_checks_itself():
    with pytest.raises(ValueError, match="never started"):
        GameSetting(name="x", mode="mix", search_ms=900, deadline=2.0)
    with pytest.raises(ValueError):
        GameSetting(name="", mode="mix")
    with pytest.raises(ValueError):
        GameSetting(name="x", mode="tree")
    with pytest.raises(ValueError):
        GameSetting(name="x", mode="mix", trust=1.0)
    assert GameSetting(name="x", mode="net", search_ms=900, deadline=0.1).today()["ms"] == 900   # no search: no deadline to keep
    assert GameSetting(name="x", mode="mix-deep").own_budget() == sp.DEEP_BUDGET                # ARM_BUDGETS' when none is given


# =============================================================================================
# The picker
# =============================================================================================

def _six():
    return [GameSetting(name=n, mode="mix") for n in SIX]


def test_the_setting_picker_keeps_the_settings_level_in_rounds():
    picker = SettingPicker(_six(), rng=random.Random(3))
    given = [picker("battle-%d" % i).name for i in range(30)]
    for start in range(0, 30, 6):
        assert sorted(given[start:start + 6]) == sorted(SIX)               # every six games are one of each
    assert [picker("battle-%d" % i).name for i in range(30)] == given       # a battle keeps its setting
    assert picker.counts == {n: 5 for n in SIX}


def test_the_setting_picker_gives_a_new_game_a_setting_with_no_game_in_progress():
    picker = SettingPicker(_six(), rng=random.Random(7))
    playing = {}
    for i in range(6):
        playing["g%d" % i] = picker("g%d" % i).name
    picker.in_play = lambda: list(playing)
    assert sorted(playing.values()) == sorted(SIX)
    rng = random.Random(11)
    for i in range(6, 60):           # the hardest order: any game in progress may be the next to end
        done = rng.choice(sorted(playing))                                  # a game ends, the next one starts
        playing.pop(done)
        fewest = min(picker.counts.values())
        least = {n for n, c in picker.counts.items() if c == fewest}
        free = least - set(playing.values())
        tag = "g%d" % i
        playing[tag] = picker(tag).name
        assert playing[tag] in (free or least)                              # the fewest games; among them one not in progress
        counts = list(picker.counts.values())
        assert max(counts) - min(counts) <= 1                               # never more than one game ahead
    picker.in_play = lambda: 1 / 0                                          # a failing in_play is no preference, not an error
    assert picker("late").name in SIX


def test_with_games_of_ladder_lengths_the_games_in_progress_are_one_of_each_setting():
    """
    A sweep of 30 games, 6 at once: each game lasts its setting's time give or take 30%, and the next game starts
    when one ends. The settings always end level (5 games each, the balance rule); and with the preference for a
    setting with no game in progress, the 6 games in progress are one of each setting about 3 times as often as
    without it: nearly always when the settings' games last about as long, two times in three when the 6 s games
    last 14 minutes and the 2 s ones 8 (then a fast setting can finish its next game before a slow one has finished
    this one, and the balance rule comes first).
    """
    def share(prefer, minutes, seeds=50):
        distinct = full = 0
        for seed in range(seeds):
            rng = random.Random(seed)
            picker = SettingPicker(_six(), rng=random.Random(seed))
            ends = {}                                   # tag -> (end time, setting)
            if prefer:
                picker.in_play = lambda: list(ends)
            started, now = 0, 0.0
            while started < 30 or ends:
                while started < 30 and len(ends) < 6:
                    tag = "g%d" % started
                    name = picker(tag).name
                    ends[tag] = (now + minutes[name] * rng.uniform(0.7, 1.3), name)
                    started += 1
                    if len(ends) == 6:
                        full += 1
                        distinct += len({n for _e, n in ends.values()}) == 6
                tag = min(ends, key=lambda t: ends[t][0])
                now = ends.pop(tag)[0]
            assert picker.counts == {n: 5 for n in SIX}, seed
        return distinct / full

    alike = dict(zip(SIX, (9.0, 9.0, 10.0, 10.0, 11.0, 11.0)))
    apart = dict(zip(SIX, (8.0, 9.0, 10.0, 11.0, 12.0, 14.0)))
    assert share(True, alike) >= 0.8 and share(True, apart) >= 0.55
    assert share(True, apart) >= 2 * share(False, apart) and share(True, alike) >= 2 * share(False, alike)


def test_the_setting_picker_starts_from_the_games_already_played_and_refuses_bad_lists():
    picker = SettingPicker(_six()[:2], counts={"2s-deeper": 4, "2s-wider": 2, "other": 9}, rng=random.Random(1))
    assert [picker("b%d" % i).name for i in range(2)] == ["2s-wider", "2s-wider"]
    assert picker.counts == {"2s-deeper": 4, "2s-wider": 4}
    with pytest.raises(ValueError):
        SettingPicker([])
    with pytest.raises(ValueError):
        SettingPicker([GameSetting(name="a", mode="mix"), GameSetting(name="a", mode="net")])


def test_setting_counts_follow_the_experiment_and_count_only_games_with_a_result():
    games = [{"setting": "a", "checkpoint": "c.pt", "experiment": "e", "won": True},
             {"setting": "a", "checkpoint": "c.pt", "experiment": "e", "won": False},
             {"setting": "b", "checkpoint": "c.pt", "experiment": "e", "won": None, "lost_track": "gone"},
             {"setting": "b", "checkpoint": "c.pt", "experiment": "other", "won": True},
             {"setting": "b", "checkpoint": "d.pt", "experiment": "e", "won": True},
             {"arm": "mix", "checkpoint": "c.pt", "experiment": "e", "won": True}]
    assert ladder.setting_counts(games, "c.pt", "e") == {"a": 2}


# =============================================================================================
# On: a game of a setting plays as a player of that setting alone
# =============================================================================================

def _alone_kwargs(s: "ladder.ArmSetting"):
    """The player ladder.make_player makes for a run started with this setting's arguments alone (as search_player kwargs)."""
    a = s.args
    kw = {"mode": a.player, "search_ms": a.search_ms, "deadline": a.deadline,
          "mix": MixSettings(min_iterations=0, trust=float(a.trust))}
    budgets = ladder.arm_budgets(a, [a.player])
    if budgets:
        kw["budgets"] = budgets
    return kw


@tsp.needs_player
def test_three_settings_at_once_each_play_as_that_setting_alone(classes):
    battles = [e for e in tsp.load_fixture() if e[-1]["username"] == "recA3ace"]
    args = load(app_settings(["2s-deeper", "6s-wider"]) + app_settings(["4s-deeper"], mode="lookahead"), ("--parallel", "3"))
    by_name = {s.name: s for s in args.arm_settings}
    bridge = tsp.FakeBridge(battles)
    try:
        picker = SettingPicker([s.game for s in args.arm_settings], rng=random.Random(2))
        # The player's own search_ms and deadline are not any setting's: a game that used them would show 20 ms.
        player = made(classes["new"], "recA3ace", bridge, mode="mix", setting_picker=picker, search_ms=20, deadline=0.5,
                      mix=MixSettings(min_iterations=0))
        picker.in_play = lambda: [t for t, b in player.battles.items() if not b.finished]
        tsp.play(player, *battles)                                         # the three battles at the same time
        together = {}
        for events in battles:
            room = events[0]["room"]
            together[room] = (player.battle_setting(room).name, what_it_did(player, bridge, events))
    finally:
        bridge.close()
    assert sorted(name for name, _ in together.values()) == sorted(by_name)  # one battle of each setting
    for events in battles:
        room = events[0]["room"]
        name, got = together[room]
        s = by_name[name]
        bridge = tsp.FakeBridge([events])
        try:
            alone = made(classes["new"], "recA3ace", bridge, **_alone_kwargs(s))
            tsp.play(alone, events)
            want = what_it_did(alone, bridge, events)
        finally:
            bridge.close()
        assert got["sent"] == want["sent"], name
        assert got["asked"] == want["asked"], name
        assert [strip(d, {"setting"}) for d in got["decisions"]] == want["decisions"], name
        assert all(d["setting"] == name for d in got["decisions"])
        assert strip(got["summary"], SETTING_FIELDS) == want["summary"], name
        summary = got["summary"]
        assert summary["setting"] == name and summary["setting_budget"] == s.game.asks()
        searched = [d for d in got["decisions"] if d["asked"]]
        assert searched and len(got["asked"]) == len(searched)
        own = s.game.own_budget()
        if own is None:                                                    # mix-trust, mix: search_ms within deadline
            assert "setting_fallback" not in summary and "budget" not in summary
            assert summary["budget_asked"] == {"setting": len(searched), "fallback": 0}
            assert all(ms == s.game.search_ms and "parallelism" not in dict(more) for _r, ms, more in got["asked"])
            assert all(d["asked_ms"] == s.game.search_ms and d["worlds_x"] == 1 for d in searched)
        else:                                                              # mix-deep: its own budget, then (fallen back) today's
            assert summary["budget"] == own.as_dict() and summary["setting_fallback"] == s.game.today()
            mine = [ms == own.search_ms and dict(more).get("parallelism") == own.worlds_x for _r, ms, more in got["asked"]]
            n = sum(mine)
            assert n >= 1 and mine == [True] * n + [False] * (len(mine) - n)    # its own budget until a fall back, if any
            assert summary["budget_asked"] == {"setting": n, "fallback": len(mine) - n}
            assert all(ms == s.game.search_ms for _r, ms, more in got["asked"][n:])
            assert (summary["deep_fallback"] is not None) == (n < len(mine))
            assert all(d["asked_ms"] == own.search_ms and d["worlds_x"] == own.worlds_x for d in searched[:n])
        trust = 0.03 if s.game.mode == "mix-trust" else s.game.trust
        assert summary.get("trust", 0.0) == trust, name
        for d in searched:                                                 # the guard reached the decision
            if d["source"] == "mix" and trust > 0:
                q = sp.merged_prior(d["probs"], d["mask"])
                top = max(range(14), key=lambda a: (q[a], -a) if d["mask"][a] else (-1.0, 0))
                assert q[d["played"]] >= trust - 1e-3 or sp.base_action(d["played"]) == sp.base_action(top), (name, d["turn"])


@tsp.needs_player
def test_a_mode_picker_and_a_setting_picker_cannot_go_together(classes):
    bridge = tsp.FakeBridge([])
    try:
        with pytest.raises(ValueError, match="cannot go together"):
            made(classes["new"], "x", bridge, mode="mix", mode_picker=sp.ArmPicker(["mix"]),
                 setting_picker=SettingPicker([GameSetting(name="a", mode="mix")]))
    finally:
        bridge.close()


def test_a_settings_game_line_says_its_setting_and_its_time_per_world():
    # ladder.py's result line of a settings game (drive's report: arm_fields with the setting's ms).
    summary = {"arm": "mix-trust", "setting": "6s-deeper", "setting_budget": {"ms": 2700, "worlds_x": 1, "threads": 1, "deadline": 7.2},
               "budget_asked": {"setting": 20, "fallback": 0}, "trust": 0.03}
    line = ladder.arm_fields(summary, "p1", int(summary["setting_budget"]["ms"]), "veto")
    assert (line["setting"], line["search_ms"], line["trust"]) == ("6s-deeper", 2700, 0.03)
    deep = {"arm": "mix-deep", "setting": "6s-wider", "budget": {"ms": 500, "worlds_x": 5, "threads": 1, "deadline": 6.7},
            "setting_budget": {"ms": 500, "worlds_x": 5, "threads": 1, "deadline": 6.7}}
    assert ladder.arm_fields(deep, "p2", 500, "veto")["search_ms"] == 500
    rows = [dict(summary, won=True, decisions=10, searched=9, fallbacks=0), dict(deep, won=False, decisions=12, searched=12, fallbacks=1)]
    text = ladder.arms_text(rows, key=lambda d: d["setting"]).splitlines()
    assert [t.split()[0] for t in text] == ["6s-deeper", "6s-wider"]


# =============================================================================================
# tools/search_sweep.py, together
# =============================================================================================

def test_the_sweep_builds_the_six_settings_with_what_each_must_record():
    settings = sweep.build_settings(sweep.read_think_table(THINK), [2.0, 4.0, 6.0], ["deeper", "wider"], "guarded")
    assert [s["name"] for s in settings] == SIX
    want = {"2s-deeper": (900, 1, 2.7), "2s-wider": (500, 2, 3.0), "4s-deeper": (1800, 1, 5.0), "4s-wider": (500, 4, 5.5),
            "6s-deeper": (2700, 1, 7.2), "6s-wider": (500, 5, 6.7)}
    for s in settings:
        assert (s["expect"]["ms"], s["expect"]["worlds_x"], s["expect"]["deadline"]) == want[s["name"]]
        assert s["expect"]["trust"] == 0.03
    checked = sweep.ladder_check(settings, 6, "public", "x.pt", "someone")
    for s, c in zip(settings, checked):                                    # ladder.py plans what the app's table says
        assert sweep.as_planned({"setting_budget": c.game.asks(), "trust": 0.03}, s["expect"]), s["name"]
    with pytest.raises(SystemExit, match="at most 6"):
        sweep.plan_together(type("A", (), {"checkpoint": "x", "games_per": 5, "server": "public"})(), settings + [dict(settings[0], name="7")],
                            "u", "guarded", "s")


def _plan(tmp_path, settings, games_per=2):
    a = type("A", (), {"checkpoint": "/x/c.pt", "games_per": games_per, "server": "local:11900"})()
    plan = sweep.plan_together(a, settings, "u", "guarded", "20261020-000000")
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return plan


def test_the_sweep_counts_reports_and_commands_by_setting(tmp_path):
    settings = sweep.build_settings(sweep.read_think_table(THINK), [2.0, 6.0], ["deeper", "wider"], "guarded")
    plan = _plan(tmp_path, settings)
    assert plan["parallel"] == 4 and plan["experiment"] == "sweep-20261020-000000"
    cmd = sweep.together_command(plan, str(tmp_path), "py", 8)
    assert cmd[cmd.index("--parallel") + 1] == "4" and cmd[cmd.index("--helpers") + 1] == "5" and cmd[cmd.index("--games") + 1] == "8"
    assert cmd[cmd.index("--arm-settings") + 1] == os.path.join(str(tmp_path), "settings.json") and "--password-stdin" in cmd
    exp = {s["name"]: s["expect"] for s in settings}
    lines = []
    for i, (name, won) in enumerate([("2s-deeper", True), ("2s-wider", False), ("6s-deeper", True), ("6s-wider", True),
                                     ("2s-deeper", False), ("6s-wider", False)]):
        e = exp[name]
        lines.append({"battle": f"b{i}", "setting": name, "experiment": plan["experiment"], "checkpoint": "c.pt", "won": won,
                      "rating_before": 1500, "rating_after": 1510, "opponent_rating": 1500, "turns": 20, "asked": 20, "fallbacks": 1,
                      "decision_ms_mean": 2000.0, "iterations_per_decision": 300000, "time": f"2026-10-20T10:{10 + i:02d}:00Z",
                      "seconds": 600.0, "trust": 0.03, "setting_budget": {"ms": e["ms"], "worlds_x": e["worlds_x"], "threads": 1,
                                                                            "deadline": e["deadline"]},
                      **({"setting_fallback": {"ms": 500}, "deep_fallback": None} if "wider" in name else {})})
    lines.append(dict(lines[0], battle="lost", won=None))                  # no result seen: played again
    lines.append(dict(lines[0], battle="old", experiment="another"))       # another sweep
    lines[3]["deep_fallback"] = {"why": "the battle timer said 80 s left (under 90)", "turn": 30}
    lines[2]["setting_budget"] = dict(lines[2]["setting_budget"], ms=500)   # not as planned
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    assert sweep.finished_counts(str(tmp_path), plan) == {"2s-deeper": 2, "2s-wider": 1, "6s-deeper": 1, "6s-wider": 2}
    text = sweep.report(str(tmp_path))
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]] for line in text.splitlines()
            if line.startswith("| ") and "s-" in line.split("|")[1]}
    assert rows["2s-deeper"][:2] == ["2", "1-1"] and rows["2s-deeper"][-1] == "2 of 2"
    assert rows["6s-deeper"][-1] == "0 of 1"
    assert rows["6s-wider"][8] == "1 of 2 games" and rows["2s-deeper"][8] == "-"
    assert rows["2s-deeper"][7] == "2 of 40 (5.0%)"
    assert "+-69 points" in text and os.path.exists(tmp_path / "REPORT.md")
    # b0 from 10:00 to 10:10, b1..b5 starting a minute after each other: (9 + 8 + 7 + 6 + 5) / 10 games beside it
    assert math.isclose(sweep.at_once([lines[0]], lines[:6])[0], 3.5)


def test_the_sweep_stops_itself_on_a_game_whose_budget_is_not_its_settings(tmp_path, capsys):
    settings = sweep.build_settings(sweep.read_think_table(THINK), [2.0, 6.0], ["deeper", "wider"], "guarded")
    plan = _plan(tmp_path, settings)
    checked = {c.name: c.game for c in sweep.ladder_check(settings, 4, "public", "x.pt", "someone")}

    def line(name, **change):
        g = checked[name]
        r = {"battle": "b-" + name, "setting": name, "won": True, "turns": 20, "rating_before": 1000, "rating_after": 1020,
             "arm": g.mode, "search_ms": g.asks()["ms"], "setting_budget": g.asks(), "trust": 0.03}
        if g.own_budget() is not None:
            r.update(budget=g.own_budget().as_dict(), setting_fallback=g.today(), deep_fallback=None)
        r.update(change)
        return r

    for name in ("2s-deeper", "2s-wider", "6s-deeper", "6s-wider"):           # what ladder.py records for its settings
        assert sweep.note_game(str(tmp_path), plan, line(name), 1), name
    fell = line("6s-wider", deep_fallback={"why": "the battle timer said 80 s left (under 90)", "turn": 31})
    assert sweep.note_game(str(tmp_path), plan, fell, 2)                      # a fall back is the setting's own guard
    assert not os.path.exists(tmp_path / "STOP")
    out = capsys.readouterr().out
    assert "recorded 2700 ms x1 within 7.2 s, trust 0.03 (mix-trust); the setting's 2700 ms x1 within 7.2 s, trust 0.03 (mix-trust): " \
           "as planned" in out and "fell back to the 1 s search at turn 31" in out
    for bad in (line("6s-deeper", search_ms=500, setting_budget=dict(line("6s-deeper")["setting_budget"], ms=500)),
                line("2s-deeper", trust=None), line("6s-wider", budget={"ms": 500, "worlds_x": 1, "threads": 1, "deadline": 2.5}),
                line("2s-wider", arm="mix")):
        assert not sweep.check_game(bad, {s["name"]: s for s in settings}[bad["setting"]])[1], bad
    assert not sweep.note_game(str(tmp_path), plan, line("2s-deeper", trust=None), 3)
    assert "MISMATCH" in capsys.readouterr().out and "2s-deeper" in (tmp_path / "STOP").read_text(encoding="utf-8")


def test_the_sweep_dry_run_prints_the_one_ladder_command_and_writes_nothing(tmp_path):
    ckpt = tmp_path / "net.pt"
    ckpt.write_bytes(b"not a network")
    before = set(os.listdir(os.path.join(REPO, "runs", "sweep"))) if os.path.isdir(os.path.join(REPO, "runs", "sweep")) else set()
    out = subprocess.run([sys.executable, os.path.join(REPO, "tools", "search_sweep.py"), "--dry-run", "--think-table", THINK,
                          "--mode", "guarded", "--user", "someone", "--checkpoint", str(ckpt)],
                         capture_output=True, text=True, timeout=120, cwd=REPO)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "30 games, all 6 settings at once in one ladder.py (6 games at once, 7 search helpers)" in out.stdout
    command = [l for l in out.stdout.splitlines() if l.startswith("command:")][0]
    assert "--parallel 6 --helpers 7 --arm-settings" in command and "--games 30" in command and "--server public" in command
    after = set(os.listdir(os.path.join(REPO, "runs", "sweep"))) if os.path.isdir(os.path.join(REPO, "runs", "sweep")) else set()
    assert after == before
    seq = subprocess.run([sys.executable, os.path.join(REPO, "tools", "search_sweep.py"), "--dry-run", "--sequential", "--think-table",
                          THINK, "--mode", "guarded", "--user", "someone", "--checkpoint", str(ckpt), "--seed", "1"],
                         capture_output=True, text=True, timeout=120, cwd=REPO)
    assert seq.returncode == 0 and "30 games, one at a time, in 5 shuffled rounds of 6" in seq.stdout, seq.stdout + seq.stderr
