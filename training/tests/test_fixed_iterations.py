"""
Foul Play at a fixed number of iterations (evaluate.py --foulplay-iterations N, bench/foulplay.py
fixed_iterations; bench/notes/FIXED_ITERATIONS.md).

    python -m pytest tests/test_fixed_iterations.py -q -p no:cacheprovider

Held here:
  * the child replaces Foul Play's search call so that poke-engine gets iterations=N whatever
    duration Foul Play passes, and says so in its log; its own option never reaches run.py;
  * run_matches with fixed_iterations: the child command carries the option, the result says N
    and the smallest and largest count of every "Iterations" line, and counts lines that are not
    N. Without it the command, the result and the row are what they were;
  * evaluate.py: every Foul Play row of an evaluation at N says N (also a row of a match that
    failed), a row whose logs show another count is not trusted, the report never compares or
    pools rows of different N (pool() refuses outright), and the slow-search warning leaves rows
    at fixed iterations alone;
  * watch mode: eval.foulplay.iterations reaches the evaluation's command line, and a config
    without it starts the same command as before.

The fake Foul Play of tests/test_foulplay.py stands in for the real one; the real one was run on
the box (bench/notes/FIXED_ITERATIONS.md has the smoke run). Nothing here deletes anything.
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate as ev  # noqa: E402
from bench import foulplay  # noqa: E402
import test_evaluate as te  # noqa: E402
import test_foulplay as tf  # noqa: E402
from test_foulplay import fake  # noqa: E402,F401  (the fixture)

N = 20000


# --------------------------------------------------------------------------- the child side


def test_the_child_hands_poke_engine_exactly_n_iterations(monkeypatch, capsys):
    calls = []

    def original(state, duration_ms=1000, iterations=0, threads=1):
        calls.append((state, duration_ms, iterations, threads))
        return types.SimpleNamespace(total_visits=iterations or duration_ms * 700)

    fp_pkg = types.ModuleType("fp")
    search_pkg = types.ModuleType("fp.search")
    search_main = types.ModuleType("fp.search.main")
    search_main.monte_carlo_tree_search = original
    fp_pkg.search, search_pkg.main = search_pkg, search_main
    for name, mod in (("fp", fp_pkg), ("fp.search", search_pkg), ("fp.search.main", search_main)):
        monkeypatch.setitem(sys.modules, name, mod)
    asked = []
    import multiprocessing
    monkeypatch.setattr(multiprocessing, "set_start_method", lambda method, *a, **k: asked.append(method))
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none=False: None)

    foulplay._fix_iterations(N)
    # Foul Play's own call (fp/search/main.py get_result_from_mcts): the duration positionally, threads by name
    res = search_main.monte_carlo_tree_search("state", 200, threads=1)
    assert calls == [("state", 200, N, 1)] and res.total_visits == N
    search_main.monte_carlo_tree_search("s2", 100, 5, 3)          # an explicit count is overridden too
    assert calls[-1] == ("s2", 100, N, 3)
    out = capsys.readouterr().out
    assert foulplay._RE_FIXED.match(out.splitlines()[-1]) and "Fixed iterations: %d " % N in out
    if sys.platform.startswith("linux"):
        assert asked == ["fork"]                                  # the search workers must inherit the replacement


def test_a_bad_child_option_stops_before_anything_happens(capsys):
    assert foulplay._child_main([os.path.join(REPO, "no_such_dir"), "--fixed-iterations", "many"]) == 2
    assert foulplay._child_main([os.path.join(REPO, "no_such_dir"), "--fixed-iterations"]) == 2
    assert "--fixed-iterations N" in capsys.readouterr().out


def test_the_command_carries_the_option_only_when_asked():
    info = {"python": "PY", "dir": "/fp", "commit": "x"}
    player = types.SimpleNamespace(username="tester", battles={})
    plain = foulplay._Run(player, info, "127.0.0.1", 1, 200, 1, "/logs", "gen9randombattle", 240.0, 60.0, 2)
    fixed = foulplay._Run(player, info, "127.0.0.1", 1, 200, 1, "/logs", "gen9randombattle", 240.0, 60.0, 2, N)
    a, b = plain.command("FPs0r0", 3), fixed.command("FPs0r0", 3)
    assert "--fixed-iterations" not in a
    assert b[:6] == a[:5] + ["--fixed-iterations"] and b[6] == str(N) and b[7:] == a[5:]
    assert a == ["PY", "-u", foulplay._THIS_FILE, "--child", "/fp", "--websocket-uri", "ws://127.0.0.1:1/showdown/websocket",
                 "--ps-username", "FPs0r0", "--bot-mode", "challenge_user", "--user-to-challenge", "tester",
                 "--pokemon-format", "gen9randombattle", "--search-time-ms", "200", "--search-parallelism", "1",
                 "--run-count", "3", "--log-level", "INFO"]


def test_the_log_is_checked_line_by_line():
    p = foulplay._Proc(0, 0, "FPs0r0", 1, None, "/x", fixed_iterations=N)
    for line in ("INFO     Fixed iterations: %d per sampled world (search time ignored)" % N,
                 "INFO     Iterations 0: %d" % N, "INFO     Iterations 1: %d" % N, "INFO     Iterations 0: 19999",
                 "INFO     Iterations 0: 1000"):
        p.parse_line(line)
    # 1,000: a root where Foul Play had one option (a forced switch) ends after poke-engine's first batch, in either mode
    assert (p.fixed_said, p.iterations_min, p.iterations_max, p.iterations_off, p.iterations_one_option, p.iterations_n) ==         (N, 19999, N, 1, 1, 4)
    q = foulplay._Proc(0, 0, "FPs0r0", 1, None, "/x")
    q.parse_line("INFO     Iterations 0: 5")
    q.parse_line("INFO     Fixed iterations: 5 per sampled world (search time ignored)")
    assert (q.fixed_said, q.iterations_min, q.iterations_off, q.iterations_n) == (None, None, 0, 1)


# --------------------------------------------------------------------------- through the fake Foul Play

FAKE_SEARCH = '''
class _Result:
    def __init__(self, n):
        self.total_visits = n


def monte_carlo_tree_search(state, duration_ms=1000, iterations=0, threads=1):
    return _Result({result})
'''


def _with_search(ns, honest: bool = True) -> None:
    """The fake checkout gets fp/search/main.py, and its games log what that search returns."""
    tf.write(os.path.join(ns.checkout, "fp", "search", "__init__.py"), "")
    # honest=False: a search that ignores the count it is given (the replacement did not take)
    tf.write(os.path.join(ns.checkout, "fp", "search", "main.py"),
             FAKE_SEARCH.replace("{result}", "iterations if iterations > 0 else duration_ms * 700" if honest else "777"))
    main = tf.FAKE_MAIN.replace(
        '        print("INFO     Iterations 0: 1000", flush=True)\n        print("INFO     Iterations 0: 1000", flush=True)\n',
        '        import fp.search.main as sm\n'
        '        visits = sm.monte_carlo_tree_search(None, 200, threads=1).total_visits\n'
        '        print("INFO     Iterations 0: %d" % visits, flush=True)\n'
        '        print("INFO     Iterations 0: %d" % visits, flush=True)\n')
    assert main != tf.FAKE_MAIN
    tf.write(os.path.join(ns.checkout, "fp", "main.py"), main)


def test_run_matches_at_fixed_iterations(fake):
    _with_search(fake)
    result, _ = tf.play(fake, 3, parallel=2, fixed_iterations=N)
    assert result["games"] == 3 and result["errors"] == 0, result
    assert (result["fixed_iterations"], result["iterations_min"], result["iterations_max"], result["iterations_off"],
            result["iterations_one_option"]) == (N, N, N, 0, 0)
    assert result["foulplay_mean_iterations"] == N and "iterations_unconfirmed" not in result
    text = tf.logs_text(fake)
    assert "Fixed iterations: %d per sampled world" % N in text
    argv = [line for line in text.splitlines() if "argv: " in line]
    assert argv and not any("--fixed-iterations" in line for line in argv)      # run.py never sees the child's own option


def test_run_matches_on_the_search_time_is_unchanged(fake):
    _with_search(fake)
    result, _ = tf.play(fake, 2, parallel=2)
    assert result["games"] == 2 and result["foulplay_mean_iterations"] == 200 * 700
    for key in ("fixed_iterations", "iterations_min", "iterations_max", "iterations_off", "iterations_unconfirmed"):
        assert key not in result
    assert "Fixed iterations" not in tf.logs_text(fake)


def test_a_count_poke_engine_cannot_search_exactly_is_refused(fake):
    result, _ = tf.play(fake, 2, parallel=2, fixed_iterations=N + 500)
    assert result["games"] == 0 and result["errors"] == 2 and "not a multiple of 1000" in result["error"], result
    assert os.listdir(fake.logs) == []                                     # nothing was started


def test_a_search_that_ignores_the_count_is_caught(fake):
    _with_search(fake, honest=False)
    result, _ = tf.play(fake, 2, parallel=2, fixed_iterations=N)
    assert result["games"] == 2 and result["iterations_off"] == 4 and result["iterations_min"] == 777
    tally = ev.foulplay_tally(result, 2)
    row = ev.result_line(te.ResultLineTests.META, "foulplay", tally, sample=False, extras={"foulplay_iterations": N})
    assert row["valid"] is False and any("not a fixed-iteration measurement" in p for p in row["problems"]), row["problems"]


# --------------------------------------------------------------------------- evaluate.py


def _fp_result(n=N, off=0, games=20, wins=8):
    out = {"games": games, "wins": wins, "losses": games - wins, "ties": 0, "errors": 0, "seconds": 300.0,
           "search_time_ms": 200, "parallel": 4, "foulplay_commit": "8f541f8", "log_agrees": True,
           "foulplay_mean_iterations": n or 88000}
    if n:
        out.update({"fixed_iterations": n, "iterations_min": n - (1 if off else 0), "iterations_max": n, "iterations_off": off})
    return out


def _subject():
    return types.SimpleNamespace(battles={}, audit={"requests": 500, "invalid_choices": 0, "unavailable_choices": 0,
                                                    "other_errors": 0, "exceptions": 0, "forced_random": 0},
                                 counts={"decisions": 500, "fallbacks": 0}, audit_endings={}, audit_messages=[])


def test_the_runner_is_asked_for_n_and_the_row_says_n(tmp_path):
    calls = []

    async def run_matches(player, n_games, host, port, *, parallel=4, search_time_ms=200, log_dir=None, fixed_iterations=0):
        calls.append(fixed_iterations)
        return _fp_result(fixed_iterations)

    name = te.fake_module("fake_fp_fixed", lambda: (True, "foul-play 8f541f8 at /x"), run_matches)
    ctx = te.make_ctx(str(tmp_path), name)
    ctx.foulplay_iterations = N
    tally, _text = asyncio.run(ev.play_foulplay(ctx, 20, make_subject=_subject))
    assert calls == [N]
    row = ev.result_line(te.ResultLineTests.META, "foulplay", tally, sample=False, extras={"foulplay_iterations": N})
    assert (row["foulplay_iterations"], row["foulplay_iterations_min"], row["foulplay_iterations_max"], row["foulplay_iterations_off"]) == (N, N, N, 0)
    assert row["valid"] is True and row["problems"] == []
    assert ev.play_settings(row) == (200, 4, "8f541f8", N)


def test_without_the_option_the_runner_and_the_row_are_as_before(tmp_path):
    calls = []

    async def run_matches(player, n_games, host, port, *, parallel=4, search_time_ms=200, log_dir=None):   # the old contract
        calls.append(search_time_ms)
        return _fp_result(0)

    name = te.fake_module("fake_fp_plain", lambda: (True, "foul-play 8f541f8 at /x"), run_matches)
    tally, _text = asyncio.run(ev.play_foulplay(te.make_ctx(str(tmp_path), name), 20, make_subject=_subject))
    assert calls == [200]
    row = ev.result_line(te.ResultLineTests.META, "foulplay", tally, sample=False)
    assert not [k for k in row if k.startswith("foulplay_iterations")]
    assert ev.play_settings(row) == (200, 4, "8f541f8") and ev.fixed_iterations_of(row) == 0 and row["valid"] is True


def test_every_row_of_an_evaluation_at_n_says_n_even_when_foul_play_failed(tmp_path, monkeypatch):
    async def run_matches(player, n_games, host, port, **kw):
        raise RuntimeError("foul play fell over")

    async def nothing(*_a, **_k):
        return None

    monkeypatch.setattr(ev, "_make_subject", lambda *_a, **_k: types.SimpleNamespace())
    monkeypatch.setattr(ev, "_wait_logged_in", nothing)
    monkeypatch.setattr(ev, "_disconnect", nothing)
    name = te.fake_module("fake_fp_fixed_raises", lambda: (True, "v"), run_matches)
    ckpt = ev.stat_ckpt(te.write_file(os.path.join(str(tmp_path), "run", "checkpoints", "learner_update_000001.pt")))
    paths = ev.RunPaths(os.path.join(str(tmp_path), "run"))
    ctx = te.make_ctx(str(tmp_path), name, ckpt)
    ctx.foulplay_iterations = N
    args = types.SimpleNamespace(also_hybrid=False, games=10, foulplay_games=5)
    meta = {"checkpoint": ckpt.name, "update": 1, "phase": "ppo", "total_steps": 1}
    lines, _code = asyncio.run(ev._run_evaluation(args, paths, ctx, meta, ["foulplay"], {}))
    assert len(lines) == 1 and lines[0]["foulplay_iterations"] == N and lines[0]["games"] == 0
    assert ev.read_jsonl(paths.results)[0]["foulplay_iterations"] == N


def test_rows_whose_logs_show_another_count_are_not_trusted():
    bad = ev.result_line(te.ResultLineTests.META, "foulplay", ev.foulplay_tally(_fp_result(N, off=3), 20), sample=False)
    assert bad["valid"] is False and "not a fixed-iteration measurement" in "; ".join(bad["problems"])
    unread = _fp_result(N)
    for key in ("iterations_min", "iterations_max"):
        unread.pop(key)
    row = ev.result_line(te.ResultLineTests.META, "foulplay", ev.foulplay_tally(unread, 20), sample=False)
    assert row["valid"] is False and "no search to check that by" in "; ".join(row["problems"])


def _row(n, t, games=200, wins=100, name="learner_update_000010.pt", mtime=1000.0):
    extra = {"opponent_info": {"search_time_ms": 200, "parallel": 4, "foulplay_commit": "8f541f8"}}
    if n:
        extra.update(foulplay_iterations=n, foulplay_iterations_min=n, foulplay_iterations_max=n, foulplay_iterations_off=0)
    return te.line("foulplay", games=games, wins=wins, t=t, name=name, mtime=mtime, **extra)


def test_the_report_never_pools_or_compares_rows_of_different_n():
    a = _row(N, "2026-10-15T00:00:00Z", wins=90, name="learner_update_000010.pt", mtime=1.0)
    b = _row(N, "2026-10-15T01:00:00Z", wins=110, name="learner_update_000020.pt", mtime=2.0)
    other = _row(90000, "2026-10-15T02:00:00Z", wins=150, name="learner_update_000030.pt", mtime=3.0)
    timed = _row(0, "2026-10-15T03:00:00Z", wins=60, name="learner_update_000040.pt", mtime=4.0)
    assert all(r["valid"] for r in (a, b, other, timed))
    assert ev.pool([a, b])["games"] == 400
    for rows in ([a, other], [a, timed], [a, b, other]):
        with pytest.raises(ValueError, match="different fixed iterations"):
            ev.pool(rows)
    assert ev.pool([te.line("simple"), te.line("simple")])["games"] == 400          # other opponents: as before
    # the newest row at N=90,000 has nothing at its own N to be compared with
    v = ev.opponent_verdict([a, b, other])
    assert v["latest"] is other and v["previous"] is None and v["other_settings"] == 2
    v = ev.opponent_verdict([a, b])
    assert v["latest"] is b and v["previous"] is a and v["vs_previous"] is not None
    v = ev.opponent_verdict([a, timed])
    assert v["latest"] is timed and v["previous"] is None and v["other_settings"] == 1
    report = ev.build_report([a, b, other, timed])
    assert "20,000 iterations fixed" in report and "90,000 iterations fixed" in report
    assert "its fixed iterations" in report


def test_the_slow_search_warning_leaves_fixed_rows_alone():
    slow = _row(0, "2026-10-15T00:00:00Z")
    slow["foulplay_mean_iterations"] = 100000
    fixed_rows = [_row(N, "2026-10-15T01:00:00Z") for _ in range(3)]
    for r in fixed_rows:
        r["foulplay_mean_iterations"] = N
    # a match at its search time is compared with the earlier ones at its search time only
    assert ev.foulplay_slow_search(60000, 200, [slow] + fixed_rows) is None
    assert "less than half" in ev.foulplay_slow_search(40000, 200, [slow] + fixed_rows)
    # a match at fixed iterations is never "slow"
    assert ev.foulplay_slow_search(5, 200, [slow], fixed_iterations=N) is None


def test_watch_mode_passes_n_on_and_leaves_the_command_alone_without_it():
    cfg, warnings = ev.eval_config({"eval": {"foulplay": {"games": 50, "iterations": N}}})
    assert warnings == [] and cfg["foulplay"]["iterations"] == N
    plain, warnings = ev.eval_config({"eval": {"foulplay": {"games": 50}}})
    assert warnings == [] and "iterations" not in plain["foulplay"]
    zero, warnings = ev.eval_config({"eval": {"foulplay": {"iterations": 0}}})
    assert warnings == [] and "iterations" not in zero["foulplay"]
    for bad in ("lots", 20500, -1000):
        cfg_bad, warnings = ev.eval_config({"eval": {"foulplay": {"iterations": bad}}})
        assert any("eval.foulplay.iterations" in w for w in warnings) and "iterations" not in cfg_bad["foulplay"], bad
    over, _ = ev.eval_config({}, {"foulplay": {"iterations": 30000}})
    assert over["foulplay"]["iterations"] == 30000

    ck = ev.Ckpt("/run/checkpoints/learner_update_000010.pt", "learner_update_000010.pt", 5.0, 1)
    me = types.SimpleNamespace(paths=types.SimpleNamespace(run="/run"),
                               args=types.SimpleNamespace(foulplay_module=ev.FOULPLAY_MODULE, single_inference=False))
    plan = ev.Plan("evaluate", "interval", ck, ("simple", "foulplay"))
    snap = ev.Snapshot(now=0.0, phase=None, newest=ck, n_checkpoints=1, results=(), final_done=None)
    with_n = ev.Watcher.command(me, plan, cfg, snap)
    without = ev.Watcher.command(me, plan, plain, snap)
    assert with_n[-2:] == ["--foulplay-iterations", str(N)] and with_n[:-2] == without
    assert "--foulplay-iterations" not in without


def test_the_command_line_option(capsys):
    args = ev.build_parser().parse_args(["--run-dir", "/r", "--foulplay-iterations", str(N)])
    assert args.foulplay_iterations_override == N
    assert ev.build_parser().parse_args(["--run-dir", "/r"]).foulplay_iterations_override is None
    for bad in ("-1000", "20500"):
        assert ev.main(["--run-dir", "/r", "--foulplay-iterations", bad]) == ev.EXIT_USAGE
        assert "--foulplay-iterations must be 0 (off) or a multiple of 1,000" in capsys.readouterr().out
