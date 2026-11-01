"""
What worker.py does when the server calls a choice invalid (bench/MEMORY.md).

Release r1 answered every "|error|[Invalid choice]" twice: its patch of poke-env's handler sent
"/choose default", and poke-env asked the player again and sent what it chose. The server takes
the first; the second replaces it or lands on the next turn, which leaves the client a decision
behind the server, and once the side can choose nothing (it has a wait request, the turn is
being played, the game is over) every choice it still sends is refused and every refusal was
answered twice again. One refused choice in a training run on the laptop made 28,280 round
trips in 18 seconds and put 44 steps that were never played into the battle's episode; with
scripted players on a real server, 43,122 in 16 seconds (bench/MEMORY.md).

Now a refused choice gets one answer, "/choose default", when a default can be taken, at most
once a request, and the policy is not asked a second time. The battle's episode is then the
episode of the same battle without the error.

Same harness as tests/test_worker_memory.py: both versions of worker.py, real poke-env players
that never connect, recorded battle streams, the real RolloutWorker loop, no Ray. The server's
side is played from server/room-battle.ts (choose(): a side that has a wait request, or whose
turn is being played, or whose game is over is told "[Invalid choice] There's nothing to
choose" or "Sorry, too late ..." for everything it sends).
"""
from __future__ import annotations

import pytest

from test_worker_memory import (Harness, NOTHING, cut_after_requests, ending, recorded_battles, same_episodes,  # noqa: F401
                                tmp_root, worker_module)
from test_worker_memory import pytestmark  # noqa: F401


REFUSED = [
    "|error|[Invalid choice] There's nothing to choose",
    "|error|[Invalid choice] Sorry, too late to make a different move; the next turn has already started",
    "|error|[Invalid choice] Can't do anything: It's not your turn",
    "|error|[Invalid choice] Can't do anything: The game is over",
    "|error|[Invalid choice] Can't undo: A trapping/disabling effect would cause undo to leak information",
]
REPAIRABLE = [
    "|error|[Invalid choice] Can't move: Your Garchomp doesn't have a move matching earthpower",
    "|error|[Invalid choice] Can't switch: You can't switch to an active Pokémon",
    "|error|[Invalid choice] Can't move: You can only Terastallize once per battle.",
    "|error|[Invalid choice] The battle crashed",
]
WAIT_REQUEST = '|request|{"wait":true,"side":{"name":"x","id":"p1","pokemon":[]},"rqid":99}'


def harness(mod):
    h = Harness(mod)
    return h


def player_of(h, username):
    player = h.player(username)
    # poke-env sends a default instead of asking again once in a thousand times: not in a test.
    player.DEFAULT_CHOICE_CHANCE = 0.0
    return player


def steps(h):
    return sum(len(t["act"]) for t in h.worker._traj.values())


def defaults_held(h):
    """Rooms that are still marked as having had their default (per client): must be 0 when battles are over."""
    return sum(len(p.ps_client.__dict__.get("_default_sent", ())) for p in h.players.values())


def test_which_errors_a_default_cannot_repair(tmp_root):
    mod = worker_module("new", tmp_root)
    for line in REFUSED:
        assert mod.default_would_be_refused(">battle-gen9randombattle-1\n" + line)
    for line in REPAIRABLE:
        assert not mod.default_would_be_refused(">battle-gen9randombattle-1\n" + line)
    # One of each in a message: the repairable one is answered.
    assert not mod.default_would_be_refused(">battle-gen9randombattle-1\n" + REFUSED[0] + "\n" + REPAIRABLE[0])
    assert not mod.default_would_be_refused(">battle-gen9randombattle-1\n|turn|3")
    # "[Unavailable choice]" is not an invalid choice: neither version answers it with a default.
    assert not mod.default_would_be_refused(">battle-gen9randombattle-1\n|error|[Unavailable choice] Can't switch: The active Pokémon is trapped")


def _after(mod, username, room, messages, line):
    """Plays a stream up to its third request, then the error line. -> (messages sent because of it, steps added)"""
    h = harness(mod)
    try:
        player = player_of(h, username)
        h.feed(username, room, cut_after_requests(messages, 3))
        h.settle()
        sent_before, steps_before = len(player.sent), steps(h)
        h.feed(username, room, [line], timeout=30.0)
        h.settle()
        return [m for _room, m in player.sent[sent_before:]], steps(h) - steps_before
    finally:
        h.close()


@pytest.mark.parametrize("line", REPAIRABLE)
def test_a_refused_choice_is_answered_with_one_default_and_the_policy_is_not_asked_again(tmp_root, line):
    username, room, messages = recorded_battles()[0]
    new_sent, new_steps = _after(worker_module("new", tmp_root), username, room, messages, line)
    old_sent, old_steps = _after(worker_module("r1", tmp_root), username, room, messages, line)
    assert new_sent == ["/choose default"] and new_steps == 0
    # r1: the same default first, then a second choice from a second asking, recorded as a step.
    assert old_sent[0] == "/choose default" and len(old_sent) == 2 and old_sent[1].startswith("/choose ")
    assert old_steps == 1


@pytest.mark.parametrize("line", REFUSED)
def test_an_error_that_says_nothing_can_be_chosen_is_not_answered_at_all(tmp_root, line):
    username, room, messages = recorded_battles()[0]
    new_sent, new_steps = _after(worker_module("new", tmp_root), username, room, messages, line)
    old_sent, old_steps = _after(worker_module("r1", tmp_root), username, room, messages, line)
    assert new_sent == [] and new_steps == 0
    assert old_sent[0] == "/choose default" and len(old_sent) == 2 and old_steps == 1      # two choices that must be refused


def test_an_unavailable_choice_is_left_to_poke_env_as_before(tmp_root):
    """"[Unavailable choice]" (a trapped switch): the server sends a new request, and that asks again. Untouched."""
    username, room, messages = recorded_battles()[0]
    line = "|error|[Unavailable choice] Can't switch: The active Pokémon is trapped"
    new = _after(worker_module("new", tmp_root), username, room, messages, line)
    old = _after(worker_module("r1", tmp_root), username, room, messages, line)
    assert new == old == ([], 0)


def _bounce(mod, username, room, messages, first_error, then_wait_request, limit=200):
    """
    The third request is answered; the server refuses that choice with `first_error` and from then
    on behaves as room-battle.ts does: the first choice that arrives is taken (and, with
    `then_wait_request`, completes the turn: the other side must switch, this side gets a wait
    request); every later choice is refused with "There's nothing to choose", for as long as the
    other side does not move. It never moves here. -> (choices sent after the first error, steps added)
    """
    h = harness(mod)
    try:
        player = player_of(h, username)
        h.feed(username, room, cut_after_requests(messages, 3))
        h.settle()
        seen, steps_before = len(player.sent), steps(h)
        h.feed(username, room, [first_error], timeout=30.0)
        h.settle()
        taken, count = False, 0
        while count < limit:
            choices = [m for _r, m in player.sent[seen:] if m.startswith("/choose")]
            seen = len(player.sent)
            if not choices:
                break
            for _choice in choices:
                if count >= limit:
                    break
                count += 1
                if not taken:
                    taken = True                                  # the side's move for this turn
                    if then_wait_request:
                        h.feed(username, room, [WAIT_REQUEST], timeout=30.0)
                    continue
                h.feed(username, room, ["|error|[Invalid choice] There's nothing to choose"], timeout=30.0)
            h.settle()
        return count, steps(h) - steps_before
    finally:
        h.close()


@pytest.mark.parametrize("then_wait_request", [True, False])
def test_one_refused_choice_ends_with_one_default(tmp_root, then_wait_request):
    """
    The flood (bench/MEMORY.md; what swells the cloud box's workers, as far as could be told from
    outside them). A side that can choose nothing: r1 sends a default and a second choice, the
    second is refused, that refusal is answered again, without end (cut off at 200 here; 43,122 in
    16 s with scripted players on a real server), and while the side has no wait request every
    refusal also asks the policy again: a step in the episode each time. Now: the default, and
    that is all.
    """
    username, room, messages = recorded_battles()[0]
    error = REPAIRABLE[0]
    new = _bounce(worker_module("new", tmp_root), username, room, messages, error, then_wait_request)
    old = _bounce(worker_module("r1", tmp_root), username, room, messages, error, then_wait_request)
    assert new == (1, 0)                         # one choice sent (the default), no step that was not played
    assert old[0] == 200 and old[1] >= 1         # without end, and askings recorded as steps
    if not then_wait_request:
        assert old[1] >= 50                      # no wait request: every refusal is one more asking, one more step


def test_a_stray_choice_while_waiting_is_refused_once(tmp_root):
    """A side with a wait request that sends a choice is refused every time. r1 answered each refusal with a default."""
    username, room, messages = recorded_battles()[0]
    bounces = {}
    for which in ("r1", "new"):
        h = harness(worker_module(which, tmp_root))
        try:
            player = player_of(h, username)
            h.feed(username, room, cut_after_requests(messages, 3))
            h.settle()
            h.feed(username, room, [WAIT_REQUEST])
            seen = len(player.sent)
            player.sent.append((room, "/choose default"))        # one stray choice while waiting
            count = 0
            while count < 200:                                   # the other side never moves: cut off at 200
                new_choices = [m for _r, m in player.sent[seen:] if m.startswith("/choose")]
                seen = len(player.sent)
                if not new_choices:
                    break
                for _ in new_choices:
                    count += 1
                    h.feed(username, room, ["|error|[Invalid choice] There's nothing to choose"], timeout=30.0)
            bounces[which] = count
        finally:
            h.close()
    assert bounces["r1"] == 200          # without end
    assert bounces["new"] == 1           # the stray choice is refused once, and that is all


def test_one_default_a_request_whatever_the_error_says(tmp_root):
    """
    No error text can start a bounce: a room gets one default per request. ("The battle crashed"
    is an invalid choice that a default does not repair and that is not on the list.)
    """
    mod = worker_module("new", tmp_root)
    username, room, messages = recorded_battles()[0]
    crashed = REPAIRABLE[3]
    h = harness(mod)
    try:
        player = player_of(h, username)
        three, four = cut_after_requests(messages, 3), cut_after_requests(messages, 4)
        h.feed(username, room, three)
        h.settle()
        seen = len(player.sent)
        for _ in range(5):
            h.feed(username, room, [crashed], timeout=30.0)
        assert [m for _r, m in player.sent[seen:]] == ["/choose default"]
        assert defaults_held(h) == 1 and player.choices_refused == 5
        # The next request: the policy is asked as always, and a refusal of THAT choice gets its default.
        steps_before = steps(h)
        h.feed(username, room, four[len(three):])
        h.settle()
        assert steps(h) == steps_before + 1 and defaults_held(h) == 0
        seen = len(player.sent)
        h.feed(username, room, [REPAIRABLE[0], REPAIRABLE[0]], timeout=30.0)
        assert [m for _r, m in player.sent[seen:]] == ["/choose default"]
        # The battle ends: nothing is kept for the room, the mark included.
        h.feed(username, room, ending("win", username, messages)[len(four):])
        h.settle()
        assert h.kept() == NOTHING and defaults_held(h) == 0
        assert len(h.learner.episodes) == 1
    finally:
        h.close()


def _play_with_error(mod, username, room, messages, after_request, lines, kind="win"):
    """The whole recorded battle with `lines` put in after its n-th request has been answered. -> harness"""
    h = harness(mod)
    try:
        player_of(h, username)
        stream = ending(kind, username, messages)
        head = cut_after_requests(messages, after_request)
        h.feed(username, room, head)
        h.settle()
        if lines:
            h.feed(username, room, lines, timeout=30.0)
            h.settle()
        h.feed(username, room, stream[len(head):])
        h.settle()
        return h
    finally:
        h.close()


@pytest.mark.parametrize("kind", ["win", "tie", "forfeit"])
@pytest.mark.parametrize("lines", [[REPAIRABLE[0]], [REFUSED[0]], [REPAIRABLE[1], REFUSED[0], REFUSED[1], REFUSED[0]]])
def test_the_episode_of_a_battle_with_a_refused_choice_is_the_episode_without_it(tmp_root, kind, lines):
    """
    What the learner is handed. A refused choice adds no step and changes no later observation:
    the episode is, array for array, the one of the same battle without the error (which is
    r1's episode for that battle, tests/test_worker_memory.py). r1 with the error hands over the
    same battle with one step more for every time it asked again.
    """
    username, room, messages = recorded_battles()[0]
    new_mod, old_mod = worker_module("new", tmp_root), worker_module("r1", tmp_root)
    clean = _play_with_error(new_mod, username, room, messages, 3, [], kind)
    clean_r1 = _play_with_error(old_mod, username, room, messages, 3, [], kind)
    with_error = _play_with_error(new_mod, username, room, messages, 3, lines, kind)
    r1_with_error = _play_with_error(old_mod, username, room, messages, 3, lines, kind)
    assert len(clean.learner.episodes) == 1
    same_episodes(clean.learner.episodes, clean_r1.learner.episodes)
    same_episodes(with_error.learner.episodes, clean.learner.episodes)
    assert len(r1_with_error.learner.episodes[0][1]) == len(clean.learner.episodes[0][1]) + len(lines)
    assert with_error.kept() == NOTHING and defaults_held(with_error) == 0
    assert sum(p.choices_refused for p in with_error.players.values()) == len(lines)


def test_a_refused_choice_in_a_room_the_server_then_closes_leaves_nothing(tmp_root):
    username, room, messages = recorded_battles()[0]
    h = _play_with_error(worker_module("new", tmp_root), username, room, messages, 3, [REPAIRABLE[0]], "expire")
    assert h.learner.episodes == [] and h.kept() == NOTHING and defaults_held(h) == 0


# ---------------------------------------------------------------------------------------------
# after the review: a refusal that shares its message, and what the trainer's log is told
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("where", ["before", "after"])
def test_a_refusal_that_shares_its_message_with_other_lines_takes_only_itself_away(tmp_root, where):
    """
    The server sends an "[Invalid choice]" line alone. Should it ever come in one message with
    other lines, only the error line is kept from poke-env: here it is put in front of, and
    behind, the lines of the message that follows a request, and the episode has to be the one
    of the battle without it. (The review's deliberate break "the whole message is dropped"
    passed every test before this one.)
    """
    mod = worker_module("new", tmp_root)
    username, room, messages = recorded_battles()[0]
    clean = _play_with_error(mod, username, room, messages, 3, [], "win")
    n_requests = sum(1 for m in messages for line in m.split("\n") if line.startswith("|request|{") and '"wait":true' not in line)
    assert n_requests > 6
    for k in (1, 3, n_requests - 1):
        h = harness(mod)
        try:
            player_of(h, username)
            stream = ending("win", username, messages)
            head = len(cut_after_requests(messages, k))
            nxt = stream[head]
            assert "|error|" not in nxt and nxt.strip()
            shared = REPAIRABLE[0] + "\n" + nxt if where == "before" else nxt + "\n" + REPAIRABLE[0]
            h.feed(username, room, stream[:head])
            h.settle()
            h.feed(username, room, [shared] + stream[head + 1:])
            h.settle()
            same_episodes(h.learner.episodes, clean.learner.episodes)
            assert [m for _r, m in h.players[username].sent].count("/choose default") == \
                [m for _r, m in clean.players[username].sent].count("/choose default") + 1
            assert h.kept() == NOTHING and defaults_held(h) == 0
            assert sum(p.choices_refused for p in h.players.values()) == 1
        finally:
            h.close()


def test_a_refused_choice_and_a_closed_room_each_print_one_line_and_never_many(tmp_root, capsys):
    """
    The trainer's log is told, in the server's own words, of every choice it refuses and of
    every room it closes: the first ten of a kind in a worker, then every thousandth (a wrong
    legal-move mask must not fill the log). The lines change nothing that is handed over
    (the comparisons with r1 above and in tests/test_worker_memory.py run with them).
    """
    mod = worker_module("new", tmp_root)
    username, room, messages = recorded_battles()[0]
    saved = dict(mod.REPORTED)
    mod.REPORTED.clear()
    try:
        capsys.readouterr()
        h = _play_with_error(mod, username, room, messages, 3, [REPAIRABLE[0], REFUSED[0]], "expire")
        err = capsys.readouterr().err
        lines = [line for line in err.split("\n") if line.startswith("[rollout worker ")]
        refused = [line for line in lines if "] choice refused, number " in line]
        closed = [line for line in lines if "] room closed, number " in line]
        assert len(refused) == 2 and len(closed) == 1, lines
        assert refused[0].endswith(REPAIRABLE[0].split("|error|")[1]) and f"{username} {room} (turn " in refused[0], refused[0]
        assert "number 1 in this worker" in refused[0] and "number 2 in this worker" in refused[1]
        assert f"{username} {room} at turn " in closed[0] and "after its first decision" in closed[0], closed[0]
        assert h.learner.episodes == [] and h.kept() == NOTHING
        # never many: 5,000 more of a kind print ten minus the two above, and one line a thousand
        for _ in range(5000):
            mod.report_rare("choice refused", "x")
        err = capsys.readouterr().err
        assert err.count("choice refused, number") == (mod.REPORT_FIRST - 2) + 5 and mod.REPORTED["choice refused"] == 5002
        # a stream that cannot be written to is not the battle's problem
        import sys as _sys
        real = _sys.stderr
        _sys.stderr = None
        try:
            mod.report_rare("room closed", "y")
        finally:
            _sys.stderr = real
    finally:
        mod.REPORTED.clear()
        mod.REPORTED.update(saved)
