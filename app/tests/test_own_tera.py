"""
The own Tera type as an input: obs_tera.py, tera_worker.py, train.py --own-tera-type, the entry
in the learner's checkpoints, and the players that read it.

    python -m pytest tests/test_own_tera.py -q -p no:cacheprovider -rs

What is held:

  * where the type comes from: the request's "teraType" of each own Pokemon, filed by poke-env
    under the Pokemon's ident, so every Pokemon of the team shows its own type whatever the
    order of the request, after a switch, and next to a Pokemon that has Terastallized (which
    shows the type it took, as before); a Pokemon without a type in the request and a type the
    vocabulary lacks write nothing and are counted; the opponent's blocks are never written
  * the bytes are those of ObservationAssembler(own_tera_type=True) in commit 53c2809, where the
    option was first written into obs_assembler.py itself
  * off is what it was: make_assembler() is ObservationAssembler, obs_assembler.py is release
    r1's file, train.py without the flag imports nothing of this, builds the actors it built
    before, and the learner writes the checkpoint it wrote before
  * the flag reaches the workers' assemblers: with Ray stood in for by a stub, every worker of
    a run with --own-tera-type is a class whose constructor points worker.py's assemblers at
    the one that shows the type, also the workers with Foul Play pairs; through the worker's own
    rollout loop on a recorded battle the learner is handed observations that differ from a run
    without the flag in the own Tera blocks only, and the same actions, rewards and episode ends
  * the checkpoint round trip: the learner writes "obs_settings" only when told to, a run with
    the flag that resumes a checkpoint without the entry says "on" from then on (and says so in
    its log), a checkpoint without the entry loads as "off"
  * hybrid.PolicyNet builds the observation the checkpoint was trained with, and with it the
    players of evaluate.py and ladder.py

No Showdown server and no Ray are started. Nothing is removed.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import json
import logging
import os
import queue
import subprocess
import sys
import time
import types
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

pytest.importorskip("numpy")
pytest.importorskip("poke_env")
pytest.importorskip("ray")
pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from poke_env.battle import Battle  # noqa: E402

import config  # noqa: E402
import obs_tera  # noqa: E402
import worker  # noqa: E402
from bench import micro_ref  # noqa: E402
from obs_assembler import ObservationAssembler  # noqa: E402
from obs_tera import (CHECKPOINT_KEY, OwnTeraObservationAssembler, checkpoint_obs_settings,  # noqa: E402
                      checkpoint_own_tera_type, make_assembler, own_tera_type_name)

TERA_COMMIT = "53c2809"            # branch "search": the option as a parameter of ObservationAssembler
OLD_CKPT = os.environ.get("PSPPO_OLD_CKPT") or r"C:\Users\you\ps-ppo\runs\vm\checkpoints\learner_update_002575.pt"

with open(os.path.join(REPO, "tests", "fixtures", "tera_request.json"), encoding="utf-8") as _f:
    REQUEST = json.load(_f)["request"]


# --------------------------------------------------------------------------- battles


def make_battle(request=None) -> Battle:
    battle = Battle("battle-gen9randombattle-1", "tester", logging.getLogger("test_own_tera"), gen=9)
    battle.parse_request(copy.deepcopy(request if request is not None else REQUEST))
    return battle


def name_of(entry) -> str:
    return entry["ident"].split(": ", 1)[1]


def switched_request(to: int):
    """The request after the active Pokemon has switched to side.pokemon[to]: the server lists the new active one first."""
    request = copy.deepcopy(REQUEST)
    mons = request["side"]["pokemon"]
    mons[0]["active"], mons[to]["active"] = False, True
    mons[0], mons[to] = mons[to], mons[0]
    request["active"] = [{"moves": [{"move": m, "id": m, "pp": 8, "maxpp": 8, "target": "normal", "disabled": False}
                                    for m in mons[0]["moves"]], "canTerastallize": mons[0]["teraType"]}]
    request["rqid"] = int(REQUEST.get("rqid", 0)) + 2
    return request


def switched_battle(to: int = 2) -> Battle:
    battle = make_battle()
    entry = REQUEST["side"]["pokemon"][to]
    battle.parse_message(["", "switch", f"p1a: {name_of(entry)}", entry["details"], entry["condition"]])
    battle.parse_request(switched_request(to))
    return battle


def terastallized_battle() -> Battle:
    battle = make_battle()
    active = REQUEST["side"]["pokemon"][0]
    battle.parse_message(["", "-terastallize", f"p1a: {name_of(active)}", active["teraType"]])
    request = copy.deepcopy(REQUEST)
    del request["active"][0]["canTerastallize"]
    request["side"]["pokemon"][0]["terastallized"] = active["teraType"]
    request["rqid"] = int(REQUEST.get("rqid", 0)) + 2
    battle.parse_request(request)
    return battle


def with_an_opponent(battle: Battle) -> Battle:
    battle.parse_message(["", "switch", "p2a: Arceus", "Arceus, L70", "100/100"])
    return battle


STATES = {
    "first request": make_battle,
    "after a switch": switched_battle,
    "after Terastallizing": terastallized_battle,
    "with an opponent": lambda: with_an_opponent(make_battle()),
}


def tera_block(asm, obs, slot: int) -> np.ndarray:
    t0, t1 = asm.meta["feature_map"]["body"]["types_raw"]
    half = (t1 - t0) // 2
    start = asm.offsets["pokemon_body"][0] + slot * asm.pokemon_scalar_dim + t0 + half
    return obs[start:start + half]


def own_tera_positions(asm) -> np.ndarray:
    """True at every number of the observation that belongs to the Tera type block of an own Pokemon."""
    t0, t1 = asm.meta["feature_map"]["body"]["types_raw"]
    half = (t1 - t0) // 2
    inside = np.zeros(asm.total_dim, dtype=bool)
    for slot in range(6):
        start = asm.offsets["pokemon_body"][0] + slot * asm.pokemon_scalar_dim + t0 + half
        inside[start:start + half] = True
    return inside


def type_id(asm, name: str) -> int:
    return asm.vocab_map["pokemon.type"][name.lower()] + 1


@pytest.fixture(scope="module")
def plain() -> ObservationAssembler:
    return ObservationAssembler()


@pytest.fixture()
def shown() -> OwnTeraObservationAssembler:
    return make_assembler(own_tera_type=True)


# --------------------------------------------------------------------------- where the type comes from


def test_every_pokemon_shows_its_own_type_whatever_the_order_of_the_request(shown):
    """
    After a switch the server lists the new active Pokemon first. The observation's slots keep
    the order the team was first seen in (battle.team), and each slot shows the type the request
    names for THAT Pokemon: the entry is found by ident, not by position.
    """
    by_name = {name_of(p): p["teraType"] for p in REQUEST["side"]["pokemon"]}
    assert len(set(by_name.values())) > 1, "otherwise a mix-up between slots would pass"
    for to in (1, 2, 5):
        battle = switched_battle(to)
        team = list(battle.team.values())
        assert [m.name for m in team] == [name_of(p) for p in REQUEST["side"]["pokemon"]], "the slots did not move"
        assert [m.active for m in team] == [i == to for i in range(6)], "the switch was applied"
        assert [name_of(p) for p in battle.last_request["side"]["pokemon"]][0] == team[to].name, "and the request is in another order"
        obs = shown.assemble(battle, events=[])
        for slot, mon in enumerate(team):
            assert own_tera_type_name(mon) == by_name[mon.name].lower()
            assert np.flatnonzero(tera_block(shown, obs, slot)).tolist() == [type_id(shown, by_name[mon.name])], (to, slot)


def test_a_terastallized_pokemon_shows_the_type_it_took_and_the_others_keep_theirs(plain, shown):
    battle = terastallized_battle()
    team = list(battle.team.values())
    assert team[0].is_terastallized and team[0].tera_type is not None and [m.tera_type for m in team[1:]] == [None] * 5
    off, on = plain.assemble(battle, events=[]), shown.assemble(battle, events=[])
    types_ = [p["teraType"] for p in REQUEST["side"]["pokemon"]]
    # the Terastallized one: written by the encoders from what poke-env knows, the same with and without
    assert np.flatnonzero(tera_block(plain, off, 0)).tolist() == np.flatnonzero(tera_block(shown, on, 0)).tolist() == [type_id(shown, types_[0])]
    for slot in range(1, 6):
        assert not tera_block(plain, off, slot).any()
        assert np.flatnonzero(tera_block(shown, on, slot)).tolist() == [type_id(shown, types_[slot])]
    assert shown.tera_counts == {"observations": 1, "shown": 5, "unknown": 0, "corrected": 0}


def test_no_type_in_the_request_and_a_type_the_vocabulary_lacks_write_nothing_and_are_counted(shown):
    request = copy.deepcopy(REQUEST)
    del request["side"]["pokemon"][1]["teraType"]
    request["side"]["pokemon"][2]["teraType"] = "Nonsense"
    request["side"]["pokemon"][3]["teraType"] = "Stellar"          # Terapagos's: the vocabulary has it
    obs = shown.assemble(make_battle(request), events=[])
    assert not tera_block(shown, obs, 1).any() and not tera_block(shown, obs, 2).any()
    assert np.flatnonzero(tera_block(shown, obs, 3)).tolist() == [type_id(shown, "stellar")]
    for slot in (0, 4, 5):
        assert np.flatnonzero(tera_block(shown, obs, slot)).tolist() == [type_id(shown, REQUEST["side"]["pokemon"][slot]["teraType"])]
    assert shown.tera_counts == {"observations": 1, "shown": 4, "unknown": 2, "corrected": 0}


def test_a_pokemon_whose_request_entry_is_gone_writes_nothing(shown):
    """poke-env drops the entry while it corrects an Illusion (Pokemon.was_illusioned) and puts it back at once."""
    battle = make_battle()
    mon = list(battle.team.values())[4]
    kept, mon._last_request = mon._last_request, None
    assert own_tera_type_name(mon) is None
    obs = shown.assemble(battle, events=[])
    assert not tera_block(shown, obs, 4).any() and shown.tera_counts["unknown"] == 1
    mon._last_request = kept
    assert tera_block(shown, shown.assemble(battle, events=[]), 4).any()


def illusion_battle() -> Battle:
    """
    A Zoroark (here the Pokemon in slot 4, whose Tera type is Poison) Terastallizes while it looks
    like the Pokemon in slot 0. The server says "-terastallize|p1a: <slot 0>|Poison", and poke-env
    files that Tera on slot 0. The next request says what really happened: slot 4 is
    Terastallized into Poison, slot 0 is not and its Tera type is its own.
    """
    battle = make_battle()
    mons = REQUEST["side"]["pokemon"]
    battle.parse_message(["", "-terastallize", f"p1a: {name_of(mons[0])}", mons[4]["teraType"]])
    request = copy.deepcopy(REQUEST)
    del request["active"][0]["canTerastallize"]
    request["side"]["pokemon"][4]["terastallized"] = mons[4]["teraType"]
    request["rqid"] = int(REQUEST.get("rqid", 0)) + 2
    battle.parse_request(request)
    return battle


def test_a_tera_type_poke_env_filed_on_the_wrong_pokemon_is_replaced_by_the_requests(plain, shown, as_committed):
    """
    The review's finding: with an Illusion poke-env shows the Zoroark's Tera type on the Pokemon
    it was disguised as (live: 230 of 759,624 own blocks, all on teams with a Zoroark). For the own
    team the request is the truth: every own block shows the request's type, the Zoroark the one
    it took. This is the one state where the bytes differ from 53c2809's on purpose, in slot 0's
    block only; the observation without the flag is not touched (it keeps poke-env's error).
    """
    battle = illusion_battle()
    mons = REQUEST["side"]["pokemon"]
    team = list(battle.team.values())
    assert str(team[0].tera_type.name).lower() == mons[4]["teraType"].lower() != mons[0]["teraType"].lower(), "poke-env's mistake is there"
    assert team[4].tera_type is None
    off, on = plain.assemble(battle, events=[]), shown.assemble(battle, events=[])
    assert np.flatnonzero(tera_block(plain, off, 0)).tolist() == [type_id(plain, mons[4]["teraType"])], "without the flag: as before"
    for slot in range(6):
        assert np.flatnonzero(tera_block(shown, on, slot)).tolist() == [type_id(shown, mons[slot]["teraType"])], slot
    assert shown.tera_counts == {"observations": 1, "shown": 5, "unknown": 0, "corrected": 1}
    differ = off != on
    assert not differ[~own_tera_positions(plain)].any(), "nothing outside the own Tera blocks"
    ref = as_committed.ObservationAssembler(own_tera_type=True).assemble(battle, events=[])
    against_ref = np.flatnonzero(ref != on)
    t0, t1 = plain.meta["feature_map"]["body"]["types_raw"]
    slot0 = plain.offsets["pokemon_body"][0] + t0 + (t1 - t0) // 2
    assert against_ref.min() >= slot0 and against_ref.max() < slot0 + (t1 - t0) // 2, "53c2809 differs in slot 0's block only"


def test_the_type_a_pokemon_took_is_the_requests_also_where_poke_env_has_none(shown):
    """The Zoroark itself: poke-env never saw it Terastallize (the line named another Pokemon). The request's "terastallized" is shown."""
    request = copy.deepcopy(REQUEST)
    del request["active"][0]["canTerastallize"]
    request["side"]["pokemon"][2]["terastallized"] = "Stellar"          # not its teraType: the block follows "terastallized"
    request["rqid"] = int(REQUEST.get("rqid", 0)) + 2
    battle = make_battle(request)
    assert list(battle.team.values())[2].tera_type is None
    obs = shown.assemble(battle, events=[])
    assert np.flatnonzero(tera_block(shown, obs, 2)).tolist() == [type_id(shown, "stellar")]
    assert shown.tera_counts["shown"] == 6 and shown.tera_counts["corrected"] == 0


def test_only_the_own_tera_blocks_differ_and_the_opponents_are_never_written(plain, shown):
    inside = own_tera_positions(plain)
    for name, make in STATES.items():
        battle = make()
        off, on = plain.assemble(battle, events=[]), shown.assemble(battle, events=[])
        differ = off != on
        assert off.dtype == on.dtype and off.shape == on.shape
        assert int(differ.sum()) == (5 if name == "after Terastallizing" else 6), name
        assert not differ[~inside].any(), name
        for slot in range(6, 12):
            assert not tera_block(shown, on, slot).any(), name
        m0, m1 = plain.offsets["action_mask"]
        assert off[m0:m1].tolist() == on[m0:m1].tolist()


@pytest.fixture(scope="module")
def as_committed():
    """obs_assembler.py of commit 53c2809, where ObservationAssembler itself had the option."""
    try:
        source = micro_ref.text_at(TERA_COMMIT, "obs_assembler.py")
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(f"commit {TERA_COMMIT} (branch search) is not at hand to compare with: {e}")
    module = types.ModuleType("obs_assembler_53c2809")
    module.__file__ = os.path.join(REPO, "obs_assembler.py")
    exec(compile(source.decode("utf-8"), "obs_assembler.py@53c2809", "exec"), module.__dict__)
    assert "own_tera_type" in inspect.signature(module.ObservationAssembler.__init__).parameters
    return module


@pytest.mark.parametrize("fast", [True, False])
def test_the_bytes_are_those_of_the_commit_the_option_comes_from(as_committed, fast):
    """On and off, in every state above: what this module writes is what 53c2809's assembler writes."""
    ref_on = as_committed.ObservationAssembler(fast=fast, own_tera_type=True)
    ref_off = as_committed.ObservationAssembler(fast=fast)
    new_on, new_off = make_assembler(own_tera_type=True, fast=fast), make_assembler(fast=fast)
    for name, make in STATES.items():
        battle = make()
        assert new_on.assemble(battle, events=[]).tobytes() == ref_on.assemble(battle, events=[]).tobytes(), name
        assert new_off.assemble(battle, events=[]).tobytes() == ref_off.assemble(battle, events=[]).tobytes(), name
        for mon in battle.team.values():
            assert own_tera_type_name(mon) == as_committed.own_tera_type_name(mon)


# --------------------------------------------------------------------------- off is what it was


def test_off_is_the_class_and_the_file_of_before(plain):
    assert type(make_assembler()) is ObservationAssembler and type(make_assembler(own_tera_type=False)) is ObservationAssembler
    assert make_assembler.__defaults__ == (False,)
    with open(os.path.join(REPO, "obs_assembler.py"), "rb") as f:
        assert micro_ref.git_blob_id(f.read().replace(b"\r\n", b"\n")) == "c8b4799ce878076d6624dadce4ec292982e9dd85", \
            "obs_assembler.py is release r1's but for the switch names (tests/test_switch_names.py derives it)"
    # the subclass with the option off writes the base class's bytes too
    quiet = OwnTeraObservationAssembler(own_tera_type=False)
    for name, make in STATES.items():
        battle = make()
        assert quiet.assemble(battle, events=[]).tobytes() == plain.assemble(battle, events=[]).tobytes(), name
    assert quiet.tera_counts == {"observations": 0, "shown": 0, "unknown": 0, "corrected": 0}
    # and nothing but the two methods is the subclass's own
    for name in ("create_action_mask", "map_index_to_order", "map_order_to_index", "get_schema_metadata", "translate_relative_index"):
        assert getattr(OwnTeraObservationAssembler, name) is getattr(ObservationAssembler, name), name
    on = make_assembler(own_tera_type=True)
    assert on.total_dim == plain.total_dim and on.offsets == plain.offsets and on.layout == plain.layout


def _run_python(code: str) -> dict:
    """Runs `code` in a fresh interpreter in the repository and returns the JSON it prints last."""
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_without_the_flag_train_py_imports_nothing_of_it_and_the_flag_never_reaches_the_run_config():
    got = _run_python(
        "import sys, json\n"
        "import train, config, worker, obs_assembler\n"
        "args = train.parse_args([])\n"
        "on = train.parse_args(['--own-tera-type'])\n"
        "out = {'flag': [args.own_tera_type, on.own_tera_type],\n"
        "       'classes': train.own_tera_type_classes(args) is None,\n"
        "       'by_hand': train.own_tera_type_classes(__import__('argparse').Namespace()) is None,\n"
        "       'config': [train.build_config(args) == config.RunConfig.default(), train.build_config(on) == config.RunConfig.default()],\n"
        "       'worker_builds': worker.ObservationAssembler is obs_assembler.ObservationAssembler,\n"
        "       'imported': sorted(m for m in ('obs_tera', 'tera_worker', 'tera_fp_worker', 'fp_opponent', 'hybrid') if m in sys.modules)}\n"
        "print(json.dumps(out))\n")
    assert got == {"flag": [False, True], "classes": True, "by_hand": True, "config": [True, True], "worker_builds": True, "imported": []}
    names = {f.name for cls in (config.RunConfig, config.ObsConfig, config.EnvConfig, config.RolloutConfig, config.LearnerConfig,
                                config.InferenceConfig) for f in dataclasses.fields(cls)}
    assert not [n for n in names if "tera" in n.lower()], "the setting is not a field of RunConfig: test_ppo_health holds that to r1"


# --------------------------------------------------------------------------- train.py, with Ray stood in for


def _deploy(monkeypatch, caplog, argv):
    """train.main() from its first line to its last with test_fp_opponent's Ray stub: the actors made, the log, the report."""
    import foulplay_bridge
    import test_fp_opponent as tfo
    import train
    stub = tfo._RayStub()
    monkeypatch.setattr(train, "ray", stub)
    monkeypatch.setattr(train, "WeightStore", SimpleNamespace(remote=lambda: "the weight store"))
    monkeypatch.setattr(train, "time", SimpleNamespace(time=time.time, sleep=lambda _s: None))
    monkeypatch.setattr(foulplay_bridge, "available", lambda *a, **k: (True, "foul-play test at nowhere"))
    seen = {}

    async def no_loop(args, started, infer, learner, workers, run_refs, **more):
        seen.update(workers=len(workers), more=more)

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(train, "_telemetry_loop", no_loop)
    monkeypatch.setattr(train.asyncio, "sleep", no_sleep)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=train.logger.name):
        asyncio.run(train.main(train.parse_args(argv)))
    assert seen["workers"] == len(stub.made) - 2
    return stub.made, [r.getMessage() for r in caplog.records], seen["more"]


BOX_IN_SMALL = ["--mode", "ppo", "--ports", "8840-8843", "--battles", "32", "--rooms-per-pair", "4"]     # 4 workers of 2 pairs of 4 battles
STAMP = "2026-10-06 09:48:48,489 [INFO] "


def _without_actors(kwargs):
    return {k: v for k, v in kwargs.items() if k not in ("inference_actor", "learner_actor")}


def test_with_the_flag_every_worker_shows_the_type_and_the_learner_is_told(monkeypatch, caplog):
    """
    --own-tera-type: the inference actor is made as before, the learner gets one more argument
    (what to write into its checkpoints), and every rollout worker is OwnTeraRolloutWorker, made
    from the arguments worker.RolloutWorker was made from. Without the flag: the actors of
    before from the arguments of before, the telemetry loop called as before, not a line more.
    """
    import inference
    import learner
    import pipeline
    import tera_worker
    plain, plain_log, plain_more = _deploy(monkeypatch, caplog, BOX_IN_SMALL)
    assert [m[0] for m in plain] == [inference.InferenceActor, learner.LearnerActor] + [worker.RolloutWorker] * 4
    assert plain[1][3:] == (plain[0][3], {}) and plain_more == {}
    assert not [line for line in plain_log if "tera" in line.lower()]

    made, log, more = _deploy(monkeypatch, caplog, BOX_IN_SMALL + ["--own-tera-type"])
    assert [m[0] for m in made] == [inference.InferenceActor, learner.LearnerActor] + [tera_worker.OwnTeraRolloutWorker] * 4
    assert made[0][1:] == plain[0][1:], "the inference actor does not know of it"
    assert made[1][1:4] == plain[1][1:4] and made[1][4] == {"obs_settings": {"own_tera_type": True}}
    for m, p in zip(made[2:], plain[2:]):
        assert m[1:4] == p[1:4], "declared to Ray as the workers of before"
        assert set(m[4]) == set(p[4]) and _without_actors(m[4]) == _without_actors(p[4]), "and no argument more"
        assert m[4]["learner_actor"].made is made[1] and m[4]["inference_actor"].made is made[0]
    assert isinstance(more.get("own_tera_report"), tera_worker.OwnTeraReport) and set(more) == {"own_tera_report"}
    said = [line for line in log if "tera" in line.lower()]
    assert len(said) == 1 and said[0].startswith("Own Tera type: ON. Every rollout worker writes each own Pokemon's Tera type")
    assert [line for line in log if line not in said] == plain_log
    assert pipeline.parse_trainer_events(STAMP + said[0]) == []
    read = [[e for line in lines for e in pipeline.parse_trainer_events(STAMP + line)] for lines in (log, plain_log)]
    assert read[0] == read[1] and [e["kind"] for e in read[0]] == ["header", "final_saved"], "the supervisor reads the same start"


def test_with_the_flag_and_a_foul_play_share_the_foul_play_workers_show_it_too(monkeypatch, caplog):
    import fp_opponent as fo
    import tera_fp_worker
    import tera_worker
    import train
    share = ["--foulplay-opponent-share", "0.5", "--foulplay-search-ms", "50", "--foulplay-helpers", "2"]
    only_share, _log, _more = _deploy(monkeypatch, caplog, BOX_IN_SMALL + share)
    assert [m[0] for m in only_share[2:]] == [fo.FoulPlayRolloutWorker] * 2 + [worker.RolloutWorker] * 2
    made, log, _more = _deploy(monkeypatch, caplog, BOX_IN_SMALL + share + ["--own-tera-type"])
    assert [m[0] for m in made[2:]] == [tera_fp_worker.OwnTeraFoulPlayRolloutWorker] * 2 + [tera_worker.OwnTeraRolloutWorker] * 2
    for m, p in zip(made[2:], only_share[2:]):
        assert m[1:4] == p[1:4] and _without_actors(m[4]) == _without_actors(p[4]), "the same plan, the same arguments"
    assert made[1][4] == {"obs_settings": {"own_tera_type": True}}
    assert issubclass(tera_fp_worker.OwnTeraFoulPlayRolloutWorker, fo.FoulPlayRolloutWorker)
    assert len([line for line in log if line.startswith("Own Tera type: ON")]) == 1
    assert len([line for line in log if line.startswith("Foul Play opponent: 4 of 8 pairs")]) == 1
    # the Foul Play players build their assembler where every player does: in worker.SyncBridgePlayer.__init__
    assert "assembler =" not in inspect.getsource(fo), "fp_opponent.py makes no assembler of its own"
    assert issubclass(fo.NetworkSeatPlayer, worker.SyncBridgePlayer) and issubclass(fo.FoulPlayOpponentPlayer, worker.SyncBridgePlayer)
    assert train.own_tera_type_classes(train.parse_args(["--own-tera-type"]))["foulplay_worker"]() is tera_fp_worker.OwnTeraFoulPlayRolloutWorker


def test_if_its_modules_cannot_be_had_the_whole_run_goes_without_and_says_so(monkeypatch, caplog):
    """Never half: workers that do not show the type under checkpoints that say they did would be played wrongly."""
    import learner
    plain, plain_log, _ = _deploy(monkeypatch, caplog, BOX_IN_SMALL)
    monkeypatch.setitem(sys.modules, "tera_worker", None)          # importing it now raises ImportError
    made, log, more = _deploy(monkeypatch, caplog, BOX_IN_SMALL + ["--own-tera-type"])
    assert [m[0] for m in made] == [m[0] for m in plain] and made[1][0] is learner.LearnerActor
    assert made[1][3:] == plain[1][3:] and made[1][4] == {} and more == {}
    for m, p in zip(made[2:], plain[2:]):
        assert m[0] is worker.RolloutWorker and m[1:4] == p[1:4] and _without_actors(m[4]) == _without_actors(p[4])
    said = [line for line in log if "tera" in line.lower()]
    assert len(said) == 1 and said[0].startswith("Own Tera type: NOT USED, the run goes on without it and its checkpoints do not say it.")
    assert [line for line in log if line not in said] == plain_log


# --------------------------------------------------------------------------- the flag reaches the workers' assemblers


def _cfg():
    base = config.RunConfig.default()
    return dataclasses.replace(base, learner=dataclasses.replace(base.learner, mode="ppo"))


@pytest.fixture()
def worker_assembler(monkeypatch):
    """Puts worker.py's name for the assembler back after a test that pointed it elsewhere."""
    monkeypatch.setattr(worker, "ObservationAssembler", worker.ObservationAssembler)
    assert worker.ObservationAssembler is ObservationAssembler


def test_the_worker_points_every_assembler_of_its_process_at_the_one_that_shows_the_type(monkeypatch, worker_assembler):
    """
    The real constructor, with the two things it starts (the wrapper's thread of players and the
    learner client's thread) stood in for. The name is set before the wrapper is made, so every
    player the wrapper's thread makes later builds the assembler that shows the type; so does
    the worker itself. The heartbeat counts them.
    """
    import fp_opponent as fo
    import tera_worker
    import test_foulplay_bridge as tfb
    at_wrapper = []

    def wrapper(cfg, pairs, port):
        at_wrapper.append(worker.ObservationAssembler)
        return SimpleNamespace(active_pairs=[], ep_sem=SimpleNamespace(_value=7))

    monkeypatch.setattr(worker, "PokeEnvVectorWrapper", wrapper)
    monkeypatch.setattr(worker, "SyncLearnerClient", lambda actor, cfg: SimpleNamespace(q=queue.Queue()))
    rw = tera_worker.OwnTeraRolloutWorker(cfg=_cfg(), inference_actor=None, learner_actor=None, pairs=2, server_port=8840)
    assert at_wrapper == [OwnTeraObservationAssembler], "before the wrapper, and so before any player"
    assert worker.ObservationAssembler is OwnTeraObservationAssembler and type(rw.assembler) is OwnTeraObservationAssembler

    seat = dict(cfg=_cfg(), event_queue=queue.Queue(), action_futures={}, ep_sem=asyncio.Semaphore(4))
    players = [tfb.offline_player(cls, name, **seat) for cls, name in ((worker.SyncBridgePlayer, "ta"), (worker.SyncBridgePlayer, "tb"),
                                                                       (fo.NetworkSeatPlayer, "tc"))]
    for p in players:
        assert type(p.assembler) is OwnTeraObservationAssembler and p.assembler.own_tera_type is True
    obs = players[0].assembler.assemble(make_battle(), events=[])
    assert [int(tera_block(players[0].assembler, obs, s).sum()) for s in range(12)] == [1] * 6 + [0] * 6

    rw.vec_env.active_pairs = [(players[0], players[1]), (players[2], players[2])]
    beat = rw.heartbeat()
    assert beat["own_tera_type"] == {"players": 4, "on": 4, "observations": 1, "shown": 6, "unknown": 0, "corrected": 0}
    assert {"active_battles_worker", "learner_q_size", "traj_in_memory", "ep_sem_value"} <= set(beat), "the heartbeat of before, and one entry more"
    # a player that does not show it is counted as such
    players[1].assembler = ObservationAssembler()
    assert rw.heartbeat()["own_tera_type"]["on"] == 3
    # a heartbeat always answers
    rw._own_tera_beat = lambda: 1 / 0
    beat = rw.heartbeat()
    assert "ZeroDivisionError" in beat["own_tera_type"]["report_error"] and beat["learner_q_size"] == 0


def test_the_worker_classes_run_the_base_classes_own_code():
    import fp_opponent as fo
    import tera_fp_worker
    import tera_worker
    for name in ("run", "_guarded_rl_loop", "_run_rl_loop", "_finalize_trajectory", "_cleanup_trajectory"):
        assert getattr(tera_worker.OwnTeraRolloutWorker, name) is getattr(worker.RolloutWorker, name), name
        assert getattr(tera_fp_worker.OwnTeraFoulPlayRolloutWorker, name) is getattr(fo.FoulPlayRolloutWorker, name), name
    assert tera_worker.OwnTeraRolloutWorker.__mro__[:3] == (tera_worker.OwnTeraRolloutWorker, tera_worker.OwnTeraWorkerMixin, worker.RolloutWorker)
    assert set(vars(tera_worker.OwnTeraWorkerMixin)) & {"run", "_run_rl_loop", "_finalize_trajectory"} == set()


def test_ray_can_name_the_worker_classes_in_another_process():
    """An actor class is pickled for the worker's process. Both are found there by module and name, and are the classes."""
    import ray.cloudpickle as cloudpickle
    import tera_fp_worker
    import tera_worker
    blobs = {}
    for cls in (tera_worker.OwnTeraRolloutWorker, tera_fp_worker.OwnTeraFoulPlayRolloutWorker):
        data = cloudpickle.dumps(cls)
        assert cloudpickle.loads(data) is cls and len(data) < 200, "pickled by name, not by value"
        blobs[cls.__name__] = data.hex()
    got = _run_python(
        "import json, sys\n"
        "import ray.cloudpickle as cloudpickle\n"
        f"blobs = {blobs!r}\n"
        "out = {}\n"
        "for name, data in blobs.items():\n"
        "    cls = cloudpickle.loads(bytes.fromhex(data))\n"
        "    import worker, obs_tera\n"
        "    out[name] = [cls.__name__, cls.__module__, issubclass(cls, worker.RolloutWorker), cls.__init__.__qualname__]\n"
        "print(json.dumps(out))\n")
    assert got == {"OwnTeraRolloutWorker": ["OwnTeraRolloutWorker", "tera_worker", True, "OwnTeraWorkerMixin.__init__"],
                   "OwnTeraFoulPlayRolloutWorker": ["OwnTeraFoulPlayRolloutWorker", "tera_fp_worker", True, "OwnTeraWorkerMixin.__init__"]}


def test_through_the_workers_own_rollout_loop_the_learner_gets_the_observation_with_the_type(monkeypatch, worker_assembler):
    """
    One recorded battle through worker.RolloutWorker's real rollout loop, reward code and learner
    client (test_fp_opponent's harness: the inference actor and the learner are stand-ins),
    three times: a run without the flag, a run with it, and the network's seat of a Foul Play
    pair in a run with it. With the flag the learner is handed the same actions, rewards, ends of
    episodes and lengths, and observations that differ in the own Tera blocks only. (The
    stand-in's log-probability, value and variance are read off the observation, so they move
    by exactly what was written: the learner gets what inference answered for the observation
    it is given.)
    """
    import fp_opponent as fo
    import tera_fp_worker
    import tera_worker
    import test_foulplay_bridge as tfb
    import test_fp_opponent as tfo
    if not os.path.isfile(tfb.FIXTURE):
        pytest.skip("needs tests/fixtures/foulplay_bridge_trace.jsonl.gz")
    monkeypatch.chdir(REPO)
    events = tfo._battles()[0]
    username = events[-1]["username"]
    decisions = len([e for e in events if e["t"] == "order"])

    def run(worker_class, seat_class):
        harness = tfo._LoopHarness(monkeypatch, worker_class)
        player = tfb.offline_player(seat_class, username, **harness.seat_kwargs())
        tfo._feed(player, events)
        return harness.settle(), player

    plain, plain_player = run(worker.RolloutWorker, worker.SyncBridgePlayer)
    assert type(plain_player.assembler) is ObservationAssembler and len(plain.sent) == 1

    tera_worker.show_own_tera_type_in_this_process()             # what OwnTeraWorkerMixin.__init__ does (the harness skips constructors)
    shown, shown_player = run(tera_worker.OwnTeraRolloutWorker, worker.SyncBridgePlayer)
    fp_net, fp_player = run(tera_fp_worker.OwnTeraFoulPlayRolloutWorker, fo.NetworkSeatPlayer)
    for player in (shown_player, fp_player):
        assert type(player.assembler) is OwnTeraObservationAssembler
        assert player.assembler.tera_counts["observations"] == decisions and player.assembler.tera_counts["unknown"] == 0

    names = ("obs", "act", "logp", "val", "var", "rew", "done", "lengths")
    a, b, c = (dict(zip(names, h.sent[0])) for h in (plain, shown, fp_net))
    assert len(shown.sent) == len(fp_net.sent) == 1 and shown.asked == plain.asked
    for name in names:
        assert b[name].dtype == a[name].dtype and b[name].shape == a[name].shape, name
        assert c[name].tobytes() == b[name].tobytes(), f"{name}: the network's seat of a Foul Play pair is a self-play seat here too"
    for name in ("act", "rew", "done", "lengths"):
        assert b[name].tobytes() == a[name].tobytes(), name
    assert a["lengths"].tolist() == [decisions]
    differ = a["obs"] != b["obs"]
    inside = own_tera_positions(plain_player.assembler)
    lit = differ.sum(axis=1)
    assert not differ[:, ~inside].any(), "nothing outside the six own Tera type blocks"
    assert lit.min() >= 1 and lit.max() <= 6 and int(lit.sum()) == shown_player.assembler.tera_counts["shown"]
    assert (b["obs"][differ] == 1.0).all() and (a["obs"][differ] == 0.0).all()
    for name, scale in (("logp", 1e-3), ("val", 2e-3), ("var", 3e-3)):
        assert np.allclose(b[name] - a[name], lit * scale, atol=2e-4), name


# --------------------------------------------------------------------------- the line in the trainer's log


def test_the_line_counts_the_players_and_an_absent_one_is_an_error():
    import pipeline
    import tera_worker
    plain = {"active_battles_worker": 128, "learner_q_size": 0, "traj_in_memory": 60, "ep_sem_value": 100}

    def beat(**tera):
        return dict(plain, own_tera_type=dict({"players": 8, "on": 8, "observations": 1000, "shown": 5700, "unknown": 0}, **tera))

    report = tera_worker.OwnTeraReport(every_s=60.0)
    wrong, line = report.line([beat(), beat()], 2, now=1000.0)
    assert wrong is False
    assert line == ("[OwnTera] shown by 16 of 16 players on 2 of 2 workers | 2000 observations, a Tera type written for 11400 own Pokemon "
                    "(100.00% of those that had not Terastallized; 0 had no type in the request)")
    assert pipeline.parse_trainer_events(STAMP + line) == []
    assert report.line([beat(), beat()], 2, now=1030.0) is None, "nothing new and the minute is not over"
    assert report.line([beat(), beat()], 2, now=1061.0) is not None

    wrong, line = report.line([beat(on=6), beat()], 2, now=1062.0)
    assert wrong is True and "NOT SHOWN BY 2 PLAYERS" in line, "said at once, whatever the minute"
    wrong, line = report.line([beat(), dict(plain)], 2, now=1063.0)
    assert wrong is True and "1 WORKERS DO NOT REPORT IT" in line
    wrong, line = report.line([beat(), dict(plain, own_tera_type={"report_error": "boom"})], 2, now=1064.0)
    assert wrong is True and "1 workers could not report: boom" in line
    wrong, line = tera_worker.OwnTeraReport().line([beat(players=0, on=0, observations=0, shown=0)], 1, now=5.0)
    assert wrong is False and line == "[OwnTera] shown by 0 of 0 players on 1 of 1 workers | no observation yet"
    wrong, line = tera_worker.OwnTeraReport().line([beat(shown=5600, unknown=100)], 1, now=5.0)
    assert "(98.25% of those that had not Terastallized; 100 had no type in the request)" in line
    wrong, line = tera_worker.OwnTeraReport().line([beat(corrected=3)], 1, now=5.0)
    assert wrong is False and line.endswith("0 had no type in the request), 3 blocks set to the request's type where poke-env had another (an Illusion)")


def test_the_telemetry_loop_logs_the_line_after_the_one_the_supervisor_reads(monkeypatch, caplog, tmp_path):
    """The "[Telemetry]" line is the line of before, character for character. A report that fails costs its own line only."""
    import pipeline
    import tera_worker
    import test_fp_opponent as tfo
    import train
    base = {"active_battles_worker": 128, "learner_q_size": 0, "traj_in_memory": 60, "ep_sem_value": 100}
    plain = tfo._telemetry_lines(monkeypatch, caplog, tmp_path, [dict(base), dict(base)])
    assert len(plain) == 1 and plain[0].startswith("[Telemetry] ACTIVE: 256 | ")

    def lines(beats, report):
        monkeypatch.setattr(train, "ray", tfo._RayStub())
        stop = tmp_path / ("stop_tera_%d" % len(list(tmp_path.iterdir())))
        args = train.parse_args(["--own-tera-type", "--stop-file", str(stop)])

        async def one_round_only(_s):
            stop.write_text("stop")

        monkeypatch.setattr(train.asyncio, "sleep", one_round_only)
        infer = SimpleNamespace(get_stats=SimpleNamespace(remote=lambda: {"batches": 5, "avg_batch_size": 301.5}))
        learner = SimpleNamespace(get_stats=SimpleNamespace(remote=lambda: {"update": 7, "episodes": 9, "steps_in_dataset": 11, "total_steps": 128000000}))
        workers = [SimpleNamespace(heartbeat=SimpleNamespace(remote=lambda beat=beat: beat)) for beat in beats]
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=train.logger.name):
            asyncio.run(asyncio.wait_for(train._telemetry_loop(args, time.time(), infer, learner, workers, ["run"] * len(workers),
                                                               own_tera_report=report), timeout=60))
        return [(r.levelname, r.getMessage()) for r in caplog.records][:-1]

    tera = {"players": 8, "on": 8, "observations": 10, "shown": 58, "unknown": 0}
    good = lines([dict(base, own_tera_type=dict(tera)), dict(base, own_tera_type=dict(tera))], tera_worker.OwnTeraReport())
    assert [g[1] for g in good][0] == plain[0] and len(good) == 2
    assert good[1][0] == "INFO" and good[1][1].startswith("[OwnTera] shown by 16 of 16 players on 2 of 2 workers | 20 observations")
    read = [e for _lvl, line in good for e in pipeline.parse_trainer_events(STAMP + line)]
    assert read == pipeline.parse_trainer_events(STAMP + plain[0]) and read[0]["kind"] == "telemetry"

    bad = lines([dict(base, own_tera_type=dict(tera, on=7)), dict(base)], tera_worker.OwnTeraReport())
    assert bad[0][1] == plain[0] and bad[1][0] == "ERROR" and "NOT SHOWN BY 1 PLAYERS" in bad[1][1] and "1 WORKERS DO NOT REPORT IT" in bad[1][1]

    class Broken:
        def line(self, *_a, **_k):
            raise RuntimeError("boom")

    broken = lines([dict(base, own_tera_type=dict(tera))], Broken())
    assert broken[0][1].startswith("[Telemetry] ACTIVE: 128 | ") and len(broken) == 2
    assert broken[1] == ("WARNING", "The own Tera type report failed (training is not affected): RuntimeError('boom')")


# --------------------------------------------------------------------------- the checkpoint


KEYS_OF_BEFORE = {"model", "target_model", "optimizer", "scheduler", "update_idx", "total_steps", "run_cfg", "total_episodes",
                  "torch_rng", "numpy_rng"}


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """Two files written by LearnerActor._save_checkpoint itself, with the real network: a run without the flag, and one with."""
    import learner
    folder = tmp_path_factory.mktemp("own_tera_ckpt")
    torch.manual_seed(11)
    net = config.RunConfig.default().make_model()
    paths = {}
    for name, settings in (("off", None), ("on", checkpoint_obs_settings(True))):
        la = object.__new__(learner.LearnerActor)
        la.net, la.sched, la.run_cfg = net, None, config.RunConfig.default()
        la.opt = SimpleNamespace(state_dict=lambda: {})
        la.update_idx, la.total_steps, la.total_episodes = 7, 128, 3
        if name == "on":
            la.obs_settings = settings
        # "off" has no such attribute at all: a learner object built by hand, as tests/test_speed_update.py builds them
        paths[name] = str(folder / ("%s" % name) / "learner_update_000007.pt")
        os.makedirs(os.path.dirname(paths[name]))
        la._save_checkpoint(paths[name])
    return paths


def test_the_learner_writes_the_entry_only_when_the_run_shows_the_type(checkpoints):
    assert checkpoint_obs_settings(False) is None and checkpoint_obs_settings(True) == {"own_tera_type": True}
    off = torch.load(checkpoints["off"], map_location="cpu", weights_only=False)
    on = torch.load(checkpoints["on"], map_location="cpu", weights_only=False)
    assert set(off) == KEYS_OF_BEFORE, "without the flag a checkpoint has the entries it had"
    assert set(on) == KEYS_OF_BEFORE | {CHECKPOINT_KEY} and on[CHECKPOINT_KEY] == {"own_tera_type": True}
    assert checkpoint_own_tera_type(off) is False and checkpoint_own_tera_type(on) is True
    for k in ("update_idx", "total_steps", "total_episodes", "run_cfg"):
        assert off[k] == on[k]
    for k, v in off["model"].items():
        assert torch.equal(v, on["model"][k]), k
    # what is not a learner checkpoint, or says something else, is "off"
    for other in (None, [], "x", {}, {CHECKPOINT_KEY: None}, {CHECKPOINT_KEY: "yes"}, {CHECKPOINT_KEY: {}},
                  {CHECKPOINT_KEY: {"own_tera_type": False}}, {CHECKPOINT_KEY: {"own_tera_type": "yes"}}, {"model": {}}):
        assert checkpoint_own_tera_type(other) is False, other


def test_the_learners_constructor_takes_the_setting_and_is_what_it_was_without(tmp_path):
    import learner
    params = inspect.signature(learner.LearnerActor.__init__).parameters
    # search_teacher came after (search_teacher.py, tests/test_search_teacher.py): None, its default, is the learner of before.
    # imitation_switch_weight came after that (tests/test_obs_abil.py): None, its default, is the learner of before.
    # human_imitation after that (tests/test_human_imitation.py): None, its default, is the learner of before.
    assert list(params) == ["self", "cfg", "weight_store", "obs_settings", "search_teacher", "imitation_switch_weight",
                            "human_imitation"] and params["obs_settings"].default is None
    assert params["human_imitation"].default is None
    assert params["search_teacher"].default is None
    assert params["imitation_switch_weight"].default is None

    class Quiet(learner.LearnerActor):
        async def _loop(self):
            return None

        async def _maybe_resume_latest(self):
            return None

    async def build(*args, **kwargs):
        la = Quiet(*args, **kwargs)
        await asyncio.sleep(0)
        return la

    base = config.RunConfig.default()
    cfg = dataclasses.replace(base, learner=dataclasses.replace(base.learner, device="cpu", ckpt_dir=str(tmp_path / "ckpt")))
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        assert loop.run_until_complete(build(cfg, "store")).obs_settings is None
        assert loop.run_until_complete(build(cfg, "store", obs_settings={})).obs_settings is None
        given = {"own_tera_type": True}
        la = loop.run_until_complete(build(cfg, "store", obs_settings=given))
        assert la.obs_settings == given and la.obs_settings is not given
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_a_run_with_the_flag_resumes_a_checkpoint_without_it_and_from_then_on_says_on(monkeypatch, capsys, tmp_path):
    """
    The normal case: the trainer is started again with --own-tera-type on a checkpoint trained
    without. LearnerActor._load_checkpoint takes the weights and everything else as it always
    did and says the one line; the next save carries the entry. The other three cases too:
    without the flag on an old checkpoint nothing is said and nothing is written, on on on
    nothing is said, and a checkpoint that says "on" resumed without the flag is a warning.
    """
    import learner
    import test_speed_update as tsu
    run = tsu._new_learner("ppo")
    run.update_idx, run.total_steps, run.total_episodes = 5, 1234, 77
    old = tsu._checkpoint_in_memory(run)
    assert CHECKPOINT_KEY not in old
    saved = []

    def keep(payload, path):
        saved.append(payload)
        with open(path, "wb") as f:
            f.write(b"stand-in for the 219 MB")

    def resume(ckpt, settings):
        back = tsu._new_learner("ppo")
        if settings is not None:
            back.obs_settings = settings
        capsys.readouterr()
        with monkeypatch.context() as patched:
            patched.setattr(torch, "load", lambda *a, **k: ckpt)
            back._load_checkpoint("held in memory")
            said = [line for line in capsys.readouterr().out.splitlines() if "obs_settings" in line]
            patched.setattr(torch, "save", keep)
            back._save_checkpoint(str(tmp_path / ("learner_update_%06d.pt" % len(saved))))
        return back, said, saved[-1]

    back, said, payload = resume(old, {"own_tera_type": True})
    assert said == ["[learner] obs_settings.own_tera_type: the checkpoint was trained without it, this run has it on. "
                    "Checkpoints from here on say True."]
    assert payload[CHECKPOINT_KEY] == {"own_tera_type": True} and set(payload) == KEYS_OF_BEFORE | {CHECKPOINT_KEY}
    assert (back.update_idx, back.total_steps, back.total_episodes) == (5, 1234, 77), "resumed as any checkpoint is"
    for (k, v), (_k2, w) in zip(run.net.state_dict().items(), back.net.state_dict().items()):
        assert torch.equal(v.cpu(), w.cpu()), k
    assert checkpoint_own_tera_type(payload) is True

    _back, said, payload = resume(old, None)
    assert said == [] and set(payload) == KEYS_OF_BEFORE, "no flag, old checkpoint: as before"

    on = dict(old, **{CHECKPOINT_KEY: {"own_tera_type": True}})
    _back, said, payload = resume(on, {"own_tera_type": True})
    assert said == [] and payload[CHECKPOINT_KEY] == {"own_tera_type": True}

    _back, said, payload = resume(on, None)
    assert len(said) == 1 and said[0].startswith("[learner] WARNING obs_settings.own_tera_type: the checkpoint was trained with True, this run has False")
    assert set(payload) == KEYS_OF_BEFORE, "this run does not show the type, and its checkpoints do not say it does"

    # the line is a line of the log: whatever the entry holds, the resume goes through
    _back, said, _payload = resume(dict(old, **{CHECKPOINT_KEY: ["not", "a", "dict"]}), {"own_tera_type": True})
    assert len(said) == 1 and "trained without it" in said[0]
    import pipeline
    for line in ("[learner] obs_settings.own_tera_type: the checkpoint was trained without it, this run has it on. Checkpoints from here on say True.",
                 "[learner] WARNING obs_settings.own_tera_type: the checkpoint was trained with True, this run has False (the flag is not given)."):
        assert pipeline.parse_trainer_events("\x1b[36m(LearnerActor pid=5996)\x1b[0m " + line) == [], "the supervisor reads nothing into it"
    assert "self._say_obs_settings(ckpt.get(\"obs_settings\"))" in inspect.getsource(learner.LearnerActor._load_checkpoint)


# --------------------------------------------------------------------------- the players that load a checkpoint


def test_the_policy_builds_the_observation_the_checkpoint_was_trained_with(checkpoints):
    from hybrid import HybridConfig, PolicyNet
    assert HybridConfig().own_tera_type is None, "every player: as the checkpoint says"
    off = PolicyNet(HybridConfig(checkpoint=checkpoints["off"], device="cpu"))
    on = PolicyNet(HybridConfig(checkpoint=checkpoints["on"], device="cpu"))
    assert off.own_tera_type is False and type(off.assembler) is ObservationAssembler
    assert on.own_tera_type is True and type(on.assembler) is OwnTeraObservationAssembler
    battle = make_battle()
    a, b = off.assembler.assemble(battle, events=[]), on.assembler.assemble(battle, events=[])
    assert [int(tera_block(off.assembler, a, s).sum()) for s in range(6)] == [0] * 6
    assert [int(tera_block(on.assembler, b, s).sum()) for s in range(6)] == [1] * 6
    for k, v in off.net.state_dict().items():
        assert torch.equal(v, on.net.state_dict()[k]), k
    # for measurements the setting can be forced, either way
    assert type(PolicyNet(HybridConfig(checkpoint=checkpoints["off"], device="cpu", own_tera_type=True)).assembler) is OwnTeraObservationAssembler
    assert type(PolicyNet(HybridConfig(checkpoint=checkpoints["on"], device="cpu", own_tera_type=False)).assembler) is ObservationAssembler
    if not os.path.isdir(os.path.join(REPO, "checkpoints")):
        # an untrained network (plumbing tests) has no checkpoint to say anything
        assert type(PolicyNet(HybridConfig(random_init=True, device="cpu", checkpoint=None)).assembler) is ObservationAssembler


@pytest.mark.skipif(not os.path.isfile(OLD_CKPT), reason="no checkpoint of the running training on this machine (PSPPO_OLD_CKPT)")
def test_a_checkpoint_of_the_running_training_loads_as_off():
    from hybrid import HybridConfig, PolicyNet
    ckpt = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    assert CHECKPOINT_KEY not in ckpt and checkpoint_own_tera_type(ckpt) is False
    del ckpt
    policy = PolicyNet(HybridConfig(checkpoint=OLD_CKPT, device="cpu"))
    assert policy.own_tera_type is False and type(policy.assembler) is ObservationAssembler


@pytest.fixture()
def no_connection(monkeypatch):
    """Players that are built as evaluate.py and ladder.py build them, and never connect anywhere."""
    import poke_env.ps_client as ps_client

    async def listen(self):
        return None

    monkeypatch.setattr(ps_client.PSClient, "listen", listen)


def test_evaluate_py_and_ladder_py_pick_the_setting_up_from_the_checkpoint(checkpoints, no_connection):
    """
    evaluate.py's own function for the player it measures (_make_subject, which goes through
    play.build_player) and ladder.py's make_player, each given a checkpoint that says "on" and
    one that says nothing: the player's assembler is the one the checkpoint was trained with.
    """
    import evaluate as ev
    import ladder

    def subject(path, batch=True):
        name = os.path.basename(path)
        ctx = ev.EvalContext(
            host="127.0.0.1", port=8879, battle_format="gen9randombattle", concurrency=2,
            checkpoint=ev.Ckpt(path, name, os.path.getmtime(path), 7), sample=False, device="cpu", batch=batch,
            stall_seconds=90.0, foulplay_parallel=1, foulplay_search_ms=50, foulplay_module="no_such_module",
            foulplay_log_dir=os.path.dirname(path), server=None)
        return ev._make_subject(ctx, "policy", 2), ctx

    for batch in (True, False):
        player, ctx = subject(checkpoints["on"], batch)
        assert player.policy.own_tera_type is True and type(player.assembler) is OwnTeraObservationAssembler
        assert player.assembler is player.policy.assembler
        assert [n for n in ctx.notes if "trained with the own Tera types shown" in n], ctx.notes
        obs = player.assembler.assemble(make_battle(), events=[])
        assert [int(tera_block(player.assembler, obs, s).sum()) for s in range(6)] == [1] * 6

        player, ctx = subject(checkpoints["off"], batch)
        assert player.policy.own_tera_type is False and type(player.assembler) is ObservationAssembler
        assert not [n for n in ctx.notes if "Tera" in n], ctx.notes

    def ladder_player(path):
        args = SimpleNamespace(server="local:8879", checkpoint=path, device="cpu", think_ms=0, format="gen9randombattle",
                               parallel=1, timer=False)
        return ladder.make_player(args, "tester", None)

    assert type(ladder_player(checkpoints["on"]).assembler) is OwnTeraObservationAssembler
    assert type(ladder_player(checkpoints["off"]).assembler) is ObservationAssembler
