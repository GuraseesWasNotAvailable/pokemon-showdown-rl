"""
"[Invalid choice] Can't switch: You can't switch to an active Pokemon", always with Ditto active
(the live run, about one move in 100,000): switch_names.py and the mask.

    python -m pytest tests/test_switch_names.py -q -p no:cacheprovider -rs

What is held, on a real position captured from the box's own Showdown
(tests/fixtures/ditto_transform_switch.json: Imposter Ditto has Transformed into the foe's
Garchomp, and our own Garchomp is on the bench):

  * the cause: poke-env offers the switch to our Garchomp (it is legal, and not the active
    Pokemon) and sends it as "/choose switch Garchomp"; the server gives that name to the active
    Ditto, whose species is now Garchomp, and refuses it. "/choose switch 2" brings Garchomp in
  * the mask (ObservationAssembler and foulplay_bridge.legal_mask) no longer offers that switch,
    and every switch it does offer is one the server's rule resolves to that very Pokemon
  * nothing else changes: against release r1's obs_assembler.py the observation differs in that
    one mask entry only, and is byte for byte r1's in the same battle before the Transform, after
    the switch, with Ditto Transformed into a species we do not have, and without the fixture
  * obs_assembler.py is the base commit's file but for this change (branch local-app)
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

np = pytest.importorskip("numpy")
pytest.importorskip("poke_env")

from poke_env.battle import Battle  # noqa: E402
from poke_env.player.battle_order import SingleBattleOrder  # noqa: E402

import foulplay_bridge  # noqa: E402
from obs_assembler import ObservationAssembler  # noqa: E402
from switch_names import switch_name_taken  # noqa: E402

with open(os.path.join(REPO, "tests", "fixtures", "ditto_transform_switch.json"), encoding="utf-8") as _f:
    FIX = json.load(_f)

# Branch local-app was made from search at this commit; obs_assembler.py is its file with the two lines of this change.
BASE_COMMIT = "e65a7e7"
# Release r1 (what the box ran PPO with) and its obs_assembler.py; this tree's other obs_* modules are r1's.
R1_COMMIT = "8227671"
R1_OBS_ASSEMBLER_BLOB = "c0c8a87c04edec7e8e61c721a02f2e38dd4777ba"


class ReferenceUnavailable(RuntimeError):
    """git, the repository or the commit is missing: there is nothing to compare against."""


def text_at(commit: str, path: str) -> bytes:
    """A file of a commit as git holds it (bench/micro_ref.text_at on fptrain)."""
    import subprocess
    try:
        return subprocess.run(["git", "-C", REPO, "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise ReferenceUnavailable(f"git show {commit}:{path} failed: {e!r}") from e


def git_blob_id(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
EVENTS = ("switch", "-transform", "turn")


def battle_at(request=None, upto: str = "|turn|1", skip=()) -> Battle:
    """The fixture's battle as poke-env holds it when p1 is asked to choose: the request, then the log."""
    battle = Battle("battle-gen9randombattle-1", "A", logging.getLogger("test_switch_names"), gen=9)
    battle.parse_request(copy.deepcopy(request if request is not None else FIX["request"]))
    for line in FIX["messages"]:
        split = line.split("|")
        if len(split) > 1 and split[1] in EVENTS and line not in skip:
            battle.parse_message(split)
        if line == upto:
            break
    return battle


def slot(battle, name):
    return [m.name for m in battle.team.values()].index(name)


def order_text(mon) -> str:
    return SingleBattleOrder(mon).message


def server_resolves(text: str, request, species_now):
    """Side.chooseSwitch (sim/side.ts) for "/choose switch <text>": the first of the side's Pokemon, in the request's order, by name or species id."""
    from poke_env.data import to_id_str
    for i, entry in enumerate(request["side"]["pokemon"]):
        name = entry["ident"].split(": ", 1)[1]
        if text.lower() == name.lower() or to_id_str(text) == species_now[entry["ident"]]:
            return i
    return -1


def species_now(request, transformed_into=None):
    from poke_env.data import to_id_str
    out = {e["ident"]: to_id_str(e["details"].split(",")[0]) for e in request["side"]["pokemon"]}
    if transformed_into:
        active = [e["ident"] for e in request["side"]["pokemon"] if e["active"]][0]
        out[active] = transformed_into
    return out


@pytest.fixture(scope="module")
def r1_module():
    """Release r1's obs_assembler.py (from git), run against this tree's obs_* modules (which are r1's)."""
    try:
        source = text_at(R1_COMMIT, "obs_assembler.py")
    except ReferenceUnavailable as e:
        pytest.skip(f"release r1 is not at hand to compare with: {e}")
    assert git_blob_id(source.replace(b"\r\n", b"\n")) == R1_OBS_ASSEMBLER_BLOB
    module = types.ModuleType("obs_assembler_r1")
    module.__file__ = os.path.join(REPO, "obs_assembler.py")
    exec(compile(source.decode("utf-8"), "obs_assembler.py@r1", "exec"), module.__dict__)
    return module


def test_the_server_refuses_the_name_and_takes_the_slot():
    """What the box's Showdown did with three choices for the fixture's request (captured, not computed here)."""
    answers = FIX["answers"]
    assert answers["switch Garchomp"] == "|error|[Invalid choice] Can't switch: You can't switch to an active Pokémon"
    assert answers["switch 2"].startswith("|switch|p1a: Garchomp|Garchomp, L77")
    assert answers["switch Pikachu"].startswith("|switch|p1a: Pikachu|")
    assert "|-transform|p1a: Ditto|p2a: Garchomp|[from] ability: Imposter" in FIX["messages"]
    side = FIX["request"]["side"]["pokemon"]
    assert [e["ident"] for e in side] == ["p1: Ditto", "p1: Garchomp", "p1: Pikachu"] and side[0]["active"]
    assert side[0]["details"] == "Ditto, L84", "the request does not say what Ditto has become"


def test_poke_env_offers_a_legal_switch_and_names_it_as_the_server_refuses(r1_module):
    battle = battle_at()
    active = battle.active_pokemon
    assert active.name == "Ditto" and active.transformed and active.ability == "imposter"
    garchomp = battle.team["p1: Garchomp"]
    assert battle.available_switches == [garchomp, battle.team["p1: Pikachu"]], "poke-env is right: both are legal, neither is active"
    assert order_text(garchomp) == "/choose switch Garchomp" and "switch Garchomp" in FIX["answers"]
    request = FIX["request"]
    assert server_resolves("Garchomp", request, species_now(request, "garchomp")) == 0, "the name reaches the active Ditto"
    old = r1_module.ObservationAssembler().create_action_mask(battle, list(battle.team.values()))
    assert old[4 + slot(battle, "Garchomp")] == 1.0, "release r1 offers it: the refused order"


@pytest.mark.parametrize("fast", [True, False])
def test_the_mask_does_not_offer_it_and_every_switch_offered_reaches_its_pokemon(fast):
    battle = battle_at()
    asm = ObservationAssembler(fast=fast)
    obs = asm.assemble(battle, events=[])
    m0, m1 = asm.offsets["action_mask"]
    mask = obs[m0:m1]
    assert mask[4 + slot(battle, "Garchomp")] == 0.0
    assert mask[4 + slot(battle, "Pikachu")] == 1.0
    assert mask[4 + slot(battle, "Ditto")] == 0.0
    # The moves are r1's, untouched: poke-env gives a Transformed Pokemon the moves it KNEW of the
    # foe (none yet), so none of the request's four matches and r1's fallback offers slot 0 only.
    assert mask[:4].tolist() == [1.0, 0.0, 0.0, 0.0] and mask[10:14].tolist() == [0.0, 0.0, 0.0, 0.0]   # (and no Tera with it)
    assert foulplay_bridge.legal_mask(battle) == [int(x) for x in mask], "the bridge's copy of the mask says the same"
    request, now = FIX["request"], species_now(FIX["request"], "garchomp")
    team = list(battle.team.values())
    for idx in np.flatnonzero(mask[4:10]):
        mon = team[idx]
        at = server_resolves(order_text(mon).split(" ", 2)[2], request, now)
        assert at >= 0 and request["side"]["pokemon"][at]["ident"] == "p1: " + mon.name and not request["side"]["pokemon"][at]["active"]


def test_the_check_itself():
    battle = battle_at()
    ditto, garchomp, pikachu = (battle.team["p1: " + n] for n in ("Ditto", "Garchomp", "Pikachu"))
    assert switch_name_taken(ditto, garchomp) and not switch_name_taken(ditto, pikachu)
    assert not switch_name_taken(ditto, ditto) and not switch_name_taken(None, garchomp) and not switch_name_taken(ditto, None)
    before = battle_at(upto="|switch|p2a: Garchomp|Garchomp, L77, F|100/100")
    assert not before.active_pokemon.transformed and not switch_name_taken(before.active_pokemon, before.team["p1: Garchomp"])
    ditto.terastallize("Normal")                 # a Transformed Pokemon that Terastallizes loses _temporary_types, not the rest
    assert switch_name_taken(ditto, garchomp)
    ditto.switch_out(battle.fields)              # leaving the field ends the Transform
    assert not switch_name_taken(ditto, garchomp)


def _states():
    """(name, battle) of states around the fixture's, in which the mask must be r1's."""
    request = FIX["request"]
    yield "before the Transform", battle_at(upto="|switch|p2a: Garchomp|Garchomp, L77, F|100/100")
    # Ditto Transformed into a species we do not have: the foe's Snorlax
    b = Battle("battle-gen9randombattle-2", "A", logging.getLogger("test_switch_names"), gen=9)
    b.parse_request(copy.deepcopy(request))
    for line in ("|switch|p1a: Ditto|Ditto, L84|200/200", "|switch|p2a: Snorlax|Snorlax, L85, M|100/100",
                 "|-transform|p1a: Ditto|p2a: Snorlax|[from] ability: Imposter", "|turn|1"):
        b.parse_message(line.split("|"))
    assert b.active_pokemon.transformed
    yield "Transformed into Snorlax", b
    # our Garchomp has come in (the request after "switch 2"): Ditto is on the bench, nothing Transformed
    after = copy.deepcopy(request)
    mons = after["side"]["pokemon"]
    mons[0]["active"], mons[1]["active"] = False, True
    mons[0]["moves"], mons[0]["ability"] = ["transform"], "imposter"
    mons[0], mons[1] = mons[1], mons[0]
    after["active"][0]["canTerastallize"] = "Steel"
    b = battle_at()
    b.parse_message(["", "switch", "p1a: Garchomp", "Garchomp, L77, M", "277/277"])
    b.parse_request(after)
    assert b.active_pokemon.name == "Garchomp" and not b.team["p1: Ditto"].transformed
    yield "after the switch", b


def test_nothing_else_of_the_observation_changes(r1_module):
    """Against release r1's assembler: one mask entry in the fixture's state, nothing in the others."""
    for fast in (True, False):
        new, old = ObservationAssembler(fast=fast), r1_module.ObservationAssembler(fast=fast)
        m0, _ = new.offsets["action_mask"]
        battle = battle_at()
        a, b = new.assemble(battle, events=[]), old.assemble(battle_at(), events=[])
        assert np.flatnonzero(a != b).tolist() == [m0 + 4 + slot(battle, "Garchomp")], fast
        assert b[m0 + 4 + slot(battle, "Garchomp")] == 1.0 and a[m0 + 4 + slot(battle, "Garchomp")] == 0.0
        for name, state in _states():
            assert new.assemble(state, events=[]).tobytes() == old.assemble(state, events=[]).tobytes(), (name, fast)
            assert foulplay_bridge.legal_mask(state) == [int(x) for x in old.create_action_mask(state, list(state.team.values()))], name


def test_obs_assembler_is_this_branchs_file_but_for_this_change():
    """
    Branch local-app: obs_assembler.py is the file of the branch it was made from (search, e65a7e7,
    which also carries the own-Tera-type option, off by default) with exactly the two edits brought
    over from fptrain (f08e030). On fptrain the same test derives the file from release r1's.
    """
    with open(os.path.join(REPO, "obs_assembler.py"), "rb") as f:
        text = f.read().replace(b"\r\n", b"\n")
    try:
        base = text_at(BASE_COMMIT, "obs_assembler.py").replace(b"\r\n", b"\n").decode("utf-8")
    except ReferenceUnavailable as e:
        pytest.skip(f"the base commit is not at hand to derive the file from: {e}")
    imp = "from obs_transitions import encode_transitions_inplace, transition_id_dim, transition_scalar_dim\n"
    old_if = "                    if idx < 6:\n                        mask[4 + idx] = 1.0\n"
    new_if = ("                    # Not one whose name the server gives to an active Pokemon that Transformed\n"
              "                    # into its species (switch_names.py): that order is refused.\n"
              "                    if idx < 6 and not switch_name_taken(active_mon, available_switch):\n"
              "                        mask[4 + idx] = 1.0\n")
    assert base.count(imp) == 1 and base.count(old_if) == 1
    derived = base.replace(imp, imp + "from switch_names import switch_name_taken\n").replace(old_if, new_if)
    assert derived.encode("utf-8") == text, "obs_assembler.py is the base's with exactly the two edits of this change"
