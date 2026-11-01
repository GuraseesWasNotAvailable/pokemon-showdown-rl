"""
The observation assembler's table-driven encoders (MoveEncoder, PokemonBodyEncoder,
AbilityEncoder) against the original encode_*_inplace functions: every observation must come
out byte for byte the same.

    python -m pytest tests/test_speed_obs.py -q -p no:cacheprovider -s

The main test replays recorded battles through the real worker path (poke-env's message
handler with worker.py's patches, SyncBridgePlayer, choose_move) and, at every decision, hands
the same battle object and the same turn events to two assemblers: the working tree's, and
the reference commit's (bench/micro_ref.py), which is the code before the change with its
own copies of every obs_* module and of utils. The two byte strings must be equal, and so
must the counter of names missing from the vocabulary that utils.get_id keeps.

It needs recorded player streams (bench/micro_showdown_sim.py --record N writes them):

    PSPPO_STREAMS        one or more streams.jsonl files, separated by the path separator
    PSPPO_SHOWDOWN_DIR   or: a built Pokemon Showdown checkout, to record 160 battles now
    PSPPO_OBS_MIN        decisions the replay has to reach (default 10000)

Without either the replay is skipped and only the small checks below it run.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402

import obs_abilities  # noqa: E402
import obs_assembler  # noqa: E402
import obs_moves  # noqa: E402
import obs_pokemon  # noqa: E402
import utils  # noqa: E402
from bench import micro_ref  # noqa: E402

ORIGINALS = (("obs_moves", "encode_moves_inplace"), ("obs_moves", "get_accuracy_int"), ("obs_moves", "get_move_metadata"),
             ("obs_pokemon", "encode_pokemon_body_inplace"), ("obs_pokemon", "get_combat_bounds"), ("obs_pokemon", "estimate_stat"),
             ("obs_pokemon", "get_pokemon_metadata"), ("obs_abilities", "encode_ability_inplace"),
             ("obs_global", "encode_global_inplace"), ("obs_transitions", "encode_transitions_inplace"),
             ("obs_assembler", "ObservationAssembler.create_action_mask"))


@pytest.fixture(scope="module")
def ref():
    try:
        return micro_ref.Reference.export()
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(f"no reference code to compare with: {e}")


@pytest.fixture(scope="module")
def ref_assembler_class(ref):
    mod = ref.load("obs_assembler")
    assert not hasattr(mod, "MoveEncoder") and mod is not obs_assembler
    return mod.ObservationAssembler


def _streams():
    given = [p for p in os.environ.get("PSPPO_STREAMS", "").split(os.pathsep) if p.strip()]
    if given:
        missing = [p for p in given if not os.path.isfile(p)]
        if missing:
            pytest.fail(f"PSPPO_STREAMS names files that are not there: {missing}")
        return given
    ps_dir = os.environ.get("PSPPO_SHOWDOWN_DIR")
    if not ps_dir:
        pytest.skip("no recorded battles: set PSPPO_STREAMS or PSPPO_SHOWDOWN_DIR")
    out = tempfile.mkdtemp(prefix="psppo_streams_")
    r = subprocess.run([sys.executable, os.path.join(REPO, "bench", "micro_showdown_sim.py"), "--ps-dir", ps_dir, "--battles", "160",
                        "--record", "160", "--seed", "777", "--ipc-messages", "0", "--out", out], capture_output=True, text=True)
    path = os.path.join(out, "streams.jsonl")
    if r.returncode != 0 or not os.path.isfile(path):
        pytest.fail("could not record battles: " + (r.stdout + r.stderr)[-800:])
    return [path]


# --------------------------------------------------------------------------- the replay


def test_every_recorded_decision_gives_the_same_bytes(ref, ref_assembler_class):
    paths = _streams()
    need = int(os.environ.get("PSPPO_OBS_MIN", "10000"))
    from bench import micro_obs
    from poke_env.concurrency import POKE_LOOP

    ref_utils = ref.load("utils")
    assert ref_utils is not utils
    utils.UNKNOWN_ENTITIES.clear()
    ref_utils.UNKNOWN_ENTITIES.clear()
    with ref.active():
        old_asm = ref_assembler_class()
    stats = Counter()
    species, move_ids = set(), set()
    first_bad = []

    def watch(player):
        new_asm = player.assembler
        assert new_asm.fast and isinstance(new_asm, obs_assembler.ObservationAssembler)
        inner = new_asm.assemble

        def assemble(battle, events=None):
            old_err = new_err = None
            try:
                want = old_asm.assemble(battle, events=events)
            except Exception as e:  # noqa: BLE001
                old_err = e
            try:
                got = inner(battle, events=events)
            except Exception as e:  # noqa: BLE001
                new_err = e
            stats["decisions"] += 1
            if old_err is not None or new_err is not None:
                stats["old code raised"] += old_err is not None
                stats["new code raised"] += new_err is not None
                if type(old_err) is not type(new_err) or str(old_err) != str(new_err):
                    stats["different"] += 1
                    first_bad.append(("exception", repr(old_err), repr(new_err)))
                if new_err is not None:
                    raise new_err
                return got
            if want.dtype != got.dtype or want.shape != got.shape or want.tobytes() != got.tobytes():
                stats["different"] += 1
                if len(first_bad) < 5:
                    cols = np.flatnonzero(want.view(np.uint16) != got.view(np.uint16))[:12].tolist()
                    first_bad.append((battle.battle_tag, battle.turn, cols, want[cols].tolist(), got[cols].tolist()))
            # what this decision exercised
            mons = list(battle.team.values()) + list(battle.opponent_team.values())
            stats["a Terastallized Pokemon on the field or bench"] += any(m.is_terastallized for m in mons)
            stats["a transformed Pokemon (Transform, Imposter)"] += any(m.transformed for m in mons)
            stats["Zoroark in a team (Illusion)"] += any(m.species.startswith("zoroark") for m in mons)
            stats["an Illusion ending this turn (replace event)"] += any(len(e) > 1 and e[1] == "replace" for e in (events or []))
            stats["forced switch"] += bool(battle.force_switch)
            stats["team preview"] += bool(getattr(battle, "teampreview", False))
            avail = [m.id for m in battle.available_moves]
            stats["Struggle offered"] += "struggle" in avail
            stats["Recharge offered (a move with no table entry)"] += "recharge" in avail
            stats["no move offered"] += not avail
            stats["a Pokemon with more than four known moves"] += any(len(m.moves) > 4 for m in mons)
            stats["a move missing from the vocabulary"] += any(utils.normalize_name(mv.id) not in new_asm.vocab_map["move.id"]
                                                               for m in mons for mv in m.moves.values())
            stats["a Pokemon with a status"] += any(m.status for m in mons)
            stats["a Pokemon with volatile effects"] += any(m.effects for m in mons)
            stats["weather or a field"] += bool(battle.weather or battle.fields)
            stats["one legal action"] += float(got[new_asm.offsets["action_mask"][0]:new_asm.offsets["action_mask"][1]].sum()) == 1.0
            for m in mons:
                species.add(m.species)
                move_ids.update(m.moves)
            return got

        new_asm.assemble = assemble

    battles = 0
    for path in paths:
        _cfg, players, queue = micro_obs.build_players("ppo", "simple", 0, 7)
        for p in players.values():
            watch(p)
        msgs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                msgs.append(json.loads(line))
        battles += len({d["b"] for d in msgs})
        asyncio.run_coroutine_threadsafe(micro_obs.replay(players, msgs, {"request": micro_obs.Acc(), "update": micro_obs.Acc()}), POKE_LOOP).result()
        stats["decisions answered by the worker path"] += queue.steps

    print("\n" + json.dumps({"files": len(paths), "battles": battles, "species": len(species), "moves": len(move_ids),
                             "names missing from the vocabulary (kinds)": len(utils.UNKNOWN_ENTITIES),
                             "names missing from the vocabulary (count)": sum(utils.UNKNOWN_ENTITIES.values()),
                             **{k: int(v) for k, v in sorted(stats.items())}}, indent=2))
    assert stats["different"] == 0, first_bad
    assert stats["decisions"] == stats["decisions answered by the worker path"] + stats["new code raised"]
    assert stats["decisions"] >= need, f"only {stats['decisions']} decisions replayed, {need} wanted"
    assert dict(utils.UNKNOWN_ENTITIES) == dict(ref_utils.UNKNOWN_ENTITIES), "the count of names missing from the vocabulary moved"
    assert stats["a Terastallized Pokemon on the field or bench"] > 0 and stats["forced switch"] > 0
    assert stats["a move missing from the vocabulary"] > 0 and stats["team preview"] == 0
    if stats["decisions"] >= 40000:
        # A sample this large has always held these. A smaller one may not.
        assert stats["a transformed Pokemon (Transform, Imposter)"] > 0 and stats["Zoroark in a team (Illusion)"] > 0


# --------------------------------------------------------------------------- small checks, no recorded battles needed


def test_the_original_functions_are_untouched(ref):
    """The encoders fall back on them and fill their tables with them, so they must be the old code."""
    for mod_name, fn_name in ORIGINALS:
        new_obj, old_obj = sys.modules.get(mod_name) or __import__(mod_name), ref.load(mod_name)
        for part in fn_name.split("."):
            new_obj, old_obj = getattr(new_obj, part), getattr(old_obj, part)
        assert inspect.getsource(new_obj).replace("\r\n", "\n") == inspect.getsource(old_obj).replace("\r\n", "\n"), (mod_name, fn_name)


class _Mon:
    """Enough of a poke-env Pokemon for the encoders, with plain attributes."""

    def __init__(self, rng: random.Random, moves, types, status_values, genders, effects_values):
        stats = ("hp", "atk", "def", "spa", "spd", "spe")
        self.base_stats = {s: rng.choice([1, 5, 20, 45, 70, 80, 100, 120, 150, 200, 255]) for s in stats}
        if rng.random() < 0.05:
            del self.base_stats[rng.choice(stats)]                  # the functions default a missing stat to 100
        self.level = rng.choice([1, 50, 73, 80, 84, 88, 92, 100])
        self.item = rng.choice([None, "", "unknown_item", "choicescarf", "choiceband", "choicespecs", "assaultvest", "eviolite",
                                "lightball", "thickclub", "leftovers", "heavydutyboots"])
        self.ability = rng.choice([None, "swiftswim", "chlorophyll", "hugepower", "purepower", "furcoat", "unburden", "intimidate", "levitate"])
        pool = ["Swift Swim", "chlorophyll", "Huge Power", "furcoat", "Sand Rush", "intimidate", "Pure Power", "overgrow", "unburden"]
        self.possible_abilities = rng.sample(pool, rng.randint(0, 3)) if rng.random() < 0.9 else None
        self.moves = {m.id: m for m in rng.sample(moves, rng.randint(0, 5))}
        self.species = rng.choice(["pikachu", "ditto", "zoroark", "garchomp", "notapokemon", "ironvaliant"])
        self.current_hp_fraction = rng.choice([0, 0.01, 0.337, 0.5, 0.999, 1.0, 1])
        self.boosts = {k: rng.randint(-6, 6) for k in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion") if rng.random() < 0.8}
        self.weight = rng.choice([0.1, 0.05, 6, 6.0, 9.9, 10, 60.5, 100, 999.9, 1000, 0])
        self.height = rng.choice([0.1, 0.3, 1, 1.7, 2.0, 14.5, 20, 25])
        self.active = rng.random() < 0.3
        self.fainted = rng.random() < 0.1
        self.is_terastallized = rng.random() < 0.2
        self.status_counter = rng.choice([0, 0, 1, 3, 8, 9, 15, -1])
        self.protect_counter = rng.choice([0, 1, 2])
        self.first_turn = rng.random() < 0.3
        self.type_1 = rng.choice(types)
        self.type_2 = rng.choice(types + [None, None])
        self.tera_type = rng.choice(types + [None])
        self.effects = {e: 1 for e in rng.sample(effects_values, rng.randint(0, 3))}
        self.status = rng.choice(status_values + [None, None])
        self.gender = rng.choice(genders + [None])


def _fake_mons(n: int, seed: int):
    from poke_env.battle import Effect, Move, PokemonGender, PokemonType, Status
    from poke_env.data import GenData
    rng = random.Random(seed)
    ids = sorted(GenData.from_gen(9).moves)
    picked = rng.sample(ids, 220) + ["trickroom", "gyroball", "foulplay", "bodypress", "rapidspin", "earthquake", "tackle",
                                     "struggle", "triplekick", "bulletseed", "swordsdance", "thunderwave", "scald", "hiddenpower"]
    moves = []
    for i in picked:
        try:
            moves.append(Move(i, gen=9))
        except Exception:  # noqa: BLE001  (a few table entries cannot be built as moves)
            pass
    for raw in ("hiddenpowerfire60", "hiddenpowerice70", "hiddenpower60"):
        try:                                                      # poke-env keeps the number as a power of its own
            moves.append(Move("hiddenpower", gen=9, raw_id=raw))
        except Exception:  # noqa: BLE001
            pass
    args = (moves, list(PokemonType), list(Status), list(PokemonGender), list(Effect)[:40])
    return [_Mon(rng, *args) for _ in range(n)], moves


@pytest.fixture(scope="module")
def fake_mons():
    return _fake_mons(4000, 11)


def test_encoders_equal_the_original_functions_on_made_up_pokemon(ref, ref_assembler_class, fake_mons):
    """
    Thousands of made-up Pokemon (every stat, item, ability, boost and move combination the
    stat bounds look at, and far more variety than recorded games hold) through the reference
    commit's functions and through the encoders, slot by slot.
    """
    mons, _moves = fake_mons
    with ref.active():
        old = ref_assembler_class()
    old_moves, old_pok, old_abil = ref.load("obs_moves"), ref.load("obs_pokemon"), ref.load("obs_abilities")
    ref_utils = ref.load("utils")
    new = obs_assembler.ObservationAssembler()
    utils.UNKNOWN_ENTITIES.clear()
    ref_utils.UNKNOWN_ENTITIES.clear()
    rng = random.Random(5)
    a = np.zeros(new.total_dim, dtype=np.float16)
    b = np.zeros(new.total_dim, dtype=np.float16)
    bad = failed = 0
    for round_ in range(3):                     # the second and third rounds read the tables the first one filled
        for n, mon in enumerate(mons):
            slot = (n + round_) % 12
            is_self = slot < 6
            avail = [m for m in mon.moves if rng.random() < 0.6] if rng.random() < 0.8 else []
            who = None if n % 97 == 0 else mon
            a.fill(0), b.fill(0)
            old_err = new_err = None
            try:
                old_pok.encode_pokemon_body_inplace(who, a, slot, old.pokemon_scalar_dim, old.offsets, old.vocab_map, old.vocab_lists)
                old_abil.encode_ability_inplace(who, a, slot, old.offsets, old.vocab_map)
                old_moves.encode_moves_inplace(who, a, slot, old.move_scalar_dim, old.offsets, old.vocab_map, old.vocab_lists,
                                               is_self=is_self, available_move_ids=avail)
            except Exception as e:  # noqa: BLE001  (a move the original code cannot encode: both must fail alike)
                old_err = e
            try:
                new._body_encoder.encode(who, b, slot)
                new._ability_encoder.encode(who, b, slot)
                new._move_encoder.encode(who, b, slot, is_self=is_self, available_move_ids=avail)
            except Exception as e:  # noqa: BLE001
                new_err = e
            if old_err is not None or new_err is not None:
                failed += 1
                bad += type(old_err) is not type(new_err) or str(old_err) != str(new_err)
            else:
                bad += a.tobytes() != b.tobytes()
    assert bad == 0
    assert failed < len(mons) // 4, "most made-up Pokemon should encode"
    assert dict(utils.UNKNOWN_ENTITIES) == dict(ref_utils.UNKNOWN_ENTITIES) and len(utils.UNKNOWN_ENTITIES) > 0
    assert len(new._move_encoder._rows) > 200 and len(obs_pokemon._STAT_BLOCK) > 1000


def test_stat_block_matches_the_bounds_functions_and_survives_its_limit(fake_mons, monkeypatch):
    mons, _moves = fake_mons
    enc = obs_assembler.ObservationAssembler()._body_encoder
    monkeypatch.setattr(obs_pokemon, "_STAT_BLOCK_LIMIT", 50)           # start over every 50 entries
    obs_pokemon._STAT_BLOCK.clear()
    for mon in mons[:1500]:
        for _ in range(2):
            max_hp, block = enc._stats(mon, mon.moves, mon.item, getattr(mon, "ability", None))
            want = [obs_pokemon.estimate_stat(mon, "hp")]
            for stat in ("atk", "def", "spa", "spd", "spe"):
                want += list(obs_pokemon.get_combat_bounds(mon, stat))
            assert max_hp == want[0] and block.dtype == np.float16 and block.tolist() == [float(np.float16(x)) for x in want]
        assert len(obs_pokemon._STAT_BLOCK) <= 50


def test_other_buffers_and_odd_objects_take_the_original_path(fake_mons):
    """A float32 buffer, a slot outside the twelve, a move that is not a poke-env move."""
    mons, moves = fake_mons
    new = obs_assembler.ObservationAssembler()
    args_b = (new.pokemon_scalar_dim, new.offsets, new.vocab_map, new.vocab_lists)
    args_m = (new.move_scalar_dim, new.offsets, new.vocab_map, new.vocab_lists)
    multi = next(m for m in moves if m.id == "bulletseed")              # 3.1667 expected hits: not a float16 value
    mon = next(m for m in mons if len(m.moves) >= 2)
    mon.moves = dict(list(mon.moves.items())[:1] + [("bulletseed", multi)])
    a32 = np.zeros(new.total_dim, dtype=np.float32)
    b32 = np.zeros(new.total_dim, dtype=np.float32)
    obs_pokemon.encode_pokemon_body_inplace(mon, a32, 3, *args_b)
    obs_moves.encode_moves_inplace(mon, a32, 3, *args_m, is_self=True, available_move_ids=["bulletseed"])
    new._body_encoder.encode(mon, b32, 3)
    new._move_encoder.encode(mon, b32, 3, is_self=True, available_move_ids=["bulletseed"])
    assert a32.tobytes() == b32.tobytes()
    meta = obs_moves.get_move_metadata(new.vocab_lists)
    eh = new.offsets["move_scalars"][0] + 3 * new.move_scalar_dim + meta["single_move_dim"] + meta["feature_map"]["expected_hits"]
    assert a32[eh] != np.float32(np.float16(a32[eh])), "the float32 path would be the place a rounded row shows"

    # a move object without the attributes of a poke-env move
    odd = SimpleNamespace(id="tackle", accuracy=0.95, base_power=40, current_pp=3, expected_hits=1.0, boosts=None, target=None,
                          priority=0, type=None, category=None, status=None, secondary=[])
    holder = mons[0]
    holder.moves = {"tackle": odd}
    a = np.zeros(new.total_dim, dtype=np.float16)
    b = np.zeros(new.total_dim, dtype=np.float16)
    obs_pokemon.encode_pokemon_body_inplace(holder, a, 7, *args_b)
    obs_moves.encode_moves_inplace(holder, a, 7, *args_m, is_self=False, available_move_ids=[])
    new._body_encoder.encode(holder, b, 7)
    new._move_encoder.encode(holder, b, 7, is_self=False, available_move_ids=[])
    assert a.tobytes() == b.tobytes()


def test_assembler_can_be_asked_for_the_original_path():
    slow = obs_assembler.ObservationAssembler(fast=False)
    fast = obs_assembler.ObservationAssembler()
    assert slow.fast is False and fast.fast is True and slow.offsets == fast.offsets and slow.total_dim == fast.total_dim


def test_remembered_lookups_count_unknown_names_every_time():
    vocab = {"x": {"known": 4}}
    look = obs_abilities.IdLookup(vocab, "x")
    utils.UNKNOWN_ENTITIES.clear()
    want = Counter()
    for value in ["known", "Known", "nope", "nope", None, "", "none", "nope", 1, True, 1.0, ["unhashable"], "known"]:
        before = Counter(utils.UNKNOWN_ENTITIES)
        direct = utils.get_id(vocab, "x", value)
        counted = Counter(utils.UNKNOWN_ENTITIES) - before
        utils.UNKNOWN_ENTITIES.subtract(counted)                 # take the direct call's count out again
        want.update(counted)
        assert look(value) == direct, value
    assert +Counter(utils.UNKNOWN_ENTITIES) == want and sum(want.values()) == 7
