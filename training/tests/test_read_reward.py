"""
Tests for read_reward.py, the stage 2 read tracker.

  - hand-made turns for the owner's example and every exclusion in the plan;
  - real ladder records (tests/fixtures/reads/<battle>.jsonl.gz: the app's records cut down to the
    protocol lines and the decisions) with every judged hit checked by hand against the lines;
  - the zero-sum property: both seats of the same games, the bonuses cancel exactly. On the self-play
    fixture always, and on 200 fresh self-play games from Showdown's simulator when node and
    ~/pokemon-showdown (PS_DIR) are there (tests/selfplay_logs.js).

    pytest tests/test_read_reward.py -s
"""

import glob
import gzip
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from read_reward import (OPP_MOVED, OPP_NA, OPP_UNSEEN, ReadTracker, other, protocol_decisions,  # noqa: E402
                         read_ladder_record, track_steps)
from bench.read_stats import selfplay_steps  # noqa: E402
from matchup import load_randbats_sets  # noqa: E402

SETS = load_randbats_sets()
needs_sets = pytest.mark.skipif(not SETS, reason="heuristic/cache/gen9randombattle.json missing")

FIX = os.path.join(ROOT, "tests", "fixtures", "reads")
PS_DIR = os.environ.get("PS_DIR", os.path.expanduser("~/pokemon-showdown"))


# ------------------------------------------------------------------------------------------- helpers

def header(p1="Vaporeon", p2="Arcanine", p1_extra=(), p2_extra=()):
    return ["|player|p1|Alice|1|", "|player|p2|Bob|2|", "|teamsize|p1|6", "|teamsize|p2|6", "|gen|9",
            "|tier|[Gen 9] Random Battle", "|", "|start",
            f"|switch|p1a: {p1}|{p1}, L84|100/100", *p1_extra,
            f"|switch|p2a: {p2}|{p2}, L84|100/100", *p2_extra, "|turn|1"]


def D1(action, forced=None):
    return ("p1", action, forced)


def D2(action, forced=None):
    return ("p2", action, forced)


def run(script, seat="p1", worker=False, infer_forced=False, sets=None):
    """Feed a script (protocol lines and per-seat decisions) to a tracker for `seat`. worker=True drops
    the '|turn|' lines as worker.py's event harvest does and passes the lines as split parts."""
    t = ReadTracker(seat=seat, keep_lines=True, sets=sets)
    for item in script:
        if isinstance(item, tuple):
            if item[0] == seat:
                t.decide(item[1], forced=None if infer_forced else item[2])
            continue
        if worker:
            if item.startswith(("|turn|", "|t:|", "|request|")):
                continue
            t.observe([item.split("|")])
        else:
            t.observe([item])
    return t.finalize(), t


def all_modes(script, seat, sets=None):
    """The labels must not depend on how the lines arrive."""
    base, t = run(script, seat, sets=sets)
    for kw in ({"worker": True}, {"infer_forced": True}, {"worker": True, "infer_forced": True}):
        other_labels, _ = run(script, seat, sets=sets, **kw)
        assert [(d.read_tier, d.being_read_tier, d.opp_action, d.forced) for d in other_labels] == \
            [(d.read_tier, d.being_read_tier, d.opp_action, d.forced) for d in base], kw
    return base, t


END = ["|", "|upkeep", "|turn|9"]


def owner_example(p1="Vaporeon", out="Arcanine", inc="Sceptile", move="Ice Beam", extra=()):
    """We (p1, a Water type) attack; they (p2) switch their Fire type out to a Grass type."""
    return header(p1, out) + [
        D1(1), D2(4),
        "|", "|t:|1", f"|switch|p2a: {inc}|{inc}, L87, F|100/100", *extra,
        f"|move|p1a: {p1}|{move}|p2a: {inc}", f"|-supereffective|p2a: {inc}", f"|-damage|p2a: {inc}|20/100",
    ] + END


# ------------------------------------------------------------------------------------- the definition

def test_owner_example_water_fire_grass():
    # Ice Beam: resisted by the Fire type that left, super effective on the Grass type that came in.
    labels, t = all_modes(owner_example(), "p1")
    d = labels[0]
    assert (d.read_tier, d.being_read_tier, d.opp_action) == (1.0, 0.0, OPP_UNSEEN)
    assert d.read.move == "icebeam" and d.read.label == "resisted"
    # The other seat sees the same turn as being read.
    labels2, _ = all_modes(owner_example(), "p2")
    assert (labels2[0].read_tier, labels2[0].being_read_tier, labels2[0].opp_action) == (0.0, 1.0, OPP_MOVED)
    assert t.net() + run(owner_example(), "p2")[1].net() == 0


def test_tiers_neutral_and_also_super_effective():
    labels, _ = all_modes(owner_example(out="Snorlax"), "p1")         # Ice neutral on Snorlax
    assert labels[0].read_tier == 0.6
    labels, _ = all_modes(owner_example(out="Garchomp"), "p1")        # Ice also super effective on Garchomp
    assert labels[0].read_tier == 0.0 and labels[0].read is not None and labels[0].read.label == "super"


def test_opp_action_slots_follow_the_team_order():
    script = owner_example() + [
        D1(0), D2(4),
        "|", "|switch|p2a: Arcanine|Arcanine, L84|100/100",
        "|move|p1a: Vaporeon|Surf|p2a: Arcanine", "|-supereffective|p2a: Arcanine", "|-damage|p2a: Arcanine|30/100",
        "|", "|upkeep", "|turn|10", D1(0), D2(0),
        "|", "|move|p1a: Vaporeon|Surf|p2a: Arcanine", "|-supereffective|p2a: Arcanine", "|-damage|p2a: Arcanine|0 fnt",
        "|faint|p2a: Arcanine", "|", "|upkeep",
    ]
    labels, _ = all_modes(script, "p1")
    # Turn 1: an unseen Pokemon; turn 9: back to Arcanine, slot 0 of the opponent's team; then it stayed.
    assert [d.opp_action for d in labels] == [OPP_UNSEEN, 1, OPP_MOVED]
    # Surf into Arcanine from Sceptile (resisted): a read too.
    assert labels[1].read_tier == 1.0
    # With the observation's own order passed in.
    t = ReadTracker(seat="p1")
    for item in owner_example():
        if isinstance(item, tuple):
            if item[0] == "p1":
                t.decide(item[1])
        else:
            t.observe([item])
    t.decide(0, forced=False, opp_team=["p2: Sceptile", "p2: Arcanine"])
    t.observe(["|", "|switch|p2a: Arcanine|Arcanine, L84|100/100", "|move|p1a: Vaporeon|Surf|p2a: Arcanine",
               "|", "|upkeep"])
    assert t.finalize()[1].opp_action == 2


# ------------------------------------------------------------------------------------------- exclusions

def test_faint_replacement_is_not_a_switch_by_choice():
    script = header() + [
        D1(0), D2(0),
        "|", "|move|p1a: Vaporeon|Surf|p2a: Arcanine", "|-supereffective|p2a: Arcanine",
        "|-damage|p2a: Arcanine|0 fnt", "|faint|p2a: Arcanine", "|", "|upkeep",
        D2(4, True),
        "|", "|switch|p2a: Sceptile|Sceptile, L87, F|100/100", "|turn|2", D1(1), D2(0),
        "|", "|move|p1a: Vaporeon|Ice Beam|p2a: Sceptile", "|-supereffective|p2a: Sceptile",
        "|-damage|p2a: Sceptile|20/100", "|", "|upkeep",
    ]
    for seat in ("p1", "p2"):
        labels, t = all_modes(script, seat)
        assert all(d.read_tier == 0 and d.being_read_tier == 0 for d in labels)
        assert not t.events
    labels, _ = run(script, "p1")
    assert [d.opp_action for d in labels] == [OPP_MOVED, OPP_MOVED]
    labels, _ = run(script, "p2")
    assert [d.forced for d in labels] == [False, True, False] and labels[1].opp_action == OPP_NA


@pytest.mark.parametrize("pivot", ["U-turn", "Volt Switch", "Flip Turn", "Parting Shot", "Teleport",
                                   "Chilly Reception", "Shed Tail", "Baton Pass"])
def test_pivots_are_not_switches_by_choice(pivot):
    script = header() + [
        D1(1), D2(0),
        "|", f"|move|p2a: Arcanine|{pivot}|p1a: Vaporeon", "|-damage|p1a: Vaporeon|90/100", D2(4, True),
        "|switch|p2a: Sceptile|Sceptile, L87, F|100/100",
        "|move|p1a: Vaporeon|Ice Beam|p2a: Sceptile", "|-supereffective|p2a: Sceptile",
        "|-damage|p2a: Sceptile|20/100",
    ] + END
    labels, t = all_modes(script, "p1")
    assert labels[0].read_tier == 0 and not t.events
    assert labels[0].opp_action == OPP_MOVED
    labels, _ = all_modes(script, "p2")
    assert [d.forced for d in labels] == [False, True]


def test_drags_are_not_switches_by_choice():
    # Red Card drags our attacker out; the slower foe then hits the Pokemon dragged in.
    script = header("Vaporeon", "Arcanine") + [
        D1(0), D2(1),
        "|", "|move|p1a: Vaporeon|Surf|p2a: Arcanine", "|-supereffective|p2a: Arcanine", "|-damage|p2a: Arcanine|40/100",
        "|-enditem|p2a: Arcanine|Red Card|[of] p1a: Vaporeon", "|drag|p1a: Garchomp|Garchomp, L80|100/100",
        "|move|p2a: Arcanine|Dragon Pulse|p1a: Garchomp", "|-supereffective|p1a: Garchomp",
        "|-damage|p1a: Garchomp|50/100",
    ] + END
    for seat in ("p1", "p2"):
        labels, t = all_modes(script, seat)
        assert not t.events and all(d.read_tier == d.being_read_tier == 0 for d in labels)
    # A drag line, even one before any move, is never a switch by choice.
    script = header() + [
        D1(1), D2(0),
        "|", "|drag|p2a: Sceptile|Sceptile, L87, F|100/100",
        "|move|p1a: Vaporeon|Ice Beam|p2a: Sceptile", "|-supereffective|p2a: Sceptile",
    ] + END
    labels, t = all_modes(script, "p1")
    assert not t.events and labels[0].opp_action == OPP_MOVED


@pytest.mark.parametrize("trigger", ["|-enditem|p2a: Arcanine|Eject Button", "|-enditem|p2a: Arcanine|Eject Pack",
                                     "|-activate|p2a: Arcanine|ability: Emergency Exit",
                                     "|-activate|p2a: Arcanine|ability: Wimp Out"])
def test_item_and_ability_forced_switches(trigger):
    # The switch comes before any move line, but an item or ability forced it (Eject Pack after an
    # Intimidate, for one): not a switch by choice, and what the opponent chose is unknown.
    script = header() + [
        D1(1), D2(0),
        "|", "|-ability|p1a: Vaporeon|Intimidate|boost", "|-unboost|p2a: Arcanine|atk|1", trigger,
        "|switch|p2a: Sceptile|Sceptile, L87, F|100/100",
        "|move|p1a: Vaporeon|Ice Beam|p2a: Sceptile", "|-supereffective|p2a: Sceptile",
        "|-damage|p2a: Sceptile|20/100",
    ] + END
    labels, t = all_modes(script, "p1")
    assert labels[0].read_tier == 0 and not t.events
    assert labels[0].opp_action == OPP_NA


def test_eject_button_after_a_hit_keeps_the_read():
    # They switch by choice, our hit is super effective on the switch-in, which then leaves by Eject Button.
    script = owner_example()[:-3] + [
        "|-enditem|p2a: Sceptile|Eject Button", "|switch|p2a: Snorlax|Snorlax, L84|100/100"] + END
    labels, _ = all_modes(script, "p1")
    assert labels[0].read_tier == 1.0 and labels[0].opp_action == OPP_UNSEEN


def test_illusion_same_turn():
    script = owner_example()[:-3] + [
        "|replace|p2a: Zoroark|Zoroark, L80, M", "|-end|p2a: Zoroark|Illusion"] + END
    labels, t = all_modes(script, "p1")
    assert labels[0].read_tier == 0 and t.events[0].excluded == "illusion"


def test_illusion_revealed_later_cancels_the_read():
    script = owner_example() + [
        D1(0), D2(0),
        "|", "|move|p1a: Vaporeon|Surf|p2a: Sceptile", "|-resisted|p2a: Sceptile", "|-damage|p2a: Sceptile|10/100",
        "|replace|p2a: Zoroark|Zoroark-Hisui, L80, M", "|-end|p2a: Zoroark|Illusion", "|", "|upkeep",
    ]
    labels, t = all_modes(script, "p1")
    assert labels[0].read_tier == 0 and t.events[0].excluded == "illusion"
    labels, _ = all_modes(script, "p2")
    assert labels[0].being_read_tier == 0


def test_illusion_broken_earlier_counts():
    # Zoroark-Hisui was revealed on turn 1; on turn 9 it leaves as itself and Close Combat is a read.
    script = header("Kommo-o", "Gengar") + [
        D1(0), D2(0),
        "|", "|move|p1a: Kommo-o|Dragon Claw|p2a: Gengar", "|-damage|p2a: Gengar|60/100",
        "|replace|p2a: Zoroark|Zoroark-Hisui, L80, M", "|-end|p2a: Zoroark|Illusion", "|", "|upkeep", "|turn|9",
        D1(1), D2(4),
        "|", "|switch|p2a: Tyranitar|Tyranitar, L80|100/100",
        "|move|p1a: Kommo-o|Close Combat|p2a: Tyranitar", "|-supereffective|p2a: Tyranitar",
    ] + END
    labels, t = all_modes(script, "p1")
    assert labels[1].read_tier == 1.0 and t.events[0].out_species == "Zoroark-Hisui"
    assert t.events[0].label == "immune"


def test_double_switch():
    script = header() + [
        D1(4), D2(4),
        "|", "|switch|p1a: Garchomp|Garchomp, L80|100/100", "|switch|p2a: Sceptile|Sceptile, L87, F|100/100",
    ] + END
    for seat in ("p1", "p2"):
        labels, t = all_modes(script, seat)
        assert not t.events and labels[0].read_tier == labels[0].being_read_tier == 0
        assert labels[0].opp_action == OPP_UNSEEN


@pytest.mark.parametrize("hit", [
    ["|move|p1a: Vaporeon|Toxic|p2a: Sceptile", "|-status|p2a: Sceptile|tox"],                       # status move
    ["|move|p1a: Vaporeon|Ice Beam|p2a: Sceptile|[miss]", "|-miss|p1a: Vaporeon|p2a: Sceptile"],     # miss
    ["|move|p1a: Vaporeon|Ice Beam|p2a: Sceptile", "|-immune|p2a: Sceptile"],                        # immune
    ["|move|p1a: Vaporeon|Ice Beam|p2a: Sceptile", "|-activate|p2a: Sceptile|move: Protect"],        # Protect
    ["|move|p1a: Vaporeon|Sucker Punch|p2a: Sceptile|[still]", "|-fail|p1a: Vaporeon"],               # failed
    ["|move|p1a: Vaporeon|Surf|p2a: Sceptile", "|-resisted|p2a: Sceptile"],                          # not super effective
    ["|move|p1a: Vaporeon|Sleep Talk|p1a: Vaporeon",                                                  # called move
     "|move|p1a: Vaporeon|Ice Beam|p2a: Sceptile|[from]move: Sleep Talk", "|-supereffective|p2a: Sceptile"],
    ["|move|p1a: Vaporeon|Ice Beam|p2a: Sceptile|[from]lockedmove", "|-supereffective|p2a: Sceptile"],  # locked move
])
def test_no_read_without_our_own_super_effective_hit(hit):
    script = header() + [D1(1), D2(4), "|", "|switch|p2a: Sceptile|Sceptile, L87, F|100/100", *hit] + END
    labels, t = all_modes(script, "p1")
    assert labels[0].read_tier == 0 and not t.events
    assert labels[0].opp_action == OPP_UNSEEN


def test_future_sight_hit_is_not_our_move():
    script = header() + [
        D1(1), D2(4),
        "|", "|switch|p2a: Sceptile|Sceptile, L87, F|100/100",
        "|move|p1a: Vaporeon|Calm Mind|p1a: Vaporeon", "|-boost|p1a: Vaporeon|spa|1",
        "|", "|-end|p2a: Sceptile|move: Future Sight", "|-supereffective|p2a: Sceptile", "|-damage|p2a: Sceptile|50/100",
    ] + END
    labels, t = all_modes(script, "p1")
    assert labels[0].read_tier == 0 and not t.events


# ------------------------------------------------------------------------------ type rules on the switch

def test_tera_blast_counts_with_its_tera_type():
    # Our Clefable Terastallized Fire earlier. Tera Blast: Fire, resisted by the Azumarill that left,
    # super effective on Scizor. Without the Tera it would be Normal (neutral on Azumarill, 0.6).
    script = header("Clefable", "Azumarill") + [
        D1(10), D2(0),
        "|", "|-terastallize|p1a: Clefable|Fire", "|move|p1a: Clefable|Moonblast|p2a: Azumarill",
        "|-resisted|p2a: Azumarill", "|-damage|p2a: Azumarill|90/100", "|", "|upkeep", "|turn|2",
        D1(1), D2(4),
        "|", "|switch|p2a: Scizor|Scizor, L80, M|100/100",
        "|move|p1a: Clefable|Tera Blast|p2a: Scizor|[anim] Tera Blast Fire", "|-supereffective|p2a: Scizor",
    ] + END
    labels, t = all_modes(script, "p1")
    assert labels[1].read_tier == 1.0
    assert t.events[0].move_type == "Fire" and t.events[0].label == "resisted"


def test_tera_of_the_outgoing_pokemon():
    # The Arcanine that left had Terastallized Water: Ice is resisted either way, Electric is not.
    base = header("Vaporeon", "Arcanine") + [
        D1(0), D2(0), "|", "|-terastallize|p2a: Arcanine|Water", "|move|p2a: Arcanine|Flare Blitz|p1a: Vaporeon",
        "|-resisted|p1a: Vaporeon", "|", "|upkeep", "|turn|2", D1(1), D2(4),
        "|", "|switch|p2a: Gyarados|Gyarados, L80, M|100/100",
        "|move|p1a: Vaporeon|Thunderbolt|p2a: Gyarados", "|-supereffective|p2a: Gyarados",
    ] + END
    labels, t = all_modes(base, "p1")
    # Thunderbolt on Tera-Water Arcanine: super effective, so not a read.
    assert labels[1].read_tier == 0.0 and t.events[0].label == "super"


def test_typechange_and_neutralizing_gas():
    # Soak made the Arcanine that left a Water type: Thunderbolt would have been super effective on it.
    script = header("Vaporeon", "Arcanine") + [
        D1(0), D2(0), "|", "|move|p1a: Vaporeon|Soak|p2a: Arcanine",
        "|-start|p2a: Arcanine|typechange|Water|[from] move: Soak", "|", "|upkeep", "|turn|2",
        D1(1), D2(4), "|", "|switch|p2a: Gyarados|Gyarados, L80, M|100/100",
        "|move|p1a: Vaporeon|Thunderbolt|p2a: Gyarados", "|-supereffective|p2a: Gyarados"] + END
    labels, t = all_modes(script, "p1")
    assert labels[1].read_tier == 0.0 and t.events[0].label == "super"
    # Our Neutralizing Gas would have switched the outgoing Cryogonal's Levitate off: neutral, 0.6.
    sets = {"Cryogonal": {"abilities": ["Levitate"]}, "Heatran": {"abilities": ["Flash Fire"]}}
    script = header("Weezing", "Cryogonal", p1_extra=["|-ability|p1a: Weezing|Neutralizing Gas"]) + [
        D1(1), D2(4), "|", "|switch|p2a: Heatran|Heatran, L80, M|100/100",
        "|move|p1a: Weezing|Earthquake|p2a: Heatran", "|-supereffective|p2a: Heatran"] + END
    labels, t = all_modes(script, "p1", sets=sets)
    assert labels[0].read_tier == 0.6 and t.events[0].label == "neutral"


def test_levitate_on_the_outgoing_pokemon():
    # Cryogonal (Ice) would take Earthquake neutrally, but its only random-battle ability is Levitate:
    # immune, tier 1.0.
    sets = {"Cryogonal": {"abilities": ["Levitate"]}, "Heatran": {"abilities": ["Flash Fire"]}}
    labels, t = all_modes(owner_example("Garchomp", "Cryogonal", "Heatran", "Earthquake"), "p1", sets=sets)
    assert labels[0].read_tier == 1.0 and t.events[0].label == "immune"
    # Without the set data Levitate is not assumed: Earthquake is neutral on an Ice type, 0.6.
    labels, t = all_modes(owner_example("Garchomp", "Cryogonal", "Heatran", "Earthquake"), "p1", sets={})
    assert labels[0].read_tier == 0.6 and t.events[0].label == "neutral"


def test_levitate_counts_when_revealed_or_the_only_choice():
    # Sets with two abilities, one of them Levitate: unrevealed, Levitate is not assumed (Earthquake is
    # super effective on Poison Weezing); revealed by an earlier immunity, it is.
    sets = {"Weezing": {"abilities": ["Levitate", "Neutralizing Gas"]}, "Heatran": {"abilities": ["Flash Fire"]}}
    switch_turn = [D1(1), D2(4), "|", "|switch|p2a: Heatran|Heatran, L80, M|100/100",
                   "|move|p1a: Garchomp|Earthquake|p2a: Heatran", "|-supereffective|p2a: Heatran"] + END
    labels, t = all_modes(header("Garchomp", "Weezing") + switch_turn, "p1", sets=sets)
    assert labels[0].read_tier == 0.0 and t.events[0].label == "super"
    revealed = header("Garchomp", "Weezing") + [
        D1(0), D2(0), "|", "|move|p1a: Garchomp|Earthquake|p2a: Weezing",
        "|-immune|p2a: Weezing|[from] ability: Levitate", "|", "|upkeep", "|turn|2"] + switch_turn
    labels, t = all_modes(revealed, "p1", sets=sets)
    assert labels[1].read_tier == 1.0 and t.events[0].label == "immune"


@pytest.mark.parametrize("balloon,tier", [("revealed", 1.0), ("popped", 0.0), ("none", 0.0)])
def test_air_balloon_on_the_outgoing_pokemon(balloon, tier):
    p2_extra = ["|-item|p2a: Heatran|Air Balloon"] if balloon != "none" else []
    pre = []
    if balloon == "popped":
        pre = [D1(0), D2(0), "|", "|move|p1a: Garchomp|Dragon Claw|p2a: Heatran", "|-resisted|p2a: Heatran",
               "|-damage|p2a: Heatran|90/100", "|-enditem|p2a: Heatran|Air Balloon", "|", "|upkeep", "|turn|2"]
    script = header("Garchomp", "Heatran", p2_extra=p2_extra) + pre + [
        D1(1), D2(4), "|", "|switch|p2a: Arcanine|Arcanine, L84|100/100",
        "|move|p1a: Garchomp|Earthquake|p2a: Arcanine", "|-supereffective|p2a: Arcanine",
    ] + END
    labels, t = all_modes(script, "p1")
    assert labels[-1].read_tier == tier
    assert t.events[0].label == ("immune" if balloon == "revealed" else "super")


# ------------------------------------------------------------------------------------- decisions

def test_forced_decisions_get_no_labels_and_are_inferred():
    script = header() + [
        D1(0), D2(0),
        "|", "|move|p2a: Arcanine|Wild Charge|p1a: Vaporeon", "|-supereffective|p1a: Vaporeon",
        "|-damage|p1a: Vaporeon|0 fnt", "|faint|p1a: Vaporeon", "|", "|upkeep",
        D1(4, True),
        "|", "|switch|p1a: Garchomp|Garchomp, L80|100/100", "|turn|2",
        D1(1), D2(0),
        "|", "|move|p1a: Garchomp|U-turn|p2a: Arcanine", "|-damage|p2a: Arcanine|60/100", D1(4, True),
        "|switch|p1a: Vaporeon2|Vaporeon, L84|100/100", "|move|p2a: Arcanine|Flare Blitz|p1a: Vaporeon2",
        "|-resisted|p1a: Vaporeon2",
    ] + END
    labels, _ = all_modes(script, "p1")
    assert [d.forced for d in labels] == [False, True, False, True]
    assert [d.opp_action for d in labels] == [OPP_MOVED, OPP_NA, OPP_MOVED, OPP_NA]


def test_action_must_be_a_move_for_a_read():
    script = owner_example()
    script = [D1(5) if (isinstance(x, tuple) and x[0] == "p1") else x for x in script]
    labels, t = run(script, "p1")
    assert labels[0].read_tier == 0 and t.action_mismatch == 1


def test_game_ending_before_the_opponent_acts():
    script = header() + [D1(0), D2(0), "|", "|-message|Bob forfeited.", "|win|Alice"]
    labels, _ = all_modes(script, "p1")
    assert labels[0].opp_action == OPP_NA


# --------------------------------------------------------------------------------- real ladder games

# Every hit on a switch-in that the tracker judges in these records, checked by hand against the lines
# (turn, reader, outgoing, incoming, move, tier).
GOLDEN = {
    # Reuniclus' Focus Blast into Urshifu: Zoroark-Hisui (revealed two turns before) is immune to it.
    "2694973334": [(4, "p1", "Zoroark-Hisui", "Urshifu", "focusblast", 1.0)],
    # A reconnection replays the battle (second |init|): each turn is still read once.
    # Focus Blast: resisted by Heracross, super effective on Incineroar. Sludge Wave: Iron Crown (Steel)
    # is immune, Arboliva (Grass) is weak.
    "2697643669": [(6, "p1", "Heracross", "Incineroar", "focusblast", 1.0),
                   (16, "p1", "Iron Crown", "Arboliva", "sludgewave", 1.0)],
    # Earthquake and Ground Tera Blast are also super effective on Toxapex: 0. Turn 24: the Clodsire that
    # left had Terastallized Flying earlier: immune, 1.0.
    "2697830575": [(15, "p2", "Toxapex", "Klefki", "earthquake", 0.0),
                   (21, "p2", "Toxapex", "Clodsire", "terablast", 0.0),
                   (24, "p2", "Clodsire", "Toxapex", "terablast", 1.0)],
    # Earth Power: neutral on the Zoroark (revealed on turn 22) that left, 4x on the real Salazzle.
    "2698004805": [(24, "p1", "Zoroark", "Salazzle", "earthpower", 0.6)],
    # Ice Punch: Iron Bundle (Water/Ice) resists, Krookodile is weak. Lash Out: Chien-Pao (Dark) resists,
    # Sinistcha (Ghost) is weak. Sacred Sword: super effective on Oinkologne too (Cresselia had Tera Steel).
    "2698044488": [(4, "p1", "Iron Bundle", "Krookodile", "icepunch", 1.0),
                   (19, "p1", "Chien-Pao", "Sinistcha-Masterpiece", "lashout", 1.0),
                   (40, "p2", "Oinkologne", "Cresselia", "sacredsword", 0.0)],
    # Knock Off: super effective on Chandelure too. Air Slash: Rotom-Wash resists, Vespiquen is weak, 3 times.
    "2698791356": [(10, "p1", "Chandelure", "Oricorio-Sensu", "knockoff", 0.0),
                   (33, "p2", "Rotom-Wash", "Vespiquen", "airslash", 1.0),
                   (37, "p2", "Rotom-Wash", "Vespiquen", "airslash", 1.0),
                   (41, "p2", "Rotom-Wash", "Vespiquen", "airslash", 1.0)],
}


@needs_sets
@pytest.mark.parametrize("battle", sorted(GOLDEN))
def test_real_ladder_games(battle):
    if not os.path.exists(os.path.join(FIX, f"{battle}.jsonl.gz")):
        pytest.skip("recorded public ladder games are not published with the code")
    seat, steps = read_ladder_record(os.path.join(FIX, f"{battle}.jsonl.gz"))
    assert seat in ("p1", "p2")
    t = track_steps(steps, seat)
    got = [(e.turn, e.reader, e.out_species, e.in_species, e.move, e.tier) for e in t.events if e.counts]
    assert got == GOLDEN[battle]
    # Our seat's labels carry exactly its reads and the reads against it, on decisions of the right kind.
    labels = t.labels()
    for d in labels:
        if d.read is not None:
            assert d.read.reader == seat and (0 <= d.action <= 3 or 10 <= d.action <= 13)
        if d.being_read is not None:
            assert d.being_read.switcher == seat and 4 <= d.action <= 9
    assert sum(d.read_tier for d in labels) == pytest.approx(sum(g[5] for g in GOLDEN[battle] if g[1] == seat))
    assert sum(d.being_read_tier for d in labels) == pytest.approx(
        sum(g[5] for g in GOLDEN[battle] if g[1] != seat))
    # The other seat, with decisions made up from the protocol, sees the mirror image.
    o = track_steps(protocol_decisions(steps, other(seat)), other(seat))
    assert t.net() + o.net() == pytest.approx(0, abs=1e-12)
    # Forced decisions inferred from the lines match the record's.
    inferred = track_steps(steps, seat, infer_forced=True)
    assert [d.forced for d in inferred.labels()] == [d.forced for d in labels]


# ------------------------------------------------------------------------------------------ zero-sum

def _zero_sum(game):
    """Both seats over one self-play game, in the three ways lines can arrive; returns the reads."""
    out = []
    for drop_turn in (False, True):
        for infer in (False, True):
            ts = {s: track_steps(selfplay_steps(game[s], drop_turn_lines=drop_turn), s, infer_forced=infer)
                  for s in ("p1", "p2")}
            for s in ("p1", "p2"):
                o = other(s)
                reads = sorted(d.read_tier for d in ts[s].labels() if d.read_tier)
                against = sorted(d.being_read_tier for d in ts[o].labels() if d.being_read_tier)
                assert reads == against, (game.get("game"), s, drop_turn, infer)
            assert ts["p1"].net() + ts["p2"].net() == pytest.approx(0, abs=1e-12)
            out.append(sum(1 for d in ts["p1"].labels() if d.read_tier) + sum(
                1 for d in ts["p2"].labels() if d.read_tier))
    assert len(set(out)) == 1          # the same reads whichever way the lines came
    return out[0]


def test_zero_sum_on_selfplay_fixture():
    with gzip.open(os.path.join(FIX, "selfplay3.jsonl.gz"), "rt", encoding="utf-8") as f:
        games = [json.loads(line) for line in f if line.strip()]
    assert len(games) == 3
    assert sum(_zero_sum(g) for g in games) >= 6


@pytest.mark.skipif(not (shutil.which("node") and os.path.isdir(os.path.join(PS_DIR, "dist", "sim"))),
                    reason="needs node and a built Showdown in PS_DIR (~/pokemon-showdown)")
def test_zero_sum_on_200_selfplay_games(tmp_path):
    out = tmp_path / "selfplay.jsonl"
    r = subprocess.run(["node", os.path.join(ROOT, "tests", "selfplay_logs.js"), "200", str(out), "11"],
                       capture_output=True, text=True, timeout=900, env=dict(os.environ, PS_DIR=PS_DIR))
    assert r.returncode == 0, r.stderr[-2000:]
    games = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(games) == 200
    reads = sum(_zero_sum(g) for g in games)
    print(f"\n200 self-play games: {reads} reads, both seats cancel in every game")
    assert reads > 50
