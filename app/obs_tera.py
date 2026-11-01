"""
The player's own Tera type as an input the network can see.

poke-env 0.15 never reads "teraType" from the request, so Pokemon.tera_type is None for the own
team of a random battle until the "-terastallize" line, and the observation's Tera type block of
an own Pokemon stays empty until then: the policy chooses whether to Terastallize without seeing
into which type (bench/notes/tera.md on branch "search": 0 of 490,108 decisions showed it).

    OwnTeraObservationAssembler   ObservationAssembler that also writes the Tera type the
                                  request names for every own Pokemon into that block
    make_assembler(own_tera_type) the assembler for a setting: ObservationAssembler itself when
                                  it is off, so "off" is the code of before, not a copy of it
    checkpoint_obs_settings()     what a learner checkpoint stores about it
    checkpoint_own_tera_type()    what a checkpoint says: a checkpoint without the entry is "off"

Nothing of this is in obs_assembler.py, on purpose: tests/test_ppo_health.py holds that file to
release r1 byte for byte, and with the setting off no line of this module runs. The bytes
written are those of ObservationAssembler(own_tera_type=True) in commit 53c2809 (branch
"search"), where this was first written into obs_assembler.py itself; tests/test_own_tera.py
compares the two on a real request.

The layout and the size of the observation do not change (the block was there, empty), so a
network trained without the input can be trained further with it. train.py --own-tera-type
switches it on for the rollout workers (tera_worker.py) and the learner then writes
"obs_settings": {"own_tera_type": True} into its checkpoints; hybrid.PolicyNet reads that, so
evaluate.py, play.py and ladder.py build the observation the way the checkpoint was trained.

Importing this module pulls in numpy and obs_assembler, not torch and not ray.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from obs_assembler import ObservationAssembler

# The key of the entry in a learner checkpoint, beside "model", "run_cfg" and the rest.
CHECKPOINT_KEY = "obs_settings"


def own_tera_type_name(mon: Any) -> Optional[str]:
    """
    The Tera type of one of the player's own Pokemon, lower case, or None when it is not known.

    poke-env 0.15 keeps the Pokemon's entry of the last request (Pokemon._last_request) but never
    reads its "teraType", so Pokemon.tera_type stays None in a random battle until the Pokemon
    Terastallizes. This reads it: first what poke-env knows, then the request.

    The entry is the Pokemon's own: poke-env files every entry of side.pokemon under its
    "ident", so the order of the request (the active Pokemon first, changing with every switch)
    does not matter, and a Zoroark is filed under its own name whatever it looks like.
    """
    known = getattr(mon, "tera_type", None)
    if known is not None:
        return str(getattr(known, "name", known)).lower()
    request = getattr(mon, "_last_request", None)
    name = request.get("teraType") if isinstance(request, dict) else None
    return str(name).lower() if name else None


def request_tera_type_name(mon: Any) -> Optional[str]:
    """
    The Tera type the server's last request names for one of the player's own Pokemon, lower
    case: the type it took if it has Terastallized ("terastallized"), else its "teraType". None
    when the Pokemon has no request entry or the entry names none.

    For the own team the request is the truth, and poke-env's Pokemon.tera_type is not always:
    a Zoroark that Terastallizes while disguised is reported as "-terastallize|p1a: <the
    Pokemon it looks like>|<type>", and poke-env files that Tera type, and the "tera:X" in
    later switch details, on the Pokemon it looked like. The review measured it on live
    self-play: 230 of 759,624 own blocks wrong, every one on a team with a Zoroark.
    """
    request = getattr(mon, "_last_request", None)
    if not isinstance(request, dict):
        return None
    name = request.get("terastallized") or request.get("teraType")
    return str(name).lower() if name else None


class OwnTeraObservationAssembler(ObservationAssembler):
    """
    ObservationAssembler that shows the player its own Pokemon's Tera types.

    Everything is the base class's. After it has assembled an observation, the Tera type block of
    each own Pokemon (where the encoders write Pokemon.tera_type) shows the type the server's
    request names for it: the type it took if it has Terastallized, else its Tera type
    (request_tera_type_name). Where poke-env has no Tera type for the Pokemon the block was
    empty and the type is written. Where poke-env has one and it is the request's (after
    Terastallizing, a team from the teambuilder) the block is left as the encoders wrote it.
    Where poke-env has one and the request names another (a Zoroark's Tera filed on the Pokemon
    it was disguised as) the block is cleared and the request's type written. A Pokemon without a
    request entry, or whose entry names a type the vocabulary lacks, keeps what the encoders wrote.
    Nothing else of the observation is touched: the six own blocks at most.

    A Pokemon that has not Terastallized keeps its type in the block after another of the team
    has (the server keeps naming it). Whether a Pokemon IS Terastallized is its Tera flag in the
    body, and whether the side still can is the global flag, as before (neither is changed here).

    tera_counts: how often the type was written where poke-env had none ("shown"), how often
    there was none to write ("unknown"), and how often poke-env's type was replaced by the
    request's ("corrected"), for the worker's heartbeat (tera_worker.py). Plain numbers, read
    from another thread a decision out of date at worst.
    """

    def __init__(self, vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True, own_tera_type: bool = True):
        super().__init__(vocab, fast)
        self.own_tera_type = bool(own_tera_type)
        # Where a Pokemon's Tera type block starts inside its body: the second half of types_raw.
        types_start, types_end = self.meta["feature_map"]["body"]["types_raw"]
        self._tera_block = types_start + (types_end - types_start) // 2
        self._tera_body_base = self.offsets["pokemon_body"][0]
        self._tera_block_len = (types_end - types_start) // 2
        self._tera_type_ids = self.vocab_map.get("pokemon.type", {})
        self.tera_counts = {"observations": 0, "shown": 0, "unknown": 0, "corrected": 0}

    def assemble(self, battle: Any, events: List[List[str]] = None):
        obs = super().assemble(battle, events)
        if self.own_tera_type:
            self._show_own_tera_types(battle, obs)
        return obs

    def _show_own_tera_types(self, battle: Any, obs: Any) -> None:
        counts = self.tera_counts
        counts["observations"] += 1
        ids = self._tera_type_ids
        dim = self.pokemon_scalar_dim
        at = self._tera_body_base + self._tera_block
        width = self._tera_block_len
        # The same list, in the same order, the base class filled the six own slots from.
        for slot, mon in enumerate(list(battle.team.values())[:6]):
            if mon is None:
                continue
            name = request_tera_type_name(mon)
            t_idx = (ids.get(name, -1) + 1) if name else 0
            start = at + slot * dim
            known = mon.tera_type
            if known is None:
                if t_idx > 0:
                    obs[start:start + width] = 0.0             # empty already: the encoders write only a known type
                    obs[start + t_idx] = 1.0
                    counts["shown"] += 1
                else:
                    counts["unknown"] += 1
            elif t_idx > 0 and str(getattr(known, "name", known)).lower() != name:
                obs[start:start + width] = 0.0                 # poke-env's type, filed here by an Illusion: replaced
                obs[start + t_idx] = 1.0
                counts["corrected"] += 1


def make_assembler(own_tera_type: bool = False, **kwargs: Any) -> ObservationAssembler:
    """
    The assembler for a setting. Off: ObservationAssembler(**kwargs), the class and the code
    every checkpoint so far was trained on. On: OwnTeraObservationAssembler.
    """
    if own_tera_type:
        return OwnTeraObservationAssembler(own_tera_type=True, **kwargs)
    return ObservationAssembler(**kwargs)


def checkpoint_obs_settings(own_tera_type: bool) -> Optional[Dict[str, Any]]:
    """
    What a run tells its learner to store in every checkpoint, or None when there is nothing to
    say: with the setting off the learner is built and saves exactly as it did before.
    """
    return {"own_tera_type": True} if own_tera_type else None


def checkpoint_own_tera_type(ckpt: Any) -> bool:
    """
    Whether the network of a loaded checkpoint (the dict torch.load gives) was trained with its
    own Tera types shown. A checkpoint without the entry, which is every checkpoint written
    before this existed, means "off"; so does anything that is not a learner checkpoint.
    """
    if not isinstance(ckpt, dict):
        return False
    settings = ckpt.get(CHECKPOINT_KEY)
    if not isinstance(settings, dict):
        return False
    return settings.get("own_tera_type") is True
