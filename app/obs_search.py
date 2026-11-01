"""
The observation of the "search" branch's own-Tera switch, so its checkpoints can be played here.

Two lines of code train networks that see their own Tera type, and they do not write it the same:

    this line (release r5 and later)   "obs_settings": {"own_tera_type": True, ...} in the checkpoint,
                                       built by obs_tera.OwnTeraObservationAssembler (the type the
                                       request names, corrected for a Zoroark's Illusion)
    branch "search" from 73f524b on    "run_cfg": {"obs": {"own_tera_type": True}}, no "obs_settings",
    (the baseline run of 2026-10-13)   built by ObservationAssembler(own_tera_type=True) of that
                                       branch: the type where poke-env has none, AND the STAB flag
                                       put back on moves of a Terastallized Pokemon's original types
                                       (either side), which this line never does

A "search" checkpoint loaded by this line's players used to be read as "trained without it" (no
"obs_settings"): it loaded, and played on inputs it was not trained with. hybrid.PolicyNet now asks
checkpoint_search_own_tera() and builds SearchOwnTeraObservationAssembler for such a checkpoint.

    SearchOwnTeraObservationAssembler   the bytes of the search branch's ObservationAssembler(
                                        own_tera_type=True), on top of this line's ObservationAssembler
    checkpoint_search_own_tera()        whether a loaded checkpoint is one of that line's

One difference is kept on purpose: this line's action mask leaves out a switch the server refuses
(switch_names.py, a Pokemon whose name an active Transformed Pokemon carries). The search branch
offers it, and its run logs the refusal ("Can't switch: You can't switch to an active Pokemon").

tests/test_obs_search.py compares the bytes with the search branch's own file where one is at hand
(PSPPO_SEARCH_REF: a directory holding that branch's obs_assembler.py).

Importing this module pulls in numpy and obs_assembler, not torch and not ray.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from obs_assembler import ObservationAssembler
from obs_tera import CHECKPOINT_KEY, own_tera_type_name

# What hybrid.PolicyNet.obs_line says for a checkpoint of that branch.
LINE = "search"


def checkpoint_search_own_tera(ckpt: Any) -> bool:
    """
    Whether a loaded checkpoint was trained by the "search" branch with its own-Tera switch on:
    its run_cfg says so (config.checkpoint_own_tera_type of that branch) and it has no
    "obs_settings" entry, which only this line's learner writes and which then decides.
    """
    if not isinstance(ckpt, dict) or isinstance(ckpt.get(CHECKPOINT_KEY), dict):
        return False
    run_cfg = ckpt.get("run_cfg")
    obs = run_cfg.get("obs") if isinstance(run_cfg, dict) else None
    return bool(obs.get("own_tera_type", False)) if isinstance(obs, dict) else False


class SearchOwnTeraObservationAssembler(ObservationAssembler):
    """
    ObservationAssembler plus what the search branch's own_tera_type=True adds (its commits
    c213830 and 16a975a), applied to the finished observation instead of inside the slot loop.
    Nothing that runs after those writes in the search branch reads the buffer, so the bytes are
    the same.

      * An own Pokemon for which poke-env has no Tera type gets the type the request names
        ("teraType") in its Tera type block. One poke-env has a type for is left as encoded.
      * A Terastallized Pokemon, own or the opponent's, gets the STAB flag set on every move that
        shares one of its original types (poke-env reports only the Tera type after the Tera, so
        the move encoders drop the flag; gen 9 keeps the bonus).
    """

    def __init__(self, vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True):
        super().__init__(vocab, fast)
        self.own_tera_type = True
        # Where a Pokemon's Tera type block starts inside its body: the second half of types_raw.
        types_start, types_end = self.meta["feature_map"]["body"]["types_raw"]
        self._tera_block = types_start + (types_end - types_start) // 2

    def assemble(self, battle: Any, events: List[List[str]] = None):
        obs = super().assemble(battle, events)
        # The same lists, in the same order, the base class filled the twelve slots from.
        self_team = list(battle.team.values())
        opp_team = list(battle.opponent_team.values())
        for i in range(12):
            is_self = i < 6
            mon = self_team[i] if (is_self and i < len(self_team)) else \
                  (opp_team[i - 6] if (not is_self and (i - 6) < len(opp_team)) else None)
            if mon is None:
                continue
            if is_self:
                self._write_own_tera_type(obs, mon, i)
            if getattr(mon, "is_terastallized", False):
                self._restore_original_stab(obs, mon, i)
        return obs

    def _write_own_tera_type(self, obs: Any, mon: Any, slot: int) -> None:
        if mon.tera_type is not None:
            return
        name = own_tera_type_name(mon)
        if not name:
            return
        t_idx = self.vocab_map.get("pokemon.type", {}).get(name, -1) + 1
        if t_idx > 0:
            obs[self.offsets["pokemon_body"][0] + slot * self.pokemon_scalar_dim + self._tera_block + t_idx] = 1.0

    def _restore_original_stab(self, obs: Any, mon: Any, slot: int) -> None:
        original = set(getattr(mon, "base_types", None) or [])
        if not original:
            return
        stab = self.meta["feature_map"]["move"]["stab_flag"]
        single = self.move_scalar_dim // 4
        start = self.offsets["move_scalars"][0] + slot * self.move_scalar_dim
        for m_idx, move in enumerate(list(mon.moves.values())[:4]):
            if move is not None and move.type in original:
                obs[start + m_idx * single + stab] = 1.0
