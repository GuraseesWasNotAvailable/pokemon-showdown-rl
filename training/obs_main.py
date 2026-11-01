"""
The main line's observation (release r5 / r6, the network M24381) on this code line, with this line's
matchup and moveset blocks appended after it.

    ObsConfig(main_line=True, own_tera_type=False, matchup=True, movesets=True)   (train.py --obs-main-line)

Why. The strongest network so far, main update 024381 (M24381), was trained on the main line's
observation: obs_tera.py's own Tera type (the request's type, a Zoroark's corrected, no STAB restored
after a Tera), the six fixes of obs_abil.py (abil_sets, abil_announced, abil_history, item_unknown,
spe_conditions, regenerator) and switch_names.py in the action mask. This line's own observation
differs from it in those places (its own_tera_type writes poke-env's type and restores STAB, and it
has none of the fixes), so a network trained on one plays the other on inputs it was not trained
with. With main_line the first 8,567 numbers are the main line's, byte for byte (the window app's
AbilityObservationAssembler, branch local-app, on the same battle), and the blocks follow as on this
line, so a widened M24381 (tools/widen_checkpoint.py --main-line) plays exactly as M24381 on day one.

How. obs_tera.py, obs_abil.py, switch_names.py and randbats_abilities_gen9.json are the app's files,
byte for byte (git hash-object equals local-app's). Their classes are put on top of MainLineBase, which
is this line's ObservationAssembler with:
  * the blocks of the run (matchup, movesets) and this line's own_tera_type OFF: obs_tera's overlay
    sets the attribute own_tera_type to True after the base's __init__, so the two places of the base
    that read it (the v2 own Tera write and the STAB restore) are no-ops here;
  * create_action_mask as the app's obs_assembler.py has it: a switch whose name the server would give
    to an active Pokemon that Transformed into its species is not offered (switch_names.py).
The overlays write only into the first part (their offsets are the base layout's), after the base has
assembled the whole observation, blocks included. Nothing of this module runs unless main_line is on:
config.RunConfig.make_assembler builds ObservationAssembler exactly as before without it.

Proof: bench/league_exactness.py (the worker's own players on 300 recorded battles against the app's
player: 12,048 decisions, the prefix byte for byte, logits and values bit for bit) and
tests/test_obs_main.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from obs_assembler import ObservationAssembler

SETTING = "main_line"
# What the first part is: the app's AbilityObservationAssembler with these settings (M24381's obs_settings).
MAIN_LINE_SETTINGS: Dict[str, bool] = {
    "own_tera_type": True, "abil_sets": True, "abil_announced": True, "abil_history": True,
    "item_unknown": True, "spe_conditions": True, "regenerator": True,
}
MAIN_LINE_DIM = 8567          # the first part's width (the base layout without the blocks)


class MainLineBase(ObservationAssembler):
    """
    This line's ObservationAssembler as the main line's base: the blocks as asked, this line's own Tera
    type never written, the app's action mask. Built by MainLineObservationAssembler only.
    """

    _main_matchup: bool = False
    _main_movesets: bool = False

    def __init__(self, vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True):
        super().__init__(vocab, fast, own_tera_type=False, matchup=self._main_matchup, movesets=self._main_movesets)

    # obs_tera.OwnTeraObservationAssembler sets self.own_tera_type = True after this __init__; the base
    # then calls these two. The main line's own Tera type is obs_tera's overlay alone, and it never
    # restored STAB after a Tera.
    def _write_own_tera_type(self, mon: Any, slot: int) -> None:
        return None

    def _restore_original_stab(self, mon: Any, slot: int) -> None:
        return None

    def create_action_mask(self, battle: Any, self_team: List[Any]) -> np.ndarray:
        """
        The app's ObservationAssembler.create_action_mask (branch local-app, obs_assembler.py), line for
        line: this line's, and a switch the server would refuse (switch_names.switch_name_taken) is not
        offered.
        """
        from switch_names import switch_name_taken

        mask = np.zeros(self.action_dim, dtype=np.float32)
        active_mon = battle.active_pokemon

        if not active_mon:
            return mask

        # 1. HANDLE MOVES & TERA (0-3 & 10-13)
        if battle.available_moves:
            mon_move_ids = [m.id for m in active_mon.moves.values()]
            for available_move in battle.available_moves:
                if available_move.id in mon_move_ids:
                    idx = mon_move_ids.index(available_move.id)
                    if idx < 4:
                        mask[idx] = 1.0
                        if battle.can_tera:
                            mask[idx + 10] = 1.0
                elif available_move.id == "struggle":
                    mask[0] = 1.0

            # Safety fallback for extreme edge cases (e.g., Transform bugs)
            if not mask[:4].any():
                mask[0] = 1.0

        # 2. HANDLE SWITCHES & REVIVALS (4-9)
        if battle.available_switches:
            for available_switch in battle.available_switches:
                try:
                    idx = self_team.index(available_switch)
                    # Not one whose name the server gives to an active Pokemon that Transformed
                    # into its species (switch_names.py): that order is refused.
                    if idx < 6 and not switch_name_taken(active_mon, available_switch):
                        mask[4 + idx] = 1.0
                except ValueError:
                    pass

        # 3. SAFETY FALLBACK
        if not mask.any():
            mask[0] = 1.0

        return mask


_CLASSES: Dict[tuple, type] = {}


def assembler_class(matchup: bool, movesets: bool) -> type:
    """
    The main line's assembler class with these blocks: obs_abil.AbilityObservationAssembler (and through
    it obs_tera.OwnTeraObservationAssembler) on MainLineBase, built with no arguments as worker.py builds
    every assembler. The MRO is AbilityObservationAssembler, OwnTeraObservationAssembler, MainLineBase,
    ObservationAssembler: obs_tera's super().__init__(vocab, fast) reaches MainLineBase.
    """
    if movesets and not matchup:
        raise ValueError("the moveset block needs the matchup block")
    key = (bool(matchup), bool(movesets))
    cls = _CLASSES.get(key)
    if cls is not None:
        return cls
    from obs_abil import AbilityObservationAssembler, parse_settings

    abil = parse_settings({k: v for k, v in MAIN_LINE_SETTINGS.items() if k != "own_tera_type"})
    base = type(f"MainLineBase_{int(key[0])}{int(key[1])}", (MainLineBase,),
                {"_main_matchup": key[0], "_main_movesets": key[1]})

    class MainLineObservationAssembler(AbilityObservationAssembler, base):   # type: ignore[valid-type, misc]
        """The main line's observation (obs_abil with every fix and the own Tera type), then the blocks."""

        main_line = True

        def __init__(self, vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True):
            AbilityObservationAssembler.__init__(self, vocab, fast, own_tera_type=True, settings=abil)

    MainLineObservationAssembler.__name__ = MainLineObservationAssembler.__qualname__ = (
        "MainLineObservationAssembler" + ("" if not key[0] else "_matchup" + ("_movesets" if key[1] else "")))
    _CLASSES[key] = MainLineObservationAssembler
    return MainLineObservationAssembler


def make_assembler(matchup: bool, movesets: bool) -> ObservationAssembler:
    """An assembler of the main line's observation with these blocks (config.RunConfig.make_assembler with main_line)."""
    return assembler_class(matchup, movesets)()


def checkpoint_main_line(ckpt: Any) -> bool:
    """Whether a learner checkpoint's network sees the main line's observation (run_cfg["obs"]["main_line"]). No key: it does not."""
    run_cfg = ckpt.get("run_cfg") if isinstance(ckpt, dict) else None
    obs = run_cfg.get("obs") if isinstance(run_cfg, dict) else None
    return bool(obs.get(SETTING, False)) if isinstance(obs, dict) else False


def checkpoint_has_main_settings(ckpt: Any) -> bool:
    """
    Whether a checkpoint of the main line itself (M24381: no run_cfg key, an "obs_settings" entry with the
    own Tera type and all six of obs_abil's fixes) was trained on the first part of this observation.
    """
    saved = ckpt.get("obs_settings") if isinstance(ckpt, dict) else None
    return isinstance(saved, dict) and all(saved.get(k) is True for k in MAIN_LINE_SETTINGS)


def checkpoint_obs_settings(matchup: bool, movesets: bool) -> Dict[str, bool]:
    """
    The "obs_settings" entry a learner with main_line writes into its checkpoints: the main line's
    settings and the blocks. The window app (branch local-app, hybrid.PolicyNet) reads that entry first
    and refuses settings it does not build ("matchup", "movesets", "main_line"), so it never plays such a
    network on its own 8,567-number observation by mistake.
    """
    out = dict(MAIN_LINE_SETTINGS)
    out[SETTING] = True
    if matchup:
        out["matchup"] = True
    if movesets:
        out["movesets"] = True
    return out
