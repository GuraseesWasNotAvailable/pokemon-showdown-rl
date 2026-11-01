"""
obs_league: the networks of the league run (branch league, ~/runs/v2/league), played here exactly as
that branch trained them. The league run started from M24381 widened: it sees this app's own
observation, the main line's (obs_tera.py's own Tera type, obs_abil.py's six fixes, switch_names.py in
the action mask: what M24381 sees), with train-v2's matchup and moveset blocks appended after it.

The league branch builds that observation in its obs_main.py: the app's obs_tera.py, obs_abil.py and
switch_names.py (byte for byte) on its MainLineBase, which is its ObservationAssembler with the blocks
and its own Tera write made a no-op. Its base assembles the whole observation, blocks included, and the
overlays (obs_tera, obs_abil) then fix the first part. Its learner writes into every checkpoint an
"obs_settings" entry with the main line's settings and "main_line", "matchup", "movesets" named
(obs_main.checkpoint_obs_settings), which hybrid.unknown_obs_settings refuses: before this module the
app refused such a network instead of playing it on its 8,567 numbers.

hybrid.PolicyNet asks checkpoint_league_blocks() first, and for such a checkpoint builds

    the network     RunConfig.make_model(matchup=True, movesets=...), as for obs_v2: the first part is
                    8,567 numbers wide on both lines, so the input columns are the same
    the weights     the checkpoint's, without the training-only heads (obs_v2.AUX_HEAD_PREFIXES)
    the assembler   league_assembler(movesets): this app's AbilityObservationAssembler with the main
                    line's settings (M24381's) on BlocksBase, this app's ObservationAssembler with the
                    blocks appended after the action mask and written at the end of its assemble, as
                    the league's base writes them. The MRO is the league's: AbilityObservationAssembler,
                    OwnTeraObservationAssembler, BlocksBase, ObservationAssembler, so the overlays fix
                    the first part after the blocks are written, in the league's order.

The action mask is this app's own (create_action_mask with switch_names), which is what the league
trained on: the player chooses from the observation's mask, as for M24381.

obs_matchup.py, obs_movesets.py and matchup.py are the files obs_v2 already uses: the league branch has
them byte for byte but for the one import line of own_tera_type_name (from its obs_assembler; the same
function as obs_tera's). bench/v2_exactness.py app against the league branch's bench/league_exactness.py
ref on recorded battles holds the observation byte for byte and the logits and values bit for bit.

Importing this module pulls in numpy and obs_assembler, not torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from obs_assembler import ObservationAssembler

# What hybrid.PolicyNet.obs_line says for such a checkpoint.
LINE = "league"

# The main line's settings (M24381's obs_settings), which the league's obs_main.MAIN_LINE_SETTINGS
# names: the first part of the observation is the app's AbilityObservationAssembler with these.
MAIN_LINE_SETTINGS: Dict[str, bool] = {
    "own_tera_type": True, "abil_sets": True, "abil_announced": True, "abil_history": True,
    "item_unknown": True, "spe_conditions": True, "regenerator": True,
}
MAIN_LINE_DIM = 8567          # the first part's width (the layout without the blocks)

# The names the league's learner adds to "obs_settings" (obs_main.checkpoint_obs_settings).
LEAGUE_SETTINGS = ("main_line", "matchup", "movesets")


def checkpoint_league_blocks(ckpt: Any) -> Optional[Dict[str, bool]]:
    """
    {"matchup": True, "movesets"} of a league checkpoint: "obs_settings" names main_line and the matchup
    block, with exactly the main line's settings on. None for every other checkpoint. A checkpoint that
    names main_line with other settings raises: it is a network this code does not build.
    """
    saved = ckpt.get("obs_settings") if isinstance(ckpt, dict) else None
    if not isinstance(saved, dict) or not saved.get("main_line"):
        return None
    on = {k for k, v in saved.items() if v}
    want = {k for k, v in MAIN_LINE_SETTINGS.items() if v} | {"main_line", "matchup"}
    extra = on - want - {"movesets"}
    missing = want - on
    if extra or missing:
        raise RuntimeError(
            "a main-line checkpoint with other observation settings than the league run's: "
            f"missing {sorted(missing)}, not known {sorted(extra)}")
    run_cfg = ckpt.get("run_cfg")
    obs = run_cfg.get("obs") if isinstance(run_cfg, dict) else None
    if isinstance(obs, dict) and (not obs.get("main_line") or bool(obs.get("movesets")) != bool(saved.get("movesets"))):
        raise RuntimeError(f"the checkpoint's run_cfg obs {obs} does not agree with its obs_settings {saved}")
    return {"matchup": True, "movesets": bool(saved.get("movesets"))}


def describe(blocks: Dict[str, bool]) -> str:
    """One line for the log: what the observation holds."""
    parts = ["the main line's observation (M24381's: own Tera types, the six ability and item fixes)",
             "the matchup block"]
    if blocks.get("movesets"):
        parts.append("the moveset block")
    return ", ".join(parts)


class BlocksBase(ObservationAssembler):
    """
    This app's ObservationAssembler with the matchup block (and the moveset block) appended after the
    action mask, written at the end of assemble as the league's base writes them. Built by
    league_assembler() only, under the overlays.
    """

    _movesets: bool = True

    def __init__(self, vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True):
        import obs_matchup
        import obs_movesets

        super().__init__(vocab, fast)
        # As obs_v2.V2ObservationAssembler (train-v2's and the league's obs_assembler.py): the blocks
        # after the action mask, the matchup block first, and their meta for the network.
        self.matchup = True
        self.movesets = bool(self._movesets)
        self.layout.append(("matchup", (obs_matchup.BLOCK_DIM,), obs_matchup.BLOCK_DIM))
        self.meta["matchup"] = obs_matchup.block_meta()
        if self.movesets:
            self.layout.append(("movesets", (obs_movesets.BLOCK_DIM,), obs_movesets.BLOCK_DIM))
            self.meta["movesets"] = obs_movesets.block_meta()
        curr = 0
        for name, _, size in self.layout:
            if name in self.offsets and self.offsets[name] != (curr, curr + size):
                raise RuntimeError(f"the offset of {name} would move: {self.offsets[name]} -> {(curr, curr + size)}")
            self.offsets[name] = (curr, curr + size)
            curr += size
        if self.offsets["matchup"][0] != MAIN_LINE_DIM:
            raise RuntimeError(f"the first part is {self.offsets['matchup'][0]} numbers wide, the league's is {MAIN_LINE_DIM}")
        self.meta["offsets"] = self.offsets
        self.total_dim = curr
        self.calc_buf = np.zeros(self.total_dim, dtype=np.float16)
        self._matchup_encoder = obs_matchup.MatchupEncoder(self.vocab_map, full_types=self.movesets)
        if self.movesets:
            self._moveset_encoder = obs_movesets.MovesetEncoder(self.vocab_map, self._matchup_encoder)

    def assemble(self, battle: Any, events: List[List[str]] = None):
        obs = super().assemble(battle, events)
        off = self.offsets
        # 5. THE MATCHUP BLOCK, 6. THE MOVESET BLOCK (the league's obs_assembler.py, in its order)
        infos = self._matchup_encoder.encode(battle, obs, off["matchup"][0])
        if self.movesets:
            self._moveset_encoder.encode(battle, obs, off["movesets"][0], infos)
        return obs


_CLASSES: Dict[bool, type] = {}


def league_assembler_class(movesets: bool) -> type:
    """
    The league's observation with these blocks: obs_abil.AbilityObservationAssembler (and through it
    obs_tera.OwnTeraObservationAssembler) on BlocksBase, built with no arguments. The MRO is
    AbilityObservationAssembler, OwnTeraObservationAssembler, BlocksBase, ObservationAssembler:
    obs_tera's super().__init__(vocab, fast) reaches BlocksBase.
    """
    key = bool(movesets)
    cls = _CLASSES.get(key)
    if cls is not None:
        return cls
    from obs_abil import AbilityObservationAssembler, parse_settings

    abil = parse_settings({k: v for k, v in MAIN_LINE_SETTINGS.items() if k != "own_tera_type"})
    base = type(f"BlocksBase_{int(key)}", (BlocksBase,), {"_movesets": key})

    class LeagueObservationAssembler(AbilityObservationAssembler, base):   # type: ignore[valid-type, misc]
        """The main line's observation (obs_abil with every fix and the own Tera type), then the blocks."""

        main_line = True

        def __init__(self, vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True):
            AbilityObservationAssembler.__init__(self, vocab, fast, own_tera_type=True, settings=abil)

    LeagueObservationAssembler.__name__ = LeagueObservationAssembler.__qualname__ = (
        "LeagueObservationAssembler_matchup" + ("_movesets" if key else ""))
    _CLASSES[key] = LeagueObservationAssembler
    return LeagueObservationAssembler


def league_assembler(movesets: bool = True) -> ObservationAssembler:
    """An assembler of the league's observation with these blocks."""
    return league_assembler_class(movesets)()
