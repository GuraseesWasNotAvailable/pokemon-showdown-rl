"""
obs_v2: the networks of branch train-v2 that see its matchup block (and moveset block), played here
exactly as that branch trained them. The from-scratch run (~/runs/v2/scratch) is one of them.

train-v2 records what its observation holds in every checkpoint's run_cfg["obs"] (own_tera_type,
matchup, movesets; no "obs_settings" entry), builds the observation with its
ObservationAssembler(own_tera_type, matchup, movesets), and the network with extra input columns
for the blocks (PokeTransformer with meta["matchup"] / meta["movesets"]). hybrid.PolicyNet asks
checkpoint_v2_blocks() first, and for such a checkpoint builds

    the network     RunConfig.make_model(matchup=True, movesets=...): ppo_core.PokeTransformer
                    with train-v2's input columns and its _wide_subnet (ported line for line);
                    refused when the run's run_cfg["model"] built another one (architecture_mismatch)
    the weights     the checkpoint's, without the training-only heads (AUX_HEAD_PREFIXES: the
                    SetHead and the opponent-action head, which no player reads)
    V2ObservationAssembler, that branch's observation byte for byte:
      * before the action mask: this line's ObservationAssembler and, with own_tera_type, the
        search branch's own Tera type and STAB after a Tera (obs_search.py). train-v2's own-Tera
        switch is the search branch's, so the bytes are the same;
      * the action mask as train-v2 writes it (trained_action_mask), which offers a switch the
        server refuses (switch_names.py: a Pokemon whose name an active Transformed Pokemon
        carries). The network sees that mask; the player chooses from choice_mask(), this line's
        mask (create_action_mask), which leaves it out, and so does the search (foulplay_bridge);
      * the matchup and moveset blocks after the mask, laid out and written as train-v2's
        obs_assembler.py does: obs_matchup.py, obs_movesets.py and matchup.py are train-v2's files
        (e429f6f, the release ~/rel/v2-movesets), unchanged but for where they import
        own_tera_type_name from (obs_tera here, the same function), with randbats_sets_gen9.json.

Nothing else changes: obs_assembler.py is untouched, a train-v2 checkpoint without the matchup
block is a search-line checkpoint (obs_search.py), and this line's own checkpoints carry
"obs_settings" (obs_tera.py, obs_abil.py).

bench/v2_exactness.py holds the app's player against train-v2's own code on recorded battles:
the observation byte for byte, the logits and values bit for bit.

Importing this module pulls in numpy and obs_assembler, not torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from obs_assembler import ObservationAssembler
from obs_search import SearchOwnTeraObservationAssembler
from obs_tera import CHECKPOINT_KEY

# What hybrid.PolicyNet.obs_line says for such a checkpoint.
LINE = "v2"

# The state-dict prefixes of train-v2's training-only heads (its ppo_core.AUX_HEAD_PREFIXES).
AUX_HEAD_PREFIXES = ("set_head.", "opp_action_head.")


def checkpoint_v2_blocks(ckpt: Any) -> Optional[Dict[str, bool]]:
    """
    {"own_tera_type", "matchup", "movesets"} of a train-v2 checkpoint whose run_cfg["obs"] has the
    matchup block on; None for every other checkpoint (no run_cfg, the block off, or an
    "obs_settings" entry, which only this line's learner writes and which then decides).
    """
    if not isinstance(ckpt, dict) or isinstance(ckpt.get(CHECKPOINT_KEY), dict):
        return None
    run_cfg = ckpt.get("run_cfg")
    obs = run_cfg.get("obs") if isinstance(run_cfg, dict) else None
    if not isinstance(obs, dict) or not obs.get("matchup", False):
        return None
    return {
        "own_tera_type": bool(obs.get("own_tera_type", False)),
        "matchup": True,
        "movesets": bool(obs.get("movesets", False)),
    }


def architecture_mismatch(ckpt: Any, net: Any) -> List[str]:
    """
    What the checkpoint's run_cfg["model"] says that the network built here is not ([] when it
    agrees). This line builds the network from its own defaults, train-v2 from the run's config.
    A different size of a tensor already stops the load; these settings change no tensor's size
    and would play another network without a word: the attention heads, the layers, the history
    length (kv_cache_len) and the feed-forward width (ff_expansion, checked here for the message).
    The value support is a buffer of the checkpoint and is loaded with it.
    """
    run_cfg = ckpt.get("run_cfg") if isinstance(ckpt, dict) else None
    model = run_cfg.get("model") if isinstance(run_cfg, dict) else None
    if not isinstance(model, dict):
        return []
    layer = net.transformer[0] if len(getattr(net, "transformer", [])) else None
    built = {
        "n_heads": int(net.n_heads),
        "n_layers": len(net.transformer),
        "kv_cache_len": int(net.kv_cache_len),
        "ff_expansion": (float(layer.ff[0].out_features) / float(net.d_model)) if layer is not None else None,
    }
    out = []
    for key, have in built.items():
        want = model.get(key)
        if want is None or have is None:
            continue
        if float(want) != float(have):
            out.append(f"{key} {want} (this code builds {have})")
    return out


def describe(blocks: Dict[str, bool]) -> str:
    """One line for the log: what the observation holds."""
    parts = ["the matchup block"]
    if blocks.get("movesets"):
        parts.append("the moveset block")
    if blocks.get("own_tera_type"):
        parts.append("own Tera types (with STAB kept after a Tera)")
    return ", ".join(parts)


class V2ObservationAssembler(SearchOwnTeraObservationAssembler):
    """
    train-v2's ObservationAssembler(own_tera_type, matchup=True, movesets) on this line's code.

    own_tera_type  the search branch's own Tera type and STAB after a Tera (obs_search.py), as
                   train-v2's own-Tera switch writes them
    movesets       the moveset block after the matchup block
    """

    def __init__(self, own_tera_type: bool = True, movesets: bool = True,
                 vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True):
        import obs_matchup
        import obs_movesets

        super().__init__(vocab, fast)
        self.own_tera_type = bool(own_tera_type)
        # train-v2's obs_assembler.py: the blocks appended to the layout after the action mask, the
        # matchup block first, and their meta for the network. The offsets before them stay; the
        # encoders built above keep the same offsets dict, which only gains the new entries.
        self.matchup = True
        self.movesets = bool(movesets)
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
        self.meta["offsets"] = self.offsets
        self.total_dim = curr
        self.calc_buf = np.zeros(self.total_dim, dtype=np.float16)
        # With the moveset block, the move tokens get every move's full type (abilities, forms, weather);
        # without it they stay what every matchup-only checkpoint was trained on (train-v2's words).
        self._matchup_encoder = obs_matchup.MatchupEncoder(self.vocab_map, full_types=self.movesets)
        if self.movesets:
            self._moveset_encoder = obs_movesets.MovesetEncoder(self.vocab_map, self._matchup_encoder)

    def assemble(self, battle: Any, events: List[List[str]] = None):
        # Everything before the blocks, in the buffer zeroed from end to end (the blocks' encoders
        # need their part zeroed, as train-v2's does it).
        if self.own_tera_type:
            obs = super().assemble(battle, events)
        else:
            obs = ObservationAssembler.assemble(self, battle, events)
        off = self.offsets
        m_start, m_end = off["action_mask"]
        obs[m_start:m_end] = self.trained_action_mask(battle, list(battle.team.values()))
        # 5. THE MATCHUP BLOCK, 6. THE MOVESET BLOCK (train-v2's obs_assembler.py, in its order)
        infos = self._matchup_encoder.encode(battle, obs, off["matchup"][0])
        if self.movesets:
            self._moveset_encoder.encode(battle, obs, off["movesets"][0], infos)
        return obs

    def choice_mask(self, battle: Any, obs: Any) -> np.ndarray:
        """What the player may choose from: this line's mask, without a switch the server refuses."""
        return self.create_action_mask(battle, list(battle.team.values()))

    def trained_action_mask(self, battle: Any, self_team: List[Any]) -> np.ndarray:
        """
        train-v2's ObservationAssembler.create_action_mask, line for line: this line's mask but for
        the switch the server refuses, which it offers (only the observation has it).
        """
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
                    if idx < 6:
                        mask[4 + idx] = 1.0
                except ValueError:
                    pass

        # 3. SAFETY FALLBACK
        if not mask.any():
            mask[0] = 1.0

        return mask
