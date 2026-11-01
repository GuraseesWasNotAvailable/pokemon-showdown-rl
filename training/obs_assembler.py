"""
Observation Assembler for Pokémon Showdown Reinforcement Learning.

This module converts complex Battle objects into structured, flattened NumPy 
arrays suitable for neural network input. It manages the global schema, 
feature offsets, and action space mapping.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from obs_abilities import AbilityEncoder, encode_ability_inplace
from obs_global import encode_global_inplace
from obs_moves import MoveEncoder, encode_moves_inplace
from obs_pokemon import PokemonBodyEncoder, encode_pokemon_body_inplace
from obs_transitions import encode_transitions_inplace, transition_id_dim, transition_scalar_dim

# Configure logging
logger = logging.getLogger(__name__)

def own_tera_type_name(mon: Any) -> Optional[str]:
    """
    The Tera type of one of the player's own Pokemon, lower case, or None when it is not known.

    poke-env 0.15 keeps the Pokemon's entry of the last request (Pokemon._last_request) but never
    reads its "teraType", so Pokemon.tera_type stays None in a random battle until the Pokemon
    Terastallizes. This reads it: first what poke-env knows, then the request.
    """
    known = getattr(mon, "tera_type", None)
    if known is not None:
        return str(getattr(known, "name", known)).lower()
    request = getattr(mon, "_last_request", None)
    name = request.get("teraType") if isinstance(request, dict) else None
    return str(name).lower() if name else None


class ObservationAssembler:
    """
    Orchestrates the conversion of Battle states into numerical observations.
    
    This class handles the layout of the observation vector and provides 
    utilities for mapping model indices back to game-engine commands.
    """

    def __init__(self, vocab: Optional[Dict[str, List[str]]] = None, fast: bool = True,
                 own_tera_type: bool = False, matchup: bool = False, movesets: bool = False):
        """
        Initializes the assembler with vocabulary mappings and schema metadata.

        Args:
            vocab: A dictionary containing lists of species, items, moves, etc.
                  If None, attempts to load from 'vocab.json'.
            fast: Encode the Pokemon, their abilities and their moves with the encoder
                  classes (MoveEncoder, PokemonBodyEncoder, AbilityEncoder), which remember
                  what does not change between decisions. False uses the original
                  encode_*_inplace functions for every decision. Both write the same bytes
                  (tests/test_speed_obs.py); False is there for that comparison.
            own_tera_type: Write the Tera type of the player's own Pokemon into their Tera type
                  block before they Terastallize. poke-env 0.15 does not read "teraType" from the
                  request, so Pokemon.tera_type is None for the own team of a random battle until
                  the "-terastallize" line, and the block stays empty: the policy chooses whether
                  to Terastallize without seeing into which type (bench/notes/tera.md). False
                  keeps the observation every checkpoint so far was trained on. True is for
                  training a network that sees it; on a network trained without it, it is an
                  input that network has never seen.
            matchup: Append the matchup and set-candidate block (obs_matchup.py) after the action
                  mask: move effectiveness against the other side, ability and item candidates.
                  Nothing before it moves. False (the default) writes exactly the observation
                  every checkpoint so far was trained on. True is for a network built with the
                  block (meta["matchup"]; tools/widen_checkpoint.py makes one from a checkpoint).
            movesets: Append the moveset block (obs_movesets.py) after the matchup block: the moves,
                  Tera types and roles each Pokemon may have, and its threat to the other side.
                  Needs matchup=True. False (the default) writes exactly the observation without it.
        """
        if movesets and not matchup:
            raise ValueError("the moveset block (movesets=True) needs the matchup block (matchup=True)")
        if vocab is None:
            vocab = self._load_vocab("vocab.json")
        
        self.vocab_lists = vocab
        self.vocab_map = {
            category: {item.lower(): i for i, item in enumerate(items)}
            for category, items in vocab.items()
        }
        
        # Metadata defines the dynamic sizes of sub-vectors (e.g., number of types)
        self.meta = self.get_schema_metadata(vocab)
        
        self.pokemon_scalar_dim = self.meta["dim_pokemon_body"]
        self.move_scalar_dim = self.meta["dim_move_scalars"]
        self.global_dim = self.meta["dim_global_scalars"]
        self.transition_id_dim = self.meta["dim_transition_ids"]
        self.transition_scalar_dim = self.meta["dim_transition_scalars"]
        self.action_dim = self.meta["action_dim"]

        # Define the layout of the flattened observation vector
        self.layout = [
            ("pokemon_body", (12, self.pokemon_scalar_dim), 12 * self.pokemon_scalar_dim),
            ("pokemon_ids", (12, 2), 24),
            ("ability_ids", (12, 4), 48),
            ("move_ids", (12, 4), 48),
            ("move_scalars", (12, self.move_scalar_dim), 12 * self.move_scalar_dim),
            ("global_scalars", (self.global_dim,), self.global_dim),
            ("transition_ids", (self.transition_id_dim,), self.transition_id_dim),
            ("transition_scalars", (self.transition_scalar_dim,), self.transition_scalar_dim),
            ("action_mask", (self.action_dim,), self.action_dim)
        ]
        self.matchup = bool(matchup)
        if self.matchup:
            import obs_matchup
            self.layout.append(("matchup", (obs_matchup.BLOCK_DIM,), obs_matchup.BLOCK_DIM))
            self.meta["matchup"] = obs_matchup.block_meta()
        self.movesets = bool(movesets)
        if self.movesets:
            import obs_movesets
            self.layout.append(("movesets", (obs_movesets.BLOCK_DIM,), obs_movesets.BLOCK_DIM))
            self.meta["movesets"] = obs_movesets.block_meta()

        self.offsets = {}
        curr = 0
        for name, _, size in self.layout:
            self.offsets[name] = (curr, curr + size)
            curr += size
        self.meta["offsets"] = self.offsets
        self.total_dim = curr
        self.calc_buf = np.zeros(self.total_dim, dtype=np.float16)

        self.fast = bool(fast)
        self.own_tera_type = bool(own_tera_type)
        # Where a Pokemon's Tera type block starts inside its body: the second half of types_raw.
        types_start, types_end = self.meta["feature_map"]["body"]["types_raw"]
        self._tera_block = types_start + (types_end - types_start) // 2
        self._body_encoder = PokemonBodyEncoder(self.pokemon_scalar_dim, self.offsets, self.vocab_map, self.vocab_lists)
        self._ability_encoder = AbilityEncoder(self.offsets, self.vocab_map)
        self._move_encoder = MoveEncoder(self.move_scalar_dim, self.offsets, self.vocab_map, self.vocab_lists)
        if self.matchup:
            # With the moveset block, the move tokens get every move's full type (abilities, forms, weather);
            # without it they stay what every matchup-only checkpoint was trained on.
            self._matchup_encoder = obs_matchup.MatchupEncoder(self.vocab_map, full_types=self.movesets)
        if self.movesets:
            self._moveset_encoder = obs_movesets.MovesetEncoder(self.vocab_map, self._matchup_encoder)

    def _load_vocab(self, path: str) -> Dict[str, List[str]]:
        """Loads and validates the vocabulary file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        with open(path, "r") as f:
            return json.load(f)

    def assemble(self, battle: Any, events: List[List[str]] = None) -> np.ndarray:
        """
        Converts a Battle state into a flattened NumPy array.

        Args:
            battle: A poke_env Battle object.

        Returns:
            np.ndarray: A flattened vector of shape (total_dim,).
        """
        self.calc_buf.fill(0)
        off = self.offsets

        # Extract teams (ensuring internal server consistency)
        self_team = list(battle.team.values())
        opp_team = list(battle.opponent_team.values())

        # 1. ENCODE POKEMON (12 Slots: 6 Self, 6 Opponent)
        fast = self.fast
        for i in range(12):
            is_self = i < 6
            mon = self_team[i] if (is_self and i < len(self_team)) else \
                  (opp_team[i-6] if (not is_self and (i-6) < len(opp_team)) else None)

            # Encode physical properties, stats, and health
            if fast:
                self._body_encoder.encode(mon, self.calc_buf, i)
            else:
                encode_pokemon_body_inplace(
                    mon, self.calc_buf, i, self.pokemon_scalar_dim,
                    off, self.vocab_map, self.vocab_lists
                )

            if is_self and mon is not None and self.own_tera_type:
                self._write_own_tera_type(mon, i)

            # Encode categorical IDs (Species, Item, Ability)
            if fast:
                self._ability_encoder.encode(mon, self.calc_buf, i)
            else:
                encode_ability_inplace(mon, self.calc_buf, i, off, self.vocab_map)

            # Encode categorical IDs (Species, Item, Ability)
            available_move_ids = []
            if is_self and mon and mon.active and battle.available_moves:
                available_move_ids = [m.id for m in battle.available_moves]

            # Encode individual move properties
            if fast:
                self._move_encoder.encode(mon, self.calc_buf, i, is_self=is_self, available_move_ids=available_move_ids)
            else:
                encode_moves_inplace(
                    mon, self.calc_buf, i, self.move_scalar_dim,
                    off, self.vocab_map, self.vocab_lists,
                    is_self=is_self, # Pass the boolean directly
                    available_move_ids=available_move_ids
                )

            # After Terastallizing, poke-env's type_1 / type_2 report only the Tera type, so both
            # move encoders drop STAB on the original types. Gen 9 keeps it. Corrected only under
            # own_tera_type, because every checkpoint so far was trained on the old flag.
            if self.own_tera_type and mon is not None and getattr(mon, 'is_terastallized', False):
                self._restore_original_stab(mon, i)

        # 2. ENCODE GLOBAL STATE (Weather, Terrain, Hazards)
        encode_global_inplace(
            battle, self.calc_buf, off["global_scalars"], 
            self.vocab_map, self.vocab_lists
        )

        # 3. ENCODE TRANSITIONS (Move History & Type Effectiveness)
        
        player_role = getattr(battle, "player_role", None)
        if player_role not in ("p1", "p2"):
            player_role = "p1"  # safe fallback; you can also raise here if preferred
        
        encode_transitions_inplace(
            events,
            self.calc_buf,
            off["transition_ids"][0],
            off["transition_scalars"][0],
            self.vocab_map,
            self_is_p1=(player_role == "p1"),
        )

        # 4. ENCODE ACTION MASK
        m_start, m_end = off["action_mask"]
        self.calc_buf[m_start:m_end] = self.create_action_mask(battle, self_team)

        # 5. THE MATCHUP BLOCK (obs_matchup.py), only with matchup=True
        if self.matchup:
            infos = self._matchup_encoder.encode(battle, self.calc_buf, off["matchup"][0])
            # 6. THE MOVESET BLOCK (obs_movesets.py), only with movesets=True
            if self.movesets:
                self._moveset_encoder.encode(battle, self.calc_buf, off["movesets"][0], infos)

        return self.calc_buf.copy()

    def _restore_original_stab(self, mon: Any, slot: int) -> None:
        """Sets the STAB flag of moves that share one of the Pokemon's original types."""
        original = set(getattr(mon, 'base_types', None) or [])
        if not original:
            return
        stab = self.meta["feature_map"]["move"]["stab_flag"]
        single = self.move_scalar_dim // 4
        start = self.offsets["move_scalars"][0] + slot * self.move_scalar_dim
        for m_idx, move in enumerate(list(mon.moves.values())[:4]):
            if move is not None and move.type in original:
                self.calc_buf[start + m_idx * single + stab] = 1.0

    def _write_own_tera_type(self, mon: Any, slot: int) -> None:
        """
        The Tera type the server's request names for one of the player's own Pokemon, written
        where the encoders write Pokemon.tera_type. Only when poke-env does not know it already
        (it does after Terastallizing, and for a team that came from the teambuilder).
        """
        if mon.tera_type is not None:
            return
        name = own_tera_type_name(mon)
        if not name:
            return
        t_idx = self.vocab_map.get("pokemon.type", {}).get(name, -1) + 1
        if t_idx > 0:
            self.calc_buf[self.offsets["pokemon_body"][0] + slot * self.pokemon_scalar_dim + self._tera_block + t_idx] = 1.0

    def create_action_mask(self, battle: Any, self_team: List[Any]) -> np.ndarray:
        """
        Generates a valid action mask using poke-env 0.15's native engine.
        
        0-3: Moves, 4-9: Switches, 10-13: Terastallize + Moves.
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
        # In 0.15, available_switches natively handles trapped, reviving, and fainting states.
        if battle.available_switches:
            for available_switch in battle.available_switches:
                try:
                    # Map the available switch to the static 6-slot self_team array
                    idx = self_team.index(available_switch)
                    if idx < 6:
                        mask[4 + idx] = 1.0
                except ValueError:
                    pass

        # 3. SAFETY FALLBACK
        # Ensure at least one action is always valid so the NN never deadlocks
        if not mask.any():
            mask[0] = 1.0
            
        return mask

    def map_index_to_order(self, index: int, battle: Any) -> Tuple[Union[str, Any], Dict]:
        """Maps a predicted index to a valid game engine command."""
        active_mon = battle.active_pokemon
        if not active_mon:
            return "DEFAULT", {}

        # Moves and Terastallization
        if (0 <= index <= 3) or (10 <= index <= 13):
            mon_move_ids = [m.id for m in active_mon.moves.values()]
            move_idx = index if index <= 3 else index - 10
            
            if move_idx < len(mon_move_ids):
                target_id = mon_move_ids[move_idx]
                for am in battle.available_moves:
                    if am.id == target_id:
                        return am, {"terastallize": True} if index >= 10 else {}
                    elif am.id == "struggle" and move_idx == 0:
                        return am, {}
            
            # Fallbacks
            if battle.available_moves: return battle.available_moves[0], {}
            if battle.available_switches: return battle.available_switches[0], {}
            return "DEFAULT", {}

        # Switches
        elif 4 <= index <= 9:
            team_list = list(battle.team.values())
            slot_idx = index - 4
            if slot_idx < len(team_list) and team_list[slot_idx]:
                return team_list[slot_idx], {}

        return "DEFAULT", {}

    def map_order_to_index(self, order: Any, battle: Any) -> int:
        """Reverse-maps a 0.15 BattleOrder to a policy index (used for Imitation Learning)."""
        if not order:
            return 0 # Handle string/DEFAULT orders
        choice = getattr(order, "order", None)
        if not choice:
            return 0
        # Case A: Move/Tera (0.15 exposes Move objects)
        if hasattr(choice, "id"):  # Safer check than base_power
            active_mon = battle.active_pokemon
            if not active_mon: return 0
            
            mon_move_ids = [m.id for m in active_mon.moves.values()]
            if choice.id in mon_move_ids:
                move_slot = mon_move_ids.index(choice.id)
                if getattr(order, "terastallize", False): return move_slot + 10
                return move_slot
            elif choice.id == "struggle":
                return 0
            return 0
                
        # Case B: Switch (0.15 exposes Pokemon objects)
        elif hasattr(choice, "current_hp"):
            team_list = list(battle.team.values())
            try: 
                return 4 + team_list.index(choice)
            except ValueError: 
                return 4

        return 0

    @staticmethod
    def get_schema_metadata(vocab_lists: Dict[str, List[str]]=None) -> Dict[str, Any]:
        if vocab_lists is None:
            with open("vocab.json", "r") as f:
                vocab_lists = json.load(f)
        from obs_moves import get_move_metadata
        from obs_pokemon import get_pokemon_metadata
        move_meta = get_move_metadata(vocab_lists)
        pok_meta = get_pokemon_metadata(vocab_lists)
        
        v_type = len(vocab_lists.get("pokemon.type", [])) + 1
        
        return {
            "dim_pokemon_body": pok_meta["scalar_dim"],
            "dim_move_scalars": move_meta["single_move_dim"] * 4,
            "dim_transition_ids": transition_id_dim(),
            "dim_transition_scalars": transition_scalar_dim(),
            "dim_global_scalars": 3 + (len(vocab_lists.get("global.weather", [])) + 1 + 10) + 
                                  (len(vocab_lists.get("global.field", [])) + 1) + 
                                  ((len(vocab_lists.get("global.side_condition", [])) + 1) * 2),
            "feature_map": {
                "body": pok_meta["feature_map"],
                "move": move_meta["feature_map"],
                "global": {"turn_int": 0, "remainder_raw": (1, None)},
                "transition": {
                    "move_ids": (0, 2), "pokemon_ids": (2, 6), "ability_ids": (6, 8), "item_ids": (8, 10),
                },
            },
            "faint_internal_idx": pok_meta["faint_internal_idx"], # Index 113
            "vocab_pokemon": len(vocab_lists.get("pokemon.species", [])) + 1,
            "vocab_item": len(vocab_lists.get("pokemon.item", [])) + 1,
            "vocab_ability": len(vocab_lists.get("pokemon.ability", [])) + 1,
            "vocab_move": len(vocab_lists.get("move.id", [])) + 1,
            "vocab_type": v_type,
            "action_dim": 14,
            "n_pokemon_slots": 12, "n_move_slots": 4, "n_ability_slots": 4,
            "n_transition_ids": transition_id_dim(),
        }
    
    def translate_relative_index(self, block_name: str, relative_idx: int) -> str:
        """Translates a raw block index into a human-readable feature name."""
        meta = self.meta
        
        try:
            if block_name == "pokemon_body":
                single_dim = self.pokemon_scalar_dim
                slot = relative_idx // single_dim
                idx_in_mon = relative_idx % single_dim
                
                for k, v in meta["feature_map"]["body"].items():
                    if isinstance(v, tuple) and v[0] <= idx_in_mon < v[1]:
                        return f"P{slot//6 + 1}_Mon{slot%6}.{k}[{idx_in_mon - v[0]}]"
                    elif v == idx_in_mon:
                        return f"P{slot//6 + 1}_Mon{slot%6}.{k}"
                        
            elif block_name == "move_scalars":
                # dim_move_scalars is the total size for all 4 moves of ONE pokemon
                single_move_dim = meta["dim_move_scalars"] // 4
                slot = relative_idx // meta["dim_move_scalars"]
                idx_in_mon = relative_idx % meta["dim_move_scalars"]
                move_num = idx_in_mon // single_move_dim
                idx_in_move = idx_in_mon % single_move_dim
                
                for k, v in meta["feature_map"]["move"].items():
                    if isinstance(v, tuple) and v[0] <= idx_in_move < v[1]:
                        return f"P{slot//6 + 1}_Mon{slot%6}.Move{move_num}.{k}[{idx_in_move - v[0]}]"
                    elif v == idx_in_move:
                        return f"P{slot//6 + 1}_Mon{slot%6}.Move{move_num}.{k}"
                        
            elif block_name == "global_scalars":
                if relative_idx == 0: return "turn_int"
                if relative_idx == 1: return "p1_tera_flag"
                if relative_idx == 2: return "p2_tera_flag"
                return f"weather_or_side_condition[{relative_idx}]"
            
            elif block_name == "transition_ids":
                names = ["P1_Move", "P2_Move", "P1_Actor", "P1_Target", "P2_Actor", "P2_Target", "P1_Abil", "P2_Abil", "P1_Item", "P2_Item"]
                return names[relative_idx] if relative_idx < len(names) else f"TransID[{relative_idx}]"
            
            elif block_name == "transition_scalars":
                side = "P1" if relative_idx < 45 else "P2"
                idx = relative_idx % 45
                return f"{side}_Transition_Scalar[{idx}]"
                
        except Exception:
            pass
            
        return f"Unknown[{relative_idx}]"
