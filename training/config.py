"""
Configuration module for PokeTransformer Reinforcement Learning.

This module defines the schema and default parameters for observation processing,
neural network architecture, environment interaction, and the PPO learner.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple, Literal, Final
from pathlib import Path

# Setup logging for configuration errors
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ObsConfig:
    """Configuration for observation parsing and tokenization schema."""
    n_slots: int = 6
    n_move_slots: int = 4
    hp_bins: int = 11  # Categorical bins for HP percentage
    vocab_path: str = "vocab.json"
    # Show each own Pokemon its Tera type before it Terastallizes (ObservationAssembler(own_tera_type=True)).
    # poke-env 0.15 and 0.16.1 never read the request's "teraType", so without this the policy decides
    # whether to Terastallize without knowing into what (bench/notes/tera.md).
    # Default ON: the network should see everything a human player sees. The observation SIZE does not
    # change (the Tera type block of every Pokemon already exists; it was only empty for the own team
    # until the Tera), so any checkpoint loads and runs with it on. Checkpoints from before the switch
    # (no key in their run_cfg) keep the observation they were trained on in the players
    # (checkpoint_own_tera_type), and continuing PPO/imitation from one turns it on: the learner prints
    # a notice and the network learns the input from there. No weight surgery is needed (zero-initialising
    # the Tera block's input weights would destroy features they learned from opposing and post-Tera
    # Pokemon). --no-own-tera-type reproduces the old observation.
    # The same switch also restores STAB on a Terastallized Pokemon's original types: poke-env reports
    # only the Tera type after the Tera, so the move encoders dropped it (gen 9 keeps it).
    # The value is saved in every checkpoint's run_cfg; evaluation, ladder and search players
    # (hybrid.PolicyNet) read it from there, so the training switch is the only switch.
    own_tera_type: bool = True
    # The matchup and set-candidate block (obs_matchup.py, bench/notes/OBS_MATCHUP.md): 1,068 numbers
    # appended after the action mask, and the network's extra input columns for them. Off by default:
    # the observation and the network are then byte for byte what they were. A checkpoint trained
    # without it gets the block by widening (tools/widen_checkpoint.py), never by switching it on in
    # place. Saved in every checkpoint's run_cfg (obs_matchup.checkpoint_obs_matchup).
    matchup: bool = False
    # The moveset block (obs_movesets.py, bench/notes/OBS_MOVESETS.md): 660 numbers after the matchup block
    # (the moves, Tera types and roles each Pokemon may have, its threat to the other side), and 103 more
    # input columns of pokemon_net[0]. Needs `matchup`. Off by default: byte for byte what it was without it.
    # A matchup checkpoint gets it by widening (tools/widen_checkpoint.py --movesets). Saved in run_cfg.
    movesets: bool = False
    # The main line's observation (obs_main.py; release r5/r6, the network M24381) in the place of this
    # line's own first part: obs_tera's own Tera type (no STAB restored after a Tera), obs_abil's six fixes
    # and switch_names in the action mask, byte for byte the window app's AbilityObservationAssembler;
    # the matchup and moveset blocks follow as without it. The size does not change. It carries its own
    # own Tera type, so it needs own_tera_type False (train.py --obs-main-line sets both). Off by default:
    # the observation is then byte for byte what it was, and the key is not in run_cfg (RunConfig.as_dict).
    main_line: bool = False

    def __post_init__(self):
        if self.movesets and not self.matchup:
            raise ValueError("ObsConfig.movesets needs ObsConfig.matchup (the moveset block reads the matchup block's sets)")
        if self.main_line and self.own_tera_type:
            raise ValueError("ObsConfig.main_line carries the main line's own Tera type (obs_tera.py): own_tera_type must be "
                             "False with it (this line's own_tera_type writes another Tera type and restores STAB)")


@dataclass(frozen=True)
class ModelConfig:
    """Parameters for the PokeTransformer/PokeNet architecture."""
    
    # Dedicated Embedding Sizes
    emb_dims: Dict[str, int] = field(default_factory=lambda: {
        "pokemon": 64,
        "item": 32,
        "ability": 64,
        "move": 64,
        "action": 14,
    })
    
    # Dedicated Subnet Output Sizes
    out_dims: Dict[str, int] = field(default_factory=lambda: {
        "ability_vec": 64,
        "pokemon_vec": 512,
        "global_vec": 96,
        "transition_vec": 416,
    })
    
    # Universal Embedding Bank Sizes
    bank_dims: Dict[str, int] = field(default_factory=lambda: {
        "val_100": 32,  # HP, Level, Acc, PP
        "stat": 64,     # Base Stats, Weight, Height
        "power": 32,    # Move Power
    })
    
    # Vocabulary Safety Caps
    bank_ranges: Dict[str, int] = field(default_factory=lambda: {
        "val_100": 101,
        "stat": 800,
        "power": 251,
    })
    
    dropout: float = 0.0
    n_layers: int = 3
    n_heads: int = 8
    # Width of each feed-forward block, as a multiple of d_model. This read 4.0 but was never
    # passed to the layers, which were all built at 2.0, so every checkpoint so far has 2.0
    # sized blocks. A larger value does not load into them and needs a fresh run.
    ff_expansion: float = 2.0
    kv_cache_len: int = 64


@dataclass(frozen=True)
class EnvConfig:
    """Pokemon Showdown environment settings."""
    battle_format: str = "gen9randombattle"
    act_dim: int = 14  # 4 moves + 4 tera moves + 6 switches


@dataclass(frozen=True)
class RolloutConfig:
    """Settings for distributed rollout workers and batching logic."""
    target_concurrent_battles: int = 800
    rooms_per_pair: int = 32

    # Timeouts
    infer_timeout_s: float = 15.0
    open_timeout: float = 30.0

    # Batching constraints
    infer_min_batch: int = 32
    infer_max_batch: int = 2048
    infer_wait_ms: float = 3.0
    infer_max_pending: int = 20000

    learn_min_episodes: int = 1
    learn_max_episodes: int = 32
    learn_wait_ms: float = 1.0
    learn_max_pending_episodes: int = 220
    learn_max_pending_batches: int = 5 

    # Who the imitation and JEPA modes learn from:
    #   "psbot"   the PSBot heuristic (psbot/psbot.js, run headless by heuristic/engine.js)
    #   "simple"  poke-env's SimpleHeuristicsPlayer, the original teacher
    imitation_teacher: str = "psbot"

    def worker_kwargs(self) -> Dict[str, Any]:
        """Returns a dictionary suitable for RolloutWorker initialization."""
        return asdict(self)


@dataclass(frozen=True)
class InferenceConfig:
    """Config for the centralized inference server/actor."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    server_min_batch: int = 0
    server_max_wait: float = 0

    def kwargs(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "server_min_batch": self.server_min_batch,
            "server_max_wait": self.server_max_wait,
        }

@dataclass(frozen=True)
class LearnerConfig:
    """Core PPO and Hyperparameter settings."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Supported Modes:
    #   imitation
    #   jepa_pretraining
    #   warmup
    #   warmup_with_actor_reset
    #   ppo
    #   ppo_frozen_backbone
    mode: str = "ppo_with_jepa"
    
    # Reinforcement Learning Math
    gamma: float = 0.999
    gae_lambda: float = 0.75
    use_dynamic_lambda: bool = True
    dynamic_lambda_min: float = 0.55
    dynamic_lambda_max: float = 0.95
    dynamic_lambda_eps: float = 1e-8
    
    jepa_ema_tau: float = 0.99
    
    # Distributional Value (Two-Hot Encoding)
    use_twohot_value: bool = True
    v_min: float = -1.5
    v_max: float = 1.5
    v_bins: int = 51
    
    # Schedules
    temp_start: float = 1.0
    temp_end: float = 1.0
    temp_total_steps: int = 500_000
    
    # Optimizer settings
    lr: float = 3e-4
    lr_warmup_steps: int = 1_000
    lr_hold_steps: int = 500_000
    # Optimizer steps at which the anneal ends at 1/27 of the rate. It used to read 1_00_000,
    # smaller than the hold, so the whole anneal happened in one step: at optimizer step
    # 501,001 of a phase (about 192 million decisions into PPO at 96 steps per update) the
    # rate fell 27 times at once. 1_000_000 is what bench/SPEED_AUDIT.md section 5 works out:
    # after the hold the rate glides down to 1/27 by step 1,000,000 and stays there. Nothing
    # differs before step 501,001, the schedule is rebuilt from this file at every trainer
    # start, and only the step count is restored from a checkpoint, so a run started with the
    # old value picks this one up at its next restart.
    lr_total_steps: int = 1_000_000
    # The same schedule counted in this phase's decisions (the learner's total_steps, the counter
    # the supervisor's target_decisions reads) and not in optimizer steps: the full rate until
    # lr_hold_decisions, then 1/(8p+1)^1.5 with p = (D - hold) / (total - hold), 1/27 from
    # lr_total_decisions on. The warmup ramp stays lr_warmup_steps optimizer steps. An update
    # takes about 49.6 optimizer steps while it stops early on KL and 96 once it does not, so a
    # schedule in steps lands far from where it was planned in decisions; this one cannot.
    # Both 0 (the default): off, and lr_hold_steps / lr_total_steps rule as before.
    lr_hold_decisions: int = 0
    lr_total_decisions: int = 0
    weight_decay: float = 1e-2
    
    # Layer-specific LR multipliers (initialized in __post_init__)
    lr_backbone_mult: float = field(init=False)
    lr_pi_mult: float = field(init=False)
    lr_v_mult: float = field(init=False)
    
    # PPO Specifics
    update_epochs: int = 2
    minibatch_size: int = 768
    # Minibatches per optimizer step. This used to read 4 but was never passed to the update
    # (it was missing from ppo_kwargs), so every run so far has effectively used 1. It is now
    # passed through, and 1 keeps the default behaviour unchanged. On a small GPU, a smaller
    # minibatch with a matching value here gives the same effective batch.
    grad_accum_steps: int = 1
    batch_seq_len: int = 256
    clip_coef: float = 0.1
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    jepa_coef: float = 1.0
    clip_vloss: bool = False
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.02
    steps_per_update: int = 36864

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    save_every_updates: int = 25
    keep_last: int = 500
    resume: bool = True
    
    # Auxiliary Tasks
    use_hp_aux: bool = False
    hp_aux_coef: float = 0.4

    # Stage 2-4 training features (the plan "teach the bot reads, set-up play, abilities and items").
    # Every one is off at its default, and off the code paths are the ones before them.
    # The read bonus (read_reward.py): adv += b * read for the decision only, before the advantages are
    # normalised, never into the value targets; b = read_bonus until read_anneal_start decisions of this
    # arm, then linear to 0 at read_anneal_end; the sum of one game's bonuses kept within +-read_cap.
    read_bonus: float = 0.0
    read_anneal_start: int = 20_000_000
    read_anneal_end: int = 45_000_000
    read_cap: float = 0.5
    # A head on the opponent tokens (Z_t 7-12) and the field token that predicts what the opponent did
    # next (read_reward's opp_action: moved, switched to slot k, switched to an unseen Pokemon).
    opp_action_aux_coef: float = 0.0
    # The policy loss of rows where the opponent's active Pokemon has a stat at +1 or more weighs this
    # much against 1 for the others, normalised to a mean of 1 per minibatch.
    boosted_row_weight: float = 1.0
    # The SetHead (ppo_core.SetHead): the opposing abilities and items, from tokens 7-12. Needs the
    # matchup block (ObsConfig.matchup).
    set_head_coef: float = 0.0

    def __post_init__(self):
        """Sets gradient multipliers based on the current TrainingMode."""
        # Use object.__setattr__ because the dataclass is frozen
        multipliers = {
            "imitation": (1.0, 1.0, 0.6), # backbone, actor, critic
            "warmup": (0.0, 0.0, 1.0), # backbone, actor, critic
            "ppo": (0.5, 1.0, 2.0), # backbone, actor, critic
            "warmup_with_actor_reset": (0.0, 1.0, 0.001)
        }
        backbone, pi, v = multipliers.get(self.mode, (1.0, 1.0, 1.0))
        object.__setattr__(self, "lr_backbone_mult", backbone)
        object.__setattr__(self, "lr_pi_mult", pi)
        object.__setattr__(self, "lr_v_mult", v)
        
        if self.lr_hold_decisions or self.lr_total_decisions:
            # The decisions schedule (off at 0 and 0): the anneal must end after the hold.
            if not 0 <= int(self.lr_hold_decisions) < int(self.lr_total_decisions):
                raise ValueError(f"lr_hold_decisions {self.lr_hold_decisions} and lr_total_decisions {self.lr_total_decisions}: "
                                 f"need 0 <= hold < total (or both 0, which is off)")
        if self.batch_seq_len <= 0:
            raise ValueError("batch_seq_len must be > 0")
        if self.minibatch_size % self.batch_seq_len != 0:
            raise ValueError(
                f"minibatch_size ({self.minibatch_size}) must be divisible by "
                f"batch_seq_len ({self.batch_seq_len})"
            )

    def get_temp(self, global_step: int) -> float:
        """Calculates linear annealing of the action temperature."""
        if global_step >= self.temp_total_steps:
            return self.temp_end
        frac = global_step / self.temp_total_steps
        return self.temp_start + frac * (self.temp_end - self.temp_start)

    def ppo_kwargs(self) -> Dict[str, Any]:
        """Extracts PPO-specific hyperparameters for the optimizer."""
        keys = [
            "update_epochs", "minibatch_size", "batch_seq_len", "grad_accum_steps",
            "clip_coef", "ent_coef", "hp_aux_coef",
            "vf_coef", "clip_vloss", "max_grad_norm", "target_kl"
        ]
        return {k: getattr(self, k) for k in keys}


@dataclass(frozen=True)
class RewardConfig:
    """
    Defines the reward signal for the agent. RL_REWARD.md has the reasoning and the evidence.

    shaping:
      "potential"  win/loss at the end, plus scale * (gamma * Phi(next) - Phi(now)) on every step,
                   where Phi is the fitted position value in rl_reward/. The shaping terms cancel
                   over a game, so the best policy is the best policy for win/loss alone.
      "faint"      the upstream reward: win/loss plus faint_self / faint_opp per fainted Pokemon.
                   Pays more for wide wins than for narrow ones.
      "none"       win/loss only.
    """
    terminal_win: float = 1.0
    terminal_loss: float = -1.0
    terminal_tie: float = 0.0
    shaping: str = "potential"
    # Largest |Phi| is the cap in rl_reward/weights.json (about 0.85), so with 0.5 every return
    # lies inside the value head's [-1.5, 1.5] support (LearnerConfig.v_min / v_max).
    shaping_scale: float = 0.5
    potential_file: str = ""  # empty: rl_reward/weights.json
    use_faint_reward: bool = False  # kept for old configs; "faint" above is what selects it
    faint_self: float = -0.1
    faint_opp: float = +0.1

    def __post_init__(self):
        if self.shaping not in ("potential", "faint", "none"):
            raise ValueError("RewardConfig.shaping must be 'potential', 'faint' or 'none'")
        if not 0.0 <= self.shaping_scale <= 1.0:
            raise ValueError("RewardConfig.shaping_scale must be between 0 and 1")


@dataclass(frozen=True)
class RunConfig:
    """The master configuration object coordinating all sub-configs."""
    obs: ObsConfig
    model: ModelConfig
    env: EnvConfig
    rollout: RolloutConfig
    infer: InferenceConfig
    learner: LearnerConfig
    reward: RewardConfig

    @classmethod
    def default(cls) -> RunConfig:
        """Factory method for default settings."""
        return cls(
            obs=ObsConfig(),
            model=ModelConfig(),
            env=EnvConfig(),
            rollout=RolloutConfig(),
            infer=InferenceConfig(),
            learner=LearnerConfig(),
            reward=RewardConfig(),
        )

    def make_model(self) -> nn.Module:
        """
        Instantiates the PokeTransformer model based on current configuration.
        
        Returns:
            nn.Module: An initialized model ready for training or inference.
        
        Raises:
            ImportError: If required core modules are missing.
        """
        try:
            from obs_assembler import ObservationAssembler
            from ppo_core import PokeTransformer
        except ImportError as e:
            logger.error(f"Failed to import model components: {e}")
            raise

        # Derive schema metadata (Source of Truth for input sizes). The matchup block's meta is there
        # only with ObsConfig.matchup (and then the network has the block's input columns).
        if getattr(self.obs, "movesets", False):
            assembler = ObservationAssembler(matchup=True, movesets=True)
        elif getattr(self.obs, "matchup", False):
            assembler = ObservationAssembler(matchup=True)
        else:
            assembler = ObservationAssembler()

        return PokeTransformer(
            act_dim=int(self.env.act_dim),
            meta=assembler.meta,
            emb_dims=self.model.emb_dims,
            out_dims=self.model.out_dims,
            bank_dims=self.model.bank_dims,
            bank_ranges=self.model.bank_ranges,
            n_heads=self.model.n_heads,
            n_layers=self.model.n_layers,
            v_bins=int(self.learner.v_bins),
            v_min=float(self.learner.v_min),
            v_max=float(self.learner.v_max),
            ff_expansion=self.model.ff_expansion,
            dropout=self.model.dropout,
            kv_cache_len=self.model.kv_cache_len,
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Converts the entire configuration tree to a dictionary. ObsConfig.main_line is in it only when it
        is on, so every run without it saves the run_cfg it always did, key for key.
        """
        out = asdict(self)
        if not out.get("obs", {}).get("main_line", False):
            out.get("obs", {}).pop("main_line", None)
        return out

    def make_assembler(self) -> Any:
        """The ObservationAssembler this run's networks see (ObsConfig.own_tera_type, ObsConfig.matchup, ObsConfig.movesets,
        ObsConfig.main_line)."""
        if getattr(self.obs, "main_line", False):
            # obs_main.py: the main line's observation, then the blocks. Only with main_line.
            import obs_main
            return obs_main.make_assembler(bool(getattr(self.obs, "matchup", False)), bool(getattr(self.obs, "movesets", False)))
        from obs_assembler import ObservationAssembler
        if getattr(self.obs, "movesets", False):
            return ObservationAssembler(own_tera_type=self.obs.own_tera_type, matchup=True, movesets=True)
        if getattr(self.obs, "matchup", False):
            return ObservationAssembler(own_tera_type=self.obs.own_tera_type, matchup=True)
        return ObservationAssembler(own_tera_type=self.obs.own_tera_type)


def checkpoint_own_tera_type(ckpt: Any) -> bool:
    """
    Whether the network in a learner checkpoint was trained with ObsConfig.own_tera_type on.
    Checkpoints from before the switch existed have no such key: they were trained without it.
    """
    run_cfg = ckpt.get("run_cfg") if isinstance(ckpt, dict) else None
    obs = run_cfg.get("obs") if isinstance(run_cfg, dict) else None
    return bool(obs.get("own_tera_type", False)) if isinstance(obs, dict) else False