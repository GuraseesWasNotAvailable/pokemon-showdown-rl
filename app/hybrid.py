"""
The merged bot: the PokeTransformer policy and the PSBot heuristic deciding together.

Each turn both give an opinion on the same 14 actions:

    policy     log-probabilities from the transformer (ppo_core.PokeTransformer)
    heuristic  PSBot's expected material swing for every legal option (psbot/psbot.js)

and the action played is the one with the best combined score

    policy_weight * log p(action)  +  beta * heuristic(action)

after dropping any action PSBot rates more than `veto_gap` below its own best. So the policy
chooses among the options that are tactically sound, and PSBot's damage calculation stops it
from walking into an immunity or a KO.

Without a checkpoint there is no policy, and the player is PSBot on its own. The repository
this was forked from does not ship trained weights, so that is the state until a model has
been trained (see HYBRID.md).
"""

from __future__ import annotations

import contextlib
import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from poke_env.player import Player

from psbot_bridge import Advice, PSBotMixin, choice_to_order

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
ACTIONS = 14

# PSBot does not switch or Terastallize just because the raw score is a little higher: it asks
# for these margins first. The prior mirrors them so that its best action is PSBot's own pick.
SWITCH_MARGIN = 0.3
TERA_MARGIN = 0.3


@dataclass
class HybridConfig:
    checkpoint: Optional[str] = None   # path to a learner checkpoint; None = newest in checkpoints/
    beta: float = 4.0                  # nats of preference per Pokemon of material PSBot expects
    veto_gap: Optional[float] = 0.75   # drop actions this far below PSBot's best; None = no veto
    policy_weight: float = 1.0         # weight of the policy's log-probabilities
    greedy: bool = True                # take the best action instead of sampling
    temperature: float = 1.0           # policy temperature before mixing
    device: Optional[str] = None       # "cpu" or "cuda"; None = cuda when available
    random_init: bool = False          # build an untrained network (plumbing tests only)
    # The own Pokemon's Tera types in the observation (obs_tera.py). None, which is what every
    # player uses: as the checkpoint was trained, read from its "obs_settings" entry (a
    # checkpoint without one was trained without). True or False overrides the checkpoint and is
    # for measurements only (bench/tera_input_check.py).
    own_tera_type: Optional[bool] = None
    # The ability / item / switching fixes of the observation (obs_abil.py). None, which is what
    # every player uses: as the checkpoint was trained (its "obs_settings"; a checkpoint without
    # them was trained without). "all", "abil_sets,item_unknown" or {} overrides the checkpoint,
    # for measurements only: a network reads a fix right only after training with it.
    obs_abilities: Optional[Any] = None


@contextlib.contextmanager
def _in_repo_root():
    """The upstream modules open vocab.json relative to the working directory."""
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Finds the model weights inside a learner checkpoint (same rules as inference.py)."""
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Expected dict from checkpoint, got {type(ckpt)}")
    for k in ("model", "state_dict", "net", "policy", "actor_critic"):
        v = ckpt.get(k)
        if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
            return v
    if any(isinstance(x, torch.Tensor) for x in ckpt.values()):
        return ckpt
    raise RuntimeError("Could not find a model state_dict inside the checkpoint file.")


def find_checkpoint(path: Optional[str] = None) -> Optional[str]:
    """The given checkpoint, or the newest learner_update_*.pt under checkpoints/."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    found = glob.glob(str(REPO_ROOT / "checkpoints" / "learner_update_*.pt"))
    if not found:
        return None
    return max(found, key=os.path.getmtime)


def unknown_obs_settings(ckpt: Any) -> list:
    """The settings switched on in a checkpoint's "obs_settings" that this code cannot build ([] for none)."""
    from obs_abil import SETTINGS
    saved = ckpt.get("obs_settings") if isinstance(ckpt, dict) else None
    if not isinstance(saved, dict):
        return []
    known = {"own_tera_type", *SETTINGS}
    return sorted(k for k, v in saved.items() if v and k not in known)


def value_settings_from_checkpoint(ckpt: Any) -> Dict[str, Any]:
    """
    The reward a checkpoint's value head was trained on, from its "run_cfg" (learner._save_checkpoint
    stores RunConfig.as_dict()): {"shaping": "potential" | "faint" | "none", "scale": the shaping
    scale (Phi's weight in the return), "gamma", "potential_file", "from": "checkpoint" | "default"}.
    Without a run_cfg (or for None) the defaults of config.RewardConfig and LearnerConfig.
    """
    out: Dict[str, Any] = {"shaping": "potential", "scale": 0.5, "gamma": 0.999, "potential_file": "", "from": "default"}
    try:
        from config import LearnerConfig, RewardConfig

        out.update(shaping=RewardConfig.shaping, scale=float(RewardConfig.shaping_scale), gamma=float(LearnerConfig.gamma))
    except Exception:
        pass
    run = ckpt.get("run_cfg") if isinstance(ckpt, dict) else None
    if isinstance(run, dict):
        reward = run.get("reward") if isinstance(run.get("reward"), dict) else {}
        learner = run.get("learner") if isinstance(run.get("learner"), dict) else {}
        if reward:
            shaping = reward.get("shaping")
            if shaping is None:
                shaping = "faint" if reward.get("use_faint_reward") else out["shaping"]
            out.update(shaping=str(shaping), scale=float(reward.get("shaping_scale", out["scale"]) or 0.0),
                       potential_file=str(reward.get("potential_file") or ""), **{"from": "checkpoint"})
        if learner.get("gamma") is not None:
            out["gamma"] = float(learner["gamma"])
    return out


class PolicyNet:
    """The transformer, loaded for one-battle-at-a-time inference."""

    def __init__(self, cfg: HybridConfig):
        from config import RunConfig
        from obs_abil import make_observation_assembler, parse_settings, settings_from_checkpoint
        from obs_search import LINE as SEARCH_LINE, SearchOwnTeraObservationAssembler, checkpoint_search_own_tera
        from obs_tera import checkpoint_own_tera_type
        from obs_league import LEAGUE_SETTINGS, LINE as LEAGUE_LINE, MAIN_LINE_SETTINGS, checkpoint_league_blocks, league_assembler
        from obs_v2 import AUX_HEAD_PREFIXES, LINE as V2_LINE, V2ObservationAssembler, architecture_mismatch, checkpoint_v2_blocks

        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.path = find_checkpoint(cfg.checkpoint)
        ckpt = torch.load(self.path, map_location="cpu", weights_only=False) if self.path else None
        # A network of branch train-v2 with its matchup block (and moveset block) in the observation
        # (obs_v2.py): its run_cfg["obs"] says so. None for every other checkpoint.
        self.v2_blocks: Optional[Dict[str, bool]] = checkpoint_v2_blocks(ckpt) if ckpt is not None else None
        # A network of the league run (branch league): this line's own observation, the main line's,
        # with train-v2's blocks after it (obs_league.py): its "obs_settings" names main_line. None
        # for every other checkpoint.
        try:
            self.league_blocks: Optional[Dict[str, bool]] = checkpoint_league_blocks(ckpt) if ckpt is not None else None
        except RuntimeError as e:
            raise RuntimeError(f"Checkpoint {self.path}: {e}") from None
        blocks = self.v2_blocks if self.v2_blocks is not None else self.league_blocks
        with _in_repo_root():
            if blocks is not None:
                self.net = RunConfig.default().make_model(matchup=True, movesets=blocks["movesets"])
            else:
                self.net = RunConfig.default().make_model()
        if blocks is not None:
            # train-v2 and the league build their network from the run's config, this line from its
            # defaults: a setting that changes no tensor's size (attention heads, history length) must agree.
            differs = architecture_mismatch(ckpt, self.net)
            if differs:
                raise RuntimeError(f"Checkpoint {self.path} is a {'train-v2' if self.v2_blocks is not None else 'league'} "
                                   f"network this code does not build: {'; '.join(differs)}")
        # How the value head's numbers turn into a win estimate (search_player.value_fields): the
        # reward the checkpoint was trained on, from its "run_cfg"; the defaults without one.
        self.value_settings: Dict[str, Any] = value_settings_from_checkpoint(None)
        asked = getattr(cfg, "own_tera_type", None)
        self.own_tera_type = bool(asked) if asked is not None else False
        asked_abil = getattr(cfg, "obs_abilities", None)
        self.obs_abilities = parse_settings(asked_abil) if asked_abil is not None else {}
        # Whose observation: "own" (this line's settings, above), "search" (a checkpoint of the
        # search branch trained with its own-Tera switch, which writes other bytes: obs_search.py) or
        # "v2" (a train-v2 checkpoint with the matchup block: obs_v2.py) or "league" (a league run
        # checkpoint: this line's observation with the blocks, obs_league.py).
        self.obs_line = "own"
        # Tensors of training-only heads left out at load (a train-v2 checkpoint's SetHead and
        # opponent-action head): no player reads them.
        self.dropped_heads = 0
        if ckpt is not None:
            # An input this code does not know would be left out of the observation with no error (the
            # tensors have the same shapes): the network would play on inputs it was not trained with.
            # The stage2 review measured that kind of mismatch at 37-63 against the same network.
            unknown = unknown_obs_settings(ckpt)
            if self.league_blocks is not None:
                # The names the league's learner adds (checkpoint_league_blocks checked the rest).
                unknown = [k for k in unknown if k not in LEAGUE_SETTINGS]
            if unknown:
                raise RuntimeError(
                    f"Checkpoint {self.path} was trained with observation settings this code does not build: {unknown}. "
                    f"Play it with code that has them (obs_tera.py, obs_abil.py of the release that trained it).")
            if self.v2_blocks is not None:
                # A train-v2 network with the matchup block: its observation is the one its run_cfg
                # names, nothing else (the overrides below are for this line's own checkpoints).
                if asked is not None or asked_abil is not None:
                    raise RuntimeError(
                        f"Checkpoint {self.path} is a train-v2 network with the matchup block: it is played only "
                        f"on the observation it was trained with (own_tera_type and obs_abilities cannot be set)")
                self.obs_line = V2_LINE
                self.own_tera_type = self.v2_blocks["own_tera_type"]
            elif self.league_blocks is not None:
                # A league network: the main line's observation with M24381's settings, then the
                # blocks, nothing else (the overrides below are for this line's own checkpoints).
                if asked is not None or asked_abil is not None:
                    raise RuntimeError(
                        f"Checkpoint {self.path} is a league network: it is played only on the observation it "
                        f"was trained with (own_tera_type and obs_abilities cannot be set)")
                self.obs_line = LEAGUE_LINE
                self.own_tera_type = True
            elif asked is None and asked_abil is None and checkpoint_search_own_tera(ckpt):
                # A checkpoint of the search branch (the baseline run): it records the setting in its
                # run_cfg, not in "obs_settings", and its observation also restores STAB after a Tera.
                self.obs_line = SEARCH_LINE
                self.own_tera_type = True
            elif asked is None:
                # The observation is built the way this checkpoint was trained (release r5 and later:
                # the own Tera types and the ability / item fixes; older checkpoints: neither).
                self.own_tera_type = checkpoint_own_tera_type(ckpt)
            if self.obs_line == LEAGUE_LINE:
                self.obs_abilities = parse_settings({k: v for k, v in MAIN_LINE_SETTINGS.items() if k != "own_tera_type"})
            elif asked_abil is None:
                self.obs_abilities = settings_from_checkpoint(ckpt)
            self.value_settings = value_settings_from_checkpoint(ckpt)
            state = _extract_state_dict(ckpt)
            del ckpt
            # A checkpoint whose tensors do not fit used to load with a warning and then play
            # with untrained weights, and the win rate was reported as if it were the policy's.
            # Buffers that are rebuilt at start (attention masks) may be absent; learned tensors
            # may not. The training-only heads of train-v2 (AUX_HEAD_PREFIXES) are left out: the
            # network built here has none, and only the learner reads them.
            state = {k: v for k, v in state.items() if "attn_mask" not in k}
            heads = [k for k in state if k.startswith(AUX_HEAD_PREFIXES)]
            for k in heads:
                del state[k]
            self.dropped_heads = len(heads)
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            learned = {n for n, _p in self.net.named_parameters()}
            untrained = [k for k in missing if k in learned]
            if untrained or unexpected:
                raise RuntimeError(
                    f"Checkpoint {self.path} does not fit the network: "
                    f"{len(untrained)} learned tensors absent, {len(unexpected)} not recognised")
            if missing:
                logger.warning("Checkpoint %s: %d buffers are not in it and keep their built-in values", self.path, len(missing))
        elif not cfg.random_init:
            raise FileNotFoundError("No checkpoint to load")
        with _in_repo_root():
            # All off: ObservationAssembler(), as before. Own Tera type alone: obs_tera's class, as
            # before. Any obs_abil setting: obs_abil's class (with the own Tera type when on).
            if self.obs_line == V2_LINE:
                self.assembler = V2ObservationAssembler(own_tera_type=self.own_tera_type, movesets=self.v2_blocks["movesets"])
            elif self.obs_line == LEAGUE_LINE:
                self.assembler = league_assembler(movesets=self.league_blocks["movesets"])
            elif self.obs_line == SEARCH_LINE:
                self.assembler = SearchOwnTeraObservationAssembler()
            else:
                self.assembler = make_observation_assembler(own_tera_type=self.own_tera_type, abil=self.obs_abilities)
        self.net = self.net.float().to(self.device).eval()
        self._cache: Dict[str, Any] = {}

    @torch.no_grad()
    def logits(self, tag: str, obs: np.ndarray) -> np.ndarray:
        """Action logits for one observation. `tag` keys the per-battle history cache."""
        x = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        kv, kv_len = self._cache.get(tag, (None, None))
        out, _v_logits, _v, new_kv, new_len = self.net(x, kv_cache=kv, kv_cache_len=kv_len)
        self._cache[tag] = (new_kv, new_len)
        return out[0].float().cpu().numpy()

    def forget(self, tag: str) -> None:
        self._cache.pop(tag, None)


def choice_mask(assembler: Any, battle: Any, obs: np.ndarray) -> np.ndarray:
    """
    The 14 actions a player may choose from at this decision, as float32 flags: the observation's
    own action mask. An assembler whose observation offers a switch the server refuses, because the
    network was trained on that mask (obs_v2.V2ObservationAssembler), says itself what may be chosen.
    """
    pick = getattr(assembler, "choice_mask", None)
    if pick is not None:
        return pick(battle, obs)
    m0, m1 = assembler.offsets["action_mask"]
    return obs[m0:m1].astype(np.float32)


def heuristic_prior(player: Any, assembler: Any, battle: Any, advice: Advice) -> np.ndarray:
    """PSBot's scores laid out on the 14 actions. NaN marks actions it did not score."""
    h = np.full(ACTIONS, np.nan, dtype=np.float64)
    for opt in advice.options:
        order = choice_to_order(player, battle, opt.choice)
        if order is None:
            continue
        idx = assembler.map_order_to_index(order, battle)
        v = opt.value
        if opt.is_switch:
            v -= SWITCH_MARGIN
        elif " terastallize" in opt.choice:
            v -= TERA_MARGIN
        if np.isnan(h[idx]) or v > h[idx]:
            h[idx] = v
    chosen = choice_to_order(player, battle, advice.choice)
    if chosen is not None:
        idx = assembler.map_order_to_index(chosen, battle)
        top = np.nanmax(h) if not np.all(np.isnan(h)) else 0.0
        h[idx] = max(top, h[idx] if not np.isnan(h[idx]) else top) + 1e-3
    return h


def combine(logits: Optional[np.ndarray], prior: Optional[np.ndarray], mask: np.ndarray, cfg: HybridConfig,
            rng: Optional[np.random.Generator] = None) -> int:
    """Picks the action from the policy's logits and the heuristic prior."""
    legal = mask > 0.5
    if not legal.any():
        return 0
    score = np.zeros(ACTIONS, dtype=np.float64)
    if logits is not None and cfg.policy_weight > 0:
        z = np.where(legal, logits.astype(np.float64) / max(cfg.temperature, 1e-4), -np.inf)
        z = z - z[legal].max()
        logp = z - np.log(np.exp(z[legal]).sum())
        score += cfg.policy_weight * np.where(legal, logp, 0.0)
    if prior is not None and not np.all(np.isnan(prior[legal])):
        known = ~np.isnan(prior)
        floor = np.nanmin(prior[legal]) - 0.25  # legal but unscored by PSBot: mildly disfavoured
        h = np.where(known, prior, floor)
        score += cfg.beta * h
        if cfg.veto_gap is not None:
            best = h[legal].max()
            score = np.where(h < best - cfg.veto_gap, -np.inf, score)
    score = np.where(legal, score, -np.inf)
    if cfg.greedy or rng is None:
        return int(np.argmax(score))
    p = np.exp(score - score[np.isfinite(score)].max())
    p = np.where(np.isfinite(score), p, 0.0)
    return int(rng.choice(ACTIONS, p=p / p.sum()))


class HybridPlayer(PSBotMixin, Player):
    """Plays the combined choice of the transformer policy and PSBot."""

    def __init__(self, hybrid: Optional[HybridConfig] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.hybrid = hybrid or HybridConfig()
        self.rng = np.random.default_rng()
        self._events: Dict[str, List[List[str]]] = {}
        self.counts = {"decisions": 0, "policy_agrees": 0, "heuristic_missing": 0, "fallbacks": 0}
        self.policy: Optional[PolicyNet] = None
        try:
            self.policy = PolicyNet(self.hybrid)
            logger.info("Policy loaded from %s on %s", self.policy.path or "random init", self.policy.device)
            tera = bool(getattr(self.policy, "own_tera_type", False))
            abil = getattr(self.policy, "obs_abilities", None) or {}
            if getattr(self.policy, "obs_line", "own") == "v2":
                from obs_v2 import describe
                logger.info("The observation is built as the checkpoint was trained (branch train-v2): %s; "
                            "%d training-only head tensors left out", describe(self.policy.v2_blocks),
                            getattr(self.policy, "dropped_heads", 0))
            elif getattr(self.policy, "obs_line", "own") == "league":
                from obs_league import describe as describe_league
                logger.info("The observation is built as the checkpoint was trained (the league run): %s; "
                            "%d training-only head tensors left out", describe_league(self.policy.league_blocks),
                            getattr(self.policy, "dropped_heads", 0))
            elif getattr(self.policy, "obs_line", "own") != "own":
                logger.info("The observation is built as the checkpoint was trained: the %s branch's own Tera types "
                            "(with STAB kept after a Tera)", self.policy.obs_line)
            elif tera or abil:
                logger.info("The observation is built as the checkpoint was trained: own Tera types %s, ability fixes %s",
                            "on" if tera else "off", ",".join(sorted(k for k, v in abil.items() if v)) or "off")
        except FileNotFoundError:
            logger.warning("No trained checkpoint found: playing as PSBot alone.")
        if self.policy is None:
            from obs_assembler import ObservationAssembler
            with _in_repo_root():
                self.assembler = ObservationAssembler()
        else:
            self.assembler = self.policy.assembler
        # With no heuristic weight there is nothing to ask PSBot for.
        self.psbot_enabled = self.policy is None or self.hybrid.beta > 0 or self.hybrid.veto_gap is not None

    async def _handle_battle_message(self, split_messages: List[List[str]]):
        # The observation's "what happened last turn" block is built from the raw protocol lines.
        if self.policy is not None and split_messages and split_messages[0] and split_messages[0][0].startswith(">"):
            room = split_messages[0][0][1:].strip()
            for parts in split_messages:
                if len(parts) > 1 and parts[0] == "" and parts[1] not in ("t:", "request", "turn"):
                    self._events.setdefault(room, []).append(parts)
        await super()._handle_battle_message(split_messages)

    def _battle_finished_callback(self, battle):
        self._events.pop(battle.battle_tag, None)
        if self.policy is not None:
            self.policy.forget(battle.battle_tag)
        super()._battle_finished_callback(battle)

    async def choose_move(self, battle: Any):
        self.counts["decisions"] += 1
        advice: Optional[Advice] = None
        if self.psbot_enabled:
            try:
                advice = await self.psbot_advice(battle)
            except Exception as e:
                logger.warning("PSBot advice failed in %s: %s", battle.battle_tag, e)
        if advice is None:
            self.counts["heuristic_missing"] += 1

        logits = None
        if self.policy is not None:
            obs = self.assembler.assemble(battle, events=self._events.pop(battle.battle_tag, []))
            logits = self.policy.logits(battle.battle_tag, obs)
            # The observation's action mask (obs_v2's leaves out a switch the server refuses here).
            mask = choice_mask(self.assembler, battle, obs)
        else:
            mask = self.assembler.create_action_mask(battle, list(battle.team.values()))

        prior = heuristic_prior(self, self.assembler, battle, advice) if advice is not None else None
        action = combine(logits, prior, mask, self.hybrid, None if self.hybrid.greedy else self.rng)
        if logits is not None and prior is not None and not np.all(np.isnan(prior)):
            masked = np.where(mask > 0.5, logits, -np.inf)
            if int(np.argmax(masked)) == int(np.nanargmax(np.where(mask > 0.5, prior, np.nan))):
                self.counts["policy_agrees"] += 1

        target, kwargs = self.assembler.map_index_to_order(action, battle)
        if target == "DEFAULT":
            # Nothing mapped (a forced state the 14 actions do not cover): take PSBot's own order.
            order = choice_to_order(self, battle, advice.choice) if advice is not None else None
            if order is None:
                self.counts["fallbacks"] += 1
                return self.choose_random_move(battle)
        else:
            order = self.create_order(target, **kwargs)
        if self.psbot_enabled:
            self.psbot_note(battle, order)
        return order

    def hybrid_stats(self) -> str:
        c = self.counts
        src = "no checkpoint (PSBot alone)" if self.policy is None else (self.policy.path or "untrained network")
        agree = f", policy's first choice matched PSBot's on {100 * c['policy_agrees'] / max(c['decisions'], 1):.0f}% of turns" if self.policy is not None else ""
        return f"{c['decisions']} decisions, model: {src}{agree}, {c['heuristic_missing']} without PSBot advice, {c['fallbacks']} random fallbacks"
