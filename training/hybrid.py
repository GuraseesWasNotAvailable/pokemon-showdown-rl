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


class PolicyNet:
    """The transformer, loaded for one-battle-at-a-time inference."""

    def __init__(self, cfg: HybridConfig):
        from config import RunConfig, checkpoint_own_tera_type
        from obs_assembler import ObservationAssembler

        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.path = find_checkpoint(cfg.checkpoint)
        ckpt = torch.load(self.path, map_location="cpu", weights_only=False) if self.path else None
        # The observation the network was trained on: own Tera types shown or not (config.ObsConfig).
        # Read from the checkpoint, so a network trained with them sees them here too, and the
        # older ones (no key) keep the observation they were trained on.
        own_tera_type = checkpoint_own_tera_type(ckpt) if ckpt is not None else RunConfig.default().obs.own_tera_type
        # The matchup block (obs_matchup.py), likewise: only a network trained with it (a widened
        # checkpoint and the runs from one) sees it, and its input layers are built for it.
        from obs_matchup import checkpoint_obs_matchup
        from obs_movesets import checkpoint_obs_movesets
        matchup = checkpoint_obs_matchup(ckpt) if ckpt is not None else False
        # And the moveset block (obs_movesets.py), which comes only on top of the matchup block.
        movesets = matchup and (checkpoint_obs_movesets(ckpt) if ckpt is not None else False)
        # The main line's observation in the first part (obs_main.py; ObsConfig.main_line): only a network
        # trained or widened with it has the key in its run_cfg. Without it nothing here changes.
        from obs_main import checkpoint_main_line
        main_line = checkpoint_main_line(ckpt) if ckpt is not None else False
        with _in_repo_root():
            run_cfg = RunConfig.default()
            if main_line:
                import dataclasses
                import obs_main
                run_cfg = dataclasses.replace(run_cfg, obs=dataclasses.replace(run_cfg.obs, matchup=matchup, movesets=movesets))
                self.assembler = obs_main.make_assembler(matchup, movesets)
            elif movesets:
                import dataclasses
                run_cfg = dataclasses.replace(run_cfg, obs=dataclasses.replace(run_cfg.obs, matchup=True, movesets=True))
                self.assembler = ObservationAssembler(own_tera_type=own_tera_type, matchup=True, movesets=True)
            elif matchup:
                import dataclasses
                run_cfg = dataclasses.replace(run_cfg, obs=dataclasses.replace(run_cfg.obs, matchup=True))
                self.assembler = ObservationAssembler(own_tera_type=own_tera_type, matchup=True)
            else:
                self.assembler = ObservationAssembler(own_tera_type=own_tera_type)
            self.net = run_cfg.make_model()
        if self.path:
            from ppo_core import AUX_HEAD_PREFIXES
            state = _extract_state_dict(ckpt)
            # A checkpoint whose tensors do not fit used to load with a warning and then play
            # with untrained weights, and the win rate was reported as if it were the policy's.
            # Buffers that are rebuilt at start (attention masks) may be absent; learned tensors
            # may not. The learner's auxiliary heads (SetHead, OppActionHead) are training-only
            # and are left out, as the inference actor leaves them out.
            state = {k: v for k, v in state.items() if "attn_mask" not in k and not str(k).startswith(AUX_HEAD_PREFIXES)}
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
            mask = obs[self.assembler.offsets["action_mask"][0]:self.assembler.offsets["action_mask"][1]].astype(np.float32)
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
