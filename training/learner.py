"""
Learner Actor Module for Distributed Proximal Policy Optimization (PPO).

The LearnerActor is the central training hub. It aggregates experience from 
remote workers, performs vectorized GAE (Generalized Advantage Estimation), 
executes PPO updates, and manages the lifecycle of model checkpoints and 
the self-play league.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import random
import time
import traceback
from typing import Dict, List, Optional, Tuple, Final

import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import RunConfig
import ppo_core
from ppo_core import AsyncEpisodeDataset, gae_from_episode, ppo_update


# Configure logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------
# The read bonus (stage 2, read_reward.py). Pure functions, so the schedule and the cap can be tested
# without a learner.
# ---------------------------------------------------------------------------------------------------

def read_bonus_at(decisions: int, bonus: float, start: int, end: int) -> float:
    """b after `decisions` decisions of the arm: `bonus` before `start`, then linear to 0 at `end`, 0 after."""
    if not bonus or bonus <= 0.0:
        return 0.0
    if decisions < start:
        return float(bonus)
    if decisions >= end or end <= start:
        return 0.0
    return float(bonus) * float(end - decisions) / float(end - start)


def read_bonus_terms(read: torch.Tensor, ep_ids: torch.Tensor, b: float, cap: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    The advantage bonus of every step: b * read (the decision's read tier minus the tier it was read
    by, read_reward.py), with one game's bonuses scaled down together where their sum would leave
    [-cap, cap] (cap <= 0: no cap). Scaling keeps the two seats of a game opposite: their sums are
    equal and opposite before, so they get the same factor. Returns (float32 [n], numbers for the log).
    """
    r = read.reshape(-1).to(torch.float32)
    bonus = r * float(b)
    info = {"read_n": float((r != 0).sum()), "read_capped": 0.0}
    if cap and cap > 0.0 and bonus.numel():
        _uniq, inv = torch.unique(ep_ids.reshape(-1), return_inverse=True)
        sums = torch.zeros(int(inv.max()) + 1, dtype=torch.float32).index_add_(0, inv, bonus)
        over = sums.abs() > float(cap)
        if bool(over.any()):
            scale = torch.where(over, float(cap) / sums.abs().clamp(min=1e-12), torch.ones_like(sums))
            bonus = bonus * scale[inv]
            info["read_capped"] = float(over.sum())
    info["read_sum"] = float(bonus.sum())
    info["read_abs"] = float(bonus.abs().sum())
    return bonus, info


def decision_rate_share(decisions: int, hold: int, total: int) -> float:
    """
    The learning-rate schedule in decisions (LearnerConfig.lr_hold_decisions / lr_total_decisions):
    the share of the full rate after `decisions` decisions of the phase. 1 below `hold`, then
    1/(8p+1)^1.5 with p = min(1, (decisions - hold) / (total - hold)): the step schedule's anneal,
    down to 1/27 at `total` and after. cloud/check_restart.py holds a copy, tested to the letter.
    """
    if decisions < hold:
        return 1.0
    progress = max(0.0, min(1.0, (decisions - hold) / max(1, total - hold)))
    return 1.0 / ((8 * progress + 1) ** 1.5)


AUX_GROUP_PREFIX = "aux_"


def optimizer_state_for_layout(saved: dict, current: dict) -> Optional[dict]:
    """
    A saved MasterWeightOptimizer state laid out for the current optimizer when the two differ only
    in their auxiliary heads (the "aux" zone, whose groups are the last ones): every other group must
    have the same name and size in both. Those keep their Adam state and fp32 masters; the current aux
    groups start fresh (masters = their weights, no Adam state). None when the layouts differ otherwise.
    """
    if not isinstance(saved, dict) or "inner" not in saved or "masters" not in saved:
        return None
    sg, cg = saved["inner"]["param_groups"], current["inner"]["param_groups"]

    def base(groups):
        names = [str(g.get("name", "")) for g in groups]
        k = next((i for i, n in enumerate(names) if n.startswith(AUX_GROUP_PREFIX)), len(names))
        if any(not n.startswith(AUX_GROUP_PREFIX) for n in names[k:]):
            return None
        return groups[:k]

    sb, cb = base(sg), base(cg)
    if sb is None or cb is None or len(sb) != len(cb):
        return None
    if any(a.get("name") != b.get("name") or len(a["params"]) != len(b["params"]) for a, b in zip(sb, cb)):
        return None
    n_base = sum(len(g["params"]) for g in sb)
    if len(saved["masters"]) < n_base or len(current["masters"]) < n_base:
        return None
    saved_ids = [i for g in sb for i in g["params"]]
    cur_ids = [i for g in cb for i in g["params"]]
    state = {}
    for si, ci in zip(saved_ids, cur_ids):
        if si in saved["inner"]["state"]:
            state[ci] = saved["inner"]["state"][si]
    groups = []
    for a, b in zip(sb, cb):
        g = dict(a)
        g["params"] = list(b["params"])
        groups.append(g)
    groups.extend(dict(g) for g in cg[len(cb):])
    return {"inner": {"state": state, "param_groups": groups},
            "masters": list(saved["masters"][:n_base]) + list(current["masters"][n_base:])}


class LearnerActor:
    """
    Ray actor responsible for model optimization and weight distribution.
    
    Attributes:
        run_cfg: The master configuration (RunConfig).
        dataset: CPU-side buffer for asynchronous experience collection.
        net: The central neural network being optimized.
        opt: AdamW optimizer with layer-specific parameter groups.
    """

    def __init__(self, cfg: RunConfig, weight_store: ray.actor.ActorHandle,
                 human_imitation: Optional[Dict[str, object]] = None, fast_update: bool = False,
                 fast_ingest: bool = False, lag_stats: bool = False):
        self.run_cfg = cfg
        self.cfg = cfg.learner
        self.weight_store = weight_store
        # train.py --lag-stats (league.py): how old each update's data is, in seconds and in updates
        # (the workers' per-step column "t_act" against the times this learner pushed its weights), and
        # the KL of the update's first minibatch before its first step (kl0: the behaviour policy against
        # the policy the update starts from). Printed at the end of the update line, the column dropped
        # before the update. False, the default: nothing of it runs and train.py does not pass it.
        self.lag_stats = bool(lag_stats)
        self._push_times: List[float] = []
        # train.py --fast-update: ppo_update's sync-free minibatch loop (ppo_core.ppo_update,
        # fast_update). The same update bit for bit; False, the default, calls ppo_update as before.
        # Not part of RunConfig, so nothing of it reaches a checkpoint.
        self.fast_update = bool(fast_update)
        # train.py --fast-ingest (fast_ingest.py): each message's observations go to the GPU as they
        # come in (copied by a thread of their own), the advantages of all an update's messages at
        # once when it starts, the weight push on a thread of its own. The same tensors in the same
        # order reach every update. False, the default: nothing of it is imported and every line
        # below runs as before. Not part of RunConfig, so no checkpoint has it.
        self.fast_ingest = bool(fast_ingest)
        self._obs_buf = None            # fast_ingest.GpuObsBuffer, made at the first message
        self._pusher = None             # fast_ingest.WeightPusher
        self._gae_pending = None        # fast_ingest.PendingAdvantages
        self._fast_fallbacks = 0        # updates whose observations were joined on the CPU after all
        if self.fast_ingest:
            import fast_ingest
            self._pusher = fast_ingest.WeightPusher(weight_store)
            self._gae_pending = fast_ingest.PendingAdvantages()
            print("[learner] fast_ingest on: observations to the GPU on arrival, advantages of an update's "
                  "messages at once, weight push on its own thread", flush=True)
        # train.py --human-imitation-coef: the top players' decisions as an auxiliary cross-entropy in
        # the PPO updates (human_imitation.py). None, which is what every run without it passes:
        # nothing of it is imported and ppo_update gets no `human`. It never stops a start: data that
        # cannot be had are logged and the run goes on without the term.
        self.human = None
        if human_imitation:
            try:
                from human_imitation import HumanImitation
                self.human = HumanImitation(dict(human_imitation), cfg.obs)
            except Exception as e:
                logger.error(f"Human imitation: NOT USED, the run goes on without it: {e!r}")
                print(f"[learner] human imitation OFF: {e!r}", flush=True)
                self.human = None

        # Device verification
        if self.cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("LearnerActor configured for CUDA, but no GPU was detected.")

        self.net: Optional[nn.Module] = None
        self.opt: Optional[optim.Optimizer] = None
        self.sched: Optional[optim.lr_scheduler.LambdaLR] = None
        
        # CPU-side storage for experience; moved to GPU only during PPO minibatches
        self.dataset = AsyncEpisodeDataset(act_dim=self.run_cfg.env.act_dim, device="cpu")

        self.update_idx = 0
        self.total_episodes = 0
        self.total_steps = 0

        # Backpressure-aware queue for incoming trajectory batches
        self._q: asyncio.Queue = asyncio.Queue(
            maxsize=int(self.run_cfg.rollout.learn_max_pending_batches)
        )

        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        # The read bonus counts this arm's decisions from here (_load_checkpoint sets it from the checkpoint).
        self.read_origin = 0

        # Initialize main loops
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._loop())
        self._task.add_done_callback(self._on_loop_done)
        self._startup_task = loop.create_task(self._maybe_resume_latest())
        
        self.ep_counter = 0

    def _init_optimizer(self):
        """
        Initializes the optimizer applying gradients and learning rates 
        based on the active training mode.
        """
        if self.net is None:
            return
            
        zones = self.net.get_parameter_zones()
        active_grads, lr_mults = self._get_zone_config()
        
        # 1. APPLY GRADIENT TOGGLES
        for zone_name, params in zones.items():
            requires_grad = active_grads[zone_name]
            for _, p in params:
                p.requires_grad = requires_grad
                
        # 2. BUILD PARAMETER GROUPS WITH WEIGHT DECAY LOGIC
        wd_val = float(getattr(self.cfg, "weight_decay", 0.01))
        base_lr = float(self.cfg.lr)
        
        param_groups = []
        master_pairs = []  # (bf16 model param, fp32 master) pairs

        for zone_name, params in zones.items():
            if not active_grads[zone_name]:
                continue # Skip frozen zones entirely

            decay, no_decay = [], []
            for name, p in params:
                # fp32 master copy: AdamW steps this; it is rounded back into
                # the bf16 model param after each step (MasterWeightOptimizer).
                master = p.detach().to(torch.float32).clone()
                master_pairs.append((p, master))

                # Standard Transformer Rule: No decay on LayerNorm, Bias, or 1D tensors
                no_decay_condition = (any(x in name for x in ["_emb", "norm"]) or name.endswith(".bias") or p.ndim == 1)

                if no_decay_condition:
                    no_decay.append(master)
                else:
                    decay.append(master)
                    
            # Pull the mode-specific multiplier from the grid
            target_lr = base_lr * lr_mults[zone_name]
            
            if decay:
                param_groups.append({"params": decay, "lr": target_lr, "weight_decay": wd_val, "name": f"{zone_name}_wd"})
            if no_decay:
                param_groups.append({"params": no_decay, "lr": target_lr, "weight_decay": 0.0, "name": f"{zone_name}_stable"})

        inner = optim.AdamW(param_groups, eps=1e-5)
        self.opt = ppo_core.MasterWeightOptimizer(inner, master_pairs)
        self._init_scheduler()
        
    def _get_zone_config(self) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Returns mode-aware gradient toggles and LR multipliers."""
        mode = self.cfg.mode
        
        # 1. Gradient toggles (True = Unfrozen, False = Frozen)
        grad_toggles = {
            "imitation":               {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": False, "readout": True,  "pi": True,  "v": True},
            "imitation_with_jepa":     {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": True, "readout": True,  "pi": True,  "v": True},
            "imitation_frozen_backbone":{"embeddings": False, "subnets": False, "transformer": False, "jepa": False, "readout": True,  "pi": True,  "v": True},
            "jepa_pretraining":        {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": True,  "readout": False, "pi": False, "v": False},
            "warmup":                  {"embeddings": False, "subnets": False, "transformer": False, "jepa": False, "readout": False, "pi": False, "v": True},
            "warmup_with_actor_reset": {"embeddings": False, "subnets": False, "transformer": False, "jepa": False, "readout": False, "pi": True,  "v": True},
            "ppo_frozen_backbone":     {"embeddings": False, "subnets": False, "transformer": False, "jepa": False, "readout": True,  "pi": True,  "v": True},
            "ppo":                     {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": False, "readout": True,  "pi": True,  "v": True},
            "ppo_with_jepa":           {"embeddings": True,  "subnets": True,  "transformer": True,  "jepa": True, "readout": True,  "pi": True,  "v": True},
        }
        # The auxiliary heads (the "aux" zone: SetHead, OppActionHead; empty without them) learn
        # wherever the backbone does, at the full rate: they start from nothing.
        aux_on = {"jepa_pretraining": False, "warmup": False, "warmup_with_actor_reset": False}
        for m, row in grad_toggles.items():
            row["aux"] = aux_on.get(m, True)

        # 2. LR Multipliers (Mode-Aware)
        lr_mult_configs = {
            "imitation":               {"embeddings": 1.0, "subnets": 1.0, "transformer": 1.0, "jepa": 0.0, "readout": 1.0, "pi": 1.0, "v": 1.0},
            "imitation_frozen_backbone":{"embeddings": 0.0, "subnets": 0.0, "transformer": 0.0, "jepa": 0.0, "readout": 1.0, "pi": 1.0, "v": 2.0},
            "imitation_with_jepa":     {"embeddings": 1.0, "subnets": 1.0, "transformer": 1.0, "jepa": 1.0, "readout": 1.0, "pi": 1.0, "v": 1.0},
            "jepa_pretraining":        {"embeddings": 1.0, "subnets": 1.0, "transformer": 1.0, "jepa": 1.0, "readout": 0.0, "pi": 0.0, "v": 0.0},
            "warmup":                  {"embeddings": 0.0, "subnets": 0.0, "transformer": 0.0, "jepa": 0.0, "readout": 0.0, "pi": 0.0, "v": 1.0},
            "warmup_with_actor_reset": {"embeddings": 0.0, "subnets": 0.0, "transformer": 0.0, "jepa": 0.0, "readout": 0.0, "pi": 1.0, "v": 1.0},
            "ppo_frozen_backbone":     {"embeddings": 0.0, "subnets": 0.0, "transformer": 0.0, "jepa": 0.0, "readout": 1.0, "pi": 1.0, "v": 2.0},
            "ppo_with_jepa":           {"embeddings": 0.5, "subnets": 0.5, "transformer": 0.5, "jepa": 1.0, "readout": 1.0, "pi": 1.0, "v": 2.0},
            "ppo":                     {"embeddings": 0.1, "subnets": 0.1, "transformer": 0.1, "jepa": 0.0, "readout": 1.0, "pi": 1.0, "v": 2.0},
        }
        for m, row in lr_mult_configs.items():
            row["aux"] = 1.0 if grad_toggles.get(m, {}).get("aux") else 0.0

        if mode not in grad_toggles or mode not in lr_mult_configs:
            raise ValueError(f"Unknown training mode configured: {mode}")

        return grad_toggles[mode], lr_mult_configs[mode]

    def _lr_by_decisions(self) -> bool:
        """True when the rate is scheduled in decisions (lr_total_decisions > 0); off by default."""
        return int(getattr(self.cfg, "lr_total_decisions", 0) or 0) > 0

    def _init_decisions_scheduler(self) -> None:
        """
        The LambdaLR of the decisions schedule: lambda(step) = warm(step) x share(D), warm the same
        ramp over the first lr_warmup_steps optimizer steps of a phase as the step schedule's, D
        this phase's decisions as _apply_decision_rate last set it (self._lr_dec). D only changes
        between updates, so every optimizer step of an update has the same share, whether the
        update takes 49.6 steps (it stops early on KL) or 96.
        """
        w_steps = int(getattr(self.cfg, "lr_warmup_steps", 0))
        hold, total = int(self.cfg.lr_hold_decisions), int(self.cfg.lr_total_decisions)
        if not hasattr(self, "_lr_dec"):
            self._lr_dec = 0

        def lr_lambda_decisions(step: int) -> float:
            warm = float(step + 1) / float(w_steps) if step < w_steps else 1.0
            return warm * decision_rate_share(self._lr_dec, hold, total)

        sched_opt = self.opt.inner if isinstance(self.opt, ppo_core.MasterWeightOptimizer) else self.opt
        self.sched = optim.lr_scheduler.LambdaLR(sched_opt, lr_lambda=lr_lambda_decisions)
        print(f"[learner] learning-rate schedule in decisions of this phase: full rate (after a ramp over the first "
              f"{w_steps:,} optimizer steps) until {hold:,} decisions, then down to 1/27 at {total:,} and after", flush=True)

    def _apply_decision_rate(self) -> None:
        """
        Before an update, with the decisions schedule: D = total_steps (this phase's decisions, the
        counter the supervisor's target_decisions reads), and every group's rate where the schedule
        has it for that D at the scheduler's optimizer step. A pure function of D and the step, so
        a restart cannot move it.
        """
        self._lr_dec = int(self.total_steps)
        if self.sched is None:
            return
        rates = [base * fn(self.sched.last_epoch) for fn, base in zip(self.sched.lr_lambdas, self.sched.base_lrs)]
        for pg, lr in zip(self.sched.optimizer.param_groups, rates):
            pg["lr"] = lr
        self.sched._last_lr = rates

    def _init_scheduler(self):
        """Sets up the LambdaLR scheduler based on configured warmup and hold steps."""
        if self._lr_by_decisions():
            # --lr-hold-decisions / --lr-total-decisions. Without them everything below is as it was.
            self._init_decisions_scheduler()
            return
        w_steps = int(getattr(self.cfg, "lr_warmup_steps", 0))
        h_steps = int(getattr(self.cfg, "lr_hold_steps", 0))
        t_steps = int(getattr(self.cfg, "lr_total_steps", 0))

        def lr_lambda(step: int) -> float:
            if step < w_steps:
                return float(step + 1) / float(w_steps)
            if step < (w_steps + h_steps):
                return 1.0
            
            anneal_start = w_steps + h_steps
            # Kept between 0 and 1 whatever the config says. A total at or below the end of the
            # hold leaves no room to anneal in: the rate then drops to 1/27 in one step (warned
            # about below), as it always did.
            progress = max(0.0, min(1.0, (step - anneal_start) / max(1, t_steps - anneal_start)))
            return 1.0 / ((8 * progress + 1) ** 1.5)

        if t_steps <= w_steps + h_steps:
            logger.warning(
                f"lr_total_steps={t_steps} is not past warmup+hold={w_steps + h_steps}: "
                f"the rate will fall 27 times in one step at optimizer step {w_steps + h_steps + 1}"
            )

        # LambdaLR requires a torch Optimizer instance; the wrapper's
        # param_groups are the same objects, so LR updates flow through.
        sched_opt = self.opt.inner if isinstance(self.opt, ppo_core.MasterWeightOptimizer) else self.opt
        self.sched = optim.lr_scheduler.LambdaLR(sched_opt, lr_lambda=lr_lambda)
        
    def _enable_compiled_flex_attention(self):
        """
        Compile flex_attention only inside the learner actor process.
        Safe because Ray actors are separate Python processes.
        """
        if getattr(ppo_core, "_learner_flex_compiled", False):
            return

        ppo_core.flex_attention = torch.compile(ppo_core.flex_attention)
        ppo_core._learner_flex_compiled = True
        logger.info("LearnerActor enabled compiled flex_attention")

    async def _loop(self):
        """
        The main training orchestration loop.
        
        Continually consumes packed experience batches, computes advantages, 
        and triggers PPO updates when sufficient steps are collected.
        """
        self._init_if_needed()
        
        while True:
            # item = ("packed", obs, act, logp, val, rew, done, lengths)
            msgs = [await self._q.get()]
            while not self._q.empty():
                try:
                    msgs.append(self._q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            for msg in msgs:
                if not isinstance(msg, tuple) or msg[0] != "packed":
                    continue
                if getattr(self, "fast_ingest", False):
                    # --fast-ingest: the same columns, the observations also on their way to the GPU.
                    self._ingest_packed_fast(msg)
                    continue
    
                _, obs_cat, act_cat, logp_cat, val_cat, var_cat, rew_cat, done_cat, lengths = msg[:9]
                # Named per-step columns, when a worker sent any (submit_packed_batch's extras).
                extras = msg[9] if len(msg) > 9 else None

                # Convert to CPU Tensors
                obs_all = torch.from_numpy(obs_cat)
                act_all = torch.from_numpy(act_cat).long()
                val_all = torch.from_numpy(val_cat).float()
                var_all = torch.from_numpy(var_cat).float()
                rew_all = torch.from_numpy(rew_cat).float()
                done_all = torch.from_numpy(done_cat).float()
    
                # Vectorized GAE Calculation
                adv_chunks, ret_chunks, id_chunks = [], [], []
                curr = 0
                for length in lengths.tolist():
                    end = curr + int(length)
                    adv, ret = gae_from_episode(
                        rewards=rew_all[curr:end], 
                        values=val_all[curr:end], 
                        dones=done_all[curr:end],
                        gamma=self.cfg.gamma, 
                        lam=self.cfg.gae_lambda,
                        variances=var_all[curr:end] if self.cfg.use_dynamic_lambda else None,
                        lam_min=self.cfg.dynamic_lambda_min,
                        lam_max=self.cfg.dynamic_lambda_max,
                        lam_eps=self.cfg.dynamic_lambda_eps
                    )
                    adv_chunks.append(adv)
                    ret_chunks.append(ret)
                    # Generate matching Episode IDs for this chunk
                    id_chunks.append(torch.full((int(length),), self.ep_counter, dtype=torch.long))
                    
                    curr = end
                    self.ep_counter += 1 # Increment unique global episode ID
    
                columns = (
                    obs_all, act_all, torch.from_numpy(logp_cat), val_all,
                    torch.cat(adv_chunks), torch.cat(ret_chunks), torch.zeros(len(act_all)),
                    torch.cat(id_chunks)
                )
                cols = self._extra_columns(extras, len(act_all)) if extras is not None else None
                if cols:
                    self.dataset.add_steps(*columns, extras=cols)
                else:
                    self.dataset.add_steps(*columns)

                self.total_episodes += len(lengths)
                self.total_steps += len(act_all)

            # Trigger Update
            if len(self.dataset) >= self.cfg.steps_per_update:
                await self._perform_update()

    def _ingest_packed_fast(self, msg) -> None:
        """
        _loop's handling of one "packed" message with --fast-ingest. The CPU dataset gets the same
        eight columns and extras as without it. The advantages and returns: two empty tensors, which
        fast_ingest.PendingAdvantages fills for all the update's messages at once when the update
        takes its data (gae_batch, the same float32 operations as gae_from_episode); a message
        gae_batch does not take gets that loop itself here, as without the flag. The episode ids by
        np.repeat. The observations stay in the CPU dataset by reference (no copy) and are also
        handed to the GPU buffer's copy thread, from which the update takes them.
        """
        import fast_ingest

        _, obs_cat, act_cat, logp_cat, val_cat, var_cat, rew_cat, done_cat, lengths = msg[:9]
        extras = msg[9] if len(msg) > 9 else None

        obs_all = torch.from_numpy(obs_cat)
        act_all = torch.from_numpy(act_cat).long()
        val_all = torch.from_numpy(val_cat).float()
        var_all = torch.from_numpy(var_cat).float()
        rew_all = torch.from_numpy(rew_cat).float()
        done_all = torch.from_numpy(done_cat).float()

        variances = var_all if self.cfg.use_dynamic_lambda else None
        plain = fast_ingest.plain_message(rew_all, val_all, done_all, lengths, variances)
        if plain is not None:
            # filled by self._gae_pending.flush() before the update reads them
            adv_all, ret_all = torch.empty_like(val_all), torch.empty_like(val_all)
        else:
            adv_all, ret_all = fast_ingest.gae_per_episode(rew_all, val_all, done_all, variances, lengths, self.cfg)
        ids = fast_ingest.episode_ids(lengths, self.ep_counter)
        self.ep_counter += len(lengths)

        columns = (obs_all, act_all, torch.from_numpy(logp_cat), val_all, adv_all, ret_all, torch.zeros(len(act_all)), ids)
        cols = self._extra_columns(extras, len(act_all)) if extras is not None else None
        if cols:
            self.dataset.add_steps(*columns, extras=cols)
        else:
            self.dataset.add_steps(*columns)
        if plain is not None:
            if getattr(self, "_gae_pending", None) is None:
                self._gae_pending = fast_ingest.PendingAdvantages()
            self._gae_pending.add(fast_ingest.gae_settings(self.cfg, variances is not None), rew_all, val_all, done_all,
                                  variances, plain, adv_all, ret_all)

        # The GPU copy, only once the dataset has the rows (so the two always hold the same ones).
        if self._obs_buf is None and self.net is not None:
            dev = next(self.net.parameters()).device
            if dev.type == "cuda":
                self._obs_buf = fast_ingest.GpuObsBuffer(dev, int(1.5 * int(self.cfg.steps_per_update)))
        if self._obs_buf is not None:
            self._obs_buf.append(obs_all)

        self.total_episodes += len(lengths)
        self.total_steps += len(act_all)

    def _take_update_data(self):
        """
        _perform_update's (data, names) with --fast-ingest: first the advantages and returns of all
        the update's messages (fast_ingest.PendingAdvantages), then the observations from the GPU
        buffer when it holds this update's rows, every other column the CPU dataset's as always.
        When the buffer cannot give them (no room on the GPU, another layout) the update joins them
        on the CPU as without the flag, and a line says so. Returns (data, names, from_gpu).
        """
        import fast_ingest

        if getattr(self, "_gae_pending", None) is not None:
            self._gae_pending.flush()
        n = len(self.dataset)
        view = self._obs_buf.take(n) if self._obs_buf is not None else None
        if view is None:
            if self._obs_buf is not None:
                self._fast_fallbacks += 1
                if self._fast_fallbacks <= 5 or self._fast_fallbacks % 100 == 0:
                    print(f"[learner] fast_ingest: update {self.update_idx + 1} joins its observations on the CPU "
                          f"({self._obs_buf.reason}; {self._fast_fallbacks} such updates)", flush=True)
            data, names = self.dataset.swap_out_columns()
            return data, names, False
        data, names = fast_ingest.swap_out_columns_without_obs(self.dataset)
        return (view,) + tuple(data[1:]), names, True

    def _extra_columns(self, extras, n: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        A batch's named per-step columns (step_columns.py) as tensors for its n steps, or None.
        A column that is not usable (another number of rows, not numbers) is left out and
        counted in extra_problems, the first few of each kind printed. It never raises: an
        exception here would end the learner's loop.
        """
        problems = getattr(self, "extra_problems", None)
        if problems is None:
            problems = self.extra_problems = {}
        try:
            from step_columns import check_columns

            cols, said = check_columns(extras, n)
        except Exception as e:      # noqa: BLE001
            cols, said = None, [f"extra columns could not be read: {type(e).__name__}"]
        for text in said:
            key = text.split(":")[0][:80]
            problems[key] = problems.get(key, 0) + 1
            if problems[key] <= 3:
                print(f"[learner] {text} (left out)", flush=True)
        if not cols:
            return None
        return {name: torch.from_numpy(np.ascontiguousarray(arr)) for name, arr in cols.items()}

    @staticmethod
    def _stats_1d(x: torch.Tensor, prefix: str) -> Dict[str, float]:
        """
        Basic robust stats for a 1D tensor. Always returns finite floats when possible.
        """
        if x is None:
            return {f"{prefix}_n": 0.0}
        x = x.detach()
        if x.numel() == 0:
            return {f"{prefix}_n": 0.0}
        x = x.float().view(-1)

        # guard against NaNs/Infs
        finite = torch.isfinite(x)
        if not bool(finite.all()):
            xf = x[finite]
        else:
            xf = x
        if xf.numel() == 0:
            return {
                f"{prefix}_n": float(x.numel()),
                f"{prefix}_finite": 0.0,
            }

        def q(p: float) -> float:
            # torch.quantile is fine on GPU/CPU, but convert to float at the end
            return float(torch.quantile(xf, torch.tensor(p, device=xf.device)))

        out = {
            f"{prefix}_n": float(x.numel()),
            f"{prefix}_finite": float(xf.numel()),
            f"{prefix}_mean": float(xf.mean()),
            f"{prefix}_std": float(xf.std(unbiased=False)),
            f"{prefix}_min": float(xf.min()),
            f"{prefix}_p05": q(0.05),
            f"{prefix}_p50": q(0.50),
            f"{prefix}_p95": q(0.95),
            f"{prefix}_max": float(xf.max()),
            f"{prefix}_abs_mean": float(xf.abs().mean()),
            f"{prefix}_abs_max": float(xf.abs().max()),
        }
        return out

    @staticmethod
    def _fmt(d: Dict[str, float], keys) -> str:
        parts = []
        for k in keys:
            v = d.get(k, None)
            if v is None:
                continue
            # ints encoded as float: print without decimals
            if k.endswith("_n") or k.endswith("_finite"):
                parts.append(f"{k}={int(v)}")
            else:
                parts.append(f"{k}={v:.4g}")
        return " ".join(parts)

    async def _perform_update(self):
        """Executes a PPO update and synchronizes weights with Inference."""
        try:
            # From here until the line below is printed this actor answers nothing: no
            # episodes are taken in and no telemetry is given out.
            started = time.perf_counter()

            # 1. Prepare training data
            obs_on_gpu = False
            if getattr(self, "fast_ingest", False):
                # --fast-ingest: the observations from the GPU buffer, the same rows as the join below.
                data, names, obs_on_gpu = self._take_update_data()
            else:
                data, names = self.dataset.swap_out_columns()
            obs_u, act_u, logp_u, val_u, adv_u, ret_u, next_hp_u, ep_ids_u = data[:8]
            # The named per-step columns, row for row with the eight ({} when the workers sent none).
            extra_cols = dict(zip(names, data[8:]))
            lag_info = None
            if getattr(self, "lag_stats", False):
                # --lag-stats: the decisions' wall times leave the data here; the update never sees them.
                lag_info = self._lag_info(extra_cols.pop("t_act", None))

            diag = {}
            diag.update(self._stats_1d(adv_u, "adv_pre"))
            diag.update(self._stats_1d(ret_u, "ret"))
            diag.update(self._stats_1d(val_u, "v_old"))
            diag.update(self._stats_1d(logp_u, "logp_old"))

            # The read bonus (--read-bonus): added to the advantages of the decisions it belongs to,
            # before they are normalised; the returns (the critic's targets) never see it.
            read_info = None
            if float(getattr(self.cfg, "read_bonus", 0.0) or 0.0) > 0.0 and "read" in extra_cols:
                b = read_bonus_at(self.total_steps - int(getattr(self, "read_origin", 0)), self.cfg.read_bonus,
                                  int(self.cfg.read_anneal_start), int(self.cfg.read_anneal_end))
                bonus, read_info = read_bonus_terms(extra_cols["read"], ep_ids_u, b, float(self.cfg.read_cap))
                read_info["b"] = b
                if b > 0.0:
                    adv_u = adv_u + bonus.to(adv_u.dtype)

            # 2. Advantage Normalization
            adv_u = (adv_u - adv_u.mean()) / (adv_u.std() + 1e-8)
            adv_u = torch.clamp(adv_u, -10.0, 10.0)
            
            diag["adv_norm_mean"] = float(adv_u.mean())
            diag["adv_norm_std"]  = float(adv_u.std(unbiased=False))
            
            if obs_on_gpu:
                # The observations stay on the GPU, every other column on the CPU as below.
                import fast_ingest
                train_ds = fast_ingest.DeviceEpisodeDataset(self.run_cfg.env.act_dim, "cpu")
            else:
                train_ds = AsyncEpisodeDataset(self.run_cfg.env.act_dim, obs_u.device)
            if extra_cols:
                train_ds.add_steps(obs_u, act_u, logp_u, val_u, adv_u, ret_u, next_hp_u, ep_ids_u, extras=extra_cols)
            else:
                train_ds.add_steps(obs_u, act_u, logp_u, val_u, adv_u, ret_u, next_hp_u, ep_ids_u)

            # 3. PPO Update Step
            # The top players' imitation term (human_imitation.py), only when it is configured and the
            # mode trains on the PPO loss. Without it ppo_update is called exactly as before.
            human = getattr(self, "human", None)
            human_kw = {}
            if human is not None and self.cfg.mode in ("ppo", "ppo_with_jepa"):
                human.start(self.net, int(self.cfg.batch_seq_len))
                human.begin_update()
                human_kw["human"] = human
            else:
                human = None
            if self._lr_by_decisions():
                # The schedule in decisions (--lr-hold-decisions): this update's rate from this
                # phase's decisions. Off (the default), nothing here runs.
                self._apply_decision_rate()
            # --fast-update: only when it is on does ppo_update get the argument.
            speed_kw = {"fast_update": True} if getattr(self, "fast_update", False) else {}
            if lag_info is not None:
                speed_kw["record_first_kl"] = True          # --lag-stats: kl0, the first minibatch's KL before its step
            stats = ppo_update(
                net=self.net, opt=self.opt, dataset=train_ds, scheduler=self.sched,
                mode=self.cfg.mode, target_net=getattr(self, "target_net", None), **self.cfg.ppo_kwargs(),
                v_min=self.cfg.v_min, v_max=self.cfg.v_max, v_bins=self.cfg.v_bins, **self._aux_update_kwargs(),
                **human_kw, **speed_kw
            )
            
            if self.cfg.mode in ("jepa_pretraining", "imitation_with_jepa", "ppo_with_jepa"):
                tau = float(getattr(self.cfg, "jepa_ema_tau", 0.99))
                with torch.no_grad():
                    for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                        target_param.data.mul_(tau).add_(param.data, alpha=1.0 - tau)

            self.update_idx += 1
            
            # 4. Broadcast weights to Inference (via WeightStore)
            if getattr(self, "_pusher", None) is not None:
                # --fast-ingest: the same two calls in the same order, from the push thread.
                self._pusher.push(self.net.state_dict(), self.update_idx, self.cfg.get_temp(self.total_steps))
            else:
                weights = {k: v.cpu().detach() for k, v in self.net.state_dict().items()}
                self.weight_store.update.remote(weights, self.update_idx)

                # Update Temperature
                new_temp = self.cfg.get_temp(self.total_steps)
                self.weight_store.set_temp.remote(new_temp)
            if lag_info is not None:
                self._push_times.append(time.time())
                if len(self._push_times) > 4096:
                    del self._push_times[:2048]

            logger.info(f"Update {self.update_idx}: Loss={stats.total_loss:.3f}, KL={stats.approx_kl:.4f}")
            # New keys go at the end: pipeline.py, status.py and bench/throughput.py read this
            # line by key and were written against the first seven. sec is how long the update
            # took, data preparation and weight broadcast included. acc is printed in the
            # imitation modes only: the share of steps where the policy's best legal action
            # was the teacher's.
            extra = f" sec={time.perf_counter() - started:.1f}"
            if stats.acc is not None:
                extra += f" acc={stats.acc:.3f}"
            # Two more, again at the end. adv_pre_std: the spread of this update's advantages
            # before they are normalised (the reward's scale as the policy sees it; a bonus added
            # to the advantages is sized against it). lr: the policy head's rate after this
            # update (0 in a mode that does not train the head), which a learning-rate restart
            # (cloud/stage_lr.py) is checked by.
            extra += f" adv_pre_std={diag.get('adv_pre_std', float('nan')):.4g} lr={self._policy_lr():.2e}"
            # The stage 2-4 numbers, only for the features that are on (again at the end).
            if read_info is not None:
                extra += (f" read_n={int(read_info['read_n'])} read_sum={read_info['read_sum']:.4g}"
                          f" read_abs={read_info['read_abs']:.4g} read_capped={int(read_info['read_capped'])} b={read_info['b']:.4g}")
            for k, v in (getattr(stats, "aux", None) or {}).items():
                extra += f" {k}={int(v)}" if k.endswith("_n") else f" {k}={v:.4g}"
            if lag_info is not None:
                # --lag-stats, after everything else on the line (and before the human term's numbers).
                kl0 = getattr(stats, "kl_first", None)
                if kl0 is not None:                         # not in a mode without the PPO loss (no NaN on the line)
                    extra += f" kl0={kl0:.4g}"
                extra += "".join(f" {k}={v:.4g}" for k, v in lag_info.items())
            if human is not None:
                # The top players' imitation term, at the very end and only when it is configured
                # (human_imitation.HumanImitation.report: hum_n, hum_ce, hum_agree, hum_ho ...). The
                # held-out agreement is measured at the first update the data are ready, then every
                # holdout_every updates.
                try:
                    every = max(1, int(human.s.holdout_every))
                    if human.status == "on" and (human.ho["upd"] is None or self.update_idx % every == 0):
                        human.holdout_agreement(self.net, next(self.net.parameters()).device, obs_u.dtype, self.update_idx)
                except Exception as e:
                    logger.warning(f"human imitation: held-out agreement not measured: {e!r}")
                try:
                    extra += human.report()
                except Exception as e:
                    logger.warning(f"human imitation: no report: {e!r}")
            print(
                f"[learner] upd={self.update_idx} "
                f"kl={stats.approx_kl:.4f} clip={stats.clip_frac:.3f} ent={stats.entropy:.3f} "
                f"vloss={stats.v_loss:.3f} ploss={stats.pg_loss:.3f} loss={stats.total_loss:.3f} "
                f"n_mb={stats.n_mb}" + extra
            )

            # 5. Checkpointing
            if self.update_idx % self.cfg.save_every_updates == 0:
                self._save_checkpoint(self._ckpt_path_for_update(self.update_idx))

        except Exception as e:
            logger.error(f"PPO Update failed: {e}")
            self.dataset.clear()
            if getattr(self, "_obs_buf", None) is not None:
                self._obs_buf.reset()
            if getattr(self, "_gae_pending", None) is not None:
                self._gae_pending.clear()
            
    def _lag_info(self, t_act) -> Dict[str, float]:
        """
        --lag-stats: how old this update's decisions are. age_s: seconds from the decision to now (the
        update's start); lag: how many of this learner's weight pushes came after the decision, i.e. how
        many updates behind the policy that chose it is. A push is stamped when the learner hands the
        weights over and the inference actors load it up to --infer-sync-s later, so a decision made in
        between counts one too few: lag is a lower bound, by at most one. Means and the 50th / 90th
        percentiles over the rows that carry the column (lag_rows); never raises.
        """
        out: Dict[str, float] = {"lag_rows": 0.0}
        try:
            if t_act is None:
                return out
            t = t_act.double().reshape(-1).cpu().numpy() if torch.is_tensor(t_act) else np.asarray(t_act, dtype=np.float64).reshape(-1)
            t = t[t > 0]
            out["lag_rows"] = float(t.size)
            if not t.size:
                return out
            now = time.time()
            age = now - t
            pushes = np.asarray(self._push_times, dtype=np.float64)
            lag = (pushes.size - np.searchsorted(pushes, t, side="right")).astype(np.float64)
            out.update(age_s=float(age.mean()), age_p50=float(np.percentile(age, 50)), age_p90=float(np.percentile(age, 90)),
                       lag_mean=float(lag.mean()), lag_p50=float(np.percentile(lag, 50)), lag_p90=float(np.percentile(lag, 90)))
        except Exception as e:      # noqa: BLE001  (a measurement never costs an update)
            out["lag_error"] = 1.0
            logger.warning(f"--lag-stats: not measured this update: {e!r}")
        return out

    def _aux_update_kwargs(self) -> Dict[str, float]:
        """ppo_update's settings for the stage 3-4 terms, only those that are on (none by default)."""
        out = {}
        w = float(getattr(self.cfg, "boosted_row_weight", 1.0) or 1.0)
        if w != 1.0:
            out["boosted_row_weight"] = w
        for name in ("opp_action_aux_coef", "set_head_coef"):
            c = float(getattr(self.cfg, name, 0.0) or 0.0)
            if c > 0.0:
                out[name] = c
        return out

    def _attach_aux_heads(self) -> None:
        """The auxiliary heads this run trains, on the network only (never the EMA target), after make_model and before the optimizer."""
        if float(getattr(self.cfg, "set_head_coef", 0.0) or 0.0) > 0.0:
            if not getattr(self.net, "matchup", False):
                raise ValueError("set_head_coef needs the matchup block in the observation (ObsConfig.matchup, --obs-matchup)")
            self.net.attach_set_head(float(self.cfg.set_head_coef))
        if float(getattr(self.cfg, "opp_action_aux_coef", 0.0) or 0.0) > 0.0:
            self.net.attach_opp_action_head(float(self.cfg.opp_action_aux_coef))

    def _base_state_dict(self) -> Dict[str, torch.Tensor]:
        """The network's weights without its auxiliary heads (what the EMA target and the players hold)."""
        return {k: v for k, v in self.net.state_dict().items() if not k.startswith(ppo_core.AUX_HEAD_PREFIXES)}

    def _policy_lr(self) -> float:
        """The rate the policy head's parameter groups hold now. 0.0 when the mode does not train the head."""
        for pg in self.opt.param_groups:
            if str(pg.get("name", "")).startswith("pi_"):
                return float(pg["lr"])
        return 0.0

    def submit_episode(self, tokens, tmask, amask, act, logp, val, rew, done):
        """Legacy manual submission endpoint."""
        self._q.put_nowait((tokens, tmask, amask, act, logp, val, rew, done))

    async def save_now(self, path: Optional[str] = None) -> str:
        """Forces an immediate checkpoint to disk."""
        if self.net is None or self.opt is None:
            raise RuntimeError("Cannot save: model not initialized yet.")
        if getattr(self, "_resume_failed", None):
            raise RuntimeError(f"Not saving: the resume from {self._resume_failed} failed, so these weights are untrained.")
        if path is None:
            path = self._ckpt_path_for_update(self.update_idx)
        self._save_checkpoint(path)
        return path
            
    async def submit_packed_batch(
        self,
        obs_cat: np.ndarray,
        act_cat: np.ndarray,   # [S]
        logp_cat: np.ndarray,  # [S]
        val_cat: np.ndarray,   # [S]
        var_cat: np.ndarray,   # [S]
        rew_cat: np.ndarray,   # [S]
        done_cat: np.ndarray,  # [S]
        lengths: np.ndarray,   # [B]
        extras=None,           # named per-step columns {name: [S] or [S, ...]} (step_columns.py); None for none
    ):
        if extras is None:
            await self._q.put(("packed", obs_cat, act_cat, logp_cat, val_cat, var_cat, rew_cat, done_cat, lengths))
        else:
            await self._q.put(("packed", obs_cat, act_cat, logp_cat, val_cat, var_cat, rew_cat, done_cat, lengths, extras))
        return True
    
    def _ckpt_path_for_update(self, update_idx: int) -> str:
        return os.path.join(self.cfg.ckpt_dir, f"learner_update_{update_idx:06d}.pt")
    
    def _save_checkpoint(self, path: str):
        """Serializes model, optimizer, and RNG states to disk."""
        payload = {
            "model": self.net.state_dict(),
            "target_model": self.target_net.state_dict() if hasattr(self, "target_net") else None,
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict() if self.sched else None,
            "update_idx": self.update_idx,
            "total_steps": self.total_steps,
            "run_cfg": self.run_cfg.as_dict(),
            "total_episodes": self.total_episodes,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }
        if float(getattr(getattr(self, "cfg", None), "read_bonus", 0.0) or 0.0) > 0.0:
            # Where this arm's decisions are counted from for the read bonus's schedule.
            payload["read_anneal_origin"] = int(getattr(self, "read_origin", 0))
        if getattr(self, "human", None) is not None:
            # Which top-player term trained these weights (absent: none did). Nothing reads it to resume.
            payload["human_imitation"] = self.human.s.as_dict()
        obs_cfg = getattr(getattr(self, "run_cfg", None), "obs", None)
        if getattr(obs_cfg, "main_line", False):
            # The main line's observation (obs_main.py): its "obs_settings" entry as the main line's learner
            # wrote it, with the blocks named, so the window app (which reads this entry first) refuses the
            # network instead of playing it on its own observation. Absent without main_line, as always.
            import obs_main
            payload["obs_settings"] = obs_main.checkpoint_obs_settings(bool(getattr(obs_cfg, "matchup", False)),
                                                                       bool(getattr(obs_cfg, "movesets", False)))
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        # Make the bytes durable before the file takes its final name. Without this a box that
        # loses power right after the rename (a cloud spot interruption) can come back with
        # the new name pointing at a file whose contents never reached the disk: a torn
        # newest checkpoint, which is the one every resume picks.
        with open(tmp, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        self._sync_directory(os.path.dirname(os.path.abspath(path)))

    @staticmethod
    def _sync_directory(directory: str) -> None:
        """Makes a rename in `directory` durable. POSIX only: Windows cannot open a directory this way."""
        if os.name != "posix":
            return
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _init_if_needed(self):
        """Lazy initialization of the network and optimizer."""
        if self.net is None:
            self._enable_compiled_flex_attention()
            # Model runs in bf16 for fast forward/backward, but the optimizer
            # steps fp32 master weights (see MasterWeightOptimizer in ppo_core)
            # so small updates aren't rounded away by bf16.
            self.net = self.run_cfg.make_model().to(self.cfg.device).train()
            self.net.enable_bf16_recurrent_path()

            self.target_net = self.run_cfg.make_model().to(self.cfg.device).eval()
            self.target_net.enable_bf16_recurrent_path()
            self.target_net.load_state_dict(self.net.state_dict())
            for p in self.target_net.parameters():
                p.requires_grad = False

            # The auxiliary heads (none by default): after the target is in step, before the optimizer.
            self._attach_aux_heads()

            self._init_optimizer()

            # The top players' imitation term (only when configured): its loader starts reading the
            # shards now, beside the resume, so the first updates do not wait for it.
            human = getattr(self, "human", None)
            if human is not None and self.cfg.mode in ("ppo", "ppo_with_jepa"):
                human.start(self.net, int(self.cfg.batch_seq_len))
            
    def _latest_ckpt_path(self) -> Optional[str]:
        # The newest file, not the highest number. A change of phase restarts the update count
        # at zero, so after one the highest-numbered file is the last checkpoint of the phase
        # before, and resuming from it would throw the new phase's progress away on every restart.
        paths = glob.glob(os.path.join(self.cfg.ckpt_dir, "learner_update_*.pt"))
        return max(paths, key=os.path.getmtime) if paths else None
            
    async def _maybe_resume_latest(self):
        # Always initialize so we have *some* weights to publish.
        self._init_if_needed()
        assert self.net is not None
    
        loaded = False
        if self.cfg.resume:
            path = self._latest_ckpt_path()
            if path:
                try:
                    self._load_checkpoint(path)
                    loaded = True
                    print(f"[learner] resumed from {path}", flush=True)
                except Exception as e:
                    print(f"[learner] resume failed from {path}: {e!r}", flush=True)
                    # The network now holds fresh weights. A final save on the way out (train.py's
                    # finally, or a SIGTERM) would write them as the newest checkpoint and the
                    # next start would resume from nothing. save_now refuses while this is set.
                    self._resume_failed = path
    
        # IMPORTANT: publish policy 0 exactly once at startup.
        # - If resume worked: pushes resumed weights
        # - Else: pushes fresh init weights
        sd_cpu = {k: v.detach().to("cpu") for k, v in self.net.state_dict().items()}
        self.weight_store.update.remote(sd_cpu, self.update_idx)
    
        if loaded:
            print("[learner] pushed resumed policy 0 to inference", flush=True)
        else:
            print("[learner] pushed init policy 0 to inference", flush=True)

    def _on_loop_done(self, task: asyncio.Task):
        """Safety callback to detect and log crashes in the main training loop."""
        if not task.cancelled() and task.exception():
            logger.critical("Learner training loop CRASHED!")
            traceback.print_exception(None, task.exception(), None)
            
    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        self._init_if_needed()

        assert self.net is not None and self.opt is not None
        
        from obs_matchup import checkpoint_obs_matchup
        was_mu, now_mu = checkpoint_obs_matchup(ckpt), bool(getattr(self.run_cfg.obs, "matchup", False))
        if was_mu != now_mu:
            # The network's input layers differ in size: nothing loads. A checkpoint gets the block by
            # widening (tools/widen_checkpoint.py), and loses it never.
            msg = (f"{path} was trained with obs matchup={was_mu}, this run has matchup={now_mu}: "
                   + ("widen it first (tools/widen_checkpoint.py --in ... --out ...)" if now_mu else "start with --obs-matchup"))
            print(f"[learner] REFUSED: {msg}", flush=True)
            raise RuntimeError(msg)
        from obs_movesets import checkpoint_obs_movesets
        was_ms, now_ms = checkpoint_obs_movesets(ckpt), bool(getattr(self.run_cfg.obs, "movesets", False))
        if was_ms != now_ms:
            # pokemon_net's input layer differs in size, as for the matchup block.
            msg = (f"{path} was trained with obs movesets={was_ms}, this run has movesets={now_ms}: "
                   + ("widen it first (tools/widen_checkpoint.py --movesets --in ... --out ...)" if now_ms else "start with --obs-movesets"))
            print(f"[learner] REFUSED: {msg}", flush=True)
            raise RuntimeError(msg)

        from obs_main import checkpoint_main_line, checkpoint_has_main_settings
        was_ml, now_ml = checkpoint_main_line(ckpt), bool(getattr(self.run_cfg.obs, "main_line", False))
        if was_ml != now_ml and not (now_ml and checkpoint_has_main_settings(ckpt)):
            # The same size, other inputs: a network trained on the main line's observation (obs_main.py)
            # would play this line's, or the other way round, without an error. Refused: a main_line run
            # loading a checkpoint without it (unless it is a main-line network's own, M24381's obs_settings),
            # and a run without main_line loading a checkpoint of a main_line run. A raw main-line checkpoint
            # (no run_cfg key) into a run without main_line loads as it always did (off is as before).
            msg = (f"{path} was trained with obs main_line={was_ml}, this run has main_line={now_ml}: "
                   + ("widen M24381-like checkpoints with tools/widen_checkpoint.py --main-line" if now_ml
                      else "continue it with --obs-main-line"))
            print(f"[learner] REFUSED: {msg}", flush=True)
            raise RuntimeError(msg)

        from config import checkpoint_own_tera_type
        was, now = checkpoint_own_tera_type(ckpt), bool(self.run_cfg.obs.own_tera_type)
        if was != now:
            # Not refused: switching it on for a checkpoint trained without it is how a network gets
            # the input (config.ObsConfig). Switching it off again is almost certainly a mistake.
            print(f"[learner] {path}: trained with own_tera_type={was}, continuing with own_tera_type={now}"
                  + ("" if now else "  <-- the observation loses the own Tera types it was trained on"), flush=True)

        ckpt_mode = ckpt.get("run_cfg", {}).get("learner", {}).get("mode", "unknown")
        current_mode = self.cfg.mode
        
        should_reset_opt = (current_mode != ckpt_mode)
        model_weights = {k: v for k, v in ckpt["model"].items() if "attn_mask" not in k}
        
        # FIX: We intentionally scrubbed attn_mask, so strict must be False
        self.net.load_state_dict(model_weights, strict=False)

        # Re-seed fp32 masters from the loaded (bf16) weights. If the checkpoint
        # carries exact fp32 masters, opt.load_state_dict below overwrites these.
        self.opt.sync_masters_from_model()

        if hasattr(self, "target_net"):
            if "target_model" in ckpt and ckpt["target_model"] is not None:
                # Same scrubbing logic for the target net
                target_weights = {k: v for k, v in ckpt["target_model"].items() if "attn_mask" not in k}
                self.target_net.load_state_dict(target_weights, strict=False)
                print("[learner] Loaded target network EMA state from checkpoint.")
            else:
                self.target_net.load_state_dict(self._base_state_dict())
                print("[learner] No target network in ckpt. Synced target_net to main net.")

        if should_reset_opt:
            print(f"[learner] 🛑 PHASE CHANGE DETECTED ({ckpt_mode} -> {current_mode}).")
            
            # --- Reinitialize actor head weights on warmup ---
            if current_mode == "warmup_with_actor_reset":
                print("[learner] 🧹 Resetting actor head parameters for warmup phase.")
                if hasattr(self.net, "pi_head"):
                    nn.init.normal_(self.net.pi_head.weight, std=0.02)
                    if hasattr(self.net.pi_head, "bias") and self.net.pi_head.bias is not None:
                        nn.init.zeros_(self.net.pi_head.bias)
                # Masters must mirror the freshly reset head weights
                self.opt.sync_masters_from_model()

            print(f"[learner] 🧹 SKIPPING Optimizer/Scheduler load to enforce fresh LR.")
            
            # We treat this as a fresh start, just with pre-trained weights.
            self.update_idx = 0
            self.total_episodes = 0
            self.total_steps = 0
            self.read_origin = 0
            if self._lr_by_decisions():
                self._lr_dec = 0                # the decisions schedule starts with the phase
            self._say_read_schedule()
            return 
        else:
            # Normal Resume Logic (Same Phase)
            if self._lr_by_decisions():
                # The decisions schedule reads this phase's decisions: the checkpoint's, so the rate
                # put in place below (and the line printed) is the one the run had when it saved.
                self._lr_dec = int(ckpt.get("total_steps", 0))
            try:
                self.opt.load_state_dict(ckpt["optimizer"])
                print("[learner] Optimizer state loaded successfully.")
            except Exception as e:
                # The auxiliary heads came or went since the checkpoint: every other parameter keeps
                # its Adam state, the heads start fresh.
                laid = optimizer_state_for_layout(ckpt["optimizer"], self.opt.state_dict())
                try:
                    if laid is None:
                        raise e
                    self.opt.load_state_dict(laid)
                    print("[learner] Optimizer state loaded; the auxiliary heads, new since the checkpoint, start without one.")
                except Exception:
                    print(f"[learner] Optimizer topology mismatch. Resetting optimizer state.")

            # 1. ALWAYS LOAD SCHEDULER FIRST
            if self.sched is not None and "scheduler" in ckpt:
                try:
                    self.sched.load_state_dict(ckpt["scheduler"])
                    print("[learner] Scheduler state loaded successfully.")
                except:
                    pass

            # 2. APPLY OVERRIDE AFTERWARD (Updated for Dynamic Parameter Zones)
            try:
                new_base_lr = float(self.cfg.lr)
                active_grads, lr_mults = self._get_zone_config()

                for pg in self.opt.param_groups:
                    group_name = pg.get("name", "")
                    zone = group_name.split("_")[0] 
                    
                    # Only apply LR jump if the zone exists and is active in the current mode
                    if zone in lr_mults and active_grads.get(zone, False):
                        target_lr = new_base_lr * lr_mults[zone]
                        pg["lr"] = target_lr
                        pg["initial_lr"] = target_lr  # Updates the baseline for the LambdaLR
                
                # 3. CRITICAL: Force the scheduler to accept the new baselines
                if self.sched is not None:
                    self.sched.base_lrs = [pg["initial_lr"] for pg in self.opt.param_groups]
                    # The loop above wrote the full rate into every group. The scheduler only
                    # sets a group's rate when it steps, which is after the next optimizer
                    # step, so the first step after every restart used to run at the full
                    # rate: 11 times the scheduled one for a checkpoint 89 steps into the
                    # warmup, up to 27 times late in the anneal. Put the rate where the
                    # schedule has it for the step this checkpoint was saved at.
                    scheduled = [base * fn(self.sched.last_epoch)
                                 for fn, base in zip(self.sched.lr_lambdas, self.sched.base_lrs)]
                    for pg, lr in zip(self.opt.param_groups, scheduled):
                        pg["lr"] = lr
                    self.sched._last_lr = scheduled
                    
                try:
                    new_wd = float(getattr(self.cfg, "weight_decay", 0.01))
                    for pg in self.opt.param_groups:
                        if pg["weight_decay"] > 0:
                            pg["weight_decay"] = new_wd
                    print(f"[learner] ⚡ Weight Decay Override: Updated to {new_wd}")
                except Exception as e:
                    print(f"[learner] Weight Decay override failed: {e!r}")
                
                print(f"[learner] ⚡ LR Jump Applied Dynamically Across Zones. Base LR={new_base_lr:.2g}")
                if self.sched is not None and self._lr_by_decisions():
                    print(f"[learner] policy rate at the resume: {self._policy_lr():.3e} at optimizer step "
                          f"{self.sched.last_epoch} and {self._lr_dec:,} decisions of this phase, schedule in decisions "
                          f"(lr={self.cfg.lr:g} lr_warmup_steps={self.cfg.lr_warmup_steps} "
                          f"lr_hold_decisions={self.cfg.lr_hold_decisions} lr_total_decisions={self.cfg.lr_total_decisions})",
                          flush=True)
                elif self.sched is not None:
                    # What a restart with new --lr / --lr-hold-steps / --lr-total-steps resumes at
                    # (cloud/stage_lr.py plans it, cloud/check_restart.py checks it beforehand).
                    print(f"[learner] policy rate at the resume: {self._policy_lr():.3e} at optimizer step "
                          f"{self.sched.last_epoch} (lr={self.cfg.lr:g} lr_warmup_steps={self.cfg.lr_warmup_steps} "
                          f"lr_hold_steps={self.cfg.lr_hold_steps} lr_total_steps={self.cfg.lr_total_steps})", flush=True)
            except Exception as e:
                print(f"[learner] LR Jump override failed: {e!r}")

        self.update_idx = int(ckpt.get("update_idx", 0))
        self.total_episodes = int(ckpt.get("total_episodes", 0))
        self.total_steps = int(ckpt.get("total_steps", 0))
        # The read bonus's schedule counts the arm's own decisions: from where this checkpoint says,
        # or, for a checkpoint of another run (no such key), from here.
        self.read_origin = int(ckpt.get("read_anneal_origin", self.total_steps))
        self._say_read_schedule()

        # Push resumed temperature to the WeightStore
        new_temp = self.cfg.get_temp(self.total_steps)
        self.weight_store.set_temp.remote(new_temp)

        if "torch_rng" in ckpt:
            rng = ckpt["torch_rng"]
            if isinstance(rng, torch.Tensor):
                rng = rng.detach().to("cpu")
                if rng.dtype != torch.uint8:
                    rng = rng.to(torch.uint8)
            torch.set_rng_state(rng)

        if "numpy_rng" in ckpt:
            np.random.set_state(ckpt["numpy_rng"])
        if "python_rng" in ckpt:
            random.setstate(ckpt["python_rng"])
            
    def _say_read_schedule(self) -> None:
        if float(getattr(self.cfg, "read_bonus", 0.0) or 0.0) > 0.0:
            print(f"[learner] read bonus {self.cfg.read_bonus:g}: full until {self.cfg.read_anneal_start:,} decisions of this arm, "
                  f"0 at {self.cfg.read_anneal_end:,}, per-game cap {self.cfg.read_cap:g}; this arm's decisions count from "
                  f"total_steps {self.read_origin:,} (now {self.total_steps - self.read_origin:,})", flush=True)

    async def get_stats(self) -> dict:
        return {
            "update": self.update_idx,
            "episodes": self.total_episodes,
            "steps_in_dataset": len(self.dataset),
            "total_steps": self.total_steps,
        }