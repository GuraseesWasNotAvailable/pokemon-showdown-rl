"""
Inference Server and Model Management for Distributed RL.

This module provides the InferenceActor, which handles large-batch GPU forward 
passes
"""

from __future__ import annotations
import time
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Final

import numpy as np
import ray
import torch
from config import RunConfig
import ppo_core
from ppo_core import masked_sample
from obs_assembler import ObservationAssembler

# Setup logging
logger = logging.getLogger(__name__)

# Constants
CKPT_UPDATE_PATTERN: Final[re.Pattern] = re.compile(r"learner_update_(\d+)\.pt$")

# -----------------------------------------------------------------------------
# CHECKPOINT UTILITIES
# -----------------------------------------------------------------------------

def _extract_state_dict(ckpt: dict) -> dict:
    """
    Surgically extracts the model state dictionary from various checkpoint formats.
    
    Args:
        ckpt: A dictionary loaded via torch.load.
        
    Returns:
        dict: The raw state_dict containing model parameters.
        
    Raises:
        RuntimeError: If no valid state_dict structure is detected.
    """
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Expected dict from checkpoint, got {type(ckpt)}")

    # Check common wrapper keys used in RL frameworks
    for k in ("model", "state_dict", "net", "policy", "actor_critic"):
        v = ckpt.get(k)
        if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
            return v
    
    # If no wrapper, check if the dict itself is the state_dict
    if any(isinstance(x, torch.Tensor) for x in ckpt.values()):
        return ckpt
        
    raise RuntimeError("Could not find a model state_dict inside the checkpoint file.")
@dataclass
class InferenceStats:
    """Container for monitoring inference performance and throughput."""
    flushes: int = 0
    total_requests: int = 0
    avg_batch_size: float = 0.0

# -----------------------------------------------------------------------------
# ACTORS
# -----------------------------------------------------------------------------

@ray.remote
class WeightStore:
    def __init__(self):
        self.weights = None
        self.version = -1
        self.temp = 1.0

    def update(self, weights, version):
        self.weights = weights
        self.version = version

    def get_version(self):
        return self.version

    def get_weights(self):
        return self.weights

    def set_temp(self, temp: float):
        self.temp = float(temp)

    def get_temp(self):
        return self.temp

    def get_meta(self):
        return self.version, self.temp

class InferenceActor:
    """
    High-throughput inference server responsible for GPU kernel execution.
    
    Supports 'Snapshot Inference'—running multiple different model versions 
    within the same batch by grouping observations by their assigned policy_id.
    """
    def __init__(self, cfg: RunConfig, weight_store: ray.actor.ActorHandle):
        self.cfg = cfg
        self.weight_store = weight_store
        self.device: Final[str] = str(cfg.infer.device)
        
        self.current_version = -1
        self.current_temp = cfg.learner.temp_start
        self.stats = InferenceStats()
        self.assembler = ObservationAssembler()
        
        # Build main model and league models
        ppo_core.flex_attention = torch.compile(ppo_core.flex_attention)
        self.net = self.cfg.make_model().to(self.device).eval()
        self.net.enable_bf16_recurrent_path()
        self._last_sync_time = None

        # Every battle side's history (the field token going into each layer, for its last
        # kv_cache_len turns) lives in one tensor, one row per side. It used to be a list of
        # three small tensors per side, cloned after every call and concatenated before the
        # next: about 100 tiny allocations for a batch of 32. Now a call reads its rows with
        # one index and writes them back with one.
        self._kv_slot: Dict[str, int] = {}        # tag -> row
        self._kv_free: List[int] = []             # rows no tag holds, the next to hand out last
        self._kv_pool: Optional[torch.Tensor] = None   # [layers, rows, kv_cache_len, d_model]
        self._kv_len: Optional[torch.Tensor] = None    # [rows], turns of history each row holds
        self._kv_batch: Optional[Tuple[torch.Tensor, Optional[List[int]]]] = None
        self._grow_kv_pool(self._initial_kv_rows())
        
        # Initial load
        self.resume_from_disk()
        
    # Rows the history pool never starts below, whatever the configured number of battles.
    KV_POOL_MIN_ROWS: Final[int] = 64

    def _initial_kv_rows(self) -> int:
        """
        One row per battle side the run can have alive at once, and a quarter more. train.py
        rounds the battles up to whole pairs of rooms_per_pair, and each battle has two sides.
        A row is 0.2 MB (3 layers x 64 turns x 512 bfloat16), so 128 battles take 63 MB.
        """
        ro = self.cfg.rollout
        rooms = max(1, int(ro.rooms_per_pair))
        battles = -(-int(ro.target_concurrent_battles) // rooms) * rooms
        return max(self.KV_POOL_MIN_ROWS, (2 * battles * 5) // 4)

    def _grow_kv_pool(self, rows: int) -> None:
        """Makes the pool at least `rows` long, keeping what the rows in use hold."""
        have = 0 if self._kv_pool is None else int(self._kv_pool.shape[1])
        if rows <= have:
            return
        net = self.net
        pool = torch.zeros(net.n_layers, rows, net.kv_cache_len, net.d_model, device=self.device, dtype=net.recurrent_dtype())
        lens = torch.zeros(rows, dtype=torch.long, device=self.device)
        if have:
            pool[:, :have] = self._kv_pool
            lens[:have] = self._kv_len
            # Not expected in a healthy run: the pool starts with room for every side. Sides
            # whose worker died never send their clear_cache, and this is where that shows.
            logger.warning(f"InferenceActor history pool grown from {have} to {rows} rows ({len(self._kv_slot)} sides alive)")
        self._kv_pool, self._kv_len = pool, lens
        self._kv_free = list(range(rows - 1, have - 1, -1)) + self._kv_free

    def clear_cache(self, tag: str):
        row = self._kv_slot.pop(tag, None)
        if row is not None:
            # What the row holds stays until the row is handed out again, and is wiped then.
            self._kv_free.append(row)

    def clear_all_caches(self):
        self._kv_slot.clear()
        self._kv_free = list(range(int(self._kv_pool.shape[1]) - 1, -1, -1))

    def _build_batched_kv_cache(self, tags: List[str]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        slot = self._kv_slot
        rows: List[int] = []
        fresh: List[int] = []
        fresh_tags: List[str] = []
        try:
            for tag in tags:
                row = slot.get(tag)
                if row is None:
                    if not self._kv_free:
                        self._grow_kv_pool(2 * int(self._kv_pool.shape[1]))
                    row = self._kv_free.pop()
                    slot[tag] = row
                    fresh.append(row)
                    fresh_tags.append(tag)
                rows.append(row)

            if fresh:
                # A side seen for the first time starts from zeros and no history, exactly as it
                # did when each side had its own freshly made tensors. The zeros matter: the model
                # multiplies the unused turns by zero, and a row's previous owner may have left
                # anything there, including values that do not multiply to +0.
                f = torch.tensor(fresh, dtype=torch.long, device=self.device)
                self._kv_pool[:, f] = 0
                self._kv_len[f] = 0
        except BaseException:
            # Growing the pool or wiping the rows failed (the GPU is out of memory). A side that
            # was given a row in this call must not keep it: the row still holds its previous
            # owner's history and length, and on its next decision the side would be found here
            # and treated as known. Hand the rows back, in the order that leaves the free list
            # as it was, so a failed call stores nothing, as it never did.
            for tag, row in zip(reversed(fresh_tags), reversed(fresh)):
                slot.pop(tag, None)
                self._kv_free.append(row)
            raise

        idx = torch.tensor(rows, dtype=torch.long, device=self.device)
        # The same side twice in one call (the worker never sends that) reads the same history
        # twice. Which of the two is written back is settled in _store_new_kv_cache.
        self._kv_batch = (idx, rows if len(set(rows)) != len(rows) else None)
        batched = self._kv_pool[:, idx]                      # [layers, B, K, D], a copy
        return list(batched.unbind(0)), self._kv_len[idx]

    def _store_new_kv_cache(
        self,
        tags: List[str],
        new_kv_cache: List[torch.Tensor],
        new_kv_cache_len: torch.Tensor,
    ) -> None:
        idx, dup_rows = self._kv_batch
        self._kv_batch = None
        new_rows = torch.stack(new_kv_cache, dim=0)          # [layers, B, K, D]
        if dup_rows is not None:
            # The last entry of a repeated side wins, as it did when the entries were stored
            # one after another. An indexed write with a repeated index promises no order.
            last = {row: b for b, row in enumerate(dup_rows)}
            keep = torch.tensor(sorted(last.values()), dtype=torch.long, device=self.device)
            idx, new_rows, new_kv_cache_len = idx[keep], new_rows[:, keep], new_kv_cache_len[keep]
        self._kv_pool[:, idx] = new_rows
        self._kv_len[idx] = new_kv_cache_len

    @torch.no_grad()
    def infer_batch(self, tags: List[str], obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes a batch forward pass for a single policy, ignoring legacy policy IDs.
        """
        self._sync_weights()
        
        B = obs_np.shape[0]
        if len(tags) != B:
            raise ValueError(f"len(tags)={len(tags)} != batch size {B}")
            
        self.stats.flushes += 1
        self.stats.total_requests += B
        self.stats.avg_batch_size = (self.stats.avg_batch_size * 0.95) + (B * 0.05)

        # The workers send the observation as the float16 the assembler writes it in (half the
        # bytes to pickle and to move). It is widened here, on the GPU, to the float32 the
        # model has always been given: every float16 is a float32, so nothing changes.
        obs = torch.from_numpy(obs_np).to(self.device, non_blocking=True).float()
        
        kv_cache, kv_cache_len = self._build_batched_kv_cache(tags)

        logits, v_logits, v_exp, new_kv_cache, new_kv_cache_len = self.net(
            obs,
            kv_cache=kv_cache,
            kv_cache_len=kv_cache_len,
        )
        
        v_probs = torch.softmax(v_logits, dim=-1)
        v_var = (v_probs * (self.net.v_support - v_exp.unsqueeze(-1))**2).sum(dim=-1)
        
        # Extract Action Mask
        m_start, m_end = self.assembler.offsets["action_mask"]
        mask_sub = obs[:, m_start:m_end].float()
        
        bad = (mask_sub.sum(dim=1) <= 0.49)
        if bad.any():
            mask_sub = mask_sub.clone()
            mask_sub[bad, 0] = 1.0 # Force struggle/default if no valid moves
        
        # Sample actions
        act_game, logp_game, _ = masked_sample(logits, mask_sub, temp=self.current_temp)
        self._store_new_kv_cache(tags, new_kv_cache, new_kv_cache_len)

        return (
            act_game.cpu().numpy(),
            logp_game.cpu().numpy(),
            v_exp.cpu().numpy(),
            v_var.cpu().numpy()
        )
    
    def set_temp(self, temp: float):
        self.current_temp = temp

    def _sync_weights(self):
        """Pulls updated weights from the WeightStore if a new version is available."""
        # --- THE THROTTLE ---
        now = time.time()
        if self._last_sync_time:
            if now - self._last_sync_time < 5.0:
                return
        self._last_sync_time = now
        # --------------------

        try:
            # Use a lightweight version check before pulling the heavy state_dict
            latest_v = ray.get(self.weight_store.get_version.remote(), timeout=1.0)
            
            if latest_v > self.current_version:
                weights = ray.get(self.weight_store.get_weights.remote(), timeout=2.0)
                if weights:
                    self.net.load_state_dict(weights, strict=False)
                    self.current_version = latest_v
                    self.clear_all_caches()
                    
                    # --- SYNC TEMPERATURE ---
                    self.current_temp = ray.get(self.weight_store.get_temp.remote())
                    # ------------------------
                    
                    logger.info(f"InferenceActor synced to version {latest_v} | Temp: {self.current_temp:.3f}")
        except Exception:
            pass # Prevent timeout errors from crashing the GPU loop

    def resume_from_disk(self):
        """Initializes the actor with the latest weights found in the checkpoint directory."""
        if not getattr(self.cfg.learner, "resume", False):
            return

        ckpt_dir = self.cfg.learner.ckpt_dir
        if not os.path.exists(ckpt_dir):
            return

        # Find all checkpoints and load the most recent one
        import glob
        paths = glob.glob(os.path.join(ckpt_dir, "learner_update_*.pt"))
        
        if not paths:
            return
            
        latest_ckpt = max(paths, key=os.path.getmtime)  # newest file, not highest number (see LearnerActor._latest_ckpt_path)
        try:
            st = _extract_state_dict(torch.load(latest_ckpt, map_location="cpu", weights_only=False))
            self.net.load_state_dict(st)
            logger.info(f"InferenceActor resumed from {latest_ckpt}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest_ckpt}: {e}")
            
    def refresh_snapshots_from_disk(self):
        """
        Legacy endpoint for the LearnerActor. 
        In a single-policy setup, dynamic disk reloading is unnecessary because 
        the latest weights are synced continuously via RAM (WeightStore).
        """
        pass
                
    def get_stats(self) -> dict:
        """Returns diagnostic metrics."""
        return {
            "total_requests": self.stats.total_requests,
            "avg_batch_size": round(self.stats.avg_batch_size, 2),
            "model_version": self.current_version
        }