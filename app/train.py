"""
Cluster Driver for Distributed Pokémon Showdown Reinforcement Learning.

This script initializes the Ray cluster, spawns fractional GPU actors for 
inference and training, and distributes rollout workers across multiple 
Pokémon Showdown server instances.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import math
import os
import sys
import time
from typing import List, Final, Optional

import ray
from config import RunConfig
from inference import InferenceActor, WeightStore
from learner import LearnerActor
from worker import RolloutWorker

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# List of ports where local Pokémon Showdown server instances are running
SERVER_PORTS: Final[List[int]] = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009]



def parse_ports(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Everything here overrides config.py for this run only. With no arguments the run is exactly
    what config.py says, on the ten default ports.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=None, help="LearnerConfig.mode: imitation, imitation_with_jepa, jepa_pretraining, warmup, ppo, ppo_with_jepa, ...")
    ap.add_argument("--ports", default=None, help="Showdown servers to play on, e.g. 8000-8003. One rollout worker per port.")
    ap.add_argument("--battles", type=int, default=None, help="RolloutConfig.target_concurrent_battles")
    ap.add_argument("--rooms-per-pair", type=int, default=None, help="RolloutConfig.rooms_per_pair")
    ap.add_argument("--teacher", choices=("psbot", "simple"), default=None, help="RolloutConfig.imitation_teacher")
    ap.add_argument("--steps-per-update", type=int, default=None, help="LearnerConfig.steps_per_update")
    ap.add_argument("--minibatch", type=int, default=None, help="LearnerConfig.minibatch_size (a multiple of the sequence length)")
    ap.add_argument("--seq-len", type=int, default=None, help="LearnerConfig.batch_seq_len")
    ap.add_argument("--grad-accum", type=int, default=None, help="LearnerConfig.grad_accum_steps")
    ap.add_argument("--save-every", type=int, default=None, help="LearnerConfig.save_every_updates")
    ap.add_argument("--ckpt-dir", default=None, help="LearnerConfig.ckpt_dir")
    ap.add_argument("--shaping", choices=("potential", "faint", "none"), default=None, help="RewardConfig.shaping")
    ap.add_argument("--minutes", type=float, default=0.0, help="stop cleanly after this many minutes (0 = run until Ctrl+C)")
    ap.add_argument("--stop-file", default=None, help="stop cleanly, the way --minutes does, as soon as this file exists (pipeline.py creates it)")
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig.default()
    learner, rollout, reward = {}, {}, {}
    for field, value in (("mode", args.mode), ("steps_per_update", args.steps_per_update), ("minibatch_size", args.minibatch),
                         ("batch_seq_len", args.seq_len), ("grad_accum_steps", args.grad_accum),
                         ("save_every_updates", args.save_every), ("ckpt_dir", args.ckpt_dir)):
        if value is not None:
            learner[field] = value
    for field, value in (("target_concurrent_battles", args.battles), ("rooms_per_pair", args.rooms_per_pair),
                         ("imitation_teacher", args.teacher)):
        if value is not None:
            rollout[field] = value
    if args.shaping is not None:
        reward["shaping"] = args.shaping
    if learner:
        cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, **learner))
    if rollout:
        cfg = dataclasses.replace(cfg, rollout=dataclasses.replace(cfg.rollout, **rollout))
    if reward:
        cfg = dataclasses.replace(cfg, reward=dataclasses.replace(cfg.reward, **reward))
    return cfg


async def main(args: Optional[argparse.Namespace] = None):
    """
    Main entry point for the training cluster.
    
    Orchestrates the following:
    1. Ray initialization and configuration pinning.
    2. Fractional GPU allocation for Learner (0.65) and Inference (0.35).
    3. Distribution of self-play pairs across available local servers.
    4. Real-time telemetry and monitoring.
    """
    try:
        ray.init(ignore_reinit_error=True)
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        return

    # 1. Global Configuration (Single Source of Truth)
    args = args or parse_args([])
    cfg = build_config(args)
    server_ports = parse_ports(args.ports) if args.ports else list(SERVER_PORTS)
    rooms_per_pair = int(cfg.rollout.rooms_per_pair)
    target_concurrent = int(cfg.rollout.target_concurrent_battles)
    logger.info(
        f"mode={cfg.learner.mode} teacher={cfg.rollout.imitation_teacher} reward={cfg.reward.shaping} "
        f"ports={server_ports} steps_per_update={cfg.learner.steps_per_update} "
        f"minibatch={cfg.learner.minibatch_size}x{cfg.learner.grad_accum_steps} seq={cfg.learner.batch_seq_len} ckpt={cfg.learner.ckpt_dir}"
    )

    # Calculate distribution
    total_pairs = math.ceil(target_concurrent / rooms_per_pair)
    n_workers = len(server_ports)
    
    # Balance pairs across worker actors
    base_pairs = total_pairs // n_workers
    remainder = total_pairs % n_workers
    pairs_per_worker = [base_pairs + (1 if i < remainder else 0) for i in range(n_workers)]

    logger.info(
        f"Initializing Run: Target Battles={target_concurrent}, "
        f"Total Pairs={total_pairs}, Workers={n_workers}"
    )

    # 2. Initialize Shared Infrastructure
    # We pin the config in Ray's object store to prevent unnecessary serialization
    cfg_ref = ray.put(cfg)
    weight_store = WeightStore.remote()

    # Define Remote Actors with Resource Constraints
    # Splitting the GPU between Learner (0.65) and Inference (0.35)
    InferRemote = ray.remote(num_gpus=0.35)(InferenceActor)
    LearnerRemote = ray.remote(num_gpus=0.65)(LearnerActor)
    
    # Workers no longer need num_gpus because they just pass data!
    WorkerRemote = ray.remote(num_cpus=1)(RolloutWorker).options(
        max_restarts=-1,  
        max_task_retries=0
    )

    # Spawn Core Actors 
    infer = InferRemote.remote(cfg_ref, weight_store)
    learner = LearnerRemote.remote(cfg_ref, weight_store)

    # 3. Spawn Rollout Workers
    workers = []
    for i, port in enumerate(server_ports):
        count = pairs_per_worker[i]
        if count <= 0:
            continue

        worker_actor = WorkerRemote.remote(
            cfg=cfg_ref,
            inference_actor=infer,
            learner_actor=learner,
            pairs=count,
            server_port=port,
        )
        workers.append(worker_actor)

    logger.info(f"Deployment complete. Distributed {len(workers)} rollout workers.")

    # 4. Staggered Startup
    # Allow core actors to initialize and load checkpoints before starting battles
    logger.info("Staggering initialization of actors...")
    await asyncio.sleep(2)
    
    # Launch asynchronous rollout tasks
    run_refs = []
    for w in workers:
        run_refs.append(w.run.remote())
        time.sleep(4) # required to prevent overloading communication channels

    # 5. Monitoring Loop
    logger.info("Training started. Entering telemetry loop.")
    started = time.time()
    try:
        await _telemetry_loop(args, started, infer, learner, workers, run_refs)
    finally:
        # Checkpoints are otherwise written only every save_every_updates updates, so a stop
        # between two of them used to lose everything since the last one.
        try:
            path = ray.get(learner.save_now.remote(), timeout=180)
            logger.info(f"Saved a final checkpoint: {path}")
        except Exception as e:
            logger.warning(f"Could not save a final checkpoint: {e!r}")
        ray.shutdown()


async def _telemetry_loop(args, started, infer, learner, workers, run_refs):
    while True:
        if args.minutes and time.time() - started > args.minutes * 60:
            logger.info(f"Reached --minutes {args.minutes}. Shutting down.")
            return
        # Ctrl+C from a parent process is unreliable on Windows, so a supervisor asks for the
        # same clean exit (and the checkpoint main() saves on the way out) by creating a file.
        if getattr(args, "stop_file", None) and os.path.exists(args.stop_file):
            logger.info(f"Stop file {args.stop_file} found. Shutting down.")
            return
        try:
            finished, _ = ray.wait(run_refs, num_returns=len(run_refs), timeout=0)
            if finished:
                for ref in finished:
                    try:
                        ray.get(ref) # This will raise the hidden exception
                    except Exception as e:
                        logger.critical(f"FATAL: A RolloutWorker crashed in run(): {e}")
                        import traceback
                        traceback.print_exception(type(e), e, e.__traceback__)
                
                # Stop the run so you don't stare at 0s
                logger.critical("Shutting down cluster due to worker crash.")
                raise SystemExit(1)
            # Gather telemetry from all actors in parallel
            istats_ref = infer.get_stats.remote()
            lstats_ref = learner.get_stats.remote()
            wstats_refs = [w.heartbeat.remote() for w in workers]

            # Ray 2.0+ pattern: Combine refs and fetch via ray.get in a thread
            results = await asyncio.to_thread(ray.get, [istats_ref] + [lstats_ref] + wstats_refs)
            
            istats = results[0]
            lstats = results[1]
            wstats_list = results[2:]

            # Aggregate worker-side metrics
            total_active = sum(w.get("active_battles_worker", 0) for w in wstats_list)
            #max_q = max((w.get("learner_q_size", 0) for w in wstats_list), default=0)
            #total_traj = sum(w.get("traj_in_memory", 0) for w in wstats_list)

            # Professional telemetry print
            stats_msg = (
                f"[Telemetry] ACTIVE: {total_active} | "
                #f"LQ: {max_q} | TRAJ_BUF: {total_traj} | "
                f"INFER: {istats} | TRAIN: {lstats}"
            )
            logger.info(stats_msg)

        except Exception as e:
            logger.warning(f"Telemetry loop encountered an error: {e}")
            # Do not exit; the cluster might still be healthy
        
        await asyncio.sleep(5.0)

if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Cluster terminated.")