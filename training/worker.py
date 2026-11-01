"""
Distributed Rollout Worker for Pokémon Showdown RL.

This module uses a Thread-Bridge architecture to quarantine poke-env's async 
websocket logic into a background thread. The main thread operates completely 
synchronously, gathering perfect numpy batches and dispatching them to a central 
GPU Inference Actor via zero-copy Ray RPCs.
"""

from __future__ import annotations

import random
import asyncio
import functools
import logging
import os
import sys
import time
import traceback
import threading
import queue
import secrets
from typing import Any, Dict, List, Optional, Tuple, Final
from collections import Counter, OrderedDict

import numpy as np
import ray
import poke_env
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import DefaultBattleOrder
from poke_env.player.baselines import SimpleHeuristicsPlayer
from config import RunConfig
from obs_assembler import ObservationAssembler
from players import corrected_stat_estimation as _corrected_stat_estimation
from psbot_bridge import PSBotMixin
from rl_reward import load_potential, shaped_rewards

# Global Constants
ZOMBIE_LIMIT_S: Final[float] = 300.0  
REPAIR_INTERVAL_S: Final[float] = 30.0

# train.py --lag-stats (league.py): every PPO step also records the wall time of its decision ("t_act",
# a per-step column the learner reads to measure how old its data is, then drops). Set by
# RolloutWorker.__init__ in this worker's process before its players are made; False, the default,
# leaves every player as it was.
LAG_STATS: bool = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MONKEY PATCHES & SYSTEM FIXES
# ---------------------------------------------------------------------------

# Apply the Monkey Patch
SimpleHeuristicsPlayer._stat_estimation = _corrected_stat_estimation

_original_handle_message = poke_env.ps_client.PSClient._handle_message

# What happens when the server calls a choice invalid (bench/MEMORY.md has the measurements).
#
# r1 answered every "|error|[Invalid choice]" twice: this module's patch sent "/choose default"
# at once, and poke-env asked the player again and sent what it chose a decision later. The
# server takes the default. The second choice replaces it, or, when the turn has gone on by
# then, is taken as the NEXT turn's move: from there the client is one decision behind the
# server, some of its late choices are refused, and every refusal adds a default and one more
# asking that waits its turn on the room's lock. When the game ends on the server the client
# still has those askings ahead of its "|win|". Everything it sends now is refused, the patch
# answers each refusal with another default before it takes the lock, and that goes round until
# the "|win|" has come through the queue: 43,122 refusals in 16 seconds with scripted players
# on a real server (4,478 a second at the peak, 40,304 handler tasks waiting on the lock,
# about 3 KB each); 28,280 in 18 seconds in a training run, whose battle went to the learner with
# 66 steps for 22 turns: every asking-again is recorded as a step, though its choice is not
# what was played.
#
# Now a refused choice gets exactly one answer, "/choose default", and only when a default can
# be taken:
#   * not for the errors that say this side cannot choose anything at this moment
#     (NOTHING_TO_CHOOSE): a default is one more choice and gets the same error back;
#   * not twice for one request (a room is in the client's "_default_sent" set from its default
#     until its next request), so no error text, known or not, can start a bounce;
#   * and the training player does not ask the policy again (SyncBridgePlayer.
#     _handle_battle_message keeps these errors from poke-env): the server has the default, and
#     a second choice would replace it, be refused, or be taken as the NEXT turn's move unseen.
#
# "It has a wait request" ("There's nothing to choose", from the server's room-battle.ts, also
# when the game is over), "the turn is already being played" ("Sorry, too late ..."), "it is not
# its turn" ("Can't do anything: ...", from the simulator), "its choice cannot be taken back"
# ("Can't undo: ..."):
NOTHING_TO_CHOOSE: Final[Tuple[str, ...]] = (
    "|error|[Invalid choice] There's nothing to choose",
    "|error|[Invalid choice] Sorry, too late to make a different move",
    "|error|[Invalid choice] Can't do anything",
    "|error|[Invalid choice] Can't undo",
)
INVALID_CHOICE: Final[str] = "|error|[Invalid choice]"
# Since this process started: "/choose default" sent after an invalid choice, and not sent
# (the error was one of the above, or this request had its default already).
CHOICE_REPAIR: Dict[str, int] = {"defaults_sent": 0, "defaults_withheld": 0}


def default_would_be_refused(message: str) -> bool:
    """True when every "[Invalid choice]" error in this message is one that "/choose default" cannot repair."""
    errors = [line for line in message.split("\n") if line.startswith(INVALID_CHOICE)]
    return bool(errors) and all(line.startswith(NOTHING_TO_CHOOSE) for line in errors)


def default_is_due(client: Any, message: str) -> bool:
    """
    Called for every message a client receives, in the order they arrive. True when the message
    carries an invalid choice that "/choose default" should answer: it is one a default can
    repair, and this room has had no default since its last request.
    """
    sent = client.__dict__.get("_default_sent")
    invalid = INVALID_CHOICE in message
    if not sent and not invalid:
        return False                                             # every ordinary message ends here
    room_id = message.split("\n", 1)[0].replace(">", "").strip()
    if sent and room_id in sent and ("|request|{" in message or "|deinit" in message
                                     or "|win|" in message or "|tie" in message):
        sent.discard(room_id)                                    # a new request, or the room is over
    if not invalid:
        return False
    if default_would_be_refused(message) or (sent is not None and room_id in sent):
        CHOICE_REPAIR["defaults_withheld"] += 1
        return False
    if sent is None:
        sent = client.__dict__.setdefault("_default_sent", set())
    if len(sent) > 4096:                                         # cannot happen (see above); never a leak
        sent.clear()
    sent.add(room_id)
    CHOICE_REPAIR["defaults_sent"] += 1
    return True


# The trainer's log has to say how often the server refuses a choice or closes a room, and why:
# on the cloud box neither was known (bench/MEMORY.md). One line on stderr each, which Ray copies
# into the trainer's log under "(RolloutWorker pid=...)": the first REPORT_FIRST of a kind in a
# worker, then every REPORT_EVERY-th, so that a wrong legal-move mask (thousands of refusals)
# cannot fill the log. Nothing else is done with it.
REPORT_FIRST: Final[int] = 10
REPORT_EVERY: Final[int] = 1000
REPORTED: Dict[str, int] = {}


def report_rare(kind: str, text: str) -> None:
    try:
        n = REPORTED[kind] = REPORTED.get(kind, 0) + 1
        if n <= REPORT_FIRST or n % REPORT_EVERY == 0:
            print(f"[rollout worker {os.getpid()}] {kind}, number {n} in this worker: {text}", file=sys.stderr, flush=True)
    except Exception:       # a diagnostic line must never stop a battle
        pass


def _apply_patches():
    """Applies necessary patches to poke_env and asyncio for stability."""
    original_handle = poke_env.ps_client.PSClient._handle_message
    async def loud_handle(self, message):
        try:
            if default_is_due(self, message):
                room_id = message.split("\n")[0].replace(">", "").strip()
                logger.debug(f"NN chose invalid move in {room_id}. Forcing /choose default.")
                await self.send_message("/choose default", room=room_id)
            await original_handle(self, message)
        except Exception:
            logger.critical(f"Poke-env crash on message: {message[:100]}...")
            traceback.print_exc()
            raise
    poke_env.ps_client.PSClient._handle_message = loud_handle

    if sys.platform == "win32":
        import asyncio.proactor_events
        def silence_proactor_error(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                try: return func(self, *args, **kwargs)
                except (AssertionError, OSError): pass
            return wrapper
        asyncio.proactor_events._ProactorBaseWritePipeTransport._loop_writing = \
            silence_proactor_error(asyncio.proactor_events._ProactorBaseWritePipeTransport._loop_writing)

_apply_patches()

def make_server_conf(host: str, port: int) -> ServerConfiguration:
    ws_url = f"ws://{host}:{port}/showdown/websocket"
    http_action = f"http://{host}:{port}/action.php?"
    return ServerConfiguration(ws_url, http_action)

def battle_tag_for(battle) -> str:
    tag = getattr(battle, "battle_tag", None)
    if not tag: tag = f"pyid_{id(battle)}"
    return str(tag)

def mk_name(run_tag: str, p: int, side: str) -> str:
    return f"p{run_tag}{p:03d}{side}"

# --- RESTORED NETWORKING HELPERS ---
async def wait_for_login(player: SyncBridgePlayer, timeout_s: float = 30.0) -> None:
    c = getattr(player, "ps_client", None)
    if c is None: raise RuntimeError(f"{player.username} has no ps_client")
    fn = getattr(c, "wait_for_login", None)
    if not callable(fn): raise RuntimeError(f"{player.username}.ps_client has no wait_for_login()")
    await asyncio.wait_for(fn(), timeout=timeout_s)

async def safe_send(player: SyncBridgePlayer, message: str, room: str, *, retries: int = 8) -> None:
    await wait_for_login(player)
    last_err = None
    for i in range(retries):
        try:
            await player.ps_client.send_message(message, room=room)
            return
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.05 * (i + 1))
    raise RuntimeError(f"[safe_send] failed: {message!r} ({last_err!r})")

async def join_lobby(player: SyncBridgePlayer) -> None:
    await safe_send(player, "/join lobby", room="")

# ---------------------------------------------------------------------------
# SYNC LEARNER CLIENT
# ---------------------------------------------------------------------------

class SyncLearnerClient:
    # Episodes handed in and dropped as the oldest (counts only, read by the heartbeat with --lag-stats). Class
    # defaults, so a client made without __init__ (tests do) counts as well instead of failing.
    submitted = 0
    evicted = 0

    def __init__(self, learner_actor, cfg: RunConfig, maxsize: Optional[int] = None):
        """
        maxsize: episodes this worker holds for the learner before it drops the oldest (train.py
        --learn-queue-episodes, league.py). None, the default: learn_max_pending_batches x
        learn_max_episodes (5 x 32 = 160), as always.
        """
        self.learner_actor = learner_actor
        self.cfg = cfg.rollout
        if maxsize is None:
            self.q = queue.Queue(maxsize=self.cfg.learn_max_pending_batches * self.cfg.learn_max_episodes)
        else:
            self.q = queue.Queue(maxsize=int(maxsize))
        self.submitted = 0
        self.evicted = 0
        threading.Thread(target=self._worker, daemon=True).start()

    def submit_episode(self, obs: np.ndarray, act: np.ndarray, logp: np.ndarray,
                       val: np.ndarray, var: np.ndarray, rew: np.ndarray, done: np.ndarray,
                       extras: "Dict[str, np.ndarray] | None" = None) -> bool:
        """extras: named per-step columns of this episode (step_columns.py), None for none."""
        # If the queue is full, evict the oldest trajectory
        self.submitted += 1
        if self.q.full():
            try:
                self.q.get_nowait()
                self.evicted += 1
            except queue.Empty:
                pass

        # Now there is guaranteed room for the absolute newest data
        if extras is None:
            self.q.put_nowait((obs, act, logp, val, var, rew, done))
        else:
            self.q.put_nowait((obs, act, logp, val, var, rew, done, extras))
        return True

    def _worker(self):
        while True:
            items = [self.q.get()]
            while len(items) < self.cfg.learn_max_episodes:
                try: items.append(self.q.get_nowait())
                except queue.Empty: break
            
            packed = self._prepare_batch(items)
            try:
                ray.get(self.learner_actor.submit_packed_batch.remote(*packed))
            except Exception as e:
                logger.error(f"Learner submission failed: {e}")

    @staticmethod
    def _prepare_batch(items: List[Tuple]) -> Tuple:
        lengths = np.asarray([it[1].shape[0] for it in items], dtype=np.int32)
        packed = (
            np.concatenate([it[0] for it in items], axis=0),
            np.concatenate([it[1] for it in items], axis=0).astype(np.int64),
            np.concatenate([it[2] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[3] for it in items], axis=0).astype(np.float32),
            np.concatenate([it[4] for it in items], axis=0).astype(np.float32), # var_cat
            np.concatenate([it[5] for it in items], axis=0).astype(np.float32), # rew_cat
            np.concatenate([it[6] for it in items], axis=0).astype(np.float32), # done_cat
            lengths
        )
        if any(len(it) > 7 for it in items):
            # Named per-step columns: joined in the episodes' order, a fill value for the steps of
            # an episode that came without one (step_columns.pack_columns). The learner takes them
            # as submit_packed_batch's extras. Without any, the batch is the eight as always.
            from step_columns import pack_columns

            extras = pack_columns([it[7] if len(it) > 7 else None for it in items], lengths.tolist())
            if extras is not None:
                packed = packed + (extras,)
        return packed

# ---------------------------------------------------------------------------
# TOLLBOOTH PLAYER (RUNS IN ASYNC THREAD)
# ---------------------------------------------------------------------------

IMITATION_MODES: Final = ("imitation", "imitation_frozen_backbone", "jepa_pretraining", "imitation_with_jepa")


class SyncBridgePlayer(PSBotMixin, SimpleHeuristicsPlayer):
    def __init__(self, cfg: RunConfig, event_queue: queue.Queue, action_futures: dict, ep_sem: asyncio.Semaphore, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        # PSBot only has work to do while it is the teacher. In self-play its engine stays off.
        self.psbot_enabled = (
            getattr(cfg.learner, "mode", "ppo") in IMITATION_MODES
            and getattr(cfg.rollout, "imitation_teacher", "simple") == "psbot"
        )
        self.event_queue = event_queue
        self.action_futures = action_futures
        self.ep_sem = ep_sem
        self.assembler = cfg.make_assembler()   # ObsConfig.own_tera_type
        # The position value the shaped reward is built from, read at every decision.
        self.potential = None
        if getattr(cfg.reward, "shaping", "none") == "potential":
            self.potential = load_potential(cfg.reward.potential_file or None)
        self._battle_starts = {}
        self._last_act_time = {}
        self._episode_slot_state = {} 
        self._battle_events = {}
        # Rooms whose battle has ended and been dropped (newest last, bounded).
        self._finished_rooms: "OrderedDict[str, None]" = OrderedDict()
        # Battles given up because the server closed their room without a result (see _room_closed).
        self.rooms_closed = 0
        # Choices the server called invalid ("|error|[Invalid choice]", see _handle_battle_message).
        self.choices_refused = 0
        # The other seat of this player's pair (set by the vector wrapper): in self-play it holds the
        # opposing team's true abilities and items, the SetHead's labels.
        self.partner = None
        # The stage 2-4 per-step columns this player records for the learner (step_columns.py), only
        # in PPO and only for the features that are on (config.LearnerConfig): read and opp_action
        # (read_reward.ReadTracker), boosted, set_labels. All off: nothing here runs.
        lc = getattr(cfg, "learner", None)
        ppo = getattr(lc, "mode", "ppo") not in IMITATION_MODES
        self.reads_on = ppo and (float(getattr(lc, "read_bonus", 0.0) or 0.0) > 0.0
                                 or float(getattr(lc, "opp_action_aux_coef", 0.0) or 0.0) > 0.0)
        self.boosted_on = ppo and float(getattr(lc, "boosted_row_weight", 1.0) or 1.0) != 1.0
        self.set_labels_on = ppo and float(getattr(lc, "set_head_coef", 0.0) or 0.0) > 0.0 and bool(
            getattr(getattr(cfg, "obs", None), "matchup", False))
        # With the moveset block the labels also hold the true moves and Tera type (6 x 8, not 6 x 3).
        self.set_labels_movesets = bool(getattr(getattr(cfg, "obs", None), "movesets", False))
        # --lag-stats (LAG_STATS above): the decision's wall time as the column "t_act". Off: not recorded.
        self.lag_on = ppo and LAG_STATS
        self.step_features = self.reads_on or self.boosted_on or self.set_labels_on or self.lag_on
        self._read_trackers: Dict[str, Any] = {}     # tag -> read_reward.ReadTracker
        self._step_cols: Dict[str, Dict[str, list]] = {}   # tag -> column name -> one entry per STEP

    def _get_unique_tag(self, battle):
        return f"{self.username}_{battle_tag_for(battle)}"
    
    async def _handle_battle_message(self, split_messages: list[list[str]]):
        """Corrected for 0.15: Processes a batch of pre-split protocol lines."""
        if not split_messages or not split_messages[0]:
            await super()._handle_battle_message(split_messages)
            return

        # 1. Extract Room ID from the first line
        room_id = None
        if split_messages[0] and split_messages[0][0].startswith(">"):
            room_id = split_messages[0][0][1:].strip()

        # A finished battle is removed from self._battles to free memory, but the server still
        # sends that room a few more lines (the deinit after /leave, for one). poke-env's handler
        # waits for a battle that will never come back, so every finished game used to leave a
        # coroutine parked forever, and every new battle woke all of them, so the cost grew
        # with every game played. Nothing is waiting for these lines: drop them.
        if room_id and room_id in self._finished_rooms:
            is_init = len(split_messages) > 1 and len(split_messages[1]) > 1 and split_messages[1][1] == "init"
            if not is_init:
                return

        # "|deinit": the server has taken this player out of the room. After a battle that ended
        # with a result that is the answer to our own /leave, and the room is in _finished_rooms
        # (above). Here the battle has NOT ended for poke-env: rlspawn.ts closes every battle room
        # that is five minutes old ("|expire|", then "|deinit", never a "|win|"), so no finished
        # battle is ever reported and r1 kept everything about it for good (see _room_closed).
        closed = bool(room_id) and any(parts[1:] == ["deinit"] for parts in split_messages[1:])
        if closed and room_id not in self._battles:
            # Nothing of this room is held (it was never started here). Its lines must not wait
            # in poke-env for a battle that will not come.
            self._note_finished_room(room_id)
            self._battle_events.pop(room_id, None)
            return

        # "|error|[Invalid choice]": the server refused this side's choice. poke-env answers by
        # asking the player again (choose_move: one more STEP for the worker, one more choice
        # for the server). The server already has this request's "/choose default" (the patch at
        # the top of this file), so the second choice replaces the default, or is refused with
        # another error, or arrives after the next request and is played unseen as that turn's
        # move (which leaves the client one decision behind the server, and ends in the flood of
        # refusals described at the top of this file), and every asking is a step in the episode
        # that was never played. r1 did that. Here the error stops: poke-env
        # does not see it and it is not harvested, so the next decision is asked for by the next
        # request and observed exactly as in a battle without the error.
        if room_id and any(len(parts) > 2 and parts[1] == "error" and parts[2].startswith("[Invalid choice]")
                           for parts in split_messages[1:]):
            kept = [parts for parts in split_messages[1:]
                    if not (len(parts) > 2 and parts[1] == "error" and parts[2].startswith("[Invalid choice]"))]
            self.choices_refused += len(split_messages) - 1 - len(kept)
            self._report_refused(room_id, split_messages)
            if not kept:
                return
            split_messages = [split_messages[0]] + kept

        # 2. Harvest Events
        if room_id:
            for parts in split_messages:
                # Valid protocol lines start with an empty string (from the '|' split)
                if len(parts) > 1 and parts[0] == "":
                    cmd = parts[1]
                    # Filter out structural tags we don't need for transitions
                    if cmd not in ("t:", "request", "turn"):
                        # Save the parts exactly as obs_transitions expects them
                        self._battle_events.setdefault(room_id, []).append(parts)

        # 3. Pass to parent so poke-env can update its internal Battle object
        await super()._handle_battle_message(split_messages)

        # 4. The room is gone and poke-env reported no finished battle for it: give it back.
        if closed and room_id in self._battles:
            self._room_closed(room_id)

    def _report_refused(self, room_id: str, split_messages: list) -> None:
        """The server's own words for a refused choice, into the trainer's log (see report_rare)."""
        try:
            battle = self._battles.get(room_id)
            active = getattr(getattr(battle, "active_pokemon", None), "species", None)
            where = (f"turn {getattr(battle, 'turn', '?')}, active {active}"
                     + (", forced switch" if getattr(battle, "force_switch", False) else "")
                     + (", trapped" if getattr(battle, "trapped", False) else ""))
            for parts in split_messages[1:]:
                if len(parts) > 2 and parts[1] == "error" and parts[2].startswith("[Invalid choice]"):
                    report_rare("choice refused", f"{self.username} {room_id} ({where}): {'|'.join(parts[2:])[:200]}")
        except Exception:
            pass

    def _note_finished_room(self, room_id: str) -> None:
        self._finished_rooms[room_id] = None
        while len(self._finished_rooms) > 20000:
            self._finished_rooms.popitem(last=False)

    def _room_closed(self, room_id: str) -> None:
        """
        The server closed this battle's room and there was no result.

        r1 never noticed: poke-env reports a finished battle only on "|win|" or "|tie", and the
        repair loop that was meant to find such battles (_handle_query) is never called by
        poke-env 0.15. So the Battle object, its seat in poke-env's count of running battles
        (max_concurrent_battles of them: once all are taken, the next battle's first message
        waits for a seat forever and holds its room's lock), its episode slot, its steps in the
        worker's _traj and its history in the inference actor all stayed for the life of the
        worker.

        What the learner gets does not change: r1 handed over nothing for such a battle (its
        steps sat in _traj and never left), and nothing is handed over here. The steps go the
        way r1's own CLEANUP event sends them: dropped, with the inference history and the slot.
        """
        battle = self._battles.get(room_id)
        tag = self._get_unique_tag(battle) if battle is not None else f"{self.username}_{room_id}"
        self.rooms_closed += 1
        started = self._battle_starts.get(tag)
        report_rare("room closed", f"{self.username} {room_id} at turn {getattr(battle, 'turn', '?')}, "
                    + (f"{time.time() - started:.0f} s after its first decision" if started else "before its first decision"))
        # poke-env's seat: taken in _create_battle, given back only where it sees a result.
        try:
            self._battle_count_queue.get_nowait()
            self._battle_count_queue.task_done()
        except (asyncio.QueueEmpty, ValueError):
            pass
        if self.psbot_enabled:
            try:
                self.psbot.send({"op": "end", "key": self._psbot_key(room_id)})
            except Exception:
                pass
        self._note_finished_room(room_id)
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._drop_step_columns(tag)
        # The episode slot is taken at the battle's first decision. A battle closed before that
        # holds none, has no steps and no history: there is nothing for the worker to clean up,
        # and a CLEANUP would give back a slot that was never taken.
        had_slot = self._episode_slot_state.pop(tag, None) is not None
        self.action_futures.pop(tag, None)
        self._battles.pop(room_id, None)
        self._battle_events.pop(room_id, None)
        if had_slot:
            self.event_queue.put_nowait(("CLEANUP", tag, None, None))

    async def choose_move(self, battle):
        try:
            tag = self._get_unique_tag(battle)
            if getattr(battle, 'finished', False):
                return DefaultBattleOrder()
                
            now = time.time()
            self._last_act_time[tag] = now
            self._battle_starts.setdefault(tag, now)

            if tag not in self._episode_slot_state:
                await self.ep_sem.acquire()
                if tag not in self._battle_starts:
                    self._safe_release()
                    return DefaultBattleOrder()
                self._episode_slot_state[tag] = "acquired"
            
            events = self._battle_events.pop(battle.battle_tag, [])
            obs_flat = self.assembler.assemble(battle, events=events)
            # Taken now, on this thread: the battle object moves on before the worker reads it.
            phi = self.potential(battle) if self.potential is not None else 0.0
            mode = getattr(self.cfg.learner, "mode", "ppo")
            read_idx = self._note_step(tag, battle, events) if getattr(self, "step_features", False) and mode not in IMITATION_MODES else None
            if mode in IMITATION_MODES:
                if mode == "jepa_pretraining" and random.random() < 0.10:
                    order = self.choose_random_move(battle)
                    if self.psbot_enabled:
                        self.psbot_note(battle, order)
                else:
                    order = await self._teacher_order(battle)
                action_idx = self.assembler.map_order_to_index(order, battle)
                
                # Send directly to the trajectory compiler
                self.event_queue.put_nowait(("IMITATION", tag, obs_flat, (action_idx, phi)))
                
                action_obj, kwargs = self.assembler.map_index_to_order(action_idx, battle)
                if action_obj == "DEFAULT": return DefaultBattleOrder()
                return self.create_order(action_obj, **kwargs)

            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self.action_futures[tag] = (fut, loop) 

            self.event_queue.put_nowait(("STEP", tag, obs_flat, phi))

            action_idx = await fut
            if read_idx is not None:
                self._note_action(tag, read_idx, action_idx)

            action_obj, kwargs = self.assembler.map_index_to_order(action_idx, battle)
            if action_obj == "DEFAULT": return DefaultBattleOrder()
            return self.create_order(action_obj, **kwargs)

        except Exception as e:
            logger.error(f"SyncBridgePlayer crashed for {battle.battle_tag}: {e}. Using fallback.")
            return self.choose_random_move(battle)

    async def _teacher_order(self, battle):
        """The move the imitation teacher plays: PSBot when enabled, SimpleHeuristics otherwise."""
        if self.psbot_enabled:
            order = await self.psbot_order(battle)
            if order is not None:
                return order
            # The engine had no legal answer for this request. Fall back to the old teacher.
        return super().choose_move(battle)

    # ---- the stage 2-4 per-step columns (off unless a feature is on) ----------------------------

    def _note_step(self, tag: str, battle, events: list):
        """
        Records this decision's columns, called once per STEP (the step the worker will append to
        the trajectory), before the network is asked. Returns the read tracker's decision index, or
        None. Never raises: a column that cannot be worked out is left out of the episode.
        """
        try:
            cols = self._step_cols.setdefault(tag, {})
            idx = None
            if self.reads_on:
                tr = self._read_trackers.get(tag)
                if tr is None:
                    from read_reward import ReadTracker
                    tr = self._read_trackers[tag] = ReadTracker(seat=getattr(battle, "player_role", None))
                tr.observe(events)
                idx = tr.decide(None, forced=bool(getattr(battle, "force_switch", False)),
                                opp_team=list(battle.opponent_team.keys()))
            if self.boosted_on:
                from setup_moves import largest_boost
                cols.setdefault("boosted", []).append(1 if largest_boost(getattr(battle, "opponent_active_pokemon", None)) >= 1 else 0)
            if self.set_labels_on:
                cols.setdefault("set_labels", []).append(self._set_labels(battle))
            if getattr(self, "lag_on", False):
                cols.setdefault("t_act", []).append(time.time())
            return idx
        except Exception as e:      # noqa: BLE001
            report_rare("step column failed", f"{getattr(self, 'username', '?')} {getattr(battle, 'battle_tag', '?')}: {e!r}")
            self._read_trackers.pop(tag, None)
            self._step_cols[tag] = {"broken": []}
            return None

    def _note_action(self, tag: str, idx: int, action: int) -> None:
        tr = self._read_trackers.get(tag)
        if tr is not None:
            try:
                tr.set_action(idx, int(action))
            except Exception:
                pass

    def _set_label_width(self) -> int:
        from obs_matchup import LABEL_WIDTH, LABEL_WIDTH_MOVESETS, N_SIDE
        return N_SIDE * (LABEL_WIDTH_MOVESETS if getattr(self, "set_labels_movesets", False) else LABEL_WIDTH)

    def _set_labels(self, battle) -> np.ndarray:
        """The SetHead's 18 labels for this decision (48 with the moveset block): the partner seat's own team holds the truth."""
        from obs_matchup import build_set_labels
        partner = self.partner
        pb = getattr(partner, "_battles", {}).get(battle.battle_tag) if partner is not None else None
        team = getattr(pb, "team", None) if pb is not None else None
        if not team:
            return np.zeros(self._set_label_width(), dtype=np.int32)
        if getattr(self, "set_labels_movesets", False):
            return build_set_labels(battle, team, vocab_map=getattr(self.assembler, "vocab_map", None), movesets=True).reshape(-1)
        return build_set_labels(battle, team, vocab_map=getattr(self.assembler, "vocab_map", None)).reshape(-1)

    def _finish_step_columns(self, tag: str, battle):
        """The episode's columns at its end (the read tracker reads the last turn's lines first), or None."""
        cols = self._step_cols.pop(tag, None)
        tr = self._read_trackers.pop(tag, None)
        if cols is None and tr is None:
            return None
        cols = cols or {}
        if "broken" in cols:
            return None
        try:
            if tr is not None:
                tr.observe(self._battle_events.get(battle_tag_for(battle), []))
                labels = tr.finalize()
                cols["read"] = np.asarray([d.read_tier - d.being_read_tier for d in labels], dtype=np.float32)
                cols["opp_action"] = np.asarray([d.opp_action for d in labels], dtype=np.int8)
            out = {}
            for name, values in cols.items():
                if name == "set_labels":
                    out[name] = np.stack(values, axis=0).astype(np.int32) if values else np.zeros((0, self._set_label_width()), np.int32)
                elif name == "boosted":
                    out[name] = np.asarray(values, dtype=np.int8)
                else:
                    out[name] = values
            return out or None
        except Exception as e:      # noqa: BLE001
            report_rare("step columns failed at the end", f"{getattr(self, 'username', '?')} {tag}: {e!r}")
            return None

    def _drop_step_columns(self, tag: str) -> None:
        if getattr(self, "step_features", False):
            self._read_trackers.pop(tag, None)
            self._step_cols.pop(tag, None)

    def _battle_finished_callback(self, battle):
        tag = self._get_unique_tag(battle)
        cols = self._finish_step_columns(tag, battle) if getattr(self, "step_features", False) else None
        self.event_queue.put_nowait(("DONE", tag, cols, battle))
        self._finished_rooms[battle_tag_for(battle)] = None
        while len(self._finished_rooms) > 20000:
            self._finished_rooms.popitem(last=False)
        
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._episode_slot_state.pop(tag, None)
        
        self.action_futures.pop(tag, None)
        base_tag = tag.split("_", 1)[1] if "_" in tag else tag
        if hasattr(self, "_battles"):
            self._battles.pop(base_tag, None)
        self._battle_events.pop(base_tag, None)
        
        super()._battle_finished_callback(battle)

    def _cleanup_local_battle(self, tag: str):
        self._battle_starts.pop(tag, None)
        self._last_act_time.pop(tag, None)
        self._episode_slot_state.pop(tag, None)
        self._drop_step_columns(tag)
        
        self.action_futures.pop(tag, None) 
        
        self.event_queue.put_nowait(("CLEANUP", tag, None, None))
        
        base_tag = tag.split("_", 1)[1] if "_" in tag else tag
        if hasattr(self, "_battles"):
            self._battles.pop(base_tag, None)
        self._battle_events.pop(base_tag, None)

    def _safe_release(self):
        try: self.ep_sem.release()
        except ValueError: pass

    async def run_reconciliation(self):
        try: await self.ps_client.send_message("/rlactive", room="")
        except Exception: pass

    def _handle_query(self, query_type: str, data: Any) -> None:
        super()._handle_query(query_type, data)
        if query_type == "rlactive":
            if not data or isinstance(data, list):
                server_ids = set()
            else:
                server_ids = set(str(data).split(","))
            
            server_ids.discard("")
            local_map = {tag: (tag.split("_", 1)[1] if "_" in tag else tag) for tag in self._battle_starts.keys()}
            local_base_ids = set(local_map.values())
            
            ghosts = list(server_ids - local_base_ids)
            for bid in ghosts:
                asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))
            
            now = time.time()
            stalls = [tag for tag, bid in local_map.items() if bid in server_ids and (now - self._last_act_time.get(tag, 0)) > 60.0]
            for tag in stalls:
                bid = local_map[tag]
                self._cleanup_local_battle(tag)
                asyncio.create_task(self.ps_client.send_message(f"/rlrescue {bid}", room=""))
                asyncio.create_task(self.ps_client.send_message(f"/join {bid}", room=""))
                
            zombies = [tag for tag, bid in local_map.items() if bid not in server_ids and (now - self._battle_starts.get(tag, 0)) > ZOMBIE_LIMIT_S]
            for tag in zombies:
                self._cleanup_local_battle(tag)

# ---------------------------------------------------------------------------
# VECTOR WRAPPER (THE BRIDGE)
# ---------------------------------------------------------------------------

class PokeEnvVectorWrapper:
    def __init__(self, cfg: RunConfig, pairs_count: int, server_port: int):
        self.cfg = cfg
        self.pairs_count = pairs_count
        self.server_conf = make_server_conf("127.0.0.1", server_port)
        self.run_tag = secrets.token_hex(3)
        
        self.event_queue = queue.Queue()
        self.action_futures = {} 
        self.active_pairs = []

        self.async_loop = asyncio.new_event_loop()
        self.bg_thread = threading.Thread(target=self._run_bg_loop, daemon=True)
        self.bg_thread.start()

    def _run_bg_loop(self):
        asyncio.set_event_loop(self.async_loop)
        self.ep_sem = asyncio.Semaphore(self.cfg.rollout.learn_max_pending_episodes)
        self.async_loop.run_until_complete(self._init_players_and_spawn())
        self.async_loop.run_forever()

    async def _init_players_and_spawn(self):
        for i in range(self.pairs_count):
            pA = SyncBridgePlayer(
                cfg=self.cfg,
                event_queue=self.event_queue, action_futures=self.action_futures, ep_sem=self.ep_sem,
                account_configuration=AccountConfiguration(mk_name(self.run_tag, i, "a"), None),
                server_configuration=self.server_conf, max_concurrent_battles=self.cfg.rollout.rooms_per_pair,
                battle_format=self.cfg.env.battle_format, start_listening=True, log_level=40
            )
            pB = SyncBridgePlayer(
                cfg=self.cfg,
                event_queue=self.event_queue, action_futures=self.action_futures, ep_sem=self.ep_sem,
                account_configuration=AccountConfiguration(mk_name(self.run_tag, i, "b"), None),
                server_configuration=self.server_conf, max_concurrent_battles=self.cfg.rollout.rooms_per_pair,
                battle_format=self.cfg.env.battle_format, start_listening=True, log_level=40
            )
            pA.partner, pB.partner = pB, pA
            self.active_pairs.append((pA, pB))

        players = [p for pair in self.active_pairs for p in pair]
        
        async def wait_for_login_retry(player, attempts=5, delay=1.0):
            last_err = None
            for i in range(attempts):
                try:
                    await wait_for_login(player)
                    return
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(delay)
            raise last_err

        for p in players:
            await wait_for_login_retry(p)
            await asyncio.sleep(0.05)
            
        await asyncio.gather(*[join_lobby(p) for p in players])
        
        for (pA, pB) in self.active_pairs:
            await safe_send(pA, f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {self.cfg.rollout.rooms_per_pair}", room="lobby")

        self.async_loop.create_task(self._maintenance_loop())

    async def _maintenance_loop(self):
        loop_idx = 0
        while True:
            await asyncio.sleep(REPAIR_INTERVAL_S)
            loop_idx += 1
            for pA, pB in self.active_pairs:
                asyncio.create_task(pA.run_reconciliation())
                asyncio.create_task(pB.run_reconciliation())
                
            if loop_idx % 6 == 0:
                for pA, pB in self.active_pairs:
                    asyncio.create_task(
                        safe_send(pA, f"/rlautospawn {pA.username}, {pB.username}, {self.cfg.env.battle_format}, {self.cfg.rollout.rooms_per_pair}", room="lobby")
                    )

    def release_slot(self):
        def _safe_release():
            try: self.ep_sem.release()
            except ValueError: pass
        self.async_loop.call_soon_threadsafe(_safe_release)

    def step(self, actions_dict: dict) -> List[Tuple]:
        for tag, action_idx in actions_dict.items():
            if tag in self.action_futures:
                fut, loop = self.action_futures.pop(tag)
                if not fut.done():
                    loop.call_soon_threadsafe(fut.set_result, action_idx)

        events = [self.event_queue.get()]
        time.sleep(0.003)
        
        while True:
            try: events.append(self.event_queue.get_nowait())
            except queue.Empty: break

        return events

# ---------------------------------------------------------------------------
# CPU PRIORITY OF A ROLLOUT WORKER (train.py --rollout-nice; off unless asked)
# ---------------------------------------------------------------------------
#
# With --rollout-nice N, train.py starts Ray's workers at nice 0 instead of Ray's 15, so the
# learner and the inference actors keep the CPU they need, and has every rollout worker put its
# own threads at N (RolloutWorker.set_nice), the nice the Showdown servers are started at
# (servers.command under `nice -n N`). On Linux the nice value belongs to each thread, and a new
# thread takes the nice of the thread that starts it, so after one walk every thread the worker
# starts later is at N as well; heartbeat() walks again for a thread started by one that was not
# yet raised. Raising one's own nice needs no privilege. Lowering it does (ulimit -e is 0 on the
# boxes), so nothing here ever lowers a thread.
#
# The processes a worker starts are walked too (raise_children_nice). A child process takes the
# nice of the thread that starts it, and with the flag Ray starts the worker at 0, so a helper
# started in the worker's constructor, before set_nice can reach it, starts at 0, the learner's
# nice: Foul Play's search helpers (foulplay_bridge, own session) and PSBot's node engine
# (psbot_bridge.shared_engine) for opponent seats of those kinds. Every thread of every process
# under the worker is raised to N as well, at set_nice and at every heartbeat.

# Texts of threads that could not be raised: at most this many per walk, and logged at most this
# many times per worker.
NICE_PROBLEMS_KEPT: Final[int] = 3


def raise_thread_nice(n: int, pid: Optional[int] = None) -> Dict[str, Any]:
    """
    Every thread of process `pid` (None: this process) whose nice is below n is put at n
    (os.setpriority on its thread id). A thread at n or above is left as it is. Never raises.

    -> {"threads": found, "raised": put at n by this call, "failed": could not be, "errors": the
    first NICE_PROBLEMS_KEPT texts}. A thread that ends during the walk is not counted. For another
    process that has ended (its /proc entry is gone), nothing is counted and "gone" is True.
    """
    out: Dict[str, Any] = {"threads": 0, "raised": 0, "failed": 0, "errors": []}
    where = "/proc/self/task" if pid is None else f"/proc/{int(pid)}/task"
    try:
        tids = sorted(int(t) for t in os.listdir(where))
    except Exception as e:      # noqa: BLE001  (no /proc: not Linux; or the other process has ended)
        if pid is not None and isinstance(e, (FileNotFoundError, ProcessLookupError)):
            out["gone"] = True
            return out
        out["failed"] = 1
        out["errors"].append(f"cannot list {where}: {e!r}")
        return out
    for tid in tids:
        try:
            current = os.getpriority(os.PRIO_PROCESS, tid)
            out["threads"] += 1
            if current >= n:
                continue
            os.setpriority(os.PRIO_PROCESS, tid, n)
            out["raised"] += 1
        except ProcessLookupError:
            continue                                     # the thread ended during the walk
        except Exception as e:  # noqa: BLE001
            out["failed"] += 1
            if len(out["errors"]) < NICE_PROBLEMS_KEPT:
                out["errors"].append(f"thread {tid}: {e!r}" if pid is None else f"pid {pid} thread {tid}: {e!r}")
    return out


# /proc/<pid>/task/<tid>/children lists the processes a thread started (CONFIG_PROC_CHILDREN; on in
# the boxes' kernels). None until first needed; without it the parents are read from every
# /proc/<pid>/stat instead.
_CHILDREN_FILE: Optional[bool] = None


def _has_children_file() -> bool:
    global _CHILDREN_FILE
    if _CHILDREN_FILE is None:
        _CHILDREN_FILE = os.path.exists(f"/proc/self/task/{threading.get_native_id()}/children")
    return _CHILDREN_FILE


def _children_by_file(pid: int) -> List[int]:
    """The processes started by any thread of `pid`. FileNotFoundError when `pid` has ended."""
    kids: List[int] = []
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            with open(f"/proc/{pid}/task/{tid}/children") as f:
                kids.extend(int(x) for x in f.read().split())
        except (FileNotFoundError, ProcessLookupError):
            continue                                     # the thread ended
    return kids


def _ppid_of_stat(text: str) -> int:
    """The parent pid in a /proc/<pid>/stat line: "pid (comm) state ppid ...". comm may hold spaces and ')'."""
    return int(text[text.rindex(")") + 1:].split()[1])


def _children_by_stat() -> Dict[int, List[int]]:
    """parent pid -> its child pids, from every /proc/<pid>/stat (without the children files)."""
    kids: Dict[int, List[int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                ppid = _ppid_of_stat(f.read())
        except (OSError, ValueError, IndexError):
            continue                                     # ended, or unreadable
        kids.setdefault(ppid, []).append(int(name))
    return kids


def descendant_pids(root: Optional[int] = None, visit=None) -> List[int]:
    """
    Every process under `root` (None: this process), children before grandchildren, `root` itself
    not in it. `visit(pid)`, when given, is called on each before its own children are listed, so a
    process it starts after the visit takes what the visit did. Raises OSError when /proc cannot be
    read at all (not Linux).
    """
    root = os.getpid() if root is None else int(root)
    if _has_children_file():
        step = _children_by_file
    else:
        table = _children_by_stat()
        step = lambda p: table.get(p, [])               # noqa: E731
    seen = {root}
    order: List[int] = []
    todo = [root]
    while todo:
        p = todo.pop(0)
        if p != root and visit is not None:
            visit(p)
        try:
            kids = step(p)
        except (FileNotFoundError, ProcessLookupError):
            if p == root:
                raise
            continue                                     # it ended
        for k in kids:
            if k not in seen:
                seen.add(k)
                order.append(k)
                todo.append(k)
    return order


def raise_children_nice(n: int) -> Dict[str, Any]:
    """
    Every thread of every process under this one (descendant_pids) raised to n, as raise_thread_nice:
    a worker's helpers (Foul Play's, PSBot's engine). Each process is raised before its own children
    are listed. Never raises.

    -> {"children": processes walked, "threads", "raised", "failed", "errors"} over all of them.
    """
    out: Dict[str, Any] = {"children": 0, "threads": 0, "raised": 0, "failed": 0, "errors": []}

    def visit(pid: int) -> None:
        c = raise_thread_nice(n, pid)
        if c.get("gone"):
            return
        out["children"] += 1
        for k in ("threads", "raised", "failed"):
            out[k] += c[k]
        out["errors"].extend(c["errors"][:max(0, NICE_PROBLEMS_KEPT - len(out["errors"]))])

    try:
        descendant_pids(visit=visit)
    except Exception as e:      # noqa: BLE001
        out["failed"] += 1
        if len(out["errors"]) < NICE_PROBLEMS_KEPT:
            out["errors"].append(f"cannot list the child processes: {e!r}")
    return out


# ---------------------------------------------------------------------------
# ROLLOUT WORKER (RAY ACTOR) - PURE SYNC MAIN THREAD
# ---------------------------------------------------------------------------

class RolloutWorker:
    # train.py --rollout-nice: the nice this worker keeps its threads at; None until set_nice is called.
    _rollout_nice: Optional[int] = None

    def __init__(self, cfg: RunConfig, inference_actor, learner_actor, pairs: int, server_port: int,
                 learn_queue: Optional[int] = None, lag_stats: bool = False):
        """
        learn_queue, lag_stats: train.py --learn-queue-episodes and --lag-stats (league.py). train.py passes
        them only when they are given; at their defaults the worker is what it always was.
        """
        self.cfg = cfg
        self._lag_stats = bool(lag_stats)
        if self._lag_stats:
            global LAG_STATS
            LAG_STATS = True                     # read by this process's players, made below
        if learn_queue is None:
            self.learner_client = SyncLearnerClient(learner_actor, cfg)
        else:
            self.learner_client = SyncLearnerClient(learner_actor, cfg, maxsize=learn_queue)
        
        # --- NEW: Store the Central Inference Actor ---
        self.inference_actor = inference_actor
        
        self.vec_env = PokeEnvVectorWrapper(cfg, pairs, server_port)
        self._traj = {}
        self.assembler = cfg.make_assembler()
        # Memory diagnostics, only when PSPPO_MEMWATCH names a directory (see memwatch.py).
        import memwatch
        self._memwatch = memwatch.start_if_asked(self, server_port)

    async def run(self):
        threading.Thread(target=self._run_rl_loop, daemon=True).start()
        while True:
            await asyncio.sleep(1.0)

    def _run_rl_loop(self):
        actions_dict = {}
        
        while True:
            events = self.vec_env.step(actions_dict)
            actions_dict = {}
            
            step_tags = []
            step_obs = []
            step_phi = []
            
            for event_type, tag, payload, extra_data in events:
                if event_type == "STEP":
                    step_tags.append(tag)
                    step_phi.append(float(extra_data or 0.0))   # extra is the potential at this decision
                    step_obs.append(payload) 
                elif event_type == "IMITATION":
                    traj = self._traj.setdefault(tag, {"obs": [], "act": [], "logp": [], "val": [], "var": [], "phi": []})
                    action_idx, phi = extra_data                 # extra is (action_idx, potential)
                    traj["obs"].append(payload)    # payload is obs_flat
                    traj["act"].append(action_idx)
                    traj["logp"].append(0.0)
                    traj["val"].append(0.0)
                    traj["var"].append(0.0)
                    traj["phi"].append(float(phi))
                elif event_type == "DONE":
                    if payload is not None and tag in self._traj:
                        # The stage 2-4 columns the player recorded for this battle (SyncBridgePlayer._note_step).
                        self._traj[tag]["extras"] = payload
                    self._finalize_trajectory(tag, extra_data)
                elif event_type == "CLEANUP":
                    self._cleanup_trajectory(tag)

            if step_obs:
                # The observations go out as the float16 the assembler made them. Widening
                # them to float32 here cost 0.7 ms a call on the interpreter lock this
                # worker's parsing thread needs, and doubled the bytes. The inference actor
                # widens them on the GPU, to the same float32 numbers.
                obs_batch = np.stack(step_obs, axis=0)

                # --- FIRE TO CENTRAL INFERENCE GPU ---
                # This blocks the worker thread until Ray returns the answers.
                # Because obs_batch is a contiguous array, Ray zero-copies it!
                try:
                    acts, logps, vals, var = ray.get(
                        self.inference_actor.infer_batch.remote(step_tags, obs_batch)
                    )
                except Exception as e:
                    logger.error(f"Inference Actor call failed: {e}")
                    # Fallback if InferenceActor crashes (avoids complete deadlock)
                    N = len(step_obs)
                    acts = np.zeros(N, dtype=np.int64)
                    logps = np.zeros(N, dtype=np.float32)
                    vals = np.zeros(N, dtype=np.float32)
                    var = np.zeros(N, dtype=np.float32)
                
                for i, tag in enumerate(step_tags):
                    actions_dict[tag] = int(acts[i])
                    
                    traj = self._traj.setdefault(tag, {"obs": [], "act": [], "logp": [], "val": [], "var": [], "phi": []})
                    traj["obs"].append(step_obs[i])
                    traj["act"].append(acts[i])
                    traj["logp"].append(logps[i])
                    traj["val"].append(vals[i])
                    traj["var"].append(var[i])
                    traj["phi"].append(step_phi[i])

    def _cleanup_trajectory(self, tag: str):
        """Called when a Zombie/Stall is forcibly killed by the async maintenance thread."""
        self.inference_actor.clear_cache.remote(tag) # Fire and forget
        self._traj.pop(tag, None)
        self.vec_env.release_slot()

    def _finalize_trajectory(self, tag: str, battle):
        """Calculates rewards and sends to learner."""
        self.inference_actor.clear_cache.remote(tag) # Fire and forget
        
        if tag in self._traj: 
            buf = self._traj.pop(tag)
            if buf["act"]: 
                T = len(buf["act"])
                obs_stacked = np.stack(buf["obs"], axis=0)
                
                rc = self.cfg.reward
                # A tie or a game with no result is neither a win nor a loss.
                if battle.won:
                    terminal_reward = rc.terminal_win
                elif getattr(battle, "lost", False):
                    terminal_reward = rc.terminal_loss
                else:
                    terminal_reward = getattr(rc, "terminal_tie", 0.0)
                rewards = np.zeros(T, dtype=np.float32)
                shaping = getattr(rc, "shaping", "faint" if rc.use_faint_reward else "none")
                
                if shaping == "potential" and len(buf.get("phi", [])) == T:
                    # Win/loss at the end plus the discounted change in position value on every
                    # step. The learner's own gamma is used so the shaping terms cancel exactly.
                    rewards = shaped_rewards(buf["phi"], 0.0, self.cfg.learner.gamma, rc.shaping_scale)
                elif shaping == "faint":
                    b_start, b_end = self.assembler.offsets["pokemon_body"]
                    faint_idx = self.assembler.meta["faint_internal_idx"]
                    
                    body_history = obs_stacked[:, b_start:b_end].reshape(T, 12, self.assembler.meta["dim_pokemon_body"])
                    is_fainted = body_history[:, :, faint_idx] > 0.5
                    
                    ds = np.diff(is_fainted[:, :6].sum(axis=1))
                    do = np.diff(is_fainted[:, 6:].sum(axis=1))

                    # Apply to rewards[:-1] to align the consequence with the action
                    rewards[:-1] += (np.maximum(0, ds) * float(self.cfg.reward.faint_self)) + \
                                    (np.maximum(0, do) * float(self.cfg.reward.faint_opp))

                rewards[-1] += terminal_reward
                dones = np.zeros(T, dtype=np.float32)
                dones[-1] = 1.0

                extras = self._checked_extras(tag, battle, buf, T)
                if extras is None:
                    self.learner_client.submit_episode(
                        obs_stacked, np.array(buf["act"]), np.array(buf["logp"]),
                        np.array(buf["val"]), np.array(buf["var"]), rewards, dones
                    )
                else:
                    self.learner_client.submit_episode(
                        obs_stacked, np.array(buf["act"]), np.array(buf["logp"]),
                        np.array(buf["val"]), np.array(buf["var"]), rewards, dones, extras=extras
                    )

        self.vec_env.release_slot()

    def episode_extras(self, tag: str, battle, buf: dict, T: int):
        """
        Named per-step columns for this episode: a dict of name -> array of T rows (integers,
        bools or floats), or None. They reach the learner beside the eight fixed columns and
        stay aligned with their steps into ppo_update's minibatches (step_columns.py). None here:
        a later stage overrides this (or fills buf["extras"], a dict of name -> list with one
        entry per step, which is returned as it is).
        """
        cols = buf.get("extras")
        return {name: np.asarray(values) for name, values in cols.items()} if cols else None

    def _checked_extras(self, tag: str, battle, buf: dict, T: int):
        """episode_extras, checked (step_columns.check_columns). A column that cannot be used is left
        out and counted in extra_problems; an exception is counted too. Never raises."""
        try:
            cols = self.episode_extras(tag, battle, buf, T)
            if cols is None:
                return None
            from step_columns import check_columns

            cols, problems = check_columns(cols, T)
        except Exception as e:      # noqa: BLE001
            cols, problems = None, [f"episode_extras raised {type(e).__name__}"]
        if problems:
            counts = self.__dict__.setdefault("extra_problems", Counter())
            for text in problems:
                key = text.split(":")[0][:80]
                counts[key] += 1
                if counts[key] <= 3:
                    logger.warning(f"{text} (left out)")
        return cols

    def heartbeat(self):
        """Called by Train.py telemetry."""
        sem_val = getattr(self.vec_env, "ep_sem", None)
        beat = {
            "active_battles_worker": sum(len(p._battles) for pA, pB in self.vec_env.active_pairs for p in (pA, pB) if hasattr(p, "_battles")),
            "learner_q_size": self.learner_client.q.qsize(),
            "traj_in_memory": len(self._traj),
            "ep_sem_value": sem_val._value if sem_val else -1
        }
        if self._rollout_nice is not None:
            # --rollout-nice: threads started since the last walk go to this worker's nice too.
            self._nice_walk()
            beat["nice"] = dict(self._nice_totals, nice=self._rollout_nice)
        if getattr(self, "_lag_stats", False):
            # --lag-stats: what this worker's queue to the learner held and dropped since it started.
            lc = self.learner_client
            beat["lag"] = {"submitted": int(lc.submitted), "evicted": int(lc.evicted), "q": int(lc.q.qsize()),
                           "maxsize": int(lc.q.maxsize)}
        return beat

    def set_nice(self, n: int) -> Dict[str, Any]:
        """
        train.py --rollout-nice N: every thread of this worker at nice N (raise_thread_nice), and
        every thread of every process it has started (raise_children_nice: the opponent seats'
        helpers, which the constructor starts at Ray's nice 0); through heartbeat() also every
        thread and process started later. Idempotent; it never lowers a thread and never raises
        (a thread that cannot be raised is logged and counted).

        -> {"pid", "nice", "threads", "raised", "failed", "children", "child_threads", "child_raised"}
        of this call's walk: threads and raised are this worker's own, children the processes under
        it and child_threads / child_raised their threads, failed both together.
        """
        self._rollout_nice = int(n)
        if getattr(self, "_nice_totals", None) is None:
            # raised_later: threads the heartbeats found below the nice (started by a thread not yet
            # raised), this worker's and its child processes'. children / child_threads: the last walk's.
            self._nice_totals = {"threads": 0, "raised_later": 0, "failed": 0, "walks": 0,
                                 "children": 0, "child_threads": 0}
            self._nice_logged = 0
        counts = self._nice_walk(first=True)
        return {"pid": os.getpid(), "nice": self._rollout_nice, "threads": counts["threads"],
                "raised": counts["raised"], "failed": counts["failed"], "children": counts["children"],
                "child_threads": counts["child_threads"], "child_raised": counts["child_raised"]}

    def _nice_walk(self, first: bool = False) -> Dict[str, Any]:
        """
        One walk at this worker's nice: its own threads (raise_thread_nice), then the processes under
        it (raise_children_nice), added to the totals the heartbeat reports.
        """
        try:
            counts = raise_thread_nice(self._rollout_nice)
        except Exception as e:  # noqa: BLE001  (raise_thread_nice does not raise; the last guard)
            counts = {"threads": 0, "raised": 0, "failed": 1, "errors": [repr(e)]}
        try:
            kids = raise_children_nice(self._rollout_nice)
        except Exception as e:  # noqa: BLE001  (raise_children_nice does not raise; the last guard)
            kids = {"children": 0, "threads": 0, "raised": 0, "failed": 1, "errors": [repr(e)]}
        counts = {"threads": counts["threads"], "raised": counts["raised"],
                  "failed": counts["failed"] + kids["failed"],
                  "children": kids["children"], "child_threads": kids["threads"], "child_raised": kids["raised"],
                  "errors": (counts["errors"] + kids["errors"])[:2 * NICE_PROBLEMS_KEPT]}
        t = self._nice_totals
        t["threads"] = counts["threads"]
        t["children"] = counts["children"]
        t["child_threads"] = counts["child_threads"]
        t["walks"] += 1
        t["failed"] += counts["failed"]
        if not first:
            t["raised_later"] += counts["raised"] + counts["child_raised"]
        if counts["errors"] and self._nice_logged < NICE_PROBLEMS_KEPT:
            self._nice_logged += 1
            logger.warning(f"--rollout-nice {self._rollout_nice}: {counts['failed']} thread(s) of pid {os.getpid()} "
                           f"or its child processes not raised: {'; '.join(counts['errors'])}")
        return counts