"""
train.py --fast-ingest: the learner's work around its updates, made cheaper. Off by default.

Three parts, each giving the same numbers as the code it stands in for (tests/test_fast_ingest.py,
bench/fast_ingest_bench.py):

  GpuObsBuffer / DeviceEpisodeDataset
      Each message's observations are copied to the learner's GPU as the message comes in, into one
      buffer that is kept from update to update. A thread of the buffer's own makes the copies, on
      the learner's CUDA stream, so the learner's thread only hands them over and goes on to the
      next message. At the update the learner waits for the last copy and takes a view of the
      buffer's first n rows. Without it the update joined about 200 CPU chunks into one 0.6 GB
      tensor (torch.cat, 0.59 s on box a's crowded memory) and then copied all of it to the GPU a
      second time. Every other column (actions, log-probabilities, values, advantages, returns,
      episode ids, the named extras) stays in the CPU dataset exactly as before, so the advantage
      normalisation, the read bonus and the statistics run on the CPU as before. The minibatches
      index the same rows in the same order. The CPU dataset also keeps a reference to every
      message's observations (no copy), so an update whose buffer failed joins them as before.

  gae_batch / PendingAdvantages
      ppo_core.gae_from_episode's float32 arithmetic for many episodes at once, not one episode at
      a time: the same operations on the same numbers, so the same bits. The backward recurrence
      runs over a [max_len x episodes] float32 array with every episode's last step in the last
      row, one numpy multiply and one add per row. The learner hands each message's columns to
      PendingAdvantages as the message comes in, with the two empty tensors the data set holds for
      that message's advantages and returns, and the update fills them all with one gae_batch over
      the update's episodes before it reads them (5 to 25 ms on box c where one message at a time
      took 0.12 to 0.26 s per update). A message gae_batch does not take (another dtype, lengths that
      do not add up) gets the per-episode loop as it comes in, as without the flag.

  WeightPusher
      After an update the weights go to the inference actors through the weight store. Before, the
      update's thread copied them to the CPU one tensor at a time (a wait on the GPU for each) and
      serialised them for Ray itself. Now one batched copy goes into pinned CPU buffers and a single
      background thread waits for it, hands Ray a dict of CPU tensors equal to the old one and calls
      weight_store.update and set_temp, in update order. At most one push is in flight. The tensors
      go to Ray as OutOfBandTensor: their bytes out of band, which Ray copies without the Python lock,
      so the push no longer holds the lock for Ray's pickling of all 32 MB (about 60 ms on box c)
      while the learner's thread takes in the next update's messages. The store unpickles the same
      dict as before, and faster.

Nothing here changes a loss, a gradient, a checkpoint or the order of anything an update reads.
The one timing change: the inference actors see each new set of weights a little earlier or later
than before (well inside their 1 s sync interval).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ppo_core import AsyncEpisodeDataset, gae_from_episode

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------
# Advantages for many episodes at once
# ------------------------------------------------------------------------------------------------

def _plain_lengths(lengths, n_steps: int) -> Optional[np.ndarray]:
    """The episode lengths as int64, or None when they are not plain (not integers, negative, none, or not summing to n_steps)."""
    try:
        L = np.asarray(lengths)
    except Exception:       # noqa: BLE001
        return None
    if L.ndim != 1 or L.size == 0 or not np.issubdtype(L.dtype, np.integer):
        return None
    L = L.astype(np.int64)
    if int(L.min()) < 0 or int(L.sum()) != int(n_steps):
        return None
    return L


def plain_message(rewards, values, dones, lengths, variances=None) -> Optional[np.ndarray]:
    """
    The episode lengths (int64) when gae_batch takes these columns: float32 CPU tensors of one
    dimension and one length, no gradient, integer lengths that add up to it. None otherwise.
    """
    ts = [rewards, values, dones] + ([variances] if variances is not None else [])
    if not all(torch.is_tensor(t) and t.dtype == torch.float32 and t.device.type == "cpu" and t.ndim == 1
               and not t.requires_grad for t in ts):
        return None
    S = int(rewards.shape[0])
    if any(int(t.shape[0]) != S for t in ts):
        return None
    return _plain_lengths(lengths, S)


def _backward(d: np.ndarray, coef: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    acc = d[t] + coef[t] * acc from each episode's last step back to its first, acc starting at 0.0:
    gae_from_episode's loop, as float32 multiplies and adds on whole rows. Episodes are columns of a
    [max_len x episodes] array with every episode's last step in the last row, where acc is 0.0 as
    the loop starts it. The rows above an episode's first step (zeros) come after its steps in the
    backward pass and are never read.
    """
    S = int(d.shape[0])
    out = np.empty(S, dtype=np.float32)
    if S == 0:
        return out
    nz = L > 0
    Ls = L[nz]
    starts = (np.cumsum(L) - L)[nz]
    E, T = int(Ls.size), int(Ls.max())
    col = np.repeat(np.arange(E, dtype=np.int64), Ls)
    row = (T - np.repeat(Ls, Ls)) + (np.arange(S, dtype=np.int64) - np.repeat(starts, Ls))
    D = np.zeros((T, E), dtype=np.float32)
    C = np.zeros((T, E), dtype=np.float32)
    D[row, col] = d
    C[row, col] = coef
    OUT = np.empty((T, E), dtype=np.float32)
    tmp = np.empty(E, dtype=np.float32)
    acc = np.zeros(E, dtype=np.float32)
    for j in range(T - 1, -1, -1):
        np.multiply(C[j], acc, out=tmp)         # coef[t] * acc
        acc = OUT[j]
        np.add(D[j], tmp, out=acc)              # d[t] + (that)
    out[:] = OUT[row, col]
    return out


@torch.no_grad()
def gae_batch(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, lengths, gamma: float, lam: float,
              variances: Optional[torch.Tensor] = None, lam_min: float = 0.55, lam_max: float = 0.95,
              lam_eps: float = 1e-8, last_value: float = 0.0) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    The advantages and returns of a message's episodes, laid one after the other as the learner gets
    them: (adv, ret), float32 [S], bit for bit what torch.cat of ppo_core.gae_from_episode's results
    for each episode gives (the learner calls it with last_value 0.0, the default here too).

    rewards, values, dones, variances: float32 CPU tensors [S]; variances None for the static lambda.
    lengths: the episodes' lengths, integers summing to S.

    None when the input is anything else (another dtype or device, lengths that do not add up, no
    episodes): the caller then takes the per-episode loop, which handles it as it always has.
    """
    L = plain_message(rewards, values, dones, lengths, variances)
    if L is None:
        return None
    S = int(rewards.shape[0])

    # The value after each step: the next step's, and last_value after an episode's last step. As in
    # gae_from_episode: torch.cat([values[1:], torch.tensor([last_value])]), the float32 last_value.
    nv = torch.empty_like(values)
    if S > 1:
        nv[:-1] = values[1:]
    ends = np.cumsum(L)[L > 0] - 1
    nv[torch.from_numpy(ends)] = torch.tensor([last_value], dtype=torch.float32).expand(int(ends.size))

    if variances is not None:
        # gae_from_episode, branch A: the same torch expressions on the whole message.
        next_nonterminal = 1.0 - dones
        deltas = rewards + gamma * nv * next_nonterminal - values
        delta_sq = deltas ** 2
        dynamic_lambda = delta_sq / (delta_sq + variances + lam_eps)
        dynamic_lambda = torch.clamp(dynamic_lambda, min=lam_min, max=lam_max)
        d = deltas.numpy()
        coef = (np.float32(gamma) * dynamic_lambda.numpy()) * next_nonterminal.numpy()
    else:
        # Branch B: its NumPy float32 scalars, as arrays.
        r, v, dn = rewards.numpy(), values.numpy(), dones.numpy()
        g = np.float32(gamma)
        gl = np.float32(gamma * lam)
        one = np.float32(1.0)
        nonterminal = one - dn
        d = r + g * nv.numpy() * nonterminal - v
        coef = gl * nonterminal

    adv = torch.from_numpy(_backward(np.ascontiguousarray(d, dtype=np.float32),
                                     np.ascontiguousarray(coef, dtype=np.float32), L))
    ret = adv + values
    return adv, ret


def gae_per_episode(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, variances: Optional[torch.Tensor],
                    lengths, cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    """The learner's own per-episode loop (learner.LearnerActor._loop), for the messages gae_batch does not take."""
    adv_chunks, ret_chunks = [], []
    curr = 0
    for length in lengths.tolist():
        end = curr + int(length)
        adv, ret = gae_from_episode(
            rewards=rewards[curr:end], values=values[curr:end], dones=dones[curr:end],
            gamma=cfg.gamma, lam=cfg.gae_lambda,
            variances=variances[curr:end] if variances is not None else None,
            lam_min=cfg.dynamic_lambda_min, lam_max=cfg.dynamic_lambda_max, lam_eps=cfg.dynamic_lambda_eps)
        adv_chunks.append(adv)
        ret_chunks.append(ret)
        curr = end
    return torch.cat(adv_chunks), torch.cat(ret_chunks)


def episode_ids(lengths, first: int) -> torch.Tensor:
    """first, first+1, ... repeated over each episode's steps (int64): the learner's torch.full chunks, joined."""
    L = np.asarray(lengths).astype(np.int64).reshape(-1)
    return torch.from_numpy(np.repeat(np.arange(int(first), int(first) + int(L.size), dtype=np.int64), L))


def gae_settings(cfg, dynamic: bool) -> Tuple:
    """What the learner passes gae_from_episode, besides the columns (LearnerConfig fields)."""
    return (float(cfg.gamma), float(cfg.gae_lambda), bool(dynamic), float(cfg.dynamic_lambda_min),
            float(cfg.dynamic_lambda_max), float(cfg.dynamic_lambda_eps))


class _Settings:
    """gae_per_episode's view of a gae_settings tuple (the LearnerConfig fields it reads)."""

    def __init__(self, s: Tuple):
        self.gamma, self.gae_lambda, _dyn, self.dynamic_lambda_min, self.dynamic_lambda_max, self.dynamic_lambda_eps = s


class PendingAdvantages:
    """
    An update's advantages and returns, computed for all its messages at once when the update takes
    its data, into the tensors the data set already holds for them.

    add() takes one message's columns, its episode lengths (plain_message's) and two empty float32
    tensors of its length, which the caller has put in the data set as the message's advantages and
    returns. flush() fills every one of them: one gae_batch over all the messages' episodes laid one
    after the other (one per run of messages with the same settings, which is one run in a run that
    does not change them). The numbers are gae_from_episode's for each episode, bit for bit, since
    gae_batch is that arithmetic and nothing in it mixes episodes. Should the batch fail for any
    reason, each message is computed by the per-episode loop instead (as without the flag) and the
    failure is counted and logged.
    """

    def __init__(self):
        self.items: List[Tuple] = []
        self.rows = 0
        self.flush_s = 0.0          # time flush() took (on the learner's thread, inside the update)
        self.flushes = 0
        self.fallbacks = 0
        self.last_error = ""

    def __len__(self) -> int:
        return len(self.items)

    def add(self, settings: Tuple, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
            variances: Optional[torch.Tensor], lengths: np.ndarray, adv_out: torch.Tensor, ret_out: torch.Tensor) -> None:
        self.items.append((settings, rewards, values, dones, variances, lengths, adv_out, ret_out))
        self.rows += int(rewards.shape[0])

    def clear(self) -> None:
        self.items = []
        self.rows = 0

    @staticmethod
    def _one_by_one(group) -> None:
        for settings, r, v, d, var, L, adv_out, ret_out in group:
            adv, ret = gae_per_episode(r, v, d, var, L, _Settings(settings))
            adv_out.copy_(adv)
            ret_out.copy_(ret)

    def flush(self) -> None:
        """Fills every pending message's advantages and returns. Afterwards nothing is pending."""
        if not self.items:
            return
        t0 = time.perf_counter()
        items, self.items, self.rows = self.items, [], 0
        try:
            # runs of consecutive messages with the same settings (the learner's never change)
            groups: List[List[Tuple]] = []
            for it in items:
                if groups and groups[-1][0][0] == it[0] and (groups[-1][0][4] is None) == (it[4] is None):
                    groups[-1].append(it)
                else:
                    groups.append([it])
            for group in groups:
                try:
                    self._batch(group)
                except Exception as e:      # noqa: BLE001
                    self.fallbacks += 1
                    self.last_error = f"{type(e).__name__}: {e}"
                    if self.fallbacks <= 5 or self.fallbacks % 100 == 0:
                        logger.warning(f"advantages of {len(group)} messages one by one ({self.last_error})")
                        print(f"[learner] fast_ingest: the advantages of {len(group)} messages are computed one episode "
                              f"at a time ({self.last_error}; {self.fallbacks} such times)", flush=True)
                    self._one_by_one(group)
            self.flushes += 1
        finally:
            self.flush_s += time.perf_counter() - t0

    @staticmethod
    def _batch(group) -> None:
        settings = group[0][0]
        gamma, lam, _dyn, lam_min, lam_max, lam_eps = settings
        cat = (lambda xs: xs[0] if len(xs) == 1 else torch.cat(xs))
        r = cat([it[1] for it in group])
        v = cat([it[2] for it in group])
        d = cat([it[3] for it in group])
        var = cat([it[4] for it in group]) if group[0][4] is not None else None
        L = np.concatenate([it[5] for it in group])
        got = gae_batch(r, v, d, L, gamma=gamma, lam=lam, variances=var, lam_min=lam_min, lam_max=lam_max, lam_eps=lam_eps)
        if got is None:
            raise ValueError("gae_batch did not take the update's columns")
        adv, ret = got
        a = 0
        for it in group:
            b = a + int(it[1].shape[0])
            it[6].copy_(adv[a:b])
            it[7].copy_(ret[a:b])
            a = b


# ------------------------------------------------------------------------------------------------
# Observations on the GPU from the time they come in
# ------------------------------------------------------------------------------------------------

class GpuObsBuffer:
    """
    One update's observations on the learner's device, written as the messages come in.

    Allocated at the first message, in its dtype and width, with rows_hint rows (1.5 x
    steps_per_update); an update that runs past that grows it once with one copy on the device.
    take(n) hands the update a view of the first n rows and starts the next update at row 0.

    The copies. append() runs on the learner's thread: it checks the message, finds its rows (and
    grows the buffer when it has to, there and then) and hands the copy to the buffer's own thread,
    which makes it on the CUDA stream the learner's thread was on when it handed it over. A copy
    from pageable memory holds its thread until the bytes are on their way, and torch lets go of the
    Python lock meanwhile, so the learner's thread goes on with the next message. take() waits until
    every copy handed over has been made, so the update, queued on the same stream after them, reads
    them all. The next update's copies are handed over only after the update's kernels have been
    queued (the learner takes no message during an update), so they land on the stream after them
    and one buffer is enough. Without the thread (background=False, or should it have died) append()
    makes the copy itself, as the first version of this did.

    It never raises. Anything that goes wrong (no room on the device, a message of another dtype or
    width, a copy that fails on the thread) leaves this update's observations to the CPU dataset,
    which always has them: take() then returns None, says why in `reason`, and the learner joins
    them as before.
    """

    WAIT_LOG_S = 60.0

    def __init__(self, device, rows_hint: int, background: bool = True):
        self.device = torch.device(device)
        self.rows_hint = max(1, int(rows_hint))
        self.buf: Optional[torch.Tensor] = None
        self.n = 0
        self.ok = True
        self.reason = ""
        self.grows = 0
        self.failures = 0
        self.copy_s = 0.0          # time append() spent on the caller's thread
        self.thread_s = 0.0        # time the copy thread spent copying
        self.wait_s = 0.0          # time take() / reset() waited for the copy thread
        self.handed = 0            # copies handed to the thread
        self.inline = 0            # copies made on the caller's thread
        self.background = bool(background)
        self._cv = threading.Condition()
        self._jobs: "queue.SimpleQueue" = queue.SimpleQueue()
        self._pending = 0          # copies handed over and not made yet
        self._thread_error = ""    # the first copy that failed on the thread since the last take()
        self._streams: List = []   # the streams copies went to since the last take()
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    @property
    def capacity(self) -> int:
        return 0 if self.buf is None else int(self.buf.shape[0])

    def nbytes(self) -> int:
        return 0 if self.buf is None else int(self.buf.numel() * self.buf.element_size())

    def _fail(self, why: str) -> bool:
        if self.ok:
            self.failures += 1
        self.ok = False
        self.reason = why
        return False

    # -- the copy thread ------------------------------------------------------------------------

    def _thread_alive(self) -> bool:
        if not self.background or self._closed:
            return False
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="learner-obs-copy", daemon=True)
            self._thread.start()
        return self._thread.is_alive()

    @staticmethod
    def _copy(dst: torch.Tensor, src: torch.Tensor, stream) -> None:
        """One message's rows into the buffer, on `stream` (the learner thread's stream when it handed them over)."""
        if stream is not None:
            with torch.cuda.stream(stream):
                dst.copy_(src, non_blocking=True)
        else:
            dst.copy_(src, non_blocking=True)

    def _run(self) -> None:
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device.index)
        while True:
            job = self._jobs.get()
            if job is None:
                return
            dst, src, stream = job
            t0 = time.perf_counter()
            err = ""
            try:
                self._copy(dst, src, stream)
            except torch.cuda.OutOfMemoryError as e:
                err = f"no room on {self.device}: {str(e).splitlines()[0][:160]}"
            except Exception as e:      # noqa: BLE001
                err = f"the copy failed on its thread: {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
            finally:
                del dst, src, job
                with self._cv:
                    self.thread_s += time.perf_counter() - t0
                    if err and not self._thread_error:
                        self._thread_error = err
                    self._pending -= 1
                    self._cv.notify_all()

    def _drain(self) -> None:
        """Waits until every copy handed over has been made (logs every minute it waits)."""
        if self._pending == 0:
            return
        t0 = time.perf_counter()
        said = t0
        with self._cv:
            while self._pending > 0:
                if self._cv.wait(1.0):
                    continue
                if not (self._thread is not None and self._thread.is_alive()):
                    # nothing will make the copies left: this update joins its observations on the CPU
                    self._pending = 0
                    if not self._thread_error:
                        self._thread_error = "the copy thread is gone"
                    break
                if time.perf_counter() - said >= self.WAIT_LOG_S:
                    said = time.perf_counter()
                    msg = f"waited {said - t0:.0f} s for {self._pending} observation copies"
                    logger.warning(msg)
                    print(f"[learner] fast_ingest: {msg}", flush=True)
        self.wait_s += time.perf_counter() - t0

    def close(self, timeout: float = 10.0) -> None:
        """Makes the copies handed over and ends the thread; later copies are made on the caller's thread."""
        self._drain()
        self._closed = True
        if self._thread is not None and self._thread.is_alive():
            self._jobs.put(None)
            self._thread.join(timeout)

    # -- the learner's side ---------------------------------------------------------------------

    def append(self, obs: torch.Tensor) -> bool:
        """Copies obs [rows, width] in after the rows already there. False when this update's buffer has given up."""
        if not self.ok:
            return False
        t0 = time.perf_counter()
        try:
            if not torch.is_tensor(obs) or obs.ndim != 2:
                return self._fail(f"observations of shape {tuple(getattr(obs, 'shape', ()))}")
            rows, width = int(obs.shape[0]), int(obs.shape[1])
            if self.buf is not None and (self.buf.dtype != obs.dtype or int(self.buf.shape[1]) != width):
                if self.n > 0:
                    return self._fail(f"a message of {obs.dtype} x {width} after {self.buf.dtype} x {int(self.buf.shape[1])}")
                self._drain()
                self.buf = None                       # a new layout at the start of an update: start over
            if self.buf is None:
                self.buf = torch.empty((max(self.rows_hint, rows), width), dtype=obs.dtype, device=self.device)
            stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
            if self.n + rows > self.capacity:
                # every copy into the old storage made first, then the rows so far moved on this stream
                self._drain()
                self._order_after_copies(stream)
                grown = torch.empty((max(self.n + rows, int(self.capacity * 1.5)), width), dtype=obs.dtype, device=self.device)
                grown[:self.n].copy_(self.buf[:self.n])
                self.buf = grown
                self.grows += 1
            dst = self.buf[self.n:self.n + rows]
            if self._thread_alive():
                with self._cv:
                    self._pending += 1
                if stream is not None and all(s != stream for s in self._streams):
                    self._streams.append(stream)
                self._jobs.put((dst, obs, stream))
                self.handed += 1
            else:
                self._copy(dst, obs, stream)
                self.inline += 1
            self.n += rows
            return True
        except torch.cuda.OutOfMemoryError as e:
            return self._fail(f"no room on {self.device}: {str(e).splitlines()[0][:160]}")
        except Exception as e:      # noqa: BLE001
            return self._fail(f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}")
        finally:
            self.copy_s += time.perf_counter() - t0

    def _order_after_copies(self, stream) -> None:
        """Work queued on `stream` from now on comes after every copy made so far (on whatever stream it went)."""
        if stream is not None:
            for s in self._streams:
                if s != stream:
                    stream.wait_stream(s)
        self._streams = [stream] if stream is not None else []

    def _settle(self) -> None:
        """Every copy made; a copy that failed on the thread fails this update."""
        self._drain()
        with self._cv:
            err, self._thread_error = self._thread_error, ""
        if err:
            self._fail(err)
        self._order_after_copies(torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None)

    def take(self, n: int) -> Optional[torch.Tensor]:
        """A view of the first n rows when the buffer holds exactly this update's n rows, else None. Either way the next update starts at row 0."""
        self._settle()
        good = self.ok and self.buf is not None and self.n == int(n) and n > 0
        if not good and self.ok:
            self.reason = f"the buffer holds {self.n} rows and the update has {int(n)}"
        view = self.buf[:int(n)] if good else None
        self.n = 0
        self.ok = True
        return view

    def reset(self) -> None:
        """Drops this update's rows (the learner cleared its dataset)."""
        self._settle()
        self.n = 0
        self.ok = True
        self.reason = ""


class DeviceEpisodeDataset(AsyncEpisodeDataset):
    """
    AsyncEpisodeDataset whose add_steps keeps the observations where they are (the learner's GPU
    buffer). Every other column goes to self.device as in the base class (the CPU in the learner).
    on_device then moves only the small columns, and iter_minibatches cuts the same rows in the same
    order.
    """

    def add_steps(self, obs, act, logp, val, adv, ret, next_hp, ep_ids, extras=None):
        # The base class stores every other column exactly as always; it is handed an empty slice of
        # the observations (moving no rows costs nothing) and the real ones replace it.
        super().add_steps(obs[:0], act, logp, val, adv, ret, next_hp, ep_ids, extras=extras)
        self.obs[-1] = obs.detach()


def swap_out_columns_without_obs(ds: AsyncEpisodeDataset) -> Tuple[Tuple, Tuple[str, ...]]:
    """
    ds.swap_out_columns() without joining the observations: every other column is the dataset's own
    tensorize() output, column for column. Index 0 of the result is an empty observation tensor (the
    caller puts the update's observations there). The dataset is cleared, as by swap_out_columns.
    """
    if ds.obs:
        ds.obs = [ds.obs[0][:0]]                  # one chunk: tensorize's join hands it back as it is
    ds._tensor_cache = None
    return ds.swap_out_columns()


# ------------------------------------------------------------------------------------------------
# The weights on the wire
# ------------------------------------------------------------------------------------------------

def _tensor_from_bytes(raw, dtype: torch.dtype, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    The receiving end of OutOfBandTensor, in the weight store's process: a tensor in CPU memory of
    its own (a copy of the bytes Ray delivered, which sit read-only in its object store), with the
    dtype and shape it was sent with. What unpickling the tensor itself gives: the same dtype, shape,
    contiguous strides, values and bytes, no gradient, plain (not pinned) memory owned by torch.
    """
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    out = torch.empty(int(raw.size), dtype=torch.uint8)
    if raw.size:
        np.copyto(out.numpy(), raw)
    return out.view(dtype).reshape(shape)


class OutOfBandTensor:
    """
    A CPU tensor as the weight push hands it to Ray: pickled, it is its raw bytes as a NumPy uint8
    view, which pickle protocol 5 (Ray's) hands over out of band, plus its dtype and shape. Unpickled,
    it is a torch tensor again (_tensor_from_bytes), equal to the one sent.

    Why: Ray pickles a torch tensor through torch.save into a bytes object and copies that into the
    pickle, holding the Python lock throughout: about 60 ms for the policy's 32 MB on box c, which
    the push thread took from the learner's thread while it took in the next update's messages (the
    review of v2-fast-ingest, 2026-10-18). Out of band, the pickle is 6 KB and Ray copies the bytes
    into its object store itself. With a real Ray (bench/fast_ingest_bench.py wire, box c) a push
    took the push thread 42 ms quiet and 77 ms under the box-a emulation as tensors, 24 and 27 ms
    out of band; the weight store, a nice-15 Ray worker, ran an update in 0.67 s against 2.9 s
    (median, emulation). Only the encoding between the two processes changes: the weight store
    gets the same dict as before (same keys in the same order, same dtypes, shapes, strides, values).
    """

    __slots__ = ("t",)

    def __init__(self, t: torch.Tensor):
        self.t = t

    def __reduce__(self):
        t = self.t
        return (_tensor_from_bytes, (t.reshape(-1).view(torch.uint8).numpy(), t.dtype, tuple(t.shape)))


def wire_ready(t) -> bool:
    """
    True when t can go as OutOfBandTensor and come back the same: a plain dense CPU tensor that owns
    exactly its storage from its start (as the push's clones do), contiguous, with no gradient and
    no conjugate or negative bit. Anything else goes to Ray as the tensor itself, as before.
    """
    return (type(t) is torch.Tensor and t.device.type == "cpu" and t.layout == torch.strided and not t.requires_grad
            and not t.is_quantized and not t.is_conj() and not t.is_neg() and t.is_contiguous()
            and t.storage_offset() == 0 and t.untyped_storage().nbytes() == t.numel() * t.element_size())


def for_the_wire(weights: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """weights with every wire_ready tensor as an OutOfBandTensor: same keys, same order."""
    return {k: (OutOfBandTensor(v) if wire_ready(v) else v) for k, v in weights.items()}


# ------------------------------------------------------------------------------------------------
# The weight push, off the update's thread
# ------------------------------------------------------------------------------------------------

class WeightPusher:
    """
    weight_store.update(weights, version) and weight_store.set_temp(temp) after every update, from one
    background thread, in update order. push() takes the snapshot (one batched copy into pinned CPU
    buffers and a CUDA event) and returns; the thread waits for the event, clones the buffers into a
    dict of plain CPU tensors (the dict the learner used to build: same keys in the same order, same
    dtypes, shapes, strides and values) and makes the two calls, the tensors as OutOfBandTensor so
    that Ray copies their bytes without the Python lock (the store unpickles the same dict). At most
    one push is in flight: push() waits for the one before to have been handed to Ray.

    A snapshot that cannot be taken (a tensor that is not one, a copy that fails) is pushed on the
    caller's thread exactly as the learner always did, and a dead thread sends every push that way.

    The out-of-band weights are unpickled by fast_ingest._tensor_from_bytes in the store's process,
    which imports fast_ingest from the same code directory as inference.py. The first out-of-band
    update's reply is kept, and every later push looks at it without waiting (the store is a nice-15
    Ray worker among busy processes and its first update can take a minute: 68 s on box c under the
    box-a emulation, so waiting for it stalled the learner in a first version of this). Once it has run
    the out-of-band push is known to work and nothing is looked at again. Should it have failed (the
    store could not unpickle the weights), the version at hand and every later one go as the tensors
    themselves, as before the change, and a line says so; the inference actors then kept the weights
    before that first version until this push.
    """

    WAIT_LOG_S = 60.0

    def __init__(self, store, name: str = "learner", out_of_band: bool = True):
        self.store = store
        self.name = name
        # True: the weights go to Ray as OutOfBandTensor (bench/fast_ingest_bench.py compares both ways)
        self.out_of_band = bool(out_of_band)
        self.oob_confirmed = False  # an out-of-band update is known to have run in the store
        self.oob_refused = ""       # why the store did not take them, when it did not (then out_of_band is False)
        self._first = None          # (the reply to the first out-of-band update, its version) until it is known
        self._get = None            # ray.get (tests put a stand-in here)
        self._jobs: "queue.Queue" = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        self._keys: Optional[List[str]] = None
        self._sig = None
        self._pinned: Optional[List[torch.Tensor]] = None
        self.pushed = 0
        self.errors = 0
        self.sync_pushes = 0
        self.wait_s = 0.0           # time push() waited for the push before it
        self.snap_s = 0.0           # time push() took on the caller's thread for the snapshot
        self.thread_s = 0.0         # time the thread spent per push (wait for the copy, clone, hand to Ray)
        self.last_error = ""
        self._thread = threading.Thread(target=self._run, name=f"{name}-weight-push", daemon=True)
        self._thread.start()

    def _sync_push(self, state_dict, version: int, temp: float) -> None:
        """What the learner always did, on the caller's thread."""
        weights = {k: v.cpu().detach() for k, v in state_dict.items()}
        self.store.update.remote(weights, version)
        self.store.set_temp.remote(temp)
        self.sync_pushes += 1

    def _wait_idle(self) -> None:
        t0 = time.perf_counter()
        while not self._idle.wait(self.WAIT_LOG_S):
            msg = f"the weight push before is still in flight after {time.perf_counter() - t0:.0f} s; waiting"
            logger.warning(f"{self.name}: {msg}")
            print(f"[learner] fast_ingest: {msg}", flush=True)
            if not self._thread.is_alive():
                break
        self.wait_s += time.perf_counter() - t0

    def push(self, state_dict: Dict[str, torch.Tensor], version: int, temp: float) -> None:
        self._wait_idle()
        t0 = time.perf_counter()
        if not self._thread.is_alive():
            self._sync_push(state_dict, version, temp)
            return
        keys = list(state_dict.keys())
        vals = [state_dict[k] for k in keys]
        if not all(torch.is_tensor(v) for v in vals):
            self._sync_push(state_dict, version, temp)
            return
        try:
            sig = [(k, v.dtype, tuple(v.shape), tuple(v.stride()), v.device.type) for k, v in zip(keys, vals)]
            cuda = [v for v in vals if v.device.type == "cuda"]
            if sig != self._sig:
                self._pinned = [torch.empty_strided(tuple(v.shape), tuple(v.stride()), dtype=v.dtype, device="cpu",
                                                    pin_memory=bool(cuda)) for v in vals]
                self._keys, self._sig = keys, sig
            with torch.no_grad():
                torch._foreach_copy_(self._pinned, [v.detach() for v in vals], non_blocking=True)
            event = None
            if cuda:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(cuda[0].device))
        except Exception as e:      # noqa: BLE001
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning(f"{self.name}: weight snapshot failed ({e!r}); pushing on this thread as before")
            self._sig = None
            self._sync_push(state_dict, version, temp)
            return
        self._idle.clear()
        self._jobs.put((event, version, float(temp)))
        self.snap_s += time.perf_counter() - t0

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                self._idle.set()
                return
            event, version, temp = job
            t0 = time.perf_counter()
            try:
                if event is not None:
                    event.synchronize()
                weights = {k: t.clone() for k, t in zip(self._keys, self._pinned)}
                self._update(weights, version)
                self.store.set_temp.remote(temp)
                self.pushed += 1
            except Exception as e:      # noqa: BLE001
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                logger.error(f"{self.name}: weight push of version {version} failed: {e!r}")
                print(f"[learner] fast_ingest: weight push of version {version} failed: {e!r}", flush=True)
            finally:
                self.thread_s += time.perf_counter() - t0
                self._idle.set()

    def _update(self, weights: Dict[str, torch.Tensor], version: int) -> None:
        """store.update(weights, version), out of band when it is on (Ray copies the bytes without the Python lock)."""
        if self.out_of_band:
            self._look_at_first(version)
        if not self.out_of_band:
            self.store.update.remote(weights, version)
            return
        ref = self.store.update.remote(for_the_wire(weights), version)
        if self._first is None and not self.oob_confirmed and type(ref).__name__ == "ObjectRef":
            self._first = (ref, version)            # a Ray store's reply (a stand-in store returns none)

    def _look_at_first(self, version: int) -> None:
        """Whether the first out-of-band update has run in the store, without waiting: confirmed, refused or not known yet."""
        if self.oob_confirmed or self._first is None:
            return
        ref, first = self._first
        try:
            if self._get is None:
                import ray
                self._get = ray.get
            self._get(ref, timeout=0)
            self.oob_confirmed = True
            self._first = None
        except Exception as e:      # noqa: BLE001
            if type(e).__name__ == "GetTimeoutError":
                return                              # not run yet: the next push looks again
            self._first = None
            self.out_of_band = False
            self.oob_refused = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
            msg = (f"the weight store could not take version {first} out of band ({self.oob_refused}); from version "
                   f"{version} on the weights go as the tensors themselves")
            logger.error(f"{self.name}: {msg}")
            print(f"[learner] fast_ingest: {msg}", flush=True)

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Waits until the last push has been handed to Ray. True when it has."""
        return self._idle.wait(timeout)

    def close(self, timeout: float = 10.0) -> None:
        self.flush(timeout)
        self._jobs.put(None)
        self._thread.join(timeout)
