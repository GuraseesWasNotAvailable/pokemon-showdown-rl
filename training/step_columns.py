"""
Named per-step columns that travel with the training data beside the eight fixed ones.

A rollout worker can attach columns to an episode (RolloutWorker.episode_extras: a dict of name ->
array with one row per step, integer, bool or float). They go to the learner with the episode
(SyncLearnerClient packs them, LearnerActor.submit_packed_batch takes them), and the learner's
data set keeps them aligned with their rows through the advantage pass, the update's data on the
CPU or the GPU, and the minibatches cut by whole episodes (ppo_core.AsyncEpisodeDataset). In
ppo_update every minibatch has them as a dict, mb_extras. Nothing reads them yet: later stages add
their columns (read_tier, opp_action, boosted) and the code that uses them.

With no columns anywhere, nothing of this runs: the episodes, the packed batches, the queue items
and the data set's tensors are what they were, and an update is bit-identical (tests/test_extras.py).

A step without a value for a column (its episode came without it, in a batch where others had it)
gets the column's fill value: FILL[name], 0 when the name is not registered. A stage whose column
needs another "no label" value registers it here, e.g. FILL["opp_action"] = -1.

NumPy only: the worker side imports this without torch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Fill value per column name, for steps that came without the column. Unregistered names fill with 0.
# opp_action: -1 is "no label" (read_reward.OPP_NA); read, boosted and set_labels fill with 0 (no bonus,
# not boosted, no label).
FILL: Dict[str, float] = {"opp_action": -1}


def fill_value(name: str) -> float:
    return FILL.get(name, 0)


def _as_column(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.int8)
    elif arr.dtype.kind == "u" and arr.dtype.itemsize > 1:
        arr = arr.astype(np.int64)          # torch takes uint8, and not every wider unsigned type
    return arr


def check_columns(cols: Any, n: int) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    """
    The usable columns of `cols` for n steps: each a NumPy array whose first dimension is n, of
    integers or floats (bools become int8). Returns (columns or None when none is usable, problems):
    a column of another length, of another kind (text, objects, complex), with a name that is not
    text, or with a value that is not finite is left out and said. Never raises.
    """
    problems: List[str] = []
    if cols is None:
        return None, problems
    if not isinstance(cols, dict):
        return None, [f"extra columns are a {type(cols).__name__}, not a dict"]
    out: Dict[str, np.ndarray] = {}
    for name, value in cols.items():
        if not isinstance(name, str) or not name:
            problems.append(f"an extra column's name is {name!r}, not text")
            continue
        try:
            arr = _as_column(value)
        except Exception as e:      # noqa: BLE001
            problems.append(f"extra column {name}: {type(e).__name__}")
            continue
        if arr.dtype.kind not in "iuf":
            problems.append(f"extra column {name} holds {arr.dtype}, not integers or floats")
            continue
        if arr.ndim < 1 or arr.shape[0] != n:
            problems.append(f"extra column {name} has {arr.shape[0] if arr.ndim else 'no'} rows for {n} steps")
            continue
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            problems.append(f"extra column {name} has values that are not finite")
            continue
        out[name] = arr
    return (out or None), problems


def pack_columns(items: Sequence[Optional[Dict[str, np.ndarray]]], lengths: Sequence[int]) -> Optional[Dict[str, np.ndarray]]:
    """
    The columns of several episodes, joined in their order: one array per name with sum(lengths)
    rows. An episode without a column gets its fill value for its steps, so every row keeps its
    own value. None when no episode has any column.
    """
    names = sorted({name for cols in items if cols for name in cols})
    if not names:
        return None
    out: Dict[str, np.ndarray] = {}
    for name in names:
        have = [cols[name] for cols in items if cols and name in cols]
        dtype = np.result_type(*have)
        tail = have[0].shape[1:]
        parts = []
        for cols, length in zip(items, lengths):
            if cols and name in cols and cols[name].shape[1:] == tail:
                parts.append(cols[name].astype(dtype, copy=False))
            else:
                parts.append(np.full((int(length),) + tuple(tail), fill_value(name), dtype=dtype))
        out[name] = np.concatenate(parts, axis=0)
    return out
