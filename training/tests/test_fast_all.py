"""
v2-fast: the three speed flags together (--fast-update, --fast-ingest, --rollout-nice), held to
v2-humanaux 4cfd496 bit for bit.

    python -m pytest tests/test_fast_all.py -q -p no:cacheprovider -rs
    PSPPO_BASE_DIR=<a directory with 4cfd496's files> python -m pytest tests/test_fast_all.py ...

Each flag has its own suite (test_fast_update.py, test_fast_ingest.py, test_rollout_nice.py). What
only the merge can break is held here:
  * the switches: all three parse together, none of them reaches RunConfig (so a checkpoint's
    run_cfg is the same with or without them), and the learner gets exactly the keywords of the
    flags that are on;
  * the two learner flags on one update: --fast-ingest hands ppo_update a data set whose
    observations already sit on the GPU (fast_ingest.DeviceEpisodeDataset), and --fast-update runs
    the sync-free loop over it. Two whole updates in a row from the same weights, data and seed:
    every weight, fp32 master, Adam moment, the schedule and every returned figure the same bits as
    4cfd496's ppo_update over the CPU data set it gets today, including a KL stop at the first
    minibatch of the second epoch and rows with an all-zero action mask.

The live-sized check of the same (the newest live checkpoint, recorded worker messages through the
learner's own loop, the weight pushes and the checkpoint written after) is
bench/fast_ingest_bench.py check --fast-update. Nothing here starts Ray, a server or a training run;
nothing is removed.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

import torch  # noqa: E402

import fast_ingest  # noqa: E402
import learner as new_learner  # noqa: E402
import ppo_core as new_core  # noqa: E402
import test_speed_update as tsu  # noqa: E402
from bench import micro_ref  # noqa: E402
from test_fast_update import FIGURES, STOP_AT_ONCE, _cols, _extras, _update, base, data, needs_gpu  # noqa: E402,F401

DEV = tsu.DEV

# The live run's PPO command line (run.json of the scratch run, the top-player term off as now).
LIVE = ["--mode", "ppo", "--battles", "2304", "--rooms-per-pair", "16", "--teacher", "psbot", "--steps-per-update", "36864",
        "--minibatch", "768", "--seq-len", "256", "--grad-accum", "1", "--save-every", "50", "--obs-matchup", "--obs-movesets",
        "--infer-actors", "3", "--infer-sync-s", "1", "--set-head-coef", "0.1", "--opp-action-aux-coef", "0.1",
        "--read-bonus", "0.3", "--read-anneal", "20000000:45000000", "--opponent-seats", "setupnet:0.15",
        "--boosted-row-weight", "2", "--lr-hold-steps", "479713", "--lr-total-steps", "1113566"]
FLAGS = ["--fast-update", "--fast-ingest", "--rollout-nice", "5"]


# --------------------------------------------------------------------------- the switches


def test_the_three_flags_parse_together_and_none_reaches_run_config():
    import train
    off = train.parse_args(LIVE)
    on = train.parse_args(LIVE + FLAGS)
    assert (off.fast_update, off.fast_ingest, off.rollout_nice) == (False, False, None)
    assert (on.fast_update, on.fast_ingest, on.rollout_nice) == (True, True, 5)
    assert train.learner_speed_kwargs(off) == {} and train.fast_ingest_kwargs(off) == {}
    assert train.rollout_nice_of(off) is None
    assert train.learner_speed_kwargs(on) == {"fast_update": True}
    assert train.fast_ingest_kwargs(on) == {"fast_ingest": True}
    assert train.rollout_nice_of(on) == 5
    assert train.build_config(off).as_dict() == train.build_config(on).as_dict()
    # each flag alone, and every pair: the others stay off
    for i, f in enumerate(["--fast-update", "--fast-ingest"]):
        one = train.parse_args(LIVE + [f])
        assert (one.fast_update, one.fast_ingest, one.rollout_nice) == ((True, False, None) if i == 0 else (False, True, None))


def test_the_learner_takes_both_keywords_off_by_default():
    params = inspect.signature(new_learner.LearnerActor.__init__).parameters
    assert params["fast_update"].default is False and params["fast_ingest"].default is False
    assert params["human_imitation"].default is None
    # the order 4cfd496 had for its positional arguments is kept
    assert list(params)[:4] == ["self", "cfg", "weight_store", "human_imitation"]


# --------------------------------------------------------------------------- the two learner flags on one update


def _update_from_gpu_buffer(la, cols, extras, seed: int):
    """ppo_update as the learner calls it with both learner flags: the observations on the GPU (the
    buffer's view), every other column on the CPU, in a DeviceEpisodeDataset, and fast_update."""
    ds = fast_ingest.DeviceEpisodeDataset(14, "cpu")
    obs = cols[0].to(DEV)
    if extras is None:
        ds.add_steps(obs, *cols[1:])
    else:
        ds.add_steps(obs, *cols[1:], extras=extras)
    assert ds.obs[-1].device == obs.device and obs.device.type == "cuda"     # not moved back to the CPU
    torch.manual_seed(seed)
    return new_core.ppo_update(net=la.net, opt=la.opt, dataset=ds, scheduler=la.sched, mode=la.cfg.mode, target_net=None,
                               **la.cfg.ppo_kwargs(), v_min=la.cfg.v_min, v_max=la.cfg.v_max, v_bins=la.cfg.v_bins,
                               **la._aux_update_kwargs(), fast_update=True)


# variant: (mode, learner overrides, extras?, share of rows with an all-zero action mask)
VARIANTS = {
    "ppo": ("ppo", {"update_epochs": 2}, False, 0.0),
    "ppo_aux_live": ("ppo", {"update_epochs": 2, "opp_action_aux_coef": 0.1, "boosted_row_weight": 2.0}, True, 0.0),
    "ppo_zero_masks": ("ppo", {"update_epochs": 2}, False, 0.03),
    "ppo_stop_at_once": ("ppo", {"update_epochs": 2, "target_kl": STOP_AT_ONCE}, True, 0.0),
}


@needs_gpu
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_both_learner_flags_give_4cfd496s_update_bit_for_bit(base, data, variant, caplog):
    mode, over, with_extras, zero_share = VARIANTS[variant]
    obs_np, ep_np, _kind = data
    n = 1536                                            # 6 minibatches of 256 an epoch, grad_accum 3
    updates = [_cols(obs_np, ep_np, n, 30, zero_share), _cols(obs_np[n:], ep_np[n:], n, 31, zero_share)]
    xs = [_extras(n, 40), _extras(n, 41)] if with_extras else [None, None]
    old = tsu._old_learner(base, mode, **over)
    both = tsu._new_learner(mode, **over)
    tsu._copy_state(old, both)
    for step, (cols, x) in enumerate(zip(updates, xs)):
        sa = _update(base.core, old, cols, x, 70 + step)
        want = tsu._state(old)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=new_core.__name__):
            sb = _update_from_gpu_buffer(both, cols, x, 70 + step)
        got = tsu._state(both)
        assert want.keys() == got.keys()
        diff = [k for k in want if not micro_ref.same_bits(want[k], got[k])]
        assert diff == [], (variant, step, diff[:5])
        for f in FIGURES:
            assert getattr(sa, f) == getattr(sb, f), (variant, step, f, getattr(sa, f), getattr(sb, f))
        zero_lines = [r.getMessage() for r in caplog.records if "Zero mask detected" in r.getMessage()]
        assert len(zero_lines) == (1 if zero_share > 0.0 else 0), zero_lines
        if over.get("target_kl", 0.02) == STOP_AT_ONCE:
            assert sa.n_mb == sb.n_mb == 7              # the first minibatch of the second epoch
        else:
            assert sa.n_mb == sb.n_mb
    assert any(m.fast_unbind for m in both.net.modules() if isinstance(m, new_core.FlexEncoderLayer))
    assert both.opt.foreach_copy
    del old, both
    torch.cuda.empty_cache()
