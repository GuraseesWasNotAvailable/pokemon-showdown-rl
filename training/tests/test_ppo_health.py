"""
The PPO health work: the option to keep each battle's history when new weights arrive, the
diagnostics at the end of the learner's update line, the hyperparameter flags of train.py,
and a rollout worker whose loop has died ending its run().

    python -m pytest tests/test_ppo_health.py -q -p no:cacheprovider -s -rs

What is held, and to what:

  * with nothing new asked for, the code is release r1 (commit 8227671, what the cloud box
    runs PPO with, bench/micro_ref.py): a whole update in imitation, warmup and PPO leaves
    the same weights, optimizer state and loss figures bit for bit, the inference actor
    answers a long random run with new weights arriving in it identically, and the files the
    observations come from are r1's
  * --keep-history is what bench/check_history.py measured as "(ii)" before the option
    existed (an actor whose clear_all_caches does nothing), on the same sequences
  * kl0 and clip0 are the first minibatch's figures, computed before the update's first
    optimizer step, and stop and ep say what the early stop did
  * every flag reaches its LearnerConfig field, none of them moves a default when absent,
    and a schedule that cannot be is refused at start, but only when a schedule flag was given
  * a worker whose rollout loop raises ends its run() with that error (it used to sleep on)
  * after the review of this work: the restart it settled on (--keep-history and, as the
    one choice of schedule that changes nothing before the hold ends, --lr-total-steps
    1000000) on the box's own config, the "ppo settings:" line by which a restart is
    checked, and bench/check_history.py saying of its box tables that they are simulated,
    printing no share of a "live" figure it was not given, and reading a run's own kl0 and
    clip0 per start of the trainer (--metrics)

Optional inputs: PSPPO_SELFPLAY (episodes recorded by bench/check_history.py --record) or
PSPPO_OBS_SAMPLES (bench/micro_obs.py --out) give real observations, and PSPPO_CKPT_A and
PSPPO_CKPT_B two consecutive PPO checkpoints (default: runs/vm/checkpoints/
learner_update_000070.pt and ..._000071.pt when they are there). Without them made-up
observations and perturbed weights are used. PSPPO_RAY_TESTS=1 adds the one test that starts
Ray. Nothing is removed.
"""
from __future__ import annotations

import ast
import asyncio
import dataclasses
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO)          # vocab.json is opened by a relative path

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config as new_config  # noqa: E402
import inference as new_inf  # noqa: E402
import learner as new_learner  # noqa: E402
import ppo_core as new_core  # noqa: E402
import train as new_train  # noqa: E402
import worker as new_worker  # noqa: E402
import test_speed_update as tsu  # noqa: E402  (its learner and update helpers)
# bench/check_history.py (the run-history checker) did not come over to this code line with these
# tests: without it the module is skipped instead of failing to import.
check_history = pytest.importorskip("bench.check_history")
from bench import micro_ref  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU = DEV == "cuda"
needs_gpu = pytest.mark.skipif(not GPU, reason="flex_attention has no backward on a CPU")
NEVER = 1e18
CORE_FILES = ("config", "ppo_core", "inference", "learner")       # the modules this work changed


def _default_ckpt(n: int) -> str:
    return os.path.join(REPO, "runs", "vm", "checkpoints", f"learner_update_{n:06d}.pt")


CKPT_A = os.environ.get("PSPPO_CKPT_A") or _default_ckpt(70)
CKPT_B = os.environ.get("PSPPO_CKPT_B") or _default_ckpt(71)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def r1():
    """Release r1's modules, sharing one compiled flex_attention with the working tree's."""
    try:
        ref = micro_ref.Reference.export(micro_ref.R1_COMMIT)
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(f"release r1 is not at hand to compare with: {e}")
    core, lrn, cfg, inf = ref.load("ppo_core"), ref.load("learner"), ref.load("config"), ref.load("inference")
    assert lrn.ppo_core is core and inf.ppo_core is core and core is not new_core and cfg is not new_config
    # r1 is the code before this work: no option, no diagnostics
    assert "clear_history_on_sync" not in {f.name for f in dataclasses.fields(cfg.InferenceConfig)}
    assert "kl0" not in {f.name for f in dataclasses.fields(core.PPOUpdateStats)}
    assert hasattr(core, "update_data_fits") and hasattr(inf.InferenceActor, "_kv_pool") is False and hasattr(inf.InferenceActor, "_grow_kv_pool")
    if GPU:
        micro_ref.shared_compiled_flex_attention(core, new_core)
    core._learner_flex_compiled = True
    new_core._learner_flex_compiled = True
    return SimpleNamespace(core=core, learner=lrn, config=cfg, inference=inf, ref=ref)


@pytest.fixture(scope="module")
def data():
    obs, ep, kind = micro_ref.observations(n_min=4096)
    return obs, ep, kind


@pytest.fixture(scope="module")
def sequences():
    """Decision sequences of battle sides: recorded self-play when PSPPO_SELFPLAY names a file, made-up observations otherwise."""
    path = os.environ.get("PSPPO_SELFPLAY")
    if path and os.path.isfile(path):
        seqs, _extra = check_history.sequences_from_npz(path, max_episodes=160)
        return seqs, "recorded self-play"
    obs, ep, kind = micro_ref.observations(n_min=4096)
    cuts = np.flatnonzero(np.diff(ep) != 0) + 1
    bounds = [0] + cuts.tolist() + [int(ep.shape[0])]
    seqs = [np.ascontiguousarray(obs[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2][:60]
    return seqs, kind


def _perturbed(sd, scale: float, seed: int):
    """The weights a small step away, on the CPU, which is where the learner publishes them from."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sd.items():
        v = v.detach().cpu()
        out[k] = (v.float() + scale * torch.randn(v.shape, generator=g)).to(v.dtype) if v.is_floating_point() else v.clone()
    return out


@pytest.fixture(scope="module")
def two_weights():
    """Weights A and B: two consecutive PPO checkpoints when they are on this machine, else a model and a small step away from it."""
    if os.path.isfile(CKPT_A) and os.path.isfile(CKPT_B):
        return check_history.load_weights(CKPT_A), check_history.load_weights(CKPT_B), "checkpoints"
    torch.manual_seed(3)
    net = new_config.RunConfig.default().make_model()
    net.enable_bf16_recurrent_path()
    a = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return a, _perturbed(a, 0.01, 4), "made up"


# --------------------------------------------------------------------------- the defaults are r1: the files


def test_only_the_four_core_modules_differ_from_r1(r1):
    """The observation code (every obs_* module, utils) is r1's file for file, so the observations are r1's."""
    differ = r1.ref.differs_from_tree()
    assert set(differ) <= set(CORE_FILES), differ
    for m in ("utils", "obs_assembler", "obs_moves", "obs_pokemon", "obs_abilities", "obs_global", "obs_transitions"):
        assert m not in differ


def _defs(source: str):
    """name -> ast dump of every top-level function and class, and of every method as Class.method."""
    tree = ast.parse(source)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = ast.dump(sub)
                else:
                    out.setdefault(f"{node.name}.<body>", []).append(ast.dump(sub))
        else:
            out.setdefault("<module>", []).append(ast.dump(node))
    return out


def test_worker_is_r1_but_for_run():
    """
    worker.py is where an observation is made and a trajectory is put together. Against r1
    RolloutWorker.run changed and _guarded_rl_loop was added (PPO health), and for the memory
    work (bench/MEMORY.md): the player gives back a battle whose room the server closed
    (_handle_battle_message, _room_closed, _note_finished_room, two counters in __init__), a
    choice the server refuses is answered with one "/choose default" and no second asking of
    the policy (_apply_patches, default_is_due, default_would_be_refused, the module constants,
    _handle_battle_message), and RolloutWorker.__init__ starts memwatch.py when asked.
    Everything else is r1's code, to the letter: choose_move (the observation), the rollout
    loop, the reward, what is sent to the learner and when a finished battle is finalized.
    tests/test_worker_memory.py and tests/test_worker_choice_storm.py compare what the two
    versions hand to the learner and send to the server on recorded battles.
    """
    try:
        old = micro_ref.text_at(micro_ref.R1_COMMIT, "worker.py")
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(str(e))
    assert micro_ref.git_blob_id(old) == micro_ref.R1_OTHER_BLOBS["worker"]
    with open(os.path.join(REPO, "worker.py"), "r", encoding="utf-8") as f:
        new = f.read()
    a, b = _defs(old.decode("utf-8")), _defs(new)
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert changed == ["<module>", "RolloutWorker.__init__", "RolloutWorker._guarded_rl_loop", "RolloutWorker.run",
                       "SyncBridgePlayer.__init__", "SyncBridgePlayer._handle_battle_message",
                       "SyncBridgePlayer._note_finished_room", "SyncBridgePlayer._room_closed",
                       "_apply_patches", "default_is_due", "default_would_be_refused"], changed
    for same in ("SyncBridgePlayer.choose_move", "SyncBridgePlayer._battle_finished_callback", "RolloutWorker._run_rl_loop",
                 "RolloutWorker._finalize_trajectory", "RolloutWorker._cleanup_trajectory", "SyncLearnerClient.submit_episode",
                 "SyncLearnerClient._prepare_batch", "PokeEnvVectorWrapper.step"):
        assert same in a and a[same] == b[same], same
    assert "RolloutWorker._guarded_rl_loop" not in a and "RolloutWorker._run_rl_loop" in a


def test_train_py_without_flags_builds_r1s_config(r1):
    """No flag, the config of config.py, which apart from the one new field is r1's field for field."""
    args = new_train.parse_args([])
    for name in ("clip_coef", "update_epochs", "lr", "target_kl", "ent_coef", "lr_total_steps", "lr_hold_steps"):
        assert getattr(args, name) is None, name
    assert args.keep_history is False
    cfg = new_train.build_config(args)
    assert cfg == new_config.RunConfig.default()
    with r1.ref.active():
        old = r1.config.RunConfig.default().as_dict()
    new = cfg.as_dict()
    assert new["infer"].pop("clear_history_on_sync") is True
    assert new == old
    lc = cfg.learner
    assert (lc.clip_coef, lc.update_epochs, lc.lr, lc.target_kl, lc.ent_coef) == (0.1, 2, 3e-4, 0.02, 0.02)
    assert (lc.lr_warmup_steps, lc.lr_hold_steps, lc.lr_total_steps) == (1_000, 500_000, 100_000), "the committed schedule is not this work's to move"
    assert cfg.learner.ppo_kwargs() == r1.config.LearnerConfig().ppo_kwargs()


# --------------------------------------------------------------------------- the defaults are r1: a whole update


def _stats_equal(sa, sb, where) -> None:
    for f in ("approx_kl", "clip_frac", "entropy", "v_loss", "pg_loss", "total_loss", "n_mb", "acc"):
        assert getattr(sa, f) == getattr(sb, f), (where, f, getattr(sa, f), getattr(sb, f))


@needs_gpu
@pytest.mark.parametrize("mode", ["imitation", "warmup", "ppo"])
def test_default_update_is_bit_identical_to_r1(r1, data, mode):
    """
    r1's ppo_update and the working tree's from the same weights, data and seed, two updates
    each (the second starts from the Adam moments of the first): every weight, fp32 master,
    Adam moment, step count, rate and loss figure must be equal. The new diagnostics are
    read off numbers the update already had, so they may not move any of it.
    """
    obs_np, ep_np, _kind = data
    n = 1536
    updates = [tsu._update_data(obs_np, ep_np, n, seed=10), tsu._update_data(obs_np[n:], ep_np[n:], n, seed=11)]
    a, a2, b = tsu._old_learner(r1, mode), tsu._old_learner(r1, mode), tsu._new_learner(mode)
    for other in (a2, b):
        tsu._copy_state(a, other)
    start = tsu._state(a)
    for step, cols in enumerate(updates):
        sa = tsu._run_update(r1.core, a, cols, seed=50 + step)
        want = tsu._state(a)
        tsu._run_update(r1.core, a2, cols, seed=50 + step)
        again = tsu._state(a2)
        assert [k for k in want if not micro_ref.same_bits(want[k], again[k])] == [], "r1 does not repeat its own update here"
        sb = tsu._run_update(new_core, b, cols, seed=50 + step)
        got = tsu._state(b)
        assert want.keys() == got.keys()
        diff = [k for k in want if not micro_ref.same_bits(want[k], got[k])]
        assert diff == [], (mode, step, diff[:5])
        _stats_equal(sa, sb, (mode, step))
        assert not hasattr(sa, "kl0")
        if mode == "ppo":
            assert np.isfinite(sb.kl0) and 0 <= sb.clip0 <= 1 and sb.epochs == sb.n_mb / 6 and sb.stopped == (sb.n_mb < 12)
        else:
            assert (sb.kl0, sb.clip0, sb.stopped, sb.epochs) == (0.0, 0.0, False, 2.0)
    trained = [k for k in want if k.startswith("master/")]
    assert len(trained) == (2 if mode == "warmup" else 92)
    assert all(not micro_ref.same_bits(start[k], want[k]) for k in trained)
    del a, a2, b
    torch.cuda.empty_cache()


@needs_gpu
def test_default_update_in_the_boxs_shape_is_bit_identical_to_r1(r1, data):
    """The same in PPO with the box's minibatch: 768 steps as 3 sequences of 256, no accumulation, the early stop live."""
    obs_np, ep_np, _kind = data
    n = 768 * 3
    cols = tsu._update_data(obs_np, ep_np, n, seed=21)
    over = dict(minibatch_size=768, grad_accum_steps=1, steps_per_update=n)
    a, b = tsu._old_learner(r1, "ppo", **over), tsu._new_learner("ppo", **over)
    tsu._copy_state(a, b)
    try:
        sa = tsu._run_update(r1.core, a, cols, seed=77)
        sb = tsu._run_update(new_core, b, cols, seed=77)
    except torch.cuda.OutOfMemoryError:
        del a, b
        torch.cuda.empty_cache()
        pytest.skip("a minibatch of 768 does not fit this GPU")
    want, got = tsu._state(a), tsu._state(b)
    assert [k for k in want if not micro_ref.same_bits(want[k], got[k])] == []
    _stats_equal(sa, sb, "box shape")
    assert sb.epochs == sb.n_mb / 3 and sb.stopped == (sb.n_mb < 6)
    del a, b
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- the diagnostics


def _stale_data(obs_np, ep_np, n: int, seed: int):
    """
    tsu._update_data with the stored log-probabilities moved away from the model's: as they
    come, they are those of a uniform policy, which is what a freshly made model plays, so
    the ratio would be 1 everywhere and there would be nothing for kl0 and clip0 to show.
    """
    cols = list(tsu._update_data(obs_np, ep_np, n, seed))
    g = torch.Generator().manual_seed(seed)
    cols[2] = cols[2] + 0.3 * torch.randn(n, generator=g).abs()      # upwards only, so the mean of old - new is well above zero
    return tuple(cols)


@needs_gpu
@pytest.mark.parametrize("accum,epochs,n", [(1, 1, 256), (1, 2, 1024), (3, 2, 1536)])
def test_kl0_and_clip0_are_the_first_minibatch_before_any_step(data, accum, epochs, n):
    """
    Recomputed from outside: the first minibatch the update draws, through the weights the
    update starts from. With accumulation 3 the first optimizer step comes after three
    minibatches, with 1 right after the first: either way the first minibatch's own forward
    pass has seen no step.
    """
    obs_np, ep_np, _kind = data
    la = tsu._new_learner("ppo", grad_accum_steps=accum, update_epochs=epochs, target_kl=None)
    cols = _stale_data(obs_np, ep_np, n, seed=20)
    ds = new_core.AsyncEpisodeDataset(14, "cpu")
    ds.add_steps(*cols)
    torch.manual_seed(7)
    mb = next(iter(ds.iter_minibatches(256, shuffle_episodes=True)))
    m0, m1 = la.net.unpacker.offsets["action_mask"]
    with torch.no_grad():
        obs = mb[0].to(DEV)
        logits = la.net(obs, mb[7].to(DEV), batch_seq_len=256)[0]
        logp, _ent = new_core.masked_logprob_entropy(logits.float(), obs[:, m0:m1], mb[1].to(DEV))
        old = mb[2].to(DEV)
        want_kl = float((old - logp).mean())
        want_clip = float((((logp - old).exp() - 1.0).abs() > la.cfg.clip_coef).float().mean())
    stats = tsu._run_update(new_core, la, cols, seed=7)
    assert stats.n_mb == epochs * (n // 256) and stats.stopped is False and stats.epochs == float(epochs)
    assert abs(stats.kl0 - want_kl) <= 1e-6 * max(1.0, abs(want_kl)), (stats.kl0, want_kl)
    assert abs(stats.clip0 - want_clip) <= 1e-6, (stats.clip0, want_clip)
    assert want_clip > 0.5, "the made-up log-probabilities are far from the model's: the figure must show it"
    if stats.n_mb > 1:
        assert stats.kl0 != stats.approx_kl, "kl0 is one minibatch, kl the mean of all"
    del la
    torch.cuda.empty_cache()


@needs_gpu
def test_stop_and_ep_say_what_the_early_stop_did(data):
    obs_np, ep_np, _kind = data
    n = 1024                                                    # 4 minibatches an epoch
    cols = _stale_data(obs_np, ep_np, n, seed=22)
    # a target no update can stay under: stops at the first minibatch of the second epoch
    la = tsu._new_learner("ppo", grad_accum_steps=1, target_kl=1e-9)
    s = tsu._run_update(new_core, la, cols, seed=8)
    assert s.stopped is True and s.n_mb == 5 and s.epochs == 1.25
    # no target: both epochs
    lb = tsu._new_learner("ppo", grad_accum_steps=1, target_kl=None)
    s = tsu._run_update(new_core, lb, cols, seed=8)
    assert s.stopped is False and s.n_mb == 8 and s.epochs == 2.0
    # a mode without a ratio: nothing to stop on, and no kl0
    lc = tsu._new_learner("imitation", grad_accum_steps=1)
    s = tsu._run_update(new_core, lc, cols, seed=8)
    assert (s.stopped, s.n_mb, s.epochs, s.kl0, s.clip0) == (False, 8, 2.0, 0.0, 0.0)
    del la, lb, lc
    torch.cuda.empty_cache()


NEW_KEYS = ["kl0", "clip0", "stop", "ep", "lr"]


@needs_gpu
@pytest.mark.parametrize("mode", ["ppo", "imitation", "warmup"])
def test_update_line_ends_with_the_five_new_keys_and_every_parser_reads_it(data, capsys, mode):
    import pipeline
    from bench import throughput
    obs_np, ep_np, _kind = data
    la = tsu._new_learner(mode, save_every_updates=10**9)
    la.dataset.add_steps(*_stale_data(obs_np, ep_np, 1024, seed=30))
    la.total_steps += 1024
    asyncio.run(la._perform_update())
    out = capsys.readouterr().out
    assert "PPO Update failed" not in out
    line = next(ln for ln in out.splitlines() if ln.startswith("[learner] upd="))
    keys = [kv.split("=")[0] for kv in line.split()[1:]]
    old_keys = ["upd", "kl", "clip", "ent", "vloss", "ploss", "loss", "n_mb", "sec"] + (["acc"] if mode == "imitation" else [])
    assert keys == old_keys + NEW_KEYS, line
    vals = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in line.split()[1:]}
    assert vals["stop"] in (0.0, 1.0) and vals["ep"] == round(vals["n_mb"] / 4, 2)
    if mode == "ppo":
        assert vals["kl0"] > 0.01 and vals["clip0"] > 0.5 and vals["stop"] == (1.0 if vals["n_mb"] < 8 else 0.0)
    else:
        assert vals["kl0"] == 0 and vals["clip0"] == 0 and vals["stop"] == 0 and vals["ep"] == 2.0
    # the policy head's rate: one update of 4 minibatches at accumulation 3 is 4 optimizer steps into the warmup
    pi = [pg["lr"] for pg in la.opt.param_groups if pg["name"].startswith("pi_")]
    if mode == "warmup":
        assert pi == [] and vals["lr"] == 0.0, "the warmup does not train the policy head"
    else:
        assert pi and float(f"{pi[0]:.2e}") == vals["lr"] and 0 < vals["lr"] < 3e-4

    ray_line = f"\x1b[36m(LearnerActor pid=5996)\x1b[0m {line}"
    glued = ray_line + ("2026-10-05 09:48:48,489 [INFO] [Telemetry] ACTIVE: 256 | INFER: {'total_requests': 9, 'avg_batch_size': 2.0, "
                        "'model_version': 1} | TRAIN: {'update': 1, 'episodes': 3, 'steps_in_dataset': 0, 'total_steps': 1024}")
    for text in (line, ray_line, ray_line + "\r\n", glued):
        evs = pipeline.parse_trainer_events(text)
        assert [e["kind"] for e in evs] == ["learner_update"] + (["telemetry"] if text is glued else [])
        ev = evs[0]
        assert ev["update"] == 1 and ev["nonfinite"] == []
        assert set(ev["values"]) == set(keys[1:]), "pipeline.py reads every key, the old ones as before"
        for k in keys[1:]:
            assert ev["values"][k] == vals[k], (k, text[-60:])
        tp = throughput.parse_learner(text)
        assert tp["upd"] == 1
        for k in old_keys[1:] + ["stop", "ep", "lr"]:
            assert tp[k] == vals[k], (k, text[-60:])
        # bench/throughput.py takes key names of letters only, so it does not see kl0 and clip0. It
        # must not take them for something else either.
        assert set(tp) - {"actor", "pid"} <= set(keys)
    del la
    torch.cuda.empty_cache()


def test_lines_of_the_old_format_still_parse():
    """What r1 prints, which is what the box's log holds up to the restart: both parsers must go on reading it."""
    import pipeline
    from bench import throughput
    for line, n in (("[learner] upd=93 kl=0.0249 clip=0.293 ent=0.488 vloss=2.165 ploss=0.014 loss=1.087 n_mb=55 sec=4.9", 9),
                    ("\x1b[36m(LearnerActor pid=7)\x1b[0m [learner] upd=3 kl=0.0000 clip=0.000 ent=0.000 vloss=3.780 ploss=1.796 loss=3.686 n_mb=128 sec=41.9 acc=0.412", 10)):
        ev = pipeline.parse_trainer_line(line)
        assert ev["kind"] == "learner_update" and len(ev["values"]) == n - 1 and ev["nonfinite"] == []
        assert "kl0" not in ev["values"] and throughput.parse_learner(line)["n_mb"] == ev["values"]["n_mb"]


# --------------------------------------------------------------------------- the flags


def test_every_flag_reaches_its_field_and_only_its_field():
    base = new_config.RunConfig.default()
    cases = [
        (["--clip-coef", "0.2"], "learner", "clip_coef", 0.2),
        (["--update-epochs", "1"], "learner", "update_epochs", 1),
        (["--lr", "1e-4"], "learner", "lr", 1e-4),
        (["--target-kl", "0.05"], "learner", "target_kl", 0.05),
        (["--target-kl", "none"], "learner", "target_kl", None),
        (["--ent-coef", "0.005"], "learner", "ent_coef", 0.005),
        (["--ent-coef", "0"], "learner", "ent_coef", 0.0),
        (["--lr-total-steps", "644000"], "learner", "lr_total_steps", 644_000),
        (["--lr-hold-steps", "50000"], "learner", "lr_hold_steps", 50_000),
        (["--keep-history"], "infer", "clear_history_on_sync", False),
    ]
    for argv, section, field, value in cases:
        cfg = new_train.build_config(new_train.parse_args(argv))
        assert getattr(getattr(cfg, section), field) == value, argv
        back = dataclasses.replace(cfg, **{section: dataclasses.replace(getattr(cfg, section), **{field: getattr(getattr(base, section), field)})})
        assert back == base, (argv, "something else moved")
    # the four that the update reads go through ppo_kwargs
    cfg = new_train.build_config(new_train.parse_args(["--clip-coef", "0.15", "--update-epochs", "3", "--target-kl", "none", "--ent-coef", "0.01"]))
    kw = cfg.learner.ppo_kwargs()
    assert (kw["clip_coef"], kw["update_epochs"], kw["target_kl"], kw["ent_coef"]) == (0.15, 3, None, 0.01)
    # together with the flags that were there before
    cfg = new_train.build_config(new_train.parse_args(["--mode", "ppo", "--battles", "64", "--lr", "2e-4", "--keep-history", "--lr-total-steps", "700000"]))
    assert (cfg.learner.mode, cfg.rollout.target_concurrent_battles, cfg.learner.lr, cfg.infer.clear_history_on_sync, cfg.learner.lr_total_steps) == \
        ("ppo", 64, 2e-4, False, 700_000)
    # a Namespace made by hand, as it was before these flags existed, still builds
    import argparse
    old_style = argparse.Namespace(mode="warmup", ports=None, battles=None, rooms_per_pair=None, teacher=None, steps_per_update=None, minibatch=None,
                                   seq_len=None, grad_accum=None, save_every=None, ckpt_dir=None, shaping=None, minutes=0.0, stop_file=None)
    assert new_train.build_config(old_style) == dataclasses.replace(base, learner=dataclasses.replace(base.learner, mode="warmup"))


def test_a_schedule_that_cannot_be_is_refused_at_start_only_when_asked_for(capsys):
    # the committed pair (hold 500,000, total 100,000) is such a schedule, and stays startable
    assert new_train.build_config(new_train.parse_args([])).learner.lr_total_steps == 100_000
    assert new_train.build_config(new_train.parse_args(["--lr", "1e-4", "--clip-coef", "0.2"])).learner.lr_hold_steps == 500_000
    bad = [["--lr-hold-steps", "600000"],                       # total stays 100,000
           ["--lr-total-steps", "400000"],                      # hold stays 500,000
           ["--lr-total-steps", "501000"],                      # exactly warmup + hold: still one step
           ["--lr-total-steps", "100000"],                      # the committed value, asked for by name
           ["--lr-hold-steps", "200000", "--lr-total-steps", "201000"],
           ["--lr-hold-steps", "-5", "--lr-total-steps", "9000"]]
    import argparse
    plain = vars(new_train.parse_args([]))
    for argv in bad:
        with pytest.raises(SystemExit) as e:
            new_train.parse_args(argv)
        assert e.value.code == 2, argv
        err = capsys.readouterr().err
        assert "learning-rate schedule cannot be" in err or "must not be negative" in err, (argv, err[-300:])
        # and build_config itself, which is what a caller that skips parse_args meets
        asked = {argv[i].lstrip("-").replace("-", "_"): int(argv[i + 1]) for i in range(0, len(argv), 2)}
        with pytest.raises(ValueError):
            new_train.build_config(argparse.Namespace(**{**plain, **asked}))
    good = [(["--lr-total-steps", "501001"], 500_000, 501_001), (["--lr-total-steps", "1000000"], 500_000, 1_000_000),
            (["--lr-hold-steps", "50000"], 50_000, 100_000), (["--lr-hold-steps", "0", "--lr-total-steps", "1001"], 0, 1_001),
            (["--lr-hold-steps", "143000", "--lr-total-steps", "644000"], 143_000, 644_000)]
    for argv, hold, total in good:
        lc = new_train.build_config(new_train.parse_args(argv)).learner
        assert (lc.lr_hold_steps, lc.lr_total_steps) == (hold, total), argv
    for argv in (["--target-kl", "0"], ["--target-kl", "-0.1"], ["--target-kl", "soon"], ["--clip-coef", "0"], ["--clip-coef", "1.5"],
                 ["--update-epochs", "0"], ["--lr", "0"], ["--lr", "-1e-4"], ["--ent-coef", "-0.01"]):
        with pytest.raises(SystemExit) as e:
            new_train.parse_args(argv)
        assert e.value.code == 2, argv
    capsys.readouterr()


def test_the_flags_change_what_the_learner_builds():
    """--lr and the two schedule flags, as the learner's optimizer and scheduler have them."""
    cfg = new_train.build_config(new_train.parse_args(["--mode", "ppo", "--lr", "1e-4", "--lr-hold-steps", "20000", "--lr-total-steps", "60000"]))
    cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, device=DEV, resume=False))
    la = tsu._make_learner(new_learner, new_core, cfg)
    mult = {"embeddings": 0.1, "subnets": 0.1, "transformer": 0.1, "readout": 1.0, "pi": 1.0, "v": 2.0}
    for pg, base in zip(la.opt.param_groups, la.sched.base_lrs):
        assert abs(base - 1e-4 * mult[pg["name"].split("_")[0]]) < 1e-18
    fn = la.sched.lr_lambdas[0]
    assert [fn(s) for s in (0, 999, 1_000, 21_000)] == [0.001, 1.0, 1.0, 1.0]
    assert 0.999 < fn(21_001) < 1.0 and abs(fn(60_000) - 1 / 27) < 1e-12 and abs(fn(10**7) - 1 / 27) < 1e-12
    worst = max(1 - fn(s + 1) / fn(s) for s in range(21_000, 60_000, 7))
    assert worst < 4e-4, "no step falls by more than a few hundredths of a percent"
    assert la._policy_lr() == [pg["lr"] for pg in la.opt.param_groups if pg["name"].startswith("pi_")][0]
    del la


def test_the_supervisor_passes_the_flags_through_extra_train_args():
    """A phase's extra_train_args go onto train.py's command line as they are, and train.py takes them."""
    import pipeline
    path = os.path.join(REPO, "cloud", "run_laptop_test.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    extra = ["--keep-history", "--target-kl", "none", "--lr-total-steps", "644000", "--lr-hold-steps", "143000", "--clip-coef", "0.15",
             "--update-epochs", "1", "--lr", "2e-4", "--ent-coef", "0.01"]
    raw["phases"]["ppo"]["extra_train_args"] = list(extra)
    cfg = pipeline.validate_config(raw)
    paths = pipeline.Paths(os.path.join(tsu.TEST_ROOT, "extra_args"))
    cmd = pipeline.build_trainer_command(cfg, paths, "ppo")
    assert cmd[-len(extra):] == extra
    lc_cfg = new_train.build_config(new_train.parse_args(cmd[2:]))
    lc = lc_cfg.learner
    assert (lc.mode, lc.target_kl, lc.lr_total_steps, lc.lr_hold_steps, lc.clip_coef, lc.update_epochs, lc.lr, lc.ent_coef) == \
        ("ppo", None, 644_000, 143_000, 0.15, 1, 2e-4, 0.01)
    assert lc_cfg.infer.clear_history_on_sync is False
    # the other phases of the same config are untouched
    other = new_train.build_config(new_train.parse_args(pipeline.build_trainer_command(cfg, paths, "warmup")[2:]))
    assert other.infer.clear_history_on_sync is True and other.learner.target_kl == 0.02


# --------------------------------------------------------------------------- the inference actor


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)


class FakeStore:
    def __init__(self):
        self.version, self.weights, self.temp = 0, None, 1.0
        self.get_version = _Remote(lambda: self.version)
        self.get_weights = _Remote(lambda: self.weights)
        self.get_temp = _Remote(lambda: self.temp)


class FakeRay:
    @staticmethod
    def get(x, timeout=None):
        return x


def _small(cfg, **infer_over):
    return dataclasses.replace(
        cfg,
        learner=dataclasses.replace(cfg.learner, resume=False),
        infer=dataclasses.replace(cfg.infer, device=DEV, **infer_over),
        rollout=dataclasses.replace(cfg.rollout, target_concurrent_battles=2, rooms_per_pair=2),
    )


def _new_actor(monkeypatch, store, **infer_over):
    monkeypatch.setattr(new_inf, "ray", FakeRay)
    if GPU:
        micro_ref.shared_compiled_flex_attention(new_core)
    saved = new_core.flex_attention
    try:
        actor = new_inf.InferenceActor(_small(new_config.RunConfig.default(), **infer_over), store)
    finally:
        new_core.flex_attention = saved
    actor._last_sync_time = NEVER
    return actor


def _spy(actor, log):
    inner = actor.net.forward

    def forward(obs, **kw):
        out = inner(obs, **kw)
        log.append((obs, out[0], out[1], out[2]))
        return out

    actor.net.forward = forward


def _same_pool(a, b) -> None:
    assert a._kv_slot == b._kv_slot and a._kv_free == b._kv_free
    assert torch.equal(a._kv_len, b._kv_len) and micro_ref.same_bits(a._kv_pool, b._kv_pool)


def test_default_inference_actor_is_bit_identical_to_r1(r1, data, monkeypatch):
    """
    r1's InferenceActor and the working tree's with nothing asked for, driven by the same
    random run: battles starting, ending and coming back, the pool growing, new weights and a
    new temperature arriving six times. Every answer, the logits behind it and the whole
    history pool (which rows, which lengths, which bits) must be equal throughout: with the
    option at its default new weights still wipe every history.
    """
    obs_np, _ep, _kind = data
    monkeypatch.setattr(r1.inference, "ray", FakeRay)
    monkeypatch.setattr(r1.inference.InferenceActor, "KV_POOL_MIN_ROWS", 8)
    monkeypatch.setattr(new_inf.InferenceActor, "KV_POOL_MIN_ROWS", 8)
    store = FakeStore()
    saved_old = r1.core.flex_attention
    torch.manual_seed(7)
    try:
        with r1.ref.active():
            old = r1.inference.InferenceActor(_small(r1.config.RunConfig.default()), store)
    finally:
        r1.core.flex_attention = saved_old
    new = _new_actor(monkeypatch, store)
    assert new._clear_history_on_sync is True and type(old.net) is r1.core.PokeTransformer and type(new.net) is new_core.PokeTransformer
    new.net.load_state_dict(old.net.state_dict(), strict=True)
    old._last_sync_time = NEVER
    seen_old, seen_new = [], []
    _spy(old, seen_old), _spy(new, seen_new)
    rng, nrng = random.Random(20261006), np.random.default_rng(20261006)
    steps = 700 if GPU else 120
    weight_steps = set(range(90, steps, 110)) if GPU else {30, 70, 100}
    # two sides that are in nearly every call and never end, so that a history outlives the
    # 64-turn window between two arrivals of weights and is wiped by each of them
    marathon = ["pmar000a_battle-gen9randombattle-1", "pmar000b_battle-gen9randombattle-1"]
    alive, ended, made, counts = [], [], 0, {"calls": 0, "rows": 0, "weights": 0, "ended": 0, "cleared_at_sync": 0, "max_history": 0}
    for step in range(steps):
        if rng.random() < 0.10 and alive and step not in weight_steps:
            for _ in range(rng.randint(1, 4)):
                if alive:
                    tag = alive.pop(rng.randrange(len(alive)))
                    old.clear_cache(tag), new.clear_cache(tag)
                    ended.append(tag)
                    counts["ended"] += 1
            continue
        if step in weight_steps:
            sd = old.net.state_dict()
            store.weights = _perturbed(sd, 0.02, step)
            store.version += 1
            store.temp = rng.choice([1.0, 0.8, 1.25])
            old._last_sync_time = new._last_sync_time = None
            counts["weights"] += 1
            counts["cleared_at_sync"] += len(new._kv_slot)
        tags = [m for m in marathon if rng.random() < 0.9]
        for _ in range(rng.choice([1, 2, 3, 5, 8, 13, 21, 32])):
            k = rng.random()
            if k < 0.15 or not alive:
                made += 1
                tag = f"pabc{made % 7:03d}a_battle-gen9randombattle-{made}"
            elif k < 0.19 and ended:
                tag = ended.pop(rng.randrange(len(ended)))
            else:
                tag = rng.choice(alive)
            if tag in tags:
                continue
            tags.append(tag)
            if tag not in alive:
                alive.append(tag)
        obs16 = np.ascontiguousarray(obs_np[nrng.integers(0, obs_np.shape[0], len(tags))])
        syncing = old._last_sync_time is None
        torch.manual_seed(1_000_000 + step)
        a = old.infer_batch(list(tags), obs16)
        torch.manual_seed(1_000_000 + step)
        b = new.infer_batch(list(tags), obs16)
        if syncing:
            assert old.current_version == new.current_version == store.version and old.current_temp == new.current_temp == store.temp
            assert set(new._kv_slot) == set(tags), "new weights were meant to leave only this call's sides with a history"
            assert int(new._kv_len[[new._kv_slot[t] for t in tags]].max()) == 1, "a history survived new weights"
            old._last_sync_time = new._last_sync_time = NEVER
        (o_obs, o_pi, o_v, o_ve), (n_obs, n_pi, n_v, n_ve) = seen_old.pop(), seen_new.pop()
        assert torch.equal(o_obs, n_obs) and torch.equal(o_pi, n_pi) and torch.equal(o_v, n_v) and torch.equal(o_ve, n_ve), step
        for name, x, y in zip(("actions", "log-probabilities", "values", "value variance"), a, b):
            assert micro_ref.same_bits(x, y), (name, step)
        counts["calls"] += 1
        counts["rows"] += len(tags)
        counts["max_history"] = max(counts["max_history"], int(new._kv_len.max()))
        if step % 50 == 0 or step == steps - 1 or syncing:
            _same_pool(old, new)
    print("\n" + str(counts) + f", pool rows {int(new._kv_pool.shape[1])}")
    assert counts["weights"] >= 3 and counts["cleared_at_sync"] > 20 and counts["ended"] > 20 and int(new._kv_pool.shape[1]) > 8
    if GPU:
        assert counts["max_history"] == new.net.kv_cache_len, "no side lived past the history window between two arrivals of weights"
    assert new._kv_last_use == {}, "with histories cleared nothing is kept about when a side was last seen"


def test_keep_history_is_what_was_measured_as_ii(sequences, two_weights):
    """
    bench/check_history.py first measured "(ii)" on an actor whose clear_all_caches was made
    to do nothing, before the option existed. The option must be that, on the same sequences:
    the same distribution at every decision, bit for bit, the same sampled actions and stored
    log-probabilities, and histories of the same length at the end. And both must differ
    from the actor that clears, or the test would compare nothing.
    """
    seqs, kind = sequences
    wa, wb, wkind = two_weights
    assert check_history.keep_option_exists()
    rng = np.random.default_rng(5)
    lens = [int(s.shape[0]) for s in seqs]
    sync_at = np.array([int(rng.integers(1, n)) for n in lens])
    T = int(sync_at.max())
    offsets = (T - sync_at).tolist()
    events = {T: (1, wb)}
    runs = {}
    for name, kw in (("patch", dict(keep=True, keep_impl="patch")), ("option", dict(keep=True, keep_impl="option")), ("clear", dict(keep=False))):
        runs[name] = check_history.run_actor(seqs, offsets, wa, events, device=DEV, seed=3, **kw)
    assert (runs["patch"].keep_impl, runs["option"].keep_impl, runs["clear"].keep_impl) == ("patch", "option", "clear")
    for e in range(len(seqs)):
        for field in ("lp", "act", "logp", "version", "since", "nsync"):
            assert micro_ref.same_bits(getattr(runs["patch"], field)[e], getattr(runs["option"], field)[e]), (field, e)
    assert runs["patch"].final_len == runs["option"].final_len == [min(n, 64) for n in lens]
    assert runs["clear"].final_len == [min(n - s, 64) for n, s in zip(lens, sync_at)]
    after = np.concatenate(runs["clear"].since) > 0
    lp_keep, lp_clear = np.concatenate(runs["option"].lp), np.concatenate(runs["clear"].lp)
    assert after.sum() > 100 and not np.array_equal(lp_keep[after], lp_clear[after]), "keeping and clearing gave the same answers"
    assert np.array_equal(lp_keep[~after], lp_clear[~after]), "before the sync point the two are the same actor"
    # and what the measurement said about it: next to the weights' own distribution, a kept history is far closer than a cleared one
    ref = np.concatenate(check_history.run_actor(seqs, offsets, wb, device=DEV, seed=3).lp)
    kl_keep, _ = check_history.kl_clip(lp_keep[after], ref[after])
    kl_clear, _ = check_history.kl_clip(lp_clear[after], ref[after])
    print(f"\n{kind}, weights {wkind}: {int(after.sum())} decisions after the sync; KL to the new weights' own, kept {kl_keep.mean():.6f}, cleared {kl_clear.mean():.6f}")
    if wkind == "checkpoints" and kind != "synthetic":
        assert kl_keep.mean() < 0.2 * kl_clear.mean()


def test_new_weights_leave_a_kept_history_alone_and_wipe_a_cleared_one(data, monkeypatch):
    """The pool itself: which sides hold which rows, how long each history is and what it holds, across a sync."""
    obs_np, _ep, _kind = data
    store = FakeStore()
    torch.manual_seed(11)
    keep = _new_actor(monkeypatch, store, clear_history_on_sync=False)
    clear = _new_actor(monkeypatch, store)
    clear.net.load_state_dict(keep.net.state_dict(), strict=True)
    assert keep._clear_history_on_sync is False and clear._clear_history_on_sync is True
    tags = ["A", "B", "C"]
    for turn in range(5):
        obs = np.ascontiguousarray(obs_np[10 * turn:10 * turn + 3])
        for actor in (keep, clear):
            torch.manual_seed(turn)
            actor.infer_batch(list(tags), obs)
    keep.clear_cache("C"), clear.clear_cache("C")                       # one battle ends before the sync
    _same_pool(keep, clear)
    before = (dict(keep._kv_slot), list(keep._kv_free), keep._kv_len.clone(), keep._kv_pool.clone())
    assert set(keep._kv_last_use) == {"A", "B"} and clear._kv_last_use == {}

    store.weights = _perturbed(keep.net.state_dict(), 0.02, 1)
    store.version += 1
    store.temp = 0.8
    old_w = {k: v.clone() for k, v in keep.net.state_dict().items()}
    for actor in (keep, clear):
        actor._last_sync_time = None
        actor._sync_weights()
        assert actor.current_version == 1 and actor.current_temp == 0.8
        assert any(not torch.equal(v, old_w[k]) for k, v in actor.net.state_dict().items()), "the weights did not change"
    assert (keep._kv_slot, keep._kv_free) == before[:2] and torch.equal(keep._kv_len, before[2]) and micro_ref.same_bits(keep._kv_pool, before[3])
    assert clear._kv_slot == {} and sorted(clear._kv_free) == list(range(int(clear._kv_pool.shape[1])))

    # the next decision: the kept side reads five turns of history, the cleared one none
    obs = np.ascontiguousarray(obs_np[100:102])
    for actor in (keep, clear):
        actor._last_sync_time = NEVER
        torch.manual_seed(50)
        actor.infer_batch(["A", "B"], obs)
    assert [int(keep._kv_len[keep._kv_slot[t]]) for t in ("A", "B")] == [6, 6]
    assert [int(clear._kv_len[clear._kv_slot[t]]) for t in ("A", "B")] == [1, 1]
    rows = list(keep._kv_slot.values())
    assert len(rows) == len(set(rows)) and sorted(rows + keep._kv_free) == list(range(int(keep._kv_pool.shape[1])))


def test_a_kept_history_is_given_up_when_its_side_has_gone_quiet(data, monkeypatch):
    """
    Clearing at every sync also freed the row of any side whose clear_cache never came. With
    histories kept, a side that has not asked for a decision for KV_IDLE_DROP_S gives its row
    back at the next sync, and nothing else does.
    """
    obs_np, _ep, _kind = data
    store = FakeStore()
    torch.manual_seed(12)
    actor = _new_actor(monkeypatch, store, clear_history_on_sync=False)
    obs = np.ascontiguousarray(obs_np[:3])
    for turn in range(3):
        torch.manual_seed(turn)
        actor.infer_batch(["lost", "alive", "ended"], obs)
    actor.clear_cache("ended")
    assert set(actor._kv_last_use) == {"lost", "alive"}, "a side that ended is forgotten at once"
    row_lost = actor._kv_slot["lost"]
    actor._kv_last_use["lost"] -= actor.KV_IDLE_DROP_S + 1.0            # its worker died a quarter of an hour ago
    actor._kv_last_use["alive"] -= actor.KV_IDLE_DROP_S - 60.0          # quiet, but inside the limit

    def publish():
        store.weights = {k: v.clone() for k, v in actor.net.state_dict().items()}
        store.version += 1
        actor._last_sync_time = None
        actor._sync_weights()
        actor._last_sync_time = NEVER

    publish()
    assert "lost" not in actor._kv_slot and "lost" not in actor._kv_last_use and row_lost in actor._kv_free
    assert "alive" in actor._kv_slot and int(actor._kv_len[actor._kv_slot["alive"]]) == 3
    torch.manual_seed(9)
    actor.infer_batch(["alive", "lost"], obs[:2])                       # were it to come back, it starts again from nothing
    assert int(actor._kv_len[actor._kv_slot["alive"]]) == 4 and int(actor._kv_len[actor._kv_slot["lost"]]) == 1
    publish()
    assert set(actor._kv_slot) == {"alive", "lost"}, "a sync by itself drops nobody"
    rows = list(actor._kv_slot.values())
    assert sorted(rows + actor._kv_free) == list(range(int(actor._kv_pool.shape[1])))
    assert 600.0 <= actor.KV_IDLE_DROP_S, "above the worker's own limits (60 s without an action, 300 s for a battle the server lost)"
    assert new_worker.ZOMBIE_LIMIT_S < actor.KV_IDLE_DROP_S


# --------------------------------------------------------------------------- a rollout loop that dies


class _Boom(RuntimeError):
    pass


def _worker_whose_loop_dies(after_steps: int = 0):
    """A RolloutWorker without Ray or a server: its environment raises on the given step."""

    class Env:
        calls = 0

        def step(self, actions):
            Env.calls += 1
            if Env.calls > after_steps:
                raise _Boom("the environment broke")
            time.sleep(0.01)
            return []

    w = object.__new__(new_worker.RolloutWorker)
    w.inference_actor, w.vec_env, w._traj = None, Env(), {}
    return w, Env


def test_a_dead_rollout_loop_ends_run():
    """
    The loop runs on a thread of its own. When it raised, the thread ended, run() went on
    sleeping and the actor went on answering heartbeats: train.py never learned of it and
    that worker's battles stopped without a word. run() must end with the error instead, so
    train.py's "A RolloutWorker crashed in run()" path fires. On r1 this test fails with a
    timeout: run() is still asleep six seconds after the loop has died.
    """
    w, env = _worker_whose_loop_dies(after_steps=3)

    async def go():
        return await asyncio.wait_for(w.run(), timeout=6.0)

    t0 = time.time()
    with pytest.raises(RuntimeError) as err:
        asyncio.run(go())
    assert "rollout loop" in str(err.value) and "the environment broke" in str(err.value)
    assert isinstance(err.value.__cause__, _Boom)
    assert env.calls == 4 and time.time() - t0 < 5.0


def test_a_live_rollout_loop_keeps_run_going():
    """The other half: while the loop is alive run() never returns."""
    w, env = _worker_whose_loop_dies(after_steps=10**9)

    async def go():
        return await asyncio.wait_for(w.run(), timeout=2.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
    assert env.calls > 20


def test_a_loop_that_returns_ends_run_too():
    """The loop never returns by itself. If it ever did, the worker would be just as idle: run() says so."""
    w = object.__new__(new_worker.RolloutWorker)
    w._run_rl_loop = lambda: None

    async def go():
        return await asyncio.wait_for(w.run(), timeout=6.0)

    with pytest.raises(RuntimeError, match="ended without an error"):
        asyncio.run(go())


@pytest.mark.skipif(os.environ.get("PSPPO_RAY_TESTS") != "1", reason="starts Ray: set PSPPO_RAY_TESTS=1")
def test_under_ray_a_dead_loop_makes_train_py_exit_with_1():
    """
    The whole path with real Ray actors: a RolloutWorker actor whose loop raises, and
    train.py's own telemetry loop watching its run(). It must leave with SystemExit(1) through
    the "A RolloutWorker crashed in run()" branch, which is what the supervisor restarts on.
    """
    import ray

    class BrokenWorker(new_worker.RolloutWorker):
        def __init__(self, fail_after: float):
            self._fail_after = fail_after

        def _run_rl_loop(self):
            time.sleep(self._fail_after)
            raise RuntimeError("the rollout loop broke (made up by the test)")

        def heartbeat(self):
            return {"active_battles_worker": 0}

    class Stats:
        def get_stats(self):
            return {"update": 0, "total_steps": 0}

    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        stats = ray.remote(Stats).remote()
        workers = [ray.remote(num_cpus=0)(BrokenWorker).options(max_restarts=-1, max_task_retries=0).remote(2.0)]
        refs = [w.run.remote() for w in workers]
        args = SimpleNamespace(minutes=0.0, stop_file=None)
        t0 = time.time()

        async def go():
            await asyncio.wait_for(new_train._telemetry_loop(args, time.time(), stats, stats, workers, refs), timeout=90.0)

        with pytest.raises(SystemExit) as e:
            asyncio.run(go())
        assert e.value.code == 1
        took = time.time() - t0
        assert 2.0 <= took < 60.0
        with pytest.raises(Exception, match="rollout loop of this worker died"):
            ray.get(refs[0], timeout=10)
        assert ray.get(workers[0].heartbeat.remote(), timeout=10) == {"active_battles_worker": 0}, "the actor itself is still there, as before"
        print(f"\ntrain.py's telemetry loop left with SystemExit(1) {took:.1f} s after the worker started (its loop died after 2.0 s)")
    finally:
        ray.shutdown()


# --------------------------------------------------------------------------- bench/check_history.py's clock


def test_the_boxs_clock_puts_decisions_and_syncs_where_the_arithmetic_says():
    """
    box_timeline is plain arithmetic, and every figure bench/check_history.py reports for the
    box rests on it: who decided when, with which weights, how many versions behind the learner.
    """
    rng = np.random.default_rng(1)
    lens = [1, 7, 8, 30, 31, 60, 100]
    # a learner that is never idle, an actor that takes new weights at once: 9 s over 1.3 s a decision
    offsets, events, version0, notes = check_history.box_timeline(lens, 6, 9.0, 1.3, 0.0, rng)
    assert notes["busy_seconds"] == 9.0 and notes["sync_delay_seconds"] == [0.0] * 6
    ends = [o + n - 1 for o, n in zip(offsets, lens)]
    assert min(offsets) == 0 and max(ends) - min(ends) <= 7, "every side ends inside one period of about 7 decisions"
    steps = sorted(events)
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    assert set(gaps) <= {6, 7} and abs(sum(gaps) / len(gaps) - 9.0 / 1.3) < 0.5
    assert [events[g] for g in steps] == list(range(version0 + 1, version0 + 1 + len(steps))), "versions arrive one after another"
    # the newest version the actor ever takes is one behind the learner's, and it has it before any side ends
    assert max(events.values(), default=version0) == 5 and max(steps) <= min(ends)
    # the first decision of the longest side lies 99 decisions back: far more than five periods, so it is decided with version 0
    assert version0 == 0 and offsets[lens.index(100)] == 0

    def versions_at_the_end(busy, check, seed):
        offsets, events, version0, _notes = check_history.box_timeline([20] * 4000, 6, 9.0, 1.3, check, np.random.default_rng(seed), busy)
        out = []
        for o in offsets:
            end = o + 19
            out.append(max([v for g, v in events.items() if g <= end], default=version0))
        return np.array(out)

    # never idle, weights taken at once: every side ends with the version before the learner's
    assert set(versions_at_the_end(None, 0.0, 2).tolist()) == {5}
    # the actor looks only every 5 s: some sides end before it has looked, two versions behind
    late = versions_at_the_end(None, 5.0, 2)
    assert set(late.tolist()) == {4, 5} and 0.05 < (late == 4).mean() < 0.56
    # an update that computes for 4.5 of the 9 s: half the sides end after the learner's own weights came out
    idle = versions_at_the_end(4.5, 0.0, 2)
    assert set(idle.tolist()) == {5, 6} and 0.4 < (idle == 6).mean() < 0.6


# --------------------------------------------------------------------------- after the review
#
# The review of this work found nothing wrong in the code and three things wrong in the plan
# for the live run: the schedule's one-step fall was left to land unattended (R1), the watch
# list judged the box by figures simulated on a policy it had left hours before (R2), and
# the restart was planned around a supervisor that restarts the trainer by itself with the
# arguments it read at its own start (R3). What follows holds the parts of the answers that
# are code: the arguments the restart is made with, the one line a restart is checked by,
# and what bench/check_history.py says about its own figures.

# What goes into phases.ppo.extra_train_args. --lr-total-steps is the owner's decision
# (TRAINING.md, "Learning rate"): this is the value that leaves every rate as it is until the
# hold ends at optimizer step 501,000 and then falls over 499,000 steps and not in one.
RESTART_ARGS = ["--keep-history", "--lr-total-steps", "1000000"]
# The line release r1 would print if it printed one: what a start without any new flag reads.
SETTINGS_DEFAULT = ("ppo settings: clip_coef=0.1 update_epochs=2 lr=0.0003 target_kl=0.02 ent_coef=0.02 lr_warmup_steps=1000 "
                    "lr_hold_steps=500000 lr_total_steps=100000 clear_history_on_sync=True")


def _box_command(extra):
    """train.py's command line as the supervisor builds it from cloud/run_big.json, with these extra_train_args in the PPO phase."""
    import pipeline
    with open(os.path.join(REPO, "cloud", "run_big.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["phases"]["ppo"]["extra_train_args"] = list(extra)
    cfg = pipeline.validate_config(raw)
    return pipeline.build_trainer_command(cfg, pipeline.Paths(os.path.join(tsu.TEST_ROOT, "box_command")), "ppo")


def test_the_restart_arguments_move_two_fields_and_no_rate_before_the_hold_ends(monkeypatch):
    """
    R1 and R3. The box's own config with the restart's arguments, through the supervisor's
    command builder, train.py's parser and the learner's resume, at the optimizer step the
    box's newest checkpoint here holds (130,592 at update 2575):

      * against the same config without them, only infer.clear_history_on_sync and
        learner.lr_total_steps differ, and nothing ppo_update is handed
      * every parameter group resumes at the rate it would have had without them
      * without --lr-total-steps the rate is whole at step 501,000 and 1/27 of it at 501,001
      * with it the rate is whole at 501,000, falls by less than 0.003% from any step to the
        next, is 0.24 of the full rate at 600,000 and 1/27 at 1,000,000
      * a misspelt argument passes the supervisor and is refused by train.py (exit 2), which
        on the box would be a trainer that never starts: check the arguments before the start
    """
    import pipeline  # noqa: F401  (the command builder)
    plain_cmd, cmd = _box_command([]), _box_command(RESTART_ARGS)
    assert cmd[:len(plain_cmd)] == plain_cmd and cmd[len(plain_cmd):] == RESTART_ARGS
    plain, new = new_train.build_config(new_train.parse_args(plain_cmd[2:])), new_train.build_config(new_train.parse_args(cmd[2:]))
    lc = plain.learner
    assert lc.mode == "ppo" and lc.minibatch_size % lc.batch_seq_len == 0       # whatever shape the config gives the box (768 x 1 as this is written)
    assert plain.infer.clear_history_on_sync is True and (lc.lr_warmup_steps, lc.lr_hold_steps, lc.lr_total_steps) == (1_000, 500_000, 100_000)
    assert new.infer.clear_history_on_sync is False and new.learner.lr_total_steps == 1_000_000
    back = dataclasses.replace(new, infer=plain.infer, learner=dataclasses.replace(new.learner, lr_total_steps=100_000))
    assert back == plain, "something other than the two fields moved"
    assert new.learner.ppo_kwargs() == plain.learner.ppo_kwargs()

    def learner_of(cfg):
        cfg = dataclasses.replace(cfg, learner=dataclasses.replace(cfg.learner, device=DEV, resume=False,
                                                                   ckpt_dir=os.path.join(tsu.TEST_ROOT, "ckpt_unused")))
        return tsu._make_learner(new_learner, new_core, cfg)

    step = 130_592
    run = learner_of(plain)
    run.opt.inner.step()                                   # LambdaLR wants an optimizer step before its own
    run.sched.last_epoch = step - 1
    run.sched.step()
    run.update_idx = 2575
    ck = tsu._checkpoint_in_memory(run)
    assert ck["scheduler"]["last_epoch"] == step and ck["run_cfg"]["learner"]["lr_total_steps"] == 100_000
    monkeypatch.setattr(torch, "load", lambda *a, **k: ck)
    a, b = learner_of(plain), learner_of(new)
    for la in (a, b):
        la._load_checkpoint("held in memory")
        assert la.sched.last_epoch == step and la.update_idx == 2575
    rates_a = {pg["name"]: pg["lr"] for pg in a.opt.param_groups}
    rates_b = {pg["name"]: pg["lr"] for pg in b.opt.param_groups}
    assert rates_a == rates_b, "the restart's arguments changed a rate at the resume"
    assert (a._policy_lr(), rates_a["transformer_wd"], rates_a["v_wd"]) == (3e-4, 3e-4 * 0.1, 3e-4 * 2.0), "policy, backbone and value at their full rates"
    fa, fb = a.sched.lr_lambdas[0], b.sched.lr_lambdas[0]
    for s in list(range(step, 501_001, 9_973)) + [500_999, 501_000]:
        assert fa(s) == fb(s) == 1.0, s
    assert abs(fa(501_001) - 1 / 27) < 1e-12 and abs(fa(809_000) - 1 / 27) < 1e-12, "without the flag: the whole fall in one step"
    assert 0.9999 < fb(501_001) < 1.0 and abs(fb(600_000) - 0.2403) < 1e-4 and abs(fb(1_000_000) - 1 / 27) < 1e-12
    assert 0.06 < fb(809_000) < 0.08, "where a 600M-decision run ends at 50 optimizer steps an update"
    assert max(1 - fb(s + 1) / fb(s) for s in range(500_990, 1_000_010, 11)) < 3e-5
    del run, a, b

    typo = _box_command(["--keep-histroy"])
    with pytest.raises(SystemExit) as e:
        new_train.parse_args(typo[2:])
    assert e.value.code == 2


class _StopHere(Exception):
    pass


def test_the_start_log_has_the_line_a_restart_is_checked_by(monkeypatch, caplog):
    """
    R2 and R3. Nothing the update line prints can say whether histories are kept: kl0 and
    clip0 move with the policy and with every restart. What says it is the "ppo settings:"
    line train.py logs at every start, the supervisor's planned restarts included. This runs
    train.main up to the point where it would hand the config to Ray and reads that line:
    for a start without flags, for --keep-history, and for the restart's arguments on the
    box's own command line. And the supervisor takes the line for nothing else.
    """
    import logging
    import pipeline

    class Ray:
        @staticmethod
        def init(**_k):
            return None

        @staticmethod
        def put(_cfg):
            raise _StopHere()

    monkeypatch.setattr(new_train, "ray", Ray)

    def settings_of(argv):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=new_train.logger.name):
            with pytest.raises(_StopHere):
                asyncio.run(new_train.main(new_train.parse_args(argv)))
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("ppo settings:")]
        assert len(lines) == 1, [r.getMessage() for r in caplog.records]
        header = [r.getMessage() for r in caplog.records if r.getMessage().startswith("mode=")]
        assert len(header) == 1
        stamp = "2026-10-06 03:30:01,123 [INFO] "
        assert pipeline.parse_trainer_events(stamp + lines[0]) == [], "the supervisor read the settings line as something"
        assert [e["kind"] for e in pipeline.parse_trainer_events(stamp + header[0])] == ["header"]
        return lines[0]

    assert settings_of([]) == SETTINGS_DEFAULT
    assert settings_of(["--keep-history"]) == SETTINGS_DEFAULT.replace("clear_history_on_sync=True", "clear_history_on_sync=False")
    assert settings_of(_box_command([])[2:]) == SETTINGS_DEFAULT, "the box's command line without new arguments is a default start"
    box = settings_of(_box_command(RESTART_ARGS)[2:])
    assert box == SETTINGS_DEFAULT.replace("lr_total_steps=100000", "lr_total_steps=1000000").replace("=True", "=False")
    assert box.endswith("lr_total_steps=1000000 clear_history_on_sync=False")
    # every flag of this work shows in it
    line = settings_of(["--clip-coef", "0.15", "--update-epochs", "1", "--lr", "1e-4", "--target-kl", "none", "--ent-coef", "0.01",
                        "--lr-hold-steps", "143000", "--lr-total-steps", "644000"])
    assert line == ("ppo settings: clip_coef=0.15 update_epochs=1 lr=0.0001 target_kl=None ent_coef=0.01 lr_warmup_steps=1000 "
                    "lr_hold_steps=143000 lr_total_steps=644000 clear_history_on_sync=True")


def _timeline_result(checkpoints):
    row = lambda kl, clip: {"vs_learner_training_forward": {"n": 100, "kl": kl, "clip": clip}}  # noqa: E731
    return {"period_seconds": 9.0, "busy_seconds": 9.0, "decision_seconds": 1.3, "check_seconds": 5.0, "chain_length": len(checkpoints),
            "decisions": 28378, "versions_behind_the_learner": {"1": 0.13, "2": 0.22}, "mean_versions_behind": 3.79,
            "share_after_a_sync_in_their_episode": 0.871, "checkpoints": list(checkpoints),
            "floor_same_weights_whole_history": row(0.0002, 0.003), "stale_weights_only": row(0.0233, 0.283),
            "keep": row(0.0234, 0.283), "clear": row(0.0297, 0.304)}


def test_a_simulated_timeline_says_so_and_prints_no_share_of_a_figure_it_was_not_given(capsys):
    """
    R2. The box tables of bench/check_history.py are a simulation on the checkpoints handed
    to it. They used to be printed with every row as a share "of live KL" and "of live
    clip", against 0.02 and 0.29 built in as defaults: the box's figures at update 93,
    which it had left (0.030 and 0.26) by the time the tables were read. Now the table says
    it is simulated and on which checkpoints, and prints a share only of figures it is given.
    """
    import inspect
    args = check_history.build_parser().parse_args(["--selfplay", "x.npz", "--ckpt-a", "a.pt", "--ckpt-b", "b.pt"])
    assert args.live_kl is None and args.live_clip is None, "a live figure is built in again"
    sig = inspect.signature(check_history.print_timeline).parameters
    assert sig["live_kl"].default is None and sig["live_clip"].default is None

    chain = ["learner_update_000070.pt", "learner_update_000071.pt", "learner_update_000072.pt"]
    check_history.print_timeline(_timeline_result(chain), "self-play, the chain of 3 checkpoints", args.live_kl, args.live_clip)
    out = capsys.readouterr().out
    assert out.isascii() and "SIMULATED, not measured on a running trainer" in out
    assert ", ".join(chain) in out and "the last is the learner's" in out
    assert "%" not in out.replace("87.1%", ""), "a share of a live figure nobody gave:\n" + out
    assert "of live" not in out and "today" not in out
    last_two = lambda name: next(ln for ln in out.splitlines() if ln.strip().startswith(name)).split()[-2:]  # noqa: E731
    assert last_two("history kept") == ["0.0234", "0.283"] and last_two("history cleared (the default)") == ["0.0297", "0.304"]
    assert last_two("stale weights, whole histories") == ["0.0233", "0.283"] and last_two("nothing stale, nothing cleared") == ["0.0002", "0.003"]

    # weights held at one checkpoint: named once, and said to be the cleared history alone
    check_history.print_timeline(_timeline_result(["learner_update_002575.pt"] * 8), "self-play, weights held at B")
    out = capsys.readouterr().out
    assert out.count("learner_update_002575.pt") == 1 and "the weights never change" in out

    # given a run's figures, the shares are of those, and are said not to be like for like
    check_history.print_timeline(_timeline_result(chain), "self-play", 0.030, 0.26)
    out = capsys.readouterr().out
    assert "of the kl given" in out and "--live-kl 0.03 and --live-clip 0.26" in out and "Not like for like" in out
    cleared = next(ln for ln in out.splitlines() if "history cleared" in ln).split()
    assert cleared[-2:] == ["99%", "117%"], cleared            # 0.0297 / 0.030 and 0.304 / 0.26
    assert check_history.SIMULATED in out
    # one of the two without the other is refused, not left out without a word
    for argv in (["--live-kl", "0.03"], ["--live-clip", "0.26"], ["--live-kl", "0", "--live-clip", "0.26"]):
        with pytest.raises(SystemExit) as e:
            check_history.main(["--selfplay", "x.npz", "--ckpt-a", "a.pt", "--ckpt-b", "b.pt"] + argv)
        assert e.value.code == 2, argv
    assert "together, or neither" in capsys.readouterr().err


def _metric_rows():
    """
    A metrics.jsonl as pipeline.py writes it: an earlier phase, a PPO start on release r1 (no
    kl0), then two starts on this code. In each start the first 60 updates carry the climb
    after a restart (figures far below the level), which must not reach the means.
    """
    rows, t = [], 1791215700.0

    def add(phase, segment, n, first_update, level, new_keys):
        nonlocal t
        for i in range(n):
            t += 7.9
            climb = i < 60
            rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)), "wall_time": t, "phase": phase, "update": first_update + i,
                   "segment": segment, "kl": 0.004 if climb else level["kl"] + (0.002 if i % 2 else -0.002),
                   "clip": 0.10 if climb else level["clip"], "ent": 0.44, "n_mb": 96.0 if climb else 49.0, "sec": 4.5}
            if new_keys:
                rec.update({"kl0": 0.0005 if climb else level["kl0"] + (0.006 if i % 2 else -0.006),
                            "clip0": 0.004 if climb else level["clip0"] + (0.01 if i % 3 == 0 else -0.005),
                            "stop": 0.0 if climb else 1.0, "ep": 2.0 if climb else level["ep"], "lr": 3e-4})
            rows.append(rec)

    add("warmup", 1, 30, 1, {"kl": 0.0, "clip": 0.0}, False)
    add("ppo", 2, 200, 72, {"kl": 0.030, "clip": 0.27}, False)
    add("ppo", 3, 150, 272, {"kl": 0.030, "clip": 0.27, "kl0": 0.029, "clip0": 0.262, "ep": 1.02}, True)
    add("ppo", 4, 260, 422, {"kl": 0.027, "clip": 0.26, "kl0": 0.022, "clip0": 0.247, "ep": 1.10}, True)
    return rows


def test_a_runs_own_kl0_and_clip0_are_read_per_start_over_the_same_window(capsys):
    """
    R2. What kl0 and clip0 read on the box was never measured, and a before and an after
    there is two starts of the trainer on this code, one without --keep-history and one
    with it. --metrics reads them from the supervisor's metrics.jsonl: per start, the climb
    after the restart left out, the same number of updates of each, the difference between
    the last two with its standard error. A start on release r1 has no kl0 and is not compared.
    """
    path = os.path.join(tsu.TEST_ROOT, "metrics_made_up.jsonl")
    rows = _metric_rows()
    with open(path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(rows):
            f.write(json.dumps(rec) + "\n")
            if i == 40:
                f.write("{\"time\": \"2026-10-05T16:00:00Z\", \"update\": 5, \"kl\"\n")        # a line cut short by a full disk
    read = check_history.read_metrics(path)
    assert len(read) == len(rows) and read[0]["phase"] == "warmup"
    blocks = check_history.metrics_by_start(read, skip=60, use=80)
    assert [(b["segment"], b["updates"], b["used"]) for b in blocks] == [(2, 200, 80), (3, 150, 80), (4, 260, 80)], "the warmup phase is not PPO"
    r1_start, before, after = blocks
    assert "kl0" not in r1_start["keys"] and "clip0" not in r1_start["keys"] and abs(r1_start["keys"]["kl"]["mean"] - 0.030) < 1e-12
    assert r1_start["keys"]["n_mb"]["mean"] == 49.0, "an update of the climb reached the mean"
    assert (before["first_update"], before["last_update"]) == (272, 421) and before["from"] == rows[30 + 200 + 60]["time"]
    for b, kl0, clip0, ep in ((before, 0.029, 0.262, 1.02), (after, 0.022, 0.247, 1.10)):
        k = b["keys"]
        assert k["kl0"]["n"] == 80 and abs(k["kl0"]["mean"] - kl0) < 1e-12 and abs(k["ep"]["mean"] - ep) < 1e-12 and k["stop"]["mean"] == 1.0
        assert abs(k["kl0"]["sd"] - 0.006 * (80 / 79) ** 0.5) < 1e-12 and abs(k["kl0"]["se"] - k["kl0"]["sd"] / 80 ** 0.5) < 1e-15
        want = clip0 + (0.01 * 27 - 0.005 * 53) / 80 if b is before else None
        if want is not None:
            assert abs(k["clip0"]["mean"] - want) < 1e-12        # updates 60..139 of the start: 27 of them are multiples of 3
    d = check_history.metrics_difference(before, after)
    assert abs(d["kl0"]["difference"] + 0.007) < 1e-12 and abs(d["kl0"]["se"] - (2 * before["keys"]["kl0"]["se"] ** 2) ** 0.5) < 1e-15
    assert 7.0 < d["kl0"]["in_se"] < 7.5 and abs(d["clip0"]["difference"] + 0.015) < 1e-12 and abs(d["ep"]["difference"] - 0.08) < 1e-12

    # with everything after the climb taken, the two starts are read over windows of different length, and the tool says so
    all_of = check_history.metrics_by_start(read, skip=60)
    assert [b["used"] for b in all_of] == [140, 90, 200]
    # and with nothing left out the climb is in the means: that is what --skip is for
    assert check_history.metrics_by_start(read, skip=0, use=80)[1]["keys"]["kl0"]["mean"] < 0.01

    assert check_history.main(["--metrics", path, "--skip", "60", "--use", "80"]) == 0
    out = capsys.readouterr().out
    assert out.isascii() and "start 4 against start 3" in out and "kl0 -0.0070" in out and "clip0 -0.0150" in out
    assert "ppo settings:" in out, "it has to say where the flags of each start are found"
    line2 = next(ln for ln in out.splitlines() if ln.startswith("   2 "))
    assert line2.split().count("-") == 5, "a start on r1 printed kl0, clip0, stop, ep or lr:\n" + line2
    assert check_history.main(["--metrics", path, "--skip", "60"]) == 0
    assert "windows of different length (90 and 200 updates)" in capsys.readouterr().out
    # a log of release r1 alone: nothing to compare
    only_r1 = os.path.join(tsu.TEST_ROOT, "metrics_r1_only.jsonl")
    with open(only_r1, "w", encoding="utf-8") as f:
        for rec in rows[30:230]:
            f.write(json.dumps(rec) + "\n")
    assert check_history.main(["--metrics", only_r1]) == 0
    assert "Fewer than two starts printed kl0" in capsys.readouterr().out
