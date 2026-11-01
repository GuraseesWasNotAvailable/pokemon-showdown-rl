"""
tools/widen_checkpoint.py: a checkpoint trained without the matchup block, widened to take it,
must play exactly like the original on day one.

    PSPPO_STREAMS=<streams.jsonl> CUDA_VISIBLE_DEVICES=3 python -m pytest tests/test_widen.py -q -p no:cacheprovider -s

    PSPPO_WIDEN_CKPT   the checkpoint (default: ckpt_armc.pt in the repository, a copy of the full
                       run's arm C, search observation with own Tera types)
    PSPPO_STREAMS      recorded player streams (bench/micro_showdown_sim.py --record N)

Checked:
  * the widened checkpoint: every tensor as it was, the new columns of move_net.0 and
    pokemon_net.0 zero in the model, the target model, the fp32 masters and both Adam moments;
    counters, scheduler and RNG states untouched; the setting recorded;
  * 2,000+ recorded decisions (whole sides of battles, each stepped through with its own
    history the way the app and the inference actor do it, and packed the way the learner
    trains): the original network on the observation without the block and the widened one on
    the observation with it give the same logits, value logits and values: bit for bit in fp32,
    within 1e-3 in bf16 (they are bit for bit there too, and the test says so);
  * the block does reach the logits once its columns are not zero.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(REPO)

import numpy as np  # noqa: E402

torch = pytest.importorskip("torch")

import obs_matchup as om  # noqa: E402
import widen_checkpoint as wc  # noqa: E402
from obs_assembler import ObservationAssembler  # noqa: E402

CKPT = os.environ.get("PSPPO_WIDEN_CKPT") or os.path.join(REPO, "ckpt_armc.pt")
MIN_DECISIONS = 2000
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def original():
    if not os.path.isfile(CKPT):
        pytest.skip(f"no checkpoint at {CKPT} (PSPPO_WIDEN_CKPT)")
    return torch.load(CKPT, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def widened(original):
    return wc.widen(original, source=CKPT)


@pytest.fixture(scope="module")
def widened_sh(original):
    return wc.widen(original, set_head=True, source=CKPT)


def _same(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def _check_widened_tensor(old, new, name):
    assert new.dtype == old.dtype and new.shape[0] == old.shape[0] and new.shape[1] > old.shape[1], name
    k0 = old.shape[1]
    assert torch.equal(new[:, :k0], old), name
    assert not new[:, k0:].any(), name


# --------------------------------------------------------------------------- the checkpoint itself


def test_widened_checkpoint_contents(original, widened):
    assert set(widened) == set(original) | {"widened"}
    for part in ("model", "target_model"):
        o, w = original[part], widened[part]
        assert list(o) == list(w), part
        for k in o:
            if k in wc.WIDENED:
                _check_widened_tensor(o[k], w[k], f"{part}.{k}")
            else:
                assert _same(o[k], w[k]), f"{part}.{k}"
    assert widened["model"]["move_net.0.weight"].shape[1] - original["model"]["move_net.0.weight"].shape[1] == 18
    assert widened["model"]["pokemon_net.0.weight"].shape[1] - original["model"]["pokemon_net.0.weight"].shape[1] == 99

    oo, ow = original["optimizer"], widened["optimizer"]
    assert len(oo["masters"]) == len(ow["masters"])
    n_wide = 0
    for a, b in zip(oo["masters"], ow["masters"]):
        if a.shape == b.shape:
            assert torch.equal(a, b)
        else:
            _check_widened_tensor(a, b, "master")
            n_wide += 1
    assert n_wide == 2
    assert _same(oo["inner"]["param_groups"], ow["inner"]["param_groups"])
    so, sw = oo["inner"]["state"], ow["inner"]["state"]
    assert sorted(so) == sorted(sw)
    n_wide = 0
    for i in so:
        assert so[i].keys() == sw[i].keys()
        for k, v in so[i].items():
            if torch.is_tensor(v) and v.dim() == 2 and v.shape != sw[i][k].shape:
                _check_widened_tensor(v, sw[i][k], f"state {i} {k}")
                n_wide += 1
            else:
                assert _same(v, sw[i][k]), (i, k)
    assert n_wide == 4                                    # exp_avg and exp_avg_sq of the two widened weights
    for k in ("scheduler", "update_idx", "total_steps", "total_episodes", "torch_rng", "numpy_rng"):
        assert _same(original[k], widened[k]), k
    rc_o, rc_w = original["run_cfg"], widened["run_cfg"]
    assert rc_w["obs"] == {**rc_o["obs"], "matchup": True}
    assert {k: v for k, v in rc_w.items() if k != "obs"} == {k: v for k, v in rc_o.items() if k != "obs"}
    assert om.checkpoint_obs_matchup(widened) and not om.checkpoint_obs_matchup(original)
    assert not om.checkpoint_set_head(widened)
    print("\n" + json.dumps(widened["widened"], indent=2))
    with pytest.raises(ValueError):
        wc.widen(widened)                                 # twice is refused


def test_widened_with_a_set_head(original, widened, widened_sh):
    m = widened_sh["model"]
    head = [k for k in m if k.startswith("set_head.")]
    assert head == ["set_head.ability.weight", "set_head.ability.bias", "set_head.item.weight", "set_head.item.bias"]
    assert all(_same(m[k], widened["model"][k]) for k in widened["model"])
    assert all(torch.equal(widened_sh["target_model"][k], m[k]) for k in head)
    assert len(widened_sh["optimizer"]["masters"]) == len(widened["optimizer"]["masters"]) + 4
    assert om.checkpoint_set_head(widened_sh)
    # the learner's optimizer takes it, with the head's masters equal to its weights and no Adam state for them yet
    cfg = wc.run_config_for(original)
    with wc.in_repo_root():
        net = om.build_model(cfg, matchup=True, set_head=True)
    net.load_state_dict(m, strict=True)
    opt = wc.learner_optimizer(net, cfg.learner)
    opt.load_state_dict(widened_sh["optimizer"])
    names = {id(p): n for n, p in net.named_parameters()}
    for p, master in opt.pairs:
        if names[id(p)].startswith("set_head."):
            assert torch.equal(master, m[names[id(p)]].float()) and not opt.inner.state.get(master)


def test_the_tool_writes_the_same_checkpoint(widened, tmp_path):
    out = tmp_path / "widened.pt"
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "widen_checkpoint.py"), "--in", CKPT, "--out", str(out)],
                       capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    got = torch.load(str(out), map_location="cpu", weights_only=False)
    for part in ("model", "target_model", "optimizer", "scheduler", "run_cfg"):
        assert _same(got[part], widened[part]), part
    r2 = subprocess.run([sys.executable, os.path.join(REPO, "tools", "widen_checkpoint.py"), "--in", CKPT, "--out", str(out)],
                        capture_output=True, text=True, cwd=str(tmp_path))
    assert r2.returncode != 0 and "not overwritten" in (r2.stdout + r2.stderr)


# --------------------------------------------------------------------------- the network on recorded decisions


@pytest.fixture(scope="module")
def decisions():
    import matchup_replay
    if not matchup_replay.stream_paths():
        pytest.skip("no recorded battles: set PSPPO_STREAMS")
    out = matchup_replay.replay({"matchup": ObservationAssembler(own_tera_type=True, matchup=True)})
    recs = out["records"]
    plain = matchup_replay.sequences(recs, "default", MIN_DECISIONS)
    block = matchup_replay.sequences(recs, "matchup", MIN_DECISIONS)
    assert sum(len(s) for s in plain) >= MIN_DECISIONS
    return plain, block


def _run_stepwise(net, seqs):
    """Each side of a battle stepped through with its own history (kv cache), all sides at once. Returns (pi, v, v_exp) per decision, in sequence order."""
    n, T = len(seqs), max(len(s) for s in seqs)
    L, K, D = net.n_layers, net.kv_cache_len, net.d_model
    kv = torch.zeros(L, n, K, D, device=DEV, dtype=net.recurrent_dtype())
    kv_len = torch.zeros(n, dtype=torch.long, device=DEV)
    got = [[None] * len(s) for s in seqs]
    with torch.no_grad():
        for t in range(T):
            idx = [i for i in range(n) if len(seqs[i]) > t]
            ii = torch.tensor(idx, device=DEV)
            x = torch.from_numpy(np.stack([seqs[i][t] for i in idx]).astype(np.float32)).to(DEV)
            pi, v, ve, new_kv, new_len = net(x, kv_cache=[kv[l, ii] for l in range(L)], kv_cache_len=kv_len[ii])
            for l in range(L):
                kv[l, ii] = new_kv[l].to(kv.dtype)
            kv_len[ii] = new_len
            for j, i in enumerate(idx):
                got[i][t] = (pi[j], v[j], ve[j])
    flat = [g for s in got for g in s]
    return tuple(torch.stack([g[k] for g in flat]) for k in range(3))


def _run_packed(net, seqs, L=32, per_call=4):
    """The learner's path: whole episodes back to back, cut into rows of L decisions, with episode ids."""
    rows = np.concatenate(seqs).astype(np.float32)
    ep = np.concatenate([np.full(len(s), i, dtype=np.int64) for i, s in enumerate(seqs)])
    pad = (-len(rows)) % L
    if pad:
        rows = np.concatenate([rows, np.repeat(rows[-1:], pad, axis=0)])
        ep = np.concatenate([ep, np.full(pad, len(seqs), dtype=np.int64)])
    outs = []
    with torch.no_grad():
        for s in range(0, len(rows), L * per_call):
            x = torch.from_numpy(rows[s:s + L * per_call]).to(DEV)
            e = torch.from_numpy(ep[s:s + L * per_call]).to(DEV)
            pi, v, ve, _a, _b = net(x, episode_ids=e, batch_seq_len=L)
            outs.append((pi, v, ve))
    n = sum(len(s) for s in seqs)
    return tuple(torch.cat([o[k] for o in outs])[:n] for k in range(3))


def _diff(a, b) -> float:
    return max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, b))


def _bits(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_widened_network_gives_identical_logits_and_values(original, widened, widened_sh, decisions):
    plain, block = decisions
    n = sum(len(s) for s in plain)
    cfg = wc.run_config_for(original)
    with wc.in_repo_root():
        old = om.build_model(cfg, matchup=False)
        new = om.build_model(cfg, matchup=True)
        new_sh = om.build_model(cfg, matchup=True, set_head=True)
    old.load_state_dict({k: v for k, v in original["model"].items() if "attn_mask" not in k}, strict=True)
    new.load_state_dict(widened["model"], strict=True)
    new_sh.load_state_dict(widened_sh["model"], strict=True)
    old, new, new_sh = (m.float().to(DEV).eval() for m in (old, new, new_sh))
    report = {"decisions": n, "battle sides": len(plain), "device": DEV}

    # fp32, stepped (the app's path): bit for bit
    a = _run_stepwise(old, plain)
    a2 = _run_stepwise(old, plain)
    assert _bits(a, a2), "the original network is not repeatable on this device: nothing to compare against"
    b = _run_stepwise(new, block)
    report["fp32 stepped max |diff|"] = _diff(a, b)
    assert _bits(a, b), report
    assert _bits(a, _run_stepwise(new_sh, block)), "the SetHead changed the logits"
    # fp32, packed (the learner's layout)
    ap, bp = _run_packed(old, plain), _run_packed(new, block)
    report["fp32 packed max |diff|"] = _diff(ap, bp)
    assert _bits(ap, bp), report

    # the block is cut off by the zero columns, and reaches the logits once they are not zero
    scrambled = [s.copy() for s in block]
    dim0 = scrambled[0].shape[1] - om.BLOCK_DIM
    rng = np.random.default_rng(0)
    for s in scrambled:
        s[:, dim0:] = rng.integers(0, 3, size=s[:, dim0:].shape).astype(np.float16)
    assert _bits(a, _run_stepwise(new, scrambled))
    with torch.no_grad():
        for lin in (new.move_net[0], new.pokemon_net[0]):
            k0 = lin.weight.shape[1] - (new.mu_move_in if lin is new.move_net[0] else new.mu_pok_in)
            lin.weight[:, k0:] = torch.randn_like(lin.weight[:, k0:]) * 0.02
    moved = _run_stepwise(new, block)
    report["with non-zero block columns, max |logit diff|"] = _diff(a[:1], moved[:1])
    assert _diff(a[:1], moved[:1]) > 1e-4
    new.load_state_dict(widened["model"], strict=True)

    # bf16 (the inference actor's and the learner's precision): within 1e-3
    old.enable_bf16_recurrent_path()
    new.enable_bf16_recurrent_path()
    a16, b16 = _run_stepwise(old, plain), _run_stepwise(new, block)
    ap16, bp16 = _run_packed(old, plain), _run_packed(new, block)
    report["bf16 stepped max |diff|"] = _diff(a16, b16)
    report["bf16 packed max |diff|"] = _diff(ap16, bp16)
    report["bf16 bit for bit"] = _bits(a16, b16) and _bits(ap16, bp16)
    print("\n" + json.dumps(report, indent=2))
    assert report["bf16 stepped max |diff|"] <= 1e-3 and report["bf16 packed max |diff|"] <= 1e-3


# =========================================================================== --movesets: a matchup checkpoint widened to the moveset block
#
#   PSPPO_WIDEN_MS_CKPT     a checkpoint WITH the matchup block (e.g. the newest of the from-scratch run)
#   PSPPO_WIDEN_MS_SH_CKPT  optional: one with the matchup block and a SetHead (and an OppActionHead): a PPO
#                           checkpoint of a run with --set-head-coef (and --opp-action-aux-coef)

import obs_movesets as oms  # noqa: E402

MS_CKPT = os.environ.get("PSPPO_WIDEN_MS_CKPT", "")
MS_SH_CKPT = os.environ.get("PSPPO_WIDEN_MS_SH_CKPT", "")


def _load_or_skip(path, what):
    if not path or not os.path.isfile(path):
        pytest.skip(f"no {what}")
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def mu_original():
    ck = _load_or_skip(MS_CKPT, "matchup checkpoint (PSPPO_WIDEN_MS_CKPT)")
    assert om.checkpoint_obs_matchup(ck) and not oms.checkpoint_obs_movesets(ck)
    return ck


@pytest.fixture(scope="module")
def ms_widened(mu_original):
    return wc.widen(mu_original, movesets=True, source=MS_CKPT)


def _check_ms_contents(original, widened, n_wide_state, by_index=True):
    """Everything as it was but pokemon_net.0.weight (103 zero columns) and the SetHead's new layers.
    by_index: the optimizer state compared position by position (only when no parameter was added)."""
    assert set(widened) == set(original) | {"widened_movesets"}
    for part in ("model", "target_model"):
        o, w = original[part], widened[part]
        for k in o:
            if k == "pokemon_net.0.weight":
                _check_widened_tensor(o[k], w[k], f"{part}.{k}")
                assert w[k].shape[1] - o[k].shape[1] == 64 + 39
            elif "attn_mask" not in k:
                assert _same(o[k], w[k]), f"{part}.{k}"
        assert all(k.startswith(("set_head.moves.", "set_head.tera.")) for k in w if k not in o), part
    if by_index:
        so, sw = original["optimizer"]["inner"]["state"], widened["optimizer"]["inner"]["state"]
        n_wide = 0
        for i in so:
            assert so[i].keys() == sw[i].keys()
            for k, v in so[i].items():
                if torch.is_tensor(v) and v.dim() == 2 and v.shape != sw[i][k].shape:
                    _check_widened_tensor(v, sw[i][k], f"state {i} {k}")
                    n_wide += 1
                else:
                    assert _same(v, sw[i][k]), (i, k)
        assert n_wide == n_wide_state                              # exp_avg and exp_avg_sq of pokemon_net.0.weight
    for k in ("scheduler", "update_idx", "total_steps", "total_episodes"):
        assert _same(original[k], widened[k]), k
    rc_o, rc_w = original["run_cfg"], widened["run_cfg"]
    assert rc_w["obs"] == {**rc_o["obs"], "movesets": True}
    assert {k: v for k, v in rc_w.items() if k != "obs"} == {k: v for k, v in rc_o.items() if k != "obs"}
    assert oms.checkpoint_obs_movesets(widened) and om.checkpoint_obs_matchup(widened)
    print("\n" + json.dumps(widened["widened_movesets"], indent=2))


def test_movesets_widened_checkpoint_contents(mu_original, ms_widened):
    o_masters, w_masters = mu_original["optimizer"]["masters"], ms_widened["optimizer"]["masters"]
    assert len(o_masters) == len(w_masters)
    assert _same(mu_original["optimizer"]["inner"]["param_groups"], ms_widened["optimizer"]["inner"]["param_groups"])
    n = 0
    for a, b in zip(o_masters, w_masters):
        if a.shape == b.shape:
            assert torch.equal(a, b)
        else:
            _check_widened_tensor(a, b, "master")
            n += 1
    assert n == 1
    _check_ms_contents(mu_original, ms_widened, 2)
    assert ms_widened["model"]["move_net.0.weight"].shape == mu_original["model"]["move_net.0.weight"].shape
    with pytest.raises(ValueError):
        wc.widen(ms_widened, movesets=True)                        # twice is refused
    no_mu = dict(mu_original, run_cfg={**mu_original["run_cfg"], "obs": {**mu_original["run_cfg"]["obs"], "matchup": False}})
    with pytest.raises(ValueError):
        wc.widen(no_mu, movesets=True)                             # the matchup block first


def test_movesets_widen_keeps_a_set_head_and_adds_its_new_layers():
    original = _load_or_skip(MS_SH_CKPT, "matchup checkpoint with a SetHead (PSPPO_WIDEN_MS_SH_CKPT)")
    assert om.checkpoint_set_head(original)
    w = wc.widen(original, movesets=True, source=MS_SH_CKPT)
    m = w["model"]
    old_head = [k for k in original["model"] if k.startswith("set_head.")]
    new_head = [k for k in m if k.startswith("set_head.")]
    assert new_head == old_head + ["set_head.moves.weight", "set_head.moves.bias", "set_head.tera.weight", "set_head.tera.bias"]
    for k in old_head:
        assert torch.equal(m[k], original["model"][k]), k
    assert not m["set_head.moves.bias"].any() and float(m["set_head.moves.weight"].std()) > 0
    _check_ms_contents(original, w, 2, by_index=False)
    # the learner's optimizer takes it: the old head's masters and Adam state kept, the new layers fresh
    cfg = wc.run_config_for(original)
    had_opp = any(k.startswith("opp_action_head.") for k in original["model"])
    with wc.in_repo_root():
        net = om.build_model(cfg, matchup=True, movesets=True, set_head=True, opp_action_head=had_opp)
        old_net = om.build_model(cfg, matchup=True, set_head=True, opp_action_head=had_opp)
    net.load_state_dict(m, strict=True)
    old_net.load_state_dict({k: v for k, v in original["model"].items() if "attn_mask" not in k}, strict=True)
    opt = wc.learner_optimizer(net, cfg.learner)
    opt.load_state_dict(w["optimizer"])
    opt_old = wc.learner_optimizer(old_net, cfg.learner)
    opt_old.load_state_dict(original["optimizer"])
    old_names = {id(p): n for n, p in old_net.named_parameters()}
    old_state = {old_names[id(p)]: opt_old.inner.state.get(mm, {}) for p, mm in opt_old.pairs}
    names = {id(p): n for n, p in net.named_parameters()}
    n_kept = n_fresh = n_wide = 0
    for p, master in opt.pairs:
        name = names[id(p)]
        st = opt.inner.state.get(master, {})
        assert torch.equal(master.to(m[name].dtype), m[name]), name
        if name.startswith(("set_head.moves", "set_head.tera")):
            assert not st, name
            n_fresh += 1
            continue
        old = old_state[name]
        assert st.keys() == old.keys(), name
        for k, v in old.items():
            if torch.is_tensor(v) and v.dim() == 2 and v.shape != st[k].shape:
                assert name == "pokemon_net.0.weight"
                _check_widened_tensor(v, st[k], f"{name} {k}")
                n_wide += 1
            else:
                assert _same(v, st[k]), (name, k)
        n_kept += name.startswith("set_head.") and bool(old)
    assert n_kept == 4 and n_fresh == 4 and n_wide == 2
    # repeatable: the new layers are seeded
    w2 = wc.widen(original, movesets=True, source=MS_SH_CKPT)
    assert torch.equal(w2["model"]["set_head.moves.weight"], m["set_head.moves.weight"])


def test_the_tool_writes_the_same_movesets_checkpoint(ms_widened, tmp_path):
    out = tmp_path / "widened_ms.pt"
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "widen_checkpoint.py"), "--movesets", "--in", MS_CKPT,
                        "--out", str(out)], capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    got = torch.load(str(out), map_location="cpu", weights_only=False)
    for part in ("model", "target_model", "optimizer", "scheduler", "run_cfg"):
        assert _same(got[part], ms_widened[part]), part
    print("\n" + r.stdout.strip())


@pytest.fixture(scope="module")
def ms_decisions():
    import matchup_replay
    if not matchup_replay.stream_paths():
        pytest.skip("no recorded battles: set PSPPO_STREAMS")
    out = matchup_replay.replay({"matchup": ObservationAssembler(own_tera_type=True, matchup=True),
                                 "movesets": ObservationAssembler(own_tera_type=True, matchup=True, movesets=True)})
    recs = out["records"]
    return (matchup_replay.sequences(recs, "matchup", MIN_DECISIONS), matchup_replay.sequences(recs, "movesets", MIN_DECISIONS))


def test_movesets_widened_network_gives_identical_logits_and_values(mu_original, ms_widened, ms_decisions):
    mu_obs, ms_seqs = ms_decisions
    # With the moveset block the matchup block's move tokens use every move's full type (abilities, forms,
    # weather): the observation before the block is the matchup-only one but for those tokens. The original
    # network reads that prefix; the widened one the whole observation.
    m0 = ObservationAssembler(own_tera_type=True, matchup=True).offsets["matchup"][0]
    dim0 = ms_seqs[0].shape[1] - oms.BLOCK_DIM
    mu_seqs = [np.ascontiguousarray(s[:, :dim0]) for s in ms_seqs]
    changed = 0
    for a_, b_ in zip(mu_obs, mu_seqs):
        assert a_.shape == b_.shape
        diff = np.flatnonzero((a_.view(np.uint16) != b_.view(np.uint16)).any(axis=0))
        assert ((diff >= m0) & (diff < m0 + om.MOVE_BLOCK)).all(), diff[:10]
        changed += int((a_.view(np.uint16) != b_.view(np.uint16)).any(axis=1).sum())
    n = sum(len(s) for s in mu_seqs)
    cfg = wc.run_config_for(mu_original)
    with wc.in_repo_root():
        old = om.build_model(cfg, matchup=True)
        new = om.build_model(cfg, matchup=True, movesets=True)
        new_sh = om.build_model(cfg, matchup=True, movesets=True, set_head=True)
    old.load_state_dict({k: v for k, v in mu_original["model"].items() if "attn_mask" not in k}, strict=True)
    new.load_state_dict(ms_widened["model"], strict=True)
    sh_sd = new_sh.state_dict()
    sh_sd.update(ms_widened["model"])
    new_sh.load_state_dict(sh_sd, strict=True)
    old, new, new_sh = (m.float().to(DEV).eval() for m in (old, new, new_sh))
    report = {"decisions": n, "battle sides": len(mu_seqs), "device": DEV,
              "decisions whose matchup move tokens change with the full types": changed}

    # fp32, stepped (the app's path) and packed (the learner's): bit for bit
    a = _run_stepwise(old, mu_seqs)
    assert _bits(a, _run_stepwise(old, mu_seqs)), "the original network is not repeatable on this device"
    b = _run_stepwise(new, ms_seqs)
    report["fp32 stepped max |diff|"] = _diff(a, b)
    assert _bits(a, b), report
    assert _bits(a, _run_stepwise(new_sh, ms_seqs)), "the SetHead changed the logits"
    ap, bp = _run_packed(old, mu_seqs), _run_packed(new, ms_seqs)
    report["fp32 packed max |diff|"] = _diff(ap, bp)
    assert _bits(ap, bp), report

    # the block is cut off by the zero columns, and reaches the logits once they are not zero
    scrambled = [s.copy() for s in ms_seqs]
    rng = np.random.default_rng(0)
    for s in scrambled:
        s[:, dim0:] = rng.integers(0, 3, size=s[:, dim0:].shape).astype(np.float16)
    assert _bits(a, _run_stepwise(new, scrambled))
    with torch.no_grad():
        lin = new.pokemon_net[0]
        lin.weight[:, -new.ms_pok_in:] = torch.randn_like(lin.weight[:, -new.ms_pok_in:]) * 0.02
    moved = _run_stepwise(new, ms_seqs)
    report["with non-zero block columns, max |logit diff|"] = _diff(a[:1], moved[:1])
    assert _diff(a[:1], moved[:1]) > 1e-4
    new.load_state_dict(ms_widened["model"], strict=True)

    # bf16 (the inference actor's and the learner's precision)
    old.enable_bf16_recurrent_path()
    new.enable_bf16_recurrent_path()
    a16, b16 = _run_stepwise(old, mu_seqs), _run_stepwise(new, ms_seqs)
    ap16, bp16 = _run_packed(old, mu_seqs), _run_packed(new, ms_seqs)
    report["bf16 stepped max |diff|"] = _diff(a16, b16)
    report["bf16 packed max |diff|"] = _diff(ap16, bp16)
    report["bf16 bit for bit"] = _bits(a16, b16) and _bits(ap16, bp16)
    print("\n" + json.dumps(report, indent=2))
    assert report["bf16 stepped max |diff|"] <= 1e-3 and report["bf16 packed max |diff|"] <= 1e-3
