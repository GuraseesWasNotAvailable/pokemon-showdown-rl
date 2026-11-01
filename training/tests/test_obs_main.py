"""
The main line's observation on this line (obs_main.py, ObsConfig.main_line, train.py --obs-main-line,
tools/widen_checkpoint.py --main-line). The proof that it is the app's observation byte for byte, and
that a widened M24381 gives M24381's logits and values bit for bit, is bench/league_exactness.py on 300
recorded battles (bench/notes/LEAGUE.md); here: everything off by default, and the pieces.

    python -m pytest tests/test_obs_main.py -q
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("poke_env")

import config  # noqa: E402
import obs_main  # noqa: E402

# git hash-object of the app's files (branch local-app 00c5373): these four are those files, byte for byte.
APP_FILES = {
    "obs_tera.py": "cbe71df9db7e55fe2177890fcc59cc57660db714",
    "obs_abil.py": "e4aa020362e181bf5205b6ab93f61720d34c51a6",
    "switch_names.py": "5634fdc3534632ff4e8e16d1058300fa276fa2d2",
    "randbats_abilities_gen9.json": "8c715e6e6a82979770ae4d0268d2a9f02db1afb7",
}


def _git_blob_sha(path: str) -> str:
    data = open(path, "rb").read().replace(b"\r\n", b"\n")      # a Windows checkout has CRLF; the blob is LF
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.mark.parametrize("name", sorted(APP_FILES))
def test_the_main_lines_files_are_the_apps_byte_for_byte(name):
    assert _git_blob_sha(os.path.join(REPO, name)) == APP_FILES[name]


def test_off_by_default_and_not_in_run_cfg():
    cfg = config.RunConfig.default()
    assert cfg.obs.main_line is False
    d = cfg.as_dict()
    assert "main_line" not in d["obs"], "a run without it saves the run_cfg it always did"
    plain = dataclasses.asdict(cfg)
    plain["obs"].pop("main_line")
    assert d == plain
    for mu, ms in ((False, False), (True, False), (True, True)):
        c = dataclasses.replace(cfg, obs=dataclasses.replace(cfg.obs, matchup=mu, movesets=ms))
        assert "main_line" not in c.as_dict()["obs"]
        here = os.getcwd()
        os.chdir(REPO)
        try:
            asm = c.make_assembler()
        finally:
            os.chdir(here)
        from obs_assembler import ObservationAssembler
        assert type(asm) is ObservationAssembler and asm.own_tera_type is True and asm.matchup is mu and asm.movesets is ms


def test_on_it_is_in_run_cfg_and_needs_own_tera_type_off():
    cfg = config.RunConfig.default()
    with pytest.raises(ValueError):
        dataclasses.replace(cfg.obs, main_line=True)            # own_tera_type is True by default
    obs = dataclasses.replace(cfg.obs, own_tera_type=False, main_line=True, matchup=True, movesets=True)
    on = dataclasses.replace(cfg, obs=obs)
    assert on.as_dict()["obs"]["main_line"] is True and on.as_dict()["obs"]["own_tera_type"] is False


def test_the_assembler_is_obs_abil_on_this_lines_base_with_the_blocks(monkeypatch):
    monkeypatch.chdir(REPO)
    from obs_abil import AbilityObservationAssembler
    from obs_tera import OwnTeraObservationAssembler
    from obs_assembler import ObservationAssembler
    for mu, ms, dim in ((False, False, obs_main.MAIN_LINE_DIM), (True, False, None), (True, True, None)):
        asm = obs_main.make_assembler(mu, ms)
        mro = type(asm).__mro__
        assert mro.index(AbilityObservationAssembler) < mro.index(OwnTeraObservationAssembler) < mro.index(obs_main.MainLineBase) \
            < mro.index(ObservationAssembler)
        assert asm.own_tera_type is True, "obs_tera's own Tera type is on"
        assert asm.matchup is mu and asm.movesets is ms
        assert asm.settings == {k: True for k in obs_main.MAIN_LINE_SETTINGS if k != "own_tera_type"}
        # the first part's layout is the base layout: the blocks only come after it
        assert asm.offsets["action_mask"][1] == obs_main.MAIN_LINE_DIM
        if dim is not None:
            assert asm.total_dim == dim
        ref = ObservationAssembler(matchup=mu, movesets=ms)
        assert asm.total_dim == ref.total_dim and asm.offsets == ref.offsets and asm.meta == ref.meta
    a, b = obs_main.assembler_class(True, True), obs_main.assembler_class(True, True)
    assert a is b, "one class per set of blocks"
    with pytest.raises(ValueError):
        obs_main.assembler_class(False, True)


def test_this_lines_own_tera_type_and_stab_restore_never_run():
    base = obs_main.MainLineBase.__new__(obs_main.MainLineBase)
    assert base._write_own_tera_type(object(), 0) is None and base._restore_original_stab(object(), 0) is None


def test_make_assembler_follows_the_config(monkeypatch):
    monkeypatch.chdir(REPO)
    cfg = config.RunConfig.default()
    obs = dataclasses.replace(cfg.obs, own_tera_type=False, main_line=True, matchup=True, movesets=True)
    asm = dataclasses.replace(cfg, obs=obs).make_assembler()
    assert getattr(asm, "main_line", False) and asm.movesets


def test_checkpoint_helpers():
    full = dict(obs_main.MAIN_LINE_SETTINGS)
    assert obs_main.checkpoint_has_main_settings({"obs_settings": full})
    assert not obs_main.checkpoint_has_main_settings({"obs_settings": dict(full, regenerator=False)})
    assert not obs_main.checkpoint_has_main_settings({})
    assert obs_main.checkpoint_main_line({"run_cfg": {"obs": {"main_line": True}}})
    assert not obs_main.checkpoint_main_line({"run_cfg": {"obs": {}}}) and not obs_main.checkpoint_main_line({})
    s = obs_main.checkpoint_obs_settings(True, True)
    assert s["main_line"] and s["matchup"] and s["movesets"] and all(s[k] for k in obs_main.MAIN_LINE_SETTINGS)
    # the window app refuses settings it does not build (its hybrid.unknown_obs_settings knows the main line's seven)
    assert {"main_line", "matchup", "movesets"} <= set(s) - set(obs_main.MAIN_LINE_SETTINGS)


def test_train_py_flag():
    import train
    off = train.build_config(train.parse_args([]))
    assert off.obs.main_line is False and off.obs.own_tera_type is True
    assert "obs_main_line" not in train.features_line(off)
    on = train.build_config(train.parse_args(["--obs-matchup", "--obs-movesets", "--obs-main-line"]))
    assert on.obs.main_line is True and on.obs.own_tera_type is False and on.obs.matchup and on.obs.movesets
    assert train.features_line(on).endswith(" obs_movesets=True obs_main_line=True")
    with pytest.raises(SystemExit):
        train.parse_args(["--obs-main-line", "--own-tera-type"])


def test_check_restart_names_main_line_only_when_on():
    sys.path.insert(0, os.path.join(REPO, "cloud"))
    import check_restart as cr
    ck = {"run_cfg": {"obs": {"own_tera_type": True, "matchup": True, "movesets": True}}}
    assert "main_line" not in cr.checkpoint_obs_settings(ck)
    ck_on = {"run_cfg": {"obs": {"own_tera_type": False, "matchup": True, "movesets": True, "main_line": True}}}
    assert cr.checkpoint_obs_settings(ck_on)["main_line"] is True
    import train
    on = train.build_config(train.parse_args(["--obs-matchup", "--obs-movesets", "--obs-main-line"]))
    assert cr.obs_settings_of_config(on) == {"own_tera_type": False, "matchup": True, "movesets": True, "main_line": True}
    off = train.build_config(train.parse_args(["--obs-matchup", "--obs-movesets"]))
    assert "main_line" not in cr.obs_settings_of_config(off)
    assert cr.obs_change(cr.checkpoint_obs_settings(ck_on), cr.obs_settings_of_config(on)) == ([], [])


def test_the_widen_tool_refuses_what_is_not_the_main_line():
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import widen_checkpoint as wc
    with pytest.raises(ValueError):
        wc.widen_main_line({"model": {}, "run_cfg": {"obs": {"matchup": True}}})
    with pytest.raises(ValueError):
        wc.widen_main_line({"model": {}, "run_cfg": {"obs": {}}, "obs_settings": {"own_tera_type": True}})
    import torch
    assert wc.ff_expansion_of({"transformer.0.ff.0.weight": torch.zeros(1024, 512)}) == 2.0
    assert wc.ff_expansion_of({}) is None
