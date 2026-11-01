"""
The old code the other tests/test_speed_*.py files compare against (bench/micro_ref.py), on a
machine that cannot ask git for it.

The cloud box gets the code as a tar file, without .git. Reference.export() used to raise
there even with every old file already in place, and every comparison with the old code was
skipped: a run of the test files came back green having compared nothing. Now a directory
that holds the reference commit's files is taken, but only when each file has exactly the
bytes git names in micro_ref.REF_BLOBS, so nothing else can pass for the old code.

    python -m pytest tests/test_speed_ref.py -q -p no:cacheprovider

No GPU, no Ray, no servers. Nothing is removed: the small directories the tests write stay
under $PSPPO_TEST_DIR (default: a new directory in the system temp folder).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from bench import micro_ref  # noqa: E402

TEST_ROOT = os.environ.get("PSPPO_TEST_DIR") or tempfile.mkdtemp(prefix="psppo_speed_test_")
os.makedirs(TEST_ROOT, exist_ok=True)
_made = [0]


def _new_dir(label: str) -> str:
    _made[0] += 1
    d = os.path.join(TEST_ROOT, f"ref_{label}_{os.getpid()}_{int(time.time())}_{_made[0]}")
    os.makedirs(d)
    return d


def _copy_reference(src: str, label: str, leave_out=()) -> str:
    d = _new_dir(label)
    for m in micro_ref.MODULES:
        if m in leave_out:
            continue
        with open(os.path.join(src, m + ".py"), "rb") as f:
            data = f.read()
        with open(os.path.join(d, m + ".py"), "wb") as f:
            f.write(data)
    return d


@pytest.fixture(scope="module")
def exported():
    """The reference as this machine can get it: from git, or from a checked directory."""
    try:
        return micro_ref.Reference.export()
    except micro_ref.ReferenceUnavailable as e:
        pytest.skip(f"no reference code on this machine: {e}")


def _git_cannot_be_asked(monkeypatch, how: str) -> None:
    def run(cmd, *a, **k):
        if how == "no git":
            raise FileNotFoundError(2, "No such file or directory", "git")
        raise subprocess.CalledProcessError(128, cmd, b"", b"fatal: not a git repository")

    monkeypatch.setattr(micro_ref.subprocess, "run", run)


def test_the_blob_ids_are_the_commits():
    """REF_BLOBS against the repository, where there is one. A typo here would make every fallback refuse."""
    assert set(micro_ref.REF_BLOBS) == set(micro_ref.MODULES)
    try:
        out = subprocess.run(["git", "-C", REPO, "ls-tree", micro_ref.REF_COMMIT, "--"] + [m + ".py" for m in micro_ref.MODULES],
                             capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"no git history here to check the ids against: {e!r}")
    seen = {line.split("\t")[1][:-3]: line.split()[2] for line in out.splitlines()}
    assert seen == micro_ref.REF_BLOBS
    assert micro_ref.git_blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"      # git's well-known name for the empty file


@pytest.mark.parametrize("how", ["no git", "no repository"])
def test_export_without_git_takes_a_directory_that_holds_the_commit(monkeypatch, exported, how):
    d = _copy_reference(exported.dir, "complete")
    _git_cannot_be_asked(monkeypatch, how)
    ref = micro_ref.Reference.export(dest=d)
    assert ref.dir == os.path.abspath(d) and "checked against the commit's blob id" in ref.source
    # through the environment variable, which is how the tests and the box name the directory
    monkeypatch.setenv("PSPPO_REF_DIR", d)
    assert micro_ref.Reference.export().dir == os.path.abspath(d)
    # and what it loads is the old code: its schedule constant, and no trace of the speed work
    old_config, old_core = ref.load("config"), ref.load("ppo_core")
    assert os.path.dirname(os.path.abspath(old_config.__file__)) == os.path.abspath(d)
    assert old_config.LearnerConfig().lr_total_steps == 100_000
    assert not hasattr(old_core, "update_data_fits") and not hasattr(old_core, "episodic_block_mask")
    import ppo_core
    assert hasattr(ppo_core, "update_data_fits") and ppo_core is not old_core


def test_export_without_git_refuses_what_is_not_the_commit(monkeypatch, exported):
    _git_cannot_be_asked(monkeypatch, "no git")
    empty = _new_dir("empty")
    with pytest.raises(micro_ref.ReferenceUnavailable, match=r"utils\.py is not there"):
        micro_ref.Reference.export(dest=empty)

    partial = _copy_reference(exported.dir, "partial", leave_out=("obs_global",))
    with pytest.raises(micro_ref.ReferenceUnavailable, match=r"obs_global\.py is not there"):
        micro_ref.Reference.export(dest=partial)

    # today's file under the old name: the mistake that would compare the new code with itself
    todays = _copy_reference(exported.dir, "todays")
    with open(os.path.join(REPO, "ppo_core.py"), "rb") as f:
        data = f.read()
    with open(os.path.join(todays, "ppo_core.py"), "wb") as f:
        f.write(data)
    with pytest.raises(micro_ref.ReferenceUnavailable, match=r"ppo_core\.py is not the commit's file"):
        micro_ref.Reference.export(dest=todays)

    # the right text with converted line endings is refused too: bit-for-bit means the commit's bytes
    crlf = _copy_reference(exported.dir, "crlf")
    path = os.path.join(crlf, "config.py")
    with open(path, "rb") as f:
        data = f.read()
    swapped = data.replace(b"\r\n", b"\n") if b"\r\n" in data else data.replace(b"\n", b"\r\n")
    assert swapped != data
    with open(path, "wb") as f:
        f.write(swapped)
    with pytest.raises(micro_ref.ReferenceUnavailable, match=r"config\.py is not the commit's file"):
        micro_ref.Reference.export(dest=crlf)

    # another commit has no known ids, so a directory cannot stand in for it
    complete = _copy_reference(exported.dir, "other_commit")
    with pytest.raises(micro_ref.ReferenceUnavailable, match="only the files of"):
        micro_ref.Reference.export(commit="fa61d1f", dest=complete)


def test_with_git_the_export_still_writes_and_repairs_the_files(exported):
    """The path this machine normally takes is unchanged: a wrong file in the directory is replaced from git."""
    if exported.source != "git show":
        pytest.skip("git cannot be asked on this machine")
    d = _copy_reference(exported.dir, "repair", leave_out=("utils",))
    with open(os.path.join(d, "config.py"), "ab") as f:
        f.write(b"\n# not the commit's file any more\n")
    assert micro_ref.Reference.why_not_the_commit(d) is not None
    ref = micro_ref.Reference.export(dest=d)
    assert ref.source == "git show" and micro_ref.Reference.why_not_the_commit(d) is None
