#!/usr/bin/env python
"""
The code as it was before the speed changes, loaded next to the working tree, so that old and
new can be run on the same inputs in one process and compared bit for bit.

"Before" is commit REF_COMMIT: what the cloud box ran its imitation phase with. Its copies of
the files below are written once into a directory outside the repository (`git show`, nothing
is checked out and nothing is ever removed), and imported from there under their own names
while the working tree's modules of the same names are set aside:

    ref = Reference.export()            # or Reference(<a directory holding the old files>)
    old_core = ref.load("ppo_core")     # the old module; `import ppo_core` still gives the new
    with ref.active():                  # inside, `import ppo_core` means the old file
        old_net = old_config.RunConfig.default().make_model()

The `with` matters for anything that imports late: config.make_model imports ppo_core when it
is called, so a model built outside the block would be the new class.

Also here: where the tests get observations from (real ones saved by bench/micro_obs.py, or
made-up ones that stay inside every embedding table), and small comparison helpers.

    python bench/micro_ref.py            # export the reference and say what differs from it

A machine without the repository's history (the cloud box is deployed as a tar file) cannot
run `git show`. There, Reference.export() takes a directory that already holds the commit's
files, checked byte for byte against git's own names for them (REF_BLOBS):

    PSPPO_REF_DIR=<dir> python bench/micro_ref.py     # here, where the history is; then send <dir>
    PSPPO_REF_DIR=<dir> python -m pytest tests/...     # there
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The commit the cloud box trained with before the speed work. For every file in MODULES it is
# byte for byte the same as fa61d1f, the commit the speed work started from.
REF_COMMIT = "b5680bd"

# Every repository module the old files import from one another. worker.py is left out on
# purpose: importing it patches poke-env, and a second copy would patch it twice.
MODULES: Tuple[str, ...] = (
    "utils", "config", "ppo_core", "inference", "learner",
    "obs_assembler", "obs_moves", "obs_pokemon", "obs_abilities", "obs_global", "obs_transitions",
)


# git's own names for REF_COMMIT's copies of MODULES (`git ls-tree b5680bd`). They let a
# directory of those files be recognised on a machine that has no history to ask: the cloud
# box gets the code as a tar file, without .git.
REF_BLOBS: Dict[str, str] = {
    "utils": "ef3181b0341dbd26eb139408f8dc3e5736501ace",
    "config": "d5a00578e2814f922bd8ae192f6361b4b8d4f436",
    "ppo_core": "86726923400d20ea7ace500dfd7bd3429cf0fabe",
    "inference": "978acb6eec6cad861faf4bd70dbf8d1200edb08d",
    "learner": "9b079a190fb045dca8510d151ab22a4b3a070e48",
    "obs_assembler": "1e1a44c4c379841adc924d06daa1d55556fa946a",
    "obs_moves": "272b4ff56f9984f7441a5cd1daf449a83cf180fe",
    "obs_pokemon": "8ef42768a0dcfa89ace14e370d7aa27644f20715",
    "obs_abilities": "0a3fd695882a6e3c43191868676c53d85aa79707",
    "obs_global": "19f77c107e8c039a916615994d8b2739a64f9810",
    "obs_transitions": "4391972725a1446f50e8e2e8932d8b5fd72887d8",
}


def git_blob_id(data: bytes) -> str:
    """The name git gives a file with these bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


class ReferenceUnavailable(RuntimeError):
    """git, the repository or the commit is missing: there is nothing to compare against."""


class Reference:
    def __init__(self, src_dir: str, source: str = "a directory given by the caller"):
        self.dir = os.path.abspath(src_dir)
        self.source = source
        missing = [m for m in MODULES if not os.path.isfile(os.path.join(self.dir, m + ".py"))]
        if missing:
            raise ReferenceUnavailable(f"{self.dir} lacks {missing}")
        self._mods: Dict[str, Any] = {}

    @staticmethod
    def why_not_the_commit(src_dir: str, commit: str = REF_COMMIT) -> Optional[str]:
        """None when src_dir holds the commit's copy of every module, byte for byte. Otherwise what is wrong with it."""
        if commit != REF_COMMIT:
            return f"only the files of {REF_COMMIT} can be recognised without git, not those of {commit}"
        for m in MODULES:
            path = os.path.join(src_dir, m + ".py")
            if not os.path.isfile(path):
                return f"{m}.py is not there"
            with open(path, "rb") as f:
                if git_blob_id(f.read()) != REF_BLOBS[m]:
                    return f"{m}.py is not the commit's file (changed, or its line endings were converted)"
        return None

    @classmethod
    def export(cls, commit: str = REF_COMMIT, dest: Optional[str] = None, repo: str = REPO_ROOT) -> "Reference":
        """
        Writes the commit's copies of MODULES into dest (default: a directory named after the
        commit in the system temp folder, or $PSPPO_REF_DIR) and returns a Reference on it. A
        file that is already there with the same bytes is left alone.

        Where git cannot be asked (no git, no history, no such commit: the cloud box), dest is
        used as it is when it already holds every file of REF_COMMIT with exactly the commit's
        bytes. So run this once where the history is, with $PSPPO_REF_DIR set, and send that
        directory along. Anything else raises ReferenceUnavailable, as it always did.
        """
        dest = dest or os.environ.get("PSPPO_REF_DIR") or os.path.join(tempfile.gettempdir(), f"psppo_ref_{commit}")
        os.makedirs(dest, exist_ok=True)
        for m in MODULES:
            try:
                data = subprocess.run(["git", "-C", repo, "show", f"{commit}:{m}.py"], capture_output=True, check=True).stdout
            except (OSError, subprocess.CalledProcessError) as e:
                why = cls.why_not_the_commit(dest, commit)
                if why is None:
                    return cls(dest, source="files that were already there, each checked against the commit's blob id (git could not be asked)")
                raise ReferenceUnavailable(f"git show {commit}:{m}.py failed: {e!r}, and {dest} cannot stand in: {why}") from e
            path = os.path.join(dest, m + ".py")
            old = None
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    old = f.read()
            if old != data:
                with open(path, "wb") as f:
                    f.write(data)
        return cls(dest, source="git show")

    @contextlib.contextmanager
    def active(self) -> Iterator["Reference"]:
        """Inside the block the names in MODULES mean the old files, for imports made in it."""
        saved = {m: sys.modules.get(m) for m in MODULES}
        for m in MODULES:
            if m in self._mods:
                sys.modules[m] = self._mods[m]
            else:
                sys.modules.pop(m, None)
        sys.path.insert(0, self.dir)
        try:
            yield self
        finally:
            try:
                sys.path.remove(self.dir)
            except ValueError:
                pass
            for m in MODULES:
                mod = sys.modules.pop(m, None)
                if mod is not None:
                    self._mods[m] = mod
                if saved[m] is not None:
                    sys.modules[m] = saved[m]

    def load(self, name: str) -> Any:
        if name not in MODULES:
            raise KeyError(name)
        with self.active():
            mod = importlib.import_module(name)
        src = os.path.abspath(getattr(mod, "__file__", ""))
        if os.path.dirname(src) != self.dir:
            raise ReferenceUnavailable(f"{name} came from {src}, not from {self.dir}")
        return mod

    def differs_from_tree(self, repo: str = REPO_ROOT) -> List[str]:
        """The modules whose working-tree file is not the reference file (line endings aside)."""
        out = []
        for m in MODULES:
            with open(os.path.join(self.dir, m + ".py"), "rb") as f:
                a = f.read().replace(b"\r\n", b"\n")
            with open(os.path.join(repo, m + ".py"), "rb") as f:
                b = f.read().replace(b"\r\n", b"\n")
            if a != b:
                out.append(m)
        return out


# --------------------------------------------------------------------------- observations


def real_observations(path: Optional[str] = None):
    """
    (obs float16 [N, dim], episode id int64 [N]) saved by bench/micro_obs.py --out, in learner
    order (whole episodes back to back), or None when no file is given. The file comes from
    the argument or from $PSPPO_OBS_SAMPLES.
    """
    import numpy as np
    path = path or os.environ.get("PSPPO_OBS_SAMPLES")
    if not path or not os.path.isfile(path):
        return None
    d = np.load(path)
    return d["obs"], d["ep"].astype(np.int64)


def synthetic_observations(n: int, seed: int = 0, mean_episode: int = 40):
    """
    Made-up observations for when no recorded ones are at hand. Every number the model looks
    up in an embedding table is inside that table and the action mask always allows something,
    so the model runs on them as it does on real ones. They are noise, not Pokemon: good for
    comparing two implementations of the same arithmetic, useless for judging a policy.
    Episode lengths are drawn around mean_episode, with some far longer than the 64-turn
    history window.
    """
    import numpy as np
    from obs_assembler import ObservationAssembler

    a = ObservationAssembler()
    meta, off = a.meta, a.offsets
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, a.total_dim), dtype=np.float16)

    def put(name: str, values) -> None:
        s, e = off[name]
        obs[:, s:e] = values.reshape(n, e - s)

    body_dim = meta["dim_pokemon_body"]
    bmap = meta["feature_map"]["body"]
    body = (rng.random((n, 12, body_dim)) < 0.08).astype(np.float16)
    body[:, :, bmap["hp_int"]] = rng.integers(0, 101, (n, 12))
    s0, s1 = bmap["stats_int"]
    body[:, :, s0:s1] = rng.integers(0, 800, (n, 12, s1 - s0))
    body[:, :, bmap["level_int"]] = rng.integers(1, 101, (n, 12))
    body[:, :, bmap["weight_int"]] = rng.integers(0, 21, (n, 12))
    body[:, :, bmap["height_int"]] = rng.integers(0, 201, (n, 12))
    put("pokemon_body", body)
    ids = np.stack([rng.integers(0, meta["vocab_pokemon"], (n, 12)), rng.integers(0, meta["vocab_item"], (n, 12))], axis=-1)
    put("pokemon_ids", ids.astype(np.float16))
    put("ability_ids", rng.integers(0, meta["vocab_ability"], (n, 48)).astype(np.float16))
    put("move_ids", rng.integers(0, meta["vocab_move"], (n, 48)).astype(np.float16))
    mdim = meta["dim_move_scalars"] // 4
    mmap = meta["feature_map"]["move"]
    mv = (rng.random((n, 48, mdim)) < 0.08).astype(np.float16)
    mv[:, :, mmap["acc_int"]] = rng.integers(0, 101, (n, 48))
    mv[:, :, mmap["pwr_int"]] = rng.integers(0, 251, (n, 48))
    mv[:, :, mmap["pp_int"]] = rng.integers(0, 65, (n, 48))
    put("move_scalars", mv)
    g0, g1 = off["global_scalars"]
    glob = (rng.random((n, g1 - g0)) < 0.1).astype(np.float16)
    glob[:, 0] = rng.integers(0, 251, n)
    put("global_scalars", glob)
    tmap = meta["feature_map"]["transition"]
    tid = np.zeros((n, meta["dim_transition_ids"]), dtype=np.float16)
    for key, top in (("move_ids", "vocab_move"), ("pokemon_ids", "vocab_pokemon"), ("ability_ids", "vocab_ability"), ("item_ids", "vocab_item")):
        lo, hi = tmap[key]
        tid[:, lo:hi] = rng.integers(0, meta[top], (n, hi - lo))
    put("transition_ids", tid)
    t0, t1 = off["transition_scalars"]
    put("transition_scalars", (rng.random((n, t1 - t0)) < 0.1).astype(np.float16))
    mask = (rng.random((n, meta["action_dim"])) < 0.5).astype(np.float16)
    mask[np.arange(n), rng.integers(0, meta["action_dim"], n)] = 1.0
    put("action_mask", mask)

    ep = np.zeros(n, dtype=np.int64)
    i, e = 0, 0
    while i < n:
        length = int(rng.choice([1, 2, 3, 150])) if rng.random() < 0.15 else max(1, int(rng.normal(mean_episode, 15)))
        ep[i:i + length] = e
        i += length
        e += 1
    return obs, ep


def observations(n_min: int = 0, seed: int = 0):
    """Real observations when $PSPPO_OBS_SAMPLES names a file with at least n_min rows, made-up ones otherwise. Returns (obs, ep, kind)."""
    real = real_observations()
    if real is not None and real[0].shape[0] >= n_min:
        return real[0], real[1], "real"
    obs, ep = synthetic_observations(max(n_min, 4096), seed)
    return obs, ep, "synthetic"


# --------------------------------------------------------------------------- comparison helpers

BLOCK_MASK_TENSORS: Tuple[str, ...] = (
    "kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
    "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices",
)


def block_mask_differences(a: Any, b: Any) -> List[str]:
    """Names of the BlockMask fields that are not the same in a and b (empty: identical)."""
    import torch
    bad = []
    for name in BLOCK_MASK_TENSORS:
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            if x is not y:
                bad.append(name)
            continue
        if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device or not torch.equal(x, y):
            bad.append(name)
        elif x.is_contiguous() != y.is_contiguous():
            bad.append(name + " (memory layout)")
    if tuple(a.seq_lengths) != tuple(b.seq_lengths):
        bad.append("seq_lengths")
    if tuple(a.BLOCK_SIZE) != tuple(b.BLOCK_SIZE):
        bad.append("BLOCK_SIZE")
    return bad


def same_bits(a: Any, b: Any) -> bool:
    """Same shape, dtype and bytes. NaN equals NaN here, and +0.0 does not equal -0.0."""
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy() if a.dtype == torch.bfloat16 else a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().float().cpu().numpy() if b.dtype == torch.bfloat16 else b.detach().cpu().numpy()
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def shared_compiled_flex_attention(*modules: Any) -> None:
    """
    Gives every given ppo_core module the one compiled flex_attention, the way both training
    actors compile it before their first forward, so old and new run the same kernels. The
    recompile limit is raised: one process pushing two copies of each mask rule through one
    compiled function passes torch's default of 8, after which it would silently run eagerly.
    """
    import torch
    from torch.nn.attention.flex_attention import flex_attention
    try:
        import torch._dynamo
        torch._dynamo.config.recompile_limit = 64
    except Exception:  # noqa: BLE001
        pass
    compiled = getattr(shared_compiled_flex_attention, "_fn", None)
    if compiled is None:
        compiled = torch.compile(flex_attention)
        shared_compiled_flex_attention._fn = compiled
    for m in modules:
        m.flex_attention = compiled


def main() -> int:
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)
    ref = Reference.export()
    print(f"reference {REF_COMMIT} in {ref.dir} (from {ref.source})")
    print("modules that differ from it in the working tree:", ref.differs_from_tree() or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
