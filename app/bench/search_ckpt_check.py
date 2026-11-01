"""
A real checkpoint of the search branch's baseline run, through two code trees: do they give the
same numbers?

Run from the root of the code tree that is to load the checkpoint (it imports that tree's
hybrid.py, not the one beside this file):

    python <here>/search_ckpt_check.py obs OUT.npz              (this line only) observations of a
                                                                few battle states, built by
                                                                obs_search's assembler
    python <here>/search_ckpt_check.py logits CKPT IN.npz OUT.npz   that tree's PolicyNet on those
                                                                observations, one battle per state
    python <here>/search_ckpt_check.py compare A.npz B.npz      largest difference, and whether
                                                                the best action is the same
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), "tests"))

import numpy as np  # noqa: E402


def cmd_obs(out: str) -> None:
    import test_obs_search as tos
    from obs_search import SearchOwnTeraObservationAssembler

    asm = SearchOwnTeraObservationAssembler()
    names, rows = [], []
    for name, make in tos.STATES.items():
        battle = make()
        # three decisions of one battle, so the per-battle history is used too
        for step in range(3):
            names.append(name)
            rows.append(asm.assemble(battle, events=[]).astype(np.float32))
    np.savez(out, names=np.array(names), obs=np.stack(rows))
    print(f"{len(rows)} observations of {len(tos.STATES)} states -> {out}")


def cmd_logits(ckpt: str, inp: str, out: str) -> None:
    from hybrid import HybridConfig, PolicyNet

    data = np.load(inp)
    pol = PolicyNet(HybridConfig(checkpoint=ckpt, device="cpu"))
    print(json.dumps({"tree": os.getcwd(), "checkpoint": os.path.basename(ckpt), "assembler": type(pol.assembler).__name__,
                      "own_tera_type": bool(getattr(pol.assembler, "own_tera_type", getattr(pol, "own_tera_type", False))),
                      "obs_line": getattr(pol, "obs_line", None), "obs_abilities": getattr(pol, "obs_abilities", None)}))
    rows = [pol.logits(str(name), obs) for name, obs in zip(data["names"], data["obs"])]
    np.savez(out, names=data["names"], logits=np.stack(rows))
    print(f"{len(rows)} rows of logits -> {out}")


def cmd_compare(a: str, b: str) -> None:
    x, y = np.load(a), np.load(b)
    assert list(x["names"]) == list(y["names"])
    la, lb = x["logits"], y["logits"]
    diff = float(np.abs(la - lb).max())
    same = int((la.argmax(1) == lb.argmax(1)).sum())
    print(f"rows {len(la)} | largest difference {diff:.3g} | same best action in {same} of {len(la)}")
    sys.exit(0 if diff < 1e-4 and same == len(la) else 1)


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    {"obs": cmd_obs, "logits": cmd_logits, "compare": cmd_compare}[mode](*args)
