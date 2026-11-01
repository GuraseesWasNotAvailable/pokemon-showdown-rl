"""
Export the newest checkpoint of v2 arms for the window app: the network only (no optimizer, no
target network, no training-only heads), with run_cfg kept, named

    v2-<arm>_update_<N>_<M>M.pt

M is millions of decisions the arm trained:
  * an arm of the waves, started from S (learner_update_022640, 838,091,147 decisions):
    (total_steps - S) / 1e6;
  * the from-scratch run (arm "scratch", random weights): update x steps_per_update / 1e6, with
    steps_per_update from the checkpoint's run_cfg["learner"] (36,864 in that run);
  * the league run (arm "league", from M24381 widened with its counters at 0): total_steps / 1e6,
    the PPO decisions of that run.

League networks (the main line's observation with the blocks; "obs_settings" names main_line) are
exported the same way: branch local-app plays them from obs_league.py on, and their "obs_settings"
entry is kept.

Networks with train-v2's matchup block (run_cfg["obs"]["matchup"], the from-scratch run and
musethead) are exported like the others: branch decide plays them from obs_v2.py on, and run_cfg
tells it which observation the network was trained on, so run_cfg must stay in the file. The
training-only heads (set_head.*, opp_action_head.*) are left out; decide drops them at load too.

A checkpoint written in the last 60 seconds may still be being written: the newest older one is
taken instead. The file is written as <name>.part, read back and compared with the checkpoint
(every network tensor bit for bit, run_cfg equal), and only then renamed to <name>.

Runs on a training box (box a: ~/dev/export_v2.py; a copy of this file), reading the arms under
~/runs/v2/<arm>/checkpoints and writing ~/dev/export (or --out):

    ~/venv/bin/python ~/dev/export_v2.py scratch
    ~/venv/bin/python ~/dev/export_v2.py a0 readhead --out ~/dev/export
    ~/venv/bin/python ~/dev/export_v2.py league --file ~/dev/league/bars/L300.pt --out ~/dev/league/export

--file exports that checkpoint (a copy of a run's checkpoint, e.g. from the S3 backup) instead of the
arm's newest; one arm then.
"""
import argparse
import glob
import os
import sys
import time

import torch

S_TOTAL = 838_091_147                 # decisions in S, the start of every wave arm
FROM_SCRATCH = {"scratch"}            # arms trained from random weights: M counts every update
FROM_ZERO = {"league"}                # arms whose total_steps started at 0 (the league start): M is total_steps
HEAD_PREFIXES = ("set_head.", "opp_action_head.")
SETTLE_S = 60.0


def newest_checkpoint(ck_dir: str) -> str:
    found = glob.glob(os.path.join(ck_dir, "learner_update_*.pt"))
    settled = [f for f in found if time.time() - os.path.getmtime(f) >= SETTLE_S]
    if not settled:
        raise FileNotFoundError(f"no settled learner_update_*.pt in {ck_dir}")
    return max(settled, key=os.path.getmtime)


def millions(arm: str, ck: dict) -> int:
    if arm in FROM_SCRATCH:
        learner = (ck.get("run_cfg") or {}).get("learner") or {}
        per_update = int(learner.get("steps_per_update") or 36_864)
        return round(int(ck["update_idx"]) * per_update / 1e6)
    if arm in FROM_ZERO:
        return round(int(ck["total_steps"]) / 1e6)
    return round((int(ck["total_steps"]) - S_TOTAL) / 1e6)


def check(path: str, ck: dict, model: dict) -> None:
    """The written file, read back: every network tensor bit for bit the checkpoint's, run_cfg the same."""
    back = torch.load(path, map_location="cpu", weights_only=False)
    if set(back["model"]) != set(model):
        raise SystemExit(f"{path}: other tensors than the checkpoint's network")
    for k, v in model.items():
        w = back["model"][k]
        if w.dtype != v.dtype or w.shape != v.shape or not torch.equal(w, v):
            raise SystemExit(f"{path}: {k} is not the checkpoint's")
    if back.get("run_cfg") != ck.get("run_cfg"):
        raise SystemExit(f"{path}: run_cfg is not the checkpoint's")


def export(arm: str, runs: str, out_dir: str, file: str = "") -> str:
    newest = file or newest_checkpoint(os.path.join(runs, arm, "checkpoints"))
    ck = torch.load(newest, map_location="cpu", weights_only=False)
    model = {k: v for k, v in ck["model"].items() if not k.startswith(HEAD_PREFIXES)}
    keep = {k: ck[k] for k in ("run_cfg", "update_idx", "total_steps", "total_episodes", "obs_settings") if k in ck}
    keep["model"] = model
    obs = (ck.get("run_cfg") or {}).get("obs") or {}
    keep["exported_from"] = {"arm": arm, "file": os.path.basename(newest), "obs_settings_in_source": "obs_settings" in ck,
                             "obs": {k: obs.get(k) for k in ("own_tera_type", "main_line", "matchup", "movesets") if k in obs}}
    name = f"v2-{arm}_update_{int(ck['update_idx']):06d}_{millions(arm, ck)}M.pt"
    path = os.path.join(out_dir, name)
    tmp = path + ".part"
    torch.save(keep, tmp)
    check(tmp, ck, model)
    os.replace(tmp, path)
    print(arm, os.path.basename(newest), "->", path, "| dropped", len(ck["model"]) - len(model), "head tensors",
          "| obs:", keep["exported_from"]["obs"], "| total_steps", ck.get("total_steps"), flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--runs", default=os.path.expanduser("~/runs/v2"))
    ap.add_argument("--out", default=os.path.expanduser("~/dev/export"))
    ap.add_argument("--file", default="", help="export this checkpoint file instead of the arm's newest (one arm)")
    args = ap.parse_args()
    if args.file and len(args.arms) != 1:
        ap.error("--file takes one arm")
    os.makedirs(args.out, exist_ok=True)
    for arm in args.arms:
        export(arm, args.runs, args.out, os.path.expanduser(args.file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
