"""
A live check of the opponent seats (opponent_seats.py), and the gate that picks setupnet's p.

    python start_servers.py --dir ~/pokemon-showdown --ports 10300-10305 &      # servers of its own
    python bench/seats_smoke.py --ckpt ckpt.pt --port 10300 --pairs 3 --rooms 8 --seats setupnet:0.34 --games 20
    python bench/seats_smoke.py --ckpt ckpt.pt --port 10301 --pairs 6 --rooms 16 --seats setupnet:1 --setupnet-p 0.5 --games 200

It runs the training's own rollout code in one process, without Ray: OpponentSeatsRolloutWorker
(worker.RolloutWorker's own rollout loop, reward code and learner client) on the seats' vector
env, against a Showdown server. The inference actor is inference.InferenceActor itself, made in
this process on the GPU with the checkpoint's weights (it samples from the policy as in
training); the learner is a stand-in that keeps what it is sent. Nothing is trained.

What it prints at the end (and writes to --out as JSON):
  * per seat kind: games, the trained network's wins (as p1 and p2), the seat's decisions, the
    rule moves (setupnet), the set-up moves played and the network's own set-up choices
  * what reached the learner: episodes, and which players they came from. An opponent seat's
    battle must never be among them (opponent_seat_episodes = 0), its battles end in CLEANUPs
  * crashes, refused choices, rooms the server closed

With --seats setupnet:1 every pair is a setupnet pair, and the network's win rate is the gate:
the training's p is one at which the network wins 55 to 80% (bench/notes/OPPONENT_SEATS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


class _Call:
    def __init__(self, fn):
        self.remote = fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--rooms", type=int, default=8, help="battles at once per pair")
    ap.add_argument("--seats", default="setupnet:0.34", help="as --opponent-seats: kind:share of this worker's pairs")
    ap.add_argument("--setupnet-p", type=float, default=0.5)
    ap.add_argument("--setupnet-max-stage", type=int, default=4)
    ap.add_argument("--seat-foulplay-iterations", type=int, default=0)
    ap.add_argument("--seat-foulplay-ms", type=int, default=200)
    ap.add_argument("--games", type=int, default=20, help="stop once the opponent seats have finished this many counted games")
    ap.add_argument("--minutes", type=float, default=20.0, help="stop after this long whatever the count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch

    import config
    import dataclasses
    import inference
    import opponent_seats as seats
    import worker

    started = time.time()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base = config.RunConfig.default()
    cfg = dataclasses.replace(
        base,
        learner=dataclasses.replace(base.learner, mode="ppo", resume=False),
        rollout=dataclasses.replace(base.rollout, rooms_per_pair=int(args.rooms), target_concurrent_battles=int(args.pairs * args.rooms)),
        obs=dataclasses.replace(base.obs, own_tera_type=config.checkpoint_own_tera_type(ckpt)),
    )
    actor = inference.InferenceActor(cfg, SimpleNamespace())
    actor.net.load_state_dict(inference._extract_state_dict(ckpt))
    del ckpt
    print(f"[smoke] network loaded from {args.ckpt} on {actor.device} (own_tera_type {cfg.obs.own_tera_type}, temp {actor.current_temp}) "
          f"in {time.time() - started:.0f} s", flush=True)

    sent_batches = []
    finalized, cleaned = [], []
    infer_lock = threading.Lock()

    def infer(tags, obs):
        with infer_lock:
            return actor.infer_batch(tags, obs)

    def clear(tag):
        with infer_lock:
            actor.clear_cache(tag)

    worker.ray = SimpleNamespace(get=lambda value, **_k: value)     # the actors below are called in this process
    inference_shim = SimpleNamespace(infer_batch=_Call(infer), clear_cache=_Call(clear))
    learner_shim = SimpleNamespace(submit_packed_batch=_Call(lambda *packed: sent_batches.append(packed[-1].tolist())))

    class SmokeWorker(seats.OpponentSeatsRolloutWorker):
        """The seats' worker, noting which battles it finalizes (DONE) and drops (CLEANUP)."""

        def _finalize_trajectory(self, tag, battle):
            finalized.append((tag, len(self._traj.get(tag, {}).get("act", []))))
            super()._finalize_trajectory(tag, battle)

        def _cleanup_trajectory(self, tag):
            cleaned.append((tag, len(self._traj.get(tag, {}).get("act", []))))
            super()._cleanup_trajectory(tag)

    settings = seats.OpponentSeatsConfig(
        shares=seats.parse_shares(args.seats),
        setupnet=seats.SetupNetSettings(p=args.setupnet_p, max_stage=args.setupnet_max_stage),
        foulplay=seats.FoulPlaySeatSettings(iterations=args.seat_foulplay_iterations, sizing_ms=args.seat_foulplay_ms),
        seed=args.seed)
    plan = seats.plan_seats([args.pairs], args.rooms, settings)
    entry = plan["workers"][0]
    print("[smoke] " + seats.describe_plan(plan), flush=True)
    rw = SmokeWorker(cfg, inference_shim, learner_shim, args.pairs, args.port, seats=entry)
    threading.Thread(target=rw._run_rl_loop, daemon=True, name="rl-loop").start()
    vec = rw.vec_env

    def counted_games(beat):
        st = beat.get("stats") or {}
        return sum(int(st.get(k, {}).get("games_p1", 0)) + int(st.get(k, {}).get("games_p2", 0)) + int(st.get(k, {}).get("mixed_games", 0))
                   for k in seats.SEAT_KINDS)

    deadline = started + 60.0 * args.minutes
    last = 0.0
    beat = {}
    while time.time() < deadline:
        time.sleep(2.0)
        beat = rw.heartbeat().get("opponent_seats") or {}
        if counted_games(beat) >= args.games:
            break
        if time.time() - last > 30.0:
            last = time.time()
            print(f"[smoke] {time.time() - started:.0f} s: " + seats.report_line([beat]) +
                  f" | learner episodes {sum(len(b) for b in sent_batches)}", flush=True)
    beat = rw.heartbeat().get("opponent_seats") or {}
    elapsed = time.time() - started

    seat_names = {p.username for p in vec.seat_players}
    net_names = {p.username for pair in vec.active_pairs for p in pair} - seat_names

    def owner(tag):
        return tag.split("_", 1)[0]

    result = {
        "args": vars(args), "elapsed_s": round(elapsed, 1), "plan": seats.describe_plan(plan),
        "line": seats.report_line([beat]), "stats": beat.get("stats"), "unavailable": beat.get("unavailable"),
        "failed_start": beat.get("failed_start"), "rooms_closed": beat.get("rooms_closed"),
        "learner_batches": len(sent_batches), "learner_episodes": sum(len(b) for b in sent_batches),
        "learner_steps": sum(sum(b) for b in sent_batches),
        "finalized_total": len(finalized),
        "opponent_seat_episodes": sum(1 for t, _n in finalized if owner(t) in seat_names),
        "trained_seat_episodes": sum(1 for t, n in finalized if owner(t) in net_names and n),
        "opponent_seat_cleanups": sum(1 for t, _n in cleaned if owner(t) in seat_names),
        "opponent_seat_steps_dropped": sum(n for t, n in cleaned if owner(t) in seat_names),
        "trained_seat_cleanups": sum(1 for t, _n in cleaned if owner(t) in net_names),
        "choices_refused": sum(int(getattr(p, "choices_refused", 0)) for pair in vec.active_pairs for p in pair),
        "choice_repair": dict(worker.CHOICE_REPAIR),
        "traj_in_memory": len(rw._traj),
    }
    st = (beat.get("stats") or {}).get("setupnet") or {}
    if st.get("decisions"):
        d = float(st["decisions"])
        games = st.get("games_p1", 0) + st.get("games_p2", 0)
        result["setupnet"] = {
            "p": args.setupnet_p, "max_stage": args.setupnet_max_stage, "games": games,
            "network_win_rate": round((st.get("wins_p1", 0) + st.get("wins_p2", 0) + 0.5 * (st.get("ties_p1", 0) + st.get("ties_p2", 0))) / games, 4) if games else None,
            "decisions": int(d), "rule_ready_share": round(st.get("rule_ready", 0) / d, 4),
            "rule_move_share": round(st.get("bot_decisions", 0) / d, 4),
            "setup_played_share": round(st.get("setup_played", 0) / d, 4),
            "network_own_setup_share": round(st.get("net_setup", 0) / d, 4),
            "overrides": {k: st.get("override_" + k, 0) for k in ("setup", "sleep", "recover", "sleeptalk")},
            "crashes": st.get("crashes", 0),
        }
    print("[smoke] RESULT " + json.dumps(result, default=str), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, default=str)
    try:
        vec.close_seats()
        import psbot_bridge
        if psbot_bridge._shared is not None:
            psbot_bridge._shared.close()
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)                       # the players' loop and the rollout loop are daemon threads on open sockets


if __name__ == "__main__":
    sys.exit(main())
