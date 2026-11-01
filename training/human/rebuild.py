#!/usr/bin/env python
"""
A top player's own view of a public gen9randombattle replay, decision by decision, exactly as the
network sees it in play: the observation of the live rollout worker (worker.SyncBridgePlayer with
its ObservationAssembler and its event history), the legal mask, and the action that player took
on our 14 actions (0-3 moves, 4-9 switches, 10-13 Tera moves).

HOW. A replay's "inputlog" names the Showdown commit the game ran on, the seeds and every choice.
Re-running it in that commit (human/resim.js) gives back each player's own stream with the real
|request| messages: the whole team with stats, moves, PP, items, abilities, Tera types and exact
HP, with no inference. The opponent's side holds only what the player was shown (|split| lines
are resolved per player, as the server does). The stream is fed as websocket frames, in the
order the server sends them (the turn's log, then the request, with the server's rqid put back),
into the REAL worker path, poke-env's PSClient._handle_message -> SyncBridgePlayer (the only fakes
are the websocket and the inference call, as in bench/micro_obs.py). Every choose_move the worker
makes is one step of the episode, as in self-play, so the history the transformer sees is the same.

A game is kept only if the re-simulated log equals the replay's log line for line (|-message|
lines aside: the server writes "X forfeited." there). Labels come from the side's inputlog lines,
in order, one per request that is not a wait. The randbats set data is not needed for this.

BAR. A side is used when its own pre-game Elo in the inputlog is >= --bar (2100) in a rated
gen9randombattle game. Its sequence is written whole; "train" marks the steps to imitate: the
label maps onto the 14 actions, is legal under the observation's own mask, and there was more
than one legal action.

    nice -n 15 ~/venv/bin/python human/rebuild.py own  --replay <replay.json> --recording <ladder battle .jsonl>
    nice -n 15 ~/venv/bin/python human/rebuild.py build --replays '<glob of replay .json>' --out <dir> [--procs 10]

Box only (the owner's rule), in ~/dev/human-rebuild (HUMAN_DIR). Usernames never leave memory:
the stream is fed with the names humanp1 / humanp2, and the shards carry salted hashes.

Branch baseline-run (REPORT_LADDER_AND_BASELINE_RUN.md, the imitation test's arm B): the oursearch copy,
changed only so the observation is THIS branch's: the worker's own assembler (RunConfig.obs,
ObsConfig.own_tera_type, STAB after a Tera), never obs_abil.py or obs_tera.py; --obs-abilities is
refused; the material count is not stored (NaN), the turn and Phi are.

Branch oursearch (bench/WINHEAD.md section 3.1): copied from branch stage2 and changed only so:
  * the observation is this branch's (obs_abil.py with the absorb fix); the assembler class is set
    on worker.ObservationAssembler directly (stage2's tera_worker.py is not on this branch);
  * every step also stores, read when the observation is assembled (as bench/value_gate.py reads
    them): the turn, Phi (rl_reward's potential) and the material count (value_gate.material).
    The observation bytes do not change;
  * build --from-done takes exactly the replay list of an earlier build's done.json;
  * build --exclude-ids FILE leaves out replays whose battle id is listed (the gate and A/B games),
    and the environment variable WH_EXCLUDE_USERS (comma separated user ids, kept in memory only)
    leaves out replays where either player is one of them (the bot's own ladder accounts). Both
    are counted in the summary;
  * own --obs-abilities, so the own check runs with the observation of the build;
  * own and --only-built never build Showdown: a version without a cached build is an error.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.environ.get("HUMAN_DIR", os.path.expanduser("~/dev/human-rebuild"))
PS = os.path.join(HOME, "ps")
BUILDS = os.path.join(HOME, "builds")
WT = os.path.join(HOME, "wt")
RESIM_JS = os.path.join(REPO, "human", "resim.js")
GITHUB = "https://github.com/smogon/pokemon-showdown.git"
ALIAS = {"p1": "humanp1", "p2": "humanp2"}
FORMAT = "gen9randombattle"
NL = chr(10)


def log(*a: Any) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =============================================================================================
# The replay
# =============================================================================================

def parse_inputlog(text: str) -> Dict[str, Any]:
    """{"version", "start": {...}, "players": {"p1": {...}, "p2": {...}}, "choices": {"p1": [...], "p2": [...]}, "other": [...]}"""
    out: Dict[str, Any] = {"version": None, "start": {}, "players": {}, "choices": {"p1": [], "p2": []}, "other": []}
    for line in (text or "").split("\n"):
        if line.startswith(">version "):
            out["version"] = line.split()[1]
        elif line.startswith(">start "):
            out["start"] = json.loads(line[7:])
        elif line.startswith(">player "):
            side, js = line[8:10], line[11:]
            out["players"][side] = json.loads(js)
        elif line[:4] in (">p1 ", ">p2 "):
            out["choices"][line[1:3]].append(line[4:].strip())
        elif line.startswith(">"):
            out["other"].append(line)
    return out


def side_elo(il: Dict[str, Any], side: str) -> Optional[int]:
    try:
        r = il["players"][side].get("rating")
        return int(r) if r not in (None, "") else None
    except Exception:
        return None


def is_rated(il: Dict[str, Any]) -> bool:
    return bool(il["start"].get("rated"))


_DROP_NORM = re.compile(r"^\|(j|J|l|L|n|N|c|c:|chat|t:|raw|html|uhtml|uhtmlchange|inactive|inactiveoff|badge|player|title|init|"
                        r"rated|gametype|gen|tier|rule|teamsize|-message|)$")


def norm_log(lines: List[str]) -> List[str]:
    return [l for l in lines if l.startswith("|") and not _DROP_NORM.match("|" + l.split("|")[1])]


def same_log(replay_log: str, omniscient_chunks: List[str]) -> Tuple[bool, Optional[int]]:
    a = norm_log(replay_log.split("\n"))
    b = norm_log("\n".join(omniscient_chunks).split("\n"))
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return (i == len(a) == len(b)), (None if i == len(a) == len(b) else i)


def salted(kind: str, text: str) -> str:
    salt_path = os.path.join(HOME, "salt")
    if not os.path.exists(salt_path):
        os.makedirs(HOME, exist_ok=True)
        with open(salt_path, "w") as f:
            f.write(os.urandom(16).hex())
    with open(salt_path) as f:
        salt = f.read().strip()
    return kind + hashlib.sha256((salt + "|" + text.lower()).encode()).hexdigest()[:12]


# =============================================================================================
# Showdown builds, one per distinct sim/data/formats content
# =============================================================================================

def _git(*args: str, cwd: str = PS, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args), r.stderr.strip()[:300]))
    return r.stdout.strip()


def has_commit(version: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", version + "^{commit}"], cwd=PS, capture_output=True).returncode == 0


def build_key(version: str) -> Optional[str]:
    if not has_commit(version):
        return None
    trees = []
    for p in ("sim", "data", "config/formats.ts", "lib"):
        r = subprocess.run(["git", "rev-parse", "%s:%s" % (version, p)], cwd=PS, capture_output=True, text=True)
        trees.append(r.stdout.strip() if r.returncode == 0 else "-")
    return hashlib.sha1(" ".join(trees).encode()).hexdigest()[:16]


def ensure_build(version: str, slot: int = 0) -> Tuple[Optional[str], str]:
    """(build dir holding dist/, note). The build is shared by every version with the same sim/data/formats/lib."""
    key = build_key(version)
    if key is None:
        return None, "commit not in the clone"
    dest = os.path.join(BUILDS, key)
    if os.path.exists(os.path.join(dest, "dist", "sim", "battle-stream.js")):
        return dest, "cached"
    wt = os.path.join(WT, "s%d" % slot)
    if not os.path.isdir(wt):
        os.makedirs(WT, exist_ok=True)
        _git("worktree", "add", "-q", "--detach", wt, version)
    else:
        _git("checkout", "-q", "-f", version, cwd=wt)
    nm = os.path.join(wt, "node_modules")
    if not os.path.lexists(nm):
        os.symlink(os.path.join(BUILDS, "node_modules"), nm)
    if os.path.exists(os.path.join(wt, "dist")):           # a failed build's leftovers: moved aside, never deleted
        os.makedirs(os.path.join(HOME, "old"), exist_ok=True)
        shutil.move(os.path.join(wt, "dist"), os.path.join(HOME, "old", "dist-%s-%d" % (key, int(time.time()))))
    with open(os.path.join(HOME, "build-%s.log" % key), "w") as fh:
        r = subprocess.run(["nice", "-n", "15", "node", "build"], cwd=wt, stdout=fh, stderr=subprocess.STDOUT, timeout=900)
    if r.returncode != 0 or not os.path.exists(os.path.join(wt, "dist", "sim", "battle-stream.js")):
        return None, "build failed (build-%s.log)" % key
    os.makedirs(dest + ".tmp", exist_ok=True)
    shutil.move(os.path.join(wt, "dist"), os.path.join(dest + ".tmp", "dist"))
    os.replace(dest + ".tmp", dest)
    return dest, "built"


def fetch_missing(versions: List[str]) -> None:
    missing = [v for v in versions if not has_commit(v)]
    if missing:
        log("fetching %d missing Showdown commits from GitHub" % len(missing))
        subprocess.run(["git", "fetch", "-q", GITHUB, "master"], cwd=PS)


def resim(build: str, jobs: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    payload = "".join(json.dumps(j) + "\n" for j in jobs)
    r = subprocess.run(["nice", "-n", "15", "node", RESIM_JS, build], input=payload, capture_output=True, text=True,
                       timeout=60 + 30 * len(jobs))
    out = {}
    for line in r.stdout.splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["key"]] = d
    return out


# =============================================================================================
# The frames a player's websocket would have carried
# =============================================================================================

def rename_line(line: str, names: Dict[str, str]) -> Optional[str]:
    """A protocol line with the players' names replaced by ALIAS; None for lines that only carry chat or names."""
    parts = line.split("|")
    if len(parts) < 2:
        return line
    kind = parts[1]
    if kind in ("c", "c:", "chat", "raw", "html", "uhtml", "uhtmlchange", "-message", "message", "j", "J", "l", "L", "n", "N",
                "inactive", "inactiveoff", "title", "init", "badge"):
        return None
    if kind == "player" and len(parts) > 3:
        parts[3] = ALIAS.get(parts[2], parts[3])
        return "|".join(parts)
    if kind == "win" and len(parts) > 2:
        winner = "|".join(parts[2:])
        for s, n in names.items():
            if n == winner:
                return "|win|" + ALIAS[s]
        return "|win|somebody"
    for s, n in names.items():
        if n and ("%s: %s" % (s, n)) in line:
            line = re.sub(r"(?<![\w])%s: %s(?=\||$)" % (s, re.escape(n)), "%s: %s" % (s, ALIAS[s]), line)
    return line


def frames_for_side(room: str, side: str, chunks: List[str], names: Dict[str, str]) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """[(websocket frame, request dict or None)] for one player, rqid as the server numbers it (2k+2 for p1, 2k+3 for p2)."""
    frames: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    head = ">%s\n" % room
    frames.append((head + "|init|battle\n|title|%s vs. %s\n|j|☆%s\n|j|☆%s" % (ALIAS["p1"], ALIAS["p2"], ALIAS["p1"], ALIAS["p2"]), None))
    k = 0
    for chunk in chunks:
        lines, req = [], None
        for line in chunk.split("\n"):
            if line.startswith("|request|"):
                body = line[9:]
                if not body.strip():
                    continue
                req = json.loads(body)
                req["rqid"] = 2 * k + (2 if side == "p1" else 3)
                k += 1
                lines.append("|request|" + json.dumps(req, separators=(",", ":"), ensure_ascii=False))
            else:
                new = rename_line(line, names)
                if new is not None:
                    lines.append(new)
        if lines:
            frames.append((head + "\n".join(lines), req))
    return frames


def frames_from_recording(path: str) -> Tuple[str, str, List[Tuple[str, Optional[Dict[str, Any]]]], List[Dict[str, Any]]]:
    """(room, username, frames, decision notes) of one of our ladder recordings (ladder.py), as the bot received them."""
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    note = next(r["data"] for r in rows if r["dir"] == "note" and "battle" in r["data"] and "user" in r["data"])
    room, user = note["battle"], note["user"]
    frames, notes = [], []
    for r in rows:
        if r["dir"] == "in":
            req = None
            for line in r["data"].split("\n"):
                if line.startswith("|request|") and line[9:].strip():
                    req = json.loads(line[9:])
            frames.append((">%s\n%s" % (room, r["data"]), req))
        elif r["dir"] == "note" and "decision" in r["data"]:
            notes.append(r["data"]["decision"])
    return room, user, frames, notes


# =============================================================================================
# The real worker path, fed from frames
# =============================================================================================

class _Queue:
    """The worker's event queue: a STEP is recorded and answered at once with the action `pick` returns."""

    def __init__(self, harness: "Harness") -> None:
        self.h = harness

    def put_nowait(self, item: Any) -> None:
        kind, tag, payload, extra = item
        if kind == "STEP":
            action = self.h._on_step(tag, payload)
            fut, _loop = self.h.action_futures.pop(tag)
            fut.set_result(int(action))
        elif kind == "DONE":
            owner = self.h.players.get(tag.split("_", 1)[0])
            if owner is not None:
                owner._safe_release()


async def _noop_send(*_a: Any, **_k: Any) -> None:
    return None


class Harness:
    def __init__(self, own_tera_type: bool, obs_abilities: str = "") -> None:
        os.chdir(REPO)                                   # vocab.json is opened by a relative path
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        import worker                                     # applies worker.py's poke-env patches
        if obs_abilities:
            raise SystemExit("this branch has no --obs-abilities: its observation is the worker's own")
        self.obs_abilities = ""
        # turn and Phi per step (Phi: rl_reward's potential, as the rollout worker reads it)
        from rl_reward import load_potential
        self.potential = load_potential(None)
        self.material = None
        from config import RunConfig
        self.worker = worker
        self.cfg = RunConfig.default()
        # The worker builds its assembler from cfg.obs (SyncBridgePlayer: cfg.make_assembler()).
        self.cfg = dataclasses.replace(self.cfg, learner=dataclasses.replace(self.cfg.learner, mode="ppo"),
                                       obs=dataclasses.replace(self.cfg.obs, own_tera_type=bool(own_tera_type)))
        self.own_tera_type = own_tera_type
        self.action_futures: Dict[str, Any] = {}
        self.sem = asyncio.Semaphore(10 ** 9)
        self.players: Dict[str, Any] = {}
        self.queue = _Queue(self)
        self.steps: List[Dict[str, Any]] = []
        self.pick = None
        self.current_request: Optional[Dict[str, Any]] = None
        self.current_player: Any = None
        self.assembler = None

    def player(self, username: str) -> Any:
        if username not in self.players:
            from poke_env import AccountConfiguration, ServerConfiguration
            p = self.worker.SyncBridgePlayer(
                cfg=self.cfg, event_queue=self.queue, action_futures=self.action_futures, ep_sem=self.sem,
                account_configuration=AccountConfiguration(username, None),
                server_configuration=ServerConfiguration("ws://127.0.0.1:1/showdown/websocket", "http://127.0.0.1:1/action.php?"),
                max_concurrent_battles=0, battle_format=FORMAT, start_listening=False, log_level=40)
            p.ps_client.send_message = _noop_send
            self.players[username] = p
            self.assembler = p.assembler
        return self.players[username]

    def _on_step(self, tag: str, obs: np.ndarray) -> int:
        room = tag.split("_", 1)[1]
        battle = self.current_player._battles.get(room)
        a0, a1 = self.assembler.offsets["action_mask"]
        mask = obs[a0:a1].astype(np.float32)
        step = {"obs": obs, "mask": mask, "request": self.current_request, "battle": battle}
        # The battle is as it was when this observation was assembled (the worker sends the step
        # right after assembling): the moment value_gate.py reads Phi and the material.
        try:
            step["turn"] = int(getattr(battle, "turn", 0) or 0)
            step["phi"] = float(self.potential(battle))
            step["mat"] = float(self.material(battle)) if self.material is not None else float("nan")
        except Exception:                                 # a reading never changes what is rebuilt
            step.setdefault("turn", -1)
            step["phi"], step["mat"] = float("nan"), float("nan")
        action = self.pick(step) if self.pick is not None else None
        if action is None or not (0 <= action < len(mask)) or mask[action] < 0.5:
            legal = np.flatnonzero(mask > 0.5)
            action = int(legal[0]) if len(legal) else 0
        step["battle"] = None
        self.steps.append(step)
        return action

    def run(self, username: str, frames: List[Tuple[str, Optional[Dict[str, Any]]]], pick=None) -> List[Dict[str, Any]]:
        """Feeds the frames to the player `username`; one step per choose_move. pick(step) -> action index or None."""
        from poke_env.concurrency import POKE_LOOP
        player = self.player(username)
        self.steps, self.pick, self.current_player = [], pick, player

        async def go() -> None:
            for frame, req in frames:
                if req is not None:
                    self.current_request = req
                await player.ps_client._handle_message(frame)

        asyncio.run_coroutine_threadsafe(go(), POKE_LOOP).result(timeout=600)
        return self.steps


# =============================================================================================
# The action a choice line names, on our 14 actions
# =============================================================================================

def map_choice(choice: str, request: Optional[Dict[str, Any]], battle: Any) -> Tuple[Optional[int], str]:
    """(action index, "") or (None, why not)."""
    parts = choice.split()
    if not parts or request is None or battle is None:
        return None, "no_request"
    kind = parts[0]
    if kind == "move" and len(parts) >= 2:
        arg = parts[1]
        tera = "terastallize" in parts[2:]
        if any(x in parts[2:] for x in ("mega", "zmove", "dynamax", "max", "ultra")):
            return None, "gimmick_not_tera"
        if arg.isdigit():
            try:
                moveid = request["active"][0]["moves"][int(arg) - 1]["id"]
            except Exception:
                return None, "move_number_not_in_request"
        else:
            moveid = re.sub(r"[^a-z0-9]", "", arg.lower())
        mon = battle.active_pokemon
        if mon is None:
            return None, "no_active"
        ids = [m.id for m in mon.moves.values()]
        if moveid in ids:
            idx = ids.index(moveid)
        elif moveid in ("struggle", "recharge"):
            idx = 0                                       # as the live mapping (map_order_to_index) has them
        else:
            return None, "move_not_in_known_moves"
        if idx >= 4:
            return None, "move_slot_over_4"
        return idx + (10 if tera else 0), ""
    if kind == "switch" and len(parts) >= 2:
        try:
            n = int(parts[1])
            ident = request["side"]["pokemon"][n - 1]["ident"]
        except Exception:
            return None, "switch_not_in_request"
        team = list(battle.team.values())
        mon = battle.team.get(ident)
        if mon is None:
            try:
                mon = battle.get_pokemon(ident)
            except Exception:
                mon = None
        if mon is None or mon not in team:
            return None, "switch_target_not_in_team"
        idx = team.index(mon)
        if idx >= 6:
            return None, "switch_slot_over_6"
        return 4 + idx, ""
    return None, "choice_" + kind


# =============================================================================================
# One game
# =============================================================================================

def rebuild_side(h: Harness, replay_id: str, side: str, sim: Dict[str, Any], il: Dict[str, Any]) -> Dict[str, Any]:
    names = {s: str(il["players"].get(s, {}).get("name") or "") for s in ("p1", "p2")}
    room = "battle-%s-%s" % (FORMAT, replay_id.rsplit("-", 1)[-1])
    frames = frames_for_side(room, side, sim[side], names)
    choices = list(il["choices"][side])
    reasons: Counter = Counter()
    labels: List[int] = []
    pos = [0]

    def pick(step: Dict[str, Any]) -> Optional[int]:
        i = pos[0]
        pos[0] += 1
        if i >= len(choices):
            labels.append(-1)
            reasons["no_choice_line"] += 1
            return None
        a, why = map_choice(choices[i], step["request"], step["battle"])
        if a is None:
            labels.append(-1)
            reasons[why] += 1
            return None
        if step["mask"][a] < 0.5:
            labels.append(-1)
            reasons["label_not_legal_in_mask"] += 1
            return None
        labels.append(a)
        return a

    steps = h.run(ALIAS[side], frames, pick)
    nonwait = sum(1 for _f, r in frames if r is not None and not r.get("wait") and not r.get("teamPreview"))
    aligned = len(choices) in (len(steps), len(steps) - 1) and len(steps) == nonwait
    winner = None
    for chunk in sim["omniscient"]:
        for line in chunk.split("\n"):
            if line.startswith("|win|"):
                winner = line[5:]
            elif line.startswith("|tie"):
                winner = ""
    won = -1 if winner is None or winner == "" else (1 if winner == names[side] else 0)
    return {"steps": steps, "labels": labels, "reasons": reasons, "aligned": aligned, "n_choices": len(choices),
            "n_requests": nonwait, "won": won}


# =============================================================================================
# Validation on our own recorded ladder game(s) that an opponent made public
# =============================================================================================

def field_groups(asm: Any) -> Dict[str, Tuple[int, int]]:
    g: Dict[str, Tuple[int, int]] = {}
    for name, (a, b) in asm.offsets.items():
        if name in ("pokemon_body", "pokemon_ids", "ability_ids", "move_ids", "move_scalars"):
            half = (b - a) // 2
            g[name + ".own"] = (a, a + half)
            g[name + ".opp"] = (a + half, b)
        else:
            g[name] = (a, b)
    return g


def compare_steps(asm: Any, truth: List[np.ndarray], rebuilt: List[np.ndarray]) -> Dict[str, Any]:
    groups = field_groups(asm)
    n = min(len(truth), len(rebuilt))
    out: Dict[str, Any] = {"decisions_truth": len(truth), "decisions_rebuilt": len(rebuilt), "groups": {}}
    whole = 0
    for i in range(n):
        whole += int(np.array_equal(truth[i], rebuilt[i]))
    out["whole_vector_identical"] = [whole, n]
    for g, (a, b) in groups.items():
        same = sum(int(np.array_equal(truth[i][a:b], rebuilt[i][a:b])) for i in range(n))
        out["groups"][g] = [same, n]
    return out


def compare_game(h: Harness, rep: Dict[str, Any], recording: str) -> Dict[str, Any]:
    """One recorded side of a game (the truth: the frames as received) against its rebuild from the inputlog."""
    room, user, live_frames, notes = frames_from_recording(recording)
    il = parse_inputlog(rep["inputlog"])
    side = next(s for s in ("p1", "p2") if re.sub(r"[^a-z0-9]", "", str(il["players"][s]["name"]).lower()) == user)
    key = build_key(il["version"] or rep.get("version"))
    build = os.path.join(BUILDS, key) if key else None
    if build is None or not os.path.exists(os.path.join(build, "dist", "sim", "battle-stream.js")):
        return {"error": "no cached build (this copy never builds Showdown)"}
    sim = resim(build, [{"key": rep["id"], "inputlog": rep["inputlog"]}])[rep["id"]]
    ok, at = same_log(rep["log"], sim["omniscient"])
    truth_steps = h.run(user, live_frames)
    rb = rebuild_side(h, rep["id"], side, sim, il)
    res = compare_steps(h.assembler, [s["obs"] for s in truth_steps], [s["obs"] for s in rb["steps"]])
    played = [d.get("played") for d in notes]
    n = min(len(played), len(rb["labels"]))
    res.update({"log_identical": ok, "log_first_mismatch": at, "side": side,
                "masks_identical": [sum(int(np.array_equal(a["mask"], b["mask"])) for a, b in zip(truth_steps, rb["steps"])), min(len(truth_steps), len(rb["steps"]))],
                "recorded_mask_equals_rebuilt": [sum(int(list(map(int, d["mask"])) == list(map(int, s["mask"]))) for d, s in zip(notes, rb["steps"]) if "mask" in d), sum(1 for d in notes[:len(rb["steps"])] if "mask" in d)],
                "action_equal_to_played": [sum(int(a == b) for a, b in zip(played, rb["labels"])), n],
                "label_reasons": dict(rb["reasons"]), "aligned": rb["aligned"], "choices": rb["n_choices"], "requests": rb["n_requests"]})
    mism = [i for i in range(n) if played[i] != rb["labels"][i]]
    res["action_mismatches"] = [{"i": i, "played": played[i], "rebuilt": rb["labels"][i],
                                 "choice": il["choices"][side][i] if i < len(il["choices"][side]) else None} for i in mism[:5]]
    return res


def add_up(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tot: Dict[str, Any] = {"games_sides": len(results), "errors": sum(1 for r in results if "error" in r)}
    ok = [r for r in results if "error" not in r]
    for key in ("whole_vector_identical", "masks_identical", "recorded_mask_equals_rebuilt", "action_equal_to_played"):
        tot[key] = [sum(r[key][0] for r in ok), sum(r[key][1] for r in ok)]
    tot["groups"] = {}
    for r in ok:
        for g, (a, b) in r["groups"].items():
            x = tot["groups"].setdefault(g, [0, 0])
            x[0] += a
            x[1] += b
    tot["decisions_truth"] = sum(r["decisions_truth"] for r in ok)
    tot["decisions_rebuilt"] = sum(r["decisions_rebuilt"] for r in ok)
    tot["logs_identical"] = sum(1 for r in ok if r["log_identical"])
    tot["aligned"] = sum(1 for r in ok if r["aligned"])
    tot["label_reasons"] = dict(sum((Counter(r["label_reasons"]) for r in ok), Counter()))
    tot["action_mismatches"] = [m for r in ok for m in r["action_mismatches"]][:10]
    return tot


def cmd_own(args: argparse.Namespace) -> int:
    own_tera = resolve_own_tera(args.own_tera_type)
    h = Harness(own_tera, abilities_setting(getattr(args, "obs_abilities", None)))
    rep = json.load(open(args.replay, encoding="utf-8"))
    res = compare_game(h, rep, args.recording)
    res["own_tera_type"] = own_tera
    res["obs_abilities"] = h.obs_abilities
    print(json.dumps(res, indent=1, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return 0


def cmd_dev(args: argparse.Namespace) -> int:
    own_tera = resolve_own_tera(args.own_tera_type)
    h = Harness(own_tera)
    logs = {}
    for p in glob.glob(os.path.join(os.path.expanduser(args.logs), "**", "*.log.json"), recursive=True):
        d = json.load(open(p, encoding="utf-8"))
        rid = str(d.get("roomid") or os.path.basename(p)[:-9]).replace("battle-", "", 1)
        logs[rid] = {"id": rid, "inputlog": NL.join(d["inputLog"]) if isinstance(d["inputLog"], list) else d["inputLog"],
                     "log": NL.join(d["log"]) if isinstance(d["log"], list) else d["log"], "version": args.version}
    results = []
    for rec in sorted(glob.glob(os.path.join(os.path.expanduser(args.dev), "*.jsonl"))):
        rid = os.path.basename(rec).split(".")[0].replace("battle-", "", 1)
        if rid not in logs:
            results.append({"error": "no server log"})
            continue
        try:
            results.append(compare_game(h, logs[rid], rec))
        except Exception as e:
            results.append({"error": "%s: %s" % (type(e).__name__, str(e)[:200])})
    tot = add_up(results)
    tot["own_tera_type"] = own_tera
    tot["server_logs"] = len(logs)
    tot["error_kinds"] = dict(Counter(r["error"] for r in results if "error" in r))
    print(json.dumps(tot, indent=1, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"total": tot, "games": results}, f, indent=1, default=str)
    return 0


# =============================================================================================
# The dataset
# =============================================================================================

def resolve_own_tera(setting: str) -> bool:
    if setting in ("on", "off"):
        return setting == "on"
    ckpts = sorted(glob.glob(os.path.expanduser("~/runs/main/checkpoints/learner_update_*.pt")))
    if not ckpts:
        return False
    import torch
    sys.path.insert(0, REPO)
    from config import checkpoint_own_tera_type
    ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    return checkpoint_own_tera_type(ck)


def load_replay(path: str) -> Optional[Dict[str, Any]]:
    try:
        if path.endswith(".gz"):
            import gzip
            with gzip.open(path, "rt", encoding="utf-8") as f:
                d = json.load(f)
        else:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
    except Exception:
        return None
    if not isinstance(d, dict) or not d.get("inputlog") or not d.get("log") or not str(d.get("id", "")).startswith(FORMAT + "-"):
        return None
    return d


def plan(paths: List[str], bar: int) -> Tuple[Dict[str, List[Tuple[str, List[str]]]], Counter]:
    """{version: [(path, [qualifying sides])]} and the counts of what was left out and why."""
    by_version: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    c: Counter = Counter()
    seen = set()
    for p in paths:
        d = load_replay(p)
        if d is None:
            c["not_a_gen9randombattle_replay_with_inputlog"] += 1
            continue
        if d["id"] in seen:
            c["duplicate"] += 1
            continue
        seen.add(d["id"])
        c["replays"] += 1
        il = parse_inputlog(d["inputlog"])
        if il["start"].get("formatid") != FORMAT:
            c["format_not_" + FORMAT] += 1
            continue
        if not is_rated(il):
            c["unrated"] += 1
            continue
        sides = [s for s in ("p1", "p2") if (side_elo(il, s) or 0) >= bar]
        c["sides_at_bar_%d" % len(sides)] += 1
        if not sides or not il["version"]:
            continue
        by_version[il["version"]].append((p, sides))
    return by_version, c


def abilities_setting(value: Optional[str]) -> str:
    """--obs-abilities as the shards record it: "" (off) or the obs_abil.py settings, comma separated, in order."""
    if not value:
        return ""
    raise SystemExit("this branch has no --obs-abilities: its observation is the worker's own")


def work_version(task: Tuple) -> Dict[str, Any]:
    version, items, slot, own_tera, out_dir, shard_no = task[:6]
    abil = task[6] if len(task) > 6 else ""
    exclude_ids = set(task[7]) if len(task) > 7 else set()
    exclude_users = {u for u in os.environ.get("WH_EXCLUDE_USERS", "").split(",") if u}
    counts: Counter = Counter()
    reasons: Counter = Counter()
    key = build_key(version)
    build = os.path.join(BUILDS, key) if key else None
    note = "cached"
    if build is None or not os.path.exists(os.path.join(build, "dist", "sim", "battle-stream.js")):
        build, note = None, "not built"
        counts["games_no_build"] += len(items)
        return {"version": version, "counts": counts, "reasons": reasons, "note": note, "shards": []}
    replays = {}
    for p, sides in items:
        d = load_replay(p)
        if d["id"] in exclude_ids:
            counts["games_excluded_listed_battle"] += 1
            continue
        if exclude_users:
            il0 = parse_inputlog(d["inputlog"])
            ids = {re.sub(r"[^a-z0-9]", "", str(il0["players"].get(s, {}).get("name") or "").lower()) for s in ("p1", "p2")}
            if ids & exclude_users:
                counts["games_excluded_bot_account"] += 1
                continue
        replays[d["id"]] = (d, sides)
    if not replays:
        return {"version": version, "counts": counts, "reasons": reasons, "note": note, "shards": []}
    sims = resim(build, [{"key": k, "inputlog": d["inputlog"]} for k, (d, _s) in replays.items()])
    h = Harness(own_tera, abil)
    eps: List[Dict[str, Any]] = []
    for rid, (d, sides) in replays.items():
        sim = sims.get(rid)
        if sim is None or sim.get("err"):
            counts["games_resim_error"] += 1
            continue
        ok, _at = same_log(d["log"], sim["omniscient"])
        if not ok:
            counts["games_log_differs"] += 1
            continue
        il = parse_inputlog(d["inputlog"])
        counts["games_kept"] += 1
        for side in sides:
            try:
                rb = rebuild_side(h, rid, side, sim, il)
            except Exception as e:
                counts["sides_harness_error"] += 1
                reasons["harness_error " + type(e).__name__] += 1
                continue
            if not rb["aligned"]:
                counts["sides_choices_misaligned"] += 1
                continue
            reasons.update(rb["reasons"])
            other = "p2" if side == "p1" else "p1"
            obs = np.stack([s["obs"] for s in rb["steps"]]).astype(np.float16) if rb["steps"] else None
            if obs is None:
                counts["sides_no_steps"] += 1
                continue
            masks = np.stack([s["mask"] for s in rb["steps"]]) > 0.5
            act = np.array(rb["labels"] + [-1] * (len(rb["steps"]) - len(rb["labels"])), dtype=np.int16)
            legal = masks.sum(1).astype(np.int8)
            train = (act >= 0) & (legal > 1)
            counts["sides"] += 1
            counts["steps"] += len(act)
            counts["steps_single_legal_action"] += int((legal <= 1).sum())
            counts["steps_unlabelled"] += int((act < 0).sum())
            counts["steps_train"] += int(train.sum())
            counts["steps_train_tera"] += int((train & (act >= 10)).sum())
            counts["steps_train_switch"] += int((train & (act >= 4) & (act <= 9)).sum())
            eps.append({"obs": obs, "act": act, "mask": masks.astype(np.uint8), "train": train.astype(np.uint8),
                        "turn": np.array([s.get("turn", -1) for s in rb["steps"]], dtype=np.int16),
                        "phi": np.array([s.get("phi", np.nan) for s in rb["steps"]], dtype=np.float32),
                        "mat": np.array([s.get("mat", np.nan) for s in rb["steps"]], dtype=np.float32),
                        "game": salted("g", rid), "player": salted("u", str(il["players"][side].get("name"))),
                        "side": 1 if side == "p1" else 2, "elo": side_elo(il, side) or 0, "opp_elo": side_elo(il, other) or 0,
                        "won": rb["won"], "time": int(d.get("uploadtime") or 0), "version": version[:10]})
    shards = []
    if eps:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "shard-%05d-%s.npz" % (shard_no, version[:10]))
        starts = np.cumsum([0] + [len(e["act"]) for e in eps]).astype(np.int64)
        np.savez_compressed(
            path + ".tmp.npz",
            obs=np.concatenate([e["obs"] for e in eps]), act=np.concatenate([e["act"] for e in eps]),
            mask=np.concatenate([e["mask"] for e in eps]), train=np.concatenate([e["train"] for e in eps]),
            turn=np.concatenate([e["turn"] for e in eps]), phi=np.concatenate([e["phi"] for e in eps]),
            mat=np.concatenate([e["mat"] for e in eps]),
            ep_start=starts, ep_game=np.array([e["game"] for e in eps]), ep_player=np.array([e["player"] for e in eps]),
            ep_side=np.array([e["side"] for e in eps], dtype=np.int8), ep_elo=np.array([e["elo"] for e in eps], dtype=np.int16),
            ep_opp_elo=np.array([e["opp_elo"] for e in eps], dtype=np.int16), ep_won=np.array([e["won"] for e in eps], dtype=np.int8),
            ep_time=np.array([e["time"] for e in eps], dtype=np.int64), ep_version=np.array([e["version"] for e in eps]),
            own_tera_type=np.array(own_tera), obs_abilities=np.array(abil))
        os.replace(path + ".tmp.npz", path)
        shards.append(os.path.basename(path))
    return {"version": version, "counts": counts, "reasons": reasons, "note": note, "shards": shards}


def cmd_build(args: argparse.Namespace) -> int:
    import multiprocessing as mp
    paths = sorted(set(p for pat in (args.replays or []) for p in glob.glob(os.path.expanduser(pat))))
    if args.from_done:
        with open(os.path.expanduser(args.from_done)) as f:
            paths = sorted(set(paths) | {p for d in json.load(f).values() for p in d.get("paths", [])})
    exclude_ids: List[str] = []
    if args.exclude_ids:
        with open(os.path.expanduser(args.exclude_ids)) as f:
            exclude_ids = sorted({l.strip().replace("battle-", "", 1) for l in f if l.strip()})
    log("excluded battle ids %d, excluded user ids %d (in memory)" % (
        len(exclude_ids), len([u for u in os.environ.get("WH_EXCLUDE_USERS", "").split(",") if u])))
    by_version, c = plan(paths, args.bar)
    log("replay files %d; %s" % (len(paths), dict(c)))
    if not getattr(args, "only_built", False):
        fetch_missing(list(by_version))
    own_tera = resolve_own_tera(args.own_tera_type)
    abil = abilities_setting(getattr(args, "obs_abilities", None))
    log("own_tera_type", own_tera, "obs_abilities", abil or "off")
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, "done.json")
    done = json.load(open(done_path)) if os.path.exists(done_path) else {}
    # Builds first, one version at a time per slot (a build is 1 core), then the games in parallel.
    versions = sorted(by_version, key=lambda v: -len(by_version[v]))
    tasks = []
    shard_no = max([int(x.split("-")[1]) for d in done.values() for x in d.get("shards", [])] + [-1]) + 1
    for i, v in enumerate(versions):
        items = [it for it in by_version[v] if it[0] not in set(done.get(v, {}).get("paths", []))]
        if not items:
            continue
        for j in range(0, len(items), args.games_per_shard):
            tasks.append((v, items[j:j + args.games_per_shard], 0, own_tera, out_dir, shard_no, abil, exclude_ids))
            shard_no += 1
    # Each version is built once before its tasks run (slot = process index keeps worktrees apart).
    total: Counter = Counter()
    reasons: Counter = Counter()
    notes: Counter = Counter()
    started = time.time()
    for v in versions:
        if getattr(args, "only_built", False):
            # Read only: a version without a cached build is left out (its games count as
            # games_no_build), nothing is fetched, checked out or built under HUMAN_DIR.
            key = build_key(v)
            ok = bool(key) and os.path.exists(os.path.join(BUILDS, key, "dist", "sim", "battle-stream.js"))
            notes["cached" if ok else "not built (--only-built)"] += 1
            continue
        try:
            b, note = ensure_build(v, 0)
        except Exception as e:
            b, note = None, "error %s" % e
        notes[note] += 1
    log("builds: %s" % dict(notes))
    with mp.get_context("spawn").Pool(args.procs) as pool:
        for k, res in enumerate(pool.imap_unordered(work_version, tasks)):
            total.update(res["counts"])
            reasons.update(res["reasons"])
            d = done.setdefault(res["version"], {"paths": [], "shards": []})
            d["shards"] += res["shards"]
            if k % 10 == 0:
                log("%d/%d tasks, %s" % (k + 1, len(tasks), dict(total)))
    for t in tasks:
        done.setdefault(t[0], {"paths": [], "shards": []})["paths"] += [p for p, _s in t[1]]
    with open(done_path, "w") as f:
        json.dump(done, f)
    summary = {"bar": args.bar, "own_tera_type": own_tera, "obs_abilities": abil, "plan": dict(c), "versions": len(versions), "builds": dict(notes),
               "counts": dict(total), "label_reasons": dict(reasons), "seconds": round(time.time() - started, 1)}
    with open(os.path.join(out_dir, "summary-%s.json" % time.strftime("%Y%m%d-%H%M%S")), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("own", help="compare a rebuilt game with our own recording of it")
    o.add_argument("--replay", required=True)
    o.add_argument("--recording", required=True)
    o.add_argument("--out")
    o.add_argument("--own-tera-type", default="auto", choices=["auto", "on", "off"])
    o.add_argument("--obs-abilities", default=None, help="obs_abil.py settings, as for build")
    v = sub.add_parser("dev", help="compare dev-server games (human/devgames.py) with their rebuilds")
    v.add_argument("--dev", default=os.path.join(HOME, "dev"))
    v.add_argument("--logs", default=os.path.join(HOME, "server", "logs"))
    v.add_argument("--out")
    v.add_argument("--version", default="14546894d86f9589ac11130c510bbe73b6968665", help="the dev server's commit (its logs carry no >version line)")
    v.add_argument("--own-tera-type", default="auto", choices=["auto", "on", "off"])
    b = sub.add_parser("build", help="rebuild replays into npz shards")
    b.add_argument("--replays", nargs="+", default=[], help="globs of replay .json files (Showdown's replay JSON)")
    b.add_argument("--from-done", default="", help="also every replay path listed in this done.json (an earlier build's list)")
    b.add_argument("--exclude-ids", default="", help="file of battle ids to leave out, one per line")
    b.add_argument("--out", default=os.path.join(HOME, "data"))
    b.add_argument("--bar", type=int, default=2100)
    b.add_argument("--procs", type=int, default=10)
    b.add_argument("--games-per-shard", type=int, default=200)
    b.add_argument("--own-tera-type", default="auto", choices=["auto", "on", "off"])
    b.add_argument("--obs-abilities", default=None,
                   help="obs_abil.py settings ('all' or a list), as train.py --obs-abilities: the observation of a run "
                        "with them. Recorded in every shard (obs_abilities; absent or '' = off)")
    b.add_argument("--only-built", action="store_true",
                   help="never fetch, check out or build Showdown under HUMAN_DIR: versions without a cached build are "
                        "left out (for a second rebuild beside the collector's own)")
    args = ap.parse_args()
    return {"own": cmd_own, "dev": cmd_dev, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
