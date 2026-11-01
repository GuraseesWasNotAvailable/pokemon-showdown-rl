# One bot from two: PokeTransformer + PSBot

This branch merges two Pokémon Showdown random-battle bots into one: the transformer policy
from [Nebraskinator/ps-ppo](https://github.com/Nebraskinator/ps-ppo) (this repository's `main`)
and PSBot, a rule-based bot with a full damage calculator that plays inside the Showdown web
client.

## What each side brings

| | PokeTransformer (upstream) | PSBot |
|---|---|---|
| What it is | A transformer policy trained by imitation, then PPO self-play | A hand-written evaluator: damage calc, speed and priority, switch and Tera logic |
| Strength | Upstream reports 1900+ ladder Elo after about two days on an RTX 3090 | Plays at once, no training |
| What it is missing | No trained weights are published, and it has no damage calculator | It looks one turn ahead and cannot learn |
| Runs in | Python (poke-env, Ray, PyTorch) | The browser, and Node for the headless engine |

## How they are merged

There is one implementation of the heuristic, `psbot/psbot.js`. The browser bot is that file.
`heuristic/engine.js` runs the same file under Node, on top of the Showdown client's own battle
code, so Python can ask it for a move. It is used in three places.

1. **Teacher.** Upstream bootstraps the policy by imitating poke-env's `SimpleHeuristicsPlayer`.
   `worker.py` can now imitate PSBot instead (`RolloutConfig.imitation_teacher`, default
   `"psbot"`). PSBot beats that original teacher in about nine games out of ten (numbers below),
   and unlike it, PSBot Terastallizes, so the policy starts from Tera play it would otherwise have
   to discover.
2. **Guardrail and prior at play time.** `hybrid.py` scores each of the 14 actions as
   `policy_weight * log p(action) + beta * heuristic(action)` and drops any action PSBot rates
   more than `veto_gap` below its own best. The policy chooses among the tactically sound
   options; the damage calculator keeps it out of immunities and KOs.
3. **Fallback.** With no checkpoint, the merged player is PSBot alone.

## State of the weights

The upstream repository publishes code only. No checkpoint exists in its history, releases or
forks, on Hugging Face, or in the author's Reddit thread, where the author describes the repo as
a record of the method and not a runnable package. So the policy half of the merged bot has to
be trained before it contributes anything. Until then `--agent hybrid` and `--agent psbot` play
the same moves.

Training has not been run on this branch. What has been checked is everything around it: the
model builds and runs a forward pass, the teacher feeds the worker legal actions, and the hybrid
player runs end to end with an untrained network.

## Layout

| Path | What it is |
|---|---|
| `psbot/` | The browser bot: `psbot.js`, its position tests, the Windows launcher. See `psbot/README.md`. |
| `heuristic/engine.js` | Runs `psbot.js` headless. JSON lines on stdin and stdout. |
| `heuristic/setup.js` | Fetches the Showdown client files and set data the engine needs into `heuristic/cache/`. |
| `heuristic/test.js` | Runs the position tests headless. |
| `heuristic/replay.js` | Replays games the browser bot recorded and shows where the current bot would choose differently. |
| `psbot_bridge.py` | The engine as a Python object, and `PSBotMixin`, which gives any poke-env player PSBot's opinion. |
| `players.py` | `PSBotPlayer`, and the stat fix for `SimpleHeuristicsPlayer` (moved here from `worker.py`). |
| `hybrid.py` | `HybridPlayer`: the policy and PSBot deciding together. |
| `play.py` | Plays matches on a local server and reports the win rate with its interval. |
| `check_teacher.py` | Checks the imitation teacher inside the worker's own player class, without Ray. |
| `rl_reward/` | The training reward: win/loss plus potential-based shaping on a position value fitted to 2987 expert games. `RL_REWARD.md` has the design and the evidence. |
| `check_reward.py` | Checks the reward's properties on live games through the worker's own player class, without Ray. |

Changed upstream files: `worker.py` (teacher switch, PSBot mixin) and `config.py`
(`imitation_teacher`). Everything else on `main` is untouched.

## Setup

```
pip install -r requirements.txt
node heuristic/setup.js            # once; add --all-gens for generations before 9
```

A local Showdown server, the same one training uses:

```
git clone https://github.com/smogon/pokemon-showdown
cd pokemon-showdown && node pokemon-showdown start --no-security
```

For training, copy `pokemon-showdown/server/chat-plugins/rlspawn.ts` from this repo into that
server first, as upstream requires.

## Commands

```
node heuristic/test.js                                   # 23 position tests
node heuristic/replay.js <battles folder> --losses       # the current bot on recorded games
python check_teacher.py --games 40                       # teacher actions are legal
python play.py --agent psbot  --opponent simple --games 300
python play.py --agent hybrid --opponent simple --games 300          # PSBot alone until a checkpoint exists
python play.py --agent hybrid --checkpoint checkpoints/learner_update_200.pt --opponent psbot
python play.py --agent policy --checkpoint checkpoints/learner_update_200.pt --opponent simple
```

`--beta`, `--veto-gap` and `--policy-weight` set the mix. They are untuned: they can only be
tuned against a trained checkpoint. Start from the defaults and measure with `play.py`.

Training itself is unchanged from upstream: start the servers (`start_servers.sh`), set
`LearnerConfig.mode` to `imitation`, run `train.py`, then switch the mode to `ppo`.

## Results measured on this branch

All games are gen9randombattle on a local server. Intervals are 95% Wilson intervals.

| Check | Result |
|---|---|
| Position tests (`node heuristic/test.js`) | 23 of 23 pass, the same file and results as in the browser |
| PSBot against the upstream teacher (`SimpleHeuristicsPlayer` with the stat fix) | 2,717 of 3,000 games, 90.6% (89.5 to 91.6) on the current build; the builds before it measured between 89.7% and 90.8% |
| PSBot as the imitation teacher inside `worker.SyncBridgePlayer` | 40 games, 2,482 recorded steps, every action inside the action mask, 77 of them Tera moves |
| The original teacher through the same path | 20 games, 872 steps, all legal, no Tera moves |
| Upstream model on this machine | builds (14.5M parameters at the current config), forward pass runs on CPU and CUDA |
| Hybrid player with an untrained network and PSBot (`beta` 4) | 28 of 30 against the upstream teacher: the prior carries it, as it should |
| Policy alone, untrained network | 3 of 20 against the upstream teacher: the policy path runs end to end and plays at random, as it should |
| Engine speed | about 170 decisions a second per engine process, with no fallback in more than 300,000 decisions |

For scale, upstream reports that its trained agent wins more than 85% against the same
`SimpleHeuristicsPlayer`. That is a comparison against one fixed opponent, not a ladder rating.

Not measured: anything involving a trained policy, and anything on the public ladder from Python.

## Credits and licences

The transformer, training code and observation encoding are Nebraskinator's, MIT licensed (see
`LICENSE`). The additions on this branch are under the same licence. The Showdown client's battle
code is not part of this repository: `heuristic/setup.js` downloads it from
play.pokemonshowdown.com when you run it.
