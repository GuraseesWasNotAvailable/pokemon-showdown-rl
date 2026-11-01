# Pokemon Showdown RL

A 14.5M-parameter transformer that plays Pokemon Showdown's gen 9 random battles. It is trained by PPO
self-play on local Showdown servers and played on the ladder by a Windows app that can put a
Monte Carlo search beside it. It reached a peak ladder rating of **2,466**.

The full story is in **[docs/DETAILS.md](docs/DETAILS.md)**: every training run, every test and what
did not work. How the game became a training environment is in
**[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)**.

## What is in here

| Folder | What it is |
|---|---|
| [`training/`](training/) | The RL environment and the trainer: local Showdown servers, the poke-env client and its fixes, the observation, the reward, PPO on Ray, and the league, imitation and evaluation code |
| [`app/`](app/) | The playing side: the window app "PSBot Lookahead", the ladder runner `ladder.py`, the bridge to Foul Play's search, and PSBot, the hand-written heuristic bot used as the first teacher |
| [`docs/`](docs/) | The write-ups |
| [Release v1.0](https://github.com/GuraseesWasNotAvailable/pokemon-showdown-rl/releases/tag/v1.0) | The trained weights and the built app |

## Results

| | |
|---|---|
| Public ladder | 1,407 rated games, peak **2,466**, playing level about 2,330 to 2,400 |
| Best network alone vs Foul Play (88,000 iterations a sampled world) | **60.2%** of 400 games, +67 to +75 Elo |
| With the search beside it | 59 to 66% vs the same Foul Play |
| Training speed | 4,300 to 6,200 learned decisions a second, 2,304 battles at once, one cloud machine |
| Correctness | the app and the trainer agree byte for byte on 12,048 recorded decisions |

Every network from five separate training lines ends within about 30 Elo of the others. The
write-up explains what moved the number (the learning rate coming down) and what did not.

## Weights

All in [release v1.0](https://github.com/GuraseesWasNotAvailable/pokemon-showdown-rl/releases/tag/v1.0), about 32 MB each, network only.

| File | What it is |
|---|---|
| `learner_update_024381.pt` | **Best.** Main run final, 900M decisions. First in every rating fit |
| `learner_update_011850.pt` | Main run at 437M. Played the 2,466 ladder peak |
| `v2-league_update_008124_300M.pt` | **Latest.** League run final: 24381 trained 300M more against frozen past networks |
| `v2-league_update_005600_207M.pt` | League run at 207M, its best checkpoint test (54.8% vs 24381) |
| `v2-topstart_update_016127_595M.pt` | Top-player run final: started from 140k games of strong human players |
| `v2-scratch_update_016041_591M.pt` | From-scratch run final, every technique on |
| `PSBot.Lookahead.exe` | The window app, built from `app/app/PSBotLookahead.cs` |

## Quick start

**Play it in the window app (Windows).**
1. `cd app`, then `python -m venv .venv` and `.venv\Scripts\pip install -r requirements.txt`.
2. Put `PSBot.Lookahead.exe` in `app/` and the weights in `app/weights/`.
3. Start the exe, pick a network and "network only". The search modes also need Foul Play inside WSL
   (`bench/install_foulplay.sh`).
4. Test on a local server first: `PSBot.Lookahead.exe --test-server 8000 --test-user tester`.

**Train.** Follow [`training/TRAINING.md`](training/TRAINING.md): install the `rlspawn` plugin into a
Pokemon Showdown checkout, start the servers with `start_servers.py`, then run imitation, warmup
and PPO with `train.py`, or let `pipeline.py` run all three.

## How it works, in five lines

1. **Environment.** Local Showdown servers keep thousands of random battles running between our own
   players. poke-env parses the protocol, and the code fixes what it gets wrong.
2. **Observation.** 10,295 numbers a decision: 12 Pokemon, 48 moves, the field, last turn's events,
   the legal actions, plus blocks of type matchups and likely movesets.
3. **Model.** 64 tokens a turn through a 3-layer transformer that also sees the last 64 turns. It
   outputs 14 actions (4 moves, 6 switches, 4 Tera moves) and a 51-bin value.
4. **Training.** It imitates PSBot first, then plays PPO self-play with a win/loss reward plus
   potential-based shaping fitted on top players' games.
5. **Play.** The app runs the network on the ladder. In search mode, Foul Play's tree search vetoes
   moves it finds clearly bad.

## Responsible use

Since 20 October 2026 Showdown only matches accounts registered as bots against players who opted in
with `/allowbots`. Train and test on local servers. On the public ladder, use a registered bot
account and do not try to get around the opt-in. The app plays one bot per account, sends no chat
and keeps the battle timer on.

## Credits and licence

Built by **Gurasees Singh Dhanoa** ([@GuraseesWasNotAvailable](https://github.com/GuraseesWasNotAvailable))
and **[@UberMayinch](https://github.com/UberMayinch)**, equal contributors. It builds on
**[Nebraskinator/ps-ppo](https://github.com/Nebraskinator/ps-ppo)**, the transformer and PPO setup
it started from.

MIT, see [LICENSE](LICENSE). Not included and licensed separately:
- [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) (MIT);
- [poke-env](https://github.com/hsahovic/poke-env) (MIT);
- [Foul Play](https://github.com/pmariglia/foul-play) (GPL-3.0). The bridge files import it at run
  time, so a program that combines them is covered by the GPL;
- random-battle set data from [pkmn/randbats](https://github.com/pkmn/randbats).

Pokemon is a trademark of Nintendo, Creatures and Game Freak. This project is not affiliated with
them or with Pokemon Showdown.
