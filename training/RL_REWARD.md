# The reward for training the policy

The reward is the result of the game (+1 win, -1 loss, 0 tie) plus, on every decision, the
discounted change in a fitted estimate of how good the position is:

```
r_t = scale * (gamma * Phi(s_{t+1}) - Phi(s_t))        every decision but the last
r_T = result + scale * (0 - Phi(s_T))                   the last decision
```

`Phi` runs from about -1 (lost) to +1 (won). `scale` is 0.5 and `gamma` is the learner's own
discount (0.999). The shaping terms cancel over a game, so the policy that maximises this reward
is exactly the policy that wins most. What the shaping adds is timing: the agent hears about a
good or bad turn on that turn, not twenty decisions later with the result.

It is on by default (`RewardConfig.shaping = "potential"` in `config.py`). The code is in
`rl_reward/`, and `worker.py` calls it.

## Why this form

Summed over a game with the learner's discount, the rewards above come to

```
gamma^(T-1) * result  -  scale * Phi(s_0)
```

`Phi(s_0)` is the position at the first decision and nothing the agent does changes it. So two
ways of playing are ranked by this reward exactly as they are ranked by winning. This is
potential-based shaping, and it is the only form of added reward with that guarantee
([Ng, Harada and Russell, 1999](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html)).
Three details make it hold here:

| Detail | Why it matters |
|---|---|
| The finished game has potential 0 | In a game that ends, a non-zero potential at the end changes what is being optimised ([Grzes, 2017](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf)). The old per-faint reward fails on exactly this. |
| The shaping uses the learner's gamma | With a different discount the terms no longer cancel. |
| `Phi` has no constant term and is an odd function of "mine minus theirs" | The same position from the other seat has the opposite potential, so self-play stays zero-sum. |

Shaping of this kind is also the same thing as starting the critic at `Phi` instead of at zero
([Wiewiora, 2003](https://arxiv.org/pdf/1106.5267)): the critic only has to learn what the
potential gets wrong.

Returns lie in `[-1 - scale * cap, 1 + scale * cap]`. With scale 0.5 and the fitted cap of 0.98
that is +/-1.49, inside the value head's existing `[-1.5, 1.5]` support.

## What Phi is

`Phi(s) = cap * tanh(w . f(s))`. `f(s)` is 19 numbers, each "mine minus theirs" (`rl_reward/features.py`):
Pokemon left, HP left, each of those again as a share of the total (one Pokemon matters more at
2 v 1 than at 6 v 5), the active Pokemon's HP, counts of each status, boosts and drops on the
active Pokemon, hazards and screens on each side, who still has a Tera, and how many Pokemon
each side has not shown yet. Unseen Pokemon count as alive at full HP.

The weights were fitted on the 2987 highest-rated public Gen 9 random battle replays (rated 2351
to 2547, 91,293 turn states), against who went on to win. The largest weights:

| Feature | Weight | | Feature | Weight |
|---|---|---|---|---|
| HP share | +1.44 | | Burned | -0.95 |
| HP difference | +1.28 | | Paralysed | -0.94 |
| Pokemon left | +1.19 | | Poisoned | -0.88 |
| Screens up | +0.61 | | Asleep or frozen | -0.64 |
| Attack, Sp. Atk and Speed boosts | +0.51 | | Spikes on my side | -0.44 |
| Tera still unused | +0.29 | | Stealth Rock on my side | -0.17 |

The games are parsed by poke-env's own protocol parser and the features come from the same
function the training worker calls on the live battle, so the potential the agent trains on is
the one that was fitted and checked.

## Evidence

All numbers below are out-of-fold (five folds split by game) and scored from both seats.

**How well each potential predicts the winner.** A coin flip scores 0.693.

| Potential | Log loss | AUC | Early third | Middle | Late |
|---|---|---|---|---|---|
| Pokemon left only (all a per-faint reward sees) | 0.644 | 0.635 | 0.691 | 0.669 | 0.576 |
| Pokemon left, HP, status (Metamon and poke-env style) | 0.626 | 0.684 | 0.691 | 0.654 | 0.537 |
| Full (this one) | 0.598 | 0.718 | 0.681 | 0.631 | 0.490 |

It is calibrated: positions it calls 54%, 65%, 75%, 85% and 94% were won 54%, 63%, 72%, 82%
and 92% of the time. On 420 ladder games PSBot recorded, which it was not fitted on, it scores
0.587 against 0.632 for Pokemon left only.

**What each reward design does on those games.**

| Design | Critic's target (mean square) | First-half reward vs result | 6-0 win minus 6-5 win |
|---|---|---|---|
| Win/loss only | 0.965 | none | 0 |
| +/-0.1 per faint (upstream ps-ppo) | 1.154 | 0.23 | 0.50 |
| Metamon's default, rescaled | 0.983 | 0.26 | 0.05 |
| Potential, scale 0.5 (this one) | 0.850 | 0.36 | 0 |

- The per-faint reward makes the critic's job harder than no shaping at all (1.154 against
  0.965) and pays half a win extra for a wide margin.
- The potential lowers the critic's target by 12% and its first-half total tracks the result
  best.
- One number does not favour it. Per unit of reward paid, a faint points at the eventual winner
  slightly more often (0.18) than a change in potential does (0.14), because the potential also
  moves on small HP changes.

**Two facts about the game that set expectations.**

- The side that took the first knockout won 60.2% of the 2965 expert games. Early material is
  a weak sign of who wins.
- In the first third of a game no potential beats a coin flip by more than 0.012 of log loss.
  Shaping helps mostly in the middle and late game. It will not teach openings.

**Live check.** `check_reward.py` plays games through the worker's own player class on a local
server. Over 30 games (1334 decisions): no potential outside the cap, no return outside the
value head's support, the cancellation identity held to 5e-8, the potential at the last decision
had the sign of the result in 54 of 60, and the two seats' potentials correlated at -0.84.

## What others used, and what happened

| Agent | Reward | What they reported |
|---|---|---|
| [Metamon](https://arxiv.org/abs/2504.04395) (UT Austin, Gen 1 to 4 and 9 OU) | `r_hp + 0.5 r_stat + r_faint + 100 r_win`, gamma 0.999 | "We do find some qualitative evidence of models exploiting the shaped terms. For example, our agents tend to cling to life in clearly lost positions by using recovery moves." Their code also ships [a second reward](https://github.com/UT-Austin-RPL/metamon) that pays 200 for a win and 0 for a loss, described as discouraging the "tendency to cling to lost positions. Gamble and try to win!" |
| [ps-ppo](https://github.com/Nebraskinator/ps-ppo) (upstream of this repo, Gen 9 random) | "shaped to promote winning (+/-1 for win/loss) by a large margin (+/-0.1 for faint)" | Author reports 1900+ on the ladder. |
| [Huang and Lee, 2019](https://www.yuzeh.com/assets/CoG-2019-Pkmn.pdf) (Gen 7 random) | +/-1, -0.0125 when own Pokemon faints, +0.0025 for a super effective move | 1677 Glicko-1. |
| [VGC-Bench](https://arxiv.org/abs/2506.10326) (Gen 9 doubles) | +/-1 at the end only, gamma 1.0 | A benchmark comparing behaviour cloning, self-play and population methods. |
| [OpenAI Five](https://cdn.openai.com/dota-2.pdf) | Shaped, "modeled loosely after potential-based shaping functions, though the guarantees therein do not apply", made zero-sum by subtracting the enemy's reward | Used shaping to simplify credit assignment. |
| [Suphx](https://arxiv.org/abs/2003.13590) (Mahjong) | A learned predictor of the final result supplies the training signal for each round | The closest precedent for a fitted potential. |

I could not open the MIT thesis on Gen 4 random battles
([Wang, 2024](https://dspace.mit.edu/handle/1721.1/153888)), so its reward is not in the table.

## What was rejected, and why

| Idea | Why not |
|---|---|
| +/-0.1 per faint | Not zero at the end of the game, so it pays for margin, and it raises the critic's target variance. |
| HP gained and damage dealt as direct reward | Metamon's own report: the agent heals in lost positions to collect it. As a potential the same information is safe. |
| A bonus for an action (super effective move, setting hazards) | It pays for doing a thing, not for a position, so it can always be farmed. |
| A penalty per turn | Breaks zero-sum, and pushes toward rushing. Random battles end on their own. |
| Win 200, loss 0 | Under discounting it removes the pull to drag out a lost game, which is real but only a few percent of a game's value at gamma 0.999. Worth an ablation with gamma, not a default. |
| Scale 1.0 | Lowers the critic's target a little more (0.853 against 0.883, both without discounting) but needs the value head widened to +/-2. |

## Settings

```python
RewardConfig(
    terminal_win=1.0, terminal_loss=-1.0, terminal_tie=0.0,
    shaping="potential",     # "potential", "faint" (upstream), or "none"
    shaping_scale=0.5,
    potential_file="",       # empty: rl_reward/weights.json
)
```

Use the same reward in every phase (imitation, warmup, PPO). The critic is trained in all of
them, and changing its target between phases throws that work away.

## Ablations worth running once training is up

| Run | Change | What it answers |
|---|---|---|
| A | `shaping="none"` | Does shaping help at all at this scale of training? |
| B | `shaping="faint"` | The upstream reward, as the baseline. |
| C | default | This reward. |
| D | `shaping_scale=1.0`, `v_min=-2`, `v_max=2` | Is the stronger signal worth the wider value head? |
| E | `gamma=1.0` | Does removing the discount stop stalling in lost games? |

Judge them by win rate against a fixed opponent pool at equal environment steps, not by reward:
the designs pay on different scales.

## Running it

```
node heuristic/toplay.js fetch --pages 60                 # public replays, highest rated first
.venv/Scripts/python.exe -m rl_reward.fit_potential --write   # fit, report, write weights.json
.venv/Scripts/python.exe -m rl_reward.compare_rewards         # the design comparison above
.venv/Scripts/python.exe -m rl_reward.test_reward             # 9 property tests
.venv/Scripts/python.exe check_reward.py --games 30           # live check, needs the local server
```

## Limits

- No policy has been trained with it yet. Everything above is measured on recorded games and on
  live games played by scripted players.
- The potential explains about 15% of the result's variance. That is what the game allows from
  these features, not a tuning problem: the outcome of a random battle is mostly not decided by
  material until late.
- The reward is not the main thing standing between this repo and a strong policy. Compute and
  the opponent pool are. Upstream reports two days on an RTX 3090 for 250M states.
- The potential was fitted on human games. If the trained agent plays very differently, the
  potential may read its positions less well. That costs learning speed, not correctness, since
  any potential leaves the best policy unchanged.
