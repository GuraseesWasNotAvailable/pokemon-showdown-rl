# The moveset block (what each Pokemon can do)

One block appended after the matchup block (bench/notes/OBS_MATCHUP.md). Nothing before it moves. It
is written only with the observation setting `movesets` (`ObservationAssembler(matchup=True,
movesets=True)`, `run_cfg["obs"]["movesets"]`, `train.py --obs-matchup --obs-movesets`), which needs
the matchup block. Off (the default) the observation, the network and a PPO update are byte for byte
and bit for bit what efe43f7 gives (tests/test_obs_movesets.py, 12,048 recorded decisions).

Code: `obs_movesets.py` (the block, the draws, the SetHead's candidate tables), `obs_matchup.py`
(the role filter it shares, the labels), `ppo_core.py` (the model's use of it, the SetHead's move and
Tera parts), `tools/widen_checkpoint.py --movesets`.

## Layout

The observation with own Tera types is 8,567 numbers; with the matchup block 9,635; with this block
10,295.

| Part | Offsets | Size |
|---|---|---|
| everything before (body ... action mask, matchup block) | 0 - 9,635 | unchanged |
| Pokemon | 9,635 - 10,295 | 12 x 55 = 660 |

**Pokemon** `i` (the observation's order: ours 0-5, theirs 6-11) at `9,635 + 55*i`:

| Within | What |
|---|---|
| 0-7 | candidate moves not shown yet: move vocabulary ids (`vocab.json` "move.id", +1; 0 = none), most likely first (ties by name) |
| 8-15 | their probabilities (top 8; the rest of the mass is left out) |
| 16-34 | Tera type: a probability over bug, dark, dragon, electric, fairy, fighting, fire, flying, ghost, grass, ground, ice, normal, poison, psychic, rock, steel, water, stellar. All 0 once it has Terastallized |
| 35-40 | threat to each of the 6 Pokemon on the OTHER side: expected best log2 damage multiplier of its damaging moves (shown and not shown); a status move counts as -4, so a Pokemon with no damaging move scores -4. 0 for a slot not seen yet |
| 41-46 | probability that it has a super-effective move (multiplier above 1) against each of the 6 |
| 47-54 | probability that it has: a set-up move, a priority attack, recovery, a pivot move, a status move (sleep, paralysis, burn, poison, Yawn), a hazard, hazard removal, phazing (Haze counts) |

**Our own side is exact** (the request names everything): its 4 moves with probability 1, its Tera
type one-hot from the request (`obs_assembler.own_tera_type_name`), threat and roles exact.

**Theirs**: the random battle roles of its species that fit what it has shown, exactly the roles
`obs_matchup.opponent_guess` keeps (`obs_matchup.kept_roles`), equally likely. Within a role, the
moves not shown are drawn as `4 - (moves shown)` of the role's moves not shown yet, every subset
equally likely, so a move's probability is the role-weighted `min(1, free / left)`, the best of a
draw is worked out exactly (`top_probs`: the i-th best of n is the best drawn with probability
C(n-i, s-1)/C(n, s)), and "has a move of kind X" is `1 - C(n-k, s)/C(n, s)`. Not modelled: the
generator's own rules (a role that always takes its set-up move, STAB first, and so on); the
SetHead learns what the uniform draw misses. Tera: each role's Tera types equally likely within it.

The multipliers are the matchup block's: `obs_matchup.effectiveness` (matchup.py) with what is known
of the defender (revealed or only-possible ability, known item), the same numbers as the move
tokens. Every move gets its full type, `obs_matchup.move_type(full=True)`, as `matchup.move_type`
has it: the attacker's ability (Pixilate, Galvanize, Aerilate, Refrigerate, Normalize, Liquid
Voice), Tera Starstorm of a Terastallized Terapagos (Stellar), Revelation Dance (the user's first
type, its Tera type after), Aura Wheel, Weather Ball in the battle's weather (a known Utility Umbrella
hides sun and rain), besides Tera Blast, Ivy Cudgel, Raging Bull and Judgment. A move not shown is
typed and judged with its role's ability while the Pokemon's is not known (Primarina's Psychic Noise
is Water in its Liquid Voice role, Psychic in no role). Not modelled: Air Lock / Cloud Nine, Morpeko's
Hangry form (poke-env keeps the species "morpeko" through Hunger Switch: Aura Wheel reads Electric).

**The matchup block with this block on**: its move tokens use the same full types
(`MatchupEncoder(full_types=True)`, set by `ObservationAssembler(movesets=True)`), so our side's
threat is the best of its own move tokens. Without this block the move tokens keep efe43f7's types
(Tera Blast, Ivy Cudgel, Raging Bull, Judgment only), which every matchup checkpoint so far was
trained on: the matchup-only observation is byte for byte efe43f7's. The difference touches the
species whose sets hold such a move or ability (Sylveon, Golem-Alola, Primarina, the four Oricorio,
Pelipper, Politoed, Sunflora, Victreebel, Terapagos; Morpeko's Aura Wheel reads Electric either
way): on the recorded streams 1,909 move
tokens (of 12,048 decisions) are retyped and 961 of them change numbers. Before it, Sylveon's Hyper
Voice read Normal (immune on a Ghost, neutral on a Dragon), Galvanize Double-Edge Normal, Revelation
Dance Normal, Liquid Voice Psychic Noise Psychic, Stellar Tera Starstorm Normal.

**Illusion** (`obs_movesets.illusion_adjust`): poke-env leaves the moves an Illusion user used in
disguise on the Pokemon it posed as, also after `|replace|`. Once an Illusion user (Zoroark,
Zoroark-Hisui: a species whose sets have Illusion) is in the opposing team, every other opposing
Pokemon (not Transformed, with sets) leaves out of its moveset numbers the shown moves its species'
sets do not hold, and the Illusion user counts as shown those of them its own sets hold. A move both
sets hold stays where poke-env put it (exact attribution needs the `|replace|` line, which the
encoder does not see). Every battle without a revealed Illusion user is untouched. The matchup block
(its move tokens, `opponent_guess`) still reads poke-env's moves, as in efe43f7.

Roles of the moves (`obs_movesets.move_flags`): set-up from `setup_moves.is_setup` (a Ghost's Curse
is not); priority: a damaging move with priority above 0 (Protect is not); recovery: Recover, Roost,
Slack Off, Soft-Boiled, Milk Drink, Moonlight, Morning Sun, Synthesis, Shore Up, Strength Sap, Wish,
Rest, Pain Split (and Jungle Healing, Lunar Blessing, Life Dew, Heal Order); pivot: U-turn, Volt
Switch, Flip Turn, Parting Shot, Teleport, Chilly Reception, Shed Tail; status: a status move whose
effect is sleep, paralysis, burn or poison, and Yawn (Nuzzle is an attack); hazards: Stealth Rock,
Spikes, Toxic Spikes, Sticky Web, Stone Axe, Ceaseless Edge; removal: Rapid Spin, Defog, Mortal Spin,
Tidy Up, Court Change; phazing: Roar, Whirlwind, Dragon Tail, Circle Throw, Haze.

## The model

`PokeTransformer` turns it on when `meta` has `"movesets"` (and then needs `"matchup"`).
`pokemon_net[0]` gets 103 more input columns AFTER the matchup block's: `sum_k p_k * move_emb(id_k)`
of the 8 candidates with the existing move embedding (64), then the block's 39 dense numbers (Tera,
threat, super-effective, roles). In: 1,741 -> 1,844. `_wide_subnet` adds one product per part
(`W_old x_old + b + W_mu x_mu + W_ms x_ms`), so zero new columns give exactly the matchup network's
numbers.

## SetHead

With the block, `attach_set_head` builds the head with two more layers after the ability and item
ones: `moves` (Linear d_model -> 351, one logit per move id) and `tera` (Linear d_model -> 20),
limited at the loss to what the species' sets allow (`obs_movesets.candidate_tables`, not saved).

- Labels: `build_set_labels(..., movesets=True)` -> int32 `[6, 8]` per decision = the worker's
  48-integer `set_labels` column (18 without the block): species, ability, item, then the 4 true
  moves (`+id` not shown yet: a label; `-id` shown: neither label nor candidate; 0 none), then the
  true Tera type (index + 1, 0 once it has Terastallized).
- Loss: per opposing Pokemon with at least one true move among its candidates, the binary
  cross-entropy averaged over its candidates (the species' set moves not shown yet); the Tera type a
  cross-entropy over the species' Tera types. Each counts as one entry with the ability and item
  ones: `loss = (sum of all) / (number of entries)`.
- Logged: `set_mv_n`, `set_mv_acc` (mean precision at k: the share of its k highest candidates that
  are true, k = its true moves not shown), `set_tera_n`, `set_tera_acc`, next to `set_ab_acc` /
  `set_it_acc`.

## Widening a matchup checkpoint

    python tools/widen_checkpoint.py --movesets --in S_matchup.pt --out S_movesets.pt [--set-head]

Copies every tensor; `pokemon_net.0.weight` gets 103 zero columns in `model`, `target_model`, the
fp32 master and both Adam moments; `move_net` is untouched. An existing SetHead keeps its weights and
Adam state, its new `moves` / `tera` layers start fresh (seeded, no Adam state); an OppActionHead is
kept. Writes `run_cfg["obs"]["movesets"] = True` and `widened_movesets`. Refuses a checkpoint without
the matchup block, or one that has the moveset block already. The learner refuses a checkpoint whose
movesets setting is not the run's; `cloud/check_restart.py` refuses a start that switches it (exit 2).

**Applying it to a pipeline run** (e.g. the from-scratch run, ~/runs/v2/scratch): the trainer and
the servers start from `repo_dir` of run.json, so a release of this code, the widened checkpoint as
the newest `learner_update_NNNNNN.pt`, and `--obs-movesets` in `train_defaults.extra_train_args`:

    # on the laptop: the release (R) on the box
    git -C <worktree> archive <commit> | ssh <box> 'mkdir -p ~/rel/v2-movesets-<commit> && tar -x -C ~/rel/v2-movesets-<commit>'
    # on the box
    R=~/rel/v2-movesets-<commit>; RUN=~/runs/v2/scratch; CK=$RUN/checkpoints
    mkdir -p $R/heuristic/cache && cp ~/rel/v2-0bf9e93/heuristic/cache/gen9randombattle.json $R/heuristic/cache/
    sudo systemctl stop psppo-v2-scratch-train                  # the supervisor saves and stops the trainer
    N=$(ls -t $CK/learner_update_*.pt | head -1)
    mkdir -p $RUN/pre_movesets && cp -p $N $RUN/run.json $RUN/pre_movesets/
    cd $R && ~/venv/bin/python tools/widen_checkpoint.py --movesets --in $N --out $RUN/pre_movesets/widened.pt
    NEXT=$(printf 'learner_update_%06d.pt' $(( 10#$(basename $N .pt | tr -dc 0-9) + 1 )))
    cp $RUN/pre_movesets/widened.pt $CK/$NEXT                    # the newest file is the one resumed
    ~/venv/bin/python -c "import json,sys; p=sys.argv[1]; c=json.load(open(p)); c['repo_dir']=sys.argv[2]; \
        a=c['train_defaults']['extra_train_args']; a.append('--obs-movesets') if '--obs-movesets' not in a else None; \
        json.dump(c, open(p,'w'), indent=1)" $RUN/run.json $R
    ~/venv/bin/python $R/cloud/check_restart.py --config $RUN/run.json --run-dir $RUN   # must end "OK to start"
    sudo systemctl start psppo-v2-scratch-train

The evaluator unit (if installed) runs evaluate.py from its own release directory: point it at `$R` too
(`sudo sed -i "s#$HOME/rel/v2-0bf9e93#$R#g" /etc/systemd/system/psppo-v2-scratch-eval.service`,
`sudo systemctl daemon-reload`, restart it), or it cannot load the widened network.

## Cost

The encoder (`MovesetEncoder`) reads the MatchupEncoder's guesses and defenders for the same
decision and keeps what it works out in one store per process, shared by every assembler there (a
rollout worker has one per player): a profile per (Pokemon, what it has shown), the log2 code per
(move, defender), the threat numbers per (profile, defender) and the 55 numbers per (profile, the six
defenders). A defender is keyed by what a multiplier reads of it (types, Tera state, known ability,
known item), a move without an attacker ability matchup.py ignores, an opposing Pokemon's profile by
the MatchupEncoder's cached guess for it. Each cache is emptied on its own at its limit
(`_LIMITS`, at most some 600 MB per process). `encode_reference` works everything out again with no
cache; the two write the same bytes on every recorded decision.

On the recorded streams (12,048 decisions, box c, one process per arm, 8 assemblers as in a worker):
0.86 ms per assemble with the block against 0.72 with the matchup block alone (0.36 without either):
+0.14 ms a decision. Most of what is left is first-time work: verdicts for (move, defender) pairs not
seen before (~35%) and profiles of new Pokemon states (~20%).

Live the workers are the bottleneck (single-threaded, ~93% busy while the 16-vCPU box is half idle),
and there everything costs about 2.5 times its replay time (contention, the GIL shared with the
worker's inference thread): inside the workers, with the store, the block's encoder took 0.33-0.36 ms
a decision and the garbage collector 0.07 ms more (instrumented runs; the first version, one cache
per assembler that emptied all at once, took 0.43-0.47).

Decisions per second of the smoke runs below (updates 2-20, ~680k decisions each), in the order
A B B A, each on freshly started servers:

| Run | --obs-movesets | decisions/s | s per update |
|---|---|---|---|
| A6 | no | 1,782 | 7.7 |
| B6 | yes | 1,609 | 8.1 |
| B7 | yes | 1,603 | 8.1 |
| A7 | no | 1,793 | 7.8 |

10% fewer decisions a second on this worker-bound box (the update itself is unchanged). Earlier
runs on servers that had been up for several runs, with the first version of the encoder: 13 to 17%.
On a box whose rollout workers are not the bottleneck (the imitation phase is held by the PSBot
teacher) the cost is smaller; with spare cores, one more server and worker buys it back.

The SetHead's labels with moves and Tera type cost 0.02 ms a decision more than without (0.044
against 0.022).

## Smoke run

    python tools/widen_checkpoint.py --movesets --in ckpts/scratch_matchup_000211.pt --out ckpts/ms/learner_update_000211.pt
    python train.py --mode ppo --ports 10700-10703 --battles 256 --rooms-per-pair 16 --minibatch 768 --seq-len 256 \
        --steps-per-update 36864 --ckpt-dir <copy> --obs-matchup --obs-movesets --set-head-coef 0.1

from the from-scratch run's imitation checkpoint 211 (matchup block, no SetHead; the SetHead starts
fresh), 20 updates each, box c (g6e.4xlarge, 1 L40S, 16 vCPU, 4 servers). The same without
`--obs-movesets` from the unwidened checkpoint is the comparison.

The SetHead at update 20 (its labels: the partner seat's team, as for abilities and items):

| | B6 | B7 | A6 | A7 |
|---|---|---|---|---|
| set_mv_acc (precision at k over the moves not shown) | 0.706 | 0.713 | | |
| set_tera_acc | 0.432 | 0.441 | | |
| set_ab_acc | 0.681 | 0.692 | 0.666 | 0.646 |
| set_it_acc | 0.631 | 0.638 | 0.617 | 0.629 |

The ability and item heads are no worse with the move and Tera layers beside them. Not measured: what
the observation's own candidate probabilities alone would score on the same labels (the baseline the
move and Tera numbers should be read against).

## Tests (tests/test_obs_movesets.py, tests/test_widen.py)

    PSPPO_STREAMS=<streams.jsonl> PSPPO_MOVESETS_BASE=<git archive efe43f7> PSPPO_MOVESETS_CKPT=<matchup ckpt> \
    PSPPO_WIDEN_MS_CKPT=<matchup ckpt> PSPPO_WIDEN_MS_SH_CKPT=<matchup ckpt with a SetHead> CUDA_VISIBLE_DEVICES=0 \
        python -m pytest tests/test_obs_movesets.py tests/test_widen.py -q -p no:cacheprovider -s

- the block on hand-made battles: our side exact (its 4 moves, Tera type, roles, threat; the
  threat equals the matchup block's own move tokens); an opponent's candidates and roles narrow as
  it shows moves; the Tera vector is 0 after the Tera; all against a brute force that enumerates every
  draw of every role from the sets file;
- full types: Sylveon (-1 on Gengar, +1 on Garchomp and Dragonite), Primarina's Psychic Noise shown
  and not, Terapagos-Stellar's Tera Starstorm (as "terapagosstellar" and as a Terastallized
  "terapagos"), Oricorio-Sensu's Revelation Dance before and after its Tera, Golem-Alola's Galvanize,
  one Sunflora's Weather Ball through sun, no weather, rain, sand and an Umbrella; the matchup
  block's token with and without the block;
- Illusion, on a poke-env battle fed the protocol lines: in disguise nothing changes; after
  `|replace|` Garchomp's and Zoroark's rows are those of a Garchomp that showed nothing and a Zoroark
  that showed the three moves, and stay so when the real Garchomp shows two of its own;
- 12,048 recorded decisions: fast path = reference path byte for byte; a second assembler sharing the
  store on half the decisions writes the same bytes; 7,213 opposing Pokemon against the brute force,
  10,512 of ours against matchup.py directly; fewer candidates the more moves are shown (6.25, 4.47,
  3.28, 2.13, 0 for 0-4 shown); before the block, the observation is the matchup-only one except move
  tokens whose full type differs from efe43f7's (1,909 retyped, 961 with other numbers);
- setting off: the plain and the matchup observation byte for byte efe43f7's on all 12,048 decisions;
  the network bit for bit (fp32 and bf16); a whole learner update with matchup + SetHead (+
  OppActionHead) and an imitation update bit for bit efe43f7's (once in six runs move_emb's Adam
  moments differed in the last bits, the GPU summing its gradient over repeated ids in no fixed order;
  weights and masters stayed equal);
- widening the from-scratch run's checkpoint 211: logits, value logits and values bit for bit on
  2,063 recorded decisions, fp32 and bf16, stepped and packed (the original network reads the
  moveset observation's prefix, which differs from the matchup-only observation only in retyped move
  tokens, as the test checks); a scrambled block changes nothing, a
  non-zero column does; the tool's file equals `widen()`'s; a PPO checkpoint with a SetHead keeps the
  head's weights and Adam state and gets fresh, seeded move and Tera layers;
- the label column (48 integers), the head's loss by hand, its gradient only on labelled tokens,
  an update that trains the head and logs set_mv_acc / set_tera_acc; train.py's flag, check_restart's
  refusal, the learner's refusal, hybrid.PolicyNet and evaluate's checks on a widened checkpoint.
