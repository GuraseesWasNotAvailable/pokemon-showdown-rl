# The matchup block (stage 4: abilities and items)

One block appended after the action mask of the flat observation. Nothing before it moves.
It is written only with the observation setting `matchup` (`ObservationAssembler(matchup=True)`,
`run_cfg["obs"]["matchup"]` in a checkpoint). Without it the observation and the network are
byte for byte and bit for bit what they were (tests/test_obs_matchup.py, 11,841 recorded decisions).

Code: `obs_matchup.py` (the block, the matchup function, set candidates, SetHead labels),
`randbats_sets_gen9.json` (the pinned set data), `ppo_core.py` (the model's use of it, SetHead),
`tools/widen_checkpoint.py`.

## Layout

The observation with own Tera types (this code line) is 8,567 numbers; with the block, 9,635.

| Part | Offsets | Size |
|---|---|---|
| everything before (body ... action mask) | 0 - 8,567 | unchanged |
| move tokens | 8,567 - 9,431 | 48 x 18 = 864 |
| Pokemon | 9,431 - 9,635 | 12 x 17 = 204 |

**Move token** `t = 4*i + k` (Pokemon `i` in the observation's order, its `k`-th move in
`list(mon.moves.values())`, the same order as the move tokens), at `8,567 + 18*t`:

| Within | What |
|---|---|
| 0-5 | log2 of the damage multiplier against each of the 6 Pokemon of the OTHER side (our moves: the opponent's slots 6-11; theirs: ours 0-5). Immune: -4. Otherwise clamped to [-3, 3]. Status moves: 0. |
| 6-11 | possible immunity: the probability that an ability or item the defender may have but has not shown makes it immune (Levitate, Water Absorb, Air Balloon...). 0 when the answer is known. |
| 12-17 | 1 where that defender is known (seen), 0 for a slot not seen yet (its numbers are 0 too). |

**Pokemon** `i` at `9,431 + 17*i`:

| Within | What |
|---|---|
| 0-2 | ability candidates, vocabulary ids (`vocab.json` "pokemon.ability", +1; 0 = none) |
| 3-5 | their probabilities, most likely first (ties by name) |
| 6-9 | item candidates, vocabulary ids ("pokemon.item", +1) |
| 10-13 | their probabilities (top 4; the rest of the mass is left out, so the sum can be below 1) |
| 14 | item unknown (poke-env still says `unknown_item`): the corrected flag |
| 15 | a Choice item possible (known Choice item, or a Choice item among the candidates) |
| 16 | item gone (knocked off, used, popped; or our own Pokemon holds nothing) |

Our own side is exact (the request names ability and item). The opponent's: a revealed ability or
item is exact; otherwise the species' random battle roles, kept when they hold every move shown
(and the shown ability, item and Tera type), equally likely, each role's abilities and items
equally likely within it. A form the sets do not list falls back on its Pokedex battleOnly,
changesFrom, baseSpecies. An ability that is the only one the kept roles allow counts as known.

The old item flag (`obs_pokemon.py`, `1.0 if not mon.item`: 0 for an unknown item, 1 for a
removed one) and the Pokedex ability slots are left as they are.

**The matchup function** (`obs_matchup.effectiveness`, `possible_immunity`): poke-env's gen 9
type chart; immunities Levitate, Flash Fire, Well-Baked Body, Water Absorb, Volt Absorb, Storm
Drain, Lightning Rod, Motor Drive, Dry Skin, Sap Sipper, Earth Eater, Wind Rider, Bulletproof,
Soundproof, Wonder Guard, Air Balloon; halving Thick Fat, Heatproof, Purifying Salt; attacker
Scrappy/Mind's Eye, Tinted Lens, Freeze-Dry, Flying Press, Thousand Arrows, Tera Blast (and
Stellar), Ivy Cudgel, Raging Bull, Judgment; fixed-damage moves count only an immunity; a Stellar
Tera keeps the original types for defence. Not modelled: Mold Breaker and kin, Weather Ball,
Gravity, Iron Ball, Magnet Rise. With the moveset block on (`ObservationAssembler(movesets=True)`,
`MatchupEncoder(full_types=True)`) the move tokens take every move's full type,
`move_type(full=True)`: the attacker's ability (Pixilate, Galvanize, Liquid Voice and the others), Tera
Starstorm, Revelation Dance, Weather Ball in the battle's weather (bench/notes/OBS_MOVESETS.md);
without it they stay as above, byte for byte what the matchup checkpoints were trained on. When `matchup.py` (branch v2-reads) is merged these two
functions should call it; any difference changes the block, so switch before a network trains
on it.

**Set data**: `randbats_sets_gen9.json`, a pinned copy of `heuristic/cache/gen9randombattle.json`
(509 species, md5 e6385a98c69048be4f57e5207d4bf350 recorded in the file and in a widened
checkpoint's `widened.sets_md5`). The cache is not read at run time, so a new fetch cannot change
the observation. `python obs_matchup.py --build-sets <gen9randombattle.json>` rewrites it.

Cost: about 0.13 ms per decision on the box over 0.29 ms for the rest of the observation with the old local chart (fast path; caches warm up over a run). With matchup.py inside (train-v2) the block costs 0.26 ms per decision on the recorded streams (0.57 against 0.31 ms per assemble), with one process-wide cache of matchup verdicts. Live, that is about 25% of the rollout's decisions per second: 2,470 against 3,300 in 50-update smoke runs on 4 servers (the update itself is unchanged, 3.4 s). The SetHead labels cost nothing measurable (2,518 with them, 2,499 without).

## The model

`PokeTransformer` turns the block on when `meta` has `"matchup"` (the assembler's meta with
matchup=True):
- `move_net[0]` gets 18 more input columns: the move token's 18 numbers;
- `pokemon_net[0]` gets 99 more: `sum_k p_k * ability_emb(id_k)` (64), `sum_k p_k * item_emb(id_k)`
  (32) with the existing embeddings, and the three flags.

The new columns are the last ones. `_wide_subnet` computes `W_old x_old + b + W_new x_new`, the
first product with the old shapes, so zero new columns give exactly the old numbers.

## Widening a checkpoint

    python tools/widen_checkpoint.py --in S.pt --out S_matchup.pt [--set-head]

Copies every tensor; the new columns of `move_net.0.weight` (242 -> 260) and
`pokemon_net.0.weight` (1,642 -> 1,741) are zero in `model`, `target_model`, the fp32 masters and
both Adam moments; `step` and every hyperparameter stay; the optimizer's groups are laid out by
`LearnerActor._init_optimizer` itself and every master is checked against its weight. Writes
`run_cfg["obs"]["matchup"] = True` and a `widened` entry. `--set-head` adds a fresh SetHead (no
Adam state). On the full run's arm C checkpoint, 2,002 recorded decisions: logits, value logits
and values bit for bit equal in fp32 and in bf16, stepped with history and packed
(tests/test_widen.py).

## SetHead

`net.attach_set_head(coef)` (after `make_model`, before the optimizer is built: its parameters
come last and land in the "transformer" zone). Linear heads over the ability and item
vocabularies on the opponent tokens Z_t[7:13], limited at the loss to the species' set
candidates (`obs_matchup.set_candidate_tables`, non-persistent buffers).
- Loss: `ppo_core.compute_set_head_loss(net, z_tokens, labels, coef=None)` -> `(coef * loss,
  stats)`; `z_tokens` = Z_t `[B, 64, D]`, `labels` `[B, 6, 3]` = (species id, true ability id,
  true item id); 0 = no label. Coefficient 0 by default. Stats are tensors (no GPU sync).
- Labels: `obs_matchup.build_set_labels(battle, true_team, vocab_map=assembler.vocab_map)`:
  labels only what the observing side has not seen. `true_team`: in self-play the partner seat's
  `battle.team` (same keys as `battle.opponent_team`); for Foul Play seats that seat's team; for
  rebuilt top-player games a dict of the same keys -> `{"species", "ability", "item"}`.

## Wired into training (train-v2)

- **config.py**: `ObsConfig.matchup` (default False); `make_model` and `make_assembler` build with the
  block when it is on. `obs_matchup.build_model` still works (it only sets the default).
- **train.py**: `--obs-matchup` and `--set-head-coef C` (refused without `--obs-matchup`).
- **learner.py**: refuses a checkpoint whose `run_cfg.obs.matchup` differs from the run's (the input
  layers differ in size); attaches the SetHead after `make_model` and before the optimizer, never to the
  EMA target; its loss is `compute_set_head_loss` on Z_t from `forward(..., return_z=True)` times C.
- **ppo_core.py**: the heads' parameters are in their own zone, "aux" (the learner's last optimizer
  groups, `aux_wd` / `aux_stable`, at the full rate in every mode that trains). A checkpoint without a
  head resumes with every other parameter's Adam state; the head starts fresh
  (`learner.optimizer_state_for_layout`). `tools/widen_checkpoint.py --set-head` lays the head out the
  same way.
- **worker.py**: with the head on, every STEP records `build_set_labels(battle, partner.team)`, the
  partner being the other seat of the pair (`SyncBridgePlayer.partner`, set by the vector wrapper and by
  the seats' wrapper), as an 18-integer `set_labels` column.
- **obs_matchup.py**: `effectiveness` and `possible_immunity` call `matchup.matchup` (with the move's
  type as worked out here), so the block and the read reward use one function.
- **cloud/check_restart.py**: `obs_settings` include `matchup`; a start that switches it on or off is
  refused (exit code 2) even with `--allow-obs-change`, since no checkpoint loads across it.

**A widened start**, exactly:

    cd ~/dev/<arm>
    ~/venv/bin/python tools/widen_checkpoint.py --in S.pt --out S_matchup.pt
    mkdir -p ckpt && cp S_matchup.pt ckpt/learner_update_000000.pt     # the newest file in --ckpt-dir is resumed
    ~/venv/bin/python train.py --mode ppo --ckpt-dir ckpt --obs-matchup --set-head-coef 0.1 ...

## What the wiring had to do (kept for reference)

- **config.py**: `ObsConfig.matchup: bool = False`; `make_model` builds from
  `ObservationAssembler(matchup=self.obs.matchup)`, `make_assembler` passes it too. Then
  `obs_matchup.build_model` can call `make_model` directly.
- **train.py**: `--obs-matchup` (refuse it on a checkpoint without the block unless it was
  widened; `check_restart.py` refuses an observation change without `--allow-obs-change`),
  `--set-head-coef`.
- **learner.py**: read `checkpoint_obs_matchup` like `checkpoint_own_tera_type`; with a SetHead,
  `attach_set_head(coef)` on `net` between `make_model` and `_init_optimizer` (the EMA zip in
  the target update stays aligned since its parameters come last); add the term from Z_t
  (stage 2's `forward(..., return_z=True)`) and the `set_labels` extras column.
- **worker.py**: with the head on, at `choose_move` build `build_set_labels(battle,
  partner_battle.team)` and send it as an 18-integer extras column per step.
- **human/rebuild.py**: `--obs-matchup` (assembler with the block; labels from the exact sets),
  rebuild the shards; `human/imitate.py` checks the setting like `own_tera_type`.
- **decide app** (`hybrid.PolicyNet`, `obs_search`): for a checkpoint with
  `obs_matchup.checkpoint_obs_matchup(ckpt)`:
  1. build the network with the block: `obs_matchup.build_model(RunConfig.default(), matchup=True)`
     (or `make_model` after the config change), and drop `set_head.*` keys before the strict
     load check (or `attach_set_head()` first);
  2. this line: `ObservationAssembler(own_tera_type=..., matchup=True)`; the decide line's
     `SearchOwnTeraObservationAssembler`: keep its observation and append
     `obs_matchup.MatchupEncoder(assembler.vocab_map).block(battle)` (one encoder per assembler)
     after it, the same 1,068 numbers;
  3. ppo_core there needs the same model support (`mu_meta` in `PokeTransformer.__init__` and
     `ObservationUnpacker`, `_wide_subnet`, the two branches of `encode_features`);
  4. test that the app's observation equals the worker's byte for byte (same obs_matchup.py,
     same randbats_sets_gen9.json, same poke-env).
