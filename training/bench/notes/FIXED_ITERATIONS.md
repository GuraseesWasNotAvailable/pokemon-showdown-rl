# Foul Play at a fixed number of iterations

**Recommended: `evaluate.py --foulplay-iterations 88000`** (watch mode: `"eval": {"foulplay": {"iterations": 88000}}`),
with the search time left at 200 ms.

## What the option does

- Every sampled world is searched for exactly N iterations, whatever the clock says. The Foul Play child
  hands poke-engine its own `iterations` argument, the path `bench/fp_bridge_server.py` takes for a
  decision that asks for a count (`bench/foulplay.py`, `_fix_iterations`). Foul Play's checkout is not edited.
- The search time still decides how many worlds Foul Play samples: 4 early in a game (before the
  opponent's active Pokemon has shown a move), 2 after. It no longer decides how long each is searched.
- N must be a multiple of 1,000: poke-engine searches in batches of 1,000 and would round anything else up.
- A world where Foul Play has a single option (a forced switch) ends after the first batch of 1,000, in
  either mode. Those searches are counted apart (`foulplay_iterations_one_option`), about 2% of them.
- Every result row of such an evaluation carries `foulplay_iterations` (N), and
  `foulplay_iterations_min`, `_max`, `_off` (searches that logged another count), `_one_option`.
- A row with `_off` above 0, or with no search in its logs to check, is not trusted.
- The report never compares or pools rows of different N (`play_settings` has N as a fourth value;
  `pool()` raises on a mix). Rows at the search time alone count as their own setting.

## How N was chosen: the median per world at 200 ms

Foul Play's iterations per sampled world at its full 200 ms, forced switches left out:

| Source | Worlds | Median | Middle half |
|---|---:|---:|---|
| Full run's evaluations, 10 of them, 14 to 15 Oct (nice 5, beside training) | 122,766 | 88,000 | 80,000 to 100,000 |
| Baseline run's evaluations, 5 of them, 13 to 14 Oct | 62,428 | 88,000 | 80,000 to 100,000 |
| One run of 16 games today, nice 15 (`~/dev/v2-found-smoke/timed`) | 1,008 | 85,000 | 76,000 to 97,000 |

Each of the 15 evaluations had its own median between 87,000 and 90,000. The box (192 threads) carried a
load of about 21 throughout, so its cores were mostly idle; 88,000 is the idle-box figure on this machine.

## Smoke runs (port 10110, GPU 2, checkpoint `~/runs/baseline/checkpoints/learner_update_022640.pt`)

| Run | Games | Every non-forced search | Forced (1,000) | Wall time | Valid |
|---|---:|---|---:|---:|---|
| `--foulplay-iterations 20000`, 4 processes | 4 | exactly 20,000 (484 log lines) | 12 lines | 6 s | yes |
| `--foulplay-iterations 88000`, 8 processes | 16 | exactly 88,000 | 72 lines | 51 s | yes |
| 200 ms on the clock, 8 processes | 16 | 40,500 to 610,000 | 26 lines | 49 s | yes |

At N = 88,000 an evaluation takes about as long as at 200 ms (17.5 s against 16.7 s a game, the same CPU).

## Two differences from 200 ms to keep in mind

- **Early game.** At 200 ms Foul Play gives its 4 early worlds half the time each (median 42,000). At a flat
  N they get N, about twice that. These are about 9% of its decisions.
- **Simple late positions.** At 200 ms a small position can reach 600,000 iterations or more (the top 5%).
  At N = 88,000 it gets 88,000. MCTS settles quickly in such positions, so this should cost little.

So 88,000 is not exactly the 200 ms opponent. It is the same opponent every time, on a loaded box and an
idle one, which is what the arms are compared on. Win rates at N are not compared with rows measured at
200 ms; the report keeps them apart.

Measured by reading the Foul Play logs (each decision logs `Sampling K battles at M ms each`, then an
`Iterations i: n` line per world), read only, under `~/runs/full/eval/logs/foulplay` and
`~/runs/baseline/eval/logs/foulplay` on 15 Oct.
