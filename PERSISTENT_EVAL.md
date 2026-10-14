# Persistent intervention evaluation

**Status: implemented in `mech_int/reproduce_persistent_interventions.py`.**

Test whether next-move predictions continue to match an edited board after a
one-time intervention: **intervene once, preserve the affected KV cache, then
continue without further steering**.

## Procedure

Use the released cases in `data/intervention_benchmark.pkl` and the trained
layer-5 probe. Reuse the settings in `mech_int/reproduce_interventions.py`:
FLIP strength **2.3** at blocks **2–7**; ERASE strength **1.3** at blocks **0–7**.

1. Replay the original history. Copy the board and apply the selected FLIP or
   ERASE edit, preserving the turn state. Exclude central-square ERASE edits,
   since those squares have no move tokens.
2. Process the original history twice, creating independent KV caches: one with
   steering at the final prefix token, and one without steering as a baseline.
   Remove the steering hooks after this prefix pass.
3. Score both models' immediate next-move predictions against the edited board.
   This is **h = 0**.
4. Uniformly sample a legal move from the edited board using seed **42**. Apply it
   to the simulator, including captures, and feed the same token to both models
   using their respective caches. Apply no further steering.
5. Score the resulting predictions against the **updated board's** legal moves.
   This is **h = 1**. Repeat through **h = 10**, stopping earlier if the game ends
   or the total input length reaches **59** tokens.

Allow the edited square to change naturally as moves are played. Handle implicit
passes using the existing simulator; do not add a pass token.

Using identical continuation tokens keeps token choice from confounding the cache
comparison. Some moves may be illegal under the original board, so the baseline
then measures behavior on that same counterfactual history.

## Scoring

At each horizon, let G be the updated board's legal moves and N = |G|. Take the
model's top-N playable tokens, excluding padding, and convert them to set P.

- **Exact-set accuracy:** whether P = G.
- **Error:** |P minus G| + |G minus P|, matching the existing intervention metric.

Report both metrics for patched and baseline runs, separately for FLIP and ERASE,
at each horizon. Include h = 0 failures and give the number of scored cases and
stopping reasons. Exclude terminal boards with N = 0 from accuracy/error averages.
Later horizons may contain fewer cases, so show their denominators.

Persistence means continued agreement with the evolving edited board after h = 0.
Improvement over the baseline provides stronger evidence; matching predictions
where both boards have the same legal moves does not distinguish their states.

## Ordinary-game control

`mech_int/reproduce_clean_continuations.py` adds an ordinary-game capability
reference. Replay each of the same benchmark prefixes, apply **no board edit and
no steering**, then sample legal continuation moves from the evolving original
board. Score the unsteered model against that board at h=0 through h=10, using
the same real KV cache, top-N playable-token metric, input limit, implicit-pass
convention, and terminal exclusion as the intervention runs.

One control trajectory is retained per benchmark record, including repeated
prefixes, to preserve the original record weighting. Both edit panels use this
same control curve. An independent seed-42 NumPy generator visits case IDs in
ascending order and samples from sorted legal moves before model batching.
At h=0 all three arms have identical input histories; later the ordinary-game
control follows its own legal original-board continuation. Its comparison with
the edited-board arms provides a normal-performance reference. The patched
versus unsteered edited-board comparison remains the paired test of steering.

The three reporting labels distinguish model treatment from the scoring target:

| Arm | Steering | Continuation moves | Scoring target |
| --- | --- | --- | --- |
| Patched, edited board | Once during prefix | Edited board | Edited board |
| Unsteered, edited board | None | Same edited-board moves | Edited board |
| Unsteered, original board | None | Original board | Original board |

The ordinary-game control saves its own configuration, per-case and per-step
predictions, full playable-token rankings, summaries, numerical checks, and
independent audit under `persistent_interventions/ordinary_game_control/`.
Full rankings allow all h=0 predictions to be re-scored against the edited
targets and checked against the existing unsteered baseline.

The ordinary-game control reports three separate notions of accuracy:

- **Top-1 move legality:** the full-vocabulary argmax is legal on the unedited
  board. Padding is an illegal prediction.
- **Exact sampled-next-token accuracy:** that same argmax equals the actual
  uniformly sampled next legal move. The reference chance rate at each step is
  `1 / number_of_legal_moves`; legal alternatives can be equally valid.
- **Exact legal-set accuracy:** the top-N playable tokens recover the entire
  original board's N legal moves, matching the intervention metric.

At each step before the final horizon, the sampled-next-token target is the
next continuation move already used to evolve the ordinary game. At the final
reached nonterminal step, a separate lookahead draw supplies a target without
adding another input token or changing existing trajectories. Its NumPy seed is
`SeedSequence([seed, case_id, 0x434C4541])`. Full 61-token logits are retained for
independent checking of argmax and next-token negative log likelihood.

## Essential checks

Use a real autoregressive KV cache, not the diagnostic `ActivationCache`.
The compact model's `forward_cached` returns new-token logits and a
`FrozenKVCache` containing one key/value pair per transformer block. It offsets
position embeddings and causal masks by the retained prefix length. Independent
baseline and patched caches are created during the prefix pass; subsequent
calls receive only the shared continuation tokens.

Before evaluation, verify that:

- Clean cached predictions match ordinary full-history recomputation.
- Immediate patched predictions match the existing intervention runner.
- Patched cached predictions match a full-history reference edited only at the
  **original prefix position**, with no edit on continuation tokens.

Save per-step predictions, target legal sets, scores, and stopping reasons, plus
one summary row per edit and horizon, under
`output/linear_probe_reproduction/persistent_interventions/`.

## Running and verifying

```bash
.venv/bin/python scripts/verify_persistent_interventions.py --device cpu
.venv/bin/python -u -m mech_int.reproduce_persistent_interventions --device auto --batch-size 64
.venv/bin/python scripts/audit_persistent_interventions.py
.venv/bin/python -u -m mech_int.reproduce_clean_continuations --device auto --batch-size 64
.venv/bin/python scripts/audit_clean_continuations.py
.venv/bin/python scripts/compare_persistent_controls.py
```

The numerical verifier compares sampled clean and patched cached predictions
with full histories from TransformerLens 1.6.0 at every horizon. Its reference
hook stays at the original prefix position. It also checks batched chunked
decoding, cache filtering, repeated ERASE tokens, the context limit, and simulator
pass/terminal behavior. Use `--device both` to check CPU and Apple GPU.

The runner saves `config.json`, `results.json`, `summary.csv`,
`per_case_results.json`, and `per_step_results.csv`. An independent audit replays
every edited continuation and recomputes all stored scores and denominators.
If compatible prior immediate-intervention artifacts exist, all h=0 prediction
sets are also checked against that runner. `--limit-cases` produces a clearly
labeled diagnostic run; omit it for the full benchmark.

Sampling uses one NumPy generator seeded with 42, visiting case IDs in ascending
order, FLIP then ERASE, and sorted legal-move candidates. Trajectories are
precomputed before model batching, so changing batch size does not change them.
Each step records the evolving board, turn, legal moves, incoming token and
implicit-pass flag. A terminal step is saved with null scores and excluded from
averages. Stopping checks prioritize terminal boards, then the input-length
limit, then h=10.

An optional standalone plot can be generated from saved results in any Python
environment with Matplotlib:

```bash
python scripts/plot_persistent_interventions.py output/linear_probe_reproduction/persistent_interventions
```

This produces `persistence.png` and `persistence.svg` without rerunning inference.
When ordinary-game control results are present, the plot includes their accuracy
and mean-error curves alongside the two edited-board arms.
`compare_persistent_controls.py` writes `comparison_with_ordinary_control.json`
and CSV, with explicit arm labels, scoring targets, per-horizon denominators,
and hashes of both input reports.
