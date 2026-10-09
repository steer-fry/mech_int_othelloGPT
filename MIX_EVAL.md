# Mixed FLIP/ERASE intervention evaluation

This experiment tests whether the trained single-tile linear probe directions
compose when several tiles are edited simultaneously. It uses the newly trained
layer-5 relative-colour probe and the released single-tile intervention strengths
and hook locations. No probe training or intervention-strength tuning is performed.

## Prefixes and sampling

The source is `data/intervention_benchmark.pkl`. Deduplicate exact move histories,
retaining their first occurrence in the released file, and keep prefixes with at
least eight played moves. This gives **854 distinct prefixes**, with lengths 8–30.
The same prefixes are used for all three values of **k = 2, 4, 8**.

For each prefix, reconstruct its original board with the repository's Othello
simulator. Eligible intervention tiles are occupied squares excluding the four
central starting squares (indices 27, 28, 35, 36). Erasing a central square could
create a legal next move that has no corresponding model vocabulary token.

Use **one sample per prefix**:

1. List eligible square indices in ascending order.
2. Randomly permute the indices and take the first eight distinct squares.
3. Draw eight independent fair coins. Assign FLIP when the coin is 1 and ERASE
   when it is 0.
4. For each k, use the first k positions and their already assigned actions.

Sampling uses NumPy `default_rng(42)` (PCG64), in retained prefix order, with
`rng.permutation(eligible)[:8]` followed by `rng.integers(0, 2, size=8)` for each
prefix. The sampled order and all eight actions are saved in the manifest.

There are **854 x 3 = 2,562 evaluation cases**. The k=2 edits are a subset of the
k=4 edits, which are a subset of the k=8 edits. No additional all-FLIP or all-ERASE
control arms are created. Natural coin outcomes containing only one action type
remain in the evaluation; independent coins do not force a balanced mixture.

## Target boards and model inputs

Construct each target board directly from the original board. For every selected
tile, FLIP reverses its original colour and ERASE sets it to empty. Apply the
edits simultaneously; these are board-state edits rather than legal Othello moves,
so they do not trigger captures. Preserve the original turn state before asking
the simulator for the next legal moves.

The model receives the **original, unmodified move history** for every k.
Only its internal activations are edited. The counterfactual target board defines
the correct next-move set; it is not fed to the model.

## Combined intervention

Load `output/linear_probe_reproduction/resid_5_linear.pth`. Probe classes are
`[EMPTY, YOURS, MINE]`; use the same history-parity mapping for a target flipped
colour as the existing single-tile runner.

Normalize each selected tile's target-class probe vector independently to unit
L2 norm. At the final input token's `blocks.<layer>.hook_attn_out`, add:

- Blocks 0–1: the sum of all ERASE vectors, each scaled by **1.3**.
- Blocks 2–7: the sum of all FLIP vectors, each scaled by **2.3**, plus all ERASE
  vectors, each scaled by **1.3**.

Reuse the layer-5 probe at every patched block. Keep the combined sum
**unnormalized**, preserving the per-tile strengths. Log the resulting vector
norms to help interpret how intervention magnitude changes with k.

## Scoring and aggregation

Let G be the edited board's legal next moves and N = |G|. Take the model's top-N
predictions over its 60 playable move tokens, excluding padding. Let that set be P.

- False positives: |P minus G|.
- False negatives: |G minus P|.
- Paper-compatible error: **FP + FN**. One incorrect top-N move contributes two
  errors because both sets have size N.
- Exact legal-move-set accuracy: the fraction of cases with P = G.
- Normalized error: **FP / N**, for N > 0.

Report these metrics separately for each k. Include results for all nonterminal
cases, the subset whose legal-move set changes, and strata by the numbers of FLIP
and ERASE edits. Preserve cases requiring an implicit pass and flag them; legal
targets follow the simulator's existing implicit-pass convention. Cases with no
legal moves for either player are retained and counted separately, but excluded
from accuracy/error aggregates because top-0 scoring is automatically perfect.

Use a separate NumPy RNG with seed **43** for **1,000 bootstrap resamples** of
whole prefixes. Report 95% percentile confidence intervals and paired differences
between k values. Each prefix is sampled as one unit so the nested cases remain
paired. Report the denominator used for each aggregate and paired comparison.

## Execution and artifacts

```bash
.venv/bin/python -m mech_int.reproduce_mixed_interventions --device auto --seed 42 --batch-size 64
```

The model and directions use float32. Before the full run, verify the combined
interventions against actual TransformerLens 1.6.0 `run_with_hooks` on selected
generated cases, including batched CPU and Apple GPU inference.

Artifacts are saved under `output/linear_probe_reproduction/mixed_interventions/`:
the sampling manifest, configuration and SHA256 provenance, per-case JSON/CSV,
aggregate results and summary CSV, the run log, and implementation/result audits.
The model checkpoint and newly trained probe hashes are recorded with the run.

The implementation is `mech_int/reproduce_mixed_interventions.py`. To generate
the manifest without model inference, add `--generate-only`. The runner reuses a
matching saved manifest on subsequent runs. To rerun the independent hook check:

```bash
HF_HOME=/tmp/othello-hf-cache TRANSFORMERS_CACHE=/tmp/othello-hf-cache/hub .venv/bin/python scripts/verify_mixed_interventions.py --device both
```

## Results

Executed all **2,562 cases** with seed 42, the newly trained layer-5 probe, and
float32 inference on the Apple GPU (PyTorch 2.8.0). No additional control arms
were run. Each k has 854 cases; one k=4 target has no legal moves and is excluded
from metric denominators. The reported intervals are 95% percentile intervals
from 1,000 paired prefix-bootstrap resamples with seed 43.

| k | Scored cases | Mean FP + FN (95% CI) | Exact legal-move sets | Exact-set percentage (95% CI) | Mean FP / N |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 854 | 0.1382 (0.0960–0.1850) | 810 / 854 | 94.85% (93.21–96.37%) | 0.006460 |
| 4 | 853 | 0.5909 (0.4754–0.7222) | 682 / 853 | 79.95% (77.16–82.91%) | 0.028895 |
| 8 | 854 | 3.2974 (3.0140–3.5975) | 284 / 854 | 33.26% (30.09–36.42%) | 0.176475 |

Total FP + FN counts are **118**, **504**, and **2,816** for k=2, 4, and 8.
Mean FP + FN is an error count per case, not a percentage. The final column is
the macro average of each case's normalized error.

The paired mean-error differences are:

| Comparison | Paired prefixes | Increase in mean FP + FN (95% CI) |
| --- | ---: | ---: |
| k=4 minus k=2 | 853 | 0.4525 (0.3423–0.5745) |
| k=8 minus k=4 | 853 | 2.7104 (2.4244–3.0035) |
| k=8 minus k=2 | 854 | 3.1593 (2.8665–3.4592) |

Natural coin compositions and target-board counts are:

| k | Both action types | All FLIP | All ERASE | Forced passes | No legal moves | Changed legal-move sets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 411 | 216 | 227 | 1 | 0 | 818 |
| 4 | 749 | 54 | 51 | 2 | 1 | 852 |
| 8 | 851 | 3 | 0 | 1 | 0 | 854 |

These composition groups come from the original fair-coin draws. They are not
separately constructed control arms. Composition, forced-pass, and changed-set
metrics are included in the saved summary files. Padding appeared in zero
full-vocabulary top-N predictions.

Under the prescribed fixed per-tile strengths, exact prediction declines as k
increases. The mean combined vector norm at blocks 2–7 increases from **2.5846**
to **3.6989** to **5.2704**. Interactions between directions, growing perturbation
magnitude, and changes in target-board difficulty can all contribute to this
performance decline.

Verification against actual TransformerLens 1.6.0 passed on **24 generated cases**
across k=2, 4, and 8, using CPU and MPS. Selected cases cover natural pure/mixed
coin outcomes, both parity conventions, forced passes, and the zero-legal-move
case. The maximum absolute logit difference was **0.0000293**, and every tested
top-N set matched. All 2,562 target boards and nested edit sets passed independent
simulator checks.

An independent result audit regenerated every seed-42 tile permutation and coin
draw, replayed all target boards, and recomputed every case score. All **29
aggregate groups**, **96 bootstrap intervals**, and **three paired comparisons**
matched the saved results. The audit used prefix multiplicity weights and its own
metric calculations rather than the runner's summary functions.

Files:

- [Sampling manifest](output/linear_probe_reproduction/mixed_interventions/cases_manifest.json)
- [Summary CSV](output/linear_probe_reproduction/mixed_interventions/summary.csv)
- [Full aggregate results](output/linear_probe_reproduction/mixed_interventions/results.json)
- [Per-case CSV](output/linear_probe_reproduction/mixed_interventions/per_case_results.csv)
- [Per-case JSON](output/linear_probe_reproduction/mixed_interventions/per_case_results.json)
- [Implementation verification](output/linear_probe_reproduction/mixed_interventions/implementation_verification.json)
- [Independent result audit](output/linear_probe_reproduction/mixed_interventions/result_audit.json)
- [Results, probe, manifest and implementation ZIP](output/linear_probe_reproduction/mixed_intervention_results.zip)
