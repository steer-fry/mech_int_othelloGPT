# Mechanistic Interpretations of OthelloGPT

This repository provides the experiments in Emergent Linear Representations in World Models of Self-Supervised Sequence Models.


## Abstract

> How do sequence models represent their decision-making process? Prior work suggests that Othello-playing neural network learned nonlinear models of the board state (Li et al., 2023a). In this work, we provide evidence of a closely related linear representation of the board. In particular, we show that probing for “my colour” vs. “opponent’s colour” may be a simple yet powerful way to interpret the model’s internal state. This precise understanding of the internal representations allows us to control the model’s behaviour with simple vector arithmetic. Linear representations enable significant interpretability progress, which we demonstrate with further exploration of how the world model is computed.

## Installation

```
conda env create -f environment.yml
conda activate mech_int_othello
```

Download relevent data:

* [OthelloGPT](https://drive.google.com/drive/folders/1bpnwJnccpr9W-N_hzXSm59hT7Lij4HxZ): We only analyze the synthetic model. Save this checkpoint to the root directory of this repo.
* [sequence data](https://github.com/likenneth/othello_world/blob/master/mechanistic_interpretability/tl_initial_exploration.py#L20-L58): Refer to the code here to generate training+validation data. Keep this data in `./data`.


## Relevant Files

This repo uses bits and pieces from [Othello World](https://github.com/likenneth/othello_world).
All of the experiments in our paper can be found in the `mech_int` directory.

* `board_probe.py` contains training + evaluation scripts for our linear probes. See `train()` and `evaluate()`.
* `train_flipped.py` contains training + evaluation scripts for our `Flipped` probes. See `train()` and `evaluate()`.
* `intervene.py`, `intervene_blank.py`, `intervene_flipped.py` contain our intervention experiments. 
* `tl_othello_utils.py` contains various utility functions.
* `./figures/` contains various notebooks that were used to create our figures.
* `./probes/` contains all of our probes.

## Reproducing the Section 3 linear probe

`mech_int/reproduce_linear_probe.py` reproduces only the relative-colour linear
probe row in Table 1 of [the paper](https://aclanthology.org/2023.blackboxnlp-1.2.pdf).
It trains eight separate bias-free probes on frozen post-block residual streams,
saves the lowest-validation-loss checkpoint for each layer, and reports test
accuracy across all 59 input positions and all 64 squares.

The protocol uses AdamW with learning rate 0.01, weight decay 0.01, betas
(0.9, 0.99), 512 validation games, 1,000 held-out test games, and patience 10.
Batch size 128 and probe initialization follow the released code. Its
`valid_every=200` condition checks the **sample offset**, so validation runs after
updates 1, 26, 51, etc. This reproduction preserves that behavior. Training
continues until patience expires for every layer, rather than stopping after one
epoch.

The paper does not publish exact game identities, a split seed, or a fixed
training-pool size. This reproduction uses seed 42 and a pool of one million
uniformly random legal games, retains only complete 60-move games, and makes all
three game splits distinct. Labels preserve the released parity-based convention,
including after implicit passes. The reported approximately 100,000 samples is
an observed convergence count, not a prescribed training-split size.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-probe-reproduction.txt
curl -L --fail 'https://huggingface.co/NeelNanda/Othello-GPT-Transformer-Lens/resolve/905ca1a68b9f7dff77adc56af1962e5f6fcac274/synthetic_model.pth' -o synthetic_model.pth
clang++ -O3 -std=c++17 scripts/generate_probe_games.cpp -o /tmp/generate_probe_games
/tmp/generate_probe_games --output-directory data/probe_reproduction --train-games 1000000 --val-games 512 --test-games 1000 --seed 42
.venv/bin/python scripts/verify_probe_games.py data/probe_reproduction
.venv/bin/python -m mech_int.reproduce_linear_probe --mode verify
.venv/bin/python -m mech_int.reproduce_linear_probe --device auto
```

The generator refuses to overwrite an existing dataset. The trainer verifies the
authors' checkpoint SHA256. Its compact frozen transformer is numerically checked
against TransformerLens 1.6.0; sharing its forward pass across layers does not
share probe weights or optimizer state. PyTorch 2.8.0 supports the local Apple
GPU; the original PyTorch 2.0.1 build crashed on this machine. Computation uses
float32 throughout.

Artifacts are saved under `output/linear_probe_reproduction/`: eight
`resid_<layer>_linear.pth` tensors compatible with the existing probe code,
`accuracy.csv`, detailed `test_results.json`, configuration, validation history,
and verification reports. Checkpoints can be loaded with
`torch.load(path, map_location="cpu", weights_only=True)` and have shape
`[1, 512, 8, 8, 3]`, with class order `[EMPTY, YOURS, MINE]`. Add `--resume` to
continue from the most recently saved validation state using the same settings;
use `--mode evaluate` to evaluate the selected checkpoints again.

## Running the linear intervention benchmarks

Run both Table 2 interventions with the newly trained layer-5 probe:

```bash
.venv/bin/python -m mech_int.reproduce_interventions --device auto --batch-size 64
```

The runner defaults to `output/linear_probe_reproduction/resid_5_linear.pth`.
FLIP adds its normalized target-colour direction with strength 2.3 at blocks 2–7;
ERASE adds its normalized EMPTY direction with strength 1.3 at blocks 0–7.
Each addition affects only the final input token's attention output. Both
experiments use every record in `data/intervention_benchmark.pkl`: the released
file contains 1,001 cases, although the paper describes 1,000. Results for the
first 1,000 are also saved as a separately labeled diagnostic aggregate.

The error is the number of false positives plus false negatives when comparing
the model's top-N next moves with the edited board's N legal moves, averaged
across cases. A single incorrect move in the top-N set contributes two errors.
The null baseline uses the unmodified model's predictions against the same
edited board.

This run used float32 on the Apple GPU, without tuning the released settings:

| Edit | New probe mean errors | Paper mean errors | Exact legal-move sets |
| --- | ---: | ---: | ---: |
| FLIP | 0.087912 (88 / 1,001) | 0.10 | 962 / 1,001 (96.10%) |
| ERASE | 0.021978 (22 / 1,001) | 0.02 | 992 / 1,001 (99.10%) |

Null baseline means were 2.723277 for FLIP and 2.733267 for ERASE, matching the
paper's 2.723 and 2.73 at its displayed precision. Artifacts in
`output/linear_probe_reproduction/interventions/` include `results.json`,
`table2_comparison.csv`, detailed `per_case_results.json` and CSV, the run log,
and configuration with hashes of the probe, model, benchmark, and runner.

`scripts/verify_interventions.py` checks the original and patched forward passes
against actual TransformerLens 1.6.0. Twenty benchmark cases spanning history
lengths 5–30 and all original-colour/parity groups passed on CPU and MPS; all
patched top-N predictions matched. To rerun that check:

```bash
HF_HOME=/tmp/othello-hf-cache TRANSFORMERS_CACHE=/tmp/othello-hf-cache/hub .venv/bin/python scripts/verify_interventions.py --device both
```

The nested mixed FLIP/ERASE evaluation is specified in [MIX_EVAL.md](MIX_EVAL.md).
It uses one sampled set of eight tiles per eligible prefix, with nested k=2, 4,
and 8 edits and no additional all-FLIP or all-ERASE control arms.


