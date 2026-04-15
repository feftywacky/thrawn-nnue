# thrawn-nnue

`thrawn-nnue` is a trainer/export pipeline for a fixed HalfKP chess NNUE aimed at direct centipawn outputs for engine pruning.

## Architecture

The repo now targets one production shape only:

- feature set: `halfkp`
- feature transform: `40960 -> 256`
- dense path: `512 -> 32 -> 32 -> 1`
- output perspective: side to move
- raw exported output: direct centipawns

Training uses classic HalfKP with training-time `P` factorization. Exported `.nnue` files contain only coalesced real HalfKP weights.

## Installation

Install Python 3.11 and the repo in editable mode:

```bash
python3.11 -m pip install -e .
```

The native `.binpack` bridge builds automatically on first use.

## Quick Start

1. Edit [default.toml](/Users/feiyulin/Code/thrawn-nnue/configs/default.toml) and point `train_datasets` / `validation_datasets` at your Jan-May / June `.binpack` files.

2. Inspect a dataset:

```bash
thrawn-nnue inspect-binpack --path /absolute/path/to/train.binpack
```

`inspect-binpack` reports score percentiles, WDL saturation diagnostics, and a starting recommendation for `score_clip` / `wdl_scale`.

3. Train:

```bash
thrawn-nnue train --config configs/default.toml
```

4. Resume if needed:

```bash
thrawn-nnue resume --checkpoint runs/halfkp_baseline/checkpoints/step_00001000.pt
```

5. Export the best checkpoint:

```bash
thrawn-nnue export --checkpoint runs/halfkp_baseline/checkpoints/best.pt --out runs/halfkp_baseline/model.nnue
```

6. Verify checkpoint/export parity and sanity scores:

```bash
thrawn-nnue verify-export --checkpoint runs/halfkp_baseline/checkpoints/best.pt --nnue runs/halfkp_baseline/model.nnue
```

7. Summarize the run and generate plots:

```bash
thrawn-nnue metrics --run-dir runs/halfkp_baseline
```

## Training Notes

- `feature_set = "halfkp"` is the only supported feature set.
- `score_clip` clips teacher centipawns directly; there is no `score_scale`.
- The training loss is `Huber(cp) + wdl_lambda * auxiliary_wdl`.
- `verify-export` includes a fixed material sanity ladder so you can quickly check `pawn < knight < rook < queen`.
- The engine-side contract is documented in [nnue_spec.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_spec.md).

## Scope

- This repository is trainer/export/spec only.
- Engine loader and search integration stay in your engine repo.
