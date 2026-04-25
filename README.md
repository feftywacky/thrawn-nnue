# thrawn-nnue

`thrawn-nnue` is a trainer/export pipeline for a fixed HalfKP chess NNUE aimed at direct centipawn outputs for engine pruning.

## Architecture

The repo targets classic HalfKP with a larger v2 production shape:

- feature set: `halfkp`
- v2 feature transform: `40960 -> 1024`
- v2 dense path: `2048 -> 256 -> 64 -> 1`
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

1. Edit [v2.toml](/Users/feiyulin/Code/thrawn-nnue/configs/v2.toml) and point `train_datasets` / `validation_datasets` at your Jan-May / June `.binpack` files.

2. Inspect a dataset:

```bash
thrawn-nnue inspect-binpack --path /absolute/path/to/train.binpack
```

`inspect-binpack` reports score percentiles, WDL saturation diagnostics, and a starting recommendation for `score_clip` / `wdl_scale`.

3. Train v2:

```bash
thrawn-nnue train --config configs/v2.toml
```

4. Resume if needed:

```bash
thrawn-nnue resume --checkpoint runs/v2/checkpoints/step_00001000.pt
```

5. Export the best checkpoint:

```bash
thrawn-nnue export --checkpoint runs/v2/checkpoints/best.pt --out runs/v2/model.nnue
```

6. Verify checkpoint/export parity and sanity scores:

```bash
thrawn-nnue verify-export --checkpoint runs/v2/checkpoints/best.pt --nnue runs/v2/model.nnue
```

7. Summarize the run and generate plots:

```bash
thrawn-nnue metrics --run-dir runs/v2
```

## Training Notes

- `feature_set = "halfkp"` is the only supported feature set.
- Binpack scores are converted from Stockfish internal units to true centipawns before inspection and training: `score_cp = raw_score * 100 / 208`.
- `score_clip` clips teacher centipawns directly; there is no `score_scale`.
- The training loss is expectation-space MSE on a blended target. For a cp value:
  `win = sigmoid((cp - offset) / scale)`, `loss = sigmoid((-cp - offset) / scale)`, `draw = 1 - win - loss`, `expectation = win + 0.5 * draw`.
- `wdl_lambda` weights the teacher expectation, so `wdl_lambda = 0.9` means 90% teacher expectation and 10% game result.
- `sanity_anchor_weight` adds a small zero-cp anchor for neutral start/bare-king positions. It is enabled in v2.
- LR decay is epoch-based cosine annealing: `epoch_positions` defines the fixed position budget per epoch, and scheduler steps occur at completed epoch boundaries.
- `verify-export` includes a fixed material sanity ladder so you can quickly check `pawn < knight < rook < queen`.
- The engine-side contract is documented in [nnue_spec.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_spec.md).

## Scope

- This repository is trainer/export/spec only.
- Engine loader and search integration stay in your engine repo.
