# thrawn-nnue

`thrawn-nnue` is a training scaffold for a dual-perspective chess NNUE using a simple A-768 feature set and a `.nnue` export format for [thrawn](https://github.com/feftywacky/thrawn).

## What is included

- Native `.binpack` ingestion via [native_binpack/](/Users/feiyulin/Code/thrawn-nnue/native_binpack)
- A dual-perspective `768 -> 256 -> 32 -> 1` NNUE in PyTorch
- Exact-resume checkpoints with config and RNG state snapshots
- A `.nnue` binary export plus a verification command
- Dataset inspection and a native test fixture generator for smoke tests

## Installation

Install Python 3.11 and then install the package in editable mode:

```bash
python3.11 -m pip install -e .
```

This repo expects `numpy` and `torch`. The native `.binpack` bridge is built automatically on first use.

## Device Selection

Training is configurable for:

- `cuda`
- `mps` for Apple Silicon GPUs
- `cpu`
- `auto`

`auto` prefers:

1. CUDA
2. Apple `mps`
3. CPU

For an Apple M4 Pro MacBook, the usual choice is `device = "mps"` or just `device = "auto"` if your PyTorch build has MPS support.

Mixed precision (`amp = true`) is currently only used on CUDA. On Apple Silicon, training runs in standard precision on `mps`.

## Quick Start

1. Edit [default.toml](/Users/feiyulin/Code/thrawn-nnue/configs/default.toml) and point `train_datasets` at one or more `.binpack` files.
2. Inspect a dataset:

```bash
thrawn-nnue inspect-binpack --path /absolute/path/to/train.binpack
```

`inspect-binpack` now reports score percentiles, absolute-score tail counts, WDL saturation diagnostics for common `wdl_scale` values, and a recommended starting normalization setup.

3. Train:

```bash
thrawn-nnue train --config configs/default.toml
```

4. Validation runs automatically every `validation_every` steps if `validation_datasets` is configured, and the best validation checkpoint is written to `runs/.../checkpoints/best.pt`.

5. Resume later if needed:

```bash
thrawn-nnue resume --checkpoint runs/default/checkpoints/step_00001000.pt
```

6. Export the best checkpoint:

```bash
thrawn-nnue export --checkpoint runs/default/checkpoints/best.pt --out runs/default/model.nnue
```

7. Verify:

```bash
thrawn-nnue verify-export --checkpoint runs/default/checkpoints/step_00010000.pt --nnue runs/default/model.nnue
```

8. Summarize the run and generate plots:

```bash
thrawn-nnue metrics --run-dir runs/default
```

## Documentation

- Tests: [docs/testing.md](/Users/feiyulin/Code/thrawn-nnue/docs/testing.md)
- Trainer workflow: [docs/trainer_workflow.md](/Users/feiyulin/Code/thrawn-nnue/docs/trainer_workflow.md)
- Exported binary format: [docs/nnue_format.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_format.md)
- Engine integration: [docs/engine_integration.md](/Users/feiyulin/Code/thrawn-nnue/docs/engine_integration.md)

## Repo layout

- [src/thrawn_nnue](/Users/feiyulin/Code/thrawn-nnue/src/thrawn_nnue): Python package, model, training loop, checkpoints, export, CLI
- [native_binpack/](/Users/feiyulin/Code/thrawn-nnue/native_binpack): native `.binpack` reader and feature extraction bridge
- [configs/](/Users/feiyulin/Code/thrawn-nnue/configs): training configs
- [docs/](/Users/feiyulin/Code/thrawn-nnue/docs): longer-form documentation
- [tests/](/Users/feiyulin/Code/thrawn-nnue/tests): test suite

## Notes

- This repository is trainer-only.
- The engine-side loader/inference code belongs in your engine repo.
- The exported `.nnue` format is trainer-owned and documented in [docs/nnue_format.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_format.md).
- Training metrics are logged to `metrics.jsonl`, and `thrawn-nnue metrics --run-dir ...` generates summary output plus PNG plots in `plots/`.
- Dataset inspection is intended to drive `wdl_scale`, `score_clip`, and `score_scale` choices before a long training run.
