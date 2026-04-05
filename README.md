# thrawn-nnue

`thrawn-nnue` is a CLI framework for training a dual-perspective chess NNUE using a simple A-768 feature set for [thrawn](https://github.com/feftywacky/thrawn).

## Installation

Install Python 3.11 and then install the package in editable mode:

```bash
python3.11 -m pip install -e .
```

This repo expects `numpy` and `torch`. The native `.binpack` bridge is built automatically on first use.

## Quick Start

1. Edit [default.toml](/Users/feiyulin/Code/thrawn-nnue/configs/default.toml) and point `train_datasets` at one or more `.binpack` files, a dataset directory, or a glob such as `"/data/train/**/*.binpack"`.
2. Inspect a dataset:

```bash
thrawn-nnue inspect-binpack --path /absolute/path/to/train.binpack
```

`inspect-binpack` now reports score percentiles, absolute-score tail counts, WDL saturation diagnostics for common `wdl_scale` values, and a recommended starting normalization setup.

3. Train:

```bash
thrawn-nnue train --config configs/default.toml
```

Training now defaults to a live single-line progress bar. It shows step progress, epoch, ETA, latest train loss, latest validation loss, and checkpoint notices. If you prefer plain summaries instead, set `console_mode = "text"` in [default.toml](/Users/feiyulin/Code/thrawn-nnue/configs/default.toml) or run:

```bash
thrawn-nnue train --config configs/default.toml --console-mode text
```

4. Validation runs automatically when `validation_datasets` is configured. Use `validation_every` for step-based validation or `validation_every = 0` for end-of-epoch validation. The best validation checkpoint is written to `runs/.../checkpoints/best.pt`.

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
- The engine-side loader and inference code belongs in your engine repo.
- Training metrics are logged to `metrics.jsonl`, and `thrawn-nnue metrics --run-dir ...` generates summary output plus PNG plots in `plots/`.
- Multiple `train_datasets` are opened as one combined cyclic training stream, not processed one file at a time.
- Dataset lists can contain individual files, directories, or glob patterns. Directories are expanded recursively to `.binpack` files.
- `steps_per_epoch = 0` means one full training-corpus pass per epoch.
- `validation_every = 0` means validate at the end of each epoch.
- `validation_steps = 0` means one full validation-corpus pass.
- Dataset inspection is intended to drive `wdl_scale`, `score_clip`, and `score_scale` choices before a long training run.
