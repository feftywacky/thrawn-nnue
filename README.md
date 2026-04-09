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

You can also prepare a filtered shard before long runs:

```bash
thrawn-nnue prepare-binpack --path /absolute/path/to/train.binpack --out /absolute/path/to/prepared.binpack
```

3. Train:

```bash
thrawn-nnue train --config configs/default.toml
```

Training now defaults to a live single-line progress bar. It shows positions seen out of the configured budget, optimizer step, superbatch index, latest train loss, latest validation loss, and checkpoint notices. If you prefer plain summaries instead, set `console_mode = "text"` in [default.toml](/Users/feiyulin/Code/thrawn-nnue/configs/default.toml) or run:

```bash
thrawn-nnue train --config configs/default.toml --console-mode text
```

4. Validation runs automatically when `validation_datasets` is configured. Use `validation_interval_positions` for explicit position-based validation, or set `validation_interval_positions = 0` to validate at each `superbatch_positions` boundary. The best validation checkpoint is written to `runs/.../checkpoints/best.pt`.

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

8. Calibrate engine centipawn scaling from validation data:

```bash
thrawn-nnue calibrate-scale --nnue runs/default/model.nnue --validation-path /absolute/path/to/valid.binpack
```

This prints JSON containing `cp_per_raw` and `raw_per_cp`, plus fit quality metrics and a small hardcoded-position sanity block.
For version-4 dual-head exports it also reports per-head fit metrics and a `preferred_head`.

9. Summarize the run and generate plots:

```bash
thrawn-nnue metrics --run-dir runs/default
```

## Operating The Trainer

- Treat `total_train_positions` as the real run budget. The trainer is no longer organized around epochs.
- Multiple `train_datasets` are interleaved through the native chunk reader, so a Jan-May list is sampled across all shards instead of being consumed one file at a time.
- Use `superbatch_positions` as a reporting boundary and as the default validation boundary when `validation_interval_positions = 0`.
- Use `validation_positions = 0` for a full held-out pass, or set it to a smaller fixed position budget for faster iteration.
- Run `inspect-binpack` on a representative shard before long training runs to choose `wdl_scale`, `score_clip`, and `score_scale`.
- Watch validation metrics, especially blended loss, `wdl_accuracy`, and `teacher_result_disagreement_rate`, rather than train loss alone.
- `feature_set = "a768"` is the preferred config spelling; the older `a768_dual` alias is still accepted.
- Set `output_buckets = 8` for production-style runs so the final layer can separate opening and endgame behavior while keeping the same dual-accumulator update path.
- Use [test80_a768_v9.toml](/Users/feiyulin/Code/thrawn-nnue/configs/test80_a768_v9.toml) as the scratch-training reference for the dual-head v9 run.
- Export `checkpoints/best.pt` by default, then verify parity and engine strength in the engine repo.
- The unified trainer/export/engine contract lives in [nnue_spec.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_spec.md).

## Notes

- This repository is trainer-only.
- The engine-side loader and inference code belongs in your engine repo.
- Training metrics are logged to `metrics.jsonl`, and `thrawn-nnue metrics --run-dir ...` generates summary output plus PNG plots in `plots/`.
- Multiple `train_datasets` are opened as one combined cyclic training stream, not processed one file at a time.
- Dataset lists can contain individual files, directories, or glob patterns. Directories are expanded recursively to `.binpack` files.
- `total_train_positions` is the primary run budget.
- `superbatch_positions` is a reporting and default-validation boundary, not an epoch.
- `validation_interval_positions = 0` means validate at each superbatch boundary.
- `validation_positions = 0` means one full validation-corpus pass.
- `validation_positions > 0` is now respected exactly, including the last partial batch.
- Dataset inspection is intended to drive `wdl_scale`, `score_clip`, and `score_scale` choices before a long training run.
- Train and validation shard lists must not overlap; same-game dedup remains a data-prep responsibility outside the trainer.
