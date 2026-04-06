# Trainer Workflow

This is the intended end-to-end workflow for operating `thrawn-nnue`.

## 1. Prepare `.binpack` data

This trainer expects Stockfish-style `.binpack` files. Each entry provides:

- a packed board position
- the side to move
- a teacher move
- a teacher score
- the game ply
- the final game result

The trainer uses:

- `score` as the eval target
- `result` as the WDL target
- the board position to derive white-perspective and black-perspective A-768 features

## 2. Configure a run

Edit [default.toml](/Users/feiyulin/Code/thrawn-nnue/configs/default.toml) and set at least:

- `train_datasets`
- `validation_datasets` if you have validation data
- `total_train_positions`
- `superbatch_positions`
- `validation_interval_positions`
- `validation_positions`
- `output_dir`
- `batch_size`
- `checkpoint_every`
- `console_mode`

The score-target controls are now:

- `score_clip`
- `score_scale`
- `wdl_scale`

Example:

```toml
train_datasets = ["/absolute/path/to/train/**/*.binpack"]
validation_datasets = ["/absolute/path/to/valid"]
output_dir = "runs/my_first_run"
device = "auto"
total_train_positions = 20000000000
superbatch_positions = 4000000000
validation_interval_positions = 0
validation_positions = 1048576
console_mode = "progress"
score_clip = 0.0
score_scale = 1.0
wdl_scale = 410.0
```

Dataset entries may be:

- explicit `.binpack` file paths
- directories, which are expanded recursively to `.binpack` files
- glob patterns such as `"/data/train/**/*.binpack"`

Budget controls:

- `total_train_positions` is the primary stop condition
- `superbatch_positions` is a reporting boundary, not a full-pass epoch
- `validation_interval_positions = 0` runs validation at each superbatch boundary
- `validation_positions = 0` runs one full validation-corpus pass
- train and validation shard lists must not overlap

Network sizing:

- `feature_set = "a768"` and `num_features = 768` describe the sparse input encoding
- the legacy alias `feature_set = "a768_dual"` is still accepted
- `ft_size` is the accumulator width for one perspective
- the first dense layer therefore sees `2 * ft_size` inputs because the trainer feeds `[stm_acc, nstm_acc]`
- `output_buckets` controls how many phase buckets share the same hidden layer and split only at the final output layer

## Device configuration

The trainer accepts four device modes:

- `device = "auto"`
- `device = "cuda"`
- `device = "mps"`
- `device = "cpu"`

`auto` prefers CUDA first, then Apple `mps`, then CPU.

For Apple Silicon training, use:

```toml
device = "mps"
```

or leave:

```toml
device = "auto"
```

if your installed PyTorch build supports MPS.

Notes:

- The trainer now uses PyTorch's generic AMP API, so `amp = true` enables autocast/scaling on whichever backend your local PyTorch build supports.
- If a backend does not expose AMP support in that build, the trainer falls back to fp32 rather than pretending AMP is active.
- If you request `cuda` or `mps` explicitly and that backend is unavailable, the trainer raises an error instead of silently falling back.

## 3. Inspect a dataset

Before training, validate that the dataset is readable and sane:

```bash
thrawn-nnue inspect-binpack --path /absolute/path/to/train_01.binpack
```

This prints:

- number of entries
- white-to-move versus black-to-move counts
- win/draw/loss distribution
- min/max score
- mean signed score
- average absolute score
- average piece count
- score percentiles
- absolute-score percentiles
- score-tail counts and fractions
- WDL saturation diagnostics for common `wdl_scale` values
- a recommended starting `wdl_scale`, `score_clip`, and `score_scale`

`inspect-binpack` is a manual preflight step. It is not automatically run by `train`.

Use the recommendation block to choose score normalization for your run:

- `score_clip` limits extreme tails before target construction
- `score_scale` rescales unusually large score magnitudes
- `wdl_scale` controls how sharply normalized scores map into WDL space

## 4. Train

Start training with:

```bash
thrawn-nnue train --config configs/default.toml
```

By default, training uses a live progress bar. It tracks `positions_seen` out of `total_train_positions`, shows the current optimizer step and superbatch index, latest train loss, latest validation loss, and emits short notices when validation runs or checkpoints are saved.

If you want plain text summaries instead:

```bash
thrawn-nnue train --config configs/default.toml --console-mode text
```

Operational guidance:

- Think in positions, not epochs. `positions_seen / total_train_positions` is the main measure of run progress.
- Keep `train_datasets` and `validation_datasets` disjoint at the shard level. The trainer enforces path-level separation only.
- Use smaller `validation_positions` during tuning to validate more frequently, then switch to `0` for full held-out passes on serious runs.
- Choose `superbatch_positions` large enough that validation cadence is meaningful rather than noisy.
- Resume from a saved step checkpoint to continue a run exactly; export `best.pt` when validation has clearly peaked.

During training the trainer:

1. Streams `.binpack` entries through the native loader.
   All configured `train_datasets` are opened together as one combined cyclic stream.
   The native reader samples chunk reads across the shard list and shuffles buffered entries, so training is interleaved across the whole date range instead of consuming one file front-to-back.
2. Extracts white and black A-768 active feature lists.
3. Builds two accumulators with a shared feature-transformer table.
4. Orders them as `[stm_acc, nstm_acc]`.
5. Runs the configured `2 * ft_size -> hidden_size -> output_buckets` network and selects the phase bucket for each position at the final layer.
6. Applies teacher-score preprocessing:
   clip, then scale
7. Converts normalized score targets into WDL space.
8. Blends teacher loss and result loss using `eval_lambda`.
9. Saves periodic training checkpoints.
10. Runs validation whenever `positions_seen` crosses the next configured position threshold.
11. Updates `checkpoints/best.pt` when validation loss improves.
12. Writes JSONL training and validation metrics.

Outputs go under the configured `output_dir`, including:

- `checkpoints/`
- `metrics.jsonl`
- `plots/` after running the metrics report command

Validation uses only `validation_datasets`. Those shards should be held out from `train_datasets`.
The trainer enforces disjoint shard paths, but it cannot detect same-game leakage across separately prepared shards.

Each validation pass:

- opens the held-out `.binpack` files in non-cyclic mode
- evaluates positions until `validation_positions` is reached or the dataset is exhausted
- if `validation_positions = 0`, auto-sizes to a full held-out pass
- averages blended loss, teacher loss, and result loss
- reports `wdl_accuracy` against the game result
- reports `teacher_result_disagreement_rate`
- logs a validation record to `metrics.jsonl`
- updates `checkpoints/best.pt` if blended validation loss is the best seen so far

## 5. Inspect metrics and plots

After or during a run, summarize the run and generate PNG plots:

```bash
thrawn-nnue metrics --run-dir runs/my_first_run
```

This reads `metrics.jsonl`, prints a compact summary, and writes plots under `runs/my_first_run/plots/`.

The first version generates:

- `train_loss.png`
- `validation_loss.png` if validation metrics exist
- `lr.png`
- `loss_overview.png` if both train and validation metrics exist

## 6. Resume

Resume from a saved checkpoint with:

```bash
thrawn-nnue resume --checkpoint runs/my_first_run/checkpoints/step_00001000.pt
```

You can also override the console mode on resume:

```bash
thrawn-nnue resume --checkpoint runs/my_first_run/checkpoints/step_00001000.pt --console-mode text
```

Checkpoints contain:

- model weights
- optimizer state
- scheduler state
- AMP scaler state
- config snapshot
- RNG state
- optimizer step counter
- `positions_seen`
- `superbatch_index`
- current best validation loss and best validation position

That makes resume suitable for continuing a run exactly, not just reloading weights.

## 7. Export

Export a trained checkpoint to `.nnue`:

```bash
thrawn-nnue export --checkpoint runs/my_first_run/checkpoints/best.pt --out runs/my_first_run/model.nnue
```

The binary format is documented in [nnue_format.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_format.md).

## 8. Verify the export

Compare exported inference to checkpoint inference:

```bash
thrawn-nnue verify-export --checkpoint runs/my_first_run/checkpoints/step_00010000.pt --nnue runs/my_first_run/model.nnue
```

This reports parity metrics on fixture positions.
