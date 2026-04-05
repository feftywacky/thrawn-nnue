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
- `validation_every`
- `validation_steps`
- `output_dir`
- `batch_size`
- `steps_per_epoch`
- `max_epochs`
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
steps_per_epoch = 0
validation_every = 0
validation_steps = 256
console_mode = "progress"
score_clip = 0.0
score_scale = 1.0
wdl_scale = 410.0
```

Dataset entries may be:

- explicit `.binpack` file paths
- directories, which are expanded recursively to `.binpack` files
- glob patterns such as `"/data/train/**/*.binpack"`

Sizing controls support auto mode:

- `steps_per_epoch = 0` runs one full training-corpus pass per epoch
- `validation_every = 0` runs validation at the end of each epoch
- `validation_steps = 0` runs one full validation-corpus pass

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

- CUDA training uses AMP when `amp = true`.
- Apple `mps` training currently runs without AMP in this scaffold.
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

By default, training uses a live progress bar. It tracks `global_step` out of `max_epochs * steps_per_epoch`, shows the current epoch/step, latest train loss, latest validation loss, and emits short notices when validation runs or checkpoints are saved.

If you want plain text summaries instead:

```bash
thrawn-nnue train --config configs/default.toml --console-mode text
```

During training the trainer:

1. Streams `.binpack` entries through the native loader.
   All configured `train_datasets` are opened together as one combined cyclic stream.
2. Extracts white and black A-768 active feature lists.
3. Builds two accumulators with a shared feature-transformer table.
4. Orders them as `[stm_acc, nstm_acc]`.
5. Runs the `512 -> 32 -> 1` network.
6. Applies teacher-score preprocessing:
   clip, then scale
7. Converts normalized score targets into WDL space.
8. Blends eval loss and result loss using `result_lambda`.
9. Saves periodic training checkpoints.
10. Runs validation every `validation_every` steps if `validation_datasets` is configured.
11. Updates `checkpoints/best.pt` when validation loss improves.
12. Writes JSONL training and validation metrics.

Outputs go under the configured `output_dir`, including:

- `checkpoints/`
- `metrics.jsonl`
- `plots/` after running the metrics report command

Validation uses only `validation_datasets`. Those shards should be held out from `train_datasets`.

Each validation pass:

- opens the held-out `.binpack` files in non-cyclic mode
- evaluates `validation_steps` batches or stops earlier at dataset exhaustion
- if `validation_steps = 0`, auto-sizes to a full held-out pass
- averages blended loss, eval loss, and result loss
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
- epoch and step counters
- current best validation loss and best validation step

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
