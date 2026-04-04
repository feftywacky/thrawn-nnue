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
- `output_dir`
- `batch_size`
- `steps_per_epoch`
- `max_epochs`
- `checkpoint_every`

Example:

```toml
train_datasets = ["/absolute/path/to/train_01.binpack", "/absolute/path/to/train_02.binpack"]
validation_datasets = ["/absolute/path/to/valid.binpack"]
output_dir = "runs/my_first_run"
device = "auto"
```

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
- average absolute score
- average piece count

## 4. Train

Start training with:

```bash
thrawn-nnue train --config configs/default.toml
```

During training the trainer:

1. Streams `.binpack` entries through the native loader.
2. Extracts white and black A-768 active feature lists.
3. Builds two accumulators with a shared feature-transformer table.
4. Orders them as `[stm_acc, nstm_acc]`.
5. Runs the `512 -> 32 -> 1` network.
6. Converts centipawn predictions into WDL space.
7. Blends eval loss and result loss using `result_lambda`.
8. Writes checkpoints and JSONL metrics.

Outputs go under the configured `output_dir`, including:

- `checkpoints/`
- `metrics.jsonl`

## 5. Resume

Resume from a saved checkpoint with:

```bash
thrawn-nnue resume --checkpoint runs/my_first_run/checkpoints/step_00001000.pt
```

Checkpoints contain:

- model weights
- optimizer state
- scheduler state
- AMP scaler state
- config snapshot
- RNG state
- epoch and step counters

That makes resume suitable for continuing a run exactly, not just reloading weights.

## 6. Export

Export a trained checkpoint to `.nnue`:

```bash
thrawn-nnue export --checkpoint runs/my_first_run/checkpoints/step_00010000.pt --out runs/my_first_run/model.nnue
```

The binary format is documented in [nnue_format.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_format.md).

## 7. Verify the export

Compare exported inference to checkpoint inference:

```bash
thrawn-nnue verify-export --checkpoint runs/my_first_run/checkpoints/step_00010000.pt --nnue runs/my_first_run/model.nnue
```

This reports parity metrics on fixture positions.
