# thrawn-nnue

`thrawn-nnue` is an NNUE trainer, exporter, and dataset inspection tool for [Thrawn](https://github.com/feftywacky/thrawn).

It has two main pieces:

1. A Python training/export pipeline built on PyTorch.
2. A native C++ bridge for reading `.binpack` datasets efficiently.

## Architecture

High-level layout:

- `src/thrawn_nnue/`
  - training loop, config loading, export, metrics, CLI
- `native_binpack/`
  - C++20 shared library for `.binpack` streaming and inspection
- `configs/`
  - training configs and run artifacts
- `docs/`
  - engine-facing integration notes
- `tests/`
  - unit and regression tests

Current training shape:

```text
HalfKAv2_hm sparse features: 22528
feature transformer: 22528 -> 1024 per perspective
concat [us_acc | them_acc] -> 2048
fc0: 2048 -> 31 hidden lanes + 1 forward lane
SCReLU(31) || CReLU(31) -> 62
fc1: 62 -> 32
CReLU
fc2: 32 -> 1
final output: fc2 + forward lane
output perspective: side to move
output unit: Stockfish internal score
```

The trainer, native `.binpack` bridge, and exporter use the same Stockfish-style `HalfKAv2_hm` feature order. There is no legacy compatibility path or factorized FT export path.

For the exact exported file format, score units, `HalfKAv2_hm` indexing, accumulator rules, SIMD notes, and engine integration contract, see [docs/nnue_spec.md](docs/nnue_spec.md).

## Requirements

### Python

- Python `>= 3.11`
- `torch >= 2.2`
- `numpy >= 1.26`
- `matplotlib >= 3.8`
- `tqdm >= 4.66`

The Python package provides the `thrawn-nnue` CLI.

### Native C++ bridge

The `.binpack` reader is a separate shared library built from `native_binpack/`.

Requirements:

- CMake `>= 3.20`
- a C++20 compiler
- enough RAM and disk for dataset inspection/training

Platform notes:

- Windows: Visual Studio Build Tools / MSVC with the C++ workload
- macOS: Apple clang with Xcode command line tools
- Linux: recent `g++` or `clang++`

## Python Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## PyTorch: CUDA vs CPU-only

Install the PyTorch build you actually want before installing this repo.

### CPU-only

```bash
python -m pip install torch
```

### CUDA

Example CUDA wheels from the PyTorch index:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

or:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
```

After that, install this repo.

If you want to keep the exact Torch build you already installed:

```bash
python -m pip install --no-deps -e .
python -m pip install "numpy>=1.26" "matplotlib>=3.8" "tqdm>=4.66"
```

If you are fine letting pip resolve dependencies:

```bash
python -m pip install -e .
```

Verify the install:

```bash
thrawn-nnue --help
```

Verify CUDA visibility if applicable:

```bash
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda)"
```

Notes:

- `accelerator = "cuda"` in your TOML requires a CUDA-capable Torch build.
- `amp = true`, `tf32 = true`, `pin_memory = true`, and `fused_optimizer = true` are CUDA-oriented optimizations.
- CPU training works, but it is much slower.

## Building the Native `.binpack` Bridge

The native bridge is required for:

- `thrawn-nnue train`
- `thrawn-nnue resume`
- `thrawn-nnue inspect-binpack`
- `thrawn-nnue inspect-binpack-dir`

It is not required for:

- `thrawn-nnue export`
- `thrawn-nnue verify-export`
- `thrawn-nnue metrics`

The library can build automatically on first use, but manual build is often clearer.

### Manual build

```bash
cmake -S native_binpack -B build/native_binpack
cmake --build build/native_binpack --config Release
```

### Verify the native build

```bash
python -c "from thrawn_nnue.native import build_native_extension; print(build_native_extension())"
```

If Windows compilation fails, the usual issue is missing MSVC build tools or opening PowerShell without the C++ toolchain available.

## Configuration

The main training config in this repo is currently [configs/v4.toml](configs/v4.toml).

That config currently points at:

- `data/nodes5000pv2_UHO.binpack`
- output directory `configs/runs/v4`
- Stockfish-style baseline schedule: `max_epochs = 200`, `epoch_size = 100,000,000`, `batch_size = 16,384`
- `HalfKAv2_hm` features with `1024x2 -> 31+1 -> 32 -> 1`
- Lambda schedule: `start_lambda = 1.0` to `end_lambda = 1.0`

The current second-stage fine-tune config is:

- [configs/v5.toml](configs/v5.toml): Stockfish-style T60/T70 IsRight Farseer retrain config after a stable `nodes5000pv2_UHO` baseline.

It starts from a v4 checkpoint and trains on:

- `data/T60T70wIsRightFarseer.binpack`
- output directories under `configs/runs/`

Important config themes:

- dataset paths and validation size
- epoch-derived position budgets
- device/runtime settings
- training budget and checkpoint cadence
- LR schedule
- feature set
- Stockfish-style dataloader skipping: tactical positions, WDL/score mismatch, random FEN skipping, hard/soft early-ply skipping, simple-eval skipping, and dynamic piece-count weighting
- loader throughput: `num_workers` controls native `.binpack` reader concurrency; `data_loader_queue_size` controls Python-side prefetch depth
- WDL target shaping

The exported runtime score contract is documented in [docs/nnue_spec.md](docs/nnue_spec.md), not duplicated here.

## Engine Inference Notes

The runtime path is designed around cache-local incremental accumulators and a compact dense tail:

- Keep one 1024-lane `int16` accumulator per perspective, aligned to 64 bytes.
- Refresh by streaming contiguous FT rows; patch by add/subtracting only changed feature rows.
- Fully refresh the perspective whose own king moved; patch enemy king moves as normal feature deltas.
- Keep `fc0`'s 32 outputs, the forward lane, the 62 activation values, `fc1`, and `fc2` in stack/register-local buffers.
- AVX2 builds should target `-march=x86-64-v3` or `/arch:AVX2`; NEON dot-product builds should target `armv8.2-a+dotprod` when available and fall back to plain NEON.

See [docs/nnue_spec.md](docs/nnue_spec.md) for exact SIMD, cache locality, and compiler-target details.

## `thrawn-nnue` CLI

### Train

Train from a TOML config:

```bash
thrawn-nnue train --config configs/v4.toml
```

Optional:

```bash
thrawn-nnue train --config configs/v4.toml --console-mode text
thrawn-nnue train --config configs/v4.toml --init-checkpoint configs/runs/v4/checkpoints/best.pt
```

`--init-checkpoint` warm-starts model weights but starts a fresh optimizer/scheduler state for the new run.

### Fine-tune

Fine-tune from an existing checkpoint with a new config and a fresh optimizer:

```bash
thrawn-nnue fine-tune --config configs/v5.toml --checkpoint configs/runs/v4/checkpoints/best.pt
```

This is equivalent to `train --init-checkpoint`, but names the workflow directly. Use it when changing datasets or optimizer settings while keeping trained weights.

### Resume

Resume from a saved checkpoint:

```bash
thrawn-nnue resume --checkpoint configs/runs/v4/checkpoints/step_00010000.pt
```

Optional:

```bash
thrawn-nnue resume --checkpoint configs/runs/v4/checkpoints/step_00010000.pt --console-mode text
```

### Export

Export writes the current `HalfKAv2_hm` v8 file format directly. The payload stores the coalesced 22528x1024 feature transformer plus `fc0`, `fc1`, `fc2`, and the forward lane contract documented in [docs/nnue_spec.md](docs/nnue_spec.md).

Export a checkpoint to `.nnue`:

```bash
thrawn-nnue export --checkpoint configs/runs/v4/checkpoints/best.pt --out configs/runs/v4/model.nnue
```

Export and immediately verify checkpoint/export parity:

```bash
thrawn-nnue export --checkpoint configs/runs/v4/checkpoints/best.pt --out configs/runs/v4/model.nnue --verify
```

Checkpoint selection notes:

- `best.pt` is continuously replaced with the best checkpoint by validation `score_mae`.
- `epoch_####_best.pt` is retained beside it as the epoch-stamped copy of the same best checkpoint.

### Verify export

Compare a PyTorch checkpoint against the exported `.nnue` file. The default output is a concise parity report; add `--json` for the full prediction and quantization diagnostics.

```bash
thrawn-nnue verify-export --checkpoint configs/runs/v4/checkpoints/best.pt --nnue configs/runs/v4/model.nnue
```

You can also provide one or more custom FENs:

```bash
thrawn-nnue verify-export --checkpoint configs/runs/v4/checkpoints/best.pt --nnue configs/runs/v4/model.nnue --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
```

### Inspect one dataset

Full scan:

```bash
thrawn-nnue inspect-binpack --path C:/path/to/data.binpack
```

Fast sampled scan:

```bash
thrawn-nnue inspect-binpack --path C:/path/to/data.binpack --sample-entries 1000000
```

Optional diagnostic filters:

```bash
thrawn-nnue inspect-binpack --path C:/path/to/data.binpack --skip-wdl-score-mismatch
```

### Inspect a directory of datasets

```bash
thrawn-nnue inspect-binpack-dir --path C:/path/to/data
```

With parallel inspection and sampling:

```bash
thrawn-nnue inspect-binpack-dir --path C:/path/to/data --jobs 4 --sample-entries 1000000
```

### Metrics and plots

Summarize a run and regenerate plots:

```bash
thrawn-nnue metrics --run-dir configs/runs/v4
```

JSON output:

```bash
thrawn-nnue metrics --run-dir configs/runs/v4 --json
```

Run artifacts typically include:

- `metrics.jsonl`
- `plots/loss.png`
- `plots/mae.png`

### Run tests

Run the full unittest suite through the CLI:

```bash
thrawn-nnue test
```

Useful options:

```bash
thrawn-nnue test --verbosity 1
thrawn-nnue test --failfast
thrawn-nnue test --pattern "test_cli.py"
```

## Typical Workflow

1. Install Python dependencies and the native bridge.
2. Inspect a dataset before training.
3. Edit a TOML config under `configs/`.
4. Start training.
5. Fine-tune with `thrawn-nnue fine-tune` when switching to better data.
6. Resume if interrupted.
7. Inspect `thrawn-nnue metrics`.
8. Export the `best.pt` checkpoint or another chosen checkpoint with `--verify`.
9. Integrate using [docs/nnue_spec.md](docs/nnue_spec.md).

## Tests

The repo uses `unittest`.

Preferred entrypoint:

```bash
thrawn-nnue test
```

Direct Python discovery is equivalent:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run one test file directly:

```bash
python -m unittest discover -s tests -p "test_native.py"
python -m unittest discover -s tests -p "test_training_validation.py"
python -m unittest discover -s tests -p "test_metrics_reporting.py"
```

Current test modules in `tests/`:

- `test_accumulator.py`
- `test_checkpoint_metadata.py`
- `test_cli.py`
- `test_export_format.py`
- `test_features.py`
- `test_inspect_analysis.py`
- `test_metrics_reporting.py`
- `test_native.py`
- `test_training_validation.py`
- `test_validation_config.py`

On Windows with a local venv, that often looks like:

```powershell
thrawn-nnue test
```

or:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## Troubleshooting

### `thrawn-nnue train` or `inspect-binpack` fails immediately

The native bridge likely did not build or load. Re-run:

```bash
python -c "from thrawn_nnue.native import build_native_extension; print(build_native_extension())"
```

### CUDA is not detected

You likely installed a CPU-only Torch wheel or a mismatched CUDA wheel. Reinstall Torch first, then reinstall this repo if needed.

### Exported cp and exported score are different

That is expected.

- `score` is the native exported runtime score.
- `cp` is a display conversion.

Current conversion:

```text
score_cp = score * 100 / 208
```

See [docs/nnue_spec.md](docs/nnue_spec.md) for the engine-facing score contract.
