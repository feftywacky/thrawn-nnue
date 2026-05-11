# thrawn-nnue

`thrawn-nnue` is a HalfKP NNUE trainer, exporter, and dataset inspection tool for [Thrawn](https://github.com/feftywacky/thrawn).

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

Current production network shape:

```text
HalfKP FT: 40960 -> 1024
concat [us_acc | them_acc] -> 2048
dense: 2048 -> 256
dense: 256 -> 64
dense: 64 -> 1
output perspective: side to move
output unit: Stockfish internal score
```

The trainer uses training-time factorization for the feature transform. Exported `.nnue` files contain coalesced HalfKP weights only.

For the exact exported file format, score units, HalfKP indexing, accumulator rules, and engine integration contract, see [docs/nnue_spec.md](docs/nnue_spec.md).

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

- `device = "cuda"` in your TOML requires a CUDA-capable Torch build.
- `amp = true`, `cuda_tf32 = true`, `cuda_pin_memory = true`, and `cuda_fused_optimizer = true` are CUDA-only optimizations.
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

Important config themes:

- dataset paths and validation split
- device/runtime settings
- training budget and checkpoint cadence
- optimizer / LR schedule
- network sizes
- WDL target shaping
- export quantization scales

The exported runtime score contract is documented in [docs/nnue_spec.md](docs/nnue_spec.md), not duplicated here.

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

Export a checkpoint to `.nnue`:

```bash
thrawn-nnue export --checkpoint configs/runs/v4/checkpoints/best.pt --out configs/runs/v4/model.nnue
```

Checkpoint selection notes:

- `best.pt` is the best stable engine-facing checkpoint according to validation score metrics.
- `best_loss.pt` is the checkpoint with the lowest blended validation loss.

### Verify export

Compare a PyTorch checkpoint against the exported `.nnue` file:

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
thrawn-nnue inspect-binpack --path C:/path/to/data.binpack --skip-capture-positions --skip-wdl-score-mismatch --max-abs-score 2000
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
- `plots/loss_overview.png`
- `plots/train_loss.png`
- `plots/validation_loss.png`
- `plots/lr.png`

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
5. Resume if interrupted.
6. Inspect `thrawn-nnue metrics`.
7. Export `best.pt` or another chosen checkpoint.
8. Run `verify-export`.
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
