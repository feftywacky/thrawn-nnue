# thrawn-nnue

`thrawn-nnue` is an NNUE trainer, exporter, and dataset inspection tool for [Thrawn](https://github.com/feftywacky/thrawn).

- Python training/export pipeline built on PyTorch
- Native C++ bridge for reading `.binpack` datasets efficiently

## Layout

```
src/thrawn_nnue/   training loop, config, export, metrics, CLI
native_binpack/    C++20 shared library for .binpack streaming
configs/           training configs and run artifacts
docs/              engine integration notes
tests/             unit and regression tests
```

## Setup

### 1. Python environment

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch

CPU-only:
```bash
pip install torch
```

CUDA (example):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install this repo

If you want pip to manage all dependencies:
```bash
pip install -e .
```

If you want to keep an existing Torch build:
```bash
pip install --no-deps -e .
pip install "numpy>=1.26" "matplotlib>=3.8" "tqdm>=4.66"
```

Verify:
```bash
thrawn-nnue --help
```

### 4. Build the native `.binpack` bridge

Required for `train`, `resume`, and `inspect-binpack*` commands.

```bash
cmake -S native_binpack -B build/native_binpack
cmake --build build/native_binpack --config Release
```

Verify:
```bash
python -c "from thrawn_nnue.native import build_native_extension; print(build_native_extension())"
```

**Platform notes:** Windows requires Visual Studio Build Tools with the C++ workload. macOS uses Apple clang. Linux uses a recent `g++` or `clang++`. CMake `>= 3.20` and a C++20 compiler are required on all platforms.

## Configuration

Configs live under `configs/`. The current training config is `configs/v4.toml` (baseline on `nodes5000pv2_UHO.binpack`). Fine-tuning from a v4 checkpoint uses `configs/v5.toml` (T60/T70 IsRight Farseer data).

Key config knobs: dataset paths, epoch/batch size, LR schedule, device settings, dataloader filtering, and WDL target shaping.

## CLI

### Train

```bash
thrawn-nnue train --config configs/v4.toml
thrawn-nnue train --config configs/v4.toml --init-checkpoint configs/runs/v4/checkpoints/best.pt
```

`--init-checkpoint` warm-starts weights but resets optimizer/scheduler state.

### Fine-tune

```bash
thrawn-nnue fine-tune --config configs/v5.toml --checkpoint configs/runs/v4/checkpoints/best.pt
```

Use when switching datasets or optimizer settings while keeping trained weights.

### Resume

```bash
thrawn-nnue resume --checkpoint configs/runs/v4/checkpoints/step_00010000.pt
```

### Export

```bash
thrawn-nnue export --checkpoint configs/runs/v4/checkpoints/best.pt --out configs/runs/v4/model.nnue
thrawn-nnue export --checkpoint configs/runs/v4/checkpoints/best.pt --out configs/runs/v4/model.nnue --verify
```

`best.pt` is continuously replaced with the best validation checkpoint. `epoch_####_best.pt` is the epoch-stamped copy of the same checkpoint.

### Verify export

```bash
thrawn-nnue verify-export --checkpoint configs/runs/v4/checkpoints/best.pt --nnue configs/runs/v4/model.nnue
thrawn-nnue verify-export ... --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
```

Add `--json` for full quantization diagnostics.

### Inspect datasets

```bash
thrawn-nnue inspect-binpack --path /path/to/data.binpack
thrawn-nnue inspect-binpack --path /path/to/data.binpack --sample-entries 1000000
thrawn-nnue inspect-binpack-dir --path /path/to/data --jobs 4 --sample-entries 1000000
```

### Metrics and plots

```bash
thrawn-nnue metrics --run-dir configs/runs/v4
thrawn-nnue metrics --run-dir configs/runs/v4 --json
```

### Tests

```bash
thrawn-nnue test
```

## Typical Workflow

1. Build the native bridge.
2. Inspect your dataset with `inspect-binpack`.
3. Edit a TOML config under `configs/`.
4. Train with `thrawn-nnue train`.
5. Fine-tune with `thrawn-nnue fine-tune` when switching to better data.
6. Resume with `thrawn-nnue resume` if interrupted.
7. Check `thrawn-nnue metrics` for loss/MAE plots.
8. Export with `thrawn-nnue export --verify`.
9. Integrate the `.nnue` file — see [docs/nnue_spec.md](docs/nnue_spec.md).

## Troubleshooting

**`train` or `inspect-binpack` fails immediately** — the native bridge didn't build or load. Re-run the CMake build and verify with:
```bash
python -c "from thrawn_nnue.native import build_native_extension; print(build_native_extension())"
```

**CUDA not detected** — you likely have a CPU-only or mismatched Torch wheel. Reinstall Torch first, then reinstall this repo.

**`score` and `cp` differ in export output** — that is expected. `cp` is a display conversion (`score * 100 / 208`). See [docs/nnue_spec.md](docs/nnue_spec.md) for the engine-facing score contract.
