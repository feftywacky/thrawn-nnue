# thrawn-nnue

`thrawn-nnue` is a CLI trainer for a HalfKP chess NNUE.

## Architecture

The repo targets classic HalfKP with a larger v2 production shape:

- feature set: `halfkp`
- v2 feature transform: `40960 -> 1024`
- v2 dense path: `2048 -> 256 -> 64 -> 1`
- output perspective: side to move
- raw exported output: direct centipawns

Training uses classic HalfKP with training-time `P` factorization. Exported `.nnue` files contain only coalesced real HalfKP weights.

## Requirements

- Python `>=3.11`
- PyTorch `>=2.2`
- NumPy `>=1.26`
- Matplotlib `>=3.8`
- tqdm `>=4.66`

## Build Overview

There are two separate pieces in this repo:

1. Python package: installs the CLI and the training/export code.
2. Native `.binpack` bridge: a C++ shared library used to read and inspect `.binpack` datasets.

Use this quick map:

- Python only: `thrawn-nnue export`, `thrawn-nnue verify-export`, `thrawn-nnue metrics`
- Python + native C++ bridge: `thrawn-nnue inspect-binpack`, `thrawn-nnue inspect-binpack-dir`, `thrawn-nnue train`, `thrawn-nnue resume`

The native bridge can build automatically on first use, but you can also build it manually.

## Python Setup

### 1. Create a virtual environment

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Windows Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

### 2. Install PyTorch

CPU only:

```bash
python -m pip install torch
```

CUDA 12.1:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install this repo

If you already chose a Torch build and do not want pip to replace it, use:

```bash
python -m pip install --no-deps -e .
python -m pip install "numpy>=1.26" "matplotlib>=3.8" "tqdm>=4.66"
```

If you are fine letting pip resolve everything:

```bash
python -m pip install -e .
```

### 4. Confirm the Python install

```bash
thrawn-nnue --help
```

If you installed CUDA, also confirm Torch sees it:

```bash
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda)"
```

If `cuda_available= False`, you likely installed a CPU-only wheel or your CUDA runtime does not match the PyTorch wheel.

## Native `.binpack` Bridge Build

### What it needs

- CMake `>=3.20`
- A C++20 compiler
- On Windows: Visual Studio Build Tools / MSVC

### Manual build

From the repo root:

```bash
cmake -S native_binpack -B build/native_binpack
cmake --build build/native_binpack --config Release
```

### Manual build on Windows

The same commands work in PowerShell:

```powershell
cmake -S native_binpack -B build/native_binpack
cmake --build build/native_binpack --config Release
```

If CMake cannot find a compiler, install Visual Studio Build Tools with the C++ workload and retry from a Developer PowerShell or a shell where MSVC is on `PATH`.

### Verify the native build

This forces the Python package to build or load the native library and prints the resulting shared library path:

```bash
python -c "from thrawn_nnue.native import build_native_extension; print(build_native_extension())"
```

## Windows Compatibility Notes

The native bridge is intended to work on Windows, but it is more sensitive there because it depends on a local C++ toolchain.

Current notes:

- The Python package itself is cross-platform.
- The native bridge now uses Windows-safe compiler flags in CMake.
- If native compilation fails on Windows, the usual cause is missing MSVC build tools or CMake not finding the generator/toolchain.
- Commands that touch `.binpack` data depend on the native bridge, so Python may install correctly even when dataset commands still fail.

## Quick Start

1. Edit [v2.toml](/Users/feiyulin/Code/thrawn-nnue/configs/v2.toml) and point `train_datasets` / `validation_datasets` at your `.binpack` files.

2. Optionally verify the native bridge first:

```bash
python -c "from thrawn_nnue.native import build_native_extension; print(build_native_extension())"
```

3. Inspect one dataset:

```bash
thrawn-nnue inspect-binpack --path /absolute/path/to/train.binpack
```

`inspect-binpack` reports score percentiles, WDL saturation diagnostics, and a starting recommendation for `score_clip` / `wdl_scale`.

4. Train v2:

```bash
thrawn-nnue train --config configs/v2.toml
```

5. Resume if needed:

```bash
thrawn-nnue resume --checkpoint runs/v2/checkpoints/step_00001000.pt
```

6. Export the best checkpoint:

```bash
thrawn-nnue export --checkpoint runs/v2/checkpoints/best.pt --out runs/v2/model.nnue
```

7. Verify checkpoint/export parity and sanity scores:

```bash
thrawn-nnue verify-export --checkpoint runs/v2/checkpoints/best.pt --nnue runs/v2/model.nnue
```

8. Summarize the run and generate plots:

```bash
thrawn-nnue metrics --run-dir runs/v2
```
