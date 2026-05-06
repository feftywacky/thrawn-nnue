# thrawn-nnue

`thrawn-nnue` is a cli trainer for a HalfKP chess NNUE.

## Architecture

The repo targets classic HalfKP with a larger v2 production shape:

- feature set: `halfkp`
- v2 feature transform: `40960 -> 1024`
- v2 dense path: `2048 -> 256 -> 64 -> 1`
- output perspective: side to move
- raw exported output: direct centipawns

Training uses classic HalfKP with training-time `P` factorization. Exported `.nnue` files contain only coalesced real HalfKP weights.

## Installation

This repo is packaged with `pyproject.toml`. Current project/package versions declared in the repo:

- `thrawn-nnue`: `0.1.0`
- Python: `>=3.11`
- PyTorch: `>=2.2`
- NumPy: `>=1.26`
- Matplotlib: `>=3.8`
- tqdm: `>=4.66

If you want CUDA training, install a CUDA-enabled PyTorch wheel first, then install this repo without re-resolving dependencies. That avoids accidentally pulling a CPU-only PyTorch build from the default index.

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install CUDA PyTorch.

For CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. Install this repo in editable mode without replacing your chosen Torch build:

```bash
pip install --no-deps -e .
```

4. Install the remaining runtime dependencies:

```bash
pip install "numpy>=1.26" "matplotlib>=3.8" "tqdm>=4.66"
```

5. Confirm that PyTorch can see CUDA:

```bash
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda)"
```

You should see `cuda_available= True`. If it prints `False`, you likely installed a CPU-only build or your CUDA driver/runtime does not match the wheel you installed.

For CPU-only installation, you can simply run:

```bash
python3.11 -m pip install -e .
```

The native `.binpack` bridge builds automatically on first use. Training config files can require CUDA explicitly: [v2.toml](/Users/feiyulin/Code/thrawn-nnue/configs/v2.toml) sets `device = "cuda"`, and training will fail fast if the installed PyTorch build does not include CUDA support.

## Quick Start

1. Edit [v2.toml](/Users/feiyulin/Code/thrawn-nnue/configs/v2.toml) and point `train_datasets` / `validation_datasets` at your Jan-May / June `.binpack` files.

2. Inspect a dataset:

```bash
thrawn-nnue inspect-binpack --path /absolute/path/to/train.binpack
```

`inspect-binpack` reports score percentiles, WDL saturation diagnostics, and a starting recommendation for `score_clip` / `wdl_scale`.

3. Train v2:

```bash
thrawn-nnue train --config configs/v2.toml
```

4. Resume if needed:

```bash
thrawn-nnue resume --checkpoint runs/v2/checkpoints/step_00001000.pt
```

5. Export the best checkpoint:

```bash
thrawn-nnue export --checkpoint runs/v2/checkpoints/best.pt --out runs/v2/model.nnue
```

6. Verify checkpoint/export parity and sanity scores:

```bash
thrawn-nnue verify-export --checkpoint runs/v2/checkpoints/best.pt --nnue runs/v2/model.nnue
```

7. Summarize the run and generate plots:

```bash
thrawn-nnue metrics --run-dir runs/v2
```

