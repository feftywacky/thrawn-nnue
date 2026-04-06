# Using the Trained `.nnue` In Your Engine Repo

This repository is trainer-only. The output artifact your engine needs is the exported `.nnue` file.

If you want AVX2, AVX-512, or NEON optimization, that work belongs in the engine repo, not here. This trainer produces and exports weights; your engine owns quantized inference, SIMD kernels, accumulator update speed, and cross-architecture parity.

For a concrete engine-side optimization guide, see [simd_optimization.md](/Users/feiyulin/Code/thrawn-nnue/docs/simd_optimization.md).

## What the native code in this repo is for

The code in [native_binpack/](/Users/feiyulin/Code/thrawn-nnue/native_binpack) is trainer-side infrastructure, not engine inference code.

Its job is to:

- read Stockfish-style `.binpack` files efficiently
- decode packed positions using upstream-compatible logic
- extract white-perspective and black-perspective A-768 feature indices
- return batched arrays to Python for training
- generate a tiny `.binpack` fixture for smoke tests

The split of responsibilities is:

- native C++: dataset parsing and feature extraction
- Python/PyTorch: model definition, training loop, checkpoints, export, verification

## What your engine repo needs to do

Your engine needs four pieces.

## 1. Load the exported file

Read the header and tensor layout documented in [nnue_format.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_format.md).

Validate:

- magic is `THNNUE\0\1`
- version is `2` for current exports
- feature set is `a768_dual_v1`
- dimensions match your engine build

## 2. Recreate the same A-768 features

Your engine must generate the same perspective-relative piece-square features as the trainer:

- 6 piece types
- own/opponent color relative to the perspective
- 64 oriented squares
- one pass from White's perspective
- one pass from Black's perspective

Maintain:

- one White-perspective accumulator
- one Black-perspective accumulator

and update both on make and unmake.

## 3. Recreate the same forward pass

At evaluation time:

1. Start each accumulator from `ft_bias`.
2. Add FT rows for each active feature in that perspective.
3. Order them as `[stm_acc, nstm_acc]`.
4. Apply clipped ReLU to `[0, 1]`.
5. Apply the first dense layer.
6. Apply clipped ReLU again.
7. Apply the final output layer.

If you want integer inference, keep the quantized weights and use the scales from [nnue_format.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_format.md).

## 4. Interpret the output

The exported network predicts a side-to-move-relative scalar.

In practice your engine will usually:

- treat positive values as good for side to move
- map the scalar to your internal eval scale
- let the side-to-move ordering handle the sign naturally

## SIMD Responsibilities

Trainer-side responsibilities:

- train a network shape that fits your engine budget
- export weights and scales consistently
- optionally evolve toward more quantization-aware training in the future

Engine-side responsibilities:

- implement fixed-point inference
- pack weights and activations into SIMD-friendly layouts
- write AVX2/NEON accumulator and dense-layer kernels
- ensure x86 and ARM produce matching results
- benchmark NPS impact and tune cache behavior

## Practical plan

1. Add a `ThrawnNnue` file loader.
2. Add A-768 feature extraction shared by full refresh and incremental update code.
3. Add a two-accumulator state object to your position or search stack.
4. Add the forward pass over the exported tensors.
5. Validate parity against the trainer on a few known FENs before optimizing.
