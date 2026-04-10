# NNUE Specification

This document describes the production NNUE contract for the v10 mainline network in this repository and how to use it well in an engine.

## 1. Recommended Network

The preferred v10 network is a small scalar `a768` NNUE:

- feature space: `768`
- per-perspective accumulator size: `256`
- hidden layer size: `32`
- output buckets: `8`
- output head: scalar only
- output perspective: side to move

The architecture is:

```text
white_accumulator: 768 -> 256
black_accumulator: 768 -> 256
combine by side to move: [us_acc | them_acc] -> 512
SCReLU
Linear(512, 32)
Clipped ReLU
Linear(32, 8)
select one bucket by material phase
```

This shape is the mainline balance for v10: fast incremental updates, small dense layers, and enough phase specialization to avoid a single global output for every material regime.

## 2. Feature Indexing Contract

The feature space is always:

- `6` piece types
- `2` relative colors
- `64` oriented squares

Per-piece indexing:

```text
piece_type_index:
P=0 N=1 B=2 R=3 Q=4 K=5
```

The important rule is that feature meaning is defined per accumulator, not as one global STM-relative tensor.

For the white-perspective accumulator:

- white pieces use relative-color bit `0`
- black pieces use relative-color bit `1`
- squares are not flipped

For the black-perspective accumulator:

- black pieces use relative-color bit `0`
- white pieces use relative-color bit `1`
- squares are flipped vertically

The feature index formula is:

```text
feature_index = (piece_type_index * 2 + relative_color_bit) * 64 + oriented_square
```

This repo does **not** use one friendly/enemy tensor that is rewritten into STM-relative form before PyTorch. It always produces two accumulators, one from white’s perspective and one from black’s perspective.

## 3. Accumulator And Inference Rules

Accumulator refresh/update semantics:

```text
white_acc = ft_bias + sum(ft_weight[white_features])
black_acc = ft_bias + sum(ft_weight[black_features])
```

Side to move only affects concatenation order:

```text
if stm == white:
    combined = [white_acc | black_acc]
else:
    combined = [black_acc | white_acc]
```

Dense inference is:

```text
hidden0 = square(clamp(combined, 0, 1))
hidden1 = clamp(hidden0 @ l1_weight + l1_bias, 0, 1)
outputs = hidden1 @ out_weight + out_bias
```

The selected scalar output is chosen by total piece count:

```text
clamped_piece_count = clamp(piece_count, 2, 32)
phase_progress = 32 - clamped_piece_count
bucket = min(output_buckets - 1, (phase_progress * output_buckets) / 31)
```

For best engine performance:

- keep the exact accumulator semantics above
- keep SCReLU and clamp behavior identical
- use incremental accumulator updates in search
- keep trainer/export/engine bucket selection identical

## 4. Export Contract

The preferred v10 format is scalar export version `3`.

Header:

```text
magic[8]                = "THNNUE\0\1"
uint32 version          = 3
char feature_set[16]    = "a768_dual_v1"
uint32 num_features
uint32 ft_size
uint32 hidden_size
uint32 output_buckets
uint32 output_perspective
float  ft_scale
float  dense_scale
float  wdl_scale
uint32 description_length
```

Payload:

```text
description bytes
int16  ft_bias[ft_size]
int16  ft_weight[num_features][ft_size]
int32  l1_bias[hidden_size]
int8   l1_weight[ft_size * 2][hidden_size]
int32  out_bias[output_buckets]
int16  out_weight[hidden_size][output_buckets]
```

## 5. Score Interpretation And Calibration

The selected output bucket produces a raw scalar, not a centipawn score by itself.

Before shipping a net to the engine:

1. Export the best checkpoint.
2. Run `thrawn-nnue verify-export` to confirm checkpoint/export parity.
3. Run `thrawn-nnue calibrate-scale` on the June holdout shard.
4. Use the fitted `cp_per_raw` or its derived normalization constant in the engine.

Recommended workflow:

- calibrate on the same held-out month used for validation
- keep raw output in NNUE space internally until the final engine score conversion
- store the fitted normalization constant alongside the net version you ship

## 6. Practical Usage Guidance

For strongest results with this NNUE:

- prefer `checkpoints/best.pt` rather than the last checkpoint
- keep `8` buckets enabled in both trainer and engine
- preserve the small `256 -> 32` dense path for speed
- treat the engine loader and the trainer export path as one contract, not two similar implementations

Before integrating a net into search, validate a fixed sanity suite:

- starting position
- white up pawn
- white up knight
- white up bishop
- white up rook
- white up queen
- simple winning endgames
- reduced-material drawish cases

The sanity suite should confirm:

- outputs are not constant
- materially better positions score above the start position
- side-to-move symmetry has the expected sign behavior
- export quantization has not destroyed the network signal
