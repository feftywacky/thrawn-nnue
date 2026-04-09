# NNUE Specification

This document is the unified trainer, export, and engine contract for the NNUE models produced by this repository after the v9 training rework.

## 1. Scope

This specification covers:

- the v9 scratch-training pipeline
- dataset filtering and preparation
- the `a768` dual-perspective network architecture
- the version-4 export format
- engine inference semantics for both heads
- validation and sanity requirements

## 2. Architecture Contract

The network uses:

- feature set: `a768`
- active features per perspective: `32`
- two shared accumulators: white perspective and black perspective
- side-to-move ordered concatenation
- SCReLU on the concatenated accumulator vector
- trunk: `1536 -> 128`
- piece-count output buckets: `8`
- head type: `dual_value_wdl`

The v9 dense head is:

- value head: `128 -> 8`
- WDL head: `128 -> 8 x 3`

The output bucket is selected from total piece count:

```text
clamped_piece_count = clamp(piece_count, 2, 32)
phase_progress = 32 - clamped_piece_count
bucket = min(output_buckets - 1, (phase_progress * output_buckets) / 31)
```

## 3. Feature Semantics

The feature space remains:

- 6 piece types
- 2 relative colors
- 64 oriented squares

Indexing rules:

```text
piece_type_index:
P=0 N=1 B=2 R=3 Q=4 K=5

relative_color_bit:
white perspective: white=0 black=1
black perspective: black=0 white=1

square orientation:
white perspective: unchanged
black perspective: vertical flip only
```

Feature index:

```text
bucket = piece_type_index * 2 + relative_color_bit
feature_index = bucket * 64 + oriented_square
```

## 4. Training Pipeline

The trainer is position-budget based, not epoch based.

The default v9 config is [test80_a768_v9.toml](/Users/feiyulin/Code/thrawn-nnue/configs/test80_a768_v9.toml) with:

- `total_train_positions = 500000000`
- `validation_positions = 20000000`
- `batch_size = 16384`
- `optimizer = "ranger"` with `RAdam` fallback when a full Ranger implementation is unavailable
- `teacher_lambda_start = 1.0`
- `teacher_lambda_end = 0.75`
- `warmup_positions = 50000000`

### 4.1 Runtime filtering

The loader may filter positions before they reach Python using:

- `filter_min_ply`
- `filter_max_abs_score_cp`
- `filter_skip_bestmove_captures`
- `filter_wld_skip`

`filter_wld_skip` removes positions where the packed result is too unlikely under the upstream Stockfish-style score/result model already embedded in the binpack tooling.

### 4.2 Offline preparation

The repository provides:

```bash
thrawn-nnue prepare-binpack --path input.binpack --out prepared.binpack
```

This command:

- applies the same filtering policy as runtime training
- records rejection reasons
- reports bucket occupancy before and after rebalancing
- can duplicate underrepresented buckets up to `rebalance_cap`

### 4.3 Losses

The value head uses Stockfish-style scalar-in-WDL-space loss:

```text
pred_wdl = sigmoid(prediction_cp / wdl_scale)
target_wdl = sigmoid(target_cp / wdl_scale)
value_loss =
    teacher_lambda * mse(pred_wdl, target_wdl)
  + (1 - teacher_lambda) * mse(pred_wdl, result_wdl)
```

The WDL head uses hard-label cross entropy on:

- loss
- draw
- win

The consistency term ties the two heads together:

```text
expected = Pwin + 0.5 * Pdraw
wdl_raw = logit(clamp(expected))
consistency_loss = mse(sigmoid(value_head / wdl_scale), sigmoid(wdl_raw / wdl_scale))
```

Training phases:

- warmup phase: first `50M` train positions, value loss only
- main phase: value loss + `wdl_ce_weight * wdl_ce_loss` + `head_consistency_weight * consistency_loss`

## 5. Export Format

Version 3 remains the scalar-only export format.

Version 4 is the dual-head export format and is the preferred format for v9.

### 5.1 Header

Little-endian header:

```text
magic[8]                = "THNNUE\0\1"
uint32 version          = 4
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

### 5.2 Version 4 payload

```text
description bytes
int16  ft_bias[ft_size]
int16  ft_weight[num_features][ft_size]
int32  l1_bias[hidden_size]
int8   l1_weight[ft_size * 2][hidden_size]
int32  out_bias[output_buckets]
int16  out_weight[hidden_size][output_buckets]
int32  wdl_out_bias[output_buckets * 3]
int16  wdl_out_weight[hidden_size][output_buckets * 3]
```

Version 3 omits the final two tensors and is interpreted as `head_type = scalar`.

## 6. Engine Inference Contract

Accumulator and trunk semantics are unchanged:

```text
white_acc = ft_bias + sum(ft_weight[white_features])
black_acc = ft_bias + sum(ft_weight[black_features])

combined =
    [white_acc | black_acc] if stm == white
    [black_acc | white_acc] if stm == black

hidden0 = square(clamp(combined, 0, 1))
hidden1 = clamp(hidden0 @ l1_weight + l1_bias, 0, 1)
```

The engine then evaluates the selected bucket:

- value head: `value_raw = hidden1 @ out_weight + out_bias`
- WDL head: `wdl_logits = hidden1 @ wdl_out_weight + wdl_out_bias`

To collapse the WDL head into a scalar:

```text
probabilities = softmax(wdl_logits)
expected = Pwin + 0.5 * Pdraw
wdl_raw = logit(clamp(expected))
```

The engine MUST NOT use `logit(Pwin - Ploss)`.

## 7. Calibration

The repository calibrates against validation `.binpack` data:

```bash
thrawn-nnue calibrate-scale --nnue model.nnue --validation-path /path/to/validation.binpack
```

For version-4 exports the calibrator fits both:

- the value head
- the WDL-collapsed head

It reports:

- per-head `cp_per_raw`
- per-head fit metrics
- a `preferred_head`
- hardcoded sanity positions

## 8. Validation And Sanity

Minimum validation requirements:

1. export a checkpoint
2. verify checkpoint vs export parity
3. validate opening, middlegame, and endgame positions
4. validate both sides to move
5. validate incremental updates against refresh

The trainer and engine SHOULD track a fixed sanity suite containing:

- starting position
- white up pawn
- white up knight
- white up bishop
- white up rook
- white up queen
- simple won endgames
- reduced-material drawish positions

At minimum, materially winning positions SHOULD not score below the start position on either the value head or the collapsed WDL head.

## 9. Performance Priorities

Implementation order SHOULD be:

1. scalar refresh path
2. incremental accumulator updates
3. export parity checks
4. SIMD accumulator kernels
5. SCReLU packing
6. SIMD `1536 -> 128` dense kernel
7. optional SIMD head kernels

The accumulator path remains the highest-value optimization target.
