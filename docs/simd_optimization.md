# SIMD Optimization For Engine-Side NNUE

This document describes how to optimize inference for exported `thrawn-nnue` networks in your engine using SIMD on x86 (`AVX2`) and ARM (`NEON`).

This work belongs in the engine repo. The trainer exports quantized weights and a fixed tensor layout, but it does not provide engine inference kernels.

## Scope

The exported network in this repo has this shape:

- feature set: `a768_dual_v1`
- sparse input count: `num_features = 768`
- one accumulator per perspective of width `ft_size`
- first dense layer input width: `2 * ft_size`
- hidden layer width: `hidden_size`
- final output width: `output_buckets` with one selected scalar per position

The exported file layout is documented in [nnue_format.md](/Users/feiyulin/Code/thrawn-nnue/docs/nnue_format.md).

## Forward Pass Structure

At inference time:

1. Start each perspective accumulator from `ft_bias`.
2. Add one `ft_weight[feature]` row per active feature.
3. Order the accumulators as `[stm_acc, nstm_acc]`.
4. Apply clipped ReLU to `[0, 1]`.
5. Compute the first dense layer.
6. Apply clipped ReLU again.
7. Compute the output layer.

In this repo's float reference path, that logic is:

- accumulator build in [`export.py`](/Users/feiyulin/Code/thrawn-nnue/src/thrawn_nnue/export.py#L170)
- dense path in [`export.py`](/Users/feiyulin/Code/thrawn-nnue/src/thrawn_nnue/export.py#L181)

## Quantized Tensors

Current exports use:

- `ft_bias`: `int16[ft_size]`
- `ft_weight`: `int16[num_features][ft_size]`
- `l1_bias`: `int32[hidden_size]`
- `l1_weight`: `int8[ft_size * 2][hidden_size]`
- `out_bias`: `int32[1]`
- `out_weight`: `int16[hidden_size][1]`

The scales are:

- `ft_scale` for the feature transformer tensors
- `dense_scale` for the dense layers

Useful implications:

- accumulator math is naturally `int16` row-add heavy
- the first dense layer is naturally a dot-product from SCReLU accumulator activations into `int32`
- the output layer is tiny and usually not the bottleneck

## What To Optimize First

Usually the wins come in this order:

1. Incremental accumulator updates on make/unmake
2. Full refresh of the two accumulators
3. The first dense layer dot product
4. Weight layout and cache locality
5. Cross-architecture parity and saturation handling

If your engine still does full accumulator rebuilds every node, fix that before chasing dense-layer micro-optimizations.

## Recommended Integer Inference Model

Use a fixed-point pipeline that mirrors the exported representation closely.

Suggested approach:

- keep accumulator state in `int16` or `int32`
- add and subtract FT rows as raw quantized integers
- apply SCReLU to accumulator activations before the first dense layer
- widen to `int32` for dense accumulation
- apply biases in `int32`
- rescale only where necessary for your final eval convention

Two common patterns work:

- `int16` accumulators with widening dot products into `int32`
- `int32` accumulators for simpler correctness, then narrow or clamp before the dense layer

If you are just getting started, prefer the simpler path that is easiest to verify against the trainer.

## Accumulator Layout

For engine-side NNUE, the accumulator is the hot structure.

Recommended layout:

- store White-perspective and Black-perspective accumulators separately
- align each accumulator to at least 32 bytes for `AVX2`
- pad `ft_size` upward to your SIMD width if that simplifies kernels
- store FT rows contiguous by feature: `ft_weight[feature][lane]`

That lets make/unmake do:

- `acc_white += ft_weight[added_white_feature]`
- `acc_white -= ft_weight[removed_white_feature]`
- `acc_black += ft_weight[added_black_feature]`
- `acc_black -= ft_weight[removed_black_feature]`

with simple streaming loads.

## AVX2 Strategy

`AVX2` gives you 256-bit registers, so a natural kernel shape is:

- 16 lanes for `int16`
- 32 lanes for `int8`
- 8 lanes for `int32`

### 1. Accumulator Updates

For FT row adds/subtracts:

- load 16 `int16` values from the accumulator
- load 16 `int16` values from the FT row
- add or subtract with `_mm256_add_epi16` / `_mm256_sub_epi16`
- store back

That loop is simple, predictable, and often memory-bandwidth bound.

Implementation notes:

- align FT rows to 32 bytes if possible
- unroll by 2 to 4 vectors if `ft_size` is large
- prefetch only after measuring; manual prefetching often does nothing or hurts

### 2. SCReLU Before Dense

Your float reference uses SCReLU on the concatenated accumulators: clamp to `[0, 1]`, then square. In integer inference you will usually map that to an integer activation range and reproduce the same transform there.

The important part is consistency:

- choose one fixed-point activation scale
- clamp and square identically on every architecture
- test parity against the float export path

### 3. First Dense Layer

For the `2 * ft_size -> hidden_size` layer:

- treat it as a dot product between the concatenated SCReLU accumulators and each hidden neuron's weight vector
- accumulate into `int32`

Two good layouts:

- neuron-major: one contiguous weight vector per hidden neuron
- blocked: pack several neurons together so one pass over the accumulator feeds multiple outputs

For `AVX2`, blocked layouts are often better because they reduce rereads of the accumulator.

If you stay with neuron-major initially:

- keep the activation vector contiguous
- load 32 `int8` weights at a time
- widen/multiply/accumulate into `int32`

If you later want more speed, repack `l1_weight` into blocks such as 4 or 8 output neurons.

## NEON Strategy

On Apple Silicon and other ARM targets, `NEON` is the right baseline.

Natural kernel shapes:

- 8 lanes for `int16x8_t`
- 16 lanes for `int8x16_t`
- 4 lanes for `int32x4_t`

### 1. Accumulator Updates

The FT row-add kernel maps cleanly:

- load `int16x8_t` chunks from the accumulator and FT row
- add/subtract with `vaddq_s16` / `vsubq_s16`
- store back

Because Apple cores are strong on load/store throughput, keeping the accumulator compact and aligned matters more than clever instruction tricks.

### 2. First Dense Layer

Use widening multiply-accumulate patterns:

- unpack or widen activations and weights as needed
- accumulate into `int32x4_t`
- horizontally reduce at the end

If your target supports ARM dot-product instructions, you can add a faster path later, but a plain `NEON` implementation is the best portable starting point.

## Weight Packing

The exported file is stored for portability, not for final inference speed.

A good engine pipeline is:

1. Load the exported tensors.
2. Validate dimensions and scales.
3. Repack weights into engine-native SIMD layouts.
4. Keep both the original and packed shapes only if debugging requires it.

Recommended packing ideas:

- FT: keep feature-major rows, optionally padded to SIMD width
- L1: repack by output blocks
- output layer: leave simple unless profiling proves otherwise

Avoid premature fancy packing. Repack only the tensors that show up hot in profiling.

## Incremental Update Model

The biggest engine-side gain usually comes from not rebuilding accumulators from scratch.

You want:

- a full-refresh path for root load, TT verification, and debugging
- an incremental path for make/unmake
- a parity test that proves both paths produce identical accumulators

A typical search stack stores:

- White accumulator
- Black accumulator
- dirty-piece metadata sufficient to update both perspectives

Then make/unmake applies row adds and subtracts based on the changed piece-square features.

## Saturation And Numeric Safety

Be careful with saturation semantics.

Areas to watch:

- `int16` accumulator overflow if the network or activation scale grows
- architecture differences between saturating and wrapping arithmetic
- clipping behavior before the first dense layer
- final eval scaling back to your engine's score domain

Practical advice:

- use non-saturating arithmetic only when you have proven the value ranges are safe
- otherwise clamp explicitly at defined points
- log min/max accumulator values during debug builds
- compare x86 and ARM outputs on the same FEN corpus

## Parity Testing

Do not optimize blind. Keep a slow reference path and test against it.

Good validation steps:

1. Load one exported `.nnue`.
2. Run the Python-side export verifier on a few FENs.
3. In the engine, implement a scalar reference path first.
4. Add SIMD kernels.
5. Compare scalar vs SIMD on thousands of random positions.
6. Compare x86 vs ARM on the same position set.

Track:

- exact accumulator equality after full refresh
- exact accumulator equality after incremental updates
- exact hidden-layer outputs if you stay fully integer
- final eval differences, ideally zero or tightly bounded

## Suggested Rollout Plan

1. Implement a scalar loader and scalar forward pass.
2. Implement scalar incremental accumulator updates.
3. Add parity tests against the trainer export path.
4. Add `AVX2` accumulator update kernels.
5. Add `NEON` accumulator update kernels.
6. Optimize the first dense layer only after accumulator updates are correct and fast.
7. Repack weights only after profiling identifies the bottleneck.

## Architecture Notes

### x86-64 / AVX2

- prioritize 32-byte alignment
- use blocked dense kernels if hidden size is large
- benchmark on your real search node mix, not only synthetic eval loops

### ARM64 / NEON

- prioritize compact data layout and predictable memory access
- keep a portable baseline `NEON` path before adding chip-specific fast paths
- on Apple Silicon, measure whole-engine NPS, not just kernel throughput

## Common Mistakes

- optimizing the output layer first
- rebuilding accumulators every node
- using a fancy packed layout before proving correctness
- mixing different clamp or scale conventions across architectures
- comparing only final evals instead of intermediate accumulators

## Minimal Deliverables For A First Fast Engine Port

You do not need everything at once.

A strong first version is:

- scalar reference inference
- incremental accumulators
- `AVX2` row-add kernel
- `NEON` row-add kernel
- scalar dense layer
- parity tests across loader, refresh, and incremental update paths

That is often enough to get most of the practical gain while keeping the codebase understandable.
