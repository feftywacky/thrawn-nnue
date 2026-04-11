# NNUE Specification

This document describes the production NNUE contract for the v11 mainline network in this repository and how to integrate it into a C++ engine.

## 1. Recommended Network

The preferred v11 network is a small scalar `a768` NNUE:

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

This shape is the mainline balance for v11: fast incremental updates, small dense layers, and enough phase specialization to avoid a single global output for every material regime.

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

Vertical flip for the black accumulator:

```text
oriented_square = (7 - rank) * 8 + file
```

This repo does **not** use one friendly/enemy tensor that is rewritten into STM-relative form before PyTorch. It always produces two accumulators, one from white's perspective and one from black's perspective.

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

The preferred v11 format is scalar export version `3`.

Header (all little-endian):

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

For v11 the quantization scales are:

| Parameter | Scale | Stored Type | Notes |
|---|---|---|---|
| `ft_bias` | `ft_scale` (127) | int16 | `quantized = round(float_value * 127)` |
| `ft_weight` | `ft_scale` (127) | int16 | one row per feature |
| `l1_weight` | `dense_scale` (96) | int8 | `quantized = round(float_value * 96)` |
| `l1_bias` | `dense_scale` (96) | int32 | |
| `out_weight` | `dense_scale` (96) | int16 | |
| `out_bias` | `dense_scale` (96) | int32 | |

To dequantize any parameter back to float: `float_value = quantized / scale`.

## 5. Quantized Integer Inference (Update–Evaluate)

This section describes how to evaluate the network using **only integer arithmetic**, suitable for a C++ engine targeting AVX2 or NEON.

Define the two quantization constants from the export header:

```text
QA = ft_scale   = 127   (accumulator quantization unit)
QB = dense_scale = 96    (dense layer quantization unit)
```

### 5.1 Accumulator Refresh

Each perspective accumulator is a vector of `ft_size` (256) int16 values:

```cpp
// Full refresh from scratch
void refresh(int16_t acc[256], const int16_t ft_bias[256],
             const int16_t ft_weight[][256], const int* features, int count)
{
    memcpy(acc, ft_bias, 256 * sizeof(int16_t));
    for (int f = 0; f < count; f++)
        for (int i = 0; i < 256; i++)
            acc[i] += ft_weight[features[f]][i];
}
```

All values stay in int16 range because the trainer clips feature transform weights during training.

### 5.2 Incremental Accumulator Update

When a move adds/removes pieces, patch the accumulator instead of recomputing:

```cpp
void update(int16_t acc[256], const int16_t ft_weight[][256],
            const int* removed, int n_removed,
            const int* added, int n_added)
{
    for (int f = 0; f < n_removed; f++)
        for (int i = 0; i < 256; i++)
            acc[i] -= ft_weight[removed[f]][i];
    for (int f = 0; f < n_added; f++)
        for (int i = 0; i < 256; i++)
            acc[i] += ft_weight[added[f]][i];
}
```

A quiet move (no capture, no promotion) changes exactly one feature per perspective (remove old square, add new square). A capture changes one feature in the moving side's accumulator and two in the opponent's. Castling changes two features per perspective (king + rook). The caller is responsible for computing which feature indices changed.

Both perspectives must be maintained independently. Only the concatenation order changes based on side to move.

### 5.3 Dense Forward Pass (Integer)

After the accumulators are ready, the dense forward pass converts them into a single scalar score. Every operation uses integer arithmetic with controlled scales.

```cpp
int32_t evaluate(const int16_t white_acc[256], const int16_t black_acc[256],
                 bool white_to_move, int piece_count,
                 const int8_t  l1_weight[512][32],
                 const int32_t l1_bias[32],
                 const int16_t out_weight[32][8],
                 const int32_t out_bias[8])
{
    // --- Step 1: Perspective concatenation ---
    // combined[0..255]   = us  accumulator (int16, scale = QA)
    // combined[256..511] = them accumulator (int16, scale = QA)
    const int16_t* us   = white_to_move ? white_acc : black_acc;
    const int16_t* them = white_to_move ? black_acc : white_acc;

    // --- Step 2: SCReLU ---
    // clamp each value to [0, QA], then square.
    // output range: [0, QA*QA] = [0, 16129]
    int32_t screlu[512];
    for (int i = 0; i < 256; i++) {
        int16_t c = std::clamp(us[i],   (int16_t)0, (int16_t)QA);
        screlu[i] = (int32_t)c * c;
    }
    for (int i = 0; i < 256; i++) {
        int16_t c = std::clamp(them[i], (int16_t)0, (int16_t)QA);
        screlu[256 + i] = (int32_t)c * c;
    }

    // --- Step 3: L1 linear + CReLU ---
    // dot product has scale QB * QA^2.
    // bias is stored at scale QB, so multiply by QA^2 before adding.
    // divide the sum by QA^2 to get scale = QB.
    // clamp to [0, QB] for CReLU.
    int32_t hidden[32];
    for (int j = 0; j < 32; j++) {
        int64_t sum = 0;
        for (int i = 0; i < 512; i++)
            sum += (int64_t)screlu[i] * l1_weight[i][j];
        sum += (int64_t)l1_bias[j] * (QA * QA);
        sum /= (QA * QA);                           // now scale = QB
        hidden[j] = std::clamp((int32_t)sum, 0, QB); // CReLU
    }

    // --- Step 4: Output linear ---
    // dot product has scale QB * QB = QB^2.
    // bias is stored at scale QB, so multiply by QB before adding.
    int bucket = output_bucket(piece_count);
    int64_t out = 0;
    for (int j = 0; j < 32; j++)
        out += (int64_t)hidden[j] * out_weight[j][bucket];
    out += (int64_t)out_bias[bucket] * QB;

    // out is now in QB^2 (= 9216) scale.
    // return it — the caller applies the normalization divisor.
    return (int32_t)out;
}
```

### 5.4 Output Bucket Selection

```cpp
int output_bucket(int piece_count) {
    int clamped = std::clamp(piece_count, 2, 32);
    int phase   = 32 - clamped;
    return std::min(OUTPUT_BUCKETS - 1, (phase * OUTPUT_BUCKETS) / 31);
}
```

`piece_count` is the total number of pieces on the board (both sides, including kings). Range: 2 (K vs K) to 32 (starting position).

### 5.5 Score Normalization

The integer output from `evaluate()` is in `QB²` (9216) scale. Converting to centipawns is a two-step process that maps directly to the `calibrate-scale` output.

**Step 1 — Dequantize to raw float units.**

Divide by `QB²` to recover the same raw value that `calibrate-scale` reports:

```cpp
// raw_float ≈ the "raw" values in calibrate-scale output
double raw_float = (double)evaluate(...) / (QB * QB);  // / 9216
```

**Step 2 — Convert raw to centipawns using the calibration normalization constant.**

`calibrate-scale` outputs a `normalization_constant_rounded` which is defined as `round(100 / cp_per_raw)`. This means: `normalization_constant` raw units = 100 centipawns.

```cpp
// normalization_constant_rounded from calibrate-scale
static constexpr int NORM = 50;  // v11 value

int score_cp = (int)raw_float * 100 / NORM;
```

**Combined as pure integer arithmetic:**

The two divisions can be folded into one to avoid floating point entirely:

```cpp
static constexpr int QB_SQ = 96 * 96;                  // 9216
static constexpr int NORM  = 50;                        // from calibrate-scale

int score_cp = (int)((int64_t)evaluate(...) * 100 / ((int64_t)QB_SQ * NORM));
// = raw_int * 100 / 460800
```

When you retrain or recalibrate, only `NORM` changes. `QB_SQ` only changes if you change `dense_scale` in the export config.

## 6. SIMD Implementation Notes

### 6.1 Accumulator Update (AVX2 / NEON)

The accumulator is 256 int16 values = 512 bytes. This fits in 2 AVX2 registers (256 bit) or 8 NEON registers (128 bit).

```text
AVX2:  16 x int16 per __m256i, so 256 values = 16 registers.
       Use _mm256_add_epi16 / _mm256_sub_epi16 for incremental updates.
       One weight row (256 int16) = 16 loads + 16 adds + 16 stores.

NEON:  8 x int16 per int16x8_t, so 256 values = 32 registers.
       Use vaddq_s16 / vsubq_s16 for incremental updates.
       One weight row = 32 loads + 32 adds + 32 stores.
```

Accumulator update is the hottest path in search. Vectorize it first.

### 6.2 SCReLU (AVX2 / NEON)

Clamp and square 512 int16 values into 512 int32 values:

```text
AVX2:  _mm256_max_epi16(x, zero)           // clamp low
       _mm256_min_epi16(x, qa_vec)          // clamp high
       _mm256_madd_epi16(clamped, clamped)  // square + horizontal pair-add
       Note: _mm256_madd_epi16 computes a[i]*b[i] + a[i+1]*b[i+1] as int32,
       which is NOT what we want for element-wise square.
       Instead use _mm256_mullo_epi16 for low 16 bits and combine, or
       widen to int32 first with _mm256_cvtepi16_epi32 and use _mm256_mullo_epi32.

NEON:  vmaxq_s16(x, zero) + vminq_s16(x, qa_vec) for clamping.
       vmull_s16 (low half) and vmull_high_s16 (high half) to widen to int32 and multiply.
```

### 6.3 L1 Dot Product (AVX2 / NEON)

512 int32 values dot with 512 int8 weights per output neuron. This is the main compute cost after the accumulator.

```text
AVX2:  Widen int8 weights to int16, multiply with int16 clamped values
       (before squaring) using _mm256_maddubs_epi16, then horizontal sum.
       Alternatively, keep screlu as int32 and use sequential multiply-add.
       32 output neurons × 512 inputs = 16384 multiply-adds.

NEON:  Use vmull_s8 or widen-and-multiply. SDOT (int8 dot product) on
       ARMv8.2+ can be very efficient if you restructure the loop.
```

For 32 hidden neurons and 512 inputs, the L1 dot product is small enough that a straightforward vectorized loop is sufficient. No need for blocking or tiling.

### 6.4 Output Layer

32 int32 values dot with 32 int16 weights, once per bucket (or once for the selected bucket). This is trivial — a single vectorized dot product.

## 7. Score Interpretation And Calibration

The selected output bucket produces a raw scalar, not a centipawn score by itself.

Before shipping a net to the engine:

1. Export the best checkpoint.
2. Run `thrawn-nnue verify-export` to confirm checkpoint/export parity.
3. Run `thrawn-nnue calibrate-scale` on the June holdout shard.
4. Compute `engine_divisor = round(QB² / cp_per_raw)` from the calibration result.
5. Embed `engine_divisor` as a constant in the engine.

Recommended workflow:

- calibrate on the same held-out month used for validation
- keep the quantized integer output in the engine until the final division
- store the engine divisor alongside the net version you ship
- if you retrain or change `dense_scale`, recalibrate and recompute the divisor

## 8. Practical Usage Guidance

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
