# Thrawn NNUE Engine Integration Guide

This document is the engine-side integration contract for Thrawn's exported HalfKP NNUE files. It is written for a C or C++ chess engine with efficiently updatable accumulators and SIMD inference paths for AVX2 and NEON.

The exported network is a scalar evaluator. It returns a centipawn-like score from the side-to-move perspective. WDL is used only during training loss computation and is not part of runtime inference.

## 1. Runtime Contract

The exported architecture is described by the header dimensions. The current production shape is:

| Config | Feature Transformer | Dense Path |
|---|---:|---:|
| v2 | `40960 -> 1024` | `2048 -> 256 -> 64 -> 1` |

This keeps classic HalfKP features and spends most of the extra capacity in the feature transformer, which is the part that stores king-piece feature knowledge. The dense path is deliberately wider than older HalfKP nets, but keeps a tapering shape so the first dense layer can mix the larger concatenated accumulators before the compact scalar head.

For the current v2 config, the runtime graph is:

```text
HalfKP feature transformer: 40960 -> 1024
two perspective accumulators: white_acc[1024], black_acc[1024]
side-to-move concat: [us_acc | them_acc] -> 2048
clipped ReLU
dense: 2048 -> 256
clipped ReLU
dense: 256 -> 64
clipped ReLU
output: 64 -> 1
```

Example current v2 constants:

```cpp
static constexpr int NumFeatures = 40960;
static constexpr int NumFactorFeatures = 640;
static constexpr int MaxActiveFeatures = 30;
static constexpr int FtSize = 1024;
static constexpr int L1Size = 256;
static constexpr int L2Size = 64;
```

The trainer uses a virtual `P` factor table. Exported `.nnue` files contain only coalesced real HalfKP rows; the engine does not load or apply the factor table.

The runtime output is direct cp:

```text
eval(position) -> score_cp_from_side_to_move
```

Do not invert a sigmoid or WDL transform. The search should consume the scalar output directly, then apply the engine's usual value clamping, contempt, draw scaling, mate bounds, or tempo policy.

Training binpack scores are converted from Stockfish internal units before loss computation:

```text
score_cp = raw_score * 100 / 208
```

The WDL teacher target is an expected score derived from Stockfish's win/draw/loss transform, not a plain sigmoid over cp:

```text
win = sigmoid((cp - offset) / scale)
loss = sigmoid((-cp - offset) / scale)
draw = 1 - win - loss
expectation = win + 0.5 * draw
```

With `wdl_lambda = 0.9`, training uses 90% teacher expectation and 10% game-result expectation.

## 2. File Format

All multi-byte values are little-endian. Export version is `6`.

Header:

```text
char   magic[8]             = "THNNUE\0\1"
uint32 version              = 6
char   feature_set[16]      = "halfkp_v1\0..."  // feature-family id, not the net size
uint32 num_features         = 40960
uint32 ft_size              // 1024
uint32 l1_size              // 256
uint32 l2_size              // 64
uint32 output_perspective   = 1
float  ft_scale
float  l1_scale
float  l2_scale
float  out_scale
uint32 description_length
```

Payload follows immediately:

```text
uint8  description[description_length]
int16  ft_bias[ft_size]
int16  ft_weight[num_features][ft_size]
int32  l1_bias[l1_size]
int8   l1_weight[ft_size * 2][l1_size]
int32  l2_bias[l2_size]
int8   l2_weight[l1_size][l2_size]
int32  out_bias[1]
int8   out_weight[l2_size]
```

`output_perspective = 1` means side-to-move output. Reject any other value.

Recommended loader checks:

- `magic == "THNNUE\0\1"`
- `version == 6`
- `feature_set == "halfkp_v1"` (the classic HalfKP feature-family id)
- dimensions match the engine-supported production shape above
- `output_perspective == 1`
- file has no trailing short reads
- all tensors are aligned or copied into aligned engine-owned storage

## 3. Quantization Model

Stored weights are quantized by simple symmetric scaling:

```text
float_value = integer_value / scale
integer_value = round(float_value * scale)
```

Scales are stored in the header:

```cpp
float ft_scale;
float l1_scale;
float l2_scale;
float out_scale;
```

Tensor scales:

| Tensor | Stored Type | Scale |
|---|---:|---:|
| `ft_bias` | `int16` | `ft_scale` |
| `ft_weight` | `int16` | `ft_scale` |
| `l1_bias` | `int32` | `l1_scale` |
| `l1_weight` | `int8` | `l1_scale` |
| `l2_bias` | `int32` | `l2_scale` |
| `l2_weight` | `int8` | `l2_scale` |
| `out_bias` | `int32` | `out_scale` |
| `out_weight` | `int8` | `out_scale` |

The simplest correct integer inference keeps activations in the scale of the previous layer:

```text
accumulator scale: ft_scale
clipped0 range: [0, ft_scale]
h1 range: [0, l1_scale]
h2 range: [0, l2_scale]
output raw scale: out_scale
final cp = round(raw_output / out_scale)
```

The fast integer formulas below assume the exported scales are integral or very close to integral. The default export settings are intended to produce integral scales such as `127` and `64`. If a scale is backed off to a non-integral value to avoid quantization clipping, either use the float reference path, use fixed-point scale reciprocals, or re-export/retrain with enough headroom for integral runtime scales.

## 4. Feature Indexing

Square indexing is `a1 = 0`, `b1 = 1`, ..., `h8 = 63`.

```cpp
int file_of(int sq) { return sq & 7; }
int rank_of(int sq) { return sq >> 3; }
int flip_vertical(int sq) { return (7 - rank_of(sq)) * 8 + file_of(sq); }
```

HalfKP is computed separately for the white and black perspectives. Kings are never active pieces.

Piece type index:

```text
P = 0
N = 1
B = 2
R = 3
Q = 4
```

Relative color bit:

```text
0 = friendly piece from this perspective
1 = enemy piece from this perspective
```

Piece bucket:

```cpp
bucket = piece_type_index * 2 + relative_color_bit; // 0..9
```

Perspective orientation:

```text
white perspective: no square flip, white pieces friendly
black perspective: vertical square flip, black pieces friendly
```

Index formulas:

```cpp
int oriented_king = perspective == White ? king_sq : flip_vertical(king_sq);
int oriented_piece = perspective == White ? piece_sq : flip_vertical(piece_sq);
int p_index = bucket * 64 + oriented_piece;        // 0..639
int halfkp_index = oriented_king * 640 + p_index;  // 0..40959
```

At most 30 non-king pieces are active per perspective.

## 5. Accumulator Model

Each search stack entry should carry two accumulators:

```cpp
struct alignas(64) Accumulator {
    int16_t white[FtSize];
    int16_t black[FtSize];
    bool white_valid;
    bool black_valid;
};
```

Refresh:

```cpp
void refresh_perspective(
    int16_t acc[FtSize],
    const int16_t ft_bias[FtSize],
    const int16_t (*ft_weight)[FtSize],
    const int* features,
    int count
) {
    memcpy(acc, ft_bias, FtSize * sizeof(int16_t));
    for (int n = 0; n < count; ++n) {
        const int16_t* row = ft_weight[features[n]];
        for (int i = 0; i < FtSize; ++i)
            acc[i] += row[i];
    }
}
```

Patch:

```cpp
void patch_perspective(
    int16_t acc[FtSize],
    const int16_t (*ft_weight)[FtSize],
    const int* removed,
    int removed_count,
    const int* added,
    int added_count
) {
    for (int n = 0; n < removed_count; ++n) {
        const int16_t* row = ft_weight[removed[n]];
        for (int i = 0; i < FtSize; ++i)
            acc[i] -= row[i];
    }
    for (int n = 0; n < added_count; ++n) {
        const int16_t* row = ft_weight[added[n]];
        for (int i = 0; i < FtSize; ++i)
            acc[i] += row[i];
    }
}
```

## 6. Efficiently Updatable Evaluation

UE is the point of NNUE: do not rebuild both accumulators at every node.

For every move, compute feature diffs for both perspectives:

```text
removed_white[], added_white[]
removed_black[], added_black[]
```

For non-king moves, captures, en passant, castling rook movement, and promotions, update the relevant piece features incrementally:

```text
remove old piece feature
remove captured piece feature, if any
add new piece feature
add promoted piece feature instead of pawn, if promotion
move rook feature for castling
remove ep-captured pawn from its real square
```

King moves are special because every feature for that king's perspective depends on the king square.

Rules:

- White king move: refresh `white` accumulator from scratch.
- Black king move: refresh `black` accumulator from scratch.
- The opposite perspective does not refresh just because the enemy king moved, because kings are excluded from the active piece list.
- If castling, also patch the rook feature in both perspectives.

Practical lazy-valid strategy:

```cpp
struct Dirty {
    bool white_refresh;
    bool black_refresh;
    SmallList white_removed, white_added;
    SmallList black_removed, black_added;
};

if (dirty.white_refresh)
    refresh white from board;
else
    patch white;

if (dirty.black_refresh)
    refresh black from board;
else
    patch black;
```

For search performance, store accumulators in the stack and derive child accumulators from the parent. Do not allocate during evaluation.

## 7. Reference Integer Forward Pass

This is the canonical integer path. SIMD implementations must match it within rounding tolerance.

```cpp
int round_div_i64(int64_t x, int64_t d) {
    return x >= 0 ? int((x + d / 2) / d) : int((x - d / 2) / d);
}

int evaluate_int_reference(
    const Net& net,
    const int16_t white_acc[FtSize],
    const int16_t black_acc[FtSize],
    bool white_to_move
) {
    const int16_t* us = white_to_move ? white_acc : black_acc;
    const int16_t* them = white_to_move ? black_acc : white_acc;
    const int ft_scale = int(std::lround(net.ft_scale));
    const int l1_scale = int(std::lround(net.l1_scale));
    const int l2_scale = int(std::lround(net.l2_scale));
    const int out_scale = int(std::lround(net.out_scale));

    int16_t clipped0[FtSize * 2];
    for (int i = 0; i < FtSize; ++i) {
        clipped0[i] = std::clamp<int>(us[i], 0, ft_scale);
        clipped0[FtSize + i] = std::clamp<int>(them[i], 0, ft_scale);
    }

    int16_t h1[L1Size];
    for (int j = 0; j < L1Size; ++j) {
        int64_t sum = int64_t(net.l1_bias[j]) * int64_t(ft_scale);
        for (int i = 0; i < FtSize * 2; ++i)
            sum += int64_t(clipped0[i]) * int64_t(net.l1_weight[i][j]);
        int v = round_div_i64(sum, int64_t(ft_scale));
        h1[j] = std::clamp<int>(v, 0, l1_scale);
    }

    int16_t h2[L2Size];
    for (int j = 0; j < L2Size; ++j) {
        int64_t sum = int64_t(net.l2_bias[j]) * int64_t(l1_scale);
        for (int i = 0; i < L1Size; ++i)
            sum += int64_t(h1[i]) * int64_t(net.l2_weight[i][j]);
        int v = round_div_i64(sum, int64_t(l1_scale));
        h2[j] = std::clamp<int>(v, 0, l2_scale);
    }

    int64_t out = int64_t(net.out_bias[0]) * int64_t(l2_scale);
    for (int i = 0; i < L2Size; ++i)
        out += int64_t(h2[i]) * int64_t(net.out_weight[i]);

    int raw = round_div_i64(out, int64_t(l2_scale)); // out_scale units
    return round_div_i64(raw, int64_t(out_scale));   // cp
}
```

If you need exact parity with the Python float verifier, use a float reference path during bring-up, then switch to integer once quantized parity is tested.

## 8. AVX2 Integration

### 8.1 Data Layout

Keep FT rows contiguous:

```cpp
alignas(64) int16_t ft_weight[NumFeatures][FtSize];
alignas(64) int16_t ft_bias[FtSize];
```

For dense layers, the exported layout is row-major by input:

```text
l1_weight[FtSize * 2][L1Size]
l2_weight[L1Size][L2Size]
out_weight[L2Size]
```

For AVX2, also build transposed or packed copies at load time if that simplifies dot products:

```text
l1_weight_t[L1Size][FtSize * 2]
l2_weight_t[L2Size][L1Size]
```

The loader may keep both original and packed forms; the file format stays unchanged.

### 8.2 Accumulator Patches

One FT row is `1024` `int16_t`, exactly 64 AVX2 vectors or 128 NEON `int16x8_t` vectors.

```cpp
#include <immintrin.h>

void add_row_avx2(int16_t acc[FtSize], const int16_t row[FtSize]) {
    for (int i = 0; i < FtSize; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i*)(acc + i));
        __m256i r = _mm256_load_si256((const __m256i*)(row + i));
        _mm256_store_si256((__m256i*)(acc + i), _mm256_add_epi16(a, r));
    }
}

void sub_row_avx2(int16_t acc[FtSize], const int16_t row[FtSize]) {
    for (int i = 0; i < FtSize; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i*)(acc + i));
        __m256i r = _mm256_load_si256((const __m256i*)(row + i));
        _mm256_store_si256((__m256i*)(acc + i), _mm256_sub_epi16(a, r));
    }
}
```

Use unaligned loads only if your allocator cannot guarantee 32-byte alignment. Prefer `alignas(64)` for stack entries and network rows.

### 8.3 Clipping and Packing

If `ft_scale <= 127` or `ft_scale <= 255`, you can pack clipped FT activations to bytes for faster dense multiplication. The exported default usually targets `ft_scale = 127`.

```text
clipped0_i16 = clamp(acc, 0, ft_scale)
clipped0_u8  = narrow/clamp to uint8
```

Dense weights are signed int8. A fast AVX2 implementation can use one of these:

- `_mm256_maddubs_epi16` after arranging unsigned activations and signed weights.
- `_mm256_madd_epi16` after widening activations/weights to int16.
- On AVX512/VNNI targets, a separate implementation can use dot-product instructions.

AVX2 byte dot-product outline:

```cpp
// activation: uint8, weight: int8
// maddubs: pairs u8*s8 -> i16 pair sums
// madd: i16 pair sums -> i32 quad sums
__m256i prod16 = _mm256_maddubs_epi16(act_u8, w_i8);
__m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1));
sum32 = _mm256_add_epi32(sum32, prod32);
```

Keep a scalar or int16 AVX2 fallback for unusual scales where byte packing is not valid.

### 8.4 Dense Layer Sizes

The dense path is `2048 -> 256 -> 64 -> 1`. The first dense layer is intentionally wide enough to mix the larger `us|them` accumulator pair, while the second dense layer keeps the scalar head moderate. All dimensions are divisible by common AVX2 and NEON lane groups. The critical path is usually accumulator maintenance and the first dense layer.

## 9. NEON Integration

### 9.1 Accumulator Patches

One FT row is `FtSize` `int16_t`; the production `FtSize = 1024` is divisible by 8.

```cpp
#include <arm_neon.h>

void add_row_neon(int16_t acc[FtSize], const int16_t row[FtSize]) {
    for (int i = 0; i < FtSize; i += 8) {
        int16x8_t a = vld1q_s16(acc + i);
        int16x8_t r = vld1q_s16(row + i);
        vst1q_s16(acc + i, vaddq_s16(a, r));
    }
}

void sub_row_neon(int16_t acc[FtSize], const int16_t row[FtSize]) {
    for (int i = 0; i < FtSize; i += 8) {
        int16x8_t a = vld1q_s16(acc + i);
        int16x8_t r = vld1q_s16(row + i);
        vst1q_s16(acc + i, vsubq_s16(a, r));
    }
}
```

### 9.2 Dot Products

On ARMv8.4-A with dot product support, use `sdot`/`vdotq_s32` for int8 dense layers. For broader NEON support, use widening multiply-add:

```cpp
int16x8_t act = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(act_u8)));
int16x8_t wt = vmovl_s8(vget_low_s8(w_i8));
int32x4_t lo = vmull_s16(vget_low_s16(act), vget_low_s16(wt));
int32x4_t hi = vmull_s16(vget_high_s16(act), vget_high_s16(wt));
sum = vaddq_s32(sum, lo);
sum = vaddq_s32(sum, hi);
```

Recommended dispatch:

```text
Apple Silicon / ARM dotprod available: NEON dot-product path
generic ARM64: NEON widening path
fallback: scalar reference
```

## 10. Search Integration

Evaluation should be side-to-move:

```cpp
int evaluate(const Position& pos) {
    const Accumulator& acc = ensure_accumulator(pos);
    int cp = nnue.evaluate(acc.white, acc.black, pos.side_to_move() == WHITE);
    return clamp_to_engine_value(cp);
}
```

If your engine stores scores from White's perspective, convert at the boundary:

```cpp
int stm_cp = nnue.evaluate(...);
int white_pov_cp = pos.side_to_move() == WHITE ? stm_cp : -stm_cp;
```

For alpha-beta negamax, side-to-move cp is the natural representation.

Suggested search clamping:

```cpp
static constexpr int EvalLimit = 32000;
static constexpr int MateScore = 30000;
cp = std::clamp(cp, -EvalLimit, EvalLimit);
cp = std::clamp(cp, -MateScore + max_ply, MateScore - max_ply);
```

Do not mix mate scores into NNUE output. Mate values are search values, not eval values.

## 11. Validation Checklist

Before enabling the net in search:

1. Load header and dimensions.
2. Run a float reference evaluator against the Python `verify-export` output.
3. Run integer scalar against float reference.
4. Run AVX2 against integer scalar.
5. Run NEON against integer scalar.
6. Verify accumulator refresh equals incremental update after random legal move sequences.
7. Verify king moves refresh only the moving king perspective.
8. Verify en passant, castling, promotions, and captures.
9. Verify side-to-move behavior on symmetric positions.
10. Run the material sanity ladder:

```text
starting_position < white_up_pawn < white_up_knight < white_up_rook < white_up_queen
```

The repository's export verifier reports:

```text
checkpoint_predictions
exported_predictions
max_abs_error
mean_abs_error
material_ordering_ok
starting_position_near_zero
```

Use it as the source of truth while wiring engine-side parity tests.

## 12. Common Bugs

Wrong square indexing:

- This spec uses `a1 = 0`.
- FEN parsing often iterates ranks from 8 to 1; convert carefully.

Wrong black orientation:

- Black perspective flips vertically only.
- Do not rotate 180 degrees.

Including kings as active pieces:

- Kings determine the HalfKP block.
- Kings are not active piece features.

Applying WDL at inference:

- Do not. The net output is cp-like scalar eval.
- WDL transforms are training loss machinery only.

Refreshing both accumulators on king moves:

- Only the moving king's perspective needs a king-square refresh.
- The opposite perspective may still need ordinary piece patches for rook movement or captures.

Scale mistakes:

- Biases are already stored in their layer's output scale.
- For integer dense layers, multiply bias by the input activation scale before adding products.
- Convert final raw output from `out_scale` units to cp by rounded division.

Silent overflow:

- Use at least `int32` for dense accumulation.
- Use `int64` in scalar reference and tests.
- Accumulator rows are `int16`; patching should remain in range for valid exported nets, but debug builds should assert.

## 13. Implementation Plan

Recommended order:

1. Implement file loader and scalar float evaluator.
2. Implement feature generation and full accumulator refresh.
3. Compare engine float eval with Python verifier on fixed FENs.
4. Implement integer scalar evaluator.
5. Add parent-to-child accumulator patching.
6. Fuzz incremental accumulators against full refresh.
7. Add AVX2 accumulator row add/sub.
8. Add AVX2 first dense layer.
9. Add NEON accumulator row add/sub.
10. Add NEON first dense layer.
11. Dispatch at startup by CPU capability.
12. Run search with scalar/integer/SIMD parity assertions in debug builds.

Keep the scalar integer path permanently. It is the reference for debugging SIMD and for unsupported targets.
