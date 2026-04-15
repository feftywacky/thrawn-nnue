# NNUE Specification

This document describes the production HalfKP NNUE contract for this repository and how to integrate it into a C++ engine.

## 1. Recommended Network

The production network is a fixed scalar HalfKP NNUE:

- feature space: `40960`
- factor space: `640` training-only `P` factors
- per-perspective accumulator size: `256`
- dense path: `512 -> 32 -> 32 -> 1`
- output perspective: side to move
- raw output: direct centipawns

The architecture is:

```text
HalfKP FT: 40960 -> 256
two accumulators: white, black
combine by side to move: [us_acc | them_acc] -> 512
Clipped ReLU
Linear(512, 32)
Clipped ReLU
Linear(32, 32)
Clipped ReLU
Linear(32, 1)
```

The trainer uses classic HalfKP with training-time `P` factorization. Exported nets contain only coalesced real HalfKP weights, never the factor table.

## 2. Feature Indexing Contract

Each perspective uses its own king square and excludes both kings from the active piece list.

### 2.1 Piece Buckets

Per non-king piece:

```text
piece_type_index:
P=0 N=1 B=2 R=3 Q=4
```

Relative color bit:

- `0` = friendly piece from that perspective
- `1` = enemy piece from that perspective

Bucket formula:

```text
piece_bucket = piece_type_index * 2 + relative_color_bit
```

This gives `10` buckets total.

### 2.2 Orientation

For the white perspective:

- white king is `our king`
- white pieces are friendly
- black pieces are enemy
- squares are not flipped

For the black perspective:

- black king is `our king`
- black pieces are friendly
- white pieces are enemy
- squares are flipped vertically

Vertical flip:

```text
oriented_square = (7 - rank) * 8 + file
```

### 2.3 HalfKP Index

Let:

- `ksq` = our king square for the perspective
- `sq` = the non-king piece square
- `bucket` = piece bucket from above
- `oriented_ksq` = oriented king square
- `oriented_sq` = oriented piece square

Then:

```text
p_index      = bucket * 64 + oriented_sq
halfkp_index = oriented_ksq * 640 + p_index
```

Because:

```text
64 king squares * 10 buckets * 64 piece squares = 40960 features
```

The factor-space `P` index is exactly `p_index`.

### 2.4 Active Features

Active features are:

- all non-king pieces for the white perspective, indexed with the white king square
- all non-king pieces for the black perspective, indexed with the black king square

At most `30` features are active per perspective.

## 3. Accumulator Rules

Per perspective:

```text
acc = ft_bias + sum(ft_weight[halfkp_features])
```

Side to move affects only concatenation order:

```text
if stm == white:
    combined = [white_acc | black_acc]
else:
    combined = [black_acc | white_acc]
```

Forward pass:

```text
hidden0 = clamp(combined, 0, 1)
hidden1 = clamp(hidden0 @ l1_weight + l1_bias, 0, 1)
hidden2 = clamp(hidden1 @ l2_weight + l2_bias, 0, 1)
output  = hidden2 @ out_weight + out_bias
```

The output is already a centipawn value from the side-to-move perspective. The engine should not apply any additional calibration, normalization constant, or start-position bias subtraction.

### 3.1 Incremental Update Semantics

Non-king moves:

- update the perspective whose non-king feature changed with add/remove row operations
- if the moved/captured/promoted piece is not a king, the opposing accumulator usually changes only where that piece appears as friendly/enemy

King moves:

- refresh the moving side's perspective accumulator from scratch because every HalfKP feature depends on `our king square`
- the opposite perspective does not require a king-square refresh because enemy kings are excluded from the feature list

In practice:

- white king move: refresh white accumulator, patch black accumulator only if another non-king piece changed
- black king move: refresh black accumulator, patch white accumulator only if another non-king piece changed

## 4. Training-Time Factorization

Training uses a virtual `P` factor table:

```text
P: 640 -> 256
```

During training, each active feature contributes:

```text
effective_row = ft_weight[halfkp_index] + p_weight[p_index]
```

So:

```text
acc = ft_bias + sum(ft_weight[halfkp_index] + p_weight[p_index])
```

At export time the factor table is coalesced:

```text
coalesced_ft_weight[halfkp_index] =
    ft_weight[halfkp_index] + p_weight[halfkp_index % 640]
```

Only `coalesced_ft_weight` is written to the `.nnue` file.

## 5. Export Contract

The format is export version `5`.

Header (all little-endian):

```text
magic[8]                = "THNNUE\0\1"
uint32 version          = 5
char feature_set[16]    = "halfkp_v1"
uint32 num_features
uint32 ft_size
uint32 l1_size
uint32 l2_size
uint32 output_perspective
float  ft_scale
float  l1_scale
float  l2_scale
float  out_scale
float  wdl_scale
uint32 description_length
```

Payload:

```text
description bytes
int16  ft_bias[ft_size]
int16  ft_weight[num_features][ft_size]
int32  l1_bias[l1_size]
int8   l1_weight[ft_size * 2][l1_size]
int32  l2_bias[l2_size]
int8   l2_weight[l1_size][l2_size]
int32  out_bias[1]
int8   out_weight[l2_size]
```

Quantization:

| Parameter | Scale | Stored Type |
|---|---|---|
| `ft_bias` | `ft_scale` | int16 |
| `ft_weight` | `ft_scale` | int16 |
| `l1_bias` | `l1_scale` | int32 |
| `l1_weight` | `l1_scale` | int8 |
| `l2_bias` | `l2_scale` | int32 |
| `l2_weight` | `l2_scale` | int8 |
| `out_bias` | `out_scale` | int32 |
| `out_weight` | `out_scale` | int8 |

Dequantization:

```text
float_value = quantized / scale
```

## 6. Scalar Reference Inference

This is the reference engine-side float logic:

```cpp
float evaluate_float(const Net& net,
                     const int16_t white_acc[256],
                     const int16_t black_acc[256],
                     bool white_to_move) {
    float combined[512];
    const int16_t* us   = white_to_move ? white_acc : black_acc;
    const int16_t* them = white_to_move ? black_acc : white_acc;

    for (int i = 0; i < 256; ++i) {
        combined[i]       = std::clamp(float(us[i])   / net.ft_scale, 0.0f, 1.0f);
        combined[256 + i] = std::clamp(float(them[i]) / net.ft_scale, 0.0f, 1.0f);
    }

    float h1[32];
    for (int j = 0; j < 32; ++j) {
        float sum = float(net.l1_bias[j]) / net.l1_scale;
        for (int i = 0; i < 512; ++i)
            sum += combined[i] * (float(net.l1_weight[i][j]) / net.l1_scale);
        h1[j] = std::clamp(sum, 0.0f, 1.0f);
    }

    float h2[32];
    for (int j = 0; j < 32; ++j) {
        float sum = float(net.l2_bias[j]) / net.l2_scale;
        for (int i = 0; i < 32; ++i)
            sum += h1[i] * (float(net.l2_weight[i][j]) / net.l2_scale);
        h2[j] = std::clamp(sum, 0.0f, 1.0f);
    }

    float out = float(net.out_bias[0]) / net.out_scale;
    for (int i = 0; i < 32; ++i)
        out += h2[i] * (float(net.out_weight[i]) / net.out_scale);
    return out; // direct centipawns from STM
}
```

## 7. Quantized Integer Inference

Define:

```text
QA = ft_scale
Q1 = l1_scale
Q2 = l2_scale
QO = out_scale
```

### 7.1 Accumulator Refresh

```cpp
void refresh(int16_t acc[256], const int16_t ft_bias[256],
             const int16_t ft_weight[][256], const int* features, int count)
{
    memcpy(acc, ft_bias, 256 * sizeof(int16_t));
    for (int f = 0; f < count; ++f)
        for (int i = 0; i < 256; ++i)
            acc[i] += ft_weight[features[f]][i];
}
```

### 7.2 Incremental Update

```cpp
void update(int16_t acc[256], const int16_t ft_weight[][256],
            const int* removed, int n_removed,
            const int* added, int n_added)
{
    for (int f = 0; f < n_removed; ++f)
        for (int i = 0; i < 256; ++i)
            acc[i] -= ft_weight[removed[f]][i];
    for (int f = 0; f < n_added; ++f)
        for (int i = 0; i < 256; ++i)
            acc[i] += ft_weight[added[f]][i];
}
```

If the moving side king square changes, do a full refresh for that perspective instead of trying to patch all king-conditioned rows.

### 7.3 Dense Forward Pass

```cpp
int evaluate_int(const Net& net,
                 const int16_t white_acc[256],
                 const int16_t black_acc[256],
                 bool white_to_move)
{
    const int16_t* us   = white_to_move ? white_acc : black_acc;
    const int16_t* them = white_to_move ? black_acc : white_acc;

    int16_t clipped0[512];
    for (int i = 0; i < 256; ++i) {
        clipped0[i]       = std::clamp(us[i],   (int16_t)0, (int16_t)net.ft_scale);
        clipped0[256 + i] = std::clamp(them[i], (int16_t)0, (int16_t)net.ft_scale);
    }

    int16_t h1[32];
    for (int j = 0; j < 32; ++j) {
        int64_t sum = 0;
        for (int i = 0; i < 512; ++i)
            sum += (int64_t)clipped0[i] * net.l1_weight[i][j];
        sum += (int64_t)net.l1_bias[j] * net.ft_scale;
        sum /= net.ft_scale; // now in Q1 scale
        h1[j] = (int16_t)std::clamp<int64_t>(sum, 0, (int64_t)net.l1_scale);
    }

    int16_t h2[32];
    for (int j = 0; j < 32; ++j) {
        int64_t sum = 0;
        for (int i = 0; i < 32; ++i)
            sum += (int64_t)h1[i] * net.l2_weight[i][j];
        sum += (int64_t)net.l2_bias[j] * net.l1_scale;
        sum /= net.l1_scale; // now in Q2 scale
        h2[j] = (int16_t)std::clamp<int64_t>(sum, 0, (int64_t)net.l2_scale);
    }

    int64_t out = 0;
    for (int i = 0; i < 32; ++i)
        out += (int64_t)h2[i] * net.out_weight[i];
    out += (int64_t)net.out_bias[0] * net.l2_scale;
    out /= net.l2_scale; // now in QO scale

    return (int)out;
}
```

Final centipawns:

```cpp
int score_cp = evaluate_int(net, white_acc, black_acc, white_to_move) / net.out_scale;
```

If you want rounding instead of truncation:

```cpp
int score_cp = (raw_int >= 0)
    ? (raw_int + net.out_scale / 2) / net.out_scale
    : (raw_int - net.out_scale / 2) / net.out_scale;
```

## 8. C++ Performance Notes

### 8.1 Memory Layout

- Keep `ft_weight` rows contiguous. One HalfKP feature row is exactly one accumulator add/sub patch.
- Store accumulators as `alignas(64) int16_t acc[256]`.
- Store clipped dense buffers as stack arrays, not heap allocations.
- Keep one network blob with already-swizzled little-endian arrays after load.

### 8.2 Cache Locality

- The accumulator update path is the hottest code in search. Optimize row adds/subs first.
- Access FT rows sequentially for the changed features of one move before touching dense layers.
- Keep white and black accumulators adjacent in the node stack to reduce pointer chasing.
- Reuse the same scratch buffers for `clipped0`, `h1`, and `h2` across evaluations.

### 8.3 SIMD

- AVX2 is the primary target.
- Accumulator updates: vectorize `int16` row adds/subs with `_mm256_add_epi16` and `_mm256_sub_epi16`.
- First dense layer: if `ft_scale <= 255`, pack `clipped0` to `uint8_t` and use byte-wise loads with widened multiply-adds; otherwise keep the first clipped buffer as `int16_t`.
- Second dense layer and output layer naturally fit packed byte activations because the clipped outputs live in `[0, l1_scale]` / `[0, l2_scale]` and the default dense scales keep them byte-sized.
- Keep a scalar fallback that is bit-for-bit equivalent to the reference logic above.

### 8.4 Refresh Strategy

- Quiet non-king moves should patch both accumulators with a small number of row adds/subs.
- Promotions and captures still patch incrementally unless the moving side king square changed.
- King moves should trigger a full refresh for the moving side perspective only.

## 9. Sanity Suite

Before shipping a net to the engine, confirm:

- starting position
- white up pawn
- white up knight
- white up rook
- white up queen

Expected behavior:

- start position is near `0cp`
- `start < pawn < knight < rook < queen`
- checkpoint/export parity is tight
- opposite side-to-move flips the sign in symmetric positions
