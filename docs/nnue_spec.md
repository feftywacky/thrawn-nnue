# Thrawn NNUE Spec

This is the engine-side contract for Thrawn's exported HalfKP `.nnue` files.

## Runtime Output

The exported net returns a side-to-move score in Stockfish internal score units:

```text
eval(position) -> score_stm
```

Current training/export scale:

```text
score_stm = raw_output * score_scale
score_scale = nnue2score = 600.0
```

Human-facing cp conversion is:

```text
score_cp = score_stm * 100 / 208
```

Use exactly one unit consistently in the engine:

- `score_stm`: if your search constants are tuned in Stockfish-style internal units.
- `score_cp`: if your search constants are tuned in classical centipawns.

Do not apply any sigmoid or WDL transform at runtime.

## Current Net Shape

Current production shape:

```text
HalfKP FT: 40960 -> 1024
concat: [us_acc | them_acc] -> 2048
clipped ReLU
dense: 2048 -> 256
clipped ReLU
dense: 256 -> 64
clipped ReLU
output: 64 -> 1
```

Current constants:

```cpp
static constexpr int NumFeatures = 40960;
static constexpr int NumFactorFeatures = 640;
static constexpr int MaxActiveFeatures = 30;
static constexpr int FtSize = 1024;
static constexpr int L1Size = 256;
static constexpr int L2Size = 64;
```

The trainer uses a virtual `P` factor table. Export does not. Each exported FT row is already coalesced.

## File Format

All values are little-endian.

Header:

```text
char   magic[8]           = "THNNUE\0\1"
uint32 version            = 7
char   feature_set[16]    = "halfkp_v1\0..."
uint32 num_features       = 40960
uint32 ft_size            = 1024
uint32 l1_size            = 256
uint32 l2_size            = 64
uint32 output_perspective = 1
float  ft_scale
float  l1_scale
float  l2_scale
float  out_scale
float  score_scale
uint32 description_length
uint8  description[description_length]
```

Payload:

```text
int16  ft_bias[ft_size]
int16  ft_weight[num_features][ft_size]
int32  l1_bias[l1_size]
int8   l1_weight[ft_size * 2][l1_size]
int32  l2_bias[l2_size]
int8   l2_weight[l1_size][l2_size]
int32  out_bias[1]
int8   out_weight[l2_size]
```

Required loader checks:

- `magic == "THNNUE\0\1"`
- `version == 7`
- `feature_set == "halfkp_v1"`
- dimensions match the engine build
- `output_perspective == 1`

## Quantization

Stored weights use symmetric scaling:

```text
float_value   = integer_value / scale
integer_value = round(float_value * scale)
```

Scales:

| Tensor | Type | Scale |
|---|---:|---:|
| `ft_bias` | `int16` | `ft_scale` |
| `ft_weight` | `int16` | `ft_scale` |
| `l1_bias` | `int32` | `l1_scale` |
| `l1_weight` | `int8` | `l1_scale` |
| `l2_bias` | `int32` | `l2_scale` |
| `l2_weight` | `int8` | `l2_scale` |
| `out_bias` | `int32` | `out_scale` |
| `out_weight` | `int8` | `out_scale` |

## HalfKP Indexing

Square indexing:

```text
a1 = 0, b1 = 1, ..., h8 = 63
```

Kings are not active features.

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
0 = friendly from this perspective
1 = enemy from this perspective
```

Bucket:

```cpp
bucket = piece_type_index * 2 + relative_color_bit; // 0..9
```

Black perspective uses vertical flip only:

```cpp
int flip_vertical(int sq) { return (7 - (sq >> 3)) * 8 + (sq & 7); }
```

Index formulas:

```cpp
int oriented_king  = perspective == White ? king_sq  : flip_vertical(king_sq);
int oriented_piece = perspective == White ? piece_sq : flip_vertical(piece_sq);
int p_index        = bucket * 64 + oriented_piece;       // 0..639
int halfkp_index   = oriented_king * 640 + p_index;      // 0..40959
```

At most 30 non-king pieces are active per perspective.

## Accumulators

Store one accumulator per perspective:

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
acc = ft_bias
for each active feature:
    acc += ft_weight[feature]
```

Patch:

```cpp
for removed feature: acc -= ft_weight[feature]
for added feature:   acc += ft_weight[feature]
```

King moves are special:

- White king move: refresh `white`
- Black king move: refresh `black`
- Opposite perspective does not refresh just because the enemy king moved

## Reference Forward Pass

Input ordering is side-to-move:

```text
us_acc   = stm ? white_acc : black_acc
them_acc = stm ? black_acc : white_acc
```

Reference flow:

```text
clip accumulator to [0, ft_scale]
concat [us | them]
dense + clipped ReLU
dense + clipped ReLU
output dense
divide by out_scale
multiply by score_scale
```

The repository verifier is the parity target:

```text
checkpoint_score
exported_score
checkpoint_cp
exported_cp
```

Use `score` values for engine integration. `cp` values are display conversions.

## Engine Integration

Side-to-move form is the native search value:

```cpp
int score_stm = nnue.evaluate(acc.white, acc.black, pos.side_to_move() == WHITE);
```

If your engine wants White POV:

```cpp
int score_white = pos.side_to_move() == WHITE ? score_stm : -score_stm;
```

If your search is cp-based:

```cpp
int eval_cp = round(score_stm * 100.0 / 208.0);
```

Do not mix mate scores into NNUE output. Mate bounds stay search-side.

## Validation Checklist

Before turning the net on in search:

1. Verify loader/header parsing.
2. Verify scalar parity against `thrawn-nnue verify-export`.
3. Verify incremental accumulator patching against full refresh.
4. Verify king moves, castling, promotions, en passant, captures.
5. Verify side-to-move sign handling.
6. Verify material sanity ordering:

```text
equal_material < white_up_pawn < white_up_knight < white_up_rook < white_up_queen
```

