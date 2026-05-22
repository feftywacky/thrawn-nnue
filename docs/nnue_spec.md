# Thrawn NNUE Spec

This is the engine-side contract for Thrawn's exported `HalfKAv2_hm` `.nnue` files.

## Runtime Output

The exported net returns a side-to-move score in Stockfish internal score units:

```text
eval(position) -> score_stm
score_stm = raw_output * score_scale
score_scale = nnue2score = 600.0
```

Human-facing centipawn conversion:

```text
score_cp = score_stm * 100 / 208
```

Use one unit consistently inside the engine. Do not apply sigmoid, WDL, or mate-score transforms at NNUE runtime.

## Net Shape

Current production shape:

```text
HalfKAv2_hm sparse features: 22528
feature transformer: 22528 -> 1024 per perspective
concat [us_acc | them_acc]: 2048
fc0: 2048 -> 32
split fc0: lanes 0..30 hidden, lane 31 forward
activation concat: SCReLU(hidden) || CReLU(hidden) -> 62
fc1: 62 -> 32
activation: CReLU
fc2: 32 -> 1
final: fc2_output + forward_output
```

Constants:

```cpp
static constexpr int NumFeatures = 22528;
static constexpr int PsNb = 11 * 64;
static constexpr int MaxActiveFeatures = 32;
static constexpr int FtSize = 1024;
static constexpr int HiddenSize = 31;
static constexpr int ForwardSize = 1;
static constexpr int Fc0OutputSize = HiddenSize + ForwardSize;
static constexpr int Fc1InputSize = HiddenSize * 2;
static constexpr int Fc1OutputSize = 32;
```

Activation definitions:

```cpp
crelu(x)  = clamp(x, 0, 1)
screlu(x) = crelu(x) * crelu(x)
```

## File Format

All values are little-endian.

Header:

```text
char   magic[8]            = "THNNUE\0\1"
uint32 version             = 8
char   feature_set[16]     = "HalfKAv2_hm\0..."
uint32 num_features        = 22528
uint32 ft_size             = 1024
uint32 hidden_size         = 31
uint32 forward_size        = 1
uint32 fc0_output_size     = 32
uint32 fc1_input_size      = 62
uint32 fc1_output_size     = 32
uint32 output_perspective  = 1
float  ft_scale
float  fc0_scale
float  fc1_scale
float  fc2_scale
float  score_scale
uint32 description_length
uint8  description[description_length]
```

Payload:

```text
int16  ft_bias[ft_size]
int16  ft_weight[num_features][ft_size]
int32  fc0_bias[fc0_output_size]
int8   fc0_weight[ft_size * 2][fc0_output_size]
int32  fc1_bias[fc1_output_size]
int8   fc1_weight[fc1_input_size][fc1_output_size]
int32  fc2_bias[1]
int8   fc2_weight[fc1_output_size]
```

Required loader checks:

- `magic == "THNNUE\0\1"`
- `version == 8`
- `feature_set == "HalfKAv2_hm"`
- all dimensions match the engine build
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
| `fc0_bias` | `int32` | `fc0_scale` |
| `fc0_weight` | `int8` | `fc0_scale` |
| `fc1_bias` | `int32` | `fc1_scale` |
| `fc1_weight` | `int8` | `fc1_scale` |
| `fc2_bias` | `int32` | `fc2_scale` |
| `fc2_weight` | `int8` | `fc2_scale` |

The v4/v5/v6 configs use `ft_scale = 255.0` and dense scales of `64.0`. The trainer forward pass is still floating point; quantization-aware simulation is the next accuracy step.

## HalfKAv2_hm Indexing

Square indexing:

```text
a1 = 0, b1 = 1, ..., h8 = 63
```

Active features include both kings, so a normal chess position has up to 32 active features per perspective.

Piece-square offsets:

```cpp
friendly pawn   = 0 * 64
enemy pawn      = 1 * 64
friendly knight = 2 * 64
enemy knight    = 3 * 64
friendly bishop = 4 * 64
enemy bishop    = 5 * 64
friendly rook   = 6 * 64
enemy rook      = 7 * 64
friendly queen  = 8 * 64
enemy queen     = 9 * 64
king            = 10 * 64
```

The king plane is shared by both colors. This is why the feature count is `32 * 11 * 64 = 22528`, not a 12-plane layout.

King buckets:

```cpp
static constexpr int KingBuckets[64] = {
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
     8,  9, 10, 11, 11, 10,  9,  8,
     4,  5,  6,  7,  7,  6,  5,  4,
     0,  1,  2,  3,  3,  2,  1,  0,
};
```

Index formula:

```cpp
int flip = perspective == Black ? 56 : 0;
int orient = (king_sq & 7) < 4 ? 7 : 0;
int oriented_sq = piece_sq ^ orient ^ flip;
int king_bucket = KingBuckets[king_sq ^ flip];
int index = oriented_sq + piece_square_offset + king_bucket * 704;
```

The trainer, native `.binpack` bridge, exported weight order, and engine must all use exactly this order.

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

Own king moves require a full refresh for that perspective because the bucket and horizontal mirror can change every active feature. Enemy king moves are normal feature remove/add patches for the opposite perspective.

## Reference Forward Pass

Input ordering is side-to-move:

```text
us_acc   = stm ? white_acc : black_acc
them_acc = stm ? black_acc : white_acc
```

Reference flow:

```text
x = concat [us_acc | them_acc]
fc0_out = x @ fc0_weight + fc0_bias
hidden = fc0_out[0..30]
forward = fc0_out[31]
a0 = SCReLU(hidden) || CReLU(hidden)
x = a0 @ fc1_weight + fc1_bias
x = CReLU(x)
x = x @ fc2_weight + fc2_bias
score_stm = (x + forward) * score_scale
```

The repository verifier is the parity target:

```text
checkpoint_score
exported_score
checkpoint_cp
exported_cp
```

Use `score` values for search integration. `cp` values are display conversions.

## Fast Inference Notes

Keep feature rows contiguous and 64-byte aligned. Each feature row is `1024 * int16`, so full refresh touches predictable streaming memory and incremental updates only add/subtract changed rows. Keep white and black accumulators hot in the position state and avoid heap allocation in evaluation.

The dense tail is intentionally tiny. Compute `fc0` into a fixed 32-lane buffer, keep lane 31 as the forward lane, materialize the 62 activation bytes/words in a stack buffer aligned to 64 bytes, then run `fc1` and `fc2`. The exported dense matrices are input-major; engines may transpose or tile them at load time for their SIMD kernels.

Suggested compiler targets:

- x86-64 AVX2 GCC/Clang: `-O3 -DNDEBUG -march=x86-64-v3`, or explicitly `-mavx2 -mbmi2 -mpopcnt`.
- x86-64 AVX2 MSVC: `/O2 /DNDEBUG /arch:AVX2`.
- AArch64 NEON: `-O3 -DNDEBUG -march=armv8-a+simd` or a concrete `-mcpu`.
- AArch64 dot-product NEON: `-O3 -DNDEBUG -march=armv8.2-a+dotprod` when `__ARM_FEATURE_DOTPROD` is available.

AVX2 kernels usually pack/clamp activations and use multiply-add reductions over int8 weights with int32 sums. NEON dot-product builds should prefer `vdotq_s32`/`sdot` paths for int8 dense layers, with a non-dotprod NEON fallback. Keep scalar fallbacks bit-identical for parity testing.

## Engine Integration

Side-to-move form is the native search value:

```cpp
int score_stm = nnue.evaluate(acc.white, acc.black, pos.side_to_move() == WHITE);
```

If the engine wants White POV:

```cpp
int score_white = pos.side_to_move() == WHITE ? score_stm : -score_stm;
```

If the search is cp-based:

```cpp
int eval_cp = round(score_stm * 100.0 / 208.0);
```

Do not mix mate scores into NNUE output. Mate bounds stay search-side.

## Validation Checklist

Before turning the net on in search:

1. Verify loader/header parsing.
2. Verify scalar parity against `thrawn-nnue verify-export`, or use `thrawn-nnue export --verify` when writing the file.
3. Verify incremental accumulator patching against full refresh.
4. Verify king moves, castling, promotions, en passant, captures.
5. Verify side-to-move sign handling.

## Stockfish References

- [Stockfish `HalfKAv2_hm` feature source](https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/features/half_ka_v2_hm.h)
- [Stockfish `HalfKAv2_hm::make_index`](https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/features/half_ka_v2_hm.cpp)
- [Stockfish NNUE architecture header](https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_architecture.h)
- [Stockfish `nnue-pytorch` dataloader skip config](https://github.com/official-stockfish/nnue-pytorch/blob/master/data_loader/config.py)
- [Stockfish `nnue-pytorch` skip predicate](https://github.com/official-stockfish/nnue-pytorch/blob/master/data_loader/cpp/training_data_loader.cpp)
- [Official nnue-pytorch architecture history](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
