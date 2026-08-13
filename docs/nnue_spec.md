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
FT activation (pairwise SqrCReLU): 2048 -> 1024
fc0: 1024 -> 32          (u8 x i8; 31 hidden lanes + 1 dedicated skip lane)
hidden = fc0_out[0..30]                    (the activated lanes)
skip   = fc0_out[31]                       (RAW pre-activation, never activated)
a0 = SqrCReLU(hidden) || CReLU(hidden) -> 62
fc1: 62 -> 32
a1 = SqrCReLU(fc1_out) || CReLU(fc1_out) -> 64
fc2: concat(a0, a1) = 126 -> 1
final: fc2_output + skip
```

The feature transformer output is *activated before* fc0: this is what makes fc0
a `u8 x i8` layer instead of the `i16 x i16` layer that a raw un-clamped
accumulator would force (fc0 is ~60% of engine eval time). The pairwise product
also halves the fc0 input width from `2048` to `1024`.

The dense stack is a single copy — there are no output buckets. `fc0`'s last
output lane is a **dedicated** skip lane: it is excluded from the activations
entirely (only the leading 31 lanes are activated and reach `fc1`), and its raw
`int32` value is added straight onto `fc2`'s output. Total `fc0` output width is
the SIMD-friendly 32; one of those lanes is spent on the skip. This mirrors
Stockfish's own `FC_0_OUTPUTS = 15` (+1 forward lane) = 16 layout.

Constants:

```cpp
static constexpr int NumFeatures = 22528;
static constexpr int PsNb = 11 * 64;
static constexpr int MaxActiveFeatures = 32;
static constexpr int FtSize = 1024;
static constexpr int HiddenSize = 31;                    // activated fc0 lanes
static constexpr int ForwardSize = 1;                    // dedicated skip lane
static constexpr int Fc0OutputSize = HiddenSize + ForwardSize;  // 32
static constexpr int Fc1OutputSize = 32;
static constexpr int Fc0InputSize = FtSize;              // pairwise SqrCReLU: 2*FtSize -> FtSize
static constexpr int Fc1InputSize = HiddenSize * 2;      // 62
static constexpr int Fc2InputSize = HiddenSize * 2 + Fc1OutputSize * 2;  // 126, widened output head
```

Activation definitions. Two DIFFERENT clamp ceilings are in play, because the FT
accumulator and the hidden dense-layer activations quantize onto different-width
integer grids (`FtOne = 256` vs `HiddenOneVal = 128`, matching Stockfish's own
constant names in `nnue_common.h`):

```cpp
static constexpr int FtOne       = 256;  // FT accumulator "one": float v <-> int round(256*v)
static constexpr int FtMaxVal    = 255;  // FT CReLU's integer ceiling (never reaches FtOne)
static constexpr int HiddenOneVal = 128; // hidden activation "one" (fc0/fc1 outputs, and the
                                          // FT pairwise-product output that feeds fc0)

// FT's own per-perspective CReLU, applied to each accumulator half BEFORE the
// pairwise multiply. Ceiling is FtMaxVal/FtOne = 255/256, NOT 1.0 -- the FT's
// u8 grid never reaches a true 1.0.
ft_crelu(x) = clamp(x, 0, 255.0/256.0)

// Hidden dense-layer activations (fc0_out/fc1_out's sqr_crelu/crelu components).
// Ceiling is (HiddenOneVal-1)/HiddenOneVal = 127/128, NOT 1.0.
crelu(x)     = clamp(x, 0, 127.0/128.0)
sqr_crelu(x) = clamp(x * x, 0, 127.0/128.0)   // square-THEN-clamp of the SIGNED
                                               // value, not clamp(x, 0, 127/128) ** 2.
                                               // Negative x still produces a
                                               // positive output here.

// Pairwise SqrCReLU on the feature-transformer output.
// Each perspective's FtSize accumulator is split into two FtSize/2 halves that
// are ft_crelu'd and multiplied together; the two perspectives are concatenated.
// Float reference:
ft_activation(us, them):
    for i in 0 .. FtSize/2 - 1:
        out[i]            = ft_crelu(us[i])   * ft_crelu(us[i + FtSize/2])
        out[i + FtSize/2] = ft_crelu(them[i]) * ft_crelu(them[i + FtSize/2])
    // out has width FtSize and feeds fc0. Not separately clamped: the product's
    // continuous ceiling (255/256)**2 = 0.9921875152... floor-quantizes (see
    // act_quantize below) onto the SAME integer maximum (127) as crelu/sqr_crelu's
    // own 127/128 ceiling, so no extra clipping loss is introduced here.

// Dense-layer activation concat used after fc0 and after fc1: width 2 * N for
// an N-wide raw pre-activation. At fc0 this is applied to the HIDDEN lanes
// only (raw = fc0_out[0..HiddenSize-1]); the skip lane is not part of it.
dense_activation(raw):
    return concat( sqr_crelu(raw), crelu(raw) )

// Floor an activation onto the HiddenOneVal-wide uint8 grid the engine's integer
// bitshift actually produces ((a*b) >> SHIFT in the fc0 hot path below, and
// the equivalent floor for the fc0/fc1 dense_activation outputs). This is now
// an EXACT power-of-two renormalization (see Fast Inference Notes below), not
// an approximation. The repository verifier applies this at every point
// marked "quantized" in the Reference Forward Pass below, UNCONDITIONALLY --
// the real engine always floors here, so the verifier's float reference
// always must too, regardless of whether the checkpoint being verified was
// itself trained with activation QAT. See "Quantization-aware training (QAT)" above.
act_quantize(x):
    return floor(x * HiddenOneVal + 1e-5) / HiddenOneVal
```

## File Format

All values are little-endian.

Header:

```text
char   magic[8]            = "THNNUE\0\1"
uint32 version             = 12
char   feature_set[16]     = "HalfKAv2_hm\0..."
uint32 num_features        = 22528
uint32 ft_size             = 1024
uint32 hidden_size         = 31
uint32 forward_size        = 1
uint32 fc0_output_size     = 32      (== hidden_size + forward_size)
uint32 fc0_input_size      = 1024    (== ft_size)
uint32 fc1_input_size      = 62      (== hidden_size * 2)
uint32 fc1_output_size     = 32
uint32 fc2_input_size      = 126     (== hidden_size * 2 + fc1_output_size * 2)
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
int8   fc0_weight[fc0_input_size][fc0_output_size]     // input-major
int32  fc1_bias[fc1_output_size]
int8   fc1_weight[fc1_input_size][fc1_output_size]
int32  fc2_bias[1]
int8   fc2_weight[fc2_input_size]
```

Required loader checks:

- `magic == "THNNUE\0\1"`
- `version == 12`
- `feature_set == "HalfKAv2_hm"`
- all dimensions match the engine build
- `fc0_output_size == hidden_size + forward_size`
- `fc0_input_size == ft_size`
- `fc1_input_size == hidden_size * 2` (the skip lane is not activated)
- `fc2_input_size == hidden_size * 2 + fc1_output_size * 2`
- `output_perspective == 1`

Version history: `v11` was the 8-bucket ("LayerStacks") net with a shared skip
taken from two of `fc0`'s activated lanes. `v12` drops the buckets and restores
the dedicated skip lane; both the header field list and the payload layout
changed, so `v11` files cannot be read by a `v12` loader at all.

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

One scale per layer, fit over that layer's whole weight tensor.

The current config uses `ft_scale = 256.0` and dense scales of `64.0`, matching
Stockfish's `FtOne`/`WeightScaleBits` constants -- chosen specifically so every
downstream integer renormalization (see Fast Inference Notes below) is an
exact power-of-two shift. Earlier configs (pre-v11) used `ft_scale = 255.0`,
whose renormalizations were only approximate; those exports are incompatible
with this version and cannot be loaded (see the format version bump above).

### Quantization-aware training (QAT)

The trainer's forward pass fake-quantizes weights and activations in-place
(mirroring upstream nnue-pytorch's `model/quantize.py`), so the model *is*
the quantized network rather than a pure-float approximation that only gets
hard-clamped after each optimizer step. This applies identically in train()
and eval() mode -- it is not a training-only trick.

Two straight-through-estimator (STE) primitives, both in `model.py`:

```python
def _fake_quantize_weights(x, scale):
    # weights are ROUNDed at serialization time (export._quantize uses np.rint)
    hard = (x * scale).round() / scale
    return hard.detach() + (x - x.detach())

def _fake_quantize_acts(x, scale):
    # activations are produced by a bitshift in the engine => FLOOR, not round
    hard = (x * scale + 1e-5).floor() / scale
    return hard.detach() + (x - x.detach())
```

The forward value is the hard-quantized value; the gradient passes straight
through unchanged. Both primitives force their arithmetic into float32
regardless of an active `torch.autocast` context: at fp16, `(x * scale).round()`
silently loses precision once `x * scale` exceeds fp16's ~2048 integer-exact
range, which would make the "quantized" value wrong rather than merely
imprecise.

`export.py`'s numpy-only `evaluate_export` reference (see Reference Forward
Pass below) has its own `_quantize_act`, a pure-numpy floor with the same
`floor(x * 128 + 1e-5) / 128` semantics as `_fake_quantize_acts`'s hard value
(no STE needed there -- it's forward-only, no gradients). The two are kept in
sync deliberately, not shared by import, so `evaluate_export` stays
torch-free.

Applied to:

- **Weights, scale = `ft_scale` (256):** `ft.weight`, `ft_bias`.
- **Weights, scale = `dense_scale` (64):** `fc0`/`fc1`/`fc2` weight *and*
  bias (the exporter quantizes each layer's bias at the same scale as its
  weight -- see `_exported_network_from_model` in export.py).
- **The FT's own per-perspective CReLU, ceiling = 255/256** (not weight- or
  activation-grid quantized -- it's a clamp only, applied inside
  `ft_activation` before the pairwise multiply; see Activation definitions
  above).
- **Activations, scale = 128 (`HiddenOneVal`):** the FT pairwise-product
  output feeding `fc0`, and the `sqr_crelu`/`crelu` components at both `fc0`
  (hidden lanes only) and `fc1` (after clamping to 127/128, before
  concatenation). With `ft_scale = 256`, this is now an EXACT renormalization
  (`255*255 >> 9 == 127`, see Fast Inference Notes below), unlike the old
  `ft_scale = 255` scheme's `255**2 / 512 ~= 127.002`.
- **Not applied to the skip connection** (`fc0_out[HiddenSize]`, the dedicated
  lane read from the raw pre-activation `fc0_out`) -- matches upstream's
  `fake_quantize_skip_act`, which is deliberately an identity.

Config flags (`TrainConfig`, all default enabled): `use_fake_weight_quantization`,
`use_fake_act_quantization`, and `use_fake_ft_weight_quantization`. The last
one is split out because `ft.weight` is ~23M floats (92MB fp32) and
fake-quantizing it every forward pass is measurably more expensive than the
other tensors -- roughly a 7% slower full train step (fwd+bwd+opt) and 15-20%
slower forward pass alone, measured at `batch_size=16384` on Apple Silicon
MPS (`accelerator = "mps"`), with no clearly measurable increase in peak MPS
memory beyond measurement noise. This flag lets it be disabled independently
if that cost is unacceptable for a given run.

`_clip_model_weights`'s post-optimizer-step hard clamp is unchanged by QAT
and still runs: QAT does not replace it, since weights must still stay
inside the representable range regardless of how the forward pass simulates
quantization.

Because training's forward pass now fake-quantizes weights at exactly
`export_ft_scale`/`export_dense_scale`, the export step must produce an
exported net using that same scale -- if `_fit_quantization_scale` ever had
to silently reduce the scale to avoid an int overflow, the exported net
would use a different scale than the one training simulated, invalidating
QAT with no visible symptom short of a strength regression. `export.py`
raises `QuantizationScaleMismatchError` instead of silently rescaling. The
error names the specific offending tensor (e.g. `ft_weight`, not just "the
ft group") and states the actionable fix: lower `export_ft_scale` /
`export_dense_scale`, or investigate whether the weights genuinely diverged.
In practice `fc0`/`fc1`/`fc2` weight can't trigger this -- `_clip_model_weights`
bounds them every step -- but `ft.weight`/`ft_bias` are never clamped
anywhere (upstream nnue-pytorch deliberately doesn't clamp the FT either),
so a very long or unstable run is the realistic way this could ever fire;
the error message calls that out specifically for `ft`-prefixed tensors.

`evaluate_export` -- the repository verifier's numpy reference, and the
engine's parity target (see Reference Forward Pass below) -- applies
`_quantize_act` **unconditionally**, at the same three points `model.py`'s
QAT does when `use_fake_act_quantization` is on: the FT pairwise-product
output, and the `sqr_crelu`/`crelu` components at both `fc0` and `fc1`. This
is not gated on the checkpoint's own QAT flags, because the real engine
always floors activations there regardless of how the checkpoint was
trained -- `evaluate_export` has to match the engine, not the checkpoint's
training configuration. A checkpoint trained with
`use_fake_act_quantization = false` will therefore show a real, non-vacuous
discrepancy against `evaluate_export`; that is the verifier correctly
reporting that the float model won't match the engine, not a false alarm.

**Verification** (`thrawn-nnue verify-export`, briefly-trained model with
realistic, off-quantization-grid weight magnitudes -- a genuinely default-init
model's weights are small enough that floor-quantization can coincidentally
absorb weight-rounding noise and mask these properties): with QAT off
entirely, the checkpoint's float forward pass and the exported int net's
dequantized-weight forward pass differ by double digits of score units --
multiple cp of pure uncompensated quantization loss (weight rounding *and*
the activation-floor mismatch below, both present and not simulated during
training). With the realistic full-QAT configuration (both weight and
activation QAT on, the defaults), `model.forward()` and `evaluate_export` are
now computing the *same function*: weights are rounded the same way on both
sides, and activations are floored to the same 128-wide grid on both sides.
The error collapses to near machine precision (single-digit `1e-5` score
units, float32 rounding only) -- not just a reduction, a collapse. Weight
QAT alone (activation QAT off) does **not** collapse the error: it removes
the weight-rounding component but leaves the activation-floor mismatch fully
exposed (`evaluate_export` still floors; the model no longer does), which is
the intended, useful signal from disabling activation QAT, not a regression.
The dense stack's only two lossy steps are weight rounding at serialization
and the activation floor between layers (`fc0_out`/`fc1_out` themselves need
no quantization of their own: the engine's `u8 x i8` accumulation into
`int32` is exact integer arithmetic) -- once QAT simulates both during
training, there is no third source of error left for the export step to
introduce.

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

Reference flow, as computed by the repository verifier (`evaluate_export` in
export.py) against a *loaded, dequantized* `.nnue` file -- this is the exact
function an engine's integer kernel must reproduce, so every `act_quantize`
call below is load-bearing, not optional polish:

```text
x = act_quantize(ft_activation(us_acc, them_acc))    # pairwise SqrCReLU, quantized, width FtSize

fc0_out = x @ fc0_weight + fc0_bias             # RAW, pre-activation, width Fc0OutputSize (32)
hidden  = fc0_out[0 .. HiddenSize-1]            # the 31 activated lanes
skip    = fc0_out[HiddenSize]                   # the DEDICATED lane: raw, NOT quantized,
                                                # NOT activated, and not part of `hidden`
a0      = act_quantize(dense_activation(hidden))    # width 2 * HiddenSize (62), each half quantized separately

fc1_out = a0 @ fc1_weight + fc1_bias                # RAW, pre-activation, width Fc1OutputSize
a1      = act_quantize(dense_activation(fc1_out))   # width 2 * Fc1OutputSize (64), each half quantized separately

out       = concat(a0, a1) @ fc2_weight + fc2_bias + skip   # scalar, skip unscaled
score_stm = out * score_scale
```

`act_quantize(dense_activation(raw))` means `act_quantize` is applied to
`sqr_crelu(raw)` and `crelu(raw)` **separately**, before they're concatenated
-- not to the concatenated result as a single vector (equivalent either way
since `act_quantize` is elementwise, but implementations should not fuse the
clamp/square and the floor into one pass that skips the intermediate
`[0, 127/128]` value).

`ft_activation`, `dense_activation`, and `act_quantize` are defined above.
`ft_activation`/`dense_activation` are pure float (clamp and square only);
`act_quantize` is the separate floor-to-uint8-grid step the engine's integer
bitshift performs, and the reference applies it after every
`ft_activation`/`dense_activation` call, unconditionally. `fc0_out`/`fc1_out`
themselves are never quantized directly -- in the engine, the `u8 x i8` dot
product that produces them accumulates into `int32` exactly (integer
multiply-accumulate introduces no rounding), so the float reference's plain
matmul against dequantized weights reproduces that value exactly. The only
lossy steps in the whole dense stack are (a) weight rounding at serialization
and (b) `act_quantize` between layers -- there is no third source of error to
model. The skip lane adds none of its own: it is a raw `fc0` output added
straight through.

The repository verifier is the parity target:

```text
checkpoint_score
exported_score
checkpoint_cp
exported_cp
```

Use `score` values for search integration. `cp` values are display
conversions. A checkpoint trained with `use_fake_act_quantization = false`
will *not* match this reference closely (see "Quantization-aware training
(QAT)" above) -- that is the verifier correctly reporting a real mismatch
against the engine, not a bug in the verifier.

## Fast Inference Notes

Keep feature rows contiguous and 64-byte aligned. Each feature row is `1024 * int16`, so full refresh touches predictable streaming memory and incremental updates only add/subtract changed rows. Keep white and black accumulators hot in the position state and avoid heap allocation in evaluation.

### Pad the small layers' inputs; do not special-case them

`fc0` — the layer that matters — is fully aligned: 1024 in and 32 out both
divide evenly by every target ISA's vector width, so fc0 runs 32 dot products
and its 32-lane output buffer activates in full-width passes with no remainder.
That alignment is *why* the skip lane is carved out of a 32-wide `fc0` output
(31 hidden + 1) rather than added on top of it: spending one of the 32 lanes on
the skip keeps the expensive layer clean.

The cost lands instead on the two cheap layers, whose widths derive from the 31
*activated* lanes: `fc1`'s input is `2 * 31 = 62` and `fc2`'s is `62 + 64 =
126`. Neither is a multiple of the vector width.

Handle this the way Stockfish does (`ceil_to_multiple` /
`PaddedInputDimensions` in `nnue_architecture.h` / `affine_transform.h`), not
with tail/remainder code: **pad the activation buffers and the corresponding
weight rows up to the next multiple of the vector width, with zeros.** The
padding lanes contribute exactly `0` to the dot product, so the result is
unchanged and the inner loop stays a clean sequence of full-width vectors.
The `.nnue` file always stores the true, unpadded widths (`62`, `126`) — the
padding is an engine-internal, load-time transform of the engine's own copy.

| Layer | Input width | Padded to (AVX2/AVX512/NEON) | Output width |
|---|---:|---:|---:|
| fc0 | 1024 | 1024 (already aligned) | 32 |
| fc1 | 62 | 64 / 64 / 64 | 32 |
| fc2 | 126 | 128 / 128 / 128 | 1 |

fc0 is ~94% of the engine's dense multiply-accumulates (`1024 * 32 = 32768`
vs `62 * 32 + 126 * 1 = 2110` for fc1 + fc2 combined), so it dominates eval
time and is the layer worth hand-tuning hardest -- but all three use the
identical `u8 x i8 -> i32` dot-product primitive below, just at different
widths.

### fc0 as a `u8 x i8` layer (the main hot path)

Feeding fc0 the pairwise-SqrCReLU activation keeps it a `u8 x i8` dot product
instead of the `i16 x i16` path a raw un-clamped accumulator would force, and
halves its input from `2 * FtSize` to `FtSize`. Realize the float
`ft_activation` in integers as:

```text
FtOne = 256           # fixed -- matches export_ft_scale = 256.0 (TrainConfig / HalfKAv2HmNNUE)
FtMaxVal = 255         # FtOne - 1: the FT CReLU's integer ceiling
for i in 0 .. FtSize/2 - 1:
    # us perspective
    a = clamp(us_acc[i],            0, FtMaxVal)     # int16, CReLU
    b = clamp(us_acc[i + FtSize/2], 0, FtMaxVal)
    act[i]            = (a * b) >> 9                 # uint8 in [0, 127]
    # them perspective (same, into act[i + FtSize/2])
```

`>> 9` is now a FIXED shift, not tuned per `ft_scale`: `(256x * 256y) >> 9 =
128*x*y` for any float `x, y` in `[0, 1]`, and the maximum `255 * 255 >> 9 =
127` lands exactly on `HiddenOneVal`'s (128) own integer ceiling -- an exact
renormalization, not an approximation. (The predecessor scheme used `ft_scale
= 255`, whose analogous renormalization was only `255**2 / 512 ≈ 127.002`; see
the Quantization section above.) Then `fc0_out[j] = sum_i act[i] *
fc0_weight[i][j]` accumulates in `int32`.

Bias scaling: the exported `fc0_bias` is quantized at `fc0_scale` (it matches
the float reference where the activation is in `[0, 1]`). Because the integer
product above carries an extra activation factor of `HiddenOneVal = 128`, add
the bias as `fc0_bias[j] * 128` (i.e. pre-multiply the loaded `fc0_bias` by
the uint8 "one") before the fc0 `int32` sum, or fold that factor into a
load-time bias rewrite. `fc1`/`fc2` need the identical `* 128` bias
pre-multiply, for the same reason (their inputs are also `HiddenOneVal`-scaled
`u8` activations) -- see "Layer output scale is exact and uniform" below.

The dense tail is intentionally tiny and fully fixed-size. Compute `fc0` into a
32-lane `int32` buffer, then read lane 31 of that RAW buffer as the skip
(`skip = fc0_out[HiddenSize]`) and set it aside — it is not activated and takes
no further part until the final add. Activate only lanes 0..30, materializing
the 62-byte `a0` (`sqr_crelu(hidden) || crelu(hidden)`) into a 64-byte
stack buffer aligned to 64 bytes with the 2 trailing bytes ZEROED (see
"Pad the small layers' inputs" above). Run `fc1` into a 32-lane buffer,
materialize the 64-byte `a1` the same way, concatenate `a0 || a1` into the
126-byte `fc2` input — again in a zero-padded 128-byte buffer — run `fc2`,
then add the unscaled `skip`.

One ordering caveat on that concatenation: the file's `fc2_weight` is 126
entries in the order `a0` (62) then `a1` (64), **with no gap**. If an engine
finds it more convenient to lay the `fc2` input out as `a0` padded to 64
followed by `a1` (putting the 2 pad bytes in the *middle* rather than at the
end), it must insert two matching zero weights at the same offset in its
in-memory `fc2_weight` copy at load time. The exported dense matrices are
input-major; engines may transpose, tile, or zero-pad them at load time for
their SIMD kernels, as long as activation order and weight order stay
consistent.

### Layer output scale is exact and uniform (`2^13`)

Every dense layer's raw `int32` pre-activation output sits at the SAME
integer scale: `HiddenOneVal * dense_scale = 128 * 64 = 8192 = 2^13`. `fc0_out`
is `sum(u8-at-128 * i8-at-64)`; `fc1_out` is the same shape (its inputs `a0`
are `HiddenOneVal`-scaled `u8`); `fc2_out` is the same shape again, against
`a0 || a1`. This is why `skip = fc0_out[HiddenSize]` (also at scale `8192`,
being a raw fc0-output lane) can be added directly to `fc2_out` with **no
rescaling** -- both operands are already at the identical `2^13` scale, so the
final add is a plain `int32` addition. (Stockfish's own forward lane needs a
`<< (WeightScaleBits - 1)` fixup at this point because its FT output scale
differs; ours does not. Do not copy that shift.)

`ClippedReLU` and `SqrClippedReLU` turn a `2^13`-scaled `int32` into a
`HiddenOneVal = 128`-scaled `u8`, both by an EXACT power-of-two shift:

```text
clipped_relu(out)    = clamp(out >> 6, 0, 127)             # 8192 / 2^6 = 128
sqr_clipped_relu(out) = clamp((out * out) >> 19, 0, 127)    # 8192^2 / 128 = 2^19
```

`>> 6` is `WeightScaleBits` (`log2(dense_scale)`, `dense_scale = 64 = 2^6`).
`>> 19` is `2 * WeightScaleBits + 7`: squaring a `2^13`-scaled value gives
`2^26`-scaled, and `2^26 / 2^7 = 2^19`. Both are exact for any `int32` input --
no multiply-by-reciprocal and no accumulated rounding error to reason about,
on any of the three target ISAs.

### `u8 x i8` dot products (fc0/fc1/fc2)

- **AVX2 + VNNI:** `_mm256_dpbusd_epi32(acc, a, b)` — one instruction computes
  `acc += sum(a[u8] * b[i8])` across 32 lanes, into 8×`i32` lanes of `acc`.
- **AVX512 + VNNI:** `_mm512_dpbusd_epi32(acc, a, b)`, the 64-lane analog.
- **AVX2 without VNNI** (VNNI is a distinct CPU feature from AVX2 itself --
  plenty of AVX2-capable CPUs lack it): fall back to
  `p = _mm256_maddubs_epi16(a, b); p = _mm256_madd_epi16(p, _mm256_set1_epi16(1)); acc = _mm256_add_epi32(acc, p)`.
  `maddubs_epi16` computes pairwise-summed `u8 * i8 -> i16` products (two
  lanes combined per output, safe from overflow at our activation/weight
  magnitudes); `madd_epi16` against all-ones then sums adjacent `i16` pairs
  into `i32`.
- **AVX512 without VNNI:** the same `_mm512_maddubs_epi16` +
  `_mm512_madd_epi16(_, _mm512_set1_epi16(1))` + `_mm512_add_epi32` fallback,
  at 64-lane width.
- **NEON with dot-product** (built with `-march=armv8.2-a+dotprod` or later,
  guarded by `__ARM_FEATURE_DOTPROD`): `vdotq_s32(acc, a, b)`. This is the
  *signed* dot-product intrinsic -- NEON's dot-product extension has no
  unsigned×signed form, so reinterpret the `u8` activation vector as `s8`
  before calling it (safe here since the activation range `[0, 127]` never
  sets the sign bit).
- **NEON without dot-product:** `p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b))`,
  `p1 = vmull_high_s8(a, b)` (two `i16x8` partial-product vectors), `sum =
  vpaddq_s16(p0, p1)` to pairwise-sum them, then `acc = vpadalq_s16(acc, sum)`
  to widen-accumulate into `i32x4`.

Reduce each vector accumulator to a scalar once per output lane after the dot
product (e.g. AVX2: split into two 128-bit halves, `_mm_add_epi32`, then two
`_mm_shuffle_epi32` + add steps; NEON: `vaddvq_s32` on ARMv8, or sum the four
lanes manually pre-ARMv8). Add the pre-multiplied bias (see "Bias scaling"
above) after the horizontal reduction.

### The paired activation: `sqr_crelu` + `crelu` in one pass

At both fc0 and fc1, the architecture needs BOTH `sqr_crelu(raw)` and
`crelu(raw)` of the SAME `int32` pre-activation (`dense_activation`, see
Activation definitions above). Computing them as two independent passes over
`raw` wastes a load and an `i32 -> i16` narrow that both paths need anyway.
Stockfish's `SqrClippedReLU::propagate_pair`
(`src/nnue/layers/sqr_clipped_relu.h`, guarded by `USE_PAIR_ACTIVATIONS`,
which is defined whenever AVX2 is available -- so also on AVX512 builds, which
enable AVX2 too) does exactly this fusion; the sequences below are transcribed
from that function and from the standalone AArch64 `ClippedReLU`/`SqrClippedReLU`
kernels, verified against the Stockfish source rather than reconstructed from
memory.

At fc0 the vector width (32 lanes) covers all of `fc0_out`, including the skip
lane at index 31. Running the paired activation over all 32 lanes and then
simply not copying lane 31's two activated outputs into `a0` is fine and is
what Stockfish does — the skip lane's *raw* value has already been read out
before this point, and its activated value is discarded. What must not happen
is the reverse: letting lane 31's activated value land inside `a0`.

Both paths first narrow the `int32` input to `int16`: the linear path needs
this before its `>> 6`, and the squared path needs it because `mulhi_epi16`
(the fast x86 way to compute the top 16 bits of a 16-bit-lane product) only
takes 16-bit inputs.

**AVX512** (one 512-bit register of `i16` covers 32 lanes; note the final
narrow uses `cvtsepi16_epi8`, a straight signed-saturating *convert*, not a
*pack* -- it does not interleave lanes the way `packs`/`packus` do):

```text
words     = _mm512_packs_epi32(in0, in1)                          # i32 -> i16, saturating
sqrWords  = _mm512_srli_epi16(_mm512_mulhi_epi16(words, words), 3)  # >> (2*6+7-16) = >> 3
squared   = _mm512_cvtsepi16_epi8(sqrWords)                        # i16 -> i8, saturating
clipWords = _mm512_srli_epi16(_mm512_max_epi16(words, zero), 6)     # >> WeightScaleBits
clipped   = _mm512_cvtsepi16_epi8(clipWords)
```

**AVX2** (two 256-bit halves per 32-lane chunk, since `packs_epi32`/`packs_epi16`
only span 8/16 lanes per register; the final narrow here DOES use `packs_epi16`,
which does interleave -- see the lane-order caveat below):

```text
words0  = _mm256_packs_epi32(in0, in1)      # i32 -> i16, first 16 lanes
words1  = _mm256_packs_epi32(in2, in3)      # i32 -> i16, second 16 lanes
sqr0    = _mm256_srli_epi16(_mm256_mulhi_epi16(words0, words0), 3)
sqr1    = _mm256_srli_epi16(_mm256_mulhi_epi16(words1, words1), 3)
squared = _mm256_packs_epi16(sqr0, sqr1)    # i16 -> i8, saturating
clip0   = _mm256_srli_epi16(_mm256_max_epi16(words0, zero), 6)
clip1   = _mm256_srli_epi16(_mm256_max_epi16(words1, zero), 6)
clipped = _mm256_packs_epi16(clip0, clip1)
```

The `srli_epi16` shift (`3`) is `2*WeightScaleBits + 7 - 16`: squaring via
`mulhi_epi16` gives the top 16 bits of the 32-bit product for free (an
implicit `>> 16`), and the full requirement is `>> 19` (see "Layer output
scale is exact and uniform" above), so only `19 - 16 = 3` more bits need
shifting out explicitly.

**NEON:** Stockfish does not implement a fused pair path for NEON --
`propagate_pair` is compiled only under `USE_AVX512`/`USE_AVX2`. Its separate,
existing NEON kernels are:

```text
# ClippedReLU (linear): shift-and-narrow fused into one instruction, 8 lanes/iter
pack0    = vqshrn_n_s32(in0, 6)              # i32 -> i16, >> 6, saturating, one op
pack1    = vqshrn_n_s32(in1, 6)
shifted  = vcombine_s16(pack0, pack1)        # int16x8_t, 8 lanes
clipped  = vmax_s8(vqmovn_s16(shifted), vdup_n_s8(0))   # narrow to i8, then clamp >= 0

# SqrClippedReLU (squared): plain narrow, then a "doubling" multiply-high, 16 lanes/iter
words   = vcombine_s16(vqmovn_s32(in0), vqmovn_s32(in1))   # i32 -> i16, saturating
r       = vshrq_n_s16(vqdmulhq_s16(words, words), 4)       # >> (SHIFT + 1) = >> 4
squared = vqmovn_s16(r)                                    # i16 -> i8, saturating
```

(`vqdmulhq_s16` is a *doubling* multiply-high -- it computes `(2*a*b) >> 16` --
so it needs one extra bit of right-shift versus x86's plain `mulhi_epi16`:
`SHIFT + 1 = 3 + 1 = 4`.) Stockfish's two NEON kernels process different lane
counts per loop iteration (8 for linear, 16 for squared) because each was
tuned independently; an engine that wants the AVX2/AVX512-style shared-narrow
fusion on NEON can still build it by processing both paths at the linear
kernel's 8-lane (`2×int32x4_t`) granularity, sharing one `vqmovn_s32`-based
narrow into `int16x8_t` between the two paths (giving up the fused
`vqshrn_n_s32` for the linear path in exchange for the shared load/narrow) --
profile both structures before committing, since Stockfish itself hasn't
needed to make this call.

**Lane-order caveat:** `packs_epi32`/`packs_epi16` interleave elements at
their operand's lane granularity when narrowing -- the AVX2 result above is
NOT the flat concatenation `[in0[0..7], in1[0..7], in2[0..7], in3[0..7]]`, it
is permuted by 128-bit lane groups, and AVX512's `packs_epi32` permutes by a
*different* 128-bit-lane pattern (see the differing formulas for AVX2 vs
AVX512 in `AffineTransform::get_weight_index_scrambled`,
`src/nnue/layers/affine_transform.h`). An engine must either (a) shuffle the
packed vector back into linear order before storing it (extra instructions on
the hot path), or (b) do what Stockfish does: leave the narrowed/activated
output in its natural permuted order, and PRE-PERMUTE the *next* layer's
weight rows once at load time so the permuted activation vector lines up
correctly against the permuted weight rows, with zero runtime shuffles.
Option (b) is strictly better once the permutation is precomputed; the exact
permutation differs by ISA (AVX2 vs AVX512 use different lane-swap patterns,
per the source above), so it must be generated for whichever ISA the binary
targets. AVX512's `cvtsepi16_epi8` narrow does NOT interleave (it's a
straight convert, not a pack), so only the *first* narrow step
(`packs_epi32`, `i32 -> i16`) needs this treatment on AVX512; AVX2 needs it
for both narrow steps since both use `packs`. The exported `.nnue` file itself
stores weights in plain input-major order (see File Format above) regardless
of ISA -- this permutation is purely an engine-internal, load-time transform
of the engine's own in-memory copy, not a file format concern.

### Suggested compiler flags

| Target | GCC/Clang | MSVC |
|---|---|---|
| x86-64 AVX2 | `-O3 -DNDEBUG -march=x86-64-v3` (or explicitly `-mavx2 -mfma -mbmi2 -mpopcnt`) | `/O2 /DNDEBUG /arch:AVX2` |
| x86-64 AVX512 (baseline F/CD/BW/DQ/VL) | `-O3 -DNDEBUG -march=x86-64-v4` | `/O2 /DNDEBUG /arch:AVX512` |
| x86-64 AVX512-VNNI | add `-mavx512vnni` explicitly | no dedicated switch, see note below |
| AArch64 NEON (baseline) | `-O3 -DNDEBUG -march=armv8-a+simd` (or a concrete `-mcpu=...`) | N/A — NEON is baseline on ARM64 |
| AArch64 NEON dot-product | `-O3 -DNDEBUG -march=armv8.2-a+dotprod`, guarded on `__ARM_FEATURE_DOTPROD` | N/A — gate at runtime, see note below |

Notes:

- `x86-64-v4` (the GCC/Clang microarchitecture-level shorthand for
  AVX512F/CD/BW/DQ/VL) does not reliably include VNNI on every toolchain --
  VNNI is a distinct CPU feature from baseline AVX512, and whether a given
  compiler's `x86-64-v4` pulls it in has differed across releases. Check your
  specific toolchain rather than assuming, and add `-mavx512vnni` explicitly
  if it doesn't. Runtime-dispatch the VNNI kernel behind a CPU-feature check
  regardless (e.g. `__builtin_cpu_supports("avx512vnni")` on GCC/Clang) — the
  same binary must still run correctly, via the `maddubs`+`madd` fallback
  above, on AVX512 hardware that lacks VNNI.
- MSVC's `/arch:AVX512` enables the F/CD/BW/DQ/VL baseline; MSVC has no
  VNNI-only switch. `_mm512_dpbusd_epi32` is usable once `/arch:AVX512` is set
  on a sufficiently recent MSVC, but the compiler does not itself verify the
  target CPU has VNNI — gate the VNNI path behind a runtime CPU-feature check
  exactly as with GCC/Clang, not just the compile flag.
- MSVC's ARM64 target always includes NEON (there is no `-march`-equivalent
  switch for it, unlike x86); dot-product intrinsics are declared but must
  still be runtime-gated via the platform's CPU-feature-detection API, since
  MSVC has no compile-time flag equivalent to GCC/Clang's
  `-march=armv8.2-a+dotprod` / `__ARM_FEATURE_DOTPROD`.
- Whatever the build's baseline ISA, keep a portable scalar fallback path (no
  intrinsics) that is bit-identical to the vectorized kernels — needed both
  for hardware below the build's minimum target and as the parity-testing
  reference for the vectorized paths.

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
