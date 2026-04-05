# Thrawn `.nnue` Format

This repository exports a trainer-owned binary format for `a768_dual_v1`.

## Header

All multi-byte values are little-endian.

| Field | Type | Value |
| --- | --- | --- |
| magic | 8 bytes | `THNNUE\0\1` |
| version | `u32` | `2` |
| feature_set | 16 bytes | ASCII, NUL-padded, `a768_dual_v1` |
| num_features | `u32` | `768` |
| ft_size | `u32` | usually `256` |
| hidden_size | `u32` | usually `32` |
| output_perspective | `u32` | `1` for side-to-move output |
| ft_scale | `f32` | feature-transformer quantization scale |
| dense_scale | `f32` | dense-layer quantization scale |
| wdl_scale | `f32` | training-time centipawn-to-WDL sigmoid scale |
| description_length | `u32` | bytes of UTF-8 description following the fixed header |

The description bytes come next, followed by packed tensors in this order:

1. `ft_bias`: `int16[ft_size]`
2. `ft_weight`: `int16[num_features][ft_size]`
3. `l1_bias`: `int32[hidden_size]`
4. `l1_weight`: `int8[ft_size * 2][hidden_size]`
5. `out_bias`: `int32[1]`
6. `out_weight`: `int16[hidden_size][1]` in version 2, `int8[hidden_size][1]` in version 1

## Semantics

- The feature transformer is shared between white and black perspectives.
- The first dense layer always consumes `[stm_acc, nstm_acc]`.
- Exported tensors are quantized for compact storage, but verification in this repo dequantizes them back to float for parity checks.

## Quantization

- Feature-transformer weights and bias are rounded to `int16` using `ft_scale`.
- First dense-layer weights are rounded to `int8` using `dense_scale`.
- Output-layer weights are rounded to `int16` using `dense_scale` in version 2.
- Dense-layer biases are rounded to `int32` using the product of the upstream activation scale and `dense_scale`.
