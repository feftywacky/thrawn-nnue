from __future__ import annotations

from .board import BoardState


FEATURES = "HalfKAv2_hm"

HALFKA_V2_HM_PS_NB = 11 * 64
HALFKA_V2_HM_NUM_BUCKETS = 32
HALFKA_V2_HM_NUM_FEATURES = HALFKA_V2_HM_NUM_BUCKETS * HALFKA_V2_HM_PS_NB
HALFKA_V2_HM_MAX_ACTIVE_FEATURES = 32

HALFKA_V2_HM_KING_BUCKETS = [
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
    8, 9, 10, 11, 11, 10, 9, 8,
    4, 5, 6, 7, 7, 6, 5, 4,
    0, 1, 2, 3, 3, 2, 1, 0,
]


def _validate_perspective(perspective: str) -> None:
    if perspective not in {"white", "black"}:
        raise ValueError(f"Unknown perspective: {perspective}")


def _relative_color_bit(piece: str, perspective: str) -> int:
    is_white_piece = piece.isupper()
    if perspective == "white":
        return 0 if is_white_piece else 1
    return 0 if not is_white_piece else 1


def _king_piece_for_perspective(perspective: str) -> str:
    if perspective == "white":
        return "K"
    if perspective == "black":
        return "k"
    raise ValueError(f"Unknown perspective: {perspective}")


def king_square(board_state: BoardState, perspective: str) -> int:
    king_piece = _king_piece_for_perspective(perspective)
    for square_index, piece in board_state.board.items():
        if piece == king_piece:
            return square_index
    raise ValueError(f"Board state is missing the {perspective} king")


def halfka_v2_hm_orient(square_index: int, perspective: str, king_square_index: int) -> int:
    _validate_perspective(perspective)
    orient = 7 if king_square_index % 8 < 4 else 0
    flip = 56 if perspective == "black" else 0
    return square_index ^ orient ^ flip


def halfka_v2_hm_piece_square_offset(piece: str, perspective: str) -> int:
    _validate_perspective(perspective)
    piece_name = piece.upper()
    if piece_name == "K":
        return 10 * 64
    piece_type_index = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4}.get(piece_name)
    if piece_type_index is None:
        raise ValueError(f"Unknown chess piece: {piece}")
    bucket = piece_type_index * 2 + _relative_color_bit(piece, perspective)
    return bucket * 64


def halfka_v2_hm_feature_index(
    our_king_square: int,
    square_index: int,
    piece: str,
    perspective: str,
) -> int:
    _validate_perspective(perspective)
    flip = 56 if perspective == "black" else 0
    king_bucket = HALFKA_V2_HM_KING_BUCKETS[our_king_square ^ flip]
    oriented_square = halfka_v2_hm_orient(square_index, perspective, our_king_square)
    return (
        oriented_square
        + halfka_v2_hm_piece_square_offset(piece, perspective)
        + king_bucket * HALFKA_V2_HM_PS_NB
    )


def active_feature_indices(
    board_state: BoardState,
    perspective: str,
    *,
    features: str = FEATURES,
) -> list[int]:
    canonical_feature_name(features)
    our_king_square = king_square(board_state, perspective)
    indices = [
        halfka_v2_hm_feature_index(our_king_square, square_index, piece, perspective)
        for square_index, piece in sorted(board_state.board.items())
    ]
    if len(indices) > HALFKA_V2_HM_MAX_ACTIVE_FEATURES:
        raise ValueError(
            f"Expected at most {HALFKA_V2_HM_MAX_ACTIVE_FEATURES} active features, got {len(indices)}"
        )
    return indices


def padded_feature_indices(
    board_state: BoardState,
    perspective: str,
    *,
    features: str = FEATURES,
) -> list[int]:
    max_active = feature_shape(features)["max_active_features"]
    indices = active_feature_indices(board_state, perspective, features=features)
    return indices + [-1] * (max_active - len(indices))


def canonical_feature_name(value: str) -> str:
    if value == FEATURES:
        return FEATURES
    raise ValueError("Only features='HalfKAv2_hm' is supported")


def feature_shape(value: str) -> dict[str, int]:
    canonical_feature_name(value)
    return {
        "num_features": HALFKA_V2_HM_NUM_FEATURES,
        "max_active_features": HALFKA_V2_HM_MAX_ACTIVE_FEATURES,
    }
