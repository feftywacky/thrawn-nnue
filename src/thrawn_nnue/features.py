from __future__ import annotations

from .board import BoardState, flip_vertical


HALFKP_FEATURES = "HalfKP^"
HALFKA_V2_HM_FEATURES = "HalfKAv2_hm^"

NUM_PIECE_BUCKETS = 10
NUM_FACTOR_FEATURES = NUM_PIECE_BUCKETS * 64
NUM_FEATURES = 64 * NUM_FACTOR_FEATURES
MAX_ACTIVE_FEATURES = 30

HALFKA_V2_HM_NUM_BUCKETS = 32
HALFKA_V2_HM_NUM_PLANES = 64 * 12
HALFKA_V2_HM_NUM_FEATURES = HALFKA_V2_HM_NUM_BUCKETS * HALFKA_V2_HM_NUM_PLANES
HALFKA_V2_HM_NUM_FACTOR_FEATURES = HALFKA_V2_HM_NUM_PLANES
HALFKA_V2_HM_MAX_ACTIVE_FEATURES = 32

HALFKA_V2_HM_KING_BUCKETS = [
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10, 9, 8,
    -1, -1, -1, -1, 7, 6, 5, 4,
    -1, -1, -1, -1, 3, 2, 1, 0,
]


def orient_square(square_index: int, perspective: str) -> int:
    if perspective == "white":
        return square_index
    if perspective == "black":
        return flip_vertical(square_index)
    raise ValueError(f"Unknown perspective: {perspective}")


def piece_type_index(piece: str) -> int:
    lookup = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4}
    try:
        return lookup[piece.upper()]
    except KeyError as exc:
        raise ValueError(f"HalfKP excludes kings: {piece}") from exc


def piece_type_index_with_kings(piece: str) -> int:
    lookup = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
    try:
        return lookup[piece.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown chess piece: {piece}") from exc


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


def piece_bucket_index(piece: str, perspective: str) -> int:
    return piece_type_index(piece) * 2 + _relative_color_bit(piece, perspective)


def factor_feature_index(square_index: int, piece: str, perspective: str) -> int:
    bucket = piece_bucket_index(piece, perspective)
    return bucket * 64 + orient_square(square_index, perspective)


def feature_index(our_king_square: int, square_index: int, piece: str, perspective: str) -> int:
    oriented_king = orient_square(our_king_square, perspective)
    return oriented_king * NUM_FACTOR_FEATURES + factor_feature_index(square_index, piece, perspective)


def halfka_v2_hm_orient(square_index: int, perspective: str, king_square_index: int) -> int:
    if perspective not in {"white", "black"}:
        raise ValueError(f"Unknown perspective: {perspective}")
    result = square_index
    if king_square_index % 8 < 4:
        result ^= 7
    if perspective == "black":
        result ^= 56
    return result


def halfka_v2_hm_feature_index(
    our_king_square: int,
    square_index: int,
    piece: str,
    perspective: str,
) -> int:
    piece_bucket = piece_type_index_with_kings(piece) * 2 + _relative_color_bit(piece, perspective)
    oriented_king = halfka_v2_hm_orient(our_king_square, perspective, our_king_square)
    king_bucket = HALFKA_V2_HM_KING_BUCKETS[oriented_king]
    if king_bucket < 0:
        raise ValueError("HalfKAv2_hm king orientation did not map to a valid bucket")
    oriented_square = halfka_v2_hm_orient(square_index, perspective, our_king_square)
    return oriented_square + piece_bucket * 64 + king_bucket * HALFKA_V2_HM_NUM_PLANES


def active_feature_indices(
    board_state: BoardState,
    perspective: str,
    *,
    features: str = HALFKP_FEATURES,
) -> list[int]:
    feature_name = canonical_feature_name(features)
    if feature_name == HALFKA_V2_HM_FEATURES:
        return active_halfka_v2_hm_indices(board_state, perspective)
    return active_halfkp_indices(board_state, perspective)


def active_halfkp_indices(board_state: BoardState, perspective: str) -> list[int]:
    our_king_square = king_square(board_state, perspective)
    indices = [
        feature_index(our_king_square, square_index, piece, perspective)
        for square_index, piece in sorted(board_state.board.items())
        if piece.upper() != "K"
    ]
    if len(indices) > MAX_ACTIVE_FEATURES:
        raise ValueError(f"Expected at most {MAX_ACTIVE_FEATURES} active features, got {len(indices)}")
    return indices


def active_halfka_v2_hm_indices(board_state: BoardState, perspective: str) -> list[int]:
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
    features: str = HALFKP_FEATURES,
) -> list[int]:
    feature_name = canonical_feature_name(features)
    max_active = feature_shape(feature_name)["max_active_features"]
    indices = active_feature_indices(board_state, perspective, features=feature_name)
    return indices + [-1] * (max_active - len(indices))


def canonical_feature_name(value: str) -> str:
    if value in {"HalfKP", "HalfKP^"}:
        return HALFKP_FEATURES
    if value in {"HalfKAv2_hm", "HalfKAv2_hm^"}:
        return HALFKA_V2_HM_FEATURES
    raise ValueError("Only features='HalfKP^' or features='HalfKAv2_hm^' are supported")


def feature_shape(value: str) -> dict[str, int]:
    feature_name = canonical_feature_name(value)
    if feature_name == HALFKA_V2_HM_FEATURES:
        return {
            "num_features": HALFKA_V2_HM_NUM_FEATURES,
            "num_factor_features": HALFKA_V2_HM_NUM_FACTOR_FEATURES,
            "max_active_features": HALFKA_V2_HM_MAX_ACTIVE_FEATURES,
        }
    return {
        "num_features": NUM_FEATURES,
        "num_factor_features": NUM_FACTOR_FEATURES,
        "max_active_features": MAX_ACTIVE_FEATURES,
    }
