from __future__ import annotations

from dataclasses import dataclass

from .board import BoardState, flip_vertical


NUM_PIECE_BUCKETS = 10
NUM_FACTOR_FEATURES = NUM_PIECE_BUCKETS * 64
NUM_FEATURES = 64 * NUM_FACTOR_FEATURES
MAX_ACTIVE_FEATURES = 30


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


def active_feature_indices(board_state: BoardState, perspective: str) -> list[int]:
    our_king_square = king_square(board_state, perspective)
    indices = [
        feature_index(our_king_square, square_index, piece, perspective)
        for square_index, piece in sorted(board_state.board.items())
        if piece.upper() != "K"
    ]
    if len(indices) > MAX_ACTIVE_FEATURES:
        raise ValueError(f"Expected at most {MAX_ACTIVE_FEATURES} active features, got {len(indices)}")
    return indices


def active_factor_feature_indices(board_state: BoardState, perspective: str) -> list[int]:
    indices = [
        factor_feature_index(square_index, piece, perspective)
        for square_index, piece in sorted(board_state.board.items())
        if piece.upper() != "K"
    ]
    if len(indices) > MAX_ACTIVE_FEATURES:
        raise ValueError(f"Expected at most {MAX_ACTIVE_FEATURES} active features, got {len(indices)}")
    return indices


def padded_feature_indices(board_state: BoardState, perspective: str) -> list[int]:
    indices = active_feature_indices(board_state, perspective)
    return indices + [-1] * (MAX_ACTIVE_FEATURES - len(indices))


def padded_factor_feature_indices(board_state: BoardState, perspective: str) -> list[int]:
    indices = active_factor_feature_indices(board_state, perspective)
    return indices + [-1] * (MAX_ACTIVE_FEATURES - len(indices))


@dataclass(slots=True)
class HalfKPFeatures:
    white: list[int]
    black: list[int]
    white_factor: list[int]
    black_factor: list[int]
    stm: float


def extract_halfkp(board_state: BoardState) -> HalfKPFeatures:
    return HalfKPFeatures(
        white=padded_feature_indices(board_state, "white"),
        black=padded_feature_indices(board_state, "black"),
        white_factor=padded_factor_feature_indices(board_state, "white"),
        black_factor=padded_factor_feature_indices(board_state, "black"),
        stm=1.0 if board_state.side_to_move == "w" else 0.0,
    )
