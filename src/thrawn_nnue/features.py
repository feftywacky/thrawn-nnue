from __future__ import annotations

from dataclasses import dataclass

from .board import BoardState, flip_vertical


NUM_FEATURES = 768
MAX_ACTIVE_FEATURES = 32
MIN_PIECE_COUNT = 2
MAX_PIECE_COUNT = 32


def orient_square(square_index: int, perspective: str) -> int:
    if perspective == "white":
        return square_index
    if perspective == "black":
        return flip_vertical(square_index)
    raise ValueError(f"Unknown perspective: {perspective}")


def _relative_color_bit(piece: str, perspective: str) -> int:
    is_white_piece = piece.isupper()
    if perspective == "white":
        return 0 if is_white_piece else 1
    return 0 if not is_white_piece else 1


def piece_type_index(piece: str) -> int:
    lookup = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
    return lookup[piece.upper()]


def feature_index(square_index: int, piece: str, perspective: str) -> int:
    bucket = piece_type_index(piece) * 2 + _relative_color_bit(piece, perspective)
    return bucket * 64 + orient_square(square_index, perspective)


def active_feature_indices(board_state: BoardState, perspective: str) -> list[int]:
    indices = [
        feature_index(square_index, piece, perspective)
        for square_index, piece in sorted(board_state.board.items())
    ]
    if len(indices) > MAX_ACTIVE_FEATURES:
        raise ValueError(f"Expected at most {MAX_ACTIVE_FEATURES} active features, got {len(indices)}")
    return indices


def padded_feature_indices(board_state: BoardState, perspective: str) -> list[int]:
    indices = active_feature_indices(board_state, perspective)
    return indices + [-1] * (MAX_ACTIVE_FEATURES - len(indices))


def output_bucket_index(piece_count: int, output_buckets: int) -> int:
    if output_buckets <= 1:
        return 0

    clamped_piece_count = min(MAX_PIECE_COUNT, max(MIN_PIECE_COUNT, piece_count))
    phase_progress = MAX_PIECE_COUNT - clamped_piece_count
    phase_span = MAX_PIECE_COUNT - MIN_PIECE_COUNT + 1
    return min(output_buckets - 1, (phase_progress * output_buckets) // phase_span)


@dataclass(slots=True)
class DualPerspectiveFeatures:
    white: list[int]
    black: list[int]
    stm: float


def extract_dual_perspective(board_state: BoardState) -> DualPerspectiveFeatures:
    return DualPerspectiveFeatures(
        white=padded_feature_indices(board_state, "white"),
        black=padded_feature_indices(board_state, "black"),
        stm=1.0 if board_state.side_to_move == "w" else 0.0,
    )
