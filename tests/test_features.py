from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.board import BoardState, flip_vertical, square_to_index
from thrawn_nnue.features import (
    MAX_ACTIVE_FEATURES,
    NUM_FEATURES,
    active_factor_feature_indices,
    active_feature_indices,
    extract_halfkp,
    factor_feature_index,
    feature_index,
    king_square,
    orient_square,
)


class FeatureTests(unittest.TestCase):
    def test_black_perspective_flips_vertically_and_swaps_colors(self) -> None:
        e7 = square_to_index("e7")
        self.assertEqual(orient_square(e7, "black"), flip_vertical(e7))
        self.assertEqual(factor_feature_index(square_to_index("e2"), "P", "white"), 12)
        self.assertEqual(factor_feature_index(square_to_index("e7"), "p", "black"), 12)
        self.assertEqual(
            feature_index(square_to_index("e1"), square_to_index("e2"), "P", "white"),
            feature_index(square_to_index("e8"), square_to_index("e7"), "p", "black"),
        )

    def test_active_features_exclude_kings_and_fit_halfkp_width(self) -> None:
        board = BoardState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
        white = active_feature_indices(board, "white")
        black = active_feature_indices(board, "black")
        self.assertEqual(len(white), MAX_ACTIVE_FEATURES)
        self.assertEqual(len(black), MAX_ACTIVE_FEATURES)
        self.assertTrue(all(0 <= value < NUM_FEATURES for value in white))
        self.assertTrue(all(0 <= value < NUM_FEATURES for value in black))

    def test_extract_halfkp_returns_real_and_factor_features(self) -> None:
        board = BoardState.from_fen("8/8/8/8/8/8/4p3/4K2k w - - 0 1")
        features = extract_halfkp(board)
        self.assertEqual(len(features.white), MAX_ACTIVE_FEATURES)
        self.assertEqual(len(features.black), MAX_ACTIVE_FEATURES)
        self.assertEqual(len(features.white_factor), MAX_ACTIVE_FEATURES)
        self.assertEqual(len(features.black_factor), MAX_ACTIVE_FEATURES)
        self.assertEqual(features.stm, 1.0)
        self.assertIn(-1, features.white)
        self.assertIn(-1, features.white_factor)

    def test_factor_features_match_real_feature_modulo(self) -> None:
        board = BoardState.from_fen("8/8/8/8/8/8/P7/K6k w - - 0 1")
        real = active_feature_indices(board, "white")
        factor = active_factor_feature_indices(board, "white")
        self.assertEqual([value % 640 for value in real], factor)

    def test_king_square_uses_perspective_king(self) -> None:
        board = BoardState.from_fen("8/8/8/8/8/8/4p3/4K2k b - - 0 1")
        self.assertEqual(king_square(board, "white"), square_to_index("e1"))
        self.assertEqual(king_square(board, "black"), square_to_index("h1"))


if __name__ == "__main__":
    unittest.main()
